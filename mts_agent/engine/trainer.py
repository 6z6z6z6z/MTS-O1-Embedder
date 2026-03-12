"""
Trainer for MTS-O1-Embedder
Handles:
1. Stage 1: Alignment (InfoNCE / Captioning Loss)
2. Stage 2: Reasoning SFT (Teacher Forcing)
"""
import json
import torch
import torch.nn.functional as F
import os
from torch.utils.data import DataLoader
from tqdm import tqdm

from mts_agent.data.samplers import BalancedBatchSampler
from mts_agent.retrieval.evaluate_retrieval import (
    evaluate_retrieval_from_cache, build_retrieval_cache,
    build_gallery, build_augmented_gallery,
    embed_queries, embed_queries_with_thought_generation,
    evaluate_gallery_vs_queries
)


class EmbeddingMemoryBank:
    """Circular queue of context-only gallery embeddings for O1-Embedder asymmetric training.

    Stores detached gallery embeddings from recent batches, providing a large
    pool of negatives for the asymmetric InfoNCE loss.  Gallery embeddings do
    NOT require gradients, so they can be accumulated cheaply across batches
    without storing any computation graph.
    """

    def __init__(self, size: int):
        self.size = size
        self._embs = None       # [size, dim] float32, CPU
        self._labels = [''] * size
        self._ptr = 0
        self._count = 0         # valid (filled) entries

    def update(self, embs: torch.Tensor, labels):
        """Enqueue new gallery embeddings (auto-detached, stored as fp32 on CPU)."""
        embs_cpu = embs.detach().float().cpu()
        B = embs_cpu.size(0)
        if self._embs is None:
            self._embs = torch.zeros(self.size, embs_cpu.size(1))
            self._labels = [''] * self.size
        for i in range(B):
            self._embs[self._ptr] = embs_cpu[i]
            self._labels[self._ptr] = str(labels[i])
            self._ptr = (self._ptr + 1) % self.size
            self._count = min(self._count + 1, self.size)

    def get(self, device):
        """Return (embeddings [N,dim], labels [N]) for all valid entries."""
        if self._count == 0:
            return None, []
        return self._embs[:self._count].to(device), self._labels[:self._count]

    def __len__(self):
        return self._count


class MTSTrainer:
    def __init__(self, model, train_dataset, collator, run_config, eval_dataset=None, test_dataset=None):
        self.model = model
        self.train_dataset = train_dataset  # kept for gallery construction during evaluation
        self.eval_dataset = eval_dataset    # internal split: used for val_loss early stopping
        self.test_dataset = test_dataset    # held-out test set: ONLY used in final_test_retrieval()
        self.collator = collator
        self.eval_dataset = eval_dataset
        self.run_config = dict(run_config)
        self.device = torch.device("cuda" if torch.cuda.is_available() else "cpu")
        print(f"Trainer initialized on device: {self.device}")
        self.model.to(self.device)

        self.batch_size = run_config.get('batch_size', 4)
        self.balanced_sampling = run_config.get('balanced_sampling', False)
        self.classes_per_batch = run_config.get('classes_per_batch', None)
        self.samples_per_class = run_config.get('samples_per_class', None)

        train_loader_kwargs = {
            'collate_fn': collator
        }
        if self.balanced_sampling:
            train_labels = [sample.get('label', '') for sample in getattr(train_dataset, 'samples', [])]
            valid_labels = [str(label) for label in train_labels if label != ""]
            unique_labels = sorted(set(valid_labels))
            if len(valid_labels) == len(train_labels) and len(unique_labels) >= 2:
                classes_per_batch = self.classes_per_batch or min(len(unique_labels), self.batch_size)
                classes_per_batch = min(classes_per_batch, self.batch_size)
                samples_per_class = self.samples_per_class or max(1, self.batch_size // classes_per_batch)
                if classes_per_batch * samples_per_class != self.batch_size:
                    samples_per_class = max(1, self.batch_size // classes_per_batch)
                    self.batch_size = classes_per_batch * samples_per_class
                    print(f"Adjusted effective batch size for balanced sampling to {self.batch_size}")
                batch_sampler = BalancedBatchSampler(train_labels, classes_per_batch, samples_per_class)
                train_loader_kwargs['batch_sampler'] = batch_sampler
                print(
                    f"Balanced sampling enabled: classes_per_batch={classes_per_batch}, "
                    f"samples_per_class={samples_per_class}"
                )
            else:
                train_loader_kwargs['batch_size'] = self.batch_size
                train_loader_kwargs['shuffle'] = True
                print("Balanced sampling requested but labels are missing or insufficient; falling back to shuffled batches.")
        else:
            train_loader_kwargs['batch_size'] = self.batch_size
            train_loader_kwargs['shuffle'] = True

        self.train_loader = DataLoader(
            train_dataset,
            **train_loader_kwargs
        )

        if eval_dataset:
            self.eval_loader = DataLoader(
                eval_dataset,
                batch_size=self.batch_size,
                shuffle=False,
                collate_fn=collator
            )
        else:
            self.eval_loader = None

        # Training configuration
        self.training_stage = run_config.get('training_stage', 'alignment')  # 'alignment' or 'reasoning'
        self.contrastive_weight = run_config.get('contrastive_weight', 0.5)
        self.contrastive_temperature = float(run_config.get('contrastive_temperature', 0.07))
        self.alignment_text_mode = run_config.get('alignment_text_mode', 'full')
        self.contrastive_positive_mode = run_config.get('contrastive_positive_mode', 'diagonal')
        self.hard_negative_weight = float(run_config.get('hard_negative_weight', 0.0))
        self.hard_negative_margin = float(run_config.get('hard_negative_margin', 0.05))
        self.neighbor_weight = run_config.get('neighbor_weight', 0.0)
        self.neighbor_margin = float(run_config.get('neighbor_margin', 0.1))
        self.lm_weight = run_config.get('lm_weight', 1.0)
        self.retrieval_eval_enabled = bool(run_config.get('retrieval_eval_enabled', False))
        # When cls_weight > 0, model selection uses val_loss on the internal split (eval_dataset);
        # LOO retrieval on train is still computed but only for monitoring, not for model selection.
        self.cls_val_loss_selection = False  # set True below if cls_weight > 0

        # O1-Embedder asymmetric contrastive: memory bank for gallery embeddings
        self.memory_bank_size = int(run_config.get('memory_bank_size', 64))
        self.memory_bank = EmbeddingMemoryBank(self.memory_bank_size)

        # Inference-time thought generation parameters (used in final_test_retrieval)
        self.inference_max_new_tokens = int(run_config.get('inference_max_new_tokens', 200))
        self.inference_temperature = float(run_config.get('inference_temperature', 0.7))
        self.inference_do_sample = bool(run_config.get('inference_do_sample', False))
        self.retrieval_eval_k = int(run_config.get('retrieval_eval_k', 5))
        self.retrieval_eval_alpha = float(run_config.get('retrieval_eval_alpha', 0.8))
        self.retrieval_eval_vote_strategy = run_config.get('retrieval_eval_vote_strategy', 'weighted')
        self.retrieval_eval_dtw_window_size = run_config.get('retrieval_eval_dtw_window_size', None)
        self.retrieval_eval_fast_dtw_max_len = int(run_config.get('retrieval_eval_fast_dtw_max_len', 100))
        # How often (in epochs) to run the retrieval evaluation. 1 = every epoch.
        self.retrieval_eval_interval = max(1, int(run_config.get('retrieval_eval_interval', 1)))

        # Mixed precision: use bf16 autocast on CUDA (Qwen2.5 is bf16-native, no GradScaler needed)
        self.use_amp = bool(run_config.get('use_amp', True)) and torch.cuda.is_available()
        self.amp_dtype = torch.bfloat16

        # TS-only embedding: bypass LLM for both training and retrieval.
        # Eliminates text-identity bias (all samples share nearly identical context strings).
        # When True, uses ts_encoder -> projector -> mean_pool directly as the retrieval embedding.
        self.ts_only_embedding = bool(run_config.get('ts_only_embedding', False))
        if self.ts_only_embedding:
            print("TS-only embedding mode: LLM bypassed for training and retrieval.")

        # Classification head: lightweight direct CE supervision on top of the retrieval embedding.
        # Provides strong class-level gradient even with tiny batches where InfoNCE is ineffective.
        self.cls_weight = float(run_config.get('cls_weight', 0.0))
        self.cls_head = None
        self.label_to_idx = {}
        if self.cls_weight > 0:
            all_labels = [str(s.get('label', '')) for s in getattr(train_dataset, 'samples', [])]
            unique_labels = sorted(set(l for l in all_labels if l != ''))
            if len(unique_labels) >= 2:
                self.label_to_idx = {l: i for i, l in enumerate(unique_labels)}
                self.cls_head = torch.nn.Sequential(
                    torch.nn.LayerNorm(model.llm_dim),
                    torch.nn.Dropout(0.3),
                    torch.nn.Linear(model.llm_dim, len(unique_labels))
                ).to(self.device)
                print(
                    f"Classification head enabled: {len(unique_labels)} classes {unique_labels}, "
                    f"weight={self.cls_weight}"
                )
                self.cls_val_loss_selection = True  # use CE val_loss for model selection (no test leakage)

        # Optimize only parameters that require gradients (+ cls_head if present)
        trainable_params = [p for p in self.model.parameters() if p.requires_grad]
        if self.cls_head is not None:
            trainable_params += list(self.cls_head.parameters())
        self.optimizer = torch.optim.AdamW(
            trainable_params,
            lr=run_config.get('lr', 1e-4)
        )

        self.num_epochs = run_config.get('epochs', 3)
        self.gradient_accumulation_steps = max(1, run_config.get('gradient_accumulation_steps', 1))
        # LR scheduler: cosine annealing over the full training horizon
        self.scheduler = torch.optim.lr_scheduler.CosineAnnealingLR(
            self.optimizer,
            T_max=max(1, self.num_epochs),
            eta_min=run_config.get('lr', 1e-4) * 0.05
        )
        patience = run_config.get('early_stopping_patience', None)
        self.early_stopping_patience = patience if patience is None else max(1, patience)
        self.early_stopping_min_delta = float(run_config.get('early_stopping_min_delta', 0.0))
        self.save_dir = run_config.get('save_dir', 'checkpoints')
        os.makedirs(self.save_dir, exist_ok=True)
        self.history_path = os.path.join(self.save_dir, "training_history.json")
        self.run_config_path = os.path.join(self.save_dir, "run_config.json")
        self.history = []

        self.best_val_loss = float('inf')
        self.epochs_without_improvement = 0
        self.best_retrieval_accuracy = 0.0      # LOO on train (monitoring)
        self.best_valid_retrieval_accuracy = 0.0  # valid-set retrieval (model selection)

        with open(self.run_config_path, 'w', encoding='utf-8') as f:
            json.dump(self.run_config, f, indent=2)

        # ARM (Adaptive Retrieval Mixer) config
        self.arm_enabled         = bool(run_config.get('arm_enabled', False))
        self.arm_k               = int(run_config.get('arm_k', self.retrieval_eval_k))
        self.arm_epochs          = int(run_config.get('arm_epochs', 100))
        self.arm_lr              = float(run_config.get('arm_lr', 3e-4))
        self.arm_num_heads       = int(run_config.get('arm_num_heads', 8))
        self.arm_dropout         = float(run_config.get('arm_dropout', 0.1))
        self.arm_augment_gallery = bool(run_config.get('arm_augment_gallery', False))
        self.arm_augment_factor  = int(run_config.get('arm_augment_factor', 5))
        if self.arm_enabled:
            print(
                f"ARM enabled: k={self.arm_k}, epochs={self.arm_epochs}, "
                f"lr={self.arm_lr}, augment_gallery={self.arm_augment_gallery}"
                + (f" (×{self.arm_augment_factor})" if self.arm_augment_gallery else "")
            )

        print(f"Training stage: {self.training_stage}")
        print(
            f"Loss weights - Contrastive: {self.contrastive_weight}, "
            f"HardNeg: {self.hard_negative_weight}, Neighbor: {self.neighbor_weight}, LM: {self.lm_weight}"
        )
        print(f"Contrastive temperature: {self.contrastive_temperature}")
        print(f"Alignment text mode: {self.alignment_text_mode}")
        print(f"Contrastive positive mode: {self.contrastive_positive_mode}")
        print(f"Hard-negative margin: {self.hard_negative_margin}")
        print(f"Gradient accumulation steps: {self.gradient_accumulation_steps}")
        print(f"Mixed precision (AMP): {'bf16' if self.use_amp else 'disabled'}")
        print(f"Memory bank size: {self.memory_bank_size} (asymmetric O1-Embedder contrastive)")
        if self.retrieval_eval_enabled and self.eval_dataset is not None:
            print(
                f"Retrieval validation enabled: k={self.retrieval_eval_k}, "
                f"alpha={self.retrieval_eval_alpha}, vote={self.retrieval_eval_vote_strategy}, "
                f"interval=every {self.retrieval_eval_interval} epoch(s)"
            )
        if self.early_stopping_patience is not None:
            print(
                f"Early stopping enabled: patience={self.early_stopping_patience}, "
                f"min_delta={self.early_stopping_min_delta}"
            )

    def _get_checkpoint_state_dict(self):
        """Return a compact checkpoint containing only trainable/adapted weights."""
        full_state = self.model.state_dict()
        checkpoint_state = {}

        trainable_names = {name for name, param in self.model.named_parameters() if param.requires_grad}
        for name in trainable_names:
            if name in full_state:
                checkpoint_state[name] = full_state[name].detach().cpu()

        # BatchNorm running stats are buffers, not parameters.
        for name, buffer in self.model.named_buffers():
            if name.startswith("ts_encoder."):
                checkpoint_state[name] = buffer.detach().cpu()

        # Keep LoRA adapter tensors even if naming differs from the trainable set.
        for name, tensor in full_state.items():
            if "lora_" in name:
                checkpoint_state[name] = tensor.detach().cpu()

        return checkpoint_state

    def _save_json(self, path, payload):
        with open(path, 'w', encoding='utf-8') as f:
            json.dump(payload, f, indent=2)

    def _save_checkpoint_metadata(self, path, epoch=None, metrics=None, is_best=False):
        metadata = {
            "checkpoint": os.path.basename(path),
            "epoch": epoch,
            "is_best": is_best,
            "training_stage": self.training_stage,
            "best_val_loss": self.best_val_loss,
            "best_retrieval_accuracy": self.best_retrieval_accuracy,
            "best_valid_retrieval_accuracy": self.best_valid_retrieval_accuracy,
            "run_config": self.run_config,
            "metrics": metrics or {}
        }
        self._save_json(f"{path}.meta.json", metadata)

    def _save_checkpoint(self, path, epoch=None, metrics=None, is_best=False):
        """Persist a compact checkpoint to disk."""
        torch.save(self._get_checkpoint_state_dict(), path)
        if self.cls_head is not None:
            torch.save(self.cls_head.state_dict(), path + ".cls_head.pt")
        self._save_checkpoint_metadata(path, epoch=epoch, metrics=metrics, is_best=is_best)

    def _append_history(self, epoch_metrics):
        self.history.append(epoch_metrics)
        self._save_json(self.history_path, self.history)

    def compute_contrastive_loss(self, ts_embeds, text_embeds, class_labels=None, temperature=None):
        """
        Compute InfoNCE contrastive loss between time series embeddings and text embeddings.
        """
        temperature = self.contrastive_temperature if temperature is None else temperature

        # Mean pooling
        ts_pooled = ts_embeds.mean(dim=1)  # [Batch, Dim]
        text_pooled = text_embeds.mean(dim=1)  # [Batch, Dim]

        # Normalize
        ts_norm = F.normalize(ts_pooled, p=2, dim=1)
        text_norm = F.normalize(text_pooled, p=2, dim=1)

        # Compute similarity matrix
        sim_matrix = torch.matmul(ts_norm, text_norm.T) / temperature  # [Batch, Batch]

        batch_size = ts_norm.size(0)
        if self.contrastive_positive_mode == 'label' and class_labels is not None:
            label_strings = [str(label) for label in class_labels]
            positive_mask = torch.tensor(
                [[anchor == candidate for candidate in label_strings] for anchor in label_strings],
                device=ts_norm.device,
                dtype=torch.bool
            )
            loss_ts = self.compute_multi_positive_nce(sim_matrix, positive_mask)
            loss_text = self.compute_multi_positive_nce(sim_matrix.T, positive_mask.T)
        else:
            labels = torch.arange(batch_size, device=ts_norm.device)
            loss_ts = F.cross_entropy(sim_matrix, labels)
            loss_text = F.cross_entropy(sim_matrix.T, labels)

        return (loss_ts + loss_text) / 2

    def compute_multi_positive_nce(self, logits, positive_mask):
        """Compute supervised multi-positive InfoNCE with one or more positives per anchor."""
        if logits.size(0) == 0:
            return torch.tensor(0.0, device=logits.device)

        logits = logits - logits.max(dim=1, keepdim=True).values
        log_prob = logits - torch.logsumexp(logits, dim=1, keepdim=True)

        positive_mask = positive_mask.float()
        positive_counts = positive_mask.sum(dim=1)
        valid_rows = positive_counts > 0
        if not valid_rows.any():
            return torch.tensor(0.0, device=logits.device)

        mean_log_prob_pos = (positive_mask * log_prob).sum(dim=1) / positive_counts.clamp_min(1.0)
        return -mean_log_prob_pos[valid_rows].mean()

    def compute_hard_negative_loss(self, ts_embeds, text_embeds, margin=None):
        """Push the positive pair above the hardest in-batch negative by a small margin."""
        batch_size = ts_embeds.size(0)
        if batch_size < 2:
            return torch.tensor(0.0, device=ts_embeds.device)

        margin = self.hard_negative_margin if margin is None else margin
        ts_pooled = F.normalize(ts_embeds.mean(dim=1), p=2, dim=1)
        text_pooled = F.normalize(text_embeds.mean(dim=1), p=2, dim=1)
        sim_matrix = torch.matmul(ts_pooled, text_pooled.T)

        positive_scores = sim_matrix.diag()
        negative_mask = ~torch.eye(batch_size, dtype=torch.bool, device=sim_matrix.device)
        masked_sim = sim_matrix.masked_fill(~negative_mask, float('-inf'))

        hardest_text_negative = masked_sim.max(dim=1).values
        hardest_ts_negative = masked_sim.max(dim=0).values

        loss_ts = F.relu(margin + hardest_text_negative - positive_scores)
        loss_text = F.relu(margin + hardest_ts_negative - positive_scores)
        return 0.5 * (loss_ts.mean() + loss_text.mean())

    def compute_neighbor_consistency_loss(self, ts_embeds, class_labels, margin=None):
        """Encourage same-label samples to stay closer than different-label samples."""
        if class_labels is None or len(class_labels) < 2:
            return torch.tensor(0.0, device=ts_embeds.device)

        margin = self.neighbor_margin if margin is None else margin
        ts_pooled = ts_embeds.mean(dim=1)
        ts_norm = F.normalize(ts_pooled, p=2, dim=1)
        sim_matrix = torch.matmul(ts_norm, ts_norm.T)

        label_strings = [str(label) for label in class_labels]
        losses = []
        for i, anchor_label in enumerate(label_strings):
            positive_indices = [j for j, label in enumerate(label_strings) if j != i and label == anchor_label]
            negative_indices = [j for j, label in enumerate(label_strings) if label != anchor_label]

            if not positive_indices or not negative_indices:
                continue

            hardest_positive = sim_matrix[i, positive_indices].min()
            hardest_negative = sim_matrix[i, negative_indices].max()
            losses.append(F.relu(margin + hardest_negative - hardest_positive))

        if not losses:
            return torch.tensor(0.0, device=ts_embeds.device)
        return torch.stack(losses).mean()

    def compute_contrastive_from_embeddings(self, embeddings, class_labels=None):
        """
        TS-only supervised contrastive loss using full LLM-processed embeddings.
        Directly optimizes the embedding space used at retrieval time.
        embeddings: [Batch, H] already-pooled embeddings from forward()
        """
        batch_size = embeddings.size(0)
        if batch_size < 2:
            return torch.tensor(0.0, device=embeddings.device)

        emb_norm = F.normalize(embeddings, p=2, dim=1)
        sim_matrix = torch.matmul(emb_norm, emb_norm.T) / self.contrastive_temperature

        if self.contrastive_positive_mode == 'label' and class_labels is not None:
            label_strings = [str(l) for l in class_labels]
            unique_labels = set(label_strings)
            # Need at least 2 classes for meaningful contrastive
            if len(unique_labels) < 2:
                return torch.tensor(0.0, device=embeddings.device)
            positive_mask = torch.tensor(
                [[a == c for c in label_strings] for a in label_strings],
                device=embeddings.device, dtype=torch.bool
            )
            return self.compute_multi_positive_nce(sim_matrix, positive_mask)
        else:
            labels = torch.arange(batch_size, device=embeddings.device)
            return F.cross_entropy(sim_matrix, labels)

    def compute_asymmetric_infonce(self, query_embs, gallery_embs, query_labels, gallery_labels):
        """O1-Embedder asymmetric InfoNCE.

        Pulls thought-enriched query embeddings toward same-class context-only
        gallery embeddings (from the memory bank).  Gradient flows only through
        query_embs; gallery_embs are detached.

        Args:
            query_embs:    [Q, H]  thought-enriched, requires grad
            gallery_embs:  [G, H]  context-only, detached (from memory bank)
            query_labels:  list of Q label strings
            gallery_labels: list of G label strings
        """
        Q = query_embs.size(0)
        G = gallery_embs.size(0)
        if Q == 0 or G == 0:
            return torch.tensor(0.0, device=query_embs.device)

        # fp32 for stable cosine similarity
        q_norm = F.normalize(query_embs.float(), p=2, dim=1)   # [Q, H]
        g_norm = F.normalize(gallery_embs.float(), p=2, dim=1)  # [G, H]

        sim_matrix = torch.matmul(q_norm, g_norm.T) / self.contrastive_temperature  # [Q, G]

        q_labs = [str(l) for l in query_labels]
        g_labs = [str(l) for l in gallery_labels]
        positive_mask = torch.tensor(
            [[ql == gl for gl in g_labs] for ql in q_labs],
            device=query_embs.device, dtype=torch.bool
        )

        # Only compute loss for rows that have at least one positive
        valid_rows = positive_mask.any(dim=1)
        if not valid_rows.any():
            return torch.tensor(0.0, device=query_embs.device)

        return self.compute_multi_positive_nce(
            sim_matrix[valid_rows], positive_mask[valid_rows]
        )

    def compute_classification_loss(self, query_emb, class_labels):
        """Cross-entropy loss via a lightweight head on the retrieval embedding.

        Unlike InfoNCE, CE is not batch-size-dependent, so it delivers useful
        gradients even with batch=8.  Used as the primary training signal when
        cls_weight > 0.
        """
        if self.cls_head is None or not class_labels:
            return torch.tensor(0.0, device=self.device)
        label_indices = [self.label_to_idx.get(str(l), -1) for l in class_labels]
        valid_pairs = [(i, idx) for i, idx in enumerate(label_indices) if idx >= 0]
        if not valid_pairs:
            return torch.tensor(0.0, device=self.device)
        emb_valid = torch.stack([query_emb[i] for i, _ in valid_pairs]).float()
        tgt_valid = torch.tensor([idx for _, idx in valid_pairs], dtype=torch.long, device=self.device)
        logits = self.cls_head(emb_valid)
        return F.cross_entropy(logits, tgt_valid)

    def compute_alignment_losses(self, ts_input, text_input_ids, class_labels):
        ts_feat = self.model.ts_encoder(ts_input)
        ts_embeds = self.model.projector(ts_feat)
        ts_embeds = self.model.projector_norm(ts_embeds)

        text_embeds = self.get_text_embeddings_from_model(text_input_ids)
        contrastive_loss = self.compute_contrastive_loss(ts_embeds, text_embeds, class_labels=class_labels)
        hard_negative_loss = self.compute_hard_negative_loss(ts_embeds, text_embeds)
        neighbor_loss = self.compute_neighbor_consistency_loss(ts_embeds, class_labels)
        return contrastive_loss, hard_negative_loss, neighbor_loss

    def get_text_embeddings_from_model(self, text_input_ids):
        """
        Extract text embeddings from the model's embedding layer, handling PEFT or base structures.
        """
        model_llm = self.model.llm
        with torch.no_grad():
            if hasattr(model_llm, "model") and hasattr(model_llm.model, "embed_tokens"):
                text_embeds = model_llm.model.embed_tokens(text_input_ids)
            elif hasattr(model_llm, "base_model") and hasattr(model_llm.base_model.model, "model") and hasattr(model_llm.base_model.model.model, "embed_tokens"):
                text_embeds = model_llm.base_model.model.model.embed_tokens(text_input_ids)
            else:
                text_embeds = model_llm.get_input_embeddings()(text_input_ids)
        return text_embeds

    def evaluate(self):
        """Run evaluation on the internal validation split.

        When ts_only_embedding is True, compute CE head loss on the TS-only embedding.
        When lm_weight == 0 and cls_weight > 0 (CE-only training), we skip the
        expensive LM forward (vocab-size logits → OOM on small GPUs) and instead
        compute only the CE classification loss on the embedding.
        """
        if not self.eval_loader:
            return 0.0
            
        self.model.eval()
        total_loss = 0
        steps = 0
        
        print("Running evaluation...")
        with torch.no_grad():
            for batch in self.eval_loader:
                ts_input = batch['ts_input'].to(self.device).float()
                text_input_ids = batch['text_input_ids'].to(self.device)
                alignment_input_ids = batch.get('alignment_input_ids', batch['text_input_ids']).to(self.device)
                class_labels = batch.get('label', None)
                
                attention_mask = batch['attention_mask'].to(self.device) if batch['attention_mask'] is not None else None
                labels = batch['labels'].to(self.device) if batch['labels'] is not None else None

                with torch.autocast(device_type='cuda', dtype=self.amp_dtype, enabled=self.use_amp):
                    if self.ts_only_embedding or self.cls_val_loss_selection:
                        # TS-only or CE-only mode: skip expensive LM forward.
                        # Val loss = CE head loss on the retrieval embedding.
                        if self.ts_only_embedding:
                            query_emb = self.model.get_ts_only_embedding(ts_input)
                        else:
                            _retrieval_ids = batch.get('retrieval_input_ids')
                            if _retrieval_ids is not None:
                                _retrieval_mask = batch.get('retrieval_attention_mask')
                                _retrieval_ids = _retrieval_ids.to(self.device)
                                _retrieval_mask = _retrieval_mask.to(self.device) if _retrieval_mask is not None else None
                                query_emb = self.model.get_embedding_for_training(
                                    ts_input, _retrieval_ids, attention_mask=_retrieval_mask
                                )
                            else:
                                query_emb = self.model.get_embedding_for_training(
                                    ts_input, alignment_input_ids
                                )
                        batch_loss = self.compute_classification_loss(query_emb, class_labels)
                    else:
                        lm_loss = torch.tensor(0.0, device=self.device)
                        contrastive_loss = torch.tensor(0.0, device=self.device)
                        hard_negative_loss = torch.tensor(0.0, device=self.device)
                        neighbor_loss = torch.tensor(0.0, device=self.device)
                        # Mirror training logic: skip expensive LM forward when lm_weight == 0
                        if self.lm_weight > 0 and labels is not None:
                            outputs = self.model(
                                ts_input=ts_input,
                                text_input_ids=text_input_ids,
                                attention_mask=attention_mask,
                                labels=labels
                            )
                            lm_loss = outputs.loss if outputs.loss is not None else lm_loss
                        if self.training_stage == 'alignment':
                            contrastive_loss, hard_negative_loss, neighbor_loss = self.compute_alignment_losses(
                                ts_input, alignment_input_ids, class_labels
                            )
                            batch_loss = (
                                self.contrastive_weight * contrastive_loss
                                + self.hard_negative_weight * hard_negative_loss
                                + self.neighbor_weight * neighbor_loss
                                + self.lm_weight * lm_loss
                            )
                        else:
                            batch_loss = lm_loss

                total_loss += batch_loss.item()
                steps += 1
                
        avg_loss = total_loss / steps if steps > 0 else 0.0
        self.model.train()
        if hasattr(self.model, 'gradient_checkpointing_enable'):
            self.model.gradient_checkpointing_enable()
        elif hasattr(self.model.llm, 'gradient_checkpointing_enable'):
            self.model.llm.gradient_checkpointing_enable() # Switch back to train mode
        return avg_loss

    def evaluate_retrieval_metrics(self):
        """Evaluate retrieval using leave-one-out on the TRAINING set.

        The test set is never touched here — this gives a clean early stopping
        signal without any information leakage from the held-out test data.
        Embeddings use context-only prompts to match the contrastive training
        distribution (contrastive loss is trained on retrieval_input_ids).
        """
        self.model.eval()
        with torch.no_grad():
            retriever, embeddings, labels, ts_data_list = build_retrieval_cache(
                self.model,
                self.train_dataset,
                self.collator,
                self.device,
                dtw_window_size=self.retrieval_eval_dtw_window_size,
                fast_dtw_max_len=self.retrieval_eval_fast_dtw_max_len,
                use_full_prompt=False,
                ts_only_embedding=self.ts_only_embedding,
            )
            metrics = evaluate_retrieval_from_cache(
                retriever, embeddings, labels, ts_data_list,
                k=self.retrieval_eval_k,
                alpha=self.retrieval_eval_alpha,
                vote_strategy=self.retrieval_eval_vote_strategy
            )
        self.model.train()
        if hasattr(self.model, 'gradient_checkpointing_enable'):
            self.model.gradient_checkpointing_enable()
        elif hasattr(self.model.llm, 'gradient_checkpointing_enable'):
            self.model.llm.gradient_checkpointing_enable()
        return metrics

    def evaluate_valid_retrieval_metrics(self):
        """Evaluate retrieval on the VALIDATION set: gallery=train, queries=valid.

        Unlike the LOO-on-train metric, this directly measures generalization to
        held-out data and should be used for model selection / early stopping.
        Returns None if no eval_dataset is available.
        """
        if not self.eval_dataset:
            return None
        self.model.eval()
        with torch.no_grad():
            retriever, _, _, _ = build_gallery(
                self.model,
                self.train_dataset,
                self.collator,
                self.device,
                dtw_window_size=self.retrieval_eval_dtw_window_size,
                fast_dtw_max_len=self.retrieval_eval_fast_dtw_max_len,
                use_full_prompt=False,
                ts_only_embedding=self.ts_only_embedding,
            )
            query_embeddings, query_labels, query_ts_list = embed_queries(
                self.model,
                self.eval_dataset,
                self.collator,
                self.device,
                ts_only_embedding=self.ts_only_embedding,
            )
        metrics = evaluate_gallery_vs_queries(
            retriever,
            query_embeddings,
            query_labels,
            query_ts_list,
            k=self.retrieval_eval_k,
            alpha=self.retrieval_eval_alpha,
            vote_strategy=self.retrieval_eval_vote_strategy
        )
        self.model.train()
        if hasattr(self.model, 'gradient_checkpointing_enable'):
            self.model.gradient_checkpointing_enable()
        elif hasattr(self.model.llm, 'gradient_checkpointing_enable'):
            self.model.llm.gradient_checkpointing_enable()
        return metrics

    def final_test_retrieval(self):
        """Final test evaluation: gallery = train+eval (context-only), queries = test (thought-enriched).

        Uses self.test_dataset if available (a true held-out test set never seen during training);
        falls back to self.eval_dataset if test_dataset is not provided.

        When a separate test_dataset exists, the gallery is expanded to include BOTH
        train_dataset and eval_dataset (all labeled reference data from the training side).
        The internal-val split received no gradient updates during training and is a
        legitimate additional gallery source — this recovers the full 316-sample gallery
        that would have been available without the 80/20 split.
        """
        report_dataset = self.test_dataset if self.test_dataset is not None else self.eval_dataset
        if not report_dataset:
            return None

        # Expand gallery: include internal-val in gallery when a true test set exists.
        if self.test_dataset is not None and self.eval_dataset is not None:
            from mts_agent.data.loader import MTSDataset
            combined_samples = list(self.train_dataset.samples) + list(self.eval_dataset.samples)
            gallery_ds = MTSDataset(
                self.train_dataset.data_path,
                mode='train',
                augment=False,
                domain_info=self.train_dataset.domain_info,
                samples=combined_samples,
            )
            print(f"Full gallery: {len(combined_samples)} samples (train {len(self.train_dataset)} + internal-val {len(self.eval_dataset)})")
        else:
            gallery_ds = self.train_dataset

        self.model.eval()
        with torch.no_grad():
            retriever, _, _, _ = build_gallery(
                self.model,
                gallery_ds,
                self.collator,
                self.device,
                dtw_window_size=self.retrieval_eval_dtw_window_size,
                fast_dtw_max_len=self.retrieval_eval_fast_dtw_max_len,
                use_full_prompt=False,  # gallery: context-only
                ts_only_embedding=self.ts_only_embedding,
            )
            # Asymmetric: queries use thought-enriched prompts (when available),
            # gallery always uses context-only (use_full_prompt=False above).
            print("Embedding test queries (thought-enriched when available)...")
            query_embeddings, query_labels, query_ts_list = embed_queries(
                self.model,
                report_dataset,
                self.collator,
                self.device,
                ts_only_embedding=self.ts_only_embedding,
            )
        metrics = evaluate_gallery_vs_queries(
            retriever,
            query_embeddings,
            query_labels,
            query_ts_list,
            k=self.retrieval_eval_k,
            alpha=self.retrieval_eval_alpha,
            vote_strategy=self.retrieval_eval_vote_strategy
        )
        self.model.train()
        if hasattr(self.model, 'gradient_checkpointing_enable'):
            self.model.gradient_checkpointing_enable()
        elif hasattr(self.model.llm, 'gradient_checkpointing_enable'):
            self.model.llm.gradient_checkpointing_enable()
        return metrics

    def train_and_evaluate_arm(self):
        """
        Post-training phase (TS-RAG-inspired): train a lightweight ARM on top
        of the frozen MTSEmbedder embeddings, then evaluate on the test set.

        Workflow
        --------
        1. Reload best checkpoint (avoids armed training on an overfit last epoch).
        2. Extract training-set embeddings (context-only, same as gallery).
        3. Optionally build augmented gallery (×arm_augment_factor) for richer
           neighbor diversity at test time.
        4. Train ARM with Leave-One-Out cross-entropy (backbone frozen).
        5. Evaluate ARM on test queries → arm_test_metrics.json.
        """
        from mts_agent.retrieval.arm import (
            AdaptiveRetrievalMixer, train_arm, evaluate_arm, build_label_map,
        )

        # 1. Reload best checkpoint so ARM trains on the best embedding space
        best_ckpt = os.path.join(self.save_dir, "model_best.pt")
        if os.path.exists(best_ckpt):
            print(f"  Reloading best checkpoint for ARM: {best_ckpt}")
            state = torch.load(best_ckpt, map_location=self.device)
            self.model.load_state_dict(state, strict=False)
        self.model.eval()

        # 2. Extract original (non-augmented) training embeddings for LOO training
        print("  Extracting training embeddings (context-only)...")
        with torch.no_grad():
            _, train_embs, train_labels, _ = build_gallery(
                self.model, self.train_dataset, self.collator, self.device,
                dtw_window_size=self.retrieval_eval_dtw_window_size,
                fast_dtw_max_len=self.retrieval_eval_fast_dtw_max_len,
                use_full_prompt=False,
            )

        # 3. Build label map from training labels
        label_to_idx = build_label_map(train_labels)
        num_classes  = len(label_to_idx)
        hidden_dim   = self.model.llm_dim
        print(
            f"  ARM config: hidden_dim={hidden_dim}, num_classes={num_classes}, "
            f"k={self.arm_k}, epochs={self.arm_epochs}"
        )

        # 4. Initialise and train ARM
        arm = AdaptiveRetrievalMixer(
            hidden_dim=hidden_dim,
            num_classes=num_classes,
            k=self.arm_k,
            num_heads=self.arm_num_heads,
            dropout=self.arm_dropout,
        )
        arm_save_path = os.path.join(self.save_dir, "arm.pt")
        arm = train_arm(
            arm, train_embs, train_labels, label_to_idx,
            k=self.arm_k,
            epochs=self.arm_epochs,
            lr=self.arm_lr,
            device=str(self.device),
            save_path=arm_save_path,
        )

        # 5. Evaluate on test set (if available)
        if self.eval_dataset is not None:
            print("\n  === ARM Final Test Evaluation ===")

            # Extract test embeddings
            with torch.no_grad():
                query_embs, query_labels, _ = embed_queries(
                    self.model, self.eval_dataset, self.collator, self.device
                )

            # Optionally build augmented gallery for richer test-time retrieval
            if self.arm_augment_gallery:
                print(f"  Building augmented gallery (\u00d7{self.arm_augment_factor})...")
                with torch.no_grad():
                    _, gallery_embs, gallery_labels, _ = build_augmented_gallery(
                        self.model, self.train_dataset, self.collator, self.device,
                        num_augments=self.arm_augment_factor,
                        dtw_window_size=self.retrieval_eval_dtw_window_size,
                        fast_dtw_max_len=self.retrieval_eval_fast_dtw_max_len,
                    )
            else:
                gallery_embs   = train_embs
                gallery_labels = train_labels

            metrics = evaluate_arm(
                arm, query_embs, query_labels,
                gallery_embs, gallery_labels,
                label_to_idx,
                k=self.arm_k,
                device=str(self.device),
            )

            arm_metrics_path = os.path.join(self.save_dir, "arm_test_metrics.json")
            self._save_json(arm_metrics_path, {
                "arm_test_accuracy":  metrics["accuracy"],
                "arm_test_f1_macro":  metrics["f1_macro"],
                "k":                  self.arm_k,
                "arm_epochs":         self.arm_epochs,
                "arm_augment_gallery": self.arm_augment_gallery,
                "arm_augment_factor": self.arm_augment_factor,
                "predictions":         metrics["predictions"],
                "references":          metrics["true_labels"],
            })
            print(f"  ARM test metrics saved to {arm_metrics_path}")

        # Restore training mode
        self.model.train()

    def train(self):
        print(f"Starting training loop in stage: {self.training_stage}...")
        self.model.train()
        if hasattr(self.model, 'gradient_checkpointing_enable'):
            self.model.gradient_checkpointing_enable()
        elif hasattr(self.model.llm, 'gradient_checkpointing_enable'):
            self.model.llm.gradient_checkpointing_enable()

        self.optimizer.zero_grad()
        for epoch in range(self.num_epochs):
            total_loss = 0
            total_lm_loss = 0
            total_contrastive_loss = 0
            total_hard_negative_loss = 0
            total_neighbor_loss = 0
            total_cls_loss = 0
            # Use tqdm for progress bar
            progress_bar = tqdm(self.train_loader, desc=f"Epoch {epoch+1}/{self.num_epochs}")

            for i, batch in enumerate(progress_bar):
                # Move batch to device
                ts_input = batch['ts_input'].to(self.device).float()
                text_input_ids = batch['text_input_ids'].to(self.device)
                alignment_input_ids = batch.get('alignment_input_ids', batch['text_input_ids']).to(self.device)
                class_labels = batch.get('label', None)

                attention_mask = None
                if batch['attention_mask'] is not None:
                    attention_mask = batch['attention_mask'].to(self.device)

                labels = None
                if batch['labels'] is not None:
                    labels = batch['labels'].to(self.device)

                # Route B – Symmetric training:
                # Query embedding = context-only (same as gallery), computed with gradient.
                # Memory bank provides cross-batch negatives (context-only, detached).
                # LM forward is skipped when lm_weight == 0 to save VRAM.
                lm_loss = torch.tensor(0.0, device=self.device)
                query_emb = None
                contrastive_loss = torch.tensor(0.0, device=self.device)
                hard_negative_loss = torch.tensor(0.0, device=self.device)
                neighbor_loss = torch.tensor(0.0, device=self.device)
                cls_loss = torch.tensor(0.0, device=self.device)

                if self.training_stage == 'alignment' and (
                    self.contrastive_weight > 0 or self.cls_weight > 0
                ):
                    _retrieval_ids = batch.get('retrieval_input_ids')
                    if self.ts_only_embedding:
                        # TS-only mode: bypass LLM entirely.
                        # Pure signal embedding with full gradient flow to ts_encoder + projector.
                        with torch.autocast(device_type='cuda', dtype=self.amp_dtype, enabled=self.use_amp):
                            query_emb = self.model.get_ts_only_embedding(ts_input)
                        # Contrastive loss on TS embedding
                        if self.contrastive_weight > 0:
                            contrastive_loss = self.compute_contrastive_from_embeddings(
                                query_emb, class_labels
                            )
                        if self.cls_weight > 0 and self.cls_head is not None:
                            cls_loss = self.compute_classification_loss(query_emb, class_labels)
                    else:
                        _retrieval_ids = batch.get('retrieval_input_ids')
                        _retrieval_mask = batch.get('retrieval_attention_mask')
                        if _retrieval_ids is not None:
                            _retrieval_ids = _retrieval_ids.to(self.device)
                            _retrieval_mask = _retrieval_mask.to(self.device) if _retrieval_mask is not None else None
                            # ── Asymmetric O1-Embedder training ──────────────────────
                            # Query  : thought-enriched text (text_input_ids) — WITH gradients
                            # Gallery: context-only text (retrieval_input_ids) — NO gradients
                            # Gradient flows only through the query path; the gallery path
                            # still benefits as the shared TS encoder improves.
                            with torch.autocast(device_type='cuda', dtype=self.amp_dtype, enabled=self.use_amp):
                                query_emb = self.model.get_embedding_for_training(
                                    ts_input, text_input_ids, attention_mask=attention_mask
                                )
                            with torch.no_grad():
                                with torch.autocast(device_type='cuda', dtype=self.amp_dtype, enabled=self.use_amp):
                                    gallery_emb = self.model.get_embedding_for_training(
                                        ts_input, _retrieval_ids, attention_mask=_retrieval_mask
                                    )
                            gallery_emb = gallery_emb.detach()
                            # Update memory bank with current in-batch gallery embeddings
                            if self.memory_bank_size > 0:
                                self.memory_bank.update(gallery_emb, class_labels)
                            # Asymmetric InfoNCE: same-class gallery items are positives.
                            # Augment gallery with memory bank embeddings for more negatives.
                            if self.contrastive_weight > 0:
                                mb_embs, mb_labels = self.memory_bank.get(self.device)
                                if mb_embs is not None and len(mb_embs) > 0:
                                    all_gallery_emb = torch.cat(
                                        [gallery_emb, mb_embs.to(gallery_emb.dtype)], dim=0
                                    )
                                    all_gallery_labels = list(class_labels) + list(mb_labels)
                                else:
                                    all_gallery_emb = gallery_emb
                                    all_gallery_labels = class_labels
                                contrastive_loss = self.compute_asymmetric_infonce(
                                    query_emb, all_gallery_emb, class_labels, all_gallery_labels
                                )
                            # Classification loss on query embedding (optional)
                            if self.cls_weight > 0 and self.cls_head is not None:
                                cls_loss = self.compute_classification_loss(query_emb, class_labels)

                    if self.hard_negative_weight > 0 or self.neighbor_weight > 0:
                        ts_feat = self.model.ts_encoder(ts_input)
                        ts_embeds = self.model.projector(ts_feat)
                        ts_embeds = self.model.projector_norm(ts_embeds)
                        if self.hard_negative_weight > 0:
                            text_embeds_aux = self.get_text_embeddings_from_model(alignment_input_ids)
                            hard_negative_loss = self.compute_hard_negative_loss(ts_embeds, text_embeds_aux)
                        if self.neighbor_weight > 0:
                            neighbor_loss = self.compute_neighbor_consistency_loss(ts_embeds, class_labels)

                # Optional LM loss (skipped when lm_weight == 0 to save VRAM)
                if self.lm_weight > 0 and labels is not None:
                    with torch.autocast(device_type='cuda', dtype=self.amp_dtype, enabled=self.use_amp):
                        outputs = self.model(
                            ts_input=ts_input,
                            text_input_ids=text_input_ids,
                            attention_mask=attention_mask,
                            labels=labels,
                        )
                    lm_loss = outputs.loss if outputs.loss is not None else torch.tensor(0.0, device=self.device)

                if self.training_stage == 'reasoning':
                    # Reasoning SFT: LM loss only
                    with torch.autocast(device_type='cuda', dtype=self.amp_dtype, enabled=self.use_amp):
                        outputs = self.model(
                            ts_input=ts_input,
                            text_input_ids=text_input_ids,
                            attention_mask=attention_mask,
                            labels=labels,
                        )
                    lm_loss = outputs.loss if outputs.loss is not None else torch.tensor(0.0, device=self.device)

                # Combined loss (works for both alignment and reasoning)
                loss = (
                    self.contrastive_weight * contrastive_loss
                    + self.hard_negative_weight * hard_negative_loss
                    + self.neighbor_weight * neighbor_loss
                    + self.cls_weight * cls_loss
                    + self.lm_weight * lm_loss
                )

                # Skip backward when loss has no gradient (e.g., empty memory bank on first step)
                if not loss.requires_grad:
                    continue

                # Gradient Accumulation
                loss = loss / self.gradient_accumulation_steps

                # Backward Pass
                loss.backward()

                if (i + 1) % self.gradient_accumulation_steps == 0 or (i + 1) == len(self.train_loader):
                    self.optimizer.step()
                    self.optimizer.zero_grad()

                total_loss += loss.item() * self.gradient_accumulation_steps
                total_lm_loss += lm_loss.item()
                total_contrastive_loss += contrastive_loss.item()
                total_hard_negative_loss += hard_negative_loss.item()
                total_neighbor_loss += neighbor_loss.item()
                total_cls_loss += cls_loss.item()

                # Update progress bar
                postfix = {'loss': f"{loss.item() * self.gradient_accumulation_steps:.4f}"}
                if self.training_stage == 'alignment':
                    postfix['closs'] = f"{contrastive_loss.item():.4f}"
                    if self.cls_weight > 0:
                        postfix['csloss'] = f"{cls_loss.item():.4f}"
                    postfix['hnloss'] = f"{hard_negative_loss.item():.4f}"
                    postfix['nloss'] = f"{neighbor_loss.item():.4f}"
                postfix['lloss'] = f"{lm_loss.item():.4f}"
                progress_bar.set_postfix(postfix)
            
            train_loss = total_loss / len(self.train_loader) if len(self.train_loader) > 0 else 0
            avg_lm_loss = total_lm_loss / len(self.train_loader) if len(self.train_loader) > 0 else 0
            avg_contrastive_loss = total_contrastive_loss / len(self.train_loader) if len(self.train_loader) > 0 else 0
            avg_hard_negative_loss = total_hard_negative_loss / len(self.train_loader) if len(self.train_loader) > 0 else 0
            avg_neighbor_loss = total_neighbor_loss / len(self.train_loader) if len(self.train_loader) > 0 else 0
            avg_cls_loss = total_cls_loss / len(self.train_loader) if len(self.train_loader) > 0 else 0
            self.scheduler.step()

            # Validation Step
            if self.eval_loader:
                val_loss = self.evaluate()
                retrieval_metrics = None
                valid_retrieval_metrics = None
                if self.retrieval_eval_enabled and (epoch + 1) % self.retrieval_eval_interval == 0:
                    retrieval_metrics = self.evaluate_retrieval_metrics()       # LOO on train
                    valid_retrieval_metrics = self.evaluate_valid_retrieval_metrics()  # valid set
                if self.training_stage == 'alignment':
                    print(f"Epoch {epoch+1} finished. Train Loss: {train_loss:.4f} (CL: {avg_contrastive_loss:.4f}, CS: {avg_cls_loss:.4f}, LM: {avg_lm_loss:.4f}, HN: {avg_hard_negative_loss:.4f}, NL: {avg_neighbor_loss:.4f}) | Val Loss: {val_loss:.4f}")
                else:
                    print(f"Epoch {epoch+1} finished. Train Loss: {train_loss:.4f} | Val Loss: {val_loss:.4f}")
                if retrieval_metrics is not None:
                    print(
                        f"Train LOO Retrieval - Accuracy: {retrieval_metrics['accuracy']:.2%} | "
                        f"Macro-F1: {retrieval_metrics['macro_f1']:.2%}"
                    )
            else:
                if self.training_stage == 'alignment':
                    print(f"Epoch {epoch+1} finished. Train Loss: {train_loss:.4f} (CL: {avg_contrastive_loss:.4f}, CS: {avg_cls_loss:.4f}, LM: {avg_lm_loss:.4f}, HN: {avg_hard_negative_loss:.4f}, NL: {avg_neighbor_loss:.4f})")
                else:
                    print(f"Epoch {epoch+1} finished. Train Loss: {train_loss:.4f}")

            metrics_summary = {
                "train_loss": train_loss,
                "train_lm_loss": avg_lm_loss,
                "train_contrastive_loss": avg_contrastive_loss,
                "train_cls_loss": avg_cls_loss,
                "train_hard_negative_loss": avg_hard_negative_loss,
                "train_neighbor_loss": avg_neighbor_loss,
            }
            if self.eval_loader:
                metrics_summary["val_loss"] = val_loss
            if self.eval_loader and retrieval_metrics is not None:
                metrics_summary["retrieval_accuracy"] = float(retrieval_metrics["accuracy"])
                metrics_summary["retrieval_macro_f1"] = float(retrieval_metrics["macro_f1"])
                if self.eval_loader and valid_retrieval_metrics is not None:
                    metrics_summary["valid_retrieval_accuracy"] = float(valid_retrieval_metrics["accuracy"])
                    metrics_summary["valid_retrieval_macro_f1"] = float(valid_retrieval_metrics["macro_f1"])
                    print(
                        f"Valid Retrieval  - Accuracy: {valid_retrieval_metrics['accuracy']:.2%} | "
                        f"Macro-F1: {valid_retrieval_metrics['macro_f1']:.2%}"
                    )

            self._append_history({
                "epoch": epoch + 1,
                **metrics_summary
            })

            # Save Checkpoint (Best & Last only)
            last_path = os.path.join(self.save_dir, "model_last.pt")
            self._save_checkpoint(last_path, epoch=epoch + 1, metrics=metrics_summary, is_best=False)
            print(f"Saved checkpoint to {last_path}")
            
            if self.eval_loader:
                # ── Model selection / early-stopping ──────────────────────────────────────
                # Priority (highest → lowest):
                #  1. cls_val_loss_selection: primary = valid retrieval accuracy on the
                #     INTERNAL val split (62 samples from training file, no test leakage).
                #     Falls back to CE val_loss on non-retrieval-eval epochs.
                #  2. retrieval_eval_enabled: LOO retrieval accuracy on training set.
                #  3. fallback: val_loss.
                if self.cls_val_loss_selection:
                    # Track LOO for monitoring
                    if retrieval_metrics is not None:
                        current_train_loo = float(retrieval_metrics['accuracy'])
                        self.best_retrieval_accuracy = max(self.best_retrieval_accuracy, current_train_loo)
                    if valid_retrieval_metrics is not None:
                        # Retrieval eval epoch: use internal-val retrieval accuracy (primary)
                        current_acc = float(valid_retrieval_metrics['accuracy'])
                        improved = current_acc > self.best_valid_retrieval_accuracy + self.early_stopping_min_delta
                        if improved:
                            self.best_valid_retrieval_accuracy = current_acc
                        if val_loss < self.best_val_loss:
                            self.best_val_loss = val_loss
                    else:
                        # Non-retrieval epoch: compare CE val_loss, but only save if also
                        # improves on the best retrieval score seen so far.
                        # (Prevents saving a CE-cheap epoch that hurt retrieval)
                        improved = val_loss < (self.best_val_loss - self.early_stopping_min_delta)
                        if improved:
                            self.best_val_loss = val_loss
                            # Only count as "improved" for early-stopping if no retrieval
                            # metric has been seen yet (i.e. in the first few epochs)
                            improved = self.best_valid_retrieval_accuracy == 0.0
                elif self.retrieval_eval_enabled and retrieval_metrics is not None:
                    # Contrastive training: prefer valid-set retrieval accuracy for model
                    # selection (no train distribution leakage).  Fall back to LOO on
                    # training set when the valid retrieval was not computed this epoch.
                    current_train_loo = float(retrieval_metrics['accuracy'])
                    self.best_retrieval_accuracy = max(self.best_retrieval_accuracy, current_train_loo)
                    if val_loss < self.best_val_loss:
                        self.best_val_loss = val_loss
                    if valid_retrieval_metrics is not None:
                        current_acc = float(valid_retrieval_metrics['accuracy'])
                        improved = current_acc > self.best_valid_retrieval_accuracy + self.early_stopping_min_delta
                        if improved:
                            self.best_valid_retrieval_accuracy = current_acc
                    else:
                        improved = current_train_loo > self.best_valid_retrieval_accuracy + self.early_stopping_min_delta
                        if improved:
                            self.best_valid_retrieval_accuracy = current_train_loo
                else:
                    # Fallback: val_loss (lower is better)
                    improved = val_loss < (self.best_val_loss - self.early_stopping_min_delta)
                    if improved:
                        self.best_val_loss = val_loss

                if improved:
                    self.epochs_without_improvement = 0
                    best_path = os.path.join(self.save_dir, "model_best.pt")
                    self._save_checkpoint(best_path, epoch=epoch + 1, metrics=metrics_summary, is_best=True)
                    print(f"New Best Model saved to {best_path}")
                else:
                    self.epochs_without_improvement += 1
                    if self.early_stopping_patience is not None:
                        print(
                            f"No improvement for {self.epochs_without_improvement} epoch(s) "
                            f"(patience={self.early_stopping_patience})."
                        )
                        if self.epochs_without_improvement >= self.early_stopping_patience:
                            print(
                                f"Early stopping triggered at epoch {epoch+1}. "
                                f"Best val loss: {self.best_val_loss:.4f} | "
                                f"Best train LOO retrieval accuracy: {self.best_retrieval_accuracy:.2%}"
                            )
                            break

        # ── Final test-set evaluation (run exactly once after training completes) ──
        report_ds = self.test_dataset if self.test_dataset is not None else self.eval_dataset
        if self.retrieval_eval_enabled and report_ds:
            print("\n=== Final Test Set Evaluation (gallery=train, query=test) ===")
            final_metrics = self.final_test_retrieval()
            if final_metrics:
                print(
                    f"Test Retrieval - Accuracy: {final_metrics['accuracy']:.2%} | "
                    f"Macro-F1: {final_metrics['macro_f1']:.2%}"
                )
                final_path = os.path.join(self.save_dir, "final_test_metrics.json")
                self._save_json(final_path, {
                    "test_accuracy": final_metrics['accuracy'],
                    "test_macro_f1": final_metrics['macro_f1'],
                    "k": self.retrieval_eval_k,
                    "alpha": self.retrieval_eval_alpha,
                    "vote_strategy": self.retrieval_eval_vote_strategy,
                    "predictions": final_metrics['predictions'],
                    "references": final_metrics['references']
                })
                print(f"Final test metrics saved to {final_path}")

        # ── ARM post-training (optional) ─────────────────────────────────────
        if self.arm_enabled:
            print("\n=== Training Adaptive Retrieval Mixer (ARM, TS-RAG-inspired) ===")
            self.train_and_evaluate_arm()
