"""
Trainer for MTS-O1-Embedder
Handles:
1. Stage 1: Alignment (InfoNCE / Captioning Loss)
2. Stage 2: Reasoning SFT (Teacher Forcing)

"""
import json
import torch
import torch.nn.functional as F
import torch.distributed as dist
import os
from contextlib import contextmanager
from dataclasses import dataclass
from torch.utils.data import DataLoader
from tqdm import tqdm

from mts_agent.data.samplers import BalancedBatchSampler, MultiDatasetBatchSampler
from mts_agent.retrieval.evaluate_retrieval import (
    evaluate_retrieval_from_cache, build_retrieval_cache,
    build_gallery,
    embed_queries,
    evaluate_gallery_vs_queries
)

class DistributedGatherLayer(torch.autograd.Function):
    """
    Gather tensors from all process, supporting backward propagation.
    """
    @staticmethod
    def forward(ctx, x):
        if not dist.is_initialized():
            return x
        output = [torch.zeros_like(x) for _ in range(dist.get_world_size())]
        dist.all_gather(output, x)
        return torch.cat(output, dim=0)

    @staticmethod
    def backward(ctx, grads):
        if not dist.is_initialized():
            return grads
        all_gradients = torch.stack(grads.chunk(dist.get_world_size(), dim=0))
        dist.all_reduce(all_gradients)
        return all_gradients[dist.get_rank()]


class EmbeddingMemoryBank:
    """Circular queue of context-only gallery embeddings for O1-Embedder asymmetric training.

    Stores detached gallery embeddings from recent batches, providing a large
    pool of negatives for the asymmetric InfoNCE loss.  Gallery embeddings do
    NOT require gradients, so they can be accumulated cheaply across batches
    without storing any computation graph.

    Optionally stores raw time-series data alongside embeddings so that
    SoftCLT DTW-based soft labels can be computed for all gallery pairs.
    """

    def __init__(self, size: int, store_ts: bool = False):
        self.size = size
        self.store_ts = store_ts
        self._embs = None       # [size, dim] float32, CPU
        self._labels = [''] * size
        self._ts: list = [None] * size   # raw TS numpy arrays when store_ts=True
        self._ptr = 0
        self._count = 0         # valid (filled) entries

    def update(self, embs: torch.Tensor, labels, ts: torch.Tensor = None):
        """Enqueue new gallery embeddings (auto-detached, stored as fp32 on CPU)."""
        import numpy as np
        embs_cpu = embs.detach().float().cpu()
        B = embs_cpu.size(0)
        if self._embs is None:
            self._embs = torch.zeros(self.size, embs_cpu.size(1))
            self._labels = [''] * self.size
            self._ts = [None] * self.size
        for i in range(B):
            self._embs[self._ptr] = embs_cpu[i]
            self._labels[self._ptr] = str(labels[i])
            if self.store_ts and ts is not None:
                self._ts[self._ptr] = ts[i].detach().cpu().numpy().astype(np.float32)
            self._ptr = (self._ptr + 1) % self.size
            self._count = min(self._count + 1, self.size)

    def get(self, device):
        """Return (embeddings [N,dim], labels [N]) for all valid entries."""
        if self._count == 0:
            return None, []
        return self._embs[:self._count].to(device), self._labels[:self._count]

    def get_ts(self):
        """Return list of raw TS numpy arrays for all valid entries (or None list)."""
        return self._ts[:self._count]

    def reset(self):
        """Flush all stored embeddings (e.g. at warmup→training transition)."""
        self._embs = None
        self._labels = [''] * self.size
        self._ts = [None] * self.size
        self._ptr = 0
        self._count = 0

    def __len__(self):
        return self._count



@dataclass
class EpochLossTracker:
    total_loss: float = 0.0
    total_lm_loss: float = 0.0
    total_contrastive_loss: float = 0.0
    total_cls_loss: float = 0.0
    total_proto_loss: float = 0.0
    total_forecast_loss: float = 0.0
    effective_steps: int = 0

    def update(self, losses, scaled_total_loss):
        self.total_loss += float(scaled_total_loss)
        self.total_lm_loss += float(losses['lm_loss'].item())
        self.total_contrastive_loss += float(losses['contrastive_loss'].item())
        self.total_cls_loss += float(losses['cls_loss'].item())
        self.total_proto_loss += float(losses['proto_loss'].item())
        self.total_forecast_loss += float(losses.get('forecast_loss', torch.tensor(0.0)).item())
        self.effective_steps += 1

    def averages(self):
        if self.effective_steps == 0:
            return {
                'train_loss': 0.0,
                'train_lm_loss': 0.0,
                'train_contrastive_loss': 0.0,
                'train_cls_loss': 0.0,
                'train_proto_loss': 0.0,
                'train_forecast_loss': 0.0,
            }

        denom = float(self.effective_steps)
        return {
            'train_loss': self.total_loss / denom,
            'train_lm_loss': self.total_lm_loss / denom,
            'train_contrastive_loss': self.total_contrastive_loss / denom,
            'train_cls_loss': self.total_cls_loss / denom,
            'train_proto_loss': self.total_proto_loss / denom,
            'train_forecast_loss': self.total_forecast_loss / denom,
        }


class MTSTrainer:
    def __init__(self, model, train_dataset, collator, run_config, eval_dataset=None, test_dataset=None):
        self.model = model
        self.train_dataset = train_dataset  # kept for gallery construction during evaluation
        self.eval_dataset = eval_dataset    # internal split: used for val_loss early stopping
        self.test_dataset = test_dataset    # held-out test set: ONLY used in final_test_retrieval()
        self.collator = collator
        self.run_config = dict(run_config)

        self.rank = int(os.environ.get("RANK", "0"))
        self.local_rank = int(os.environ.get("LOCAL_RANK", "0"))
        self.world_size = int(os.environ.get("WORLD_SIZE", "1"))
        self.is_main_process = self.rank == 0

        if torch.cuda.is_available():
            # In torchrun multi-process mode, each worker must bind to its own LOCAL_RANK GPU.
            torch.cuda.set_device(self.local_rank)
            self.device = torch.device(f"cuda:{self.local_rank}")
        else:
            self.device = torch.device("cpu")

        print(
            f"Trainer initialized on device: {self.device} "
            f"(rank={self.rank}, local_rank={self.local_rank}, world_size={self.world_size})"
        )
        # The LLM is loaded directly to device via device_map in _load_llm; calling
        # .to(device) on it again can interfere with the dispatch hooks and is wasteful.
        # Move only the non-LLM sub-modules (ts_encoder, projector, cls_head, …) which
        # were initialised on CPU.
        for _child_name, _child_mod in self.model.named_children():
            if _child_name != "llm":
                _child_mod.to(self.device)

        self.batch_size = run_config.get('batch_size', 4)
        self.seed = run_config.get('seed', None)
        self.balanced_sampling = run_config.get('balanced_sampling', False)
        self.classes_per_batch = run_config.get('classes_per_batch', None)
        self.samples_per_class = run_config.get('samples_per_class', None)
        self.allow_repeated_classes_in_batch = run_config.get('allow_repeated_classes_in_batch', True)

        train_loader_kwargs = {
            'collate_fn': collator
        }
        if self.balanced_sampling:
            train_samples = getattr(train_dataset, 'samples', [])
            train_labels = [sample.get('label', '') for sample in train_samples]
            train_datasets = [str(sample.get('dataset', '')) for sample in train_samples]
            unique_datasets = sorted(set(d for d in train_datasets if d))

            # Multi-dataset mode: use MultiDatasetBatchSampler
            if len(unique_datasets) > 1:
                dataset_labels = list(zip(train_datasets, train_labels))
                classes_per_batch = self.classes_per_batch or 6
                samples_per_class = self.samples_per_class or 3
                self.batch_size = classes_per_batch * samples_per_class
                batch_sampler = MultiDatasetBatchSampler(
                    dataset_labels,
                    classes_per_batch,
                    samples_per_class,
                    seed=self.seed + self.rank * 1000,
                    allow_repeated_classes=self.allow_repeated_classes_in_batch,
                )
                train_loader_kwargs['batch_sampler'] = batch_sampler
                print(
                    f"Multi-dataset balanced sampling: {len(unique_datasets)} datasets, "
                    f"classes_per_batch={classes_per_batch}, samples_per_class={samples_per_class}"
                )
            else:
                # Single-dataset mode: existing BalancedBatchSampler
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
                    batch_sampler = BalancedBatchSampler(
                        train_labels,
                        classes_per_batch,
                        samples_per_class,
                        seed=self.seed + self.rank * 1000,
                        allow_repeated_classes=self.allow_repeated_classes_in_batch,
                    )
                    train_loader_kwargs['batch_sampler'] = batch_sampler
                    print(
                        f"Balanced sampling enabled: classes_per_batch={classes_per_batch}, "
                        f"samples_per_class={samples_per_class}, "
                        f"allow_repeated_classes_in_batch={self.allow_repeated_classes_in_batch}"
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
            # Eval does not need balanced sampling; use a smaller batch to avoid OOM
            # after the training epoch leaves the GPU mostly full.
            eval_batch_size = max(1, self.batch_size // 3)
            self.eval_loader = DataLoader(
                eval_dataset,
                batch_size=eval_batch_size,
                shuffle=False,
                collate_fn=collator
            )
        else:
            self.eval_loader = None

        # Training configuration
        self.training_stage = run_config.get('training_stage', 'alignment')  # 'alignment' or 'reasoning'
        self.contrastive_weight = run_config.get('contrastive_weight', 0.5)
        _base_temp = float(run_config.get('contrastive_temperature', 0.07))
        # Learnable temperature: log_τ is an nn.Parameter optimised jointly with the model.
        # Clipped to [log(0.01), log(1.0)] to prevent collapse or uniform softmax.
        # Set learnable_temperature=false in config to use a fixed τ instead.
        if run_config.get('learnable_temperature', True):
            import math
            self.log_temperature = torch.nn.Parameter(
                torch.tensor(math.log(_base_temp), dtype=torch.float32, device=self.device)
            )
            self.contrastive_temperature = None   # sentinel: use log_temperature at runtime
            print(f"Learnable temperature enabled: init τ={_base_temp:.4f} (log_τ={math.log(_base_temp):.4f})")
        else:
            self.log_temperature = None
            self.contrastive_temperature = _base_temp
            print(f"Fixed temperature: τ={_base_temp:.4f}")
        self.alignment_text_mode = run_config.get('alignment_text_mode', 'full')
        self.contrastive_positive_mode = run_config.get('contrastive_positive_mode', 'diagonal')
        self.lm_weight = run_config.get('lm_weight', 1.0)
        self.retrieval_eval_enabled = bool(run_config.get('retrieval_eval_enabled', False))
        # When cls_weight > 0, model selection uses val_loss on the internal split (eval_dataset);
        # LOO retrieval on train is still computed but only for monitoring, not for model selection.
        self.cls_val_loss_selection = False  # set True below if cls_weight > 0

        # SoftCLT: DTW-based soft positive labels (ICLR 2024).
        # Replaces hard 0/1 positive mask with exp(-DTW(i,j)/sigma) weights.
        self.softclt_enabled = bool(run_config.get('softclt_enabled', False))
        self.softclt_sigma = float(run_config.get('softclt_sigma', 1.0))

        # O1-Embedder asymmetric contrastive: memory bank for gallery embeddings
        self.memory_bank_size = int(run_config.get('memory_bank_size', 64))
        # store_ts=True when SoftCLT needs raw TS for DTW soft weight computation
        self.memory_bank = EmbeddingMemoryBank(self.memory_bank_size, store_ts=self.softclt_enabled)

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

        # Mixed precision: configurable bf16/fp16 autocast on CUDA.
        # fp16 is often more stable on consumer Ada GPUs (e.g., 4090) for large 9B runs.
        self.use_amp = bool(run_config.get('use_amp', True)) and torch.cuda.is_available()
        amp_dtype_name = str(run_config.get('amp_dtype', 'bf16')).lower().strip()
        if amp_dtype_name not in {'bf16', 'fp16'}:
            print(f"Unknown amp_dtype='{amp_dtype_name}', fallback to bf16")
            amp_dtype_name = 'bf16'

        if amp_dtype_name == 'bf16':
            bf16_supported = bool(getattr(torch.cuda, 'is_bf16_supported', lambda: False)())
            if self.use_amp and not bf16_supported:
                print("bf16 autocast is not supported on this CUDA runtime/GPU; fallback to fp16")
                amp_dtype_name = 'fp16'

        self.amp_dtype = torch.bfloat16 if amp_dtype_name == 'bf16' else torch.float16

        # TS-only embedding: bypass LLM for both training and retrieval.
        # Eliminates text-identity bias (all samples share nearly identical context strings).
        # When True, uses ts_encoder -> projector -> mean_pool directly as the retrieval embedding.
        self.ts_only_embedding = bool(run_config.get('ts_only_embedding', False))
        if self.ts_only_embedding:
            print("TS-only embedding mode: LLM bypassed for training and retrieval.")
            if self.use_amp:
                # TS-only path is lightweight; disabling AMP avoids rare bf16/DDP kernel faults.
                self.use_amp = False
                print("TS-only mode: disabling AMP for stability.")
            if self.lm_weight == 0:
                # In pure TS-only contrastive mode, LLM is unused. Keep it off GPU to reduce
                # memory pressure and avoid sporadic CUDA kernel faults on some multi-GPU runs.
                for p in self._raw_model.llm.parameters():
                    p.requires_grad = False
                self._raw_model.llm.to("cpu")
                torch.cuda.empty_cache()
                print("TS-only mode: moved LLM backbone to CPU (lm_weight=0).")

        # Classification head: lightweight direct CE supervision on top of the retrieval embedding.
        # Provides strong class-level gradient even with tiny batches where InfoNCE is ineffective.
        self.cls_weight = float(run_config.get('cls_weight', 0.0))
        self.cls_head = None
        self.label_to_idx = {}
        if self.cls_weight > 0:
            embedding_dim = getattr(model, 'output_dim', None)
            if embedding_dim is None and hasattr(model, 'final_projection'):
                embedding_dim = int(model.final_projection.out_features)
            if embedding_dim is None:
                embedding_dim = int(getattr(model, 'llm_dim', 1024))

            all_labels = [str(s.get('label', '')) for s in getattr(train_dataset, 'samples', [])]
            unique_labels = sorted(set(l for l in all_labels if l != ''))
            if len(unique_labels) >= 2:
                self.label_to_idx = {l: i for i, l in enumerate(unique_labels)}
                self.cls_head = torch.nn.Sequential(
                    torch.nn.LayerNorm(embedding_dim),
                    torch.nn.Dropout(0.3),
                    torch.nn.Linear(embedding_dim, len(unique_labels))
                ).to(self.device)
                print(
                    f"Classification head enabled: {len(unique_labels)} classes {unique_labels}, "
                    f"weight={self.cls_weight}, embedding_dim={embedding_dim}"
                )
                # Only use val_loss for model selection when an eval split actually exists.
                # With eval_dataset=None (LOO-only mode), retrieval_metrics drives selection.
                self.cls_val_loss_selection = (eval_dataset is not None)

        # Optimize only parameters that require gradients (+ cls_head + log_temperature if present)
        base_lr = run_config.get('lr', 1e-4)
        weight_decay = float(run_config.get('weight_decay', 0.01))
        llm_lr_scale = float(run_config.get('llm_lr_scale', 1.0))

        if llm_lr_scale != 1.0 and hasattr(self.model, 'llm'):
            # Layer-wise LR decay: LLM LoRA params get a reduced LR to preserve pre-trained alignment.
            # When llm_lr_scale=0.0, also set weight_decay=0 to prevent AdamW from corrupting frozen weights.
            llm_param_ids = {id(p) for p in self.model.llm.parameters() if p.requires_grad}
            llm_params   = [p for p in self.model.parameters() if p.requires_grad and id(p) in llm_param_ids]
            other_params = [p for p in self.model.parameters() if p.requires_grad and id(p) not in llm_param_ids]
            if self.cls_head is not None:
                other_params += list(self.cls_head.parameters())
            if self.log_temperature is not None:
                other_params.append(self.log_temperature)
            llm_wd = 0.0 if llm_lr_scale == 0.0 else weight_decay
            param_groups = [
                {"params": other_params, "lr": base_lr,                 "weight_decay": weight_decay, "name": "ts_encoder+projector"},
                {"params": llm_params,   "lr": base_lr * llm_lr_scale,  "weight_decay": llm_wd,       "name": "llm_lora"},
            ]
            self.optimizer = torch.optim.AdamW(param_groups)
            print(f"Layer-wise LR: ts_encoder+projector lr={base_lr:.2e}  llm_lora lr={base_lr * llm_lr_scale:.2e}  llm_wd={llm_wd}")
        else:
            trainable_params = [p for p in self.model.parameters() if p.requires_grad]
            if self.cls_head is not None:
                trainable_params += list(self.cls_head.parameters())
            if self.log_temperature is not None:
                trainable_params.append(self.log_temperature)
            self.optimizer = torch.optim.AdamW(
                trainable_params,
                lr=base_lr,
                weight_decay=weight_decay,
            )

        self.num_epochs = run_config.get('epochs', 3)
        self.gradient_accumulation_steps = max(1, run_config.get('gradient_accumulation_steps', 1))
        self.max_grad_norm = float(run_config.get('max_grad_norm', 1.0))
        # LR scheduler: optional linear warmup → cosine annealing
        # base_lr already set above (used for optimizer param groups and eta_min)
        self.warmup_epochs = max(0, int(run_config.get('warmup_epochs', 0)))
        cosine_epochs = max(1, self.num_epochs - self.warmup_epochs)
        cosine_sched = torch.optim.lr_scheduler.CosineAnnealingLR(
            self.optimizer,
            T_max=cosine_epochs,
            eta_min=base_lr * 0.01,
        )
        if self.warmup_epochs > 0:
            warmup_sched = torch.optim.lr_scheduler.LinearLR(
                self.optimizer,
                start_factor=0.1,
                end_factor=1.0,
                total_iters=self.warmup_epochs,
            )
            self.scheduler = torch.optim.lr_scheduler.SequentialLR(
                self.optimizer,
                schedulers=[warmup_sched, cosine_sched],
                milestones=[self.warmup_epochs],
            )
            print(f"LR scheduler: {self.warmup_epochs}-epoch linear warmup → cosine annealing (eta_min={base_lr*0.01:.1e})")
        else:
            self.scheduler = cosine_sched
            print(f"LR scheduler: cosine annealing (eta_min={base_lr*0.01:.1e})")
        patience = run_config.get('early_stopping_patience', None)
        self.early_stopping_patience = patience if patience is None else max(1, patience)
        self.early_stopping_min_delta = float(run_config.get('early_stopping_min_delta', 0.0))
        self.save_dir = run_config.get('save_dir', 'checkpoints')
        os.makedirs(self.save_dir, exist_ok=True)
        self.history_path = os.path.join(self.save_dir, "training_history.json")
        self.run_config_path = os.path.join(self.save_dir, "run_config.json")
        self.history = []

        # EMA gallery encoder: tracks trainable params with slow EMA to stabilise gallery
        # embeddings while query embeddings update at full LR.  Only ~10M params tracked.
        # ema_momentum=0.0 (default) disables EMA for backward compatibility.
        self.ema_momentum = float(run_config.get('ema_momentum', 0.0))
        self.ema_state: dict = {}
        if self.ema_momentum > 0.0:
            self._init_ema()
            print(f"EMA gallery encoder enabled: momentum={self.ema_momentum:.4f} "
                  f"({len(self.ema_state)} param tensors tracked)")

        # ProtoNCE: EMA class prototypes updated from gallery embeddings each batch.
        # Pulls query toward prototype (stable class centroid) rather than noisy instances.
        self.proto_weight = float(run_config.get('proto_weight', 0.0))
        self.proto_momentum = float(run_config.get('proto_momentum', 0.9))
        self.class_prototypes: dict = {}  # {label_str: [H] float32 cpu tensor}
        if self.proto_weight > 0:
            print(f"ProtoNCE enabled: weight={self.proto_weight}, proto_momentum={self.proto_momentum}")

        # Forecasting (U+G unified training)
        self.forecast_weight = float(run_config.get('forecast_weight', 0.0))
        self.future_contrastive = bool(run_config.get('future_contrastive', False))
        self.future_sim_reg_weight = float(run_config.get('future_sim_reg_weight', 0.0))
        if self.forecast_weight > 0:
            print(f"Forecasting head enabled: weight={self.forecast_weight}, "
                  f"future_contrastive={self.future_contrastive}")
        if self.future_sim_reg_weight > 0:
            print(f"Future similarity regression enabled: weight={self.future_sim_reg_weight}")

        # Asymmetric Bi-Encoder: gallery encodes history+future, query encodes history only.
        self.asymmetric_biencoder = bool(run_config.get('asymmetric_biencoder', False))
        if self.asymmetric_biencoder:
            print("Asymmetric Bi-Encoder enabled: gallery=get_gallery_embedding(hist+fut), "
                  "query=get_ts_only_embedding(hist). InfoNCE with diagonal positives.")

        self.best_val_loss = float('inf')
        self.epochs_without_improvement = 0
        self.best_retrieval_accuracy = 0.0      # LOO on train (monitoring)
        self.best_valid_retrieval_accuracy = 0.0  # smoothed LOO used for model selection
        self.current_epoch = 0
        # Rolling window of raw LOO scores for smoothed model selection.
        # Model selection compares the window mean rather than single-epoch values,
        # preventing a one-off lucky epoch from triggering a premature best checkpoint.
        self.loo_smoothing_window = max(1, int(run_config.get('loo_smoothing_window', 3)))
        self._loo_window: list = []

        if self.is_main_process:
            with open(self.run_config_path, 'w', encoding='utf-8') as f:
                json.dump(self.run_config, f, indent=2)

        # DDP: wrap model AFTER optimizer and EMA are initialised so that:
        #   - optimizer holds references to the raw (non-DDP) parameters — still valid after wrap
        #   - EMA state uses param names without the 'module.' DDP prefix
        # _raw_model property strips the DDP wrapper for attribute access throughout training.
        if self.world_size > 1 and dist.is_initialized():
            from torch.nn.parallel import DistributedDataParallel as _DDP
            self.model = _DDP(
                self.model,
                device_ids=[self.local_rank],
                output_device=self.local_rank,
                find_unused_parameters=True,
            )
            print(f"[DDP] Model wrapped: rank={self.rank}/{self.world_size} (find_unused_parameters=True)")

        print(f"Training stage: {self.training_stage}")
        print(
            "Stage execution mode: single-stage per run "
            "(set training_stage to alignment or reasoning)."
        )
        print(
            f"Loss weights - Contrastive: {self.contrastive_weight}, LM: {self.lm_weight}"
        )
        print(f"Contrastive temperature: {self._temperature:.4f} ({'learnable' if self.log_temperature is not None else 'fixed'})")
        print(f"Alignment text mode: {self.alignment_text_mode}")
        print(f"Contrastive positive mode: {self.contrastive_positive_mode}")
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
        full_state = self._raw_model.state_dict()
        checkpoint_state = {}

        trainable_names = {name for name, param in self._raw_model.named_parameters() if param.requires_grad}
        for name in trainable_names:
            if name in full_state:
                checkpoint_state[name] = full_state[name].detach().cpu()

        # BatchNorm running stats are buffers, not parameters.
        for name, buffer in self._raw_model.named_buffers():
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
        if not self.is_main_process:
            return
        torch.save(self._get_checkpoint_state_dict(), path)
        if self.cls_head is not None:
            torch.save(self.cls_head.state_dict(), path + ".cls_head.pt")
        self._save_checkpoint_metadata(path, epoch=epoch, metrics=metrics, is_best=is_best)

    def _append_history(self, epoch_metrics):
        if not self.is_main_process:
            return
        self.history.append(epoch_metrics)
        self._save_json(self.history_path, self.history)

    def _gather_label_strings(self, class_labels):
        """Gather label strings across distributed workers to align with gathered embeddings."""
        if class_labels is None:
            return None

        local_labels = [str(label) for label in class_labels]
        if not dist.is_initialized():
            return local_labels

        gathered_labels = [None for _ in range(dist.get_world_size())]
        dist.all_gather_object(gathered_labels, local_labels)

        merged = []
        for item in gathered_labels:
            if item is None:
                continue
            merged.extend(str(label) for label in item)
        return merged

    def _set_train_mode(self):
        self.model.train()

    def _compute_softclt_weights(
        self,
        ts_query: torch.Tensor,
        query_labels,
        gallery_labels,
        ts_gallery: torch.Tensor = None,
        ts_gallery_bank=None,
    ) -> torch.Tensor:
        """Compute [Q, G] SoftCLT soft positive weight matrix.

        For same-class pairs: w_ij = exp(-DTW(ts_i, ts_j) / sigma), then row-normalised
        over same-class entries so weights sum to 1 per query.
        For different-class pairs: w_ij = 0 (hard negative — unchanged).

        ``ts_gallery`` covers in-batch gallery (same B samples as ts_query).
        ``ts_gallery_bank`` is a list of numpy arrays from the memory bank (may be None).
        """
        import numpy as np
        from scipy.spatial.distance import cdist

        Q = len(query_labels)
        G = len(gallery_labels)

        # Convert TS to flattened numpy float32 [samples, C*T]
        # Using all channels preserves spatial structure; channel-mean loses cross-channel info.
        def _to_flat(ts_tensor):
            arr = ts_tensor.detach().cpu().numpy().astype(np.float32)
            if arr.ndim == 3:
                arr = arr.reshape(arr.shape[0], -1)  # [B, C*T] — preserve all channels
            return arr  # [B, C*T]

        ts_q = _to_flat(ts_query)          # [Q, C*T]
        ts_g_batch = _to_flat(ts_gallery) if ts_gallery is not None else ts_q  # [B, C*T]

        # L2-normalise each row so distances lie in [0, 2] regardless of C*T dimensionality.
        # Without normalisation, distances scale as sqrt(C*T), making sigma non-transferable
        # across dataset shapes (e.g. NATOPS C*T=24*51=1224 gives dist~35 with sigma=1).
        def _l2norm(arr):
            norms = np.linalg.norm(arr, axis=1, keepdims=True).clip(min=1e-8)
            return arr / norms

        ts_q       = _l2norm(ts_q)
        ts_g_batch = _l2norm(ts_g_batch)

        # Build gallery TS array: in-batch rows + bank rows
        if ts_gallery_bank:
            bank_arrs = []
            for arr in ts_gallery_bank:
                if arr is None:
                    bank_arrs.append(np.zeros_like(ts_q[0]))
                elif arr.ndim == 2:
                    bank_arrs.append(arr.reshape(-1))  # [C*T] — flatten, consistent with _to_flat
                else:
                    bank_arrs.append(arr)
            ts_g_bank = np.stack(bank_arrs, axis=0)  # [N, C*T]
            ts_g_bank = _l2norm(ts_g_bank)
            ts_g_all = np.concatenate([ts_g_batch, ts_g_bank], axis=0)  # [G, C*T]
        else:
            ts_g_all = ts_g_batch  # [G, C*T]

        q_labs = [str(l) for l in query_labels]
        g_labs = [str(l) for l in gallery_labels]

        # Compute pairwise L2 distance matrix (proxy for DTW on equal-length signals)
        # Full DTW is O(T²) per pair; for T=51 and Q×G=18×108=1944 pairs this is fast,
        # but we use Euclidean distance on the raw sequence as a fast approximation.
        # Euclidean == DTW when sequences have the same length and no warping is needed,
        # which is true for fixed-length datasets like NATOPS.
        dist_matrix = cdist(ts_q, ts_g_all, metric='euclidean')  # [Q, G]

        # Build soft weight matrix
        weights = np.zeros((Q, G), dtype=np.float32)
        sigma = max(self.softclt_sigma, 1e-6)
        for i in range(Q):
            pos_cols = [j for j in range(G) if g_labs[j] == q_labs[i]]
            if not pos_cols:
                continue
            pos_dists = dist_matrix[i, pos_cols]
            pos_weights = np.exp(-pos_dists / sigma)
            total = pos_weights.sum()
            if total > 0:
                pos_weights /= total
            weights[i, pos_cols] = pos_weights

        return torch.from_numpy(weights)

    def _compute_future_softclt_weights(self, future_ts: torch.Tensor) -> torch.Tensor:
        """Compute in-batch [B, B] soft positive weights based on future trajectory distance.

        Unlike _compute_softclt_weights (which uses class labels as hard gates),
        this method treats ALL pairs as potential positives with weight proportional
        to future similarity.  The diagonal (self) pair naturally gets weight ~1.0
        after normalization.  Used for forecasting datasets without class labels.
        """
        import numpy as np
        from scipy.spatial.distance import cdist

        arr = future_ts.detach().cpu().numpy().astype(np.float32)
        B = arr.shape[0]
        flat = arr.reshape(B, -1)  # [B, C*H]
        norms = np.linalg.norm(flat, axis=1, keepdims=True).clip(min=1e-8)
        flat = flat / norms  # L2-normalise → distances in [0, 2]
        dist_matrix = cdist(flat, flat, metric='euclidean')  # [B, B]
        sigma = max(self.softclt_sigma, 1e-6)
        weights = np.exp(-dist_matrix / sigma).astype(np.float32)
        row_sums = weights.sum(axis=1, keepdims=True).clip(min=1e-8)
        weights = weights / row_sums  # row-normalised
        return torch.from_numpy(weights)

    def _compute_future_sim_regression_loss(
        self, emb: torch.Tensor, future_ts: torch.Tensor
    ) -> torch.Tensor:
        """Future Similarity Regression loss.

        Directly trains the embedding space so that pairwise cosine similarities
        match pairwise cosine similarities of the normalized future trajectories:

            L = ||S_emb - S_future||²_F / B²

        where S_emb[i,j]    = cos_sim(emb_i, emb_j)
              S_future[i,j] = cos_sim(L2_norm(future_i), L2_norm(future_j))

        This is more directly aligned with the RAF retrieval objective than InfoNCE:
        it trains a smooth, global ordering rather than a batch-local classification.
        No temperature, no sigma — the supervision is the future cosine similarity itself.
        """
        B = emb.shape[0]
        # Normalize futures: [B, C*H]
        flat = future_ts.float().reshape(B, -1)
        flat = F.normalize(flat, p=2, dim=1)
        S_future = torch.matmul(flat, flat.T)  # [B, B], values in [-1, 1]

        # Normalize embeddings
        emb_n = F.normalize(emb.float(), p=2, dim=1)
        S_emb = torch.matmul(emb_n, emb_n.T)  # [B, B], values in [-1, 1]

        return F.mse_loss(S_emb, S_future.detach())

    def _temp_tensor(self, device) -> torch.Tensor:
        """Return temperature as a scalar tensor. Gradients flow when learnable."""
        if self.log_temperature is not None:
            return self.log_temperature.clamp(min=-4.605, max=0.0).exp().to(device)
        return torch.tensor(self.contrastive_temperature, device=device)

    @property
    def _temperature(self) -> float:
        """Current contrastive temperature, clamped to [0.01, 1.0]."""
        if self.log_temperature is not None:
            return self.log_temperature.clamp(min=-4.605, max=0.0).exp().item()
        return self.contrastive_temperature

    @property
    def _raw_model(self):
        """Return the underlying model, unwrapping DDP if present.

        Use this for ALL direct attribute access (e.g. .ts_encoder, .llm, .state_dict())
        so the code is transparent to whether DDP is active.  Forward passes should
        always go through ``self.model`` so that DDP gradient synchronisation fires.
        """
        from torch.nn.parallel import DistributedDataParallel as _DDP
        return self.model.module if isinstance(self.model, _DDP) else self.model

    def _prepare_batch_tensors(self, batch):
        ts_input = batch['ts_input'].to(self.device).float()
        text_input_ids = batch['text_input_ids'].to(self.device)
        alignment_input_ids = batch.get('alignment_input_ids', batch['text_input_ids']).to(self.device)
        class_labels = batch.get('label', None)

        attention_mask = batch['attention_mask'].to(self.device) if batch.get('attention_mask') is not None else None
        labels = batch['labels'].to(self.device) if batch.get('labels') is not None else None

        retrieval_input_ids = batch.get('retrieval_input_ids')
        retrieval_attention_mask = batch.get('retrieval_attention_mask')
        if retrieval_input_ids is not None:
            retrieval_input_ids = retrieval_input_ids.to(self.device)
        if retrieval_attention_mask is not None:
            retrieval_attention_mask = retrieval_attention_mask.to(self.device)

        ts_input_view2 = batch.get('ts_input_view2')
        if ts_input_view2 is not None:
            ts_input_view2 = ts_input_view2.to(self.device).float()

        future_ts = batch.get('future_ts')
        if future_ts is not None:
            future_ts = future_ts.to(self.device).float()

        full_ts = batch.get('full_ts')
        if full_ts is not None:
            full_ts = full_ts.to(self.device).float()

        return {
            'ts_input': ts_input,
            'ts_input_view2': ts_input_view2,
            'text_input_ids': text_input_ids,
            'alignment_input_ids': alignment_input_ids,
            'class_labels': class_labels,
            'attention_mask': attention_mask,
            'labels': labels,
            'retrieval_input_ids': retrieval_input_ids,
            'retrieval_attention_mask': retrieval_attention_mask,
            'future_ts': future_ts,
            'full_ts': full_ts,
        }

    def _compute_batch_losses(self, tensors):
        ts_input = tensors['ts_input']
        ts_input_view2 = tensors.get('ts_input_view2')  # safe-aug gallery view (P12)
        text_input_ids = tensors['text_input_ids']
        alignment_input_ids = tensors['alignment_input_ids']
        class_labels = tensors['class_labels']
        attention_mask = tensors['attention_mask']
        labels = tensors['labels']
        retrieval_input_ids = tensors['retrieval_input_ids']
        retrieval_attention_mask = tensors['retrieval_attention_mask']
        future_ts = tensors.get('future_ts')  # [B, C, H] or None
        full_ts = tensors.get('full_ts')       # [B, C, T+H] or None (asymmetric bi-encoder)

        lm_loss = torch.tensor(0.0, device=self.device)
        contrastive_loss = torch.tensor(0.0, device=self.device)
        cls_loss = torch.tensor(0.0, device=self.device)
        proto_loss = torch.tensor(0.0, device=self.device)
        forecast_loss = torch.tensor(0.0, device=self.device)

        query_emb = None  # populated below when needed for contrastive or forecast loss
        future_sim_reg_loss = torch.tensor(0.0, device=self.device)

        if self.training_stage == 'alignment' and (self.contrastive_weight > 0 or self.cls_weight > 0 or self.forecast_weight > 0 or self.future_sim_reg_weight > 0):
            if self.asymmetric_biencoder and full_ts is not None:
                # Asymmetric Bi-Encoder: query encodes history only, gallery encodes full trajectory.
                # InfoNCE with diagonal positives: each history embedding is pulled toward
                # its own history+future embedding. At inference, gallery is built with full_ts.
                with torch.autocast(device_type='cuda', dtype=self.amp_dtype, enabled=self.use_amp):
                    query_emb = self._raw_model.get_ts_only_embedding(ts_input)    # [B, D]
                    gallery_emb = self._raw_model.get_gallery_embedding(full_ts)   # [B, D] — shared weights, grad flows both paths

                if self.contrastive_weight > 0:
                    q_n = F.normalize(query_emb.float(), p=2, dim=1)
                    g_n = F.normalize(gallery_emb.float(), p=2, dim=1)
                    sim = torch.matmul(q_n, g_n.T) / self._temp_tensor(query_emb.device)
                    n = min(sim.size(0), sim.size(1))
                    contrastive_loss = F.cross_entropy(sim[:n], torch.arange(n, device=sim.device))

                if self.forecast_weight > 0 and future_ts is not None:
                    fhead = getattr(self._raw_model, 'forecasting_head', None)
                    if fhead is not None:
                        pred = fhead(query_emb.to(fhead.linear.weight.dtype)).float()
                        forecast_loss = F.mse_loss(pred, future_ts.float()[:, :pred.shape[1], :])

            elif self.ts_only_embedding:
                with torch.autocast(device_type='cuda', dtype=self.amp_dtype, enabled=self.use_amp):
                    query_emb = self._raw_model.get_ts_only_embedding(ts_input)

                if self.contrastive_weight > 0:
                    if self.future_contrastive and future_ts is not None:
                        # Future-similarity soft InfoNCE for ts_only mode:
                        # soft labels = exp(-||fut_i - fut_j|| / sigma), row-normalised.
                        # This directly trains the encoder so that "similar future → similar embedding",
                        # which is the property needed for RAF to outperform raw cosine retrieval.
                        soft_weights = self._compute_future_softclt_weights(future_ts)
                        emb_n = F.normalize(query_emb.float(), p=2, dim=1)
                        sim = torch.matmul(emb_n, emb_n.T) / self._temp_tensor(query_emb.device)
                        contrastive_loss = self.compute_multi_positive_nce(
                            sim, soft_weights.to(query_emb.device)
                        )
                    else:
                        contrastive_loss = self.compute_contrastive_from_embeddings(query_emb, class_labels)
                if self.future_sim_reg_weight > 0 and future_ts is not None:
                    future_sim_reg_loss = self._compute_future_sim_regression_loss(
                        query_emb, future_ts.to(query_emb.device)
                    )
                if self.cls_weight > 0 and self.cls_head is not None:
                    cls_loss = self.compute_classification_loss(query_emb, class_labels)
            elif retrieval_input_ids is not None:
                # alignment_text_mode='full' → CoT-enriched query (asymmetric O1-Embedder)
                # otherwise → context-only query (symmetric; no CoT required at inference)
                if self.alignment_text_mode == 'full':
                    _q_ids, _q_mask = text_input_ids, attention_mask
                else:
                    _q_ids, _q_mask = retrieval_input_ids, retrieval_attention_mask
                with torch.autocast(device_type='cuda', dtype=self.amp_dtype, enabled=self.use_amp):
                    query_emb = self._raw_model.get_embedding_for_training(
                        ts_input, _q_ids, attention_mask=_q_mask
                    )
                with torch.no_grad():
                    with self._ema_gallery_context():
                        with torch.autocast(device_type='cuda', dtype=self.amp_dtype, enabled=self.use_amp):
                            # P12: when view2 is available, use it as gallery input so that
                            # gallery and query come from independently augmented views of the
                            # same sample (dual-view contrastive, TS-TCC / SoftCLT style).
                            gallery_ts = ts_input_view2 if ts_input_view2 is not None else ts_input
                            gallery_emb = self._raw_model.get_embedding(
                                gallery_ts, retrieval_input_ids, attention_mask=retrieval_attention_mask
                            )
                gallery_emb = gallery_emb.detach()

                if self.contrastive_weight > 0:
                    if class_labels is None:
                        if self.future_contrastive and future_ts is not None:
                            # Future-similarity soft InfoNCE (U+G unified):
                            # soft labels from future trajectory distance — no class labels needed.
                            soft_weights = self._compute_future_softclt_weights(future_ts)
                            q_n = F.normalize(query_emb.float(), p=2, dim=1)
                            g_n = F.normalize(gallery_emb.detach().float(), p=2, dim=1)
                            sim = torch.matmul(q_n, g_n.T) / self._temp_tensor(query_emb.device)
                            wmat = soft_weights.to(query_emb.device)
                            valid = wmat.sum(dim=1) > 0
                            if valid.any():
                                contrastive_loss = self.compute_multi_positive_nce(
                                    sim[valid], wmat[valid]
                                )
                        else:
                            # Diagonal InfoNCE: each query's positive is its own gallery entry
                            q_n = F.normalize(query_emb.float(), p=2, dim=1)
                            g_n = F.normalize(gallery_emb.detach().float(), p=2, dim=1)
                            sim = torch.matmul(q_n, g_n.T) / self._temperature
                            n = min(sim.size(0), sim.size(1))
                            contrastive_loss = F.cross_entropy(sim[:n], torch.arange(n, device=sim.device))
                    else:
                        # Get memory bank BEFORE updating: avoids current-batch duplicates.
                        # Skip bank during warmup: early embeddings are random and add noise.
                        mb_embs, mb_labels = self.memory_bank.get(self.device)
                        use_bank = (
                            mb_embs is not None
                            and len(mb_embs) > 0
                            and self.current_epoch >= self.warmup_epochs
                        )
                        if use_bank:
                            all_gallery_emb = torch.cat([gallery_emb, mb_embs.to(gallery_emb.dtype)], dim=0)
                            all_gallery_labels = list(class_labels) + list(mb_labels)
                        else:
                            all_gallery_emb = gallery_emb
                            all_gallery_labels = class_labels
                        soft_weights = None
                        if self.softclt_enabled:
                            ts_gallery_bank = self.memory_bank.get_ts() if use_bank else None
                            # P12: SoftCLT uses view2 as gallery TS to compute cross-view distances
                            ts_gal_input = ts_input_view2 if ts_input_view2 is not None else ts_input
                            soft_weights = self._compute_softclt_weights(
                                ts_input, class_labels, all_gallery_labels,
                                ts_gallery=ts_gal_input,
                                ts_gallery_bank=ts_gallery_bank,
                            )
                        contrastive_loss = self.compute_asymmetric_infonce(
                            query_emb, all_gallery_emb, class_labels, all_gallery_labels,
                            soft_weights=soft_weights,
                            add_self_positive=(
                                self.contrastive_positive_mode in ('augmentation', 'hybrid')
                                and ts_input_view2 is not None
                            ),
                        )

                # Update memory bank AFTER loss (eliminates current-batch duplication)
                if self.memory_bank_size > 0 and class_labels is not None:
                    ts_for_bank = ts_input if self.softclt_enabled else None
                    self.memory_bank.update(gallery_emb, class_labels, ts=ts_for_bank)

                # ProtoNCE: update EMA prototypes from gallery, then compute loss on query
                if self.proto_weight > 0 and class_labels is not None:
                    self._update_prototypes(gallery_emb, class_labels)
                    proto_loss = self.compute_prototype_nce_loss(query_emb, class_labels)

                if self.cls_weight > 0 and self.cls_head is not None:
                    cls_loss = self.compute_classification_loss(query_emb, class_labels)

        if self.lm_weight > 0 and labels is not None:
            with torch.autocast(device_type='cuda', dtype=self.amp_dtype, enabled=self.use_amp):
                outputs = self.model(
                    ts_input=ts_input,
                    text_input_ids=text_input_ids,
                    attention_mask=attention_mask,
                    labels=labels,
                )
            lm_loss = outputs.loss if outputs.loss is not None else torch.tensor(0.0, device=self.device)

        # Forecasting loss: query embedding → predict future (U+G generation objective)
        if self.forecast_weight > 0 and future_ts is not None:
            fhead = getattr(self._raw_model, 'forecasting_head', None)
            if fhead is not None and query_emb is not None:
                pred = fhead(query_emb.to(fhead.linear.weight.dtype))  # [B, C, H]
                target = future_ts.to(pred.device).to(pred.dtype)      # [B, C, H]
                forecast_loss = torch.nn.functional.mse_loss(pred, target)

        loss = self._combine_weighted_losses(
            lm_loss,
            contrastive_loss,
            cls_loss,
            proto_loss,
            forecast_loss,
            future_sim_reg_loss,
        )

        return {
            'loss': loss,
            'lm_loss': lm_loss,
            'contrastive_loss': contrastive_loss,
            'cls_loss': cls_loss,
            'proto_loss': proto_loss,
            'forecast_loss': forecast_loss,
            'future_sim_reg_loss': future_sim_reg_loss,
        }

    def _build_progress_postfix(self, losses):
        postfix = {'loss': f"{losses['loss'].item():.4f}"}
        if self.training_stage == 'alignment':
            postfix['closs'] = f"{losses['contrastive_loss'].item():.4f}"
            if self.log_temperature is not None:
                postfix['τ'] = f"{self._temperature:.3f}"
            if self.cls_weight > 0:
                postfix['csloss'] = f"{losses['cls_loss'].item():.4f}"
            if self.proto_weight > 0:
                postfix['ploss'] = f"{losses['proto_loss'].item():.4f}"
            if self.forecast_weight > 0:
                postfix['floss'] = f"{losses['forecast_loss'].item():.4f}"
            if self.future_sim_reg_weight > 0:
                postfix['srloss'] = f"{losses['future_sim_reg_loss'].item():.4f}"
        postfix['lloss'] = f"{losses['lm_loss'].item():.4f}"
        return postfix

    def _combine_weighted_losses(self, lm_loss, contrastive_loss, cls_loss, proto_loss=None, forecast_loss=None, future_sim_reg_loss=None):
        if proto_loss is None:
            proto_loss = torch.tensor(0.0, device=self.device)
        if forecast_loss is None:
            forecast_loss = torch.tensor(0.0, device=self.device)
        if future_sim_reg_loss is None:
            future_sim_reg_loss = torch.tensor(0.0, device=self.device)
        return (
            self.contrastive_weight * contrastive_loss
            + self.cls_weight * cls_loss
            + self.lm_weight * lm_loss
            + self.proto_weight * proto_loss
            + self.forecast_weight * forecast_loss
            + self.future_sim_reg_weight * future_sim_reg_loss
        )

    def _compute_eval_batch_loss(self, tensors):
        ts_input = tensors['ts_input']
        text_input_ids = tensors['text_input_ids']
        alignment_input_ids = tensors['alignment_input_ids']
        class_labels = tensors['class_labels']
        attention_mask = tensors['attention_mask']
        labels = tensors['labels']
        retrieval_input_ids = tensors['retrieval_input_ids']
        retrieval_attention_mask = tensors['retrieval_attention_mask']

        if self.ts_only_embedding or self.cls_val_loss_selection:
            if self.ts_only_embedding:
                query_emb = self._raw_model.get_ts_only_embedding(ts_input)
            elif retrieval_input_ids is not None:
                query_emb = self._raw_model.get_embedding(
                    ts_input,
                    retrieval_input_ids,
                    attention_mask=retrieval_attention_mask,
                )
            else:
                query_emb = self._raw_model.get_embedding(ts_input, alignment_input_ids)
            # For forecasting datasets: return forecast MSE as val loss
            future_ts = tensors.get('future_ts')
            fhead = getattr(self._raw_model, 'forecasting_head', None)
            if fhead is not None and future_ts is not None:
                pred = fhead(query_emb.to(fhead.linear.weight.dtype))
                target = future_ts.to(pred.device).to(pred.dtype)
                return F.mse_loss(pred, target)
            return self.compute_classification_loss(query_emb, class_labels)

        lm_loss = torch.tensor(0.0, device=self.device)
        contrastive_loss = torch.tensor(0.0, device=self.device)
        cls_loss = torch.tensor(0.0, device=self.device)

        if self.lm_weight > 0 and labels is not None:
            outputs = self.model(
                ts_input=ts_input,
                text_input_ids=text_input_ids,
                attention_mask=attention_mask,
                labels=labels,
            )
            lm_loss = outputs.loss if outputs.loss is not None else lm_loss

        if self.training_stage == 'alignment':
            query_embs = self._raw_model.get_embedding(
                ts_input, text_input_ids, attention_mask=attention_mask
            )
            gallery_embs = self._raw_model.get_embedding(
                ts_input,
                retrieval_input_ids if retrieval_input_ids is not None else alignment_input_ids,
                attention_mask=retrieval_attention_mask if retrieval_input_ids is not None else None,
            )
            contrastive_loss = self.compute_asymmetric_infonce(
                query_embs, gallery_embs.detach(), class_labels, class_labels
            )
            return self._combine_weighted_losses(lm_loss, contrastive_loss, cls_loss)

        return lm_loss

    def compute_multi_positive_nce(self, logits, positive_mask):
        """Supervised contrastive loss — SupCon Form 2.

        Uses log(sum_p exp(s_p/τ) / sum_all exp(s_k/τ)) / |P| per query,
        which groups all positives into a single log term.  This gives
        stronger gradients toward hard positives compared to Form 1
        (which averages individual per-positive log-probs independently).

        When ``positive_mask`` contains float values in [0, 1] (SoftCLT mode), the loss
        becomes the soft cross-entropy H(w, softmax(s/τ)) = log_Z - sum_j w_j * s_j/τ,
        where w_j is the normalised positive weight for pair (i, j).

        Reference: Khosla et al. "Supervised Contrastive Learning" NeurIPS 2020;
                   Lee et al. "SoftCLT" ICLR 2024.
        """
        if logits.size(0) == 0:
            return torch.tensor(0.0, device=logits.device)

        # SoftCLT path: float weights in [0, 1], already row-normalised over positives
        if positive_mask.dtype != torch.bool and positive_mask.is_floating_point():
            weights = positive_mask.to(logits.dtype)  # [Q, G]
            valid_rows = weights.sum(dim=1) > 0
            if not valid_rows.any():
                return torch.tensor(0.0, device=logits.device)
            logits = logits[valid_rows]
            weights = weights[valid_rows]
            logits = logits - logits.max(dim=1, keepdim=True).values
            log_denom = torch.logsumexp(logits, dim=1)           # [Q']
            weighted_logit = (weights * logits).sum(dim=1)        # [Q']  (sum_j w_j * s_j/τ)
            loss = -(weighted_logit - log_denom)
            return loss.mean()

        # Hard label path (original SupCon Form 2)
        positive_mask = positive_mask.bool()
        positive_counts = positive_mask.float().sum(dim=1)
        valid_rows = positive_counts > 0
        if not valid_rows.any():
            return torch.tensor(0.0, device=logits.device)

        logits = logits[valid_rows]
        positive_mask = positive_mask[valid_rows]
        positive_counts = positive_counts[valid_rows]

        # Numerical stability: subtract row max before exp
        logits = logits - logits.max(dim=1, keepdim=True).values

        # log(sum_all exp(s_k/τ))  — denominator
        log_denom = torch.logsumexp(logits, dim=1)  # [Q']

        # log(sum_p exp(s_p/τ))  — numerator; mask non-positives to -inf
        logits_pos = logits.masked_fill(~positive_mask, -1e9)
        log_pos_sum = torch.logsumexp(logits_pos, dim=1)  # [Q']

        # Normalize by |P|: -[log(sum_p exp) - log(|P|) - log(sum_all exp)]
        loss = -(log_pos_sum - positive_counts.clamp_min(1.0).log() - log_denom)
        return loss.mean()

    def compute_contrastive_from_embeddings(self, embeddings, class_labels=None):
        """
        TS-only supervised contrastive loss using full LLM-processed embeddings.
        Directly optimizes the embedding space used at retrieval time.
        embeddings: [Batch, H] already-pooled embeddings from forward()
        """
        # TS-only ablation prioritizes stability; avoid custom distributed gather in this path.
        if not self.ts_only_embedding:
            embeddings = DistributedGatherLayer.apply(embeddings)
        
        batch_size = embeddings.size(0)
        if batch_size < 2:
            return torch.tensor(0.0, device=embeddings.device)

        label_strings = self._gather_label_strings(class_labels) if class_labels is not None else None
            
        emb_norm = F.normalize(embeddings, p=2, dim=1)
        sim_matrix = torch.matmul(emb_norm, emb_norm.T) / self._temp_tensor(embeddings.device)

        if self.contrastive_positive_mode == 'label' and label_strings is not None:
            if len(label_strings) != batch_size:
                labels = torch.arange(batch_size, device=embeddings.device)
                return F.cross_entropy(sim_matrix, labels)
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

    def compute_asymmetric_infonce(self, query_embs, gallery_embs, query_labels, gallery_labels, soft_weights=None, add_self_positive=False):
        """O1-Embedder asymmetric InfoNCE.

        Pulls thought-enriched query embeddings toward same-class context-only
        gallery embeddings.  Gradient flows only through query_embs; gallery_embs
        are detached.  In DDP mode, embeddings from all ranks are gathered so
        every rank sees the full global batch of negatives.

        Args:
            query_embs:    [Q, H]  thought-enriched, requires grad
            gallery_embs:  [G, H]  context-only, detached
            query_labels:  list of Q label strings
            gallery_labels: list of G label strings
            soft_weights:  optional [Q, G] float tensor (SoftCLT row-normalised weights).
                           When provided and DDP is not active, replaces the hard positive
                           mask with DTW-based soft labels.  Ignored in DDP training mode
                           (soft weights are not gathered; hard labels are used instead).
        """
        Q = query_embs.size(0)
        G = gallery_embs.size(0)
        if Q == 0 or G == 0:
            return torch.tensor(0.0, device=query_embs.device)

        # DDP gather: active during training forward passes.
        # Soft weights are computed for local [Q, G] only and cannot be trivially
        # gathered across ranks, so we fall back to hard labels in DDP mode.
        ddp_active = dist.is_initialized() and torch.is_grad_enabled()
        if ddp_active:
            query_embs_global = DistributedGatherLayer.apply(query_embs)
            gallery_embs_global = DistributedGatherLayer.apply(gallery_embs.detach())
            q_labs = self._gather_label_strings(query_labels)
            g_labs = self._gather_label_strings(gallery_labels)
        else:
            query_embs_global = query_embs
            gallery_embs_global = gallery_embs
            q_labs = [str(l) for l in query_labels]
            g_labs = [str(l) for l in gallery_labels]

        if q_labs is None:
            q_labs = [str(l) for l in query_labels] * (query_embs_global.size(0) // Q or 1)
        if g_labs is None:
            g_labs = [str(l) for l in gallery_labels] * (gallery_embs_global.size(0) // G or 1)

        # fp32 for stable cosine similarity
        q_norm = F.normalize(query_embs_global.float(), p=2, dim=1)
        g_norm = F.normalize(gallery_embs_global.float(), p=2, dim=1)

        sim_matrix = torch.matmul(q_norm, g_norm.T) / self._temp_tensor(query_embs.device)

        # SoftCLT path: use DTW-based soft positive weights (single-GPU only).
        if soft_weights is not None and not ddp_active:
            weight_matrix = soft_weights.to(query_embs.device)
            # P12: inject diagonal self-positive into SoftCLT weight matrix.
            # The dual-aug view of sample i is the strongest positive; set its weight to 1.0
            # then re-normalise so all positive weights in each row sum to 1.
            if add_self_positive:
                Q_local = query_embs_global.size(0)
                G_local = gallery_embs_global.size(0)
                diag_size = min(Q_local, G_local)
                diag_idx = torch.arange(diag_size, device=weight_matrix.device)
                weight_matrix = weight_matrix.clone()
                weight_matrix[diag_idx, diag_idx] = 1.0
                row_sum = weight_matrix.sum(dim=1, keepdim=True).clamp(min=1e-8)
                weight_matrix = weight_matrix / row_sum
            valid_rows = weight_matrix.sum(dim=1) > 0
            if not valid_rows.any():
                return torch.tensor(0.0, device=query_embs.device)
            loss = self.compute_multi_positive_nce(sim_matrix[valid_rows], weight_matrix[valid_rows])
            return loss

        # Hard label path (default; also used in DDP mode).
        positive_mask = torch.tensor(
            [[ql == gl for gl in g_labs] for ql in q_labs],
            device=query_embs.device, dtype=torch.bool
        )

        # P12: add diagonal self-positives (query_i aligns with gallery_i, its dual-aug view).
        # Only applies to the in-batch portion of the gallery (first Q entries).
        if add_self_positive:
            Q_local = query_embs_global.size(0)
            G_local = gallery_embs_global.size(0)
            diag_size = min(Q_local, G_local)
            diag_idx = torch.arange(diag_size, device=positive_mask.device)
            positive_mask[diag_idx, diag_idx] = True

        valid_rows = positive_mask.any(dim=1)
        if not valid_rows.any():
            return torch.tensor(0.0, device=query_embs.device)

        loss = self.compute_multi_positive_nce(
            sim_matrix[valid_rows], positive_mask[valid_rows]
        )

        # DDP averages gradients across ranks (÷ world_size), so scale up to preserve
        # the same effective gradient magnitude as single-GPU training.
        # Only applies during training (grad_enabled); eval loss is local-only.
        if ddp_active:
            loss = loss * dist.get_world_size()

        return loss

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
                tensors = self._prepare_batch_tensors(batch)
                with torch.autocast(device_type='cuda', dtype=self.amp_dtype, enabled=self.use_amp):
                    batch_loss = self._compute_eval_batch_loss(tensors)

                total_loss += batch_loss.item()
                steps += 1
                
        avg_loss = total_loss / steps if steps > 0 else 0.0
        self._set_train_mode()
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
                self._raw_model,
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
        self._set_train_mode()
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
                self._raw_model,
                self.train_dataset,
                self.collator,
                self.device,
                dtw_window_size=self.retrieval_eval_dtw_window_size,
                fast_dtw_max_len=self.retrieval_eval_fast_dtw_max_len,
                use_full_prompt=False,
                ts_only_embedding=self.ts_only_embedding,
            )
            query_embeddings, query_labels, query_ts_list = embed_queries(
                self._raw_model,
                self.eval_dataset,
                self.collator,
                self.device,
                ts_only_embedding=self.ts_only_embedding,
                use_full_prompt=(self.alignment_text_mode == "full"),
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
        self._set_train_mode()
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
                self._raw_model,
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
                self._raw_model,
                report_dataset,
                self.collator,
                self.device,
                ts_only_embedding=self.ts_only_embedding,
                use_full_prompt=(self.alignment_text_mode == "full"),
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
        self._set_train_mode()
        return metrics

    def _log_epoch_summary(self, epoch, averages, val_loss=None):
        train_loss = averages['train_loss']
        avg_contrastive_loss = averages['train_contrastive_loss']
        avg_cls_loss = averages['train_cls_loss']
        avg_lm_loss = averages['train_lm_loss']
        avg_proto_loss = averages['train_proto_loss']

        avg_forecast_loss = averages.get('train_forecast_loss', 0.0)
        avg_future_sim_reg_loss = averages.get('train_future_sim_reg_loss', 0.0)
        if self.training_stage == 'alignment':
            proto_str = f", PL: {avg_proto_loss:.4f}" if self.proto_weight > 0 else ""
            forecast_str = f", FL: {avg_forecast_loss:.4f}" if self.forecast_weight > 0 else ""
            sim_reg_str = f", SR: {avg_future_sim_reg_loss:.4f}" if self.future_sim_reg_weight > 0 else ""
            train_msg = (
                f"Epoch {epoch+1} finished. Train Loss: {train_loss:.4f} "
                f"(CL: {avg_contrastive_loss:.4f}, CS: {avg_cls_loss:.4f}, "
                f"LM: {avg_lm_loss:.4f}{proto_str}{forecast_str}{sim_reg_str})"
            )
        else:
            train_msg = f"Epoch {epoch+1} finished. Train Loss: {train_loss:.4f}"

        if val_loss is None:
            print(train_msg)
        else:
            print(f"{train_msg} | Val Loss: {val_loss:.4f}")

    def _build_metrics_summary(self, averages, val_loss=None, retrieval_metrics=None, valid_retrieval_metrics=None):
        metrics_summary = dict(averages)
        if val_loss is not None:
            metrics_summary['val_loss'] = float(val_loss)
        if retrieval_metrics is not None:
            metrics_summary['retrieval_accuracy'] = float(retrieval_metrics['accuracy'])
            metrics_summary['retrieval_macro_f1'] = float(retrieval_metrics['macro_f1'])
        if valid_retrieval_metrics is not None:
            metrics_summary['valid_retrieval_accuracy'] = float(valid_retrieval_metrics['accuracy'])
            metrics_summary['valid_retrieval_macro_f1'] = float(valid_retrieval_metrics['macro_f1'])
        return metrics_summary

    def _run_epoch_validation(self, epoch, averages):
        """Run validation and retrieval evaluation for one epoch if eval loader exists."""
        val_loss = None
        retrieval_metrics = None
        valid_retrieval_metrics = None

        if self.eval_loader:
            torch.cuda.empty_cache()
            val_loss = self.evaluate()

        # Retrieval eval is compute-intensive; only rank 0 runs it.
        # LOO on train runs regardless of eval_loader (no internal-val split needed).
        if self.retrieval_eval_enabled and (epoch + 1) % self.retrieval_eval_interval == 0:
            if self.is_main_process:
                torch.cuda.empty_cache()
                retrieval_metrics = self.evaluate_retrieval_metrics()       # LOO on train
                if self.eval_loader:
                    torch.cuda.empty_cache()
                    valid_retrieval_metrics = self.evaluate_valid_retrieval_metrics()  # valid set
            if dist.is_initialized():
                dist.barrier()  # sync all ranks after rank-0-only retrieval eval

        self._log_epoch_summary(epoch, averages, val_loss=val_loss)

        if retrieval_metrics is not None:
            print(
                f"Train LOO Retrieval - Accuracy: {retrieval_metrics['accuracy']:.2%} | "
                f"Macro-F1: {retrieval_metrics['macro_f1']:.2%}"
            )

        if valid_retrieval_metrics is not None:
            print(
                f"Valid Retrieval  - Accuracy: {valid_retrieval_metrics['accuracy']:.2%} | "
                f"Macro-F1: {valid_retrieval_metrics['macro_f1']:.2%}"
            )

        return val_loss, retrieval_metrics, valid_retrieval_metrics

    # ------------------------------------------------------------------
    # ProtoNCE prototype contrastive helpers
    # ------------------------------------------------------------------

    @torch.no_grad()
    def _update_prototypes(self, gallery_embs: torch.Tensor, gallery_labels):
        """EMA update class prototypes from gallery embeddings (detached, CPU).

        For each class present in gallery_labels, compute the batch centroid of
        normalised gallery embeddings and blend into the running prototype via EMA.
        Prototypes are stored as normalised float32 CPU tensors.
        """
        if not gallery_labels:
            return
        g_norm = F.normalize(gallery_embs.detach().float().cpu(), p=2, dim=1)  # [G, H]
        label_strs = [str(l) for l in gallery_labels]
        unique_labels = set(label_strs)
        m = self.proto_momentum
        for lbl in unique_labels:
            indices = [i for i, l in enumerate(label_strs) if l == lbl]
            batch_centroid = g_norm[indices].mean(dim=0)  # [H]
            batch_centroid = F.normalize(batch_centroid, p=2, dim=0)
            if lbl not in self.class_prototypes:
                self.class_prototypes[lbl] = batch_centroid.clone()
            else:
                updated = m * self.class_prototypes[lbl] + (1.0 - m) * batch_centroid
                self.class_prototypes[lbl] = F.normalize(updated, p=2, dim=0)

    def compute_prototype_nce_loss(self, query_embs: torch.Tensor, query_labels) -> torch.Tensor:
        """Prototype-NCE: pull each query toward its class prototype.

        Each class prototype is the positive; all other class prototypes are
        negatives.  This gives stable gradients even with very small batches
        (as few as 1 sample per class) because the prototype is a class centroid
        rather than a single noisy instance.

        Gradient flows only through query_embs; prototypes are fixed tensors.
        """
        if not self.class_prototypes or not query_labels:
            return torch.tensor(0.0, device=self.device)

        proto_labels = list(self.class_prototypes.keys())
        if len(proto_labels) < 2:
            return torch.tensor(0.0, device=self.device)

        # Stack prototypes: [C, H] — detached, no grad; fp32 for stable cosine similarity
        proto_matrix = torch.stack(
            [self.class_prototypes[l] for l in proto_labels]
        ).to(self.device).float()  # [C, H]
        proto_matrix = F.normalize(proto_matrix, p=2, dim=1)

        q_norm = F.normalize(query_embs.float(), p=2, dim=1)  # [Q, H]
        sim = torch.matmul(q_norm, proto_matrix.T) / self._temp_tensor(query_embs.device)  # [Q, C]

        label_to_idx = {l: i for i, l in enumerate(proto_labels)}
        q_labels = [str(l) for l in query_labels]
        valid_rows = [i for i, ql in enumerate(q_labels) if ql in label_to_idx]
        if not valid_rows:
            return torch.tensor(0.0, device=self.device)

        targets = torch.tensor(
            [label_to_idx[q_labels[i]] for i in valid_rows],
            dtype=torch.long, device=self.device
        )
        return F.cross_entropy(sim[valid_rows], targets)

    # ------------------------------------------------------------------
    # EMA gallery encoder helpers
    # ------------------------------------------------------------------

    def _init_ema(self):
        """Copy all trainable parameters into self.ema_state (on the same device)."""
        self.ema_state = {
            name: param.data.clone().detach()
            for name, param in self._raw_model.named_parameters()
            if param.requires_grad
        }

    @torch.no_grad()
    def _update_ema(self):
        """Exponential moving average update: ema = m*ema + (1-m)*current."""
        if not self.ema_state:
            return
        m = self.ema_momentum
        for name, param in self._raw_model.named_parameters():
            if param.requires_grad and name in self.ema_state:
                self.ema_state[name].mul_(m).add_(param.data, alpha=1.0 - m)

    @contextmanager
    def _ema_gallery_context(self):
        """Context manager: swap to EMA parameters for gallery forward pass, then restore.

        Inside the ``with`` block the model's trainable parameters are replaced by
        their EMA copies.  On exit the original (gradient-tracking) parameters are
        restored, so no gradient is affected.
        """
        if not self.ema_state:
            yield
            return
        # Save live params, load EMA params
        saved = {}
        with torch.no_grad():
            for name, param in self._raw_model.named_parameters():
                if param.requires_grad and name in self.ema_state:
                    saved[name] = param.data.clone()
                    param.data.copy_(self.ema_state[name])
        try:
            yield
        finally:
            # Restore live params
            with torch.no_grad():
                for name, param in self._raw_model.named_parameters():
                    if param.requires_grad and name in saved:
                        param.data.copy_(saved[name])

    def _update_model_selection(self, val_loss, retrieval_metrics, valid_retrieval_metrics):
        """Return (should_save, reset_patience).

        should_save   = True when current metric >= best  (ties overwrite earlier/unstable checkpoint)
        reset_patience = True only when current metric > best + min_delta  (meaningful improvement)
        """
        should_save = False
        reset_patience = False

        if self.cls_val_loss_selection:
            if retrieval_metrics is not None:
                current_train_loo = float(retrieval_metrics['accuracy'])
                self.best_retrieval_accuracy = max(self.best_retrieval_accuracy, current_train_loo)
            if valid_retrieval_metrics is not None:
                current_acc = float(valid_retrieval_metrics['accuracy'])
                should_save = current_acc >= self.best_valid_retrieval_accuracy
                reset_patience = current_acc > self.best_valid_retrieval_accuracy + self.early_stopping_min_delta
                if should_save:
                    self.best_valid_retrieval_accuracy = current_acc
                if val_loss is not None and val_loss < self.best_val_loss:
                    self.best_val_loss = val_loss
            elif val_loss is not None:
                reset_patience = val_loss < (self.best_val_loss - self.early_stopping_min_delta)
                should_save = val_loss < self.best_val_loss
                if should_save:
                    self.best_val_loss = val_loss
                    reset_patience = reset_patience or (self.best_valid_retrieval_accuracy == 0.0)
                    should_save = reset_patience  # keep original: save only on real loss improvement
        elif self.retrieval_eval_enabled and retrieval_metrics is not None:
            current_train_loo = float(retrieval_metrics['accuracy'])
            self.best_retrieval_accuracy = max(self.best_retrieval_accuracy, current_train_loo)
            if val_loss is not None and val_loss < self.best_val_loss:
                self.best_val_loss = val_loss
            if valid_retrieval_metrics is not None:
                current_acc = float(valid_retrieval_metrics['accuracy'])
                should_save = current_acc >= self.best_valid_retrieval_accuracy
                reset_patience = current_acc > self.best_valid_retrieval_accuracy + self.early_stopping_min_delta
                if should_save:
                    self.best_valid_retrieval_accuracy = current_acc
            else:
                # No validation set: use rolling-mean LOO accuracy for model selection.
                # A single lucky epoch (e.g., ep21 spike) won't trigger a save unless
                # the mean of the last K epochs also exceeds the best mean so far.
                self._loo_window.append(current_train_loo)
                if len(self._loo_window) > self.loo_smoothing_window:
                    self._loo_window.pop(0)
                smoothed_loo = sum(self._loo_window) / len(self._loo_window)
                should_save = smoothed_loo >= self.best_valid_retrieval_accuracy
                reset_patience = smoothed_loo > self.best_valid_retrieval_accuracy + self.early_stopping_min_delta
                if should_save:
                    self.best_valid_retrieval_accuracy = smoothed_loo
        else:
            if val_loss is not None:
                should_save = val_loss < self.best_val_loss
                reset_patience = val_loss < (self.best_val_loss - self.early_stopping_min_delta)
                if should_save:
                    self.best_val_loss = val_loss

        return should_save, reset_patience

    def _maybe_save_best_and_check_early_stop(
        self,
        epoch,
        val_loss,
        retrieval_metrics,
        valid_retrieval_metrics,
        metrics_summary,
    ):
        should_save, reset_patience = self._update_model_selection(val_loss, retrieval_metrics, valid_retrieval_metrics)
        if should_save:
            best_path = os.path.join(self.save_dir, 'model_best.pt')
            self._save_checkpoint(best_path, epoch=epoch + 1, metrics=metrics_summary, is_best=True)
            print(f"New Best Model saved to {best_path}")

        if reset_patience:
            self.epochs_without_improvement = 0
            return False

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
                return True
        return False

    def _run_post_training_evaluations(self):
        """Run final retrieval test and optional ARM phase after training loop."""
        # Always evaluate using the best checkpoint, not the last-epoch model.
        best_ckpt = os.path.join(self.save_dir, "model_best.pt")
        if os.path.exists(best_ckpt):
            print(f"\n[eval] Reloading best checkpoint for final evaluation: {best_ckpt}")
            state = torch.load(best_ckpt, map_location=self.device)
            self._raw_model.load_state_dict(state, strict=False)
            print("[eval] Best checkpoint loaded.")
        else:
            print("[eval] No model_best.pt found; using last-epoch model for final evaluation.")

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

    def _finalize_epoch(
        self,
        epoch,
        averages,
        val_loss,
        retrieval_metrics,
        valid_retrieval_metrics,
    ):
        """Persist epoch artifacts and return whether training should stop early."""
        metrics_summary = self._build_metrics_summary(
            averages,
            val_loss=val_loss,
            retrieval_metrics=retrieval_metrics,
            valid_retrieval_metrics=valid_retrieval_metrics,
        )

        self._append_history({
            "epoch": epoch + 1,
            **metrics_summary,
        })

        last_path = os.path.join(self.save_dir, "model_last.pt")
        self._save_checkpoint(last_path, epoch=epoch + 1, metrics=metrics_summary, is_best=False)
        print(f"Saved checkpoint to {last_path}")

        if val_loss is None and retrieval_metrics is None:
            return False

        return self._maybe_save_best_and_check_early_stop(
            epoch,
            val_loss,
            retrieval_metrics,
            valid_retrieval_metrics,
            metrics_summary,
        )

    def train(self):
        print(f"Starting training loop in stage: {self.training_stage}...")
        self._set_train_mode()

        self.optimizer.zero_grad()
        self.current_epoch = 0
        for epoch in range(self.num_epochs):
            self.current_epoch = epoch

            # Flush memory bank at warmup→training transition.
            # Warmup embeddings are low-quality; letting them enter InfoNCE causes the
            # "loss spike" seen in all versions.  A fresh bank avoids stale negatives.
            if epoch == self.warmup_epochs and self.memory_bank_size > 0:
                self.memory_bank.reset()
                print(f"[Epoch {epoch+1}] Memory bank flushed at warmup→training transition.")

            epoch_losses = EpochLossTracker()
            # Use tqdm for progress bar
            progress_bar = tqdm(self.train_loader, desc=f"Epoch {epoch+1}/{self.num_epochs}")

            for i, batch in enumerate(progress_bar):
                tensors = self._prepare_batch_tensors(batch)
                losses = self._compute_batch_losses(tensors)
                loss = losses['loss']

                # Skip backward when loss has no gradient (e.g., empty memory bank on first step)
                if not loss.requires_grad:
                    continue

                # Guard against NaN/Inf loss — skip backward to prevent parameter corruption.
                # A single NaN loss would make all model params NaN via optimizer update,
                # permanently collapsing training.
                if not torch.isfinite(loss):
                    print(f"\nWARNING: non-finite loss ({loss.item():.4g}) at epoch {epoch} step {i}. Skipping backward.")
                    self.optimizer.zero_grad()
                    continue

                # Gradient Accumulation
                loss = loss / self.gradient_accumulation_steps

                # Backward Pass
                loss.backward()

                if (i + 1) % self.gradient_accumulation_steps == 0 or (i + 1) == len(self.train_loader):
                    torch.nn.utils.clip_grad_norm_(
                        [p for p in self.model.parameters() if p.grad is not None],
                        max_norm=self.max_grad_norm,
                    )
                    self.optimizer.step()
                    self.optimizer.zero_grad()
                    self._update_ema()  # EMA after every optimizer step

                scaled_total_loss = loss.item() * self.gradient_accumulation_steps
                epoch_losses.update(losses, scaled_total_loss)

                # Update progress bar
                progress_losses = dict(losses)
                progress_losses['loss'] = loss * self.gradient_accumulation_steps
                postfix = self._build_progress_postfix(progress_losses)
                progress_bar.set_postfix(postfix)

            averages = epoch_losses.averages()
            self.scheduler.step()
            val_loss, retrieval_metrics, valid_retrieval_metrics = self._run_epoch_validation(epoch, averages)

            should_stop = self._finalize_epoch(
                epoch,
                averages,
                val_loss,
                retrieval_metrics,
                valid_retrieval_metrics,
            )
            if should_stop:
                break

        if self.is_main_process:
            self._run_post_training_evaluations()
        if dist.is_initialized():
            dist.barrier()
