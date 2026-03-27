"""
Recall@K Evaluation for Time-Series Embedder
=============================================
Measures retrieval quality purely by neighbor purity:
  - For each query, find top-K nearest neighbors (cosine similarity, LOO).
  - Count how many of the K neighbors share the same class label.

Metrics reported:
  R@K  (Recall@K)    : fraction of queries with AT LEAST ONE correct neighbor in top-K
  P@K  (Precision@K) : mean fraction of top-K neighbors that are same-class per query
  H@K  (Hits@K)      : mean count of same-class neighbors in top-K per query

Run:
  python -m mts_agent.retrieval.eval_recall \\
      --config mts_agent/configs/natops_qwen35_4b_server.json \\
      --ckpt   checkpoints_natops_v9/model_best.pt
"""
import argparse
import os
import sys

import numpy as np
import torch


# ── patches must run before any fla/transformers import ───────────────────────

def _patch_triton_autotuner():
    try:
        from triton.runtime import autotuner as _m
        _orig = _m.Autotuner.__init__
        def _permissive(self, fn, arg_names, configs, key, *a, **kw):
            valid = [k for k in key if k in arg_names]
            dropped = [k for k in key if k not in arg_names]
            if dropped:
                print(f"[triton patch] Removed invalid autotune keys {dropped} from '{getattr(fn,'__name__',fn)}'")
            return _orig(self, fn, arg_names, configs, valid, *a, **kw)
        _m.Autotuner.__init__ = _permissive
        print("[triton patch] Autotuner patched.")
    except Exception as e:
        print(f"[triton patch] Skipped: {e}")

def _patch_fla_rms_norm():
    try:
        import fla.modules as _fla
        import torch as _t
        def _pt_fwd(self, x):
            f = x.float()
            out = (f * torch.rsqrt(f.pow(2).mean(-1, keepdim=True) + getattr(self,'eps',1e-6))).to(x.dtype)
            w = getattr(self,'weight',None)
            return out * w.to(x.dtype) if w is not None else out
        for name in ('RMSNorm','FusedRMSNorm'):
            cls = getattr(_fla, name, None)
            if cls and hasattr(cls,'forward'):
                cls.forward = _pt_fwd
        print("[fla patch] RMSNorm replaced with PyTorch fallback.")
    except Exception as e:
        print(f"[fla patch] Skipped: {e}")

def _patch_lm_loss_bf16():
    try:
        import transformers.loss.loss_utils as _lu
        import torch.nn.functional as _F
        def _bf16_loss(logits, labels, vocab_size, **kw):
            s = logits[...,:-1,:].contiguous().view(-1, vocab_size)
            t = labels[...,1:].contiguous().view(-1)
            return _F.cross_entropy(s, t.to(s.device), ignore_index=-100)
        _lu.ForCausalLMLoss = _bf16_loss
        print("[lm loss patch] bf16 logits patch applied.")
    except Exception as e:
        print(f"[lm loss patch] Skipped: {e}")

_patch_triton_autotuner()
_patch_fla_rms_norm()
_patch_lm_loss_bf16()

# ── DDP init (no-op for single-GPU) ──────────────────────────────────────────
_local_rank = int(os.environ.get("LOCAL_RANK", -1))
if _local_rank >= 0:
    torch.distributed.init_process_group(backend="nccl")
    torch.cuda.set_device(_local_rank)

# ── project imports ───────────────────────────────────────────────────────────
from mts_agent.config import ExperimentConfig
from mts_agent.data.loader import MTSDataset
from mts_agent.data.collator import MultimodalCollator
from mts_agent.data.prompt_builder import build_retrieval_prompt
from mts_agent.models.mts_embedder import MTSEmbedder
from transformers import AutoTokenizer


# ─────────────────────────────────────────────────────────────────────────────
# Embedding helpers
# ─────────────────────────────────────────────────────────────────────────────

def embed_dataset(model, dataset, collator, device):
    """Return (embeddings [N,D] float32 numpy, labels [N] list)."""
    from tqdm import tqdm
    model.eval()
    embeddings, labels = [], []
    with torch.no_grad():
        for i in tqdm(range(len(dataset)), desc="Embedding"):
            item = dataset[i]
            ts_input = item['time_series'].unsqueeze(0).to(device).float()
            context  = item['context']
            prompt   = build_retrieval_prompt(context)
            inputs   = collator.tokenizer(
                prompt, return_tensors='pt', truncation=True,
                max_length=collator.max_length
            )
            ids  = inputs.input_ids.to(device)
            mask = inputs.attention_mask.to(device)
            emb  = model.get_embedding(ts_input, ids, attention_mask=mask)
            embeddings.append(emb.cpu().float().numpy().flatten())
            labels.append(str(item['label']))
    return np.array(embeddings, dtype=np.float32), labels


# ─────────────────────────────────────────────────────────────────────────────
# Recall@K computation
# ─────────────────────────────────────────────────────────────────────────────

def recall_at_k(
    query_embs: np.ndarray,
    query_labels: list,
    gallery_embs: np.ndarray,
    gallery_labels: list,
    ks=(1, 3, 5),
    loo: bool = True,
):
    """
    Compute Recall@K, Precision@K and mean Hits@K.

    Args:
        loo: If True, treat query_embs == gallery_embs and exclude self (index i).
    Returns:
        dict: {k: {'R@K': float, 'P@K': float, 'H@K': float}}
    """
    # L2-normalise for cosine similarity
    def l2norm(x):
        n = np.linalg.norm(x, axis=1, keepdims=True)
        return x / np.maximum(n, 1e-8)

    Q_norm = l2norm(query_embs)    # [Q, D]
    G_norm = l2norm(gallery_embs)  # [G, D]

    sim = Q_norm @ G_norm.T        # [Q, G]

    max_k = max(ks)
    results = {}

    recall_at  = {k: [] for k in ks}
    precision_at = {k: [] for k in ks}
    hits_at    = {k: [] for k in ks}

    for i in range(len(query_labels)):
        row = sim[i].copy()
        if loo:
            row[i] = -2.0          # exclude self

        # sort descending; take top max_k
        top_idx = np.argsort(-row)[:max_k + (1 if loo else 0)]

        q_lbl = query_labels[i]
        for k in ks:
            neighbors = top_idx[:k]
            same = sum(1 for j in neighbors if gallery_labels[j] == q_lbl)
            recall_at[k].append(1.0 if same >= 1 else 0.0)
            precision_at[k].append(same / k)
            hits_at[k].append(float(same))

    for k in ks:
        results[k] = {
            'R@K': float(np.mean(recall_at[k])),
            'P@K': float(np.mean(precision_at[k])),
            'H@K': float(np.mean(hits_at[k])),
        }
    return results


def print_table(title, results, ks=(1, 3, 5)):
    print(f"\n{'='*56}")
    print(f"  {title}")
    print(f"{'='*56}")
    print(f"  {'K':>4}  {'R@K (≥1 correct)':>18}  {'P@K (frac correct)':>18}  {'H@K (mean count)':>16}")
    print(f"  {'-'*4}  {'-'*18}  {'-'*18}  {'-'*16}")
    for k in ks:
        r = results[k]
        print(f"  {k:>4}  {r['R@K']:>17.2%}  {r['P@K']:>17.2%}  {r['H@K']:>15.2f}")
    print(f"{'='*56}")


# ─────────────────────────────────────────────────────────────────────────────
# Main
# ─────────────────────────────────────────────────────────────────────────────

def main():
    parser = argparse.ArgumentParser(description="Recall@K evaluation for MTS Embedder")
    parser.add_argument("--config", required=True, help="Path to JSON config file")
    parser.add_argument("--ckpt",   default=None,  help="Checkpoint path (default: auto-detect model_best.pt)")
    parser.add_argument("--ks",     default="1,3,5", help="Comma-separated K values (default: 1,3,5)")
    parser.add_argument("--device", default=None,  help="Override device (e.g. cuda:0)")
    args = parser.parse_args()

    ks = [int(k) for k in args.ks.split(",")]

    # ── Config ────────────────────────────────────────────────────────────────
    cfg = ExperimentConfig.load(args.config)
    if cfg.environment.force_offline:
        os.environ["HF_HUB_OFFLINE"] = "1"
        os.environ["TRANSFORMERS_OFFLINE"] = "1"

    # ── Device ────────────────────────────────────────────────────────────────
    if args.device:
        device = torch.device(args.device)
    elif torch.cuda.is_available():
        device = torch.device(f"cuda:{max(0, _local_rank)}")
    else:
        device = torch.device("cpu")
    print(f"Device: {device}")

    # ── Tokenizer ─────────────────────────────────────────────────────────────
    tokenizer = AutoTokenizer.from_pretrained(
        cfg.model.llm_path, trust_remote_code=True, local_files_only=True
    )
    if tokenizer.pad_token is None:
        tokenizer.pad_token = tokenizer.eos_token

    # ── Collator ──────────────────────────────────────────────────────────────
    collator = MultimodalCollator(
        tokenizer, mode='train',
        max_length=cfg.data.max_length,
        alignment_text_mode='context',
    )

    # ── Model ─────────────────────────────────────────────────────────────────
    print("Loading model...")
    model = MTSEmbedder(
        cfg.model.llm_path,
        ts_hidden_dim=cfg.model.ts_hidden_dim,
        encoder_base_channels=cfg.model.encoder_base_channels,
        encoder_dropout=cfg.model.encoder_dropout,
        encoder_norm=cfg.model.encoder_norm,
        embedding_pooling=cfg.model.embedding_pooling,
        output_dim=cfg.model.output_dim,
        llm_attn_implementation=cfg.model.llm_attn_implementation,
        stem_strides=cfg.model.stem_strides,
        patch_size=cfg.model.patch_size,
    )
    from peft import get_peft_model, LoraConfig, TaskType
    lora_cfg = LoraConfig(
        task_type=TaskType.CAUSAL_LM, inference_mode=True, r=8, lora_alpha=16,
        lora_dropout=0.05,
        target_modules=["q_proj","k_proj","v_proj","o_proj",
                        "gate_proj","up_proj","down_proj","g_proj"],
    )
    model.llm = get_peft_model(model.llm, lora_cfg)

    # ── Checkpoint ────────────────────────────────────────────────────────────
    ckpt_path = args.ckpt
    if ckpt_path is None:
        auto = os.path.join(cfg.training.save_dir, "model_best.pt")
        if os.path.exists(auto):
            ckpt_path = auto
            print(f"Auto-detected checkpoint: {ckpt_path}")
        else:
            print("WARNING: No checkpoint found — using randomly initialised weights.")

    if ckpt_path and os.path.exists(ckpt_path):
        state = torch.load(ckpt_path, map_location="cpu")
        missing, unexpected = model.load_state_dict(state, strict=False)
        if missing:
            print(f"  Checkpoint missing keys: {len(missing)}")
        if unexpected:
            print(f"  Checkpoint unexpected keys: {len(unexpected)}")
        print(f"  Loaded: {ckpt_path}")

    # move non-LLM modules to device
    for name, mod in model.named_children():
        if name != "llm":
            mod.to(device)

    model.eval()

    # ── Datasets ──────────────────────────────────────────────────────────────
    train_dataset = MTSDataset(
        cfg.data.data_path, mode='train', augment=False,
        domain_info=cfg.data.domain_info,
    )
    print(f"Train set: {len(train_dataset)} samples")

    valid_path = cfg.data.data_path.replace("train", "valid")
    valid_dataset = None
    if os.path.exists(valid_path):
        valid_dataset = MTSDataset(
            valid_path, mode='train', augment=False,
            domain_info=cfg.data.domain_info,
        )
        print(f"Valid set: {len(valid_dataset)} samples")

    # ── Embed train ───────────────────────────────────────────────────────────
    print("\n[1/2] Embedding train set...")
    train_embs, train_labels = embed_dataset(model, train_dataset, collator, device)

    # ── LOO evaluation on train ───────────────────────────────────────────────
    print("\nComputing LOO Recall@K on train set...")
    loo_results = recall_at_k(
        train_embs, train_labels,
        train_embs, train_labels,
        ks=ks, loo=True,
    )
    print_table("LOO Recall@K  (train → train, exclude self)", loo_results, ks)

    # ── Cross-set evaluation (train gallery → valid queries) ──────────────────
    if valid_dataset is not None:
        print("\n[2/2] Embedding valid set...")
        valid_embs, valid_labels = embed_dataset(model, valid_dataset, collator, device)

        print("\nComputing Cross-set Recall@K (gallery=train, queries=valid)...")
        cross_results = recall_at_k(
            valid_embs, valid_labels,
            train_embs, train_labels,
            ks=ks, loo=False,
        )
        print_table("Cross-set Recall@K  (valid queries → train gallery)", cross_results, ks)
    else:
        print("\n[2/2] No valid set found — skipping cross-set evaluation.")

    # ── Per-class breakdown (train LOO) ───────────────────────────────────────
    unique_labels = sorted(set(train_labels))
    if len(unique_labels) <= 20:
        print(f"\n{'='*56}")
        print("  Per-class P@5 (train LOO)")
        print(f"{'='*56}")
        Q_norm = train_embs / np.maximum(np.linalg.norm(train_embs, axis=1, keepdims=True), 1e-8)
        sim = Q_norm @ Q_norm.T
        for lbl in unique_labels:
            indices = [i for i, l in enumerate(train_labels) if l == lbl]
            p5_list = []
            for i in indices:
                row = sim[i].copy(); row[i] = -2.0
                top5 = np.argsort(-row)[:5]
                same = sum(1 for j in top5 if train_labels[j] == lbl)
                p5_list.append(same / 5)
            print(f"  class {lbl:>12s}: P@5 = {np.mean(p5_list):.2%}  ({len(indices)} samples)")
        print(f"{'='*56}")


if __name__ == "__main__":
    main()
