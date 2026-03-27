"""
Recall@K analysis for MTS-O1-Embedder v9 on NATOPS.
Computes how many same-class samples appear in top-1, top-3, top-5 neighbors.
"""
import sys
import os
sys.path.insert(0, '/data/USTCAGI2/zzProject')
os.chdir('/data/USTCAGI2/zzProject')

import json
import numpy as np
import torch
from collections import defaultdict
from tqdm import tqdm

from mts_agent.models.mts_embedder import MTSEmbedder
from mts_agent.data.loader import MTSDataset
from mts_agent.data.collator import MultimodalCollator
from mts_agent.retrieval.evaluate_retrieval import build_gallery, embed_queries
from transformers import AutoTokenizer

# ── Config ────────────────────────────────────────────────────────────────────
LLM_PATH      = "local_model/Qwen/Qwen3___5-4B"
CKPT_PATH     = "checkpoints_natops_v9/model_best.pt"
TRAIN_PATH    = "mts_agent/data/processed/natops_train.jsonl"
TEST_PATH     = "mts_agent/data/processed/natops_valid.jsonl"
ALPHA         = 0.9
K_MAX         = 5
DEVICE        = torch.device("cuda:0")

# ── Load model ────────────────────────────────────────────────────────────────
print("Loading tokenizer...")
tokenizer = AutoTokenizer.from_pretrained(LLM_PATH, trust_remote_code=True, local_files_only=True)
if tokenizer.pad_token is None:
    tokenizer.pad_token = tokenizer.eos_token

print("Loading model...")
model = MTSEmbedder(
    LLM_PATH,
    ts_hidden_dim=128,
    output_dim=1024,
    encoder_base_channels=64,
    encoder_dropout=0.1,
    encoder_norm="group",
    embedding_pooling="last",
    stem_strides=[2],
    patch_size=5,
)
model.apply_lora(r=8, lora_alpha=16, lora_dropout=0.05)

print(f"Loading checkpoint from {CKPT_PATH}...")
state_dict = torch.load(CKPT_PATH, map_location=DEVICE)
model_state = model.state_dict()
filtered = {k: v for k, v in state_dict.items()
            if k in model_state and v.shape == model_state[k].shape}
model.load_state_dict(filtered, strict=False)
model.to(DEVICE)
model.eval()
print("Model ready.")

collator = MultimodalCollator(tokenizer, mode='inference', max_length=320)

# ── Build gallery (train, context-only) ──────────────────────────────────────
print("\nBuilding gallery from train set (context-only)...")
train_dataset = MTSDataset(TRAIN_PATH, mode='inference')
retriever, _, gallery_labels, _ = build_gallery(
    model, train_dataset, collator, DEVICE,
    dtw_window_size=None, fast_dtw_max_len=100,
    use_full_prompt=False, ts_only_embedding=False,
)

# ── Embed test queries (thought-enriched) ────────────────────────────────────
print("\nEmbedding test queries (thought-enriched)...")
test_dataset = MTSDataset(TEST_PATH, mode='inference')
query_embeddings, query_labels, query_ts_list = embed_queries(
    model, test_dataset, collator, DEVICE,
    ts_only_embedding=False, use_full_prompt=True,
)
print(f"Gallery: {len(gallery_labels)} samples | Queries: {len(query_labels)} samples")

# ── Recall@K computation ─────────────────────────────────────────────────────
print(f"\nComputing Recall@K (alpha={ALPHA})...")

# Per-class counters: hits at each K
class_names = sorted(set(str(l) for l in query_labels))
n_per_class = {c: sum(1 for l in gallery_labels if str(l) == c) for c in class_names}

# Aggregate counts
same_class_in_topk = {k: [] for k in [1, 3, 5]}   # per query
per_class_hits = {c: {k: [] for k in [1, 3, 5]} for c in class_names}

for i in tqdm(range(len(query_embeddings))):
    q_emb = query_embeddings[i]
    q_ts  = query_ts_list[i]
    true_label = str(query_labels[i])

    results = retriever.search(q_emb, q_ts, k=K_MAX, alpha=ALPHA)
    neighbor_labels = [str(r.get('label', r['id'].split('_')[-1])) for r in results]

    for k in [1, 3, 5]:
        top_k = neighbor_labels[:k]
        hits = sum(1 for l in top_k if l == true_label)
        same_class_in_topk[k].append(hits)
        per_class_hits[true_label][k].append(hits)

# ── Print results ─────────────────────────────────────────────────────────────
n_queries = len(query_labels)
n_gallery_per_class = n_queries // len(class_names)   # 30 for NATOPS

print("\n" + "="*60)
print("  Recall@K Analysis — NATOPS Test Set (gallery=train)")
print(f"  {n_queries} queries | {len(gallery_labels)} gallery | {len(class_names)} classes")
print("="*60)

print("\n── Overall ─────────────────────────────────────────────")
for k in [1, 3, 5]:
    hits = same_class_in_topk[k]
    total_hits = sum(hits)
    avg_hits   = np.mean(hits)
    recall     = avg_hits / k   # Recall@K = avg(hits@K) / K  (fraction of top-K that's relevant)
    pct_perfect = 100 * sum(1 for h in hits if h == k) / n_queries
    print(f"  Top-{k}: avg {avg_hits:.2f}/{k} same-class "
          f"| total hits={total_hits}/{n_queries*k} "
          f"| Precision@{k}={recall:.1%} "
          f"| {pct_perfect:.1f}% queries have all-{k} same-class")

print(f"\n  Hit@1 (top-1 is correct): {100*np.mean([h>0 for h in same_class_in_topk[1]]):.1f}%")
print(f"  Hit@3 (any of top-3 correct): {100*np.mean([h>0 for h in same_class_in_topk[3]]):.1f}%")
print(f"  Hit@5 (any of top-5 correct): {100*np.mean([h>0 for h in same_class_in_topk[5]]):.1f}%")

print("\n── Per Class ───────────────────────────────────────────")
print(f"  {'Class':<8} {'N_q':>4} | {'Top1 hits':>10} | {'Top3 hits':>10} | {'Top5 hits':>10} | Hit@1")
print("  " + "-"*60)
for c in class_names:
    nq = len(per_class_hits[c][1])
    h1 = sum(per_class_hits[c][1])
    h3 = sum(per_class_hits[c][3])
    h5 = sum(per_class_hits[c][5])
    hit1 = 100 * sum(1 for h in per_class_hits[c][1] if h > 0) / max(nq, 1)
    print(f"  {c:<8} {nq:>4} | {h1:>5}/{nq*1:<4} | {h3:>5}/{nq*3:<4} | {h5:>5}/{nq*5:<4} | {hit1:.1f}%")

print("\n── Distribution of same-class count in Top-5 ──────────")
from collections import Counter
dist = Counter(same_class_in_topk[5])
for cnt in sorted(dist.keys()):
    bar = '█' * dist[cnt]
    print(f"  {cnt} same-class in top-5: {dist[cnt]:3d} queries  {bar}")

print("\n" + "="*60)
