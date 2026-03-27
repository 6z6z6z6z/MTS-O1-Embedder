"""
Baseline retrieval comparison: DTW, Euclidean, Cosine on raw time series.
No LLM embeddings — pure TS distance metrics.
"""
import sys, os
sys.path.insert(0, '/data/USTCAGI2/zzProject')
os.chdir('/data/USTCAGI2/zzProject')

import json
import numpy as np
from collections import defaultdict, Counter
from tqdm import tqdm

# ── Load JSONL data ────────────────────────────────────────────────────────────
def load_jsonl(path):
    samples = []
    with open(path) as f:
        for line in f:
            line = line.strip()
            if line:
                samples.append(json.loads(line))
    return samples

def get_ts(sample):
    """Return time series as numpy array, shape (C, T)."""
    ts = sample.get('time_series') or sample.get('ts')
    if ts is not None:
        return np.array(ts, dtype=np.float32)
    raise ValueError("No time_series field found")

print("Loading data...")
train_samples = load_jsonl('mts_agent/data/processed/natops_train.jsonl')
test_samples  = load_jsonl('mts_agent/data/processed/natops_valid.jsonl')
print(f"Train: {len(train_samples)} | Test: {len(test_samples)}")

train_ts     = [get_ts(s) for s in train_samples]
train_labels = [str(s['label']) for s in train_samples]
test_ts      = [get_ts(s) for s in test_samples]
test_labels  = [str(s['label']) for s in test_samples]

# Flatten for Euclidean / Cosine: (N, C*T)
train_flat = np.stack([t.flatten() for t in train_ts])   # (180, C*T)
test_flat  = np.stack([t.flatten() for t in test_ts])    # (180, C*T)

# ── DTW (per-channel average) ──────────────────────────────────────────────────
try:
    from scipy.spatial.distance import cdist
    HAS_SCIPY = True
except ImportError:
    HAS_SCIPY = False

def dtw_distance_1d(a, b):
    """Standard DTW on 1D sequences."""
    n, m = len(a), len(b)
    dtw = np.full((n+1, m+1), np.inf)
    dtw[0, 0] = 0.0
    for i in range(1, n+1):
        for j in range(1, m+1):
            cost = abs(float(a[i-1]) - float(b[j-1]))
            dtw[i, j] = cost + min(dtw[i-1, j], dtw[i, j-1], dtw[i-1, j-1])
    return dtw[n, m]

def dtw_multichannel(a, b):
    """Average DTW across channels. a,b shape: (C, T)"""
    C = a.shape[0]
    return np.mean([dtw_distance_1d(a[c], b[c]) for c in range(C)])

# ── Recall@K evaluator ─────────────────────────────────────────────────────────
def recall_at_k(dist_matrix, train_lbls, test_lbls, ks=(1,3,5), higher_is_better=False):
    """
    dist_matrix: (N_test, N_train)
    higher_is_better: True for similarity, False for distance
    Returns dict of metrics.
    """
    n_test = len(test_lbls)
    class_names = sorted(set(test_lbls))
    hits = {k: [] for k in ks}
    per_class_hits = {c: {k: [] for k in ks} for c in class_names}

    for i in range(n_test):
        row = dist_matrix[i]
        if higher_is_better:
            order = np.argsort(-row)
        else:
            order = np.argsort(row)
        true_lbl = test_lbls[i]
        neighbor_lbls = [train_lbls[j] for j in order[:max(ks)]]
        for k in ks:
            h = sum(1 for l in neighbor_lbls[:k] if l == true_lbl)
            hits[k].append(h)
            per_class_hits[true_lbl][k].append(h)

    return hits, per_class_hits

def print_report(name, hits, per_class_hits, n_test, ks=(1,3,5)):
    class_names = sorted(per_class_hits.keys())
    print(f"\n{'='*60}")
    print(f"  {name}")
    print(f"{'='*60}")
    print(f"\n── Overall ─────────────────────────────────────────────")
    for k in ks:
        h = hits[k]
        avg = np.mean(h)
        total = sum(h)
        prec = avg / k
        pct_perfect = 100 * sum(1 for x in h if x == k) / n_test
        any_hit = 100 * sum(1 for x in h if x > 0) / n_test
        print(f"  Top-{k}: avg {avg:.2f}/{k} | Precision@{k}={prec:.1%} | "
              f"Hit@{k}={any_hit:.1f}% | All-same={pct_perfect:.1f}%")

    print(f"\n── Per Class ───────────────────────────────────────────")
    print(f"  {'Class':<6} | {'Top1':>8} | {'Top3':>8} | {'Top5':>8} | Hit@1")
    print("  " + "-"*50)
    for c in class_names:
        nq = len(per_class_hits[c][1])
        h1 = sum(per_class_hits[c][1])
        h3 = sum(per_class_hits[c][3])
        h5 = sum(per_class_hits[c][5])
        hit1 = 100 * sum(1 for h in per_class_hits[c][1] if h > 0) / max(nq, 1)
        print(f"  {c:<6} | {h1:>3}/{nq*1:<4} | {h3:>3}/{nq*3:<4} | {h5:>3}/{nq*5:<4} | {hit1:.1f}%")

    print(f"\n── Top-5 distribution ──────────────────────────────────")
    dist5 = Counter(hits[5])
    for cnt in sorted(dist5):
        bar = '█' * dist5[cnt]
        print(f"  {cnt}/5 same-class: {dist5[cnt]:3d} queries  {bar}")

# ── 1. Euclidean Distance ──────────────────────────────────────────────────────
print("\nComputing Euclidean distance matrix...")
# Normalize per sample (z-score) before computing distance
train_flat_z = (train_flat - train_flat.mean(axis=1, keepdims=True)) / (train_flat.std(axis=1, keepdims=True) + 1e-8)
test_flat_z  = (test_flat  - test_flat.mean(axis=1,  keepdims=True)) / (test_flat.std(axis=1,  keepdims=True)  + 1e-8)
euc_dist = np.sqrt(((test_flat_z[:, None, :] - train_flat_z[None, :, :]) ** 2).sum(axis=2))  # (180, 180)
hits_euc, pc_euc = recall_at_k(euc_dist, train_labels, test_labels, higher_is_better=False)
print_report("Euclidean Distance (z-score normalized)", hits_euc, pc_euc, len(test_labels))

# ── 2. Cosine Similarity ───────────────────────────────────────────────────────
print("\nComputing Cosine similarity matrix...")
train_norm = train_flat_z / (np.linalg.norm(train_flat_z, axis=1, keepdims=True) + 1e-8)
test_norm  = test_flat_z  / (np.linalg.norm(test_flat_z,  axis=1, keepdims=True) + 1e-8)
cos_sim = test_norm @ train_norm.T   # (180, 180)
hits_cos, pc_cos = recall_at_k(cos_sim, train_labels, test_labels, higher_is_better=True)
print_report("Cosine Similarity (z-score normalized)", hits_cos, pc_cos, len(test_labels))

# ── 3. DTW (full multichannel) ─────────────────────────────────────────────────
print("\nComputing DTW distance matrix (multichannel, this may take a while)...")
n_train, n_test = len(train_ts), len(test_ts)
dtw_dist = np.zeros((n_test, n_train), dtype=np.float32)
for i in tqdm(range(n_test)):
    for j in range(n_train):
        dtw_dist[i, j] = dtw_multichannel(test_ts[i], train_ts[j])
hits_dtw, pc_dtw = recall_at_k(dtw_dist, train_labels, test_labels, higher_is_better=False)
print_report("DTW Distance (multichannel avg)", hits_dtw, pc_dtw, len(test_labels))

# ── Summary comparison ─────────────────────────────────────────────────────────
print("\n" + "="*60)
print("  SUMMARY: Top-5 Precision@K comparison")
print("="*60)
print(f"  {'Method':<35} | {'P@1':>6} | {'P@3':>6} | {'P@5':>6} | {'Hit@5':>7}")
print("  " + "-"*65)

methods = [
    ("Euclidean (z-norm)",   hits_euc),
    ("Cosine (z-norm)",      hits_cos),
    ("DTW (multichannel)",   hits_dtw),
]
for name, hits in methods:
    p1 = np.mean(hits[1])
    p3 = np.mean(hits[3]) / 3
    p5 = np.mean(hits[5]) / 5
    hit5 = sum(1 for h in hits[5] if h > 0) / len(test_labels)
    print(f"  {name:<35} | {p1:>5.1%} | {p3:>5.1%} | {p5:>5.1%} | {hit5:>6.1%}")

# MTSEmbedder v9 reference
print(f"  {'MTSEmbedder v9 (Qwen3.5+LoRA)':<35} | {'80.6%':>6} | {'81.3%':>6} | {'79.7%':>6} | {'93.9%':>7}")
print("="*60)
