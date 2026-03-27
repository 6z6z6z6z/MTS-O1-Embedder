#!/usr/bin/env python3
"""Pure time-series baselines for NATOPS (gallery=train 180 / query=test 180)."""
import json, sys, numpy as np
from scipy.spatial.distance import cdist

sys.path.insert(0, '/data/USTCAGI2/zzProject')

GALLERY = 'mts_agent/data/processed/natops_train.jsonl'
QUERY   = 'mts_agent/data/processed/natops_valid.jsonl'
K = 5


def load(path):
    items = []
    with open(path) as f:
        for line in f:
            if line.strip():
                items.append(json.loads(line))
    return items


def get_ts(item):
    ts = np.array(item['time_series'], dtype=np.float32)
    if ts.ndim == 1:
        ts = ts[None]
    if ts.shape[0] > ts.shape[1]:
        ts = ts.T
    return ts  # [C, T]


def znorm_global(ts):
    m, s = ts.mean(), ts.std() + 1e-10
    return (ts - m) / s


def znorm_perchan(ts):
    m = ts.mean(1, keepdims=True)
    s = ts.std(1, keepdims=True) + 1e-10
    return (ts - m) / s


def dtw_dist_multichan(s1, s2, window=None):
    """Multivariate DTW; s1/s2: [C, T] -> treated as [T, C] feature vectors."""
    a, b = s1.T, s2.T  # [T, C]
    n, m_len = len(a), len(b)
    if window is None:
        window = max(n, m_len) // 4
    dm = cdist(a, b, 'euclidean')
    dp = np.full((n + 1, m_len + 1), np.inf)
    dp[0, 0] = 0.0
    for i in range(1, n + 1):
        j0 = max(1, i - window)
        j1 = min(m_len, i + window) + 1
        for j in range(j0, j1):
            dp[i, j] = dm[i-1, j-1] + min(dp[i-1, j], dp[i, j-1], dp[i-1, j-1])
    return dp[n, m_len]


def eval_metrics(sim_mat, q_labels, g_labels, higher_is_better=True):
    N_q = len(q_labels)
    p1, p3, p5, h3, h5, pf3, pf5, knn_ok = [], [], [], [], [], [], [], []
    for i in range(N_q):
        row = sim_mat[i]
        order = np.argsort(row)[::-1] if higher_is_better else np.argsort(row)
        ql = q_labels[i]
        top5 = [g_labels[order[j]] for j in range(5)]
        m5 = sum(l == ql for l in top5)
        m3 = sum(l == ql for l in top5[:3])
        p1.append(1 if top5[0] == ql else 0)
        p3.append(m3 / 3)
        p5.append(m5 / 5)
        h3.append(1 if m3 > 0 else 0)
        h5.append(1 if m5 > 0 else 0)
        pf3.append(1 if m3 == 3 else 0)
        pf5.append(1 if m5 == 5 else 0)
        # kNN rank-weighted vote
        votes = {}
        for rank, idx in enumerate(order[:K], 1):
            lb = g_labels[idx]
            votes[lb] = votes.get(lb, 0) + 1.0 / rank
        pred = max(votes, key=votes.get)
        knn_ok.append(1 if pred == ql else 0)

    def pct(lst):
        return np.mean(lst) * 100

    return {
        'P@1': pct(p1), 'P@3': pct(p3), 'P@5': pct(p5),
        'Hit@3': pct(h3), 'Hit@5': pct(h5),
        'Perfect@3': pct(pf3), 'Perfect@5': pct(pf5),
        'kNN@5': pct(knn_ok),
    }


def per_class_p1(sim_mat, q_labels, g_labels, higher_is_better=True):
    classes = sorted(set(q_labels))
    result = {}
    for cl in classes:
        idx = [i for i, l in enumerate(q_labels) if l == cl]
        correct = 0
        for i in idx:
            row = sim_mat[i]
            order = np.argsort(row)[::-1] if higher_is_better else np.argsort(row)
            if g_labels[order[0]] == cl:
                correct += 1
        result[cl] = correct / len(idx) * 100
    return result


def fmt(name, m, pc):
    print("\n" + "=" * 55)
    print("  " + name)
    print("=" * 55)
    print("  P@1=%.1f%%  P@3=%.1f%%  P@5=%.1f%%" % (m['P@1'], m['P@3'], m['P@5']))
    print("  Hit@3=%.1f%%  Hit@5=%.1f%%" % (m['Hit@3'], m['Hit@5']))
    print("  Perfect@3=%.1f%%  Perfect@5=%.1f%%" % (m['Perfect@3'], m['Perfect@5']))
    print("  kNN@5=%.1f%%" % m['kNN@5'])
    per_cls = "  ".join("cls%s=%.1f%%" % (c, v) for c, v in pc.items())
    print("  Per-class Hit@1: " + per_cls)


# ── Load data ─────────────────────────────────────────────────────────────────
gallery = load(GALLERY)
query   = load(QUERY)
g_ts    = [get_ts(s) for s in gallery]
q_ts    = [get_ts(s) for s in query]
g_lbl   = [str(s.get('label', '')) for s in gallery]
q_lbl   = [str(s.get('label', '')) for s in query]
print("Gallery: %d, Query: %d, TS shape: %s" % (len(g_lbl), len(q_lbl), str(g_ts[0].shape)))

# ── 1. Euclidean-raw ──────────────────────────────────────────────────────────
print("\n[1/5] Euclidean-raw (flattened C*T)...")
G1 = np.stack([ts.flatten() for ts in g_ts])
Q1 = np.stack([ts.flatten() for ts in q_ts])
D1 = cdist(Q1, G1, 'euclidean')
m1 = eval_metrics(-D1, q_lbl, g_lbl)
pc1 = per_class_p1(-D1, q_lbl, g_lbl)
fmt("Euclidean-raw", m1, pc1)

# ── 2. Euclidean-znorm-perchan ────────────────────────────────────────────────
print("\n[2/5] Euclidean-znorm-perchan...")
G2 = np.stack([znorm_perchan(ts).flatten() for ts in g_ts])
Q2 = np.stack([znorm_perchan(ts).flatten() for ts in q_ts])
D2 = cdist(Q2, G2, 'euclidean')
m2 = eval_metrics(-D2, q_lbl, g_lbl)
pc2 = per_class_p1(-D2, q_lbl, g_lbl)
fmt("Euclidean-znorm-perchan", m2, pc2)

# ── 3. Cosine-znorm-perchan ───────────────────────────────────────────────────
print("\n[3/5] Cosine-znorm-perchan...")
Gnrm = G2 / (np.linalg.norm(G2, axis=1, keepdims=True) + 1e-10)
Qnrm = Q2 / (np.linalg.norm(Q2, axis=1, keepdims=True) + 1e-10)
Sim3 = Qnrm @ Gnrm.T
m3 = eval_metrics(Sim3, q_lbl, g_lbl, higher_is_better=True)
pc3 = per_class_p1(Sim3, q_lbl, g_lbl, higher_is_better=True)
fmt("Cosine-znorm-perchan", m3, pc3)

# ── 4. DTW-multichan (global z-norm + multivariate DTW) ──────────────────────
print("\n[4/5] DTW-multichan (z-norm-global, multivariate)...")
g_zn = [znorm_global(ts) for ts in g_ts]
q_zn = [znorm_global(ts) for ts in q_ts]
D4 = np.zeros((len(q_lbl), len(g_lbl)), dtype=np.float32)
for i, q in enumerate(q_zn):
    if i % 30 == 0:
        print("  query %d/%d..." % (i, len(q_zn)))
    for j, g in enumerate(g_zn):
        D4[i, j] = dtw_dist_multichan(q, g)
m4 = eval_metrics(-D4, q_lbl, g_lbl)
pc4 = per_class_p1(-D4, q_lbl, g_lbl)
fmt("DTW-multichan (z-norm-global)", m4, pc4)

# ── 5. DTW-perchan (per-channel z-norm, average univariate DTW) ──────────────
print("\n[5/5] DTW-perchan (z-norm-perchan, avg univariate DTW per channel)...")
g_zpc = [znorm_perchan(ts) for ts in g_ts]
q_zpc = [znorm_perchan(ts) for ts in q_ts]
C = g_ts[0].shape[0]
D5 = np.zeros((len(q_lbl), len(g_lbl)), dtype=np.float32)
for i, q in enumerate(q_zpc):
    if i % 30 == 0:
        print("  query %d/%d..." % (i, len(q_zpc)))
    for j, g in enumerate(g_zpc):
        ch_d = []
        for c in range(C):
            s1 = q[c:c+1, :]  # [1, T]
            s2 = g[c:c+1, :]  # [1, T]
            ch_d.append(dtw_dist_multichan(s1, s2))
        D5[i, j] = float(np.mean(ch_d))
m5 = eval_metrics(-D5, q_lbl, g_lbl)
pc5 = per_class_p1(-D5, q_lbl, g_lbl)
fmt("DTW-perchan-avg (z-norm-perchan)", m5, pc5)

# ── Summary table ─────────────────────────────────────────────────────────────
print("\n\nSUMMARY TABLE:")
hdr = "%-38s %6s %6s %6s %7s %7s %8s %8s %7s"
print(hdr % ('Method', 'P@1', 'P@3', 'P@5', 'Hit@3', 'Hit@5', 'Perf@3', 'Perf@5', 'kNN@5'))
print("-" * 108)
row = "%-38s %6.1f %6.1f %6.1f %7.1f %7.1f %8.1f %8.1f %7.1f"
for name, m in [
    ("Euclidean-raw", m1),
    ("Euclidean-znorm-perchan", m2),
    ("Cosine-znorm-perchan", m3),
    ("DTW-multichan (z-norm-global)", m4),
    ("DTW-perchan-avg (z-norm-perchan)", m5),
]:
    print(row % (name, m['P@1'], m['P@3'], m['P@5'],
                 m['Hit@3'], m['Hit@5'], m['Perfect@3'], m['Perfect@5'], m['kNN@5']))

print("\nDone.")
