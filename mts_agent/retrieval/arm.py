"""
Adaptive Retrieval Mixer (ARM)

Inspired by TS-RAG: instead of rule-based voting over retrieved neighbors,
a small cross-attention module learns to adaptively fuse top-k neighbor
embeddings with the query embedding for classification.

Pipeline
--------
1. Build gallery embeddings from frozen MTSEmbedder (context-only prompts).
2. For each query: cosine-retrieve top-k neighbor embeddings from gallery.
3. ARM cross-attn(query [B,H], neighbors [B,k,H]) → logits [B,C].
4. Train ARM with LOO cross-entropy on the training set; backbone frozen.
"""

from __future__ import annotations

import os
import numpy as np
import torch
import torch.nn as nn
import torch.nn.functional as F
from tqdm import tqdm


# ─────────────────────────── ARM Model ───────────────────────────────────────

class AdaptiveRetrievalMixer(nn.Module):
    """
    Adaptive Retrieval Mixer.

    Cross-attends the query to its k nearest-neighbor embeddings, applies a
    residual FFN, and produces classification logits.

    Args:
        hidden_dim:   Embedding dimensionality (LLM hidden size, e.g. 896).
        num_classes:  Number of target classes.
        k:            Number of neighbors expected at inference time.
        num_heads:    Multi-head attention heads. Auto-reduced if not divisible.
        dropout:      Dropout rate inside attention and FFN.
    """

    def __init__(
        self,
        hidden_dim: int,
        num_classes: int,
        k: int = 5,
        num_heads: int = 8,
        dropout: float = 0.1,
    ):
        super().__init__()
        self.hidden_dim = hidden_dim
        self.num_classes = num_classes
        self.k = k

        # Ensure divisibility
        while hidden_dim % num_heads != 0 and num_heads > 1:
            num_heads -= 1
        self.num_heads = num_heads

        # Cross-attention: query (Q) attends to neighbors (K, V)
        self.cross_attn = nn.MultiheadAttention(
            hidden_dim, num_heads, batch_first=True, dropout=dropout
        )
        self.norm1 = nn.LayerNorm(hidden_dim)

        # Feed-forward network
        self.ffn = nn.Sequential(
            nn.Linear(hidden_dim, hidden_dim * 2),
            nn.GELU(),
            nn.Dropout(dropout),
            nn.Linear(hidden_dim * 2, hidden_dim),
        )
        self.norm2 = nn.LayerNorm(hidden_dim)

        # Classification head
        self.classifier = nn.Linear(hidden_dim, num_classes)

    def forward(
        self,
        query_emb: torch.Tensor,      # [B, H]
        neighbor_embs: torch.Tensor,  # [B, k, H]
    ):
        """
        Returns
        -------
        logits:       [B, num_classes]
        attn_weights: [B, num_heads, 1, k]  (for interpretability)
        """
        q = query_emb.unsqueeze(1)                                # [B, 1, H]
        fused, attn_w = self.cross_attn(q, neighbor_embs, neighbor_embs)
        fused = fused.squeeze(1)                                  # [B, H]

        # Residual + Norm (skip connection preserves original query info)
        out = self.norm1(query_emb + fused)
        # FFN + Residual + Norm
        out = self.norm2(out + self.ffn(out))

        return self.classifier(out), attn_w

    # ── persistence ──────────────────────────────────────────────────────────

    def save(self, path: str):
        os.makedirs(os.path.dirname(os.path.abspath(path)), exist_ok=True)
        torch.save(
            {
                "model_state": self.state_dict(),
                "hidden_dim": self.hidden_dim,
                "num_classes": self.num_classes,
                "k": self.k,
                "num_heads": self.num_heads,
            },
            path,
        )
        print(f"ARM saved to {path}")

    @classmethod
    def load(cls, path: str, device: str = "cpu") -> "AdaptiveRetrievalMixer":
        data = torch.load(path, map_location=device)
        arm = cls(
            hidden_dim=data["hidden_dim"],
            num_classes=data["num_classes"],
            k=data["k"],
            num_heads=data.get("num_heads", 8),
        )
        arm.load_state_dict(data["model_state"])
        return arm


# ─────────────────────────── retrieval helpers ───────────────────────────────

def _normed(arr: np.ndarray) -> np.ndarray:
    norms = np.linalg.norm(arr, axis=1, keepdims=True) + 1e-10
    return arr / norms


def retrieve_k_neighbors_loo(
    embeddings: np.ndarray,
    labels,
    k: int,
):
    """
    Leave-One-Out nearest-neighbor retrieval on training set.

    For each sample i, retrieves k closest neighbors (excluding itself) by
    cosine similarity.

    Returns
    -------
    list of length N, each element: (neighbor_embs [k, H], neighbor_labels [k])
    """
    emb = np.array(embeddings, dtype=np.float32)   # [N, H]
    normed = _normed(emb)
    sim = normed @ normed.T                          # [N, N]

    results = []
    N = len(emb)
    for i in range(N):
        sims_i = sim[i].copy()
        sims_i[i] = -999.0                          # exclude self (LOO)
        top_k = np.argsort(sims_i)[::-1][:k]
        results.append((emb[top_k], [labels[j] for j in top_k]))
    return results


def retrieve_k_neighbors(
    query_embs: np.ndarray,
    gallery_embs: np.ndarray,
    gallery_labels,
    k: int,
):
    """
    Retrieve top-k neighbors from gallery for each query (cosine similarity).

    Returns
    -------
    list of length Q, each element: (neighbor_embs [k, H], neighbor_labels [k])
    """
    q = np.array(query_embs,   dtype=np.float32)
    g = np.array(gallery_embs, dtype=np.float32)
    q_n = _normed(q)
    g_n = _normed(g)
    sim = q_n @ g_n.T                               # [Q, G]

    results = []
    for i in range(len(q)):
        top_k = np.argsort(sim[i])[::-1][:k]
        results.append((g[top_k], [gallery_labels[j] for j in top_k]))
    return results


# ─────────────────────────── label utilities ─────────────────────────────────

def build_label_map(labels) -> dict:
    """Build a deterministic label-str → int-index mapping."""
    unique = sorted(set(str(l) for l in labels))
    return {l: i for i, l in enumerate(unique)}


# ─────────────────────────── training ────────────────────────────────────────

def train_arm(
    arm: AdaptiveRetrievalMixer,
    train_embeddings,
    train_labels,
    label_to_idx: dict,
    k: int = 5,
    epochs: int = 100,
    lr: float = 3e-4,
    batch_size: int = 32,
    device: str = "cuda",
    save_path: str = None,
) -> AdaptiveRetrievalMixer:
    """
    Train ARM via Leave-One-Out retrieval on frozen training embeddings.

    The MTSEmbedder backbone is never updated here.  ARM is a lightweight
    plug-in on top of the fixed embedding space.

    Args:
        arm:              Initialized AdaptiveRetrievalMixer.
        train_embeddings: list / ndarray of shape [N, H].
        train_labels:     list of label strings [N].
        label_to_idx:     label-str → int index mapping.
        k:                Number of neighbors for LOO retrieval.
        epochs:           Training epochs.
        lr:               AdamW learning rate.
        batch_size:       Mini-batch size.
        device:           'cuda' or 'cpu'.
        save_path:        If set, persist the best ARM state here.

    Returns:
        ARM with best LOO-training-accuracy state loaded.
    """
    arm = arm.to(device)
    optimizer = torch.optim.AdamW(arm.parameters(), lr=lr, weight_decay=1e-4)
    scheduler = torch.optim.lr_scheduler.CosineAnnealingLR(
        optimizer, T_max=max(1, epochs), eta_min=lr * 0.05
    )

    emb_array = np.array(train_embeddings, dtype=np.float32)   # [N, H]
    label_ints = [label_to_idx[str(l)] for l in train_labels]
    N = len(train_embeddings)

    print(f"  Pre-computing LOO neighbors (k={k}, N={N})...")
    loo_results = retrieve_k_neighbors_loo(emb_array, label_ints, k)

    best_acc = 0.0
    best_state = None
    actual_k = min(k, N - 1)   # guard against tiny datasets

    arm.train()
    for epoch in range(epochs):
        total_loss = 0.0
        correct = 0
        idx_order = torch.randperm(N).tolist()

        for start in range(0, N, batch_size):
            batch_idx = idx_order[start: start + batch_size]
            B = len(batch_idx)

            query_np = np.stack([emb_array[i] for i in batch_idx])  # [B, H]
            neigh_np = np.stack([
                loo_results[i][0][:actual_k] for i in batch_idx
            ])                                                        # [B, k, H]
            tgt = [label_ints[i] for i in batch_idx]

            query_t = torch.tensor(query_np, dtype=torch.float32, device=device)
            neigh_t = torch.tensor(neigh_np, dtype=torch.float32, device=device)
            tgt_t   = torch.tensor(tgt,      dtype=torch.long,    device=device)

            optimizer.zero_grad()
            logits, _ = arm(query_t, neigh_t)
            loss = F.cross_entropy(logits, tgt_t)
            loss.backward()
            torch.nn.utils.clip_grad_norm_(arm.parameters(), 1.0)
            optimizer.step()

            total_loss += loss.item() * B
            correct    += (logits.argmax(1) == tgt_t).sum().item()

        scheduler.step()

        acc = correct / N
        if acc > best_acc:
            best_acc = acc
            best_state = {k_: v.cpu().clone() for k_, v in arm.state_dict().items()}

        if (epoch + 1) % 10 == 0 or epoch == 0:
            print(
                f"  ARM Epoch [{epoch+1:3d}/{epochs}]  "
                f"loss={total_loss/N:.4f}  LOO-train-acc={acc:.2%}"
            )

    print(f"  ARM training done.  Best LOO-train-acc: {best_acc:.2%}")
    if best_state is not None:
        arm.load_state_dict(best_state)
    if save_path:
        arm.save(save_path)
    return arm


# ─────────────────────────── evaluation ──────────────────────────────────────

def evaluate_arm(
    arm: AdaptiveRetrievalMixer,
    query_embeddings,
    query_labels,
    gallery_embeddings,
    gallery_labels,
    label_to_idx: dict,
    k: int = 5,
    device: str = "cuda",
    batch_size: int = 32,
) -> dict:
    """
    Evaluate ARM on held-out test queries against a prebuilt gallery.

    Args:
        query_embeddings:  list / ndarray [Q, H] — test set embeddings.
        gallery_embeddings: list / ndarray [G, H] — training set embeddings
                            (optionally augmented).
        label_to_idx:      label-str → int index mapping built from train labels.

    Returns:
        dict with 'accuracy', 'f1_macro', 'predictions', 'true_labels'.
    """
    from sklearn.metrics import accuracy_score, f1_score

    idx_to_label = {v: k_ for k_, v in label_to_idx.items()}
    arm = arm.to(device)
    arm.eval()

    q_arr = np.array(query_embeddings,   dtype=np.float32)
    g_arr = np.array(gallery_embeddings, dtype=np.float32)
    actual_k = min(k, len(g_arr))

    print(f"  Retrieving top-{actual_k} neighbors for {len(q_arr)} test queries...")
    neighbor_results = retrieve_k_neighbors(q_arr, g_arr, gallery_labels, actual_k)

    preds = []
    with torch.no_grad():
        for start in range(0, len(q_arr), batch_size):
            end = min(start + batch_size, len(q_arr))
            q_np = q_arr[start:end]
            n_list = []

            for i in range(start, end):
                neigh_embs, _ = neighbor_results[i]
                # Pad if gallery smaller than k (edge case)
                if len(neigh_embs) < actual_k:
                    pad = np.zeros(
                        (actual_k - len(neigh_embs), neigh_embs.shape[-1]),
                        dtype=np.float32,
                    )
                    neigh_embs = np.vstack([neigh_embs, pad])
                n_list.append(neigh_embs[:actual_k])

            q_t = torch.tensor(q_np,                  dtype=torch.float32, device=device)
            n_t = torch.tensor(np.stack(n_list),       dtype=torch.float32, device=device)

            logits, attn_w = arm(q_t, n_t)
            for p in logits.argmax(1).cpu().tolist():
                preds.append(idx_to_label.get(p, str(p)))

    refs = [str(l) for l in query_labels]
    acc  = accuracy_score(refs, preds)
    f1   = f1_score(refs, preds, average="macro", zero_division=0)
    print(f"  ARM Test  Accuracy: {acc:.2%}  Macro-F1: {f1:.2%}")
    return {"accuracy": acc, "f1_macro": f1, "predictions": preds, "true_labels": refs}
