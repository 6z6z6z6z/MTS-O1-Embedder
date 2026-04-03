"""
RAF (Retrieval-Augmented Forecasting) Evaluation
=================================================
Compares three forecasting approaches on a held-out test set:
  1. Direct Head   — ForecastingHead(embedding) → denormalize
  2. RAF@K         — top-K cosine-similar train histories,
                     weighted-average their futures → denormalize
  3. Naive Mean    — predict history mean for all horizons (strong baseline)

All metrics are computed in the ORIGINAL (denormalized) scale so results
are comparable with standard MTSF benchmarks.
"""
import torch
import torch.nn.functional as F
from torch.utils.data import DataLoader
from tqdm import tqdm


def _build_gallery(model, dataset, collator, device, batch_size, ts_only,
                   asymmetric_biencoder=False):
    """Embed all training windows; return (embs_normalized, futures_normalized).

    embs_normalized  : [N, D]     L2-normalized embeddings
    futures_normalized: [N, C, H] futures in each window's OWN normalized space

    Storing normalized futures (not original-scale) is key to correct RAF:
    averaging normalized futures and then denormalizing with the TEST window's
    stats correctly transfers relative patterns across different absolute scales.

    When asymmetric_biencoder=True, gallery embeddings are built with
    model.get_gallery_embedding(full_ts) — encoding history+future. Queries at
    test time use model.get_ts_only_embedding(history) — encoding history only.
    This asymmetry makes gallery embeddings future-informed so cosine similarity
    naturally prefers windows whose futures resemble the query's future.
    """
    loader = DataLoader(dataset, batch_size=batch_size, shuffle=False,
                        collate_fn=collator, num_workers=0)
    all_embs, all_futures = [], []

    model.eval()
    with torch.no_grad():
        for batch in tqdm(loader, desc="Building train gallery"):
            future_n = batch['future_ts'].to(device)   # [B, C, H] normalized

            if asymmetric_biencoder:
                full_ts = batch['full_ts'].to(device)  # [B, C, T+H]
                emb = model.get_gallery_embedding(full_ts)
            elif ts_only:
                ts_input = batch['ts_input'].to(device)
                emb = model.get_ts_only_embedding(ts_input)
            else:
                # Full LLM embedding using the context-only retrieval prompt
                ts_input = batch['ts_input'].to(device)
                retrieval_ids  = batch['retrieval_input_ids'].to(device)
                retrieval_mask = batch['retrieval_attention_mask'].to(device)
                emb = model.get_embedding(ts_input, retrieval_ids, attention_mask=retrieval_mask)

            all_embs.append(F.normalize(emb, dim=-1).cpu())
            all_futures.append(future_n.float().cpu())  # keep in normalized space

    return torch.cat(all_embs, dim=0), torch.cat(all_futures, dim=0)


def run_raf_eval(model, gallery_ds, test_ds, collator, device,
                 k_values=(1, 3, 5), batch_size=256, ts_only=True,
                 asymmetric_biencoder=False):
    """Full RAF evaluation.

    Args:
        model:      trained MTSEmbedder (or any model with get_ts_only_embedding / forecasting_head)
        gallery_ds: dataset whose windows form the retrieval gallery (typically train split,
                    stride=1 for dense coverage)
        test_ds:    held-out test dataset
        collator:   MultimodalCollator instance
        device:     torch.device
        k_values:   list of K values to evaluate
        batch_size: inference batch size
        ts_only:    use get_ts_only_embedding (no LLM)
        asymmetric_biencoder: use get_gallery_embedding(full_ts) for gallery construction

    Returns a dict with keys:
        n_test, naive_mse, naive_mae,
        direct_head_mse, direct_head_mae,
        raf@{k}_mse, raf@{k}_mae  for each k in k_values
    """
    fhead = getattr(model, 'forecasting_head', None)
    max_k = max(k_values)

    print("Building train gallery...")
    gallery_embs, gallery_futures = _build_gallery(
        model, gallery_ds, collator, device, batch_size=batch_size,
        ts_only=ts_only, asymmetric_biencoder=asymmetric_biencoder,
    )
    # gallery_embs   : [N, D]    (on CPU, L2-normalized)
    # gallery_futures: [N, C, H] (on CPU, original scale)
    print(f"Gallery: {gallery_embs.shape[0]} windows, emb_dim={gallery_embs.shape[1]}")

    test_loader = DataLoader(test_ds, batch_size=batch_size, shuffle=False,
                             collate_fn=collator, num_workers=0)

    # Accumulators (sum over samples; divide at end)
    head_mse_sum = 0.0
    head_mae_sum = 0.0
    naive_mse_sum = 0.0
    naive_mae_sum = 0.0
    raf_mse_sum = {k: 0.0 for k in k_values}
    raf_mae_sum = {k: 0.0 for k in k_values}
    n_total = 0

    model.eval()
    with torch.no_grad():
        for batch in tqdm(test_loader, desc="Evaluating test set"):
            ts_input  = batch['ts_input'].to(device)
            future_n  = batch['future_ts'].to(device)  # [B, C, H] normalized (test window's space)
            hist_mean = batch['hist_mean'].to(device)  # [B, C, 1]
            hist_std  = batch['hist_std'].to(device)   # [B, C, 1]

            # Ground truth in original scale
            gt = future_n.float() * hist_std.float() + hist_mean.float()  # [B, C, H]

            if ts_only:
                emb = model.get_ts_only_embedding(ts_input)
            else:
                retrieval_ids  = batch['retrieval_input_ids'].to(device)
                retrieval_mask = batch['retrieval_attention_mask'].to(device)
                emb = model.get_embedding(ts_input, retrieval_ids, attention_mask=retrieval_mask)

            emb_norm = F.normalize(emb, dim=-1)  # [B, D]
            B = emb_norm.shape[0]

            # ── Direct ForecastingHead ─────────────────────────────────────
            if fhead is not None:
                pred_n = fhead(emb.to(fhead.linear.weight.dtype)).float()  # [B, C, H] normalized
                pred_orig = pred_n * hist_std.float() + hist_mean.float()  # denormalize with TEST stats
                head_mse_sum += F.mse_loss(pred_orig, gt, reduction='sum').item() / (gt.shape[1] * gt.shape[2])
                head_mae_sum += (pred_orig - gt).abs().sum().item() / (gt.shape[1] * gt.shape[2])

            # ── Naive baseline: predict history mean ───────────────────────
            naive_pred = hist_mean.expand_as(gt)  # [B, C, H] all = hist_mean
            naive_mse_sum += F.mse_loss(naive_pred, gt, reduction='sum').item() / (gt.shape[1] * gt.shape[2])
            naive_mae_sum += (naive_pred - gt).abs().sum().item() / (gt.shape[1] * gt.shape[2])

            # ── RAF: cosine retrieval in normalized space ───────────────────
            # Key insight: gallery futures are stored in their OWN normalized space.
            # Average them to get a "relative pattern", then denormalize with TEST stats.
            # This correctly transfers "rise by X std above mean" to test window's scale.
            sims = emb_norm @ gallery_embs.T.to(device)  # [B, N_gallery]
            topk_sims, topk_idx = sims.topk(max_k, dim=-1, largest=True, sorted=True)

            for k in k_values:
                idx_k   = topk_idx[:, :k]                     # [B, k]
                sims_k  = topk_sims[:, :k]                    # [B, k]
                weights = F.softmax(sims_k, dim=-1)           # [B, k]

                # retrieved_norm: [B, k, C, H] — each in its own gallery window's normalized space
                retrieved_norm = gallery_futures[idx_k.cpu()].to(device)

                # weighted avg of normalized futures → [B, C, H]
                raf_pred_norm = (weights[:, :, None, None] * retrieved_norm).sum(dim=1)

                # denormalize using TEST window stats → original scale prediction
                raf_pred = raf_pred_norm * hist_std.float() + hist_mean.float()

                raf_mse_sum[k] += F.mse_loss(raf_pred, gt, reduction='sum').item() / (gt.shape[1] * gt.shape[2])
                raf_mae_sum[k] += (raf_pred - gt).abs().sum().item() / (gt.shape[1] * gt.shape[2])

            n_total += B

    results = {
        'n_test':         n_total,
        'naive_mse':      naive_mse_sum / n_total,
        'naive_mae':      naive_mae_sum / n_total,
        'direct_head_mse': head_mse_sum / n_total if fhead is not None else None,
        'direct_head_mae': head_mae_sum / n_total if fhead is not None else None,
    }
    for k in k_values:
        results[f'raf@{k}_mse'] = raf_mse_sum[k] / n_total
        results[f'raf@{k}_mae'] = raf_mae_sum[k] / n_total

    return results


def run_classical_raf_baselines(gallery_ds, test_ds, collator, device,
                                k_values=(1, 3, 5, 10, 20, 50, 100),
                                batch_size=256):
    """Classical retrieval baselines: Euclidean and Cosine on raw (instance-normalized) time series.

    Compared to run_raf_eval, this replaces learned embeddings with the
    flattened normalized history vector — no model needed.  Uses the same
    gallery_ds as run_raf_eval so stride/coverage is identical.
    """
    train_loader = DataLoader(gallery_ds, batch_size=batch_size, shuffle=False,
                              collate_fn=collator, num_workers=0)
    all_ts, all_futures = [], []
    for batch in tqdm(train_loader, desc="Building classical gallery"):
        all_ts.append(batch['ts_input'].float())          # [B, C, T] normalized
        all_futures.append(batch['future_ts'].float())    # [B, C, H] normalized
    gallery_ts      = torch.cat(all_ts, dim=0)            # [N, C, T]
    gallery_futures = torch.cat(all_futures, dim=0)       # [N, C, H]

    # Flatten for distance computation: [N, C*T]
    gal_flat      = gallery_ts.reshape(gallery_ts.shape[0], -1)          # [N, C*T]
    gal_flat_norm = F.normalize(gal_flat, dim=-1)                         # for cosine

    max_k = max(k_values)
    methods = ['euclidean', 'cosine']
    mse_sums = {m: {k: 0.0 for k in k_values} for m in methods}
    mae_sums = {m: {k: 0.0 for k in k_values} for m in methods}
    n_total = 0

    test_loader = DataLoader(test_ds, batch_size=batch_size, shuffle=False,
                             collate_fn=collator, num_workers=0)
    with torch.no_grad():
        for batch in tqdm(test_loader, desc="Classical eval"):
            ts_input  = batch['ts_input'].float()          # [B, C, T]
            future_n  = batch['future_ts'].float()         # [B, C, H]
            hist_mean = batch['hist_mean'].float()         # [B, C, 1]
            hist_std  = batch['hist_std'].float()          # [B, C, 1]
            gt = future_n * hist_std + hist_mean           # [B, C, H] original scale

            B = ts_input.shape[0]
            test_flat      = ts_input.reshape(B, -1).to(device)         # [B, C*T]
            test_flat_norm = F.normalize(test_flat, dim=-1)

            # ── Euclidean: similarity = -L2_distance ──────────────────────
            # Use negative distance so topk gives closest neighbors
            dists_l2 = torch.cdist(test_flat, gal_flat.to(device))      # [B, N]
            sims_euc = -dists_l2

            # ── Cosine ────────────────────────────────────────────────────
            sims_cos = test_flat_norm @ gal_flat_norm.T.to(device)      # [B, N]

            for method, sims in [('euclidean', sims_euc), ('cosine', sims_cos)]:
                topk_sims, topk_idx = sims.topk(max_k, dim=-1, largest=True, sorted=True)
                for k in k_values:
                    idx_k   = topk_idx[:, :k]
                    sims_k  = topk_sims[:, :k]
                    weights = F.softmax(sims_k, dim=-1)             # [B, k]

                    retrieved_norm = gallery_futures[idx_k.cpu()].to(device)  # [B, k, C, H]
                    pred_norm = (weights[:, :, None, None] * retrieved_norm).sum(dim=1)
                    pred = pred_norm * hist_std.to(device) + hist_mean.to(device)

                    gt_d = gt.to(device)
                    mse_sums[method][k] += F.mse_loss(pred, gt_d, reduction='sum').item() / (gt_d.shape[1] * gt_d.shape[2])
                    mae_sums[method][k] += (pred - gt_d).abs().sum().item() / (gt_d.shape[1] * gt_d.shape[2])

            n_total += B

    results = {'n_test': n_total}
    for method in methods:
        for k in k_values:
            results[f'{method}@{k}_mse'] = mse_sums[method][k] / n_total
            results[f'{method}@{k}_mae'] = mae_sums[method][k] / n_total
    return results


def print_classical_results(results, k_values=(1, 3, 5, 10, 20, 50, 100)):
    print("\n" + "=" * 60)
    print(f"  Classical Retrieval Baselines  (n_test={results['n_test']})")
    print("=" * 60)
    print(f"  {'K':<6}  {'Euclidean MSE':>14}  {'Euclidean MAE':>14}  {'Cosine MSE':>11}  {'Cosine MAE':>11}")
    print(f"  {'-'*6}  {'-'*14}  {'-'*14}  {'-'*11}  {'-'*11}")
    for k in sorted(k_values):
        print(f"  {k:<6}  {results[f'euclidean@{k}_mse']:>14.4f}  "
              f"{results[f'euclidean@{k}_mae']:>14.4f}  "
              f"{results[f'cosine@{k}_mse']:>11.4f}  "
              f"{results[f'cosine@{k}_mae']:>11.4f}")
    print("=" * 60)


def run_p2r_raf_eval(model, gallery_ds, test_ds, collator, device,
                     k_values=(1, 3, 5, 10, 20, 50, 100),
                     batch_size=256, ts_only=True):
    """Predict-then-Retrieve (P2R) RAF evaluation.

    Instead of retrieving by history-embedding similarity, predict the future
    via the forecasting head and retrieve gallery windows whose ACTUAL futures
    are most similar to the PREDICTED future.

    This implements "任务相关相似" (task-relevant similarity): the retrieved
    neighbors are those whose real futures best match what the model expects
    to happen, rather than those whose histories look similar.

    Gallery building requires NO model forward pass — only the actual futures
    are stored.  Query-time cost is: ts_encoder + fhead + cosine over N.
    """
    fhead = getattr(model, 'forecasting_head', None)
    if fhead is None:
        print("P2R skipped: no forecasting_head attached to model.")
        return {}

    max_k = max(k_values)

    # ── Build P2R gallery: actual futures only (no embedding needed) ──────
    loader = DataLoader(gallery_ds, batch_size=batch_size, shuffle=False,
                        collate_fn=collator, num_workers=0)
    all_fut_flat_norm, all_fut = [], []
    with torch.no_grad():
        for batch in tqdm(loader, desc="Building P2R gallery (futures)"):
            future_n = batch['future_ts'].float()            # [B, C, H] normalized
            flat = future_n.reshape(future_n.shape[0], -1)  # [B, C*H]
            all_fut_flat_norm.append(F.normalize(flat, dim=-1).cpu())
            all_fut.append(future_n.cpu())
    gal_fut_flat_norm = torch.cat(all_fut_flat_norm, dim=0)  # [N, C*H]
    gal_fut           = torch.cat(all_fut,           dim=0)  # [N, C, H]
    print(f"P2R Gallery: {gal_fut.shape[0]} windows, future_dim={gal_fut_flat_norm.shape[1]}")

    # ── Evaluate test set ─────────────────────────────────────────────────
    p2r_mse_sum = {k: 0.0 for k in k_values}
    p2r_mae_sum = {k: 0.0 for k in k_values}
    n_total = 0

    test_loader = DataLoader(test_ds, batch_size=batch_size, shuffle=False,
                             collate_fn=collator, num_workers=0)
    model.eval()
    with torch.no_grad():
        for batch in tqdm(test_loader, desc="P2R eval"):
            ts_input  = batch['ts_input'].to(device)
            future_n  = batch['future_ts'].to(device)
            hist_mean = batch['hist_mean'].to(device)
            hist_std  = batch['hist_std'].to(device)
            gt = future_n.float() * hist_std.float() + hist_mean.float()

            if ts_only:
                emb = model.get_ts_only_embedding(ts_input)
            else:
                retrieval_ids  = batch['retrieval_input_ids'].to(device)
                retrieval_mask = batch['retrieval_attention_mask'].to(device)
                emb = model.get_embedding(ts_input, retrieval_ids,
                                          attention_mask=retrieval_mask)

            # Predict future in normalized space
            pred_n = fhead(emb.to(fhead.linear.weight.dtype)).float()  # [B, C, H]
            B = pred_n.shape[0]

            # L2-normalize predicted future → cosine with gallery actual futures
            pred_flat_norm = F.normalize(pred_n.reshape(B, -1), dim=-1)  # [B, C*H]
            sims = pred_flat_norm @ gal_fut_flat_norm.T.to(device)        # [B, N]
            topk_sims, topk_idx = sims.topk(max_k, dim=-1, largest=True, sorted=True)

            for k in k_values:
                idx_k   = topk_idx[:, :k]
                sims_k  = topk_sims[:, :k]
                weights = F.softmax(sims_k, dim=-1)

                retrieved_norm = gal_fut[idx_k.cpu()].to(device)   # [B, k, C, H]
                raf_pred_norm  = (weights[:, :, None, None] * retrieved_norm).sum(dim=1)
                raf_pred = raf_pred_norm * hist_std.float() + hist_mean.float()

                p2r_mse_sum[k] += F.mse_loss(raf_pred, gt, reduction='sum').item() / (gt.shape[1] * gt.shape[2])
                p2r_mae_sum[k] += (raf_pred - gt).abs().sum().item() / (gt.shape[1] * gt.shape[2])

            n_total += B

    results = {'n_test': n_total}
    for k in k_values:
        results[f'p2r@{k}_mse'] = p2r_mse_sum[k] / n_total
        results[f'p2r@{k}_mae'] = p2r_mae_sum[k] / n_total
    return results


def print_p2r_results(results, k_values=(1, 3, 5, 10, 20, 50, 100)):
    print("\n" + "=" * 60)
    print(f"  Predict-then-Retrieve (P2R)  (n_test={results['n_test']})")
    print("=" * 60)
    for k in sorted(k_values):
        mse = results.get(f'p2r@{k}_mse')
        mae = results.get(f'p2r@{k}_mae')
        if mse is not None:
            print(f"  P2R@{k:<3}            — MSE: {mse:.4f}  MAE: {mae:.4f}")
    print("=" * 60)


def print_raf_results(results):
    print("\n" + "=" * 60)
    print(f"  RAF Evaluation Results  (n_test={results['n_test']})")
    print("=" * 60)
    print(f"  Naive (hist mean)  — MSE: {results['naive_mse']:.4f}  MAE: {results['naive_mae']:.4f}")
    if results.get('direct_head_mse') is not None:
        print(f"  Direct Head        — MSE: {results['direct_head_mse']:.4f}  MAE: {results['direct_head_mae']:.4f}")
    for k in sorted(k for k in results if k.startswith('raf@') and k.endswith('_mse')):
        knum = k.split('@')[1].split('_')[0]
        mse_key = f'raf@{knum}_mse'
        mae_key = f'raf@{knum}_mae'
        print(f"  RAF@{knum:<3}            — MSE: {results[mse_key]:.4f}  MAE: {results[mae_key]:.4f}")
    print("=" * 60)
