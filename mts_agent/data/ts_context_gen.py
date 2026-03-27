"""
Programmatic context generation from raw time series data.

For a universal TS embedder, the context prompt should describe WHAT the signal
looks like (not just what dataset it comes from).  This module provides:

  - generate_natops_context: NATOPS-specific (uses known channel anatomy)
  - generate_generic_context: dataset-agnostic statistical description

Both functions accept a [C, T] numpy array and return a short (~30-60 token)
natural-language string that is unique per sample and computable at inference.
"""

from __future__ import annotations
import numpy as np


# ── NATOPS channel map ────────────────────────────────────────────────────────
# 24 channels = 8 body landmarks × 3D (X=lateral, Y=vertical, Z=fore-aft)
# Landmark order (UEA NATOPS): HandLeft, HandRight, ElbowLeft, ElbowRight,
#                               WristLeft, WristRight, ThumbLeft, ThumbRight
_NATOPS_LEFT  = [0, 1, 2,  6,  7,  8, 12, 13, 14, 18, 19, 20]
_NATOPS_RIGHT = [3, 4, 5,  9, 10, 11, 15, 16, 17, 21, 22, 23]
_NATOPS_HAND_TIPS = [0, 1, 2, 3, 4, 5]   # HandLeft + HandRight (X,Y,Z each)


def generate_natops_context(ts: np.ndarray) -> str:
    """Generate a descriptive motion context for one NATOPS sample.

    Parameters
    ----------
    ts : np.ndarray, shape [24, 51]

    Returns
    -------
    str  ~30-60 tokens, unique per sample, computable at inference.
    """
    ts = np.asarray(ts, dtype=np.float32)
    C, T = ts.shape

    # ── 1. Laterality ────────────────────────────────────────────────────────
    left_act  = np.ptp(ts[_NATOPS_LEFT],  axis=1).mean()
    right_act = np.ptp(ts[_NATOPS_RIGHT], axis=1).mean()
    ratio = right_act / (left_act + 1e-8)
    if ratio > 1.3:
        laterality = "right-arm-dominant"
    elif ratio < 0.77:
        laterality = "left-arm-dominant"
    else:
        laterality = "bilateral symmetric"

    # ── 2. Dominant motion axis for hand tips ─────────────────────────────
    ht = ts[_NATOPS_HAND_TIPS]   # [6, T]
    x_var = ht[[0, 3]].var(axis=1).mean()   # X channels: lateral
    y_var = ht[[1, 4]].var(axis=1).mean()   # Y channels: vertical
    z_var = ht[[2, 5]].var(axis=1).mean()   # Z channels: fore-aft
    dom_axis = ["lateral", "vertical", "fore-aft"][int(np.argmax([x_var, y_var, z_var]))]

    # ── 3. Amplitude ─────────────────────────────────────────────────────────
    total_range = float(np.ptp(ts))
    if total_range > 2.5:
        amp = "large"
    elif total_range > 1.2:
        amp = "moderate"
    else:
        amp = "small"

    # ── 4. Temporal pattern (based on hand-tip mean trajectory) ──────────────
    ht_traj = ht.mean(axis=0)          # [T] — mean over 6 hand-tip channels
    mid = T // 2
    first_mean  = float(ht_traj[:mid].mean())
    second_mean = float(ht_traj[mid:].mean())
    diff = second_mean - first_mean

    peak_t = int(np.argmax(np.abs(ht_traj - ht_traj.mean())))
    if peak_t < T // 4:
        temporal = "peak-early then return"
    elif peak_t > 3 * T // 4:
        temporal = "rising to late peak"
    elif diff > 0.12:
        temporal = "ascending sweep"
    elif diff < -0.12:
        temporal = "descending sweep"
    else:
        temporal = "arc with mid-peak"

    # ── 5. Leading body part ─────────────────────────────────────────────────
    hand_rng  = float(np.ptp(ts[0:6],   axis=1).mean())
    elbow_rng = float(np.ptp(ts[6:12],  axis=1).mean())
    wrist_rng = float(np.ptp(ts[12:18], axis=1).mean())
    max_rng = max(hand_rng, elbow_rng, wrist_rng)
    if hand_rng >= max_rng * 0.9:
        leader = "hand tips lead"
    elif elbow_rng >= max_rng * 0.9:
        leader = "elbow-driven"
    else:
        leader = "wrist-driven"

    return (
        f"Pilot hand gesture: {laterality} motion, primarily {dom_axis}, "
        f"{amp} amplitude, {leader}. Temporal: {temporal}."
    )


def generate_generic_context(ts: np.ndarray, dataset_name: str = "") -> str:
    """Generic context for datasets without known channel semantics.

    Computes dataset-agnostic statistics: trend, variability, channel spread.

    Parameters
    ----------
    ts : np.ndarray, shape [C, T]
    dataset_name : str  (optional, used in the description prefix)

    Returns
    -------
    str  ~30-50 tokens, unique per sample.
    """
    ts = np.asarray(ts, dtype=np.float32)
    C, T = ts.shape

    # Overall signal statistics
    total_range = float(np.ptp(ts))
    ch_ranges   = np.ptp(ts, axis=1)       # [C]
    ch_vars     = ts.var(axis=1)            # [C]

    # Active vs inactive channels
    range_thresh = ch_ranges.mean() * 0.3
    n_active = int((ch_ranges > range_thresh).sum())
    active_frac = n_active / max(C, 1)

    if active_frac > 0.75:
        spread_desc = "most channels active"
    elif active_frac > 0.4:
        spread_desc = "partial channel activity"
    else:
        spread_desc = "sparse channel activity"

    # Temporal trend (global mean over all channels)
    global_mean = ts.mean(axis=0)   # [T]
    mid = T // 2
    diff = float(global_mean[mid:].mean() - global_mean[:mid].mean())
    if diff > 0.1 * total_range:
        trend = "rising trend"
    elif diff < -0.1 * total_range:
        trend = "falling trend"
    else:
        peak_t = int(np.argmax(np.abs(global_mean - global_mean.mean())))
        if T // 4 < peak_t < 3 * T // 4:
            trend = "peak in middle"
        else:
            trend = "stable or oscillating"

    # Amplitude descriptor
    if total_range > 3.0:
        amp = "high amplitude"
    elif total_range > 1.0:
        amp = "moderate amplitude"
    else:
        amp = "low amplitude"

    prefix = f"{dataset_name} signal: " if dataset_name else "Time series: "
    return f"{prefix}{spread_desc}, {amp}, {trend}."


# ── Dataset dispatcher ────────────────────────────────────────────────────────
def generate_context(ts: np.ndarray, dataset_name: str = "") -> str:
    """Route to the appropriate context generator based on dataset name."""
    name_lower = (dataset_name or "").lower()
    if "natops" in name_lower:
        return generate_natops_context(ts)
    return generate_generic_context(ts, dataset_name)
