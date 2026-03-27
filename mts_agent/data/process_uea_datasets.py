"""
UEA Multivariate ARFF → JSONL processor for universal TSEmbedder training.

Reads relational ARFF format (UEA 2018 archive), z-score normalizes per dataset,
zero-pads channels to a common dimension, and writes JSONL files compatible
with the existing MTSDataset loader.

Usage:
    python -m mts_agent.data.process_uea_datasets \
        --arff_dir mts_agent/UEA_Archive \
        --out_dir  mts_agent/data/processed \
        --target_channels 24
"""
from __future__ import annotations

import argparse
import json
import os
import sys
from typing import List, Optional, Tuple

import numpy as np


# ─── ARFF parser ─────────────────────────────────────────────────────────────

def parse_relational_arff(filepath: str) -> Tuple[np.ndarray, np.ndarray]:
    """Parse a UEA multivariate relational ARFF file.

    The UEA format encodes each sample as a quoted relational value where
    sub-instances (one per channel) are separated by the literal two-character
    sequence backslash-n (r'\\n').  Each sub-instance is a comma-separated
    list of T floating-point values.

    Returns
    -------
    X : np.ndarray, shape (N, C, T)
    y : np.ndarray, shape (N,)  — string class labels
    """
    with open(filepath, encoding="utf-8", errors="replace") as fh:
        raw = fh.read()

    # Locate @data section
    lower = raw.lower()
    data_start = lower.find("@data")
    if data_start == -1:
        raise ValueError(f"@data not found in {filepath}")

    data_section = raw[data_start + len("@data"):].strip()

    X_list: list = []
    y_list: list = []

    for line in data_section.splitlines():
        line = line.strip()
        if not line or line.startswith("%"):
            continue

        # Each line: '<relational_data>',LABEL
        # The relational data is enclosed in single quotes.
        if line.startswith("'"):
            end_quote = line.rindex("'")
            relational = line[1:end_quote]
            label_part = line[end_quote + 1:].lstrip(",").strip()
        else:
            # No quotes: comma-delimited flat row, last field is label
            parts = line.rsplit(",", 1)
            relational = parts[0]
            label_part = parts[1].strip() if len(parts) == 2 else ""

        label = label_part.strip()

        # Split channels by literal "\n" (backslash + n, not newline)
        channel_strings = relational.split("\\n")

        if len(channel_strings) == 1:
            # Flat format: no \n separators — try to detect from attribute count
            # (fall back: treat whole thing as one channel)
            channel_data = [np.array(channel_strings[0].split(","), dtype=np.float32)]
        else:
            channel_data = []
            for ch_str in channel_strings:
                ch_str = ch_str.strip()
                if ch_str:
                    vals = np.array(ch_str.split(","), dtype=np.float32)
                    channel_data.append(vals)

        X_list.append(np.stack(channel_data, axis=0))  # (C, T)
        y_list.append(label)

    X = np.stack(X_list, axis=0).astype(np.float32)  # (N, C, T)
    y = np.array(y_list)
    return X, y


def parse_dimension_arffs(folder: str, dataset: str, split: str) -> Tuple[Optional[np.ndarray], Optional[np.ndarray]]:
    """Parse per-dimension ARFF files (BasicMotionsDimension1_TRAIN.arff, …).

    Returns (X, y) or (None, None) if no per-dimension files exist.
    """
    dim = 1
    channels = []
    labels = None
    while True:
        fname = os.path.join(folder, f"{dataset}Dimension{dim}_{split}.arff")
        if not os.path.exists(fname):
            break
        with open(fname, encoding="utf-8", errors="replace") as fh:
            raw = fh.read()
        lower = raw.lower()
        data_start = lower.find("@data")
        if data_start == -1:
            break
        data_section = raw[data_start + len("@data"):].strip()
        ch_data = []
        ch_labels = []
        for line in data_section.splitlines():
            line = line.strip()
            if not line or line.startswith("%"):
                continue
            parts = line.rsplit(",", 1)
            if len(parts) != 2:
                continue
            raw_vals = [v if v != "?" else "0" for v in parts[0].split(",")]
            vals = np.array(raw_vals, dtype=np.float32)
            ch_data.append(vals)
            ch_labels.append(parts[1].strip())
        if not ch_data:
            break
        # Handle variable-length series by padding to max length
        max_len = max(len(v) for v in ch_data)
        padded = [np.pad(v, (0, max_len - len(v))) if len(v) < max_len else v for v in ch_data]
        channels.append(np.stack(padded, axis=0))  # (N, T)
        if labels is None:
            labels = np.array(ch_labels)
        dim += 1

    if not channels:
        return None, None
    # Align channels: different dimension files may have slightly different N or T
    min_n = min(ch.shape[0] for ch in channels)
    max_t = max(ch.shape[1] for ch in channels)
    channels = [np.pad(ch[:min_n], ((0, 0), (0, max_t - ch.shape[1]))) if ch.shape[1] < max_t
                else ch[:min_n] for ch in channels]
    X = np.stack(channels, axis=1)  # (N, C, T)
    labels = labels[:min_n] if labels is not None else labels
    return X.astype(np.float32), labels


def load_dataset(folder: str, dataset: str, split: str) -> Tuple[np.ndarray, np.ndarray]:
    """Load a UEA dataset split, trying per-dimension files first, then main ARFF."""
    X, y = parse_dimension_arffs(folder, dataset, split)
    if X is not None:
        return X, y
    main_arff = os.path.join(folder, f"{dataset}_{split.upper()}.arff")
    if os.path.exists(main_arff):
        return parse_relational_arff(main_arff)
    raise FileNotFoundError(f"No ARFF files found for {dataset}/{split} in {folder}")


# ─── Normalization ────────────────────────────────────────────────────────────

def zscore_fit(X_train: np.ndarray) -> Tuple[np.ndarray, np.ndarray]:
    """Compute dataset-level z-score statistics from training split.

    Returns mean and std with shape (1, C, 1) for broadcasting over (N, C, T).
    """
    mean = X_train.mean(axis=(0, 2), keepdims=True)  # (1, C, 1)
    std  = X_train.std(axis=(0, 2), keepdims=True)    # (1, C, 1)
    std  = np.where(std < 1e-8, 1.0, std)
    return mean, std


def zscore_transform(X: np.ndarray, mean: np.ndarray, std: np.ndarray) -> np.ndarray:
    return (X - mean) / std


# ─── Channel padding ──────────────────────────────────────────────────────────

def pad_channels(X: np.ndarray, target_channels: int) -> np.ndarray:
    """Zero-pad channels from C to target_channels.  Truncates if C > target.
    Pass target_channels <= 0 to skip padding (return X unchanged)."""
    N, C, T = X.shape
    if target_channels <= 0 or C == target_channels:
        return X
    if C > target_channels:
        return X[:, :target_channels, :]
    pad = np.zeros((N, target_channels - C, T), dtype=np.float32)
    return np.concatenate([X, pad], axis=1)


# ─── Context generation ───────────────────────────────────────────────────────

DATASET_CONTEXTS = {
    "NATOPS": (
        "24-channel 3D motion capture data of a pilot's hand gesture signal used in aviation communication. "
        "Channels encode spatial coordinates for left/right hand tips, elbows, wrists, and thumbs."
    ),
    "BasicMotions": (
        "6-channel smartwatch sensor data (3D accelerometer + 3D gyroscope) recording daily physical activities "
        "at 10 Hz over 10 seconds. Activities: walking, resting, running, badminton."
    ),
    "RacketSports": (
        "6-channel smartwatch sensor data (3D accelerometer + 3D gyroscope) recording racket sport strokes "
        "at 10 Hz over 3 seconds. Sports and stroke types: badminton/squash, smash/clear."
    ),
    "Epilepsy": (
        "3-channel tri-axial accelerometer data (X, Y, Z axes) from a wrist sensor "
        "recording four activities including seizure simulation."
    ),
    "ArticularyWordRecognition": (
        "9-channel electromagnetic articulograph (EMA) data measuring tongue and lip movement "
        "during spoken word articulation. Channels record spatial positions of articulators."
    ),
    "UWaveGestureLibrary": (
        "3-channel accelerometer data (X, Y, Z axes) recording 8 distinct hand gesture patterns. "
        "Time series length 315 at uniform sampling rate."
    ),
    "JapaneseVowels": (
        "12-channel linear prediction coefficients (LPC features) extracted from speech recordings "
        "of 9 Japanese-male speakers pronouncing vowels. Captures vocal tract resonance patterns."
    ),
    "Libras": (
        "2-channel motion capture data tracking hand/arm position (X, Y coordinates) "
        "during 15 classes of Brazilian sign language gestures."
    ),
    "Cricket": (
        "6-channel accelerometer and gyroscope data from a wrist sensor recording "
        "12 types of cricket umpire signals and hand gestures."
    ),
    "HandMovementDirection": (
        "10-channel magnetoencephalography (MEG) data recording brain activity "
        "during four directions of hand movement preparation."
    ),
}

DEFAULT_CONTEXT_TEMPLATE = (
    "{C}-channel multivariate time series data with {T} time steps. "
    "Dataset: {dataset}. {C} sensor channels recording temporal patterns across {num_classes} categories."
)


def get_context(dataset: str, C: int, T: int, num_classes: int) -> str:
    if dataset in DATASET_CONTEXTS:
        return DATASET_CONTEXTS[dataset]
    return DEFAULT_CONTEXT_TEMPLATE.format(
        C=C, T=T, dataset=dataset, num_classes=num_classes
    )


# ─── JSONL writer ─────────────────────────────────────────────────────────────

def write_jsonl(
    X: np.ndarray,
    y: np.ndarray,
    out_path: str,
    context: str,
    dataset: str,
    split: str,
    orig_channels: int,
):
    """Write samples to JSONL in the format expected by MTSDataset."""
    os.makedirs(os.path.dirname(out_path), exist_ok=True)
    written = 0
    with open(out_path, "w", encoding="utf-8") as fh:
        for i, (ts, label) in enumerate(zip(X, y)):
            record = {
                "id": f"{dataset}_{split}_{i:05d}",
                "dataset": dataset,
                "orig_channels": orig_channels,
                "time_series": ts.tolist(),   # (C_padded, T) nested list
                "label": str(label),
                "context": context,
                "teacher_thought": "",
                "thought": "",
                "thought_source": "none",
            }
            fh.write(json.dumps(record, ensure_ascii=False) + "\n")
            written += 1
    return written


# ─── Main ─────────────────────────────────────────────────────────────────────

# Datasets to process with their folder names in the UEA archive
DATASETS_TO_PROCESS = [
    "BasicMotions",
    "RacketSports",
    "Epilepsy",
    "ArticularyWordRecognition",
    "UWaveGestureLibrary",
    "JapaneseVowels",
    "Libras",
    "Cricket",
]


def process_dataset(
    dataset: str,
    arff_dir: str,
    out_dir: str,
    target_channels: int,
) -> dict:
    folder = os.path.join(arff_dir, dataset)
    if not os.path.isdir(folder):
        return {"dataset": dataset, "status": "skipped (folder not found)"}

    try:
        X_train, y_train = load_dataset(folder, dataset, "TRAIN")
        X_test,  y_test  = load_dataset(folder, dataset, "TEST")
    except FileNotFoundError as exc:
        return {"dataset": dataset, "status": f"error: {exc}"}

    N_train, C, T = X_train.shape
    N_test = X_test.shape[0]
    num_classes = len(np.unique(y_train))

    # z-score: fit on train, apply to both
    mean, std = zscore_fit(X_train)
    X_train = zscore_transform(X_train, mean, std)
    X_test  = zscore_transform(X_test,  mean, std)

    # Pad channels to target
    X_train = pad_channels(X_train, target_channels)
    X_test  = pad_channels(X_test,  target_channels)

    context = get_context(dataset, C, T, num_classes)

    train_path = os.path.join(out_dir, f"{dataset.lower()}_train.jsonl")
    test_path  = os.path.join(out_dir, f"{dataset.lower()}_test.jsonl")

    n_tr = write_jsonl(X_train, y_train, train_path, context, dataset, "train", C)
    n_te = write_jsonl(X_test,  y_test,  test_path,  context, dataset, "test",  C)

    return {
        "dataset": dataset,
        "status": "ok",
        "shape_train": f"({N_train}, {C}, {T})",
        "shape_test":  f"({N_test}, {C}, {T})",
        "num_classes": num_classes,
        "train_samples": n_tr,
        "test_samples":  n_te,
        "train_path": train_path,
        "test_path":  test_path,
    }


def main():
    parser = argparse.ArgumentParser(description="Process UEA datasets to JSONL")
    parser.add_argument("--arff_dir", default="UEA_Archive/Multivariate2018_arff/Multivariate_arff")
    parser.add_argument("--out_dir",  default="mts_agent/data/processed")
    parser.add_argument("--target_channels", type=int, default=24)
    parser.add_argument("--datasets", nargs="*", default=None,
                        help="Specific datasets to process (default: all in DATASETS_TO_PROCESS)")
    args = parser.parse_args()

    datasets = args.datasets if args.datasets else DATASETS_TO_PROCESS
    print(f"Processing {len(datasets)} datasets to {args.out_dir}  (target_channels={args.target_channels})")
    print()

    results = []
    for ds in datasets:
        r = process_dataset(ds, args.arff_dir, args.out_dir, args.target_channels)
        results.append(r)
        status = r.get("status", "?")
        if status == "ok":
            print(f"  OK {ds:30s}  train={r['train_samples']:4d}  test={r['test_samples']:4d}  "
                  f"shape={r['shape_train']}  classes={r['num_classes']}")
        else:
            print(f"  ERR {ds:30s}  {status}")

    print()
    ok = sum(1 for r in results if r.get("status") == "ok")
    print(f"Done: {ok}/{len(results)} datasets processed successfully.")
    return 0 if ok == len(results) else 1


if __name__ == "__main__":
    sys.exit(main())
