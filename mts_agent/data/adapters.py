"""
Dataset adapters and shape inference utilities.
"""
from __future__ import annotations

import json
import os
import re
import warnings
from typing import Callable, Dict, Optional, Tuple

import numpy as np


DATA_ARRAY_KEYS = ("X", "data", "samples", "time_series")
LABEL_ARRAY_KEYS = ("y", "label", "labels", "target", "targets")


def ensure_channel_first(ts_data):
    ts_data = np.asarray(ts_data, dtype=np.float32)

    if ts_data.ndim == 1:
        return ts_data.reshape(1, -1)

    if ts_data.ndim != 2:
        raise ValueError(f"Expected 1D or 2D time series, got shape {ts_data.shape}")

    if ts_data.shape[0] <= ts_data.shape[1]:
        return ts_data
    return ts_data.T


def infer_dataset_name(dataset_path, dataset_name=None):
    if dataset_name and dataset_name != "General_TS":
        return dataset_name

    path_name = os.path.basename(os.path.normpath(dataset_path))
    stem, _ = os.path.splitext(path_name)
    return stem or "General_TS"


def _extract_first_available(mapping, candidate_keys, value_name, source_name):
    for key in candidate_keys:
        if key in mapping:
            return mapping[key]
    if value_name == "time-series array":
        raise ValueError(f"Could not find {value_name} in {source_name}. Keys: {sorted(mapping.keys())}")
    return None


def _dataset_match_tokens(dataset_path: str, dataset_name: Optional[str]) -> set[str]:
    candidates = [dataset_name or ""]
    normalized_path = os.path.normpath(dataset_path)
    base_name = os.path.basename(normalized_path)
    stem, _ = os.path.splitext(base_name)
    parent_name = os.path.basename(os.path.dirname(normalized_path))
    candidates.extend([normalized_path, base_name, stem, parent_name])

    tokens = set()
    for value in candidates:
        if not value:
            continue
        lowered = value.lower()
        tokens.add(lowered)
        tokens.update(token for token in re.split(r"[^a-z0-9]+", lowered) if token)
    return tokens


def load_npy_or_npz(file_path):
    loaded = np.load(file_path, allow_pickle=True)

    if isinstance(loaded, np.ndarray):
        return loaded, None

    if hasattr(loaded, "files"):
        try:
            keys = set(loaded.files)
            data = _extract_first_available(loaded, DATA_ARRAY_KEYS, "time-series array", file_path)
            labels = _extract_first_available(loaded, LABEL_ARRAY_KEYS, "label array", file_path)
            return data, labels
        finally:
            loaded.close()

    if isinstance(loaded, dict):
        data = _extract_first_available(loaded, DATA_ARRAY_KEYS, "time-series array", file_path)
        labels = _extract_first_available(loaded, LABEL_ARRAY_KEYS, "label array", file_path)
        return data, labels

    raise ValueError(f"Unsupported file format: {file_path}")


def load_folder_dataset(folder_path, mode='train'):
    split_candidates = {
        'train': [('X_train.npy', 'y_train.npy')],
        'valid': [('X_valid.npy', 'y_valid.npy'), ('X_test.npy', 'y_test.npy')],
        'test': [('X_test.npy', 'y_test.npy'), ('X_valid.npy', 'y_valid.npy')]
    }

    for x_name, y_name in split_candidates.get(mode, split_candidates['train']):
        x_path = os.path.join(folder_path, x_name)
        y_path = os.path.join(folder_path, y_name)
        if os.path.exists(x_path):
            data = np.load(x_path, mmap_mode='r')
            labels = np.load(y_path, allow_pickle=True) if os.path.exists(y_path) else None
            if labels is None and mode in {'train', 'valid', 'test'}:
                warnings.warn(f"Labels file is missing for split '{mode}' in {folder_path}: expected {y_name}")
            return data, labels

    npy_files = [f for f in os.listdir(folder_path) if f.endswith('.npy') or f.endswith('.npz')]
    if len(npy_files) == 1:
        return load_npy_or_npz(os.path.join(folder_path, npy_files[0]))

    raise FileNotFoundError(f"Could not infer dataset files from folder: {folder_path}")


AdapterFn = Callable[[str, str], Tuple[np.ndarray, Optional[np.ndarray]]]


def _generic_adapter(dataset_path: str, mode: str):
    if os.path.isdir(dataset_path):
        return load_folder_dataset(dataset_path, mode=mode)
    if os.path.isfile(dataset_path):
        return load_npy_or_npz(dataset_path)
    raise FileNotFoundError(f"Dataset path not found: {dataset_path}")


DATASET_ADAPTERS: Dict[str, AdapterFn] = {
    "finger": _generic_adapter,
    "atrial": _generic_adapter,
    "natops": _generic_adapter,
    "generic": _generic_adapter,
}


def register_dataset_adapter(key: str, adapter: AdapterFn):
    """Register or override a dataset adapter by key."""
    normalized_key = key.strip().lower()
    if not normalized_key:
        raise ValueError("Adapter key must be a non-empty string")
    DATASET_ADAPTERS[normalized_key] = adapter


def dataset_adapter(key: str):
    """Decorator form of `register_dataset_adapter`."""
    def decorator(func: AdapterFn):
        register_dataset_adapter(key, func)
        return func
    return decorator


def resolve_dataset_adapter(dataset_path: str, dataset_name: Optional[str] = None) -> AdapterFn:
    tokens = _dataset_match_tokens(dataset_path, dataset_name)
    for key, adapter in DATASET_ADAPTERS.items():
        if key.lower() in tokens:
            return adapter
    return _generic_adapter


def load_dataset_by_adapter(dataset_path: str, mode='train', dataset_name: Optional[str] = None):
    adapter = resolve_dataset_adapter(dataset_path, dataset_name)
    return adapter(dataset_path, mode)


def infer_ts_input_dim_from_array(sample) -> int:
    sample = ensure_channel_first(sample)
    return int(sample.shape[0])


def infer_ts_input_dim_from_jsonl(data_path: str) -> Optional[int]:
    if not os.path.exists(data_path):
        return None
    with open(data_path, 'r', encoding='utf-8') as f:
        for line_no, line in enumerate(f, start=1):
            if not line.strip():
                continue
            try:
                item = json.loads(line)
            except json.JSONDecodeError as exc:
                raise ValueError(f"Invalid JSON on line {line_no} in {data_path}") from exc
            if not isinstance(item, dict):
                raise ValueError(f"Expected JSON object on line {line_no} in {data_path}, got {type(item).__name__}")
            if 'time_series' in item:
                try:
                    return infer_ts_input_dim_from_array(item['time_series'])
                except Exception as exc:
                    raise ValueError(f"Invalid time_series on line {line_no} in {data_path}") from exc
    return None


def infer_ts_input_dim(data_path: Optional[str] = None, raw_data_path: Optional[str] = None, dataset_name: Optional[str] = None) -> Optional[int]:
    if data_path:
        try:
            dim = infer_ts_input_dim_from_jsonl(data_path)
            if dim is not None:
                return dim
        except (OSError, ValueError) as exc:
            warnings.warn(f"Failed to infer ts_dim from processed data {data_path}: {exc}")

    if raw_data_path and os.path.exists(raw_data_path):
        try:
            data, _ = load_dataset_by_adapter(raw_data_path, mode='train', dataset_name=dataset_name)
            if len(data) > 0:
                return infer_ts_input_dim_from_array(data[0])
        except (OSError, ValueError, IndexError, KeyError) as exc:
            warnings.warn(f"Failed to infer ts_dim from raw data {raw_data_path}: {exc}")

    return None