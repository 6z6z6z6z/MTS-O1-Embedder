"""
Forecasting Dataset — sliding-window loader for standard TS forecasting benchmarks.

Supports CSV files in the ETT / Weather / Traffic format:
    date, feat1, feat2, ..., featN

Each sample is a (history, future) pair created by a sliding window.
The dataset is compatible with MTSDataset's dict output format so the
existing MultimodalCollator, trainer, and retrieval eval all work unchanged.
"""
import os
from typing import Optional

import numpy as np
import pandas as pd
import torch
from torch.utils.data import Dataset

from mts_agent.data.augmentations import TimeSeriesAugmentor


def _load_csv_features(csv_path: str, feature_cols: Optional[list] = None) -> np.ndarray:
    """Load a CSV file and return a float32 array of shape [T, C].

    The 'date' column (if present) is dropped automatically.
    ``feature_cols`` can restrict which columns to use (by name or index).
    """
    df = pd.read_csv(csv_path)
    # Drop non-numeric columns (date/timestamp)
    df = df.select_dtypes(include=[np.number])
    if feature_cols is not None:
        if isinstance(feature_cols[0], int):
            df = df.iloc[:, feature_cols]
        else:
            df = df[feature_cols]
    return df.values.astype(np.float32)  # [T, C]


class ForecastingDataset(Dataset):
    """Sliding-window dataset for univariate / multivariate forecasting.

    Returns samples in the same dict format as MTSDataset so the existing
    collator and trainer can be used without modification:
        time_series      : [C, history_len]  — input history (channel-first)
        time_series_view2: [C, history_len]  — safe-augmented view for gallery
        future           : [C, forecast_horizon] — ground-truth future
        id, dataset, context, teacher_thought, thought, label (all compatible)

    Args:
        csv_path:        path to CSV file with numeric columns
        history_len:     number of input timesteps
        forecast_horizon: number of future timesteps to predict
        stride:          sliding window stride (1 = dense, history_len = non-overlapping)
        start_idx:       first valid window start index (inclusive)
        end_idx:         last valid window start index (exclusive); None = end of data
        feature_cols:    list of column names or int indices to use; None = all numeric
        augment:         whether to apply augmentation to the history view
        dataset_name:    string name for logging
    """

    def __init__(
        self,
        csv_path: str,
        history_len: int = 96,
        forecast_horizon: int = 96,
        stride: int = 1,
        start_idx: int = 0,
        end_idx: Optional[int] = None,
        feature_cols: Optional[list] = None,
        augment: bool = False,
        dataset_name: str = "",
        use_freq_features: bool = False,
        use_decomp_features: bool = False,
        decomp_kernel: int = 25,
        context_file: Optional[str] = None,
    ) -> None:
        self.history_len = history_len
        self.forecast_horizon = forecast_horizon
        self.use_freq_features = use_freq_features
        self.use_decomp_features = use_decomp_features
        self.decomp_kernel = decomp_kernel
        self.dataset_name = dataset_name or os.path.splitext(os.path.basename(csv_path))[0]

        # Optional per-window context strings (key = str(window_start_idx))
        self.context_map: dict = {}
        if context_file and os.path.exists(context_file):
            import json
            with open(context_file) as f:
                raw = json.load(f)
            # Keys may be stored as strings in JSON; convert to int for fast lookup
            self.context_map = {int(k): v for k, v in raw.items()}
            print(f"  Loaded {len(self.context_map)} context strings from {context_file}")

        data = _load_csv_features(csv_path, feature_cols)  # [T, C]
        total_len = len(data)
        window_size = history_len + forecast_horizon

        # Compute the range of valid window start positions
        max_start = total_len - window_size  # last valid start (inclusive)
        if max_start < 0:
            raise ValueError(
                f"Dataset too short ({total_len} steps) for "
                f"history_len={history_len} + forecast_horizon={forecast_horizon}."
            )

        actual_end = min(end_idx if end_idx is not None else max_start + 1, max_start + 1)
        actual_start = max(start_idx, 0)

        # Build index list of window start positions
        self.indices = list(range(actual_start, actual_end, stride))
        self.data = data  # [T, C]
        self.num_channels = data.shape[1]

        # Per-series statistics for instance normalization (RevIN-style)
        # We normalize each window independently at __getitem__ time.

        self.augmentor = TimeSeriesAugmentor(prob=0.5) if augment else None
        self.safe_augmentor = TimeSeriesAugmentor(
            prob=0.5, sigma=0.03, scale_sigma=0.1, channel_dropout_prob=0.1,
            window_slice_min_ratio=1.0,
        ) if augment else None

        print(
            f"ForecastingDataset '{self.dataset_name}': "
            f"{len(self.indices)} windows, C={self.num_channels}, "
            f"T={history_len}+{forecast_horizon}, stride={stride}"
        )

    def __len__(self) -> int:
        return len(self.indices)

    def __getitem__(self, idx: int) -> dict:
        start = self.indices[idx]
        window = self.data[start : start + self.history_len + self.forecast_horizon]  # [T+H, C]

        # Channel-first
        history = torch.as_tensor(window[: self.history_len].T, dtype=torch.float32)   # [C, T]
        future = torch.as_tensor(window[self.history_len :].T, dtype=torch.float32)    # [C, H]

        # Instance normalisation on history (zero-mean, unit-std per channel)
        # Applied to both history and future using history statistics so the
        # forecasting head learns to predict in normalized space.
        mean = history.mean(dim=1, keepdim=True)        # [C, 1]
        std = history.std(dim=1, keepdim=True).clamp(min=1.0)
        history = (history - mean) / std
        future = (future - mean) / std
        hist_mean = mean   # [C, 1] — saved for RAF denormalization
        hist_std = std     # [C, 1]

        # Frequency-domain features (TF-C inspired): append FFT magnitude as extra channels
        # history: [C, T] → fft_mag: [C, T//2+1] → interpolate → [C, T] → cat → [2C, T]
        if self.use_freq_features:
            fft_mag = torch.abs(torch.fft.rfft(history, dim=1))          # [C, T//2+1]
            fft_mean = fft_mag.mean(dim=1, keepdim=True)
            fft_std = fft_mag.std(dim=1, keepdim=True).clamp(min=1e-6)
            fft_mag_norm = (fft_mag - fft_mean) / fft_std                # [C, T//2+1]
            fft_interp = torch.nn.functional.interpolate(
                fft_mag_norm.unsqueeze(0), size=self.history_len,
                mode='linear', align_corners=False,
            ).squeeze(0)                                                   # [C, T]
            history_aug = torch.cat([history, fft_interp], dim=0)        # [2C, T]
        elif self.use_decomp_features:
            # CoST-style time-domain decomposition: trend (moving average) + seasonal (residual)
            # history: [C, T] (already instance-normed)
            k = self.decomp_kernel
            # Pad both sides to preserve length after convolution
            pad = k // 2
            h_padded = torch.nn.functional.pad(
                history.unsqueeze(0), (pad, pad), mode='replicate'
            ).squeeze(0)                                                   # [C, T+k-1]
            # Uniform moving average via unfold
            trend = h_padded.unfold(1, k, 1).mean(dim=-1)                # [C, T]
            seasonal = history - trend                                     # [C, T]
            # Instance-normalize each component independently
            t_std = trend.std(dim=1, keepdim=True).clamp(min=1e-6)
            trend_norm = (trend - trend.mean(dim=1, keepdim=True)) / t_std
            s_std = seasonal.std(dim=1, keepdim=True).clamp(min=1e-6)
            seasonal_norm = (seasonal - seasonal.mean(dim=1, keepdim=True)) / s_std
            history_aug = torch.cat([history, trend_norm, seasonal_norm], dim=0)  # [3C, T]
        else:
            history_aug = history                                          # [C, T]

        # Augmentation on history only (future must stay clean for loss)
        if self.augmentor is not None:
            ts_view1 = self.augmentor(history_aug)
            ts_view2 = self.safe_augmentor(history_aug)
        else:
            ts_view1 = history_aug
            ts_view2 = history_aug

        # Full trajectory for asymmetric bi-encoder gallery encoder
        full_ts = torch.cat([history, future], dim=1)  # [C, T+H] — normalized

        return {
            "id": f"{self.dataset_name}_{start}",
            "dataset": self.dataset_name,
            "time_series": ts_view1,        # [C, T] or [2C, T] — view1 (query)
            "time_series_view2": ts_view2,  # [C, T] or [2C, T] — view2 (gallery)
            "future": future,               # [C, H] — ground truth (no augmentation)
            "full_ts": full_ts,             # [C, T+H] — history+future for asymmetric bi-encoder
            "hist_mean": hist_mean,         # [C, 1] — history mean for RAF denorm
            "hist_std": hist_std,           # [C, 1] — history std for RAF denorm
            "context": self.context_map.get(start, ""),
            "teacher_thought": "",
            "thought": "",
            "label": "",
        }


def build_forecasting_splits(
    csv_path: str,
    history_len: int = 96,
    forecast_horizon: int = 96,
    stride: int = 1,
    gallery_stride: int = 1,
    train_end_ratio: float = 0.6,
    val_end_ratio: float = 0.8,
    feature_cols: Optional[list] = None,
    augment_train: bool = True,
    use_freq_features: bool = False,
    use_decomp_features: bool = False,
    decomp_kernel: int = 25,
    context_file: Optional[str] = None,
    val_stride: int = 1,
) -> tuple:
    """Create train / val / test / gallery ForecastingDataset instances from one CSV.

    Temporal splits are used (no shuffling) to prevent look-ahead bias:
        train   : [0,              train_end_ratio)  — stride=stride (may be sparse)
        val     : [train_end_ratio, val_end_ratio)   — stride=1
        test    : [val_end_ratio,   1.0)             — stride=1
        gallery : [0,              train_end_ratio)  — stride=gallery_stride (always dense for RAF)

    ``gallery_stride`` is typically 1 (dense), allowing RAF to retrieve from all
    training windows even when training uses a larger stride for cleaner contrastive
    signal.  When gallery_stride == stride, gallery_ds is the same object as train_ds
    (no extra memory).

    Returns:
        (train_dataset, val_dataset, test_dataset, gallery_dataset)
    """
    # Count total valid windows to compute split boundaries
    data = _load_csv_features(csv_path, feature_cols)
    total_windows = max(0, len(data) - history_len - forecast_horizon + 1)

    train_end = int(total_windows * train_end_ratio)
    val_end = int(total_windows * val_end_ratio)

    dataset_name = os.path.splitext(os.path.basename(csv_path))[0]

    train_ds = ForecastingDataset(
        csv_path, history_len, forecast_horizon, stride,
        start_idx=0, end_idx=train_end,
        feature_cols=feature_cols, augment=augment_train,
        dataset_name=dataset_name, use_freq_features=use_freq_features,
        use_decomp_features=use_decomp_features, decomp_kernel=decomp_kernel,
        context_file=context_file,
    )
    val_ds = ForecastingDataset(
        csv_path, history_len, forecast_horizon, stride=val_stride,
        start_idx=train_end, end_idx=val_end,
        feature_cols=feature_cols, augment=False,
        dataset_name=dataset_name, use_freq_features=use_freq_features,
        use_decomp_features=use_decomp_features, decomp_kernel=decomp_kernel,
        context_file=context_file,
    )
    test_ds = ForecastingDataset(
        csv_path, history_len, forecast_horizon, stride=1,
        start_idx=val_end, end_idx=None,
        feature_cols=feature_cols, augment=False,
        dataset_name=dataset_name, use_freq_features=use_freq_features,
        use_decomp_features=use_decomp_features, decomp_kernel=decomp_kernel,
        context_file=context_file,
    )
    # Gallery: always augment=False (clean embeddings for retrieval)
    if gallery_stride == stride:
        gallery_ds = train_ds  # reuse; no extra memory
    else:
        gallery_ds = ForecastingDataset(
            csv_path, history_len, forecast_horizon, stride=gallery_stride,
            start_idx=0, end_idx=train_end,
            feature_cols=feature_cols, augment=False,
            dataset_name=dataset_name, use_freq_features=use_freq_features,
        use_decomp_features=use_decomp_features, decomp_kernel=decomp_kernel,
        context_file=context_file,
        )
    return train_ds, val_ds, test_ds, gallery_ds
