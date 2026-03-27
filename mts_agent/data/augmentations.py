import numpy as np
import torch


class TimeSeriesAugmentor:
    """
    Apply augmentations to time series data to prevent overfitting on small datasets.
    """
    def __init__(self, prob=0.5, sigma=0.03, scale_sigma=0.1, channel_dropout_prob=0.1,
                 window_slice_min_ratio=0.5):
        if not 0.0 <= prob <= 1.0:
            raise ValueError(f"prob must be in [0, 1], got {prob}")
        if sigma < 0.0:
            raise ValueError(f"sigma must be non-negative, got {sigma}")
        if scale_sigma < 0.0:
            raise ValueError(f"scale_sigma must be non-negative, got {scale_sigma}")
        self.prob = prob
        self.sigma = sigma  # For jitter
        self.scale_sigma = scale_sigma  # For scaling
        self.channel_dropout_prob = channel_dropout_prob  # Per-channel zero-out probability
        self.window_slice_min_ratio = window_slice_min_ratio  # Minimum window fraction to retain

    def _normalize_input(self, x):
        if torch.is_tensor(x):
            if x.ndim not in (1, 2):
                raise ValueError(f"time series tensor must be 1D or 2D, got shape={tuple(x.shape)}")
            return x, True

        x_np = np.asarray(x, dtype=np.float32)
        if x_np.ndim not in (1, 2):
            raise ValueError(f"time series array must be 1D or 2D, got shape={x_np.shape}")
        return x_np, False

    def _should_apply(self, x, is_tensor):
        if is_tensor:
            return bool(torch.rand((), device=x.device).item() < self.prob)
        return bool(np.random.random() < self.prob)

    def _channelwise_shape(self, x):
        if x.ndim == 1:
            return ()
        return (x.shape[0], 1)

    def jitter(self, x, is_tensor=None):
        if is_tensor is None:
            x, is_tensor = self._normalize_input(x)
        if self._should_apply(x, is_tensor):
            if is_tensor:
                return x + torch.randn_like(x) * self.sigma
            noise = np.random.normal(loc=0.0, scale=self.sigma, size=x.shape).astype(x.dtype, copy=False)
            return x + noise
        return x

    def scaling(self, x, is_tensor=None):
        if is_tensor is None:
            x, is_tensor = self._normalize_input(x)
        if self._should_apply(x, is_tensor):
            factor_shape = self._channelwise_shape(x)
            if is_tensor:
                if factor_shape:
                    factor = 1.0 + torch.randn(factor_shape, device=x.device, dtype=x.dtype) * self.scale_sigma
                else:
                    factor = 1.0 + torch.randn((), device=x.device, dtype=x.dtype) * self.scale_sigma
            else:
                factor = np.random.normal(loc=1.0, scale=self.scale_sigma, size=factor_shape or None).astype(x.dtype, copy=False)
            return x * factor
        return x

    def shift(self, x, is_tensor=None):
        if is_tensor is None:
            x, is_tensor = self._normalize_input(x)
        if self._should_apply(x, is_tensor):
            shift_shape = self._channelwise_shape(x)
            if is_tensor:
                if shift_shape:
                    shift_val = torch.randn(shift_shape, device=x.device, dtype=x.dtype) * self.sigma
                else:
                    shift_val = torch.randn((), device=x.device, dtype=x.dtype) * self.sigma
            else:
                shift_val = np.random.normal(loc=0.0, scale=self.sigma, size=shift_shape or None).astype(x.dtype, copy=False)
            return x + shift_val
        return x

    def channel_dropout(self, x, is_tensor=None):
        """Randomly zero out entire channels (simulates sensor dropout).

        Applied independently per channel with probability channel_dropout_prob.
        At least one channel is always preserved to avoid all-zero inputs.
        Only applied to 2-D inputs [C, T]; 1-D inputs are passed through unchanged.
        """
        if is_tensor is None:
            x, is_tensor = self._normalize_input(x)
        if x.ndim != 2 or self.channel_dropout_prob <= 0.0:
            return x
        C = x.shape[0]
        if C <= 1:
            return x
        if is_tensor:
            mask = (torch.rand(C, device=x.device) > self.channel_dropout_prob).to(x.dtype)
        else:
            mask = (np.random.rand(C) > self.channel_dropout_prob).astype(x.dtype)
        # Ensure at least one channel survives
        if mask.sum() == 0:
            if is_tensor:
                mask[torch.randint(C, (1,)).item()] = 1.0
            else:
                mask[np.random.randint(C)] = 1.0
        if is_tensor:
            return x * mask.unsqueeze(1)
        return x * mask[:, np.newaxis]

    def window_slicing(self, x, is_tensor=None):
        """Crop a random temporal sub-window then resample back to original length.

        Creates temporally diverse views while preserving channel and amplitude structure.
        The fraction of the original length retained is uniformly sampled from
        [window_slice_min_ratio, 1.0].  Using scipy is avoided; resampling is done
        with np.interp (numpy) or torch.nn.functional.interpolate (tensor).

        Only applied to inputs with T >= 4 to avoid degenerate resampling.
        """
        if is_tensor is None:
            x, is_tensor = self._normalize_input(x)
        if not self._should_apply(x, is_tensor):
            return x
        T = x.shape[-1]
        min_len = max(4, int(T * self.window_slice_min_ratio))
        if min_len >= T:
            return x

        win_len = int(np.random.randint(min_len, T + 1))
        start = int(np.random.randint(0, T - win_len + 1))

        if is_tensor:
            import torch.nn.functional as _F
            if x.ndim == 2:
                sliced = x[:, start:start + win_len].unsqueeze(0)   # [1, C, win_len]
                return _F.interpolate(sliced, size=T, mode='linear', align_corners=False).squeeze(0)
            else:
                sliced = x[start:start + win_len].unsqueeze(0).unsqueeze(0)  # [1,1,win_len]
                return _F.interpolate(sliced, size=T, mode='linear', align_corners=False).squeeze(0).squeeze(0)
        else:
            indices = np.linspace(0, win_len - 1, T)
            if x.ndim == 2:
                rows = [np.interp(indices, np.arange(win_len), x[c, start:start + win_len]) for c in range(x.shape[0])]
                return np.stack(rows).astype(x.dtype)
            else:
                return np.interp(indices, np.arange(win_len), x[start:start + win_len]).astype(x.dtype)

    def __call__(self, x):
        """
        x: Tensor or Numpy array [Channels, Time] or [Time]
        """
        x, is_tensor = self._normalize_input(x)
        x_aug = self.window_slicing(x, is_tensor=is_tensor)  # temporal crop first
        x_aug = self.jitter(x_aug, is_tensor=is_tensor)
        x_aug = self.scaling(x_aug, is_tensor=is_tensor)
        x_aug = self.shift(x_aug, is_tensor=is_tensor)
        x_aug = self.channel_dropout(x_aug, is_tensor=is_tensor)
        return x_aug
