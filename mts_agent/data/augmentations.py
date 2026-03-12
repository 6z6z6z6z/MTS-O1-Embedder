import numpy as np
import torch

class TimeSeriesAugmentor:
    """
    Apply augmentations to time series data to prevent overfitting on small datasets.
    """
    def __init__(self, prob=0.5, sigma=0.03, scale_sigma=0.1):
        self.prob = prob
        self.sigma = sigma  # For jitter
        self.scale_sigma = scale_sigma # For scaling

    def jitter(self, x):
        # Add random noise
        # x: [Channels, Time] or [Time]
        if np.random.random() < self.prob:
             return x + np.random.normal(loc=0., scale=self.sigma, size=x.shape)
        return x

    def scaling(self, x):
        # Multiply by random factor
        if np.random.random() < self.prob:
            factor = np.random.normal(loc=1.0, scale=self.scale_sigma, size=(x.shape[0], 1))
            return x * factor
        return x

    def shift(self, x):
        # Shift time axis locally? No, simple value shifting
        if np.random.random() < self.prob:
             shift_val = np.random.normal(loc=0., scale=self.sigma, size=(x.shape[0], 1))
             return x + shift_val
        return x

    def __call__(self, x):
        """
        x: Tensor or Numpy array [Channels, Time]
        """
        is_tensor = torch.is_tensor(x)
        if is_tensor:
            x_np = x.detach().cpu().numpy()
        else:
            x_np = x

        # Ensure correct shape handling if necessary
        # Assuming [Channels, Time]
        
        x_aug = self.jitter(x_np)
        x_aug = self.scaling(x_aug)
        x_aug = self.shift(x_aug)

        if is_tensor:
            return torch.tensor(x_aug, dtype=x.dtype)
        return x_aug
