import numpy as np
from typing import Tuple
from datetime import datetime

# S2 processing baseline offset correction
# SEE: https://sentiwiki.copernicus.eu/web/s2-products
BOA_ADD_OFFSET = -1000.0
PROCESSING_BASELINE_NAIVE = datetime(2022, 1, 25)

class WelfordAccumulator:
    """Online algorithm for computing mean and variance using Welford's method."""
    
    def __init__(self, n_channels: int):
        """Initialize accumulator for n_channels.
        
        Parameters
        ----------
        n_channels : int
            Number of channels to track statistics for
        """
        self.count = 0
        self.mean = np.zeros(n_channels, dtype=np.float64)
        self.m2 = np.zeros(n_channels, dtype=np.float64)
        self.n_channels = n_channels
    
    def update(self, data: np.ndarray, mask: np.ndarray):
        """Update statistics by batch.
        
        Parameters
        ----------
        data : (n_channels, height, width)
            Array of data
        mask : (height, width)
            Boolean mask where True indicates valid pixels
        """
        assert data.shape[0] == self.n_channels, f'Number of channels mismatch: {data.shape[0]} != {self.n_channels}'
        
        # Extract valid pixels for each channel
        valid_data = data[:, mask]  # (n_channels, n_valid_pixels)
        
        if valid_data.size == 0:
            return
        
        # mean, m2 of batch
        count = valid_data.shape[1]
        mean = np.mean(valid_data, axis=1)
        m2 = np.sum((valid_data - mean[:, np.newaxis]) ** 2, axis=1)

        # Combine statistics using Chan's parallel algorithm
        new_count = self.count + count
        delta = mean - self.mean
        new_mean = (self.count * self.mean + count * mean) / new_count
        new_m2 = (self.m2 + m2 + 
                  delta**2 * self.count * count / new_count)
        self.count = new_count
        self.mean = new_mean
        self.m2 = new_m2

    def merge(self, other: 'WelfordAccumulator'):
        """Merge another accumulator into this one.
        
        Parameters
        ----------
        other : WelfordAccumulator
            Another WelfordAccumulator to merge
        """
        if other.count == 0:
            return
        if self.count == 0:
            self.count = other.count
            self.mean = other.mean.copy()
            self.m2 = other.m2.copy()
            return
        
        # Combine statistics using Chan's parallel algorithm
        new_count = self.count + other.count
        delta = other.mean - self.mean
        new_mean = (self.count * self.mean + other.count * other.mean) / new_count
        
        new_m2 = (self.m2 + other.m2 + 
                  delta**2 * self.count * other.count / new_count)
        
        self.count = new_count
        self.mean = new_mean
        self.m2 = new_m2
    
    def finalize(self) -> Tuple[np.ndarray, np.ndarray]:
        """Return final mean and standard deviation.
        
        Returns:
            Tuple of (mean, std) arrays
        """
        if self.count < 2:
            return self.mean.astype(np.float32), np.ones_like(self.mean, dtype=np.float32)
        
        variance = self.m2 / (self.count - 1)
        std = np.sqrt(variance)
        return self.mean.astype(np.float32), std.astype(np.float32)