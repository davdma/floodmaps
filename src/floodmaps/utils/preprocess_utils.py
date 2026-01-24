import numpy as np
from typing import Tuple, List
from datetime import datetime
from math import exp
from numba import jit, prange
import torch

# S2 processing baseline offset correction
# SEE: https://sentiwiki.copernicus.eu/web/s2-products
BOA_ADD_OFFSET = -1000.0
PROCESSING_BASELINE_NAIVE = datetime(2022, 1, 25)

class MinMaxAccumulator:
    """Accumulator for tracking running min/max across batches.
    
    Supports parallel processing by allowing merging of multiple accumulators.
    """
    
    def __init__(self, n_channels: int):
        """Initialize accumulator with n_channels.
        
        Parameters
        ----------
        n_channels : int
            Number of channels to track min/max for
        """
        self.n_channels = n_channels
        self.min_vals = np.full(n_channels, np.inf, dtype=np.float32)
        self.max_vals = np.full(n_channels, -np.inf, dtype=np.float32)
    
    def update(self, arr: np.ndarray, mask: np.ndarray) -> None:
        """Update min/max with new data.
        
        Parameters
        ----------
        arr : np.ndarray
            Array of shape (n_channels, n_samples)
        mask : np.ndarray
            Boolean mask of shape (n_samples,) indicating valid pixels
        """
        for c in range(self.n_channels):
            valid_data = arr[c, mask]
            if valid_data.size > 0:
                self.min_vals[c] = min(self.min_vals[c], valid_data.min())
                self.max_vals[c] = max(self.max_vals[c], valid_data.max())
    
    def merge(self, other: 'MinMaxAccumulator') -> None:
        """Merge another accumulator into this one.
        
        Parameters
        ----------
        other : MinMaxAccumulator
            Another accumulator to merge
        """
        self.min_vals = np.minimum(self.min_vals, other.min_vals)
        self.max_vals = np.maximum(self.max_vals, other.max_vals)
    
    def finalize(self) -> Tuple[np.ndarray, np.ndarray]:
        """Return final min and max values.
        
        Returns
        -------
        Tuple[np.ndarray, np.ndarray]
            Tuple of (min_vals, max_vals)
        """
        return self.min_vals.copy(), self.max_vals.copy()


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


def compute_awei_sh(blue: np.ndarray, green: np.ndarray, nir: np.ndarray, swir1: np.ndarray, swir2: np.ndarray) -> np.ndarray:
    """Compute AWEI_sh (Automated Water Extraction Index - shadow)."""
    return blue + 2.5 * green - 1.5 * (nir + swir1) - 0.25 * swir2

def compute_awei_nsh(green: np.ndarray, swir1: np.ndarray, nir: np.ndarray, swir2: np.ndarray) -> np.ndarray:
    """Compute AWEI_nsh (Automated Water Extraction Index - no shadow)."""
    return 4 * (green - swir1) - (0.25 * nir + 2.75 * swir2)

def compute_ndwi(green: np.ndarray, nir: np.ndarray, missing_val=-999999) -> np.ndarray:
    """Compute NDWI (Normalized Difference Water Index)."""
    return np.where(
        (green + nir) != 0,
        (green - nir) / (green + nir),
        missing_val
    )

def compute_mndwi(green: np.ndarray, swir1: np.ndarray, missing_val=-999999) -> np.ndarray:
    """Compute MNDWI (Modified Normalized Difference Water Index)."""
    return np.where(
        (green + swir1) != 0,
        (green - swir1) / (green + swir1),
        missing_val
    )

def impute_missing_values(arr, missing_mask):
    """Imputes the missing values in the array using the mean of the non-missing values.
    Raises error if all values are missing.

    Missing mask should be the same shape as the array.
    
    Can take either H, W or C, H, W arrays. In the C, H, W case, each channel is imputed separately."""
    # modify copy of array
    new_arr = arr.copy()
    if arr.ndim == 2:
        H, W = arr.shape
        if missing_mask.all():
            raise ValueError(f"All values are missing for imputation")
        mean = arr[~missing_mask].mean()
        new_arr[missing_mask] = mean
        return new_arr
    elif arr.ndim == 3:
        C, H, W = arr.shape
        for i in range(C):
            if missing_mask[i].all():
                raise ValueError(f"All values are missing for imputation")
            mean = arr[i][~missing_mask[i]].mean()
            new_arr[i][missing_mask[i]] = mean
        return new_arr
    else:
        raise ValueError(f"Array must be either H, W or C, H, W")

def calculate_missing_percent(missing_mask: np.ndarray) -> float:
    """Calculate the percentage of missing values in the missing mask."""
    return (missing_mask.sum() / missing_mask.size)

def calculate_cloud_percent(scl: np.ndarray, classes: List[int]) -> float:
    """Calculate the percentage of cloud classes in the SCL array."""
    return (np.isin(scl, classes).sum() / scl.size)

DAMP_DEFAULT = 1.0
CU_DEFAULT = 0.523 # 0.447 is sqrt(1/number of looks)
CMAX_DEFAULT = 1.73 # 1.183 is sqrt(1 + 2/number of looks)

# for Enhanced Lee filter for SAR dataset
@jit(nopython=True)
def _enhanced_lee_filter_numba(image, kernel_size, d, cu, cmax):
    """Numba-optimized enhanced lee filter core."""
    height, width = image.shape
    half_size = kernel_size // 2
    filtered_image = np.zeros((height, width), dtype=np.float64)
    
    for y in range(height):
        for x in range(width):
            pix_value = image[y, x]
            if pix_value == 0:
                filtered_image[y, x] = 0
                continue

            # Window bounds
            xleft = max(0, x - half_size)
            xright = min(width, x + half_size + 1)
            yup = max(0, y - half_size)
            ydown = min(height, y + half_size + 1)

            # Compute mean and std excluding zeros
            total = 0.0
            total_sq = 0.0
            count = 0
            for yy in range(yup, ydown):
                for xx in range(xleft, xright):
                    val = image[yy, xx]
                    if val != 0:
                        total += val
                        total_sq += val * val
                        count += 1

            if count == 0:
                filtered_image[y, x] = 0
                continue

            w_mean = total / count
            if count == 1:
                w_std = 0.0
            else:
                variance = (total_sq / count) - (w_mean * w_mean)
                # Bessel correction for sample std
                variance = variance * count / (count - 1)
                w_std = np.sqrt(max(0.0, variance))

            # Weighting calculation
            ci = w_std / w_mean if w_mean != 0 else 0.0
            if ci <= cu:
                w_t = 1.0
            elif ci >= cmax:
                w_t = 0.0
            else:
                w_t = exp((-d * (ci - cu)) / (cmax - ci))

            new_pix_value = (w_mean * w_t) + (pix_value * (1.0 - w_t))
            filtered_image[y, x] = new_pix_value

    return filtered_image

def dbToPower(x, nodata=-9999):
    """Convert SAR raster from db scale to power scale.
    Missing values are set to 0.

    Parameters
    ----------
    x : ndarray
    nodata : int, optional
        No data value of the raster x.

    Returns
    -------
    x : ndarray
    """
    # set all missing values back to zero
    missing_mask = x == nodata
    nonzero_mask = x != nodata
    x[nonzero_mask] = np.float_power(10, x[nonzero_mask] / 10, dtype=np.float64)  # Inverse of log10 transformation
    x[missing_mask] = 0
    return x

def powerToDb(x, nodata=-9999):
    """Convert SAR raster from power scale to db scale. Missing values set to nodata.

    Parameters
    ----------
    x : ndarray
    nodata : int, optional
        No data value of the raster x.

    Returns
    -------
    x : ndarray
    """
    nonzero_mask = x > 0
    missing_mask = x <= 0
    x[nonzero_mask] = 10 * np.log10(x[nonzero_mask], dtype=np.float64)
    x[missing_mask] = nodata
    return x

def enhanced_lee_filter(image, kernel_size=7, d=DAMP_DEFAULT, cu=CU_DEFAULT,
                        cmax=CMAX_DEFAULT, nodata=-9999):
    """Optimized enhanced lee filter using Numba JIT.
    
    Implements the enhanced lee filter outlined here:
    https://desktop.arcgis.com/en/arcmap/latest/manage-data/raster-and-images/speckle-function.htm
    https://catalyst.earth/catalyst-system-files/help/concepts/orthoengine_c/Chapter_825.html
    https://pyradar-tools.readthedocs.io/en/latest/_modules/pyradar/filters/lee_enhanced.html#lee_enhanced_filter

    Missing data will not be used in the calculation, and if the center pixel is missing,
    then the filter output will preserve the missing data.
    
    Parameters
    ----------
    image : ndarray
        SAR image to filter.
    kernel_size : int
        Size of the kernel.
    d : float
        Damping factor.
    cu : float
        Noise variation coefficient.
    cmax : float
        Maximum noise variation coefficient.
    nodata : int, optional
        No data value of the raster image.

    Returns
    -------
    filtered_image : ndarray
        Filtered SAR image.
    """
    image = np.float64(image)
    image = dbToPower(image, nodata=nodata)
    
    if image.ndim == 2:
        filtered_image = _enhanced_lee_filter_numba(image, kernel_size, d, cu, cmax)
    elif image.ndim == 3 and image.shape[0] == 1:
        filtered_image = _enhanced_lee_filter_numba(image[0], kernel_size, d, cu, cmax)
        filtered_image = filtered_image[np.newaxis, :, :]
    else:
        raise ValueError("Image must be shape (H, W) or (1, H, W)")
    
    return powerToDb(filtered_image, nodata=nodata)


@jit(nopython=True, parallel=True)
def _enhanced_lee_filter_batch_numba(db_sar_batch, kernel_size, d, cu, cmax, nodata):
    """Parallel batch processing of Enhanced Lee filter.
    
    Uses numba prange to parallelize over the flattened B*C dimension,
    where each (H, W) slice is processed independently.
    
    Parameters
    ----------
    db_sar_batch : np.ndarray (B, C, H, W)
        SAR data in dB scale (float64)
    kernel_size : int
        Size of the filter kernel.
    d : float
        Damping factor.
    cu : float
        Noise variation coefficient.
    cmax : float
        Maximum noise variation coefficient.
    nodata : float
        No data value.
    
    Returns
    -------
    np.ndarray (B, C, H, W)
        Filtered SAR data in dB scale
    """
    B, C, H, W = db_sar_batch.shape
    filtered = np.empty((B, C, H, W), dtype=np.float64)
    
    # Parallelize over flattened B*C dimension
    for idx in prange(B * C):
        b = idx // C
        c = idx % C
        
        # Convert slice from dB to power
        slice_db = db_sar_batch[b, c]
        slice_power = np.zeros((H, W), dtype=np.float64)
        for y in range(H):
            for x in range(W):
                if slice_db[y, x] != nodata:
                    slice_power[y, x] = np.float_power(10, slice_db[y, x] / 10)
        
        # Apply filter (reuse existing core function)
        filtered_power = _enhanced_lee_filter_numba(slice_power, kernel_size, d, cu, cmax)
        
        # Convert back to dB
        for y in range(H):
            for x in range(W):
                if filtered_power[y, x] > 0:
                    filtered[b, c, y, x] = 10 * np.log10(filtered_power[y, x])
                else:
                    filtered[b, c, y, x] = nodata
    
    return filtered


def enhanced_lee_filter_batch(db_sar_batch, kernel_size=7, d=DAMP_DEFAULT, 
                               cu=CU_DEFAULT, cmax=CMAX_DEFAULT, nodata=-9999):
    """Apply Enhanced Lee filter to batched SAR data.
    
    Parallelized version using numba prange for efficient batch processing.
    Expected speedup: 4-16x depending on CPU cores.
    
    Parameters
    ----------
    db_sar_batch : np.ndarray (B, C, H, W) or torch.Tensor
        SAR data in dB scale
    kernel_size : int
        Size of the filter kernel (default: 7).
    d : float
        Damping factor (default: 1.0).
    cu : float
        Noise variation coefficient (default: 0.523).
    cmax : float
        Maximum noise variation coefficient (default: 1.73).
    nodata : float
        No data value (default: -9999).
    
    Returns
    -------
    np.ndarray (B, C, H, W)
        Filtered SAR data in dB scale
    """
    if isinstance(db_sar_batch, torch.Tensor):
        db_sar_batch = db_sar_batch.cpu().numpy()
    
    db_sar_batch = np.ascontiguousarray(db_sar_batch, dtype=np.float64)
    return _enhanced_lee_filter_batch_numba(db_sar_batch, kernel_size, d, cu, cmax, nodata)