import torch
from torch.utils.data import DataLoader

class WFChannelIndexer:
    """Abstract class for wrapping list of WorldFloodsS2 dataset channels used for input.

    The 5 available channels in order:

    1. B04 Red Reflectance
    2. B03 Green Reflectance
    3. B02 Blue Reflectance
    4. B08 Near Infrared
    5. NDWI

    Parameters
    ----------
    channels : list[bool]
        List of 5 booleans corresponding to the 11 input channels.
    """
    def __init__(self, channels):
        self.channels = channels
        self.names = ["rgb", "nir", "ndwi"]

    def has_rgb(self):
        return all(self.channels[:3])

    def has_b08(self):
        return self.channels[3]
    
    def has_nir(self):
        """Alias for has_b08() for consistency with visualization code."""
        return self.has_b08()

    def has_ndwi(self):
        return self.channels[4]

    def get_channel_names(self):
        return self.names

    def get_display_channels(self):
        """Channels specifically for sampling predictions."""
        # need to fix this hardcoding later
        display_names = []
        if self.has_rgb():
            display_names.append("rgb")
        if self.has_nir():
            display_names.append("nir")
        if self.has_ndwi():
            display_names.append("ndwi")
        return display_names

def wf_get_samples_with_wet_percentage(sample_set, num_samples, batch_size, num_workers, percent_wet_patches, rng):
    """
    Get a list of sample indices where a specified percentage has wet patches (y.sum() > 0).
    Uses vectorized operations for much faster performance than Python loops.
    
    Parameters
    ----------
    sample_set : torch.utils.data.Dataset
        The dataset to sample from
    num_samples : int
        Total number of samples to return
    batch_size : int
        Batch size for the DataLoader
    num_workers : int
        Number of workers for the DataLoader
    percent_wet_patches : float
        Percentage of samples that should have wet patches (0.0 to 1.0)
    rng : Random
        Random number generator instance
        
    Returns
    -------
    list
        List of sample indices
    """
    # Use DataLoader with batch processing for vectorized operations
    batch_size = min(batch_size, len(sample_set))  # Process in batches to avoid memory issues
    dataloader = DataLoader(sample_set, batch_size=batch_size, num_workers=num_workers, shuffle=False)
    wet_indices = []
    dry_indices = []
    
    current_idx = 0
    for batch_x, batch_y in dataloader:  # Unpack 2 values (image, label)
        # Vectorized operation: check which samples have any positive pixels
        # batch_y shape: (batch_size, channels, height, width)
        # Sum over spatial dimensions (last 2 dims) and channels to get total water pixels per sample
        water_pixels_per_sample = batch_y.sum(dim=(1, 2, 3))  # Shape: (batch_size,)
        
        # Get boolean mask for wet patches (samples with water_pixels > 0)
        is_wet = water_pixels_per_sample > 0  # Shape: (batch_size,)
        
        # Convert to indices in the original dataset
        batch_indices = torch.arange(current_idx, current_idx + batch_y.size(0))
        wet_batch_indices = batch_indices[is_wet].tolist()
        dry_batch_indices = batch_indices[~is_wet].tolist()
        
        wet_indices.extend(wet_batch_indices)
        dry_indices.extend(dry_batch_indices)
        
        current_idx += batch_y.size(0)
    
    # Calculate number of wet and dry samples needed
    num_wet_needed = int(num_samples * percent_wet_patches)
    num_dry_needed = num_samples - num_wet_needed
    
    # Sample wet patches (with replacement if needed)
    if len(wet_indices) >= num_wet_needed:
        wet_samples = rng.sample(wet_indices, num_wet_needed)
    else:
        # If not enough wet patches, take all and sample with replacement
        if len(wet_indices) > 0:
            wet_samples = wet_indices + rng.choices(wet_indices, k=num_wet_needed - len(wet_indices))
        else:
            # No wet patches available, sample from dry patches instead
            wet_samples = []
            num_dry_needed = num_samples
    
    # Sample dry patches (with replacement if needed)
    if len(dry_indices) >= num_dry_needed:
        dry_samples = rng.sample(dry_indices, num_dry_needed)
    else:
        # If not enough dry patches, take all and sample with replacement
        if len(dry_indices) > 0:
            dry_samples = dry_indices + rng.choices(dry_indices, k=num_dry_needed - len(dry_indices))
        else:
            # No dry patches available, sample from wet patches instead
            dry_samples = []
            if len(wet_indices) > 0:
                additional_wet_needed = num_dry_needed
                wet_samples.extend(rng.choices(wet_indices, k=additional_wet_needed))
    
    # Combine and shuffle the samples
    all_samples = wet_samples + dry_samples
    rng.shuffle(all_samples)
    
    return all_samples