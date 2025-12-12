import torch
from torch.utils.data import Dataset, DataLoader, Subset
import rasterio
import numpy as np
from floodmaps.training.dataset import FloodSampleMeanStd
from math import exp
import json
import copy
import random
import math
import pickle
import logging
import torch.nn.functional as F
from pathlib import Path
from matplotlib.colors import to_rgb

DAMP_DEFAULT = 1.0
CU_DEFAULT = 0.523 # 0.447 is sqrt(1/number of looks)
CMAX_DEFAULT = 1.73 # 1.183 is sqrt(1 + 2/number of looks)

class Metrics:
    """Improved object interface to store validation and/or test metrics during single experiment.
    Allows for generalized data splits and partitions within splits if enabled.

    All metrics are organized into one large dictionary with split (val, test, etc.) index first, then
    partition (shift, non-shift, etc.) as the next available index if enabled."""
    def __init__(self, use_partitions=False):
        self.use_partitions=use_partitions
        self.metrics = {}

    def save_metrics(self, split, partition=None, **kwargs):
        """Save one or more metrics for a given split (and partition if enabled).

        - If partitions are enabled, store under {split -> partition -> metric}.
        - If partitions are disabled, store under {split -> metric}.
        - Supports multiple metrics at once via **kwargs.

        Parameters
        ----------
        split : str
            The data split to store the metrics under (e.g., "train", "val", "test").
        partition : str, optional
            The partition within the split (e.g., "shifted", "non-shifted"). Required if partitions are enabled.
        **kwargs : dict
            Key-value pairs representing metric names and their values for the specified split / partition.
        """
        if split not in self.metrics:
            self.metrics[split] = {}  # Initialize split if not present

        if self.use_partitions:
            if partition is None:
                raise ValueError("Partition must be specified when partitions are enabled.")
            if partition not in self.metrics[split]:
                self.metrics[split][partition] = {}
            self.metrics[split][partition].update(kwargs)
        else:
            if partition is not None:
                raise ValueError("Partition specified but not enabled.")
            self.metrics[split].update(kwargs)

    def get_metrics(self, split=None, partition=None):
        """Retrieve metrics.

        - If partitions are enabled:
            - If both `split` & `partition` are specified, return that partition's metrics.
            - If only `split` is specified, return all partitions within that split.
        - If partitions are disabled:
            - If `split` is specified, return that split's metrics directly.
        - If nothing is specified, return all metrics.

        Parameters
        ----------
        split : str, optional
            The data split of the metrics to retrieve (e.g., "train", "val", "test").
        partition : str, optional
            The partition within the split of the metrics to retrieve (e.g., "shifted", "non-shifted").
        """
        if split:
            if self.use_partitions and partition:
                return self.metrics.get(split, {}).get(partition, {})
            return self.metrics.get(split, {})
        return self.metrics

    def to_json(self, filename):
        """Save the stored metrics to a JSON file.

        Parameters
        ----------
        filename : str
            Path to the file where metrics will be saved.
        """
        with open(filename, "w") as f:
            json.dump(self.metrics, f, indent=4)

class ChannelIndexer:
    """Abstract class for wrapping list of S2 dataset channels used for input.
    Most up to date with RGB spectral bands instead of TCI.

    The 16 available channels in order:

    1. B04 Red Reflectance
    2. B03 Green Reflectance
    3. B02 Blue Reflectance
    4. B08 Near Infrared
    5. SWIR1 (B11)
    6. SWIR2 (B12)
    7. NDWI
    8. MNDWI
    9. AWEI_sh
    10. AWEI_nsh
    11. DEM
    12. Slope Y
    13. Slope X
    14. Waterbody
    15. Roads
    16. Flowlines

    Parameters
    ----------
    channels : list[bool]
        List of 16 booleans corresponding to the 16 input channels.
    """
    def __init__(self, channels):
        self.channels = channels
        self.channel_names = ["red", "green", "blue", "nir", "swir1", "swir2", "ndwi", "mndwi",
                            "awei_sh", "awei_nsh", "dem", "slope_y", "slope_x", "waterbody",
                            "roads", "flowlines"]
    
    def has_red(self):
        return self.channels[0]

    def has_green(self):
        return self.channels[1]

    def has_blue(self):
        return self.channels[2]

    def has_rgb(self):
        return all(self.channels[:3])

    def has_b08(self):
        return self.channels[3]

    def has_swir1(self):
        return self.channels[4]

    def has_swir2(self):
        return self.channels[5]

    def has_ndwi(self):
        return self.channels[6]

    def has_mndwi(self):
        return self.channels[7]

    def has_awei_sh(self):
        return self.channels[8]

    def has_awei_nsh(self):
        return self.channels[9]

    def has_dem(self):
        return self.channels[10]

    def has_slope_y(self):
        return self.channels[11]

    def has_slope_x(self):
        return self.channels[12]

    def has_waterbody(self):
        return self.channels[13]

    def has_roads(self):
        return self.channels[14]
    
    def has_flowlines(self):
        return self.channels[15]

    def get_channel_names(self):
        """Get list of channel names included in this configuration.
        Can index the list to get the channel name of the configuration given channel index."""
        return [channel_name for channel_name, include in zip(self.channel_names, self.channels) if include]

    def get_display_channels(self):
        """Channels specifically for sampling predictions."""
        # need to fix this hardcoding later
        display_names = []
        if self.has_ndwi():
            display_names.append("ndwi")
        if self.has_mndwi():
            display_names.append("mndwi")
        if self.has_dem():
            display_names.append("dem")
        if self.has_slope_y():
            display_names.append("slope_y")
        if self.has_slope_x():
            display_names.append("slope_x")
        if self.has_waterbody():
            display_names.append("waterbody")
        if self.has_roads():
            display_names.append("roads")
        if self.has_flowlines():
            display_names.append("flowlines")
        return display_names
    
    def get_name_to_index(self):
        """Return dictionary mapping channel name to input index."""
        filtered_channels = [channel_name for channel_name, include in zip(self.channel_names, self.channels) if include]
        return {channel_name: idx for idx, channel_name in enumerate(filtered_channels)}


class ChannelIndexerDeprecated:
    """Deprecated channel indexer for old TCI input S2 models that used only 10 channels.
    Abstract class for wrapping list of S2 dataset channels used for input.

    The 10 available channels in order:

    1. TCI R (0-255)
    2. TCI G (0-255)
    3. TCI B (0-255)
    4. B08 Near Infrared
    5. NDWI
    6. DEM
    7. Slope Y
    8. Slope X
    9. Waterbody
    10. Roads

    Parameters
    ----------
    channels : list[bool]
        List of 10 booleans corresponding to the 10 input channels.
    """
    def __init__(self, channels):
        self.channels = channels
        self.names = names = ["images", "ndwi", "dem", "slope_y", "slope_x", "waterbody", "roads"]
        self.included = [all(channels[:3])] + channels[4:] # we ignore index 3 = b8 channel

    def has_image(self):
        return all(self.channels[:3])

    def has_b08(self):
        return self.channels[3]

    def has_ndwi(self):
        return self.channels[4]

    def has_dem(self):
        return self.channels[5]

    def has_slope_y(self):
        return self.channels[6]

    def has_slope_x(self):
        return self.channels[7]

    def has_waterbody(self):
        return self.channels[8]

    def has_roads(self):
        return self.channels[9]

    def get_channel_names(self):
        return [name for name, is_included in zip(self.names, self.included) if is_included]


class SARChannelIndexer:
    """Abstract class for wrapping list of S1 dataset channels used for input.

    The 8 available channels in order:

    1. SAR VV
    2. SAR VH
    3. DEM
    4. Slope Y
    5. Slope X
    6. Waterbody
    7. Roads
    8. Flowlines

    Parameters
    ----------
    channels : list[bool]
        List of 8 booleans corresponding to the 8 input channels.
    """
    def __init__(self, channels):
        self.channels = channels
        self.channel_names = ["vv", "vh", "dem", "slope_y", "slope_x", "waterbody", "roads", "flowlines"]

    def has_vv(self):
        return self.channels[0]

    def has_vh(self):
        return self.channels[1]

    def has_dem(self):
        return self.channels[2]

    def has_slope_y(self):
        return self.channels[3]

    def has_slope_x(self):
        return self.channels[4]

    def has_waterbody(self):
        return self.channels[5]

    def has_roads(self):
        return self.channels[6]

    def has_flowlines(self):
        return self.channels[7]

    def get_channel_names(self):
        return self.channel_names
    
    def get_display_channels(self):
        return [name for name, include in zip(self.channel_names, self.channels) if include]

    def get_name_to_index(self):
        """Return dictionary mapping channel name to input index."""
        return list(range())

def flatten_dict(d, parent_key='', sep='.'):
    """
    Flatten a nested dictionary for wandb logging purposes.
    
    Args:
        d: Dictionary to flatten
        parent_key: String to prepend to keys
        sep: Separator between nested keys
        
    Returns:
        Flattened dictionary with dot notation keys
    """
    items = []
    for k, v in d.items():
        new_key = f"{parent_key}{sep}{k}" if parent_key else k
        if isinstance(v, dict):
            items.extend(flatten_dict(v, new_key, sep=sep).items())
        else:
            items.append((new_key, v))
    return dict(items)

def load_model_weights(model, weight_path, device, model_name="Model"):
    """Shared weight loading util.

    Parameters
    ----------
    model : obj
        SAR classifier, autodespeckler, or optical model
    weight_path : str
    device : str
    model_name : str
    """
    if weight_path is None:
        return
    try:
        state_dict = torch.load(weight_path, map_location=device)
        model.load_state_dict(state_dict)
        print(f"{model_name} weights loaded successfully.")
    except RuntimeError as e:
        print(f"Error loading {model_name} weights: {e}")
        raise e

### VAE Beta Scheduling
def beta_cycle_linear(epoch, beta=1, period=50, n_cycle=4, ratio=0.5):
    """Beta annealing function as proposed by https://arxiv.org/pdf/1903.10145.
    After the n cycles of period epochs, the remaining epochs remain at max beta.

    Parameters
    ----------
    epoch : int
        Current training epoch.
    beta : float
        Max beta value at end of beta annealing cycle.
    period : int
        Epochs per cycle.
    n_cycle : int
        Number of total cycles for beta annealing. After final cycle, beta
        will stay at max beta value.
    ratio : float
        Proportion of cycle (between 0.0 and 1.0) with a linear ramp. Used to
        calculate point of the cycle where beta reaches max beta value.
    """
    if epoch >= n_cycle * period:
        return beta

    # scales linearly to the closest epoch of specified ratio
    t = epoch % period
    midpt = math.ceil((period - 1) * ratio)
    return beta if t >= midpt else beta * t / midpt

class BetaScheduler:
    def __init__(self, beta=1.0, period=50, n_cycle=4, ratio=0.5):
        """Beta scheduler for cyclic annealing."""
        self.beta = beta
        self.period = period
        self.n_cycle = n_cycle
        self.ratio = ratio
        self.epoch = 0
        self.cur_beta = beta_cycle_linear(0, beta=beta, period=period, n_cycle=n_cycle, ratio=ratio)

    def step(self):
        self.epoch += 1
        self.cur_beta = beta_cycle_linear(self.epoch,
                                          beta=self.beta,
                                          period=self.period,
                                          n_cycle=self.n_cycle,
                                          ratio=self.ratio)

    def get_beta(self):
        return self.cur_beta

### Model and gradient tracking helpers
def get_gradient_norm(model):
    """Calculate global gradient norm during training."""
    total_norm = 0.0
    for p in model.parameters():
        if p.grad is not None:
            param_norm = p.grad.data.norm(2)  # L2 norm of gradients
            total_norm += param_norm.item() ** 2
    return total_norm ** 0.5

def get_model_params(model):
    """Function to calculate and log parameter sizes."""
    total_params = sum(p.numel() for p in model.parameters())  # Total parameters
    trainable_params = sum(p.numel() for p in model.parameters() if p.requires_grad)  # Trainable parameters
    param_size_in_mb = total_params * 4 / (1024 ** 2)  # Assuming 32-bit (4 bytes) precision

    return total_params, trainable_params, param_size_in_mb

def print_model_params_and_grads(model, file_name='ad_param_err.json', save_to_file=True):
    """For checking model gradients are within reasonable bounds."""
    # Ensure the model is an instance of nn.Module
    if not isinstance(model, torch.nn.Module):
        raise TypeError("The input model must be an instance of torch.nn.Module")
    stats_dict = {}
    print(f"{'Parameter':>30} | {'Min':^10} | {'Max':^10} | {'Mean':^10} | {'Std':^10} | {'Grad Min':^10} | {'Grad Max':^10} | {'Grad Mean':^10} | {'Grad Std':^10}")
    print("=" * 130)  # Table separator

    # Iterate through the model's named parameters
    for name, param in model.named_parameters():
        param_data = param.data
        param_grad = param.grad
        stats = {
            'min': param_data.min().item(),
            'max': param_data.max().item(),
            'mean': param_data.abs().mean().item(),
            'std': param_data.std().item(),
            'grad_min': param_grad.min().item() if param_grad is not None else None,
            'grad_max': param_grad.max().item() if param_grad is not None else None,
            'grad_mean': param_grad.abs().mean().item() if param_grad is not None else None,
            'grad_std': param_grad.std().item() if param_grad is not None else None,
        }
        stats_dict[name] = stats
        print(f"{name:>30} | {'{0:^10.2e}'.format(stats['min'])} | {'{0:^10.2e}'.format(stats['max'])} | {'{0:^10.2e}'.format(stats['mean'])} | {'{0:^10.2e}'.format(stats['std'])} | "
      f"{'{0:^10.2e}'.format(stats['grad_min']) if stats['grad_min'] is not None else 'None':^10} | "
      f"{'{0:^10.2e}'.format(stats['grad_max']) if stats['grad_max'] is not None else 'None':^10} | "
      f"{'{0:^10.2e}'.format(stats['grad_mean']) if stats['grad_mean'] is not None else 'None':^10} | "
      f"{'{0:^10.2e}'.format(stats['grad_std']) if stats['grad_std'] is not None else 'None':^10}")

    # save stats_dict to pkl
    if save_to_file:
        with open(file_name, 'w') as json_file:
            json.dump(stats_dict, json_file, indent=4)

    return stats_dict

def wet_label(image, crop_size, num_pixel=100):
    """
    Determines whether a patch of given size has enough water pixels to qualify as a wet patch.

    Parameters
    ----------
    crop_size: int
        width and height of the patch.
    num_pixel: int
        Number of water pixels in patch for the patch to qualify as wet.

    Returns
    -------
    0 or 1
        0 meaning patch is not wet and 1 meaning patch is wet.
    """
    # number of water pixels must be > num_pixels in order to get a positive label
    label = image.view(-1, crop_size**2).sum(1).gt(num_pixel).int()
    return label

def trainMeanStd(batch_size=10, channels=[True] * 10, sample_dir='../sampling/samples_200_5_4_35/', label_dir='../sampling/labels/'):
    """Calculate mean and std across selected channels of all S2 training tiles used to sample patches.

    Note: only use for S2 dataset.

    Parameters
    ----------
    batch_size : int
    channels : list[bool]
        List of 10 booleans corresponding to the 10 input channels.
    sample_dir : str
        Directory containing raw S2 tiles for patch sampling.
    label_dir : str
        Directory containing raw S2 tile labels for patch sampling.

    Returns
    -------
    mean : Tensor
        Channel means.
    std : Tensor
        Channel stds.
    """
    def channel_collate(batch):
        # concatenate channel wise for all samples in batch
        samples = torch.cat(batch, 1)
        return samples

    train_mean_std = FloodSampleMeanStd(TRAIN_LABELS,
                                        channels=channels,
                                        sample_dir=sample_dir,
                                        label_dir=label_dir)
    loader = DataLoader(train_mean_std,
                        batch_size=batch_size,
                        num_workers=0,
                        collate_fn=channel_collate,
                        shuffle=False)

    # random crop - calculate total mean and std for each channel not including missing values
    n_channels = sum(channels)
    b_channels = sum(channels[-2:])
    pixel_count = 0
    tot_sum = torch.zeros(n_channels, dtype=torch.float64)
    for samples in loader:
        # mask out missing values and calculate mean for each channel
        mask = (samples[0] != 0)
        pixel_count += mask.sum()
        for i in range(n_channels - b_channels):
            tot_sum[i] += samples[i][mask].sum()
    mean = tot_sum / pixel_count

    # add up variance for each channel not including missing values
    tot_var = torch.zeros(n_channels, dtype=torch.float64)
    for samples in loader:
        # first stack all feature values from batch together
        mask = (samples[0] != 0)
        for i in range(n_channels - b_channels):
            tot_var[i] += ((samples[i][mask] - mean[i]) ** 2).sum()
    std = torch.sqrt(tot_var / (pixel_count - 1))

    # set mean to 0 and std to 1 for binary channels - roads and waterbody!!
    if b_channels > 0:
        mean[-b_channels:] = 0
        std[-b_channels:] = 1
    return mean, std

class EarlyStopper:
    """Stops the training early if validation loss doesn't improve after a given number of epochs.

    Parameters
    ----------
    patience : int
        Number of epochs without improvement before training is stopped.
    min_delta : float
        Loss above lowest loss + min_delta counts towards patience.
    """
    def __init__(self, patience=1, min_delta=0):
        self.patience = patience
        self.min_delta = min_delta
        self.counter = 0
        self.best = False
        self.min_validation_loss = float('inf')
        self.metrics = None
        self.best_model_weights = None
        self.best_epoch = 0

    def state_dict(self):
        """For saving the state of the early stopper."""
        return {
            "patience": self.patience,
            "min_delta": self.min_delta,
            "counter": self.counter,
            "best": self.best,
            "min_validation_loss": self.min_validation_loss,
            "metrics": self.metrics,
            "best_model_weights": self.best_model_weights, # note this is may be diff from chkpt weights
            "best_epoch": self.best_epoch
        }

    def load_state_dict(self, state):
        self.patience = state["patience"]
        self.min_delta = state["min_delta"]
        self.counter = state["counter"]
        self.best = state["best"]
        self.min_validation_loss = state["min_validation_loss"]
        self.metrics = state["metrics"]
        self.best_model_weights = state["best_model_weights"]
        self.best_epoch = state["best_epoch"]

    def step(self, validation_loss, model, epoch, metrics=None):
        if validation_loss < self.min_validation_loss:
            self.min_validation_loss = validation_loss
            self.counter = 0
            self.best = True
            self.best_model_weights = copy.deepcopy(model.state_dict())
            self.best_epoch = epoch
            self.metrics = metrics
        elif validation_loss > (self.min_validation_loss + self.min_delta):
            self.counter += 1
            self.best = False
        else:
            self.best = False

    def get_best_metrics(self):
        """Retrieving additional metrics besides loss from best epoch."""
        return self.metrics

    def is_stopped(self):
        """Check whether training has stopped."""
        return self.counter >= self.patience

    def get_min_validation_loss(self):
        return self.min_validation_loss

    def get_best_weights(self):
        """Return the best model weights."""
        return self.best_model_weights
    
    def get_best_epoch(self):
        """Return the best epoch."""
        return self.best_epoch

class ADEarlyStopper:
    """This class supports regular early stopping as well as early stopping for VAE cyclical beta annealing
    training patterns.

    Parameters
    ----------
    patience : int
        Number of epochs without improvement before training is stopped.
    min_delta : float
        Loss above lowest loss + min_delta counts towards patience.
    beta_annealing : bool
        Whether to only save and stop at the end of a beta annealing cycle.
    count_cycles : bool
        If True, then step at end of each cycle counts towards patience. Otherwise only starts
        counter when all cycles are done - forcing model to use all cycles before any early stopping.
    """
    def __init__(self, patience=1, min_delta=0, beta_annealing=False, period=None, n_cycle=None, count_cycles=False):
        self.patience = patience
        self.min_delta = min_delta
        self.counter = 0
        self.best = False
        self.min_validation_loss = float('inf')
        self.best_model_weights = None
        self.best_epoch = 0 # newly added

        # beta annealing
        self.beta_annealing = beta_annealing
        self.period = period
        self.n_cycle = n_cycle
        self.count_cycles = count_cycles
    
    def state_dict(self):
        """For saving the state of the early stopper."""
        return {
            "patience": self.patience,
            "min_delta": self.min_delta,
            "counter": self.counter,
            "best": self.best,
            "min_validation_loss": self.min_validation_loss,
            "best_model_weights": self.best_model_weights,
            "best_epoch": self.best_epoch,
            "beta_annealing": self.beta_annealing,
            "period": self.period,
            "n_cycle": self.n_cycle,
            "count_cycles": self.count_cycles
        }

    def load_state_dict(self, state):
        self.patience = state["patience"]
        self.min_delta = state["min_delta"]
        self.counter = state["counter"]
        self.best = state["best"]
        self.min_validation_loss = state["min_validation_loss"]
        self.best_model_weights = state["best_model_weights"]
        self.best_epoch = state["best_epoch"]
        self.beta_annealing = state["beta_annealing"]
        self.period = state["period"]
        self.n_cycle = state["n_cycle"]
        self.count_cycles = state["count_cycles"]

    def step(self, validation_loss, model, epoch):
        """
        Note: For beta annealing scheduling at the end of n_cycles,
        we revert to normal early stopping with fixed beta.

        Current bug: If training finishes before a single cycle is completed,
        then the early stopper will not record the best model weights. This
        could cause an error when loading the best model.

        Args:
        - val_loss: Current validation loss.
        - model: Model to save weights from.
        - epoch: Current epoch.
        """
        # if beta annealing check if end of cycle or past end of last cycle
        if self.beta_annealing and (epoch + 1) % self.period != 0 and epoch < self.n_cycle * self.period:
            return  # Do nothing

        if validation_loss < self.min_validation_loss:
            self.min_validation_loss = validation_loss
            self.counter = 0
            self.best = True
            self.best_model_weights = copy.deepcopy(model.state_dict())
            self.best_epoch = epoch
        elif validation_loss > (self.min_validation_loss + self.min_delta):
            if self.beta_annealing and not self.count_cycles:
                if epoch >= self.n_cycle * self.period:
                    self.counter += 1
            else:
                self.counter += 1
            self.best = False
        else:
            self.best = False

    def is_stopped(self):
        """Check whether training has stopped."""
        return self.counter >= self.patience

    def get_min_validation_loss(self):
        return self.min_validation_loss

    def get_best_weights(self):
        """Return the best model weights."""
        return self.best_model_weights

    def get_best_epoch(self):
        """Return the best epoch."""
        return self.best_epoch


### SAR Preprocessing Utils
def dbToPower(x):
    """Convert SAR raster from db scale to power scale.

    Parameters
    ----------
    x : ndarray

    Returns
    -------
    x : ndarray
    """
    # set all missing values back to zero
    missing_mask = x == -9999
    nonzero_mask = x != -9999
    x[nonzero_mask] = np.float_power(10, x[nonzero_mask] / 10, dtype=np.float64)  # Inverse of log10 transformation
    x[missing_mask] = 0
    return x

def powerToDb(x):
    """Convert SAR raster from power scale to db scale. Missing values set to -9999.

    Parameters
    ----------
    x : ndarray

    Returns
    -------
    x : ndarray
    """
    nonzero_mask = x > 0
    missing_mask = x <= 0
    x[nonzero_mask] = 10 * np.log10(x[nonzero_mask], dtype=np.float64)
    x[missing_mask] = -9999
    return x

def enhanced_lee_filter(image, kernel_size=7, d=DAMP_DEFAULT, cu=CU_DEFAULT,
                        cmax=CMAX_DEFAULT):
    """Implements the enhanced lee filter outlined here:
    https://desktop.arcgis.com/en/arcmap/latest/manage-data/raster-and-images/speckle-function.htm
    https://catalyst.earth/catalyst-system-files/help/concepts/orthoengine_c/Chapter_825.html
    https://pyradar-tools.readthedocs.io/en/latest/_modules/pyradar/filters/lee_enhanced.html#lee_enhanced_filter

    Missing data will not be used in the calculation, and if the center pixel is missing, then the filter output
    will preserve the missing data.

    Parameters
    ----------
    image : ndarray
    kernel_size : int
    d : float
        Enhanced lee damping factor.
    cu : float
        Enhanced lee noise variation coefficient.
    cmax : float
        Enhanced lee maximum noise variation coefficient

    Returns
    -------
    filtered_image : ndarray
    """

    # missing data: if center pixel is missing then do nothing!
    # if neighborhood pixels all missing then do nothing!

    # input should be 400 x 400!

    image = np.float64(image) # process image as float64 to avoid overflow
    image = dbToPower(image)
    height, width = image.shape

    # iterate over entire image, create window for each pixel, ignore missing data
    half_size = kernel_size // 2
    filtered_image = np.zeros((height, width), dtype=np.float64)
    for y in range(height):
        for x in range(width):
            # if center pixel is missing, then return missing value
            pix_value = image[y][x]
            if pix_value == 0:
                filtered_image[y][x] = 0
                continue

            # get window
            xleft = x - half_size
            xright = x + half_size
            yup = y - half_size
            ydown = y + half_size

            if xleft < 0:
                xleft = 0
            if xright >= width:
                xright = width

            if yup < 0:
                yup = 0
            if ydown >= height:
                ydown = height

            neighbor_pixels = image[yup:ydown, xleft:xright]
            num_samples = np.count_nonzero(neighbor_pixels)
            # unoptimized: num_samples = get_neighbors(y, x, image, height, width, kernel_size, neighbor_pixels)
            w_mean = getLocalMeanValue(neighbor_pixels)
            if num_samples == 1:
                w_std = 0
            else:
                w_std = getLocalStdValue(neighbor_pixels)

            w_t = weighting(w_mean, w_std, d, cu, cmax)

            new_pix_value = (w_mean * w_t) + (pix_value * (1.0 - w_t))
            if new_pix_value < 0:
                raise Exception("filter pixel value cannot be negative")

            filtered_image[y][x] = new_pix_value

    return powerToDb(filtered_image)

def weighting(w_mean, w_std, d, cu, cmax):
    # cu is the noise variation coefficient
    # ci is the variation coefficient in the window
    ci = w_std / w_mean
    if ci <= cu:  # use the mean value
        w_t = 1.0
    elif cu < ci < cmax:  # use the filter
        w_t = exp((-d * (ci - cu)) / (cmax - ci))
    elif ci >= cmax:  # preserve the original value
        w_t = 0.0

    return w_t

def getLocalMeanValue(neighbor_pixels):
    return np.mean(neighbor_pixels, where = neighbor_pixels != 0, dtype=np.float64)

def getLocalStdValue(neighbor_pixels):
    return np.std(neighbor_pixels, where = neighbor_pixels != 0, dtype=np.float64, ddof=1)

def get_neighbors(y, x, image, height, width, kernel_size, neighbor_pixels):
    half_size = kernel_size // 2
    num_samples = 0
    for j in range(kernel_size):
        yj = y - half_size + j
        if yj < 0 or yj >= height:
            for i in range(kernel_size):
                # zero is no data value
                neighbor_pixels[j][i] = 0
            continue

        for i in range(kernel_size):
            xi = x - half_size + i
            if xi < 0 or xi >= width:
                neighbor_pixels[j][i] = 0
            else:
                neighbor_pixels[j][i] = image[yj][xi]
                if neighbor_pixels[j][i] != 0:
                    num_samples += 1
    return num_samples

# NLCD Land Cover Visualization
# Colors match the sampling pipeline for consistency across the workflow

NLCD_COLORS = {
    11: '#486da2',    # Open Water
    12: '#e7effc',    # Perennial Ice/Snow
    21: '#e1cdce',    # Developed, Open Space
    22: '#dc9881',    # Developed, Low Intensity 
    23: '#f10100',    # Developed, Medium Intensity
    24: '#ab0101',    # Developed, High Intensity
    31: '#b3afa4',    # Barren Land
    41: '#6ca966',    # Deciduous Forest
    42: '#1d6533',    # Evergreen Forest
    43: '#bdcc93',    # Mixed Forest
    51: '#b19943',    # Dwarf Scrub
    52: '#d1bb82',    # Shrub/Scrub
    71: '#edeccd',    # Grassland/Herbaceous
    72: '#d0d181',    # Sedge/Herbaceous
    73: '#a4cc51',    # Lichens
    74: '#82ba9d',    # Moss
    81: '#ddd83d',    # Pasture/Hay
    82: '#ae7229',    # Cultivated Crops
    90: '#bbd7ed',    # Woody Wetlands
    95: '#71a4c1',     # Emergent Herbaceous Wetlands
    250: '#000000'     # Missing
}
NLCD_CODE_TO_RGB = {
    code: tuple(int(255 * c) for c in to_rgb(hex_color))
    for code, hex_color in NLCD_COLORS.items()
}

def nlcd_to_rgb(nlcd_array):
    """Convert NLCD classification array to RGB image using NLCD colormap.
    
    Optimized version using lookup table for efficient batch processing.
    
    Parameters
    ----------
    nlcd_array : numpy.ndarray
        2D array of NLCD land cover class codes (uint8)
        
    Returns
    -------
    numpy.ndarray
        3D RGB array with shape (H, W, 3), dtype uint8
    """
    # create NLCD colormap
    H, W = nlcd_array.shape
    rgb_img = np.zeros((H, W, 3), dtype=np.uint8)

    # vectorized mapping
    for code, rgb in NLCD_CODE_TO_RGB.items():
        mask = nlcd_array == code
        rgb_img[mask] = rgb
    
    return rgb_img

SCL_COLORS = {
    0: '#000000',    # No data
    1: '#ff0000',    # Saturated or defective
    2: '#2f2f2f',    # Topographic casted shadows
    3: '#643200',    # Cloud shadow
    4: '#00a000',    # Vegetation
    5: '#ffe65a',    # Not vegetated
    6: '#0000ff',    # Water
    7: '#808080',    # Unclassified
    8: '#c0c0c0',    # Cloud medium probability
    9: '#ffffff',    # Cloud high probability
    10: '#64c8ff',    # Thin cirrus
    11: '#ff96ff',    # Snow or ice
}
SCL_CODE_TO_RGB = {
    code: tuple(int(255 * c) for c in to_rgb(hex_color))
    for code, hex_color in SCL_COLORS.items()
}

def scl_to_rgb(scl_array):
    """Convert SCL classification array to RGB image using SCL colormap.
    
    Optimized version using lookup table for efficient batch processing.
    
    Parameters
    ----------
    scl_array : numpy.ndarray
        2D array of SCL class codes (uint8)
        
    Returns
    -------
    numpy.ndarray
        3D RGB array with shape (H, W, 3), dtype uint8
    """
    # create NLCD colormap
    H, W = scl_array.shape
    rgb_img = np.zeros((H, W, 3), dtype=np.uint8)

    # vectorized mapping
    for code, rgb in SCL_CODE_TO_RGB.items():
        mask = scl_array == code
        rgb_img[mask] = rgb
    
    return rgb_img

def get_samples_with_wet_percentage(sample_set, num_samples, batch_size, num_workers, percent_wet_patches, rng, cache_dir=None, dataset_name=None):
    """
    Get a list of sample indices where a specified percentage has wet patches (y.sum() > 0).
    Uses vectorized operations for much faster performance than Python loops.
    
    If cache_dir and dataset_name are provided, wet/dry indices will be cached to disk
    for faster subsequent runs. The cache file is stored at:
    {cache_dir}/{dataset_name}_wet_dry_indices.pkl
    
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
    cache_dir : Path or str, optional
        Directory to cache wet/dry indices. If None, no caching is performed.
    dataset_name : str, optional
        Name of the dataset (e.g., 'val', 'test') for cache file naming.
        Required if cache_dir is provided.
        
    Returns
    -------
    list
        List of sample indices
    """
    # Handle caching
    wet_indices = None
    dry_indices = None
    cache_path = None
    
    if cache_dir is not None and dataset_name is not None:
        cache_path = Path(cache_dir) / f'{dataset_name}_wet_dry_indices.pkl'
        if cache_path.exists():
            try:
                with open(cache_path, 'rb') as f:
                    cached = pickle.load(f)
                    cached_wet = cached['wet_indices']
                    cached_dry = cached['dry_indices']
                    # Validate cache matches dataset size
                    if len(cached_wet) + len(cached_dry) == len(sample_set):
                        wet_indices = cached_wet
                        dry_indices = cached_dry
                        logging.info(f"Loaded wet/dry indices from cache: {cache_path}")
                        logging.info(f"  Wet: {len(wet_indices)}, Dry: {len(dry_indices)}")
                    else:
                        logging.warning(f"Cache size mismatch (cached: {len(cached_wet) + len(cached_dry)}, "
                                      f"dataset: {len(sample_set)}). Recomputing indices.")
            except Exception as e:
                logging.warning(f"Failed to load cache from {cache_path}: {e}. Recomputing indices.")
    
    # Compute wet/dry indices if not cached
    if wet_indices is None or dry_indices is None:
        logging.info(f"Computing wet/dry indices for {len(sample_set)} samples...")
        batch_size = min(batch_size, len(sample_set))  # Process in batches to avoid memory issues
        dataloader = DataLoader(sample_set, batch_size=batch_size, num_workers=num_workers, shuffle=False)
        wet_indices = []
        dry_indices = []
        
        current_idx = 0
        for batch_x, batch_y, batch_supp in dataloader:
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
        
        logging.info(f"Computed wet/dry indices: Wet: {len(wet_indices)}, Dry: {len(dry_indices)}")
        
        # Cache the indices if cache_dir is provided
        if cache_path is not None:
            try:
                with open(cache_path, 'wb') as f:
                    pickle.dump({'wet_indices': wet_indices, 'dry_indices': dry_indices}, f)
                logging.info(f"Saved wet/dry indices to cache: {cache_path}")
            except Exception as e:
                logging.warning(f"Failed to save cache to {cache_path}: {e}")
    
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