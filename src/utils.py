import torch
from torch.utils.data import Dataset, DataLoader
import rasterio
import numpy as np
from dataset import FloodSampleMeanStd
from math import exp

TRAIN_LABELS = ["label_20150919_20150917_496_811.tif", "label_20150919_20150917_497_812.tif",
                "label_20151112_20151109_472_940.tif", "label_20151112_20151109_473_939.tif", 
                "label_20151112_20151109_473_940.tif", "label_20151112_20151109_473_941.tif", 
                "label_20151112_20151109_474_942.tif", "label_20160314_20160311_451_838.tif", 
                "label_20160314_20160311_451_839.tif", "label_20160307_20160311_452_846.tif",
                "label_20160314_20160311_452_845.tif", "label_20160314_20160311_453_845.tif",
                "label_20170830_20170828_462_695.tif", "label_20170830_20170828_470_694.tif",
                "label_20170830_20170828_463_695.tif", "label_20170830_20170828_484_696.tif",
                "label_20190908_20190906_396_1099.tif"]
TEST_LABELS = ["label_20150919_20150917_496_812.tif", "label_20150919_20150917_497_811.tif", 
               "label_20151112_20151109_474_941.tif", "label_20151112_20151109_485_960.tif", 
               "label_20160314_20160311_452_843.tif", "label_20160314_20160311_452_846.tif",
               "label_20170830_20170826_487_696.tif"] 

DAMP_DEFAULT = 1.0
CU_DEFAULT = 0.523 # 0.447 is sqrt(1/number of looks)
CMAX_DEFAULT = 1.73 # 1.183 is sqrt(1 + 2/number of looks)

class ChannelIndexer:
    def __init__(self, channels):
        self.channels = channels
        self.names = names = ["images", "ndwi", "dem", "slope", "waterbody", "roads"]
        self.included = [all(channels[:3])] + channels[4:] # we ignore index 3 = b8 channel

    def has_image(self):
        return all(self.channels[:3])

    def has_ndwi(self):
        return self.channels[4]

    def has_dem(self):
        return self.channels[5]

    def has_slope(self):
        return self.channels[6]

    def has_waterbody(self):
        return self.channels[7]

    def has_roads(self):
        return self.channels[8]

    def get_channel_names(self):
        return [name for name, is_included in zip(self.names, self.included) if is_included]

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

def trainMeanStd(batch_size=10, channels=[True] * 9, sample_dir='../samples_200_5_4_35/', label_dir='../labels/'):
    """Calculate mean and std across channels of all training tiles used to generate patches."""
    def channel_collate(batch):
        # concatenate channel wise for all samples in batch
        samples = torch.cat(batch, 1)
        return samples

    train_mean_std = FloodSampleMeanStd(TRAIN_LABELS, channels=channels, sample_dir=sample_dir, label_dir=label_dir)
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
    """Early stops the training if validation loss doesn't improve after a given patience."""
    def __init__(self, patience=1, min_delta=0):
        self.patience = patience
        self.min_delta = min_delta
        self.counter = 0
        self.best = True
        self.min_validation_loss = float('inf')
        self.metric = None

    def step(self, validation_loss):
        if validation_loss < self.min_validation_loss:
            self.min_validation_loss = validation_loss
            self.counter = 0
            self.best = True
        elif validation_loss > (self.min_validation_loss + self.min_delta):
            self.counter += 1
            self.best = False
        else:
            self.best = False

    def is_stopped(self):
        return self.counter >= self.patience

    def is_best_epoch(self):
        return self.best

    def store_metric(self, metric):
        self.metric = metric

    def get_metric(self):
        return self.metric

def dbToPower(x):
    """Convert SAR raster from db scale to power scale."""
    # set all missing values back to zero
    missing_mask = x == -9999
    nonzero_mask = x != -9999
    x[nonzero_mask] = np.float_power(10, x[nonzero_mask] / 10, dtype=np.float64)  # Inverse of log10 transformation
    x[missing_mask] = 0
    return x

def powerToDb(x):
    """Convert SAR raster from power scale to db scale. Missing values set to -9999."""
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
    """

    # missing data: if center pixel is missing then do nothing!
    # if neighborhood pixels all missing then do nothing!

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
