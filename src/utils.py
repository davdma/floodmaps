import torch
from torch.utils.data import Dataset, DataLoader
import rasterio
import numpy as np
from dataset import FloodSampleMeanStd

TRAIN_LABELS = ["label_20150919_20150917_496_811.tif", "label_20150919_20150917_497_812.tif",
                "label_20151112_20151109_472_940.tif", "label_20151112_20151109_473_939.tif", 
                "label_20151112_20151109_473_940.tif", "label_20151112_20151109_473_941.tif", 
                "label_20151112_20151109_474_942.tif", "label_20160314_20160311_451_838.tif", 
                "label_20160314_20160311_451_839.tif", "label_20160307_20160311_452_846.tif",
                "label_20160314_20160311_452_845.tif", "label_20160314_20160311_453_845.tif"]
TEST_LABELS = ["label_20150919_20150917_496_812.tif", "label_20150919_20150917_497_811.tif", 
               "label_20151112_20151109_474_941.tif", "label_20151112_20151109_485_960.tif", 
               "label_20160314_20160311_452_843.tif", "label_20160314_20160311_452_846.tif"] 

def wet_label(image, crop_size, num_pixel=100):
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
    def __init__(self, patience=1, min_delta=0):
        self.patience = patience
        self.min_delta = min_delta
        self.counter = 0
        self.min_validation_loss = float('inf')

    def early_stop(self, validation_loss):
        if validation_loss < self.min_validation_loss:
            self.min_validation_loss = validation_loss
            self.counter = 0
        elif validation_loss > (self.min_validation_loss + self.min_delta):
            self.counter += 1
            if self.counter >= self.patience:
                return True
        return False
