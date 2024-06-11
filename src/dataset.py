import rasterio
import torch
import numpy as np
import re
from glob import glob
from random import Random
from torch.utils.data import Dataset
from torchvision import transforms

import ctypes
import multiprocessing as mp

class FloodSampleSARDataset(Dataset):
    """An abstract class representing the SAR flood labelling dataset. The entire dataset is
    lazily loaded into CPU memory at initialization and then retrieved by indexing. Subsetting
    channels is done during retrieval.

    The class does not have knowledge of the dimensions of the patches of the dataset which are
    set during preprocessing. The class assumes the data has 8 channels with the first 7 channels
    being:
    
    1. SAR VV
    2. SAR VH
    3. DEM
    4. Slope Y
    5. Slope X
    6. Waterbody
    7. Roads

    And the last channel being the channel with corresponding label.

    Parameters
    ----------
    sample_dir : str
        Path to directory containing dataset (in npy file).
    channels : list[bool]
        List of 7 booleans corresponding to the 7 input channels.
    typ : str
        The subset of the dataset to load: train, val, test.
    transform : obj
        PyTorch transform.
    """
    def __init__(self, sample_dir, channels=[True] * 7, typ="train", transform=None):
        self.sample_dir = sample_dir
        self.channels = channels + [True] # always keep label channel
        self.typ = typ
        self.transform = transform

        # first load data in
        self.dataset = np.load(sample_dir + f"{typ}_patches.npy")

    def __len__(self):
        return self.dataset.shape[0]

    def __getitem__(self, idx):
        patch = self.dataset[idx, self.channels, :, :]
        image = torch.from_numpy(patch[:-1, :, :])
        label = torch.from_numpy(patch[-1, :, :]).unsqueeze(0)

        if self.transform:
            # for standardization only standardize the non-binary channels!
            image = self.transform(image)
            
        return image, label

class FloodSampleDataset(Dataset):
    """An abstract class representing the Sentinel-2 flood labelling dataset. The entire dataset is
    stored as individual files with the 10 input channels of each patch stored in sample_dir and the 
    corresponding label for the patch stored in label_dir. The files are loaded into memory and cached
    in a multiprocessing array to be shared between pytorch dataloader workers.

    After looping over the entire dataset, call set_use_cache(True) in order to start loading from cache.

    The 10 channels in order:

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
    
    Note: the multiprocessing implementation is currently buggy and can hang when run with pytorch 
    dataloaders, thus use num_workers=0 to be safe.
    
    Parameters
    ----------
    sample_dir : str
        Path to directory containing dataset channels.
    label_dir : str
        Path to directory containing dataset labels.
    channels : list[bool]
        List of 10 booleans corresponding to the 10 input channels.
    typ : str
        The subset of the dataset to load: train, val.
    transform : obj
        PyTorch transform.
    random_flip : bool
        Randomly flip patches (vertically or horizontally) for augmentation.
    """
    def __init__(self, sample_dir, labels_dir, channels=[True] * 10, typ="train", transform=None, random_flip=True, seed=41000, size=64):
        self.sample_dir = sample_dir
        self.labels_dir = labels_dir
        self.labels_path = glob(labels_dir + 'label_*.npy')
        self.channels = channels
        self.typ = typ
        self.transform = transform
        self.random_flip = random_flip
        self.seed = seed
        self.size = size

        # cached arrays
        saved_samples_base = mp.Array(ctypes.c_float, len(self.labels_path)*len(channels)*size*size)
        saved_samples_array = np.ctypeslib.as_array(saved_samples_base.get_obj())
        saved_samples_array = saved_samples_array.reshape(len(self.labels_path), len(channels), size, size)

        saved_label_base = mp.Array(ctypes.c_uint8, len(self.labels_path)*size*size)
        saved_label_array = np.ctypeslib.as_array(saved_label_base.get_obj())
        saved_label_array = saved_label_array.reshape(len(self.labels_path), 1, size, size)
        
        self.saved_samples = torch.from_numpy(saved_samples_array)
        self.saved_labels = torch.from_numpy(saved_label_array)
        self.use_cache = False
        
        if random_flip:
            self.random = Random(seed)

    def __len__(self):
        return len(self.labels_path)

    def set_use_cache(self, use_cache):
        self.use_cache = use_cache

    def read_sample(self, idx, filename):
        """Reads a NPY array into a N x H x W Tensor given N channels."""
        if not self.use_cache:
            # filling cache
            raster = np.load(filename)
            
            self.saved_samples[idx] = torch.from_numpy(raster)
        return self.saved_samples[idx]

    def read_label(self, idx, filename):
        """Reads a NPY array into a 1 x H x W Tensor."""
        if not self.use_cache:
            raster = np.load(filename)
            self.saved_labels[idx] = torch.from_numpy(raster)
        return self.saved_labels[idx]
        
    def __getitem__(self, idx):
        label_path = self.labels_path[idx]
        p = re.compile('label_(.+\.npy)')
        m = p.search(label_path)

        if m:
            sample_path = self.sample_dir + '/sample_' + m.group(1)
            # read from memory if already loaded from disk
            image = self.read_sample(idx, sample_path)[self.channels] # only select the channels we need
            label = self.read_label(idx, label_path) # ensure label tensor is of type torch.uint8
        else:
            raise Exception("Improper label path name")

        if self.transform:
            # for standardization only standardize the non-binary channels!
            image = self.transform(image)

        if self.random_flip and self.typ == "train":
            image, label = self.hv_random_flip(image, label)
            
        return image, label

    def hv_random_flip(self, x, y):
        # Random horizontal flipping
        if self.random.random() > 0.5:
            x = torch.flip(x, [2])
            y = torch.flip(y, [2])
            
        # Random vertical flipping
        if self.random.random() > 0.5:
            x = torch.flip(x, [1])
            y = torch.flip(y, [1])
            
        return x, y

class FloodSampleMeanStd(Dataset):
    """An abstract class used to estimate the mean and std of each channel across the entire dataset
    by using the original tiles (not preprocessed) used for generating patches.
    
    Parameters
    ----------
    labels : str
        Path to directory containing dataset channels.
    channels : list[bool]
        List of 10 booleans corresponding to the 10 input channels.
    sample_dir : str
        Path to directory containing raw dataset tiles.
    label_dir : str
        Path to directory containing raw dataset labels.
    """
    def __init__(self, labels, channels=[True] * 10, sample_dir='../sampling/samples_200_5_4_35/',
                 label_dir='../sampling/labels/'):
        self.labels = labels
        self.channels = channels
        self.sample_dir = sample_dir
        self.label_dir = label_dir

    def __len__(self):
        return len(self.labels)

    def stack_channels(self, eid, tile_date):
        """Reads multiple TIF images and stacks them into N x M shape given N channels."""
        tci_file = self.sample_dir + f'{eid}/tci_{tile_date}_{eid}.tif'
        b08_file = self.sample_dir + f'{eid}/b08_{tile_date}_{eid}.tif'
        ndwi_file = self.sample_dir + f'{eid}/ndwi_{tile_date}_{eid}.tif'
        dem_file = self.sample_dir + f'{eid}/dem_{eid}.tif'
        # slope_file = self.sample_dir + f'{eid}/slope_{eid}.tif'
        waterbody_file = self.sample_dir + f'{eid}/waterbody_{eid}.tif'
        roads_file = self.sample_dir + f'{eid}/roads_{eid}.tif'
        with rasterio.open(tci_file) as src:
            tci_raster = src.read()
            tci_raster = (tci_raster / 255).reshape((3, -1))

        with rasterio.open(b08_file) as src:
            b08_raster = src.read().reshape((1, -1))

        with rasterio.open(ndwi_file) as src:
            ndwi_raster = src.read().reshape((1, -1))
    
        with rasterio.open(dem_file) as src:
            dem_raster = src.read()
            slope = np.gradient(dem_raster, axis=(1,2))
            dem_raster = dem_raster.reshape((1, -1))
    
        # with rasterio.open(slope_file) as src:
            # slope_raster = src.read().reshape((1, -1))
        slope_y_raster = slope[0].reshape((1, -1))
        slope_x_raster = slope[1].reshape((1, -1))

        with rasterio.open(waterbody_file) as src:
            waterbody_raster = src.read().reshape((1, -1))

        with rasterio.open(roads_file) as src:
            roads_raster = src.read().reshape((1, -1))

        stack = np.vstack((tci_raster, b08_raster, ndwi_raster, dem_raster, 
                           slope_y_raster, slope_x_raster, waterbody_raster, 
                           roads_raster), dtype=np.float32)
        tensor = torch.from_numpy(stack)
        return tensor
        
    def __getitem__(self, idx):
        label_path = self.label_dir + self.labels[idx]
        p = re.compile('label_(\d{8})_(.+).tif')
        m = p.search(label_path)

        if m:
            tile_date = m.group(1)
            eid = m.group(2)
            image = self.stack_channels(eid, tile_date)[self.channels]
        else:
            raise Exception("Improper label path name")
            
        return image
