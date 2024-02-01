import rasterio
import torch
import numpy as np
import re
from glob import glob
from random import Random
from torch.utils.data import Dataset
from torchvision import transforms

class FloodSampleDataset(Dataset):
    def __init__(self, sample_dir, labels_dir, typ="train", transform=None, random_flip=True, seed=41000):
        self.sample_dir = sample_dir
        self.labels_dir = labels_dir
        self.labels_path = glob(labels_dir + 'label_*.tif')
        self.typ = typ
        self.transform = transform
        self.random_flip = random_flip
        self.seed = seed
        
        if random_flip:
            self.random = Random(seed)

    def __len__(self):
        return len(self.labels_path)

    @staticmethod
    def read_image(filename):
        """Reads a TIF image into a N x H x W Tensor given N channels."""
        with rasterio.open(filename) as src:
            raster = src.read()
    
        tensor = torch.from_numpy(raster)
        return tensor
        
    def __getitem__(self, idx):
        label_path = self.labels_path[idx]
        p = re.compile('label_(.+\.tif)')
        m = p.search(label_path)

        if m:
            sample_path = self.sample_dir + '/sample_' + m.group(1)
            image = self.read_image(sample_path)
            label = self.read_image(label_path) # ensure label tensor is of type torch.uint8
        else:
            raise Exception("Improper label path name")

        if self.transform:
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
    def __init__(self, labels, sample_dir='../samples_200_5_4_35/', label_dir='../labels/'):
        self.labels = labels
        self.sample_dir = sample_dir
        self.label_dir = label_dir

    def __len__(self):
        return len(self.labels)

    @staticmethod
    def stack_channels(tci_file, b08_file, ndwi_file):
        """Reads multiple TIF images and stacks them into N x M shape given N channels."""
        with rasterio.open(tci_file) as src:
            tci_raster = src.read()
            tci_raster = (tci_raster / 255).reshape((3, -1))

        with rasterio.open(b08_file) as src:
            b08_raster = src.read().reshape((1, -1))

        with rasterio.open(ndwi_file) as src:
            ndwi_raster = src.read().reshape((1, -1))

        stack = np.vstack((tci_raster, b08_raster, ndwi_raster), dtype=np.float32)
        tensor = torch.from_numpy(stack)
        return tensor
        
    def __getitem__(self, idx):
        label_path = self.label_dir + self.labels[idx]
        p = re.compile('label_(\d{8})_(.+).tif')
        m = p.search(label_path)

        if m:
            tile_date = m.group(1)
            eid = m.group(2)
            tci_file = self.sample_dir + f'{eid}/tci_{tile_date}_{eid}.tif'
            b08_file = self.sample_dir + f'{eid}/b08_{tile_date}_{eid}.tif'
            ndwi_file = self.sample_dir + f'{eid}/ndwi_{tile_date}_{eid}.tif'
            
            image = self.stack_channels(tci_file, b08_file, ndwi_file)
        else:
            raise Exception("Improper label path name")
            
        return image
