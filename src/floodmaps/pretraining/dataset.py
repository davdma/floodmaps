import torch
import numpy as np
from random import Random
from torch.utils.data import Dataset
from pathlib import Path

class WorldFloodsS2Dataset(Dataset):
    """An abstract class representing the world floods S2 dataset. This dataset class
    assumes that the preprocessed data has the following channels:

    1. B04 Red Reflectance
    2. B03 Green Reflectance
    3. B02 Blue Reflectance
    4. B08 Near Infrared
    5. NDWI

    1-4 are reflectance values divided by 10000, and ndwi is in [-1, 1] range.

    The last N+1 channel is the water mask label

    Parameters
    ----------
    sample_dir : str
        Path to preprocess directory containing dataset (in npy file).
    channels : list[bool]
        List of N booleans corresponding to the N input channels.
    typ : str
        The subset of the dataset to load: train, val, test.
    transform : obj
        PyTorch transform.
    random_flip : bool
        Randomly flip patches (vertically or horizontally) for augmentation.
    seed : int
        Random seed.

    Returns
    -------
    image : torch.Tensor
        The input image (11 channels).
    label : torch.Tensor
        The binary label (1 channel).
    supplementary : torch.Tensor
        The TCI (3 channels) + NLCD (1 channel).
    """
    def __init__(self, sample_dir, channels=[True] * 5, typ="train", random_flip=False, transform=None, seed=3200):
        self.sample_dir = Path(sample_dir)
        self.channels = channels + [True] # always keep label
        self.typ = typ
        self.random_flip = random_flip
        self.transform = transform
        self.seed = seed

        base = np.load(self.sample_dir / f"{typ}_patches.npy")

        # One-time channel selection to avoid per-sample advanced indexing copies
        if not all(self.channels):
            base = np.ascontiguousarray(base[:, self.channels, :, :])
        
        self.dataset = base

        if random_flip:
            self.random = Random(seed)

    def __len__(self):
        return self.dataset.shape[0]

    def __getitem__(self, idx):
        patch = self.dataset[idx]
        image = torch.from_numpy(patch[:-1, :, :])
        label = torch.from_numpy(patch[-1, :, :]).unsqueeze(0)

        if self.transform:
            image = self.transform(image)

        # add random flips and rotations? Could help prevent learning constant shift...
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