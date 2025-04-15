from glob import glob
import matplotlib.pyplot as plt
import rasterio
import numpy as np
from pathlib import Path
import re
import sys
import argparse
from random import Random
import logging
from utils import TRAIN_LABELS, VAL_LABELS, TEST_LABELS

def random_crop(label_names, size, num_samples, rng, method, sample_dir, label_dir, typ="train"):
    """Uniformly samples patches of dimension size x size across each dataset tile.

    Parameters
    ----------
    label_names : list[str]
        List of file paths to manually ground-truthed tile labels.
    size : int
        Size of the sampled patches.
    num_samples : int
        Number of patches to sample per raw S2 tile.
    rng : obj
        Random number generator.
    method : str
        Sampling method.
    sample_dir : str
        Directory containing raw S2 tiles for patch sampling.
    label_dir : str
        Directory containing raw S2 tile labels for patch sampling.
    typ : str
        Subset assigned to the saved patches: train, val, test.
    """
    pre_label_dir = f'data/s2/{method}/labels{size}_{typ}_{num_samples}/'
    pre_sample_dir = f'data/s2/{method}/samples{size}_{typ}_{num_samples}/'
    Path(pre_label_dir).mkdir(parents=True, exist_ok=True)
    Path(pre_sample_dir).mkdir(parents=True, exist_ok=True)

    for label_file in label_names:
        p = re.compile('label_(\d{8})_(.+).tif')
        m = p.search(label_file)

        if m:
            tile_date = m.group(1)
            eid = m.group(2)
        else:
            logger.info(f'Skipping {label_file}...')
            continue

        with rasterio.open(label_dir + label_file) as src:
            label_raster = src.read([1, 2, 3])
            # if label has any values != 0 or 255 then print to log!
            if np.any((label_raster > 0) & (label_raster < 255)):
                logger.debug(f'{label_file} values are not 0 or 255.')
                
            label_binary = np.where(label_raster[0] != 0, 1, 0)

            HEIGHT = src.height
            WIDTH = src.width

        tci_file = sample_dir + f'{eid}/tci_{tile_date}_{eid}.tif'
        b08_file = sample_dir + f'{eid}/b08_{tile_date}_{eid}.tif'
        ndwi_file = sample_dir + f'{eid}/ndwi_{tile_date}_{eid}.tif'
        dem_file = sample_dir + f'{eid}/dem_{eid}.tif'
        # slope_file = sample_dir + f'{eid}/slope_{eid}.tif'
        waterbody_file = sample_dir + f'{eid}/waterbody_{eid}.tif'
        roads_file = sample_dir + f'{eid}/roads_{eid}.tif'

        with rasterio.open(tci_file) as src:
            tci_raster = src.read()

        with rasterio.open(b08_file) as src:
            b08_raster = src.read()

        with rasterio.open(ndwi_file) as src:
            ndwi_raster = src.read()
    
        with rasterio.open(dem_file) as src:
            dem_raster = src.read()
    
        # with rasterio.open(slope_file) as src:
            # slope_raster = src.read()

        # calculate xy gradient with np.gradient
        slope = np.gradient(dem_raster, axis=(1,2))
        slope_y_raster, slope_x_raster = slope

        with rasterio.open(waterbody_file) as src:
            waterbody_raster = src.read()

        with rasterio.open(roads_file) as src:
            roads_raster = src.read()

        # choose n random coordinates within the tile
        sampled = 0
        while sampled < num_samples:
            x = int(rng.uniform(0, HEIGHT - size))
            y = int(rng.uniform(0, WIDTH - size))

            # check if already completed
            if Path(pre_sample_dir + f'sample_{tile_date}_{eid}_{x}_{y}.npy').exists():
                continue

            tci_tile = tci_raster[:, x : x + size, y : y + size]

            # if contains missing values, toss out and resample
            if np.any(tci_tile == 0):
                continue

            tci_tile = (tci_tile / 255).astype(np.float32)
            b08_tile = b08_raster[:, x : x + size, y : y + size]
            b08_tile = b08_tile.astype(np.float32)

            ndwi_tile = ndwi_raster[:, x : x + size, y : y + size]
            ndwi_tile = ndwi_tile.astype(np.float32)
    
            dem_tile = dem_raster[:, x : x + size, y : y + size]
            dem_tile = dem_tile.astype(np.float32)
    
            # slope_tile = slope_raster[:, x : x + size, y : y + size]
            # slope_tile = slope_tile.astype(np.float32)
            slope_y_tile = slope_y_raster[:, x : x + size, y : y + size]
            slope_y_tile = slope_y_tile.astype(np.float32)
            slope_x_tile = slope_x_raster[:, x : x + size, y : y + size]
            slope_x_tile = slope_x_tile.astype(np.float32)

            waterbody_tile = waterbody_raster[:, x : x + size, y : y + size]
            waterbody_tile = waterbody_tile.astype(np.float32)

            roads_tile = roads_raster[:, x : x + size, y : y + size]
            roads_tile = roads_tile.astype(np.float32)

            label_tile = label_binary[x : x + size, y : y + size]
            label_tile = label_tile.astype(np.uint8)
            
            stacked_tile = np.vstack((tci_tile, b08_tile, ndwi_tile, dem_tile, 
                                      slope_y_tile, slope_x_tile, waterbody_tile, 
                                      roads_tile), dtype=np.float32)

            # save 64 x 64 tile to npy file
            np.save(pre_sample_dir + f'sample_{tile_date}_{eid}_{x}_{y}.npy', stacked_tile)
            np.save(pre_label_dir + f'label_{tile_date}_{eid}_{x}_{y}.npy', label_tile)
            sampled += 1

def main(size, samples, seed, method='random', sample_dir='../samples_200_5_4_35/', label_dir='../labels/'):
    """Preprocesses raw S2 tiles and corresponding labels into smaller patches.

    Parameters
    ----------
    size : int
        Size of the sampled patches.
    samples : int
        Number of patches to sample per raw S2 tile.
    seed : int
    method : str
        Sampling method.
    sample_dir : str
        Directory containing raw S2 tiles for patch sampling.
    label_dir : str
        Directory containing raw S2 tile labels for patch sampling.
    """
    logger = logging.getLogger('preprocessing')
    logger.setLevel(logging.DEBUG)
    formatter = logging.Formatter('%(asctime)s - %(name)s - %(levelname)s - %(message)s')
    handler = logging.StreamHandler(sys.stdout)
    handler.setFormatter(formatter)
    handler.setLevel(logging.DEBUG)
    logger.addHandler(handler)
    logger.propagate = False
    
    rng = Random(seed)
    if method == 'random':
        random_crop(TRAIN_LABELS, size, samples, rng, method, sample_dir, label_dir, typ="train")
        random_crop(VAL_LABELS, size, samples, rng, method, sample_dir, label_dir, typ="val")
        random_crop(TEST_LABELS, size, samples, rng, method, sample_dir, label_dir, typ="test")

    logger.debug('Preprocessing complete.')

if __name__ == '__main__':
    parser = argparse.ArgumentParser(prog='preprocess_s2', description='Preprocesses 4km x 4km PRISM tiles into smaller tiles via random crop method.')
    parser.add_argument('-x', '--size', dest='size', type=int, default=64, help='pixel width of patch (default: 64)')
    parser.add_argument('-n', '--samples', dest='samples', type=int, default=1000, help='number of samples per image (default: 1000)')
    parser.add_argument('-s', '--seed', dest='seed', type=int, default=433002, help='random number generator seed (default: 433002)')
    parser.add_argument('-m', '--method', dest='method', default='random', choices=['random'], help='sampling method (default: random)')
    parser.add_argument('--sdir', dest='sample_dir', default='../sampling/samples_200_5_4_35/', help='(default: ../sampling/samples_200_5_4_35/)')
    parser.add_argument('--ldir', dest='label_dir', default='../sampling/labels/', help='(default: ../sampling/labels/)')
    
    args = parser.parse_args()
    sys.exit(main(args.size, args.samples, args.seed, method=args.method, sample_dir=args.sample_dir, label_dir=args.label_dir))
