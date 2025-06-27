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
import pickle

from utils.utils import TRAIN_LABELS, VAL_LABELS, TEST_LABELS, SRC_DIR, DATA_DIR, SAMPLES_DIR
### TO IMPLEMENT: SAMPLING FROM PREDICTED LABEL TILES ALSO INSTEAD OF HUMAN LABELS

def random_crop(label_names, size, num_samples, rng, pre_sample_dir, sample_dir, label_dir, typ="train"):
    """Uniformly samples patches of dimension size x size across each dataset tile, and saves
    all patches in train, val, test sets into respective npy files.

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
    pre_sample_dir : Path
        Directory to save the sampled patches.
    sample_dir : str
        Directory containing raw S2 tiles for patch sampling.
    label_dir : str
        Directory containing raw S2 tile labels for patch sampling.
    typ : str
        Subset assigned to the saved patches: train, val, test.
    """
    logger = logging.getLogger('preprocessing')
    
    # 11 channels for 10 data + 1 label channel
    total_patches = num_samples * len(label_names)
    dataset = np.empty((total_patches, 11, size, size), dtype=np.float32)

    tiles = []
    for label_file in label_names:
        p = re.compile('label_(\d{8})_(.+).tif')
        m = p.search(label_file)

        if m:
            tile_date = m.group(1)
            eid = m.group(2)
        else:
            raise ValueError(f'Label file {label_file} does not match expected format.')

        with rasterio.open(SAMPLES_DIR / label_dir / label_file) as src:
            label_raster = src.read([1, 2, 3])
            # if label has any values != 0 or 255 then print to log!
            if np.any((label_raster > 0) & (label_raster < 255)):
                logger.debug(f'{label_file} values are not 0 or 255.')
                
            label_binary = np.where(label_raster[0] != 0, 1, 0)
            label_binary = np.expand_dims(label_binary, axis = 0)

            HEIGHT = src.height
            WIDTH = src.width

        tci_file = SAMPLES_DIR / sample_dir / eid / f'tci_{tile_date}_{eid}.tif'
        b08_file = SAMPLES_DIR / sample_dir / eid / f'b08_{tile_date}_{eid}.tif'
        ndwi_file = SAMPLES_DIR / sample_dir / eid / f'ndwi_{tile_date}_{eid}.tif'
        dem_file = SAMPLES_DIR / sample_dir / eid / f'dem_{eid}.tif'
        waterbody_file = SAMPLES_DIR / sample_dir / eid / f'waterbody_{eid}.tif'
        roads_file = SAMPLES_DIR / sample_dir / eid / f'roads_{eid}.tif'

        with rasterio.open(tci_file) as src:
            tci_raster = src.read()
            tci_floats = (tci_raster / 255).astype(np.float32)

        with rasterio.open(b08_file) as src:
            b08_raster = src.read()

        with rasterio.open(ndwi_file) as src:
            ndwi_raster = src.read()
    
        with rasterio.open(dem_file) as src:
            dem_raster = src.read()

        # calculate xy gradient with np.gradient
        slope = np.gradient(dem_raster, axis=(1,2))
        slope_y_raster, slope_x_raster = slope

        with rasterio.open(waterbody_file) as src:
            waterbody_raster = src.read()

        with rasterio.open(roads_file) as src:
            roads_raster = src.read()

        # stack all tiles
        stacked_tile = np.vstack((tci_floats, b08_raster, ndwi_raster, dem_raster, 
                                    slope_y_raster, slope_x_raster, waterbody_raster, 
                                    roads_raster, label_binary), dtype=np.float32)
        tiles.append(stacked_tile)
    logger.info('Tiles loaded.')

    logger.info('Sampling random patches...')
    # choose n random coordinates within each tile
    for i, tile in enumerate(tiles):
        patches_sampled = 0
        _, HEIGHT, WIDTH = tile.shape

        while patches_sampled < num_samples:
            x = int(rng.uniform(0, HEIGHT - size))
            y = int(rng.uniform(0, WIDTH - size))

            patch = tile[:, x : x + size, y : y + size]

            # if contains missing values in tci or ndwi, toss out and resample
            if np.any(patch[0] == 0) or np.any(patch[4] == -999999):
                continue

            dataset[i * num_samples + patches_sampled] = patch[:11]
            patches_sampled += 1

    output_file = pre_sample_dir / f'{typ}_patches.npy'
    np.save(output_file, dataset)

def loadMaskedStack(img_dt, eid, sample_dir):
    """Load and mask the stack of channels.

    Parameters
    ----------
    img_dt : str
        Image date.
    eid : str
        Event id.
    sample_dir : str
        Directory containing S2 tiles for patch sampling.

    Returns
    -------
    masked_stack : ndarray
        Stack of channels with missing values masked out.
    """
    sample_path = SAMPLES_DIR / sample_dir
    tci_file = sample_path / eid / f'tci_{img_dt}_{eid}.tif'
    b08_file = sample_path / eid / f'b08_{img_dt}_{eid}.tif'
    ndwi_file = sample_path / eid / f'ndwi_{img_dt}_{eid}.tif'
    dem_file = sample_path / eid / f'dem_{eid}.tif'
    waterbody_file = sample_path / eid / f'waterbody_{eid}.tif'
    roads_file = sample_path / eid / f'roads_{eid}.tif'
    # flowlines_file = sample_path / eid / f'flowlines_{eid}.tif' - currently not used in the dataset

    with rasterio.open(tci_file) as src:
        tci_raster = src.read().reshape((3, -1))
        tci_floats = (tci_raster / 255).astype(np.float32) # first scale rgb to 0-1 then normalize
    
    with rasterio.open(b08_file) as src:
        b08_raster = src.read().reshape((1, -1))
    
    with rasterio.open(ndwi_file) as src:
        ndwi_raster = src.read().reshape((1, -1))

    with rasterio.open(dem_file) as src:
        dem_raster = src.read()

    slope = np.gradient(dem_raster, axis=(1,2))
    slope_y_raster, slope_x_raster = slope

    dem_raster = dem_raster.reshape((1, -1))
    slope_y_raster = slope_y_raster.reshape((1, -1))
    slope_x_raster = slope_x_raster.reshape((1, -1))

    with rasterio.open(waterbody_file) as src:
        waterbody_raster = src.read().reshape((1, -1))

    with rasterio.open(roads_file) as src:
        roads_raster = src.read().reshape((1, -1))

    # with rasterio.open(cloud_file) as src:
    #     cloud_raster = src.read().reshape((1, -1))

    mask = (tci_raster[0] != 0) & (ndwi_raster[0] != -999999)

    stack = np.vstack((tci_floats, b08_raster, ndwi_raster, dem_raster, slope_y_raster,
                        slope_x_raster, waterbody_raster, roads_raster), dtype=np.float32)

    masked_stack = stack[:, mask]

    return masked_stack

def trainMean(train_events, sample_dir):
    """Calculate mean and std of channels.

    Parameters
    ----------
    train_events : list[tuple[str, str]]
        List of training flood event folders where raw data tiles are stored.
        First element is the image date, second element is the event id.
    sample_dir : str
        Directory containing S2 tiles for patch sampling.

    Returns
    -------
    overall_channel_mean : ndarray
        Channel means.
    """
    logger = logging.getLogger('preprocessing')
    count = 0
    total_sum = np.zeros(10, dtype=np.float64) # 10 channels for original s2 data

    for img_dt, eid in train_events:
        masked_stack = loadMaskedStack(img_dt, eid, sample_dir)

        # calculate mean and var across channels
        channel_sums = np.sum(masked_stack, axis=1)
        total_sum += channel_sums

        count += masked_stack.shape[1]

    overall_channel_mean = total_sum / count

    # calculate final statistics
    return overall_channel_mean

def trainStd(train_events, train_means, sample_dir):
    """Calculate std of channels.

    Parameters
    ----------
    train_events : list[tuple[str, str]]
        List of training flood event folders where raw data tiles are stored.
        First element is the image date, second element is the event id.
    train_means : ndarray
        Channel means.
    sample_dir : str
        Directory containing S2 tiles for patch sampling.

    Returns
    -------
    overall_channel_std : ndarray
        Channel stds.
    """
    logger = logging.getLogger('preprocessing')
    count = 0
    variances = np.zeros(10, dtype=np.float64) # 10 channels for original s2 data

    for img_dt, eid in train_events:
        masked_stack = loadMaskedStack(img_dt, eid, sample_dir)

        # calculate var across channels
        deviations = masked_stack - np.array(train_means).reshape(-1, 1)
        squared_deviations = deviations ** 2
        channel_variances = squared_deviations.sum(axis=1)
        variances += channel_variances

        count += masked_stack.shape[1]

    overall_channel_variances = variances / count
    overall_channel_std = np.sqrt(overall_channel_variances)

    # calculate final std statistics
    return overall_channel_std

def main(size, samples, seed, method='random', sample_dir='samples_200_5_4_35/', label_dir='labels/'):
    """Preprocesses raw S2 tiles and corresponding labels into smaller patches. The data will be stored
    as separate npy files for train, val, and test sets, along with a mean_std.pkl file containing the
    mean and std of the training tiles.

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

    # make our preprocess directory
    pre_sample_dir = DATA_DIR / 's2' / f'samples_{size}_{samples}'
    pre_sample_dir.mkdir(parents=True, exist_ok=True)

    # get event directories from the training labels for mean and std calculation
    p = re.compile('label_(\d{8})_(.+).tif')
    train_events = [(p.search(label).group(1), p.search(label).group(2)) for label in TRAIN_LABELS]

    # calculate mean and std of train tiles
    logger.info('Calculating mean and std of training tiles...')
    mean = trainMean(train_events, sample_dir)
    std = trainStd(train_events, mean, sample_dir)
    logger.info('Mean and std of training tiles calculated.')

    # also store training mean std statistics in file
    stats_file = pre_sample_dir / f'mean_std_{size}_{samples}.pkl'
    with open(stats_file, 'wb') as f:
        pickle.dump((mean, std), f)
    logger.info('Training mean and std statistics saved.')
    
    if method == 'random':
        rng = Random(seed)
        random_crop(TRAIN_LABELS, size, samples, rng, pre_sample_dir, sample_dir, label_dir, typ="train")
        random_crop(VAL_LABELS, size, samples, rng, pre_sample_dir, sample_dir, label_dir, typ="val")
        random_crop(TEST_LABELS, size, samples, rng, pre_sample_dir, sample_dir, label_dir, typ="test")
        logger.info('Random samples generated.')

    logger.debug('Preprocessing complete.')

if __name__ == '__main__':
    parser = argparse.ArgumentParser(prog='preprocess_s2', description='Preprocesses 4km x 4km PRISM tiles into smaller tiles via random crop method.')
    parser.add_argument('-x', '--size', dest='size', type=int, default=64, help='pixel width of patch (default: 64)')
    parser.add_argument('-n', '--samples', dest='samples', type=int, default=1000, help='number of samples per image (default: 1000)')
    parser.add_argument('-s', '--seed', dest='seed', type=int, default=433002, help='random number generator seed (default: 433002)')
    parser.add_argument('-m', '--method', dest='method', default='random', choices=['random'], help='sampling method (default: random)')
    parser.add_argument('--sdir', dest='sample_dir', default='samples_200_5_4_35', help='data directory in the sampling folder (default: samples_200_5_4_35)')
    parser.add_argument('--ldir', dest='label_dir', default='labels', help='label directory in the sampling folder (default: labels)')
    
    args = parser.parse_args()
    sys.exit(main(args.size, args.samples, args.seed, method=args.method, sample_dir=args.sample_dir, label_dir=args.label_dir))
