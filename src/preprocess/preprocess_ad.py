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
from sklearn.model_selection import train_test_split

from utils.utils import SRC_DIR, SAMPLES_DIR, enhanced_lee_filter
### TO IMPLEMENT: PATHS WITH SRC_DIR, SAMPLES_DIR

def generate_patches(events, size, num_samples, rng, pre_sample_dir, sample_dir, kernel_size=5, typ="train"):
    """Uniformly samples sar patches of dimension size x size across each dataset tile. The
    patches are saved together into one file with enhanced lee filter.

    Parameters
    ----------
    events : list[str]
        List of flood event folders where raw data tiles are stored.
    size : int
        Size of the sampled patches.
    num_samples : int
        Number of patches to sample per raw S1 tile.
    rng : obj
        Random number generator.
    pre_sample_dir : Path
        Directory to save the preprocessed patches.
    sample_dir : str
        Directory containing raw S1 tiles for patch sampling.
    kernel_size : int
        Kernel size for enhanced lee Filter.
        See https://desktop.arcgis.com/en/arcmap/latest/manage-data/raster-and-images/speckle-function.htm.
    typ : str
        Subset assigned to the saved patches: train, val, test.
    """
    logger = logging.getLogger('preprocessing')

    # first load all samples into memory
    # SAR Preprocessing: labels will be stored in event sample folder
    logger.info('Generating patches. Loading tiles into memory...')
    tiles = []
    p1 = re.compile('\d{8}_\d+_\d+')
    p2 = re.compile('pred_(\d{8})_.+.tif')
    total_iterations = len(events)
    for i, event in enumerate(events):
        m = p1.search(event)

        if m:
            eid = m.group(0)
        else:
            logger.info(f'No matching eid. Skipping {event}...')
            continue

        # search for sar files only
        for sar_vv_file in glob(event + f'/sar_*_vv.tif'):
            sar_vh_file = sar_vv_file[:-6] + 'vh.tif'

            # DEM added
            dem_file = event + f'/dem_{eid}.tif'
            with rasterio.open(dem_file) as src:
                dem_raster = src.read()

            with rasterio.open(sar_vv_file) as src:
                vv_raster = src.read()

            with rasterio.open(sar_vh_file) as src:
                vh_raster = src.read()

            # apply speckle filter to sar:
            vv_raster_lee = np.expand_dims(enhanced_lee_filter(vv_raster[0], kernel_size=5).astype(np.float32), axis=0)
            vh_raster_lee = np.expand_dims(enhanced_lee_filter(vh_raster[0], kernel_size=5).astype(np.float32), axis=0)
            stacked_raster = np.vstack((vv_raster, vh_raster, vv_raster_lee, vh_raster_lee, dem_raster), dtype=np.float32)

            tiles.append(stacked_raster)

        logger.info(f"Loading tiles into memory: iteration {i}/{total_iterations} completed.")
    logger.info('Tiles loaded.')

    # loop over all events to generate samples for each epoch
    logger.info('Beginning patch sampling...')
    total_patches = num_samples * len(tiles)
    dataset = np.empty((total_patches, 5, size, size), dtype=np.float32)

    # sample given number of patches per tile
    for i, tile in enumerate(tiles):
        patches_sampled = 0
        _, HEIGHT, WIDTH = tile.shape

        while patches_sampled < num_samples:
            x = int(rng.uniform(0, HEIGHT - size))
            y = int(rng.uniform(0, WIDTH - size))

            patch = tile[:, x : x + size, y : y + size]

            # filter out missing vv or vh tiles
            if np.any(patch[0] == -9999) or np.any(patch[1] == -9999):
                continue

            dataset[i * num_samples + patches_sampled] = patch
            patches_sampled += 1

    np.save(pre_sample_dir / f'{typ}_patches.npy', dataset)

    logger.info('Sampling complete.')

def trainMean(train_events, sample_dir, kernel_size=5):
    """Calculate mean and std of channels, filtering out cloud pixels and missing sar pixels.

    Parameters
    ----------
    train_events : list[str]
        List of training flood event folders where raw data tiles are stored.
    sample_dir : str
        Directory containing raw S1 tiles for patch sampling.
    kernel_size : int
        Kernel size for enhanced lee filter.
        See https://desktop.arcgis.com/en/arcmap/latest/manage-data/raster-and-images/speckle-function.htm.

    Returns
    -------
    overall_channel_mean : ndarray
        Channel means.
    """
    logger = logging.getLogger('preprocessing')

    # calculate mean each event tile
    # since we have n > 500 this is good approximation of pop mean
    logger.info('Train mean calculation start.')
    count = 0
    means = np.zeros(4)

    p1 = re.compile('\d{8}_\d+_\d+')
    p2 = re.compile('pred_(\d{8})_.+.tif')
    total_iterations = len(train_events)
    for i, event in enumerate(train_events):
        m = p1.search(event)

        if m:
            eid = m.group(0)
        else:
            logger.info(f'No matching eid during mean std calculation. Skipping {event}...')
            continue

        # search for label file + sar file
        # look for labels w tci + sar pairings
        for sar_vv_file in glob(event + f'/sar_*_vv.tif'):
            sar_vh_file = sar_vv_file[:-6] + 'vh.tif'

            with rasterio.open(sar_vv_file) as src:
                vv_raster = src.read()

            with rasterio.open(sar_vh_file) as src:
                vh_raster = src.read()

            # apply speckle filter to sar: this may be slow
            vv_raster_lee = np.expand_dims(enhanced_lee_filter(vv_raster[0], kernel_size=5).astype(
                np.float32), axis=0).reshape((1, -1))
            vh_raster_lee = np.expand_dims(enhanced_lee_filter(vh_raster[0], kernel_size=5).astype(
                np.float32), axis=0).reshape((1, -1))

            vv_raster = vv_raster.reshape((1, -1))
            vh_raster = vh_raster.reshape((1, -1))

            mask = (vv_raster[0] != -9999) & (vh_raster[0] != -9999)

            stack = np.vstack((vv_raster, vh_raster, vv_raster_lee, vh_raster_lee), dtype=np.float32)

            masked_stack = stack[:, mask]

            # calculate mean and var across channels
            channel_means = np.mean(masked_stack, axis=1)
            means += channel_means

            count += 1
        logger.info(f'Calculating train mean: iteration {i}/{total_iterations} completed.')

    overall_channel_mean = means / count

    # calculate final statistics
    return overall_channel_mean

def trainStd(train_events, train_means, sample_dir, kernel_size=5):
    """Calculate mean and std of channels, filtering out cloud pixels and missing sar pixels.

    Parameters
    ----------
    train_events : list[str]
        List of training flood event folders where raw data tiles are stored.
    train_means : ndarray
        Channel means.
    sample_dir : str
        Directory containing raw S1 tiles for patch sampling.
    filter : str
        Specifying filter='raw' does not apply any filters to the patches. Using filter='lee' applies Enhanced Lee Filter.
        See https://desktop.arcgis.com/en/arcmap/latest/manage-data/raster-and-images/speckle-function.htm.

    Returns
    -------
    overall_channel_std : ndarray
        Channel stds.
    """
    logger = logging.getLogger('preprocessing')

    logger.info('Train std calculation start.')
    count = 0
    variances = np.zeros(4)

    p1 = re.compile('\d{8}_\d+_\d+')
    p2 = re.compile('pred_(\d{8})_.+.tif')
    total_iterations = len(train_events)
    for i, event in enumerate(train_events):
        m = p1.search(event)

        if m:
            eid = m.group(0)
        else:
            logger.info(f'No matching eid during mean std calculation. Skipping {event}...')
            continue

        # search for label file + sar file
        # look for labels w tci + sar pairings
        for sar_vv_file in glob(event + f'/sar_*_vv.tif'):
            sar_vh_file = sar_vv_file[:-6] + 'vh.tif'

            with rasterio.open(sar_vv_file) as src:
                vv_raster = src.read()

            with rasterio.open(sar_vh_file) as src:
                vh_raster = src.read()

            # apply speckle filter to sar:
            vv_raster_lee = np.expand_dims(enhanced_lee_filter(vv_raster[0], kernel_size=kernel_size).astype(
                np.float32), axis=0).reshape((1, -1))
            vh_raster_lee = np.expand_dims(enhanced_lee_filter(vh_raster[0], kernel_size=kernel_size).astype(
                np.float32), axis=0).reshape((1, -1))

            vv_raster = vv_raster.reshape((1, -1))
            vh_raster = vh_raster.reshape((1, -1))

            mask = (vv_raster[0] != -9999) & (vh_raster[0] != -9999)

            stack = np.vstack((vv_raster, vh_raster, vv_raster_lee, vh_raster_lee), dtype=np.float32)

            masked_stack = stack[:, mask]

            # Subtract the accurate means from the data
            deviations = masked_stack - np.array(train_means).reshape(-1, 1)
            squared_deviations = deviations ** 2
            channel_variances = squared_deviations.mean(axis=1)
            variances += channel_variances

            count += 1
            logger.debug(f'Count {count}')
        logger.info(f'Calculating train std: iteration {i}/{total_iterations} completed.')

    overall_channel_variances = variances / count
    overall_channel_std = np.sqrt(overall_channel_variances)

    # calculate final statistics
    return overall_channel_std


def main(size, samples, seed, method='random', kernel_size=5, sample_dir='samples_200_6_4_10_sar/'):
    """Preprocesses raw S1 tiles into smaller patches. This preprocessing script saves only SAR layers and
    its enhanced lee counterpart useful for despeckling tasks.

    Parameters
    ----------
    size : int
        Size of the sampled patches.
    samples : int
        Number of patches to sample per raw S2 tile.
    seed : int
    kernel_size : int
        Kernel size for enhanced lee filter.
    sample_dir : str
        Directory containing raw S1 tiles for patch sampling.
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
    pre_sample_dir = DATA_DIR / 'ad' / f'samples_{size}_{samples}_dem'
    pre_sample_dir.mkdir(parents=True, exist_ok=True)

    sample_path = SAMPLES_DIR / sample_dir

    # randomly select samples to be in train and test set
    all_events = list(sample_path.glob('[0-9]*'))
    train_events, val_test_events = train_test_split(all_events, test_size=0.2, random_state=seed - 20)
    val_events, test_events = train_test_split(val_test_events, test_size=0.5, random_state=seed + 1222)

    # calculate mean and std of train tiles
    mean = trainMean(train_events, sample_dir, kernel_size=kernel_size)
    std = trainStd(train_events, mean, sample_dir, kernel_size=kernel_size)

    # also store training mean std statistics in file
    stats_file = pre_sample_dir / f'mean_std_{kernel_size}_{size}_{samples}_dem.pkl'
    with open(stats_file, 'wb') as f:
        pickle.dump((mean, std), f)
    logger.info('Training mean and std statistics saved.')

    if method == 'random':
        rng = Random(seed)
        generate_patches(train_events, size, samples, rng, pre_sample_dir, sample_dir, kernel_size=kernel_size, typ="train")
        generate_patches(val_events, size, samples, rng, pre_sample_dir, sample_dir, kernel_size=kernel_size, typ="val")
        generate_patches(test_events, size, samples, rng, pre_sample_dir, sample_dir, kernel_size=kernel_size, typ="test")

    logger.debug('Preprocessing complete.')

if __name__ == '__main__':
    parser = argparse.ArgumentParser(prog='preprocess_ad', description='Preprocesses 4km x 4km SAR tiles into smaller patches via random cropping - enhanced lee filtered patches are saved also.')
    parser.add_argument('-x', '--size', dest='size', type=int, default=64, help='pixel width of patch (default: 64)')
    parser.add_argument('-n', '--samples', dest='samples', type=int, default=500, help='number of samples per image (default: 500)')
    parser.add_argument('-s', '--seed', dest='seed', type=int, default=433002, help='random number generator seed (default: 433002)')
    parser.add_argument('-m', '--method', dest='method', default='random', choices=['random'], help='sampling method (default: random)')
    parser.add_argument('--kernel_size', type=int, default=5,
                        help=f"kernel size for enhanced lee filter (default: 5)")
    parser.add_argument('--sdir', dest='sample_dir', default='samples_200_6_4_10_sar/', help='(default: samples_200_6_4_10_sar/)')

    args = parser.parse_args()
    sys.exit(main(args.size, args.samples, args.seed, method=args.method, kernel_size=args.kernel_size, sample_dir=args.sample_dir))
