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

from utils.utils import SRC_DIR, DATA_DIR, SAMPLES_DIR
### ADD DEMS IN THE FUTURE - CURRENTLY NOT THE SAME SHAPE

def find_vv_vh_tifs(directory):
    """Returns the vv and vh tif filepaths for a given event directory.
    
    Parameters
    ----------
    directory : Path
        Directory containing the multitemporal SAR files.
        
    Returns
    -------
    tuple[Path, Path]
        Paths to the vv and vh tif files.
    """
    directory = Path(directory)

    vv_files = list(directory.glob('vv_*.tif'))
    vh_files = list(directory.glob('vh_*.tif'))

    if len(vv_files) != 1:
        raise FileNotFoundError(f"Expected exactly one 'vv_*.tif', found {len(vv_files)} in {directory}")
    if len(vh_files) != 1:
        raise FileNotFoundError(f"Expected exactly one 'vh_*.tif', found {len(vh_files)} in {directory}")

    return vv_files[0], vh_files[0]

def generate_patches(events, size, num_samples, rng, pre_sample_dir, sample_dir, typ="train"):
    """Uniformly samples sar patches of dimension size x size across each dataset tile. The
    patches are saved together into one file with enhanced lee filter.

    Note: for each multitemporal SAR tile (e.g. 10 slices), will compute the composite
    average of the 10 slices, and then for each sampled patch the following is saved:
    - vv (slice)
    - vh (slice)
    - vv_composite (average of all slices)
    - vh_composite (average of all slices)

    If there are 10 slices, then each sampled patch location results in 10 of these
    single-composite pairs.

    Parameters
    ----------
    events : list[Path]
        List of folders where multitemporal SAR tiles are stored.
    size : int
        Size of the sampled patches.
    num_samples : int
        Number of patches to sample per raw S1 tile (multiply this by the
        number of temporal slices for total patches in dataset from the tile).
    rng : obj
        Random number generator.
    pre_sample_dir : Path
        Directory to save the preprocessed patches.
    sample_dir : str
        Directory containing raw S1 tiles for patch sampling.
    typ : str
        Subset assigned to the saved patches: train, val, test.
    """
    logger = logging.getLogger('preprocessing')

    # first load all samples into memory
    # SAR Preprocessing: labels will be stored in event sample folder
    logger.info('Generating patches. Loading tiles into memory...')
    tiles = []
    p1 = re.compile('\d{8}_\d+_\d+')
    # dem_dir = SAMPLES_DIR / 'samples_200_6_4_10_sar'
    total_iterations = len(events)
    for i, event in enumerate(events):
        # Use Path.name to get just the directory name for regex matching
        m = p1.search(event.name)

        if not m:
            logger.info(f'No matching eid. Skipping {event.name}...')
            continue

        eid = m.group(0)

        # search for vv and vh multitemporal rasters
        vv_file, vh_file = find_vv_vh_tifs(event)

        # DEM added
        # dem_file = dem_dir / eid / f'dem_{eid}.tif'
        # with rasterio.open(dem_file) as src:
        #     dem_raster = src.read()

        with rasterio.open(vv_file) as src:
            vv_raster = src.read()

        with rasterio.open(vh_file) as src:
            vh_raster = src.read()

        # get composite rasters - make sure that missing values are not included in the mean
        # if any of the slices have a missing value, then the composite will have a missing value
        vv_mask = (vv_raster == -9999).any(axis=0)
        vh_mask = (vh_raster == -9999).any(axis=0)
        vv_composite = np.mean(vv_raster, axis=0)
        vh_composite = np.mean(vh_raster, axis=0)
        vv_composite[vv_mask] = -9999
        vh_composite[vh_mask] = -9999

        slices = vv_raster.shape[0]
        for slice in range(slices):
            try:
                stacked_raster = np.stack((vv_raster[slice], vh_raster[slice], vv_composite, vh_composite), dtype=np.float32)
            except ValueError as e:
                print(f"Error stacking raster for event {event}")
                # print(f"DEM shape: {dem_raster.shape}, SAR shape: {vv_raster[slice].shape}")
                raise e
            except Exception as e:
                raise e
            tiles.append(stacked_raster)

        logger.info(f"Loaded {slices} total single-composite tile pairings into memory: iteration {i}/{total_iterations} completed.")
    logger.info('All tiles loaded.')

    # loop over all events to generate samples for each epoch
    logger.info('Beginning patch sampling...')
    total_patches = num_samples * len(tiles)
    dataset = np.empty((total_patches, 4, size, size), dtype=np.float32)

    # sample given number of patches per tile
    for i, tile in enumerate(tiles):
        patches_sampled = 0
        _, HEIGHT, WIDTH = tile.shape

        while patches_sampled < num_samples:
            x = int(rng.uniform(0, HEIGHT - size))
            y = int(rng.uniform(0, WIDTH - size))

            patch = tile[:, x : x + size, y : y + size]

            # filter out missing vv or vh tiles
            if np.any(patch == -9999):
                continue

            dataset[i * num_samples + patches_sampled] = patch
            patches_sampled += 1

    output_file = pre_sample_dir / f'{typ}_patches.npy'
    np.save(output_file, dataset)

    logger.info('Sampling complete.')

def get_min_max(all_values):
    """Calculate min and max values from a list of arrays, filtering outliers.
    
    Parameters
    ----------
    all_values : list[ndarray]
        List of arrays containing values to calculate min/max from.
        
    Returns
    -------
    tuple[float, float]
        Min and max values within the 1st to 99th percentile range.
    """
    # concatenate all values
    all_values = np.concatenate(all_values)

    # Compute percentiles
    lower = np.percentile(all_values, 1)
    upper = np.percentile(all_values, 99)

    # Filter values within 1st to 99th percentile
    filtered = all_values[(all_values >= lower) & (all_values <= upper)]

    # Get min and max in that range
    min_val = filtered.min()
    max_val = filtered.max()
    return min_val, max_val

def trainMinMax(train_events):
    """Calculate min and max of composite vv and vh (useful for PSNR, SSIM),
    filtering out missing values.
    
    Parameters
    ----------
    train_events : list[Path]
        List of training flood event folders where raw data tiles are stored.
        
    Returns
    -------
    tuple[float, float, float, float]
        Min and max values for vv and vh channels.
    """
    logger = logging.getLogger('preprocessing')

    logger.info('Train min max calculation start.')
    min_val_vv = np.inf
    max_val_vv = -np.inf
    min_val_vh = np.inf
    max_val_vh = -np.inf

    # tmp
    vv_values = []
    vh_values = []
    
    total_iterations = len(train_events)
    p1 = re.compile('\d{8}_\d+_\d+')
    for i, event in enumerate(train_events):
        # Use Path.name to get just the directory name for regex matching
        m = p1.search(event.name)

        if not m:
            logger.info(f'No matching eid during mean std calculation. Skipping {event}...')
            continue

        # search for vv and vh multitemporal rasters
        vv_file, vh_file = find_vv_vh_tifs(event)

        with rasterio.open(vv_file) as src:
            vv_raster = src.read()

        with rasterio.open(vh_file) as src:
            vh_raster = src.read()

        vv_mask = (vv_raster == -9999).any(axis=0)
        vh_mask = (vh_raster == -9999).any(axis=0)
        vv_composite = np.mean(vv_raster, axis=0)
        vh_composite = np.mean(vh_raster, axis=0)
        vv_composite[vv_mask] = -9999
        vh_composite[vh_mask] = -9999

        # ignore missing values for min
        vv_composite = vv_composite[~vv_mask]
        vh_composite = vh_composite[~vh_mask]

        vv_values.append(vv_composite.flatten())
        vh_values.append(vh_composite.flatten())

        # min_val_vv = np.min([min_val_vv, vv_composite.min()])
        # max_val_vv = np.max([max_val_vv, vv_composite.max()])
        # min_val_vh = np.min([min_val_vh, vh_composite.min()])
        # max_val_vh = np.max([max_val_vh, vh_composite.max()])

        logger.info(f'Calculating train min max: iteration {i}/{total_iterations} completed.')

    min_val_vv, max_val_vv = get_min_max(vv_values)
    min_val_vh, max_val_vh = get_min_max(vh_values)

    logger.info('Train min max calculation complete.')
    return min_val_vv, max_val_vv, min_val_vh, max_val_vh

def trainMean(train_events):
    """Calculate mean and std of single slice sar statistics, filtering out cloud pixels and missing sar pixels.

    Parameters
    ----------
    train_events : list[Path]
        List of training flood event folders where raw data tiles are stored.

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
    means = np.zeros(2) # vv and vh means

    p1 = re.compile('\d{8}_\d+_\d+')
    total_iterations = len(train_events)
    for i, event in enumerate(train_events):
        # Use Path.name to get just the directory name for regex matching
        m = p1.search(event.name)

        if not m:
            logger.info(f'No matching eid during mean std calculation. Skipping {event}...')
            continue

        # search for vv and vh multitemporal rasters
        vv_file, vh_file = find_vv_vh_tifs(event)

        with rasterio.open(vv_file) as src:
            vv_raster = src.read()

        with rasterio.open(vh_file) as src:
            vh_raster = src.read()

        vv_raster = vv_raster.reshape((1, -1))
        vh_raster = vh_raster.reshape((1, -1))

        mask = (vv_raster[0] != -9999) & (vh_raster[0] != -9999)

        stack = np.vstack((vv_raster, vh_raster), dtype=np.float32)

        masked_stack = stack[:, mask]

        # calculate mean and var across channels
        channel_means = np.mean(masked_stack, axis=1)
        means += channel_means

        count += 1
        logger.info(f'Calculating train mean: iteration {i}/{total_iterations} completed.')

    overall_channel_mean = means / count

    # calculate final statistics
    logger.info(f'Overall channel mean: vv={overall_channel_mean[0]}, vh={overall_channel_mean[1]}')
    return overall_channel_mean

def trainStd(train_events, train_means):
    """Calculate mean and std of channels, filtering out cloud pixels and missing sar pixels.

    Parameters
    ----------
    train_events : list[Path]
        List of training flood event folders where raw data tiles are stored.
    train_means : ndarray
        Channel means.

    Returns
    -------
    overall_channel_std : ndarray
        Channel stds.
    """
    logger = logging.getLogger('preprocessing')

    logger.info('Train std calculation start.')
    count = 0
    variances = np.zeros(2) # vv and vh variances

    p1 = re.compile('\d{8}_\d+_\d+')
    total_iterations = len(train_events)
    for i, event in enumerate(train_events):
        # Use Path.name to get just the directory name for regex matching
        m = p1.search(event.name)

        if not m:
            logger.info(f'No matching eid during mean std calculation. Skipping {event}...')
            continue

        # search for vv and vh multitemporal rasters
        vv_file, vh_file = find_vv_vh_tifs(event)

        with rasterio.open(vv_file) as src:
            vv_raster = src.read()

        with rasterio.open(vh_file) as src:
            vh_raster = src.read()

        vv_raster = vv_raster.reshape((1, -1))
        vh_raster = vh_raster.reshape((1, -1))

        mask = (vv_raster[0] != -9999) & (vh_raster[0] != -9999)

        stack = np.vstack((vv_raster, vh_raster), dtype=np.float32)

        masked_stack = stack[:, mask]

        # Subtract the accurate means from the data
        deviations = masked_stack - train_means.reshape(-1, 1)
        squared_deviations = np.square(deviations)
        channel_variances = squared_deviations.mean(axis=1)
        variances += channel_variances

        count += 1
        logger.debug(f'Count {count}')
        logger.info(f'Calculating train std: iteration {i}/{total_iterations} completed.')

    overall_channel_variances = variances / count
    overall_channel_std = np.sqrt(overall_channel_variances)

    # calculate final statistics
    logger.info(f'Overall channel std: vv={overall_channel_std[0]}, vh={overall_channel_std[1]}')
    return overall_channel_std


def main(size, samples, seed, method='random', sample_dir='samples_multi_sar_70_10_7/'):
    """Preprocesses multitemporal SAR tiles into paired single vs composite patches for conditional generation.

    Parameters
    ----------
    size : int
        Size of the sampled patches.
    samples : int
        Number of patches to sample per raw S2 tile.
    seed : int
        Random number generator seed.
    sample_dir : str
        Directory containing multitemporal SAR tiles for patch sampling.
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
    pre_sample_dir = DATA_DIR / 'multi' / f'samples_{size}_{samples}/'
    pre_sample_dir.mkdir(parents=True, exist_ok=True)

    # randomly select samples to be in train and test set
    sample_path = SAMPLES_DIR / sample_dir
    all_events = list(sample_path.glob('[0-9]*'))
    train_events, val_test_events = train_test_split(all_events, test_size=0.2, random_state=seed - 20)
    val_events, test_events = train_test_split(val_test_events, test_size=0.5, random_state=seed + 1222)

    # TBD: calculate min and max of train tiles for later metrics (PSNR, SSIM)
    # min_val_vv, max_val_vv, min_val_vh, max_val_vh = trainMinMax(train_events)
    # stats_dir = DATA_DIR / 'ad/stats'
    # stats_dir.mkdir(parents=True, exist_ok=True)
    # with open(stats_dir / f'multi_{size}_{samples}_range_percentile.pkl', 'wb') as f:
    #     pickle.dump((min_val_vv, max_val_vv, min_val_vh, max_val_vh), f)

    # calculate mean and std of train tiles
    mean = trainMean(train_events)
    std = trainStd(train_events, mean)

    # also store training vv, vh mean std statistics in file
    stats_file = pre_sample_dir / f'mean_std_{size}_{samples}.pkl'
    with open(stats_file, 'wb') as f:
        pickle.dump((mean, std), f)
    logger.info('Training mean and std statistics saved.')
 
    if method == 'random':
        rng = Random(seed)
        generate_patches(train_events, size, samples, rng, sample_dir, typ="train")
        generate_patches(val_events, size, samples, rng, sample_dir, typ="val")
        generate_patches(test_events, size, samples, rng, sample_dir, typ="test")

    logger.debug('Preprocessing complete.')

if __name__ == '__main__':
    parser = argparse.ArgumentParser(prog='preprocess_ad',
        description='Preprocesses 4km x 4km multitemporal SAR tiles into paired single vs composite patches for conditional generation.')
    parser.add_argument('-x', '--size', dest='size', type=int, default=64, help='pixel width of patch (default: 64)')
    parser.add_argument('-n', '--samples', dest='samples', type=int, default=500, help='number of samples per image (default: 500)')
    parser.add_argument('-s', '--seed', dest='seed', type=int, default=433002, help='random number generator seed (default: 433002)')
    parser.add_argument('-m', '--method', dest='method', default='random', choices=['random'], help='sampling method (default: random)')
    parser.add_argument('--sdir', dest='sample_dir', default='samples_multi_sar_70_10_7/', help='data directory in the sampling folder (default: samples_multi_sar_70_10_7/)')

    args = parser.parse_args()
    sys.exit(main(args.size, args.samples, args.seed, method=args.method, sample_dir=args.sample_dir))
