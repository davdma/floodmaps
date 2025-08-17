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

from utils.utils import enhanced_lee_filter, SRC_DIR, DATA_DIR, SAMPLES_DIR

def random_crop(events, size, num_samples, rng, pre_sample_dir, sample_dir, cloud_threshold, filter='raw', typ="train"):
    """Uniformly samples patches of dimension size x size across each dataset tile. The
    patches are saved together with labels into one file for minibatching.

    Parameters
    ----------
    events : list[Path]
        List of flood event folders where raw data tiles are stored.
    size : int
        Size of the sampled patches.
    num_samples : int
        Number of patches to sample per raw S1 tile.
    rng : obj
        Random number generator.
    pre_sample_dir : Path
        Directory to save the sampled patches.
    sample_dir : str
        Directory containing raw S1 tiles for patch sampling.
    cloud_threshold : float
        Maximum patch cloud percentage. Patches above this threshold are filtered out.
    filter : str
        Specifying filter='raw' does not apply any filters to the patches. Using filter='lee' applies Enhanced Lee Filter.
        See https://desktop.arcgis.com/en/arcmap/latest/manage-data/raster-and-images/speckle-function.htm.
    typ : str
        Subset assigned to the saved patches: train, val, test.
    """
    logger = logging.getLogger('preprocessing')

    # first load all samples into memory
    # SAR Preprocessing: labels will be stored in event sample folder
    logger.info('Loading tiles into memory...')
    tiles = []
    p1 = re.compile('\d{8}_\d+_\d+')
    p2 = re.compile('pred_(\d{8})_.+.tif')
    for event in events:
        # Use Path.name to get just the directory name for regex matching
        m = p1.search(event.name)

        if m:
            eid = m.group(0)
        else:
            logger.info(f'No matching eid. Skipping {event}...')
            continue

        # search for label file + sar file using Path.glob()
        # look for labels w tci + sar pairings
        for label in event.glob('pred_*.tif'):
            m = p2.search(label.name)
            img_dt = m.group(1)

            sar_vv_files = list(event.glob(f'sar_{img_dt}_*_vv.tif'))
            if len(sar_vv_files) == 0:
                # SKIP IF SAR NOT FOUND FOR A LABEL (THIS IS SOMETIMES POSSIBLE)
                logger.info(f'SAR file not found for label {label}')
                continue

            sar_vv_file = sar_vv_files[0]
            sar_vh_file = sar_vv_files[0][:-6] + 'vh.tif'

            # get associated label
            with rasterio.open(label) as src:
                label_raster = src.read([1, 2, 3])
                # if label has any values != 0 or 255 then print to log!
                if np.any((label_raster > 0) & (label_raster < 255)):
                    logger.debug(f'{label.name} values are not 0 or 255.')

                label_binary = np.where(label_raster[0] != 0, 1, 0)
                label_binary = np.expand_dims(label_binary, axis = 0)

                HEIGHT = src.height
                WIDTH = src.width

            # skip missing data w tci and sar combined
            sample_path = SAMPLES_DIR / sample_dir
            tci_file = sample_path / eid / f'tci_{img_dt}_{eid}.tif'
            dem_file = sample_path / eid / f'dem_{eid}.tif'
            waterbody_file = sample_path / eid / f'waterbody_{eid}.tif'
            roads_file = sample_path / eid / f'roads_{eid}.tif'
            cloud_file = sample_path / eid / f'clouds_{img_dt}_{eid}.tif'

            with rasterio.open(tci_file) as src:
                tci_raster = src.read()

            # tci floats
            tci_floats = (tci_raster / 255).astype(np.float32)

            # missing mask
            missing_raster = np.any(tci_raster == 0, axis = 0).astype(np.float32)
            missing_raster = np.expand_dims(missing_raster, axis = 0)

            with rasterio.open(sar_vv_file) as src:
                vv_raster = src.read()

            with rasterio.open(sar_vh_file) as src:
                vh_raster = src.read()

            # apply speckle filter to sar:
            if filter == "lee":
                vv_raster = np.expand_dims(enhanced_lee_filter(vv_raster[0], kernel_size=5).astype(np.float32), axis=0)
                vh_raster = np.expand_dims(enhanced_lee_filter(vh_raster[0], kernel_size=5).astype(np.float32), axis=0)

            with rasterio.open(dem_file) as src:
                dem_raster = src.read()

            slope = np.gradient(dem_raster, axis=(1,2))
            slope_y_raster, slope_x_raster = slope

            with rasterio.open(waterbody_file) as src:
                waterbody_raster = src.read()

            with rasterio.open(roads_file) as src:
                roads_raster = src.read()

            with rasterio.open(cloud_file) as src:
                cloud_raster = src.read()

            stacked_raster = np.vstack((vv_raster, vh_raster, dem_raster,
                                      slope_y_raster, slope_x_raster,
                                      waterbody_raster, roads_raster, tci_floats, label_binary,
                                      missing_raster, cloud_raster), dtype=np.float32)

            tiles.append(stacked_raster)
    logger.info('Tiles loaded.')

    # loop over all events to generate samples for each epoch
    logger.info('Sampling random patches...')
    # want to store 7 channels + label + tci
    total_patches = num_samples * len(tiles)
    dataset = np.empty((total_patches, 11, size, size), dtype=np.float32)

    # sample given number of patches per tile
    for i, tile in enumerate(tiles):
        patches_sampled = 0
        _, HEIGHT, WIDTH = tile.shape

        while patches_sampled < num_samples:
            x = int(rng.uniform(0, HEIGHT - size))
            y = int(rng.uniform(0, WIDTH - size))

            patch = tile[:, x : x + size, y : y + size]
            # if contains missing values, toss out and resample
            if np.any(patch[11] == 1):
                continue

            # filter out high cloud percentage patches
            if (patch[12].sum() / patch[12].size) >= cloud_threshold:
                continue

            # filter out missing vv or vh tiles
            if np.any(patch[0] == -9999) or np.any(patch[1] == -9999):
                continue

            dataset[i * num_samples + patches_sampled] = patch[:11]
            patches_sampled += 1

    output_file = pre_sample_dir / f'{typ}_patches.npy'
    np.save(output_file, dataset)

    logger.info('Sampling complete.')

def trainMean(train_events, sample_dir, filter="raw"):
    """Calculate mean and std of channels, filtering out cloud pixels and missing sar pixels.

    Parameters
    ----------
    train_events : list[Path]
        List of training flood event folders where raw data tiles are stored.
    sample_dir : str
        Directory containing raw S1 tiles for patch sampling.
    filter : str
        Specifying filter='raw' does not apply any filters to the patches. Using filter='lee' applies Enhanced Lee Filter.
        See https://desktop.arcgis.com/en/arcmap/latest/manage-data/raster-and-images/speckle-function.htm.

    Returns
    -------
    overall_channel_mean : ndarray
        Channel means.
    """
    logger = logging.getLogger('preprocessing')

    # calculate mean each event tile
    # since we have n > 500 this is good approximation of pop mean
    count = 0
    means = np.zeros(7)

    p1 = re.compile('\d{8}_\d+_\d+')
    p2 = re.compile('pred_(\d{8})_.+.tif')
    for event in train_events:
        m = p1.search(event.name)

        if m:
            eid = m.group(0)
        else:
            logger.info(f'No matching eid during mean std calculation. Skipping {event}...')
            continue

        # search for label file + sar file using Path.glob()
        # look for labels w tci + sar pairings
        for label in event.glob('pred_*.tif'):
            m = p2.search(label.name)
            img_dt = m.group(1)

            sar_vv_files = list(event.glob(f'sar_{img_dt}_*_vv.tif'))
            if len(sar_vv_files) == 0:
                # SKIP IF SAR NOT FOUND FOR A LABEL (THIS IS SOMETIMES POSSIBLE)
                logger.info(f'Mean std: SAR file not found for label {label}')
                continue

            sar_vv_file = sar_vv_files[0]
            sar_vh_file = sar_vv_files[0][:-6] + 'vh.tif'

            # skip missing data w tci and sar combined
            sample_path = SAMPLES_DIR / sample_dir
            dem_file = sample_path / eid / f'dem_{eid}.tif'
            waterbody_file = sample_path / eid / f'waterbody_{eid}.tif'
            roads_file = sample_path / eid / f'roads_{eid}.tif'
            cloud_file = sample_path / eid / f'clouds_{img_dt}_{eid}.tif'

            with rasterio.open(sar_vv_file) as src:
                vv_raster = src.read()

            with rasterio.open(sar_vh_file) as src:
                vh_raster = src.read()

            # apply speckle filter to sar:
            if filter == "lee":
                vv_raster = np.expand_dims(enhanced_lee_filter(vv_raster[0], kernel_size=5).astype(np.float32), axis=0)
                vh_raster = np.expand_dims(enhanced_lee_filter(vh_raster[0], kernel_size=5).astype(np.float32), axis=0)

            vv_raster = vv_raster.reshape((1, -1))
            vh_raster = vh_raster.reshape((1, -1))

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

            with rasterio.open(cloud_file) as src:
                cloud_raster = src.read().reshape((1, -1))

            mask = (vv_raster[0] != -9999) & (vh_raster[0] != -9999) & (cloud_raster[0] != 1)

            stack = np.vstack((vv_raster, vh_raster, dem_raster, slope_y_raster,
                               slope_x_raster, waterbody_raster, roads_raster), dtype=np.float32)

            masked_stack = stack[:, mask]

            # calculate mean and var across channels
            channel_means = np.mean(masked_stack, axis=1)
            means += channel_means

            count += 1

    overall_channel_mean = means / count

    # calculate final statistics
    return overall_channel_mean

def trainStd(train_events, train_means, sample_dir, filter="raw"):
    """Calculate mean and std of channels, filtering out cloud pixels and missing sar pixels.

    Parameters
    ----------
    train_events : list[Path]
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

    count = 0
    variances = np.zeros(7)

    p1 = re.compile('\d{8}_\d+_\d+')
    p2 = re.compile('pred_(\d{8})_.+.tif')
    for event in train_events:
        m = p1.search(event.name)

        if m:
            eid = m.group(0)
        else:
            logger.info(f'No matching eid during mean std calculation. Skipping {event}...')
            continue

        # search for label file + sar file using Path.glob()
        # look for labels w tci + sar pairings
        for label in event.glob('pred_*.tif'):
            m = p2.search(label.name)
            img_dt = m.group(1)

            sar_vv_files = list(event.glob(f'sar_{img_dt}_*_vv.tif'))
            if len(sar_vv_files) == 0:
                # SKIP IF SAR NOT FOUND FOR A LABEL (THIS IS SOMETIMES POSSIBLE)
                logger.info(f'Mean std: SAR file not found for label {label}')
                continue

            sar_vv_file = sar_vv_files[0]
            sar_vh_file = sar_vv_files[0][:-6] + 'vh.tif'

            # skip missing data w tci and sar combined
            sample_path = SAMPLES_DIR / sample_dir
            dem_file = sample_path / eid / f'dem_{eid}.tif'
            waterbody_file = sample_path / eid / f'waterbody_{eid}.tif'
            roads_file = sample_path / eid / f'roads_{eid}.tif'
            cloud_file = sample_path / eid / f'clouds_{img_dt}_{eid}.tif'

            with rasterio.open(sar_vv_file) as src:
                vv_raster = src.read()

            with rasterio.open(sar_vh_file) as src:
                vh_raster = src.read()

            # apply speckle filter to sar:
            if filter == "lee":
                vv_raster = np.expand_dims(enhanced_lee_filter(vv_raster[0], kernel_size=5).astype(np.float32), axis=0)
                vh_raster = np.expand_dims(enhanced_lee_filter(vh_raster[0], kernel_size=5).astype(np.float32), axis=0)

            vv_raster = vv_raster.reshape((1, -1))
            vh_raster = vh_raster.reshape((1, -1))

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

            with rasterio.open(cloud_file) as src:
                cloud_raster = src.read().reshape((1, -1))

            mask = (vv_raster[0] != -9999) & (vh_raster[0] != -9999) & (cloud_raster[0] != 1)

            stack = np.vstack((vv_raster, vh_raster, dem_raster, slope_y_raster,
                               slope_x_raster, waterbody_raster, roads_raster), dtype=np.float32)

            masked_stack = stack[:, mask]

            # Subtract the accurate means from the data
            deviations = masked_stack - np.array(train_means).reshape(-1, 1)
            squared_deviations = deviations ** 2
            channel_variances = squared_deviations.mean(axis=1)
            variances += channel_variances

            count += 1
            logger.debug(f'Count {count}')

    overall_channel_variances = variances / count
    overall_channel_std = np.sqrt(overall_channel_variances)

    # calculate final statistics
    return overall_channel_std


def main(size, samples, seed, method='random', cloud_threshold=0.1, filter=None, sample_dir='samples_200_6_4_10_sar/'):
    """Preprocesses raw S1 tiles and corresponding labels into smaller patches.

    Parameters
    ----------
    size : int
        Size of the sampled patches.
    samples : int
        Number of patches to sample per raw S2 tile.
    seed : int
    method : str
        Sampling method.
    filter : str
        Filter to apply to patches: 'raw' (no filter) or 'lee'.
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
    pre_sample_dir = DATA_DIR / 'sar' / f'samples_{size}_{samples}_{filter}'
    pre_sample_dir.mkdir(parents=True, exist_ok=True)

    # randomly select samples to be in train and test set
    sample_path = SAMPLES_DIR / sample_dir
    all_events = list(sample_path.glob('[0-9]*'))
    train_events, val_test_events = train_test_split(all_events, test_size=0.2, random_state=seed - 20)
    val_events, test_events = train_test_split(val_test_events, test_size=0.5, random_state=seed + 1222)

    # calculate mean and std of train tiles
    logger.info('Calculating mean and std of training tiles...')
    mean = trainMean(train_events, sample_dir, filter=filter)
    std = trainStd(train_events, mean, sample_dir, filter=filter)
    logger.info('Mean and std of training tiles calculated.')

    # set mean and std of binary channels at the end to 0 and 1
    bchannels = 3 # if flowlines included
    mean[-bchannels:] = 0
    std[-bchannels:] = 1

    # also store training mean std statistics in file
    stats_file = pre_sample_dir / f'mean_std_{size}_{samples}_{filter}.pkl'
    with open(stats_file, 'wb') as f:
        pickle.dump((mean, std), f)
    logger.info('Training mean and std statistics saved.')

    if method == 'random':
        rng = Random(seed)
        random_crop(train_events, size, samples, rng, pre_sample_dir, sample_dir, cloud_threshold, filter=filter, typ="train")
        random_crop(val_events, size, samples, rng, pre_sample_dir, sample_dir, cloud_threshold, filter=filter, typ="val")
        random_crop(test_events, size, samples, rng, pre_sample_dir, sample_dir, cloud_threshold, filter=filter, typ="test")
        logger.info('Random samples generated.')

    logger.debug('Preprocessing complete.')

if __name__ == '__main__':
    parser = argparse.ArgumentParser(prog='preprocess_s1', description='Preprocesses 4km x 4km machine labeled SAR tiles into smaller patches via random cropping.')
    parser.add_argument('-x', '--size', dest='size', type=int, default=68, help='pixel width of patch (default: 68)')
    parser.add_argument('-n', '--samples', dest='samples', type=int, default=1000, help='number of samples per image (default: 500)')
    parser.add_argument('-s', '--seed', dest='seed', type=int, default=433002, help='random number generator seed (default: 433002)')
    parser.add_argument('-m', '--method', dest='method', default='random', choices=['random'], help='sampling method (default: random)')
    parser.add_argument('-c', '--cloud_threshold', type=float, default=0.1, help='cloud percentage threshold for patch sampling (default: 0.1)')
    parser.add_argument('--filter', default='raw', choices=['lee', 'raw'],
                        help=f"filters: enhanced lee, raw (default: raw)")
    parser.add_argument('--sdir', dest='sample_dir', default='samples_200_6_4_10_sar/', help='data directory in the sampling folder (default: samples_200_6_4_10_sar/)')

    args = parser.parse_args()
    sys.exit(main(args.size, args.samples, args.seed, method=args.method, cloud_threshold=args.cloud_threshold, filter=args.filter, sample_dir=args.sample_dir))
