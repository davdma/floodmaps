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
from typing import List, Optional, Tuple

try:
    import yaml
except Exception:
    yaml = None

from utils.utils import enhanced_lee_filter, SRC_DIR, DATA_DIR, SAMPLES_DIR

def random_crop(events: List[Path], size, num_samples, rng, pre_sample_dir, cloud_threshold, filter='raw', typ="train"):
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
            tci_file = event / f'tci_{img_dt}_{eid}.tif'
            dem_file = event / f'dem_{eid}.tif'
            waterbody_file = event / f'waterbody_{eid}.tif'
            roads_file = event / f'roads_{eid}.tif'
            flowlines_file = event / f'flowlines_{eid}.tif'
            cloud_file = event / f'clouds_{img_dt}_{eid}.tif'
            nlcd_file = event / f'nlcd_{eid}.tif'

            with rasterio.open(tci_file) as src:
                tci_raster = src.read()

            # tci floats
            tci_floats = (tci_raster / 255).astype(np.float32)

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
            
            with rasterio.open(flowlines_file) as src:
                flowlines_raster = src.read()
            
            with rasterio.open(nlcd_file) as src:
                nlcd_raster = src.read()

            with rasterio.open(cloud_file) as src:
                cloud_raster = src.read()

            stacked_raster = np.vstack((vv_raster, vh_raster, dem_raster,
                                      slope_y_raster, slope_x_raster,
                                      waterbody_raster, roads_raster, flowlines_raster, 
                                      label_binary, tci_floats, nlcd_raster), dtype=np.float32)

            # missing mask and cloud mask
            missing_raster = np.any(tci_raster == 0, axis = 0).astype(np.uint8)
            missing_raster = np.expand_dims(missing_raster, axis = 0)
            missing_clouds_raster = np.vstack((missing_raster, cloud_raster), dtype=np.uint8)

            tiles.append((stacked_raster, missing_clouds_raster))
    logger.info('Tiles loaded.')

    # loop over all events to generate samples for each epoch
    logger.info('Sampling random patches...')
    total_patches = num_samples * len(tiles)
    dataset = np.empty((total_patches, 13, size, size), dtype=np.float32)

    # sample given number of patches per tile
    for i, tile in enumerate(tiles):
        tile_data, tile_mask = tile
        patches_sampled = 0
        _, HEIGHT, WIDTH = tile_data.shape

        while patches_sampled < num_samples:
            x = int(rng.uniform(0, HEIGHT - size))
            y = int(rng.uniform(0, WIDTH - size))

            patch = tile_data[:, x : x + size, y : y + size]
            # if tci contains missing values, toss out and resample
            missing_patch = tile_mask[0, x : x + size, y : y + size]
            if np.any(missing_patch == 1):
                continue

            # filter out high cloud percentage patches
            cloud_patch = tile_mask[1, x : x + size, y : y + size]
            if (cloud_patch.sum() / cloud_patch.size) >= cloud_threshold:
                continue

            # filter out missing vv or vh tiles
            if np.any(patch[0] == -9999) or np.any(patch[1] == -9999):
                continue

            dataset[i * num_samples + patches_sampled] = patch
            patches_sampled += 1

    output_file = pre_sample_dir / f'{typ}_patches.npy'
    np.save(output_file, dataset)

    logger.info('Sampling complete.')

def trainMean(train_events: List[Path], filter="raw"):
    """Calculate mean and std of non-binary channels, filtering out cloud pixels and missing sar pixels.

    Parameters
    ----------
    train_events : list[Path]
        List of training flood event folders where raw data tiles are stored.
    train_events : list[Path]
        Event directories across one or more datasets.
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
    means = np.zeros(5)

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
            dem_file = event / f'dem_{eid}.tif'
            cloud_file = event / f'clouds_{img_dt}_{eid}.tif'

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

            with rasterio.open(cloud_file) as src:
                cloud_raster = src.read().reshape((1, -1))

            mask = (vv_raster[0] != -9999) & (vh_raster[0] != -9999) & (cloud_raster[0] != 1)

            stack = np.vstack((vv_raster, vh_raster, dem_raster, slope_y_raster, slope_x_raster), dtype=np.float32)

            masked_stack = stack[:, mask]

            # calculate mean and var across channels
            channel_means = np.mean(masked_stack, axis=1)
            means += channel_means

            count += 1

    overall_channel_mean = means / count

    # calculate final statistics
    return overall_channel_mean

def trainStd(train_events: List[Path], train_means, filter="raw"):
    """Calculate mean and std of non binary channels, filtering out cloud pixels and missing sar pixels.

    Parameters
    ----------
    train_events : list[Path]
        List of training flood event folders where raw data tiles are stored.
    train_means : ndarray
        Channel means.
    train_events : list[Path]
        Event directories across one or more datasets.
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
    variances = np.zeros(5)

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
            dem_file = event / f'dem_{eid}.tif'
            cloud_file = event / f'clouds_{img_dt}_{eid}.tif'

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

            with rasterio.open(cloud_file) as src:
                cloud_raster = src.read().reshape((1, -1))

            mask = (vv_raster[0] != -9999) & (vh_raster[0] != -9999) & (cloud_raster[0] != 1)

            stack = np.vstack((vv_raster, vh_raster, dem_raster, slope_y_raster, slope_x_raster), dtype=np.float32)

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


def main(size, samples, seed, method='random', cloud_threshold=0.1, filter=None, sample_dir='samples_200_6_4_10_sar/', config: Optional[str] = None):
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

    # Resolve input directories and split ratios
    if config is not None:
        if yaml is None:
            raise ImportError("PyYAML is required to use --config. Please install with `pip install pyyaml`.")
        with open(config, 'r') as f:
            cfg = yaml.safe_load(f)
        cfg_s1 = cfg.get('s1', {})
        sample_dirs_list = cfg_s1.get('sample_dirs', [sample_dir])
        split_cfg = cfg_s1.get('split', {})
        split_seed = split_cfg.get('seed', seed)
        val_ratio = split_cfg.get('val_ratio', 0.1)
        test_ratio = split_cfg.get('test_ratio', 0.1)
    else:
        sample_dirs_list = [sample_dir]
        split_seed = seed
        val_ratio = 0.1
        test_ratio = 0.1

    # randomly select samples to be in train and test set from multiple directories
    # Deduplicate by eid, keeping the first directory where the event has required SAR assets
    selected_events: List[Path] = []
    seen_eids = set()
    p_eid = re.compile('\n?\d{8}_\d+_\d+$')
    for sd in sample_dirs_list:
        sample_path = SAMPLES_DIR / sd
        if not sample_path.is_dir():
            continue
        for event_dir in sample_path.glob('[0-9]*'):
            if not event_dir.is_dir():
                continue
            eid = event_dir.name
            if eid in seen_eids:
                continue
            # qualify: must have at least one prediction and at least one sar vv tile
            has_pred = any(event_dir.glob('pred_*.tif'))
            has_sar_vv = any(event_dir.glob('sar_*_vv.tif'))
            if has_pred and has_sar_vv:
                selected_events.append(event_dir)
                seen_eids.add(eid)

    all_events: List[Path] = selected_events

    if len(all_events) == 0:
        logger.info('No events found in provided sample directories. Exiting.')
        return 0

    # shuffle and split deterministically
    # First, hold out (val+test)
    holdout_ratio = val_ratio + test_ratio
    if holdout_ratio <= 0 or holdout_ratio >= 1:
        raise ValueError('Sum of val_ratio and test_ratio must be in (0, 1).')

    train_events, val_test_events = train_test_split(all_events, test_size=holdout_ratio, random_state=split_seed)
    # Split val and test according to proportions within the holdout
    test_prop_within_holdout = test_ratio / holdout_ratio
    val_events, test_events = train_test_split(val_test_events, test_size=test_prop_within_holdout, random_state=split_seed + 1222)

    # calculate mean and std of train tiles non-binary channels
    logger.info('Calculating mean and std of training tiles...')
    mean_cont = trainMean(train_events, filter=filter)
    std_cont = trainStd(train_events, mean_cont, filter=filter)
    logger.info('Mean and std of training tiles calculated.')

    # set mean and std of binary channels at the end to 0 and 1
    bchannels = 3 # waterbody, roads, flowlines
    mean_bin = np.zeros(bchannels)
    std_bin = np.ones(bchannels)
    mean = np.concatenate([mean_cont, mean_bin])
    std = np.concatenate([std_cont, std_bin])

    # also store training mean std statistics in file
    stats_file = pre_sample_dir / f'mean_std_{size}_{samples}_{filter}.pkl'
    with open(stats_file, 'wb') as f:
        pickle.dump((mean, std), f)
    logger.info('Training mean and std statistics saved.')

    if method == 'random':
        rng = Random(seed)
        random_crop(train_events, size, samples, rng, pre_sample_dir, cloud_threshold, filter=filter, typ="train")
        random_crop(val_events, size, samples, rng, pre_sample_dir, cloud_threshold, filter=filter, typ="val")
        random_crop(test_events, size, samples, rng, pre_sample_dir, cloud_threshold, filter=filter, typ="test")
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
    parser.add_argument('--config', dest='config', default=None, help='YAML config file path defining sample directories and split ratios')

    args = parser.parse_args()
    sys.exit(main(args.size, args.samples, args.seed, method=args.method, cloud_threshold=args.cloud_threshold, filter=args.filter, sample_dir=args.sample_dir, config=args.config))
