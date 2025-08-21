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
from typing import List, Optional, Tuple

try:
    import yaml
except Exception:
    yaml = None

from utils.utils import TRAIN_LABELS, VAL_LABELS, TEST_LABELS, SRC_DIR, DATA_DIR, SAMPLES_DIR
### TO IMPLEMENT: SAMPLING FROM PREDICTED LABEL TILES ALSO INSTEAD OF HUMAN LABELS

def _find_event_dir(img_dt: str, eid: str, sample_dirs: List[str]) -> Optional[Path]:
    """Find the first dataset directory under sampling/ that contains the
    eid directory.

    Returns the event directory Path or None if not found.
    """
    for sd in sample_dirs:
        event_dir = SAMPLES_DIR / sd / eid
        if not event_dir.is_dir():
            continue
        return event_dir
    return None

def random_crop(label_paths, size, num_samples, rng, pre_sample_dir, sample_dirs, typ="train"):
    """Uniformly samples patches of dimension size x size across each dataset tile, and saves
    all patches in train, val, test sets into respective npy files.

    Parameters
    ----------
    label_paths : list[str]
        List of label file paths relative to sampling/ to manually ground-truthed tile labels.
    size : int
        Size of the sampled patches.
    num_samples : int
        Number of patches to sample per raw S2 tile.
    rng : obj
        Random number generator.
    pre_sample_dir : Path
        Directory to save the sampled patches.
    sample_dirs : list[str]
        One or more dataset directories under sampling/ that contain the S2 tiles.
    typ : str
        Subset assigned to the saved patches: train, val, test.
    """
    logger = logging.getLogger('preprocessing')
    
    # 16 channels for 11 data + 1 label channel + 3 TCI + 1 NLCD
    total_patches = num_samples * len(label_paths)
    dataset = np.empty((total_patches, 16, size, size), dtype=np.float32)

    tiles = []
    for label_rel in label_paths:
        p = re.compile('label_(\d{8})_(.+).tif')
        m = p.search(label_rel)

        if m:
            tile_date = m.group(1)
            eid = m.group(2)
        else:
            raise ValueError(f'Label file {label_rel} does not match expected format.')

        with rasterio.open(SAMPLES_DIR / label_rel) as src:
            label_raster = src.read([1, 2, 3])
            # if label has any values != 0 or 255 then print to log!
            if np.any((label_raster > 0) & (label_raster < 255)):
                logger.debug(f'{label_rel} values are not 0 or 255.')
                
            label_binary = np.where(label_raster[0] != 0, 1, 0)
            label_binary = np.expand_dims(label_binary, axis = 0)

            HEIGHT = src.height
            WIDTH = src.width

        event_dir = _find_event_dir(tile_date, eid, sample_dirs)
        if event_dir is None:
            logger.info(f'No matching event assets found for label {label_rel}; skipping.')
            continue

        tci_file = event_dir / f'tci_{tile_date}_{eid}.tif'
        rgb_file = event_dir / f'rgb_{tile_date}_{eid}.tif'
        b08_file = event_dir / f'b08_{tile_date}_{eid}.tif'
        ndwi_file = event_dir / f'ndwi_{tile_date}_{eid}.tif'
        dem_file = event_dir / f'dem_{eid}.tif'
        waterbody_file = event_dir / f'waterbody_{eid}.tif'
        roads_file = event_dir / f'roads_{eid}.tif'
        flowlines_file = event_dir / f'flowlines_{eid}.tif'
        nlcd_file = event_dir / f'nlcd_{eid}.tif'

        with rasterio.open(tci_file) as src:
            tci_raster = src.read()
            tci_floats = (tci_raster / 255).astype(np.float32)

        with rasterio.open(rgb_file) as src:
            rgb_raster = src.read()

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

        with rasterio.open(flowlines_file) as src:
            flowlines_raster = src.read()

        with rasterio.open(nlcd_file) as src:
            nlcd_raster = src.read()

        # stack all tiles
        try:
            stacked_tile = np.vstack((rgb_raster, b08_raster, ndwi_raster, dem_raster, 
                                        slope_y_raster, slope_x_raster, waterbody_raster, 
                                        roads_raster, flowlines_raster, label_binary, tci_floats, nlcd_raster), dtype=np.float32)
        except ValueError as e:
            # print all the shapes for debugging
            print(f'Shapes do not match!')
            print(f'label file: {label_rel}')
            print(f'rgb_raster shape: {rgb_raster.shape}')
            print(f'b08_raster shape: {b08_raster.shape}')
            print(f'ndwi_raster shape: {ndwi_raster.shape}')
            print(f'dem_raster shape: {dem_raster.shape}')
            print(f'slope_y_raster shape: {slope_y_raster.shape}')
            print(f'slope_x_raster shape: {slope_x_raster.shape}')
            print(f'waterbody_raster shape: {waterbody_raster.shape}')
            print(f'label_binary shape: {label_binary.shape}')
            print(f'tci_floats shape: {tci_floats.shape}')
            print(f'nlcd_raster shape: {nlcd_raster.shape}')
            raise e

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

            dataset[i * num_samples + patches_sampled] = patch[:16]
            patches_sampled += 1

    output_file = pre_sample_dir / f'{typ}_patches.npy'
    np.save(output_file, dataset)

def loadMaskedStack(img_dt, eid, sample_dirs: List[str]):
    """Load and mask the stack of non-binary channels.

    Parameters
    ----------
    img_dt : str
        Image date.
    eid : str
        Event id.
    sample_dirs : list[str]
        Dataset directories containing S2 tiles for patch sampling.

    Returns
    -------
    masked_stack : ndarray
        Stack of channels with missing values masked out.
    """
    event_dir = _find_event_dir(img_dt, eid, sample_dirs)
    if event_dir is None:
        raise FileNotFoundError(f"Could not find assets for event {eid} across provided sample_dirs: {sample_dirs}")
    rgb_file = event_dir / f'rgb_{img_dt}_{eid}.tif'
    b08_file = event_dir / f'b08_{img_dt}_{eid}.tif'
    ndwi_file = event_dir / f'ndwi_{img_dt}_{eid}.tif'
    dem_file = event_dir / f'dem_{eid}.tif'

    with rasterio.open(rgb_file) as src:
        rgb_raster = src.read().reshape((3, -1))
    
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

    mask = (rgb_raster[0] != 0) & (ndwi_raster[0] != -999999)

    stack = np.vstack((rgb_raster, b08_raster, ndwi_raster, dem_raster, slope_y_raster, slope_x_raster), dtype=np.float32)

    masked_stack = stack[:, mask]

    return masked_stack

def trainMean(train_events, sample_dirs: List[str]):
    """Calculate mean and std of non-binary channels.

    Parameters
    ----------
    train_events : list[tuple[str, str]]
        List of training flood event folders where raw data tiles are stored.
        First element is the image date, second element is the event id.
    sample_dirs : list[str]
        Dataset directories containing S2 tiles for patch sampling.

    Returns
    -------
    overall_channel_mean : ndarray
        Channel means.
    """
    logger = logging.getLogger('preprocessing')
    count = 0
    total_sum = np.zeros(8, dtype=np.float64) # 8 non-binary channels for original s2 data

    for img_dt, eid in train_events:
        masked_stack = loadMaskedStack(img_dt, eid, sample_dirs)

        # calculate mean and var across channels
        channel_sums = np.sum(masked_stack, axis=1)
        total_sum += channel_sums

        count += masked_stack.shape[1]

    overall_channel_mean = total_sum / count

    # calculate final statistics
    return overall_channel_mean

def trainStd(train_events, train_means, sample_dirs: List[str]):
    """Calculate std of non-binary channels.

    Parameters
    ----------
    train_events : list[tuple[str, str]]
        List of training flood event folders where raw data tiles are stored.
        First element is the image date, second element is the event id.
    train_means : ndarray
        Channel means.
    sample_dirs : list[str]
        Dataset directories containing S2 tiles for patch sampling.

    Returns
    -------
    overall_channel_std : ndarray
        Channel stds.
    """
    logger = logging.getLogger('preprocessing')
    count = 0
    variances = np.zeros(8, dtype=np.float64) # 8 non-binary channels for original s2 data

    for img_dt, eid in train_events:
        masked_stack = loadMaskedStack(img_dt, eid, sample_dirs)

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

def main(size, samples, seed, method='random', sample_dir='samples_200_5_4_35/', label_dir='labels/', config: Optional[str] = None):
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
        Directory containing raw S2 tiles for patch sampling. Ignored if config is provided.
    label_dir : str
        Directory containing raw S2 tile labels for patch sampling. Ignored if config is provided.
    config : str, optional
        Path to YAML config file. If provided, overrides sample_dir and label_dir and hardcoded splits.
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

    # Resolve splits and sample dirs
    if config is not None:
        if yaml is None:
            raise ImportError("PyYAML is required to use --config. Please install with `pip install pyyaml`.")
        with open(config, 'r') as f:
            cfg = yaml.safe_load(f)
        cfg_s2 = cfg.get('s2', {})
        label_splits = cfg_s2.get('label_splits', {})
        train_labels = label_splits.get('train', [])
        val_labels = label_splits.get('val', [])
        test_labels = label_splits.get('test', [])
        sample_dirs_list = cfg_s2.get('sample_dirs', [sample_dir])
    else:
        # Backward-compatible: use hardcoded labels + single sample_dir/label_dir
        train_labels = [str(Path(label_dir) / lf) for lf in TRAIN_LABELS]
        val_labels = [str(Path(label_dir) / lf) for lf in VAL_LABELS]
        test_labels = [str(Path(label_dir) / lf) for lf in TEST_LABELS]
        sample_dirs_list = [sample_dir]

    # get event directories from the training labels for mean and std calculation
    p = re.compile('label_(\d{8})_(.+).tif')
    train_events = [(p.search(label).group(1), p.search(label).group(2)) for label in train_labels]

    # calculate mean and std of train tiles
    logger.info('Calculating mean and std of training tiles...')
    mean_cont = trainMean(train_events, sample_dirs_list)
    std_cont = trainStd(train_events, mean_cont, sample_dirs_list)
    logger.info('Mean and std of training tiles calculated.')

    # set mean and std of binary channels at the end to 0 and 1
    bchannels = 3 # waterbody, roads, flowlines
    mean_bin = np.zeros(bchannels)
    std_bin = np.ones(bchannels)
    mean = np.concatenate([mean_cont, mean_bin])
    std = np.concatenate([std_cont, std_bin])

    # also store training mean std statistics in file
    stats_file = pre_sample_dir / f'mean_std_{size}_{samples}.pkl'
    with open(stats_file, 'wb') as f:
        pickle.dump((mean, std), f)
    logger.info('Training mean and std statistics saved.')
    
    if method == 'random':
        rng = Random(seed)
        random_crop(train_labels, size, samples, rng, pre_sample_dir, sample_dirs_list, typ="train")
        random_crop(val_labels, size, samples, rng, pre_sample_dir, sample_dirs_list, typ="val")
        random_crop(test_labels, size, samples, rng, pre_sample_dir, sample_dirs_list, typ="test")
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
    parser.add_argument('--config', dest='config', default=None, help='YAML config file path defining splits and directories')
    
    args = parser.parse_args()
    sys.exit(main(args.size, args.samples, args.seed, method=args.method, sample_dir=args.sample_dir, label_dir=args.label_dir, config=args.config))
