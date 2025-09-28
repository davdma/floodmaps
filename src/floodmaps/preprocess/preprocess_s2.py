import rasterio
import numpy as np
from pathlib import Path
import re
import sys
from random import Random
import logging
import pickle
from typing import List, Optional, Tuple
import yaml
import hydra
from omegaconf import DictConfig
from datetime import datetime

def _find_event_dir(img_dt: str, eid: str, sample_dirs: List[str], cfg: DictConfig) -> Optional[Path]:
    """Find the first dataset directory under the imagery_dir that contains the
    eid directory.

    Returns the event directory Path or None if not found.
    """
    for sd in sample_dirs:
        event_dir = Path(cfg.paths.imagery_dir) / sd / eid
        if not event_dir.is_dir():
            continue
        return event_dir
    return None

def random_crop(label_paths, size, num_samples, rng, pre_sample_dir, sample_dirs, cfg, typ="train"):
    """Uniformly samples patches of dimension size x size across each dataset tile, and saves
    all patches in train, val, test sets into respective npy files.

    Parameters
    ----------
    label_paths : list[str]
        List of label file paths relative to labels_dir.
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
    cfg : DictConfig
        Configuration object.
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

        with rasterio.open(Path(cfg.paths.labels_dir) / label_rel) as src:
            label_raster = src.read([1, 2, 3])
            # if label has any values != 0 or 255 then print to log!
            if np.any((label_raster > 0) & (label_raster < 255)):
                logger.debug(f'{label_rel} values are not 0 or 255.')
                
            label_binary = np.where(label_raster[0] != 0, 1, 0)
            label_binary = np.expand_dims(label_binary, axis = 0)

            HEIGHT = src.height
            WIDTH = src.width

        event_dir = _find_event_dir(tile_date, eid, sample_dirs, cfg)
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

def loadMaskedStack(img_dt, eid, sample_dirs: List[str], cfg: DictConfig):
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
    event_dir = _find_event_dir(img_dt, eid, sample_dirs, cfg)
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

def trainMean(train_events, sample_dirs: List[str], cfg: DictConfig):
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
        masked_stack = loadMaskedStack(img_dt, eid, sample_dirs, cfg)

        # calculate mean and var across channels
        channel_sums = np.sum(masked_stack, axis=1)
        total_sum += channel_sums

        count += masked_stack.shape[1]

    overall_channel_mean = total_sum / count

    # calculate final statistics
    return overall_channel_mean

def trainStd(train_events, train_means, sample_dirs: List[str], cfg: DictConfig):
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
        masked_stack = loadMaskedStack(img_dt, eid, sample_dirs, cfg)

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


def extract_events_from_labels(label_paths: List[str]) -> List[str]:
    """Extract event IDs from label file paths.
    
    Parameters
    ----------
    label_paths : List[str]
        List of label file paths like 'labels/label_20200318_20200318_12_34.tif'
        
    Returns
    -------
    List[str]
        List of event IDs like '20200318_12_34'
    """
    p = re.compile(r'label_(\d{8})_(.+)\.tif')
    events = set()
    
    for label_path in label_paths:
        # Extract filename from path
        filename = Path(label_path).name
        m = p.search(filename)
        if m:
            img_dt, eid = m.group(1), m.group(2)
            events.add(eid)
    
    return list(sorted(events))


def save_event_splits(train_labels: List[str], val_labels: List[str], test_labels: List[str],
                      output_dir: Path, seed: int, timestamp: str, data_type: str = "s2_manual") -> None:
    """Save event splits to a YAML file for reproducibility and reference.
    
    Parameters
    ----------
    train_labels : List[str]
        List of training label file paths
    val_labels : List[str]  
        List of validation label file paths
    test_labels : List[str]
        List of test label file paths
    output_dir : Path
        Directory to save the splits file
    seed : int
        Random seed used (though splits are predefined, not random)
    timestamp : str
        Timestamp when preprocessing started
    data_type : str
        Type of data being processed (e.g., "s2_manual")
    """
    logger = logging.getLogger('preprocessing')
    
    # Extract event IDs from label paths
    train_events = extract_events_from_labels(train_labels)
    val_events = extract_events_from_labels(val_labels)
    test_events = extract_events_from_labels(test_labels)
    
    # Create splits file path
    splits_file = output_dir / f'event_splits_{data_type}_{seed}.yaml'
    
    # Prepare splits data structure
    event_splits = {
        'train': train_events,
        'val': val_events, 
        'test': test_events,
        'metadata': {
            'data_type': data_type,
            'seed': seed,
            'total_events': len(train_events) + len(val_events) + len(test_events),
            'train_count': len(train_events),
            'val_count': len(val_events),
            'test_count': len(test_events),
            'timestamp': timestamp
        },
        'label_files': {
            'train': sorted([Path(label).name for label in train_labels]),
            'val': sorted([Path(label).name for label in val_labels]),
            'test': sorted([Path(label).name for label in test_labels])
        }
    }
    
    # Save to YAML file if yaml is available
    try:
        if yaml is not None:
            with open(splits_file, 'w') as f:
                yaml.dump(event_splits, f, default_flow_style=False, sort_keys=False)
            logger.info(f'Event splits saved to {splits_file}')
        else:
            logger.warning('PyYAML not available, skipping event splits save')
    except Exception as e:
        logger.error(f'Failed to save event splits: {e}')
        raise

@hydra.main(version_base=None, config_path='pkg://configs', config_name='config.yaml')
def main(cfg: DictConfig) -> None:
    """Preprocesses raw S2 tiles and corresponding labels into smaller patches. The data will be stored
    as separate npy files for train, val, and test sets, along with a mean_std.pkl file containing the
    mean and std of the training tiles."""
    logger = logging.getLogger('preprocessing')
    logger.setLevel(logging.DEBUG)
    formatter = logging.Formatter('%(asctime)s - %(name)s - %(levelname)s - %(message)s')
    handler = logging.StreamHandler(sys.stdout)
    handler.setFormatter(formatter)
    handler.setLevel(logging.DEBUG)
    logger.addHandler(handler)
    logger.propagate = False

    # Create timestamp for logging
    timestamp = datetime.now().strftime('%Y-%m-%d %H:%M:%S')
    logger.info(f'''Starting S2 weak labeling preprocessing:
        Date:            {timestamp}
        Patch size:      {cfg.preprocess.size}
        Samples per tile: {cfg.preprocess.samples}
        Sampling method: {cfg.preprocess.method}
        Random seed:     {cfg.preprocess.seed}
        Sample dir(s):   {cfg.preprocess.s2.sample_dirs}
        Suffix:          {getattr(cfg.preprocess, 'suffix', None)}
    ''')

    # make our preprocess directory
    if getattr(cfg.preprocess, 'suffix', None):
        pre_sample_dir = Path(cfg.paths.preprocess_dir) / 's2' / f'samples_{cfg.preprocess.size}_{cfg.preprocess.samples}_{cfg.preprocess.suffix}'
    else:
        pre_sample_dir = Path(cfg.paths.preprocess_dir) / 's2' / f'samples_{cfg.preprocess.size}_{cfg.preprocess.samples}'
    pre_sample_dir.mkdir(parents=True, exist_ok=True)
    logger.info(f'Preprocess directory: {pre_sample_dir.name}')

    # Resolve splits and sample dirs
    cfg_s2 = cfg.preprocess.get('s2', {})
    label_splits = cfg_s2.get('label_splits', {})
    train_labels = label_splits.get('train', [])
    val_labels = label_splits.get('val', [])
    test_labels = label_splits.get('test', [])
    sample_dirs_list = cfg_s2.get('sample_dirs', [])

    # must be non empty
    if len(train_labels) == 0 or len(val_labels) == 0 or len(test_labels) == 0 or len(sample_dirs_list) == 0:
        raise ValueError('Train, val, test labels, and sample directories must be non empty.')

    # Save the event splits for reproducibility and reference
    save_event_splits(train_labels, val_labels, test_labels, pre_sample_dir, cfg.preprocess.seed, timestamp, "s2_manual")

    # get event directories from the training labels for mean and std calculation
    p = re.compile('label_(\d{8})_(.+).tif')
    train_events = [(p.search(label).group(1), p.search(label).group(2)) for label in train_labels]

    # calculate mean and std of train tiles
    logger.info('Calculating mean and std of training tiles...')
    mean_cont = trainMean(train_events, sample_dirs_list, cfg)
    std_cont = trainStd(train_events, mean_cont, sample_dirs_list, cfg)
    logger.info('Mean and std of training tiles calculated.')

    # set mean and std of binary channels at the end to 0 and 1
    bchannels = 3 # waterbody, roads, flowlines
    mean_bin = np.zeros(bchannels)
    std_bin = np.ones(bchannels)
    mean = np.concatenate([mean_cont, mean_bin])
    std = np.concatenate([std_cont, std_bin])

    # also store training mean std statistics in file
    stats_file = pre_sample_dir / f'mean_std_{cfg.preprocess.size}_{cfg.preprocess.samples}.pkl'
    with open(stats_file, 'wb') as f:
        pickle.dump((mean, std), f)
    logger.info('Training mean and std statistics saved.')
    
    if cfg.preprocess.method == 'random':
        rng = Random(cfg.preprocess.seed)
        random_crop(train_labels, cfg.preprocess.size, cfg.preprocess.samples, rng, pre_sample_dir, sample_dirs_list, cfg, typ="train")
        random_crop(val_labels, cfg.preprocess.size, cfg.preprocess.samples, rng, pre_sample_dir, sample_dirs_list, cfg, typ="val")
        random_crop(test_labels, cfg.preprocess.size, cfg.preprocess.samples, rng, pre_sample_dir, sample_dirs_list, cfg, typ="test")
        logger.info('Random samples generated.')

    logger.debug('Preprocessing complete.')

if __name__ == '__main__':
    main()