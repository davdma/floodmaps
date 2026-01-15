import rasterio
import numpy as np
from pathlib import Path
import re
import sys
from random import Random
import logging
from typing import List, Tuple, Dict
from datetime import datetime
import hydra
from omegaconf import DictConfig
from floodmaps.utils.preprocess_utils import calculate_missing_percent, calculate_cloud_percent
import concurrent.futures
import pandas as pd

# 14 channels in final output (see dataset.py for channel order)
SAR_DATASET_CHANNELS = 14
SAR_MISSING_VALUE = -9999
S2_RGB_MISSING_VALUE = 0
SCL_CLOUD_CLASSES = [8, 9]

def load_tile_for_sampling(tile_info: Tuple):
    """Load a tile, apply shift correction based on GCP points, and return the 
    stacked raster for patch sampling.

    Reads the GCP points file to compute the median pixel shift between SAR and S2
    products. Shifts the S2 products (label, TCI, SCL) relative to SAR channels 
    and crops both to produce aligned data.

    Result is 15 channel stack with 15th channel being the missing mask (recomputed
    after cropping).
    
    Parameters
    ----------
    tile_info : Tuple
        Tuple of (event_path, sar_vv_file, label_file, eid, s2_img_dt, s1_img_dt, points_path)
        
    Returns
    -------
    stacked_tile : np.ndarray
        Stacked raster of the tile with shape (15, H', W') where H' and W' are
        the cropped dimensions after shift correction
    shift_x : int
        Median shift in X direction (pixels)
    shift_y : int
        Median shift in Y direction (pixels)
    """
    event_path, sar_vv_file, label_file, eid, s2_img_dt, s1_img_dt, points_path = tile_info
    sar_vh_file = sar_vv_file.with_name(sar_vv_file.name.replace("_vv.tif", "_vh.tif"))
    tci_file = event_path / f'tci_{s2_img_dt}_{eid}.tif'
    rgb_file = event_path / f'rgb_{s2_img_dt}_{eid}.tif'
    dem_file = event_path / f'dem_{eid}.tif'
    waterbody_file = event_path / f'waterbody_{eid}.tif'
    roads_file = event_path / f'roads_{eid}.tif'
    flowlines_file = event_path / f'flowlines_{eid}.tif'
    scl_file = event_path / f'scl_{s2_img_dt}_{eid}.tif'
    nlcd_file = event_path / f'nlcd_{eid}.tif'

    # Read GCP points and compute median shift
    points_df = pd.read_csv(points_path)
    points_df['shiftX'] = (points_df['mapX'] - points_df['sourceX']) / 10
    points_df['shiftY'] = (points_df['mapY'] - points_df['sourceY']) / 10
    shift_x = int(round(points_df['shiftX'].median()))
    shift_y = int(round(points_df['shiftY'].median()))

    # Load all rasters
    with rasterio.open(label_file) as src: 
        label_raster = src.read([1, 2, 3])
        label_binary = np.where(label_raster[0] != 0, 1, 0)
        label_binary = np.expand_dims(label_binary, axis=0)
        HEIGHT = src.height
        WIDTH = src.width

    with rasterio.open(tci_file) as src:
        tci_raster = src.read()
    tci_floats = (tci_raster / 255).astype(np.float32)

    with rasterio.open(rgb_file) as src:
        rgb_raster = src.read()

    with rasterio.open(sar_vv_file) as src:
        vv_raster = src.read()

    with rasterio.open(sar_vh_file) as src:
        vh_raster = src.read()

    with rasterio.open(dem_file) as src:
        dem_raster = src.read()

    slope = np.gradient(dem_raster, axis=(1, 2))
    slope_y_raster, slope_x_raster = slope

    with rasterio.open(waterbody_file) as src:
        waterbody_raster = src.read()

    with rasterio.open(roads_file) as src:
        roads_raster = src.read()
    
    with rasterio.open(flowlines_file) as src:
        flowlines_raster = src.read()
    
    with rasterio.open(nlcd_file) as src:
        nlcd_raster = src.read()

    with rasterio.open(scl_file) as src:
        scl_raster = src.read()

    # Compute crop indices using unified formula (handles positive, negative, zero shifts)
    abs_sx, abs_sy = abs(shift_x), abs(shift_y)
    new_H, new_W = HEIGHT - abs_sy, WIDTH - abs_sx

    # S2 start indices (label, TCI, SCL)
    s2_y_start = max(0, -shift_y)
    s2_x_start = max(0, shift_x)

    # SAR start indices (VV, VH, DEM, slopes, waterbody, roads, flowlines, NLCD, RGB)
    sar_y_start = max(0, shift_y)
    sar_x_start = max(0, -shift_x)

    # Apply crops to S2 products (label, TCI, SCL)
    label_cropped = label_binary[:, s2_y_start:s2_y_start+new_H, s2_x_start:s2_x_start+new_W]
    tci_cropped = tci_floats[:, s2_y_start:s2_y_start+new_H, s2_x_start:s2_x_start+new_W]
    scl_cropped = scl_raster[:, s2_y_start:s2_y_start+new_H, s2_x_start:s2_x_start+new_W]

    # Apply crops to SAR-aligned products (VV, VH, DEM, slopes, waterbody, roads, flowlines, NLCD, RGB)
    vv_cropped = vv_raster[:, sar_y_start:sar_y_start+new_H, sar_x_start:sar_x_start+new_W]
    vh_cropped = vh_raster[:, sar_y_start:sar_y_start+new_H, sar_x_start:sar_x_start+new_W]
    dem_cropped = dem_raster[:, sar_y_start:sar_y_start+new_H, sar_x_start:sar_x_start+new_W]
    slope_y_cropped = slope_y_raster[:, sar_y_start:sar_y_start+new_H, sar_x_start:sar_x_start+new_W]
    slope_x_cropped = slope_x_raster[:, sar_y_start:sar_y_start+new_H, sar_x_start:sar_x_start+new_W]
    waterbody_cropped = waterbody_raster[:, sar_y_start:sar_y_start+new_H, sar_x_start:sar_x_start+new_W]
    roads_cropped = roads_raster[:, sar_y_start:sar_y_start+new_H, sar_x_start:sar_x_start+new_W]
    flowlines_cropped = flowlines_raster[:, sar_y_start:sar_y_start+new_H, sar_x_start:sar_x_start+new_W]
    nlcd_cropped = nlcd_raster[:, sar_y_start:sar_y_start+new_H, sar_x_start:sar_x_start+new_W]
    rgb_cropped = rgb_raster[:, sar_y_start:sar_y_start+new_H, sar_x_start:sar_x_start+new_W]

    # Recompute missing mask from cropped data
    missing_mask = ((vv_cropped[0] == SAR_MISSING_VALUE) | 
                    (vh_cropped[0] == SAR_MISSING_VALUE) | 
                    (rgb_cropped[0] == S2_RGB_MISSING_VALUE))
    missing_mask = np.expand_dims(missing_mask, axis=0).astype(np.float32)

    # Stack in original channel order:
    # 0-7: VV, VH, DEM, slope_y, slope_x, waterbody, roads, flowlines
    # 8: label
    # 9-11: TCI
    # 12: NLCD
    # 13: SCL
    # 14: missing_mask
    stacked_tile = np.vstack((
        vv_cropped, vh_cropped, dem_cropped,
        slope_y_cropped, slope_x_cropped,
        waterbody_cropped, roads_cropped, flowlines_cropped, 
        label_cropped, tci_cropped, nlcd_cropped, scl_cropped,
        missing_mask
    ), dtype=np.float32)
    
    return stacked_tile, shift_x, shift_y


def sample_patches_random(batch_tile_infos: List[Tuple], size: int, num_samples: int,
        missing_percent: float, cloud_percent: float, seed: int, max_attempts: int = 20000) -> List[np.ndarray]:
    """Sample patches using random uniform sampling for shift ablation tiles.
    
    Each tile is loaded with shift correction applied based on GCP points.
    Stops sampling once the max number of attempts is reached or the number 
    of patches sampled is reached. Returns list of patches for this batch.
    
    Parameters
    ----------
    batch_tile_infos : List[Tuple]
        List of tile info tuples (with points_path) to process
    size : int
        Patch size (e.g., 64)
    num_samples : int
        Number of patches to sample per tile
    missing_percent: float
        Maximum missing percentage for patch acceptance
    cloud_percent: float
        Maximum cloud percentage for patch acceptance
    seed : int
        Random seed for reproducibility
    max_attempts : int
        Maximum number of attempts to sample a patch before stopping
        
    Returns
    -------
    all_patches : List[np.ndarray]
        List of sampled patches
    """
    rng = Random(seed)
    all_patches = []
    
    for i, tile_info in enumerate(batch_tile_infos):
        event_path, _, _, eid, s2_img_dt, s1_img_dt, points_path = tile_info
        try:
            tile_data, shift_x, shift_y = load_tile_for_sampling(tile_info)
        except Exception as e:
            raise RuntimeError(f"Worker failed to load tile {i} (event_path: {event_path}, eid: {eid}, s2_img_dt: {s2_img_dt}, s1_img_dt: {s1_img_dt}): {e}") from e
        
        # Print shift values for this tile
        print(f"Tile {eid} ({s2_img_dt}_{s1_img_dt}): medianShiftX={shift_x}, medianShiftY={shift_y}")
        
        patches_sampled = 0
        _, HEIGHT, WIDTH = tile_data.shape
        
        # Check if tile is large enough for patch size
        if HEIGHT < size or WIDTH < size:
            raise RuntimeError(f"Tile {i} (event_path: {event_path}, eid: {eid}, s2_img_dt: {s2_img_dt}, s1_img_dt: {s1_img_dt}) is too small ({HEIGHT}x{WIDTH}) for patch size {size}x{size}")
        
        attempts = 0
        while patches_sampled < num_samples and attempts < max_attempts:
            attempts += 1
            x = int(rng.uniform(0, HEIGHT - size))
            y = int(rng.uniform(0, WIDTH - size))
            patch = tile_data[:, x:x+size, y:y+size]

            # Filter out missing or high cloud percentage patches
            if calculate_missing_percent(patch[14]) > missing_percent:
                continue
            
            if calculate_cloud_percent(patch[13], classes=SCL_CLOUD_CLASSES) > cloud_percent:
                continue

            all_patches.append(patch[:SAR_DATASET_CHANNELS])
            patches_sampled += 1
        
    return all_patches


def sample_patches_parallel_random(tile_infos: List[Tuple], size: int, num_samples: int, 
                            missing_percent: float, cloud_percent: float, output_file: Path, 
                            seed: int, n_workers: int = None) -> None:
    """Sample patches in parallel using the random method for shift ablation.
    
    Each worker processes a batch of tiles directly and returns patches.
    Results are combined in the main process and saved to a single output file.
    
    Parameters
    ----------
    tile_infos : List[Tuple]
        List of tile info tuples (with points_path) to process
    size : int
        Patch size (e.g., 64)
    num_samples : int
        Number of patches to sample per tile
    missing_percent: float
        Maximum missing percentage for patch acceptance
    cloud_percent: float
        Maximum cloud percentage for patch acceptance
    output_file : Path
        Path to save the output .npy file
    seed : int
        Random seed for reproducibility
    n_workers : int, optional
        Number of worker processes (defaults to 1)
    """
    logger = logging.getLogger('preprocessing')
    
    if n_workers is None:
        n_workers = 1
    
    # Ensure we don't have more workers than tiles
    n_workers = min(n_workers, len(tile_infos))
    logger.info(f'Using {n_workers} workers for {len(tile_infos)} tiles...')

    # Split tiles into balanced batches for workers
    tiles_per_worker = len(tile_infos) // n_workers
    remainder = len(tile_infos) % n_workers
    
    tile_batches = []
    start_idx = 0
    
    for i in range(n_workers):
        batch_size = tiles_per_worker + (1 if i < remainder else 0)
        end_idx = start_idx + batch_size
        
        if start_idx < len(tile_infos):
            tile_batches.append(tile_infos[start_idx:end_idx])
        
        start_idx = end_idx

    # Process tile batches in parallel
    try:
        with concurrent.futures.ProcessPoolExecutor(max_workers=n_workers) as executor:
            futures = [
                executor.submit(
                    sample_patches_random, batch, size, num_samples, 
                    missing_percent, cloud_percent, seed + i * 10000
                ) 
                for i, batch in enumerate(tile_batches)
            ]
            batch_results = [future.result() for future in futures]
    except Exception as e:
        logger.error(f"Failed during parallel random patch sampling: {e}")
        raise RuntimeError(f"Random patch sampling failed: {e}") from e

    # Combine all patches from all batches
    all_patches = []
    for batch_patches in batch_results:
        all_patches.extend(batch_patches)
    
    # Save to output file
    if len(all_patches) > 0:
        patches_array = np.array(all_patches, dtype=np.float32)
    else:
        patches_array = np.empty((0, SAR_DATASET_CHANNELS, size, size), dtype=np.float32)
    
    np.save(output_file, patches_array)
    logger.info(f'Saved {len(all_patches)} patches to {output_file}')

def sample_patches_strided(batch_tile_infos: List[Tuple], size: int, stride: int,
        missing_percent: float, cloud_percent: float) -> List[np.ndarray]:
    """Sample patches using a sliding window with stride for shift ablation tiles.
    
    Each tile is loaded with shift correction applied based on GCP points.
    Uses a sliding window approach that moves by stride pixels in both x and y 
    directions with complete edge coverage.
    
    Parameters
    ----------
    batch_tile_infos : List[Tuple]
        List of tile info tuples (with points_path) to process
    size : int
        Patch size (e.g., 64)
    stride : int
        Stride for sliding window sampling (e.g., 5)
    missing_percent: float
        Maximum missing percentage for patch acceptance
    cloud_percent: float
        Maximum cloud percentage for patch acceptance
        
    Returns
    -------
    all_patches : List[np.ndarray]
        List of sampled patches
    """
    all_patches = []
    
    for i, tile_info in enumerate(batch_tile_infos):
        event_path, _, _, eid, s2_img_dt, s1_img_dt, points_path = tile_info
        try:
            tile_data, shift_x, shift_y = load_tile_for_sampling(tile_info)
        except Exception as e:
            raise RuntimeError(f"Worker failed to load tile {i} (event_path: {event_path}, eid: {eid}, s2_img_dt: {s2_img_dt}, s1_img_dt: {s1_img_dt}): {e}") from e
        
        # Print shift values for this tile
        print(f"Tile {eid} ({s2_img_dt}_{s1_img_dt}): medianShiftX={shift_x}, medianShiftY={shift_y}")
        
        _, HEIGHT, WIDTH = tile_data.shape
        
        # Check if tile is large enough for patch size
        if HEIGHT < size or WIDTH < size:
            raise RuntimeError(f"Tile {i} (event_path: {event_path}, eid: {eid}, s2_img_dt: {s2_img_dt}, s1_img_dt: {s1_img_dt}) is too small ({HEIGHT}x{WIDTH}) for patch size {size}x{size}")
        
        # Generate all window positions with edge coverage
        x_positions = list(range(0, HEIGHT - size + 1, stride))
        # Ensure rightmost edge is included
        if x_positions and x_positions[-1] != HEIGHT - size:
            x_positions.append(HEIGHT - size)
        
        y_positions = list(range(0, WIDTH - size + 1, stride))
        # Ensure bottom edge is included
        if y_positions and y_positions[-1] != WIDTH - size:
            y_positions.append(WIDTH - size)
        
        # Sample all patches at these positions
        for x in x_positions:
            for y in y_positions:
                patch = tile_data[:, x:x+size, y:y+size]
                
                # Filter out missing or high cloud percentage patches
                if calculate_missing_percent(patch[14]) > missing_percent:
                    continue
            
                if calculate_cloud_percent(patch[13], classes=SCL_CLOUD_CLASSES) > cloud_percent:
                    continue
                
                all_patches.append(patch[:SAR_DATASET_CHANNELS])
    
    return all_patches

def sample_patches_parallel_strided(tile_infos: List[Tuple], size: int, stride: int, 
                            missing_percent: float, cloud_percent: float, output_file: Path, 
                            n_workers: int = None) -> None:
    """Sample patches in parallel using the strided method for shift ablation.
    
    Each worker processes a batch of tiles directly and returns patches.
    Results are combined in the main process and saved to a single output file.
    
    Parameters
    ----------
    tile_infos : List[Tuple]
        List of tile info tuples (with points_path) to process
    size : int
        Patch size (e.g., 64)
    stride : int
        Stride for sliding window sampling (e.g., 5)
    missing_percent: float
        Maximum missing percentage for patch acceptance
    cloud_percent: float
        Maximum cloud percentage for patch acceptance
    output_file : Path
        Path to save the output .npy file
    n_workers : int, optional
        Number of worker processes (defaults to 1)
    """
    logger = logging.getLogger('preprocessing')
    
    if n_workers is None:
        n_workers = 1
    
    # Ensure we don't have more workers than tiles
    n_workers = min(n_workers, len(tile_infos))
    logger.info(f'Using {n_workers} workers for {len(tile_infos)} tiles...')

    # Split tiles into balanced batches for workers
    tiles_per_worker = len(tile_infos) // n_workers
    remainder = len(tile_infos) % n_workers
    
    tile_batches = []
    start_idx = 0
    
    for i in range(n_workers):
        batch_size = tiles_per_worker + (1 if i < remainder else 0)
        end_idx = start_idx + batch_size
        
        if start_idx < len(tile_infos):
            tile_batches.append(tile_infos[start_idx:end_idx])
        
        start_idx = end_idx

    # Process tile batches in parallel
    try:
        with concurrent.futures.ProcessPoolExecutor(max_workers=n_workers) as executor:
            futures = [
                executor.submit(
                    sample_patches_strided, batch, size, stride, 
                    missing_percent, cloud_percent
                ) 
                for batch in tile_batches
            ]
            batch_results = [future.result() for future in futures]
    except Exception as e:
        logger.error(f"Failed during parallel strided patch sampling: {e}")
        raise RuntimeError(f"Strided patch sampling failed: {e}") from e

    # Combine all patches from all batches
    all_patches = []
    for batch_patches in batch_results:
        all_patches.extend(batch_patches)
    
    # Save to output file
    if len(all_patches) > 0:
        patches_array = np.array(all_patches, dtype=np.float32)
    else:
        patches_array = np.empty((0, SAR_DATASET_CHANNELS, size, size), dtype=np.float32)
    
    np.save(output_file, patches_array)
    logger.info(f'Saved {len(all_patches)} patches to {output_file}')

def build_label_index(label_dirs: list[str], cfg: DictConfig) -> Dict[Tuple[str, str], Path]:
    """Build a dictionary mapping (s2_img_dt, eid) to the label file path."""
    idx: Dict[Tuple[str, str], Path] = {}
    p = re.compile(r'label_(\d{8})_(\d{8}_\d+_\d+)\.tif$')
    for ld in label_dirs or []:
        base = Path(cfg.paths.labels_dir) / ld
        if not base.is_dir():
            continue
        for fp in base.glob('label_*.tif'):
            m = p.search(fp.name)
            if not m:
                continue
            s2_img_dt, eid = m.group(1), m.group(2)
            idx.setdefault((s2_img_dt, eid), fp)  # keep first occurrence
    return idx


@hydra.main(version_base=None, config_path='pkg://configs', config_name='config.yaml')
def main(cfg: DictConfig) -> None:
    """Preprocesses raw S1 tiles and coincident S2 label (machine & human) into smaller patches
    for shift ablation analysis. Manual GCP points are used to calculate the shift in pixels
    between the SAR tile and the corresponding prediction or label. The shift is applied
    and the final aligned patches are saved.
    
    This script specifically takes GCP points files from QGIS in cfg.s1.ablation_dir
    corresponding to SAR tiles and their predictions, with the naming convention:
    sar_{s2_img_dt}_{s1_img_dt}_{eid}_vh.tif.points
    
    The .points file is read in as a csv file with columns ["mapX", "mapY", "sourceX", "sourceY"]
    where sourceX is X coordinate of point in SAR tile, and mapX is X coordinate of point
    in S2 tile and label.

    NOTE: The script assumes that the SAR tiles are in the test set for ablation analysis.
    
    cfg.preprocess Parameters:
    - size: int (pixel width of patch)
    - method: str ['random', 'strided']
    - samples: int (number of samples per image for random method)
    - stride: int (for strided method)
    - missing_percent: float (maximum missing percentage for patch acceptance) (default: 0.0)
    - cloud_percent: float (maximum cloud percentage for patch acceptance) (default: 0.1)
    - seed: int (random number generator seed for random method)
    - n_workers: int (number of workers for parallel processing)
    - save_dir: str (path to save shift ablation patches)
    - s1.ablation_dir: str (path to the ablation directory)
    - s1.sample_dirs: List[str] (list of sample directories under cfg.data.imagery_dir)
    - s1.label_dirs: List[str] (list of label directories under cfg.data.imagery_dir)
    """
    # Setup logging
    logger = logging.getLogger('preprocessing')
    logger.setLevel(logging.DEBUG)
    formatter = logging.Formatter('%(asctime)s - %(name)s - %(levelname)s - %(message)s')
    handler = logging.StreamHandler(sys.stdout)
    handler.setFormatter(formatter)
    handler.setLevel(logging.DEBUG)
    logger.addHandler(handler)
    logger.propagate = False

    # Set default number of workers
    n_workers = getattr(cfg.preprocess, 'n_workers', 1)

    timestamp = datetime.now().strftime('%Y-%m-%d %H:%M:%S')
    logger.info(f'''Starting S1 shift ablation patch preprocessing:
        Date:            {timestamp}
        Save dir:        {cfg.preprocess.save_dir}
        Patch size:      {cfg.preprocess.size}
        Sampling method: {cfg.preprocess.method}
        Samples per tile (Random method): {getattr(cfg.preprocess, 'samples', None)}
        Stride (Strided method): {getattr(cfg.preprocess, 'stride', None)}
        Missing percent: {getattr(cfg.preprocess, 'missing_percent', 0.0)} (default: 0.0)
        Cloud percent: {getattr(cfg.preprocess, 'cloud_percent', 0.1)} (default: 0.1)
        Random seed:     {getattr(cfg.preprocess, 'seed', None)}
        Ablation dir:    {cfg.preprocess.s1.ablation_dir}
        Sample dir(s):   {cfg.preprocess.s1.sample_dirs}
        Label dir(s):    {cfg.preprocess.s1.label_dirs}
    ''')

    # Create preprocessing directory
    pre_sample_dir = Path(cfg.preprocess.save_dir)
    pre_sample_dir.mkdir(parents=True, exist_ok=True)
    logger.info(f'Save directory: {pre_sample_dir.name}')

    # Get directories and split ratios from config
    cfg_s1 = cfg.preprocess.get('s1', {})
    sample_dirs_list = cfg_s1.get('sample_dirs', [])
    label_dirs_list = cfg_s1.get('label_dirs', [])

    if len(sample_dirs_list) == 0:
        raise ValueError('No sample directories were provided.')

    # Build label index for manual labels
    label_idx = build_label_index(label_dirs_list, cfg)

    ablation_dir = Path(cfg.preprocess.s1.ablation_dir)
    if not ablation_dir.is_dir():
        raise ValueError(f'Ablation directory {ablation_dir} does not exist.')

    # eid to event_path dictionary
    seen_eids = set()
    eid_to_event_path = {}
    for sd in sample_dirs_list:
        sample_path = Path(cfg.paths.imagery_dir) / sd
        if not sample_path.is_dir():
            logger.debug(f'Sample directory {sd} is invalid, skipping...')
            continue
        
        for event_dir in sample_path.glob('[0-9]*_*_*'):
            if not event_dir.is_dir():
                continue
            eid = event_dir.name
            if eid in seen_eids:
                logger.debug(f'Event ID {eid} contained in multiple sample dirs, skipping...')
                continue
            
            eid_to_event_path[eid] = event_dir
            seen_eids.add(eid)
    
    # collect tile infos for each points file
    tile_infos = []
    p = re.compile(r'sar_(\d{8})_(\d{8})_(\d{8}_\d+_\d+)_.*\.tif\.points')
    for points_path in ablation_dir.glob("*.points"):
        m = p.match(points_path.name)
        if not m:
            continue
        s2_img_dt = m.group(1)
        s1_img_dt = m.group(2)
        eid = m.group(3)
        event_path = eid_to_event_path[eid]
        sar_vv_file = event_path / f'sar_{s2_img_dt}_{s1_img_dt}_{eid}_vv.tif'
        label_file = label_idx.get((s2_img_dt, eid), None)
        if label_file is None and (event_path / f'pred_{s2_img_dt}_{eid}.tif').exists():
            label_file = event_path / f'pred_{s2_img_dt}_{eid}.tif'
        tile_infos.append((event_path, sar_vv_file, label_file, eid, s2_img_dt, s1_img_dt, points_path))

    logger.info(f'Grabbed {len(tile_infos)} SAR tiles from points files.')
    if len(tile_infos) == 0:
        raise ValueError('No SAR tiles were found.')

    # Sample patches in parallel, no need for chunking or scratch
    missing_percent = getattr(cfg.preprocess, 'missing_percent', 0.0)
    cloud_percent = getattr(cfg.preprocess, 'cloud_percent', 0.1)

    if cfg.preprocess.method == 'random':
        output_file = pre_sample_dir / 'test_shift_ablation_patches.npy'
        
        sample_patches_parallel_random(
            tile_infos, cfg.preprocess.size, cfg.preprocess.samples, 
            missing_percent, cloud_percent, output_file, cfg.preprocess.seed, n_workers
        )
        
        logger.info('Parallel random patch sampling complete.')
    elif cfg.preprocess.method == 'strided':  
        output_file = pre_sample_dir / 'test_shift_ablation_patches.npy'
        
        sample_patches_parallel_strided(
            tile_infos, cfg.preprocess.size, cfg.preprocess.stride, 
            missing_percent, cloud_percent, output_file, n_workers
        )
        
        logger.info('Parallel strided patch sampling complete.')
    else:
        raise ValueError(f"Unsupported sampling method: {cfg.preprocess.method}")

    logger.info('Preprocessing complete.')

if __name__ == '__main__':
    main()