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
import multiprocessing as mp
import psutil
import shutil
from numpy.lib.format import open_memmap

from floodmaps.utils.preprocess_utils import (
    PROCESSING_BASELINE_NAIVE,
    BOA_ADD_OFFSET,
    compute_awei_sh,
    compute_awei_nsh,
    compute_ndwi,
    compute_mndwi
)

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


def loadMaskedStack(img_dt, eid, sample_dirs: List[str], cfg: DictConfig):
    """Load and mask the stack of DEM and slope channels for statistics computation.
    
    RGB, B08, SWIR1, SWIR2 are scaled by 10000 (reflectance values).
    NDWI, MNDWI, AWEI_sh, AWEI_nsh are left unnormalized in their computed ranges.
    Only DEM and slopes need statistics.

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
        Stack of DEM and slope channels (3 channels) with missing values masked out.
    """
    event_dir = _find_event_dir(img_dt, eid, sample_dirs, cfg)
    if event_dir is None:
        raise FileNotFoundError(f"Could not find assets for event {eid} across provided sample_dirs: {sample_dirs}")
    rgb_file = event_dir / f'rgb_{img_dt}_{eid}.tif'
    ndwi_file = event_dir / f'ndwi_{img_dt}_{eid}.tif'
    dem_file = event_dir / f'dem_{eid}.tif'

    with rasterio.open(rgb_file) as src:
        rgb_raster = src.read()
    
    with rasterio.open(ndwi_file) as src:
        ndwi_raster = src.read()

    with rasterio.open(dem_file) as src:
        dem_raster = src.read()

    slope = np.gradient(dem_raster, axis=(1,2))
    slope_y_raster, slope_x_raster = slope

    dem_raster = dem_raster.reshape((1, -1))
    slope_y_raster = slope_y_raster.reshape((1, -1))
    slope_x_raster = slope_x_raster.reshape((1, -1))

    mask = (rgb_raster[0] != 0) & (ndwi_raster[0] != -999999)
    mask = mask.flatten()

    # Only stack DEM and slopes for statistics computation
    stack = np.vstack((dem_raster, slope_y_raster, slope_x_raster), dtype=np.float32)

    masked_stack = stack[:, mask]

    return masked_stack

def trainMean(train_events, sample_dirs: List[str], cfg: DictConfig):
    """Calculate mean of DEM and slope channels only.
    
    RGB/B08/SWIR1/SWIR2 are scaled by 10000, spectral indices (NDWI/MNDWI/AWEI) 
    are left unnormalized, so we only compute statistics for DEM and slopes (3 channels).

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
        Channel means for DEM and slopes (3 channels).
    """
    logger = logging.getLogger('preprocessing')
    count = 0
    total_sum = np.zeros(3, dtype=np.float64)  # 3 channels: DEM, slope_y, slope_x

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
    """Calculate std of DEM and slope channels only.
    
    RGB/B08/SWIR1/SWIR2 are scaled by 10000, spectral indices (NDWI/MNDWI/AWEI) 
    are left unnormalized, so we only compute statistics for DEM and slopes (3 channels).

    Parameters
    ----------
    train_events : list[tuple[str, str]]
        List of training flood event folders where raw data tiles are stored.
        First element is the image date, second element is the event id.
    train_means : ndarray
        Channel means for DEM and slopes (3 channels).
    sample_dirs : list[str]
        Dataset directories containing S2 tiles for patch sampling.

    Returns
    -------
    overall_channel_std : ndarray
        Channel stds for DEM and slopes (3 channels).
    """
    logger = logging.getLogger('preprocessing')
    count = 0
    variances = np.zeros(3, dtype=np.float64)  # 3 channels: DEM, slope_y, slope_x

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


def load_tile_for_sampling(tile_info: Tuple):
    """Load a tile and return the stacked raster for patch sampling.
    
    Parameters
    ----------
    tile_info : Tuple
        Tuple of (label_path, sample_dirs, cfg)
        
    Returns
    -------
    stacked_raster : np.ndarray
        Stacked raster of the tile (23 channels: 22 data + 1 missing mask)
    """
    label_rel, sample_dirs, cfg = tile_info
    
    p = re.compile('label_(\d{8})_(.+).tif')
    m = p.search(label_rel)
    
    if m:
        tile_date = m.group(1)
        eid = m.group(2)
        tile_dt_obj = datetime.strptime(tile_date, '%Y%m%d')
    else:
        raise ValueError(f'Label file {label_rel} does not match expected format.')

    with rasterio.open(Path(cfg.paths.labels_dir) / label_rel) as src:
        label_raster = src.read([1, 2, 3])
        label_binary = np.where(label_raster[0] != 0, 1, 0)
        label_binary = np.expand_dims(label_binary, axis = 0)

    event_dir = _find_event_dir(tile_date, eid, sample_dirs, cfg)
    if event_dir is None:
        raise FileNotFoundError(f"Could not find assets for event {eid} across provided sample_dirs: {sample_dirs}")

    tci_file = event_dir / f'tci_{tile_date}_{eid}.tif'
    rgb_file = event_dir / f'rgb_{tile_date}_{eid}.tif'
    b08_file = event_dir / f'b08_{tile_date}_{eid}.tif'
    b11_file = event_dir / f'b11_{tile_date}_{eid}.tif'
    b12_file = event_dir / f'b12_{tile_date}_{eid}.tif'
    ndwi_file = event_dir / f'ndwi_{tile_date}_{eid}.tif'
    scl_file = event_dir / f'scl_{tile_date}_{eid}.tif'
    dem_file = event_dir / f'dem_{eid}.tif'
    waterbody_file = event_dir / f'waterbody_{eid}.tif'
    roads_file = event_dir / f'roads_{eid}.tif'
    flowlines_file = event_dir / f'flowlines_{eid}.tif'
    nlcd_file = event_dir / f'nlcd_{eid}.tif'

    with rasterio.open(tci_file) as src:
        tci_raster = src.read()
        tci_floats = (tci_raster / 255).astype(np.float32)

    with rasterio.open(rgb_file) as src:
        rgb_raster = src.read().astype(np.float32)
    
    # Extract missing values mask BEFORE applying offset (raw DN 0 = missing)
    missing_mask = (rgb_raster[0] == 0).astype(np.float32)
    missing_mask = np.expand_dims(missing_mask, axis=0)  # Shape: (1, H, W)
    
    if tile_dt_obj >= PROCESSING_BASELINE_NAIVE:
        rgb_raster_sr = np.clip((rgb_raster + BOA_ADD_OFFSET) / 10000.0, 0, 1)
    else:
        rgb_raster_sr = np.clip(rgb_raster / 10000.0, 0, 1)

    with rasterio.open(b08_file) as src:
        b08_raster = src.read().astype(np.float32)
        if tile_dt_obj >= PROCESSING_BASELINE_NAIVE:
            b08_raster_sr = np.clip((b08_raster + BOA_ADD_OFFSET) / 10000.0, 0, 1)
        else:
            b08_raster_sr = np.clip(b08_raster / 10000.0, 0, 1)

    with rasterio.open(b11_file) as src:
        b11_raster = src.read().astype(np.float32)
        if tile_dt_obj >= PROCESSING_BASELINE_NAIVE:
            b11_raster_sr = np.clip((b11_raster + BOA_ADD_OFFSET) / 10000.0, 0, 1)
        else:
            b11_raster_sr = np.clip(b11_raster / 10000.0, 0, 1)

    with rasterio.open(b12_file) as src:
        b12_raster = src.read().astype(np.float32)
        if tile_dt_obj >= PROCESSING_BASELINE_NAIVE:
            b12_raster_sr = np.clip((b12_raster + BOA_ADD_OFFSET) / 10000.0, 0, 1)
        else:
            b12_raster_sr = np.clip(b12_raster / 10000.0, 0, 1)

    # Post processing baseline, need to use different equation for ndwi
    # This is a temporary patch, but we want to fix this at the data pipeline step!
    recompute_ndwi = compute_ndwi(rgb_raster_sr[1], b08_raster_sr[0], missing_val=-999999)
    ndwi_raster = np.expand_dims(recompute_ndwi, axis = 0)
    ndwi_raster = np.where(missing_mask, -999999, ndwi_raster)

    # Compute MNDWI (Modified NDWI): (Green - SWIR1) / (Green + SWIR1)
    mndwi_raster = compute_mndwi(rgb_raster_sr[1], b11_raster_sr[0], missing_val=-999999)
    mndwi_raster = np.expand_dims(mndwi_raster, axis=0)
    mndwi_raster = np.where(missing_mask, -999999, mndwi_raster)

    # Compute AWEI_sh (Automated Water Extraction Index - shadow): Blue + 2.5*Green - 1.5*(NIR + SWIR1) - 0.25*SWIR2
    awei_sh_raster = compute_awei_sh(rgb_raster_sr[2], rgb_raster_sr[1], b08_raster_sr[0], b11_raster_sr[0], b12_raster_sr[0])
    awei_sh_raster = np.expand_dims(awei_sh_raster, axis=0)
    awei_sh_raster = np.where(missing_mask, -999999, awei_sh_raster)

    # Compute AWEI_nsh (Automated Water Extraction Index - no shadow): 4*(Green - SWIR1) - (0.25*NIR + 2.75*SWIR2_
    awei_nsh_raster = compute_awei_nsh(rgb_raster_sr[1], b11_raster_sr[0], b08_raster_sr[0], b12_raster_sr[0])
    awei_nsh_raster = np.expand_dims(awei_nsh_raster, axis=0)
    awei_nsh_raster = np.where(missing_mask, -999999, awei_nsh_raster)

    with rasterio.open(dem_file) as src:
        dem_raster = src.read().astype(np.float32)

    # calculate xy gradient with np.gradient
    slope = np.gradient(dem_raster, axis=(1,2))
    slope_y_raster, slope_x_raster = slope

    with rasterio.open(waterbody_file) as src:
        waterbody_raster = src.read().astype(np.float32)

    with rasterio.open(roads_file) as src:
        roads_raster = src.read().astype(np.float32)

    with rasterio.open(flowlines_file) as src:
        flowlines_raster = src.read().astype(np.float32)

    with rasterio.open(nlcd_file) as src:
        nlcd_raster = src.read().astype(np.float32)

    with rasterio.open(scl_file) as src:
        scl_raster = src.read().astype(np.float32)

    # Stack all tiles: 22 data channels + 1 missing mask channel (23 total)
    # New channel order: RGB(1-3), B08(4), SWIR1(5), SWIR2(6), NDWI(7), MNDWI(8), AWEI_sh(9), AWEI_nsh(10),
    #                    DEM(11), slope_y(12), slope_x(13), waterbody(14), roads(15), flowlines(16),
    #                    label(17), TCI(18-20), NLCD(21), SCL(22)
    # Channel 22 (0-indexed) is the missing mask for filtering, will be dropped before saving
    stacked_tile = np.vstack((rgb_raster_sr, b08_raster_sr, b11_raster_sr, b12_raster_sr, ndwi_raster, 
                                mndwi_raster, awei_sh_raster, awei_nsh_raster,
                                dem_raster, slope_y_raster, slope_x_raster, 
                                waterbody_raster, roads_raster, flowlines_raster, 
                                label_binary, tci_floats, nlcd_raster, scl_raster,
                                missing_mask), dtype=np.float32)
    
    return stacked_tile


def patch_contains_missing(patch: np.ndarray) -> bool:
    """Check if a patch loaded from tile contains missing values.
    
    Parameters
    ----------
    patch : np.ndarray
        Patch to check
    """
    return np.any(patch[22] == 1) or np.any(patch[6] == -999999) or np.any(patch[7] == -999999)


def sample_patches_in_mem(tile_infos: List[Tuple], size: int, num_samples: int, 
                          seed: int, max_attempts: int = 10000) -> np.ndarray:
    """Sample patches and stores them in memory.
    
    Parameters
    ----------
    tile_infos : List[Tuple]
        List of tile info tuples to process
    size : int
        Patch size (e.g., 64)
    num_samples : int
        Number of patches to sample per tile
    seed : int
        Random seed for reproducibility
    max_attempts : int
        Maximum number of attempts to sample a patch before raising error

    Returns
    -------
    dataset : np.ndarray
        Array of sampled patches
    """
    rng = Random(seed)
    total_patches = num_samples * len(tile_infos)
    dataset = np.empty((total_patches, 22, size, size), dtype=np.float32)
    
    for i, tile_info in enumerate(tile_infos):
        try:
            tile_data = load_tile_for_sampling(tile_info)
        except Exception as e:
            label_rel, _, _ = tile_info
            raise RuntimeError(f"Worker failed to load tile {i} ({label_rel}): {e}") from e
        
        patches_sampled = 0
        _, HEIGHT, WIDTH = tile_data.shape
        attempts = 0
        while patches_sampled < num_samples and attempts < max_attempts:
            attempts += 1
            x = int(rng.uniform(0, HEIGHT - size))
            y = int(rng.uniform(0, WIDTH - size))
            patch = tile_data[:, x : x + size, y : y + size]
            
            # Filter using missing mask (channel 22) and NDWI/MNDWI (channels 6, 7)
            if patch_contains_missing(patch):
                continue

            dataset[i * num_samples + patches_sampled] = patch[:22]
            patches_sampled += 1
        
        if patches_sampled < num_samples:
            label_rel, _, _ = tile_info
            raise RuntimeError(f"Worker exceeded max retries {max_attempts}, only sampled {patches_sampled}/{num_samples} patches for tile {i} ({label_rel})")

    return dataset


def sample_patches_in_disk(tile_infos: List[Tuple], size: int, num_samples: int, 
                          seed: int, scratch_dir: Path, worker_id: int, max_attempts: int = 10000) -> Path:
    """Sample patches and stores them in memory mapped file on disk.
    
    Parameters
    ----------
    tile_infos : List[Tuple]
        List of tile info tuples to process
    size : int
        Patch size (e.g., 64)
    num_samples : int
        Number of patches to sample per tile
    seed : int
        Random seed for reproducibility
    scratch_dir : Path
        Path to scratch directory
    worker_id : int
        Worker ID
    max_attempts : int
        Maximum number of attempts to fail to sample a patch before giving up

    Returns
    -------
    dataset_path : Path
        Path to the memory mapped file of sampled patches
    """
    rng = Random(seed)
    total_patches = num_samples * len(tile_infos)
    tmp_file = scratch_dir / f"tmp_{worker_id}.dat"
    dataset = np.memmap(tmp_file, dtype=np.float32, shape=(total_patches, 22, size, size), mode="w+")
    
    try:
        for i, tile_info in enumerate(tile_infos):
            try:
                tile_data = load_tile_for_sampling(tile_info)
            except Exception as e:
                label_rel, _, _ = tile_info
                raise RuntimeError(f"Worker {worker_id} failed to load tile {i} ({label_rel}): {e}") from e
            
            patches_sampled = 0
            _, HEIGHT, WIDTH = tile_data.shape
            attempts = 0
            while patches_sampled < num_samples and attempts < max_attempts:
                attempts += 1
                x = int(rng.uniform(0, HEIGHT - size))
                y = int(rng.uniform(0, WIDTH - size))
                patch = tile_data[:, x : x + size, y : y + size]
                
                # Filter using missing mask (channel 22) and NDWI/MNDWI (channels 6, 7)
                if patch_contains_missing(patch):
                    continue

                dataset[i * num_samples + patches_sampled] = patch[:22]
                patches_sampled += 1

            if patches_sampled < num_samples:
                label_rel, _, _ = tile_info
                raise RuntimeError(f"Worker {worker_id} exceeded max retries {max_attempts}, only sampled {patches_sampled}/{num_samples} patches for tile {i} ({label_rel})")
    finally:
        # Ensure cleanup even if an error occurs
        dataset.flush()
        del dataset

    return tmp_file


def sample_patches_parallel(label_paths: List[str], size: int, num_samples: int, 
                          output_file: Path, sample_dirs: List[str], cfg: DictConfig,
                          seed: int, n_workers: int = None) -> None:
    """Sample patches in parallel with random sampling. Strategy will depend on the memory available. If
    memory is comfortably below the total node memory (on improv you get ~230GB regardless
    of how many cpus requested), then we can just have each worker fill out their own
    array and then concatenate them in the parent.

    If memory required for entire array is too close or exceeds total node memory,
    then each worker will write to a memory mapped array and then combine them at the end.
    
    Parameters
    ----------
    label_paths : List[str]
        List of label file paths to process
    size : int
        Patch size (e.g., 64)
    num_samples : int
        Number of patches to sample per tile
    output_file : Path
        Path to save the output .npy file
    sample_dirs : List[str]
        List of sample directories
    cfg : DictConfig
        Configuration object
    seed : int
        Random seed for reproducibility
    n_workers : int, optional
        Number of worker processes (defaults to 1)
    """
    logger = logging.getLogger('preprocessing')
    
    if n_workers is None:
        n_workers = 1
    
    logger.info(f'Specified {n_workers} workers for patch sampling.')
    n_workers = min(n_workers, len(label_paths))
    logger.info(f'Using {n_workers} workers for {len(label_paths)} labels...')
    
    total_patches = len(label_paths) * num_samples
    array_shape = (total_patches, 22, size, size)
    array_dtype = np.float32

    # calculate how much memory required for the entire array (factor of 2 for concatenation operation)
    total_mem_required = array_shape[0] * array_shape[1] * array_shape[2] * array_shape[3] * np.dtype(array_dtype).itemsize * 2
    total_mem_available = psutil.virtual_memory().available
    logger.info(f'Total memory required for the entire array x2: {total_mem_required / 1024**3:.2f} GB')
    logger.info(f'Total memory available: {total_mem_available / 1024**3:.2f} GB')
    
    # divide up the labels into n_workers chunks
    worker_tile_infos = []
    batch_sizes = []
    start_idx = 0
    labels_per_worker = len(label_paths) // n_workers
    labels_remainder = len(label_paths) % n_workers
    for worker_id in range(n_workers):
        worker_label_count = labels_per_worker + (1 if worker_id < labels_remainder else 0)
        end_idx = start_idx + worker_label_count
        labels_chunk = label_paths[start_idx:end_idx]
        # Convert to tile_info tuples
        tile_infos_chunk = [(label_path, sample_dirs, cfg) for label_path in labels_chunk]
        worker_tile_infos.append(tile_infos_chunk)
        batch_sizes.append(worker_label_count)
        start_idx += worker_label_count
        
    # Log batch distribution
    logger.info(f'Label distribution per worker: {batch_sizes}')
    
    if total_mem_required < total_mem_available * 0.9:
        # use simple concatenation strategy
        logger.info('Total memory required is below available memory, using in-memory arrays and concatenation.')

        try:
            with mp.Pool(n_workers) as pool:
                results = pool.starmap(sample_patches_in_mem, [(tile_infos, size, num_samples, seed+i*10000) for i, tile_infos in enumerate(worker_tile_infos)])
        except Exception as e:
            logger.error(f"Failed during parallel patch sampling (in-memory): {e}")
            logger.error("One or more worker processes failed during patch sampling")
            raise RuntimeError(f"Patch sampling failed: {e}") from e

        try:
            final_arr = np.concatenate(results, axis=0)
            np.save(output_file, final_arr)
            logger.info('Sampling complete.')
        except Exception as e:
            logger.exception(f"Failed to concatenate results or save output.")
            raise RuntimeError(f"Failed to save final array: {e}") from e
    else:
        logger.info('Total memory required exceeds available memory, using memory mapping strategy.')

        # Use scratch directory for temp files
        scratch_dir = Path(f"/scratch/floodmapspre")
        scratch_dir.mkdir(parents=True, exist_ok=True)
        logger.info(f"Using scratch directory: {scratch_dir}")
        
        try:
            # each worker writes to their own file
            with mp.Pool(n_workers) as pool:
                worker_files = pool.starmap(sample_patches_in_disk, [(tile_infos, size, num_samples, seed+i*10000, scratch_dir, i) for i, tile_infos in enumerate(worker_tile_infos)])
        except Exception as e:
            logger.error(f"Failed during parallel patch sampling (memory-mapped): {e}")
            logger.error("One or more worker processes failed during patch sampling")
            # Clean up any partial files before re-raising
            if scratch_dir.exists():
                try:
                    shutil.rmtree(scratch_dir)
                    logger.info("Cleaned up scratch directory after worker failure")
                except Exception as cleanup_e:
                    logger.warning(f"Failed to clean up scratch directory: {cleanup_e}")
            raise RuntimeError(f"Patch sampling failed: {e}") from e
        
        try:
            # combine files into one memory mapped array
            final_npy = open_memmap(output_file, mode='w+', dtype='float32', shape=array_shape)
            start_idx = 0
            for i, f in enumerate(worker_files):
                labels_in_worker = len(worker_tile_infos[i])
                chunk_shape = (num_samples * labels_in_worker, 22, size, size)
                worker_data = np.memmap(f, dtype=np.float32, shape=chunk_shape, mode="r")  # load worker memmap
                end_idx = start_idx + worker_data.shape[0]
                final_npy[start_idx:end_idx] = worker_data  # copy into final memmap
                start_idx = end_idx

            # Finalize and clean up
            final_npy.flush()
            del final_npy
            logger.info('Memory-mapped sampling complete.')
        except Exception as e:
            logger.exception(f"Failed to combine worker files or save final output.")
            raise RuntimeError(f"Failed to create final memory-mapped array: {e}") from e
        finally:
            # Always clean up scratch directory
            if scratch_dir.exists():
                try:
                    shutil.rmtree(scratch_dir)
                    logger.info("Cleaned up scratch directory")
                except Exception as cleanup_e:
                    logger.warning(f"Failed to clean up scratch directory: {cleanup_e}")

def sample_patches_strided(tile_infos: List[Tuple], size: int, stride: int) -> np.ndarray:
    """Sample patches using a sliding window with stride and complete edge coverage.
    
    This function uses a sliding window approach that moves by stride pixels
    in both x and y directions. It ensures complete coverage by explicitly including
    positions that cover the right and bottom edges of each tile.
    
    Parameters
    ----------
    tile_infos : List[Tuple]
        List of tile info tuples to process
    size : int
        Patch size (e.g., 64)
    stride : int
        Stride for sliding window sampling (e.g., 5)
    
    Returns
    -------
    dataset : np.ndarray
        Array of sampled patches (filtered for valid patches only)
    """
    all_patches = []
    
    for i, tile_info in enumerate(tile_infos):
        try:
            tile_data = load_tile_for_sampling(tile_info)
        except Exception as e:
            label_rel, _, _ = tile_info
            raise RuntimeError(f"Worker failed to load tile {i} ({label_rel}): {e}") from e
        
        _, HEIGHT, WIDTH = tile_data.shape
        
        # Check if tile is large enough for patch size
        if HEIGHT < size or WIDTH < size:
            label_rel, _, _ = tile_info
            raise RuntimeError(f"Tile {i} ({label_rel}) is too small ({HEIGHT}x{WIDTH}) for patch size {size}x{size}")
        
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
                
                # Filter using missing mask (channel 22) and NDWI/MNDWI (channels 6, 7)
                if patch_contains_missing(patch):
                    continue
                
                all_patches.append(patch[:22])
    
    if len(all_patches) == 0:
        raise RuntimeError("No valid patches found after filtering")
    
    return np.array(all_patches, dtype=np.float32)

def sample_patches_parallel_strided(label_paths: List[str], size: int, stride: int, 
                          output_file: Path, sample_dirs: List[str], cfg: DictConfig,
                          n_workers: int = None) -> None:
    """Sample patches in parallel with strided sliding window sampling. Each worker samples
    patches strided, and results are concatenated.

    NOTE: No memory mapping strategy for strided sampling.
    
    Parameters
    ----------
    label_paths : List[str]
        List of label file paths to process
    size : int
        Patch size (e.g., 64)
    stride : int
        Stride for sliding window sampling (e.g., 5)
    output_file : Path
        Path to save the output .npy file
    sample_dirs : List[str]
        List of sample directories
    cfg : DictConfig
        Configuration object
    n_workers : int, optional
        Number of worker processes (defaults to 1)
    """
    logger = logging.getLogger('preprocessing')
    
    if n_workers is None:
        n_workers = 1
    
    logger.info(f'Specified {n_workers} workers for patch sampling.')
    n_workers = min(n_workers, len(label_paths))
    logger.info(f'Using {n_workers} workers for {len(label_paths)} labels...')
    
    # divide up the labels into n_workers chunks
    worker_tile_infos = []
    batch_sizes = []
    start_idx = 0
    labels_per_worker = len(label_paths) // n_workers
    labels_remainder = len(label_paths) % n_workers
    for worker_id in range(n_workers):
        worker_label_count = labels_per_worker + (1 if worker_id < labels_remainder else 0)
        end_idx = start_idx + worker_label_count
        labels_chunk = label_paths[start_idx:end_idx]
        # Convert to tile_info tuples
        tile_infos_chunk = [(label_path, sample_dirs, cfg) for label_path in labels_chunk]
        worker_tile_infos.append(tile_infos_chunk)
        batch_sizes.append(worker_label_count)
        start_idx += worker_label_count
        
    # Log batch distribution
    logger.info(f'Label distribution per worker: {batch_sizes}')
    
    logger.info('Strided sampling uses in-memory arrays and concatenation.')

    try:
        with mp.Pool(n_workers) as pool:
            results = pool.starmap(sample_patches_strided, [(tile_infos, size, stride) for i, tile_infos in enumerate(worker_tile_infos)])
    except Exception as e:
        logger.error(f"Failed during parallel strided patch sampling: {e}")
        logger.error("One or more worker processes failed during strided patch sampling")
        raise RuntimeError(f"Strided patch sampling failed: {e}") from e

    try:
        final_arr = np.concatenate(results, axis=0)
        np.save(output_file, final_arr)
        logger.info('Sampling complete.')
    except Exception as e:
        logger.exception(f"Failed to concatenate results or save output.")
        raise RuntimeError(f"Failed to save final array: {e}") from e

def save_event_splits(train_labels: List[str], val_labels: List[str], test_labels: List[str], cfg: DictConfig,
        output_dir: Path, timestamp: str, data_type: str = "s2_manual") -> None:
    """Save event splits to a YAML file for reproducibility and reference.
    
    Parameters
    ----------
    train_labels : List[str]
        List of training label file paths
    val_labels : List[str]  
        List of validation label file paths
    test_labels : List[str]
        List of test label file paths
    cfg : DictConfig
        Configuration object
    output_dir : Path
        Directory to save the splits file
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
    splits_file = output_dir / f'metadata.yaml'
    
    # Prepare splits data structure
    event_splits = {
        'train': train_events,
        'val': val_events, 
        'test': test_events,
        'metadata': {
            'data_type': data_type,
            'method': cfg.preprocess.method,
            'size': cfg.preprocess.size,
            'samples': getattr(cfg.preprocess, 'samples', None),
            'stride': getattr(cfg.preprocess, 'stride', None),
            'seed': getattr(cfg.preprocess, 'seed', None),
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
    mean and std of the training tiles.
    
    cfg.preprocess Parameters:
    - size: int (pixel width of patch)
    - method: str ['random', 'strided']
    - samples: int (number of samples per image for random method)
    - stride: int (for strided method)
    - seed: int (random number generator seed for random method)
    - n_workers: int (number of workers for parallel processing)
    - sample_dirs: List[str] (list of sample directories under cfg.data.imagery_dir)
    - suffix: str (optional suffix to append to preprocessed folder)
    """
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

    # Create timestamp for logging
    timestamp = datetime.now().strftime('%Y-%m-%d %H:%M:%S')
    logger.info(f'''Starting S2 manual labeling preprocessing:
        Date:            {timestamp}
        Patch size:      {cfg.preprocess.size}
        Sampling method: {cfg.preprocess.method}
        Samples per tile (Random method): {getattr(cfg.preprocess, 'samples', None)}
        Stride (Strided method): {getattr(cfg.preprocess, 'stride', None)}
        Random seed:     {getattr(cfg.preprocess, 'seed', None)}
        Workers:         {n_workers}
        Sample dir(s):   {cfg.preprocess.s2.sample_dirs}
        Suffix:          {getattr(cfg.preprocess, 'suffix', None)}
    ''')

    # make our preprocess directory
    sampling_param = cfg.preprocess.samples if cfg.preprocess.method == 'random' else cfg.preprocess.stride
    if getattr(cfg.preprocess, 'suffix', None):
        pre_sample_dir = Path(cfg.paths.preprocess_dir) / 's2' / f'{cfg.preprocess.method}_{cfg.preprocess.size}_{sampling_param}_{cfg.preprocess.suffix}'
    else:
        pre_sample_dir = Path(cfg.paths.preprocess_dir) / 's2' / f'{cfg.preprocess.method}_{cfg.preprocess.size}_{sampling_param}'
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
    save_event_splits(train_labels, val_labels, test_labels, cfg, pre_sample_dir, timestamp, "s2_manual")

    # get event directories from the training labels for mean and std calculation
    p = re.compile('label_(\d{8})_(.+).tif')
    train_events = [(p.search(label).group(1), p.search(label).group(2)) for label in train_labels]

    # calculate mean and std of train tiles (only for DEM and slopes)
    logger.info('Calculating mean and std of DEM and slope channels...')
    mean_dem_slopes = trainMean(train_events, sample_dirs_list, cfg)
    std_dem_slopes = trainStd(train_events, mean_dem_slopes, sample_dirs_list, cfg)
    logger.info('Mean and std of DEM and slopes calculated.')

    # Construct final mean and std arrays for 16 input channels:
    # Channels 0-9: RGB(3), B08(1), SWIR1(1), SWIR2(1), NDWI(1), MNDWI(1), AWEI_sh(1), AWEI_nsh(1)
    #               All scaled/normalized → mean=0, std=1
    # Channels 10-12: DEM, slope_y, slope_x → computed mean/std
    # Channels 13-15: waterbody, roads, flowlines → mean=0, std=1
    mean_spectral = np.zeros(10, dtype=np.float32)  # RGB + NIR + SWIR1 + SWIR2 + spectral indices
    std_spectral = np.ones(10, dtype=np.float32)
    mean_binary = np.zeros(3, dtype=np.float32)  # waterbody, roads, flowlines
    std_binary = np.ones(3, dtype=np.float32)
    
    mean = np.concatenate([mean_spectral, mean_dem_slopes, mean_binary])
    std = np.concatenate([std_spectral, std_dem_slopes, std_binary])

    # also store training mean std statistics in file
    stats_file = pre_sample_dir / f'mean_std.pkl'
    with open(stats_file, 'wb') as f:
        pickle.dump((mean, std), f)
    logger.info('Training mean and std statistics saved.')
    
    if cfg.preprocess.method == 'random':
        # Process each split using parallel sampling
        for split_name, labels in [('train', train_labels), ('val', val_labels), ('test', test_labels)]:
            if len(labels) == 0:
                logger.warning(f'No labels for {split_name} split, skipping...')
                continue
            
            output_file = pre_sample_dir / f'{split_name}_patches.npy'
            logger.info(f'Processing {split_name} split with {len(labels)} labels...')
            
            sample_patches_parallel(
                labels, cfg.preprocess.size, cfg.preprocess.samples, output_file,
                sample_dirs_list, cfg, cfg.preprocess.seed, n_workers
            )
        
        logger.info('Parallel patch sampling complete.')
    elif cfg.preprocess.method == 'strided':
        # Process each split using parallel sampling
        for split_name, labels in [('train', train_labels), ('val', val_labels), ('test', test_labels)]:
            if len(labels) == 0:
                logger.warning(f'No labels for {split_name} split, skipping...')
                continue
            
            output_file = pre_sample_dir / f'{split_name}_patches.npy'
            logger.info(f'Processing {split_name} split with {len(labels)} labels...')
            
            sample_patches_parallel_strided(
                labels, cfg.preprocess.size, cfg.preprocess.stride, output_file,
                sample_dirs_list, cfg, n_workers
            )
        
        logger.info('Parallel patch sampling complete.')
    else:
        raise ValueError(f"Unsupported sampling method: {cfg.preprocess.method}")

    logger.info('Preprocessing complete.')

if __name__ == '__main__':
    main()