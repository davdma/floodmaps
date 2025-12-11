import rasterio
import numpy as np
from pathlib import Path
import re
import sys
from random import Random
import logging
import pickle
from sklearn.model_selection import train_test_split
from typing import List, Optional, Tuple, Dict
from datetime import datetime
import hydra
from omegaconf import DictConfig
import yaml
import csv
import concurrent.futures
import shutil

from floodmaps.utils.preprocess_utils import (
    WelfordAccumulator,
    PROCESSING_BASELINE_NAIVE,
    BOA_ADD_OFFSET,
    compute_awei_sh,
    compute_awei_nsh,
    compute_ndwi,
    compute_mndwi
)

# 3 channels: DEM, slope_y, slope_x
S2_DATASET_CHANNELS = 22
S2_CHANNELS_TO_NORMALIZE = 3
S2_RGB_MISSING_VALUE = 0
NDWI_MISSING_VALUE = -999999

def load_tile_for_stats(tile_info: Tuple[Path, Path, Path, str, str]) -> Tuple[np.ndarray, np.ndarray]:
    """Load the tile data and return the array and mask for DEM and slopes only.
    
    RGB/NIR are scaled by 10000, NDWI is already normalized, so we only compute
    statistics for DEM and slopes (3 channels).
    
    Parameters
    ----------
    tile_info : Tuple[Path, Path, Path, str, str]
        Tuple of (event_path, rgb_file, label_file, eid, img_dt)
        
    Returns
    -------
        Tuple of (arr, mask) for the 3 channels (DEM, slope_y, slope_x)
    """
    event, rgb_file, label_file, eid, s2_img_dt = tile_info
    
    # Load S2 data for masking
    ndwi_file = event / f'ndwi_{s2_img_dt}_{eid}.tif'
    
    with rasterio.open(rgb_file) as src:
        rgb_raster = src.read()
    with rasterio.open(ndwi_file) as src:
        ndwi_raster = src.read()
    
    # Load ancillary data
    dem_file = event / f'dem_{eid}.tif'
    
    with rasterio.open(dem_file) as src:
        dem_raster = src.read()
    
    # Calculate slopes
    slope = np.gradient(dem_raster, axis=(1, 2))
    slope_y_raster, slope_x_raster = slope
    
    # Stack only DEM and slopes for statistics computation
    stack = np.vstack((dem_raster, slope_y_raster, slope_x_raster)).astype(np.float32)
    
    # Create mask for valid pixels (based on S2 criteria)
    mask = ((rgb_raster[0] != S2_RGB_MISSING_VALUE) & (ndwi_raster[0] != NDWI_MISSING_VALUE))
    
    return stack, mask


def process_tiles_batch_for_stats(tiles_batch: List[Tuple]) -> WelfordAccumulator:
    """Process a batch of tiles assigned to one worker using NumPy + Welford merging.
    
    Args:
        tiles_batch: List of tile tuples assigned to this worker
        
    Returns:
        WelfordAccumulator with accumulated statistics from all tiles in batch
    """
    accumulator = WelfordAccumulator(S2_CHANNELS_TO_NORMALIZE)  # 3 channels: DEM, slope_y, slope_x
    
    for tile_info in tiles_batch:
        try:
            arr, mask = load_tile_for_stats(tile_info)
            accumulator.update(arr, mask)
        except Exception as e:
            # Add context about which tile failed
            _, _, _, eid, s2_img_dt = tile_info
            raise RuntimeError(f"Worker failed processing tile (eid: {eid}, dt: {s2_img_dt}): {e}") from e
    
    return accumulator


def compute_statistics_parallel(train_tiles: List[Tuple], n_workers: int = None) -> Tuple[np.ndarray, np.ndarray]:
    """Compute mean and std using optimized parallel Welford's algorithm.
    
    RGB/NIR are scaled by 10000, NDWI is already normalized, so we only compute
    statistics for DEM and slopes (3 channels).
    
    This implementation:
    1. Distributes tiles in batches to workers for better load balancing
    2. Uses NumPy for efficient tile-level statistics computation
    3. Uses Welford's algorithm to merge tile statistics within each worker
    4. Performs final merge of worker accumulators in main process
    
    Parameters
    ----------
    train_tiles : List[Tuple]
        List of tile tuples for training set
    n_workers : int, optional
        Number of worker processes (defaults to CPU count)
        
    Returns
    -------
    mean : np.ndarray
        Mean of the 3 channels (DEM, slope_y, slope_x)
    std : np.ndarray
        Standard deviation of the 3 channels (DEM, slope_y, slope_x)
    """
    logger = logging.getLogger('preprocessing')
    
    if n_workers is None:
        n_workers = 1
    
    # Handle empty tiles case
    if len(train_tiles) == 0:
        raise ValueError('No training tiles provided for statistics computation')
    
    # Ensure we don't have more workers than tiles
    logger.info(f'Specified {n_workers} workers for statistics computation.')
    n_workers = min(n_workers, len(train_tiles))
    logger.info(f'Using {n_workers} workers for {len(train_tiles)} tiles...')
    
    # Split tiles into balanced batches for workers
    tiles_per_worker = len(train_tiles) // n_workers
    remainder = len(train_tiles) % n_workers
    
    tile_batches = []
    start_idx = 0
    
    for i in range(n_workers):
        # Some workers get one extra tile if there's a remainder
        batch_size = tiles_per_worker + (1 if i < remainder else 0)
        end_idx = start_idx + batch_size
        
        if start_idx < len(train_tiles):
            tile_batches.append(train_tiles[start_idx:end_idx])
        
        start_idx = end_idx
    
    # Log batch distribution
    batch_sizes = [len(batch) for batch in tile_batches]
    logger.info(f'Tile distribution per worker: {batch_sizes}')
    
    # Process tile batches in parallel
    try:
        with concurrent.futures.ProcessPoolExecutor(max_workers=n_workers) as executor:
            futures = [executor.submit(process_tiles_batch_for_stats, batch) for batch in tile_batches]
            worker_accumulators = [future.result() for future in futures]
    except Exception as e:
        logger.error(f"Failed during parallel statistics computation: {e}")
        logger.error("One or more worker processes failed during statistics calculation")
        raise RuntimeError(f"Statistics computation failed: {e}") from e
    
    # Merge worker accumulators (much fewer merge operations)
    final_accumulator = WelfordAccumulator(S2_CHANNELS_TO_NORMALIZE)
    total_pixels = 0
    
    for worker_acc in worker_accumulators:
        final_accumulator.merge(worker_acc)
        total_pixels += worker_acc.count
    
    mean, std = final_accumulator.finalize()
    logger.info(f'Statistics computed from {total_pixels} pixels across {len(train_tiles)} tiles')
    logger.info(f'Final statistics - Mean: {mean}, Std: {std}')
    
    return mean, std


def load_tile_for_sampling(tile_info: Tuple):
    """Load a tile and return the stacked raster for patch sampling.
    
    Parameters
    ----------
    tile_info : Tuple
        Tuple of (event_path, rgb_file, label_file, eid, img_dt)
        
    Returns
    -------
    stacked_raster : np.ndarray
        Stacked raster of the tile (23 channels: 22 data + 1 missing mask)
    """
    event_path, rgb_file, label_file, eid, img_dt = tile_info
    tci_file = event_path / f'tci_{img_dt}_{eid}.tif'
    b08_file = event_path / f'b08_{img_dt}_{eid}.tif'
    b11_file = event_path / f'b11_{img_dt}_{eid}.tif'
    b12_file = event_path / f'b12_{img_dt}_{eid}.tif'
    scl_file = event_path / f'scl_{img_dt}_{eid}.tif'
    dem_file = event_path / f'dem_{eid}.tif'
    waterbody_file = event_path / f'waterbody_{eid}.tif'
    roads_file = event_path / f'roads_{eid}.tif'
    flowlines_file = event_path / f'flowlines_{eid}.tif'
    nlcd_file = event_path / f'nlcd_{eid}.tif'

    with rasterio.open(label_file) as src: 
        label_raster = src.read([1, 2, 3])
        label_binary = np.where(label_raster[0] != 0, 1, 0)
        label_binary = np.expand_dims(label_binary, axis = 0)
        HEIGHT = src.height
        WIDTH = src.width

    with rasterio.open(tci_file) as src:
        tci_raster = src.read()

    # tci floats
    tci_floats = (tci_raster / 255).astype(np.float32)

    tile_dt_obj = datetime.strptime(img_dt, '%Y%m%d')

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
    
    # Recompute NDWI using surface reflectance: (Green - NIR) / (Green + NIR)
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

    # Compute AWEI_nsh (Automated Water Extraction Index - no shadow): 4*(Green - SWIR1) - (0.25*NIR + 2.75*SWIR2)
    awei_nsh_raster = compute_awei_nsh(rgb_raster_sr[1], b11_raster_sr[0], b08_raster_sr[0], b12_raster_sr[0])
    awei_nsh_raster = np.expand_dims(awei_nsh_raster, axis=0)
    awei_nsh_raster = np.where(missing_mask, -999999, awei_nsh_raster)

    # include water indices in missing mask
    missing_mask = (missing_mask.astype(bool) | (ndwi_raster == -999999) | 
                    (mndwi_raster == -999999) | (awei_sh_raster == -999999) |
                    (awei_nsh_raster == -999999)).astype(np.float32)

    with rasterio.open(dem_file) as src:
        dem_raster = src.read().astype(np.float32)

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
    # New channel order: RGB(0-2), B08(3), SWIR1(4), SWIR2(5), NDWI(6), MNDWI(7), AWEI_sh(8), AWEI_nsh(9),
    #                    DEM(10), slope_y(11), slope_x(12), waterbody(13), roads(14), flowlines(15),
    #                    label(16), TCI(17-19), NLCD(20), SCL(21)
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
    return np.any(patch[22] == 1)


def sample_patches_random(chunk_tile_infos: List[Tuple], size: int, num_samples: int,
        seed: int, save_file: Path, max_attempts: int = 20000) -> None:
    """Sample patches using random uniform sampling. Stops sampling once
    the max number of attempts is reached or the number of patches sampled is reached.
    If the number of patches is less than the requested num_samples, it is still kept
    as a partial array of patches. Output is saved to temporary file specified by save_file.
    
    Parameters
    ----------
    chunk_tile_infos : List[Tuple]
        List of tile info tuples in the chunk to process
    size : int
        Patch size (e.g., 64)
    num_samples : int
        Number of patches to sample per tile
    seed : int
        Random seed for reproducibility
    save_file: Path
        Path to save the output .npy file
    max_attempts : int
        Maximum number of attempts to sample a patch before raising error
    """
    rng = Random(seed)
    all_patches = []
    
    for i, tile_info in enumerate(chunk_tile_infos):
        event_path, _, _, eid, img_dt = tile_info
        try:
            tile_data = load_tile_for_sampling(tile_info)
        except Exception as e:
            raise RuntimeError(f"Worker failed to load tile {i} (event_path: {event_path}, eid: {eid}, dt: {img_dt}): {e}") from e
        
        patches_sampled = 0
        _, HEIGHT, WIDTH = tile_data.shape
        
        # Check if tile is large enough for patch size
        if HEIGHT < size or WIDTH < size:
            raise RuntimeError(f"Tile {i} (event_path: {event_path}, eid: {eid}, dt: {img_dt}) is too small ({HEIGHT}x{WIDTH}) for patch size {size}x{size}")
        
        attempts = 0
        while patches_sampled < num_samples and attempts < max_attempts:
            attempts += 1
            x = int(rng.uniform(0, HEIGHT - size))
            y = int(rng.uniform(0, WIDTH - size))
            patch = tile_data[:, x:x+size, y:y+size]

            # Filter out missing patches using missing mask (channel 22)
            if patch_contains_missing(patch):
                continue

            all_patches.append(patch[:S2_DATASET_CHANNELS])
            patches_sampled += 1
        
    # Always save file, even if empty, to make debugging easier and prevent confusion
    if len(all_patches) > 0:
        patches_array = np.array(all_patches, dtype=np.float32)
    else:
        patches_array = np.empty((0, S2_DATASET_CHANNELS, size, size), dtype=np.float32)
    np.save(save_file, patches_array)


def sample_patches_parallel_random(preprocess_dir: Path, tile_infos: List[Tuple], size: int, num_samples: int, 
                            output_file: Path, seed: int, n_workers: int = None, chunk_size: int = 100, scratch_dir: Path = None) -> None:
    """Sample patches in parallel using the random method. Each chunk is a number of tiles processed by a worker
    to be saved as a temporary npy file. Once each chunk is complete, all of the files
    are combined by streaming them into a single large memory mapped array.

    Ensure that the chunk size is large enough for speed and also small enough to not
    exceed memory limits.
    
    Parameters
    ----------
    preprocess_dir: Path
        Path to the preprocess directory
    tile_infos : List[Tuple]
        List of tile info tuples to process
    size : int
        Patch size (e.g., 64)
    num_samples : int
        Number of patches to sample per tile
    output_file : Path
        Path to save the output .npy file
    seed : int
        Random seed for reproducibility
    n_workers : int, optional
        Number of worker processes (defaults to 1)
    chunk_size : int, optional
        Number of tiles to process per worker before saving as temp file (defaults to 100)
    scratch_dir : Path, optional
        Path to the scratch directory for intermediate files and faster streaming (defaults to None)
    """
    logger = logging.getLogger('preprocessing')
    
    if n_workers is None:
        n_workers = 1
    
    logger.info(f'Specified {n_workers} workers for patch sampling.')
    chunks = (len(tile_infos) + chunk_size - 1) // chunk_size
    n_workers = min(n_workers, chunks)
    logger.info(f'Using {n_workers} workers for {len(tile_infos)} tiles in {chunks} chunks of size {chunk_size}...')

    # first clean up any previous temp files if failed to delete
    chunk_dir = preprocess_dir if scratch_dir is None else scratch_dir
    for tmp_file in chunk_dir.glob("chunk_*.npy"):
        try:
            tmp_file.unlink()
        except Exception as e:
            logger.warning(f"Failed to delete previous temp file {tmp_file}: {e}")
    
    # divide up the tiles into chunks
    chunked_tile_infos = []
    for start_idx in range(0, len(tile_infos), chunk_size):
        tiles_chunk = tile_infos[start_idx:start_idx+chunk_size]
        chunked_tile_infos.append(tiles_chunk)

    try:
        with concurrent.futures.ProcessPoolExecutor(max_workers=n_workers) as executor:
            futures = [executor.submit(sample_patches_random, tiles_chunk, size, num_samples, seed+i*10000, chunk_dir / f'chunk_{i}.npy') for i, tiles_chunk in enumerate(chunked_tile_infos)]
            for future in futures:
                future.result()
    except Exception as e:
        logger.error(f"Failed during parallel random patch sampling: {e}")
        logger.error("One or more worker processes failed during random patch sampling")
        raise RuntimeError(f"Random patch sampling failed: {e}") from e

    # for each chunk file, read in and stream into the final memory mapped array
    # one pass to get the size, another to stream into the final array
    try:
        # Prepare and get the shape of the first chunk to allocate array of correct shape
        chunk_files = sorted(chunk_dir.glob("chunk_*.npy"), key=lambda x: int(x.stem.split("_")[1]))
        if not chunk_files:
            raise RuntimeError("No temporary chunk files found for patch sampling output.")

        # Find output array shape from first file and pre-calculate total_patches
        total_patches = 0
        for tmp_file in chunk_files:
            arr = np.load(tmp_file, mmap_mode='r') # this does not actually read data into memory, just header info
            total_patches += arr.shape[0]
        logger.info(f'Total patches read from {len(chunk_files)} chunks: {total_patches}')

        # Preallocate memmapped output
        memmap_file = scratch_dir / output_file.name if scratch_dir is not None else output_file
        final_arr = np.lib.format.open_memmap(memmap_file, mode="w+", dtype=np.float32, shape=(total_patches, S2_DATASET_CHANNELS, size, size))
        patch_offset = 0
        for tmp_file in chunk_files:
            arr = np.load(tmp_file)
            chunk_shape = arr.shape
            logger.info(f'Streaming chunk {tmp_file.name} of shape {chunk_shape} into final array...')
            n_patches = arr.shape[0]
            final_arr[patch_offset:patch_offset + n_patches, ...] = arr
            patch_offset += n_patches
            arr = None

            try:
                tmp_file.unlink()
            except Exception as e:
                raise RuntimeError(f"Failed to delete chunk file {tmp_file}: {e}") from e
        
        # Now we move the final array from scratch to the final destination
        if scratch_dir is not None:
            shutil.move(memmap_file, output_file)

        logger.info('Sampling complete.')
    except Exception as e:
        logger.exception(f"Failed to stream chunk files or save output.")
        raise RuntimeError(f"Failed to save final array: {e}") from e


def sample_patches_strided(chunk_tile_infos: List[Tuple], size: int, stride: int,
        save_file: Path) -> None:
    """Sample patches using a sliding window with stride and complete edge coverage.
    Output is saved to temporary file specified by save_file.
    
    This function uses a sliding window approach that moves by stride pixels
    in both x and y directions. It ensures complete coverage by explicitly including
    positions that cover the right and bottom edges of each tile.
    
    Parameters
    ----------
    chunk_tile_infos : List[Tuple]
        List of tile info tuples in the chunk to process
    size : int
        Patch size (e.g., 64)
    stride : int
        Stride for sliding window sampling (e.g., 5)
    save_file: Path
        Path to save the output .npy file
    """
    all_patches = []
    
    for i, tile_info in enumerate(chunk_tile_infos):
        event_path, _, _, eid, img_dt = tile_info
        try:
            tile_data = load_tile_for_sampling(tile_info)
        except Exception as e:
            raise RuntimeError(f"Worker failed to load tile {i} (event_path: {event_path}, eid: {eid}, dt: {img_dt}): {e}") from e
        
        _, HEIGHT, WIDTH = tile_data.shape
        
        # Check if tile is large enough for patch size
        if HEIGHT < size or WIDTH < size:
            raise RuntimeError(f"Tile {i} (event_path: {event_path}, eid: {eid}, dt: {img_dt}) is too small ({HEIGHT}x{WIDTH}) for patch size {size}x{size}")
        
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
                
                # Filter out missing patches using missing mask (channel 22)
                if patch_contains_missing(patch):
                    continue
                
                all_patches.append(patch[:S2_DATASET_CHANNELS])
    
    # Always save file, even if empty, to make debugging easier and prevent confusion
    if len(all_patches) > 0:
        patches_array = np.array(all_patches, dtype=np.float32)
    else:
        patches_array = np.empty((0, S2_DATASET_CHANNELS, size, size), dtype=np.float32)
    np.save(save_file, patches_array)

def sample_patches_parallel_strided(preprocess_dir: Path, tile_infos: List[Tuple], size: int, stride: int, 
                            output_file: Path, n_workers: int = None, chunk_size: int = 100, scratch_dir: Path = None) -> None:
    """Sample patches in parallel using the strided method. Each chunk is a number of tiles processed by a worker
    to be saved as a temporary npy file. Once each chunk is complete, all of the files
    are combined by streaming them into a single large memory mapped array.

    Ensure that the chunk size is large enough for speed and also small enough to not
    exceed memory limits.
    
    Parameters
    ----------
    preprocess_dir: Path
        Path to the preprocess directory
    tile_infos : List[Tuple]
        List of tile info tuples to process
    size : int
        Patch size (e.g., 64)
    stride : int
        Stride for sliding window sampling (e.g., 5)
    output_file : Path
        Path to save the output .npy file
    n_workers : int, optional
        Number of worker processes (defaults to 1)
    chunk_size : int, optional
        Number of tiles to process per worker before saving as temp file (defaults to 100)
    scratch_dir : Path, optional
        Path to the scratch directory for intermediate files and faster streaming (defaults to None)
    """
    logger = logging.getLogger('preprocessing')
    
    if n_workers is None:
        n_workers = 1
    
    logger.info(f'Specified {n_workers} workers for patch sampling.')
    chunks = (len(tile_infos) + chunk_size - 1) // chunk_size
    n_workers = min(n_workers, chunks)
    logger.info(f'Using {n_workers} workers for {len(tile_infos)} tiles in {chunks} chunks of size {chunk_size}...')

    # first clean up any previous temp files if failed to delete
    chunk_dir = preprocess_dir if scratch_dir is None else scratch_dir
    for tmp_file in chunk_dir.glob("chunk_*.npy"):
        try:
            tmp_file.unlink()
        except Exception as e:
            logger.warning(f"Failed to delete previous temp file {tmp_file}: {e}")
    
    # divide up the tiles into chunks
    chunked_tile_infos = []
    for start_idx in range(0, len(tile_infos), chunk_size):
        tiles_chunk = tile_infos[start_idx:start_idx+chunk_size]
        chunked_tile_infos.append(tiles_chunk)

    try:
        with concurrent.futures.ProcessPoolExecutor(max_workers=n_workers) as executor:
            futures = [executor.submit(sample_patches_strided, tiles_chunk, size, stride, chunk_dir / f'chunk_{i}.npy') for i, tiles_chunk in enumerate(chunked_tile_infos)]
            for future in futures:
                future.result()
    except Exception as e:
        logger.error(f"Failed during parallel strided patch sampling: {e}")
        logger.error("One or more worker processes failed during strided patch sampling")
        raise RuntimeError(f"Strided patch sampling failed: {e}") from e

    # for each chunk file, read in and stream into the final memory mapped array
    # one pass to get the size, another to stream into the final array
    try:
        # Prepare and get the shape of the first chunk to allocate array of correct shape
        chunk_files = sorted(chunk_dir.glob("chunk_*.npy"), key=lambda x: int(x.stem.split("_")[1]))
        if not chunk_files:
            raise RuntimeError("No temporary chunk files found for patch sampling output.")

        # Find output array shape from first file and pre-calculate total_patches
        total_patches = 0
        for tmp_file in chunk_files:
            arr = np.load(tmp_file, mmap_mode='r') # this does not actually read data into memory, just header info
            total_patches += arr.shape[0]
        logger.info(f'Total patches read from {len(chunk_files)} chunks: {total_patches}')

        memmap_file = scratch_dir / output_file.name if scratch_dir is not None else output_file
        final_arr = np.lib.format.open_memmap(memmap_file, mode="w+", dtype=np.float32, shape=(total_patches, S2_DATASET_CHANNELS, size, size))
        patch_offset = 0
        for tmp_file in chunk_files:
            arr = np.load(tmp_file)
            chunk_shape = arr.shape
            logger.info(f'Streaming chunk {tmp_file.name} of shape {chunk_shape} into final array...')
            n_patches = arr.shape[0]
            final_arr[patch_offset:patch_offset + n_patches, ...] = arr
            patch_offset += n_patches
            arr = None

            try:
                tmp_file.unlink()
            except Exception as e:
                raise RuntimeError(f"Failed to delete chunk file {tmp_file}: {e}") from e
        
        # Now we move the final array from scratch to the final destination
        if scratch_dir is not None:
            shutil.move(memmap_file, output_file)

        logger.info('Sampling complete.')
    except Exception as e:
        logger.exception(f"Failed to stream chunk files or save output.")
        raise RuntimeError(f"Failed to save final array: {e}") from e


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


def get_manual_label_path(label_idx: Dict[Tuple[str, str], Path],
                          img_dt: str, eid: str) -> Optional[Path]:
    """Get the label file path for a given (img_dt, eid) pair."""
    return label_idx.get((img_dt, eid))


def save_event_splits(train_events: List[Path], val_events: List[Path], test_events: List[Path], cfg: DictConfig,
                      output_dir: Path, timestamp: str, data_type: str = "s2") -> None:
    """Save event splits to a YAML file for reproducibility and reference.
    
    Parameters
    ----------
    train_events : List[Path]
        List of training event directory paths
    val_events : List[Path]  
        List of validation event directory paths
    test_events : List[Path]
        List of test event directory paths
    cfg : DictConfig
        Config object
    output_dir : Path
        Directory to save the splits file
    timestamp : str
        Timestamp when preprocessing started
    data_type : str
        Type of data being processed (e.g., "sar", "s2")
    """
    logger = logging.getLogger('preprocessing')
    
    # Create splits file path
    splits_file = output_dir / f'metadata.yaml'
    
    # Prepare splits data structure
    event_splits = {
        'train': sorted([event.name for event in train_events]),
        'val': sorted([event.name for event in val_events]), 
        'test': sorted([event.name for event in test_events]),
        'metadata': {
            'data_type': data_type,
            'method': cfg.preprocess.method,
            'size': cfg.preprocess.size,
            'samples': getattr(cfg.preprocess, 'samples', None),
            'stride': getattr(cfg.preprocess, 'stride', None),
            'seed': getattr(cfg.preprocess, 'seed', None),
            'total_events': len(train_events) + len(val_events) + len(test_events),
            'split_csv': getattr(cfg.preprocess, 'split_csv', None),
            'val_ratio': getattr(cfg.preprocess, 'val_ratio', None),
            'test_ratio': getattr(cfg.preprocess, 'test_ratio', None),
            'train_count': len(train_events),
            'val_count': len(val_events),
            'test_count': len(test_events),
            'timestamp': timestamp
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

def get_tile_infos_from_events(events: List[Path], label_idx: Dict[Tuple[str, str], Path] = {}) -> List[Tuple]:
    """Get S2 rgb tile info from event directories. Tile info is grouped together as tuple
    in the format (event_path: Path, rgb_file: Path, label_file: Path, eid: str, s2_img_dt: str).
    
    Parameters
    ----------
    events: List[Path]
        List of event directory paths
    label_idx: Dict[Tuple[str, str], Path]
        Dictionary mapping (s2_img_dt, eid) to manual label paths
    
    Returns
    -------
    tile_infos: List[Tuple[Path, Path, Path, str, str]]
        List of S2 rgb tile info in a tuple
        (event_path: Path, rgb_file: Path, label_file: Path, eid: str, s2_img_dt: str)
    """
    logger = logging.getLogger('preprocessing')
    tile_infos = []
    s2_p = re.compile(r'rgb_(\d{8})_(\d{8}_\d+_\d+).tif')
    for event in events:
        # only add tiles with machine prediction / human label
        for rgb_file in event.glob('rgb_*.tif'):
            m = s2_p.match(rgb_file.name)
            if not m:
                logger.debug(f'Tile {rgb_file.name} in {event.name} does not match pattern, skipping...')
                continue
            s2_img_dt = m.group(1)
            eid = m.group(2)
            # prioritize human labels over machine labels
            label_file = label_idx.get((s2_img_dt, eid), None)
            if label_file is None and (event / f'pred_{s2_img_dt}_{eid}.tif').exists():
                label_file = event / f'pred_{s2_img_dt}_{eid}.tif'

            if label_file is not None:
                tile_info = (event, rgb_file, label_file, eid, s2_img_dt)
                tile_infos.append(tile_info)
            else:
                logger.debug(f'Tile {rgb_file.name} in {event.name} missing machine prediction / human label, skipping...')
                continue
    return tile_infos


@hydra.main(version_base=None, config_path='pkg://configs', config_name='config.yaml')
def main(cfg: DictConfig) -> None:
    """Preprocesses raw S2 tiles and corresponding labels (machine & human) into smaller patches.
    Patches with any missing values will be filtered out.

    The data will be stored as separate npy files for train, val, and test sets,
    along with a mean_std.pkl file containing the mean and std of the training tiles.

    Sample directories are s2 directories created using the sample_s2 script.
    Optional label directories can be also provided to use human labels in place of machine labels (pred_*.tif)
    where possible. If event has associated human label, then the machine label will be replaced by
    the human label.

    cfg.preprocess.split_csv should be the path to a CSV file with columns "y", "x", "split"
    where each PRISM cell coordinate (y, x) is associated with a split "train", "val", "test".
    If provided, allows for pre determined split rather than random split. This is preferred
    over the random splitting to avoid data leakage from similar dates / regions.

    NOTE: For large datasets on HPC, use the scratch directory for speed.
    
    cfg.preprocess Parameters:
    - size: int (pixel width of patch)
    - method: str ['random', 'strided']
    - samples: int (number of samples per image for random method)
    - stride: int (for strided method)
    - seed: int (random number generator seed for random method)
    - n_workers: int (number of workers for parallel processing)
    - chunk_size: int (number of tiles to process per worker before saving as temp file) (default: 100)
    - s2.sample_dirs: List[str] (list of sample directories under cfg.data.imagery_dir)
    - s2.label_dirs: List[str] (list of label directories under cfg.data.imagery_dir)
    - suffix: str (optional suffix to append to preprocessed folder)
    - split_csv: str (path to CSV file with columns "y", "x", "split" for PRISM cell coordinates)
    - val_ratio: used for random splitting if no split_json is provided
    - test_ratio: used for random splitting if no split_json is provided
    - scratch_dir: str (optional path to the scratch directory for intermediate files and faster streaming)
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
    logger.info(f'''Starting S2 weak labeling preprocessing:
        Date:            {timestamp}
        Patch size:      {cfg.preprocess.size}
        Sampling method: {cfg.preprocess.method}
        Samples per tile (Random method): {getattr(cfg.preprocess, 'samples', None)}
        Stride (Strided method): {getattr(cfg.preprocess, 'stride', None)}
        Random seed:     {getattr(cfg.preprocess, 'seed', None)}
        Workers:         {n_workers}
        Chunk size:      {getattr(cfg.preprocess, 'chunk_size', 100)} (default: 100)
        Sample dir(s):   {cfg.preprocess.s2.sample_dirs}
        Label dir(s):    {cfg.preprocess.s2.label_dirs}
        Suffix:          {getattr(cfg.preprocess, 'suffix', None)}
        Split CSV:       {getattr(cfg.preprocess, 'split_csv', None)}
        Val ratio:       {getattr(cfg.preprocess, 'val_ratio', None)}
        Test ratio:      {getattr(cfg.preprocess, 'test_ratio', None)}
        Scratch dir:     {getattr(cfg.preprocess, 'scratch_dir', None)}
    ''')

    # Create preprocessing directory
    sampling_param = cfg.preprocess.samples if cfg.preprocess.method == 'random' else cfg.preprocess.stride
    if getattr(cfg.preprocess, 'suffix', None):
        pre_sample_dir = Path(cfg.paths.preprocess_dir) / 's2_weak' / f'{cfg.preprocess.method}_{cfg.preprocess.size}_{sampling_param}_{cfg.preprocess.suffix}'
    else:
        pre_sample_dir = Path(cfg.paths.preprocess_dir) / 's2_weak' / f'{cfg.preprocess.method}_{cfg.preprocess.size}_{sampling_param}'
    pre_sample_dir.mkdir(parents=True, exist_ok=True)
    logger.info(f'Preprocess directory: {pre_sample_dir.name}')

    # Get directories and split ratios from config
    cfg_s2 = cfg.preprocess.get('s2', {})
    sample_dirs_list = cfg_s2.get('sample_dirs', [])
    label_dirs_list = cfg_s2.get('label_dirs', [])

    if len(sample_dirs_list) == 0:
        raise ValueError('No sample directories were provided.')

    # Build label index for manual labels
    label_idx = build_label_index(label_dirs_list, cfg)

    # Discover events with required S2 assets
    all_events: List[Path] = []
    total_tiles = 0
    seen_eids = set()
    s2_p = re.compile(r'rgb_(\d{8})_(\d{8}_\d+_\d+).tif')
    
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
            
            # Require all S2 tiles have either a machine prediction / human label
            if any(event_dir.glob('rgb_*.tif')):
                tiles_labeled = 0
                for s2_file in event_dir.glob('rgb_*.tif'):
                    m = s2_p.search(s2_file.name)
                    if not m:
                        continue
                    s2_img_dt = m.group(1)
                    # Check if prediction exists or manual label exists
                    if (event_dir / f'pred_{s2_img_dt}_{eid}.tif').exists() or (s2_img_dt, eid) in label_idx:
                        tiles_labeled += 1

                if tiles_labeled > 0:
                    all_events.append(event_dir)
                    seen_eids.add(eid)
                    total_tiles += tiles_labeled
                else:
                    logger.debug(f'Event {eid} in folder {event_dir.parent} missing labeled S2 tiles, skipping...')
            else:
                logger.debug(f'Event {eid} in folder {event_dir.parent} missing S2 tile, skipping...')

    logger.info(f'Found {total_tiles} labeled S2 tiles (images) across {len(all_events)} events for preprocessing')

    # Split events into train/val/test
    split_csv = getattr(cfg.preprocess, 'split_csv', None)
    if split_csv is not None:
        logger.info(f'Split provided by CSV file: {split_csv}')
        train_events = []
        val_events = []
        test_events = []
        p = re.compile(r'\d{8}_(\d+)_(\d+)')
        
        # Read CSV with columns: y, x, split
        coord_to_split = {}
        with open(split_csv, 'r') as f:
            reader = csv.DictReader(f)
            for row in reader:
                coord = (int(row['y']), int(row['x']))
                coord_to_split[coord] = row['split']

        for event in all_events:
            m = p.match(event.name)
            if m:
                y = int(m.group(1))
                x = int(m.group(2))
                match coord_to_split.get((y, x), None):
                    case 'train':
                        train_events.append(event)
                    case 'val':
                        val_events.append(event)
                    case 'test':
                        test_events.append(event)
                    case _:
                        logger.debug(f'Event {event.name} not assigned to any split, skipping...')
            else:
                logger.debug(f'Event {event.name} does not match pattern, skipping...')
            
    else:
        # random splitting of events
        val_ratio = getattr(cfg.preprocess, 'val_ratio', 0.1)
        test_ratio = getattr(cfg.preprocess, 'test_ratio', 0.1)
        logger.info(f'No split CSV provided, performing random splitting with val_ratio={val_ratio} and test_ratio={test_ratio}...')
        holdout_ratio = val_ratio + test_ratio
        if holdout_ratio <= 0 or holdout_ratio >= 1:
            raise ValueError('Sum of val_ratio and test_ratio must be in (0, 1).')
        
        assert getattr(cfg.preprocess, 'seed', None) is not None, 'cfg.preprocess.seed is required for random splitting.'

        train_events, val_test_events = train_test_split(
            all_events, test_size=holdout_ratio, random_state=cfg.preprocess.seed
        )
        test_prop_within_holdout = test_ratio / holdout_ratio
        val_events, test_events = train_test_split(
            val_test_events, test_size=test_prop_within_holdout, random_state=cfg.preprocess.seed + 1222
        )

    logger.info(f'Split: {len(train_events)} train, {len(val_events)} val, {len(test_events)} test events')
    
    # Save the event splits for reproducibility and reference
    save_event_splits(train_events, val_events, test_events, cfg, pre_sample_dir, timestamp, "s2")

    # Get list of tiles and their info for splits as events can have multiple valid tiles
    logger.info('Grabbing labeled S2 tile info for splits...')
    train_tile_infos = get_tile_infos_from_events(train_events, label_idx=label_idx)
    val_tile_infos = get_tile_infos_from_events(val_events, label_idx=label_idx)
    test_tile_infos = get_tile_infos_from_events(test_events, label_idx=label_idx)
    
    logger.info(f'Tiles: {len(train_tile_infos)} train, {len(val_tile_infos)} val, {len(test_tile_infos)} test')

    # Compute statistics using parallel Welford's algorithm (only for DEM and slopes)
    logger.info('Computing training statistics for DEM and slope channels...')
    mean_dem_slopes, std_dem_slopes = compute_statistics_parallel(train_tile_infos, n_workers)
    
    # Construct final mean and std arrays for 16 input channels (out of 22 total channels):
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
    
    # Save statistics
    stats_file = pre_sample_dir / f'mean_std.pkl'
    with open(stats_file, 'wb') as f:
        pickle.dump((mean, std), f)
    logger.info(f'Training mean std saved to {stats_file}')

    chunk_size = getattr(cfg.preprocess, 'chunk_size', 100)
    scratch_dir = Path(cfg.preprocess.scratch_dir) if getattr(cfg.preprocess, 'scratch_dir', None) is not None else None
    if scratch_dir is not None:
        scratch_dir.mkdir(parents=True, exist_ok=True)
        logger.info(f'Using scratch directory for intermediate files: {scratch_dir}')

    # Sample patches in parallel
    if cfg.preprocess.method == 'random':
        # Process each split using parallel sampling
        for split_name, tile_infos in [('train', train_tile_infos), ('val', val_tile_infos), ('test', test_tile_infos)]:
            if len(tile_infos) == 0:
                logger.warning(f'No labeled tiles for {split_name} split, skipping...')
                continue
            
            output_file = pre_sample_dir / f'{split_name}_patches.npy'
            logger.info(f'Processing {split_name} split with {len(tile_infos)} tiles...')
            
            sample_patches_parallel_random(
                pre_sample_dir, tile_infos, cfg.preprocess.size, cfg.preprocess.samples, 
                output_file, cfg.preprocess.seed, n_workers, chunk_size=chunk_size, scratch_dir=scratch_dir
            )
        
        logger.info('Parallel random patch sampling complete.')
    elif cfg.preprocess.method == 'strided':
        # Process each split using parallel sampling
        for split_name, tile_infos in [('train', train_tile_infos), ('val', val_tile_infos), ('test', test_tile_infos)]:
            if len(tile_infos) == 0:
                logger.warning(f'No labeled tiles for {split_name} split, skipping...')
                continue
            
            output_file = pre_sample_dir / f'{split_name}_patches.npy'
            logger.info(f'Processing {split_name} split with {len(tile_infos)} tiles...')
            
            sample_patches_parallel_strided(
                pre_sample_dir, tile_infos, cfg.preprocess.size, cfg.preprocess.stride,
                output_file, n_workers, chunk_size=chunk_size, scratch_dir=scratch_dir
            )
        
        logger.info('Parallel strided patch sampling complete.')
    else:
        raise ValueError(f"Unsupported sampling method: {cfg.preprocess.method}")

    logger.info('Preprocessing complete.')
    return 0


if __name__ == '__main__':
    main()