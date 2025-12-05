from glob import glob
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
import multiprocessing as mp
import psutil
import shutil
from numpy.lib.format import open_memmap
from datetime import datetime
import hydra
from omegaconf import DictConfig
import yaml
from floodmaps.utils.preprocess_utils import WelfordAccumulator

SAR_CHANNELS_TO_NORMALIZE = 5

def load_tile_for_stats(tile_info: Tuple[Path, Path, Path, str, str], filter_type: str) -> Tuple[np.ndarray, np.ndarray]:
    """Load the tile data and return the array and mask.
    
    Args:
        tile_info: Tuple of (event_path, sar_vv_file, label_file, eid, img_dt)
        filter_type: 'raw' or 'lee' for SAR filtering
        
    Returns:
        Tuple of (arr, mask) for the 5 non-binary channels
    """
    event, sar_vv_file, label_file, eid, img_dt = tile_info
    
    # Load SAR data
    sar_vh_file = sar_vv_file.with_name(sar_vv_file.name.replace("_vv.tif", "_vh.tif"))
    
    with rasterio.open(sar_vv_file) as src:
        vv_raster = src.read()
    with rasterio.open(sar_vh_file) as src:
        vh_raster = src.read()
    
    # Apply speckle filter if requested
    if filter_type == "lee":
        vv_raster = np.expand_dims(
            enhanced_lee_filter(vv_raster[0], kernel_size=5).astype(np.float32), axis=0
        )
        vh_raster = np.expand_dims(
            enhanced_lee_filter(vh_raster[0], kernel_size=5).astype(np.float32), axis=0
        )
    
    # Load ancillary data
    dem_file = event / f'dem_{eid}.tif'
    cloud_file = event / f'clouds_{img_dt}_{eid}.tif'
    
    with rasterio.open(dem_file) as src:
        dem_raster = src.read()
    with rasterio.open(cloud_file) as src:
        cloud_raster = src.read()
    
    # Calculate slopes
    slope = np.gradient(dem_raster, axis=(1, 2))
    slope_y_raster, slope_x_raster = slope
    
    # Stack the 5 non-binary channels
    stack = np.vstack((
        vv_raster, vh_raster, dem_raster, 
        slope_y_raster, slope_x_raster
    )).astype(np.float32)
    
    # Create mask for valid pixels
    mask = ((vv_raster[0] != -9999) & 
            (vh_raster[0] != -9999) & 
            (cloud_raster[0] != 1))
    
    return stack, mask


def process_tiles_batch_for_stats(tiles_batch: List[Tuple], filter_type: str) -> WelfordAccumulator:
    """Process a batch of tiles assigned to one worker using NumPy + Welford merging.
    
    Args:
        tiles_batch: List of tile tuples assigned to this worker
        filter_type: 'raw' or 'lee' for SAR filtering
        
    Returns:
        WelfordAccumulator with accumulated statistics from all tiles in batch
    """
    accumulator = WelfordAccumulator(SAR_CHANNELS_TO_NORMALIZE)  # 5 non-binary channels for SAR
    
    for tile_info in tiles_batch:
        try:
            arr, mask = load_tile_for_stats(tile_info, filter_type)
            accumulator.update(arr, mask)
        except Exception as e:
            # Add context about which tile failed
            _, _, _, eid, img_dt = tile_info
            raise RuntimeError(f"Worker failed processing tile (eid: {eid}, dt: {img_dt}): {e}") from e
    
    return accumulator


def compute_statistics_parallel(train_tiles: List[Tuple], filter_type: str, n_workers: int = None) -> Tuple[np.ndarray, np.ndarray]:
    """Compute mean and std using optimized parallel Welford's algorithm.
    
    This implementation:
    1. Distributes tiles in batches to workers for better load balancing
    2. Uses NumPy for efficient tile-level statistics computation
    3. Uses Welford's algorithm to merge tile statistics within each worker
    4. Performs final merge of worker accumulators in main process
    
    Parameters
    ----------
    train_tiles : List[Tuple]
        List of tile tuples for training set
    filter_type : str
        'raw' or 'lee' for SAR filtering
    n_workers : int, optional
        Number of worker processes (defaults to CPU count)
        
    Returns
    -------
    mean : np.ndarray
        Mean of the 5 non-binary channels
    std : np.ndarray
        Standard deviation of the 5 non-binary channels
    """
    logger = logging.getLogger('preprocessing')
    
    if n_workers is None:
        n_workers = 1
    
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
        with mp.Pool(n_workers) as pool:
            worker_accumulators = pool.starmap(
                process_tiles_batch_for_stats,
                [(batch, filter_type) for batch in tile_batches]
            )
    except Exception as e:
        logger.error(f"Failed during parallel statistics computation: {e}")
        logger.error("One or more worker processes failed during statistics calculation")
        raise RuntimeError(f"Statistics computation failed: {e}") from e
    
    # Merge worker accumulators (much fewer merge operations)
    final_accumulator = WelfordAccumulator(5)
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
        Tuple of (event_path, sar_vv_file, label_file, eid, img_dt)
        
    Returns
    -------
    stacked_raster : np.ndarray
        Stacked raster of the tile
    missing_clouds_raster : np.ndarray
        Missing mask and cloud mask of the tile
    """
    event_path, sar_vv_file, label_file, eid, img_dt = tile_info
    sar_vh_file = sar_vv_file.with_name(sar_vv_file.name.replace("_vv.tif", "_vh.tif"))
    tci_file = event_path / f'tci_{img_dt}_{eid}.tif'
    dem_file = event_path / f'dem_{eid}.tif'
    waterbody_file = event_path / f'waterbody_{eid}.tif'
    roads_file = event_path / f'roads_{eid}.tif'
    flowlines_file = event_path / f'flowlines_{eid}.tif'
    cloud_file = event_path / f'clouds_{img_dt}_{eid}.tif'
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

    with rasterio.open(sar_vv_file) as src:
        vv_raster = src.read()

    with rasterio.open(sar_vh_file) as src:
        vh_raster = src.read()

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
    cloud_raster = cloud_raster.astype(np.uint8)
    missing_clouds_raster = np.vstack((missing_raster, cloud_raster), dtype=np.uint8)
    
    return stacked_raster, missing_clouds_raster


def sample_patches_in_mem(tiles: List[Tuple], size: int, num_samples: int, cloud_threshold: float,
                          filter_type: str, seed: int, max_attempts: int = 20000) -> np.ndarray:
    """Sample patches and stores them in memory.
    
    Parameters
    ----------
    tiles : List[Tuple]
        List of tile tuples to process
    size : int
        Patch size (e.g., 64)
    num_samples : int
        Number of patches to sample per tile
    cloud_threshold : float
        Maximum cloud fraction for patch acceptance
    filter_type : str
        'raw' or 'lee' for SAR filtering
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
    total_patches = num_samples * len(tiles)
    dataset = np.empty((total_patches, 13, size, size), dtype=np.float32)
    
    for i, tile in enumerate(tiles):
        try:
            tile_data, tile_mask = load_tile_for_sampling(tile)
        except Exception as e:
            event_path, _, _, eid, img_dt = tile
            raise RuntimeError(f"Worker failed to load tile {i} (event_path: {event_path}, eid: {eid}, dt: {img_dt}): {e}") from e
        
        patches_sampled = 0
        _, HEIGHT, WIDTH = tile_data.shape
        attempts = 0
        while patches_sampled < num_samples and attempts < max_attempts:
            attempts += 1
            x = int(rng.uniform(0, HEIGHT - size))
            y = int(rng.uniform(0, WIDTH - size))
            patch = tile_data[:, x : x + size, y : y + size]
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
        
        if patches_sampled < num_samples:
            event_path, _, _, eid, img_dt = tile
            raise RuntimeError(f"Worker exceeded max sampling attempts {max_attempts}, only sampled {patches_sampled}/{num_samples} patches for tile {i} (event_path: {event_path}, eid: {eid}, dt: {img_dt})")

    return dataset


def sample_patches_in_disk(tiles: List[Tuple], size: int, num_samples: int, cloud_threshold: float,
                          seed: int, scratch_dir: Path, worker_id: int, max_attempts: int = 20000) -> Path:
    """Sample patches and stores them in memory mapped file on disk.
    
    Parameters
    ----------
    tiles : List[Tuple]
        List of tile tuples to process
    size : int
        Patch size (e.g., 64)
    num_samples : int
        Number of patches to sample per tile
    cloud_threshold : float
        Maximum cloud fraction for patch acceptance
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
    total_patches = num_samples * len(tiles)
    tmp_file = scratch_dir / f"tmp_{worker_id}.dat"
    dataset = np.memmap(tmp_file, dtype=np.float32, shape=(total_patches, 13, size, size), mode="w+")
    
    try:
        for i, tile in enumerate(tiles):
            try:
                tile_data, tile_mask = load_tile_for_sampling(tile)
            except Exception as e:
                event_path, _, _, eid, img_dt = tile
                raise RuntimeError(f"Worker {worker_id} failed to load tile {i} (event_path: {event_path}, eid: {eid}, dt: {img_dt}): {e}") from e
            
            patches_sampled = 0
            _, HEIGHT, WIDTH = tile_data.shape
            attempts = 0
            while patches_sampled < num_samples and attempts < max_attempts:
                attempts += 1
                x = int(rng.uniform(0, HEIGHT - size))
                y = int(rng.uniform(0, WIDTH - size))
                patch = tile_data[:, x : x + size, y : y + size]
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

            if patches_sampled < num_samples:
                event_path, _, _, eid, img_dt = tile
                raise RuntimeError(f"Worker {worker_id} exceeded max sampling attempts {max_attempts}, only sampled {patches_sampled}/{num_samples} patches for tile {i} (event_path: {event_path}, eid: {eid}, dt: {img_dt})")
    finally:
        # Ensure cleanup even if an error occurs
        dataset.flush()
        del dataset

    return tmp_file


def sample_patches_parallel(tiles: List[Tuple], size: int, num_samples: int, 
                          output_file: Path, cloud_threshold: float, 
                          filter_type: str, seed: int, n_workers: int = None) -> None:
    """Sample patches in parallel. Strategy will depend on the memory available. If
    memory is comfortably below the total node memory (on improv you get ~230GB regardless
    of how many cpus requested), then we can just have each worker fill out their own
    array and then concatenate them in the parent.

    If memory required for entire array is too close or exceeds total node memory,
    then each worker will write to a memory mapped array and then combine them at the end.
    
    Parameters
    ----------
    tiles : List[Tuple]
        List of tile tuples to process
    size : int
        Patch size (e.g., 64)
    num_samples : int
        Number of patches to sample per tile
    output_file : Path
        Path to save the output .npy file
    cloud_threshold : float
        Maximum cloud fraction for patch acceptance
    filter_type : str
        'raw' or 'lee' for SAR filtering
    seed : int
        Random seed for reproducibility
    n_workers : int, optional
        Number of worker processes (defaults to 1)
    """
    logger = logging.getLogger('preprocessing')
    
    if n_workers is None:
        n_workers = 1
    
    # Ensure we don't have more workers than tiles
    logger.info(f'Specified {n_workers} workers for patch sampling.')
    n_workers = min(n_workers, len(tiles))
    logger.info(f'Using {n_workers} workers for {len(tiles)} tiles...')

    total_patches = len(tiles) * num_samples
    array_shape = (total_patches, 13, size, size)
    array_dtype = np.float32

    # calculate how much memory required for the entire array (factor of 2 for concatenation operation)
    total_mem_required = array_shape[0] * array_shape[1] * array_shape[2] * array_shape[3] * np.dtype(array_dtype).itemsize * 2
    total_mem_available = psutil.virtual_memory().available
    logger.info(f'Total memory required for the entire array x2: {total_mem_required / 1024**3:.2f} GB')
    logger.info(f'Total memory available: {total_mem_available / 1024**3:.2f} GB')
    
    # divide up the tiles into n_workers chunks
    worker_tiles = []
    batch_sizes = []
    start_idx = 0
    tiles_per_worker = len(tiles) // n_workers
    tiles_remainder = len(tiles) % n_workers
    for worker_id in range(n_workers):
        worker_tile_count = tiles_per_worker + (1 if worker_id < tiles_remainder else 0)
        end_idx = start_idx + worker_tile_count
        tiles_chunk = tiles[start_idx:end_idx]
        worker_tiles.append(tiles_chunk)
        batch_sizes.append(worker_tile_count)
        start_idx += worker_tile_count

    # Log batch distribution
    logger.info(f'Tile distribution per worker: {batch_sizes}')
        
    if total_mem_required < total_mem_available * 0.9:
        # use simple concatenation strategy
        logger.info('Total memory required is below available memory, using in-memory arrays and concatenation.')

        try:
            with mp.Pool(n_workers) as pool:
                results = pool.starmap(sample_patches_in_mem, [(tile, size, num_samples, cloud_threshold, seed+i*10000) for i, tile in enumerate(worker_tiles)])
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
                worker_files = pool.starmap(sample_patches_in_disk, [(tile, size, num_samples, cloud_threshold, seed+i*10000, scratch_dir, i) for i, tile in enumerate(worker_tiles)])
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
                tiles_in_worker = len(worker_tiles[i])
                chunk_shape = (num_samples * tiles_in_worker, 13, size, size)
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
            event_path, _, _, eid, img_dt = tile_info
            raise RuntimeError(f"Worker failed to load tile {i} (event_path: {event_path}, eid: {eid}, dt: {img_dt}): {e}") from e
        
        _, HEIGHT, WIDTH = tile_data.shape
        
        # Check if tile is large enough for patch size
        if HEIGHT < size or WIDTH < size:
            event_path, _, _, eid, img_dt = tile_info
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
                
                # Filter out missing or high cloud percentage patches
                if patch_contains_missing_or_cloud(patch):
                    continue
                
                all_patches.append(patch[:22])
    
    if len(all_patches) == 0:
        raise RuntimeError("No valid patches found after filtering")
    
    return np.array(all_patches, dtype=np.float32)

def sample_patches_parallel_strided(tile_infos: List[Tuple], size: int, stride: int, 
                          output_file: Path, sample_dirs: List[str], cfg: DictConfig,
                          n_workers: int = None) -> None:
    """Sample patches in parallel with strided sliding window sampling. Each worker samples
    patches strided, and results are concatenated.

    NOTE: No memory mapping strategy for strided sampling.
    
    Parameters
    ----------
    tile_infos : List[Tuple]
        List of tile info tuples to process
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
    n_workers = min(n_workers, len(tile_infos))
    logger.info(f'Using {n_workers} workers for {len(tile_infos)} tiles...')
    
    # divide up the labels into n_workers chunks
    worker_tile_infos = []
    batch_sizes = []
    start_idx = 0
    tiles_per_worker = len(tile_infos) // n_workers
    tiles_remainder = len(tile_infos) % n_workers
    for worker_id in range(n_workers):
        worker_tile_count = tiles_per_worker + (1 if worker_id < tiles_remainder else 0)
        end_idx = start_idx + worker_tile_count
        tiles_chunk = tile_infos[start_idx:end_idx]
        worker_tile_infos.append(tiles_chunk)
        batch_sizes.append(worker_tile_count)
        start_idx += worker_tile_count
        
    # Log batch distribution
    logger.info(f'Label distribution per worker: {batch_sizes}')
    
    logger.info('Strided sampling uses in-memory arrays and concatenation.')

    try:
        with mp.Pool(n_workers) as pool:
            results = pool.starmap(sample_patches_strided, [(tile_infos, size, stride) for tile_infos in worker_tile_infos])
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


def save_event_splits(train_events: List[Path], val_events: List[Path], test_events: List[Path],
                      cfg: DictConfig, output_dir: Path, timestamp: str, data_type: str = "sar") -> None:
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
            'split_json': getattr(cfg.preprocess, 'split_json', None),
            'val_ratio': getattr(cfg.preprocess, 'val_ratio', None),
            'test_ratio': getattr(cfg.preprocess, 'test_ratio', None),
            'train_count': len(train_events),
            'val_count': len(val_events),
            'test_count': len(test_events),
            'timestamp': timestamp
        },
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
    """Get S1 SAR VV tile info from event directories. Tile info is grouped together as tuple
    in the format (event_path: Path, sar_vv_file: Path, label_file: Path, eid: str, s2_img_dt: str, s1_img_dt: str).
    
    Parameters
    ----------
    events: List[Path]
        List of event directory paths
    label_idx: Dict[Tuple[str, str], Path]
        Dictionary mapping (s2_img_dt, eid) to manual label paths
    
    Returns
    -------
    tile_infos: List[Tuple[Path, Path, Path, str, str, str]]
        List of S1 SAR VV tile info in a tuple
        (event_path: Path, sar_vv_file: Path, label_file: Path, eid: str, s2_img_dt: str, s1_img_dt: str)
    """
    logger = logging.getLogger('preprocessing')
    tile_infos = []
    sar_p = re.compile(r'sar_(\d{8})_(\d{8})_(\d{8}_\d+_\d+)_vv.tif')
    for event in events:
        # only add tiles with machine
        for sar_vv_file in event.glob('sar_*_vv.tif'):
            m = sar_p.match(sar_vv_file.name)
            if not m:
                logger.debug(f'Tile {sar_vv_file.name} in {event.name} does not match pattern, skipping...')
                continue
            s2_img_dt = m.group(1)
            s1_img_dt = m.group(2)
            eid = m.group(3)
            # prioritize human labels over machine labels
            label_file = label_idx.get((s2_img_dt, eid), None)
            if label_file is None and (event / f'pred_{s2_img_dt}_{eid}.tif').exists():
                label_file = event / f'pred_{s2_img_dt}_{eid}.tif'

            if label_file is not None:
                tile_info = (event, sar_vv_file, label_file, eid, s2_img_dt, s1_img_dt)
                tile_infos.append(tile_info)
            else:
                logger.debug(f'Tile {sar_vv_file.name} in {event.name} missing machine prediction / human label, skipping...')
                continue
    return tile_infos

@hydra.main(version_base=None, config_path='pkg://configs', config_name='config.yaml')
def main(cfg: DictConfig) -> None:
    """Preprocesses raw S1 tiles and coincident S2 label into smaller patches. The data will be stored
    as separate npy files for train, val, and test sets, along with a mean_std.pkl file containing the
    mean and std of the training tiles.

    Sample directories are s2_s1 directories containing SAR imagery with coincident S2 labels.
    Optional label directories can be also provided to use human labels in place of machine labels (pred_*.tif)
    where possible. If event has associated human label, then the machine label will be replaced by
    the human label.

    cfg.preprocess.split_json storing a dictionary mapping (y, x) prism coordinates to split,
    if provided, allows for pre determined split rather than random split.
    
    cfg.preprocess Parameters:
    - size: int (pixel width of patch)
    - method: str ['random', 'strided']
    - samples: int (number of samples per image for random method)
    - stride: int (for strided method)
    - seed: int (random number generator seed for random method)
    - n_workers: int (number of workers for parallel processing)
    - sample_dirs: List[str] (list of sample directories under cfg.data.imagery_dir)
    - label_dirs: List[str] (list of label directories under cfg.data.imagery_dir)
    - suffix: str (optional suffix to append to preprocessed folder)
    - split_json: str (path to json dictionary mapping (y, x) prism coordinates to train, val, test splits)
    - val_ratio: used for random splitting if no split_json is provided
    - test_ratio: used for random splitting if no split_json is provided
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
    logger.info(f'''Starting S1 manual labeling preprocessing:
        Date:            {timestamp}
        Patch size:      {cfg.preprocess.size}
        Sampling method: {cfg.preprocess.method}
        Samples per tile (Random method): {getattr(cfg.preprocess, 'samples', None)}
        Stride (Strided method): {getattr(cfg.preprocess, 'stride', None)}
        Random seed:     {getattr(cfg.preprocess, 'seed', None)}
        Workers:         {n_workers}
        Sample dir(s):   {cfg.preprocess.s1.sample_dirs}
        Label dir(s):    {cfg.preprocess.s1.label_dirs}
        Suffix:          {getattr(cfg.preprocess, 'suffix', None)}
        Split JSON:      {getattr(cfg.preprocess, 'split_json', None)}
        Val ratio:       {getattr(cfg.preprocess, 'val_ratio', None)}
        Test ratio:      {getattr(cfg.preprocess, 'test_ratio', None)}
    ''')

    # Create preprocessing directory
    sampling_param = cfg.preprocess.samples if cfg.preprocess.method == 'random' else cfg.preprocess.stride
    if getattr(cfg.preprocess, 'suffix', None):
        pre_sample_dir = Path(cfg.paths.preprocess_dir) / 's1_weak' / f'{cfg.preprocess.method}_{cfg.preprocess.size}_{sampling_param}_{cfg.preprocess.suffix}'
    else:
        pre_sample_dir = Path(cfg.paths.preprocess_dir) / 's1_weak' / f'{cfg.preprocess.method}_{cfg.preprocess.size}_{sampling_param}'
    pre_sample_dir.mkdir(parents=True, exist_ok=True)
    logger.info(f'Preprocess directory: {pre_sample_dir.name}')

    # Get directories and split ratios from config
    cfg_s1 = cfg.preprocess.get('s1', {})
    sample_dirs_list = cfg_s1.get('sample_dirs', [])
    label_dirs_list = cfg_s1.get('label_dirs', [])

    if len(sample_dirs_list) == 0:
        raise ValueError('No sample directories were provided.')

    # Build label index for manual labels
    label_idx = build_label_index(label_dirs_list, cfg)

    # Discover events with required SAR assets
    all_events: List[Path] = []
    total_tiles = 0
    seen_eids = set()
    sar_p = re.compile(r'sar_(\d{8})_(\d{8})_\d{8}_\d+_\d+_vv.tif')
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
            
            # Require all S1 tiles have either a machine prediction / human label
            if any(event_dir.glob('sar_*_vv.tif')):
                tiles_labeled = 0
                for sar_vv_file in event_dir.glob('sar_*_vv.tif'):
                    m = sar_p.match(sar_vv_file.name)
                    if not m:
                        continue
                    s2_img_dt = m.group(1)
                    if (event_dir / f'pred_{s2_img_dt}_{eid}.tif').exists() or (s2_img_dt, eid) in label_idx:
                        tiles_labeled += 1

                if tiles_labeled > 0:
                    all_events.append(event_dir)
                    seen_eids.add(eid)
                    total_tiles += tiles_labeled
                else:
                    logger.debug(f'Event {eid} in folder {event_dir.parent} missing labeled SAR tiles, skipping...')
            else:
                logger.debug(f'Event {eid} in folder {event_dir.parent} missing SAR tile, skipping...')

    logger.info(f'Found {total_tiles} labeled SAR tiles (images) across {len(all_events)} events for preprocessing')

    # Split events into train/val/test
    split_json = getattr(cfg.preprocess, 'split_json', None)
    if split_json is not None:
        logger.info(f'Split provided by json file: {split_json}')
        train_events = []
        val_events = []
        test_events = []
        p = re.compile(r'\d{8}_(\d+)_(\d+)')
        coord_to_split = {}
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
        logger.info(f'No json file provided, performing random splitting with val_ratio={val_ratio} and test_ratio={test_ratio}...')
        holdout_ratio = val_ratio + test_ratio
        if holdout_ratio <= 0 or holdout_ratio >= 1:
            raise ValueError('Sum of val_ratio and test_ratio must be in (0, 1).')

        train_events, val_test_events = train_test_split(
            all_events, test_size=holdout_ratio, random_state=cfg.preprocess.seed
        )
        test_prop_within_holdout = test_ratio / holdout_ratio
        val_events, test_events = train_test_split(
            val_test_events, test_size=test_prop_within_holdout, random_state=cfg.preprocess.seed + 1222
        )

    logger.info(f'Split: {len(train_events)} train, {len(val_events)} val, {len(test_events)} test events')
    
    # Save the event splits for reproducibility and reference
    save_event_splits(train_events, val_events, test_events, cfg, pre_sample_dir, timestamp, "sar")

    # Get list of tiles for splits as events can have multiple valid tiles
    logger.info('Grabbing labeled SAR tiles for splits...')
    train_tile_infos = get_tile_infos_from_events(train_events, label_idx=label_idx)
    val_tile_infos = get_tile_infos_from_events(val_events, label_idx=label_idx)
    test_tile_infos = get_tile_infos_from_events(test_events, label_idx=label_idx)
    
    logger.info(f'Tiles: {len(train_tile_infos)} train, {len(val_tile_infos)} val, {len(test_tile_infos)} test')

    # Compute statistics using parallel Welford's algorithm
    logger.info('Computing training statistics for SAR VV, VH, DEM and slope channels...')
    mean_cont, std_cont = compute_statistics_parallel(train_tile_infos, n_workers)
    
    # Add binary channel statistics (mean=0, std=1)
    mean_binary = np.zeros(3, dtype=np.float32)  # waterbody, roads, flowlines
    std_binary = np.ones(3, dtype=np.float32)
    mean = np.concatenate([mean_cont, mean_binary])
    std = np.concatenate([std_cont, std_binary])
    
    # also store training mean std statistics in file
    stats_file = pre_sample_dir / f'mean_std.pkl'
    with open(stats_file, 'wb') as f:
        pickle.dump((mean, std), f)
    logger.info(f'Training mean std saved to {stats_file}')

    # Sample patches in parallel
    if cfg.preprocess.method == 'random':
        # Process each split using parallel sampling
        for split_name, tile_infos in [('train', train_tile_infos), ('val', val_tile_infos), ('test', test_tile_infos)]:
            if len(tile_infos) == 0:
                logger.warning(f'No labeled tiles for {split_name} split, skipping...')
                continue
            
            output_file = pre_sample_dir / f'{split_name}_patches.npy'
            logger.info(f'Processing {split_name} split with {len(tile_infos)} tiles...')
            
            sample_patches_parallel(
                tile_infos, cfg.preprocess.size, cfg.preprocess.samples, output_file,
                sample_dirs_list, cfg, cfg.preprocess.seed, n_workers
            )
        
        logger.info('Parallel patch sampling complete.')
    elif cfg.preprocess.method == 'strided':
        # Process each split using parallel sampling
        for split_name, tile_infos in [('train', train_tile_infos), ('val', val_tile_infos), ('test', test_tile_infos)]:
            if len(tile_infos) == 0:
                logger.warning(f'No labeled tiles for {split_name} split, skipping...')
                continue
            
            output_file = pre_sample_dir / f'{split_name}_patches.npy'
            logger.info(f'Processing {split_name} split with {len(tile_infos)} tiles...')
            
            sample_patches_parallel_strided(
                tile_infos, cfg.preprocess.size, cfg.preprocess.stride, output_file,
                sample_dirs_list, cfg, n_workers
            )
        
        logger.info('Parallel patch sampling complete.')
    else:
        raise ValueError(f"Unsupported sampling method: {cfg.preprocess.method}")

    logger.info('Preprocessing complete.')

if __name__ == '__main__':
    main()