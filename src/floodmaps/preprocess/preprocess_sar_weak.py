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

try:
    import yaml
except Exception:
    yaml = None

from floodmaps.utils.utils import enhanced_lee_filter
from floodmaps.utils.preprocess_utils import WelfordAccumulator


def discover_all_tiles(events: List[Path], label_idx: Optional[Dict[Tuple[str, str], Path]] = None, tile_cloud_threshold: float = 0.25) -> List[Tuple]:
    """Discover all individual tiles across a list of events.
    
    Parameters
    ----------
    events: List[Path]
        List of event directory paths
    label_idx: Optional[Dict[Tuple[str, str], Path]]
        Optional dictionary mapping (img_dt, eid) to manual label paths
    tile_cloud_threshold: float
        Threshold for filtering out cloudy tiles (default: 0.25)

    Returns
    -------
    tiles: List[Tuple]
        List of tuples in form (event_path, sar_vv_file, label_file, eid, img_dt)
    """
    logger = logging.getLogger('preprocessing')
    tiles = []
    
    p1 = re.compile(r'\d{8}_\d+_\d+')
    p2 = re.compile(r'pred_(\d{8})_.+\.tif')
    
    for event in events:
        # Extract event ID
        m = p1.search(event.name)
        if not m:
            logger.error(f'No matching eid in {event.name}. Skipping...')
            continue
        eid = m.group(0)
        
        # Find all prediction files and their corresponding SAR files
        for label in event.glob('pred_*.tif'):
            m = p2.search(label.name)
            if not m:
                continue
            img_dt = m.group(1)
            
            # Find corresponding SAR VV file
            sar_vv_files = list(event.glob(f'sar_{img_dt}_*_vv.tif'))
            if len(sar_vv_files) == 0:
                logger.debug(f'SAR VV file not found for {label.name} in {event.name} (event in folder {event.parent})')
                continue
            
            sar_vv_file = sar_vv_files[0]
            sar_vh_file = sar_vv_file.with_name(sar_vv_file.name.replace("_vv.tif", "_vh.tif"))
            
            # Validate that all required files exist
            required_files = [
                sar_vv_file,
                sar_vh_file,
                event / f'tci_{img_dt}_{eid}.tif',
                event / f'dem_{eid}.tif',
                event / f'waterbody_{eid}.tif',
                event / f'roads_{eid}.tif',
                event / f'flowlines_{eid}.tif',
                event / f'clouds_{img_dt}_{eid}.tif',
                event / f'nlcd_{eid}.tif'
            ]
            
            # Check if manual label is available
            final_label = label
            if label_idx:
                manual_label_path = get_manual_label_path(label_idx, img_dt, eid)
                if manual_label_path:
                    logger.debug(f'Found manual label for tile {sar_vv_file.name}: {manual_label_path.name} (label in folder {manual_label_path.parent})')
                    final_label = manual_label_path
                    required_files.append(manual_label_path)
            else:
                required_files.append(label)
            
            # Validate all files exist
            missing_files = [f for f in required_files if not f.exists()]
            if missing_files:
                logger.warning(f'Missing files for tile {sar_vv_file.name} in {event.name}: {[f.name for f in missing_files]} (event in folder {event.parent})')
                continue

            # check if tile is under cloud threshold
            cloud_file = event / f'clouds_{img_dt}_{eid}.tif'
            if not cloud_file.exists():
                logger.warning(f'Cloud file not found for tile {sar_vv_file.name} in {event.name}: {cloud_file.name} (event in folder {event.parent})')
                continue
            tile_cloud_percent = tile_cloud_percentage(cloud_file)
            if tile_cloud_percent > tile_cloud_threshold:
                logger.debug(f'Tile {sar_vv_file.name} in {event.name} is over cloud threshold: {tile_cloud_percent} > {tile_cloud_threshold} (event in folder {event.parent})')
                continue
            
            tiles.append((event, sar_vv_file, final_label, eid, img_dt))
    
    logger.info(f'Discovered {len(tiles)} valid tiles across {len(events)} events')
    return tiles


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
    accumulator = WelfordAccumulator(5)  # 5 non-binary channels for SAR
    
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


def load_tile_for_sampling(tile_info: Tuple, filter_type: str):
    """Load a tile and return the stacked raster for patch sampling.
    
    Parameters
    ----------
    tile_info : Tuple
        Tuple of (event_path, sar_vv_file, label_file, eid, img_dt)
    filter_type : str
        'raw' or 'lee' for SAR filtering
        
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

    # apply speckle filter to sar:
    if filter_type == "lee":
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
            tile_data, tile_mask = load_tile_for_sampling(tile, filter_type)
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
            _, _, _, eid, img_dt = tile
            raise RuntimeError(f"Worker exceeded max sampling attempts {max_attempts}, only sampled {patches_sampled}/{num_samples} patches for tile {i} (eid: {eid}, dt: {img_dt})")

    return dataset


def sample_patches_in_disk(tiles: List[Tuple], size: int, num_samples: int, cloud_threshold: float,
                          filter_type: str, seed: int, scratch_dir: Path, worker_id: int, max_attempts: int = 20000) -> Path:
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
    filter_type : str
        'raw' or 'lee' for SAR filtering
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
                tile_data, tile_mask = load_tile_for_sampling(tile, filter_type)
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
                _, _, _, eid, img_dt = tile
                raise RuntimeError(f"Worker {worker_id} exceeded max sampling attempts {max_attempts}, only sampled {patches_sampled}/{num_samples} patches for tile {i} (eid: {eid}, dt: {img_dt})")
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
                results = pool.starmap(sample_patches_in_mem, [(tile, size, num_samples, cloud_threshold, filter_type, seed+i*10000) for i, tile in enumerate(worker_tiles)])
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
                worker_files = pool.starmap(sample_patches_in_disk, [(tile, size, num_samples, cloud_threshold, filter_type, seed+i*10000, scratch_dir, i) for i, tile in enumerate(worker_tiles)])
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


def build_label_index(label_dirs: list[str], cfg: DictConfig) -> Dict[Tuple[str, str], Path]:
    """Build a dictionary mapping (img_dt, eid) to the label file path."""
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
            img_dt, eid = m.group(1), m.group(2)
            idx.setdefault((img_dt, eid), fp)  # keep first occurrence
    return idx


def get_manual_label_path(label_idx: Dict[Tuple[str, str], Path],
                          img_dt: str, eid: str) -> Optional[Path]:
    """Get the label file path for a given (img_dt, eid) pair."""
    return label_idx.get((img_dt, eid))


def save_event_splits(train_events: List[Path], val_events: List[Path], test_events: List[Path],
                      output_dir: Path, split_seed: int, val_ratio: float, test_ratio: float, 
                      timestamp: str, data_type: str = "sar") -> None:
    """Save event splits to a YAML file for reproducibility and reference.
    
    Parameters
    ----------
    train_events : List[Path]
        List of training event directory paths
    val_events : List[Path]  
        List of validation event directory paths
    test_events : List[Path]
        List of test event directory paths
    output_dir : Path
        Directory to save the splits file
    split_seed : int
        Random seed used for splitting
    val_ratio : float
        Validation set ratio
    test_ratio : float
        Test set ratio
    timestamp : str
        Timestamp when preprocessing started
    data_type : str
        Type of data being processed (e.g., "sar", "s2")
    """
    logger = logging.getLogger('preprocessing')
    
    # Create splits file path
    splits_file = output_dir / f'event_splits_{data_type}_{split_seed}.yaml'
    
    # Prepare splits data structure
    event_splits = {
        'train': sorted([event.name for event in train_events]),
        'val': sorted([event.name for event in val_events]), 
        'test': sorted([event.name for event in test_events]),
        'metadata': {
            'data_type': data_type,
            'split_seed': split_seed,
            'val_ratio': val_ratio,
            'test_ratio': test_ratio,
            'total_events': len(train_events) + len(val_events) + len(test_events),
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


def tile_cloud_percentage(scl_path):
    """Calculates the percentage of pixels in SCL tile that is cloudy or null.
    
    Parameters
    ----------
    scl_path : Path
        Path to the SCL tile

    Returns
    -------
    cloud_percentage : float
        Percentage of pixels in SCL tile that is cloudy or null
    """
    with rasterio.open(scl_path) as src:
        cloud = src.read()
    return cloud.sum() / cloud.size

def has_tile_under_threshold(event: Path, tile_cloud_threshold: float):
    for scl_file in event.glob('clouds_*[0-9].tif'):
        if tile_cloud_percentage(scl_file) <= tile_cloud_threshold:
            return True
    return False

def filter_cloud_tiles_parallel(events: List[Path], tile_cloud_threshold: float, n_workers: int = None):
    """Filter out cloudy tiles in parallel.

    Parameters
    ----------
    events : List[Path]
        List of event paths
    tile_cloud_threshold : float
        Threshold for filtering out cloudy tiles
    n_workers : int
        Number of workers for parallel processing
    """
    if n_workers is None:
        n_workers = 1
    
    with mp.Pool(n_workers) as pool:
        mask = pool.starmap(has_tile_under_threshold, [(e, tile_cloud_threshold) for e in events])

    return [s for s, m in zip(events, mask) if m]

@hydra.main(version_base=None, config_path='pkg://configs', config_name='config.yaml')
def main(cfg: DictConfig) -> None:
    """Preprocesses raw S1 tiles and corresponding labels into smaller patches.

    Parameters
    ----------
    cfg : DictConfig
        Hydra configuration object containing all preprocessing parameters.
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
    if not hasattr(cfg.preprocess, 'n_workers'):
        cfg.preprocess.n_workers = 1

    # Create timestamp for logging
    timestamp = datetime.now().strftime('%Y-%m-%d %H:%M:%S')
    logger.info(f'''Starting SAR weak labeling preprocessing:
        Date:            {timestamp}
        Patch size:      {cfg.preprocess.size}
        Samples per tile: {cfg.preprocess.samples}
        Sampling method: {cfg.preprocess.method}
        Filter:          {getattr(cfg.preprocess, 'filter', 'raw')}
        Patch cloud threshold: {cfg.preprocess.cloud_threshold}
        Tile cloud threshold: {getattr(cfg.preprocess, 'tile_cloud_threshold', 0.25)}
        Random seed:     {cfg.preprocess.seed}
        Workers:         {cfg.preprocess.n_workers}
        Sample dir(s):   {cfg.preprocess.s1.sample_dirs}
        Label dir(s):    {cfg.preprocess.s1.label_dirs}
    ''')

    # Create preprocessing directory
    filter_str = getattr(cfg.preprocess, 'filter', 'raw')
    if getattr(cfg.preprocess, 'suffix', None):
        pre_sample_dir = Path(cfg.paths.preprocess_dir) / 's1_weak' / f'samples_{cfg.preprocess.size}_{cfg.preprocess.samples}_{filter_str}_{cfg.preprocess.suffix}'
    else:
        pre_sample_dir = Path(cfg.paths.preprocess_dir) / 's1_weak' / f'samples_{cfg.preprocess.size}_{cfg.preprocess.samples}_{filter_str}'
    pre_sample_dir.mkdir(parents=True, exist_ok=True)

    # Get directories and split ratios from config
    cfg_s1 = cfg.preprocess.get('s1', {})
    sample_dirs_list = cfg_s1.get('sample_dirs', [])
    label_dirs_list = cfg_s1.get('label_dirs', [])
    split_cfg = cfg_s1.get('split', {})
    val_ratio = split_cfg.get('val_ratio', 0.1)
    test_ratio = split_cfg.get('test_ratio', 0.1)

    if len(sample_dirs_list) == 0 or len(label_dirs_list) == 0:
        raise ValueError('Sample directories and label directories must be non empty.')

    # Build label index for manual labels
    label_idx = build_label_index(label_dirs_list, cfg)

    # Discover events with required SAR assets
    selected_events: List[Path] = []
    seen_eids = set()
    
    for sd in sample_dirs_list:
        sample_path = Path(cfg.paths.imagery_dir) / sd
        if not sample_path.is_dir():
            logger.debug(f'Sample directory {sd} is invalid, skipping...')
            continue
        
        for event_dir in sample_path.glob('[0-9]*'):
            if not event_dir.is_dir():
                continue
            eid = event_dir.name
            if eid in seen_eids:
                continue
            
            # Qualify: must have at least one prediction and at least one sar vv tile
            has_pred = any(event_dir.glob('pred_*.tif'))
            has_sar_vv = any(event_dir.glob('sar_*_vv.tif'))

            if has_pred and has_sar_vv:
                selected_events.append(event_dir)
                seen_eids.add(eid)
            else:
                logger.debug(f'Event {eid} in folder {event_dir.parent} did not meet reqs: pred={has_pred}, sar vv={has_sar_vv}')

    # filter cloudy tiles in parallel:
    tile_cloud_threshold = getattr(cfg.preprocess, 'tile_cloud_threshold', 0.25)
    filtered_events = filter_cloud_tiles_parallel(selected_events, tile_cloud_threshold, cfg.preprocess.n_workers)
    logger.info(f'# passed tile cloud threshold (<={tile_cloud_threshold}): {len(filtered_events)}/{len(selected_events)}')

    if len(filtered_events) == 0:
        logger.error('No events found in provided sample directories. Exiting.')
        return 1

    logger.info(f'Found {len(filtered_events)} events for processing')

    # Split events into train/val/test
    holdout_ratio = val_ratio + test_ratio
    if holdout_ratio <= 0 or holdout_ratio >= 1:
        raise ValueError('Sum of val_ratio and test_ratio must be in (0, 1).')

    train_events, val_test_events = train_test_split(
        filtered_events, test_size=holdout_ratio, random_state=cfg.preprocess.seed
    )
    test_prop_within_holdout = test_ratio / holdout_ratio
    val_events, test_events = train_test_split(
        val_test_events, test_size=test_prop_within_holdout, random_state=cfg.preprocess.seed + 1222
    )

    logger.info(f'Split: {len(train_events)} train, {len(val_events)} val, {len(test_events)} test events')
    
    # Save the event splits for reproducibility and reference
    save_event_splits(train_events, val_events, test_events, pre_sample_dir, 
                      cfg.preprocess.seed, val_ratio, test_ratio, timestamp, "sar")

    # Get list of tiles for splits as events can have multiple valid tiles
    logger.info('Discovering tiles...')
    train_tiles = discover_all_tiles(train_events, label_idx, tile_cloud_threshold)
    val_tiles = discover_all_tiles(val_events, label_idx, tile_cloud_threshold)
    test_tiles = discover_all_tiles(test_events, label_idx, tile_cloud_threshold)
    
    logger.info(f'Tiles: {len(train_tiles)} train, {len(val_tiles)} val, {len(test_tiles)} test')

    # Compute statistics using parallel Welford's algorithm
    logger.info('Computing training statistics...')
    mean_cont, std_cont = compute_statistics_parallel(train_tiles, filter_str, cfg.preprocess.n_workers)
    
    # Add binary channel statistics (mean=0, std=1)
    bchannels = 3  # waterbody, roads, flowlines
    mean_bin = np.zeros(bchannels, dtype=np.float32)
    std_bin = np.ones(bchannels, dtype=np.float32)
    mean = np.concatenate([mean_cont, mean_bin])
    std = np.concatenate([std_cont, std_bin])
    
    # Save statistics
    stats_file = pre_sample_dir / f'mean_std_{cfg.preprocess.size}_{cfg.preprocess.samples}_{filter_str}.pkl'
    with open(stats_file, 'wb') as f:
        pickle.dump((mean, std), f)
    logger.info(f'Statistics saved to {stats_file}')

    # Sample patches in parallel
    if cfg.preprocess.method == 'random':
        # Process each split
        for split_name, tiles in [('train', train_tiles), ('val', val_tiles), ('test', test_tiles)]:
            if len(tiles) == 0:
                logger.warning(f'No tiles for {split_name} split, skipping...')
                continue
            
            output_file = pre_sample_dir / f'{split_name}_patches.npy'
            logger.info(f'Processing {split_name} split with {len(tiles)} tiles...')
            
            sample_patches_parallel(
                tiles, cfg.preprocess.size, cfg.preprocess.samples, output_file, cfg.preprocess.cloud_threshold, 
                filter_str, cfg.preprocess.seed, cfg.preprocess.n_workers
            )
        
        logger.info('Parallel patch sampling complete.')
    else:
        raise ValueError(f"Unsupported sampling method: {cfg.preprocess.method}")

    logger.info('Preprocessing complete.')
    return 0


if __name__ == '__main__':
    main()