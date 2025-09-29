from glob import glob
import matplotlib.pyplot as plt
import rasterio
import numpy as np
from pathlib import Path
import re
import sys
from random import Random
import logging
import pickle
from sklearn.model_selection import train_test_split
import hydra
from omegaconf import DictConfig
from datetime import datetime
from typing import List, Tuple
import multiprocessing as mp
import shutil
import psutil
from numpy.lib.format import open_memmap
### ADD DEMS IN THE FUTURE - CURRENTLY NOT THE SAME SHAPE

from floodmaps.utils.preprocess_utils import WelfordAccumulator

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


def load_event_for_stats(vv_file: Path, vh_file: Path) -> Tuple[np.ndarray, np.ndarray]:
    """Load the event data and return the array and mask.

    Note that each event is expected to have one vv tif and one vh tif
    each containing N SAR time slices.

    Args:
        vv_file: Path of vv tif
        vh_file: Path of vh tif
        
    Returns:
        Tuple of (arr, mask) for the 2 sar channels
    """
    with rasterio.open(vv_file) as src:
        vv_raster = src.read()
    with rasterio.open(vh_file) as src:
        vh_raster = src.read()
    
    vv_raster = vv_raster.reshape((1, -1))
    vh_raster = vh_raster.reshape((1, -1))

    mask = (vv_raster[0] != -9999) & (vh_raster[0] != -9999)

    stack = np.vstack((vv_raster, vh_raster), dtype=np.float32)
    
    return stack, mask


def process_multi_batch_for_stats(multi_batch: List[Tuple[Path, Path]]) -> WelfordAccumulator:
    """Process a batch of events assigned to one worker using NumPy + Welford merging.
    
    Args:
        multi_batch: List of multitemporal event paths assigned to this worker
        
    Returns:
        WelfordAccumulator with accumulated statistics from all events in batch
    """
    accumulator = WelfordAccumulator(2)  # 2 sar channels
    
    for vv_file, vh_file in multi_batch:
        try:
            arr, mask = load_event_for_stats(vv_file, vh_file)
            accumulator.update(arr, mask)
        except Exception as e:
            raise RuntimeError(f"Worker failed processing event ({vv_file.name}, {vh_file.name}): {e}") from e
    
    return accumulator


def compute_statistics_parallel(train_multi: List[Tuple[Path, Path]], n_workers: int = None) -> Tuple[np.ndarray, np.ndarray]:
    """Compute mean and std of VV and VH channels using optimized parallel Welford's algorithm.

    Note: not by tiles but by events as multitemporal sampling folders for each date + cell
    do not have more than one sar sequence (e.g. of 10 vv and vh images).
    
    This implementation:
    1. Distributes events in batches to workers for better load balancing
    2. Uses NumPy for efficient tile-level statistics computation
    3. Uses Welford's algorithm to merge tile statistics within each worker
    4. Performs final merge of worker accumulators in main process
    
    Parameters
    ----------
    train_multi : List[Tuple[Path, Path]]
        List of multitemporal event paths for training set
    n_workers : int, optional
        Number of worker processes (defaults to CPU count)
        
    Returns
    -------
    mean : np.ndarray
        Mean of the 2 sar channels
    std : np.ndarray
        Standard deviation of the 2 sar channels
    """
    logger = logging.getLogger('preprocessing')
    
    if n_workers is None:
        n_workers = 1
    
    # Ensure we don't have more workers than tiles
    logger.info(f'Specified {n_workers} workers for statistics computation.')
    n_workers = min(n_workers, len(train_multi))
    logger.info(f'Using {n_workers} workers for {len(train_multi)} events...')
    
    # Split tiles into balanced batches for workers
    events_per_worker = len(train_multi) // n_workers
    remainder = len(train_multi) % n_workers
    
    multi_batches = []
    start_idx = 0
    
    for i in range(n_workers):
        # Some workers get one extra event if there's a remainder
        batch_size = events_per_worker + (1 if i < remainder else 0)
        end_idx = start_idx + batch_size
        
        if start_idx < len(train_multi):
            multi_batches.append(train_multi[start_idx:end_idx])
        
        start_idx = end_idx
    
    # Log batch distribution
    batch_sizes = [len(batch) for batch in multi_batches]
    logger.info(f'Multitemporal event distribution per worker: {batch_sizes}')
    
    # Process event batches in parallel
    try:
        with mp.Pool(n_workers) as pool:
            worker_accumulators = pool.map(
                process_multi_batch_for_stats,
                multi_batches
            )
    except Exception as e:
        logger.error(f"Failed during parallel statistics computation: {e}")
        logger.error("One or more worker processes failed during statistics calculation")
        raise RuntimeError(f"Statistics computation failed: {e}") from e
    
    # Merge worker accumulators (much fewer merge operations)
    final_accumulator = WelfordAccumulator(2)
    total_pixels = 0
    
    for worker_acc in worker_accumulators:
        final_accumulator.merge(worker_acc)
        total_pixels += worker_acc.count
    
    mean, std = final_accumulator.finalize()
    logger.info(f'Statistics computed from {total_pixels} pixels across {len(train_multi)} events')
    logger.info(f'Final statistics - Mean: {mean}, Std: {std}')
    
    return mean, std

def load_tiles_for_sampling(vv_file: Path, vh_file: Path, num_slices: int) -> Tuple[np.ndarray, np.ndarray]:
    """For a N length multitemporal snapshot, loads N individual tiles paired with
    their composites.

    Note that each event is expected to have one vv tif and one vh tif
    each containing N SAR time slices.

    Args:
        vv_file: Path of vv tif
        vh_file: Path of vh tif
        num_slices: Number of slices in each multitemporal event
        
    Returns:
        List of stacked arrays of shape (4, H, W) of the 2 sar channels and their composites
    """
    with rasterio.open(vv_file) as src:
        vv_raster = src.read()
    with rasterio.open(vh_file) as src:
        vh_raster = src.read()

    # check that the number of slices is correct or throw error
    if vv_raster.shape[0] != num_slices:
        raise ValueError(f"Number of slices in {vv_file} is {vv_raster.shape[0]}, expected {num_slices}")
    if vh_raster.shape[0] != num_slices:
        raise ValueError(f"Number of slices in {vh_file} is {vh_raster.shape[0]}, expected {num_slices}")

    vv_mask = (vv_raster == -9999).any(axis=0)
    vh_mask = (vh_raster == -9999).any(axis=0)
    vv_composite = np.mean(vv_raster, axis=0)
    vh_composite = np.mean(vh_raster, axis=0)
    vv_composite[vv_mask] = -9999
    vh_composite[vh_mask] = -9999

    tiles = []
    for slice in range(vv_raster.shape[0]):
        tiles.append(np.stack((vv_raster[slice], vh_raster[slice], vv_composite, vh_composite), dtype=np.float32))

    return tiles


def sample_patches_in_mem(multi: List[Tuple[Path, Path]], size: int, num_samples: int, num_slices: int,
                          seed: int, max_attempts: int = 10000) -> np.ndarray:
    """Sample patches and stores them in memory.
    
    Parameters
    ----------
    multi : List[Tuple[Path, Path]]
        List of multitemporal event tuples to process
    size : int
        Patch size (e.g., 64)
    num_samples : int
        Number of patches to sample per multitemporal event
    num_slices : int
        Number of slices in each multitemporal event
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
    total_patches = len(multi) * num_slices * num_samples
    dataset = np.empty((total_patches, 4, size, size), dtype=np.float32)
    
    for i, (vv_file, vh_file) in enumerate(multi):
        try:
            tiles = load_tiles_for_sampling(vv_file, vh_file, num_slices)
        except Exception as e:
            raise RuntimeError(f"Worker failed to load multitemporal event {i} (vv_file: {vv_file}, vh_file: {vh_file}): {e}") from e
        
        # loop over num_slices tiles
        for j, tile in enumerate(tiles):
            patches_sampled = 0
            _, HEIGHT, WIDTH = tile.shape
            attempts = 0
            while patches_sampled < num_samples and attempts < max_attempts:
                attempts += 1
                x = int(rng.uniform(0, HEIGHT - size))
                y = int(rng.uniform(0, WIDTH - size))
                patch = tile[:, x : x + size, y : y + size]
                
                # Filter out missing patches
                if np.any(patch == -9999):
                    continue

                dataset[(i * num_slices + j) * num_samples + patches_sampled] = patch
                patches_sampled += 1
            
            if patches_sampled < num_samples:
                raise RuntimeError(f"Worker exceeded max retries {max_attempts}, only sampled {patches_sampled}/{num_samples} patches for tile {i} (vv_file: {vv_file}, vh_file: {vh_file})")

    return dataset


def sample_patches_in_disk(multi: List[Tuple[Path, Path]], size: int, num_samples: int, num_slices: int,
                          seed: int, scratch_dir: Path, worker_id: int, max_attempts: int = 10000) -> Path:
    """Sample patches and stores them in memory mapped file on disk.
    
    Parameters
    ----------
    multi : List[Tuple[Path, Path]]
        List of tile tuples to process
    size : int
        Patch size (e.g., 64)
    num_samples : int
        Number of patches to sample per multitemporal event
    num_slices : int
        Number of slices in each multitemporal event
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
    total_patches = len(multi) * num_slices * num_samples
    tmp_file = scratch_dir / f"tmp_{worker_id}.dat"
    dataset = np.memmap(tmp_file, dtype=np.float32, shape=(total_patches, 4, size, size), mode="w+")
    
    try:
        for i, (vv_file, vh_file) in enumerate(multi):
            try:
                tiles = load_tiles_for_sampling(vv_file, vh_file, num_slices)
            except Exception as e:
                raise RuntimeError(f"Worker {worker_id} failed to load tile {i} (vv_file: {vv_file}, vh_file: {vh_file}): {e}") from e
            
            # loop over num_slices tiles
            for j, tile in enumerate(tiles):
                patches_sampled = 0
                _, HEIGHT, WIDTH = tile.shape
                attempts = 0
                while patches_sampled < num_samples and attempts < max_attempts:
                    attempts += 1
                    x = int(rng.uniform(0, HEIGHT - size))
                    y = int(rng.uniform(0, WIDTH - size))
                    patch = tile[:, x : x + size, y : y + size]
                    
                    # Filter out missing patches
                    if np.any(patch == -9999):
                        continue

                    dataset[(i * num_slices + j) * num_samples + patches_sampled] = patch
                    patches_sampled += 1

                if patches_sampled < num_samples:
                    raise RuntimeError(f"Worker {worker_id} exceeded max retries {max_attempts}, only sampled {patches_sampled}/{num_samples} patches for tile {i} (vv_file: {vv_file}, vh_file: {vh_file})")
    finally:
        # Ensure cleanup even if an error occurs
        dataset.flush()
        del dataset

    return tmp_file


def sample_patches_parallel(multi: List[Tuple[Path, Path]], size: int, num_samples: int,
                            num_slices: int, output_file: Path, seed: int, n_workers: int = None) -> None:
    """Uniformly samples sar patches of dimension size x size across each multitemporal event. The
    patches are saved together into one file.

    Note: for each multitemporal SAR tile (e.g. 10 slices), will compute the composite
    average of the 10 slices, and then for each sampled patch the following is saved:
    - vv (slice)
    - vh (slice)
    - vv_composite (average of all slices)
    - vh_composite (average of all slices)

    If there are 10 slices, then each sampled patch location results in 10 of these
    single-composite pairs.
    
    Samples patches in parallel. Strategy will depend on the memory available. If
    memory is comfortably below the total node memory (on improv you get ~230GB regardless
    of how many cpus requested), then we can just have each worker fill out their own
    array and then concatenate them in the parent.

    If memory required for entire array is too close or exceeds total node memory,
    then each worker will write to a memory mapped array and then combine them at the end.
    
    Parameters
    ----------
    multi : List[Tuple[Path, Path]]
        List of multitemporal events to process
    size : int
        Patch size (e.g., 64)
    num_samples : int
        Number of patches to sample per multitemporal event
    num_slices : int
        Number of slices in each multitemporal event
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
    
    logger.info(f'Specified {n_workers} workers for patch sampling.')
    n_workers = min(n_workers, len(multi))
    logger.info(f'Using {n_workers} workers for {len(multi)} multitemporal events...')
    
    total_patches = len(multi) * num_slices * num_samples
    array_shape = (total_patches, 4, size, size)
    array_dtype = np.float32

    # calculate how much memory required for the entire array (factor of 2 for concatenation operation)
    total_mem_required = array_shape[0] * array_shape[1] * array_shape[2] * array_shape[3] * np.dtype(array_dtype).itemsize * 2
    total_mem_available = psutil.virtual_memory().available
    logger.info(f'Total memory required for the entire array x2: {total_mem_required / 1024**3:.2f} GB')
    logger.info(f'Total memory available: {total_mem_available / 1024**3:.2f} GB')
    
    # divide up the tiles into n_workers chunks
    worker_multi = []
    batch_sizes = []
    start_idx = 0
    multi_per_worker = len(multi) // n_workers
    multi_remainder = len(multi) % n_workers
    for worker_id in range(n_workers):
        worker_multi_count = multi_per_worker + (1 if worker_id < multi_remainder else 0)
        end_idx = start_idx + worker_multi_count
        multi_chunk = multi[start_idx:end_idx]
        worker_multi.append(multi_chunk)
        batch_sizes.append(worker_multi_count)
        start_idx += worker_multi_count
        
    # Log batch distribution
    logger.info(f'Multitemporal event distribution per worker: {batch_sizes}')
    
    if total_mem_required < total_mem_available * 0.9:
        # use simple concatenation strategy
        logger.info('Total memory required is below available memory, using in-memory arrays and concatenation.')

        try:
            with mp.Pool(n_workers) as pool:
                results = pool.starmap(sample_patches_in_mem, [(multi, size, num_samples, num_slices, seed+i*10000) for i, multi in enumerate(worker_multi)])
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
                worker_files = pool.starmap(sample_patches_in_disk, [(multi, size, num_samples, num_slices, seed+i*10000, scratch_dir, i) for i, multi in enumerate(worker_multi)])
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
                multi_in_worker = len(worker_multi[i])
                chunk_shape = (multi_in_worker * num_slices * num_samples, 4, size, size)
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


@hydra.main(version_base=None, config_path='pkg://configs', config_name='config.yaml')
def main(cfg: DictConfig) -> None:
    """Preprocesses multitemporal SAR tiles into paired single vs composite patches for conditional generation.
    Also uses parallel workers.

    Parameters
    ----------
    cfg : DictConfig
        Hydra configuration object containing all preprocessing parameters.
    """
    # Extract parameters from config
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
        Samples per tile: {cfg.preprocess.samples}
        Multitemporal slices: {cfg.preprocess.slices}
        Sampling method: {cfg.preprocess.method}
        Random seed:     {cfg.preprocess.seed}
        Workers:         {n_workers}
        Sample dir(s):   {cfg.preprocess.multi.sample_dirs}
        Suffix:          {getattr(cfg.preprocess, 'suffix', None)}
    ''')

    # make our preprocess directory
    if getattr(cfg.preprocess, 'suffix', None):
        pre_sample_dir = Path(cfg.paths.preprocess_dir) / 'multi' / f'samples_{cfg.preprocess.size}_{cfg.preprocess.samples}_{cfg.preprocess.suffix}'
    else:
        pre_sample_dir = Path(cfg.paths.preprocess_dir) / 'multi' / f'samples_{cfg.preprocess.size}_{cfg.preprocess.samples}'
    pre_sample_dir.mkdir(parents=True, exist_ok=True)

    cfg_multi = cfg.preprocess.get('multi', {})
    sample_dirs_list = cfg_multi.get('sample_dirs', [])
    split_cfg = cfg_multi.get('split', {})
    val_ratio = split_cfg.get('val_ratio', 0.1)
    test_ratio = split_cfg.get('test_ratio', 0.1)

    if len(sample_dirs_list) == 0:
        raise ValueError('Sample directories and label directories must be non empty.')
    
    selected_multi = []
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
            
            # Qualify: must have vv and vh tile
            try:
                vv_file, vh_file = find_vv_vh_tifs(event_dir)
            except FileNotFoundError as e:
                logger.exception(f'Event {eid} in folder {event_dir.parent} did not meet reqs: {e}')
                continue

            selected_multi.append((vv_file, vh_file))
            seen_eids.add(eid)

    if len(selected_multi) == 0:
        logger.error('No multitemporal events found in provided sample directories. Exiting.')
        return 1
    
    logger.info(f'Found {len(selected_multi)} multitemporal events for processing')

    # Split events into train/val/test
    holdout_ratio = val_ratio + test_ratio
    if holdout_ratio <= 0 or holdout_ratio >= 1:
        raise ValueError('Sum of val_ratio and test_ratio must be in (0, 1).')

    train_multi, val_test_multi = train_test_split(
        selected_multi, test_size=holdout_ratio, random_state=cfg.preprocess.seed
    )
    test_prop_within_holdout = test_ratio / holdout_ratio
    val_multi, test_multi = train_test_split(
        val_test_multi, test_size=test_prop_within_holdout, random_state=cfg.preprocess.seed + 1222
    )

    logger.info(f'Split: {len(train_multi)} train, {len(val_multi)} val, {len(test_multi)} test multitemporal events')

    # TBD: calculate min and max of train tiles for later metrics (PSNR, SSIM)
    # min_val_vv, max_val_vv, min_val_vh, max_val_vh = trainMinMax(train_events)
    # stats_dir = DATA_DIR / 'ad/stats'
    # stats_dir.mkdir(parents=True, exist_ok=True)
    # with open(stats_dir / f'multi_{size}_{samples}_range_percentile.pkl', 'wb') as f:
    #     pickle.dump((min_val_vv, max_val_vv, min_val_vh, max_val_vh), f)

    # calculate mean and std of train tiles
    logger.info('Calculating training statistics...')
    mean, std = compute_statistics_parallel(train_multi, n_workers)
    logger.info('Mean and std of training tiles calculated.')

    # also store training vv, vh mean std statistics in file
    stats_file = pre_sample_dir / f'mean_std_{cfg.preprocess.size}_{cfg.preprocess.samples}.pkl'
    with open(stats_file, 'wb') as f:
        pickle.dump((mean, std), f)
    logger.info('Training mean and std statistics saved.')
 
    if cfg.preprocess.method == 'random':
        # Process each split
        for split_name, events in [('train', train_multi), ('val', val_multi), ('test', test_multi)]:
            if len(events) == 0:
                logger.warning(f'No multitemporal events for {split_name} split, skipping...')
                continue
            
            output_file = pre_sample_dir / f'{split_name}_patches.npy'
            logger.info(f'Processing {split_name} split with {len(events)} events...')
            
            sample_patches_parallel(
                events, cfg.preprocess.size, cfg.preprocess.samples, cfg.preprocess.slices, 
                output_file, cfg.preprocess.seed, n_workers
            )
        
        logger.info('Parallel patch sampling complete.')
    else:
        raise ValueError(f"Unsupported sampling method: {cfg.preprocess.method}")
    
    logger.info('Preprocessing complete.')
    return 0

if __name__ == '__main__':
    main()
