import rasterio
import numpy as np
from pathlib import Path
import re
import sys
from random import Random
import logging
import pickle
from sklearn.model_selection import train_test_split
from typing import List, Tuple, Dict, Optional
from datetime import datetime
import hydra
from omegaconf import DictConfig
import yaml
from floodmaps.utils.preprocess_utils import WelfordAccumulator, MinMaxAccumulator
import csv
import concurrent.futures
import shutil
from collections import defaultdict

# Constants
MULTI_CHANNELS = 4  # VV single, VH single, VV composite, VH composite
SAR_MISSING_VALUE = -9999

# Regex pattern for multitemporal SAR files from sample_sar_multi.py
MULTI_VV_PATTERN = re.compile(r'multi_(\d{8})-(\d{8})_(\d+_\d+)_vv\.tif$')


def load_tile_for_stats(tile_info: Tuple[Path, Path, Path]) -> Tuple[np.ndarray, np.ndarray]:
    """Load the multitemporal tile data and return the array and mask for statistics.
    
    Parameters
    ----------
    tile_info : Tuple[Path, Path, Path]
        Tuple of (cell_path, vv_file, vh_file)
        
    Returns
    -------
    Tuple[np.ndarray, np.ndarray]
        Tuple of (arr, mask) for the 2 SAR channels (VV, VH) flattened across all slices
    """
    cell_path, vv_file, vh_file = tile_info
    
    with rasterio.open(vv_file) as src:
        vv_raster = src.read()  # (N, H, W)
    with rasterio.open(vh_file) as src:
        vh_raster = src.read()  # (N, H, W)
    
    # Flatten all slices for statistics computation
    vv_flat = vv_raster.reshape((1, -1))  # (1, N*H*W)
    vh_flat = vh_raster.reshape((1, -1))  # (1, N*H*W)
    
    # Stack channels
    stack = np.vstack((vv_flat, vh_flat)).astype(np.float32)  # (2, N*H*W)
    
    # Create mask for valid pixels (exclude missing values)
    mask = (vv_flat[0] != SAR_MISSING_VALUE) & (vh_flat[0] != SAR_MISSING_VALUE)
    
    return stack, mask


def process_tiles_batch_for_stats(tiles_batch: List[Tuple]) -> Tuple[WelfordAccumulator, MinMaxAccumulator]:
    """Process a batch of tiles assigned to one worker using NumPy + Welford merging.
    
    Parameters
    ----------
    tiles_batch : List[Tuple]
        List of tile tuples assigned to this worker

    Returns
    -------
    Tuple[WelfordAccumulator, MinMaxAccumulator]
        Tuple of (welford_accumulator, minmax_accumulator) with accumulated statistics
    """
    welford_acc = WelfordAccumulator(2)  # 2 SAR channels (VV, VH)
    minmax_acc = MinMaxAccumulator(2)  # 2 SAR channels (VV, VH)
    
    for tile_info in tiles_batch:
        try:
            arr, mask = load_tile_for_stats(tile_info)
            welford_acc.update(arr, mask)
            minmax_acc.update(arr, mask)
            arr = None
            mask = None
        except Exception as e:
            cell_path, vv_file, vh_file = tile_info
            raise RuntimeError(f"Worker failed processing tile (cell: {cell_path.name}, vv: {vv_file.name}): {e}") from e
    
    return welford_acc, minmax_acc


def compute_statistics_parallel(train_tiles: List[Tuple], n_workers: int = None) -> Tuple[np.ndarray, np.ndarray, np.ndarray, np.ndarray]:
    """Compute mean, std, min, and max using optimized parallel algorithms.
    
    Parameters
    ----------
    train_tiles : List[Tuple]
        List of tile tuples for training set
    n_workers : int, optional
        Number of worker processes (defaults to 1)
        
    Returns
    -------
    mean : np.ndarray
        Mean of the 2 SAR channels
    std : np.ndarray
        Standard deviation of the 2 SAR channels
    min_vals : np.ndarray
        Minimum values of the 2 SAR channels
    max_vals : np.ndarray
        Maximum values of the 2 SAR channels
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
            worker_results = [future.result() for future in futures]
    except Exception as e:
        logger.error(f"Failed during parallel statistics computation: {e}")
        raise RuntimeError(f"Statistics computation failed: {e}") from e
    
    # Merge worker accumulators
    final_welford = WelfordAccumulator(2)
    final_minmax = MinMaxAccumulator(2)
    total_pixels = 0
    
    for welford_acc, minmax_acc in worker_results:
        final_welford.merge(welford_acc)
        final_minmax.merge(minmax_acc)
        total_pixels += welford_acc.count
    
    mean, std = final_welford.finalize()
    min_vals, max_vals = final_minmax.finalize()
    
    logger.info(f'Statistics computed from {total_pixels} pixels across {len(train_tiles)} tiles')
    logger.info(f'Final statistics - Mean: {mean}, Std: {std}')
    logger.info(f'Final statistics - Min: {min_vals}, Max: {max_vals}')
    
    return mean, std, min_vals, max_vals


def load_tile_for_sampling(tile_info: Tuple[Path, Path, Path], acquisitions: int) -> Tuple[List[np.ndarray], np.ndarray]:
    """Load a multitemporal tile and return N leave-one-out tiles for patch sampling.
    
    For each of the N acquisitions, creates a 4-channel tile with:
    - Channel 0: VV single slice
    - Channel 1: VH single slice  
    - Channel 2: VV composite (mean of other N-1 slices)
    - Channel 3: VH composite (mean of other N-1 slices)
    
    Parameters
    ----------
    tile_info : Tuple[Path, Path, Path]
        Tuple of (cell_path, vv_file, vh_file)
    acquisitions : int
        Expected number of acquisitions per stack
        
    Returns
    -------
    tiles : List[np.ndarray]
        List of N tiles, each of shape (4, H, W)
    union_missing_mask : np.ndarray
        Union of missing masks across all N acquisitions, shape (H, W)
    """
    cell_path, vv_file, vh_file = tile_info
    
    with rasterio.open(vv_file) as src:
        vv_raster = src.read().astype(np.float32)  # (N, H, W)
    with rasterio.open(vh_file) as src:
        vh_raster = src.read().astype(np.float32)  # (N, H, W)
    
    N, H, W = vv_raster.shape
    
    # Verify acquisition count matches expected
    if N != acquisitions:
        raise ValueError(f"Stack {vv_file.name} has {N} acquisitions, expected {acquisitions}")
    if vh_raster.shape[0] != acquisitions:
        raise ValueError(f"Stack {vh_file.name} has {vh_raster.shape[0]} acquisitions, expected {acquisitions}")
    
    # Compute union missing mask across ALL N acquisitions
    vv_missing = (vv_raster == SAR_MISSING_VALUE)  # (N, H, W)
    vh_missing = (vh_raster == SAR_MISSING_VALUE)  # (N, H, W)
    union_missing_mask = (vv_missing | vh_missing).any(axis=0)  # (H, W)
    
    # Generate leave-one-out tiles
    tiles = []
    indices = np.arange(N)
    
    for i in range(N):
        # Single slice
        vv_single = vv_raster[i]  # (H, W)
        vh_single = vh_raster[i]  # (H, W)
        
        # Leave-one-out: select all slices except i
        other_mask = (indices != i)
        others_vv = vv_raster[other_mask]  # (N-1, H, W)
        others_vh = vh_raster[other_mask]  # (N-1, H, W)
        
        # Compute composite: mean of N-1 slices
        # If ANY of the N-1 slices has missing value at a pixel, composite is missing
        others_vv_missing = (others_vv == SAR_MISSING_VALUE).any(axis=0)  # (H, W)
        others_vh_missing = (others_vh == SAR_MISSING_VALUE).any(axis=0)  # (H, W)
        
        vv_composite = np.mean(others_vv, axis=0)  # (H, W)
        vh_composite = np.mean(others_vh, axis=0)  # (H, W)
        
        # Set missing composite pixels
        vv_composite[others_vv_missing] = SAR_MISSING_VALUE
        vh_composite[others_vh_missing] = SAR_MISSING_VALUE
        
        # Stack into 4-channel tile
        tile = np.stack([vv_single, vh_single, vv_composite, vh_composite], axis=0)  # (4, H, W)
        tiles.append(tile)
    
    return tiles, union_missing_mask


def sample_patches_random(chunk_tile_infos: List[Tuple], size: int, num_samples: int,
                          acquisitions: int, missing_percent: float, seed: int, 
                          save_file: Path, max_attempts: int = 20000) -> None:
    """Sample patches using random uniform sampling with leave-one-out composites.
    
    For each multitemporal tile, generates N leave-one-out tiles and samples
    num_samples patches from each, resulting in N * num_samples patches per tile.
    
    Parameters
    ----------
    chunk_tile_infos : List[Tuple]
        List of tile info tuples in the chunk to process
    size : int
        Patch size (e.g., 64)
    num_samples : int
        Number of patches to sample per leave-one-out tile
    acquisitions : int
        Expected number of acquisitions per stack
    missing_percent : float
        Maximum missing percentage for patch acceptance (0.0 means no missing allowed)
    seed : int
        Random seed for reproducibility
    save_file : Path
        Path to save the output .npy file
    max_attempts : int
        Maximum number of attempts to sample patches per tile
    """
    rng = Random(seed)
    all_patches = []
    
    for i, tile_info in enumerate(chunk_tile_infos):
        cell_path, vv_file, vh_file = tile_info
        try:
            tiles, union_missing_mask = load_tile_for_sampling(tile_info, acquisitions)
        except Exception as e:
            raise RuntimeError(f"Worker failed to load tile {i} (cell: {cell_path.name}, vv: {vv_file.name}): {e}") from e
        
        N = len(tiles)
        _, HEIGHT, WIDTH = tiles[0].shape
        
        # Check if tile is large enough for patch size
        if HEIGHT < size or WIDTH < size:
            raise RuntimeError(f"Tile {i} (cell: {cell_path.name}) is too small ({HEIGHT}x{WIDTH}) for patch size {size}x{size}")
        
        # Sample patches from each leave-one-out tile
        for j, tile in enumerate(tiles):
            patches_sampled = 0
            attempts = 0
            
            while patches_sampled < num_samples and attempts < max_attempts:
                attempts += 1
                x = int(rng.uniform(0, HEIGHT - size))
                y = int(rng.uniform(0, WIDTH - size))
                
                # Check missing percentage in patch window using union mask
                patch_missing = union_missing_mask[x:x+size, y:y+size]
                patch_missing_pct = patch_missing.sum() / patch_missing.size
                
                if patch_missing_pct > missing_percent:
                    continue
                
                patch = tile[:, x:x+size, y:y+size]
                all_patches.append(patch)
                patches_sampled += 1
    
    # Save patches
    if len(all_patches) > 0:
        patches_array = np.array(all_patches, dtype=np.float32)
    else:
        patches_array = np.empty((0, MULTI_CHANNELS, size, size), dtype=np.float32)
    np.save(save_file, patches_array)


def sample_patches_parallel_random(preprocess_dir: Path, tile_infos: List[Tuple], size: int, 
                                   num_samples: int, acquisitions: int, missing_percent: float,
                                   output_file: Path, seed: int, n_workers: int = None, 
                                   chunk_size: int = 100, scratch_dir: Path = None) -> None:
    """Sample patches in parallel using the random method with chunk streaming.
    
    Parameters
    ----------
    preprocess_dir : Path
        Path to the preprocess directory
    tile_infos : List[Tuple]
        List of tile info tuples to process
    size : int
        Patch size (e.g., 64)
    num_samples : int
        Number of patches to sample per leave-one-out tile
    acquisitions : int
        Expected number of acquisitions per stack
    missing_percent : float
        Maximum missing percentage for patch acceptance
    output_file : Path
        Path to save the output .npy file
    seed : int
        Random seed for reproducibility
    n_workers : int, optional
        Number of worker processes (defaults to 1)
    chunk_size : int, optional
        Number of tiles to process per worker before saving as temp file
    scratch_dir : Path, optional
        Path to scratch directory for intermediate files
    """
    logger = logging.getLogger('preprocessing')
    
    if n_workers is None:
        n_workers = 1
    
    logger.info(f'Specified {n_workers} workers for random patch sampling.')
    chunks = (len(tile_infos) + chunk_size - 1) // chunk_size
    n_workers = min(n_workers, chunks)
    logger.info(f'Using {n_workers} workers for {len(tile_infos)} tiles in {chunks} chunks of size {chunk_size}...')

    # Clean up any previous temp files
    chunk_dir = preprocess_dir if scratch_dir is None else scratch_dir
    for tmp_file in chunk_dir.glob("chunk_*.npy"):
        try:
            tmp_file.unlink()
        except Exception as e:
            logger.warning(f"Failed to delete previous temp file {tmp_file}: {e}")
    
    # Divide tiles into chunks
    chunked_tile_infos = []
    for start_idx in range(0, len(tile_infos), chunk_size):
        tiles_chunk = tile_infos[start_idx:start_idx+chunk_size]
        chunked_tile_infos.append(tiles_chunk)

    try:
        with concurrent.futures.ProcessPoolExecutor(max_workers=n_workers) as executor:
            futures = [
                executor.submit(
                    sample_patches_random, 
                    tiles_chunk, size, num_samples, acquisitions, missing_percent,
                    seed + i * 10000, chunk_dir / f'chunk_{i}.npy'
                ) 
                for i, tiles_chunk in enumerate(chunked_tile_infos)
            ]
            for future in futures:
                future.result()
    except Exception as e:
        logger.error(f"Failed during parallel random patch sampling: {e}")
        raise RuntimeError(f"Random patch sampling failed: {e}") from e

    # Stream chunk files into final memory mapped array
    _stream_chunks_to_output(chunk_dir, output_file, size, scratch_dir, logger)


def sample_patches_strided(chunk_tile_infos: List[Tuple], size: int, stride: int,
                           acquisitions: int, missing_percent: float, save_file: Path) -> None:
    """Sample patches using a sliding window with stride and edge coverage.
    
    Parameters
    ----------
    chunk_tile_infos : List[Tuple]
        List of tile info tuples in the chunk to process
    size : int
        Patch size (e.g., 64)
    stride : int
        Stride for sliding window sampling
    acquisitions : int
        Expected number of acquisitions per stack
    missing_percent : float
        Maximum missing percentage for patch acceptance
    save_file : Path
        Path to save the output .npy file
    """
    all_patches = []
    
    for i, tile_info in enumerate(chunk_tile_infos):
        cell_path, vv_file, vh_file = tile_info
        try:
            tiles, union_missing_mask = load_tile_for_sampling(tile_info, acquisitions)
        except Exception as e:
            raise RuntimeError(f"Worker failed to load tile {i} (cell: {cell_path.name}, vv: {vv_file.name}): {e}") from e
        
        _, HEIGHT, WIDTH = tiles[0].shape
        
        # Check if tile is large enough for patch size
        if HEIGHT < size or WIDTH < size:
            raise RuntimeError(f"Tile {i} (cell: {cell_path.name}) is too small ({HEIGHT}x{WIDTH}) for patch size {size}x{size}")
        
        # Generate window positions with edge coverage
        x_positions = list(range(0, HEIGHT - size + 1, stride))
        if x_positions and x_positions[-1] != HEIGHT - size:
            x_positions.append(HEIGHT - size)
        
        y_positions = list(range(0, WIDTH - size + 1, stride))
        if y_positions and y_positions[-1] != WIDTH - size:
            y_positions.append(WIDTH - size)
        
        # Sample patches from each leave-one-out tile at all positions
        for tile in tiles:
            for x in x_positions:
                for y in y_positions:
                    # Check missing percentage in patch window
                    patch_missing = union_missing_mask[x:x+size, y:y+size]
                    patch_missing_pct = patch_missing.sum() / patch_missing.size
                    
                    if patch_missing_pct > missing_percent:
                        continue
                    
                    patch = tile[:, x:x+size, y:y+size]
                    all_patches.append(patch)
    
    # Save patches
    if len(all_patches) > 0:
        patches_array = np.array(all_patches, dtype=np.float32)
    else:
        patches_array = np.empty((0, MULTI_CHANNELS, size, size), dtype=np.float32)
    np.save(save_file, patches_array)


def sample_patches_parallel_strided(preprocess_dir: Path, tile_infos: List[Tuple], size: int,
                                    stride: int, acquisitions: int, missing_percent: float,
                                    output_file: Path, n_workers: int = None,
                                    chunk_size: int = 100, scratch_dir: Path = None) -> None:
    """Sample patches in parallel using the strided method with chunk streaming.
    
    Parameters
    ----------
    preprocess_dir : Path
        Path to the preprocess directory
    tile_infos : List[Tuple]
        List of tile info tuples to process
    size : int
        Patch size (e.g., 64)
    stride : int
        Stride for sliding window sampling
    acquisitions : int
        Expected number of acquisitions per stack
    missing_percent : float
        Maximum missing percentage for patch acceptance
    output_file : Path
        Path to save the output .npy file
    n_workers : int, optional
        Number of worker processes (defaults to 1)
    chunk_size : int, optional
        Number of tiles to process per worker before saving as temp file
    scratch_dir : Path, optional
        Path to scratch directory for intermediate files
    """
    logger = logging.getLogger('preprocessing')
    
    if n_workers is None:
        n_workers = 1
    
    logger.info(f'Specified {n_workers} workers for strided patch sampling.')
    chunks = (len(tile_infos) + chunk_size - 1) // chunk_size
    n_workers = min(n_workers, chunks)
    logger.info(f'Using {n_workers} workers for {len(tile_infos)} tiles in {chunks} chunks of size {chunk_size}...')

    # Clean up any previous temp files
    chunk_dir = preprocess_dir if scratch_dir is None else scratch_dir
    for tmp_file in chunk_dir.glob("chunk_*.npy"):
        try:
            tmp_file.unlink()
        except Exception as e:
            logger.warning(f"Failed to delete previous temp file {tmp_file}: {e}")
    
    # Divide tiles into chunks
    chunked_tile_infos = []
    for start_idx in range(0, len(tile_infos), chunk_size):
        tiles_chunk = tile_infos[start_idx:start_idx+chunk_size]
        chunked_tile_infos.append(tiles_chunk)

    try:
        with concurrent.futures.ProcessPoolExecutor(max_workers=n_workers) as executor:
            futures = [
                executor.submit(
                    sample_patches_strided,
                    tiles_chunk, size, stride, acquisitions, missing_percent,
                    chunk_dir / f'chunk_{i}.npy'
                )
                for i, tiles_chunk in enumerate(chunked_tile_infos)
            ]
            for future in futures:
                future.result()
    except Exception as e:
        logger.error(f"Failed during parallel strided patch sampling: {e}")
        raise RuntimeError(f"Strided patch sampling failed: {e}") from e

    # Stream chunk files into final memory mapped array
    _stream_chunks_to_output(chunk_dir, output_file, size, scratch_dir, logger)


def _stream_chunks_to_output(chunk_dir: Path, output_file: Path, size: int, 
                             scratch_dir: Optional[Path], logger: logging.Logger) -> None:
    """Stream chunk files into a single memory-mapped output array.
    
    Parameters
    ----------
    chunk_dir : Path
        Directory containing chunk_*.npy files
    output_file : Path
        Path to save the final output .npy file
    size : int
        Patch size for shape validation
    scratch_dir : Optional[Path]
        Scratch directory (if used, output is moved from scratch to final location)
    logger : logging.Logger
        Logger instance
    """
    try:
        chunk_files = sorted(chunk_dir.glob("chunk_*.npy"), key=lambda x: int(x.stem.split("_")[1]))
        if not chunk_files:
            raise RuntimeError("No temporary chunk files found for patch sampling output.")

        # Calculate total patches
        total_patches = 0
        for tmp_file in chunk_files:
            arr = np.load(tmp_file, mmap_mode='r')
            total_patches += arr.shape[0]
        logger.info(f'Total patches read from {len(chunk_files)} chunks: {total_patches}')

        # Preallocate memmapped output
        memmap_file = scratch_dir / output_file.name if scratch_dir is not None else output_file
        final_arr = np.lib.format.open_memmap(
            memmap_file, mode="w+", dtype=np.float32, 
            shape=(total_patches, MULTI_CHANNELS, size, size)
        )
        
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
        
        # Move from scratch to final destination if needed
        if scratch_dir is not None:
            shutil.move(memmap_file, output_file)

        logger.info('Sampling complete.')
    except Exception as e:
        logger.exception("Failed to stream chunk files or save output.")
        raise RuntimeError(f"Failed to save final array: {e}") from e


def sample_homogenous_patches(tile_infos: List[Tuple], size: int, acquisitions: int,
                              missing_percent: float, coord_to_patch: Dict[Tuple, List[Tuple]]) -> np.ndarray:
    """Extract homogenous patches from specified coordinates using sequential processing.
    
    For each tile, generates N leave-one-out tiles and extracts patches at 
    locations specified in coord_to_patch for that cell.
    
    NOTE: tile_infos should be pre-filtered to only include tiles whose cells
    have entries in coord_to_patch. This function assumes all tiles passed in
    have corresponding patch coordinates in coord_to_patch.
    
    Parameters
    ----------
    tile_infos : List[Tuple]
        List of (cell_path, vv_file, vh_file) tuples, pre-filtered to only include
        cells with homogenous patch locations in coord_to_patch
    size : int
        Patch size (e.g., 64)
    acquisitions : int
        Expected number of acquisitions per stack
    missing_percent : float
        Maximum missing percentage for patch acceptance
    coord_to_patch : Dict[Tuple, List[Tuple]]
        Dictionary mapping (y, x) cell coordinates to list of (i, j) patch locations
        
    Returns
    -------
    np.ndarray
        Array of shape (B, 4, size, size) containing extracted patches
    """
    logger = logging.getLogger('preprocessing')
    all_patches = []
    
    for tile_info in tile_infos:
        cell_path, vv_file, vh_file = tile_info
        
        # Extract cell coords from directory name (format: y_x)
        parts = cell_path.name.split('_')
        y, x = int(parts[0]), int(parts[1])
        patch_coords = coord_to_patch[(y, x)]
        
        # Load tile and generate leave-one-out tiles
        try:
            tiles, union_missing_mask = load_tile_for_sampling(tile_info, acquisitions)
        except Exception as e:
            logger.warning(f'Failed to load tile (cell: {cell_path.name}, vv: {vv_file.name}): {e}, skipping...')
            continue
        
        _, HEIGHT, WIDTH = tiles[0].shape
        
        # Extract patches from each leave-one-out tile at specified locations
        for tile in tiles:  # N leave-one-out tiles
            for (i, j) in patch_coords:
                # Validate patch bounds
                if i + size > HEIGHT or j + size > WIDTH:
                    logger.debug(f'Patch at ({i}, {j}) exceeds tile bounds ({HEIGHT}, {WIDTH}), skipping...')
                    continue
                
                # Check missing percentage in patch window
                patch_missing = union_missing_mask[i:i+size, j:j+size]
                patch_missing_pct = patch_missing.sum() / patch_missing.size
                
                if patch_missing_pct > missing_percent:
                    continue
                
                patch = tile[:, i:i+size, j:j+size]
                all_patches.append(patch)
    
    if len(all_patches) > 0:
        return np.array(all_patches, dtype=np.float32)
    else:
        return np.empty((0, MULTI_CHANNELS, size, size), dtype=np.float32)


def get_tile_infos_from_cells(cells: List[Path], acquisitions: int) -> List[Tuple[Path, Path, Path]]:
    """Get multitemporal tile info from cell directories.
    
    For each cell, finds all multi_*_vv.tif files and their corresponding vh pairs.
    Validates that VH counterpart exists and stack has expected acquisitions.
    
    Parameters
    ----------
    cells : List[Path]
        List of cell directory paths
    acquisitions : int
        Expected number of acquisitions per stack
        
    Returns
    -------
    List[Tuple[Path, Path, Path]]
        List of (cell_path, vv_file, vh_file) tuples
    """
    logger = logging.getLogger('preprocessing')
    tile_infos = []
    
    for cell in cells:
        for vv_file in cell.glob('multi_*_vv.tif'):
            m = MULTI_VV_PATTERN.match(vv_file.name)
            if not m:
                logger.debug(f'File {vv_file.name} in {cell.name} does not match pattern, skipping...')
                continue
            
            # Find corresponding VH file
            vh_file = vv_file.with_name(vv_file.name.replace('_vv.tif', '_vh.tif'))
            if not vh_file.exists():
                logger.debug(f'VH counterpart not found for {vv_file.name} in {cell.name}, skipping...')
                continue
            
            # Validate acquisition count
            try:
                with rasterio.open(vv_file) as src:
                    n_bands = src.count
                if n_bands != acquisitions:
                    logger.debug(f'Stack {vv_file.name} has {n_bands} acquisitions, expected {acquisitions}, skipping...')
                    continue
            except Exception as e:
                logger.debug(f'Failed to read {vv_file.name}: {e}, skipping...')
                continue
            
            tile_infos.append((cell, vv_file, vh_file))
    
    return tile_infos


def save_cell_splits(train_cells: List[Path], val_cells: List[Path], test_cells: List[Path],
                     cfg: DictConfig, output_dir: Path, timestamp: str) -> None:
    """Save cell splits to a YAML file for reproducibility and reference.
    
    Parameters
    ----------
    train_cells : List[Path]
        List of training cell directory paths
    val_cells : List[Path]
        List of validation cell directory paths
    test_cells : List[Path]
        List of test cell directory paths
    cfg : DictConfig
        Config object
    output_dir : Path
        Directory to save the splits file
    timestamp : str
        Timestamp when preprocessing started
    """
    logger = logging.getLogger('preprocessing')
    
    splits_file = output_dir / 'metadata.yaml'
    
    # Prepare splits data structure
    cell_splits = {
        'train': sorted([cell.name for cell in train_cells]),
        'val': sorted([cell.name for cell in val_cells]),
        'test': sorted([cell.name for cell in test_cells]),
        'metadata': {
            'data_type': 's1_multi',
            'method': cfg.preprocess.method,
            'size': cfg.preprocess.size,
            'acquisitions': cfg.preprocess.acquisitions,
            'samples': getattr(cfg.preprocess, 'samples', None),
            'stride': getattr(cfg.preprocess, 'stride', None),
            'missing_percent': getattr(cfg.preprocess, 'missing_percent', 0.0),
            'seed': getattr(cfg.preprocess, 'seed', None),
            'total_cells': len(train_cells) + len(val_cells) + len(test_cells),
            'split_csv': getattr(cfg.preprocess, 'split_csv', None),
            'val_ratio': getattr(cfg.preprocess, 'val_ratio', None),
            'test_ratio': getattr(cfg.preprocess, 'test_ratio', None),
            'train_count': len(train_cells),
            'val_count': len(val_cells),
            'test_count': len(test_cells),
            'timestamp': timestamp
        },
    }
    
    try:
        with open(splits_file, 'w') as f:
            yaml.dump(cell_splits, f, default_flow_style=False, sort_keys=False)
        logger.info(f'Cell splits saved to {splits_file}')
    except Exception as e:
        logger.error(f'Failed to save cell splits: {e}')
        raise


@hydra.main(version_base=None, config_path='pkg://configs', config_name='config.yaml')
def main(cfg: DictConfig) -> None:
    """Preprocesses multitemporal SAR tiles into paired single vs composite patches
    for conditional generation training. Uses a leave-one-out method where given a
    time interval with N SAR images, each image is paired with the composite
    from the remaining N-1 images, producing N pairings of each multitemporal stack of N
    images.

    Sample directories are s1_multi directories containing SAR imagery across time intervals
    sampled using the floodmaps/sampling/sample_s1_multi script.

    cfg.preprocess.split_csv should be the path to a CSV file with columns "y", "x", "split"
    where each PRISM cell coordinate (y, x) is associated with a split "train", "val", "test".
    If provided, allows for pre determined split rather than random split. This is preferred
    over the random splitting to avoid data leakage from similar dates / regions.

    NOTE: For large datasets on HPC, use the scratch directory for speed.
    
    cfg.preprocess Parameters:
    - size: int (pixel width of patch)
    - acquisitions: int (number of acquisitions in each multitemporal stack)
    - method: str ['random', 'strided']
    - samples: int (number of samples per image for random method)
    - stride: int (for strided method)
    - missing_percent: float (maximum missing percentage for patch acceptance) (default: 0.0)
    - seed: int (random number generator seed for random method)
    - n_workers: int (number of workers for parallel processing)
    - chunk_size: int (number of tiles to process per worker before saving as temp file) (default: 100)
    - s1.sample_dirs: List[str] (list of multitemporal sample directories under cfg.data.imagery_dir)
    - suffix: str (optional suffix to append to preprocessed folder)
    - split_csv: str (path to CSV file with columns "y", "x", "split" for PRISM cell coordinates)
    - homogenous_csv: str (path to CSV file with columns "y", "x", "i", "j" for PRISM cell coordinates and homogenous patch location)
    - val_ratio: used for random splitting if no split_csv is provided
    - test_ratio: used for random splitting if no split_csv is provided
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
    logger.info(f'''Starting S1 multitemporal preprocessing:
        Date:            {timestamp}
        Patch size:      {cfg.preprocess.size}
        Acquisitions:    {cfg.preprocess.acquisitions}
        Sampling method: {cfg.preprocess.method}
        Samples per tile (Random method): {getattr(cfg.preprocess, 'samples', None)}
        Stride (Strided method): {getattr(cfg.preprocess, 'stride', None)}
        Missing percent: {getattr(cfg.preprocess, 'missing_percent', 0.0)} (default: 0.0)
        Random seed:     {getattr(cfg.preprocess, 'seed', None)}
        Workers:         {n_workers}
        Chunk size:      {getattr(cfg.preprocess, 'chunk_size', 100)} (default: 100)
        Sample dir(s):   {cfg.preprocess.s1.sample_dirs}
        Suffix:          {getattr(cfg.preprocess, 'suffix', None)}
        Split CSV:       {getattr(cfg.preprocess, 'split_csv', None)}
        Homogenous CSV:  {getattr(cfg.preprocess, 'homogenous_csv', None)}
        Val ratio:       {getattr(cfg.preprocess, 'val_ratio', None)}
        Test ratio:      {getattr(cfg.preprocess, 'test_ratio', None)}
        Scratch dir:     {getattr(cfg.preprocess, 'scratch_dir', None)}
    ''')

    # Create preprocessing directory
    sampling_param = cfg.preprocess.samples if cfg.preprocess.method == 'random' else cfg.preprocess.stride
    if getattr(cfg.preprocess, 'suffix', None):
        pre_sample_dir = Path(cfg.paths.preprocess_dir) / 's1_multi' / f'{cfg.preprocess.method}_{cfg.preprocess.size}_{sampling_param}_{cfg.preprocess.suffix}'
    else:
        pre_sample_dir = Path(cfg.paths.preprocess_dir) / 's1_multi' / f'{cfg.preprocess.method}_{cfg.preprocess.size}_{sampling_param}'
    pre_sample_dir.mkdir(parents=True, exist_ok=True)
    logger.info(f'Preprocess directory: {pre_sample_dir}')

    # Get sample directories from config
    cfg_s1 = cfg.preprocess.get('s1', {})
    sample_dirs_list = cfg_s1.get('sample_dirs', [])

    if len(sample_dirs_list) == 0:
        raise ValueError('No sample directories were provided.')

    # Discover cells from sample directories with deduplication
    all_cells: List[Path] = []
    seen_cell_ids = set()
    cell_pattern = re.compile(r'(\d+)_(\d+)$')
    
    for sd in sample_dirs_list:
        sample_path = Path(cfg.paths.imagery_dir) / sd
        if not sample_path.is_dir():
            logger.debug(f'Sample directory {sd} is invalid, skipping...')
            continue
        
        for cell_dir in sample_path.glob('[0-9]*_[0-9]*'):
            if not cell_dir.is_dir():
                continue
            
            cell_id = cell_dir.name
            m = cell_pattern.match(cell_id)
            if not m:
                logger.debug(f'Directory {cell_id} does not match y_x pattern, skipping...')
                continue
            
            if cell_id in seen_cell_ids:
                logger.debug(f'Cell {cell_id} already seen in another sample dir, skipping duplicate...')
                continue
            
            # Check that cell has at least one valid multitemporal stack
            has_valid_stack = False
            for vv_file in cell_dir.glob('multi_*_vv.tif'):
                vh_file = vv_file.with_name(vv_file.name.replace('_vv.tif', '_vh.tif'))
                if vh_file.exists():
                    has_valid_stack = True
                    break
            
            if has_valid_stack:
                all_cells.append(cell_dir)
                seen_cell_ids.add(cell_id)
            else:
                logger.debug(f'Cell {cell_id} has no valid multi_*_vv/vh pairs, skipping...')

    logger.info(f'Found {len(all_cells)} cells with multitemporal SAR stacks for preprocessing')

    if len(all_cells) == 0:
        logger.error('No cells found with valid multitemporal SAR stacks. Exiting.')
        return

    # Split cells into train/val/test
    split_csv = getattr(cfg.preprocess, 'split_csv', None)
    if split_csv is not None:
        logger.info(f'Split provided by CSV file: {split_csv}')
        train_cells = []
        val_cells = []
        test_cells = []
        
        # Read CSV with columns: y, x, split
        coord_to_split = {}
        with open(split_csv, 'r') as f:
            reader = csv.DictReader(f)
            for row in reader:
                coord = (int(row['y']), int(row['x']))
                coord_to_split[coord] = row['split']

        for cell in all_cells:
            m = cell_pattern.match(cell.name)
            if m:
                y = int(m.group(1))
                x = int(m.group(2))
                match coord_to_split.get((y, x), None):
                    case 'train':
                        train_cells.append(cell)
                    case 'val':
                        val_cells.append(cell)
                    case 'test':
                        test_cells.append(cell)
                    case _:
                        logger.debug(f'Cell {cell.name} not assigned to any split, skipping...')
            else:
                logger.debug(f'Cell {cell.name} does not match pattern, skipping...')
    else:
        # Random splitting of cells
        val_ratio = getattr(cfg.preprocess, 'val_ratio', 0.1)
        test_ratio = getattr(cfg.preprocess, 'test_ratio', 0.1)
        logger.info(f'No split CSV provided, performing random splitting with val_ratio={val_ratio} and test_ratio={test_ratio}...')
        holdout_ratio = val_ratio + test_ratio
        if holdout_ratio <= 0 or holdout_ratio >= 1:
            raise ValueError('Sum of val_ratio and test_ratio must be in (0, 1).')

        assert getattr(cfg.preprocess, 'seed', None) is not None, 'cfg.preprocess.seed is required for random splitting.'

        train_cells, val_test_cells = train_test_split(
            all_cells, test_size=holdout_ratio, random_state=cfg.preprocess.seed
        )
        test_prop_within_holdout = test_ratio / holdout_ratio
        val_cells, test_cells = train_test_split(
            val_test_cells, test_size=test_prop_within_holdout, random_state=cfg.preprocess.seed + 1222
        )

    logger.info(f'Split: {len(train_cells)} train, {len(val_cells)} val, {len(test_cells)} test cells')

    # Save cell splits for reproducibility
    save_cell_splits(train_cells, val_cells, test_cells, cfg, pre_sample_dir, timestamp)

    # Get tile infos from cells (each multitemporal stack is one tile)
    logger.info('Grabbing multitemporal SAR tiles for splits...')
    acquisitions = cfg.preprocess.acquisitions
    train_tile_infos = get_tile_infos_from_cells(train_cells, acquisitions)
    val_tile_infos = get_tile_infos_from_cells(val_cells, acquisitions)
    test_tile_infos = get_tile_infos_from_cells(test_cells, acquisitions)

    logger.info(f'Tiles: {len(train_tile_infos)} train, {len(val_tile_infos)} val, {len(test_tile_infos)} test')

    if len(train_tile_infos) == 0:
        logger.error('No training tiles found. Exiting.')
        return

    # Compute statistics using parallel Welford's algorithm
    logger.info('Computing training statistics for SAR VV and VH channels...')
    mean, std, min_vals, max_vals = compute_statistics_parallel(train_tile_infos, n_workers)

    # Save training mean/std statistics
    stats_file = pre_sample_dir / 'mean_std.pkl'
    with open(stats_file, 'wb') as f:
        pickle.dump((mean, std), f)
    logger.info(f'Training mean std saved to {stats_file}')

    # Save training min/max statistics
    minmax_file = pre_sample_dir / 'min_max.pkl'
    with open(minmax_file, 'wb') as f:
        pickle.dump((min_vals, max_vals), f)
    logger.info(f'Training min max saved to {minmax_file}')

    # Sample patches in parallel
    missing_percent = getattr(cfg.preprocess, 'missing_percent', 0.0)
    chunk_size = getattr(cfg.preprocess, 'chunk_size', 100)
    scratch_dir = Path(cfg.preprocess.scratch_dir) if getattr(cfg.preprocess, 'scratch_dir', None) is not None else None
    if scratch_dir is not None:
        scratch_dir.mkdir(parents=True, exist_ok=True)
        logger.info(f'Using scratch directory for intermediate files: {scratch_dir}')

    if cfg.preprocess.method == 'random':
        for split_name, tile_infos in [('train', train_tile_infos), ('val', val_tile_infos), ('test', test_tile_infos)]:
            if len(tile_infos) == 0:
                logger.warning(f'No tiles for {split_name} split, skipping...')
                continue
            
            output_file = pre_sample_dir / f'{split_name}_patches.npy'
            logger.info(f'Processing {split_name} split with {len(tile_infos)} tiles...')
            
            sample_patches_parallel_random(
                pre_sample_dir, tile_infos, cfg.preprocess.size, cfg.preprocess.samples,
                acquisitions, missing_percent, output_file, cfg.preprocess.seed,
                n_workers, chunk_size=chunk_size, scratch_dir=scratch_dir
            )
        
        logger.info('Parallel random patch sampling complete.')
    elif cfg.preprocess.method == 'strided':
        for split_name, tile_infos in [('train', train_tile_infos), ('val', val_tile_infos), ('test', test_tile_infos)]:
            if len(tile_infos) == 0:
                logger.warning(f'No tiles for {split_name} split, skipping...')
                continue
            
            output_file = pre_sample_dir / f'{split_name}_patches.npy'
            logger.info(f'Processing {split_name} split with {len(tile_infos)} tiles...')
            
            sample_patches_parallel_strided(
                pre_sample_dir, tile_infos, cfg.preprocess.size, cfg.preprocess.stride,
                acquisitions, missing_percent, output_file, n_workers,
                chunk_size=chunk_size, scratch_dir=scratch_dir
            )
        
        logger.info('Parallel strided patch sampling complete.')
    else:
        raise ValueError(f"Unsupported sampling method: {cfg.preprocess.method}")
    
    # if homogenous patch csv provided, sample homogenous patches
    homogenous_csv = getattr(cfg.preprocess, 'homogenous_csv', None)
    if homogenous_csv is not None:
        logger.info(f'Homogenous patch CSV provided: {homogenous_csv}, '
                    'sampling homogenous patches for val and test splits...')
        coord_to_patch = defaultdict(list)
        with open(homogenous_csv, 'r') as f:
            reader = csv.DictReader(f)
            for row in reader:
                coord = (int(row['y']), int(row['x']))
                patch = (int(row['i']), int(row['j']))
                coord_to_patch[coord].append(patch)
        
        logger.info(f'Loaded {sum(len(v) for v in coord_to_patch.values())} homogenous patch locations '
                    f'across {len(coord_to_patch)} cells from CSV')
        
        # Filter tile_infos to only include cells that are in the CSV
        # This avoids iterating over all 500-1000 tiles when only 50-100 have homogenous patches
        homogenous_coords = set(coord_to_patch.keys())
        
        val_tile_infos_homogenous = [
            ti for ti in val_tile_infos
            if (int(ti[0].name.split('_')[0]), int(ti[0].name.split('_')[1])) in homogenous_coords
        ]
        test_tile_infos_homogenous = [
            ti for ti in test_tile_infos
            if (int(ti[0].name.split('_')[0]), int(ti[0].name.split('_')[1])) in homogenous_coords
        ]
        
        logger.info(f'Filtered to {len(val_tile_infos_homogenous)} val tiles and '
                    f'{len(test_tile_infos_homogenous)} test tiles with homogenous patch locations')
        
        # Sample val homogenous patches
        val_homogenous = sample_homogenous_patches(
            val_tile_infos_homogenous, cfg.preprocess.size, acquisitions,
            missing_percent, coord_to_patch
        )
        val_homogenous_file = pre_sample_dir / 'val_homogenous_patches.npy'
        np.save(val_homogenous_file, val_homogenous)
        logger.info(f'Saved {val_homogenous.shape[0]} val homogenous patches to {val_homogenous_file}')
        
        # Sample test homogenous patches
        test_homogenous = sample_homogenous_patches(
            test_tile_infos_homogenous, cfg.preprocess.size, acquisitions,
            missing_percent, coord_to_patch
        )
        test_homogenous_file = pre_sample_dir / 'test_homogenous_patches.npy'
        np.save(test_homogenous_file, test_homogenous)
        logger.info(f'Saved {test_homogenous.shape[0]} test homogenous patches to {test_homogenous_file}')

    logger.info('Preprocessing complete.')


if __name__ == '__main__':
    main()
