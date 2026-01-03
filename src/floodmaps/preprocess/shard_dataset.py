import numpy as np
from pathlib import Path
import sys
import logging
import shutil
import concurrent.futures
from typing import Tuple

import hydra
from omegaconf import DictConfig

# Binary convention: 1 GiB = 1024^3 bytes
BYTES_PER_GIB = 1024 ** 3


def write_shard(args: Tuple[Path, int, int, Path]) -> Tuple[int, int]:
    """Worker function to write a single shard.
    
    Parameters
    ----------
    args : Tuple[Path, int, int, Path]
        Tuple of (source_file, start_idx, end_idx, output_file)
        
    Returns
    -------
    Tuple[int, int]
        Tuple of (shard_idx, num_patches) for logging
    """
    source_file, start_idx, end_idx, output_file = args
    
    # Memory-map the source array and read the slice
    arr = np.load(source_file, mmap_mode='r')
    shard_data = np.array(arr[start_idx:end_idx])  # Load slice into memory
    np.save(output_file, shard_data)
    
    return end_idx - start_idx


@hydra.main(version_base=None, config_path='pkg://configs', config_name='config.yaml')
def main(cfg: DictConfig) -> None:
    """Shard the train dataset into smaller files for improved memory usage.
    
    cfg.preprocess Parameters
    - preprocess_dir: str (path to the preprocessed dataset containing train_patches.npy)
    - shard_mem_size: int (memory size of each shard in GiB, using binary convention 1024^3 bytes)
    - drop_last: bool (whether to drop patches to ensure shards are of equal size)
    - delete_original: bool (whether to delete the original train_patches.npy upon completion)
    - scratch_dir: str (optional, path to scratch directory for intermediate files)
    - n_workers: int (optional, number of parallel workers for writing shards)
    """
    # Setup logging
    logger = logging.getLogger('shard_dataset')
    logger.setLevel(logging.DEBUG)
    formatter = logging.Formatter('%(asctime)s - %(name)s - %(levelname)s - %(message)s')
    handler = logging.StreamHandler(sys.stdout)
    handler.setFormatter(formatter)
    handler.setLevel(logging.DEBUG)
    logger.addHandler(handler)
    logger.propagate = False

    # Parse config
    preprocess_dir = Path(cfg.preprocess.preprocess_dir)
    shard_mem_size = cfg.preprocess.shard_mem_size
    drop_last = cfg.preprocess.drop_last
    delete_original = cfg.preprocess.delete_original
    scratch_dir = Path(cfg.preprocess.scratch_dir) if getattr(cfg.preprocess, 'scratch_dir', None) else None
    n_workers = getattr(cfg.preprocess, 'n_workers', 1)

    # Validate paths
    train_patches_file = preprocess_dir / 'train_patches.npy'
    if not train_patches_file.exists():
        logger.error(f'train_patches.npy not found at {train_patches_file}')
        raise FileNotFoundError(f'train_patches.npy not found at {train_patches_file}')

    if scratch_dir is not None:
        scratch_dir.mkdir(parents=True, exist_ok=True)
        logger.info(f'Using scratch directory: {scratch_dir}')

    logger.info(f'''Starting dataset sharding:
        Preprocess dir:  {preprocess_dir}
        Shard mem size:  {shard_mem_size} GiB
        Drop last:       {drop_last}
        Delete original: {delete_original}
        Scratch dir:     {scratch_dir}
        Workers:         {n_workers}
    ''')

    # Load array metadata using memory-mapped mode
    arr = np.load(train_patches_file, mmap_mode='r')
    N, C, H, W = arr.shape
    dtype = arr.dtype
    
    logger.info(f'Loaded train_patches.npy: shape={arr.shape}, dtype={dtype}')

    # Calculate memory per patch
    bytes_per_patch = C * H * W * dtype.itemsize
    logger.info(f'Bytes per patch: {bytes_per_patch:,} bytes ({C} x {H} x {W} x {dtype.itemsize} bytes)')

    # Calculate patches per shard based on target GiB size
    shard_bytes = shard_mem_size * BYTES_PER_GIB
    patches_per_shard = shard_bytes // bytes_per_patch
    
    if patches_per_shard == 0:
        logger.error(f'Shard size {shard_mem_size} GiB is too small for patches of size {bytes_per_patch:,} bytes')
        raise ValueError(f'Shard size too small: need at least {bytes_per_patch / BYTES_PER_GIB:.6f} GiB per patch')

    # Calculate number of shards and remainder
    num_shards = N // patches_per_shard
    remainder = N % patches_per_shard

    logger.info(f'Sharding calculation:')
    logger.info(f'  Total patches (N):      {N:,}')
    logger.info(f'  Patches per shard (B):  {patches_per_shard:,}')
    logger.info(f'  Number of shards (k):   {num_shards}')
    logger.info(f'  Remainder (r):          {remainder:,}')

    if num_shards == 0:
        logger.error(f'Not enough patches ({N:,}) to create even one shard of {patches_per_shard:,} patches')
        raise ValueError(f'Not enough patches to create a shard')

    # Build shard assignments based on drop_last setting
    shard_assignments = []  # List of (start_idx, end_idx, shard_idx)
    
    if drop_last:
        # All shards have exactly patches_per_shard patches, remainder is dropped
        logger.info(f'drop_last=True: Creating {num_shards} shards of {patches_per_shard:,} patches each')
        if remainder > 0:
            logger.info(f'  Dropping {remainder:,} remainder patches')
        
        for i in range(num_shards):
            start_idx = i * patches_per_shard
            end_idx = start_idx + patches_per_shard
            shard_assignments.append((start_idx, end_idx, i))
    else:
        # Distribute remainder across first 'remainder' shards
        logger.info(f'drop_last=False: Creating {num_shards} shards, distributing {remainder:,} remainder patches')
        logger.info(f'  Base shard size: {patches_per_shard:,} patches')
        if remainder > 0:
            logger.info(f'  First {remainder} shards will have {patches_per_shard + 1:,} patches each')
        
        current_idx = 0
        for i in range(num_shards):
            # First 'remainder' shards get one extra patch
            shard_size = patches_per_shard + (1 if i < remainder else 0)
            start_idx = current_idx
            end_idx = current_idx + shard_size
            shard_assignments.append((start_idx, end_idx, i))
            current_idx = end_idx

    # Release mmap reference before parallel processing
    del arr

    # Determine output directory (scratch or final)
    output_dir = scratch_dir if scratch_dir is not None else preprocess_dir

    # Clean up any existing shard files in output directory
    for existing_shard in output_dir.glob('train_patches_*.npy'):
        try:
            existing_shard.unlink()
            logger.debug(f'Removed existing shard: {existing_shard.name}')
        except Exception as e:
            logger.warning(f'Failed to remove existing shard {existing_shard}: {e}')

    # Also clean up in final directory if using scratch
    if scratch_dir is not None:
        for existing_shard in preprocess_dir.glob('train_patches_*.npy'):
            try:
                existing_shard.unlink()
                logger.debug(f'Removed existing shard in final dir: {existing_shard.name}')
            except Exception as e:
                logger.warning(f'Failed to remove existing shard {existing_shard}: {e}')

    # Build worker arguments
    worker_args = []
    for start_idx, end_idx, shard_idx in shard_assignments:
        output_file = output_dir / f'train_patches_{shard_idx}.npy'
        worker_args.append((train_patches_file, start_idx, end_idx, output_file))

    # Write shards in parallel
    logger.info(f'Writing {len(shard_assignments)} shards using {min(n_workers, len(shard_assignments))} workers...')
    
    try:
        with concurrent.futures.ProcessPoolExecutor(max_workers=n_workers) as executor:
            futures = [executor.submit(write_shard, args) for args in worker_args]
            
            for i, future in enumerate(concurrent.futures.as_completed(futures)):
                try:
                    num_patches = future.result()
                    logger.debug(f'Completed shard {i}: {num_patches:,} patches')
                except Exception as e:
                    logger.error(f'Failed to write shard: {e}')
                    raise RuntimeError(f'Shard writing failed: {e}') from e
    except Exception as e:
        logger.error(f'Failed during parallel shard writing: {e}')
        raise RuntimeError(f'Sharding failed: {e}') from e

    # Move shards from scratch to final directory if needed
    if scratch_dir is not None:
        logger.info(f'Moving shards from scratch to {preprocess_dir}...')
        for _, _, shard_idx in shard_assignments:
            scratch_file = scratch_dir / f'train_patches_{shard_idx}.npy'
            final_file = preprocess_dir / f'train_patches_{shard_idx}.npy'
            try:
                shutil.move(str(scratch_file), str(final_file))
            except Exception as e:
                logger.error(f'Failed to move shard {shard_idx} from scratch: {e}')
                raise RuntimeError(f'Failed to move shard: {e}') from e

    # Delete original file if requested
    if delete_original:
        logger.info(f'Deleting original train_patches.npy...')
        try:
            train_patches_file.unlink()
            logger.info('Original file deleted successfully.')
        except Exception as e:
            logger.error(f'Failed to delete original file: {e}')
            raise RuntimeError(f'Failed to delete original: {e}') from e

    # Summary
    total_sharded = sum(end - start for start, end, _ in shard_assignments)
    logger.info(f'''Sharding complete:
        Shards created:    {len(shard_assignments)}
        Total patches:     {total_sharded:,}
        Patches dropped:   {N - total_sharded:,}
        Output directory:  {preprocess_dir}
    ''')


if __name__ == '__main__':
    main()
