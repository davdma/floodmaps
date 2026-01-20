from omegaconf import DictConfig, OmegaConf
import hydra
import concurrent.futures 
from concurrent.futures import as_completed
import logging
import sys
import re
import rasterio
from pathlib import Path
import numpy as np
import torch
from floodmaps.models.model import build_autodespeckler
from floodmaps.utils.utils import load_model_weights
import pickle
import multiprocessing
from tqdm import tqdm

SAR_MISSING_VALUE = -9999

def filter_image(
    model,
    image,
    train_mean,
    train_std,
    stride=64,
    *,
    weighting="uniform",
    deterministic=True,
    batch_size=16,
    missing_value=SAR_MISSING_VALUE,
    eps=1e-3,
):
    """Despeckle a full VV/VH tile via dense sliding-window CVAE inference.

    This implements overlap blending (overlap-add) to avoid seams:
    - weighting="uniform": uniform averaging over overlaps
    - weighting="hann": weighted overlap-add with a 2D Hann window (clamped by eps)

    Parameters
    ----------
    model : torch.nn.Module
        The SAR CVAE despeckling model. Must expose model.inference(X, deterministic=...).
    image : np.ndarray
        Input SAR tile in dB, shape (2, H, W) with channels [VV, VH].
    train_mean : torch.Tensor
        CVAE training mean in dB, shape (2,).
    train_std : torch.Tensor
        CVAE training std in dB, shape (2,).
    stride : int
        Sliding-window stride in pixels (<= 64 recommended for overlap).
    weighting : str
        "uniform" or "hann".
    deterministic : bool
        If True, use deterministic inference (z=0); else sample z ~ N(0,1).
    batch_size : int
        Number of patches to run per forward pass.
    missing_value : float | None
        If provided, pixels equal to missing_value are treated as nodata.
    eps : float
        Minimum window weight (prevents divide-by-zero at borders for hann).

    Returns
    -------
    filtered_image : np.ndarray
        Despeckled tile in dB, shape (2, H, W).
    """
    patch_size = 64

    if not isinstance(image, np.ndarray):
        raise TypeError(f"image must be np.ndarray, got {type(image)}")
    if image.ndim != 3 or image.shape[0] != 2:
        raise ValueError(f"image must have shape (2, H, W); got {image.shape}")

    _, H, W = image.shape
    if H < patch_size or W < patch_size:
        raise ValueError(f"image too small for {patch_size}x{patch_size} patches: {H}x{W}")
    if stride <= 0:
        raise ValueError(f"stride must be positive; got {stride}")

    model.eval()

    # Pick device from model params (defaults to CPU if no params)
    try:
        device = next(model.parameters()).device
    except StopIteration:
        device = torch.device("cpu")

    mean = train_mean.to(device=device, dtype=torch.float32).view(1, 2, 1, 1)
    std = train_std.to(device=device, dtype=torch.float32).view(1, 2, 1, 1)

    # Build window weights
    if weighting == "uniform":
        win_2d = torch.ones((patch_size, patch_size), dtype=torch.float32)
    elif weighting == "hann":
        w1 = torch.hann_window(patch_size, periodic=False, dtype=torch.float32)
        win_2d = (w1[:, None] * w1[None, :]).clamp_min(float(eps))
    else:
        raise ValueError(f"Unknown weighting={weighting!r}. Use 'uniform' or 'hann'.")

    # Accumulators on CPU
    out_sum = torch.zeros((2, H, W), dtype=torch.float32)
    w_sum = torch.zeros((H, W), dtype=torch.float32)

    # Optional nodata mask
    nodata_mask = None
    if missing_value is not None:
        nodata_mask = (image[0] == missing_value) | (image[1] == missing_value)

    # Ensure full edge coverage
    xs = list(range(0, H - patch_size + 1, stride))
    ys = list(range(0, W - patch_size + 1, stride))
    if xs[-1] != H - patch_size:
        xs.append(H - patch_size)
    if ys[-1] != W - patch_size:
        ys.append(W - patch_size)

    positions = [(x, y) for x in xs for y in ys]

    with torch.no_grad():
        for start in range(0, len(positions), batch_size):
            batch_pos = positions[start : start + batch_size]

            # Build batch on CPU then move to device
            patches = []
            for x, y in batch_pos:
                patch = image[:, x : x + patch_size, y : y + patch_size].astype(np.float32, copy=False)

                if missing_value is not None:
                    # Replace nodata with mean in dB for inference stability
                    patch = patch.copy()
                    patch_mask = (patch[0] == missing_value) | (patch[1] == missing_value)
                    patch[0, patch_mask] = float(train_mean[0])
                    patch[1, patch_mask] = float(train_mean[1])

                patches.append(torch.from_numpy(patch))

            X = torch.stack(patches, dim=0).to(device=device, dtype=torch.float32)  # (B, 2, 64, 64) in dB
            Xn = (X - mean) / std

            out_dict = model.inference(Xn, deterministic=deterministic)
            Y = out_dict["despeckler_output"]  # (B, 2, 64, 64) normalized
            Y_db = (Y * std + mean).detach().cpu()  # (B, 2, 64, 64) in dB

            # Accumulate
            win = win_2d  # CPU
            for i, (x, y) in enumerate(batch_pos):
                out_sum[:, x : x + patch_size, y : y + patch_size] += Y_db[i] * win
                w_sum[x : x + patch_size, y : y + patch_size] += win

    filtered = out_sum / w_sum.clamp_min(float(eps))

    if nodata_mask is not None:
        filtered[:, torch.from_numpy(nodata_mask)] = float(missing_value)

    return filtered.numpy()


def apply_filter_to_event(event_paths: list):
    """Apply CVAE despeckling to all VV and VH SAR images in event directories
    and save to file for later patching.
    
    Uses global variables set by init_worker:
    - model: CVAE model for despeckling
    - train_mean: Training mean for VV, VH channels
    - train_std: Training std for VV, VH channels
    - cfg: Configuration object with preprocess parameters
    
    Parameters
    ----------
    event_paths : list
        List of event directory paths to process
    """
    logger = logging.getLogger('preprocessing')
    sar_p = re.compile(r'sar_(\d{8})_(\d{8})_(\d{8}_\d+_\d+)_vv.tif')
    
    # Get preprocess parameters from config
    replace = getattr(cfg.preprocess, 'replace', False)
    stride = getattr(cfg.preprocess, 'stride', 32)
    weighting = getattr(cfg.preprocess, 'weighting', 'hann')
    deterministic = getattr(cfg.preprocess, 'deterministic', True)
    batch_size = getattr(cfg.preprocess, 'batch_size', 128)
    
    for event_dir in event_paths:
        if not event_dir.is_dir():
            continue
            
        # Find all SAR VV images in the event directory
        for sar_vv_file in event_dir.glob("sar_*_vv.tif"):
            m = sar_p.match(sar_vv_file.name)
            if not m:
                logger.debug(f"Skipping file with unexpected naming: {sar_vv_file.name}")
                continue
            
            s2_img_dt = m.group(1)
            s1_img_dt = m.group(2)
            eid = m.group(3)
            
            # Output file paths
            cvae_vv_file = event_dir / f"cvae_{s2_img_dt}_{s1_img_dt}_{eid}_vv.tif"
            cvae_vh_file = event_dir / f"cvae_{s2_img_dt}_{s1_img_dt}_{eid}_vh.tif"
            
            # Skip if not replacing and outputs already exist
            if not replace and cvae_vv_file.exists() and cvae_vh_file.exists():
                continue
            
            # Load VV raster
            sar_vh_file = event_dir / f"sar_{s2_img_dt}_{s1_img_dt}_{eid}_vh.tif"
            if not sar_vh_file.exists():
                logger.warning(f"SAR VH file not found: {sar_vh_file}, skipping tile")
                continue
            
            try:
                with rasterio.open(sar_vv_file) as src_vv:
                    vv_tile = src_vv.read().astype(np.float32)  # (1, H, W)
                    vv_profile = src_vv.profile.copy()
                    nodata = src_vv.nodata
                
                with rasterio.open(sar_vh_file) as src_vh:
                    vh_tile = src_vh.read().astype(np.float32)  # (1, H, W)
                
                # Stack VV and VH as (2, H, W)
                stacked = np.vstack([vv_tile, vh_tile])  # (2, H, W)
                
                # Apply CVAE despeckling
                filtered = filter_image(
                    model,
                    stacked,
                    train_mean,
                    train_std,
                    stride=stride,
                    weighting=weighting,
                    deterministic=deterministic,
                    batch_size=batch_size,
                    missing_value=SAR_MISSING_VALUE
                )
                
                # Split back to VV and VH
                filtered_vv = filtered[0:1, :, :]  # (1, H, W)
                filtered_vh = filtered[1:2, :, :]  # (1, H, W)
                
                # Update profile for output
                vv_profile.update(dtype=np.float32, count=1, nodata=nodata)
                
                # Save VV
                with rasterio.open(cvae_vv_file, 'w', **vv_profile) as dst:
                    dst.write(filtered_vv)
                
                # Save VH
                with rasterio.open(cvae_vh_file, 'w', **vv_profile) as dst:
                    dst.write(filtered_vh)
                    
            except Exception as e:
                logger.error(f"Error processing {sar_vv_file}: {e}")
                continue

def init_worker(cfg_dict: dict, num_threads: int):
    """Set up CVAE model and necessary variables for worker.
    
    Sets global variables:
    - model: CVAE autodespeckler model
    - train_mean: Training mean for VV, VH channels (torch.Tensor of shape (2,))
    - train_std: Training std for VV, VH channels (torch.Tensor of shape (2,))
    - cfg: OmegaConf configuration object
    
    Parameters
    ----------
    cfg_dict : dict
        Configuration dictionary (converted from OmegaConf)
    num_threads : int
        Number of threads for torch operations
    """
    global model, cfg, train_mean, train_std

    torch.set_num_threads(num_threads)
    cfg = OmegaConf.create(cfg_dict)
    
    # Build and load CVAE model
    model = build_autodespeckler(cfg).to("cpu")

    # Load pretrained weights if specified
    if hasattr(cfg.model, 'weights') and cfg.model.weights is not None:
        load_model_weights(model, cfg.model.weights, "cpu", model_name="Autodespeckler")

    model.eval()
    
    # Build path to mean_std.pkl from CVAE training data
    method = cfg.data.method
    size = cfg.data.size
    sample_param = cfg.data.samples if cfg.data.method == 'random' else cfg.data.stride
    suffix = getattr(cfg.data, 'suffix', '')
    if suffix:
        sample_dir = Path(cfg.paths.preprocess_dir) / 's1_multi' / f'{method}_{size}_{sample_param}_{suffix}/'
    else:
        sample_dir = Path(cfg.paths.preprocess_dir) / 's1_multi' / f'{method}_{size}_{sample_param}/'

    # Load training mean and std (only VV, VH channels - first 2)
    with open(sample_dir / 'mean_std.pkl', 'rb') as f:
        loaded_mean, loaded_std = pickle.load(f)
        # Extract only VV and VH channels (indices 0 and 1)
        train_mean = torch.from_numpy(loaded_mean[:2].astype(np.float32))
        train_std = torch.from_numpy(loaded_std[:2].astype(np.float32))

@hydra.main(version_base=None, config_path="pkg://configs", config_name="config.yaml")
def main(cfg: DictConfig):
    """Script to apply CVAE despeckling to all VV and VH SAR images in
    event directories and save to file for later patching.
    
    cfg.preprocess Parameters:
    - sample_dirs : List[str] (list of sample directories under cfg.paths.imagery_dir)
    - weighting : str ["uniform", "hann"] (overlap blending method)
    - stride : int (stride for sliding window inference, default 32)
    - replace : bool (if True, replace existing CVAE despeckled images)
    - deterministic : bool (if True, use deterministic inference (z=0); else sample z ~ N(0,1))
    - n_workers : int (number of workers for parallel processing)
    - batch_size : int (number of patches to run per forward pass)
    - threads_per_worker : int (number of threads per worker)

    Recommended to pick n_workers and threads_per_worker such that
    n_workers * threads_per_worker = resources_used.ncpus, and so
    that you only use as many threads as needed to keep inference within a process
    fast while still allowing for high parallelization I/O and inference.

    For batch size try to maximize it for inference speed also. With 400 x 400
    tile shape, batch size of 128 pretty much batches the entire tile at once.

    Note: run with conda environment 'floodmaps-training'.

    Parameters
    ----------
    cfg : DictConfig
        Configuration object containing autodespeckler model and data parameters.
    """
    logger = logging.getLogger('preprocessing')
    logger.setLevel(logging.DEBUG)
    formatter = logging.Formatter('%(asctime)s - %(name)s - %(levelname)s - %(message)s')
    handler = logging.StreamHandler(sys.stdout)
    handler.setFormatter(formatter)
    handler.setLevel(logging.DEBUG)
    logger.addHandler(handler)
    logger.propagate = False

    # Get configuration parameters
    n_workers = getattr(cfg.preprocess, 'n_workers', 1)
    sample_dirs_list = cfg.preprocess.get('sample_dirs', [])
    replace = getattr(cfg.preprocess, 'replace', False)
    stride = getattr(cfg.preprocess, 'stride', 32)
    weighting = getattr(cfg.preprocess, 'weighting', 'hann')
    deterministic = getattr(cfg.preprocess, 'deterministic', True)
    batch_size = getattr(cfg.preprocess, 'batch_size', 128)
    threads_per_worker = getattr(cfg.preprocess, 'threads_per_worker', 8)
    
    logger.info(f'''Starting CVAE despeckling:
        Model:           {cfg.model.autodespeckler}
        Weights:         {getattr(cfg.model, 'weights', None)}
        Weighting:       {weighting}
        Stride:          {stride}
        Replace:         {replace}
        Deterministic:   {deterministic}
        Batch size:      {batch_size}
        Workers:         {n_workers}
        Threads/worker:  {threads_per_worker}
        Sample dirs:     {sample_dirs_list}
    ''')
    
    # Discover event directories from sample_dirs (relative to cfg.paths.imagery_dir)
    event_dirs = []
    for sample_dir in sample_dirs_list:
        sample_path = Path(cfg.paths.imagery_dir) / sample_dir
        if not sample_path.is_dir():
            logger.warning(f'Sample directory not found: {sample_path}')
            continue
        event_paths = list(sample_path.glob("[0-9]*_*_*"))
        event_dirs.extend(event_paths)
    
    if len(event_dirs) == 0:
        raise ValueError("No event directories found in sample_dirs")
    
    logger.info(f'Found {len(event_dirs)} event directories for processing')

    n_workers = min(n_workers, len(event_dirs))
    logger.info(f'Using {n_workers} workers for {len(event_dirs)} event directories...')

    # Split events into chunks for worker processes
    start_idx = 0
    events_per_worker = len(event_dirs) // n_workers
    remainder = len(event_dirs) % n_workers
    worker_event_paths = []
    for i in range(n_workers):
        end_idx = start_idx + events_per_worker + (1 if i < remainder else 0)
        event_paths = event_dirs[start_idx:end_idx]
        start_idx = end_idx
        worker_event_paths.append(event_paths)

    # Process event directories in parallel with progress tracking
    cfg_dict = OmegaConf.to_container(cfg, resolve=True)
    logger.info(f"Using {threads_per_worker} threads per worker")
    
    with concurrent.futures.ProcessPoolExecutor(
        max_workers=n_workers,
        initializer=init_worker, 
        initargs=(cfg_dict, threads_per_worker),
        mp_context=multiprocessing.get_context("forkserver")
    ) as executor:
        futures = [executor.submit(apply_filter_to_event, event_paths) for event_paths in worker_event_paths]
        for future in tqdm(as_completed(futures), total=len(futures), desc="Workers in progress"):
            try:
                future.result()
            except Exception as e:
                logger.error(f'Error applying CVAE despeckling: {e}')
    
    logger.info('CVAE despeckling complete.')


if __name__ == "__main__":
    main()