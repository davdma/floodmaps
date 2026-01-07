import torch
import concurrent.futures
from concurrent.futures import as_completed
from itertools import product
from typing import List
from torchvision import transforms
import numpy as np
import re
import rasterio
import pickle
from pathlib import Path
import hydra
from omegaconf import DictConfig, OmegaConf
import multiprocessing
from tqdm import tqdm

from floodmaps.utils.utils import SARChannelIndexer
from floodmaps.models.model import SARWaterDetector
from floodmaps.utils.preprocess_utils import impute_missing_values

# SAR nodata value after dB scale conversion (from download_aoi.py pipeline_S1)
SAR_NODATA = -9999


def generate_prediction_sar(model, device, cfg: DictConfig, standardize, event_path, dt, eid, threshold=0.5, batch_size=128):
    """Generate predictions on unseen data using SAR water detector model.
    
    Predictions are made using a sliding window with 25% overlap.
    Missing data is imputed with tile-specific mean of each channel.
    Missing mask is saved to zero out predictions in final output.

    Parameters
    ----------
    model : obj
        Model object for inference.
    cfg : DictConfig
        Configuration object containing model and data parameters.
    standardize : obj
        Standardization of input channels before being fed into the model.
    event_path : Path
        Path to the event directory where the raw sample data is stored.
    dt : str
        Date that the SAR capture was taken.
    eid : str
        Event ID of the sample.
    threshold : float
        Threshold for the prediction.
    batch_size : int
        Batch size for inference.

    Returns
    -------
    label : ndarray
        Predicted label of the specified tile in shape (H, W).
    """
    layers = []
    my_channels = SARChannelIndexer([bool(int(x)) for x in cfg.data.channels])
    
    # Initialize total missing values mask
    tot_missing_vals = None
    
    # Load VV channel
    if my_channels.has_vv():
        vv_file = event_path / f'sar_{dt}_{eid}_vv.tif'
        with rasterio.open(vv_file) as src:
            vv_tile = src.read().astype(np.float32)
        
        # SAR nodata is -9999 (after dB scale)
        vv_missing = (vv_tile == SAR_NODATA)
        if not vv_missing.all():
            vv_tile = impute_missing_values(vv_tile, vv_missing)
        else:
            # If all values missing, set to 0 (will be zeroed out anyway)
            vv_tile[vv_missing] = 0
        
        if tot_missing_vals is None:
            tot_missing_vals = vv_missing.squeeze(axis=0)
        else:
            tot_missing_vals = tot_missing_vals | vv_missing.squeeze(axis=0)
        
        layers.append(vv_tile)
    
    # Load VH channel
    if my_channels.has_vh():
        vh_file = event_path / f'sar_{dt}_{eid}_vh.tif'
        with rasterio.open(vh_file) as src:
            vh_tile = src.read().astype(np.float32)
        
        # SAR nodata is -9999 (after dB scale)
        vh_missing = (vh_tile == SAR_NODATA)
        if not vh_missing.all():
            vh_tile = impute_missing_values(vh_tile, vh_missing)
        else:
            # If all values missing, set to 0 (will be zeroed out anyway)
            vh_tile[vh_missing] = 0
        
        if tot_missing_vals is None:
            tot_missing_vals = vh_missing.squeeze(axis=0)
        else:
            tot_missing_vals = tot_missing_vals | vh_missing.squeeze(axis=0)
        
        layers.append(vh_tile)

    # Load DEM (needed for slope regardless if in channels or not)
    dem_file = event_path / f'dem_{eid}.tif'
    with rasterio.open(dem_file) as src:
        dem_tile = src.read().astype(np.float32)
    
    if my_channels.has_dem():
        layers.append(dem_tile)

    # Compute slope from DEM
    slope = np.gradient(dem_tile, axis=(1, 2))
    slope_y_tile, slope_x_tile = slope
    
    if my_channels.has_slope_y():
        layers.append(slope_y_tile)
    if my_channels.has_slope_x():
        layers.append(slope_x_tile)
    
    # Load waterbody (binary mask, nodata=0)
    if my_channels.has_waterbody():
        waterbody_file = event_path / f'waterbody_{eid}.tif'
        with rasterio.open(waterbody_file) as src:
            waterbody_tile = src.read().astype(np.float32)
        layers.append(waterbody_tile)
    
    # Load roads (binary mask, nodata=0)
    if my_channels.has_roads():
        roads_file = event_path / f'roads_{eid}.tif'
        with rasterio.open(roads_file) as src:
            roads_tile = src.read().astype(np.float32)
        layers.append(roads_tile)
    
    # Load flowlines (binary mask, nodata=0)
    if my_channels.has_flowlines():
        flowlines_file = event_path / f'flowlines_{eid}.tif'
        with rasterio.open(flowlines_file) as src:
            flowlines_tile = src.read().astype(np.float32)
        layers.append(flowlines_tile)

    X = np.vstack(layers, dtype=np.float32)
    
    # Initialize missing mask if not set (no VV/VH channels)
    if tot_missing_vals is None:
        tot_missing_vals = np.zeros(X.shape[-2:], dtype=bool)

    X = torch.from_numpy(X)
    X = X.to(device)
    X = standardize(X)

    # Prediction done using sliding window with overlap
    patch_size = cfg.data.window  # Use window size (model input) not data size (training patch)
    overlap = patch_size // 4  # 25% overlap
    stride = patch_size - overlap

    # Tile discretely and make predictions
    H = X.shape[-2]
    W = X.shape[-1]
    if H < patch_size or W < patch_size:
        raise ValueError(f"Image dimensions must be at least {patch_size}x{patch_size}")

    # Initialize output prediction map (accumulate probabilities, not binaries)
    pred_map = torch.zeros((H, W), dtype=torch.float32, device=device)
    count_map = torch.zeros((H, W), dtype=torch.float32, device=device)

    with torch.no_grad():
        # Build list of y positions
        lst_y = list(range(0, H - patch_size + 1, stride))
        if lst_y[-1] != H - patch_size:
            lst_y.append(H - patch_size)

        # Build list of x positions
        lst_x = list(range(0, W - patch_size + 1, stride))
        if lst_x[-1] != W - patch_size:
            lst_x.append(W - patch_size)

        # Pre-generate all patch positions and collect patches
        patches = []
        positions = list(product(lst_y, lst_x))
        for y, x in positions:
            patches.append(X[:, y:y+patch_size, x:x+patch_size])

        all_patches = torch.stack(patches, dim=0)
        
        # Batch inference
        for i in range(0, all_patches.shape[0], batch_size):
            end_idx = min(i + batch_size, all_patches.shape[0])
            batch = all_patches[i:end_idx]

            output = model(batch)
            if isinstance(output, dict):
                pred = output['classifier_output'].squeeze(1)
            else:
                pred = output.squeeze(1)
            
            # Accumulate probabilities
            prob = torch.sigmoid(pred).float()

            for j in range(end_idx - i):
                cur_y, cur_x = positions[i + j]
                pred_map[cur_y:cur_y+patch_size, cur_x:cur_x+patch_size] += prob[j]
                count_map[cur_y:cur_y+patch_size, cur_x:cur_x+patch_size] += 1.0
    
    # Average overlapping predictions
    pred_map = torch.where(count_map > 0, pred_map / count_map, pred_map)
    
    # Convert to final binary prediction using provided threshold
    label = torch.where(pred_map >= threshold, 1.0, 0.0).byte().cpu().numpy()
    label[tot_missing_vals] = 0
    return label


def init_worker(cfg_dict: dict, num_threads: int):
    """Set up model and necessary variables for worker process.
    
    Parameters
    ----------
    cfg_dict : dict
        Configuration dictionary (serializable).
    num_threads : int
        Number of threads for this worker.
    """
    global model, cfg, standardize

    torch.set_num_threads(num_threads)
    cfg = OmegaConf.create(cfg_dict)
    
    ad_cfg = getattr(cfg, 'ad', None)
    model = SARWaterDetector(cfg, ad_cfg=ad_cfg).to("cpu")
    model.eval()
    
    # Build sample directory path
    method = cfg.data.method
    size = cfg.data.size
    sample_param = cfg.data.samples if cfg.data.method == 'random' else cfg.data.stride
    suffix = getattr(cfg.data, 'suffix', '')
    
    if suffix:
        sample_dir = Path(cfg.paths.preprocess_dir) / 's1_weak' / f'{method}_{size}_{sample_param}_{suffix}/'
    else:
        sample_dir = Path(cfg.paths.preprocess_dir) / 's1_weak' / f'{method}_{size}_{sample_param}/'

    channels = [bool(int(x)) for x in cfg.data.channels]
    with open(sample_dir / f'mean_std.pkl', 'rb') as f:
        train_mean, train_std = pickle.load(f)

        train_mean = torch.from_numpy(train_mean[channels])
        train_std = torch.from_numpy(train_std[channels])

    standardize = transforms.Compose([transforms.Normalize(train_mean, train_std)])


def label_captures(sar_vv_files: List[Path]):
    """Worker function that inferences SAR captures and saves predictions to file.
    
    Parameters
    ----------
    sar_vv_files : List[Path]
        List of SAR VV files (captures) to predict.
    """
    p = re.compile(r'sar_(\d{8})_(.+)_vv\.tif')
    data_path = Path(cfg.inference.data_dir)
    
    for sar_vv_file in sar_vv_files:
        m = p.search(sar_vv_file.name)
        if not m:
            continue
            
        dt = m.group(1)
        eid = m.group(2)

        # Skip if prediction already exists and replace is False
        if not cfg.inference.replace and (data_path / f'pred_sar_{dt}_{eid}.tif').exists():
            continue

        pred = generate_prediction_sar(
            model, "cpu", cfg, standardize, data_path,
            dt, eid, threshold=cfg.inference.threshold,
            batch_size=getattr(cfg.inference, 'batch_size', 128)
        )

        if cfg.inference.format == "tif":
            # Save result of prediction as .tif file with geotransform
            with rasterio.open(sar_vv_file) as src:
                transform = src.transform
                crs = src.crs

            mult_pred = pred * 255
            broadcasted = np.broadcast_to(mult_pred, (3, *pred.shape)).astype(np.uint8)
            with rasterio.open(
                data_path / f'pred_sar_{dt}_{eid}.tif', 'w',
                driver='Gtiff', count=3,
                height=pred.shape[-2], width=pred.shape[-1],
                dtype=np.uint8, crs=crs, transform=transform
            ) as dst:
                dst.write(broadcasted)
        elif cfg.inference.format == "npy":
            np.save(data_path / f'pred_sar_{dt}_{eid}.npy', pred)
        else:
            raise Exception("format unknown")


def run_inference_sar(cfg: DictConfig):
    """Generates inference labels on downloaded data from download_aoi.py using SAR model.

    NOTE: rasters are expected to have eid defined in file names i.e. sar_(date)_(eid)_vv.tif.

    Important cfg.inference parameters:
    cfg.inference.data_dir: Input directory containing SAR rasters
    cfg.inference.n_workers: Number of worker processes
    cfg.inference.threads_per_worker: Number of threads per worker
    cfg.inference.batch_size: Batch size for inference
    cfg.inference.threshold: Prediction threshold (default 0.5)
    cfg.inference.replace: Whether to overwrite existing predictions
    cfg.inference.format: Output format ("tif" or "npy")

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
        Configuration object containing model and data parameters.
    """
    # Get list of SAR VV captures to predict
    lst = list(Path(cfg.inference.data_dir).glob('sar_*_vv.tif'))
    
    if len(lst) == 0:
        print(f"No SAR VV files found in {cfg.inference.data_dir}")
        return

    # Split into chunks for worker processes
    n_workers = getattr(cfg.inference, 'n_workers', 1)
    n_workers = min(n_workers, len(lst))
    print(f'Using {n_workers} workers for {len(lst)} SAR captures...')

    start_idx = 0
    captures_per_worker = len(lst) // n_workers
    remainder = len(lst) % n_workers
    worker_capture_paths = []
    for i in range(n_workers):
        end_idx = start_idx + captures_per_worker + (1 if i < remainder else 0)
        capture_paths = lst[start_idx:end_idx]
        start_idx = end_idx
        worker_capture_paths.append(capture_paths)

    # Prepare config for serialization
    cfg_dict = OmegaConf.to_container(cfg, resolve=True)
    threads_per_worker = getattr(cfg.inference, 'threads_per_worker', 8)
    print(f"Using {threads_per_worker} threads per worker")
    
    with concurrent.futures.ProcessPoolExecutor(
        max_workers=n_workers,
        initializer=init_worker,
        initargs=(cfg_dict, threads_per_worker),
        mp_context=multiprocessing.get_context("forkserver")
    ) as executor:
        futures = [executor.submit(label_captures, capture_paths) for capture_paths in worker_capture_paths]
        for fut in tqdm(as_completed(futures), total=len(futures), desc="Workers in progress"):
            result = fut.result()


@hydra.main(version_base=None, config_path="pkg://configs", config_name="config.yaml")
def main(cfg: DictConfig):
    """Main entry point for inference on spatiotemporal data using SAR model.
    Generates predictions in specified folder for specific area of interest using S1 model.

    NOTE: The script ingests data downloaded from download_aoi.py, and assumes its
    file labeling convention, with sar files in the form sar_[dt]_[eid]_vv.tif.

    Parameters
    ----------
    cfg : DictConfig
        Configuration object containing model and data parameters.
    """
    run_inference_sar(cfg)


if __name__ == '__main__':
    main()
