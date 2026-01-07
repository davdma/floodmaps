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
from datetime import datetime
import multiprocessing
from tqdm import tqdm

from floodmaps.utils.utils import ChannelIndexer
from floodmaps.models.model import S2WaterDetector
from floodmaps.utils.preprocess_utils import (
    PROCESSING_BASELINE_NAIVE,
    BOA_ADD_OFFSET,
    compute_awei_sh,
    compute_awei_nsh,
    compute_ndwi,
    compute_mndwi,
    impute_missing_values
)

def generate_prediction_s2(model, device, cfg: DictConfig, standardize, event_path, dt, eid, threshold=0.5, batch_size=128):
    """Generate new predictions on unseen data using water detector model.
    Predictions are made using a sliding window with 25% overlap.

    For missing data, impute with tile specific mean of each channel in order to allow for inference.
    Save missing mask in order to zero out predictions in final output.

    NOTE: Negative reflectances are clipped to 0 but not treated as missing values.
    Only explicit DN 0 and undefined water indices are treated as missing values and imputed.

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
        Date that the RGB image was taken.
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
    img_dt_obj = datetime.strptime(dt, '%Y%m%d')
    layers = []
    my_channels = ChannelIndexer([bool(int(x)) for x in cfg.data.channels])

    rgb_file = event_path / f'rgb_{dt}_{eid}.tif'
    with rasterio.open(rgb_file) as src:
        # Rgb tile is 3, H, W
        rgb_tile = src.read().astype(np.float32)
    
        # Extract initial missing values (raw DN 0 = initial missing)
        tot_missing_vals = np.zeros(rgb_tile[0].shape, dtype=bool)
        
        if img_dt_obj >= PROCESSING_BASELINE_NAIVE:
            rgb_tile_sr = np.clip((rgb_tile + BOA_ADD_OFFSET) / 10000.0, 0, 1)
        else:
            rgb_tile_sr = np.clip(rgb_tile / 10000.0, 0, 1)
        rgb_tile_missing = (rgb_tile == 0)
        rgb_tile_sr = impute_missing_values(rgb_tile_sr, rgb_tile_missing)

    if my_channels.has_rgb():
        tot_missing_vals = tot_missing_vals | rgb_tile_missing.any(axis=0)
        layers.append(rgb_tile_sr)
    
    b08_file = event_path / f'b08_{dt}_{eid}.tif'
    with rasterio.open(b08_file) as src:
        b08_tile = src.read().astype(np.float32)
        if img_dt_obj >= PROCESSING_BASELINE_NAIVE:
            b08_tile_sr = np.clip((b08_tile + BOA_ADD_OFFSET) / 10000.0, 0, 1)
        else:
            b08_tile_sr = np.clip(b08_tile / 10000.0, 0, 1)
        b08_tile_missing = (b08_tile == 0)
        b08_tile_sr = impute_missing_values(b08_tile_sr, b08_tile_missing)

    if my_channels.has_b08():
        tot_missing_vals = tot_missing_vals | b08_tile_missing.squeeze(axis=0)
        layers.append(b08_tile_sr)

    # Load SWIR1 (B11)
    b11_file = event_path / f'b11_{dt}_{eid}.tif'
    with rasterio.open(b11_file) as src:
        b11_tile = src.read().astype(np.float32)
        if img_dt_obj >= PROCESSING_BASELINE_NAIVE:
            b11_tile_sr = np.clip((b11_tile + BOA_ADD_OFFSET) / 10000.0, 0, 1)
        else:
            b11_tile_sr = np.clip(b11_tile / 10000.0, 0, 1)
        b11_tile_missing = (b11_tile == 0)
        b11_tile_sr = impute_missing_values(b11_tile_sr, b11_tile_missing)

    if my_channels.has_swir1():
        tot_missing_vals = tot_missing_vals | b11_tile_missing.squeeze(axis=0)
        layers.append(b11_tile_sr)

    # Load SWIR2 (B12)
    b12_file = event_path / f'b12_{dt}_{eid}.tif'
    with rasterio.open(b12_file) as src:
        b12_tile = src.read().astype(np.float32)
        if img_dt_obj >= PROCESSING_BASELINE_NAIVE:
            b12_tile_sr = np.clip((b12_tile + BOA_ADD_OFFSET) / 10000.0, 0, 1)
        else:
            b12_tile_sr = np.clip(b12_tile / 10000.0, 0, 1)
        b12_tile_missing = (b12_tile == 0)
        b12_tile_sr = impute_missing_values(b12_tile_sr, b12_tile_missing)

    if my_channels.has_swir2():
        tot_missing_vals = tot_missing_vals | b12_tile_missing.squeeze(axis=0)
        layers.append(b12_tile_sr)

    if my_channels.has_ndwi():
        # Recompute NDWI using surface reflectance: (Green - NIR) / (Green + NIR)
        ndwi_tile = compute_ndwi(rgb_tile_sr[1], b08_tile_sr[0], missing_val=-999999)
        ndwi_missing = rgb_tile_missing[1] | b08_tile_missing[0] | (ndwi_tile == -999999)
        ndwi_tile = impute_missing_values(ndwi_tile, ndwi_missing)
        tot_missing_vals = tot_missing_vals | ndwi_missing
        ndwi_tile = np.expand_dims(ndwi_tile, axis = 0)
        layers.append(ndwi_tile)

    # Compute MNDWI (Modified NDWI): (Green - SWIR1) / (Green + SWIR1)
    if my_channels.has_mndwi():
        mndwi_tile = compute_mndwi(rgb_tile_sr[1], b11_tile_sr[0], missing_val=-999999)
        mndwi_missing = rgb_tile_missing[1] | b11_tile_missing[0] | (mndwi_tile == -999999)
        mndwi_tile = impute_missing_values(mndwi_tile, mndwi_missing)
        tot_missing_vals = tot_missing_vals | mndwi_missing
        mndwi_tile = np.expand_dims(mndwi_tile, axis=0)
        layers.append(mndwi_tile)

    # Compute AWEI_sh (Automated Water Extraction Index - shadow): Blue + 2.5*Green - 1.5*(NIR + SWIR1) - 0.25*SWIR2
    if my_channels.has_awei_sh():
        awei_sh_tile = compute_awei_sh(rgb_tile_sr[2], rgb_tile_sr[1], b08_tile_sr[0], b11_tile_sr[0], b12_tile_sr[0])
        awei_sh_missing = rgb_tile_missing[2] | rgb_tile_missing[1] | b08_tile_missing[0] | b11_tile_missing[0] | b12_tile_missing[0]
        awei_sh_tile = impute_missing_values(awei_sh_tile, awei_sh_missing)
        tot_missing_vals = tot_missing_vals | awei_sh_missing
        awei_sh_tile = np.expand_dims(awei_sh_tile, axis=0)
        layers.append(awei_sh_tile)

    # Compute AWEI_nsh (Automated Water Extraction Index - no shadow): 4*(Green - SWIR1) - (0.25*NIR + 2.75*SWIR2)
    if my_channels.has_awei_nsh():
        awei_nsh_tile = compute_awei_nsh(rgb_tile_sr[1], b11_tile_sr[0], b08_tile_sr[0], b12_tile_sr[0])
        awei_nsh_missing = rgb_tile_missing[1] | b11_tile_missing[0] | b08_tile_missing[0] | b12_tile_missing[0]
        awei_nsh_tile = impute_missing_values(awei_nsh_tile, awei_nsh_missing)
        tot_missing_vals = tot_missing_vals | awei_nsh_missing
        awei_nsh_tile = np.expand_dims(awei_nsh_tile, axis=0)
        layers.append(awei_nsh_tile)

    # need dem for slope regardless if in channels or not
    dem_file = event_path / f'dem_{eid}.tif'
    with rasterio.open(dem_file) as src:
        dem_tile = src.read().astype(np.float32)
    if my_channels.has_dem():
        layers.append(dem_tile)

    slope = np.gradient(dem_tile, axis=(1,2))
    slope_y_tile, slope_x_tile = slope
    if my_channels.has_slope_y():
        layers.append(slope_y_tile)
    if my_channels.has_slope_x():
        layers.append(slope_x_tile)
    if my_channels.has_waterbody():
        waterbody_file = event_path / f'waterbody_{eid}.tif'
        with rasterio.open(waterbody_file) as src:
            waterbody_tile = src.read().astype(np.float32)
        layers.append(waterbody_tile)
    if my_channels.has_roads():
        roads_file = event_path / f'roads_{eid}.tif'
        with rasterio.open(roads_file) as src:
            roads_tile = src.read().astype(np.float32)
        layers.append(roads_tile)
    if my_channels.has_flowlines():
        flowlines_file = event_path / f'flowlines_{eid}.tif'
        with rasterio.open(flowlines_file) as src:
            flowlines_tile = src.read().astype(np.float32)
        layers.append(flowlines_tile)

    X = np.vstack(layers, dtype=np.float32)

    X = torch.from_numpy(X)
    X = X.to(device)
    X = standardize(X)

    # prediction done using sliding window with overlap
    patch_size = cfg.data.size
    overlap = patch_size // 4  # 25% overlap
    stride = patch_size - overlap

    # tile discretely and make predictions
    H = X.shape[-2]
    W = X.shape[-1]
    if H < patch_size or W < patch_size:
        raise ValueError(f"Image dimensions must be at least {patch_size}x{patch_size}")

    # Initialize output prediction map (accumulate probabilities, not binaries)
    pred_map = torch.zeros((H, W), dtype=torch.float32, device=device)
    count_map = torch.zeros((H, W), dtype=torch.float32, device=device)

    with torch.no_grad():
        lst_y = list(range(0, H - patch_size + 1, stride))
        if lst_y[-1] != H - patch_size:
            lst_y.append(H - patch_size)

        lst_x = list(range(0, W - patch_size + 1, stride))
        if lst_x[-1] != W - patch_size:
            lst_x.append(W - patch_size)

        patches = []
        positions = list(product(lst_y, lst_x))
        for y, x in positions:
            # Stack the patches
            patches.append(X[:, y:y+patch_size, x:x+patch_size])

        all_patches = torch.stack(patches, dim=0)
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
    
    # Convert to final binary prediction
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
    model = S2WaterDetector(cfg).to("cpu")
    model.eval()
    method = cfg.data.method
    size = cfg.data.size
    sample_param = cfg.data.samples if cfg.data.method == 'random' else cfg.data.stride
    suffix = getattr(cfg.data, 'suffix', '') # for old model can use suffix=deprecated
    use_weak = getattr(cfg.data, 'use_weak', False)
    dataset_type = 's2_weak' if use_weak else 's2'
    if suffix:
        sample_dir = Path(cfg.paths.preprocess_dir) / dataset_type / f'{method}_{size}_{sample_param}_{suffix}/'
    else:
        sample_dir = Path(cfg.paths.preprocess_dir) / dataset_type / f'{method}_{size}_{sample_param}/'

    channels = [bool(int(x)) for x in cfg.data.channels]
    with open(sample_dir / f'mean_std.pkl', 'rb') as f:
        train_mean, train_std = pickle.load(f)

        train_mean = torch.from_numpy(train_mean[channels])
        train_std = torch.from_numpy(train_std[channels])

    standardize = transforms.Compose([transforms.Normalize(train_mean, train_std)])

def label_captures(rgb_files: List[Path]):
    """Worker function that inferences S2 captures and saves predictions to file.
    
    Parameters
    ----------
    rgb_files : List[Path]
        List of S2 rgb files (captures) to predict.
    """
    # get eid then use to find the rgb dates for each event
    p = re.compile('rgb_(\d{8})_(.+).tif')
    data_path = Path(cfg.inference.data_dir)
    for rgb_file in rgb_files:
        m = p.search(rgb_file.name)
        dt = m.group(1)
        eid = m.group(2)

        if not cfg.inference.replace and (data_path / f'pred_{dt}_{eid}.tif').exists():
            continue

        pred = generate_prediction_s2(model, "cpu", cfg, standardize, data_path,
                                        dt, eid, threshold=cfg.inference.threshold,
                                        batch_size=getattr(cfg.inference, 'batch_size', 256))

        if cfg.inference.format == "tif":
            # save result of prediction as .tif file
            # add transform
            with rasterio.open(rgb_file) as src:
                transform = src.transform
                crs = src.crs

            mult_pred = pred * 255
            broadcasted = np.broadcast_to(mult_pred, (3, *pred.shape)).astype(np.uint8)
            with rasterio.open(data_path / f'pred_{dt}_{eid}.tif', 'w', driver='Gtiff', count=3, height=pred.shape[-2], width=pred.shape[-1], dtype=np.uint8, crs=crs, transform=transform) as dst:
                dst.write(broadcasted)
        elif cfg.inference.format == "npy":
            np.save(data_path / f'pred_{dt}_{eid}.npy', pred)
        else:
            raise Exception("format unknown")

def run_inference_s2(cfg: DictConfig):
    """Generates inference labels on a downloaded time frame from download_aoi.py
    using the rgb S2 optical model.

    NOTE: rasters are expected to have eid defined in file names i.e. rgb_(date)_(eid).tif.

    Important cfg.inference parameters:
    cfg.inference.data_dir: Input directory containing SAR rasters
    cfg.inference.n_workers: number of worker processes
    cfg.inference.threads_per_worker: number of threads per worker
    cfg.inference.batch_size: batch size for inference
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
    # get list of rgb captures to predict
    lst = list(Path(cfg.inference.data_dir).glob('rgb_*.tif'))

    # split into chunks for worker processes to finish
    n_workers = getattr(cfg.inference, 'n_workers', 1)
    n_workers = min(n_workers, len(lst))
    print(f'Using {n_workers} workers for {len(lst)} S2 captures...')

    start_idx = 0
    captures_per_worker = len(lst) // n_workers
    remainder = len(lst) % n_workers
    worker_capture_paths = []
    for i in range(n_workers):
        end_idx = start_idx + captures_per_worker + (1 if i < remainder else 0)
        capture_paths = lst[start_idx:end_idx]
        start_idx = end_idx
        worker_capture_paths.append(capture_paths)

    # tqdm progress tracking
    cfg_dict = OmegaConf.to_container(cfg, resolve=True)
    threads_per_worker = getattr(cfg.inference, 'threads_per_worker', 8)
    print(f"Using {threads_per_worker} threads per worker")
    with concurrent.futures.ProcessPoolExecutor(max_workers = n_workers,
                                                initializer=init_worker, 
                                                initargs=(cfg_dict, threads_per_worker),
                                                mp_context=multiprocessing.get_context("forkserver")) as executor:
        futures = [executor.submit(label_captures, capture_paths) for capture_paths in worker_capture_paths]
        for fut in tqdm(as_completed(futures), total=len(futures), desc="Workers in progress"):
            result = fut.result()

@hydra.main(version_base=None, config_path="pkg://configs", config_name="config.yaml")
def main(cfg: DictConfig):
    """Main entry point for inference on spatiotemporal data using RGB S2 optical model.
    
    Parameters
    ----------
    cfg : DictConfig
        Configuration object containing model and data parameters.
    """
    run_inference_s2(cfg)

if __name__ == '__main__':
    main()
