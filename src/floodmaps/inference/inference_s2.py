import torch
from torchvision import transforms
import numpy as np
import re
import rasterio
import pickle
from pathlib import Path
import hydra
from omegaconf import DictConfig
from datetime import datetime

from floodmaps.utils.utils import ChannelIndexer
from floodmaps.models.model import S2WaterDetector
from floodmaps.utils.preprocess_utils import PROCESSING_BASELINE_NAIVE, BOA_ADD_OFFSET


def generate_prediction_s2(model, device, cfg, standardize, train_mean, event_path, dt, eid, threshold=0.5):
    """Generate new predictions on unseen data using water detector model.
    Predictions are made using a sliding window with 25% overlap.

    Parameters
    ----------
    model : obj
        Model object for inference.
    cfg : Config
        Configuration object containing model and data parameters.
    standardize : obj
        Standardization of input channels before being fed into the model.
    train_mean : float
        Channel means of model training set for imputing missing data.
    event_path : Path
        Path to the event directory where the raw sample data is stored.
    dt : str
        Date that the TCI of the sample was taken.
    eid : str
        Event ID of the sample.

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
        rgb_tile = src.read().astype(np.float32)
    
    # Extract missing values mask BEFORE applying offset (raw DN 0 = missing)
    missing_vals = rgb_tile[0] == 0
    
    if img_dt_obj >= PROCESSING_BASELINE_NAIVE:
        rgb_tile = rgb_tile + BOA_ADD_OFFSET

    if my_channels.has_rgb():
        rgb_tile_clipped = np.clip(rgb_tile, None, 10000.0)
        rgb_tile_clipped = rgb_tile_clipped / 10000.0  # Scale reflectance to [0, 1]
        layers.append(rgb_tile_clipped)

    b08_file = event_path / f'b08_{dt}_{eid}.tif'
    with rasterio.open(b08_file) as src:
        b08_tile = src.read().astype(np.float32)
        if img_dt_obj >= PROCESSING_BASELINE_NAIVE:
            b08_tile = b08_tile + BOA_ADD_OFFSET

    if my_channels.has_b08():
        b08_tile_clipped = np.clip(b08_tile, None, 10000.0)
        b08_tile_clipped = b08_tile_clipped / 10000.0  # Scale reflectance to [0, 1]
        layers.append(b08_tile_clipped)

    if my_channels.has_ndwi():
        ndwi_file = event_path / f'ndwi_{dt}_{eid}.tif'
        if img_dt_obj >= PROCESSING_BASELINE_NAIVE:
            # Post processing baseline, need to use different equation for ndwi
            # This is a temporary patch, but we want to fix this at the data pipeline step!
            recompute_ndwi = np.where(
                (rgb_tile[1] + b08_tile[0]) != 0,
                (rgb_tile[1] - b08_tile[0]) / (rgb_tile[1] + b08_tile[0]),
                -999999
            )
            ndwi_tile = np.expand_dims(recompute_ndwi, axis = 0)
        else:
            with rasterio.open(ndwi_file) as src:
                ndwi_tile = src.read().astype(np.float32)
        layers.append(ndwi_tile)

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

    # impute missing values in each channel with its mean
    train_mean = train_mean.tolist()
    for i, mean in enumerate(train_mean):
        X[i][missing_vals] = mean

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
    pred_map = torch.zeros((H, W), dtype=torch.float32)
    count_map = torch.zeros((H, W), dtype=torch.float32)

    with torch.no_grad():
        hit_y_edge = False
        for y in range(0, H, stride):
            # if patch moves out of image, change patch bounds to start from edges backward
            if hit_y_edge:
                break
            if y + patch_size >= H:
                y = H - patch_size
                hit_y_edge = True

            hit_x_edge = False
            for x in range(0, W, stride):
                if hit_x_edge:
                    break
                if x + patch_size >= W:
                    x = W - patch_size
                    hit_x_edge = True
                
                # Extract patch
                patch = X[:, y:y+patch_size, x:x+patch_size].unsqueeze(0)
                
                # Make prediction
                output = model(patch)
                if isinstance(output, dict):
                    pred = output['classifier_output'].squeeze()
                else:
                    pred = output.squeeze()
                
                # Accumulate probabilities for overlapping regions
                prob = torch.sigmoid(pred).cpu().float()
                pred_map[y:y+patch_size, x:x+patch_size] += prob
                count_map[y:y+patch_size, x:x+patch_size] += 1.0
    
    # Average overlapping predictions
    pred_map = torch.where(count_map > 0, pred_map / count_map, pred_map)
    
    # Convert to final binary prediction
    label = torch.where(pred_map >= threshold, 1.0, 0.0).byte().numpy()
    label[missing_vals] = 0
    return label

@hydra.main(version_base=None, config_path="pkg://configs", config_name="config.yaml")
def main(cfg: DictConfig):
    """Generates predictions in specified folder for specific area of interest using S2 model.

    Note: for custom data directories, rasters are expected to have eid defined in file names i.e. rgb_(date)_(eid).tif.

    Parameters
    ----------
    cfg : DictConfig
        Configuration object containing model and data parameters.
    """
    device = (
        "cuda"
        if torch.cuda.is_available()
        else "cpu"
    )
    print(f"Using {device} device")

    model = S2WaterDetector(cfg).to(device)
    model.eval()

    # dataset and transforms
    size = cfg.data.size
    samples = cfg.data.samples
    suffix = getattr(cfg.data, 'suffix', '')
    # Use weak labeled dataset if specified, otherwise use manual labeled dataset
    use_weak = getattr(cfg.data, 'use_weak', False)
    dataset_type = 's2_weak' if use_weak else 's2'
    if suffix:
        sample_dir = Path(cfg.paths.preprocess_dir) / dataset_type / f'samples_{size}_{samples}_{suffix}/'
    else:
        sample_dir = Path(cfg.paths.preprocess_dir) / dataset_type / f'samples_{size}_{samples}/'

    channels = [bool(int(x)) for x in cfg.data.channels]
    with open(sample_dir / f'mean_std_{size}_{samples}.pkl', 'rb') as f:
        train_mean, train_std = pickle.load(f)

        train_mean = torch.from_numpy(train_mean[channels])
        train_std = torch.from_numpy(train_std[channels])

    standardize = transforms.Compose([transforms.Normalize(train_mean, train_std)])

    # make prediction for each rgb file in the data dir
    p = re.compile('rgb_(\d{8})_(.+).tif')
    data_path = Path(cfg.inference.data_dir)
    for rgb_file in data_path.glob('rgb_*.tif'):
        m = p.search(rgb_file.name)
        dt = m.group(1)
        eid = m.group(2)

        if not cfg.inference.replace and (data_path / f'pred_{dt}_{eid}.tif').exists():
            continue

        print("Generating prediction for:", rgb_file.name)
        pred = generate_prediction_s2(model, device, cfg, standardize, train_mean, data_path, dt, eid, threshold=cfg.inference.threshold)
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
                        

if __name__ == '__main__':
    main()
