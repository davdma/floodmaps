import torch
from torch import nn
from torchvision import transforms
from datetime import datetime, timezone, timedelta
import numpy as np
import re
import rasterio
import pickle
from pathlib import Path
import hydra
from omegaconf import DictConfig

from floodmaps.models.model import S2WaterDetector, SARWaterDetector
from floodmaps.utils.utils import ChannelIndexerDeprecated, SARChannelIndexer

def single_prediction_s2(cfg: DictConfig, dt: str, eid: str, format="tif"):
    """Predict one S2 sample for quality control purposes.
    
    Parameters
    ----------
    cfg : Config
        Configuration object containing model and data parameters.
    dt : str
        Date that the TCI of the sample was taken.
    eid : str
        Event ID of the sample.
    format : str
        Output label file format: tif, npy.
    data_dir : str
        Directory path of dataset containing event tiles where predicted labels should be generated and stored.
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
    use_weak = getattr(cfg.data, 'use_weak', False)
    dataset_type = 's2_weak' if use_weak else 's2'
    if suffix:
        sample_dir = Path(cfg.paths.preprocess_dir) / dataset_type / f'samples_{size}_{samples}_{suffix}/'
    else:
        sample_dir = Path(cfg.paths.preprocess_dir) / dataset_type / f'samples_{size}_{samples}/'

    channels = [bool(int(x)) for x in cfg.data.channels]
    b_channels = sum(channels[-2:])
    with open(sample_dir / f'mean_std_{size}_{samples}.pkl', 'rb') as f:
        train_mean, train_std = pickle.load(f)

        train_mean = torch.from_numpy(train_mean[channels])
        train_std = torch.from_numpy(train_std[channels])
        # make sure binary channels are 0 mean and 1 std
        if b_channels > 0:
            train_mean[-b_channels:] = 0
            train_std[-b_channels:] = 1

    standardize = transforms.Compose([transforms.Normalize(train_mean, train_std)])

    # Convert string path to Path object for consistency
    event_path = Path(cfg.inference.data_dir) / eid
    pred = get_sample_prediction_s2(model, cfg, standardize, train_mean, event_path, dt, eid)

    if format == "tif":
        # save result of prediction as .tif file
        # copy sample transforms to the label file!!!
        with rasterio.open(event_path / f'tci_{dt}_{eid}.tif') as src:
            transform = src.transform
            crs = src.crs

        mult_pred = pred * 255
        broadcasted = np.broadcast_to(mult_pred, (3, *pred.shape)).astype(np.uint8)
        with rasterio.open(event_path / f'qc_pred_{dt}_{eid}.tif', 'w', driver='Gtiff', count=3,
                           height=pred.shape[-2], width=pred.shape[-1], dtype=np.uint8, crs=crs,
                           transform=transform) as dst:
            dst.write(broadcasted)
    elif format == "npy":
        np.save(event_path / f'qc_pred_{dt}_{eid}.npy', pred)
    else:
        raise Exception("format unknown")

def single_prediction_sar(cfg: DictConfig, s2_dt: str, s1_dt: str, eid: str, format="tif"):
    """Predict one SAR sample for quality control purposes.
    
    Parameters
    ----------
    cfg : DictConfig
        Configuration object containing model and data parameters.
    s2_dt : str
        Date that the S2 image was taken.
    s1_dt : str
        Date that the S1 image was taken.
    eid : str
        Event ID of the sample.
    format : str
        Output label file format: tif, npy.
    data_dir : str
        Directory path of dataset containing event tiles where predicted labels should be generated and stored.
    """
    device = (
        "cuda"
        if torch.cuda.is_available()
        else "cpu"
    )
    print(f"Using {device} device")

    # load model and weights
    ad_cfg = getattr(cfg, 'ad', None)
    model = SARWaterDetector(cfg, ad_cfg=ad_cfg).to(device)
    model.eval()

    # dataset and transforms
    size = cfg.data.size
    samples = cfg.data.samples
    filter_type = cfg.data.filter
    suffix = getattr(cfg.data, 'suffix', '')
    if suffix:
        sample_dir = Path(cfg.paths.preprocess_dir) / 's1_weak' / f'samples_{size}_{samples}_{filter_type}_{suffix}/'
    else:
        sample_dir = Path(cfg.paths.preprocess_dir) / 's1_weak' / f'samples_{size}_{samples}_{filter_type}/'

    channels = [bool(int(x)) for x in cfg.data.channels]
    with open(sample_dir / f'mean_std_{size}_{samples}_{filter_type}.pkl', 'rb') as f:
        train_mean, train_std = pickle.load(f)

        train_mean = torch.from_numpy(train_mean[channels])
        train_std = torch.from_numpy(train_std[channels])
    standardize = transforms.Compose([transforms.Normalize(train_mean, train_std)])

    # Convert string path to Path object for consistency
    event_path = Path(cfg.inference.data_dir) / eid
    pred = get_sample_prediction_sar(model, cfg, standardize, train_mean, event_path, s2_dt, s1_dt, eid)

    if format == "tif":
        # save result of prediction as .tif file
        # copy sample transforms to the label file!!!
        with rasterio.open(event_path / f'sar_{s2_dt}_{s1_dt}_{eid}_vv.tif') as src:
            transform = src.transform
            crs = src.crs

        mult_pred = pred * 255
        broadcasted = np.broadcast_to(mult_pred, (3, *pred.shape)).astype(np.uint8)
        with rasterio.open(event_path / f'sar_qc_pred_{s1_dt}_{eid}.tif', 'w', driver='Gtiff', count=3,
                           height=pred.shape[-2], width=pred.shape[-1], dtype=np.uint8, crs=crs,
                           transform=transform) as dst:
            dst.write(broadcasted)
    elif format == "npy":
        np.save(event_path / f'sar_qc_pred_{s1_dt}_{eid}.npy', pred)
    else:
        raise Exception("format unknown")

def get_sample_prediction_sar(model, cfg: DictConfig, standardize, train_mean, event_path, s2_dt, s1_dt, eid, device='cpu', threshold=0.5):
    """Generate new predictions on unseen SAR data using water detector model.
    Predictions are made using a sliding window with 25% overlap.

    Parameters
    ----------
    model : obj
        Model object for inference.
    cfg : DictConfig
        Configuration object containing model and data parameters.
    standardize : obj
        Standardization of input channels before being fed into the model.
    train_mean : float
        Channel means of model training set for imputing missing data.
    event_path : Path
        Path to the event directory where the raw sample data is stored.
    s2_dt : str
        Date that the S2 image was taken.
    s1_dt : str
        Date that the S1 image was taken.
    eid : str
        Event ID of the sample.
    device : str
        Device to run inference on.

    Returns
    -------
    label : ndarray
        Predicted label of the specified tile in shape (H, W).
    """
    layers = []
    my_channels = SARChannelIndexer([bool(int(x)) for x in cfg.data.channels])
    if my_channels.has_vv():
        vv_file = event_path / f'sar_{s2_dt}_{s1_dt}_{eid}_vv.tif'
        with rasterio.open(vv_file) as src:
            vv_raster = src.read()
        layers.append(vv_raster)
    if my_channels.has_vh():
        vh_file = event_path / f'sar_{s2_dt}_{s1_dt}_{eid}_vh.tif'
        with rasterio.open(vh_file) as src:
            vh_raster = src.read()
        layers.append(vh_raster)

    # need dem for slope regardless if in channels or not
    dem_file = event_path / f'dem_{eid}.tif'
    with rasterio.open(dem_file) as src:
        dem_raster = src.read().astype(np.float32)
    if my_channels.has_dem():
        layers.append(dem_raster)

    slope = np.gradient(dem_raster, axis=(1,2))
    slope_y_raster, slope_x_raster = slope
    if my_channels.has_slope_y():
        layers.append(slope_y_raster)
    if my_channels.has_slope_x():
        layers.append(slope_x_raster)
    if my_channels.has_waterbody():
        waterbody_file = event_path / f'waterbody_{eid}.tif'
        with rasterio.open(waterbody_file) as src:
            waterbody_raster = src.read().astype(np.float32)
        layers.append(waterbody_raster)
    if my_channels.has_roads():
        roads_file = event_path / f'roads_{eid}.tif'
        with rasterio.open(roads_file) as src:
            roads_raster = src.read().astype(np.float32)
        layers.append(roads_raster)

    X = np.vstack(layers, dtype=np.float32)

    # get missing values mask (to later zero out)
    missing_vals = X[0] == -9999

    # impute missing values in each channel with its mean
    train_mean = train_mean.tolist()
    for i, mean in enumerate(train_mean):
        X[i][missing_vals] = mean

    X = torch.from_numpy(X)
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

    # send to device
    X = X.to(device)

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

                # Accumulate probabilities
                prob = torch.sigmoid(pred).float()
                pred_map[y:y+patch_size, x:x+patch_size] += prob
                count_map[y:y+patch_size, x:x+patch_size] += 1.0

    # Average overlapping predictions
    pred_map = torch.where(count_map > 0, pred_map / count_map, pred_map)

    # Convert to final binary prediction
    label = torch.where(pred_map >= threshold, 1.0, 0.0).byte().to('cpu').numpy()
    label[missing_vals] = 0
    return label

def get_sample_prediction_s2(model, cfg: DictConfig, standardize, train_mean, event_path, dt, eid, threshold=0.5):
    """Generate new predictions on unseen data using water detector model.
    Predictions are made using a sliding window with 25% overlap.

    Parameters
    ----------
    model : obj
        Model object for inference.
    cfg : DictConfig
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
    threshold : float
        Threshold for the prediction.
    Returns
    -------
    label : ndarray
        Predicted label of the specified tile in shape (H, W).
    """
    layers = []
    my_channels = ChannelIndexerDeprecated([bool(int(x)) for x in cfg.data.channels])
    if my_channels.has_image():
        tci_file = event_path / f'tci_{dt}_{eid}.tif'
        with rasterio.open(tci_file) as src:
            tci_raster = src.read()
            tci_tile = (tci_raster / 255).astype(np.float32)
        layers.append(tci_tile)
    if my_channels.has_b08():
        b08_file = event_path / f'b08_{dt}_{eid}.tif'
        with rasterio.open(b08_file) as src:
            b08_tile = src.read().astype(np.float32)
        layers.append(b08_tile)
    if my_channels.has_ndwi():
        ndwi_file = event_path / f'ndwi_{dt}_{eid}.tif'
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

    X = np.vstack(layers, dtype=np.float32)

    # get missing values mask (to later zero out)
    missing_vals = X[0] == 0

    # impute missing values in each channel with its mean
    train_mean = train_mean.tolist()
    for i, mean in enumerate(train_mean):
        X[i][missing_vals] = mean

    X = torch.from_numpy(X)
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
                
                # Accumulate probabilities
                prob = torch.sigmoid(pred).float()
                pred_map[y:y+patch_size, x:x+patch_size] += prob
                count_map[y:y+patch_size, x:x+patch_size] += 1.0
    
    # Average overlapping predictions
    pred_map = torch.where(count_map > 0, pred_map / count_map, pred_map)
    
    # Convert to final binary prediction
    label = torch.where(pred_map >= threshold, 1.0, 0.0).byte().numpy()
    label[missing_vals] = 0
    return label

def parse_manual_file(manual_file):
    """Parse manual file to get EIDs to predict.
    Each line should contain one EID string in the event directory."""
    with open(manual_file, 'r') as f:
        eids = [line.strip() for line in f]
    return eids

def run_weak_labeling(cfg: DictConfig):
    """Generates machine labels for dataset using tuned S2 optical model.

    Note: run with conda environment 'floodmaps-training'.

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
    suffix = getattr(cfg.data, 'suffix', '') # for old model can use suffix=deprecated
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

    # get list of events to predict
    lst = []
    manual = getattr(cfg.inference, 'manual', None)
    if manual is not None:
        eids = parse_manual_file(manual)
        lst.extend([Path(cfg.inference.data_dir) / eid for eid in eids])
    else:
        # iterate over all events in dataset dir
        lst.extend((Path(cfg.inference.data_dir)).glob("[0-9]*"))

    # get eid then use to find the tci dates for each event
    p = re.compile('tci_(\d{8})_(\d{8})_(.+).tif')
    for event_path in lst:
        if event_path.is_dir():
            eid = event_path.name
            samples = list(event_path.glob('tci_*.tif'))
            # check for existence of predictions
            for sample in samples:
                m = p.search(sample.name)
                dt = m.group(1)

                # PRISM EVENT DATE IS ACTUALLY DEFINED AS 12:00 UTC OF THE DAY BEFORE (IMG_DT IS UTC) HENCE SUBTRACT 1 DAY
                img_dt = datetime.strptime(m.group(1), '%Y%m%d')
                event_dt = datetime.strptime(m.group(2), '%Y%m%d') - timedelta(days=1)

                # skip if pre-precipitation event
                if cfg.inference.post and img_dt < event_dt:
                    continue

                if not cfg.inference.replace and (event_path / f'pred_{dt}_{eid}.tif').exists():
                    continue
                else:
                    pred = get_sample_prediction_s2(model, cfg, standardize, train_mean, event_path, dt, eid)

                    if cfg.inference.format == "tif":
                        # save result of prediction as .tif file
                        # add transform
                        with rasterio.open(event_path / f'tci_{dt}_{eid}.tif') as src:
                            transform = src.transform
                            crs = src.crs

                        mult_pred = pred * 255
                        broadcasted = np.broadcast_to(mult_pred, (3, *pred.shape)).astype(np.uint8)
                        with rasterio.open(event_path / f'pred_{dt}_{eid}.tif', 'w', driver='Gtiff', count=3, height=pred.shape[-2], width=pred.shape[-1], dtype=np.uint8, crs=crs, transform=transform) as dst:
                            dst.write(broadcasted)
                    elif cfg.inference.format == "npy":
                        np.save(event_path / f'pred_{dt}_{eid}.npy', pred)
                    else:
                        raise Exception("format unknown")

@hydra.main(version_base=None, config_path="pkg://configs", config_name="config.yaml")
def main(cfg: DictConfig):
    """Main entry point for weak label generation using the old deprecated TCI input model."""
    run_weak_labeling(cfg)

if __name__ == '__main__':
    main()
