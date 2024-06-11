import torch
from torchvision import transforms
from model import WaterPixelDetector
from architectures.unet import UNet
from architectures.unet_plus import NestedUNet
from architectures.discriminator import Classifier1, Classifier2, Classifier3
from utils import trainMeanStd, ChannelIndexer
from glob import glob
from datetime import datetime
import numpy as np
import argparse
import re
import rasterio
import os
import sys

MODEL_NAMES = ['unet', 'unet++']

def get_detector(name, n_channels, dropout, load_classifier_path="", load_discriminator_path=""):
    """Initializes water pixel detection model on CPU given stored model files and settings.

    Note: UNet++ has been updated with dropout, so some prior UNet++ models trained without dropout may fail.
    
    Parameters
    ----------
    n_channels : int
        Number of input channels of model.
    dropout : float
    load_classifier_path : str
        Path to classifier weights.
    load_discriminator_path : Optional[str]
        Path to discriminator weights. Will be attached to the classifier. If empty will return 
        only classifier model.

    Returns
    -------
    detector : obj
    """
    if name == "unet":
        model = UNet(n_channels, dropout=dropout).to('cpu')
    elif name == "unet++":
        model = NestedUNet(n_channels, dropout=dropout, deep_supervision=True).to('cpu')
    else:
        raise Exception("model unknown")
        
    model.load_state_dict(torch.load(load_classifier_path, map_location=torch.device('cpu')))
    discriminator = Classifier1(n_channels).to('cpu')
    discriminator.load_state_dict(torch.load(load_discriminator_path, map_location=torch.device('cpu')))

    detector = WaterPixelDetector(model, n_channels=n_channels, discriminator=discriminator)
    detector.eval()
    return detector

def get_classifier(name, n_channels, dropout, load_classifier_path=""):
    """Initializes classifer model on CPU given stored model files and settings.

    Note: UNet++ has been updated with dropout, so some prior UNet++ models trained without dropout may fail.
    
    Parameters
    ----------
    n_channels : int
        Number of input channels of model.
    dropout : float
    load_classifier_path : str
        Path to classifier weights.

    Returns
    -------
    detector : obj
    """
    if name == "unet":
        model = UNet(n_channels, dropout=dropout).to('cpu')
    elif name == "unet++":
        model = NestedUNet(n_channels, dropout=dropout, deep_supervision=True).to('cpu')
    else:
        raise Exception("model unknown")
    model.load_state_dict(torch.load(load_classifier_path, map_location=torch.device('cpu')))
    detector = WaterPixelDetector(model, n_channels=n_channels)
    detector.eval()
    return detector

def get_sample_prediction(size, channels, detector, standardize, train_mean, dir_path, dt, eid):
    """Generate new predictions on unseen data using detector.

    Parameters
    ----------
    size : int
        Height and width of the discrete patches of the tile to be fed into the model for prediction.
    channels : list[bool]
        List of 10 booleans corresponding to the 10 S2 input channels.
    detector : obj
        Model object for inference.
    standardize : obj
        Standardization of input channels before being fed into the model.
    train_mean : float
        Channel means of model training set for imputing missing data.
    dir_path : str
        Directory path where the raw sample data is stored.
    dt : str
        Date that the TCI of the sample was taken.
    eid : str
        Event ID of the sample.

    Returns
    -------
    label : ndarray
        Predicted label of the specified tile.
    """
    layers = []
    my_channels = ChannelIndexer(channels)
    if my_channels.has_image():
        tci_file = dir_path + f'/tci_{dt}_{eid}.tif'
        with rasterio.open(tci_file) as src:
            tci_raster = src.read()
            tci_tile = (tci_raster / 255).astype(np.float32)
        layers.append(tci_tile)
    if my_channels.has_b08():
        b08_file = dir_path + f'/b08_{dt}_{eid}.tif'
        with rasterio.open(b08_file) as src:
            b08_tile = src.read().astype(np.float32)
        layers.append(b08_tile)
    if my_channels.has_ndwi():
        ndwi_file = dir_path + f'/ndwi_{dt}_{eid}.tif'
        with rasterio.open(ndwi_file) as src:
            ndwi_tile = src.read().astype(np.float32)
        layers.append(ndwi_tile)

    # need dem for slope regardless if in channels or not
    dem_file = dir_path + f'/dem_{eid}.tif'
    with rasterio.open(dem_file) as src:
        dem_tile = src.read().astype(np.float32)
    if my_channels.has_dem():
        layers.append(dem_tile)
        
    # if my_channels.has_slope():
        # slope_file = dir_path + f'/slope_{eid}.tif'
        # with rasterio.open(slope_file) as src:
            # slope_tile = src.read().astype(np.float32)
        # layers.append(slope_tile)
    slope = np.gradient(dem_tile, axis=(1,2))
    slope_y_tile, slope_x_tile = slope
    if my_channels.has_slope_y():
        layers.append(slope_y_tile)
    if my_channels.has_slope_x():
        layers.append(slope_x_tile)
    if my_channels.has_waterbody():
        waterbody_file = dir_path + f'/waterbody_{eid}.tif'
        with rasterio.open(waterbody_file) as src:
            waterbody_tile = src.read().astype(np.float32)
        layers.append(waterbody_tile)
    if my_channels.has_roads():
        roads_file = dir_path + f'/roads_{eid}.tif'
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

    # tile discretely and make predictions
    HEIGHT = X.shape[-2]
    WIDTH = X.shape[-1]
    label = np.zeros((HEIGHT, WIDTH))

    i = 0
    while i * size < HEIGHT:
        j = 0
        while j * size < WIDTH:
            start_row = i * size
            start_col = j * size
            end_row = (i + 1) * size if (i + 1) * size <= HEIGHT else HEIGHT
            end_col = (j + 1) * size if (j + 1) * size <= WIDTH else WIDTH
            
            dh = end_row - start_row # normally size
            dw = end_col - start_col # normally size
            if dh < size:
                # boundary tile
                patch_row = start_row - size + dh
            else:
                patch_row = start_row

            if dw < size:
                patch_col = start_col - size + dw
            else:
                patch_col = start_col

            patch = X[:, patch_row : patch_row + size, patch_col : patch_col + size].unsqueeze(0)
            
            patch_pred = detector(patch).squeeze()
            
            # stitch tiles together and convert to numpy then use boolean mask
            # want H x W
            label[start_row : end_row, start_col : end_col] = patch_pred[size - dh:, size - dw:].numpy()
            j += 1
        i += 1

    label[missing_vals] = 0
    return label

def sample_one(dir_path, dt, eid, size, channels, name="unet", format="tif", dropout=0.1081, sample_dir='../sampling/samples_200_5_4_35/', label_dir='../sampling/labels/', classifier_path="models/unet_model318.pth", discriminator_path="models/discriminator42.pth", two_head=False):
    """Predict one sample for quality control purposes."""
    n_channels = sum(channels)
    if two_head:
        detector = get_detector(name, n_channels, dropout, load_classifier_path=classifier_path, load_discriminator_path=discriminator_path)
    else:
        detector = get_classifier(name, n_channels, dropout, load_classifier_path=classifier_path)

    train_mean, train_std = trainMeanStd(channels=channels, sample_dir=sample_dir, label_dir=label_dir)
    standardize = transforms.Compose([transforms.Normalize(train_mean, train_std)])

    pred = get_sample_prediction(size, channels, detector, standardize, train_mean, dir_path, dt, eid)

    if format == "tif":
        # save result of prediction as .tif file
        # copy sample transforms to the label file!!!
        with rasterio.open(dir_path + f'/tci_{dt}_{eid}.tif') as src:
            transform = src.transform
            crs = src.crs
            
        mult_pred = pred * 255
        broadcasted = np.broadcast_to(mult_pred, (3, *pred.shape)).astype(np.uint8)
        with rasterio.open(dir_path + f'/qc_pred_{dt}_{eid}.tif', 'w', driver='Gtiff', count=3, 
                           height=pred.shape[-2], width=pred.shape[-1], dtype=np.uint8, crs=crs, 
                           transform=transform) as dst:
            dst.write(broadcasted)
    elif format == "npy":
        np.save(dir_path + f'/qc_pred_{dt}_{eid}.npy', pred)
    else:
        raise Exception("format unknown")

def sample_nodem(size, channels, name="unet", format="tif", dropout=0.2987776077544917, replace=True, infer_dir="", sample_dir="", label_dir="", post=False, two_head=False):
    """Predict one sample without using DEM channel for quality control purposes."""
    n_channels = sum(channels)
    if two_head:
        detector = get_detector(name, n_channels, dropout, load_classifier_path="models/unet_model511.pth", load_discriminator_path="models/discriminator42.pth")
    else:
        detector = get_classifier(name, n_channels, dropout, load_classifier_path="models/unet_model511.pth")
        
    train_mean, train_std = trainMeanStd(channels=channels, sample_dir=sample_dir, label_dir=label_dir)
    standardize = transforms.Compose([transforms.Normalize(train_mean, train_std)])

    # iterate over all events in infer dir
    lst = glob(infer_dir + "[0-9]*")

    # get eid from sample dir name, then use to find the tci dates
    p = re.compile('tci_(\d{8})_(\d{8})_(.+).tif')
    for dir_path in lst:
        if os.path.isdir(dir_path):
            eid = dir_path.split('/')[-1]
            # FILTER OUT ALL eids before 20180815 
            if datetime.strptime(eid[:8], '%Y%m%d') < datetime(2018, 8, 14):
                continue
            
            samples = glob(dir_path + '/tci_*.tif')
            # check for existence of predictions
            for sample in samples:
                m = p.search(sample)
                dt = m.group(1)
                img_dt = datetime.strptime(m.group(1), '%Y%m%d')
                event_dt = datetime.strptime(m.group(2), '%Y%m%d')

                # skip if pre-precipitation event
                if post and img_dt < event_dt:
                    continue
            
                if not replace and Path(dir_path + f'/pred_{dt}_{eid}.tif').exists():
                    continue
                else:
                    pred = get_sample_prediction(size, channels, detector, standardize, train_mean, dir_path, dt, eid)

                    if format == "tif":
                        # save result of prediction as .tif file
                        # add transform
                        with rasterio.open(dir_path + f'/tci_{dt}_{eid}.tif') as src:
                            transform = src.transform
                            crs = src.crs
                            
                        mult_pred = pred * 255
                        broadcasted = np.broadcast_to(mult_pred, (3, *pred.shape)).astype(np.uint8)
                        with rasterio.open(dir_path + f'/pred_{dt}_{eid}.tif', 'w', driver='Gtiff', count=3, height=pred.shape[-2], width=pred.shape[-1], dtype=np.uint8, crs=crs, transform=transform) as dst:
                            dst.write(broadcasted)
                    elif format == "npy":
                        np.save(dir_path + f'/pred_{dt}_{eid}.npy', pred)
                    else:
                        raise Exception("format unknown")

def main(size, channels, name="unet", format="tif", dropout=0.2, replace=True, infer_dir="", sample_dir="", label_dir="", post=False, two_head=False):
    """Generates machine labels for dataset using tuned S2 optical model.

    Note: run with conda environment 'floodmaps'.
    
    Parameters
    ----------
    size : int
        Height and width of the discrete patches of the tile to be fed into the model for prediction.
    channels : list[bool]
        List of 10 booleans corresponding to the 10 S2 input channels.
    name : str
        Architecture of model classifier: unet, unet++.
    format : str
        Output label file format: tif, npy.
    dropout : float
    replace : bool
        Whether to overwrite pre-existing predictions.
    infer_dir : str
        Directory path of raw tiles where predicted labels are generated and stored.
    sample_dir : str
        Directory path of raw tiles used in model training (for standardization purposes).
    label_dir : str
        Directory path of raw tile labels used in model training (for standardization purposes).
    post : bool
        Only make predictions on tiles taken post flood event.
    two_head : bool
        Option to create two head model (adding discriminator).
    """
    n_channels = sum(channels)
    if two_head:
        detector = get_detector(name, n_channels, dropout, load_classifier_path="models/unet_model318.pth", load_discriminator_path="models/discriminator42.pth")
    else:
        detector = get_classifier(name, n_channels, dropout, load_classifier_path="models/unet_model318.pth")
        
    train_mean, train_std = trainMeanStd(channels=channels, sample_dir=sample_dir, label_dir=label_dir)
    standardize = transforms.Compose([transforms.Normalize(train_mean, train_std)])

    # iterate over all events in infer dir
    lst = glob(infer_dir + "[0-9]*")

    # get eid from sample dir name, then use to find the tci dates
    p = re.compile('tci_(\d{8})_(\d{8})_(.+).tif')
    for dir_path in lst:
        if os.path.isdir(dir_path):
            eid = dir_path.split('/')[-1]
            samples = glob(dir_path + '/tci_*.tif')
            # check for existence of predictions
            for sample in samples:
                m = p.search(sample)
                dt = m.group(1)
                img_dt = datetime.strptime(m.group(1), '%Y%m%d')
                event_dt = datetime.strptime(m.group(2), '%Y%m%d')

                # skip if pre-precipitation event
                if post and img_dt < event_dt:
                    continue
            
                if not replace and Path(dir_path + f'/pred_{dt}_{eid}.tif').exists():
                    continue
                else:
                    pred = get_sample_prediction(size, channels, detector, standardize, train_mean, dir_path, dt, eid)

                    if format == "tif":
                        # save result of prediction as .tif file
                        # add transform
                        with rasterio.open(dir_path + f'/tci_{dt}_{eid}.tif') as src:
                            transform = src.transform
                            crs = src.crs
                            
                        mult_pred = pred * 255
                        broadcasted = np.broadcast_to(mult_pred, (3, *pred.shape)).astype(np.uint8)
                        with rasterio.open(dir_path + f'/pred_{dt}_{eid}.tif', 'w', driver='Gtiff', count=3, height=pred.shape[-2], width=pred.shape[-1], dtype=np.uint8, crs=crs, transform=transform) as dst:
                            dst.write(broadcasted)
                    elif format == "npy":
                        np.save(dir_path + f'/pred_{dt}_{eid}.npy', pred)
                    else:
                        raise Exception("format unknown")

if __name__ == '__main__':            
    parser = argparse.ArgumentParser(prog='inference', description='Generate 2 head model predictions for ground truthing.')

    def bool_indices(s):
        if len(s) == 10 and all(c in '01' for c in s):
            try:
                return [bool(int(x)) for x in s]
            except ValueError:
                raise argparse.ArgumentTypeError("Invalid boolean string: '{}'".format(s))
        else:
            raise argparse.ArgumentTypeError("Boolean string must be of length 10 and have binary digits")
            
    parser.add_argument('-x', '--size', dest='size', type=int, default=64, help='pixel width of patch (default: 64)')
    parser.add_argument('-c', '--channels', type=bool_indices, default="1111111111", help='string of 10 binary digits for selecting among the 10 available channels (R, G, B, B08, NDWI, DEM, SlopeY, SlopeX, Water, Roads) (default: 1111111111)')
    parser.add_argument('-f', '--format', default="tif", choices=["npy", "tif"], help='prediction label format: npy, tif (default: tif)')
    parser.add_argument('--name', default='unet', choices=MODEL_NAMES,
                        help=f"models: {', '.join(MODEL_NAMES)} (default: unet)")
    parser.add_argument('--dropout', type=float, default=0.2, help='model dropout (default: 0.2)')
    parser.add_argument('--replace', action='store_true', help='overwrite all previously made predictions (default: False)')
    parser.add_argument('--idir', dest='infer_dir', default='../sampling/samples_200_6_4_10_sar/', help='directory to make predictions in (default: ../sampling/samples_200_6_4_10_sar/)')
    parser.add_argument('--sdir', dest='sample_dir', default='../sampling/samples_200_5_4_35/', help='directory for label data (default: ../sampling/samples_200_5_4_35/)')
    parser.add_argument('--ldir', dest='label_dir', default='../sampling/labels/', help='(default: ../sampling/labels/)')
    parser.add_argument('--post', action='store_true', help='only label post-event images (default: False)')
    parser.add_argument('--two_head', action='store_true', help='use two head model (default: False)')
    args = parser.parse_args()
    sys.exit(main(args.size, args.channels, format=args.format, name=args.name, dropout=args.dropout, replace=args.replace, infer_dir=args.infer_dir, sample_dir=args.sample_dir, label_dir=args.label_dir, post=args.post, two_head=args.two_head))
