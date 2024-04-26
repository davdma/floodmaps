import torch
from torchvision import transforms
from model import WaterPixelDetector
from architectures.unet import UNet
from architectures.discriminator import Classifier1, Classifier2, Classifier3
from utils import trainMeanStd
from glob import glob
from datetime import datetime
import numpy as np
import argparse
import re
import rasterio
import os
import sys

def get_detector(n_channels=5, dropout=0.2, load_classifier_path="", load_discriminator_path=""):
    model = UNet(n_channels, dropout=dropout).to('cpu')
    model.load_state_dict(torch.load(load_classifier_path, map_location=torch.device('cpu')))
    discriminator = Classifier1(n_channels).to('cpu')
    discriminator.load_state_dict(torch.load(load_discriminator_path, map_location=torch.device('cpu')))

    detector = WaterPixelDetector(model, n_channels=n_channels, discriminator=discriminator)
    detector.eval()
    return detector

def get_classifier(n_channels=5, dropout=0.2, load_classifier_path=""):
    model = UNet(n_channels, dropout=dropout).to('cpu')
    model.load_state_dict(torch.load(load_classifier_path, map_location=torch.device('cpu')))
    detector = WaterPixelDetector(model, n_channels=n_channels)
    detector.eval()
    return detector

def get_sample_prediction(size, channels, detector, standardize, train_mean, dir_path, dt, eid):
    tci_file = dir_path + f'/tci_{dt}_{eid}.tif'
    b08_file = dir_path + f'/b08_{dt}_{eid}.tif'
    ndwi_file = dir_path + f'/ndwi_{dt}_{eid}.tif'
    dem_file = dir_path + f'/dem_{eid}.tif'
    slope_file = dir_path + f'/slope_{eid}.tif'
    waterbody_file = dir_path + f'/waterbody_{eid}.tif'
    roads_file = dir_path + f'/roads_{eid}.tif'
    with rasterio.open(tci_file) as src:
        tci_raster = src.read()
        tci_tile = (tci_raster / 255).astype(np.float32)

    with rasterio.open(b08_file) as src:
        b08_tile = src.read().astype(np.float32)

    with rasterio.open(ndwi_file) as src:
        ndwi_tile = src.read().astype(np.float32)
    
    with rasterio.open(dem_file) as src:
        dem_tile = src.read().astype(np.float32)
    
    with rasterio.open(slope_file) as src:
        slope_tile = src.read().astype(np.float32)

    with rasterio.open(waterbody_file) as src:
        waterbody_tile = src.read().astype(np.float32)

    with rasterio.open(roads_file) as src:
        roads_tile = src.read().astype(np.float32)

    stack = np.vstack((tci_tile, b08_tile, ndwi_tile, dem_tile, 
                        slope_tile, waterbody_tile, roads_tile), dtype=np.float32)

    X = stack[channels]

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
            # debug
            print(patch.shape)
            patch_pred = detector(patch).squeeze()
            
            # stitch tiles together and convert to numpy then use boolean mask
            # want H x W
            label[start_row : end_row, start_col : end_col] = patch_pred[size - dh:, size - dw:].numpy()
            j += 1
        i += 1

    label[missing_vals] = 0
    return label

def main(size, channels, dropout=0.2, replace=True, sample_dir="", label_dir="", post=False, two_head=False):
    n_channels = sum(channels)
    if two_head:
        detector = get_detector(n_channels=n_channels, dropout=dropout, load_classifier_path="models/model28.pth", load_discriminator_path="models/discriminator1.pth")
    else:
        detector = get_classifier(n_channels=n_channels, dropout=dropout, load_classifier_path="models/model28.pth")
        
    train_mean, train_std = trainMeanStd(channels=channels, sample_dir='../samples_200_5_4_35/', label_dir=label_dir)
    standardize = transforms.Compose([transforms.Normalize(train_mean, train_std)])

    # iterate over all samples in sample dir
    lst = glob(sample_dir + "[0-9]*")

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
                    # save result of prediction as .tif file
                    mult_pred = pred * 255
                    broadcasted = np.broadcast_to(mult_pred, (3, *pred.shape)).astype(np.uint8)
                    
                    with rasterio.open(dir_path + f'/pred_{dt}_{eid}.tif', 'w', driver='Gtiff', count=3, height=pred.shape[-2], width=pred.shape[-1], dtype=np.uint8) as dst:
                        dst.write(broadcasted)

if __name__ == '__main__':            
    parser = argparse.ArgumentParser(prog='inference', description='Generate 2 head model predictions for ground truthing.')

    def bool_indices(s):
        if len(s) == 9 and all(c in '01' for c in s):
            try:
                return [bool(int(x)) for x in s]
            except ValueError:
                raise argparse.ArgumentTypeError("Invalid boolean string: '{}'".format(s))
        else:
            raise argparse.ArgumentTypeError("Boolean string must be of length 9 and have binary digits")
            
    parser.add_argument('-x', '--size', dest='size', type=int, default=64, help='pixel width of patch (default: 64)')
    parser.add_argument('-c', '--channels', type=bool_indices, default="111111111", help='string of 9 binary digits for selecting among the 9 available channels (R, G, B, B08, NDWI, DEM, Slope, Water, Roads) (default: 111111111)')
    parser.add_argument('--dropout', type=float, default=0.2, help='model dropout (default: 0.2)')
    parser.add_argument('--replace', action='store_true', help='overwrite all previously made predictions (default: False)')
    parser.add_argument('--sdir', dest='sample_dir', default='../samples_200_5_4_35/', help='(default: ../samples_200_5_4_35/)')
    parser.add_argument('--ldir', dest='label_dir', default='../labels/', help='(default: ../labels/)')
    parser.add_argument('--post', action='store_true', help='only label post-event images (default: False)')
    parser.add_argument('--two_head', action='store_true', help='use two head model (default: False)')
    args = parser.parse_args()
    sys.exit(main(args.size, args.channels, dropout=args.dropout, replace=args.replace, sample_dir=args.sample_dir, label_dir=args.label_dir, post=args.post, two_head=args.two_head))
