from glob import glob
import matplotlib.pyplot as plt
import rasterio
import numpy as np
from pathlib import Path
import re
import sys
import argparse
from random import Random
import logging
from utils import enhanced_lee_filter

def random_crop(size, num_samples, rng, method, sample_dir, filter=None, typ="train"):
    """Uniformly samples patches of size x size pixels across the image."""
    pre_label_dir = f'data/sar/{method}/' + f'labels{size}_{typ}_{num_samples}/'
    pre_sample_dir = f'data/sar/{method}/' + f'samples{size}_{typ}_{num_samples}/'
    Path(pre_label_dir).mkdir(parents=True, exist_ok=True)
    Path(pre_sample_dir).mkdir(parents=True, exist_ok=True)

    # SAR Preprocessing: labels will be stored in native folder
    # iterate over all samples of sample_dir
    # assume pretrained labels provided
    event_samples = glob(sample_dir + '[0-9]*')
    p1 = re.compile('\d{8}_\d+_\d+')
    for event in event_samples:
        m = p1.search(event)

        if m:
            eid = m.group(0)
        else:
            logger.info(f'No matching eid. Skipping {event}...')
            continue

        # search for label file + sar file
        # look for labels w tci + sar pairings
        p2 = re.compile('pred_(\d{8})_.+.tif')
        for label in glob(event + f'/pred_*.tif'):
            m = p2.search(label)
            img_dt = m.group(1)
            
            sar_vv_files = glob(event + f'/sar_{img_dt}_*_vv.tif')
            if len(sar_vv_files) == 0:
                raise Exception('sar files missing for label')

            sar_vv_file = sar_vv_files[0]
            sar_vh_file = sar_vv_files[:-6] + 'vh.tif'

            # get associated label
            with rasterio.open(label) as src:
                label_raster = src.read([1, 2, 3])
                # if label has any values != 0 or 255 then print to log!
                if np.any((label_raster > 0) & (label_raster < 255)):
                    logger.debug(f'{label_file} values are not 0 or 255.')
                    
                label_binary = np.where(label_raster[0] != 0, 1, 0)
    
                HEIGHT = src.height
                WIDTH = src.width

            # skip missing data w tci and sar combined
            tci_file = sample_dir + f'{eid}/tci_{img_dt}_{eid}.tif'
            dem_file = sample_dir + f'{eid}/dem_{eid}.tif'
            slope_file = sample_dir + f'{eid}/slope_{eid}.tif'
            waterbody_file = sample_dir + f'{eid}/waterbody_{eid}.tif'
            roads_file = sample_dir + f'{eid}/roads_{eid}.tif'

            with rasterio.open(tci_file) as src:
                tci_raster = src.read()

            with rasterio.open(sar_vv_file) as src:
                vv_raster = src.read()

            with rasterio.open(sar_vh_file) as src:
                vh_raster = src.read()

            # apply speckle filter to sar:
            if not filter is None and filter == "lee":
                vv_raster = enhanced_lee_filter(vv_raster).astype(np.float32)
                vh_raster = enhanced_lee_filter(vh_raster).astype(np.float32)
                
            with rasterio.open(dem_file) as src:
                dem_raster = src.read()
        
            with rasterio.open(slope_file) as src:
                slope_raster = src.read()
    
            with rasterio.open(waterbody_file) as src:
                waterbody_raster = src.read()
    
            with rasterio.open(roads_file) as src:
                roads_raster = src.read()
    
            # choose n random coordinates within the tile
            sampled = 0
            while sampled < num_samples:
                x = int(rng.uniform(0, HEIGHT - size))
                y = int(rng.uniform(0, WIDTH - size))
    
                # check if already completed
                if Path(pre_sample_dir + f'sample_{tile_date}_{eid}_{x}_{y}.tif').exists():
                    continue
    
                tci_tile = tci_raster[:, x : x + size, y : y + size]
    
                # if contains missing values, toss out and resample
                if np.any(tci_tile == 0):
                    continue
    
                vv_tile = vv_raster[:, x : x + size, y : y + size]
                vv_tile = vv_tile.astype(np.float32)

                if np.any(vv_tile == -9999):
                    continue
                
                vh_tile = vh_raster[:, x : x + size, y : y + size]
                vh_tile = vh_tile.astype(np.float32)

                if np.any(vh_tile == -9999):
                    continue
        
                dem_tile = dem_raster[:, x : x + size, y : y + size]
                dem_tile = dem_tile.astype(np.float32)
        
                slope_tile = slope_raster[:, x : x + size, y : y + size]
                slope_tile = slope_tile.astype(np.float32)
    
                waterbody_tile = waterbody_raster[:, x : x + size, y : y + size]
                waterbody_tile = waterbody_tile.astype(np.float32)
    
                roads_tile = roads_raster[:, x : x + size, y : y + size]
                roads_tile = roads_tile.astype(np.float32)
    
                label_tile = label_binary[x : x + size, y : y + size]
                label_tile = label_tile.astype(np.uint8)
                
                stacked_tile = np.vstack((vv_tile, vh_tile, dem_tile, 
                                          slope_tile, waterbody_tile, roads_tile), dtype=np.float32)
    
                # save 64 x 64 tile to tif file
                with rasterio.open(pre_sample_dir + f'sample_{tile_date}_{eid}_{x}_{y}.tif', 'w', driver='Gtiff', count=stacked_tile.shape[0], height=size, width=size, dtype=np.float32) as dst:
                    dst.write(stacked_tile)
                        
                with rasterio.open(pre_label_dir + f'label_{tile_date}_{eid}_{x}_{y}.tif', 'w', driver='Gtiff', count=1, height=size, width=size, dtype=np.uint8) as dst:
                    dst.write(label_tile, 1)
                    
                sampled += 1
            
def main(size, samples, seed, method='random', filter=None, sample_dir='../samples_200_5_4_35/'):
    logger = logging.getLogger('preprocessing')
    logger.setLevel(logging.DEBUG)
    formatter = logging.Formatter('%(asctime)s - %(name)s - %(levelname)s - %(message)s')
    handler = logging.StreamHandler(sys.stdout)
    handler.setFormatter(formatter)
    handler.setLevel(logging.DEBUG)
    logger.addHandler(handler)
    logger.propagate = False
    
    rng = Random(seed)
    if method == 'random':
        random_crop(size, samples, rng, method, sample_dir, filter=filter, typ="train")
        random_crop(size, samples, rng, method, sample_dir, filter=filter, typ="test")

    logger.debug('Preprocessing complete.')

if __name__ == '__main__':
    parser = argparse.ArgumentParser(prog='preprocess_s1', description='Preprocesses 4km x 4km PRISM tiles into smaller tiles via random crop method.')
    parser.add_argument('-x', '--size', dest='size', type=int, default=64, choices=[64, 128], help='pixel width of patch (default: 64)')
    parser.add_argument('-n', '--samples', dest='samples', type=int, default=500, help='number of samples per image (default: 500)')
    parser.add_argument('-s', '--seed', dest='seed', type=int, default=433002, help='random number generator seed (default: 433002)')
    parser.add_argument('-m', '--method', dest='method', default='random', choices=['random'], help='sampling method (default: random)')
    parser.add_argument('--filter', default=None, choices=['lee', 'model'],
                        help=f"filters: enhanced lee, model (default: none)")
    parser.add_argument('--sdir', dest='sample_dir', default='../samples_200_5_4_36_sar/', help='(default: ../samples_200_5_4_36_sar/)')
    
    args = parser.parse_args()
    sys.exit(main(args.size, args.samples, args.seed, method=args.method, filter=args.filter, sample_dir=args.sample_dir))
