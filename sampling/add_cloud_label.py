import rasterio
import rasterio.merge
import numpy as np
from rasterio.warp import reproject, Resampling
import pystac_client
import planetary_computer
from pystac.extensions.projection import ProjectionExtension as pe
from glob import glob
from datetime import datetime, timedelta
import os
from fiona.transform import transform, transform_geom
import json
import argparse
from pathlib import Path
import re
import sys
import logging
os.environ['PC_SDK_SUBSCRIPTION_KEY'] = 'a613baefa08445269838bc3bc0dfe2d9'
PRISM_CRS = "EPSG:4269"
SEARCH_CRS = "EPSG:4326"

def download_SCL(items_s2, p_bbox, shape, tci_transform, crs, eid, save_as):
    """Given single True Color Image (TCI) file, get it's corresponding Scene Classification Layer (SCL)."""
    for item in items_s2:
        if pe.ext(item).crs_string == crs:
            item_href = planetary_computer.sign(item.assets["SCL"].href)

            conversion = transform(PRISM_CRS, crs, (p_bbox[0], p_bbox[2]), (p_bbox[1], p_bbox[3]))
            img_bbox = conversion[0][0], conversion[1][0], conversion[0][1], conversion[1][1]
            out_image, out_transform = rasterio.merge.merge([item_href], bounds=img_bbox)
            clouds = np.isin(out_image[0], [8, 9, 10]).astype(int)

            # need to resample to grid of tci
            dest = np.zeros(shape, dtype=clouds.dtype)
            reproject(
                clouds,
                dest,
                src_transform=out_transform,
                src_crs=crs,
                dst_transform=tci_transform,
                dst_crs=crs,
                resampling=Resampling.nearest)

            # only make cloud values (8, 9, 10) 1 everything else 0
            with rasterio.open(save_as + '.tif', 'w', driver='Gtiff', count=1, 
                               height=dest.shape[-2], width=dest.shape[-1], crs=crs, 
                               dtype=dest.dtype, transform=tci_transform) as dst:
                dst.write(dest, 1)
            return

    raise Exception(f'SCL not completed for: EID {eid}, file {save_as}, crs {crs}')

def main(dir_path):
    """Loop over all samples in folder and generate cloud labels by querying S2 planetary computer. Will resample from 20m to 10m."""
    logger = logging.getLogger('add_clouds')
    logger.setLevel(logging.DEBUG)
    formatter = logging.Formatter('%(asctime)s - %(name)s - %(levelname)s - %(message)s')
    handler = logging.StreamHandler(sys.stdout)
    handler.setFormatter(formatter)
    handler.setLevel(logging.DEBUG)
    logger.addHandler(handler)
    logger.propagate = False
    # loop over every sample, get eid
    # download s2 cloud raster for each sample tci
    if not Path(dir_path).is_dir():
        raise Exception('invalid directory')

    catalog = pystac_client.Client.open(
        "https://planetarycomputer.microsoft.com/api/stac/v1"
        # modifier=planetary_computer.sign_inplace
    )

    p = re.compile('tci_(\d{8})_(\d{8})_(.+).tif')
    lst = glob(dir_path + "[0-9]*")

    max_attempts = 3
    logger.debug('Sampling...')
    for sample in lst:
        logger.debug(f'Processing sample: {sample}')
        # first get the bbox, file crs from the metadata file
        with open(sample + '/metadata.json', 'r') as json_file:
            metadata = json.load(json_file)
            crs = metadata['metadata']['CRS']
            t_bbox = metadata['metadata']['Bounding Box']
            p_bbox = (t_bbox['minx'], t_bbox['miny'], t_bbox['maxx'], t_bbox['maxy'])
            conversion = transform(PRISM_CRS, SEARCH_CRS, (p_bbox[0], p_bbox[2]), (p_bbox[1], p_bbox[3]))
            s_bbox = (conversion[0][0], conversion[1][0], conversion[0][1], conversion[1][1])
        
        # SCL should match
        eid = sample.split('/')[-1]
        tcis = glob(sample + '/tci_*.tif')

        # download SCL for each TCI
        for tci in tcis:
            # first check if cloud raster already exists
            if Path(sample + f'/clouds_{dt}_{eid}.tif').is_file():
                logger.debug(f'---> Cloud raster already exists for tci on {dt}. Skipping...')
                continue
            
            m = p.search(tci)
            dt = m.group(1)
            img_dt = datetime.strptime(m.group(1), '%Y%m%d')
            event_dt = datetime.strptime(m.group(2), '%Y%m%d')

            tci_file = sample + f'/tci_{dt}_{eid}.tif'
            with rasterio.open(tci_file) as src:
                # get tci transform for resampling
                h = src.height
                w = src.width
                tci_transform = src.transform
        
            search_dt = img_dt.strftime('%Y-%m-%d')
            # Parse the input string into a datetime object
            save_as = sample + f'/clouds_{dt}_{eid}'

            for attempt in range(1, max_attempts + 1):
                try:
                    search_s2 = catalog.search(
                        collections=["sentinel-2-l2a"],
                        bbox=s_bbox,
                        datetime=search_dt,
                        query={"eo:cloud_cover": {"lt": 95}}
                    )
                    items_s2 = search_s2.item_collection()
                    # Break the loop if the code reaches this point without encountering an error
                    break
                except pystac_client.exceptions.APIError as err:
                    logger.error(f'PySTAC API Error: {err}, {type(err)}')
                    if attempt == max_attempts:
                        logger.error(f'Maximum number of attempts reached. Exiting.')
                        return False
                    else:
                        logger.info(f'Retrying ({attempt}/{max_attempts})...')
                except Exception as err:
                    logger.error(f'Catalog search failed: {err}, {type(err)}')
                    raise err
            download_SCL(items_s2, p_bbox, (h, w), tci_transform, crs, eid, save_as)
            # logger.debug(f'Sample cloud_{dt}_{eid} complete.')

    logger.debug('Cloud download complete.')

if __name__ == "__main__":
    parser = argparse.ArgumentParser(prog='add_to_sample', description='Generates additional rasters as needed')
    parser.add_argument('-d', '--dir', dest='dir_path', default='samples_200_6_4_10_sar/', help='specify a directory name for downloaded samples, format should end with backslash (default: samples_200_6_4_10_sar/)')
    args = parser.parse_args()

    main(args.dir_path)
