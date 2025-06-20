import argparse
from glob import glob
from fiona.transform import transform
from pathlib import Path
import re
import numpy as np
import json
import logging
from datetime import datetime, timedelta
from pystac.extensions.projection import ProjectionExtension as pe
import pystac_client
import planetary_computer
from rasterio.warp import calculate_default_transform, reproject, Resampling
import rasterio.merge
import rasterio
from matplotlib.cm import ScalarMappable
from matplotlib.colors import Normalize
import os
import sys

os.environ['PC_SDK_SUBSCRIPTION_KEY'] = 'a613baefa08445269838bc3bc0dfe2d9'
PRISM_CRS = "EPSG:4269"
SEARCH_CRS = "EPSG:4326"

def get_metadata(sample):
    with open(sample + 'metadata.json') as json_data:
        d = json.load(json_data)
        eid = d['metadata']['Sample ID']
        event_date = d['metadata']['Precipitation Event Date']
        minx = d['metadata']['Bounding Box']['minx']
        miny = d['metadata']['Bounding Box']['miny']
        maxx = d['metadata']['Bounding Box']['maxx']
        maxy = d['metadata']['Bounding Box']['maxy']
        
    return eid, event_date, minx, miny, maxx, maxy

def get_date_interval(event_dt, latest_dt, days_after):
    """Returns a date interval from beginning of event date to the last TCI tile date plus days after parameter.
    
    Parameters
    ----------
    event_dt : datetime object
    latest_dt : datetime object
    days_after : int

    Returns
    -------
    (str, str)
        Interval with start and end date strings formatted as YYYY-MM-DD.
    """
    delt = timedelta(days = days_after)
    end = latest_dt + delt
    return event_dt.strftime("%Y-%m-%d") + '/' + end.strftime("%Y-%m-%d")

def db_scale(x):
    nonzero_mask = x > 0
    missing_mask = x <= 0
    x[nonzero_mask] = 10 * np.log10(x[nonzero_mask])
    x[missing_mask] = -9999
    return x

def download_raster(href, filepath):
    with rasterio.open(href) as src:
        kwargs = src.meta.copy()
        with rasterio.open(filepath, 'w', **kwargs) as dst:
            dst.write(src.read())

def download_convert_raster(href, filepath, dst_crs, resampling=Resampling.nearest):  
    # for SCL we use nearest resampling - otherwise use bilinear
    with rasterio.open(href) as src:
        transform, width, height = calculate_default_transform(
            src.crs, dst_crs, src.width, src.height, *src.bounds)
        kwargs = src.meta.copy()
        kwargs.update({
            'crs': dst_crs,
            'transform': transform,
            'width': width,
            'height': height
        })

        with rasterio.open(filepath, 'w', **kwargs) as dst:
            for i in range(1, src.count + 1):
                reproject(
                    source=rasterio.band(src, i),
                    destination=rasterio.band(dst, i),
                    src_transform=src.transform,
                    src_crs=src.crs,
                    dst_transform=transform,
                    dst_crs=dst_crs,
                    resampling=resampling)

def pipeline_S1(dir_path, save_as, dst_crs, items, bbox):
    """Generates dB scale raster of SAR data in VV and VH polarizations.

    Parameters
    ----------
    dir_path : str
        Path for saving generated raster. 
    save_as : str
        Name of file to be saved (do not include extension!)
    dst_crs : obj
        Coordinate reference system of output raster.
    items : list[Item]
        List of PyStac Item objects
    bbox : (float, float, float, float)
        Tuple in the order minx, miny, maxx, maxy, representing bounding box, 
        should be in CRS specified by dst_crs.

    Returns
    -------
    shape : (int, int)
        Shape of the raster array.
    transform : rasterio.affine.Affine()
        Transformation matrix for mapping pixel coordinates in dest to coordinate system.
    """
    item_crs = []
    item_hrefs_vv = []
    item_hrefs_vh = []
    for item in items:
        item_crs.append(pe.ext(item).crs_string)
        item_hrefs_vv.append(planetary_computer.sign(item.assets["vv"].href))
        item_hrefs_vh.append(planetary_computer.sign(item.assets["vh"].href))

    if all(item == dst_crs for item in item_crs):
        # no conversions necessary
        # perform transformation on bounds to match crs
        out_image_vv, out_transform_vv = rasterio.merge.merge(item_hrefs_vv, bounds=bbox, nodata=0)
        out_image_vh, out_transform_vh = rasterio.merge.merge(item_hrefs_vh, bounds=bbox, nodata=0)
    else:
        filepaths_vv = []
        filepaths_vh = []
        for i, file in enumerate(item_hrefs_vv):
            filepath = dir_path + f'sar_tmp_vv_{i}.tif'
            if item_crs[i] == dst_crs:
                download_raster(file, filepath)
            else:
                # download each file and save with new crs
                download_convert_raster(file, filepath, dst_crs, resampling=Resampling.bilinear)
            filepaths_vv.append(filepath)

        for i, file in enumerate(item_hrefs_vh):
            filepath = dir_path + f'sar_tmp_vh_{i}.tif'
            if item_crs[i] == dst_crs:
                download_raster(file, filepath)
            else:
                # download each file and save with new crs
                download_convert_raster(file, filepath, dst_crs, resampling=Resampling.bilinear)
            filepaths_vh.append(filepath)

        out_image_vv, out_transform_vv = rasterio.merge.merge(filepaths_vv, bounds=bbox, nodata=0)
        out_image_vh, out_transform_vh = rasterio.merge.merge(filepaths_vh, bounds=bbox, nodata=0)
        
        for filepath in filepaths_vv:
            os.remove(filepath)
            
        for filepath in filepaths_vh:
            os.remove(filepath)

    with rasterio.open(dir_path + save_as + '_vv.tif', 'w', driver='Gtiff', count=1, height=out_image_vv.shape[-2], width=out_image_vv.shape[-1], crs=dst_crs, dtype=out_image_vv.dtype, transform=out_transform_vv, nodata=-9999) as dst:
        db_vv = db_scale(out_image_vv[0])
        dst.write(db_vv, 1)

    with rasterio.open(dir_path + save_as + '_vh.tif', 'w', driver='Gtiff', count=1, height=out_image_vh.shape[-2], width=out_image_vh.shape[-1], crs=dst_crs, dtype=out_image_vh.dtype, transform=out_transform_vh, nodata=-9999) as dst:
        db_vh = db_scale(out_image_vh[0])
        dst.write(db_vh, 1)

    # color maps
    with rasterio.open(dir_path + save_as + '_vv_cmap.tif', 'w', driver='Gtiff', count=4, height=out_image_vv.shape[-2], width=out_image_vv.shape[-1], crs=dst_crs, dtype=np.uint8, transform=out_transform_vv) as dst:
        img_norm = Normalize(vmin=np.min(db_vv[db_vv != -9999]), vmax=np.max(db_vv))
        img_cmap = ScalarMappable(norm=img_norm, cmap='gray')
        img = img_cmap.to_rgba(db_vv, bytes=True)
        img = np.clip(img, 0, 255).astype(np.uint8)
        # get color map
        dst.write(np.transpose(img, (2, 0, 1)))

    with rasterio.open(dir_path + save_as + '_vh_cmap.tif', 'w', driver='Gtiff', count=4, height=out_image_vh.shape[-2], width=out_image_vh.shape[-1], crs=dst_crs, dtype=np.uint8, transform=out_transform_vh) as dst:
        img_norm = Normalize(vmin=np.min(db_vh[db_vv != -9999]), vmax=np.max(db_vh))
        img_cmap = ScalarMappable(norm=img_norm, cmap='gray')
        img = img_cmap.to_rgba(db_vh, bytes=True)
        img = np.clip(img, 0, 255).astype(np.uint8)
        # get color map
        dst.write(np.transpose(img, (2, 0, 1)))

def downloadS1(sample, proximity, replace=True):
    # first need to check that the dates are close enough
    # extract event date and the dates of POST EVENT TCI tiles
    eid, event_date, minx, miny, maxx, maxy = get_metadata(sample)
    dt = datetime(int(event_date[0:4]), int(event_date[4:6]), int(event_date[6:8]))

    logger = logging.getLogger('main')
    logger.info('**********************************')
    logger.info('START OF SAR EVENT TASK LOG:')
    logger.info(f'Beginning event {eid} download at bounds: {minx}, {miny}, {maxx}, {maxy}')

    # find the post TCI files and their dates
    p = re.compile('\d{8}_\d{8}')
    tci_dates = []
    latest = dt
    for tci_file in glob(sample + 'tci_*.tif'):
        match = p.search(tci_file)
        if match:
            tci_date = match.group()[:8]
            tci_dt = datetime(int(tci_date[0:4]), int(tci_date[4:6]), int(tci_date[6:8]))
            if tci_dt >= dt:
                tci_dates.append(tci_dt)
                if tci_dt > latest:
                    latest = tci_dt

    # time of interest is event_date to last tci date + proximity
    time_of_interest = get_date_interval(dt, latest, proximity)
    
    # get bbox
    conversion = transform(PRISM_CRS, SEARCH_CRS, (minx, maxx), (miny, maxy))
    bbox = (conversion[0][0], conversion[1][0], conversion[0][1], conversion[1][1])

    # if sar file already exists do not download
    max_attempts = 3
    for attempt in range(1, max_attempts + 1):
        try:
            catalog = pystac_client.Client.open(
                "https://planetarycomputer.microsoft.com/api/stac/v1"
                # modifier=planetary_computer.sign_inplace
            )
        
            search = catalog.search(
                collections=["sentinel-1-rtc"],
                bbox=bbox,
                datetime=time_of_interest
            )
            
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
            
    logger.info('Filtering catalog search results...')
    items = search.item_collection()
    if len(items) == 0:
        logger.info(f'Zero products from query for date interval {time_of_interest}.')
        return False

    # group items by dates in dictionary
    products_by_date = dict()
    for item in items:
        dt = item.datetime.strftime('%Y%m%d')
        if dt in products_by_date:
            products_by_date[dt].append(item)
        else:
            products_by_date[dt] = [item]

    # download all that satisfy criteria
    # download both VV and VH polarizations
    dst_crs = pe.ext(list(products_by_date.values())[0][0]).crs_string
    if dst_crs != PRISM_CRS:
        conversion = transform(PRISM_CRS, dst_crs, (minx, maxx), (miny, maxy))
        cbbox = conversion[0][0], conversion[1][0], conversion[0][1], conversion[1][1]
    else:
        cbbox = (minx, miny, maxx, maxy)
        
    for dt, items in list(products_by_date.items()):
        pipeline_S1(sample, f'sar_{dt}_{eid}', dst_crs, items, cbbox)
            
    logger.debug(f'All S1 rasters downloaded.')
    return True

def main(proximity, dir_path=None, replace=True):
    # root logger
    start_time = datetime.now().strftime("%Y-%m-%d_%H-%M")
    rootLogger = logging.getLogger('main')
    rootLogger.setLevel(logging.DEBUG)
    fh = logging.FileHandler(dir_path + f'main_{start_time}.log', mode='w')
    fh.setLevel(logging.DEBUG)
    formatter = logging.Formatter('%(asctime)s - %(name)s - %(levelname)s - %(message)s')
    fh.setFormatter(formatter)
    rootLogger.addHandler(fh)
    
    if dir_path is None:
        raise Exception('Need to specify a directory')
    if not Path(dir_path).is_dir() or not dir_path:
        raise Exception('Directory invalid')
    if dir_path[-1] != '/':
        dir_path += '/'

    # loop over samples in directory
    rootLogger.info("Initializing SAR event sampling...")
    samples = glob(dir_path + '*_*_*/')
    for sample in samples:
        downloaded = downloadS1(sample, proximity, replace=replace)

if __name__ == '__main__':
    parser = argparse.ArgumentParser(prog='sampleS1', description='Samples imagery from Copernicus SENTINEL-1 (through Microsoft Planetary Computer API)')
    parser.add_argument('-p', '--proximity', dest='proximity', default=2, type=int, help='number of days surrounding tile dates allowed for download')
    parser.add_argument('-d', '--dir', dest='dir_path', help='specify a directory name for downloaded samples, format should end with backslash (default: None)')
    parser.add_argument('--replace', action='store_true', help='overwrite existing SAR files (default: False)')
    args = parser.parse_args()
    
    sys.exit(main(args.proximity, dir_path=args.dir_path, replace=args.replace))
