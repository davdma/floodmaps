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
from rasterio.warp import Resampling
import rasterio.merge
import rasterio
import os
import sys
from utils.utils import db_scale, setup_logging, colormap_to_rgb, crop_to_bounds, DateCRSOrganizer
from utils.stac_provider import get_stac_provider
from utils.validate import validate_event_rasters

PRISM_CRS = "EPSG:4269"
SEARCH_CRS = "EPSG:4326"

def get_metadata(sample):
    """Returns event folder metadata.
    
    Parameters
    ----------
    sample : str
        Path to event folder.

    Returns
    -------
    eid : str
        Event ID.
    event_date : str
        Event date.
    crs : str
        Coordinate reference system of the event.
    bbox : tuple
        Bounding box of the event in PRISM CRS.
    """
    with open(sample + 'metadata.json') as json_data:
        d = json.load(json_data)
        eid = d['metadata']['Sample ID']
        event_date = d['metadata']['Precipitation Event Date']
        crs = d['metadata']['CRS']
        minx = d['metadata']['Bounding Box']['minx']
        miny = d['metadata']['Bounding Box']['miny']
        maxx = d['metadata']['Bounding Box']['maxx']
        maxy = d['metadata']['Bounding Box']['maxy']
        
    return eid, event_date, crs, (minx, miny, maxx, maxy)

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

def sar_missing_percentage(stac_provider, item, item_crs, bbox):
    """Calculates the percentage of pixels in the bounding box of the SAR image that are missing."""
    vv_name = stac_provider.get_asset_names("s1")["vv"]
    item_href = stac_provider.sign_asset_href(item.assets[vv_name].href)

    conversion = transform(PRISM_CRS, item_crs, (bbox[0], bbox[2]), (bbox[1], bbox[3]))
    img_bbox = conversion[0][0], conversion[1][0], conversion[0][1], conversion[1][1]

    out_image, _ = rasterio.merge.merge([item_href], bounds=img_bbox)
    return int((np.sum(out_image <= 0) / out_image.size) * 100)

def pipeline_S1(stac_provider, dir_path, save_as, dst_crs, item, bbox):
    """Generates dB scale raster of SAR data in VV and VH polarizations.

    Parameters
    ----------
    stac_provider : STACProvider
        STAC provider object.
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
    vv_name = stac_provider.get_asset_names("s1")["vv"]
    vh_name = stac_provider.get_asset_names("s1")["vh"]
    item_hrefs_vv = stac_provider.sign_asset_href(item.assets[vv_name].href)
    item_hrefs_vh = stac_provider.sign_asset_href(item.assets[vh_name].href)

    out_image_vv, out_transform_vv = crop_to_bounds(item_hrefs_vv, bbox, dst_crs, nodata=0, resampling=Resampling.bilinear)
    out_image_vh, out_transform_vh = crop_to_bounds(item_hrefs_vh, bbox, dst_crs, nodata=0, resampling=Resampling.bilinear)

    with rasterio.open(dir_path + save_as + '_vv.tif', 'w', driver='Gtiff', count=1, height=out_image_vv.shape[-2], width=out_image_vv.shape[-1], crs=dst_crs, dtype=out_image_vv.dtype, transform=out_transform_vv, nodata=-9999) as dst:
        db_vv = db_scale(out_image_vv[0])
        dst.write(db_vv, 1)

    with rasterio.open(dir_path + save_as + '_vh.tif', 'w', driver='Gtiff', count=1, height=out_image_vh.shape[-2], width=out_image_vh.shape[-1], crs=dst_crs, dtype=out_image_vh.dtype, transform=out_transform_vh, nodata=-9999) as dst:
        db_vh = db_scale(out_image_vh[0])
        dst.write(db_vh, 1)

    # color maps
    img_vv_cmap = colormap_to_rgb(db_vv, cmap='gray', no_data=-9999)
    with rasterio.open(dir_path + save_as + '_vv_cmap.tif', 'w', driver='Gtiff', count=3, height=img_vv_cmap.shape[-2], width=img_vv_cmap.shape[-1], crs=dst_crs, dtype=np.uint8, transform=out_transform_vv, nodata=None) as dst:
        # get color map
        dst.write(img_vv_cmap)

    img_vh_cmap = colormap_to_rgb(db_vh, cmap='gray', no_data=-9999)
    with rasterio.open(dir_path + save_as + '_vh_cmap.tif', 'w', driver='Gtiff', count=3, height=img_vh_cmap.shape[-2], width=img_vh_cmap.shape[-1], crs=dst_crs, dtype=np.uint8, transform=out_transform_vh, nodata=None) as dst:
        dst.write(img_vh_cmap)

def coincident_days(s2_dts, s1_dt, within_days):
    """Returns the first S2 datetime that the S1 datetime is coincident with
    otherwise returns None.
    
    Parameters
    ----------
    s2_dts : list[datetime]
        List of S2 datetime objects (timezone naive).
    s1_dt : datetime
        S1 datetime object (timezone naive).
    within_days : int
        Number of days surrounding S2 dates allowed for S1 download.
    
    Returns
    -------
    datetime
        The datetime object of the S2 item that the S1 item is coincident with,
        otherwise returns None.
    """
    coincident_dt = None
    for s2_dt in s2_dts:
        time_difference = abs(s1_dt - s2_dt)
        days_difference = time_difference.days
        if days_difference <= within_days:
            coincident_dt = s2_dt
            break
    return coincident_dt

def downloadS1(stac_provider, sample, within_days, maxcoverpercentage, replace=True):
    """Downloads S1 imagery for a given event.
    
    Parameters
    ----------
    stac_provider : STACProvider
        STAC provider object.
    sample : str
        Path to event folder.
    within_days : int
        Number of days surrounding S2 dates allowed for S1 download.
    replace : bool
        Whether to overwrite existing SAR files found in the S2 sample folder.
    """
    # first need to check that the dates are close enough
    # extract event date and the dates of POST EVENT TCI tiles
    eid, event_date, crs, prism_bbox = get_metadata(sample)
    dt = datetime(int(event_date[0:4]), int(event_date[4:6]), int(event_date[6:8]))

    logger = logging.getLogger('events')
    logger.info('**********************************')
    logger.info('START OF SAR EVENT TASK LOG:')
    logger.info(f'Beginning event {eid} download at bounds: {prism_bbox}')

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

    # time of interest is event_date to last tci date + within_days
    time_of_interest = get_date_interval(dt, latest, within_days)
    
    # get bbox
    minx, miny, maxx, maxy = prism_bbox
    conversion = transform(PRISM_CRS, SEARCH_CRS, (minx, maxx), (miny, maxy))
    bbox = (conversion[0][0], conversion[1][0], conversion[0][1], conversion[1][1])

    # if sar file already exists do not download
    items_s1 = stac_provider.search_s1(bbox, time_of_interest, query={"sar:instrument_mode": {"eq": "IW"}})
            
    logger.info('Filtering catalog search results...')
    if len(items_s1) == 0:
        logger.info(f'Zero products from query for date interval {time_of_interest}.')
        return False

    logger.info(f'Checking s1 null percentage...')
    s1_by_date_crs = DateCRSOrganizer()    
    for item in items_s1:
        item_crs = pe.ext(item).crs_string
        polarizations = item.properties["sar:polarizations"]
        if "VV" not in polarizations or "VH" not in polarizations:
            logger.error(f'S1 product {item.id} VV or VH not found.')
            continue

        # filter out non coincident sar products with s2 products we've selected
        item_dt = datetime(item.datetime.year, item.datetime.month, item.datetime.day)
        coincident_dt = coincident_days(tci_dates, item_dt, within_days)
        if coincident_dt is None:
            logger.debug(f'S1 product {item.id} not coincident with any of the selected S2 products.')
            continue

        try:
            coverpercentage = sar_missing_percentage(stac_provider, item, item_crs, prism_bbox)
        except Exception as err:
            logger.error(f'Missing percentage calculation error for item {item.id}: {err}, {type(err)}')
            raise err

        if coverpercentage > maxcoverpercentage:
            logger.debug(f'SAR sample {item.id} near event {event_date}, at {minx}, {miny}, {maxx}, {maxy} rejected due to {coverpercentage}% missing data.')
            continue

        # should organize by the s2 item date that the s1 item is coincident with
        s1_by_date_crs.add_item(item, coincident_dt, item_crs)
    
    conversion = transform(PRISM_CRS, crs, (minx, maxx), (miny, maxy))
    cbbox = conversion[0][0], conversion[1][0], conversion[0][1], conversion[1][1] 
    for date in s1_by_date_crs.get_dates():
        # coincident date
        cdt = date.strftime('%Y%m%d')

        if not replace and any(Path(sample).glob(f'sar_{cdt}_*_{eid}_vv.tif')):
            logger.debug(f'S1 raster sar_{cdt}_*_{eid}_vv.tif already exists, skipping due to replace=False.')
            continue

        # item date
        item = s1_by_date_crs.get_primary_item_for_date(date, preferred_crs=crs)
        dt = item.datetime.strftime('%Y%m%d')

        pipeline_S1(stac_provider, sample, f'sar_{cdt}_{dt}_{eid}', crs, item, cbbox)
        logger.debug(f'S1 raster completed for {dt} coincident with S2 product at {cdt}.')
            
    logger.debug(f'All S1 rasters downloaded.')

    # validate raster shapes, CRS, transforms
    result = validate_event_rasters(sample, logger=logger)
    if not result.is_valid:
        logger.error(f'Raster validation failed for event {eid}. Removing directory and contents.')
        # shutil.rmtree(dir_path) - for now do not delete!
        raise Exception(f'Raster validation failed for event {eid}.')
        

    return True

def main(within_days, maxcoverpercentage, dir_path=None, replace=True, source='mpc'):
    """Samples S1 imagery on top of pre-existing S2 sample folder.
    Can run this on a S2 only directory downloaded using sample_s2.py, or to
    augment products in directory downloaded using sample_s2_s1.py.

    NOTE to developers: since specific datetimes of the already downloaded S2
    products are not known besides the day, we can only ascertain number of days between S2 and S1
    products rather than the hours. If you want to be more fine grained and precise with
    controlling S1 proximity to S2, use the sample_s2_s1.py script instead.
    A fix can also be implemented that queries the S2 product datetimes.
    
    Parameters
    ----------
    within_days : int
        Number of days surrounding S2 dates allowed for S1 download.
    dir_path : str
        Path to S2 sample folder.
    replace : bool
        Whether to overwrite existing SAR files found in the S2 sample folder.
    """
    assert dir_path is not None, 'Need to specify a directory'
    assert Path(dir_path).is_dir(), 'Directory invalid'
    
    # Ensure trailing slash
    dir_path = os.path.join(dir_path, '')

    # setup loggers
    rootLogger = setup_logging(dir_path, logger_name='main', log_level=logging.DEBUG, mode='w', include_console=False)
    logger = setup_logging(dir_path, logger_name='events', log_level=logging.DEBUG, mode='a', include_console=False)
    
    rootLogger.info(
        "S1 sampling parameters used:\n"
        f"  Within days of S2 dates: {within_days}\n"
        f"  Replace existing SAR files: {replace}"
    )

    # loop over samples in directory
    rootLogger.info("Initializing SAR event sampling...")
    samples = glob(dir_path + '*_*_*/')
    stac_provider = get_stac_provider(source, logger=logger)
    for sample in samples:
        downloadS1(stac_provider, sample, within_days, maxcoverpercentage, replace=replace)

if __name__ == '__main__':
    parser = argparse.ArgumentParser(prog='sampleS1', description='Samples imagery from Copernicus SENTINEL-1 (through a provider API) on top of pre-existing S2 sample folder.')
    parser.add_argument('-w', '--within_days', dest='within_days', default=2, type=int, help='number of days surrounding S2 dates allowed for download')
    parser.add_argument('-d', '--dir', dest='dir_path', help='specify a directory name for downloaded samples, format should end with backslash (default: None)')
    parser.add_argument('--replace', action='store_true', help='overwrite existing SAR files (default: False)')
    parser.add_argument('--source', choices=['mpc', 'aws'], default='mpc', help='Specify a provider (Microsoft Planetary Computer, AWS) for the ESA data (default: mpc)')
    parser.add_argument('--maxcover', dest='maxcoverpercentage', default=30, type=int, help='maximum SAR no data percentage (default: 30)')
    args = parser.parse_args()
    
    sys.exit(main(args.within_days, args.maxcoverpercentage, dir_path=args.dir_path, replace=args.replace, source=args.source))
