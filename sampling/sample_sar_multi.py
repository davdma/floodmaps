import argparse
from pathlib import Path
import numpy as np
import json
import logging
from datetime import datetime, timedelta
import pystac_client
from pystac.extensions.projection import ProjectionExtension as pe
import planetary_computer
import rasterio
from rasterio.warp import calculate_default_transform, reproject, Resampling
import rasterio.merge
from rasterio.vrt import WarpedVRT
import os
import sys
from glob import glob
from pathlib import Path
from typing import List, Tuple, Dict
from fiona.transform import transform

logging.basicConfig(level = logging.INFO)

# Set Planetary Computer API key
os.environ['PC_SDK_SUBSCRIPTION_KEY'] = 'a613baefa08445269838bc3bc0dfe2d9'
PRISM_CRS = "EPSG:4269"
SEARCH_CRS = "EPSG:4326"

def is_completed(sample_dir: Path) -> bool:
    """Check if the sample has already been processed."""
    return sample_dir.exists() and \
        any(sample_dir.glob("vv_*.tif")) and \
        any(sample_dir.glob("vh_*.tif")) and \
        any(sample_dir.glob("metadata.json"))

def get_bbox(sample: str) -> Tuple[float, float, float, float]:
    with open(sample + 'metadata.json') as json_data:
        d = json.load(json_data)
        event_date = d['metadata']['Precipitation Event Date']
        minx = d['metadata']['Bounding Box']['minx']
        miny = d['metadata']['Bounding Box']['miny']
        maxx = d['metadata']['Bounding Box']['maxx']
        maxy = d['metadata']['Bounding Box']['maxy']
        
    return event_date, minx, miny, maxx, maxy

def get_date_interval(start_dt, end_dt):
    """Returns a date interval from start date to end date.
    
    Parameters
    ----------
    start_dt : datetime object
    end_dt : datetime object

    Returns
    -------
    (str, str)
        Interval with start and end date strings formatted as YYYY-MM-DD.
    """
    return start_dt.strftime("%Y-%m-%d") + '/' + end_dt.strftime("%Y-%m-%d")

def setup_logging(output_dir: str) -> logging.Logger:
    """Setup logging configuration."""
    start_time = datetime.now().strftime("%Y-%m-%d_%H-%M")
    logger = logging.getLogger('multitemporal_sar')
    logger.setLevel(logging.DEBUG)
    
    # Create output directory if it doesn't exist
    Path(output_dir).mkdir(parents=True, exist_ok=True)
    
    fh = logging.FileHandler(os.path.join(output_dir, f'multitemporal_sar_{start_time}.log'))
    fh.setLevel(logging.DEBUG)
    formatter = logging.Formatter('%(asctime)s - %(name)s - %(levelname)s - %(message)s')
    fh.setFormatter(formatter)
    logger.addHandler(fh)
    
    return logger

def db_scale(x: np.ndarray) -> np.ndarray:
    """Convert SAR backscatter to dB scale."""
    nonzero_mask = x > 0
    missing_mask = x <= 0
    x[nonzero_mask] = 10 * np.log10(x[nonzero_mask])
    x[missing_mask] = -9999
    return x

def get_sar_items(bbox: Tuple[float, float, float, float], 
                  time_of_interest: str,
                  logger: logging.Logger) -> List:
    """Query Planetary Computer STAC API for Sentinel-1 items."""
    max_attempts = 3
    for attempt in range(1, max_attempts + 1):
        try:
            catalog = pystac_client.Client.open(
                "https://planetarycomputer.microsoft.com/api/stac/v1"
            )
            
            search = catalog.search(
                collections=["sentinel-1-rtc"],
                bbox=bbox,
                datetime=time_of_interest
            )
            
            items = search.item_collection()
            logger.info(f"Found {len(items)} SAR scenes")
            return items
            
        except pystac_client.exceptions.APIError as err:
            logger.error(f'PySTAC API Error: {err}, {type(err)}')
            if attempt == max_attempts:
                logger.error(f'Maximum number of attempts reached. Exiting.')
                return []
            else:
                logger.info(f'Retrying ({attempt}/{max_attempts})...')
        except Exception as err:
            logger.error(f'Catalog search failed: {err}, {type(err)}')
            raise err

def reproject(href, target_crs, resampling=Resampling.bilinear):
    """
    Reproject a single raster HREF to a target CRS using an in-memory WarpedVRT.

    Parameters:
        href (str): The path or URL to the raster (e.g., Planetary Computer asset href).
        target_crs (str or CRS): The target CRS to reproject to (e.g., 'EPSG:32633').
        resampling (Resampling): Resampling method for reprojection (default: bilinear).

    Returns:
        WarpedVRT: An in-memory reprojected raster that can be used like a dataset.
    """
    src = rasterio.open(href)
    
    # If already in the target CRS, just return the source
    if src.crs == target_crs:
        return src

    vrt = WarpedVRT(
        src,
        crs=target_crs,
        resampling=resampling,
        transform=None,  # Let rasterio calculate optimal transform
        height=None,
        width=None
    )
    return vrt

def validate_shapes(temporal_data: Dict[str, np.ndarray], logger: logging.Logger) -> Dict[str, np.ndarray]:
    """Ensure all vv and vh have the same shape."""

    if all(arr.shape == temporal_data['vv'][0].shape for arr in temporal_data['vv']) and \
        all(arr.shape == temporal_data['vh'][0].shape for arr in temporal_data['vh']):
        logger.info("All vv and vh have the same shape.")
        return temporal_data

    logger.info("Shape mismatch found. Resizing to smallest shape.")
    fixed_temporal_data = {'vv': [], 'vh': []}
    min_height_vv = min([vv.shape[0] for vv in temporal_data['vv']])
    min_width_vv = min([vv.shape[1] for vv in temporal_data['vv']])
    min_height_vh = min([vh.shape[0] for vh in temporal_data['vh']])
    min_width_vh = min([vh.shape[1] for vh in temporal_data['vh']])
    min_height = min(min_height_vv, min_height_vh)
    min_width = min(min_width_vv, min_width_vh)
    logger.info(f"Resizing to shape {min_height}x{min_width}.")
    for pol in ['vv', 'vh']:
        fixed_temporal_data[pol] = [data[:min_height, :min_width] for data in temporal_data[pol]]
    return fixed_temporal_data

def download_and_process_sar(items: List, 
                           output_dir: str,
                           bbox: Tuple[float, float, float, float],
                           acquisitions: int,
                           allow_missing: bool,
                           logger: logging.Logger) -> Dict[str, np.ndarray]:
    """Download and process SAR data for each acquisition date.
    
    Parameters
    ----------
    items : List
        List of STAC items to process.
    output_dir : str
        Output directory for saving files.
    bbox : Tuple[float, float, float, float]
        Bounding box of the search area in SEARCH_CRS = EPSG:4326.
    acquisitions : int
        Number of acquisitions to download.
    allow_missing : bool
        If True, allow missing data in the multitemporal SAR data.
    logger : logging.Logger
        Logger for logging messages.

    Returns
    -------
    temporal_data : Dict[str, np.ndarray]
        Temporal data for each polarization.
    metadata : dict
        Metadata for the multitemporal SAR data.
    """
    temporal_data = {'vv': [], 'vh': []}
    dates = [] # dates used out all acquisitions
    
    # Group items by date
    items_by_date = {}
    # choose one sample to extract reference parameters
    crs = pe.ext(items[0]).crs_string
    for item in items:
        # skip if item has no vv or vh
        if "vv" not in item.assets or "vh" not in item.assets:
            logger.info(f"Skipping {item.id} due to missing vv or vh")
            continue

        date = item.datetime.strftime('%Y-%m-%d')
        if date in items_by_date:
            items_by_date[date].append(item)
        else:
            items_by_date[date] = [item]
    
    count = 0
    # ensure items are sorted by date
    items_by_date_sorted = sorted(items_by_date.items(), key=lambda x: datetime.strptime(x[0], "%Y-%m-%d"))
    for date, date_items in items_by_date_sorted:
        logger.info(f"Processing SAR data for date: {date}")
        
        # check if need to reproject to reference crs
        vv_hrefs = []
        vh_hrefs = []
        for item in date_items:
            vv_href = planetary_computer.sign(item.assets["vv"].href)
            vh_href = planetary_computer.sign(item.assets["vh"].href)
            item_crs = pe.ext(item).crs_string
            if item_crs != crs:
                # instead of skipping, try to reproject to the reference crs
                logger.info(f"CRS mismatch for {item.id}: {item_crs} != reference {crs}")
                logger.info(f"Reprojecting {item.id} from {item_crs} to {crs}...")
                vv_reprojected = reproject(vv_href, crs, resampling=Resampling.bilinear)
                vh_reprojected = reproject(vh_href, crs, resampling=Resampling.bilinear)
                vv_hrefs.append(vv_reprojected)
                vh_hrefs.append(vh_reprojected)
                logger.info(f"{item.id} reprojected to {crs}.")
            else:
                vv_hrefs.append(vv_href)
                vh_hrefs.append(vh_href)

        # Download and merge VV polarization
        min_height = 999999
        min_width = 999999
        try:
            # use bilerp for best interpolation quality, also ensure no missing values
            # convert EPSG:4326 CRS of search to utm of data
            minx, miny, maxx, maxy = bbox
            conversion = transform(SEARCH_CRS, crs, (minx, maxx), (miny, maxy))
            item_bbox = (conversion[0][0], conversion[1][0], conversion[0][1], conversion[1][1])
            # Process VV
            vv_image, vv_transform = rasterio.merge.merge(vv_hrefs,
                                                        bounds=item_bbox,
                                                        resampling=Resampling.bilinear,
                                                        nodata=0)
            
            # Process VH
            vh_image, vh_transform = rasterio.merge.merge(vh_hrefs,
                                                        bounds=item_bbox,
                                                        resampling=Resampling.bilinear,
                                                        nodata=0)

            # skip if vv or vh contains no data
            if np.any(vv_image[0] <= 0) or np.any(vh_image[0] <= 0):
                if allow_missing:
                    logger.info(f"Missing data found for date: {date}.")
                    # only skip if missing data percentage is greater than 30%
                    missing_percentage = np.sum(vv_image[0] <= 0) / vv_image[0].size
                    if missing_percentage > 0.3:
                        logger.info(f"Missing data percentage {missing_percentage * 100}% is greater than 30%. Skipping...")
                        continue
                else:
                    logger.info(f"Missing data found for date: {date}. Skipping...")
                    continue

            vv_db = db_scale(vv_image[0])
            vh_db = db_scale(vh_image[0])
            temporal_data['vv'].append(vv_db)
            temporal_data['vh'].append(vh_db)
            
            dates.append(date)
            count += 1

            if count >= acquisitions:
                break
            
        except Exception as e:
            logger.error(f"Error processing data for date {date}: {str(e)}")
            continue

    if count < acquisitions:
        logger.debug(f"Not enough data found. Only found {count} usable acquistionsout of {len(items)} scenes.")
        return None, None

    logger.info(f"Merging completed for dates: {', '.join(dates)}")
    dates_obj = [datetime.strptime(date, '%Y-%m-%d') for date in dates]
    start_dt = min(dates_obj)
    end_dt = max(dates_obj)
    # Save metadata
    metadata = {
        'first_acquisition': start_dt.strftime('%Y-%m-%d'),
        'last_acquisition': end_dt.strftime('%Y-%m-%d'),
        'acquisition_dates': dates,
        'search_bbox': bbox,
        'transform': vv_transform,
        'search_crs': SEARCH_CRS,
        'item_crs': crs
    }
    
    return temporal_data, metadata

def save_multitemporal(temporal_data: Dict[str, np.ndarray],
                        start_dt: datetime,
                        end_dt: datetime,
                        output_dir: str,
                        eid: str,
                        metadata: dict,
                        logger: logging.Logger):
    """Save multitemporal data as a multi-band GeoTIFF."""
    logger.info(f"Saving multitemporal and composites in progress...")

    SAVE_DIR = Path(output_dir) / eid
    SAVE_DIR.mkdir(parents=True, exist_ok=True)

    # ensure all vv and vh have the same shape
    fixed_temporal_data = validate_shapes(temporal_data, logger)

    for pol in ['vv', 'vh']:
        # Stack temporal data
        stacked_data = np.stack(fixed_temporal_data[pol])
        
        # Save as multi-band GeoTIFF
        multitemporal_path = SAVE_DIR / f'{pol}_{metadata["first_acquisition"]}_{metadata["last_acquisition"]}.tif'
        
        logger.info(f"Saving multitemporal {pol} from {metadata['first_acquisition']} to {metadata['last_acquisition']}")
        with rasterio.open(
            multitemporal_path,
            'w',
            driver='GTiff',
            height=stacked_data.shape[1],
            width=stacked_data.shape[2],
            count=stacked_data.shape[0],
            dtype=stacked_data.dtype,
            crs=metadata['item_crs'],
            transform=metadata['transform'],
            nodata=-9999
        ) as dst:
            dst.write(stacked_data)
            
        logger.info(f"Multitemporal {pol} saved to {multitemporal_path}")
    
    # save metadata as json file
    metadata_path = SAVE_DIR / 'metadata.json'
    metadata.pop('transform')
    with open(metadata_path, 'w') as f:
        json.dump(metadata, f)

def download_multi(search_bbox: Tuple[float, float, float, float], search_start_dt: object,
                search_end_dt: object, event_dt: object, acquisitions: int, time_interval: int, 
                num_days: int, allow_missing: bool, output_dir: str, eid: str, logger: logging.Logger):
    """Downloads and composites SAR data for a given time interval and stack size.
    
    Parameters
    ----------
    search_bbox : Tuple[float, float, float, float]
        Bounding box of the search area.
    search_start_dt : datetime object
        Start date of the search interval.
    search_end_dt : datetime object
        End date of the search interval.
    event_dt : datetime object
        Date of the flood event.
    acquisitions : int
        Number of acquisitions to download.
    time_interval : int
        Time interval in days used as sliding window to find acquisitions.
    num_days : int
        Number of days before and after the flood event date to avoid sampling.
    allow_missing : bool
        If True, allow missing data in the multitemporal SAR data.
    output_dir : str
        Output directory for saving files.
    eid : str
        Event ID of the flood event.
    logger : logging.Logger
        Logger for logging messages.
    """
    # should log and save the time interval of composite: i.e. the date of the
    # first and last acquisitions
    logger.info(f'Downloading multitemporal sar for bbox {search_bbox} \
                with time interval {time_interval} between {search_start_dt} and {search_end_dt}.')

    # Get SAR items from STAC API
    current_start = search_start_dt
    current_end = current_start + timedelta(days=time_interval)
    found = False
    fails = 0
    while current_end <= search_end_dt:
        # if search interval within num_days of event_date, skip
        if (event_dt - current_end).days < num_days and (current_start - event_dt).days < num_days:
            logger.info(f'Skipping {current_start} to {current_end} due to overlap with event date {event_dt}...')
            current_start += timedelta(days=time_interval)
            current_end = current_start + timedelta(days=time_interval)
            continue

        time_of_interest = get_date_interval(current_start, current_end)
        logger.info(f'Searching {time_of_interest}...')
        items = get_sar_items(search_bbox, time_of_interest, logger)
        if len(items) >= acquisitions:
            logger.info(f'{len(items)} >= {acquisitions} minimum acquisitions found...')
            # Download and process SAR data
            temporal_data, metadata = download_and_process_sar(items, args.output_dir,
                                                              search_bbox, acquisitions,
                                                              allow_missing,
                                                              logger)
            if temporal_data is None:
                logger.debug(f"Not enough data found after merging. Trying next interval...")
                fails += 1
            else:
                found = True
                break

        if fails >= 3:
            logger.error(f"No SAR scenes found after 3 consecutive failed download attempts. Skipping search for bbox {search_bbox} \
                        with time interval {time_interval} between {search_start_dt} and {search_end_dt}.")
            return

        current_start += timedelta(days=time_interval)
        current_end = current_start + timedelta(days=time_interval)

    if not found:
        error_msg = f"No SAR scenes found. Skipping search for bbox {search_bbox} \
                    with time interval {time_interval} between {search_start_dt} and {search_end_dt}."
        logger.error(error_msg)
        return
    
    # Create and save multitemporal composite
    metadata.update({'search_start_dt': current_start.strftime('%Y-%m-%d'),
                     'search_end_dt': current_end.strftime('%Y-%m-%d'),
                     'event_dt': event_dt.strftime('%Y-%m-%d'),
                     'acquisitions': acquisitions,
                     'time_interval': time_interval,
                     'num_days': num_days})
    save_multitemporal(temporal_data, current_start, current_end,
                       output_dir, eid, metadata, logger)
    
    logger.info(f"Completed multitemporal SAR data collection and compositing for {search_bbox}")


def main(output_dir: str, time_interval: int, acquisitions: int, num_days: int,
        bbox: Tuple[float, float, float, float], allow_missing: bool):
    """Initializes multitemporal SAR data collection. Compositing should be done
    during preprocessing of the data, not here.

    Should specify to script a max number of acquisitions to acquire, and a time
    interval for samples. It will randomly sample a time interval that fits
    within the limits of the search space (and avoids the flood event date).

    Metadata should store the date of each acquisition in order and the bounding
    box of the roi.

    Parameters
    ----------
    output_dir : str
        Output directory for saving files.
    time_interval : int
        Max tolerated time interval in days between first and last acquisition.
    acquisitions : int
        Number of acquisitions to download in the time interval.
    num_days : int 
        Number of days before and after flood event date to avoid sampling.
    bbox : Tuple[float, float, float, float]
        Bounding box of the search area in PRISM_CRS = EPSG:4269.
    allow_missing : bool
        If True, allow missing data in the multitemporal SAR data.
    """
    # Setup logging
    logger = setup_logging(output_dir)
    logger.info("Starting multitemporal SAR data collection")
    
    # global search space:
    search_start_dt = datetime(2016, 9, 20)
    search_end_dt = datetime(2025, 4, 29)

    # only do single bbox - for testing purposes
    if bbox is not None:
        minx, miny, maxx, maxy = bbox
        conversion = transform(PRISM_CRS, SEARCH_CRS, (minx, maxx), (miny, maxy))

        event_dt = datetime(2017, 8, 26) # placeholder for testing
        eid = '20170826_487_695'

        search_bbox = (conversion[0][0], conversion[1][0], conversion[0][1], conversion[1][1])
        download_multi(search_bbox, search_start_dt, search_end_dt, event_dt,
                        acquisitions, time_interval, num_days, allow_missing, output_dir, eid, logger)
    else:
        # get all bboxes in the sample directory
        samples = glob('samples_200_6_4_10_sar/*_*_*/')
        test = 0
        for sample in samples:
            # first check if the sample has already been processed
            eid = sample.split('/')[-2]
            sample_dir = Path(output_dir) / eid
            if is_completed(sample_dir):
                logger.info(f"Sample {eid} already processed. Skipping...")
                continue

            event_date, minx, miny, maxx, maxy = get_bbox(sample)
            conversion = transform(PRISM_CRS, SEARCH_CRS, (minx, maxx), (miny, maxy))
            search_bbox = (conversion[0][0], conversion[1][0], conversion[0][1], conversion[1][1])
            event_dt = datetime(int(event_date[0:4]), int(event_date[4:6]), int(event_date[6:8]))

            logger.info(f'Downloading multitemporal sar for EID {eid}...')
            download_multi(search_bbox, search_start_dt, search_end_dt, event_dt,
                        acquisitions, time_interval, num_days, allow_missing, output_dir, eid, logger)
    
    return 0

if __name__ == '__main__':
    parser = argparse.ArgumentParser(description='Create multitemporal SAR composites from Sentinel-1 data')
    # if no bbox is provided, will use all bboxes in the sample directory
    parser.add_argument('--output_dir', type=str, required=True,
                      help='Output directory for saving composites')
    parser.add_argument('--time_interval', type=int, required=True,
                        help='Time interval in days between first and last acquisition')
    parser.add_argument('--acquisitions', type=int, required=True,
                        help='Number of acquisitions to download in the time interval')
    parser.add_argument('--num_days', type=int, required=True, 
                        help='Days before and after flood event date to avoid sampling')
    parser.add_argument('--bbox', nargs=4, type=float, default=None,
                      help='Bounding box coordinates (minx miny maxx maxy) in EPSG:4326')
    parser.add_argument('--allow_missing', action='store_true',
                        help='Allow missing data in the time interval')
    
    args = parser.parse_args()
    sys.exit(main(args.output_dir, args.time_interval, args.acquisitions, args.num_days, args.bbox, args.allow_missing))