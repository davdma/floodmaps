from pathlib import Path
import numpy as np
import json
import logging
import time
from datetime import datetime, timedelta
import pystac_client
import planetary_computer
import rasterio
from rasterio.warp import calculate_default_transform, reproject, Resampling
import rasterio.merge
from rasterio.vrt import WarpedVRT
from glob import glob
from pathlib import Path
from typing import List, Tuple, Dict
from fiona.transform import transform
import hydra
from omegaconf import DictConfig
import sys
import os
import re
import random

from floodmaps.utils.stac_providers import get_stac_provider, STACProvider

from floodmaps.utils.sampling_utils import (
    PRISMData,
    read_cell_coords,
    PRISM_CRS,
    SEARCH_CRS,
    setup_logging,
    get_item_crs,
    walltime_seconds,
    get_history,
    db_scale
)

def is_completed(sample_dir: Path) -> bool:
    """Check if the sample has already been processed."""
    return sample_dir.exists() and \
        any(sample_dir.glob("vv_*.tif")) and \
        any(sample_dir.glob("vh_*.tif")) and \
        any(sample_dir.glob("metadata.json"))

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
                           dir_path: Path,
                           bbox: Tuple[float, float, float, float],
                           acquisitions: int,
                           allow_missing: bool,
                           logger: logging.Logger) -> Dict[str, np.ndarray]:
    """Download and process SAR data for each acquisition date.
    
    Parameters
    ----------
    items : List
        List of STAC items to process.
    dir_path : Path
        Directory path for saving files.
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
    crs = get_item_crs(items[0])
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
            item_crs = get_item_crs(item)
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
        "Download Date": datetime.now().strftime('%Y-%m-%d'),
        "Sample ID": eid,
        "CRS": crs,
        "Bounding Box": bbox,
        "S1 Products": s1_products,
        "Source": source,
        'first_acquisition': start_dt.strftime('%Y-%m-%d'),
        'last_acquisition': end_dt.strftime('%Y-%m-%d'),
        'acquisition_dates': dates,
        'search_bbox': bbox,
        'transform': vv_transform,
        'search_crs': SEARCH_CRS,
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
    with open(metadata_path, 'w') as json_file:
        json.dump(metadata, json_file, indent=4)

def multi_sample(s1_stac_provider: STACProvider, y: int, x: int, crs: str, eid: str,
            dir_path: Path, prism_data: PRISMData, seed: int,
            cfg: DictConfig, logger: logging.Logger):
    """Downloads and composites SAR data for a given time interval and stack size."""

    def valid_interval_start_idx(arr, N, X, x=0):
        """
        Return start indices i (0 <= i <= L-N) such that
        arr[max(0, i-x) : i+N] has all values < X.
        """
        arr = np.asarray(arr)
        L = arr.shape[0]
        if N <= 0 or N > L:
            return np.array([], dtype=int)
        if x < 0:
            raise ValueError("x must be >= 0")

        bad = (arr >= X).astype(np.int32)                 # 1 where condition fails
        pref = np.concatenate(([0], np.cumsum(bad)))      # pref[k] = #bad in arr[:k]

        i = np.arange(L - N + 1)
        a = np.maximum(0, i - x)
        b = i + N

        bad_count = pref[b] - pref[a]                     # #bad in arr[a:b]
        return np.flatnonzero(bad_count == 0)

    # should log and save the time interval of composite: i.e. the date of the
    # first and last acquisitions
    prism_bbox = prism_data.get_bounding_box(x, y)
    minx, miny, maxx, maxy = prism_bbox
    conversion = transform(PRISM_CRS, SEARCH_CRS, (minx, maxx), (miny, maxy))
    search_bbox = (conversion[0][0], conversion[1][0], conversion[0][1], conversion[1][1])
    rng = random.Random(seed)

    logger.info(f'Downloading multitemporal sar at coordinates {y}, {x} with bbox {search_bbox} and CRS {crs} \
                with time interval={cfg.sampling.time_interval} and number of intervals={cfg.sampling.num_time_intervals}')
    
    # sample a random interval between all prism dates
    # that does not have precip above a certain threshold if threshold not None
    # prefilter here
    prism_data_for_cell = prism_data.precip_data[:, y, x]

    if cfg.sampling.threshold is not None:
        all_intervals = list(valid_interval_start_idx(prism_data_for_cell,
                                                        cfg.sampling.time_interval,
                                                        cfg.sampling.threshold,
                                                        x=cfg.sampling.num_days))
    else:
        all_intervals = list(range(0, prism_data_for_cell.shape[0]))
    
    if len(all_intervals) == 0:
        logger.error(f'No valid intervals found after prefiltering by threshold={cfg.sampling.threshold}')
        return
    
    rng.shuffle(all_intervals)

    attempts = 0
    num_intervals_sampled = 0
    target_intervals_sampled = cfg.sampling.num_time_intervals
    max_tries = cfg.sampling.max_tries
    for interval in all_intervals:
        if num_intervals_sampled >= target_intervals_sampled:
            logger.info(f"Reached target number of intervals {target_intervals_sampled}. Stopping...")
            break
        
        if attempts >= max_tries:
            logger.error(f"No SAR scenes found after {max_tries} consecutive failed download attempts. Skipping search for bbox {search_bbox} \
                        with time interval {cfg.sampling.time_interval}.")
            return

        interval_start_dt = prism_data.get_event_datetime(interval)
        interval_end_dt = interval_start_dt + timedelta(days=cfg.sampling.time_interval)

        time_of_interest = get_date_interval(interval_start_dt, interval_end_dt)
        logger.info(f'Searching {time_of_interest}...')
        items = s1_stac_provider.search_s1(search_bbox, time_of_interest, query={"sar:instrument_mode": {"eq": "IW"}})
        if len(items) >= cfg.sampling.acquisitions:
            logger.info(f'{len(items)} >= {cfg.sampling.acquisitions} minimum acquisitions found...')
            # Download and process SAR data
            temporal_data, metadata = download_and_process_sar(items, dir_path,
                                                              search_bbox, cfg.sampling.acquisitions,
                                                              cfg.sampling.allow_missing,
                                                              logger)
            if temporal_data is None:
                logger.debug(f"Not enough data found after merging. Trying next interval...")
                fails += 1
            else:
                found = True
                break


    if not found:
        error_msg = f"No SAR scenes found. Skipping search for bbox {search_bbox} \
                    with time interval {cfg.sampling.time_interval}."
        logger.error(error_msg)
        return
    
    # Create and save multitemporal composite
    metadata.update({'search_start_dt': current_start.strftime('%Y-%m-%d'),
                     'search_end_dt': current_end.strftime('%Y-%m-%d'),
                     'event_dt': event_dt.strftime('%Y-%m-%d'),
                     'acquisitions': cfg.sampling.acquisitions,
                     'time_interval': cfg.sampling.time_interval,
                     'num_days': cfg.sampling.num_days})
    save_multitemporal(temporal_data, current_start, current_end,
                       cfg.sampling.output_dir, eid, metadata, logger)
    
    logger.info(f"Completed multitemporal SAR data collection and compositing for {search_bbox}")

def get_default_dir_name(time_interval: int, num_time_intervals: int, acquisitions: int, num_days: int) -> str:
    """Default directory name."""
    return f's1_multi_{time_interval}_{num_time_intervals}_{acquisitions}_{num_days}/'

@hydra.main(version_base=None, config_path="pkg://configs", config_name="config.yaml")
def main(cfg: DictConfig) -> None:
    """Runs multitemporal SAR data collection on a specified set of PRISM cells.
    Should specify to script a number of acquisitions to acquire per PRISM cell
    (the size of the multitemporal stack), the max size of the time interval
    allowed between first and last acquisition in days, and number of intervals.
    For each PRISM cell, the script randomly searches time intervals of the
    specified size within the limits of the global dates search space while
    avoiding a flood event date if specified (plus minus cfg.sampling.num_days).

    For source it is recommended to use 'mpc' for Radiometric Terrain Correction
    data.

    NOTE: To avoid the failure mode of mixing geometries due to compositing
    ascending with descending orbits, the acquisitions for each time interval
    will only contain one orbit. De-duplication is also done for multiple
    acquisitions on the same datatake.
    
    NOTE: Compositing is done during preprocessing of the data, not here.

    Parameters
    ----------
    cfg : DictConfig
        Configuration object.
    
    cfg.sampling parameters
    -----------------------
    dir_path : str
        Path to directory where downloaded multitemporal samples will be saved.
    cells_file : str
        Path to text file containing cell coordinates in lines with format: y, x
    seed : int
        Seed for random sampling time intervals.
    acquisitions : int
        Number of acquisitions to download in each random time interval.
    time_interval : int
        Time interval in days between first and last acquisition.
    num_time_intervals : int
        Number of time intervals to sample for each cell for temporal diversity.
    threshold : int
        Precipitation threshold in mm to consider as flood event date and to
        filter out time intervals. If None then no threshold is applied.
    num_days : int
        Number of days after flood event date to avoid sampling.
    allow_missing : bool
        If True, allow missing values in the multitemporal SAR data.
    max_tries : int
        Max number of time intervals to try for each cell before giving up.
    max_runtime : str
        Maximum runtime in the format of HH:MM:SS
    source : str
        Source for S1 data. ['mpc', 'aws', 'cdse']
        
    Returns
    -------
    int
    """
    # make directory
    if cfg.sampling.dir_path is None:
        cfg.sampling.dir_path = str(Path(cfg.paths.imagery_dir) / get_default_dir_name(cfg.sampling.time_interval, cfg.sampling.num_time_intervals, cfg.sampling.acquisitions, cfg.sampling.num_days))
        Path(cfg.sampling.dir_path).mkdir(parents=True, exist_ok=True)
    else:
        # Create directory if it doesn't exist
        try:
            Path(cfg.sampling.dir_path).mkdir(parents=True, exist_ok=True)
        except (OSError, ValueError) as e:
            print(f"Invalid directory path '{cfg.sampling.dir_path}'. Using default.", file=sys.stderr)
            cfg.sampling.dir_path = str(Path(cfg.paths.imagery_dir) / get_default_dir_name(cfg.sampling.time_interval, cfg.sampling.num_time_intervals, cfg.sampling.acquisitions, cfg.sampling.num_days))
            Path(cfg.sampling.dir_path).mkdir(parents=True, exist_ok=True)
        
        # Ensure trailing slash
        cfg.sampling.dir_path = os.path.join(cfg.sampling.dir_path, '')

    # root logger
    rootLogger = setup_logging(cfg.sampling.dir_path, logger_name='main', log_level=logging.DEBUG, mode='w', include_console=False)

    # event logger
    logger = setup_logging(cfg.sampling.dir_path, logger_name='events', log_level=logging.DEBUG, mode='a', include_console=False)

    # log sampling parameters used
    rootLogger.info(
        "S1 multitemporal SAR sampling parameters used:\n"
        f"  Save directory: {cfg.sampling.dir_path}\n"
        f"  Use cells in file: {cfg.sampling.cells_file}\n"
        f"  Random seed: {cfg.sampling.seed}\n"
        f"  Number of acquisitions: {cfg.sampling.acquisitions}\n"
        f"  Time interval: {cfg.sampling.time_interval}\n"
        f"  Number of time intervals: {cfg.sampling.num_time_intervals}\n"
        f"  Precipitation threshold: {cfg.sampling.threshold}\n"
        f"  Number of days to avoid precip event date: {cfg.sampling.num_days}\n"
        f"  Allow missing data: {cfg.sampling.allow_missing}\n"
        f"  Max number of tries: {cfg.sampling.max_tries}\n"
        f"  Max runtime: {getattr(cfg.sampling, 'max_runtime', 'Unlimited')}\n"
        f"  Source: {cfg.sampling.source}\n"
    )
    logger.info("Starting multitemporal SAR data collection")
    
    # to track runtime
    # max_runtime_seconds = (
    #     walltime_seconds(cfg.sampling.max_runtime)
    #     if hasattr(cfg.sampling, 'max_runtime') and cfg.sampling.max_runtime is not None
    #     else float('inf')
    # )
    # start_time = time.time()

    # history will just be based on cell coords
    ## Set of (y, x) tuples
    history = get_history(Path(cfg.sampling.dir_path) / 'history.pickle')

    # filter out time intervals containing PRISM precip above a certain threshold (plus minus num_days)
    # more general
    # load PRISM data object
    prism_data = PRISMData.from_file(cfg.paths.prism_data)

    # first gather all the PRISM cells and dates from input directory / manual file
    ## List of (y, x, crs) tuples where crs can be None
    if Path(cfg.sampling.cells_file).exists():
        cells = read_cell_coords(cfg.sampling.cells_file)
    else:
        raise ValueError(f"Cells file {cfg.sampling.cells_file} does not exist")

    # if no cells specified raise
    if len(cells) == 0:
        raise ValueError(f"No cells found in file {cfg.sampling.cells_file} for multitemporal SAR sampling")

    logger.info(f"Found {len(cells)} cells in file {cfg.sampling.cells_file} for multitemporal SAR sampling")
    filtered_cells = [cell for cell in cells if (cell[0], cell[1]) not in history]
    logger.info(f"Filtered to {len(filtered_cells)} cells not in history for multitemporal SAR sampling")

    s1_stac_provider = get_stac_provider(cfg.sampling.source.lower(),
                                        mpc_api_key=getattr(cfg, "mpc_api_key", None),
                                        aws_access_key_id=getattr(cfg, "aws_access_key_id", None),
                                        aws_secret_access_key=getattr(cfg, "aws_secret_access_key", None),
                                        logger=logger)

    sample_dir = Path(cfg.sampling.dir_path)
    sample_dir.mkdir(parents=True, exist_ok=True)
    for cell in filtered_cells:
        y, x, crs = cell
        logger.info(f"Downloading multitemporal sar for cell {y}, {x} in CRS {crs}...")

        # first check if the sample has already been processed
        eid = f'{y}_{x}'
        dir_path = sample_dir / eid
        if is_completed(dir_path):
            logger.info(f"Sample {eid} already processed. Skipping...")
            continue

        multi_sample(s1_stac_provider, y, x, crs, eid, dir_path, prism_data, cfg, logger)


if __name__ == '__main__':
    main()