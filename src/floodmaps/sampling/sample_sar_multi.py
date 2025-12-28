from pathlib import Path
import numpy as np
import json
import logging
import time
import pickle
from datetime import datetime, timedelta
import pystac_client
import rasterio
from rasterio.warp import Resampling
from typing import List, Tuple, Dict, Optional
from fiona.transform import transform
import hydra
from omegaconf import DictConfig
import sys
import os
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
    db_scale,
    crop_to_bounds,
    MissingAssetError
)

CURRENT_DATE = datetime.now().strftime('%Y-%m-%d')


def get_date_interval(start_dt: datetime, end_dt: datetime) -> str:
    """Returns a date interval from start date to end date.
    
    Parameters
    ----------
    start_dt : datetime
        Start date.
    end_dt : datetime
        End date.

    Returns
    -------
    str
        Interval with start and end date strings formatted as YYYY-MM-DD/YYYY-MM-DD.
    """
    return start_dt.strftime("%Y-%m-%d") + '/' + end_dt.strftime("%Y-%m-%d")


def sar_missing_percentage(stac_provider: STACProvider, item, item_crs: str, prism_bbox: Tuple[float, float, float, float]) -> float:
    """Calculates the percentage of pixels in the bounding box of the SAR image that are missing.
    
    Parameters
    ----------
    stac_provider : STACProvider
        STAC provider object.
    item : Item
        PyStac Item object.
    item_crs : str
        CRS of the item.
    prism_bbox : Tuple[float, float, float, float]
        Bounding box in PRISM CRS (EPSG:4269).

    Returns
    -------
    float
        Percentage of missing pixels (0-100).
    """
    vv_name = stac_provider.get_asset_names("s1")["vv"]
    
    if vv_name not in item.assets:
        raise MissingAssetError(f"Asset '{vv_name}' not found in S1 item {item.id}")
    
    item_href = stac_provider.sign_asset_href(item.assets[vv_name].href)

    # Convert PRISM bbox to item CRS
    minx, miny, maxx, maxy = prism_bbox
    conversion = transform(PRISM_CRS, item_crs, (minx, maxx), (miny, maxy))
    img_bbox = (conversion[0][0], conversion[1][0], conversion[0][1], conversion[1][1])

    out_image, _ = crop_to_bounds(item_href, img_bbox, item_crs, nodata=0, resampling=Resampling.bilinear)
    return float((np.sum(out_image <= 0) / out_image.size) * 100)


def group_items_by_orbit(items: List) -> Dict[str, List]:
    """Groups S1 items by orbit direction (ascending/descending).
    
    Parameters
    ----------
    items : List
        List of STAC items.

    Returns
    -------
    Dict[str, List]
        Dictionary with 'ascending' and 'descending' keys containing lists of items.
    """
    grouped = {"ascending": [], "descending": []}
    
    for item in items:
        # Get orbit state from item properties
        orbit_state = item.properties.get("sat:orbit_state", "").lower()
        if orbit_state in grouped:
            grouped[orbit_state].append(item)
    
    return grouped


def dedupe_items_by_date(items: List, stac_provider: STACProvider, prism_bbox: Tuple[float, float, float, float], 
                         max_missing_pct: float, logger: logging.Logger) -> List[Tuple]:
    """De-duplicates items within a single orbit group by date, keeping the one with lowest missing percentage.
    
    For same-date items (from the same datatake), keeps only the one with the lowest missing percentage.
    Also filters out items exceeding max_missing_percentage.
    
    Parameters
    ----------
    items : List
        List of STAC items from a single orbit group.
    stac_provider : STACProvider
        STAC provider object.
    prism_bbox : Tuple[float, float, float, float]
        Bounding box in PRISM CRS.
    max_missing_pct : float
        Maximum allowed missing percentage (0-100).
    logger : logging.Logger
        Logger instance.

    Returns
    -------
    List[Tuple]
        List of (item, missing_pct, item_crs) tuples sorted by date (ascending).
    """
    if not items:
        return []
    
    # Group items by date
    items_by_date = {}
    for item in items:
        date_str = item.datetime.strftime('%Y-%m-%d')
        if date_str not in items_by_date:
            items_by_date[date_str] = []
        items_by_date[date_str].append(item)
    
    # For each date, calculate missing percentages and keep best one
    result = []
    for date_str, date_items in items_by_date.items():
        best_item = None
        best_missing_pct = float('inf')
        best_crs = None
        
        for item in date_items:
            try:
                item_crs = get_item_crs(item)
                if item_crs is None:
                    logger.debug(f"Skipping item {item.id}: no CRS found")
                    continue
                    
                # Check for required assets
                vv_name = stac_provider.get_asset_names("s1")["vv"]
                vh_name = stac_provider.get_asset_names("s1")["vh"]
                if vv_name not in item.assets or vh_name not in item.assets:
                    logger.debug(f"Skipping item {item.id}: missing vv or vh asset")
                    continue
                
                missing_pct = sar_missing_percentage(stac_provider, item, item_crs, prism_bbox)
                
                if missing_pct < best_missing_pct:
                    best_missing_pct = missing_pct
                    best_item = item
                    best_crs = item_crs
                    
            except Exception as e:
                logger.debug(f"Error processing item {item.id}: {e}")
                continue
        
        # Add best item if it passes the missing threshold
        if best_item is not None and best_missing_pct <= max_missing_pct:
            result.append((best_item, best_missing_pct, best_crs))
    
    # Sort by date ascending
    result.sort(key=lambda x: x[0].datetime)
    return result


def process_interval(stac_provider: STACProvider, items_with_info: List[Tuple], dir_path: Path, eid : str,
                     prism_bbox: Tuple[float, float, float, float], orbit_direction: str, crs : Optional[str],
                     logger: logging.Logger) -> Optional[Dict]:
    """Downloads and stacks N SAR acquisitions for a time interval.
    
    Uses the CRS of the first acquisition for all crops to ensure consistent shape.
    Stacks VV and VH bands in temporal order (earliest to latest).
    
    Parameters
    ----------
    stac_provider : STACProvider
        STAC provider object.
    items_with_info : List[Tuple]
        List of (item, missing_pct, item_crs) tuples sorted by date.
    dir_path : Path
        Directory path for saving files.
    eid : str
        Event/cell ID.
    prism_bbox : Tuple[float, float, float, float]
        Bounding box in PRISM CRS.
    orbit_direction : str
        Orbit direction ('ascending' or 'descending').
    crs : Optional[str]
        CRS to use for cropping all images in stac. If none is provided, the CRS of the first item will be used.
        This CRS determines the shape of the resulting rasters.
    logger : logging.Logger
        Logger instance.

    Returns
    -------
    Optional[Dict]
        Metadata dict for this stack, or None if processing failed.
    """
    if not items_with_info:
        return None
    
    # Use CRS of first item as reference
    ref_crs = items_with_info[0][2] if crs is None else crs
    
    # Convert PRISM bbox to reference CRS
    minx, miny, maxx, maxy = prism_bbox
    conversion = transform(PRISM_CRS, ref_crs, (minx, maxx), (miny, maxy))
    crop_bbox = (conversion[0][0], conversion[1][0], conversion[0][1], conversion[1][1])
    
    vv_stack = []
    vh_stack = []
    acquisition_dates = []
    product_ids = []
    ref_shape = None
    ref_transform = None
    
    vv_name = stac_provider.get_asset_names("s1")["vv"]
    vh_name = stac_provider.get_asset_names("s1")["vh"]
    
    for item, missing_pct, item_crs in items_with_info:
        try:
            vv_href = stac_provider.sign_asset_href(item.assets[vv_name].href)
            vh_href = stac_provider.sign_asset_href(item.assets[vh_name].href)
            
            # Crop VV and VH using reference CRS
            vv_image, vv_transform = crop_to_bounds(vv_href, crop_bbox, ref_crs, nodata=0, resampling=Resampling.bilinear)
            vh_image, _ = crop_to_bounds(vh_href, crop_bbox, ref_crs, nodata=0, resampling=Resampling.bilinear)
            
            # Set reference shape from first acquisition
            if ref_shape is None:
                ref_shape = vv_image.shape
                ref_transform = vv_transform
            else:
                # Verify shape consistency
                if vv_image.shape != ref_shape or vh_image.shape != ref_shape:
                    raise ValueError(f"Shape mismatch for item {item.id}: expected {ref_shape}, got VV={vv_image.shape}, VH={vh_image.shape}")
            
            # Convert to dB scale
            vv_db = db_scale(vv_image[0], no_data=-9999)
            vh_db = db_scale(vh_image[0], no_data=-9999)
            
            vv_stack.append(vv_db)
            vh_stack.append(vh_db)
            acquisition_dates.append(item.datetime.strftime('%Y-%m-%d'))
            product_ids.append(item.id)
            
        except Exception as e:
            logger.error(f"Error processing item {item.id}: {e}")
            return None
    
    if not vv_stack:
        return None
    
    # Stack arrays
    vv_stacked = np.stack(vv_stack)
    vh_stacked = np.stack(vh_stack)
    
    # Create output filenames
    start_date = acquisition_dates[0].replace('-', '')
    end_date = acquisition_dates[-1].replace('-', '')
    stack_id = f"multi_{start_date}-{end_date}_{eid}"
    
    vv_path = dir_path / f"{stack_id}_vv.tif"
    vh_path = dir_path / f"{stack_id}_vh.tif"
    
    # Save VV stack
    with rasterio.open(
        vv_path, 'w', driver='GTiff',
        height=vv_stacked.shape[1], width=vv_stacked.shape[2],
        count=vv_stacked.shape[0], dtype=vv_stacked.dtype,
        crs=ref_crs, transform=ref_transform, nodata=-9999
    ) as dst:
        dst.write(vv_stacked)
    
    # Save VH stack
    with rasterio.open(
        vh_path, 'w', driver='GTiff',
        height=vh_stacked.shape[1], width=vh_stacked.shape[2],
        count=vh_stacked.shape[0], dtype=vh_stacked.dtype,
        crs=ref_crs, transform=ref_transform, nodata=-9999
    ) as dst:
        dst.write(vh_stacked)
    
    logger.info(f"Saved {stack_id}_vv.tif and {stack_id}_vh.tif ({orbit_direction} orbit)")
    
    # Return metadata for this stack
    return {
        "stack_id": stack_id,
        "start_date": acquisition_dates[0],
        "end_date": acquisition_dates[-1],
        "orbit_direction": orbit_direction,
        "acquisition_dates": acquisition_dates,
        "products": product_ids,
        "crs": ref_crs
    }


def cell_sample(s1_stac_provider: STACProvider, y: int, x: int, manual_crs: Optional[str], 
                eid: str, dir_path: Path, prism_data: PRISMData, 
                cfg: DictConfig, logger: logging.Logger) -> bool:
    """Downloads M multitemporal SAR stacks for a given PRISM cell.
    
    For each time interval, groups items by orbit, de-duplicates by date within each orbit,
    and randomly selects an orbit that has N valid acquisitions.
    
    Parameters
    ----------
    s1_stac_provider : STACProvider
        STAC provider for S1 data.
    y : int
        PRISM cell y coordinate.
    x : int
        PRISM cell x coordinate.
    manual_crs : Optional[str]
        Manual CRS override, or None for automatic selection.
    eid : str
        Event/cell ID.
    dir_path : Path
        Directory path for saving files.
    prism_data : PRISMData
        PRISM data object.
    cfg : DictConfig
        Configuration object.
    logger : logging.Logger
        Logger instance.

    Returns
    -------
    bool
        True if at least one stack was successfully downloaded, False otherwise.
    """
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

    logger.info(f"Processing cell {eid}...")
    
    # Get bounding box and convert to search CRS
    prism_bbox = prism_data.get_bounding_box(x, y)
    minx, miny, maxx, maxy = prism_bbox
    conversion = transform(PRISM_CRS, SEARCH_CRS, (minx, maxx), (miny, maxy))
    search_bbox = (conversion[0][0], conversion[1][0], conversion[0][1], conversion[1][1])
    
    # Initialize random generator with seed
    rng = random.Random(cfg.sampling.seed + y * 10000 + x)  # Unique seed per cell
    
    # Get precipitation data for this cell
    prism_data_for_cell = prism_data.precip_data[:, y, x]
    
    # Find valid intervals based on precipitation threshold
    if cfg.sampling.threshold is not None:
        all_intervals = list(valid_interval_start_idx(
            prism_data_for_cell,
            cfg.sampling.time_interval,
            cfg.sampling.threshold,
            x=cfg.sampling.num_days
        ))
    else:
        all_intervals = list(range(0, prism_data_for_cell.shape[0] - cfg.sampling.time_interval))
    
    if len(all_intervals) == 0:
        logger.error(f"No valid intervals found after filtering by threshold={cfg.sampling.threshold}")
        return False
    
    logger.info(f"Found {len(all_intervals)} valid intervals for cell {eid}")
    rng.shuffle(all_intervals)
    
    # Create output directory
    dir_path.mkdir(parents=True, exist_ok=True)
    
    # Tracking variables
    attempts = 0
    stacks_sampled = 0
    target_stacks = cfg.sampling.num_time_intervals
    max_tries = cfg.sampling.max_tries
    stacks_metadata = {}
    cell_crs = manual_crs # can be None
    
    for interval_idx in all_intervals:
        if stacks_sampled >= target_stacks:
            logger.info(f"Reached target of {target_stacks} stacks for cell {eid}")
            break
        
        if attempts >= max_tries:
            logger.warning(f"Reached max attempts ({max_tries}) for cell {eid}. Stopping with {stacks_sampled}/{target_stacks} stacks.")
            break
        
        # Calculate interval dates
        interval_start_dt = prism_data.get_event_datetime(interval_idx)
        interval_end_dt = interval_start_dt + timedelta(days=cfg.sampling.time_interval)
        time_of_interest = get_date_interval(interval_start_dt, interval_end_dt)
        
        logger.info(f"Stack {stacks_sampled + 1}/{target_stacks}: Searching {time_of_interest}...")
        
        # Search for S1 items
        try:
            items = s1_stac_provider.search_s1(search_bbox, time_of_interest, query={"sar:instrument_mode": {"eq": "IW"}})
        except Exception as e:
            logger.error(f"STAC search failed: {e}")
            attempts += 1
            continue
        
        if len(items) < cfg.sampling.acquisitions:
            logger.debug(f"Only {len(items)} items found, need {cfg.sampling.acquisitions}. Skipping interval.")
            attempts += 1
            continue
        
        # Group by orbit
        grouped = group_items_by_orbit(items)
        logger.info(f"Found {len(items)} S1 items ({len(grouped['ascending'])} ascending, {len(grouped['descending'])} descending)")
        
        # De-duplicate and filter each orbit group
        valid_orbits = {}
        for orbit_dir, orbit_items in grouped.items():
            if not orbit_items:
                continue
            
            deduped = dedupe_items_by_date(
                orbit_items, s1_stac_provider, prism_bbox,
                cfg.sampling.max_missing_percentage, logger
            )
            
            if len(deduped) >= cfg.sampling.acquisitions:
                valid_orbits[orbit_dir] = deduped[:cfg.sampling.acquisitions]  # Take first N
        
        if not valid_orbits:
            logger.debug(f"No orbit has {cfg.sampling.acquisitions} valid acquisitions. Skipping interval.")
            attempts += 1
            continue
        
        # Log dedup results
        for orbit_dir, items_info in valid_orbits.items():
            logger.info(f"After dedup/filter: {len(items_info)} {orbit_dir} acquisitions (valid)")
        
        # Random orbit selection
        available_orbits = list(valid_orbits.keys())
        selected_orbit = rng.choice(available_orbits)
        selected_items = valid_orbits[selected_orbit]
        
        logger.info(f"Selected {selected_orbit.upper()} orbit with {len(selected_items)} acquisitions")
        
        # Process the interval
        try:
            stack_metadata = process_interval(
                s1_stac_provider, selected_items, dir_path, eid,
                prism_bbox, selected_orbit, cell_crs, logger
            )
            
            if stack_metadata is None:
                logger.error(f"Failed to process interval. Skipping.")
                attempts += 1
                continue
            
            # Record metadata
            stack_id = stack_metadata["stack_id"]
            stacks_metadata[stack_id] = stack_metadata
            
            # Use CRS from first successful stack
            if cell_crs is None:
                cell_crs = stack_metadata["crs"]
            
            stacks_sampled += 1
            logger.info(f"Stack {stacks_sampled}/{target_stacks} complete ({selected_orbit} orbit)")
            
        except Exception as e:
            logger.error(f"Error processing interval: {e}")
            attempts += 1
            continue
    
    # Save metadata if any stacks were downloaded
    if stacks_sampled > 0:
        metadata = {
            "Download Date": CURRENT_DATE,
            "Cell ID": eid,
            "Cell Coordinates": {"y": y, "x": x},
            "Bounding Box": {
                "minx": prism_bbox[0],
                "miny": prism_bbox[1],
                "maxx": prism_bbox[2],
                "maxy": prism_bbox[3]
            },
            "CRS": cell_crs,
            "Source": cfg.sampling.source,
            "Sampling Parameters": {
                "acquisitions": cfg.sampling.acquisitions,
                "time_interval": cfg.sampling.time_interval,
                "num_time_intervals": cfg.sampling.num_time_intervals,
                "threshold": cfg.sampling.threshold,
                "num_days": cfg.sampling.num_days,
                "max_missing_percentage": cfg.sampling.max_missing_percentage
            },
            "Stacks": stacks_metadata
        }
        
        metadata_path = dir_path / "metadata.json"
        with open(metadata_path, 'w') as f:
            json.dump(metadata, f, indent=4)
        
        logger.info(f"Cell {eid} complete: {stacks_sampled}/{target_stacks} stacks downloaded")
        return True
    else:
        logger.warning(f"No stacks downloaded for cell {eid}")
        return False


def get_default_dir_name(time_interval: int, num_time_intervals: int, acquisitions: int, num_days: int) -> str:
    """Returns default directory name based on sampling parameters.
    
    Parameters
    ----------
    time_interval : int
        Time interval in days.
    num_time_intervals : int
        Number of time intervals to sample.
    acquisitions : int
        Number of acquisitions per interval.
    num_days : int
        Number of days to avoid after precipitation.

    Returns
    -------
    str
        Default directory name.
    """
    return f's1_multi_{time_interval}_{num_time_intervals}_{acquisitions}_{num_days}/'


@hydra.main(version_base=None, config_path="pkg://configs", config_name="config.yaml")
def main(cfg: DictConfig) -> None:
    """Runs multitemporal SAR data collection on a specified set of PRISM cells.
    Should specify to script a number of acquisitions to acquire per PRISM cell
    (the size of the multitemporal stack), the max size of the time interval
    allowed between first and last acquisition in days, and number of intervals.
    For each PRISM cell, the script randomly searches time intervals of the
    specified size within the limits of the PRISM dates search space while
    avoiding extreme precipitation dates (as defined by cfg.sampling.threshold)
    to maintain temporal consistency.

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
        Path to text file containing cell coordinates in lines with format: y, x[, crs]
        where crs is the EPSG code of the CRS. If not provided, the CRS will be
        automatically determined based on the first valid acquisition in the interval.
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
        Number of days after high precipitation date to avoid.
    max_missing_percentage : float
        Maximum percentage of missing values allowed in a SAR temporal slice.
    max_tries : int
        Max failed attempts to sample time intervals for each cell before giving up.
    max_runtime : str
        Maximum runtime in the format of HH:MM:SS
    source : str
        Source for S1 data. ['mpc', 'aws', 'cdse']
        
    Returns
    -------
    int
    """
    # Setup directory
    if cfg.sampling.dir_path is None:
        cfg.sampling.dir_path = str(Path(cfg.paths.imagery_dir) / get_default_dir_name(
            cfg.sampling.time_interval, cfg.sampling.num_time_intervals, 
            cfg.sampling.acquisitions, cfg.sampling.num_days
        ))
        Path(cfg.sampling.dir_path).mkdir(parents=True, exist_ok=True)
    else:
        try:
            Path(cfg.sampling.dir_path).mkdir(parents=True, exist_ok=True)
        except (OSError, ValueError) as e:
            print(f"Invalid directory path '{cfg.sampling.dir_path}'. Using default.", file=sys.stderr)
            cfg.sampling.dir_path = str(Path(cfg.paths.imagery_dir) / get_default_dir_name(
                cfg.sampling.time_interval, cfg.sampling.num_time_intervals, 
                cfg.sampling.acquisitions, cfg.sampling.num_days
            ))
            Path(cfg.sampling.dir_path).mkdir(parents=True, exist_ok=True)
        
        # Ensure trailing slash
        cfg.sampling.dir_path = os.path.join(cfg.sampling.dir_path, '')

    # Setup loggers
    rootLogger = setup_logging(cfg.sampling.dir_path, logger_name='main', log_level=logging.DEBUG, mode='w', include_console=False)
    logger = setup_logging(cfg.sampling.dir_path, logger_name='events', log_level=logging.DEBUG, mode='a', include_console=False)

    # Log sampling parameters
    rootLogger.info(
        "S1 multitemporal SAR sampling parameters used:\n"
        f"  Save directory: {cfg.sampling.dir_path}\n"
        f"  Cells file: {cfg.sampling.cells_file}\n"
        f"  Random seed: {cfg.sampling.seed}\n"
        f"  Number of acquisitions: {cfg.sampling.acquisitions}\n"
        f"  Time interval: {cfg.sampling.time_interval}\n"
        f"  Number of time intervals: {cfg.sampling.num_time_intervals}\n"
        f"  Precipitation threshold: {cfg.sampling.threshold}\n"
        f"  Days to avoid after precip: {cfg.sampling.num_days}\n"
        f"  Max missing percentage: {cfg.sampling.max_missing_percentage}\n"
        f"  Max tries: {cfg.sampling.max_tries}\n"
        f"  Max runtime: {getattr(cfg.sampling, 'max_runtime', 'Unlimited')}\n"
        f"  Source: {cfg.sampling.source}\n"
    )
    
    # Runtime tracking
    max_runtime_seconds = (
        walltime_seconds(cfg.sampling.max_runtime)
        if hasattr(cfg.sampling, 'max_runtime') and cfg.sampling.max_runtime is not None
        else float('inf')
    )
    start_time = time.time()

    # Load history
    history_path = Path(cfg.sampling.dir_path) / 'history.pickle'
    history = get_history(history_path)

    # Load PRISM data
    prism_data = PRISMData.from_file(cfg.paths.prism_data)

    # Load cells
    if not Path(cfg.sampling.cells_file).exists():
        raise ValueError(f"Cells file {cfg.sampling.cells_file} does not exist")
    
    cells = read_cell_coords(cfg.sampling.cells_file)
    
    if len(cells) == 0:
        raise ValueError(f"No cells found in file {cfg.sampling.cells_file}")

    rootLogger.info(f"Found {len(cells)} cells in {cfg.sampling.cells_file}")
    
    # Filter cells already in history
    filtered_cells = [cell for cell in cells if (cell[0], cell[1]) not in history]
    rootLogger.info(f"Filtered to {len(filtered_cells)} cells not in history")

    # Initialize STAC provider
    s1_stac_provider = get_stac_provider(
        cfg.sampling.source.lower(),
        mpc_api_key=getattr(cfg, "mpc_api_key", None),
        aws_access_key_id=getattr(cfg, "aws_access_key_id", None),
        aws_secret_access_key=getattr(cfg, "aws_secret_access_key", None),
        logger=logger
    )

    logger.info("Starting multitemporal SAR data collection")
    
    # Process cells
    sample_dir = Path(cfg.sampling.dir_path)
    cells_processed = 0
    cells_successful = 0
    
    try:
        for y, x, manual_crs in filtered_cells:
            # Check runtime
            if time.time() - start_time > max_runtime_seconds:
                rootLogger.info(f"Maximum runtime reached. Stopping after {cells_processed} cells.")
                break
            
            eid = f'{y}_{x}'
            cell_dir = sample_dir / eid
            
            logger.info(f"Processing cell {eid} (y={y}, x={x}, crs={manual_crs})...")
            
            try:
                success = cell_sample(
                    s1_stac_provider, y, x, manual_crs, eid, cell_dir,
                    prism_data, cfg, logger
                )
                
                if success:
                    cells_successful += 1
                
                # Mark cell as processed in history
                history.add((y, x))
                cells_processed += 1
                
            except (rasterio.errors.RasterioIOError, pystac_client.exceptions.APIError) as e:
                logger.error(f"Connection error for cell {eid}: {e}")
                continue
            except Exception as e:
                logger.exception(f"Unexpected error for cell {eid}: {e}")
                history.add((y, x))  # Mark as done to avoid re-trying
                cells_processed += 1
                continue
                
    finally:
        # Save history
        with open(history_path, 'wb') as f:
            pickle.dump(history, f)
        
        rootLogger.info(f"Saved history with {len(history)} cells")
        rootLogger.info(f"Completed: {cells_successful}/{cells_processed} cells successful")


if __name__ == '__main__':
    main()
