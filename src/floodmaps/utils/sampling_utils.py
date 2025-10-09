from dataclasses import dataclass
import logging
from datetime import datetime, timedelta, timezone
from collections import defaultdict
import os
import re
from pathlib import Path
from typing import List, Tuple, Dict, Iterator
from cftime import num2date, date2num
import cftime
from netCDF4 import Dataset
import numpy as np
import numpy.ma as ma
import matplotlib.pyplot as plt
import matplotlib.colors as mcolors
import geopandas as gpd
from matplotlib.colors import to_rgb
from geopy.geocoders import Nominatim
import pickle
from rasterio.warp import Resampling
from rasterio.crs import CRS
from rasterio.transform import Affine
from rasterio import windows
from rasterio.vrt import WarpedVRT
import rasterio
from omegaconf import DictConfig
import zipfile

PRISM_CRS = "EPSG:4269"
SEARCH_CRS = "EPSG:4326"

# S2 processing baseline offset correction
# SEE: https://sentiwiki.copernicus.eu/web/s2-products
BOA_ADD_OFFSET = -1000.0
PROCESSING_BASELINE_UTC = datetime(2022, 1, 25, tzinfo=timezone.utc)

# NLCD color mapping dictionary
NLCD_COLORS = {
    11: '#486da2',    # Open Water
    12: '#e7effc',    # Perennial Ice/Snow
    21: '#e1cdce',    # Developed, Open Space
    22: '#dc9881',    # Developed, Low Intensity 
    23: '#f10100',    # Developed, Medium Intensity
    24: '#ab0101',    # Developed, High Intensity
    31: '#b3afa4',    # Barren Land
    41: '#6ca966',    # Deciduous Forest
    42: '#1d6533',    # Evergreen Forest
    43: '#bdcc93',    # Mixed Forest
    51: '#b19943',    # Dwarf Scrub
    52: '#d1bb82',    # Shrub/Scrub
    71: '#edeccd',    # Grassland/Herbaceous
    72: '#d0d181',    # Sedge/Herbaceous
    73: '#a4cc51',    # Lichens
    74: '#82ba9d',    # Moss
    81: '#ddd83d',    # Pasture/Hay
    82: '#ae7229',    # Cultivated Crops
    90: '#bbd7ed',    # Woody Wetlands
    95: '#71a4c1',     # Emergent Herbaceous Wetlands
    250: '#000000'     # Missing
}
NLCD_CODE_TO_RGB = {
    code: tuple(int(255 * c) for c in to_rgb(hex_color))
    for code, hex_color in NLCD_COLORS.items()
}

@dataclass
class PRISMData:
    """Container for PRISM netCDF data."""
    geotransform: np.ndarray
    time_info: Tuple[str, str]
    precip_data: np.ndarray

    @classmethod
    def from_file(cls, prism_file: str) -> 'PRISMData':
        """Load PRISM data from file."""
        with Dataset(prism_file, "r", format="NETCDF4") as nc:
            geotransform = nc["geotransform"][:]
            time_info = (nc["time"].units, nc["time"].calendar)
            precip_data = nc["precip"][:]
        
        return cls(geotransform, time_info, precip_data)
    
    def get_reference_date(self) -> datetime:
        """Get PRISM reference date (day 0)."""
        return datetime.fromisoformat(num2date(0, units=self.time_info[0], calendar=self.time_info[1]).isoformat())
    
    def get_event_datetime(self, time_index: int) -> datetime:
        """Get datetime.datetime object for specific time index in PRISM data."""
        return datetime.fromisoformat(num2date(time_index, units=self.time_info[0], calendar=self.time_info[1]).isoformat())

    def get_event_cftime(self, time_index: int) -> cftime.datetime:
        """Get cftime.datetime object for specific time index in PRISM data."""
        return num2date(time_index, units=self.time_info[0], calendar=self.time_info[1])

    def get_precip(self, time_index: int, y: int, x: int) -> float:
        """Get precipitation value for specific time index, y, x in PRISM data."""
        return self.precip_data[time_index, y, x]
    
    def get_precip_shape(self) -> Tuple[int, int, int]:
        """Get shape of precipitation data."""
        return self.precip_data.shape
    
    def get_bounding_box(self, x: int, y: int) -> Tuple[float, float, float, float]:
        """Get bounding box for grid coordinates (CRS=EPSG:4269)."""
        upper_left_x, x_size, x_rotation, upper_left_y, y_rotation, y_size = self.geotransform
        
        # convert latitude and longitude to (minx, miny, maxx, maxy)
        minx = x * x_size + upper_left_x
        miny = (y + 1) * y_size + upper_left_y
        maxx = (x + 1) * x_size + upper_left_x
        maxy = y * y_size + upper_left_y
        
        return (minx, miny, maxx, maxy)

    def get_masked_precip(self, mask: np.ndarray) -> np.ndarray:
        """Get masked precipitation data. Sets values outside mask to nan."""
        return np.where(mask, self.precip_data, np.nan)
    
    @property
    def shape(self) -> Tuple[int, int, int]:
        """Get shape of precipitation data."""
        return self.precip_data.shape

class SamplingPathManager:
    """Configuration class for managing data file paths and settings for
    sampling script."""
    
    def __init__(self, cfg: DictConfig):
        self.cfg = cfg
        self.validate_paths()
    
    def validate_paths(self):
        """Validate that all required data paths exist."""
        required_paths = [
            self.cfg.paths.get('prism_data'),
            self.cfg.paths.get('ceser_boundary'),
            self.cfg.paths.get('prism_meshgrid'),
            self.cfg.paths.get('nhd_wbd'),
            self.cfg.paths.get('elevation_dir'),
            self.cfg.paths.get('nlcd_dir'),
            self.cfg.paths.get('roads_dir')
        ]
        
        missing = []
        for path in required_paths:
            if path is None:
                raise KeyError(f"Required path not found in configuration.")
            if path and not Path(path).exists():
                missing.append(path)
        
        if missing:
            raise FileNotFoundError(
                f"Missing required data files/directories:\n" + "\n".join(missing)
            )
    
    def get_path(self, key: str, required: bool = True) -> str:
        """
        Get a specific path from configuration.
        
        Parameters
        ----------
        key : str
            Path key from config file
        required : bool
            Whether the path is required to exist
            
        Returns
        -------
        str
            Path value
        """
        path = self.cfg.paths.get(key)
        
        if path is None:
            if required:
                raise KeyError(f"Required path '{key}' not found in configuration")
            return ""
        
        if required and not Path(path).exists():
            raise FileNotFoundError(f"Required path does not exist: {path}")
        
        return path
    
    @property
    def prism_file(self) -> str:
        """Get PRISM data file path."""
        return self.get_path('prism_data')
    
    @property
    def ceser_boundary_file(self) -> str:
        """Get CESER boundary file path."""
        return self.get_path('ceser_boundary')
    
    @property
    def prism_meshgrid_file(self) -> str:
        """Get PRISM meshgrid file path."""
        return self.get_path('prism_meshgrid')
    
    @property
    def nhd_wbd_file(self) -> str:
        """Get NHD WBD file path."""
        return self.get_path('nhd_wbd')
    
    @property
    def elevation_directory(self) -> str:
        """Get elevation directory path."""
        return self.get_path('elevation_dir')
    
    @property
    def nlcd_directory(self) -> str:
        """Get NLCD directory path."""
        return self.get_path('nlcd_dir')
    
    @property
    def roads_directory(self) -> str:
        """Get roads directory path."""
        return self.get_path('roads_dir') 

class DateCRSOrganizer:
    """Organizes products by date first (in chronological order), then CRS (alphabetically).
    
    Note: since underlying object is a defaultdict, avoid indexing with non-existent keys.
    Also the data is indexed with timezone aware datetimes, so indexing the underlying
    data with naive datetimes will cause bugs (though the method calls are safe)."""
    def __init__(self):
        self.data = defaultdict(lambda: defaultdict(list))
        self.count = 0

    def is_empty(self):
        """Check if the organizer is empty."""
        return self.count == 0

    def _normalize_date(self, dt):
        """Convert datetime to date-only datetime (00:00:00) UTC"""
        if isinstance(dt, datetime):
            return datetime(dt.year, dt.month, dt.day, tzinfo=timezone.utc)
        else:
            raise ValueError("Date must be a datetime object")
    
    def add_item(self, item, date, crs):
        """Insert item into the organizer.
        
        Parameters
        ----------
            item: Item
                The item to insert.
            date: datetime
                The date of the item.
            crs: str
                The CRS of the item.
        """
        normalized_date = self._normalize_date(date)
        self.data[normalized_date][crs].append(item)
        self.count += 1
    
    def get_dates(self):
        """Get dates of all inserted items in chronological order."""
        return sorted(self.data.keys())

    def get_all_crs(self):
        """Get all unique CRS values across all dates, sorted alphabetically."""
        all_crs = set()
        for date_data in self.data.values():
            all_crs.update(date_data.keys())
        return sorted(all_crs)
    
    def get_crs_for_date(self, date):
        """Get all CRS values for a specific date, sorted."""
        date = self._normalize_date(date)
        if date not in self.data:
            raise KeyError(f"No items for date: {date}")
        return sorted(self.data[date].keys())
    
    def get_items(self, date, crs):
        """Get items for specific date and CRS."""
        date = self._normalize_date(date)
        if date not in self.data or crs not in self.data[date]:
            raise KeyError(f"No items for date: {date} and CRS: {crs}")
        return self.data[date][crs]

    def get_primary_item_for_date(self, date, preferred_crs=None):
        """
        Get the first item for a date based on CRS preference.
        
        Parameters
        ----------
            date: The date of item.
            preferred_crs: The preferred CRS of item.
        
        Returns
        -------
            The first item from preferred CRS if available, otherwise first item from 
            first alphabetical CRS for that date. Returns None if no items for date.
        """
        date = self._normalize_date(date)
        if date not in self.data or not self.data[date]:
            raise KeyError(f"No items for date: {date}")
        
        if preferred_crs is not None and preferred_crs in self.data[date]:
            return self.data[date][preferred_crs][0]
        else:
            first_crs = self.get_crs_for_date(date)[0]
            return self.data[date][first_crs][0]

    def get_all_primary_items(self, preferred_crs=None):
        """
        Get items for each date selected based on CRS preference.
        
        Parameters
        ----------
            preferred_crs: The preferred CRS of item.
        
        Returns
        -------
            The list of items selected for each date from preferred CRS.
        """
        lst = []
        for date in self.get_dates():
            lst.append(self.get_primary_item_for_date(date, preferred_crs))
        return lst
    
    def ordered_items(self):
        """Iterate over all item lists grouped by date and CRS."""
        for date in sorted(self.data.keys()):
            for crs in sorted(self.data[date].keys()):
                yield date, crs, self.data[date][crs]

class NoElevationError(Exception):
    """Exception raised when elevation data is not found."""
    pass

def get_default_dir_name(threshold: int, days_before: int, days_after: int, maxcoverpercentage: int, region: str = None) -> str:
    """Default directory name for sampling scripts."""
    if region is None:
        return f'samples_{threshold}_{days_before}_{days_after}_{maxcoverpercentage}/'
    else:
        return f'samples_{region}_{threshold}_{days_before}_{days_after}_{maxcoverpercentage}/'

def get_date_interval(event_date: str, days_before: int, days_after: int) -> str:
    """Returns a date interval made of a tuple of start and end date strings given event date string and 
    number of days extending before and after the event.
    
    Parameters
    ----------
    event_date : str
        Date string formatted as YYYYMMDD or YYYY-MM-DD.
    days_before : int
    days_after : int

    Returns
    -------
    str
        Interval with start and end date strings formatted as YYYY-MM-DD/YYYY-MM-DD.
    """
    # convert event date to datetime object
    dt = parse_date_string(event_date)
    if days_before + days_after == 0:
        return f"{dt.strftime('%Y-%m-%d')}"
    delt1 = timedelta(days = days_before)
    delt2 = timedelta(days = days_after)
    start = dt - delt1
    end = dt + delt2
    return start.strftime("%Y-%m-%d") + '/' + end.strftime("%Y-%m-%d")

def has_date_after_PRISM(dates: List[datetime], event_date_str: datetime) -> bool:
    """Returns whether there exists dates during or after a PRISM event date.
    Dates must be timezone aware (UTC)!

    Parameters
    ----------
    dates : list[datetime]
        List of datetime objects.
    event_date_str : str
        PRISM event date formatted as YYYYMMDD.

    Returns
    -------
    bool
        True if query has dates that lie on or after PRISM event date or False if not.
    """
    dt = datetime.strptime(event_date_str, "%Y%m%d")
    dt_1 = dt - timedelta(days=1)
    
    # PRISM DAY BEGINS AT 1200 UTC-0 THE DAY BEFORE
    prism_dt = datetime(dt_1.year, dt_1.month, dt_1.day, hour=12, tzinfo=timezone.utc)

    # Compare dates
    for date in dates:
        if date >= prism_dt:
            return True
    return False

def db_scale(x: np.ndarray) -> np.ndarray:
    """Scales SAR data to dB scale. Sets invalid values to -9999.
    
    Parameters
    ----------
    x : np.ndarray
        Array of SAR data.
    """
    nonzero_mask = x > 0
    missing_mask = x <= 0
    x[nonzero_mask] = 10 * np.log10(x[nonzero_mask])
    x[missing_mask] = -9999
    return x

def setup_logging(
    output_dir: str,
    logger_name: str = 'main',
    log_level: int = logging.DEBUG,
    mode: str = 'w',
    include_console: bool = True
) -> logging.Logger:
    """
    Setup logging configuration for a script.
    
    Parameters
    ----------
    output_dir : str
        Directory where log files will be saved
    logger_name : str
        Name of the logger (default: 'main')
    log_level : int
        Logging level (default: logging.DEBUG)
    mode : str
        File mode ('w' for overwrite, 'a' for append)
    include_console : bool
        Whether to also log to console
        
    Returns
    -------
    logging.Logger
        Configured logger instance
    """
    start_time = datetime.now().strftime("%Y-%m-%d_%H-%M")
    
    # Create output directory if it doesn't exist
    Path(output_dir).mkdir(parents=True, exist_ok=True)
    
    # Create logger
    logger = logging.getLogger(logger_name)
    logger.setLevel(log_level)
    
    # Clear any existing handlers to avoid duplicates
    logger.handlers.clear()
    
    # Create formatter
    formatter = logging.Formatter('%(asctime)s - %(name)s - %(levelname)s - %(message)s')
    
    # File handler
    log_file = Path(output_dir) / f'{logger_name}_{start_time}.log'
    fh = logging.FileHandler(log_file, mode=mode)
    fh.setLevel(log_level)
    fh.setFormatter(formatter)
    logger.addHandler(fh)
    
    # Console handler (optional)
    if include_console:
        ch = logging.StreamHandler()
        ch.setLevel(log_level)
        ch.setFormatter(formatter)
        logger.addHandler(ch)
    
    return logger

def get_history(file_path: Path) -> set:
    """Get the history of events from a file. History is a set of identifiers.
    
    file_path : Path
        Path to history file.

    Returns
    -------
    set
        History of events.
    """
    if file_path.is_file():
        with open(file_path, 'rb') as f:
            history = pickle.load(f)
        # check if it is a set
        if not isinstance(history, set):
            raise ValueError(f"History file {file_path} is not a set")
        
        return history
    else:
        return set()

def read_PRISM(prism_file: str) -> PRISMData:
    """Reads the PRISM netCDF file and return the encoded data."""
    return PRISMData.from_file(prism_file)

def parse_date_string(date_string: str) -> datetime:
    """Parse date string in YYYYMMDD or YYYY-MM-DD format."""
    formats = ["%Y-%m-%d", "%Y%m%d"]
    
    for fmt in formats:
        try:
            return datetime.strptime(date_string, fmt)
        except ValueError:
            continue

    raise ValueError(f"Date string '{date_string}' doesn't match expected formats YYYYMMDD or YYYY-MM-DD")

def read_manual_indices(manual_file: str) -> List[Tuple[datetime, int, int, str]]:
    """
    Reads in manual event date string, y, x indices specifying PRISM events from a text file.
    The function ignores empty lines and lines starting with '#'.

    Valid to take YYYYMMDD or YYYY-MM-DD format for time.

    Optional 4th parameter can specify CRS (e.g., EPSG:32617). If not provided, 
    defaults to None for automatic CRS selection.

    Parameters
    ----------
    manual_file : str
        Path to text file containing manual event date string, y, x indices, optionally with CRS.

    Returns
    -------
    List[Tuple[datetime, int, int, str]]
        List of tuples containing datetime obj, y, x indices and optional CRS string.
    """
    def validate_epsg(code: str) -> bool:
        """Validate EPSG code."""
        if not re.fullmatch(r"EPSG:\d+", code):
            return False
        try:
            CRS.from_string(code)
            return True
        except Exception:
            return False

    manual_indices = []
    with open(manual_file, 'r') as f:
        for lineno, line in enumerate(f, 1):
            stripped = line.strip()
            if not stripped or stripped.startswith('#'):  # skip empty lines and comments
                continue
            parts = [p.strip() for p in stripped.split(',')]
            if len(parts) not in [3, 4]:
                raise ValueError(f"Line {lineno}: Expected 3 or 4 comma-separated values, got {parts}")

            # check if first part is in format YYYYMMDD or YYYY-MM-DD and convert to datetime obj
            if len(parts[0]) != 8 and len(parts[0]) != 10:
                raise ValueError(f"Line {lineno}: Date string '{parts[0]}' doesn't match expected formats")
            event_date = parse_date_string(parts[0]) # check if date string is valid

            # Extract CRS if provided (4th parameter)
            crs = None
            if len(parts) == 4:
                if validate_epsg(parts[3]):
                    crs = parts[3]
                else:
                    raise ValueError(f"Line {lineno}: Not a valid CRS: {parts[3]}")

            try:
                values = tuple(map(int, parts[1:3]))
            except ValueError:
                raise ValueError(f"Line {lineno}: Invalid integers: {parts[1:3]}")
            if any(v < 0 for v in values):
                raise ValueError(f"Line {lineno}: All y, x coord values must be non-negative ints: {values}")

            y, x = values
            manual_indices.append((event_date, y, x, crs))

    return manual_indices

def get_mask(cfg: DictConfig, shape: Tuple[int, int, int]) -> np.ndarray:
    """Get the mask for the region in the PRISM meshgrid.

    Mask will be 3D array matching PRISM data shape.

    TO DO: Will want to add other regions of interest in the future.
    
    Parameters
    ----------
    cfg: DictConfig
        Configuration object.
    shape : tuple
        Shape of the PRISM data.

    Returns
    -------
    numpy.ndarray
        Mask for PRISM data.
    """
    if cfg.sampling.region == 'ceser':
        mask = get_shape_mask(cfg.paths.ceser_boundary, cfg.paths.prism_meshgrid, shape)
    else:
        mask = None
    return mask

def get_shape_mask(shape_file: str, prism_meshgrid_file: str, shape: Tuple[int, int, int]) -> np.ndarray:
    """Get the mask for any shapefile on the PRISM meshgrid. 
    Requires a shapefile and the PRISM meshgrid.
    
    Parameters
    ----------
    shape_file : str
        Path to shape file.
    prism_meshgrid_file : str
        Path to PRISM meshgrid file.
    shape : tuple
        Shape of the PRISM data.

    Returns
    -------
    numpy.ndarray
        Mask for PRISM data.
    """
    shape_gdf = gpd.read_file(shape_file)
    meshgrid = gpd.read_file(prism_meshgrid_file)
    # Reproject shape to prism grid
    if shape_gdf.crs != meshgrid.crs:
        shape_reprojected = shape_gdf.to_crs(meshgrid.crs)
        ref_shape = shape_reprojected.geometry.iloc[0]
    else:
        ref_shape = shape_gdf.geometry.iloc[0]
    intersecting_shapes = meshgrid[meshgrid.geometry.intersects(ref_shape)]
    filtered_indices = list(zip(intersecting_shapes['row'], intersecting_shapes['col']))

    # filter our PRISM array by cells inside the shape
    mask_2d = np.zeros((shape[1], shape[2]), dtype=bool)
    for i, j in filtered_indices:
        mask_2d[i, j] = True

    # Broadcast the 2D mask to 3D to match precip_data shape
    mask_3d = np.broadcast_to(mask_2d, shape)
    return mask_3d

def get_manual_events(prism_data: PRISMData, history: set, manual_file: str, logger: logging.Logger = None) -> Tuple[int, Iterator[Tuple[str, float, Tuple[float, float, float, float], str, Tuple[int, int, int], str]]]:
    """Compile info for manually specified events via list of event date strings, prism cell y, x indices and optional CRS.
    
    Parameters
    ----------
    prism_data : PRISMData
        PRISM data object.
    history : set()
        Set that holds all tuples of indices (from PRISM ndarray) of previously processed events.
    manual_file : str
        Path to text file containing manually specified events in lines with format: time, y, x, [crs].
        (crs is optional, must be in format EPSG:32617)
    logger : logging.Logger, optional
        Logger to use for logging. If None, a default logger will be created.

    Returns
    -------
    int
        Number of manual events found.
    Iterator
        Iterator aggregates event data with each tuple containing the date, cumulative day precipitation in mm, latitude longitude bounding box values, unique event id, indices tuple, and optional CRS string.
    """
    if logger is None:
        logger = logging.getLogger(__name__)
        # just print to console for now
        if not logger.handlers:
            ch = logging.StreamHandler()
            ch.setLevel(logging.INFO)
            formatter = logging.Formatter('%(asctime)s - %(name)s - %(levelname)s - %(message)s')
            ch.setFormatter(formatter)
            logger.addHandler(ch)

    # read in and validate manual indices
    manual_indices = read_manual_indices(manual_file)

    # lists for aggregating event data
    event_dates = []
    event_precip = []
    bbox = []
    eid = []
    indices = []
    crs_list = []

    min_date = prism_data.get_reference_date()
    logger.debug(f"PRISM start date: {min_date.strftime('%Y-%m-%d')}")
    count = 0
    for event_date, y, x, crs in manual_indices:
        event_date_str = event_date.strftime("%Y%m%d")

        # we allow time to be negative here!
        time = int(date2num(event_date, units=prism_data.time_info[0], calendar=prism_data.time_info[1]))

        # for manual events we allow events before PRISM start date, will just
        # use a negative time index if before prism start date (relative to prism start date)
        if (time, y, x) in history:
            continue

        event_dates.append(event_date_str)
        precip = prism_data.get_precip(time, y, x) if time >= 0 else 0 # if time index is negative, set precip to 0
        event_precip.append(precip)
        bbox.append(prism_data.get_bounding_box(x, y))
        eid.append(f'{event_date_str}_{y}_{x}')
        indices.append((time, y, x)) # indices for efficient tracking of completion
        crs_list.append(crs)  # append CRS (may be None)
        count += 1

    return count, zip(event_dates, event_precip, bbox, eid, indices, crs_list)

def get_extreme_events(prism_data: PRISMData, history, threshold=300, mask=None, n=None, logger=None):
    """
    Queries contents of PRISM netCDF file given the return value of read_PRISM(). Filters for 
    precipitation events that meet minimum threshold of precipitation and returns an iterator.

    Parameters
    ----------
    prism_data : PRISMData
        PRISM data object.
    history : set()
        Set that holds all tuples of indices (from PRISM ndarray) of previously processed events.
    threshold : int, optional
        Minimum cumulative daily precipitation in mm for filtering PRISM tiles.
    mask : np.ndarray, optional
        Mask to apply to PRISM data.
    n : int, optional
        Only prepares first n extreme precipitation events that meet threshold criteria.
    logger : logging.Logger, optional
        Logger to use for logging. If None, a default logger will be created.

    Returns
    -------
    int
        Number of extreme precipitation events found.
    Iterator
        Iterator aggregates extreme event data with each tuple containing the date, cumulative day precipitation in mm, latitude longitude bounding box values and a unique event id.
        The crs is set to None and defaults to the first CRS in alphabetical order.
    """
    if logger is None:
        logger = logging.getLogger(__name__)
        # just print to console for now
        if not logger.handlers:
            ch = logging.StreamHandler()
            ch.setLevel(logging.INFO)
            formatter = logging.Formatter('%(asctime)s - %(name)s - %(levelname)s - %(message)s')
            ch.setFormatter(formatter)
            logger.addHandler(ch)

    # for specific regions, mask the AOI
    precip_data = prism_data.get_masked_precip(mask) if mask is not None else prism_data.precip_data

    # returns tuple of arrays of time, lat indices, lon indices for filtered target events
    events = np.where(precip_data > threshold)

    # lists for aggregating event data
    event_dates = []
    event_precip = []
    bbox = []
    eid = []
    indices = []
    crs_list = []

    min_date = prism_data.get_reference_date()
    logger.debug(f"PRISM start date: {min_date.strftime('%Y-%m-%d')}")
    count = 0
    filtered_count = 0
    for time, y, x in zip(events[0], events[1], events[2]):
        if n is not None and count >= n:
            break
            
        event_date = prism_data.get_event_cftime(time)
        event_date_str = event_date.strftime("%Y%m%d")

        # must not be earlier than s2 launch, or previously queried in pipeline
        if event_date < min_date:
            filtered_count += 1
            continue
        elif (time, y, x) in history:
            filtered_count += 1
            continue

        event_dates.append(event_date_str)
        event_precip.append(prism_data.get_precip(time, y, x))
        bbox.append(prism_data.get_bounding_box(x, y))
        eid.append(f'{event_date_str}_{y}_{x}')
        indices.append((int(time), int(y), int(x))) # indices for efficient tracking of completion
        crs_list.append(None)
        count += 1

    if filtered_count > 0:
        logger.info(f"Filtered out {filtered_count} events (before min_date or already in history)")

    return count, zip(event_dates, event_precip, bbox, eid, indices, crs_list)

def get_random_events(prism_data: PRISMData, history, mask=None, n=None, random_seed=None, logger=None):
    """
    Randomly samples PRISM cells across the US. Useful for increasing diversity of samples (and non flood tiles).
    
    Parameters
    ----------
    prism_data : PRISMData
        PRISM data object.
    history : set()
        Set that holds all tuples of indices (from PRISM ndarray) of previously processed events.
    mask : np.ndarray, optional
        Mask to apply to PRISM data (e.g., CESER region mask).
    n : int, optional
        Maximum number of random events to find.
    random_seed : int, optional
        Random seed for reproducible sampling.
    logger : logging.Logger, optional
        Logger to use for logging.
        
    Returns
    -------
    int
        Number of random events found.
    Iterator
        Iterator of random event data with each tuple containing date, precipitation value, 
        bounding box, event id, indices, and crs.
    """
    if logger is None:
        logger = logging.getLogger(__name__)
        if not logger.handlers:
            ch = logging.StreamHandler()
            ch.setLevel(logging.INFO)
            formatter = logging.Formatter('%(asctime)s - %(name)s - %(levelname)s - %(message)s')
            ch.setFormatter(formatter)
            logger.addHandler(ch)
    
    # Set random seed for reproducibility
    if random_seed is not None:
        np.random.seed(random_seed)
    
    # Get PRISM data dimensions
    time_dim, y_dim, x_dim = prism_data.get_precip_shape()
    
    # Create valid coordinate arrays
    if mask is not None:
        # Get valid coordinates from mask (only spatial dimensions)
        mask_2d = mask[0, :, :]  # Take first time slice since mask is 3D
        valid_coords = np.where(mask_2d)
        valid_y_coords = valid_coords[0]
        valid_x_coords = valid_coords[1]
        logger.info(f"Using mask: {len(valid_y_coords)} valid PRISM cells available")
    else:
        # Use all non nan spatial coordinates in PRISM
        valid_coords = np.where(~np.isnan(prism_data.precip_data[0]))
        valid_y_coords = valid_coords[0]
        valid_x_coords = valid_coords[1]
        logger.info(f"No mask applied: {len(valid_y_coords)} PRISM cells available")
    
    # Randomly shuffle the coordinates
    indices = np.arange(len(valid_y_coords))
    np.random.shuffle(indices)
    
    # Lists for aggregating event data
    event_dates = []
    event_precip = []
    bbox = []
    eid = []
    event_indices = []
    crs_list = []
    
    min_date = prism_data.get_reference_date()
    logger.debug(f"PRISM start date: {min_date.strftime('%Y-%m-%d')}")
    count = 0
    filtered_count = 0
    
    for idx in indices:
        if n is not None and count >= n:
            break
            
        y = valid_y_coords[idx]
        x = valid_x_coords[idx]
        
        # Random time within reasonable range
        time = np.random.randint(0, time_dim)
        event_date = prism_data.get_event_cftime(time)
        event_date_str = event_date.strftime("%Y%m%d")
        
        # Skip if already processed
        if event_date < min_date:
            filtered_count += 1
            continue
        elif (time, y, x) in history:
            filtered_count += 1
            continue
            
        event_dates.append(event_date_str)
        event_precip.append(prism_data.get_precip(time, y, x))
        bbox.append(prism_data.get_bounding_box(x, y))
        eid.append(f'{event_date_str}_{y}_{x}')
        event_indices.append((int(time), int(y), int(x)))
        crs_list.append(None)
        count += 1
    
    if filtered_count > 0:
        logger.info(f"Filtered out {filtered_count} events (before min_date or already in history)")
    
    return count, zip(event_dates, event_precip, bbox, eid, event_indices, crs_list)

def event_completed(dir_path: Path, regex_patterns: list[str], pattern_dict: dict[str, str], logger: logging.Logger = None) -> bool:
    """Returns whether or not event directory contains all generated rasters by checking files with regex patterns."""
    if logger is None:
        logger = logging.getLogger(__name__)
        # just print to console for now
        if not logger.handlers:
            ch = logging.StreamHandler()
            ch.setLevel(logging.INFO)
            formatter = logging.Formatter('%(asctime)s - %(name)s - %(levelname)s - %(message)s')
            ch.setFormatter(formatter)
            logger.addHandler(ch)
    
    logger.info('Confirming whether event has already been successfully processed before...')
    existing_files = [f.name for f in dir_path.iterdir() if f.is_file()]
    
    # Check if each file name in the list exists in the directory
    missing_files = []

    for pattern in regex_patterns:
        pattern_matched = False
        for file_name in existing_files:
            file_path = dir_path / file_name
            if re.match(pattern, file_name) and file_path.stat().st_size > 0:
                pattern_matched = True
                break
        if not pattern_matched:
            missing_files.append(pattern)

    if len(missing_files) == 0:
        logger.info("Event has been successfully processed before. All files found!")
        return True
    else:
        logger.info(f"Prior processed event is missing files: {', '.join([pattern_dict[pattern] for pattern in missing_files])}.")
        return False

def get_state(minx: float, miny: float, maxx: float, maxy: float) -> str:
    """Fetches the US state corresponding to the longitude and latitude coordinates.
    Will try different combinations of longitude, latitude if state not found immediately.

    Parameters
    ----------
    minx : float
    miny : float
    maxx : float
    maxy : float
    
    Returns
    -------
    str or None
        US State or None if not found
    """
    combinations = [(miny, minx), (miny, maxx), (maxy, minx), (maxy, maxx)]
    geolocator = Nominatim(user_agent="argonneflood")
    
    for coord in combinations:
        location = geolocator.reverse(coord, exactly_one=True)
        result = location.raw['address'].get('state')
        if result is not None:
            break

    return result

def colormap_to_rgb(arr: np.ndarray, cmap: str = 'viridis', r: Tuple[float, float] = None, no_data: int = None) -> np.ndarray:
    """
    Converts a 2d numpy array into an 3D RGB array based on colormap.
    
    Parameters
    ----------
    arr :
        Numpy array.
    cmap :
        Matplotlib colormap string.
    r : (float, float)
        Arguments to vmin, vmax for matplotlib.colors.Normalize.
    no_data :
        No data value.

    Returns
    -------
    Numpy array :
        Color mapped RGB array of original 2d array
    """
    m_arr = ma.masked_equal(arr, no_data)
    # Normalize the input array to be in the range [0, 1]
    if r is None:
        norm = mcolors.Normalize(vmin=np.min(m_arr), vmax=np.max(m_arr))
    else:
        norm = mcolors.Normalize(vmin=r[0], vmax=r[1])
        
    colormap = plt.get_cmap(cmap)
    colored_array = (colormap(norm(m_arr)) * 255).astype(np.uint8)

    # Create an RGB array with the same shape as the input array
    rgb_array = np.zeros((3, arr.shape[0], arr.shape[1]), dtype=np.uint8)

    # Copy the RGB values from the colored_array to each channel
    # Missing data will be all black (0, 0, 0)
    for i in range(3):
        rgb_array[i, :, :] = ma.filled(colored_array[:, :, i], 0)

    return rgb_array

def crop_to_bounds(item_href, bounds, dst_crs, nodata=0, resampling=Resampling.nearest):
    """Crops provided raster using a bounding box and corresponding box CRS a single item.
    If the item is in a different CRS than the bounding box, it will be reprojected using a Warped VRT.
    Replaces rasterio.merge.merge() for faster warping and cropping.
    
    Parameters
    ----------
    item_href : str
        Path to the item.
    bounds : tuple
        Bounding box to crop item.
    dst_crs : str
        Destination CRS of the bounding box and cropped image.
    nodata : int, optional
        No data value to fill pixels in bounding box that falls outside the raster
    resampling : Resampling, optional
        Resampling method.

    Returns
    -------
    out_image : np.ndarray
        Cropped image.
    out_transform : Affine
        Transform of the cropped image.
    """
    dst_w, dst_s, dst_e, dst_n = bounds

    dst_crs = CRS.from_string(dst_crs)
    with rasterio.open(item_href) as src:
        res = src.res
        crs = src.crs
        count = src.count
        out_transform = Affine.translation(dst_w, dst_n) * Affine.scale(res[0], -res[1])
        out_width = int(round((dst_e - dst_w) / res[0]))
        out_height = int(round((dst_n - dst_s) / res[1]))
        out_shape = (count, out_height, out_width)

        if crs is None:
            raise ValueError("Source CRS is missing from file.")

        # if the file is the same crs, no need to reproject
        if dst_crs == crs:
            src_window = windows.from_bounds(dst_w, dst_s, dst_e, dst_n, src.transform)

            src_window_rnd_shp = src_window.round_lengths()
            # need to double check the read args are correct
            out_image = src.read(
                out_shape=out_shape,
                window=src_window_rnd_shp,
                boundless=True, # in order to allow bounds slightly outside the capture
                masked=False, # do not want masked array
                indexes=None,
                resampling=resampling,
                fill_value=nodata # fill empty pixels with nodata
            )
        else:
            # need to reproject
            vrt_options = {
                'resampling': resampling,
                'crs': dst_crs,
                'transform': out_transform,
                'height': out_height,
                'width': out_width,
                'nodata': nodata
            }
            with WarpedVRT(src, **vrt_options) as vrt:
                out_image = vrt.read()

    return out_image, out_transform

def unzip_file(zip_path: Path, remove_zip: bool = True, extract_to: Path = None) -> None:
    """Unzip a file to the same directory and optionally remove the original zip.
    
    Parameters
    ----------
    zip_path : Path
        Path to the zip file to extract
    remove_zip : bool, default=True
        Whether to remove the zip file after successful extraction
    extract_to : Path, optional
        Path to extract the zip file to. If not provided, will extract to the same directory as the zip file.
    """
    extract_to = zip_path.parent if extract_to is None else extract_to
    try:
        with zipfile.ZipFile(zip_path, 'r') as zip_ref:
            zip_ref.extractall(extract_to)
        print(f"Successfully extracted {zip_path.name}")
        
        if remove_zip:
            zip_path.unlink()
            print(f"Removed original zip file: {zip_path.name}")
            
    except zipfile.BadZipFile:
        print(f"Warning: {zip_path} is not a valid zip file")
    except Exception as e:
        print(f"Error extracting {zip_path}: {e}")