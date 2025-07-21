import argparse
from netCDF4 import Dataset
from datetime import datetime, timedelta, date, timezone
from pathlib import Path
from contextlib import ExitStack
import sys
from cftime import num2date
import logging
import numpy as np
import numpy.ma as ma
import os
import re
from glob import glob
import json
import pickle
import fiona
import shutil
import matplotlib.pyplot as plt
import matplotlib.colors as mcolors
from matplotlib.colors import to_rgb
from osgeo import gdal, ogr
from rasterio.transform import array_bounds
from rasterio.windows import from_bounds, Window
from rasterio.warp import calculate_default_transform, reproject, Resampling, transform_bounds
from rasterio.features import rasterize
import rasterio.merge
import rasterio
import geopandas as gpd
from geopy.geocoders import Nominatim
from fiona.transform import transform, transform_geom
from pystac.extensions.projection import ProjectionExtension as pe
from pystac.extensions.eo import EOExtension as eo
import pystac_client
import planetary_computer

os.environ['TF_CPP_MIN_LOG_LEVEL'] = '3'
os.environ['PC_SDK_SUBSCRIPTION_KEY'] = 'a613baefa08445269838bc3bc0dfe2d9'
from tensorflow.keras.layers import MaxPooling2D
import tensorflow as tf

# Import configuration
from config import DataConfig

# Initialize configuration with filepaths
config = DataConfig()

PRISM_CRS = "EPSG:4269"
SEARCH_CRS = "EPSG:4326"
NLCD_RANGE = None
ELEVATION_LAT_LONG = None
CURRENT_DATE = datetime.now().strftime('%Y-%m-%d')

# NLCD color mapping dictionary
nlcd_colors = {
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
nlcd_code_to_rgb = {
    code: tuple(int(255 * c) for c in to_rgb(hex_color))
    for code, hex_color in nlcd_colors.items()
}

# patterns for checking if an event has been processed
regex_patterns = [
    r'dem_.*\.tif',
    r'flowlines_.*\.tif',
    r'roads_.*\.tif',
    r'waterbody_.*\.tif',
    r'nlcd_.*\.tif',
    r'tci_\d{8}.*\.tif',
    r'rgb_\d{8}.*\.tif',
    r'ndwi_\d{8}.*\.tif',
    r'b08_\d{8}.*\.tif',
    r'sar_\d{8}.*\.tif',
    'metadata.json'
]
pattern_dict = {
    r'dem_.*\.tif': 'DEM',
    r'flowlines_.*\.tif': 'FLOWLINES',
    r'roads_.*\.tif': 'ROADS',
    r'waterbody_.*\.tif': 'WATERBODY',
    r'nlcd_.*\.tif': 'NLCD',
    r'tci_\d{8}.*\.tif': 'S2 IMAGERY',
    r'rgb_\d{8}.*\.tif': 'RGB',
    r'ndwi_\d{8}.*\.tif': 'NDWI',
    r'b08_\d{8}.*\.tif': 'B8 NIR',
    r'sar_\d{8}.*\.tif': 'SAR',
    'metadata.json': 'METADATA'
}

class NoElevationError(Exception):
    pass

def get_elevation_nw():
    """Extract all elevation file lat long pairs as well as the filepath in tuple."""
    global ELEVATION_LAT_LONG
    if ELEVATION_LAT_LONG is None:
        elevation_files = [str(x) for x in Path(config.elevation_directory).glob('n*w*.tif')]
        if len(elevation_files) == 0:
            raise FileNotFoundError('No elevation files found. Please run get_supplementary.py to download elevation data.')
        p = re.compile(r"n(\d*)w(\d*).tif")
        lst = []
        for file in elevation_files:
            m = p.search(file)
            if m:
                lst.append((m.group(1), m.group(2), file))
        ELEVATION_LAT_LONG = lst
    return ELEVATION_LAT_LONG

def get_nlcd_range():
    """Returns a tuple of the earliest and latest year for which NLCD data is available."""
    global NLCD_RANGE
    if NLCD_RANGE is None:
        nlcd_files = [str(x) for x in Path(config.nlcd_directory).glob('LndCov*.tif')]
        if len(nlcd_files) == 0:
            raise FileNotFoundError('No NLCD files found. Please run get_supplementary.py to download NLCD data.')
        p = re.compile(r'LndCov(\d{4}).tif')
        nlcd_years = [int(p.search(file).group(1)) for file in nlcd_files]
        NLCD_RANGE = (min(nlcd_years), max(nlcd_years))
    return NLCD_RANGE

def read_PRISM():
    """Reads the PRISM netCDF file and return the encoded data."""
    with Dataset(config.prism_file, "r", format="NETCDF4") as nc:
        geotransform = nc["geotransform"][:]
        time_info = (nc["time"].units, nc["time"].calendar)
        precip_data = nc["precip"][:]

    return (geotransform, time_info, precip_data)

def ceser_mask(precip_data):
    """Masks the CESER AOI in the PRISM meshgrid. Set values outside mask to nan.
    
    Parameters
    ----------
    precip_data : numpy.ndarray
        PRISM precipitation data ndarray with shape (time, y, x).

    Returns
    -------
    numpy.ndarray
        PRISM precipitation data ndarray with cells outside CESER boundary set to nan.
    """
    ceser_boundary = gpd.read_file(config.ceser_boundary_file)
    meshgrid = gpd.read_file(config.prism_meshgrid_file)
    # Reproject ceser to prism grid
    if ceser_boundary.crs != meshgrid.crs:
        ceser_reprojected = ceser_boundary.to_crs(meshgrid.crs)
        ref_shape = ceser_reprojected.geometry.iloc[0]
    else:
        ref_shape = ceser_boundary.geometry.iloc[0]
    intersecting_shapes = meshgrid[meshgrid.geometry.intersects(ref_shape)]
    filtered_indices = list(zip(intersecting_shapes['row'],intersecting_shapes['col']))

    # filter our PRISM array by cells inside the CESER boundary
    mask_2d = np.zeros((precip_data.shape[1], precip_data.shape[2]), dtype=bool)
    for i, j in filtered_indices:
        mask_2d[i, j] = True

    # Broadcast the 2D mask to 3D to match precip_data shape
    mask_3d = np.broadcast_to(mask_2d, precip_data.shape)

    # set any cells outside mask to nan
    masked_precip = np.where(mask_3d, precip_data, np.nan)
    return masked_precip

def get_manual_events(history, manual_file):
    """Compile info for manually specified events via list of PRISM indices.
    
    Parameters
    ----------
    history : set()
        Set that holds all tuples of indices (from PRISM ndarray) of previously processed events.
    manual_file : str
        Path to text file containing manually specified events in lines with format: time, y, x.

    Returns
    -------
    int
        Number of manual events found.
    Iterator
        Iterator aggregates event data with each tuple containing the date, cumulative day precipitation in mm, latitude longitude bounding box values and a unique event id.
    """
    # validate that the indices are within the PRISM grid
    logger = logging.getLogger('main')
    logger.info("Loading PRISM data...")
    geotransform, time_info, precip_data = read_PRISM()
    logger.info("PRISM successfully loaded.")

    # read in and validate manual indices
    manual_indices = []
    with open(manual_file, 'r') as f:
        for lineno, line in enumerate(f, 1):
            stripped = line.strip()
            if not stripped or stripped.startswith('#'):  # skip empty lines and comments
                continue
            parts = [p.strip() for p in stripped.split(',')]
            if len(parts) != 3:
                raise ValueError(f"Line {lineno}: Expected 3 comma-separated values, got {parts}")
            try:
                values = tuple(map(int, parts))
            except ValueError:
                raise ValueError(f"Line {lineno}: Invalid integers: {parts}")
            if any(v < 0 for v in values):
                raise ValueError(f"Line {lineno}: All values must be non-negative: {values}")

            time, y, x = values
            if time >= precip_data.shape[0] or y >= precip_data.shape[1] or x >= precip_data.shape[2]:
                raise ValueError(f"Index ({time}, {y}, {x}) is out of bounds for PRISM data.")
            manual_indices.append(values)
    
    # float64 geotransform coefficients for PRISM grid
    upper_left_x, x_size, x_rotation, upper_left_y, y_rotation, y_size = geotransform
    event_dates = []
    event_precip = []
    minx = []
    miny = []
    maxx = []
    maxy = []
    eid = []
    indices = []

    min_date = datetime(2016, 8, 1)
    count = 0
    for time, y, x in manual_indices:
        event_date = num2date(time, units=time_info[0], calendar=time_info[1])
        event_date_str = event_date.strftime("%Y%m%d")

        # must not be earlier than s2 launch, or previously queried in pipeline
        if event_date < min_date:
            continue
        elif (time, y, x) in history:
            continue

        event_dates.append(event_date_str)
        event_precip.append(precip_data[time, y, x])
        
        # convert latitude and longitude to
        # (minx, miny, maxx, maxy) using the equation
        minx.append(x * x_size + upper_left_x)
        miny.append((y + 1) * y_size + upper_left_y)
        maxx.append((x + 1) * x_size + upper_left_x)
        maxy.append(y * y_size + upper_left_y)
        eid.append(f'{event_date_str}_{y}_{x}')
        indices.append((time, y, x)) # indices for efficient tracking of completion
        count += 1

    return count, zip(event_dates, event_precip, minx, miny, maxx, maxy, eid, indices)

def get_extreme_events(history, threshold=300, n=None, region=None):
    """
    Queries contents of PRISM netCDF file given the return value of read_PRISM(). Filters for 
    precipitation events that meet minimum threshold of precipitation and returns an iterator.

    Parameters
    ----------
    history : set()
        Set that holds all tuples of indices (from PRISM ndarray) of previously processed events.
    threshold : int, optional
        Minimum cumulative daily precipitation in mm for filtering PRISM tiles.
    n : int, optional
        Only prepares first n extreme precipitation events that meet threshold criteria.

    Returns
    -------
    int
        Number of extreme precipitation events found.
    Iterator
        Iterator aggregates extreme event data with each tuple containing the date, cumulative day precipitation in mm, latitude longitude bounding box values and a unique event id.
    """
    logger = logging.getLogger('main')
    logger.info("Loading PRISM data...")
    geotransform, time_info, precip_data = read_PRISM()
    logger.info("PRISM successfully loaded.")

    # for specific regions, mask the AOI
    if region is not None:
        if region == 'ceser':
            precip_data = ceser_mask(precip_data)

    # returns tuple of arrays of time, lat indices, lon indices for filtered target events
    events = np.where(precip_data > threshold)

    # float64 geotransform coefficients for PRISM grid
    upper_left_x, x_size, x_rotation, upper_left_y, y_rotation, y_size = geotransform
    event_dates = []
    event_precip = []
    minx = []
    miny = []
    maxx = []
    maxy = []
    eid = []
    indices = []

    min_date = datetime(2016, 8, 1) # originally 2015, 9, 1
    count = 0
    for time, y, x in zip(events[0], events[1], events[2]):
        if n is not None and count >= n:
            break
            
        event_date = num2date(time, units=time_info[0], calendar=time_info[1])
        event_date_str = event_date.strftime("%Y%m%d")

        # must not be earlier than s2 launch, or previously queried in pipeline
        if event_date < min_date:
            continue
        elif (time, y, x) in history:
            continue

        event_dates.append(event_date_str)
        event_precip.append(precip_data[time, y, x])
        
        # convert latitude and longitude to
        # (minx, miny, maxx, maxy) using the equation
        minx.append(x * x_size + upper_left_x)
        miny.append((y + 1) * y_size + upper_left_y)
        maxx.append((x + 1) * x_size + upper_left_x)
        maxy.append(y * y_size + upper_left_y)
        eid.append(f'{event_date_str}_{y}_{x}')
        indices.append((time, y, x)) # indices for efficient tracking of completion
        count += 1

    return count, zip(event_dates, event_precip, minx, miny, maxx, maxy, eid, indices)

def event_completed(dir_path):
    """Returns whether or not event directory contains all generated rasters."""
    logger = logging.getLogger('main')
    logger.info('Confirming whether event has already been successfully processed before...')
    existing_files = os.listdir(dir_path)
    
    # Check if each file name in the list exists in the directory
    missing_files = []

    for pattern in regex_patterns:
        pattern_matched = False
        for file_name in existing_files:
            if re.match(pattern, file_name) and os.path.getsize(dir_path + file_name) > 0:
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

def get_date_interval(event_date, days_before, days_after):
    """Returns a date interval made of a tuple of start and end date strings given event date string and 
    number of days extending before and after the event.
    
    Parameters
    ----------
    event_date : str
        Date string formatted as YYYYMMDD.
    days_before : int
    days_after : int

    Returns
    -------
    (str, str)
        Interval with start and end date strings formatted as YYYY-MM-DD.
    """
    if days_before + days_after == 0:
        return f"{event_date[0:4]}-{event_date[4:6]}-{event_date[6:8]}"
    delt1 = timedelta(days = days_before)
    delt2 = timedelta(days = days_after)
    start = datetime(int(event_date[0:4]), int(event_date[4:6]), int(event_date[6:8])) - delt1
    end = datetime(int(event_date[0:4]), int(event_date[4:6]), int(event_date[6:8])) + delt2
    return start.strftime("%Y-%m-%d") + '/' + end.strftime("%Y-%m-%d")

def found_after_images(items, event_date):
    """Returns whether products are found in query during or after precipitation event date.

    Parameters
    ----------
    items : ItemCollection
    event_date : str
        Formatted as YYYYMMDD.

    Returns
    -------
    bool
        True if query has products that lie on or after event date or False if not.
    """
    dt = datetime.strptime(event_date, "%Y%m%d")
    dt_1 = dt - timedelta(days=1)
    
    # PRISM DAY BEGINS AT 1200 UTC-0 THE DAY BEFORE
    prism_dt = datetime(dt_1.year, dt_1.month, dt_1.day, hour=12, tzinfo=timezone.utc)

    # Compare dates
    for item in items:
        datetime_object = item.datetime
        if datetime_object >= prism_dt:
            return True

    return False

def get_state(minx, miny, maxx, maxy):
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

def make_box(bounds):
    """Returns a box polygon using a bounding box of longitude latitude coordinates.
    
    Parameters
    ----------
    bounds : (float, float, float, float)
        Tuple in the order minx, miny, maxx, maxy representing bounding box.

    Returns
    -------
    ogr.Geometry
        GDAL/OGR polygon object.
    """
    minX, minY, maxX, maxY = bounds
    
    # Create ring
    ring = ogr.Geometry(ogr.wkbLinearRing)
    ring.AddPoint(minX, minY)
    ring.AddPoint(maxX, minY)
    ring.AddPoint(maxX, maxY)
    ring.AddPoint(minX, maxY)
    ring.AddPoint(minX, minY)

    poly = ogr.Geometry(ogr.wkbPolygon)
    poly.AddGeometry(ring)
    return poly

def GetHU4Codes(bbox):
    """
    Queries national watershed boundary dataset for HUC 4 codes representing
    hydrologic units that intersect with bounding box.

    Parameters
    ----------
    bbox : (float, float, float, float)
        Tuple in the order minx, miny, maxx, maxy, representing bounding box.

    Returns
    -------
    list[str]
        HU4 codes.
    """
    # TODO: for newer gdal add context manager for opening file
    wbd = gdal.OpenEx(str('/vsizip/' / Path(config.nhd_wbd_file) / 'WBD_National_GDB.gdb'))
    wbdhu4 = wbd.GetLayerByIndex(4)

    lst = []
    boundpoly = make_box(bbox)
    for feature in wbdhu4:
        if feature.GetGeometryRef().Intersects(boundpoly):
            lst.append(feature.GetField('huc4'))

    return lst

def buffer_raster(arr, buffer):
    """Add buffer to 2d numpy array."""
    x = tf.constant(arr)
    x = tf.reshape(x, [1, arr.shape[0], arr.shape[1], 1])
    max_pool_2d = MaxPooling2D(pool_size=(buffer, buffer),
       strides=(1, 1), padding='same')
    pooled = max_pool_2d(x)
    buffer_arr = pooled.numpy()[0, :, :, 0]
    return buffer_arr

def colormap_to_rgb(arr, cmap='viridis', r=None, no_data=None):
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

def cloud_null_percentage(dir_path, item, item_crs, bbox):
    item_href = planetary_computer.sign(item.assets["SCL"].href)

    conversion = transform(PRISM_CRS, item_crs, (bbox[0], bbox[2]), (bbox[1], bbox[3]))
    img_bbox = conversion[0][0], conversion[1][0], conversion[0][1], conversion[1][1]

    out_image, _ = rasterio.merge.merge([item_href], bounds=img_bbox)
    return int((np.isin(out_image, [0, 8, 9]).sum() / out_image.size) * 100)

def sar_missing_percentage(dir_path, item, item_crs, bbox):
    item_href = planetary_computer.sign(item.assets["vv"].href)

    conversion = transform(PRISM_CRS, item_crs, (bbox[0], bbox[2]), (bbox[1], bbox[3]))
    img_bbox = conversion[0][0], conversion[1][0], conversion[0][1], conversion[1][1]

    out_image, _ = rasterio.merge.merge([item_href], bounds=img_bbox)
    return int((np.sum(out_image <= 0) / out_image.size) * 100)

# we will choose a UTM zone CRS already given and stick to it for rest of sample data!
def pipeline_TCI(dir_path, save_as, dst_crs, item, bbox):
    """Generates TCI (True Color Image) raster of S2 multispectral file.

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
    item_href = planetary_computer.sign(item.assets["visual"].href)

    out_image, out_transform = rasterio.merge.merge([item_href], bounds=bbox, nodata=0, resampling=Resampling.bilinear)

    with rasterio.open(dir_path + save_as + '.tif', 'w', driver='Gtiff', count=3, height=out_image.shape[-2], width=out_image.shape[-1], crs=dst_crs, dtype=out_image.dtype, transform=out_transform, nodata=None) as dst:
        dst.write(out_image)

    return (out_image.shape[-2], out_image.shape[-1]), out_transform

def pipeline_RGB(dir_path, save_as, dst_crs, item, bbox):
    """Generates B02 (B), B03 (G), B04 (R) rasters of S2 multispectral file.

    Parameters
    ----------
    dir_path : str
        Path for saving generated raster. 
    save_as : str
        Name of file to be saved (do not include extension!).
    dst_crs : obj
        Coordinate reference system of output raster.
    items : list[Item]
        List of PyStac Item objects.
    box : (float, float, float, float)
        Tuple in the order minx, miny, maxx, maxy, representing bounding box, 
        should be in CRS specified by dst_crs.
    """
    b02_item_href = planetary_computer.sign(item.assets["B02"].href) # B
    b03_item_href = planetary_computer.sign(item.assets["B03"].href) # G
    b04_item_href = planetary_computer.sign(item.assets["B04"].href) # R

    # stack the three bands as rgb channels
    blue_image, out_transform = rasterio.merge.merge([b02_item_href], bounds=bbox, nodata=0, resampling=Resampling.bilinear)
    green_image, _ = rasterio.merge.merge([b03_item_href], bounds=bbox, nodata=0, resampling=Resampling.bilinear)
    red_image, _ = rasterio.merge.merge([b04_item_href], bounds=bbox, nodata=0, resampling=Resampling.bilinear)

    out_image = np.vstack((red_image, green_image, blue_image))

    with rasterio.open(dir_path + save_as + '.tif', 'w', driver='Gtiff', count=3, height=out_image.shape[-2], width=out_image.shape[-1], crs=dst_crs, dtype=out_image.dtype, transform=out_transform, nodata=0) as dst:
        dst.write(out_image)

def pipeline_B08(dir_path, save_as, dst_crs, item, bbox):
    """Generates NIR B8 band raster of S2 multispectral file.

    Parameters
    ----------
    dir_path : str
        Path for saving generated raster. 
    save_as : str
        Name of file to be saved (do not include extension!).
    dst_crs : obj
        Coordinate reference system of output raster.
    items : list[Item]
        List of PyStac Item objects.
    box : (float, float, float, float)
        Tuple in the order minx, miny, maxx, maxy, representing bounding box, 
        should be in CRS specified by dst_crs.
    """
    item_href = planetary_computer.sign(item.assets["B08"].href)

    out_image, out_transform = rasterio.merge.merge([item_href], bounds=bbox, nodata=0, resampling=Resampling.bilinear)

    with rasterio.open(dir_path + save_as + '.tif', 'w', driver='Gtiff', count=1, height=out_image.shape[-2], width=out_image.shape[-1], crs=dst_crs, dtype=out_image.dtype, transform=out_transform, nodata=0) as dst:
        dst.write(out_image)

def pipeline_NDWI(dir_path, save_as, dst_crs, item, bbox):
    """Generates NDWI raster from S2 multispectral files.

    Parameters
    ----------
    dir_path : str
        Path for saving generated raster. 
    save_as : str
        Name of raw ndwi file to be saved (do not include extension!).
    items : list[Item]
        List of PyStac Item objects.
    bbox : (float, float, float, float)
        Tuple in the order minx, miny, maxx, maxy, representing bounding box, 
        should be in CRS specified by dst_crs.
    """
    b03_item_href = planetary_computer.sign(item.assets["B03"].href)

    out_image1, _ = rasterio.merge.merge([b03_item_href], bounds=bbox, nodata=0, resampling=Resampling.bilinear)
    green = out_image1[0].astype(np.int32)

    b08_item_href = planetary_computer.sign(item.assets["B08"].href)

    out_image2, out_transform = rasterio.merge.merge([b08_item_href], bounds=bbox, nodata=0, resampling=Resampling.bilinear)
    nir = out_image2[0].astype(np.int32)
    
    # calculate ndwi
    ndwi = np.where((green + nir) != 0, (green - nir)/(green + nir), -999999)

    # save raw
    with rasterio.open(dir_path + save_as + '.tif', 'w', driver='Gtiff', count=1, height=ndwi.shape[-2], width=ndwi.shape[-1], crs=dst_crs, dtype=ndwi.dtype, transform=out_transform, nodata=-999999) as dst:
        dst.write(ndwi, 1)

    # before writing to file, we will make matplotlib colormap!
    ndwi_colored = colormap_to_rgb(ndwi, cmap='seismic_r', r=(-1.0, 1.0), no_data=-999999)
    
    # nodata should not be set for cmap files
    with rasterio.open(dir_path + save_as + '_cmap.tif', 'w', driver='Gtiff', count=3, height=ndwi_colored.shape[-2], width=ndwi_colored.shape[-1], crs=dst_crs, dtype=ndwi_colored.dtype, transform=out_transform, nodata=None) as dst:
        dst.write(ndwi_colored)

def pipeline_roads(dir_path, save_as, dst_shape, dst_crs, dst_transform, state, buffer=0):
    """Generates raster with burned in geometries of roads given destination raster properties.

    Parameters
    ----------
    dir_path : str
        Path for saving generated raster. 
    save_as : str
        Name of file to be saved (do not include extension!).
    dst_shape : (int, int)
        Shape of output raster.
    dst_crs : obj
        Coordinate reference system of output raster.
    dst_transform : rasterio.affine.Affine()
        Transformation matrix for mapping pixel coordinates to coordinate system of output raster.
    state : str
        State of raster location.
    buffer : int, optional
        Buffer the geometry line thickness by certain number of pixels.
    """
    # find state shape file
    with fiona.open(Path(config.roads_directory) / f'{state.strip().upper()}.shp', "r") as shapefile:
        shapes = [transform_geom(shapefile.crs, dst_crs, feature["geometry"]) for feature in shapefile]

    if shapes:
        rasterize_roads = rasterize(
            [(line, 1) for line in shapes],
            out_shape=dst_shape,
            transform=dst_transform,
            fill=0,
            all_touched=True,
            dtype=np.uint8)

        if buffer > 0:
            rasterize_roads = buffer_raster(rasterize_roads, buffer)
            
        # Create RGB raster
        rgb_roads = np.zeros((3, rasterize_roads.shape[0], rasterize_roads.shape[1]), dtype=np.uint8)
    
        # Set values in the 3D array based on the binary_array
        rgb_roads[0, :, :] = rasterize_roads * 255
        rgb_roads[1, :, :] = rasterize_roads * 255
        rgb_roads[2, :, :] = rasterize_roads * 255
    else:
        # if no shapes to rasterize
        rasterize_roads = np.zeros(dst_shape, dtype=np.uint8)
        rgb_roads = np.zeros((3, *dst_shape), dtype=np.uint8)

    with rasterio.open(dir_path + save_as + '.tif', 'w', driver='Gtiff', count=1, height=rasterize_roads.shape[-2], width=rasterize_roads.shape[-1], 
                       crs=dst_crs, dtype=rasterize_roads.dtype, transform=dst_transform, nodata=0) as dst:
        dst.write(rasterize_roads, 1)

    with rasterio.open(dir_path + save_as + '_cmap.tif', 'w', driver='Gtiff', count=3, height=rgb_roads.shape[-2], width=rgb_roads.shape[-1], 
                       crs=dst_crs, dtype=rgb_roads.dtype, transform=dst_transform, nodata=0) as dst:
        dst.write(rgb_roads)

def pipeline_dem(dir_path, save_as, dst_shape, dst_crs, dst_transform, bounds):
    """Generates Digital Elevation Map raster given destination raster properties and bounding box.

    Note to developers: slope raster generation removed in favor of calling np.gradient on dem tile during preprocessing.

    Parameters
    ----------
    dir_path : str
        Path for saving generated raster. 
    save_as : str
        Name of file to be saved (do not include extension!).
    dst_shape : (int, int)
        Shape of output raster.
    dst_crs : str
        Coordinate reference system of output raster.
    dst_transform : rasterio.affine.Affine()
        Transformation matrix for mapping pixel coordinates to coordinate system of output raster.
    bounds : (float, float, float, float)
        Tuple in the order minx, miny, maxx, maxy, representing bounding box.
    """
    # buffer here accounts for missing edges after reprojection!
    buffer = 0.02
    bounds = (bounds[0] - buffer, bounds[1] - buffer, bounds[2] + buffer, bounds[3] + buffer)
    
    lst = []
    for lat, long, file in get_elevation_nw():
        # we want the range of the lats and the range of the longs to intersect the bounds
        # nxxwxx is TOP LEFT corner of tile!
        tile_bounds = (int(long) * (-1), int(lat) - 1, int(long) * (-1) + 1, int(lat))
        if tile_bounds[0] < bounds[2] and tile_bounds[2] > bounds[0] and tile_bounds[1] < bounds[3] and tile_bounds[3] > bounds[1]:
            lst.append(file)

    # if no dem tile can be found, raise error
    if not lst:
        raise NoElevationError(f'No elevation raster can be found for bounds ({bounds[0]}, {bounds[1]}, {bounds[2]}, {bounds[3]}) during generation of dem raster.')

    files_crs = []
    files_nodata = []
    for filepath in lst:
        with rasterio.open(filepath) as src:
            files_crs.append(src.crs.to_string())
            files_nodata.append(src.nodata)
            
    if len(files_crs) == 0:
        raise Exception(f"No CRS found for files: {', '.join(lst)}")
    elif len(files_nodata) == 0:
        raise Exception(f"Absent no data value found for files: {', '.join(lst)}")

    src_crs = files_crs[0]
    no_data = files_nodata[0]
        
    if all(item == files_crs[0] for item in files_crs) and all(item == files_nodata[0] for item in files_nodata):
        src_image, src_transform = rasterio.merge.merge(lst, bounds=bounds, nodata=no_data, resampling=Resampling.bilinear)
        destination = np.zeros(dst_shape, src_image.dtype)
    else: 
        raise Exception("CRS and no data values of all files must match")
    
    reproject(
        src_image,
        destination,
        src_transform=src_transform,
        src_crs=src_crs,
        dst_transform=dst_transform,
        dst_crs=dst_crs,
        resampling=Resampling.bilinear)

    # for DEM use grayscale!
    dem_cmap = colormap_to_rgb(destination, cmap='gray', no_data=no_data)

    # save dem raw
    with rasterio.open(dir_path + save_as + '.tif', 'w', driver='Gtiff', count=1, height=destination.shape[-2], width=destination.shape[-1], crs=dst_crs, dtype=destination.dtype, transform=dst_transform, nodata=no_data) as dst:
        dst.write(destination, 1)

    # save dem cmap
    with rasterio.open(dir_path + save_as + '_cmap.tif', 'w', driver='Gtiff', count=3, height=dem_cmap.shape[-2], width=dem_cmap.shape[-1], crs=dst_crs, dtype=dem_cmap.dtype, transform=dst_transform, nodata=None) as dst:
        dst.write(dem_cmap)

def pipeline_flowlines(dir_path, save_as, dst_shape, dst_crs, dst_transform, bbox, buffer=3, filter=['460\d{2}', '558\d{2}', '336\d{2}', '334\d{2}', '42801', '42802', '42805', '42806', '42809']):
    """Generates raster with burned in geometries of flowlines given destination raster properties.

    Parameters
    ----------
    dir_path : str
        Path for saving generated raster. 
    save_as : str
        Name of file to be saved (no file extension!).
    dst_shape : (int, int)
        Shape of output raster.
    dst_crs : str
        Coordinate reference system of output raster.
    dst_transform : rasterio.affine.Affine()
        Transformation matrix for mapping pixel coordinates to coordinate system of output raster.
    bbox : (float, float, float, float)
        Tuple in the order minx, miny, maxx, maxy, representing bounding box.
    buffer : int, optional
        Buffer the geometry line thickness by certain number of pixels.
    filter : list[str], optional
        List of regex patterns for filtering specific flowline features based on their NHDPlus dataset FCodes.
    """
    with ExitStack() as stack:
        files = [stack.enter_context(fiona.open(f'/vsizip/NHD/NHDPLUS_H_{code}_HU4_GDB.zip/NHDPLUS_H_{code}_HU4_GDB.gdb', layer='NHDFlowline')) for code in GetHU4Codes(bbox)]
        if not files:
            raise Exception(f"No flowline files found for HU4 codes in bounding box {bbox}")

        files_crs = {file.crs for file in files}
        if len(files_crs) > 1:
            raise Exception("CRS of all files must match")
        src_crs = files[0].crs
        
        shapes = []
        p = re.compile('|'.join(filter))
        for i, lyr in enumerate(files):
            for _, feat in lyr.items(bbox=bbox):
                m = p.match(str(feat.properties['FCode']))
                if m:
                    shapes.append((transform_geom(src_crs, dst_crs, feat.geometry), 1))

        if shapes:
            flowlines = rasterize(
                shapes,
                out_shape=dst_shape,
                transform=dst_transform,
                fill=0,
                all_touched=True,
                dtype=np.uint8)

            if buffer > 0:
                flowlines = buffer_raster(flowlines, buffer)

            # Create RGB raster
            rgb_flowlines = np.zeros((3, flowlines.shape[0], flowlines.shape[1]), dtype=np.uint8)
        
            # Set values in the 3D array based on the binary_array
            rgb_flowlines[0, :, :] = flowlines * 255
            rgb_flowlines[1, :, :] = flowlines * 255
            rgb_flowlines[2, :, :] = flowlines * 255
        else:
            # if no shapes to rasterize
            flowlines = np.zeros(dst_shape, dtype=np.uint8)
            rgb_flowlines = np.zeros((3, *dst_shape), dtype=np.uint8)

    # flowlines raw
    with rasterio.open(dir_path + save_as + '.tif', 'w', driver='Gtiff', count=1, height=flowlines.shape[-2], width=flowlines.shape[-1], 
                       crs=dst_crs, dtype=flowlines.dtype, transform=dst_transform, nodata=0) as dst:
        dst.write(flowlines, 1)

    # flowlines cmap
    with rasterio.open(dir_path + save_as + '_cmap.tif', 'w', driver='Gtiff', count=3, height=rgb_flowlines.shape[-2], width=rgb_flowlines.shape[-1], crs=dst_crs, dtype=rgb_flowlines.dtype, transform=dst_transform, nodata=None) as dst:
        dst.write(rgb_flowlines)

def pipeline_waterbody(dir_path, save_as, dst_shape, dst_crs, dst_transform, bbox):
    """Generates raster with burned in geometries of waterbodies given destination raster properties.

    Parameters
    ----------
    dir_path : str
        Path for saving generated raster. 
    save_as : str
        Name of file to be saved (no file extension!).
    dst_shape : (int, int)
        Shape of output raster.
    dst_crs : str
        Coordinate reference system of output raster.
    dst_transform : rasterio.affine.Affine()
        Transformation matrix for mapping pixel coordinates to coordinate system of output raster.
    bbox : (float, float, float, float)
        Tuple in the order minx, miny, maxx, maxy, representing bounding box.
    """
    with ExitStack() as stack:
        files = [stack.enter_context(fiona.open(f'/vsizip/NHD/NHDPLUS_H_{code}_HU4_GDB.zip/NHDPLUS_H_{code}_HU4_GDB.gdb', layer='NHDWaterbody')) for code in GetHU4Codes(bbox)]
        if not files:
            raise Exception(f"No waterbody files found for HU4 codes in bounding box {bbox}")

        files_crs = {file.crs for file in files}
        if len(files_crs) > 1:
            raise Exception("CRS of all files must match")
        src_crs = files[0].crs
        
        shapes = []
        for lyr in files:
            for _, feat in lyr.items(bbox=bbox):
                fcode = feat.properties['FCode']
                # filter out estuary
                if fcode == 49300: 
                    continue
                    
                shapes.append((transform_geom(src_crs, dst_crs, feat.geometry), 1))

        if shapes:
            waterbody = rasterize(
                shapes,
                out_shape=dst_shape,
                transform=dst_transform,
                fill=0,
                all_touched=True,
                dtype=np.uint8)

            # Create RGB raster
            rgb_waterbody = np.zeros((3, waterbody.shape[0], waterbody.shape[1]), dtype=np.uint8)
        
            # Set values in the 3D array based on the binary_array
            rgb_waterbody[0, :, :] = waterbody * 255
            rgb_waterbody[1, :, :] = waterbody * 255
            rgb_waterbody[2, :, :] = waterbody * 255
        else:
            # if no shapes to rasterize
            waterbody = np.zeros(dst_shape, dtype=np.uint8)
            rgb_waterbody = np.zeros((3, *dst_shape), dtype=np.uint8)

    # waterbody raw
    with rasterio.open(dir_path + save_as + '.tif', 'w', driver='Gtiff', count=1, height=waterbody.shape[-2], width=waterbody.shape[-1], crs=dst_crs, dtype=waterbody.dtype, transform=dst_transform, nodata=0) as dst:
        dst.write(waterbody, 1)

    # waterbody cmap
    with rasterio.open(dir_path + save_as + '_cmap.tif', 'w', driver='Gtiff', count=3, height=rgb_waterbody.shape[-2], width=rgb_waterbody.shape[-1], crs=dst_crs, dtype=rgb_waterbody.dtype, transform=dst_transform, nodata=None) as dst:
        dst.write(rgb_waterbody)

def pipeline_NLCD(dir_path, save_as, year, dst_shape, dst_crs, dst_transform):
    """Generates raster with NLCD land cover classes. Uses windowed reading of NLCD raster
    for speed (NLCD files are large).
    
    Parameters
    ----------
    dir_path : str
        Path for saving generated raster. 
    save_as : str
        Name of file to be saved (no file extension!).
    year : int
        Year of NLCD data to use.
    dst_shape : (int, int)
        Shape of output raster.
    dst_crs : str
        Coordinate reference system of output raster.
    dst_transform : rasterio.affine.Affine()
        Transformation matrix for mapping pixel coordinates to coordinate system of output raster.
    """
    # if year is after the most recent year, use the most recent year
    nlcd_range = get_nlcd_range()
    if year > nlcd_range[1]:
        year = nlcd_range[1]
    elif year < nlcd_range[0]:
        year = nlcd_range[0]

    with rasterio.open(Path(config.nlcd_directory) / f'LndCov{year}.tif') as src:
        # data array is of type uint8
        nlcd_crs = src.crs
        nlcd_transform = src.transform

        # get bounds of destination raster in NLCD CRS for making window
        dst_bounds = array_bounds(dst_shape[-2], dst_shape[-1], dst_transform)
        dst_bounds_in_nlcd_crs = transform_bounds(dst_crs, nlcd_crs, *dst_bounds)

        # now get window in nlcd raster, all in pixels not coordinates
        nlcd_window = from_bounds(*dst_bounds_in_nlcd_crs, transform=nlcd_transform)

        # pad bounds by one pixel for additional context
        padded_nlcd_window = Window(nlcd_window.col_off - 1, nlcd_window.row_off - 1, nlcd_window.width + 2, nlcd_window.height + 2)

        nlcd_data = src.read(1, window=padded_nlcd_window)
        window_transform = src.window_transform(padded_nlcd_window)

    # reproject to obtain NLCD raster in AOI
    nlcd_arr = np.empty(dst_shape, dtype=np.uint8)
    _, out_transform = reproject(
        source=nlcd_data,
        destination=nlcd_arr,
        src_transform=window_transform,
        src_crs=nlcd_crs,
        dst_transform=dst_transform,
        dst_crs=dst_crs,
        dst_shape=dst_shape,
        resampling=Resampling.nearest  # NLCD is categorical — nearest preserves class labels
    )
    
    # save NLCD raster
    with rasterio.open(dir_path + save_as + '.tif', 'w', driver='Gtiff', count=1, height=nlcd_arr.shape[-2], width=nlcd_arr.shape[-1], crs=dst_crs, dtype=nlcd_arr.dtype, transform=dst_transform, nodata=250) as dst:
        dst.write(nlcd_arr, 1)

    # create NLCD colormap
    H, W = nlcd_arr.shape
    rgb_img = np.zeros((H, W, 3), dtype=np.uint8)

    # vectorized mapping
    for code, rgb in nlcd_code_to_rgb.items():
        mask = nlcd_arr == code
        rgb_img[mask] = rgb

    rgb_img = np.transpose(rgb_img, (2, 0, 1))

    with rasterio.open(dir_path + save_as + '_cmap.tif', 'w', driver='Gtiff', count=3, height=rgb_img.shape[-2], width=rgb_img.shape[-1], crs=dst_crs, dtype=rgb_img.dtype, transform=dst_transform, nodata=None) as dst:
        dst.write(rgb_img)

def coincident(items_s2, items_s1, hours):
    """Checks if any S2 captures are within given number of hours of S1 captures."""
    for sar in items_s1:
        dt_sar = sar.datetime
        for s2 in items_s2:
            dt_s2 = s2.datetime
            time_difference = abs(dt_sar - dt_s2)
            hours_difference = time_difference.total_seconds() / 3600
            if hours_difference < hours:
                return True
    return False

def coincident_with(item_s1, items_s2, hours):
    """Returns the date of the s2 image that the sar image is coincident with."""
    coincident_dt = None
    dt_sar = item_s1.datetime
    for s2 in items_s2:
        dt_s2 = s2.datetime
        time_difference = abs(dt_sar - dt_s2)
        hours_difference = time_difference.total_seconds() / 3600
        if hours_difference < hours:
            coincident_dt = dt_s2.strftime('%Y%m%d')
            break
    return coincident_dt

def db_scale(x):
    nonzero_mask = x > 0
    missing_mask = x <= 0
    x[nonzero_mask] = 10 * np.log10(x[nonzero_mask])
    x[missing_mask] = -9999
    return x

def pipeline_S1(dir_path, save_as, dst_crs, item, bbox):
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
    item_hrefs_vv = planetary_computer.sign(item.assets["vv"].href)
    item_hrefs_vh = planetary_computer.sign(item.assets["vh"].href)

    out_image_vv, out_transform_vv = rasterio.merge.merge([item_hrefs_vv], bounds=bbox, nodata=0, resampling=Resampling.bilinear)
    out_image_vh, out_transform_vh = rasterio.merge.merge([item_hrefs_vh], bounds=bbox, nodata=0, resampling=Resampling.bilinear)

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

def event_sample_sar(threshold, days_before, days_after, maxcoverpercentage, within_hours, event_date, event_precip, minx, miny, maxx, maxy, eid, dir_path):
    """Samples S2 and S1 coincident imagery for a high precipitation event based on parameters and generates accompanying rasters.

    Note to developers: the script simplifies normalization onto a consistent grid by finding any common CRS shared by S2 and S1 products.
    Once it finds a CRS in common, all products that do not share that CRS are thrown out. The first S2 product is cropped using the bounding box
    and its dimensions (width and height) and affine transform are then used as reference for all subsequent normalization. 
    A smarter approach (that doesn't throw out products arbitrarily) would be to choose a CRS as reference and normalize everything using
    rasterio WarpedVRT, though this is not currently implemented.
    
    Parameters
    ----------
    threshold : int
        Daily cumulative precipitation threshold (mm) used to filter high precipitation event.
    days_before : int
        Number of days before high precipitation event allowed for sampling S2 imagery.
    days_after : int
        Number of days after high precipitation event allowed for sampling S2 imagery.
    maxcoverpercentage : int
        Maximum percentage of combined null data and cloud cover permitted for each sampled cell.
    within_hours : int
        Download S1 and S2 data that are within a given number of hours from each other
    event_date : str
        Date of high precipitation event in format YYYYMMDD.
    event_precip: float
        Cumulative daily precipitation in mm on event date.
    minx : float
        Bounding box value.
    miny : float
        Bounding box value.
    maxx : float
        Bounding box value.
    maxy : float
        Bounding box value.
    eid : str
        Event id.

    Returns
    -------
    bool
        Return True if successful, False if unsuccessful event download and processing.
    """
    logger = logging.getLogger('events')
    logger.info('**********************************')
    logger.info('START OF EVENT TASK LOG:')
    logger.info(f'Beginning event {eid} download with threshold {threshold}...')
    logger.info(f'Event on {event_date} with precipitation {event_precip}mm at bounds: {minx}, {miny}, {maxx}, {maxy}')

    # need to transform box from EPSG 4269 to EPSG 4326 for query
    conversion = transform(PRISM_CRS, SEARCH_CRS, (minx, maxx), (miny, maxy))
    bbox = (conversion[0][0], conversion[1][0], conversion[0][1], conversion[1][1])
    dir_path = dir_path + eid + '/'
    time_of_interest = get_date_interval(event_date, days_before, days_after)
    time_of_interest_sar = get_date_interval(event_date, 0, days_after + 1)

    # Planetary STAC catalog search
    logger.info('Beginning catalog search...')
    max_attempts = 3  # Set the maximum number of attempts
    for attempt in range(1, max_attempts + 1):
        try:
            catalog = pystac_client.Client.open(
                "https://planetarycomputer.microsoft.com/api/stac/v1"
                # modifier=planetary_computer.sign_inplace
            )
        
            search_s2 = catalog.search(
                collections=["sentinel-2-l2a"],
                bbox=bbox,
                datetime=time_of_interest,
                query={"eo:cloud_cover": {"lt": 95}}
            )

            search_s1 = catalog.search(
                collections=["sentinel-1-rtc"],
                bbox=bbox,
                datetime=time_of_interest_sar,
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
    items_s2 = search_s2.item_collection()
    items_s1 = search_s1.item_collection()
    if len(items_s2) == 0 or len(items_s1) == 0:
        if len(items_s2) == 0:
            logger.info(f'No S2 products found for date interval {time_of_interest}.')
        if len(items_s1) == 0:
            logger.info(f'No S1 products found for date interval {time_of_interest}.')
        return False
    elif not found_after_images(items_s2, event_date):
        logger.info(f'Products found but only before precipitation event date {event_date}.')
        return False
    elif not coincident(items_s2, items_s1, within_hours):
        logger.info(f'No coincident S1 and S2 products found within {within_hours} hours of each other.')
        return False

    # group items by crs first
    logger.info(f'Checking s2 cloud null percentage...')
    s2_products_by_crs = dict()
    for item in items_s2:
        # filter out those w/ cloud null via cloud null percentage checks
        item_crs = pe.ext(item).crs_string
        try:
            coverpercentage = cloud_null_percentage(dir_path, item, item_crs, (minx, miny, maxx, maxy))
        except Exception as err:
            logger.error(f'Cloud null percentage calculation error for item {item.id}: {err}, {type(err)}')
            raise err

        if coverpercentage > maxcoverpercentage:
            logger.debug(f'Sample {item.id} near event {event_date}, at {minx}, {miny}, {maxx}, {maxy} rejected due to {coverpercentage}% cloud or null cover.')
            continue

        if item_crs in s2_products_by_crs:
            s2_products_by_crs[item_crs].append(item)
        else:
            s2_products_by_crs[item_crs] = [item]

    if not s2_products_by_crs:
        logger.debug(f'No s2 images left after filtering...')
        return False

    logger.info(f'Checking s1 null percentage...')
    s1_products_by_crs = dict()
    for item in items_s1:
        item_crs = pe.ext(item).crs_string
        polarizations = item.properties["sar:polarizations"]
        if "VV" not in polarizations or "VH" not in polarizations:
            logger.error(f'S1 product {item.id} VV or VH not found.')
            continue

        # filter out non coincident sar products
        if not item_crs in s2_products_by_crs:
            logger.error(f"S1 product {item.id} crs {item_crs} not found in any of the s2 products: {', '.join(s2_products_by_crs.keys())}.")
            continue
        if not coincident(s2_products_by_crs[item_crs], [item], within_hours):
            continue
        
        try:
            coverpercentage = sar_missing_percentage(dir_path, item, item_crs, (minx, miny, maxx, maxy))
        except Exception as err:
            logger.error(f'Missing percentage calculation error for item {item.id}: {err}, {type(err)}')
            raise err

        if coverpercentage > maxcoverpercentage:
            logger.debug(f'SAR sample {item.id} near event {event_date}, at {minx}, {miny}, {maxx}, {maxy} rejected due to {coverpercentage}% missing data.')
            continue

        if item_crs in s1_products_by_crs:
            s1_products_by_crs[item_crs].append(item)
        else:
            s1_products_by_crs[item_crs] = [item]

    if not s1_products_by_crs:
        logger.debug(f'No s1 images left after filtering...')
        return False
        
    # ensure one CRS still has post-event coincident image after filters
    # choose that to be the crs we work with
    logger.info('Checking once more for coincidence...')
    valid_crs = None
    for crs, s2_products in s2_products_by_crs.items():
        if not found_after_images(s2_products, event_date):
            continue
        if crs not in s1_products_by_crs:
            continue

        valid_crs = crs
        break

    if valid_crs is None:
        logger.debug(f'No valid crs found that satisfies criteria after filtering.')
        return False
    else:
        Path(dir_path).mkdir(parents=True, exist_ok=True)
        
    # begin raster generation with chosen CRS!
    try:
        state = get_state(minx, miny, maxx, maxy)
        if state is None:
            raise Exception(f'State not found for {event_date}, at {minx}, {miny}, {maxx}, {maxy}')
    except Exception as err:
        logger.error(f'Error fetching state: {err}, {type(err)}. Id: {eid}. Removing directory and contents.')
        shutil.rmtree(dir_path)
        return False

    # raster generation
    logger.info('Beginning raster generation...')
    try:
        if valid_crs != PRISM_CRS:
            conversion = transform(PRISM_CRS, valid_crs, (minx, maxx), (miny, maxy))
            cbbox = conversion[0][0], conversion[1][0], conversion[0][1], conversion[1][1]
        else:
            cbbox = (minx, miny, maxx, maxy)

        s2_completed_dt = set()
        for item in s2_products_by_crs[valid_crs]:
            dt = item.datetime.strftime('%Y%m%d') 
            if dt in s2_completed_dt:
                # Most of the time there are multiple overlapping s2 products on same date, can just choose one
                logger.debug(f'For valid crs two s2 rasters found on same date.')
                continue
            else:
                s2_completed_dt.add(dt)
            dst_shape, dst_transform = pipeline_TCI(dir_path, f'tci_{dt}_{eid}', valid_crs, item, cbbox)
            logger.debug(f'TCI raster completed for {dt}.')
            pipeline_RGB(dir_path, f'rgb_{dt}_{eid}', valid_crs, item, cbbox)
            logger.debug(f'RGB raster completed for {dt}.')
            pipeline_B08(dir_path, f'b08_{dt}_{eid}', valid_crs, item, cbbox)
            logger.debug(f'B08 raster completed for {dt}.')
            pipeline_NDWI(dir_path, f'ndwi_{dt}_{eid}', valid_crs, item, cbbox)
            logger.debug(f'NDWI raster completed for {dt}.')
            
        logger.debug(f'All S2, B08, NDWI rasters completed successfully.')

        s1_completed_dt = set()
        for item in s1_products_by_crs[valid_crs]:
            dt = item.datetime.strftime('%Y%m%d') 
            if dt in s1_completed_dt:
                logger.debug(f'For valid crs two s1 rasters found on same date.')
                continue
            else:
                s1_completed_dt.add(dt)
            cdt = coincident_with(item, s2_products_by_crs[valid_crs], within_hours)
            if cdt is None:
                logger.debug(f'No coincident date found for s1 product at {dt}. SAR not downloaded.')
                continue
            pipeline_S1(dir_path, f'sar_{cdt}_{dt}_{eid}', valid_crs, item, cbbox)

        logger.debug(f'All coincident S1 rasters completed successfully.')

        # save all supplementary rasters in raw and rgb colormap
        pipeline_roads(dir_path, f'roads_{eid}', dst_shape, valid_crs, dst_transform, state, buffer=1)
        logger.debug(f'Roads raster completed successfully.')
        pipeline_dem(dir_path, f'dem_{eid}', dst_shape, valid_crs, dst_transform, (minx, miny, maxx, maxy))
        logger.debug(f'DEM raster completed successfully.')
        pipeline_flowlines(dir_path, f'flowlines_{eid}', dst_shape, valid_crs, dst_transform, (minx, miny, maxx, maxy), buffer=1)
        logger.debug(f'Flowlines raster completed successfully.')
        pipeline_waterbody(dir_path, f'waterbody_{eid}', dst_shape, valid_crs, dst_transform, (minx, miny, maxx, maxy))
        logger.debug(f'Waterbody raster completed successfully.')
        pipeline_NLCD(dir_path, f'nlcd_{eid}', int(eid[:4]), dst_shape, valid_crs, dst_transform)
        logger.debug(f'NLCD raster completed successfully.')
    except Exception as err:
        logger.error(f'Raster generation error: {err}, {type(err)}')
        logger.error(f'Raster generation failed for {event_date}, at {minx}, {miny}, {maxx}, {maxy}. Id: {eid}. Removing directory and contents.')
        shutil.rmtree(dir_path)
        raise err
        
    # lastly generate metadata file
    logger.info(f'Generating metadata file...')
    metadata = {
        "metadata": {
            "Download Date": CURRENT_DATE,
            "Sample ID": eid,
            "Precipitation Event Date": event_date,
            "Cumulative Daily Precipitation (mm)": float(event_precip),
            "Precipitation Threshold (mm)": threshold,
            "CRS": valid_crs,
            "State": state,
            "Bounding Box": {
                "minx": minx,
                "miny": miny,
                "maxx": maxx,
                "maxy": maxy
            },
            "S2, S1 Coincident Within (hours)": within_hours,
            "Max Cloud/Nodata Cover Percentage (%)": maxcoverpercentage,
        }
    }

    with open(dir_path + 'metadata.json', "w") as json_file:
        json.dump(metadata, json_file, indent=4)
    
    logger.info('Metadata and raster generation completed. Event finished.')
    return True
    
def main(threshold, days_before, days_after, maxcoverpercentage, maxevents, dir_path=None, within_hours=12, region=None, config_file=None, manual=None):
    """
    Samples imagery of events queried from PRISM using a given minimum precipitation threshold.
    Downloaded samples will contain multispectral data and sar data from within specified interval of event date, 
    their respective NDWI rasters. Samples will also have a raster of roads from TIGER roads dataset, DEM raster from USGS, 
    flowlines and waterbody rasters from NHDPlus dataset. All rasters will be approximately 4km x 4km at 10m resolution.

    Note: In the future if more L2A data become available, some previously processed and skipped events become viable.
    In that case do not use history object during run.
    
    Parameters
    ----------
    threshold : int
    days_before : int
        Number of days of interest before precipitation event.
    days_after : int
        Number of days of interest following precipitation event.
    cloudcoverpercentage : (int, int)
        Desired cloud cover percentage range for querying Copernicus Sentinel-2.
    maxevents: int or None
        Specify a limit to the number of extreme precipitation events to process.
    config_file : str, optional
        Path to custom configuration file.
    manual : str, optional
        Path to text file containing manual event indices in lines with format: time, y, x.
    Returns
    -------
    int
    """
    # Reinitialize config with custom file if provided
    global config
    if config_file:
        config = DataConfig(config_file)
    
    def get_default_dir_path(region, threshold, days_before, days_after, maxcoverpercentage):
        if region is None:
            return f'samples_{threshold}_{days_before}_{days_after}_{maxcoverpercentage}/'
        else:
            return f'samples_{region}_{threshold}_{days_before}_{days_after}_{maxcoverpercentage}/'
    
    # make directory
    if dir_path is None:
        dir_path = get_default_dir_path(region, threshold, days_before, days_after, maxcoverpercentage)
        Path(dir_path).mkdir(parents=True, exist_ok=True)
    else:
        # edge case sanity checks (empty or invalid path strings)
        if not dir_path:
            dir_path = get_default_dir_path(region, threshold, days_before, days_after, maxcoverpercentage)
            print(f"Invalid directory path specified. Defaulting to {dir_path} directory path.", file=sys.stderr)
        else:
            if not Path(dir_path).is_dir():
                Path(dir_path).mkdir(parents=True, exist_ok=True)
            if dir_path[-1] != '/':
                dir_path += '/'
            
    start_time = datetime.now().strftime("%Y-%m-%d_%H-%M")
    
    # root logger
    rootLogger = logging.getLogger('main')
    rootLogger.setLevel(logging.DEBUG)
    fh = logging.FileHandler(dir_path + f'main_{start_time}.log', mode='w')
    fh.setLevel(logging.DEBUG)
    formatter = logging.Formatter('%(asctime)s - %(name)s - %(levelname)s - %(message)s')
    fh.setFormatter(formatter)
    rootLogger.addHandler(fh)

    # event logger
    logger = logging.getLogger('events')
    logger.setLevel(logging.DEBUG)
    fh = logging.FileHandler(dir_path + f'events_{start_time}.log', mode='a')
    fh.setLevel(logging.DEBUG)
    formatter = logging.Formatter('%(asctime)s - %(name)s - %(levelname)s - %(message)s')
    fh.setFormatter(formatter)
    logger.addHandler(fh)

    # log sampling parameters used
    rootLogger.info(
        "Sampling parameters used:\n"
        f"  Threshold: {threshold}\n"
        f"  Days before precipitation event: {days_before}\n"
        f"  Days after precipitation event: {days_after}\n"
        f"  Max cloud/nodata cover percentage: {maxcoverpercentage}\n"
        f"  Max events to sample: {maxevents}\n"
        f"  S1 and S2 coincidence must be within # hours: {within_hours}\n"
        f"  Region: {region}\n"
        f"  Manual indices: {manual}"
    )

    if os.path.isfile(dir_path + 'history.pickle'):
        with open(dir_path + 'history.pickle', "rb") as f:
            history = pickle.load(f)
    else:
        history = set()

    # get PRISM event indices and event data
    if not manual:
        rootLogger.info("Finding candidate extreme precipitation events...")
        num_candidates, events = get_extreme_events(history, threshold=threshold, region=region)
        rootLogger.info(f"Found {num_candidates} candidate extreme precipitation events.")
    else:
        rootLogger.info("Using manual indices...")
        num_candidates, events = get_manual_events(history, manual)
        rootLogger.info(f"Found {num_candidates} events from {manual}.")

    rootLogger.info("Initializing event sampling...")
    count = 0
    alr_completed = 0
    try:
        for event_date, event_precip, minx, miny, maxx, maxy, eid, indices in events:
            if Path(dir_path + eid + '/').is_dir():
                if event_completed(dir_path + eid + '/'):
                    rootLogger.debug(f'Event {eid} has already been processed before. Moving on to the next event...')
                    alr_completed += 1
                    history.add(indices)
                    continue
                else:
                    rootLogger.debug(f'Event {eid} has already been processed before but unsuccessfully. Reprocessing...')

            max_attempts = 3
            for attempt in range(1, max_attempts + 1):
                try:
                    if event_sample_sar(threshold, days_before, days_after, 
                                        maxcoverpercentage, within_hours, event_date, event_precip, 
                                        minx, miny, maxx, maxy, eid, dir_path):
                        count += 1
                    history.add(indices)
                    break
                except (rasterio.errors.WarpOperationError, rasterio.errors.RasterioIOError, pystac_client.exceptions.APIError) as err:
                    rootLogger.error(f"Connection error: {type(err)}")
                    if attempt == max_attempts:
                        rootLogger.error(f'Maximum number of attempts reached, skipping event...')
                    else:
                        rootLogger.info(f'Retrying ({attempt}/{max_attempts})...')
                except NoElevationError as err:
                    rootLogger.error(f'Elevation file missing, skipping event...')
                    history.add(indices)
                    break
                except Exception as err:
                    raise err

            # once sampled maxevents, stop pipeline
            if count >= maxevents:
                rootLogger.info(f"Maximum number of events = {maxevents} reached. Stopping event sampling...")
                break
    except Exception as err:
        rootLogger.error(f"Unexpected error during event sampling: {err}, {type(err)}")
    finally:
        # store all previously processed events
        with open(dir_path + 'history.pickle', 'wb') as f:
            pickle.dump(history, f)

    rootLogger.debug(f"Number of events already completed: {alr_completed}")
    rootLogger.debug(f"Number of successful events sampled from this run: {count}")
    rootLogger.debug(f"Searched through {num_candidates} PRISM indices.")
    return 0

if __name__ == '__main__':
    parser = argparse.ArgumentParser(prog='sampleS2', description='Samples imagery from Copernicus SENTINEL-2 and SENTINEL-1 (through Microsoft Planetary Computer API) for top precipitation events and generates additional accompanying rasters for each event.')
    parser.add_argument('threshold', type=int, help='minimum daily cumulative precipitation (mm) threshold for search')
    parser.add_argument('-b', '--before', dest='days_before', default=2, type=int, help='number of days allowed for download before precipitation event (default: 2)')
    parser.add_argument('-a', '--after', dest='days_after', default=4, type=int, help='number of days allowed for download following precipitation event (default: 4)')
    parser.add_argument('-c', '--maxcover', dest='maxcoverpercentage', default=30, type=int, help='maximum cloud and no data cover percentage (default: 30)')
    parser.add_argument('-s', '--maxevents', dest='maxevents', default=100, type=int, help='maximum number of extreme precipitation events to attempt downloading (default: 100)')
    parser.add_argument('-d', '--dir', dest='dir_path', help='specify a directory name for downloaded samples, format should end with backslash (default: None)')
    parser.add_argument('-w', '--within_hours', dest='within_hours', default=12, type=int, help='Floods event must have S1 and S2 data within this many hours. (default: 12)')
    parser.add_argument('-r', '--region', default=None, choices=['ceser'], help='sample from supported regions: ["ceser"]. (default: None)')
    parser.add_argument('-m', '--manual', default=None, help='text file for parsing manual event indices with format: time, y, x (default: None)')
    parser.add_argument('-f', '--config', dest='config_file', help='specify a custom configuration file path')
    args = parser.parse_args()
    
    sys.exit(main(args.threshold, args.days_before, args.days_after, args.maxcoverpercentage, args.maxevents, dir_path=args.dir_path, within_hours=args.within_hours, region=args.region, config_file=args.config_file, manual=args.manual))
