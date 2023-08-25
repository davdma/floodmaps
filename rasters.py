import requests
from contextlib import ExitStack
from osgeo import gdal, ogr
from rasterio.warp import reproject, Resampling
import richdem as rd
from tensorflow.keras.layers import MaxPooling2D
import tensorflow as tf
from rasterio.features import rasterize
from zipfile import ZipFile
from pathlib import Path
from geopy.geocoders import Nominatim
from datetime import datetime, timedelta, date
import rasterio.merge
import rasterio
import numpy as np
import os
import fiona
import re
import logging
import shutil
from fiona.transform import transform, transform_geom

PRISM_CRS = "EPSG:4269"

def found_after_images(products, event_date):
    """Returns whether products are found in OData query during or after precipitation event date.

    Parameters
    ----------
    products : list[]
        List of dictionaries with each dictionary containing a specific product's catalog information.
    event_date : str
        Formatted as YYYYMMDD.

    Returns
    -------
    bool
        True if query has products that lie on or after event date or False if not.
    """
    dt = datetime.strptime(event_date, "%Y%m%d")
    for product in products:
        datetime_object = datetime.strptime(product['ContentDate']['Start'], "%Y-%m-%dT%H:%M:%S.%fZ")
        if datetime_object >= dt:
            return True

    return False

def download_all(products, dir_path, loggername):
    """Downloads all products in catalog returned by OData API. Note that the function will only 
    be able to download for 60 mins before tokens including the refresh expire, which can lead to 
    errors when downloading goes beyond that time frame.

    Parameters
    ----------
    products : list
        List of products stored in the 'value' key of json obtained from OData catalog API.
    dir_path : str
        Directory path where downloaded files will be stored
    """
    logger = logging.getLogger(loggername + ".rasters")
    # Token expires in 60 mins, do not request token every time!
    data = {
                'grant_type': 'password',
                'username': 'dma@anl.gov',
                'password': 'uR29W9Z@q7*GB9n',
                'client_id': 'cdse-public',
            }

    response = requests.post('https://identity.dataspace.copernicus.eu/auth/realms/CDSE/protocol/openid-connect/token', data=data)
    access_token = response.json()['access_token']
    refresh_token = response.json()['refresh_token']

    logger.info('Acquired access token. Starting download authorization...')
    
    session = requests.Session()
    session.headers.update({'Authorization': f'Bearer {access_token}'})

    # refresh token - can refresh maximum 6 times
    refresh_count = 0
    def auto_refresh(r, *args, **kwargs):
        nonlocal refresh_count
        nonlocal access_token
        if r.status_code == 401 and r.json()['detail'] == 'Expired signature!' and refresh_count < 6:
            logger.info("Refreshing token as the previous access token expired")
            data = {
                'grant_type': 'refresh_token',
                'refresh_token': refresh_token,
                'client_id': 'cdse-public',
            }

            new_response = requests.post('https://identity.dataspace.copernicus.eu/auth/realms/CDSE/protocol/openid-connect/token', data=data)
            access_token = new_response.json()['access_token']
            session.headers.update({"Authorization": f"Bearer {access_token}"})
            r.request.headers["Authorization"] = session.headers["Authorization"]

            refresh_count += 1
            
            # deregister hook to break loop!
            session.hooks['response'] = []
            return session.send(r.request, verify=False)

    Path(dir_path).mkdir(parents=True, exist_ok=True)
    for product in products:
        logger.info(f'Attempting download for product {product["Id"]}...')
    
        if not session.hooks['response']:
            logger.info('Appending new session hook')
            session.hooks['response'].append(auto_refresh)
        
        url = f'https://catalogue.dataspace.copernicus.eu/odata/v1/Products({product["Id"]})/$value'
        response = session.get(url, allow_redirects=False, timeout=180)
        while response.status_code in (301, 302, 303, 307):
            url = response.headers['Location']
            response = session.get(url, allow_redirects=False)

        if response.status_code != 200:
            logger.info(f'Total token refreshes before unexpected status: {refresh_count}')
            raise Exception(f'Unsuccessful request with status code: {response.status_code}, {response.json()}.')

        try:
            file = session.get(url, verify=False, allow_redirects=True)
        except requests.exceptions.ChunkedEncodingError as err:
            # retry
            logger.info(f'Encountered {err}, {type(err)}. Retrying once more...')
            file = session.get(url, verify=False, allow_redirects=True)
        except Exception as err:
            raise err
            
        with open(dir_path + f"{product['Name'][:-5]}.zip", 'wb') as p:
            p.write(file.content)

        logger.info(f'Download completed for {product["Id"]}.')
        logger.info(f'Total token refreshes: {refresh_count}')

def download_Sentinel_2(footprint, event_date, date_interval, dir_path, loggername):
    """Returns None if search yields no desired results or a dictionary with product filenames grouped by date.

    Parameters
    ----------
    footprint : str
        The area of interest formatted as a Well-Known Text string. Must be in ESPG4326.
    event_date : str
        Formatted as YYYYMMDD.
    date_interval : tuple of (str or datetime) or str, optional
        A time interval filter based on the Sensing Start Time of the products.
    dir_path : str
        Path to directory where Sentinel 2 multispectral products will be downloaded.

    Returns
    -------
    dict[str, list] or None
        Product filenames are grouped by their respective dates into a list and inserted into a dictionary using the date string
        as the dictionary key. If no products found or products only found before the event date, then None will be returned.
    """
    logger = logging.getLogger(loggername + ".rasters")
    logger.info('Beginning catalog search...')
    try:
        search = requests.get(f"https://catalogue.dataspace.copernicus.eu/odata/v1/Products?$filter=Collection/Name eq 'SENTINEL-2' and Attributes/OData.CSC.StringAttribute/any(att:att/Name eq 'productType' and att/OData.CSC.StringAttribute/Value eq 'S2MSI2A') and OData.CSC.Intersects(area=geography'SRID=4326;{footprint}') and ContentDate/Start ge {date_interval[0]} and ContentDate/Start le {date_interval[1]}&$top=100",
                            timeout=20).json()
    except Exception as err:
        logger.error(f"Catalog search failed: {err}, {type(err)}")
        return None

    products = [product for product in search['value'] if product['Online']]
    if len(products) == 0:
        logger.info(f'Zero products from query for date interval {date_interval[0]} to {date_interval[1]}.')
        return None
    elif not found_after_images(products, event_date):
        logger.info(f'Products found but only before precipitation event date {event_date}.')
        return None

    # throw exception if download fails here
    logger.info('Products found. Beginning product download...')
    try:
        # provide list of product ids to download in command 
        download_all(products, dir_path, loggername)
    except Exception as err:
        logger.error(f'Attempted download failed: {err}, {type(err)}')
        logger.warning(f'Removing directory for {event_date} at {dir_path}')
        # remove if exist
        if os.path.isdir(dir_path):
            shutil.rmtree(dir_path)
        return None

    logger.info('All product downloads complete.')

    # group filenames by dates in dictionary
    # POTENTIAL ISSUE: could be a problem if multiple tiles are on the edge of different dates if taken at midnight
    products_by_date = dict()
    for product in products:
        dt = datetime.strptime(product['ContentDate']['Start'], "%Y-%m-%dT%H:%M:%S.%fZ").date().strftime("%Y%m%d")
        filename = product['Name'][:-5] + ".zip" # remove .SAFE
        if dt in products_by_date:
            products_by_date[dt].append(filename)
        else:
            products_by_date[dt] = [filename]
    
    return products_by_date

def get_state(lon, lat):
    """Fetches the US state corresponding to the longitude and latitude coordinates.

    Parameters
    ----------
    lon : float
    lat : float

    Returns
    -------
    str or None
        US State or None if not found
    """
    geolocator = Nominatim(user_agent="argonneflood")
    location = geolocator.reverse((lat, lon), exactly_one=True)
    return location.raw['address'].get('state')

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
    delt1 = timedelta(days = days_before)
    delt2 = timedelta(days = days_after)
    start = datetime(int(event_date[0:4]), int(event_date[4:6]), int(event_date[6:8])) - delt1
    end = datetime(int(event_date[0:4]), int(event_date[4:6]), int(event_date[6:8])) + delt2
    return (start.strftime("%Y-%m-%d"), end.strftime("%Y-%m-%d"))

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

def GetHU4Codes(bounds):
    """
    Queries national watershed boundary dataset for HUC 4 codes representing
    hydrologic units that intersect with bounding box.

    Parameters
    ----------
    bounds : (float, float, float, float)
        Tuple in the order minx, miny, maxx, maxy, representing bounding box.

    Returns
    -------
    list[str]
        HU4 codes.
    """
    wbd = gdal.OpenEx('/vsizip/NHD/WBD_National.zip/WBD_National_GDB.gdb')
    wbdhu4 = wbd.GetLayerByIndex(4)
    
    lst = []
    boundpoly = make_box(bounds)
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

# get filepath within zip file
def get_TCI10m_filepath(dir_path, filename):
    """Gets internal TCI 10m resolution filepath of a SAFE zip file."""
    archive = ZipFile(dir_path + filename, 'r')
    files = archive.namelist()
    # pattern match
    p = re.compile(f"{os.path.splitext(filename)[0]}.SAFE/GRANULE/L2A_.*/IMG_DATA/R10m/.*_TCI_10m.jp2")

    for file in files:
        match = p.search(file)
        if match:
            return file
    
    raise Exception("TCI file not found")

def get_B03_B08_filepath(dir_path, filename):
    """Gets internal B03 and B08 band filepaths of a SAFE zip file."""
    archive = ZipFile(dir_path + filename, 'r')
    files = archive.namelist()
    p1 = re.compile(f"{os.path.splitext(filename)[0]}.SAFE/GRANULE/L2A_.*/IMG_DATA/R10m/.*_B03_10m.jp2")
    p2 = re.compile(f"{os.path.splitext(filename)[0]}.SAFE/GRANULE/L2A_.*/IMG_DATA/R10m/.*_B08_10m.jp2")
    b03_path = None
    b08_path = None
    
    for file in files:
        match1 = p1.search(file)
        match2 = p2.search(file)
        if match1 and b03_path is None:
            b03_path = file
        if match2 and b08_path is None:
            b08_path = file
            
    if b03_path is None or b08_path is None:
        raise Exception("files not found")
        
    return b03_path, b08_path

def get_SCL20m_filepath(dir_path, filename):
    """Gets internal SCL 20m resolution filepath of a SAFE zip file."""
    archive = ZipFile(dir_path + filename, 'r')
    files = archive.namelist()
    # pattern match
    p = re.compile(f"{os.path.splitext(filename)[0]}.SAFE/GRANULE/L2A_.*/IMG_DATA/R20m/.*_SCL_20m.jp2")
    for file in files:
        match = p.search(file)
        if match:
            return file
    
    raise Exception("SCL file not found")

def cloud_null_percentage(dir_path, filenames, bounds):
    """Calculates percentage of image covered by cloud or no data pixels.

    Parameters
    ----------
    dir_path : str
        Path for saving generated raster. 
    filenames : list[str]
        List of s2 zip files for stitching together SCL files.
    bounds : (float, float, float, float)
        Tuple in the order minx, miny, maxx, maxy, representing bounding box.

    Returns
    -------
    int
        Percentage of image covered by cloud or no data pixels.
    """
    filepaths = []
    for filename in filenames:
        # stitch together SCL
        filepaths.append(f"zip://{dir_path + filename}!/{get_SCL20m_filepath(dir_path, filename)}")

    with ExitStack() as stack:
        files = [stack.enter_context(rasterio.open(filepath)) for filepath in filepaths]

        # check if all crs is equal before doing merge
        files_crs = {file.crs for file in files}
        if len(files_crs) > 1:
            raise Exception("CRS of all files must match")
        
        img_crs = files[0].crs
        # perform transformation on bounds to match crs
        if bounds is not None:
            conversion = transform(PRISM_CRS, img_crs, (bounds[0], bounds[2]), (bounds[1], bounds[3]))
            bounds = conversion[0][0], conversion[1][0], conversion[0][1], conversion[1][1]

        out_image, _ = rasterio.merge.merge(files, bounds=bounds)
    return int((np.isin(out_image, [0, 8, 9]).sum() / out_image.size) * 100)

def pipeline_S2(dir_path, save_as, filenames, bounds):
    """Generates RGB raster of S2 multispectral file.

    Parameters
    ----------
    dir_path : str
        Path for saving generated raster. 
    save_as : str
        Name of file to be saved. Must have .tif extension.
    filenames : list[str]
        List of s2 files to stitch together.
    bounds : (float, float, float, float)
        Tuple in the order minx, miny, maxx, maxy, representing bounding box.
    zip : bool, optional

    Returns
    -------
    shape : (int, int)
        Shape of the raster array.
    crs : str
        Coordinate reference system for the raster.
    transform : rasterio.affine.Affine()
        Transformation matrix for mapping pixel coordinates in dest to coordinate system.
    """
    filepaths = []
    for filename in filenames:
        filepaths.append(f"zip://{dir_path + filename}!/{get_TCI10m_filepath(dir_path, filename)}")

    with ExitStack() as stack:
        files = [stack.enter_context(rasterio.open(filepath)) for filepath in filepaths]

        # check if all crs is equal before doing merge
        files_crs = {file.crs for file in files}
        if len(files_crs) > 1:
            raise Exception("CRS of all files must match")

        img_crs = files[0].crs

        # perform transformation on bounds to match crs
        conversion = transform(PRISM_CRS, img_crs, (bounds[0], bounds[2]), (bounds[1], bounds[3]))
        cbounds = conversion[0][0], conversion[1][0], conversion[0][1], conversion[1][1]

        # No data value for TCI is 0
        out_image, out_transform = rasterio.merge.merge(files, bounds=cbounds, nodata=0)

        with rasterio.open(dir_path + save_as, 'w', driver='Gtiff', count=3, height=out_image.shape[-2], width=out_image.shape[-1], crs=img_crs, dtype=out_image.dtype, 
                           transform=out_transform, nodata=0) as dst:
            dst.write(out_image)

    return (out_image.shape[-2], out_image.shape[-1]), img_crs, out_transform

def pipeline_NDWI(dir_path, save_as, filenames, bounds):
    """Generates NDWI raster from S2 multispectral files.

    Parameters
    ----------
    dir_path : str
        Path for saving generated raster. 
    save_as : str
        Name of file to be saved. Must have .tif extension.
    filenames : list[str]
        List of s2 files to stitch together.
    bounds : (float, float, float, float)
        Tuple in the order minx, miny, maxx, maxy, representing bounding box.
    zip : bool, optional
    """
    b03_paths = []
    b08_paths = []

    for filename in filenames:
        b03_path, b08_path = get_B03_B08_filepath(dir_path, filename)
        b03_paths.append(f"zip://{dir_path + filename}!/{b03_path}")
        b08_paths.append(f"zip://{dir_path + filename}!/{b08_path}")

    with ExitStack() as stack:
        files = [stack.enter_context(rasterio.open(filepath)) for filepath in b03_paths]

        files_crs = {file.crs for file in files}
        if len(files_crs) > 1:
            raise Exception("CRS of all files must match")

        img_crs = files[0].crs
        
        conversion = transform(PRISM_CRS, img_crs, (bounds[0], bounds[2]), (bounds[1], bounds[3]))
        cbounds = conversion[0][0], conversion[1][0], conversion[0][1], conversion[1][1]
        
        out_image1, _ = rasterio.merge.merge(files, bounds=cbounds, nodata=0)
        green = out_image1[0]

    with ExitStack() as stack:
        files = [stack.enter_context(rasterio.open(filepath)) for filepath in b08_paths]

        files_crs = files_crs.union({file.crs for file in files})
        if len(files_crs) > 1:
            raise Exception("CRS of all files must match")

        img_crs = files[0].crs
        
        conversion = transform(PRISM_CRS, img_crs, (bounds[0], bounds[2]), (bounds[1], bounds[3]))
        cbounds = conversion[0][0], conversion[1][0], conversion[0][1], conversion[1][1]
        
        out_image2, out_transform = rasterio.merge.merge(files, bounds=cbounds, nodata=0)
        nir = out_image2[0]

    checker = green + nir
    ndwi = np.empty(green.shape)
    ndwi[:] = -999999
    np.divide(green - nir, green + nir, out = ndwi, where = checker != 0)
    with rasterio.open(dir_path + save_as, 'w', driver='Gtiff', count=1, height=ndwi.shape[-2], width=ndwi.shape[-1], crs=img_crs, dtype=ndwi.dtype, transform=out_transform, nodata=-999999) as dst:
        dst.write(ndwi, 1)

def pipeline_roads(dir_path, save_as, dst_shape, dst_crs, dst_transform, state, buffer=0):
    """Generates raster with burned in geometries of roads given destination raster properties.

    Parameters
    ----------
    dir_path : str
        Path for saving generated raster. 
    save_as : str
        Name of file to be saved. Must have .tif extension.
    dst_shape : (int, int)
        Shape of output raster.
    dst_crs : str
        Coordinate reference system of output raster.
    dst_transform : rasterio.affine.Affine()
        Transformation matrix for mapping pixel coordinates to coordinate system of output raster.
    state : str
        State of raster location.
    buffer : int, optional
        Buffer the geometry line thickness by certain number of pixels.
    """
    # find state shape file
    with fiona.open(f'Roads/{state.strip().upper()}.shp', "r") as shapefile:
        shapes = [transform_geom(shapefile.crs, dst_crs, feature["geometry"]) for feature in shapefile]
        
    rasterize_roads = rasterize(
        [(line, 1) for line in shapes],
        out_shape=dst_shape,
        transform=dst_transform,
        fill=0,
        all_touched=True)

    if buffer > 0:
        rasterize_roads = buffer_raster(rasterize_roads, buffer)

    with rasterio.open(dir_path + save_as, 'w', driver='Gtiff', count=1, height=rasterize_roads.shape[-2], width=rasterize_roads.shape[-1], 
                       crs=dst_crs, dtype=rasterize_roads.dtype, transform=dst_transform, nodata=0) as dst:
        dst.write(rasterize_roads, 1)

def pipeline_dem_slope(dir_path, save_as, dst_shape, dst_crs, dst_transform, bounds):
    """Generates Digital Elevation Map raster given destination raster properties and bounding box.

    Parameters
    ----------
    dir_path : str
        Path for saving generated raster. 
    save_as : (str, str)
        Tuple of names of dem and slope files to be saved. Must have .tif extension. First string will be for dem raster, second
        string will be for slope raster.
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
    
    dir_names = os.listdir('Elevation/')
    lst = []
    p = re.compile(f"n(\d*)w(\d*).tif")
    for file in dir_names:
        m = p.match(file)
        # we want the range of the lats and the range of the longs to intersect the bounds
        if m:
            # nxxwxx is TOP LEFT corner of tile!
            tile_bounds = (int(m.group(2)) * (-1), int(m.group(1)) - 1, int(m.group(2)) * (-1) + 1, int(m.group(1)))
            if tile_bounds[0] < bounds[2] and tile_bounds[2] > bounds[0] and tile_bounds[1] < bounds[3] and tile_bounds[3] > bounds[1]:
                lst.append('Elevation/' + file)

    with ExitStack() as stack:
        files = [stack.enter_context(rasterio.open(filepath)) for filepath in lst]
        
        # check if all crs is equal before doing merge
        files_crs = {file.crs for file in files}
        files_nodata = {file.nodata for file in files}
        if len(files_crs) > 1:
            raise Exception("CRS of all files must match")
        elif len(files_nodata) > 1:
            raise Exception("No data values of all files must match")
        
        src_crs = files[0].crs
        no_data = files[0].nodata
        
        src_image, src_transform = rasterio.merge.merge(files, bounds=bounds, nodata=no_data)
        destination = np.zeros(dst_shape, src_image.dtype)

        reproject(
            src_image,
            destination,
            src_transform=src_transform,
            src_crs=src_crs,
            dst_transform=dst_transform,
            dst_crs=dst_crs,
            resampling=Resampling.bilinear)
        
        with rasterio.open(dir_path + save_as[0], 'w', driver='Gtiff', count=1, height=destination.shape[-2], width=destination.shape[-1], 
                           crs=dst_crs, dtype=destination.dtype, transform=dst_transform, nodata=no_data) as dst:
            dst.write(destination, 1)

        rda = rd.rdarray(destination, no_data=no_data)
        slope = rd.TerrainAttribute(rda, attrib='slope_riserun')
        nprda = np.array(slope)

        with rasterio.open(dir_path + save_as[1], 'w', driver='Gtiff', count=1, height=nprda.shape[-2], width=nprda.shape[-1], 
                           crs=dst_crs, dtype=nprda.dtype, transform=dst_transform, nodata=slope.no_data) as dst:
            dst.write(nprda, 1)

def pipeline_flowlines(dir_path, save_as, dst_shape, dst_crs, dst_transform, bounds, buffer=3, filter=['460\d{2}', '558\d{2}', '336\d{2}', '334\d{2}']):
    """Generates raster with burned in geometries of flowlines given destination raster properties.

    Parameters
    ----------
    dir_path : str
        Path for saving generated raster. 
    save_as : str
        Name of file to be saved. Must have .tif extension.
    dst_shape : (int, int)
        Shape of output raster.
    dst_crs : str
        Coordinate reference system of output raster.
    dst_transform : rasterio.affine.Affine()
        Transformation matrix for mapping pixel coordinates to coordinate system of output raster.
    bounds : (float, float, float, float)
        Tuple in the order minx, miny, maxx, maxy, representing bounding box.
    buffer : int, optional
        Buffer the geometry line thickness by certain number of pixels.
    filter : list[str], optional
        List of regex patterns for filtering specific flowline features based on their NHDPlus dataset FCodes.
    """
    with ExitStack() as stack:
        files = [stack.enter_context(fiona.open(f'/vsizip/NHD/NHDPLUS_H_{code}_HU4_GDB.zip/NHDPLUS_H_{code}_HU4_GDB.gdb', layer='NHDFlowline')) for code in GetHU4Codes(bounds)]

        files_crs = {file.crs for file in files}
        if len(files_crs) > 1:
            raise Exception("CRS of all files must match")
        src_crs = files[0].crs
        
        shapes = []
        p = re.compile('|'.join(filter))
        for i, lyr in enumerate(files):
            for _, feat in lyr.items(bbox=bounds):
                m = p.match(str(feat.properties['FCode']))
                if m:
                    shapes.append((transform_geom(src_crs, dst_crs, feat.geometry), 1))
        
        flowlines = rasterize(
            shapes,
            out_shape=dst_shape,
            transform=dst_transform,
            fill=0,
            all_touched=True)

        if buffer > 0:
            flowlines = buffer_raster(flowlines, buffer)

    with rasterio.open(dir_path + save_as, 'w', driver='Gtiff', count=1, height=flowlines.shape[-2], width=flowlines.shape[-1], 
                       crs=dst_crs, dtype=flowlines.dtype, transform=dst_transform, nodata=0) as dst:
        dst.write(flowlines, 1)

def pipeline_waterbody(dir_path, save_as, dst_shape, dst_crs, dst_transform, bounds):
    """Generates raster with burned in geometries of flowlines given destination raster properties.

    Parameters
    ----------
    dir_path : str
        Path for saving generated raster. 
    save_as : str
        Name of file to be saved. Must have .tif extension.
    dst_shape : (int, int)
        Shape of output raster.
    dst_crs : str
        Coordinate reference system of output raster.
    dst_transform : rasterio.affine.Affine()
        Transformation matrix for mapping pixel coordinates to coordinate system of output raster.
    bounds : (float, float, float, float)
        Tuple in the order minx, miny, maxx, maxy, representing bounding box.
    buffer : int, optional
        Buffer the geometry line thickness by certain number of pixels.
    """
    with ExitStack() as stack:
        files = [stack.enter_context(fiona.open(f'/vsizip/NHD/NHDPLUS_H_{code}_HU4_GDB.zip/NHDPLUS_H_{code}_HU4_GDB.gdb', layer='NHDWaterbody')) for code in GetHU4Codes(bounds)]

        files_crs = {file.crs for file in files}
        if len(files_crs) > 1:
            raise Exception("CRS of all files must match")
        src_crs = files[0].crs
        
        shapes = []
        for lyr in files:
            for _, feat in lyr.items(bbox=bounds):
                shapes.append((transform_geom(src_crs, dst_crs, feat.geometry), 1))
        
        waterbody = rasterize(
            shapes,
            out_shape=dst_shape,
            transform=dst_transform,
            fill=0,
            all_touched=True)

    with rasterio.open(dir_path + save_as, 'w', driver='Gtiff', count=1, height=waterbody.shape[-2], width=waterbody.shape[-1], 
                       crs=dst_crs, dtype=waterbody.dtype, transform=dst_transform, nodata=0) as dst:
        dst.write(waterbody, 1)