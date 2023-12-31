import requests
from contextlib import ExitStack
from osgeo import gdal, ogr
from rasterio.warp import calculate_default_transform, reproject, Resampling
import richdem as rd
from rasterio.features import rasterize
from zipfile import ZipFile
from pathlib import Path
from geopy.geocoders import Nominatim
from datetime import datetime, timedelta, date
import rasterio.merge
import rasterio
import numpy as np
import fiona
import re
import logging
import shutil
import json
import os
import sys
import time
from fiona.transform import transform, transform_geom
import sample_exceptions

os.environ['TF_CPP_MIN_LOG_LEVEL'] = '3'
from tensorflow.keras.layers import MaxPooling2D
import tensorflow as tf

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

def download_all(products, dir_path, loggername, max_retries=3):
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
    auth_data = {
                'grant_type': 'password',
                'username': 'dma@anl.gov',
                'password': 'uR29W9Z@q7*GB9n',
                'client_id': 'cdse-public',
                }

    response = requests.post('https://identity.dataspace.copernicus.eu/auth/realms/CDSE/protocol/openid-connect/token', data=auth_data)
    access_token = response.json()['access_token']
    refresh_token = response.json()['refresh_token']

    logger.info('Acquired access token. Starting download authorization...')
    
    session = requests.Session()
    session.headers.update({'Authorization': f'Bearer {access_token}'})

    # if refresh token expires, must reauthenticate!
    def auto_refresh(r, *args, **kwargs):
        nonlocal access_token
        nonlocal refresh_token
        nonlocal auth_data
        if r.status_code == 401 and r.json()['detail'] == 'Expired signature!':
            logger.info("Refreshing token as the previous access token expired")
            refresh_data = {
                'grant_type': 'refresh_token',
                'refresh_token': refresh_token,
                'client_id': 'cdse-public',
            }

            try:
                new_response = requests.post('https://identity.dataspace.copernicus.eu/auth/realms/CDSE/protocol/openid-connect/token', data=refresh_data)
                if new_response.status_code == 400 and 'error' in new_response.json() and new_response.json()['error'] == 'invalid_grant':
                    # reauth if refresh token expires also
                    logger.info("Reauthorizing for new access and refresh token.")
                    new_response = requests.post('https://identity.dataspace.copernicus.eu/auth/realms/CDSE/protocol/openid-connect/token', data=auth_data)
                access_token = new_response.json()['access_token']
                refresh_token = new_response.json()['refresh_token']
            except KeyError as err:
                logger.error(f'Unexpected json: {new_response.json()}')
            except Exception as err:
                raise err
                
            session.headers.update({"Authorization": f"Bearer {access_token}"})
            r.request.headers["Authorization"] = session.headers["Authorization"]
            
            # deregister hook to break loop!
            session.hooks['response'] = []
            return session.send(r.request, verify=False)

    def download(product, session, dir_path):
        logger.info(f'Downloading product {product["Name"]} with product id {product["Id"]}...')
        url = f'https://catalogue.dataspace.copernicus.eu/odata/v1/Products({product["Id"]})/$value'
        
        response = session.get(url, allow_redirects=False)
        while response.status_code in (301, 302, 303, 307):
            url = response.headers['Location']
            response = session.get(url, allow_redirects=False)
    
        file = session.get(url, verify=False, allow_redirects=True)
        if file.status_code != 200:
            # Do not download if credentials have expired - may need to reauth to retry
            raise sample_exceptions.BadStatusError(f'Unsuccessful request with status code {file.status_code}')
            
        with open(dir_path + f"{product['Name'][:-5]}.zip", 'wb') as p:
            p.write(file.content)

    Path(dir_path).mkdir(parents=True, exist_ok=True)
    for product in products:
        logger.info(f'Attempting download for product {product["Name"]} with product id {product["Id"]}...')
        # check if product already downloaded previously
        if os.path.isfile(dir_path + f"{product['Name'][:-5]}.zip") :
            logger.info(f'Product {product["Name"]} with product id {product["Id"]} already downloaded. Skipping download...')
            continue
        
        retries = 0
        while retries < max_retries:
            try:
                if not session.hooks['response']:
                    logger.info('Appending new session hook')
                    session.hooks['response'].append(auto_refresh)
                    
                download(product, session, dir_path)
                break
            except (requests.exceptions.ChunkedEncodingError, sample_exceptions.BadStatusError) as err:
                logger.info(f'Encountered {err}, {type(err)}. Retrying once more...')
                retries += 1
            except Exception as err:
                logger.info(f'Encountered unexpected {err}, {type(err)}. Not handled, cancelling operations.')
                raise err

        if retries == max_retries:
            raise Exception(f'Max retries reached for {product["Name"]} with product id {product["Id"]}.')

        logger.info(f'Download completed for {product["Id"]}.')

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
        response = requests.get(f"https://catalogue.dataspace.copernicus.eu/odata/v1/Products?$filter=Collection/Name eq 'SENTINEL-2' and Attributes/OData.CSC.StringAttribute/any(att:att/Name eq 'productType' and att/OData.CSC.StringAttribute/Value eq 'S2MSI2A') and OData.CSC.Intersects(area=geography'SRID=4326;{footprint}') and ContentDate/Start ge {date_interval[0]} and ContentDate/Start le {date_interval[1]}&$top=100",
                            timeout=20)
        search = response.json()
    except requests.exceptions.JSONDecodeError as err:
        logger.error(f"The response does not contain valid JSON: {err}, {type(err)}")
        logger.debug(f"Retrying once...")
        try:
            response = requests.get(f"https://catalogue.dataspace.copernicus.eu/odata/v1/Products?$filter=Collection/Name eq 'SENTINEL-2' and Attributes/OData.CSC.StringAttribute/any(att:att/Name eq 'productType' and att/OData.CSC.StringAttribute/Value eq 'S2MSI2A') and OData.CSC.Intersects(area=geography'SRID=4326;{footprint}') and ContentDate/Start ge {date_interval[0]} and ContentDate/Start le {date_interval[1]}&$top=100", timeout=20)
            search = response.json()
        except Exception as err:
            logger.error(f"Retry failed: {err}, {type(err)}")
            return None
    except requests.exceptions.RequestException as err:
        logger.error(f"Requests error: {err}, {type(err)}")
        return None
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

def get_B08_filepath(dir_path, filename):
    """Gets internal B08 band filepath of a SAFE zip file."""
    archive = ZipFile(dir_path + filename, 'r')
    files = archive.namelist()
    p = re.compile(f"{os.path.splitext(filename)[0]}.SAFE/GRANULE/L2A_.*/IMG_DATA/R10m/.*_B08_10m.jp2")
    b08_path = None
    
    for file in files:
        match = p.search(file)
        if match:
            b08_path = file
            break
            
    if b08_path is None:
        raise Exception("B8 band file not found")
        
    return b08_path

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

def get_TCI10m_crs(dir_path, filename):
    with rasterio.open(f"zip://{dir_path + filename}!/{get_TCI10m_filepath(dir_path, filename)}") as src:
        src_crs = src.crs
    
    return src_crs

def convert_raster_crs(filepath, new_filepath, dst_crs, resampling=Resampling.nearest):
    # for SCL we use nearest resampling - otherwise use bilinear
    with rasterio.open(filepath) as src:
        transform, width, height = calculate_default_transform(
            src.crs, dst_crs, src.width, src.height, *src.bounds)
        kwargs = src.meta.copy()
        kwargs.update({
            'crs': dst_crs,
            'transform': transform,
            'width': width,
            'height': height
        })

        with rasterio.open(new_filepath, 'w', **kwargs) as dst:
            for i in range(1, src.count + 1):
                reproject(
                    source=rasterio.band(src, i),
                    destination=rasterio.band(dst, i),
                    src_transform=src.transform,
                    src_crs=src.crs,
                    dst_transform=transform,
                    dst_crs=dst_crs,
                    resampling=resampling)

def cloud_null_percentage_dep(dir_path, filenames, bounds):
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
    # For cloud null percentage, it could make more sense to not do any merging!
    filepaths = []
    for filename in filenames:
        # stitch together SCL
        filepaths.append(f"zip://{dir_path + filename}!/{get_SCL20m_filepath(dir_path, filename)}")

    with ExitStack() as stack:
        files = [stack.enter_context(rasterio.open(filepath)) for filepath in filepaths]

        # check if all crs is equal before doing merge
        files_crs = {file.crs.to_string() for file in files}
        if len(files_crs) > 1:
            raise Exception(f"CRS of all files must match: {', '.join(files_crs)}. Filenames: {', '.join(filenames)}.")
        
        img_crs = files[0].crs
        # perform transformation on bounds to match crs
        if bounds is not None:
            conversion = transform(PRISM_CRS, img_crs, (bounds[0], bounds[2]), (bounds[1], bounds[3]))
            bounds = conversion[0][0], conversion[1][0], conversion[0][1], conversion[1][1]

        out_image, _ = rasterio.merge.merge(files, bounds=bounds)
    return int((np.isin(out_image, [0, 8, 9]).sum() / out_image.size) * 100)

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
    files_crs = []
    filepaths = []
    for filename in filenames:
        # stitch together SCL
        filepath = f"zip://{dir_path + filename}!/{get_SCL20m_filepath(dir_path, filename)}"
        filepaths.append(filepath)

        with rasterio.open(filepath) as src:
            files_crs.append(src.crs.to_string())

    if len(files_crs) == 0:
        raise Exception(f"No CRS found for files: {', '.join(filenames)}")
        
    img_crs = files_crs[0]
    if bounds is not None:
        conversion = transform(PRISM_CRS, img_crs, (bounds[0], bounds[2]), (bounds[1], bounds[3]))
        bounds = conversion[0][0], conversion[1][0], conversion[0][1], conversion[1][1]
        
    if all(item == files_crs[0] for item in files_crs):
        # perform transformation on bounds to match crs
        out_image, _ = rasterio.merge.merge(filepaths, bounds=bounds)
    else:
        new_filepaths = []
        for i, file in enumerate(filepaths[1:]):
            # write new files, then update stack context
            new_filepath = dir_path + f'scl_tmp_{i}.tif'
            convert_raster_crs(file, new_filepath, img_crs)
            new_filepaths.append(new_filepath)

        filepaths = [filepaths[0]] + new_filepaths
        out_image, _ = rasterio.merge.merge(filepaths, bounds=bounds)
        
        # then remove new tmp files when done
        for file in new_filepaths:
            os.remove(file)

    return int((np.isin(out_image, [0, 8, 9]).sum() / out_image.size) * 100)

def pipeline_S2_dep(dir_path, save_as, filenames, bounds):
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

# we will choose a UTM zone CRS already given and stick to it for rest of sample data!
def pipeline_S2(dir_path, save_as, dst_crs, filenames, bounds):
    """Generates RGB raster of S2 multispectral file.

    Parameters
    ----------
    dir_path : str
        Path for saving generated raster. 
    save_as : str
        Name of file to be saved. Must have .tif extension.
    dst_crs : obj
        Coordinate reference system of output raster.
    filenames : list[str]
        List of s2 files to stitch together.
    bounds : (float, float, float, float)
        Tuple in the order minx, miny, maxx, maxy, representing bounding box.
    zip : bool, optional

    Returns
    -------
    shape : (int, int)
        Shape of the raster array.
    transform : rasterio.affine.Affine()
        Transformation matrix for mapping pixel coordinates in dest to coordinate system.
    """
    # iterate through files - if a file does not have same crs as dst_crs, then must reproject and keep track
    filepaths = []
    for filename in filenames:
        filepath = f"zip://{dir_path + filename}!/{get_TCI10m_filepath(dir_path, filename)}"
        filepaths.append(filepath)

    final_filepaths = []
    new_filepaths = []
    for i, filepath in enumerate(filepaths):
        with rasterio.open(filepath) as src:
            # check to see if using equality operator works, or can just check string equality
            if src.crs == dst_crs:
                final_filepaths.append(filepath)
            else:
                new_filepath = dir_path + f's2_tmp_{i}.tif'
                convert_raster_crs(filepath, new_filepath, dst_crs, resampling=Resampling.bilinear)
                new_filepaths.append(new_filepath)

    out_image, out_transform = rasterio.merge.merge(final_filepaths + new_filepaths, bounds=bounds, nodata=0)
    for filepath in new_filepaths:
        os.remove(filepath)

    with rasterio.open(dir_path + save_as, 'w', driver='Gtiff', count=3, height=out_image.shape[-2], width=out_image.shape[-1], crs=dst_crs, dtype=out_image.dtype, transform=out_transform, nodata=0) as dst:
            dst.write(out_image)

    return (out_image.shape[-2], out_image.shape[-1]), out_transform
    
def pipeline_B08_dep(dir_path, save_as, filenames, bounds):
    """Generates NIR B8 band raster of S2 multispectral file.

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
    filepaths = []
    for filename in filenames:
        filepaths.append(f"zip://{dir_path + filename}!/{get_B08_filepath(dir_path, filename)}")

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

        with rasterio.open(dir_path + save_as, 'w', driver='Gtiff', count=1, height=out_image.shape[-2], width=out_image.shape[-1], crs=img_crs, dtype=out_image.dtype, 
                           transform=out_transform, nodata=0) as dst:
            dst.write(out_image)

def pipeline_B08(dir_path, save_as, dst_crs, filenames, bounds):
    """Generates NIR B8 band raster of S2 multispectral file.

    Parameters
    ----------
    dir_path : str
        Path for saving generated raster. 
    save_as : str
        Name of file to be saved. Must have .tif extension.
    dst_crs : obj
        Coordinate reference system of output raster.
    filenames : list[str]
        List of s2 files to stitch together.
    bounds : (float, float, float, float)
        Tuple in the order minx, miny, maxx, maxy, representing bounding box.
    zip : bool, optional
    """
    # iterate through files - if a file does not have same crs as dst_crs, then must reproject and keep track
    filepaths = []
    for filename in filenames:
        filepath = f"zip://{dir_path + filename}!/{get_B08_filepath(dir_path, filename)}"
        filepaths.append(filepath)

    final_filepaths = []
    new_filepaths = []
    for i, filepath in enumerate(filepaths):
        with rasterio.open(filepath) as src:
            # check to see if using equality operator works, or can just check string equality
            if src.crs == dst_crs:
                final_filepaths.append(filepath)
            else:
                new_filepath = dir_path + f'b08_tmp_{i}.tif'
                convert_raster_crs(filepath, new_filepath, dst_crs, resampling=Resampling.bilinear)
                new_filepaths.append(new_filepath)

    # No data value for TCI is 0
    out_image, out_transform = rasterio.merge.merge(final_filepaths + new_filepaths, bounds=bounds, nodata=0)
    for filepath in new_filepaths:
        os.remove(filepath)

    with rasterio.open(dir_path + save_as, 'w', driver='Gtiff', count=1, height=out_image.shape[-2], width=out_image.shape[-1], crs=dst_crs, dtype=out_image.dtype, transform=out_transform, nodata=0) as dst:
        dst.write(out_image)

def pipeline_NDWI_dep(dir_path, save_as, filenames, bounds):
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

def pipeline_NDWI(dir_path, save_as, dst_crs, filenames, bounds):
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
    b03_filepaths = []
    b08_filepaths = []

    for filename in filenames:
        b03_path, b08_path = get_B03_B08_filepath(dir_path, filename)
        b03_filepaths.append(f"zip://{dir_path + filename}!/{b03_path}")
        b08_filepaths.append(f"zip://{dir_path + filename}!/{b08_path}")
    
    # get b03 raster
    b03_final_filepaths = []
    b03_new_filepaths = []
    for i, filepath in enumerate(b03_filepaths):
        with rasterio.open(filepath) as src:
            # check to see if using equality operator works, or can just check string equality
            if src.crs == dst_crs:
                b03_final_filepaths.append(filepath)
            else:
                b03_new_filepath = dir_path + f'ndwi_b03_tmp_{i}.tif'
                convert_raster_crs(filepath, b03_new_filepath, dst_crs, resampling=Resampling.bilinear)
                b03_new_filepaths.append(b03_new_filepath)

    out_image1, _ = rasterio.merge.merge(b03_final_filepaths + b03_new_filepaths, bounds=bounds, nodata=0)
    green = out_image1[0]
    for filepath in b03_new_filepaths:
        os.remove(filepath)

    # get b08 raster
    b08_final_filepaths = []
    b08_new_filepaths = []
    for i, filepath in enumerate(b08_filepaths):
        with rasterio.open(filepath) as src:
            # check to see if using equality operator works, or can just check string equality
            if src.crs == dst_crs:
                b08_final_filepaths.append(filepath)
            else:
                b08_new_filepath = dir_path + f'ndwi_b08_tmp_{i}.tif'
                convert_raster_crs(filepath, b08_new_filepath, dst_crs, resampling=Resampling.bilinear)
                b08_new_filepaths.append(b08_new_filepath)

    out_image2, out_transform = rasterio.merge.merge(b08_final_filepaths + b08_new_filepaths, bounds=bounds, nodata=0)
    nir = out_image2[0]
    for filepath in b08_new_filepaths:
        os.remove(filepath)

    # calculate ndwi
    checker = green + nir
    ndwi = np.empty(green.shape)
    ndwi[:] = -999999
    np.divide(green - nir, green + nir, out = ndwi, where = checker != 0)
    with rasterio.open(dir_path + save_as, 'w', driver='Gtiff', count=1, height=ndwi.shape[-2], width=ndwi.shape[-1], crs=dst_crs, dtype=ndwi.dtype, transform=out_transform, nodata=-999999) as dst:
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
    with fiona.open(f'Roads/{state.strip().upper()}.shp', "r") as shapefile:
        shapes = [transform_geom(shapefile.crs, dst_crs, feature["geometry"]) for feature in shapefile]

    if shapes:
        rasterize_roads = rasterize(
            [(line, 1) for line in shapes],
            out_shape=dst_shape,
            transform=dst_transform,
            fill=0,
            all_touched=True)
    else:
        # if no shapes to rasterize
        rasterize_roads = np.zeros(dst_shape, dtype=int)
        

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

    # if no dem tile can be found, raise error
    if not lst:
        raise Exception(f'No elevation raster can be found for bounds ({bounds[0]}, {bounds[1]}, {bounds[2]}, {bounds[3]}) during generation of dem, slope raster.')
    
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

        if shapes:
            flowlines = rasterize(
                shapes,
                out_shape=dst_shape,
                transform=dst_transform,
                fill=0,
                all_touched=True)
        else:
            # if no shapes to rasterize
            flowlines = np.zeros(dst_shape, dtype=int)

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
                all_touched=True)
        else:
            # if no shapes to rasterize
            waterbody = np.zeros(dst_shape, dtype=int)

    with rasterio.open(dir_path + save_as, 'w', driver='Gtiff', count=1, height=waterbody.shape[-2], width=waterbody.shape[-1], 
                       crs=dst_crs, dtype=waterbody.dtype, transform=dst_transform, nodata=0) as dst:
        dst.write(waterbody, 1)