import sys
import argparse
from datetime import datetime, timedelta, date
from tqdm import tqdm
from netCDF4 import Dataset
from cftime import num2date
from logging import basicConfig
from pathlib import Path
import logging
import numpy as np
import pandas as pd
import os
import re
from glob import glob
import json
import pickle
import sample_exceptions
from fiona.transform import transform, transform_geom

PRISM_CRS = "EPSG:4269"

def read_PRISM():
    """Reads the PRISM netCDF file and return the encoded data."""
    with Dataset("PRISM/prismprecipnew.nc", "r", format="NETCDF4") as nc:
        geotransform = nc["geotransform"][:]
        time_info = (nc["time"].units, nc["time"].calendar)
        precip_data = nc["precip"][:]

    return (geotransform, time_info, precip_data)

def general_query_PRISM(prism, threshold=300, n=None):
    """
    Queries contents of PRISM netCDF file given the return value of read_PRISM(). Filters for 
    precipitation events that meet minimum threshold of precipitation.

    Parameters
    ----------
    prism : tuple returned by read_PRISM()
    threshold : int, optional
        Minimum cumulative daily precipitation in mm for filtering PRISM tiles.
    n : int, optional
        Selects only first n events that meet threshold criteria.

    Returns
    -------
    Dataframe
        pandas dataframe containing events labeled with date, cumulative day precipitation in mm, latitude longitude 
        bounding box values and a unique event id.
    """
    geotransform, time_info, precip_data = prism
    events = np.where(precip_data > threshold) #lat indices, lon indices, and time indices for target events

    upper_left_x, x_size, x_rotation, upper_left_y, y_rotation, y_size = geotransform
    event_dates = []
    event_precip = []
    minx = []
    miny = []
    maxx = []
    maxy = []
    eid = []

    min_date = datetime(2015, 7, 1)
    count = 0
    for time, y, x in zip(events[0], events[1], events[2]):
        if n is not None and count >= n:
            break
            
        event_date = num2date(time, units=time_info[0], calendar=time_info[1])

        # must not be earlier than s2 launch
        if event_date < min_date:
            continue

        event_date_str = event_date.strftime("%Y%m%d")
        event_dates.append(event_date_str)
        event_precip.append(precip_data[time, y, x])
        
        # convert latitude and longitude to
        # (minx, miny, maxx, maxy) using the equation
        minx.append(x * x_size + upper_left_x)
        miny.append((y + 1) * y_size + upper_left_y)
        maxx.append((x + 1) * x_size + upper_left_x)
        maxy.append(y * y_size + upper_left_y)
        eid.append(f'{threshold}_{event_date_str}_{y}_{x}')
        count += 1

    df = pd.DataFrame({"Date": event_dates,
                       "Precipitation (mm)": event_precip,
                       "minx": minx,
                       "miny": miny,
                       "maxx": maxx,
                       "maxy": maxy,
                       "eid": eid})

    return df

def general_query_filter_PRISM(prism, days_before, days_after, history, threshold=300, n=None, cloudcover=80, max_retries=3):
    """
    Queries contents of PRISM netCDF file given the return value of read_PRISM(). Filters for 
    precipitation events that meet minimum threshold of precipitation.

    Parameters
    ----------
    prism : tuple returned by read_PRISM()
    days_before : int
    days_after : int
    history : set()
        Set that holds all eids of previously processed events.
    threshold : int, optional
        Minimum cumulative daily precipitation in mm for filtering PRISM tiles.
    n : int, optional
        Selects only first n events that meet threshold criteria.

    Returns
    -------
    Dataframe
        pandas dataframe containing events labeled with date, cumulative day precipitation in mm, latitude longitude 
        bounding box values and a unique event id.
    """
    import rasters
    import shapely as sp
    from fiona.transform import transform
    import requests
    PRISM_CRS = "EPSG:4269"
    SEARCH_CRS = "EPSG:4326"

    rootLogger = logging.getLogger('main')
    geotransform, time_info, precip_data = prism
    events = np.where(precip_data > threshold) #lat indices, lon indices, and time indices for target events

    upper_left_x, x_size, x_rotation, upper_left_y, y_rotation, y_size = geotransform
    event_dates = []
    event_precip = []
    minx = []
    miny = []
    maxx = []
    maxy = []
    eid = []

    min_date = datetime(2015, 7, 1)
    count = 0
    if n is not None:
        pbar = tqdm(total = n)

    error_count = {'JSONDecodeError': 0, 'RequestsError': 0, 'GeneralError': 0}
    rootLogger.debug('Catalog search beginning...')
    for time, y, x in zip(events[0], events[1], events[2]):
        if n is not None and count >= n:
            break
            
        event_date = num2date(time, units=time_info[0], calendar=time_info[1])
        event_date_str = event_date.strftime("%Y%m%d")
        
        # must not be earlier than s2 launch
        if event_date < min_date:
            continue
        elif f'{event_date_str}_{y}_{x}' in history:
            continue

        # filter out events unavailable on copernicus
        date_interval = rasters.get_date_interval(event_date_str, days_before, days_after)
        c_minx = x * x_size + upper_left_x
        c_miny = (y + 1) * y_size + upper_left_y
        c_maxx = (x + 1) * x_size + upper_left_x
        c_maxy = y * y_size + upper_left_y
        conversion = transform(PRISM_CRS, SEARCH_CRS, (c_minx, c_maxx), (c_miny, c_maxy))
        box = sp.to_wkt(sp.geometry.box(conversion[0][0], conversion[1][0], conversion[0][1], conversion[1][1]))

        query_url = f"https://catalogue.dataspace.copernicus.eu/odata/v1/Products?$filter=Collection/Name eq 'SENTINEL-2' and Attributes/OData.CSC.StringAttribute/any(att:att/Name eq 'productType' and att/OData.CSC.StringAttribute/Value eq 'S2MSI2A') and OData.CSC.Intersects(area=geography'SRID=4326;{box}') and Attributes/OData.CSC.DoubleAttribute/any(att:att/Name eq 'cloudCover' and att/OData.CSC.DoubleAttribute/Value lt {cloudcover}.00) and ContentDate/Start ge {date_interval[0]} and ContentDate/Start le {date_interval[1]}&$top=100"

        # have catalog search just track numbers of errors instead of logging every error
        try:
            # filter cloud cover here!
            response = requests.get(query_url, timeout=20)
            if response.status_code != 200:
                raise sample_exceptions.BadStatusError(str(response.status_code))
            else:
                search = response.json()
        except sample_exceptions.BadStatusError as err:
            if 'BadStatusError_' + str(err) in error_count:
                error_count['BadStatusError_' + str(err)] += 1
            else:
                error_count['BadStatusError_' + str(err)] = 1
    
            try:
                response = requests.get(query_url, timeout=20)
                if response.status_code != 200:
                    continue
                else:
                    search = response.json()
            except Exception as err:
                continue
        except requests.exceptions.JSONDecodeError as err:
            error_count['JSONDecodeError'] += 1
            try:
                response = requests.get(query_url, timeout=20)
                if response.status_code != 200:
                    continue
                else:
                    search = response.json()
            except Exception as err:
                continue
        except requests.exceptions.RequestException as err:
            error_count['RequestsError'] += 1
            continue
        except Exception as err:
            error_count['GeneralError'] += 1
            continue

        products = [product for product in search['value'] if product['Online']]
        if len(products) == 0:
            continue
        elif not rasters.found_after_images(products, event_date_str):
            continue

        event_dates.append(event_date_str)
        event_precip.append(precip_data[time, y, x])
        
        # convert latitude and longitude to
        # (minx, miny, maxx, maxy) using the equation
        minx.append(c_minx)
        miny.append(c_miny)
        maxx.append(c_maxx)
        maxy.append(c_maxy)
        eid.append(f'{threshold}_{event_date_str}_{y}_{x}')

        count += 1
        if n is not None:
            pbar.update(1)
        else:
            print(f"Extreme precipitation events found: {count}", end='\r')

    if n is not None:
        pbar.close()

    rootLogger.debug('Catalog search complete.')
    rootLogger.debug(', '.join(f'{key}: {value}' for key, value in error_count.items()))
        
    df = pd.DataFrame({"Date": event_dates,
                       "Precipitation (mm)": event_precip,
                       "minx": minx,
                       "miny": miny,
                       "maxx": maxx,
                       "maxy": maxy,
                       "eid": eid})

    return df

def event_completed(dir_path):
    """Returns whether or not event directory contains all generated rasters."""
    logger = logging.getLogger('main')
    logger.info('Confirming whether event has already been successfully processed before...')
    regex_patterns = ['dem_.*\.tif', 'flowlines_.*\.tif', 'roads_.*\.tif', 'slope_.*\.tif', 'waterbody_.*\.tif', 'tci_\d{8}.*\.tif', 'ndwi_\d{8}.*\.tif', 'b08_\d{8}.*\.tif']
    pattern_dict = {'dem_.*\.tif': 'DEM', 'flowlines_.*\.tif': 'FLOWLINES', 'roads_.*\.tif': 'ROADS', 'slope_.*\.tif': 'SLOPE', 'waterbody_.*\.tif': 'WATERBODY', 'tci_\d{8}.*\.tif': 'S2 IMAGERY', 'ndwi_\d{8}.*\.tif': 'NDWI', 'b08_\d{8}.*\.tif': 'B8 NIR'}
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

def event_download(threshold, days_before, days_after, maxcoverpercentage, event_date, event_precip, minx, miny, maxx, maxy, eid):
    """Downloads S2 imagery for a high precipitation event based on parameters and generates accompanying rasters.
    
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
    from datetime import datetime, timedelta, date
    from logging import basicConfig
    import logging
    import shapely as sp
    import rasters
    import os
    import shutil
    import json
    from fiona.transform import transform

    PRISM_CRS = "EPSG:4269"
    SEARCH_CRS = "EPSG:4326"

    logger = logging.getLogger('events')
    logger.info('**********************************')
    logger.info('START OF EVENT TASK LOG:')
    logger.info(f'Beginning event {eid} download...')
    logger.info(f'Event on {event_date} at bounds: {minx}, {miny}, {maxx}, {maxy}')

    # need to transform box from EPSG 4269 to EPSG 4326 for OData query
    conversion = transform(PRISM_CRS, SEARCH_CRS, (minx, maxx), (miny, maxy))
    box = sp.to_wkt(sp.geometry.box(conversion[0][0], conversion[1][0], conversion[0][1], conversion[1][1]))
    
    dir_path = f'samples_{threshold}_{days_before}_{days_after}_{maxcoverpercentage}/' + eid + '/'
    products_downloaded = rasters.download_Sentinel_2(box, event_date, rasters.get_date_interval(event_date, days_before, days_after), dir_path, 'events')

    # do not download unless we have the products that we want - i.e. criteria is met
    # only download if products found, products exist during or after precip event, otherwise skip
    if products_downloaded is None:
        logger.info(f'Skipping event on {event_date}, at {minx}, {miny}, {maxx}, {maxy}.')
        return False

    # filter out dates with high cloud or no data cover
    # otherwise remove files and skip to next sample
    bounds = (minx, miny, maxx, maxy)
    has_after = False
    logger.info(f'Checking cloud null percentage...')
    
    for dt, filenames in list(products_downloaded.items()):
        # if a date contains high cloud percentage or null data values, toss date out
        try:
            coverpercentage = rasters.cloud_null_percentage(dir_path, filenames, bounds)
            if coverpercentage > maxcoverpercentage:
                logger.debug(f'Sample {dt} near event {event_date}, at {minx}, {miny}, {maxx}, {maxy} rejected due to {coverpercentage}% cloud or null cover.')
                del products_downloaded[dt]
                for filename in filenames:
                    os.remove(dir_path + filename)
            else:
                if datetime.strptime(dt, "%Y%m%d").date() >= datetime.strptime(event_date, "%Y%m%d").date():
                    has_after = True
        except Exception as err:
            logger.error(f'Cloud null percentage calculation error for files at {dt}: {err}, {type(err)}')
            shutil.rmtree(dir_path)
            return False

    # ensure still has post-event image after filters appplied
    if not has_after:
        logger.debug(f'Skipping {event_date}, at {minx}, {miny}, {maxx}, {maxy} due to lack of usable post-event imagery.')
        shutil.rmtree(dir_path)
        return False

    state = rasters.get_state(minx, miny, maxx, maxy)
    if state is None:
        logger.warning(f'State not found for {event_date}, at {minx}, {miny}, {maxx}, {maxy}. Id: {eid}. Removing directory and contents.')
        shutil.rmtree(dir_path)
        return False

    logger.info('Beginning raster generation...')
    try:
        # new - choose dst_crs by picking first file from first product
        dst_crs = rasters.get_TCI10m_crs(dir_path, list(products_downloaded.values())[0][0])
        conversion = transform(PRISM_CRS, dst_crs, (bounds[0], bounds[2]), (bounds[1], bounds[3]))
        cbounds = conversion[0][0], conversion[1][0], conversion[0][1], conversion[1][1]
        
        for dt, filenames in products_downloaded.items():
            dst_shape, dst_transform = rasters.pipeline_S2(dir_path, f'tci_{dt}_{eid}.tif', dst_crs, filenames, cbounds)
            logger.debug(f'S2 raster completed for {dt}.')
            rasters.pipeline_B08(dir_path, f'b08_{dt}_{eid}.tif', dst_crs, filenames, cbounds)
            logger.debug(f'B08 raster completed for {dt}.')
            rasters.pipeline_NDWI(dir_path, f'ndwi_{dt}_{eid}.tif', dst_crs, filenames, cbounds)
            logger.debug(f'NDWI raster completed for {dt}.')
            for filename in filenames:
                os.remove(dir_path + filename)
        logger.debug(f'All S2, B08, NDWI rasters completed successfully.')

        rasters.pipeline_roads(dir_path, f'roads_{eid}.tif', dst_shape, dst_crs, dst_transform, state, buffer=2)
        logger.debug(f'Roads raster completed successfully.')
        rasters.pipeline_dem_slope(dir_path, (f'dem_{eid}.tif', f'slope_{eid}.tif'), dst_shape, dst_crs, dst_transform, bounds)
        logger.debug(f'DEM, slope rasters completed successfully.')
        rasters.pipeline_flowlines(dir_path, f'flowlines_{eid}.tif', dst_shape, dst_crs, dst_transform, bounds, buffer=2)
        logger.debug(f'Flowlines raster completed successfully.')
        rasters.pipeline_waterbody(dir_path, f'waterbody_{eid}.tif', dst_shape, dst_crs, dst_transform, bounds)
        logger.debug(f'Waterbody raster completed successfully.')
    except Exception as err:
        logger.error(f'Raster generation error: {err}, {type(err)}')
        logger.error(f'Raster generation failed for {event_date}, at {minx}, {miny}, {maxx}, {maxy}. Id: {eid}. Removing directory and contents.')
        shutil.rmtree(dir_path)
        return False

    # lastly generate metadata file
    logger.info(f'Generating metadata file...')
    metadata = {
        "metadata": {
            "Sample ID": eid,
            "Precipitation Event Date": event_date,
            "Cumulative Daily Precipitation (mm)": event_precip,
            "Precipitation Threshold (mm)": threshold,
            "State": state,
            "Bounding Box": {
                "minx": minx,
                "miny": miny,
                "maxx": maxx,
                "maxy": maxy
            }
        }
    }

    with open(dir_path + 'metadata.json', "w") as json_file:
        json.dump(metadata, json_file, indent=4)
    
    logger.info('Metadata and raster generation completed. Event finished.')
    return True

def main(threshold, days_before, days_after, maxcoverpercentage, maxevents):
    """
    Samples imagery of events queried from PRISM using a given minimum precipitation threshold.
    Downloaded samples will contain multispectral data from within specified interval of event date, their respective
    NDWI rasters. Samples will also have a raster of roads from TIGER roads dataset, DEM raster from USGS, 
    flowlines and waterbody rasters from NHDPlus dataset. All rasters will be 4km x 4km at 10m resolution.

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
        
    Returns
    -------
    int
    """
    # make directory
    dir_path = f'samples_{threshold}_{days_before}_{days_after}_{maxcoverpercentage}/'
    Path(dir_path).mkdir(parents=True, exist_ok=True)

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
    
    prism = read_PRISM()
    rootLogger.info("PRISM successfully loaded.")

    if os.path.isfile(dir_path + 'history.pickle'):
        with open(dir_path + 'history.pickle', "rb") as f:
            history = pickle.load(f)
    else:
        history = set()
        
    events = general_query_filter_PRISM(prism, days_before, days_after, history, threshold=threshold, n=maxevents)
    # events = general_query_PRISM(prism, threshold=threshold, n=maxevents)

    rootLogger.info("Initializing event downloads...")
    
    # TEST:
    # events = pd.DataFrame.from_dict({'Date': ['20201010', '20201016', '20201109'], 'minx': [-92.770833, -81.437500, -80.395833], 'miny': [30.062500, 26.104167, 25.937500], 'maxx': [-92.729167, -81.395833, -80.354167], 'maxy': [30.10416, 26.145833, 25.979167], 'eid': ['test1', 'test2', 'test3']})

    count = 0
    alr_completed = 0
    print(f'{len(events.index)} possible extreme precipitation events found. Beginning data collection...')
    try:
        pbar = tqdm(total = len(events.index))
        pbar.set_description(f"Successful samples: {count}")
        for event_date, event_precip, minx, miny, maxx, maxy, eid in zip(events['Date'], events['Precipitation (mm)'], events['minx'], events['miny'], events['maxx'], events['maxy'], events['eid']):
            if Path(dir_path + eid + '/').is_dir():
                if event_completed(dir_path + eid + '/'):
                    rootLogger.debug('Event has already been processed before. Moving on to the next event...')
                    alr_completed += 1
                    history.add(eid)
                    continue
                else:
                    rootLogger.debug('Event has already been processed before but unsuccessfully. Reprocessing...')

            try:
                if event_download(threshold, days_before, days_after, maxcoverpercentage, event_date, event_precip, minx, miny, maxx, maxy, eid):
                    count += 1
            except Exception as err:
                raise err
            else:
                history.add(eid)
                
            pbar.set_description(f"Successful samples: {count}")
            pbar.update(1)

        pbar.close()
        
        rootLogger.debug(f"Number of events already completed: {alr_completed}")
        rootLogger.debug(f"Number of events skipped from this run: {len(events.index) - count - alr_completed} out of {len(events.index)}")
    except Exception as err:
        rootLogger.error(f"Unexpected error: {err}, {type(err)}")
    finally:
        # store all previously processed events
        with open(dir_path + 'history.pickle', 'wb') as f:
            pickle.dump(history, f)
        
    return 0

if __name__ == '__main__':
    parser = argparse.ArgumentParser(prog='sampleS2', description='Samples imagery from Copernicus SENTINEL-2 for top precipitation events and generates additional accompanying rasters for each event.')
    parser.add_argument('threshold', type=int, help='minimum daily cumulative precipitation (mm) threshold for search')
    parser.add_argument('-b', '--before', dest='days_before', default=2, type=int, help='number of days allowed for download before precipitation event (default: 2)')
    parser.add_argument('-a', '--after', dest='days_after', default=4, type=int, help='number of days allowed for download following precipitation event (default: 4)')
    parser.add_argument('-c', '--maxcover', dest='maxcoverpercentage', default=30, type=int, help='maximum cloud and no data cover percentage (default: 30)')
    parser.add_argument('-s', '--maxevents', dest='maxevents', type=int, help='maximum number of extreme precipitation events to attempt downloading (default: None)')
    args = parser.parse_args()
    
    sys.exit(main(args.threshold, args.days_before, args.days_after, args.maxcoverpercentage, args.maxevents))
