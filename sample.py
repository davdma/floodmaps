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

# parallelization
import parsl
from parsl import bash_app, python_app
from parsl.config import Config
from parsl.providers import SlurmProvider
from parsl.executors import HighThroughputExecutor
from parsl.addresses import address_by_hostname, address_by_interface
from parsl.channels import LocalChannel
from parsl.launchers import SrunLauncher, AprunLauncher, MpiRunLauncher
from parsl.data_provider.files import File

import traceback
from glob import glob
import json

def read_PRISM():
    """Reads the PRISM netCDF file and return the encoded data."""
    with Dataset("PRISM/prismprecip.nc", "r", format="NETCDF4") as nc:
        geotransform = nc["geotransform"][:]
        time_info = (nc["time"].units, nc["time"].calendar)
        precip_data = nc["precip"][:]

    return (geotransform, time_info, precip_data)

def general_query_PRISM(prism, threshold=300):
    """
    Queries contents of PRISM netCDF file given the return value of read_PRISM(). Filters for 
    precipitation events that meet minimum threshold of precipitation.

    Parameters
    ----------
    prism : tuple returned by read_PRISM()
    threshold : int, optional
        Minimum cumulative daily precipitation in mm for filtering PRISM tiles.

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
    id = []

    min_date = datetime(2015, 7, 1)
    for time, y, x in zip(events[0], events[1], events[2]):
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
    regex_patterns = ['dem\.tif', 'flowlines\.tif', 'roads\.tif', 'slope\.tif', 'waterbody\.tif', 's2_\d{8}\.tif', 'ndwi_\d{8}\.tif']
    pattern_dict = {'dem\.tif': 'DEM', 'flowlines\.tif': 'FLOWLINES', 'roads\.tif': 'ROADS', 'slope\.tif': 'SLOPE', 'waterbody\.tif': 'WATERBODY', 's2_\d{8}\.tif': 'S2 IMAGERY', 'ndwi_\d{8}\.tif': 'NDWI'}
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

@python_app
def event_download(threshold, days_before, days_after, maxcoverpercentage, event_date, minx, miny, maxx, maxy, eid, log_file):
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
    log_file : str
        File to use for logging.

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
    from fiona.transform import transform

    PRISM_CRS = "EPSG:4269"
    SEARCH_CRS = "EPSG:4326"

    loggername = f'{event_date}_{eid}'
    logger = logging.getLogger(loggername)
    logger.setLevel(logging.DEBUG)
    fh = logging.FileHandler(log_file, mode='w')
    fh.setLevel(logging.DEBUG)
    formatter = logging.Formatter('%(asctime)s - %(name)s - %(levelname)s - %(message)s')
    fh.setFormatter(formatter)
    logger.addHandler(fh)
    logger.info(f'Beginning event {eid} download...')

    # need to transform box from EPSG 4269 to EPSG 4326 for OData query
    conversion = transform(PRISM_CRS, SEARCH_CRS, (minx, maxx), (miny, maxy))
    box = sp.to_wkt(sp.geometry.box(conversion[0][0], conversion[1][0], conversion[0][1], conversion[1][1]))
    
    dir_path = f'samples_{threshold}_{days_before}_{days_after}_{maxcoverpercentage}/' + eid + '/'
    products_downloaded = rasters.download_Sentinel_2(box, event_date, rasters.get_date_interval(event_date, days_before, days_after), dir_path, loggername)

    # do not download unless we have the products that we want - i.e. criteria is met
    # only download if products found, products exist during or after precip event, otherwise skip
    if products_downloaded is None:
        logger.info(f'Skipping event on {event_date}, at {minx}, {miny}, {maxx}, {maxy}.')
        return False

    # filter out dates with high cloud or no data cover
    # otherwise remove files and skip to next sample
    bounds = (minx, miny, maxx, maxy)
    has_after = False
    for dt, filenames in list(products_downloaded.items()):
        # if a date contains high cloud percentage or null data values, toss date out
        coverpercentage = rasters.cloud_null_percentage(dir_path, filenames, bounds)
        if coverpercentage > maxcoverpercentage:
            logger.debug(f'Sample {dt} near event {event_date}, at {minx}, {miny}, {maxx}, {maxy} rejected due to {coverpercentage}% cloud or null cover.')
            del products_downloaded[dt]
            for filename in filenames:
                os.remove(dir_path + filename)
        else:
            if datetime.strptime(dt, "%Y%m%d").date() >= datetime.strptime(event_date, "%Y%m%d").date():
                has_after = True

    # ensure still has post-event image after filters appplied
    if not has_after:
        logger.debug(f'Skipping {event_date}, at {minx}, {miny}, {maxx}, {maxy} due to lack of usable post-event imagery.')
        shutil.rmtree(dir_path)
        return False

    state = rasters.get_state(minx, miny)
    if state is None:
        logger.warning(f'State not found for {event_date}, at {minx}, {miny}, {maxx}, {maxy}. Id: {eid}. Removing directory and contents.')
        shutil.rmtree(dir_path)
        return False

    logger.info('Beginning raster generation...')
    for dt, filenames in products_downloaded.items():
        dst_shape, dst_crs, dst_transform = rasters.pipeline_S2(dir_path, f's2_{dt}.tif', filenames, bounds)
        rasters.pipeline_NDWI(dir_path, f'ndwi_{dt}.tif', filenames, bounds)
        for filename in filenames:
            os.remove(dir_path + filename)
            
    try:
        rasters.pipeline_roads(dir_path, 'roads.tif', dst_shape, dst_crs, dst_transform, state, buffer=3)
        rasters.pipeline_dem_slope(dir_path, ('dem.tif', 'slope.tif'), dst_shape, dst_crs, dst_transform, bounds)
        rasters.pipeline_flowlines(dir_path, 'flowlines.tif', dst_shape, dst_crs, dst_transform, bounds, buffer=3)
        rasters.pipeline_waterbody(dir_path, 'waterbody.tif', dst_shape, dst_crs, dst_transform, bounds)
    except Exception as err:
        logger.error(f'Raster generation of roads, DEM, flowlines, waterbody failed for {event_date}, at {minx}, {miny}, {maxx}, {maxy}. Id: {eid}. Removing directory and contents.')
        shutil.rmtree(dir_path)
        return False
    
    return True # whether successful or not

@bash_app
def compile_logs(log_files, dir_path):
    """Compiles all listed log files into one single log file."""
    return "sed -s -e '1i----------------' -e '$a----------------' {0} > {1}events.log && rm {0}".format(' '.join(log_files), dir_path)

def main(threshold, days_before, days_after, maxcoverpercentage):
    """
    Samples imagery of events queried from PRISM using a given minimum precipitation threshold.
    Downloaded samples will contain multispectral data from within specified interval of event date, their respective
    NDWI rasters. Samples will also have a raster of roads from TIGER roads dataset, DEM raster from USGS, 
    flowlines and waterbody rasters from NHDPlus dataset. All rasters will be 4km x 4km at 10m resolution.
    
    Parameters
    ----------
    threshold : int
    days_before : int
        Number of days of interest before precipitation event.
    days_after : int
        Number of days of interest following precipitation event.
    cloudcoverpercentage : (int, int)
        Desired cloud cover percentage range for querying Copernicus Sentinel-2.
        
    Returns
    -------
    int
    """
    # make directory
    dir_path = f'samples_{threshold}_{days_before}_{days_after}_{maxcoverpercentage}/'
    Path(dir_path).mkdir(parents=True, exist_ok=True)
        
    rootLogger = logging.getLogger('main')
    rootLogger.setLevel(logging.DEBUG)
    fh = logging.FileHandler(dir_path + 'main.log', mode='w')
    fh.setLevel(logging.DEBUG)
    formatter = logging.Formatter('%(asctime)s - %(name)s - %(levelname)s - %(message)s')
    fh.setFormatter(formatter)
    rootLogger.addHandler(fh)
    
    prism = read_PRISM()
    rootLogger.info("PRISM successfully loaded.")
    events = general_query_PRISM(prism, threshold=threshold)

    rootLogger.info("Initializing event downloads...")

    # parallelization
    parsl.set_file_logger(dir_path + 'parsl.log', level=logging.DEBUG)
    config = Config(
        retries=1,
        executors=[
            HighThroughputExecutor(
                label="bebopslurm",
                cores_per_worker=36,
                provider=SlurmProvider(
                    partition='bdwall',
        		    account='STARTUP-DMA',
        		    launcher=SrunLauncher(),
                    scheduler_options='#SBATCH --job-name=flood_samples',
                    min_blocks=1,
                    init_blocks=1,
                    max_blocks=4,
                    walltime="03:00:00",
                    nodes_per_block=1,
                    parallelism=1,
                    cmd_timeout=60,
                    worker_init='''
    cd /lcrc/hydrosm/dma
    source activate floodmaps
    export PYTHONPATH=/lcrc/hydrosm/dma/:$PYTHONPATH
    '''
                ),
            )
        ]
    )
    parsl.load(config)
    
    # TEST:
    # events = pd.DataFrame.from_dict({'Date': ['20201010', '20201016', '20201109'], 'minx': [-92.770833, -81.437500, -80.395833], 'miny': [30.062500, 26.104167, 25.937500], 'maxx': [-92.729167, -81.395833, -80.354167], 'maxy': [30.10416, 26.145833, 25.979167], 'eid': ['test1', 'test2', 'test3']})
    
    futures = []
    log_files = []
    alr_completed = 0
    for event_date, minx, miny, maxx, maxy, eid in tqdm(zip(events['Date'], events['minx'], events['miny'], events['maxx'], events['maxy'], events['eid']), total=len(events.index)):
        if Path(dir_path + eid + '/').is_dir():
            if event_completed(dir_path + eid + '/'):
                rootLogger.debug('Event has already been processed before. Moving on to the next event...')
                alr_completed += 1
                continue
            else:
                rootLogger.debug('Event has already been processed before but unsuccessfully. Reprocessing...')
            
        log_file = dir_path + f'{event_date}_{eid}.log'
        log_files.append(log_file)
        futures.append(event_download(threshold, days_before, days_after, maxcoverpercentage, event_date, minx, miny, maxx, maxy, eid, log_file))
        
    results = [future.result() for future in futures]
    rootLogger.debug(f"Number of events already completed: {alr_completed}")
    rootLogger.debug(f"Number of events skipped from this run: {len(results) - sum(results)} out of {len(results) + alr_completed}")

    # concatenate log files here with separator in between using bash script
    if len(log_files) > 0:
        compile_logs(log_files, dir_path).result()
    
    return 0

if __name__ == '__main__':
    parser = argparse.ArgumentParser(prog='sampleS2', description='Samples imagery from Copernicus SENTINEL-2 for top precipitation events and generates additional accompanying rasters for each event.')
    parser.add_argument('threshold', type=int, help='minimum daily cumulative precipitation (mm) threshold for search')
    parser.add_argument('-b', '--before', dest='days_before', default=2, type=int, help='number of days allowed for download before precipitation event (default: 2)')
    parser.add_argument('-a', '--after', dest='days_after', default=4, type=int, help='number of days allowed for download following precipitation event (default: 4)')
    parser.add_argument('-c', '--maxcover', dest='maxcoverpercentage', default=30, type=int, help='maximum cloud and no data cover percentage (default: 30)')
    args = parser.parse_args()
    
    sys.exit(main(args.threshold, args.days_before, args.days_after, args.maxcoverpercentage))
