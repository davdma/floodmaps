from glob import glob
import numpy as np
import re
import os
from osgeo import gdal
import multiprocessing
from netCDF4 import Dataset
from tqdm import tqdm
from cftime import num2date, date2num
from datetime import date, datetime, timedelta
import requests
from pathlib import Path
import argparse
import sys

PATH_PRISM_DIR = Path('../sampling/PRISM/')

def download_prism_zips(start_date=date(2024, 7, 31), end_date=date(2024, 11, 30)):
    """Downloads prism precip zip files through HTTP.
    Note that we use 2016/8/1 as the 0 point of the netcdf time dimension as S2 data
    only becomes frequently available after that date.
    
    Parameters
    ----------
    start_date : date
        Start date to download prism precip zip files from.
    end_date : date
        End date to download prism precip zip files to. Use the most recent stable date
        which can be found here: https://data.prism.oregonstate.edu/daily/ppt/
    """
    # Before: 2014/4/3 until 2023/01/31
    # Latest: 2016/8/1 until 2024/11/30
    delta = timedelta(days=1)

    # iterate over range of dates
    while (start_date <= end_date):
        dt = start_date.strftime("%Y%m%d")
        # skip if already exists
        if (PATH_PRISM_DIR / f'PRISM_ppt_stable_4kmD2_{dt}_bil.zip').exists():
            start_date += delta
            continue
        
        response = requests.get(f'http://services.nacse.org/prism/data/public/4km/ppt/{dt}')
        with open(PATH_PRISM_DIR / f'PRISM_ppt_stable_4kmD2_{dt}_bil.zip', 'wb') as p:
            p.write(response.content)

        start_date += delta

def load_prism_zips():
    """Read all downloaded precip data from PRISM directory into python list.
    
    Returns
    -------
    daily_precip : list
        List of tuples containing date, precip data, and geotransform.
    """
    # alphabetical sorting is chronological here
    _paths_prism_daily = np.sort(glob(str(PATH_PRISM_DIR / '*_bil.zip')))
    def read_prism(_path_prism_zip):
        p = re.compile('\d{8}')
        match = p.search(_path_prism_zip)
        if match:
            date = match.group()
        else:
            raise Exception('No formatted date found for file:', filename)
        
        compressed_filename = os.path.basename(_path_prism_zip)[:-3] + 'bil'

        # NOTE: For gdal path_to_file after vsizip can be relative or absolute
        prism_file = gdal.Open('/vsizip/' + _path_prism_zip + '/' + compressed_filename)

        prism_raw = prism_file.GetRasterBand(1).ReadAsArray()

        prism_raw[prism_raw == -9999] = np.nan

        return (date, prism_raw, prism_file.GetGeoTransform())
        
    pbar = tqdm(range(len(_paths_prism_daily)))

    with multiprocessing.Pool(10) as pool:    
        tasks = [pool.apply_async(read_prism, args=(path,)) for path in _paths_prism_daily[:]]
        
        prev_ready = 0
        num_ready = sum(task.ready() for task in tasks)
        
        while num_ready != len(tasks):
            if num_ready > prev_ready:
                pbar.update(num_ready - prev_ready)
            prev_ready = num_ready
            num_ready = sum(task.ready() for task in tasks)

        daily_precip = [task.get() for task in tasks]

    return daily_precip

def store_netcdf(daily_precip, filename="prismprecip_20160801_20241130.nc"):
    """Store daily precip data into netcdf file.
    
    Parameters
    ----------
    daily_precip : list
        List of tuples containing date, precip data, and geotransform.
    """
    arr = np.empty((len(daily_precip), len(daily_precip[0][1]), len(daily_precip[0][1][0])))
    for i, entry in tqdm(enumerate(daily_precip)):
        date, data, geo = entry
        arr[i, :, :] = data

    with Dataset(PATH_PRISM_DIR / filename, "w", format="NETCDF4") as nc:
        nc.description = "PRISM Precipitation Dataset"
        time = nc.createDimension("time", len(daily_precip))
        lat = nc.createDimension("y", len(daily_precip[0][1]))
        lon = nc.createDimension("x", len(daily_precip[0][1][0]))
        coeff = nc.createDimension("coeff", 6)
        precip_var = nc.createVariable("precip", "f4", ("time", "y", "x"), zlib=True)
        time_var = nc.createVariable("time", "u4", ("time",))
        time_var[:] = np.arange(0, len(daily_precip))
        time_var.units = "days since 2016-08-01 00:00:00"
        time_var.calendar = "gregorian"
        geotransform = nc.createVariable("geotransform", "f8", ("coeff",), zlib=True)
        geotransform[:] = daily_precip[0][2]
        precip_var[:, :, :] = arr

def main(start_date, end_date):
    # Convert string dates to date objects
    start_date_obj = datetime.strptime(start_date, '%Y%m%d').date()
    end_date_obj = datetime.strptime(end_date, '%Y%m%d').date()
    
    download_prism_zips(start_date=start_date_obj, end_date=end_date_obj)
    daily_precip = load_prism_zips()
    store_netcdf(daily_precip, filename=f"prismprecip_20160801_{end_date}.nc")

if __name__ == "__main__":
    parser = argparse.ArgumentParser(prog='getprism', description="""This script
        can be used to setup PRISM data or to update existing PRISM data to match
        newly available dates. For setting up PRISM data from scratch, the start_date should be
        20160801 as that is the 0 point of the netcdf time dimension used for the project dataset.
        Otherwise to add more dates, set start_date as the date of the most recently downloaded
        precipitation file.""")
    parser.add_argument('--start_date', default='20160801', help='start date to download PRISM precip data from (default: 20160801)')
    parser.add_argument('--end_date', default='20241130', help='end date to download PRISM precip data to (default: 20241130)')
    args = parser.parse_args()
    
    sys.exit(main(args.start_date, args.end_date))