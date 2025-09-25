from glob import glob
import numpy as np
import re
import os
from osgeo import gdal
import multiprocessing
from netCDF4 import Dataset
from cftime import num2date, date2num
from datetime import date, datetime, timedelta
import requests
from pathlib import Path
import hydra
from omegaconf import DictConfig
from typing import List, Tuple

from floodmaps.utils.sampling_utils import parse_date_string, unzip_file

def download_prism_zips(cfg: DictConfig, start_date=date(2024, 7, 31), end_date=date(2024, 11, 30)):
    """Downloads prism precip zip files through HTTP and then extracts the zip to bil file.
    Note that we use 2016/8/1 as the 0 point of the netcdf time dimension as S2 data
    only becomes frequently available after that date. This can however be changed as a setting,
    but it is recommended to stick to one consistently.
    
    Parameters
    ----------
    cfg : DictConfig
        Configuration dictionary.
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
        if (Path(cfg.paths.prism_dir) / f'PRISM_ppt_stable_4kmD2_{dt}_bil.bil').exists():
            print(f'BIL file for date {dt} already exists, skipping')
            start_date += delta
            continue
        
        max_attempts = 3
        for attempt in range(max_attempts):
            try:
                response = requests.get(f'http://services.nacse.org/prism/data/public/4km/ppt/{dt}')
                response.raise_for_status()
                break
            except requests.exceptions.RequestException as e:
                if attempt == max_attempts - 1:
                    raise Exception(f'Failed to download PRISM zip file for date {dt} from url: http://services.nacse.org/prism/data/public/4km/ppt/{dt}') from e
                else:
                    print(f'Failed to download PRISM zip file for date {dt} from url: http://services.nacse.org/prism/data/public/4km/ppt/{dt}. Retrying...')
                    continue
        
        with open(Path(cfg.paths.prism_dir) / f'PRISM_ppt_stable_4kmD2_{dt}_bil.zip', 'wb') as p:
            p.write(response.content)
        print(f'Downloaded zip file PRISM_ppt_stable_4kmD2_{dt}_bil.zip for date {dt}. Extracting...')

        unzip_file(Path(cfg.paths.prism_dir) / f'PRISM_ppt_stable_4kmD2_{dt}_bil.zip', remove_zip=True)

        start_date += delta

def read_prism(_path_bil_file):
    """Helper to read PRISM bil file into numpy array.
    
    Parameters
    ----------
    _path_bil_file : Path
        Path to PRISM bil file.

    Returns
    -------
    tuple
        Tuple containing date string YYYYMMDD, precip data, and geotransform.
    """
    p = re.compile('\d{8}')
    match = p.search(_path_bil_file.name)
    if match:
        dt = match.group()
    else:
        raise Exception('No formatted date found for file:', _path_bil_file)

    # NOTE: For gdal path_to_file after vsizip can be relative or absolute
    with gdal.Open(_path_bil_file) as prism_file:
        if prism_file is None:
            raise Exception(f'Failed to open PRISM bil file: {_path_bil_file}')

        prism_raw = prism_file.GetRasterBand(1).ReadAsArray()
        geotransform = prism_file.GetGeoTransform()

    prism_raw[prism_raw == -9999] = np.nan

    return (dt, prism_raw, geotransform)

def load_prism_bils(cfg: DictConfig):
    """Read all downloaded precip data from PRISM directory into python list.

    Parameters
    ----------
    cfg : DictConfig
        Configuration dictionary.
    
    Returns
    -------
    daily_precip : list
        List of tuples containing date, precip data, and geotransform.
    """
    # alphabetical sorting is chronological here
    _paths_prism_daily = sorted(Path(cfg.paths.prism_dir).glob('*_bil.bil')) # this globs the extracted files

    with multiprocessing.Pool(10) as pool:    
        tasks = [pool.apply_async(read_prism, args=(path,)) for path in _paths_prism_daily[:]]
        daily_precip = [task.get() for task in tasks]

    return daily_precip

def store_netcdf(cfg: DictConfig, daily_precip: List[Tuple[str, np.ndarray, np.ndarray]],
                filename: str = "prismprecip_20160801_20241130.nc",
                start_date: date = date(2016, 8, 1)) -> None:
    """Store daily precip data into netcdf file.
    
    Parameters
    ----------
    cfg : DictConfig
        Configuration dictionary.
    daily_precip : list
        List of tuples containing date, precip data, and geotransform.
    filename : str
        Name of the netcdf file to store the data in.
    start_date : date
        Start date of the PRISM data.
    """
    if len(daily_precip) == 0:
        raise Exception('No daily precipitation data found')
    
    arr = np.empty((len(daily_precip), len(daily_precip[0][1]), len(daily_precip[0][1][0])))
    # validate that the dates are chronological and increments a day at a time
    for i, entry in enumerate(daily_precip):
        dt, data, geo = entry

        ref_date = start_date + timedelta(days=i)
        if datetime.strptime(dt, "%Y%m%d").date() != ref_date:
            raise Exception(f'Dates are not chronological and increments a day at a time. Found date {dt} but expected {ref_date.strftime("%Y%m%d")}')
        
        arr[i, :, :] = data

    print(f'Read into array. Storing as netcdf file {filename}...')
    with Dataset(Path(cfg.paths.prism_dir) / filename, "w", format="NETCDF4") as nc:
        nc.description = "PRISM Precipitation Dataset"
        time = nc.createDimension("time", len(daily_precip))
        lat = nc.createDimension("y", len(daily_precip[0][1]))
        lon = nc.createDimension("x", len(daily_precip[0][1][0]))
        coeff = nc.createDimension("coeff", 6)
        precip_var = nc.createVariable("precip", "f4", ("time", "y", "x"), zlib=True)
        time_var = nc.createVariable("time", "u4", ("time",))
        time_var[:] = np.arange(0, len(daily_precip))
        time_var.units = f"days since {start_date.strftime('%Y-%m-%d')} 00:00:00"
        time_var.calendar = "gregorian"
        geotransform = nc.createVariable("geotransform", "f8", ("coeff",), zlib=True)
        geotransform[:] = daily_precip[0][2]
        precip_var[:, :, :] = arr


@hydra.main(version_base=None, config_path="pkg://configs", config_name="config.yaml")
def main(cfg: DictConfig):
    """
    Main function to download PRISM precipitation data and store as NetCDF.
    
    Two hydra config args start_date and end_date of PRISM precip data can be provided via CLI:
        python -m floodmaps.sampling.get_prism +start_date=2016-08-01 +end_date=2024-11-30
    
    For setting up PRISM data from scratch, the start_date should be
    2016-08-01 as that is the 0 point of the netcdf time dimension used for the project dataset.
    Otherwise to add more dates, set start_date as the date of the most recently downloaded
    precipitation file.
    """
    # Get dates from config with defaults - raise error if not provided
    start_date = cfg.get('start_date', '2016-08-01')
    end_date = cfg.get('end_date', '2024-11-30')
    print(f'Using start_date: {start_date}, end_date: {end_date}')
    
    # Convert string dates to date objects
    start_date_obj = parse_date_string(start_date).date()
    end_date_obj = parse_date_string(end_date).date()

    # Make prism directory if it doesn't exist
    Path(cfg.paths.prism_dir).mkdir(parents=True, exist_ok=True)
    
    download_prism_zips(cfg, start_date=start_date_obj, end_date=end_date_obj)
    daily_precip = load_prism_bils(cfg)
    store_netcdf(cfg, daily_precip, filename=f"prismprecip_{start_date_obj.strftime('%Y%m%d')}_{end_date_obj.strftime('%Y%m%d')}.nc",
                 start_date=start_date_obj)

if __name__ == "__main__":
    main()