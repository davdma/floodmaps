from glob import glob
import re
import os
import geopandas as gpd
from tqdm import tqdm
import requests
from pathlib import Path
from omegaconf import DictConfig

def download_url(url, save_path, chunk_size=128) -> None:
    r = requests.get(url, stream=True)
    with open(save_path, 'wb') as fd:
        for chunk in r.iter_content(chunk_size=chunk_size):
            fd.write(chunk)

def download_roads(cfg: DictConfig) -> None:
    """Download all TIGER roads data and stitch together into one big file."""
    if cfg.sampling.roads_dir is None:
        raise ValueError('Roads directory not specified in configuration.')

    # make Roads directory if it doesn't exist
    Path(cfg.sampling.roads_dir).mkdir(parents=True, exist_ok=True)

    r = requests.get('https://www2.census.gov/geo/tiger/TIGER_RD18/LAYER/ROADS/')
    p = re.compile("tl_rd22_\d*_roads.zip")
    for filename in p.findall(r.text):
        download_url(f'https://www2.census.gov/geo/tiger/TIGER_RD18/LAYER/ROADS/{filename}', Path(cfg.sampling.roads_dir) / filename)

    # statefips.txt is a file that contains the FIPS codes for each state.
    # e.g. 17,ILLINOIS
    statefips = dict()
    with open(Path(cfg.paths.setup_dir) / 'statefips.txt', 'r') as file:
        for line in file:
            digits, state = line.split(',')
            statefips[digits] = state.rstrip()

    # stitch all roads shapefiles into one by state
    dir_names = os.listdir(cfg.sampling.roads_dir)
    for digits, state in statefips.items():
        lst = []
        p = re.compile(f"tl_rd22_{digits}\d*_roads.zip")
        for file in dir_names:
            m = p.match(file)
            if m:
                lst.append(gpd.read_file(f'/vsizip/{cfg.sampling.roads_dir}/{file}/{file[:-4]}.shp'))
        
        stateshps = gpd.pd.concat(lst)
        stateshps.to_file(Path(cfg.sampling.roads_dir) / f"{state.strip()}.shp")

    # remove roads files
    for file in dir_names:
        os.remove(Path(cfg.sampling.roads_dir) / file)

def download_dem(cfg: DictConfig) -> None:
    """Download National Elevation Dataset (NED) for every tile across united states.
    Accessed via 3DEP 1/3 arc-second DEM.
    
    Use https://apps.nationalmap.gov/downloader/ to filter for 1/3 arc-second (10m) current
    DEM. Then download the search results into setup/neddownload.txt."""
    # there are no duplicates in download links
    # make Elevation directory if it doesn't exist
    Path(cfg.sampling.elevation_dir).mkdir(parents=True, exist_ok=True)

    current_files = list(Path(cfg.sampling.elevation_dir).glob('*.tif'))
    current_products = set()
    p = re.compile('n\d*w\d*')
    for file in current_files:
        match = p.search(file.name)
        if match:
            current_products.add(match.group())

    all_products = []
    with open(Path(cfg.paths.setup_dir) / 'neddownload.txt', 'r') as file:
        for line in file:
            all_products.append(line.rstrip())

    for product in tqdm(all_products):
        match = p.search(product)
        if match and match.group() not in current_products:
            download_url(product, Path(cfg.sampling.elevation_dir) / f'{match.group()}.tif')

def download_nhd(cfg: DictConfig) -> None:
    """Download WBD and all NHDPlus GDB datasets.
    
    NOTE: You will need to unzip each zip file in the NHD directory. Reading compressed .zip files
    is extremely slow so the scripts will not work with them.
    """
    Path(cfg.sampling.nhd_dir).mkdir(parents=True, exist_ok=True)

    # Download WBD
    download_url("https://prd-tnm.s3.amazonaws.com/StagedProducts/Hydrography/WBD/National/GDB/WBD_National_GDB.zip", Path(cfg.sampling.nhd_dir) / 'WBD_National.zip')
    
    # download all NHDPlus GDB datasets
    p = re.compile("NHDPLUS_H_\d{4}_HU4_GDB.zip")
    with open(Path(cfg.paths.setup_dir) / 'nhdfilenames.txt', 'r') as f:
        for filename in p.findall(f.read()):
            download_url(f'https://prd-tnm.s3.amazonaws.com/StagedProducts/Hydrography/NHDPlusHR/Beta/GDB/{filename}', Path(cfg.sampling.nhd_dir) / filename)

def download_nlcd(cfg: DictConfig) -> None:
    """Download NLCD data for each year. May need to update the list if new NLCD years
    are released.
    
    The NLCD data catalog is found here: https://www.mrlc.gov/data.
    """
    Path(cfg.sampling.nlcd_dir).mkdir(parents=True, exist_ok=True)

    years = ['2016', '2017', '2018', '2019', '2020', '2021', '2022', '2023']
    for year in years:
        download_url(f'https://s3-us-west-2.amazonaws.com/mrlc/Annual_NLCD_LndCov_{year}_CU_C1V0.tif', Path(cfg.sampling.nlcd_dir) / f'LndCov{year}.tif')

def main(cfg: DictConfig) -> None:
    download_nlcd(cfg)

if __name__ == "__main__":
    """This script is used for setting up supplementary data required for the sampling scripts. Downloads TIGER roads,
        NHD, DEM, and NLCD data. Run inside the sampling directory as it will create subfolders."""
    main()