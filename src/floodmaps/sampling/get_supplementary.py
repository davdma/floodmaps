import re
import os
import shutil
import geopandas as gpd
from tqdm import tqdm
import requests
from pathlib import Path
import hydra
from omegaconf import DictConfig
import zipfile

def download_url(url, save_path, chunk_size=128) -> None:
    r = requests.get(url, stream=True)
    with open(save_path, 'wb') as fd:
        for chunk in r.iter_content(chunk_size=chunk_size):
            fd.write(chunk)

def unzip_file(zip_path: Path, remove_zip: bool = True) -> None:
    """Unzip a file to the same directory and optionally remove the original zip.
    
    Parameters
    ----------
    zip_path : Path
        Path to the zip file to extract
    remove_zip : bool, default=True
        Whether to remove the zip file after successful extraction
    """
    extract_to = zip_path.parent
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

def download_roads(cfg: DictConfig) -> None:
    """Download all TIGER roads data and stitch together into one big file."""
    if cfg.paths.roads_dir is None:
        raise ValueError('Roads directory not specified in configuration.')

    # make Roads directory if it doesn't exist
    Path(cfg.paths.roads_dir).mkdir(parents=True, exist_ok=True)

    r = requests.get('https://www2.census.gov/geo/tiger/TIGER_RD18/LAYER/ROADS/')
    p = re.compile("tl_rd22_\d*_roads.zip")
    for filename in p.findall(r.text):
        zip_path = Path(cfg.paths.roads_dir) / filename
        download_url(f'https://www2.census.gov/geo/tiger/TIGER_RD18/LAYER/ROADS/{filename}', zip_path)
        unzip_file(zip_path, remove_zip=True)

    # statefips.txt is a file that contains the FIPS codes for each state.
    # e.g. 17,ILLINOIS
    statefips = dict()
    with open(Path(cfg.paths.setup_dir) / 'statefips.txt', 'r') as file:
        for line in file:
            digits, state = line.split(',')
            statefips[digits] = state.rstrip()

    # stitch all roads shapefiles into one by state
    dir_names = os.listdir(cfg.paths.roads_dir)
    for digits, state in statefips.items():
        lst = []
        p = re.compile(f"tl_rd22_{digits}\d*_roads")
        for item in dir_names:
            # Look for unzipped directories that match the pattern
            item_path = Path(cfg.paths.roads_dir) / item
            if item_path.is_dir() and p.match(item):
                shp_file = item_path / f"{item}.shp"
                if shp_file.exists():
                    lst.append(gpd.read_file(shp_file))
        
        if lst:  # Only process if we found matching files
            stateshps = gpd.pd.concat(lst)
            stateshps.to_file(Path(cfg.paths.roads_dir) / f"{state.strip()}.shp")

    # remove extracted directories
    for item in dir_names:
        item_path = Path(cfg.paths.roads_dir) / item
        if item_path.is_dir() and item.startswith('tl_rd22_'):
            # Remove extracted directory and its contents
            shutil.rmtree(item_path)

def download_dem(cfg: DictConfig) -> None:
    """Download National Elevation Dataset (NED) for every tile across united states.
    Accessed via 3DEP 1/3 arc-second DEM.
    
    Use https://apps.nationalmap.gov/downloader/ to filter for 1/3 arc-second (10m) current
    DEM. Then download the search results into setup/neddownload.txt."""
    # there are no duplicates in download links
    # make Elevation directory if it doesn't exist
    Path(cfg.paths.elevation_dir).mkdir(parents=True, exist_ok=True)

    current_files = list(Path(cfg.paths.elevation_dir).glob('*.tif'))
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
            download_url(product, Path(cfg.paths.elevation_dir) / f'{match.group()}.tif')

def download_nhd(cfg: DictConfig) -> None:
    """Download WBD and all NHDPlus GDB datasets and unzip them.
    
    The zip files are automatically unzipped after download since reading compressed .zip files
    is extremely slow. The original zip files are removed after successful extraction.
    """
    Path(cfg.paths.nhd_dir).mkdir(parents=True, exist_ok=True)

    # Download and unzip WBD
    wbd_zip_path = Path(cfg.paths.nhd_dir) / 'WBD_National_GDB.zip'
    download_url("https://prd-tnm.s3.amazonaws.com/StagedProducts/Hydrography/WBD/National/GDB/WBD_National_GDB.zip", wbd_zip_path)
    unzip_file(wbd_zip_path, remove_zip=True)
    
    # download and unzip all NHDPlus GDB datasets
    p = re.compile("NHDPLUS_H_\d{4}_HU4_GDB.zip")
    with open(Path(cfg.paths.setup_dir) / 'nhdfilenames.txt', 'r') as f:
        for filename in p.findall(f.read()):
            zip_path = Path(cfg.paths.nhd_dir) / filename
            download_url(f'https://prd-tnm.s3.amazonaws.com/StagedProducts/Hydrography/NHDPlusHR/Beta/GDB/{filename}', zip_path)
            unzip_file(zip_path, remove_zip=True)

def download_nlcd(cfg: DictConfig) -> None:
    """Download NLCD data for each year. May need to update the list if new NLCD years
    are released.
    
    The NLCD data catalog is found here: https://www.mrlc.gov/data.
    """
    Path(cfg.paths.nlcd_dir).mkdir(parents=True, exist_ok=True)

    years = ['2016', '2017', '2018', '2019', '2020', '2021', '2022', '2023']
    for year in years:
        download_url(f'https://s3-us-west-2.amazonaws.com/mrlc/Annual_NLCD_LndCov_{year}_CU_C1V0.tif', Path(cfg.paths.nlcd_dir) / f'LndCov{year}.tif')

@hydra.main(version_base=None, config_path="pkg://configs", config_name="config.yaml")
def main(cfg: DictConfig) -> None:
    """This script is used for setting up supplementary data required for the sampling scripts. Downloads TIGER roads,
        NHD, DEM, and NLCD data. Run inside the sampling directory as it will create subfolders.

        Note: Use floodmaps-sampling environment.
        
        Parameters
        ----------
        cfg : DictConfig
            Configuration object.
    """
    download_roads(cfg)
    download_dem(cfg)
    download_nhd(cfg)
    download_nlcd(cfg)

if __name__ == "__main__":
    main()