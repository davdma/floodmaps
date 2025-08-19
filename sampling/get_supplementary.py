from glob import glob
import re
import os
import geopandas as gpd
from tqdm import tqdm
import requests
import argparse

def download_url(url, save_path, chunk_size=128):
    r = requests.get(url, stream=True)
    with open(save_path, 'wb') as fd:
        for chunk in r.iter_content(chunk_size=chunk_size):
            fd.write(chunk)

def download_roads():
    """Download all TIGER roads data and stitch together into one big file."""

    # make Roads directory if it doesn't exist
    os.makedirs('Roads', exist_ok=True)

    r = requests.get('https://www2.census.gov/geo/tiger/TIGER_RD18/LAYER/ROADS/')
    p = re.compile("tl_rd22_\d*_roads.zip")
    for filename in p.findall(r.text):
        download_url(f'https://www2.census.gov/geo/tiger/TIGER_RD18/LAYER/ROADS/{filename}', f'Roads/{filename}')

    # statefips.txt is a file that contains the FIPS codes for each state.
    # e.g. 17,ILLINOIS
    statefips = dict()
    with open('statefips.txt', 'r') as file:
        for line in file:
            digits, state = line.split(',')
            statefips[digits] = state.rstrip()

    # stitch all roads shapefiles into one by state
    dir_names = os.listdir('Roads/')
    for digits, state in statefips.items():
        lst = []
        p = re.compile(f"tl_rd22_{digits}\d*_roads.zip")
        for file in dir_names:
            m = p.match(file)
            if m:
                lst.append(gpd.read_file(f'/vsizip/Roads/{file}/{file[:-4]}.shp'))
        
        stateshps = gpd.pd.concat(lst)
        stateshps.to_file(f"Roads/{state.strip()}.shp")

    # remove roads files
    for file in dir_names:
        os.remove('Roads/' + file)

def download_dem():
    """Download National Elevation Dataset (NED) for every tile across united states.
    Accessed via 3DEP 1/3 arc-second DEM.
    
    Use https://apps.nationalmap.gov/downloader/ to filter for 1/3 arc-second (10m) current
    DEM. Then download the search results into neddownload.txt."""
    # there are no duplicates in download links
    # make Elevation directory if it doesn't exist
    os.makedirs('Elevation', exist_ok=True)

    current_files = glob('Elevation/*.tif')
    current_products = set()
    p = re.compile('n\d*w\d*')
    for file in current_files:
        match = p.search(file)
        if match:
            current_products.add(match.group())

    all_products = []
    with open("neddownload.txt", 'r') as file:
        for line in file:
            all_products.append(line.rstrip())

    for product in tqdm(all_products):
        match = p.search(product)
        if match and match.group() not in current_products:
            download_url(product, f'Elevation/{match.group()}.tif')

def download_nhd():
    """Download WBD and all NHDPlus GDB datasets.
    
    NOTE: You will need to unzip each zip file in the NHD directory. Reading compressed .zip files
    is extremely slow so the scripts will not work with them.
    """
    os.makedirs('NHD', exist_ok=True)

    # Download WBD
    download_url("https://prd-tnm.s3.amazonaws.com/StagedProducts/Hydrography/WBD/National/GDB/WBD_National_GDB.zip", 'NHD/WBD_National.zip')
    
    # download all NHDPlus GDB datasets
    p = re.compile("NHDPLUS_H_\d{4}_HU4_GDB.zip")
    with open('nhdfilenames.txt', 'r') as f:
        for filename in p.findall(f.read()):
            download_url(f'https://prd-tnm.s3.amazonaws.com/StagedProducts/Hydrography/NHDPlusHR/Beta/GDB/{filename}', f'NHD/{filename}')

def download_nlcd():
    """Download NLCD data for each year. May need to update the list if new NLCD years
    are released.
    
    The NLCD data catalog is found here: https://www.mrlc.gov/data.
    """
    os.makedirs('NLCD', exist_ok=True)

    years = ['2016', '2017', '2018', '2019', '2020', '2021', '2022', '2023']
    for year in years:
        download_url(f'https://s3-us-west-2.amazonaws.com/mrlc/Annual_NLCD_LndCov_{year}_CU_C1V0.tif', f'NLCD/LndCov{year}.tif')

def main(roads=True, dem=True, nhd=True, nlcd=True):
    if roads:
        download_roads()
    if dem:
        download_dem()
    if nhd:
        download_nhd()
    if nlcd:
        download_nlcd()

if __name__ == "__main__":
    parser = argparse.ArgumentParser(prog='getsupplementary', description="""This script is used
        for setting up supplementary data required for the sampling scripts. Downloads TIGER roads,
        NHD, DEM, and NLCD data. Run inside the sampling directory as it will create subfolders.""")
    parser.add_argument('--roads', action='store_true', help='download TIGER roads data')
    parser.add_argument('--dem', action='store_true', help='download DEM data')
    parser.add_argument('--nhd', action='store_true', help='download NHD data')
    parser.add_argument('--nlcd', action='store_true', help='download NLCD data')
    args = parser.parse_args()
    main(args.roads, args.dem, args.nhd, args.nlcd)