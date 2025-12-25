from datetime import datetime
from pathlib import Path
import sys
import logging
import numpy as np
import os
import re
import json
import pickle
import fiona
import shutil
from osgeo import gdal, ogr, osr
from rasterio.transform import array_bounds
from rasterio.windows import from_bounds, Window
from rasterio.warp import calculate_default_transform, reproject, Resampling, transform_bounds
import rasterio.merge
import rasterio
from fiona.transform import transform
import pystac_client
import hydra
from omegaconf import DictConfig
import time

from floodmaps.utils.sampling_utils import (
    PRISM_CRS,
    SEARCH_CRS,
    BOA_ADD_OFFSET,
    PROCESSING_BASELINE_UTC,
    NLCD_CODE_TO_RGB,
    PRISMData,
    DateCRSOrganizer,
    get_date_interval,
    has_date_after_PRISM,
    setup_logging,
    get_history,
    get_mask,
    get_manual_events,
    get_extreme_events,
    event_completed,
    get_state,
    compute_ndwi,
    colormap_to_rgb,
    NoElevationError,
    crop_to_bounds,
    get_item_crs,
    MissingAssetError,
    walltime_seconds
)
from floodmaps.utils.stac_providers import get_stac_provider
from floodmaps.utils.validate import validate_event_rasters

NLCD_RANGE = None
ELEVATION_LAT_LONG = None
CURRENT_DATE = datetime.now().strftime('%Y-%m-%d')

# patterns for checking if an S2 event has been processed
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
    r'b11_\d{8}.*\.tif',
    r'b12_\d{8}.*\.tif',
    r'scl_\d{8}.*\.tif',
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
    r'b11_\d{8}.*\.tif': 'SWIR1',
    r'b12_\d{8}.*\.tif': 'SWIR2',
    r'scl_\d{8}.*\.tif': 'SCL',
    'metadata.json': 'METADATA'
}

def get_elevation_nw(cfg):
    """Extract all elevation file lat long pairs as well as the filepath in tuple."""
    global ELEVATION_LAT_LONG
    if ELEVATION_LAT_LONG is None:
        elevation_files = [str(x) for x in Path(cfg.paths.elevation_dir).glob('n*w*.tif')]
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

def get_nlcd_range(cfg):
    """Returns a tuple of the earliest and latest year for which NLCD data is available."""
    global NLCD_RANGE
    if NLCD_RANGE is None:
        nlcd_files = [str(x) for x in Path(cfg.paths.nlcd_dir).glob('LndCov*.tif')]
        if len(nlcd_files) == 0:
            raise FileNotFoundError('No NLCD files found. Please run get_supplementary.py to download NLCD data.')
        p = re.compile(r'LndCov(\d{4}).tif')
        nlcd_years = [int(p.search(file).group(1)) for file in nlcd_files]
        NLCD_RANGE = (min(nlcd_years), max(nlcd_years))
    return NLCD_RANGE

def GetHU4Codes(prism_bbox, cfg):
    """
    Queries national watershed boundary dataset for HUC 4 codes representing
    hydrologic units that intersect with bounding box.

    Parameters
    ----------
    prism_bbox : (float, float, float, float)
        Tuple in the order minx, miny, maxx, maxy, representing bounding box in PRISM_CRS = EPSG:4269.

    Returns
    -------
    list[str]
        HU4 codes.
    """
    with gdal.OpenEx(cfg.paths.nhd_wbd) as ds:
        minx, miny, maxx, maxy = prism_bbox
        layer = ds.GetLayerByIndex(4)
        layer.SetSpatialFilterRect(minx, miny, maxx, maxy)
        
        huc4_codes = []
        for feature in layer:
            huc4_code = feature.GetField('huc4')
            if huc4_code:
                huc4_codes.append(huc4_code)

    return huc4_codes

def cloud_null_percentage(stac_provider, item, item_crs, bbox):
    """Calculates the percentage of pixels in the bounding box of the S2 image that are cloud or null."""
    scl_name = stac_provider.get_asset_names("s2")["SCL"]
    
    if scl_name not in item.assets:
        raise MissingAssetError(f"Asset '{scl_name}' not found in S2 item {item.id}")
    
    item_href = stac_provider.sign_asset_href(item.assets[scl_name].href)

    conversion = transform(PRISM_CRS, item_crs, (bbox[0], bbox[2]), (bbox[1], bbox[3]))
    img_bbox = conversion[0][0], conversion[1][0], conversion[0][1], conversion[1][1]

    out_image, _ = rasterio.merge.merge([item_href], bounds=img_bbox)
    return int((np.isin(out_image, [0, 8, 9]).sum() / out_image.size) * 100)

# we will choose a UTM zone CRS already given and stick to it for rest of sample data!
def pipeline_TCI(stac_provider, dir_path: Path, save_as, dst_crs, item, bbox):
    """Generates TCI (True Color Image) raster of S2 multispectral file.

    Parameters
    ----------
    stac_provider : STACProvider
        STAC provider object.
    dir_path : Path
        Path for saving generated raster. 
    save_as : str
        Name of file to be saved (do not include extension!)
    dst_crs : str
        Coordinate reference system of output raster.
    item : Item
        PyStac Item object.
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
    visual_name = stac_provider.get_asset_names("s2")["visual"]
    
    if visual_name not in item.assets:
        raise MissingAssetError(f"Asset '{visual_name}' not found in S2 item {item.id}")
    
    item_href = stac_provider.sign_asset_href(item.assets[visual_name].href)

    out_image, out_transform = crop_to_bounds(item_href, bbox, dst_crs, nodata=0, resampling=Resampling.bilinear)

    with rasterio.open(dir_path / f'{save_as}.tif', 'w', driver='Gtiff', count=3, height=out_image.shape[-2], width=out_image.shape[-1], crs=dst_crs, dtype=out_image.dtype, transform=out_transform, nodata=None) as dst:
        dst.write(out_image)

    return (out_image.shape[-2], out_image.shape[-1]), out_transform

def pipeline_SCL(stac_provider, dir_path: Path, save_as, dst_shape, dst_crs, dst_transform, item, bbox):
    """Generates Scene Classification Layer raster of S2 multispectral file and resamples to 10m resolution.

    Parameters
    ----------
    stac_provider : STACProvider
        STAC provider object.
    dir_path : Path
        Path for saving generated raster. 
    save_as : str
        Name of file to be saved (do not include extension!).
    dst_shape : (int, int)
        Shape of output raster.
    dst_crs : str
        Coordinate reference system of output raster.
    dst_transform : rasterio.affine.Affine()
        Transform of TCI raster.
    item : Item
        PyStac Item object.
    bbox : (float, float, float, float)
        Tuple in the order minx, miny, maxx, maxy, representing bounding box, 
        should be in CRS specified by dst_crs.
    """
    scl_name = stac_provider.get_asset_names("s2")["SCL"]
    
    if scl_name not in item.assets:
        raise MissingAssetError(f"Asset '{scl_name}' not found in S2 item {item.id}")
    
    item_href = stac_provider.sign_asset_href(item.assets[scl_name].href)

    # Single-step resampling directly to TCI grid for perfect pixel alignment
    h, w = dst_shape
    dest = np.zeros((1, h, w), dtype=np.uint8)
    
    with rasterio.open(item_href) as src:
        reproject(
            source=rasterio.band(src, 1),
            destination=dest,
            src_transform=src.transform,
            src_crs=src.crs,
            dst_transform=dst_transform,
            dst_crs=dst_crs,
            resampling=Resampling.nearest,
            dst_nodata=0
        )

    # save SCL raster with all class values preserved
    with rasterio.open(dir_path / f'{save_as}.tif', 'w', driver='Gtiff', count=1, height=h, width=w, crs=dst_crs, 
                        dtype=dest.dtype, transform=dst_transform, nodata=0) as dst:
        dst.write(dest)

    # create RGB colormap from SCL using utility function from sampling_utils
    from floodmaps.utils.sampling_utils import scl_to_rgb
    rgb_scl = scl_to_rgb(dest[0])
    
    with rasterio.open(dir_path / f'{save_as}_cmap.tif', 'w', driver='Gtiff', count=3, height=rgb_scl.shape[-2], width=rgb_scl.shape[-1], 
                       crs=dst_crs, dtype=rgb_scl.dtype, transform=dst_transform, nodata=None) as dst:
        dst.write(rgb_scl)

def pipeline_RGB(stac_provider, dir_path: Path, save_as, dst_crs, item, bbox):
    """Generates B02 (B), B03 (G), B04 (R) rasters of S2 multispectral file.

    Parameters
    ----------
    stac_provider : STACProvider
        STAC provider object.
    dir_path : Path
        Path for saving generated raster. 
    save_as : str
        Name of file to be saved (do not include extension!).
    dst_crs : str
        Coordinate reference system of output raster.
    item : Item
        PyStac Item object.
    bbox : (float, float, float, float)
        Tuple in the order minx, miny, maxx, maxy, representing bounding box,
        should be in CRS specified by dst_crs.
    """
    b02_name = stac_provider.get_asset_names("s2")["B02"]
    b03_name = stac_provider.get_asset_names("s2")["B03"]
    b04_name = stac_provider.get_asset_names("s2")["B04"]
    
    if b02_name not in item.assets:
        raise MissingAssetError(f"Asset '{b02_name}' not found in S2 item {item.id}")
    if b03_name not in item.assets:
        raise MissingAssetError(f"Asset '{b03_name}' not found in S2 item {item.id}")
    if b04_name not in item.assets:
        raise MissingAssetError(f"Asset '{b04_name}' not found in S2 item {item.id}")
    
    b02_item_href = stac_provider.sign_asset_href(item.assets[b02_name].href) # B
    b03_item_href = stac_provider.sign_asset_href(item.assets[b03_name].href) # G
    b04_item_href = stac_provider.sign_asset_href(item.assets[b04_name].href) # R

    # stack the three bands as rgb channels
    blue_image, out_transform = crop_to_bounds(b02_item_href, bbox, dst_crs, nodata=0, resampling=Resampling.bilinear)
    green_image, _ = crop_to_bounds(b03_item_href, bbox, dst_crs, nodata=0, resampling=Resampling.bilinear)
    red_image, _ = crop_to_bounds(b04_item_href, bbox, dst_crs, nodata=0, resampling=Resampling.bilinear)

    out_image = np.vstack((red_image, green_image, blue_image))

    with rasterio.open(dir_path / f'{save_as}.tif', 'w', driver='Gtiff', count=3, height=out_image.shape[-2], width=out_image.shape[-1], crs=dst_crs, dtype=out_image.dtype, transform=out_transform, nodata=0) as dst:
        dst.write(out_image)

def pipeline_B08(stac_provider, dir_path: Path, save_as, dst_crs, item, bbox):
    """Generates NIR B8 band raster of S2 multispectral file.

    Parameters
    ----------
    stac_provider : STACProvider
        STAC provider object.
    dir_path : Path
        Path for saving generated raster. 
    save_as : str
        Name of file to be saved (do not include extension!).
    dst_crs : str
        Coordinate reference system of output raster.
    item : Item
        PyStac Item object.
    bbox : (float, float, float, float)
        Tuple in the order minx, miny, maxx, maxy, representing bounding box,
        should be in CRS specified by dst_crs.
    """
    b08_name = stac_provider.get_asset_names("s2")["B08"]
    
    if b08_name not in item.assets:
        raise MissingAssetError(f"Asset '{b08_name}' not found in S2 item {item.id}")
    
    item_href = stac_provider.sign_asset_href(item.assets[b08_name].href)

    out_image, out_transform = crop_to_bounds(item_href, bbox, dst_crs, nodata=0, resampling=Resampling.bilinear)

    with rasterio.open(dir_path / f'{save_as}.tif', 'w', driver='Gtiff', count=1, height=out_image.shape[-2], width=out_image.shape[-1], crs=dst_crs, dtype=out_image.dtype, transform=out_transform, nodata=0) as dst:
        dst.write(out_image)

def pipeline_B11(stac_provider, dir_path: Path, save_as, dst_shape, dst_crs, dst_transform, item, bbox):
    """Generates SWIR1 B11 band raster of S2 multispectral file (resampled to 10m).

    Parameters
    ----------
    stac_provider : STACProvider
        STAC provider object.
    dir_path : Path
        Path for saving generated raster.
    save_as : str
        Name of file to be saved (do not include extension!).
    dst_shape : (int, int)
        Shape of output raster.
    dst_crs : str
        Coordinate reference system of output raster.
    dst_transform : rasterio.affine.Affine()
        Transform of TCI raster.
    item : Item
        PyStac Item object.
    bbox : (float, float, float, float)
        Tuple in the order minx, miny, maxx, maxy, representing bounding box,
        should be in CRS specified by dst_crs.
    """
    b11_name = stac_provider.get_asset_names("s2")["B11"]
    
    if b11_name not in item.assets:
        raise MissingAssetError(f"Asset '{b11_name}' not found in S2 item {item.id}")
    
    item_href = stac_provider.sign_asset_href(item.assets[b11_name].href)

    # Single-step resampling directly to TCI grid for perfect pixel alignment
    h, w = dst_shape
    dest = np.zeros((1, h, w), dtype=np.uint16)
    
    with rasterio.open(item_href) as src:
        reproject(
            source=rasterio.band(src, 1),
            destination=dest,
            src_transform=src.transform,
            src_crs=src.crs,
            dst_transform=dst_transform,
            dst_crs=dst_crs,
            resampling=Resampling.bilinear,
            dst_nodata=0
        )

    with rasterio.open(dir_path / f'{save_as}.tif', 'w', driver='Gtiff', count=1, height=h, width=w, crs=dst_crs, dtype=dest.dtype, transform=dst_transform, nodata=0) as dst:
        dst.write(dest)

def pipeline_B12(stac_provider, dir_path: Path, save_as, dst_shape, dst_crs, dst_transform, item, bbox):
    """Generates SWIR2 B12 band raster of S2 multispectral file (resampled to 10m).

    Parameters
    ----------
    stac_provider : STACProvider
        STAC provider object.
    dir_path : Path
        Path for saving generated raster.
    save_as : str
        Name of file to be saved (do not include extension!).
    dst_shape : (int, int)
        Shape of output raster.
    dst_crs : str
        Coordinate reference system of output raster.
    dst_transform : rasterio.affine.Affine()
        Transform of TCI raster.
    item : Item
        PyStac Item object.
    bbox : (float, float, float, float)
        Tuple in the order minx, miny, maxx, maxy, representing bounding box,
        should be in CRS specified by dst_crs.
    """
    b12_name = stac_provider.get_asset_names("s2")["B12"]
    
    if b12_name not in item.assets:
        raise MissingAssetError(f"Asset '{b12_name}' not found in S2 item {item.id}")
    
    item_href = stac_provider.sign_asset_href(item.assets[b12_name].href)

    # Single-step resampling directly to TCI grid for perfect pixel alignment
    h, w = dst_shape
    dest = np.zeros((1, h, w), dtype=np.uint16)
    
    with rasterio.open(item_href) as src:
        reproject(
            source=rasterio.band(src, 1),
            destination=dest,
            src_transform=src.transform,
            src_crs=src.crs,
            dst_transform=dst_transform,
            dst_crs=dst_crs,
            resampling=Resampling.bilinear,
            dst_nodata=0
        )

    with rasterio.open(dir_path / f'{save_as}.tif', 'w', driver='Gtiff', count=1, height=h, width=w, crs=dst_crs, dtype=dest.dtype, transform=dst_transform, nodata=0) as dst:
        dst.write(dest)

def pipeline_NDWI(stac_provider, dir_path: Path, save_as, dst_crs, item, bbox):
    """Generates NDWI raster from S2 multispectral files.

    Parameters
    ----------
    stac_provider : STACProvider
        STAC provider object.
    dir_path : Path
        Path for saving generated raster. 
    save_as : str
        Name of raw ndwi file to be saved (do not include extension!).
    dst_crs : str
        Coordinate reference system of output raster.
    item : Item
        PyStac Item object.
    bbox : (float, float, float, float)
        Tuple in the order minx, miny, maxx, maxy, representing bounding box, 
        should be in CRS specified by dst_crs.
    """
    b03_name = stac_provider.get_asset_names("s2")["B03"]
    b08_name = stac_provider.get_asset_names("s2")["B08"]
    
    if b03_name not in item.assets:
        raise MissingAssetError(f"Asset '{b03_name}' not found in S2 item {item.id}")
    if b08_name not in item.assets:
        raise MissingAssetError(f"Asset '{b08_name}' not found in S2 item {item.id}")
    
    b03_item_href = stac_provider.sign_asset_href(item.assets[b03_name].href)

    out_image1, _ = crop_to_bounds(b03_item_href, bbox, dst_crs, nodata=0, resampling=Resampling.bilinear)
    green = out_image1[0].astype(np.int32)

    b08_item_href = stac_provider.sign_asset_href(item.assets[b08_name].href)

    out_image2, out_transform = crop_to_bounds(b08_item_href, bbox, dst_crs, nodata=0, resampling=Resampling.bilinear)
    nir = out_image2[0].astype(np.int32)

    missing_mask = (green == 0) | (nir == 0)

    # calculate ndwi with BOA offset for baseline-or-later captures
    if item.datetime >= PROCESSING_BASELINE_UTC:
        green_sr = (green.astype(np.float32) + BOA_ADD_OFFSET) / 10000.0
        nir_sr = (nir.astype(np.float32) + BOA_ADD_OFFSET) / 10000.0
        green_sr = np.clip(green_sr, 0, 1)
        nir_sr = np.clip(nir_sr, 0, 1)
        ndwi = compute_ndwi(green_sr, nir_sr, missing_val=-999999)
    else:
        green_sr = green.astype(np.float32) / 10000.0
        nir_sr = nir.astype(np.float32) / 10000.0
        green_sr = np.clip(green_sr, 0, 1)
        nir_sr = np.clip(nir_sr, 0, 1)
        ndwi = compute_ndwi(green, nir, missing_val=-999999)
    
    ndwi = np.where(missing_mask, -999999, ndwi)

    # save raw
    with rasterio.open(dir_path / f'{save_as}.tif', 'w', driver='Gtiff', count=1, height=ndwi.shape[-2], width=ndwi.shape[-1], crs=dst_crs, dtype=ndwi.dtype, transform=out_transform, nodata=-999999) as dst:
        dst.write(ndwi, 1)

    # before writing to file, we will make matplotlib colormap!
    ndwi_colored = colormap_to_rgb(ndwi, cmap='seismic_r', r=(-1.0, 1.0), no_data=-999999)
    
    # nodata should not be set for cmap files
    with rasterio.open(dir_path / f'{save_as}_cmap.tif', 'w', driver='Gtiff', count=3, height=ndwi_colored.shape[-2], width=ndwi_colored.shape[-1], crs=dst_crs, dtype=ndwi_colored.dtype, transform=out_transform, nodata=None) as dst:
        dst.write(ndwi_colored)

def pipeline_roads(dir_path: Path, save_as, dst_shape, dst_crs, dst_transform, state, prism_bbox, cfg):
    """Generates raster with burned in geometries of roads given destination raster properties.

    Parameters
    ----------
    dir_path : Path
        Path for saving generated raster. 
    save_as : str
        Name of file to be saved (do not include extension!).
    dst_shape : (int, int)
        Shape of output raster.
    dst_crs : str
        Coordinate reference system of output raster.
    dst_transform : rasterio.affine.Affine()
        Transformation matrix for mapping pixel coordinates to coordinate system of output raster.
    state : str
        State of raster location.
    prism_bbox : (float, float, float, float)
        Tuple in the order minx, miny, maxx, maxy, representing bounding box, 
        in PRISM CRS.
    cfg : DictConfig
        Configuration object.
    """
    logger = logging.getLogger('events')
    mem_ds = None
    raster_ds = None
    minx, miny, maxx, maxy = prism_bbox

    # if state file does not exist, save blank raster
    if (Path(cfg.paths.roads_dir) / f'{state.strip().upper()}.shp').exists():
        try:
            with gdal.OpenEx(Path(cfg.paths.roads_dir) / f'{state.strip().upper()}.shp') as ds:
                layer = ds.GetLayer()
                layer.SetSpatialFilterRect(minx, miny, maxx, maxy)

                dst_srs = osr.SpatialReference()
                dst_srs.SetFromUserInput(dst_crs)

                ct = osr.CoordinateTransformation(layer.GetSpatialRef(), dst_srs)

                # Create in-memory layer with transformed geometries
                mem_driver = ogr.GetDriverByName('Memory')
                mem_ds = mem_driver.CreateDataSource('memData')
                mem_layer = mem_ds.CreateLayer('roads', dst_srs, ogr.wkbLineString)

                # Add a field for burn values
                field_defn = ogr.FieldDefn('burn_value', ogr.OFTInteger)
                mem_layer.CreateField(field_defn)
                
                # get transformed geometries in GeoJSON like format or Obj
                for feat in layer:
                    geom = feat.GetGeometryRef().Clone()
                    geom.Transform(ct)
                    out_feature = ogr.Feature(mem_layer.GetLayerDefn())
                    out_feature.SetGeometry(geom)
                    out_feature.SetField('burn_value', 1)  # Value to burn into raster
                    mem_layer.CreateFeature(out_feature)
                    out_feature = None
                    geom = None

            # Check if we have any features
            feature_count = mem_layer.GetFeatureCount()
            if feature_count > 0:
                # Create in-memory raster dataset
                height, width = dst_shape
                
                # Convert rasterio transform to GDAL geotransform
                # rasterio: (x_offset, x_scale, xy_skew, y_offset, yx_skew, y_scale)
                # GDAL: (x_offset, x_scale, xy_skew, y_offset, yx_skew, y_scale)
                geotransform = (
                    dst_transform.c,    # x_offset (top-left x)
                    dst_transform.a,    # x_scale (pixel width)
                    dst_transform.b,    # xy_skew (rotation)
                    dst_transform.f,    # y_offset (top-left y)
                    dst_transform.d,    # yx_skew (rotation) 
                    dst_transform.e     # y_scale (pixel height, usually negative)
                )
                
                # Create raster dataset in memory
                raster_driver = gdal.GetDriverByName('MEM')
                raster_ds = raster_driver.Create('', width, height, 1, gdal.GDT_Byte)
                raster_ds.SetGeoTransform(geotransform)
                raster_ds.SetProjection(dst_srs.ExportToWkt())
                
                # Get the raster band
                band = raster_ds.GetRasterBand(1)
                band.SetNoDataValue(0)
                band.Fill(0)  # Initialize with 0
                
                # Burn value of 1 for all features
                gdal.RasterizeLayer(raster_ds, [1], mem_layer, 
                                    burn_values=[1], options=['ALL_TOUCHED=TRUE'])

                # Read the rasterized data
                rasterize_roads = band.ReadAsArray()

                # Create RGB raster
                rgb_roads = np.zeros((3, rasterize_roads.shape[0], rasterize_roads.shape[1]), dtype=np.uint8)
                
                # Set values in the 3D array based on the binary_array
                rgb_roads[0, :, :] = rasterize_roads * 255
                rgb_roads[1, :, :] = rasterize_roads * 255
                rgb_roads[2, :, :] = rasterize_roads * 255
            else:
                # if no shapes to rasterize
                logger.info(f'No road geometries to rasterize, saving blank roads raster.')
                rasterize_roads = np.zeros(dst_shape, dtype=np.uint8)
                rgb_roads = np.zeros((3, *dst_shape), dtype=np.uint8)
        except Exception as e:
            raise e
        finally:
            if mem_ds:
                mem_ds = None
            if raster_ds:
                raster_ds = None
    else:
        logger.info(f'State file does not exist for {state}, saving blank raster.')
        rasterize_roads = np.zeros(dst_shape, dtype=np.uint8)
        rgb_roads = np.zeros((3, *dst_shape), dtype=np.uint8)

    with rasterio.open(dir_path / f'{save_as}.tif', 'w', driver='Gtiff', count=1, height=rasterize_roads.shape[-2], width=rasterize_roads.shape[-1], 
                       crs=dst_crs, dtype=rasterize_roads.dtype, transform=dst_transform, nodata=0) as dst:
        dst.write(rasterize_roads, 1)

    with rasterio.open(dir_path / f'{save_as}_cmap.tif', 'w', driver='Gtiff', count=3, height=rgb_roads.shape[-2], width=rgb_roads.shape[-1], 
                       crs=dst_crs, dtype=rgb_roads.dtype, transform=dst_transform, nodata=None) as dst:
        dst.write(rgb_roads)

def pipeline_dem(dir_path: Path, save_as, dst_shape, dst_crs, dst_transform, bounds, cfg):
    """Generates Digital Elevation Map raster given destination raster properties and bounding box.

    Note to developers: slope raster generation removed in favor of calling np.gradient on dem tile during preprocessing.

    Parameters
    ----------
    dir_path : Path
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
    cfg : DictConfig
        Configuration object.
    """
    # buffer here accounts for missing edges after reprojection!
    buffer = 0.02
    bounds = (bounds[0] - buffer, bounds[1] - buffer, bounds[2] + buffer, bounds[3] + buffer)
    
    lst = []
    for lat, long, file in get_elevation_nw(cfg):
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
    with rasterio.open(dir_path / f'{save_as}.tif', 'w', driver='Gtiff', count=1, height=destination.shape[-2], width=destination.shape[-1], crs=dst_crs, dtype=destination.dtype, transform=dst_transform, nodata=no_data) as dst:
        dst.write(destination, 1)

    # save dem cmap
    with rasterio.open(dir_path / f'{save_as}_cmap.tif', 'w', driver='Gtiff', count=3, height=dem_cmap.shape[-2], width=dem_cmap.shape[-1], crs=dst_crs, dtype=dem_cmap.dtype, transform=dst_transform, nodata=None) as dst:
        dst.write(dem_cmap)

def pipeline_flowlines(dir_path: Path, save_as, dst_shape, dst_crs, dst_transform, prism_bbox, cfg,
exact_fcodes=['46000', '46003', '46006', '46007', '55800', '33600', '33601', '33603', '33400', '42801', '42802', '42805', '42806', '42809']):
    """Generates raster with burned in geometries of flowlines given destination raster properties.

    NOTE: FCode for flowlines can be referenced here: https://files.hawaii.gov/dbedt/op/gis/data/NHD%20Complete%20FCode%20Attribute%20Value%20List.pdf

    Parameters
    ----------
    dir_path : Path
        Path for saving generated raster. 
    save_as : str
        Name of file to be saved (no file extension!).
    dst_shape : (int, int)
        Shape of output raster.
    dst_crs : str
        Coordinate reference system of output raster.
    dst_transform : rasterio.affine.Affine()
        Transformation matrix for mapping pixel coordinates to coordinate system of output raster.
    prism_bbox : (float, float, float, float)
        Tuple in the order minx, miny, maxx, maxy, representing bounding box, 
        in PRISM CRS.
    cfg : DictConfig
        Configuration object.
    exact_fcodes : list[str], optional
        List of flowline feature FCodes to filter for e.g. ["42801", "42805", "46002"].
    """
    logger = logging.getLogger('events')
    where_clause = f"FCode IN ({','.join(exact_fcodes)})"
    minx, miny, maxx, maxy = prism_bbox

    mem_ds = None
    raster_ds = None

    try:
        # Create in-memory layer with transformed geometries
        dst_srs = osr.SpatialReference()
        dst_srs.SetFromUserInput(dst_crs)
        mem_driver = ogr.GetDriverByName('Memory')
        mem_ds = mem_driver.CreateDataSource('memData')
        mem_layer = mem_ds.CreateLayer('flowlines', dst_srs, ogr.wkbMultiLineString)
        
        # Add a field for burn values
        field_defn = ogr.FieldDefn('burn_value', ogr.OFTInteger)
        mem_layer.CreateField(field_defn)

        for code in GetHU4Codes(prism_bbox, cfg):
            if (Path(cfg.paths.nhd_dir) / f'NHDPLUS_H_{code}_HU4_GDB.gdb').exists():
                with gdal.OpenEx(Path(cfg.paths.nhd_dir) / f'NHDPLUS_H_{code}_HU4_GDB.gdb') as ds:
                    layer = ds.GetLayerByName('NHDFlowline')
                    layer.SetSpatialFilterRect(minx, miny, maxx, maxy)
                    if exact_fcodes:
                        layer.SetAttributeFilter(where_clause)

                    ct = osr.CoordinateTransformation(layer.GetSpatialRef(), dst_srs)

                    for feat in layer:
                        geom = feat.GetGeometryRef().Clone()
                        geom.Transform(ct)
                        out_feature = ogr.Feature(mem_layer.GetLayerDefn())
                        out_feature.SetGeometry(geom)
                        out_feature.SetField('burn_value', 1)  # Value to burn into raster
                        mem_layer.CreateFeature(out_feature)
                        out_feature = None
                        geom = None
            else:
                logger.info(f'NHD file does not exist for code {code}, skipping.')
        
        # Check if we have any features
        feature_count = mem_layer.GetFeatureCount()
        if feature_count > 0:
            # Create in-memory raster dataset
            height, width = dst_shape
            
            # Convert rasterio transform to GDAL geotransform
            # rasterio: (x_offset, x_scale, xy_skew, y_offset, yx_skew, y_scale)
            # GDAL: (x_offset, x_scale, xy_skew, y_offset, yx_skew, y_scale)
            geotransform = (
                dst_transform.c,    # x_offset (top-left x)
                dst_transform.a,    # x_scale (pixel width)
                dst_transform.b,    # xy_skew (rotation)
                dst_transform.f,    # y_offset (top-left y)
                dst_transform.d,    # yx_skew (rotation) 
                dst_transform.e     # y_scale (pixel height, usually negative)
            )
            
            # Create raster dataset in memory
            raster_driver = gdal.GetDriverByName('MEM')
            raster_ds = raster_driver.Create('', width, height, 1, gdal.GDT_Byte)
            raster_ds.SetGeoTransform(geotransform)
            raster_ds.SetProjection(dst_srs.ExportToWkt())
            
            # Get the raster band
            band = raster_ds.GetRasterBand(1)
            band.SetNoDataValue(0)
            band.Fill(0)  # Initialize with 0
            
            # Burn value of 1 for all features
            gdal.RasterizeLayer(raster_ds, [1], mem_layer, 
                                burn_values=[1], options=['ALL_TOUCHED=TRUE'])

            # Read the rasterized data
            flowlines = band.ReadAsArray()

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
    except Exception as e:
        raise e
    finally:
        if mem_ds:
            mem_ds = None
        if raster_ds:
            raster_ds = None
    
    # flowlines raw
    with rasterio.open(dir_path / f'{save_as}.tif', 'w', driver='Gtiff', count=1, height=flowlines.shape[-2], width=flowlines.shape[-1], 
                       crs=dst_crs, dtype=flowlines.dtype, transform=dst_transform, nodata=0) as dst:
        dst.write(flowlines, 1)

    # flowlines cmap
    with rasterio.open(dir_path / f'{save_as}_cmap.tif', 'w', driver='Gtiff', count=3, height=rgb_flowlines.shape[-2], width=rgb_flowlines.shape[-1], crs=dst_crs, dtype=rgb_flowlines.dtype, transform=dst_transform, nodata=None) as dst:
        dst.write(rgb_flowlines)

def pipeline_waterbody(dir_path: Path, save_as, dst_shape, dst_crs, dst_transform, prism_bbox, cfg):
    """Generates raster with burned in geometries of waterbodies given destination raster properties.

    Parameters
    ----------
    dir_path : Path
        Path for saving generated raster. 
    save_as : str
        Name of file to be saved (no file extension!).
    dst_shape : (int, int)
        Shape of output raster.
    dst_crs : str
        Coordinate reference system of output raster.
    dst_transform : rasterio.affine.Affine()
        Transformation matrix for mapping pixel coordinates to coordinate system of output raster.
    prism_bbox : (float, float, float, float)
        Tuple in the order minx, miny, maxx, maxy, representing bounding box, 
        in PRISM CRS.
    cfg : DictConfig
        Configuration object.
    """
    logger = logging.getLogger('events')
    # filter out estuary
    where_clause = "FCode <> 49300"

    minx, miny, maxx, maxy = prism_bbox
    mem_ds = None
    raster_ds = None

    try:
        # Create in-memory layer with transformed geometries
        dst_srs = osr.SpatialReference()
        dst_srs.SetFromUserInput(dst_crs)
        mem_driver = ogr.GetDriverByName('Memory')
        mem_ds = mem_driver.CreateDataSource('memData')
        mem_layer = mem_ds.CreateLayer('waterbody', dst_srs, ogr.wkbMultiPolygon)
        
        # Add a field for burn values
        field_defn = ogr.FieldDefn('burn_value', ogr.OFTInteger)
        mem_layer.CreateField(field_defn)

        for code in GetHU4Codes(prism_bbox, cfg):
            if (Path(cfg.paths.nhd_dir) / f'NHDPLUS_H_{code}_HU4_GDB.gdb').exists():
                with gdal.OpenEx(Path(cfg.paths.nhd_dir) / f'NHDPLUS_H_{code}_HU4_GDB.gdb') as ds:
                    layer = ds.GetLayerByName('NHDWaterbody')
                    layer.SetSpatialFilterRect(minx, miny, maxx, maxy)
                    layer.SetAttributeFilter(where_clause)

                    ct = osr.CoordinateTransformation(layer.GetSpatialRef(), dst_srs)

                    for feat in layer:
                        geom = feat.GetGeometryRef().Clone()
                        geom.Transform(ct)
                        out_feature = ogr.Feature(mem_layer.GetLayerDefn())
                        out_feature.SetGeometry(geom)
                        out_feature.SetField('burn_value', 1)  # Value to burn into raster
                        mem_layer.CreateFeature(out_feature)
                        out_feature = None
                        geom = None
            else:
                logger.info(f'NHD file does not exist for code {code}, skipping.')
        
        # Check if we have any features
        feature_count = mem_layer.GetFeatureCount()
        if feature_count > 0:
            # Create in-memory raster dataset
            height, width = dst_shape
            
            # Convert rasterio transform to GDAL geotransform
            # rasterio: (x_offset, x_scale, xy_skew, y_offset, yx_skew, y_scale)
            # GDAL: (x_offset, x_scale, xy_skew, y_offset, yx_skew, y_scale)
            geotransform = (
                dst_transform.c,    # x_offset (top-left x)
                dst_transform.a,    # x_scale (pixel width)
                dst_transform.b,    # xy_skew (rotation)
                dst_transform.f,    # y_offset (top-left y)
                dst_transform.d,    # yx_skew (rotation) 
                dst_transform.e     # y_scale (pixel height, usually negative)
            )
            
            # Create raster dataset in memory
            raster_driver = gdal.GetDriverByName('MEM')
            raster_ds = raster_driver.Create('', width, height, 1, gdal.GDT_Byte)
            raster_ds.SetGeoTransform(geotransform)
            raster_ds.SetProjection(dst_srs.ExportToWkt())
            
            # Get the raster band
            band = raster_ds.GetRasterBand(1)
            band.SetNoDataValue(0)
            band.Fill(0)  # Initialize with 0
            
            # Burn value of 1 for all features
            gdal.RasterizeLayer(raster_ds, [1], mem_layer, 
                                burn_values=[1], options=['ALL_TOUCHED=TRUE'])

            # Read the rasterized data
            waterbody = band.ReadAsArray()

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
    except Exception as e:
        raise e
    finally:
        if mem_ds:
            mem_ds = None
        if raster_ds:
            raster_ds = None
    
    # waterbody raw
    with rasterio.open(dir_path / f'{save_as}.tif', 'w', driver='Gtiff', count=1, height=waterbody.shape[-2], width=waterbody.shape[-1], crs=dst_crs, dtype=waterbody.dtype, transform=dst_transform, nodata=0) as dst:
        dst.write(waterbody, 1)

    # waterbody cmap
    with rasterio.open(dir_path / f'{save_as}_cmap.tif', 'w', driver='Gtiff', count=3, height=rgb_waterbody.shape[-2], width=rgb_waterbody.shape[-1], crs=dst_crs, dtype=rgb_waterbody.dtype, transform=dst_transform, nodata=None) as dst:
        dst.write(rgb_waterbody)

def pipeline_NLCD(dir_path: Path, save_as, year, dst_shape, dst_crs, dst_transform, cfg):
    """Generates raster with NLCD land cover classes. Uses windowed reading of NLCD raster
    for speed (NLCD files are large).
    
    Parameters
    ----------
    dir_path : Path
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
    cfg : DictConfig
        Configuration object.
    """
    # if year is after the most recent year, use the most recent year
    nlcd_range = get_nlcd_range(cfg)
    if year > nlcd_range[1]:
        year = nlcd_range[1]
    elif year < nlcd_range[0]:
        year = nlcd_range[0]

    with rasterio.open(Path(cfg.paths.nlcd_dir) / f'LndCov{year}.tif') as src:
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
    with rasterio.open(dir_path / f'{save_as}.tif', 'w', driver='Gtiff', count=1, height=nlcd_arr.shape[-2], width=nlcd_arr.shape[-1], crs=dst_crs, dtype=nlcd_arr.dtype, transform=dst_transform, nodata=250) as dst:
        dst.write(nlcd_arr, 1)

    # create NLCD colormap
    H, W = nlcd_arr.shape
    rgb_img = np.zeros((H, W, 3), dtype=np.uint8)

    # vectorized mapping
    for code, rgb in NLCD_CODE_TO_RGB.items():
        mask = nlcd_arr == code
        rgb_img[mask] = rgb

    rgb_img = np.transpose(rgb_img, (2, 0, 1))

    with rasterio.open(dir_path / f'{save_as}_cmap.tif', 'w', driver='Gtiff', count=3, height=rgb_img.shape[-2], width=rgb_img.shape[-1], crs=dst_crs, dtype=rgb_img.dtype, transform=dst_transform, nodata=None) as dst:
        dst.write(rgb_img)

def event_sample(stac_provider, event_date, event_precip, prism_bbox, eid, dir_path: Path, cfg, manual_crs=None):
    """Samples S2 imagery for a high precipitation event based on parameters and generates accompanying rasters.
    
    Note to developers: the script simplifies normalization onto a consistent grid by finding any common CRS shared by S2 and S1 products.
    Once it finds a CRS in common, all products that do not share that CRS are thrown out. The first S2 product is cropped using the bounding box
    and its dimensions (width and height) and affine transform are then used as reference for all subsequent normalization. 
    A smarter approach (that doesn't throw out products arbitrarily) would be to choose a CRS as reference and normalize everything using
    rasterio WarpedVRT, though this is not currently implemented.

    Parameters
    ----------
    stac_provider : STACProvider
        STAC provider object.
    event_date : str
        Date of high precipitation event in format YYYYMMDD.
    event_precip: float
        Cumulative daily precipitation in mm on event date.
    prism_bbox : (float, float, float, float)
        Tuple in the order minx, miny, maxx, maxy, representing bounding box.
    eid : str
        Event id.
    dir_path : Path
        Path for sampling folder of events.
    manual_crs : str, optional
        Manual CRS specification for event processing. If provided, this CRS will be used
        instead of automatically selecting (alphabetically) from available products. The CRS
        determines the resulting shape of the generated rasters, so specifying the CRS prevents
        any shape ambiguity. This is especially important for downloading events for previously
        labeled products.

    Returns
    -------
    bool
        Return True if successful, False if unsuccessful event download and processing.
    """
    logger = logging.getLogger('events')
    logger.info('**********************************')
    logger.info('START OF EVENT TASK LOG:')
    logger.info(f'Beginning event {eid} download...')

    minx, miny, maxx, maxy = prism_bbox
    logger.info(f'Event on {event_date} with precipitation {event_precip}mm at bounds: {minx}, {miny}, {maxx}, {maxy}')

    # need to transform box from EPSG 4269 to EPSG 4326 for query
    conversion = transform(PRISM_CRS, SEARCH_CRS, (minx, maxx), (miny, maxy))
    bbox = (conversion[0][0], conversion[1][0], conversion[0][1], conversion[1][1])
    event_dir_path = dir_path / eid
    time_of_interest = get_date_interval(event_date, cfg.sampling.before, cfg.sampling.after)

    # STAC catalog search
    logger.info('Beginning catalog search...')
    items_s2 = stac_provider.search_s2(bbox, time_of_interest)
            
    logger.info('Filtering catalog search results...')
    if len(items_s2) == 0:
        logger.info(f'Zero products from query for date interval {time_of_interest}.')
        return False
    elif not has_date_after_PRISM([item.datetime for item in items_s2], event_date):
        logger.info(f'Products found but only before precipitation event date {event_date}.')
        return False

    # group items by dates in dictionary
    logger.info(f'Checking s2 cloud null percentage...')
    s2_by_date_crs = DateCRSOrganizer()
    for item in items_s2:
        # filter out those w/ cloud null via cloud null percentage checks
        item_crs = get_item_crs(item)
        try:
            coverpercentage = cloud_null_percentage(stac_provider, item, item_crs, prism_bbox)
            logger.info(f'Cloud null percentage for item {item.id}: {coverpercentage}')
        except Exception as err:
            logger.exception(f'Cloud null percentage calculation error for item {item.id}.')
            raise err

        if coverpercentage > cfg.sampling.maxcoverpercentage:
            logger.debug(f'Sample {item.id} near event {event_date}, at {minx}, {miny}, {maxx}, {maxy} rejected due to {coverpercentage}% cloud or null cover.')
            continue

        s2_by_date_crs.add_item(item, item.datetime, item_crs)

    # ensure one CRS still has post-event image after filters
    if s2_by_date_crs.is_empty():
        logger.debug(f'No s2 images left after filtering...')
        return False
    elif not has_date_after_PRISM(s2_by_date_crs.get_dates(), event_date):
        logger.debug(f'No s2 images post event date {event_date} after filtering...')
        return False
    
    # either use specified CRS or choose first CRS in alphabetical order for rasters
    main_crs = s2_by_date_crs.get_all_crs()[0] if manual_crs is None else manual_crs
    logger.debug(f'Using CRS for raster generation: {main_crs}')

    try:
        state = get_state(minx, miny, maxx, maxy)
        if state is None:
            raise Exception(f'State not found for {event_date}, at {minx}, {miny}, {maxx}, {maxy}')
    except Exception as err:
        logger.exception(f'Error fetching state. Id: {eid}.')
        return False

    # raster generation with chosen CRS
    logger.info('Beginning raster generation...')
    event_dir_path.mkdir(parents=True, exist_ok=True)
    s2_date_to_product = dict()
    try:
        # convert prism bbox to main CRS
        conversion = transform(PRISM_CRS, main_crs, (minx, maxx), (miny, maxy))
        cbbox = conversion[0][0], conversion[1][0], conversion[0][1], conversion[1][1]
            
        s2_dates = s2_by_date_crs.get_dates()
        for date in s2_dates:
            logger.debug(f'Processing S2 date: {date}')
            dt = date.strftime('%Y%m%d') 

            # use one item for date in preferred CRS
            item = s2_by_date_crs.get_primary_item_for_date(date, preferred_crs=main_crs)
            
            dst_shape, dst_transform = pipeline_TCI(stac_provider, event_dir_path, f'tci_{dt}_{eid}', main_crs, item, cbbox)
            logger.debug(f'TCI raster completed for {dt}.')
            pipeline_RGB(stac_provider, event_dir_path, f'rgb_{dt}_{eid}', main_crs, item, cbbox)
            logger.debug(f'RGB raster completed for {dt}.')
            pipeline_B08(stac_provider, event_dir_path, f'b08_{dt}_{eid}', main_crs, item, cbbox)
            logger.debug(f'B08 raster completed for {dt}.')
            pipeline_B11(stac_provider, event_dir_path, f'b11_{dt}_{eid}', dst_shape, main_crs, dst_transform, item, cbbox)
            logger.debug(f'B11 raster completed for {dt}.')
            pipeline_B12(stac_provider, event_dir_path, f'b12_{dt}_{eid}', dst_shape, main_crs, dst_transform, item, cbbox)
            logger.debug(f'B12 raster completed for {dt}.')
            pipeline_NDWI(stac_provider, event_dir_path, f'ndwi_{dt}_{eid}', main_crs, item, cbbox)
            logger.debug(f'NDWI raster completed for {dt}.')
            pipeline_SCL(stac_provider, event_dir_path, f'scl_{dt}_{eid}', dst_shape, main_crs, dst_transform, item, cbbox)
            logger.debug(f'SCL raster completed for {dt}.')

            # record S2 product ID for this date
            s2_date_to_product[date.strftime('%Y-%m-%d')] = item.id
            
        logger.debug(f'All S2, B08, NDWI, SCL rasters completed successfully.')

        # save all supplementary rasters in raw and rgb colormap
        pipeline_roads(event_dir_path, f'roads_{eid}', dst_shape, main_crs, dst_transform, state, prism_bbox, cfg)
        logger.debug(f'Roads raster completed successfully.')
        pipeline_dem(event_dir_path, f'dem_{eid}', dst_shape, main_crs, dst_transform, prism_bbox, cfg)
        logger.debug(f'DEM raster completed successfully.')
        pipeline_flowlines(event_dir_path, f'flowlines_{eid}', dst_shape, main_crs, dst_transform, prism_bbox, cfg)
        logger.debug(f'Flowlines raster completed successfully.')
        pipeline_waterbody(event_dir_path, f'waterbody_{eid}', dst_shape, main_crs, dst_transform, prism_bbox, cfg)
        logger.debug(f'Waterbody raster completed successfully.')
        pipeline_NLCD(event_dir_path, f'nlcd_{eid}', int(eid[:4]), dst_shape, main_crs, dst_transform, cfg)
        logger.debug(f'NLCD raster completed successfully.')
    except Exception as err:
        logger.error(f'Raster generation error: {err}, {type(err)}')
        logger.error(f'Raster generation failed for {event_date}, at {minx}, {miny}, {maxx}, {maxy}. Id: {eid}. Removing directory and contents.')
        shutil.rmtree(event_dir_path)
        raise err
    
    # validate raster shapes, CRS, transforms
    result = validate_event_rasters(event_dir_path, logger=logger)
    if not result.is_valid:
        logger.error(f'Raster validation failed for event {eid}. Removing directory and contents.')
        # shutil.rmtree(dir_path) - for now do not delete!
        raise Exception(f'Raster validation failed for event {eid}.')
        
    # lastly generate metadata file
    logger.info(f'Generating metadata file...')
    metadata = {
        "Download Date": CURRENT_DATE,
        "Sample ID": eid,
        "Precipitation Event Date": datetime.strptime(event_date, '%Y%m%d').strftime('%Y-%m-%d'),
        "Cumulative Daily Precipitation (mm)": float(event_precip),
        "CRS": main_crs,
        "State": state,
        "Bounding Box": {
            "minx": minx,
            "miny": miny,
            "maxx": maxx,
            "maxy": maxy
        },
        "S2 Products": s2_date_to_product,
        "Max Cloud/Nodata Cover Percentage (%)": cfg.sampling.maxcoverpercentage,
        "Source": cfg.sampling.source
    }

    with open(event_dir_path / 'metadata.json', "w") as json_file:
        json.dump(metadata, json_file, indent=4)
    
    logger.info('Metadata and raster generation completed. Event finished.')
    return True

def get_default_dir_name(threshold: int, days_before: int, days_after: int, maxcoverpercentage: int, region: str = None) -> str:
    """Default directory name."""
    if region is None:
        return f's2_{threshold}_{days_before}_{days_after}_{maxcoverpercentage}/'
    else:
        return f's2_{region}_{threshold}_{days_before}_{days_after}_{maxcoverpercentage}/'

@hydra.main(version_base=None, config_path="pkg://configs", config_name="config.yaml")
def main(cfg: DictConfig) -> None:
    """
    Samples imagery of events queried from PRISM using a given minimum precipitation threshold.
    Downloaded samples will contain multispectral data from within specified interval of event date, their respective
    NDWI rasters. Samples will also have a raster of roads from TIGER roads dataset, DEM raster from USGS, 
    flowlines and waterbody rasters from NHDPlus dataset. All rasters will be 4km x 4km at 10m resolution.

    Manual file
    -----------
    Sometimes flood events are located outside high precipitation tiles. Use manual file to circumvent precipitation
    filter and sample from specific PRISM indices.

    Regions
    -------
    Regions mask the PRISM cell grid to only sample from specific regions.

    Note: In the future if more L2A data become available, some previously processed and skipped events become viable.
    In that case do not use history object during run.

    Parameters
    ----------
    cfg : DictConfig
        Configuration object.
    
    cfg.sampling parameters
    -----------------------
    threshold : int
    days_before : int
        Number of days of interest before precipitation event.
    days_after : int
        Number of days of interest following precipitation event.
    cloudcoverpercentage : (int, int)
        Desired cloud cover percentage range for querying Copernicus Sentinel-2.
    maxevents: int or None
        Specify a limit to the number of extreme precipitation events to process.
    dir_path : str, optional
        Path to directory where samples will be saved.
    region : str, optional
        Region to sample from.
    manual : str, optional
        Path to text file containing manual event indices in lines with format: time, y, x.
        
    Returns
    -------
    int
    """
    # make directory
    if cfg.sampling.dir_path is None:
        cfg.sampling.dir_path = str(Path(cfg.paths.imagery_dir) / get_default_dir_name(cfg.sampling.threshold, cfg.sampling.before, cfg.sampling.after, cfg.sampling.maxcoverpercentage, cfg.sampling.region))
        Path(cfg.sampling.dir_path).mkdir(parents=True, exist_ok=True)
    else:
        # Create directory if it doesn't exist
        try:
            Path(cfg.sampling.dir_path).mkdir(parents=True, exist_ok=True)
        except (OSError, ValueError) as e:
            print(f"Invalid directory path '{cfg.sampling.dir_path}'. Using default.", file=sys.stderr)
            cfg.sampling.dir_path = str(Path(cfg.paths.imagery_dir) / get_default_dir_name(cfg.sampling.threshold, cfg.sampling.before, cfg.sampling.after, cfg.sampling.maxcoverpercentage, cfg.sampling.region))
            Path(cfg.sampling.dir_path).mkdir(parents=True, exist_ok=True)

        # Ensure trailing slash
        cfg.sampling.dir_path = os.path.join(cfg.sampling.dir_path, '')
    
    # root logger
    rootLogger = setup_logging(cfg.sampling.dir_path, logger_name='main', log_level=logging.DEBUG, mode='w', include_console=False)

    # event logger
    logger = setup_logging(cfg.sampling.dir_path, logger_name='events', log_level=logging.DEBUG, mode='a', include_console=False)

    # log sampling parameters used
    rootLogger.info(
        "S2 sampling parameters used:\n"
        f"  Threshold: {cfg.sampling.threshold}\n"
        f"  Days before precipitation event: {cfg.sampling.before}\n"
        f"  Days after precipitation event: {cfg.sampling.after}\n"
        f"  Max cloud/nodata cover percentage: {cfg.sampling.maxcoverpercentage}\n"
        f"  Max events to sample: {cfg.sampling.maxevents}\n"
        f"  Max runtime: {getattr(cfg.sampling, 'max_runtime', 'Unlimited')}\n"
        f"  Region: {cfg.sampling.region}\n"
        f"  Manual indices: {cfg.sampling.manual}\n"
        f"  Source: {cfg.sampling.source}\n"
    )

    # to track runtime
    max_runtime_seconds = (
        walltime_seconds(cfg.sampling.max_runtime)
        if hasattr(cfg.sampling, 'max_runtime') and cfg.sampling.max_runtime is not None
        else float('inf')
    )
    start_time = time.time()

    # history set of tuples
    history = get_history(Path(cfg.sampling.dir_path) / 'history.pickle')

    # load PRISM data object
    prism_data = PRISMData.from_file(cfg.paths.prism_data)

    # get PRISM event indices and event data
    if cfg.sampling.manual:
        rootLogger.info("Using manual indices...")
        num_candidates, events = get_manual_events(prism_data, history, cfg.sampling.manual, logger=rootLogger)
        rootLogger.info(f"Found {num_candidates} events from {cfg.sampling.manual}.")
    else:
        mask = get_mask(cfg, prism_data.shape)
        rootLogger.info("Finding candidate extreme precipitation events...")
        num_candidates, events = get_extreme_events(prism_data, history, threshold=cfg.sampling.threshold, mask=mask, logger=rootLogger)
        rootLogger.info(f"Found {num_candidates} candidate extreme precipitation events.")

    rootLogger.info("Initializing event sampling...")
    count = 0
    search_count = 0
    alr_completed = 0
    try:
        rootLogger.info(f"Searching through {num_candidates} candidate indices/events...")
        # get stac provider
        stac_provider = get_stac_provider(cfg.sampling.source.lower(),
                                    mpc_api_key=getattr(cfg, "mpc_api_key", None),
                                    aws_access_key_id=getattr(cfg, "aws_access_key_id", None),
                                    aws_secret_access_key=getattr(cfg, "aws_secret_access_key", None),
                                    logger=logger)
        for event_date, event_precip, prism_bbox, eid, indices, crs in events:
            if time.time() - start_time > max_runtime_seconds:
                rootLogger.info(f"Maximum runtime of {getattr(cfg.sampling, 'max_runtime', 'Unlimited')} reached. Stopping event sampling...")
                break

            search_count += 1
            if (Path(cfg.sampling.dir_path) / eid).is_dir():
                if event_completed(Path(cfg.sampling.dir_path) / eid, regex_patterns, pattern_dict, logger=rootLogger):
                    rootLogger.debug(f'Event {eid} index {indices} has already been processed before. Moving on to the next event...')
                    alr_completed += 1
                    history.add(indices)
                    continue
                else:
                    rootLogger.debug(f'Event {eid} index {indices} has already been processed before but unsuccessfully. Reprocessing...')

            max_attempts = 3
            for attempt in range(1, max_attempts + 1):
                try:
                    if event_sample(stac_provider, event_date, event_precip, 
                                    prism_bbox, eid, Path(cfg.sampling.dir_path), cfg, manual_crs=crs):
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
                except MissingAssetError as err:
                    rootLogger.error(f'Missing required asset: {err}. Skipping event...')
                    history.add(indices)
                    break
                except Exception as err:
                    raise err

            # once sampled maxevents, stop pipeline
            if count >= cfg.sampling.maxevents:
                rootLogger.info(f"Maximum number of events = {cfg.sampling.maxevents} reached. Stopping event sampling...")
                break
    except Exception as err:
        rootLogger.exception("Unexpected error during event sampling")
    finally:
        # store all previously processed events
        with open(Path(cfg.sampling.dir_path) / 'history.pickle', 'wb') as f:
            pickle.dump(history, f)

    rootLogger.debug(f"Number of events already completed: {alr_completed}")
    rootLogger.debug(f"Number of successful events sampled from this run: {count}/{search_count}")

if __name__ == '__main__':
    main()
