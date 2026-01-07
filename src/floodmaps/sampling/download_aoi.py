from datetime import datetime
from pathlib import Path
import logging
import numpy as np
import re
import json
from osgeo import gdal, ogr, osr
import geopandas as gpd
from rasterio.transform import array_bounds
from rasterio.windows import from_bounds, Window
from rasterio.warp import reproject, Resampling, transform_bounds
import rasterio.merge
import rasterio
from fiona.transform import transform
from pystac.extensions.projection import ProjectionExtension as pe
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
    setup_logging,
    db_scale,
    get_state,
    colormap_to_rgb,
    NoElevationError,
    crop_to_bounds,
    scl_to_rgb,
    get_item_crs,
    compute_ndwi
)
from floodmaps.utils.stac_providers import get_stac_provider
from floodmaps.utils.validate import validate_event_rasters

NLCD_RANGE = None
ELEVATION_LAT_LONG = None
CURRENT_DATE = datetime.now().strftime('%Y-%m-%d')

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
    logger = logging.getLogger('main')
    visual_name = stac_provider.get_asset_names("s2")["visual"]
    item_href = stac_provider.sign_asset_href(item.assets[visual_name].href)

    max_attempts = 3
    for attempt in range(1, max_attempts + 1):
        try:
            out_image, out_transform = crop_to_bounds(item_href, bbox, dst_crs, nodata=0, resampling=Resampling.bilinear)

            # Check if data is all zeros
            if np.all(out_image == 0):
                raise ValueError("TCI raster is all zeros - likely failed download")

            with rasterio.open(dir_path / f'{save_as}.tif', 'w', driver='Gtiff', count=3, height=out_image.shape[-2], width=out_image.shape[-1], crs=dst_crs, dtype=out_image.dtype, transform=out_transform, nodata=None) as dst:
                dst.write(out_image)

            return (out_image.shape[-2], out_image.shape[-1]), out_transform

        except (rasterio.errors.RasterioIOError, ValueError) as err:
            if attempt == max_attempts:
                raise RuntimeError(f"Failed to process TCI after {max_attempts} attempts: {err}") from err
            else:
                logger.warning(f"TCI processing attempt {attempt}/{max_attempts} failed: {err}. Retrying...")
                time.sleep(2 ** attempt)

def pipeline_SCL(stac_provider, dir_path: Path, save_as, dst_shape, dst_crs, dst_transform, item, bbox, cfg):
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
    cfg : DictConfig
        Configuration object.
    """
    logger = logging.getLogger('main')
    scl_name = stac_provider.get_asset_names("s2")["SCL"]
    item_href = stac_provider.sign_asset_href(item.assets[scl_name].href)

    h, w = dst_shape
    max_attempts = 3
    for attempt in range(1, max_attempts + 1):
        try:
            # Single-step resampling directly to TCI grid for perfect pixel alignment
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

            # Check if data is all zeros
            if np.all(dest == 0):
                raise ValueError("SCL raster is all zeros - likely failed download")

            # save SCL raster with all class values preserved
            with rasterio.open(dir_path / f'{save_as}.tif', 'w', driver='Gtiff', count=1, height=h, width=w, crs=dst_crs, 
                                dtype=dest.dtype, transform=dst_transform, nodata=0) as dst:
                dst.write(dest)
            
            if cfg.sampling.include_cmap:
                # create RGB colormap from SCL using utility function
                rgb_scl = scl_to_rgb(dest[0])
                
                with rasterio.open(dir_path / f'{save_as}_cmap.tif', 'w', driver='Gtiff', count=3, height=rgb_scl.shape[-2], width=rgb_scl.shape[-1], 
                                crs=dst_crs, dtype=rgb_scl.dtype, transform=dst_transform, nodata=None) as dst:
                    dst.write(rgb_scl)
                    
            break

        except (rasterio.errors.RasterioIOError, ValueError) as err:
            if attempt == max_attempts:
                raise RuntimeError(f"Failed to process SCL after {max_attempts} attempts: {err}") from err
            else:
                logger.warning(f"SCL processing attempt {attempt}/{max_attempts} failed: {err}. Retrying...")
                time.sleep(2 ** attempt)

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
    logger = logging.getLogger('main')
    b02_name = stac_provider.get_asset_names("s2")["B02"]
    b03_name = stac_provider.get_asset_names("s2")["B03"]
    b04_name = stac_provider.get_asset_names("s2")["B04"]
    b02_item_href = stac_provider.sign_asset_href(item.assets[b02_name].href) # B
    b03_item_href = stac_provider.sign_asset_href(item.assets[b03_name].href) # G
    b04_item_href = stac_provider.sign_asset_href(item.assets[b04_name].href) # R

    max_attempts = 3
    for attempt in range(1, max_attempts + 1):
        try:
            # stack the three bands as rgb channels
            blue_image, out_transform = crop_to_bounds(b02_item_href, bbox, dst_crs, nodata=0, resampling=Resampling.bilinear)
            green_image, _ = crop_to_bounds(b03_item_href, bbox, dst_crs, nodata=0, resampling=Resampling.bilinear)
            red_image, _ = crop_to_bounds(b04_item_href, bbox, dst_crs, nodata=0, resampling=Resampling.bilinear)

            # Check if any channel is all zeros
            if np.all(blue_image == 0):
                raise ValueError("RGB Blue (B02) channel is all zeros - likely failed download")
            if np.all(green_image == 0):
                raise ValueError("RGB Green (B03) channel is all zeros - likely failed download")
            if np.all(red_image == 0):
                raise ValueError("RGB Red (B04) channel is all zeros - likely failed download")

            out_image = np.vstack((red_image, green_image, blue_image))

            with rasterio.open(dir_path / f'{save_as}.tif', 'w', driver='Gtiff', count=3, height=out_image.shape[-2], width=out_image.shape[-1], crs=dst_crs, dtype=out_image.dtype, transform=out_transform, nodata=0) as dst:
                dst.write(out_image)

            break

        except (rasterio.errors.RasterioIOError, ValueError) as err:
            if attempt == max_attempts:
                raise RuntimeError(f"Failed to process RGB after {max_attempts} attempts: {err}") from err
            else:
                logger.warning(f"RGB processing attempt {attempt}/{max_attempts} failed: {err}. Retrying...")
                time.sleep(2 ** attempt)

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
    logger = logging.getLogger('main')
    b08_name = stac_provider.get_asset_names("s2")["B08"]
    item_href = stac_provider.sign_asset_href(item.assets[b08_name].href)

    max_attempts = 3
    for attempt in range(1, max_attempts + 1):
        try:
            out_image, out_transform = crop_to_bounds(item_href, bbox, dst_crs, nodata=0, resampling=Resampling.bilinear)

            # Check if data is all zeros
            if np.all(out_image == 0):
                raise ValueError("B08 raster is all zeros - likely failed download")

            with rasterio.open(dir_path / f'{save_as}.tif', 'w', driver='Gtiff', count=1, height=out_image.shape[-2], width=out_image.shape[-1], crs=dst_crs, dtype=out_image.dtype, transform=out_transform, nodata=0) as dst:
                dst.write(out_image)

            break

        except (rasterio.errors.RasterioIOError, ValueError) as err:
            if attempt == max_attempts:
                raise RuntimeError(f"Failed to process B08 after {max_attempts} attempts: {err}") from err
            else:
                logger.warning(f"B08 processing attempt {attempt}/{max_attempts} failed: {err}. Retrying...")
                time.sleep(2 ** attempt)

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
    logger = logging.getLogger('main')
    b11_name = stac_provider.get_asset_names("s2")["B11"]
    item_href = stac_provider.sign_asset_href(item.assets[b11_name].href)

    h, w = dst_shape
    max_attempts = 3
    for attempt in range(1, max_attempts + 1):
        try:
            # Single-step resampling directly to TCI grid for perfect pixel alignment
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

            # Check if data is all zeros
            if np.all(dest == 0):
                raise ValueError("B11 raster is all zeros - likely failed download")

            with rasterio.open(dir_path / f'{save_as}.tif', 'w', driver='Gtiff', count=1, height=h, width=w, crs=dst_crs, dtype=dest.dtype, transform=dst_transform, nodata=0) as dst:
                dst.write(dest)

            break

        except (rasterio.errors.RasterioIOError, ValueError) as err:
            if attempt == max_attempts:
                raise RuntimeError(f"Failed to process B11 after {max_attempts} attempts: {err}") from err
            else:
                logger.warning(f"B11 processing attempt {attempt}/{max_attempts} failed: {err}. Retrying...")
                time.sleep(2 ** attempt)

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
    logger = logging.getLogger('main')
    b12_name = stac_provider.get_asset_names("s2")["B12"]
    item_href = stac_provider.sign_asset_href(item.assets[b12_name].href)

    h, w = dst_shape
    max_attempts = 3
    for attempt in range(1, max_attempts + 1):
        try:
            # Single-step resampling directly to TCI grid for perfect pixel alignment
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

            # Check if data is all zeros
            if np.all(dest == 0):
                raise ValueError("B12 raster is all zeros - likely failed download")

            with rasterio.open(dir_path / f'{save_as}.tif', 'w', driver='Gtiff', count=1, height=h, width=w, crs=dst_crs, dtype=dest.dtype, transform=dst_transform, nodata=0) as dst:
                dst.write(dest)

            break

        except (rasterio.errors.RasterioIOError, ValueError) as err:
            if attempt == max_attempts:
                raise RuntimeError(f"Failed to process B12 after {max_attempts} attempts: {err}") from err
            else:
                logger.warning(f"B12 processing attempt {attempt}/{max_attempts} failed: {err}. Retrying...")
                time.sleep(2 ** attempt)

def pipeline_NDWI(stac_provider, dir_path: Path, save_as, dst_crs, item, bbox, cfg):
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
    item: Item
        PyStac Item object.
    bbox : (float, float, float, float)
        Tuple in the order minx, miny, maxx, maxy, representing bounding box,
        should be in CRS specified by dst_crs.
    cfg : DictConfig
        Configuration object.
    """
    logger = logging.getLogger('main')
    b03_name = stac_provider.get_asset_names("s2")["B03"]
    b08_name = stac_provider.get_asset_names("s2")["B08"]
    b03_item_href = stac_provider.sign_asset_href(item.assets[b03_name].href)
    b08_item_href = stac_provider.sign_asset_href(item.assets[b08_name].href)

    max_attempts = 3
    for attempt in range(1, max_attempts + 1):
        try:
            out_image1, _ = crop_to_bounds(b03_item_href, bbox, dst_crs, nodata=0, resampling=Resampling.bilinear)
            green = out_image1[0].astype(np.int32)

            # Check if B03 is all zeros
            if np.all(green == 0):
                raise ValueError("NDWI B03 (green) band is all zeros - likely failed download")

            out_image2, out_transform = crop_to_bounds(b08_item_href, bbox, dst_crs, nodata=0, resampling=Resampling.bilinear)
            nir = out_image2[0].astype(np.int32)

            # Check if B08 is all zeros
            if np.all(nir == 0):
                raise ValueError("NDWI B08 (NIR) band is all zeros - likely failed download")

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
            
            if cfg.sampling.include_cmap:
                # before writing to file, we will make matplotlib colormap!
                ndwi_colored = colormap_to_rgb(ndwi, cmap='seismic_r', r=(-1.0, 1.0), no_data=-999999)

                # nodata should not be set for cmap files
                with rasterio.open(dir_path / f'{save_as}_cmap.tif', 'w', driver='Gtiff', count=3, height=ndwi_colored.shape[-2], width=ndwi_colored.shape[-1], crs=dst_crs, dtype=ndwi_colored.dtype, transform=out_transform, nodata=None) as dst:
                    dst.write(ndwi_colored)

            break

        except (rasterio.errors.RasterioIOError, ValueError) as err:
            if attempt == max_attempts:
                raise RuntimeError(f"Failed to process NDWI after {max_attempts} attempts: {err}") from err
            else:
                logger.warning(f"NDWI processing attempt {attempt}/{max_attempts} failed: {err}. Retrying...")
                time.sleep(2 ** attempt)

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
                if cfg.sampling.include_cmap:
                    rgb_roads = np.zeros((3, rasterize_roads.shape[0], rasterize_roads.shape[1]), dtype=np.uint8)

                    # Set values in the 3D array based on the binary_array
                    rgb_roads[0, :, :] = rasterize_roads * 255
                    rgb_roads[1, :, :] = rasterize_roads * 255
                    rgb_roads[2, :, :] = rasterize_roads * 255
            else:
                # if no shapes to rasterize
                logger.info(f'No road geometries to rasterize, saving blank roads raster.')
                rasterize_roads = np.zeros(dst_shape, dtype=np.uint8)
                if cfg.sampling.include_cmap:
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
        if cfg.sampling.include_cmap:
            rgb_roads = np.zeros((3, *dst_shape), dtype=np.uint8)

    with rasterio.open(dir_path / f'{save_as}.tif', 'w', driver='Gtiff', count=1, height=rasterize_roads.shape[-2], width=rasterize_roads.shape[-1],
                       crs=dst_crs, dtype=rasterize_roads.dtype, transform=dst_transform, nodata=0) as dst:
        dst.write(rasterize_roads, 1)

    if cfg.sampling.include_cmap:
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

    # save dem raw
    with rasterio.open(dir_path / f'{save_as}.tif', 'w', driver='Gtiff', count=1, height=destination.shape[-2], width=destination.shape[-1], crs=dst_crs, dtype=destination.dtype, transform=dst_transform, nodata=no_data) as dst:
        dst.write(destination, 1)

    if cfg.sampling.include_cmap:
        # for DEM use grayscale!
        dem_cmap = colormap_to_rgb(destination, cmap='gray', no_data=no_data)

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
            if cfg.sampling.include_cmap:
                rgb_flowlines = np.zeros((3, flowlines.shape[0], flowlines.shape[1]), dtype=np.uint8)

                # Set values in the 3D array based on the binary_array
                rgb_flowlines[0, :, :] = flowlines * 255
                rgb_flowlines[1, :, :] = flowlines * 255
                rgb_flowlines[2, :, :] = flowlines * 255
        else:
            # if no shapes to rasterize
            flowlines = np.zeros(dst_shape, dtype=np.uint8)
            if cfg.sampling.include_cmap:
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
    if cfg.sampling.include_cmap:
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

            if cfg.sampling.include_cmap:
                # Create RGB raster
                rgb_waterbody = np.zeros((3, waterbody.shape[0], waterbody.shape[1]), dtype=np.uint8)

                # Set values in the 3D array based on the binary_array
                rgb_waterbody[0, :, :] = waterbody * 255
                rgb_waterbody[1, :, :] = waterbody * 255
                rgb_waterbody[2, :, :] = waterbody * 255
        else:
            # if no shapes to rasterize
            waterbody = np.zeros(dst_shape, dtype=np.uint8)

            if cfg.sampling.include_cmap:
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
    if cfg.sampling.include_cmap:
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
    if cfg.sampling.include_cmap:
        H, W = nlcd_arr.shape
        rgb_img = np.zeros((H, W, 3), dtype=np.uint8)

        # vectorized mapping
        for code, rgb in NLCD_CODE_TO_RGB.items():
            mask = nlcd_arr == code
            rgb_img[mask] = rgb

        rgb_img = np.transpose(rgb_img, (2, 0, 1))

        with rasterio.open(dir_path / f'{save_as}_cmap.tif', 'w', driver='Gtiff', count=3, height=rgb_img.shape[-2], width=rgb_img.shape[-1], crs=dst_crs, dtype=rgb_img.dtype, transform=dst_transform, nodata=None) as dst:
            dst.write(rgb_img)


def pipeline_S1(stac_provider, dir_path: Path, save_as, dst_crs, item, bbox, cfg):
    """Generates dB scale raster of SAR data in VV and VH polarizations.

    Note: This function returns the shape and transform as opposed to other sampling scripts
    because we may need to generate supplementary rasters even when only S1 and no S2 data is
    downloaded for inferencing.

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
        PyStac Item object
    bbox : (float, float, float, float)
        Tuple in the order minx, miny, maxx, maxy, representing bounding box,
        should be in CRS specified by dst_crs.
    cfg : DictConfig
        Configuration object.

    Returns
    -------
    shape : (int, int)
        Shape of the raster array.
    transform : rasterio.affine.Affine()
        Transformation matrix for mapping pixel coordinates in dest to coordinate system.
    """
    logger = logging.getLogger('main')
    vv_name = stac_provider.get_asset_names("s1")["vv"]
    vh_name = stac_provider.get_asset_names("s1")["vh"]
    item_hrefs_vv = stac_provider.sign_asset_href(item.assets[vv_name].href)
    item_hrefs_vh = stac_provider.sign_asset_href(item.assets[vh_name].href)

    max_attempts = 3
    for attempt in range(1, max_attempts + 1):
        try:
            out_image_vv, out_transform_vv = crop_to_bounds(item_hrefs_vv, bbox, dst_crs, nodata=0, resampling=Resampling.bilinear)
            out_image_vh, out_transform_vh = crop_to_bounds(item_hrefs_vh, bbox, dst_crs, nodata=0, resampling=Resampling.bilinear)

            # Check if VV polarization is all zeros
            if np.all(out_image_vv == 0):
                raise ValueError("S1 VV polarization is all zeros - likely failed download")

            # Check if VH polarization is all zeros
            if np.all(out_image_vh == 0):
                raise ValueError("S1 VH polarization is all zeros - likely failed download")

            with rasterio.open(dir_path / f'{save_as}_vv.tif', 'w', driver='Gtiff', count=1, height=out_image_vv.shape[-2], width=out_image_vv.shape[-1], crs=dst_crs, dtype=out_image_vv.dtype, transform=out_transform_vv, nodata=-9999) as dst:
                db_vv = db_scale(out_image_vv[0], no_data=-9999)
                dst.write(db_vv, 1)

            with rasterio.open(dir_path / f'{save_as}_vh.tif', 'w', driver='Gtiff', count=1, height=out_image_vh.shape[-2], width=out_image_vh.shape[-1], crs=dst_crs, dtype=out_image_vh.dtype, transform=out_transform_vh, nodata=-9999) as dst:
                db_vh = db_scale(out_image_vh[0], no_data=-9999)
                dst.write(db_vh, 1)

            # color maps
            if cfg.sampling.include_cmap:
                img_vv_cmap = colormap_to_rgb(db_vv, cmap='gray', no_data=-9999)
                with rasterio.open(dir_path / f'{save_as}_vv_cmap.tif', 'w', driver='Gtiff', count=3, height=img_vv_cmap.shape[-2], width=img_vv_cmap.shape[-1], crs=dst_crs, dtype=np.uint8, transform=out_transform_vv, nodata=None) as dst:
                    # get color map
                    dst.write(img_vv_cmap)

                img_vh_cmap = colormap_to_rgb(db_vh, cmap='gray', no_data=-9999)
                with rasterio.open(dir_path / f'{save_as}_vh_cmap.tif', 'w', driver='Gtiff', count=3, height=img_vh_cmap.shape[-2], width=img_vh_cmap.shape[-1], crs=dst_crs, dtype=np.uint8, transform=out_transform_vh, nodata=None) as dst:
                    dst.write(img_vh_cmap)

            return (out_image_vv.shape[-2], out_image_vv.shape[-1]), out_transform_vv

        except (rasterio.errors.RasterioIOError, ValueError) as err:
            if attempt == max_attempts:
                raise RuntimeError(f"Failed to process S1 after {max_attempts} attempts: {err}") from err
            else:
                logger.warning(f"S1 processing attempt {attempt}/{max_attempts} failed: {err}. Retrying...")
                time.sleep(2 ** attempt)

def sar_missing_percentage(stac_provider, item, item_crs, bbox):
    """Calculates the percentage of pixels in the bounding box of the SAR image that are missing."""
    vv_name = stac_provider.get_asset_names("s1")["vv"]
    item_href = stac_provider.sign_asset_href(item.assets[vv_name].href)

    conversion = transform(PRISM_CRS, item_crs, (bbox[0], bbox[2]), (bbox[1], bbox[3]))
    img_bbox = conversion[0][0], conversion[1][0], conversion[0][1], conversion[1][1]

    out_image, _ = rasterio.merge.merge([item_href], bounds=img_bbox)
    return int((np.sum(out_image <= 0) / out_image.size) * 100)

def s2_missing_percentage(stac_provider, item, item_crs, bbox):
    """Calculates the percentage of pixels in the bounding box of the SAR image that are missing."""
    rgb_name = stac_provider.get_asset_names("s2")["B02"]
    item_href = stac_provider.sign_asset_href(item.assets[rgb_name].href)

    conversion = transform(PRISM_CRS, item_crs, (bbox[0], bbox[2]), (bbox[1], bbox[3]))
    img_bbox = conversion[0][0], conversion[1][0], conversion[0][1], conversion[1][1]

    out_image, _ = rasterio.merge.merge([item_href], bounds=img_bbox)
    return int((np.sum(out_image == 0) / out_image.size) * 100)

def deduplicate_items_by_date(items, stac_provider, sensor_type, bbox):
    """Keep only one item per date, choosing the one with least missing data.
    
    Parameters
    ----------
    items : list
        List of STAC items.
    stac_provider : STACProvider
        STAC provider object for accessing asset names.
    sensor_type : str
        Either 's2' or 's1' to determine which assets to check.
    bbox : tuple
        Bounding box in PRISM CRS EPSG:4269.
    
    Returns
    -------
    list
        List of items with only one item per date, selected by least missing percentage.
    """
    date_to_items = {}
    for item in items:
        date_str = item.datetime.strftime('%Y%m%d')
        if date_str not in date_to_items:
            date_to_items[date_str] = []
        date_to_items[date_str].append(item)
    
    # For each date, select the item with least missing data
    selected_items = []
    
    for date_str, date_items in date_to_items.items():
        if len(date_items) == 1:
            selected_items.append(date_items[0])
        else:
            # Select item with minimum missing percentage
            if sensor_type == 's2':
                best_item = min(date_items, key=lambda x: s2_missing_percentage(stac_provider, x, get_item_crs(x), bbox))
            else:  # s1
                best_item = min(date_items, key=lambda x: sar_missing_percentage(stac_provider, x, get_item_crs(x), bbox))
            selected_items.append(best_item)
    
    return selected_items

def download_area(stac_provider, bbox, dir_path, area_id, start_date, end_date, 
                  product_ids_s2, product_ids_s1, crs, cfg):
    """Downloads S2 and S1 imagery for a specific study area within a date range.

    Downloads all S2 and S1 products within the specified date interval. For each date,
    selects only one product of each type (S2/S1) based on least missing data percentage.
    If product ID lists are provided, only downloads those specific products.

    Parameters
    ----------
    stac_provider : STACProvider
        STAC provider object.
    bbox : tuple
        Bounding box in PRISM CRS EPSG:4269.
    dir_path : str or Path
        Output directory for downloaded data.
    area_id : str
        Area identifier for file naming.
    start_date : str
        Start date of interval (YYYY-MM-DD).
    end_date : str
        End date of interval (YYYY-MM-DD).
    product_ids_s2 : list
        List of S2 product IDs to filter for (empty list = all products).
    product_ids_s1 : list
        List of S1 product IDs to filter for (empty list = all products).
    crs : str or None
        CRS for output rasters (None = auto-detect from first product).
    cfg : DictConfig
        Configuration object for system-wide settings (paths, include_cmap, etc.).

    Returns
    -------
    bool
        Return True if successful, False if unsuccessful download and processing.
    """
    logger = logging.getLogger('main')
    minx, miny, maxx, maxy = bbox # assume bbox is in EPSG:4269
    
    # need to transform box from EPSG 4269 to EPSG 4326 for query
    conversion = transform(PRISM_CRS, SEARCH_CRS, (minx, maxx), (miny, maxy))
    search_bbox = (conversion[0][0], conversion[1][0], conversion[0][1], conversion[1][1])
    time_of_interest = f"{start_date}/{end_date}"

    # STAC catalog search
    logger.info('Beginning catalog search...')
    items_s2 = stac_provider.search_s2(search_bbox, time_of_interest).items
    items_s1 = stac_provider.search_s1(search_bbox, time_of_interest, query={"sar:instrument_mode": {"eq": "IW"}}).items

    logger.info('Filtering catalog search results...')
    if len(items_s2) == 0 and len(items_s1) == 0:
        logger.info(f'No S2 and S1 products found for date interval {time_of_interest}.')
        return False
    logger.info(f'Found {len(items_s2)} S2 products and {len(items_s1)} S1 products for date interval {time_of_interest}.')

    try:
        state = get_state(minx, miny, maxx, maxy)
        if state is None:
            raise Exception(f'State not found for bbox: {minx}, {miny}, {maxx}, {maxy}')
    except Exception as err:
        logger.exception(f'Error fetching state for area {area_id}.')
        return False

    logger.info('Beginning download...')
    area_dir_path = Path(dir_path)
    s2_date_to_product = {}  # Map date string to S2 product ID
    s1_date_to_product = {}  # Map date string to S1 product ID
    
    # Handle product ID filtering if specified
    if product_ids_s2:
        logger.info(f'Filtering for S2 product IDs: {product_ids_s2}')
        items_s2 = [item for item in items_s2 if any(pid in item.id for pid in product_ids_s2)]
        logger.info(f'Filtered to {len(items_s2)} S2 products.')
    
    if product_ids_s1:
        logger.info(f'Filtering for S1 product IDs: {product_ids_s1}')
        items_s1 = [item for item in items_s1 if any(pid in item.id for pid in product_ids_s1)]
        logger.info(f'Filtered to {len(items_s1)} S1 products.')
    
    # Deduplicate items by date - keep only one per date based on least missing data
    if items_s2:
        items_s2 = deduplicate_items_by_date(items_s2, stac_provider, 's2', bbox)
    if items_s1:
        items_s1 = deduplicate_items_by_date(items_s1, stac_provider, 's1', bbox)
    
    logger.info(f'After deduplication: {len(items_s2)} S2 products and {len(items_s1)} S1 products.')
    
    # Determine consistent CRS across all products
    all_items = items_s2 + items_s1
    if not all_items:
        logger.error('No products to download after filtering.')
        return False
    
    # Use specified CRS or extract from first item
    main_crs = crs if crs else get_item_crs(all_items[0])
    logger.info(f'Using CRS: {main_crs} for all rasters.')
    
    # Transform bbox to target CRS
    conversion = transform(PRISM_CRS, main_crs, (minx, maxx), (miny, maxy))
    cbbox = conversion[0][0], conversion[1][0], conversion[0][1], conversion[1][1]
    
    # Download S2 products
    for item in items_s2:
        dt = item.datetime.strftime('%Y%m%d')
        logger.info(f'Processing S2 product {item.id} for date {dt}')
        
        dst_shape, dst_transform = pipeline_TCI(stac_provider, area_dir_path, f'tci_{dt}_{area_id}', main_crs, item, cbbox)
        logger.debug(f'TCI raster completed for {item.id} on {dt}.')
        
        pipeline_RGB(stac_provider, area_dir_path, f'rgb_{dt}_{area_id}', main_crs, item, cbbox)
        logger.debug(f'RGB raster completed for {item.id} on {dt}.')
        
        pipeline_B08(stac_provider, area_dir_path, f'b08_{dt}_{area_id}', main_crs, item, cbbox)
        logger.debug(f'B08 raster completed for {item.id} on {dt}.')
        
        pipeline_B11(stac_provider, area_dir_path, f'b11_{dt}_{area_id}', dst_shape, main_crs, dst_transform, item, cbbox)
        logger.debug(f'B11 raster completed for {item.id} on {dt}.')
        
        pipeline_B12(stac_provider, area_dir_path, f'b12_{dt}_{area_id}', dst_shape, main_crs, dst_transform, item, cbbox)
        logger.debug(f'B12 raster completed for {item.id} on {dt}.')
        
        pipeline_NDWI(stac_provider, area_dir_path, f'ndwi_{dt}_{area_id}', main_crs, item, cbbox, cfg)
        logger.debug(f'NDWI raster completed for {item.id} on {dt}.')
        
        pipeline_SCL(stac_provider, area_dir_path, f'scl_{dt}_{area_id}', dst_shape, main_crs, dst_transform, item, cbbox, cfg)
        logger.debug(f'SCL raster completed for {item.id} on {dt}.')

        # Record S2 product ID for this date
        s2_date_to_product[item.datetime.strftime('%Y-%m-%d')] = item.id

    # Download S1 products
    for item in items_s1:
        dt = item.datetime.strftime('%Y%m%d')
        logger.info(f'Processing S1 product {item.id} for date {dt}')

        dst_shape, dst_transform = pipeline_S1(stac_provider, area_dir_path, f'sar_{dt}_{area_id}', main_crs, item, cbbox, cfg)
        logger.debug(f'S1 raster completed for {item.id} on {dt}.')

        # Record S1 product ID for this date
        s1_date_to_product[item.datetime.strftime('%Y-%m-%d')] = item.id
    
    logger.info('Processing supplementary rasters...')
    if not (area_dir_path / f'roads_{area_id}.tif').exists():
        pipeline_roads(area_dir_path, f'roads_{area_id}', dst_shape, main_crs, dst_transform, state, bbox, cfg)
        logger.debug(f'Roads raster completed successfully.')
    if not (area_dir_path / f'dem_{area_id}.tif').exists():
        pipeline_dem(area_dir_path, f'dem_{area_id}', dst_shape, main_crs, dst_transform, bbox, cfg)
        logger.debug(f'DEM raster completed successfully.')
    if not (area_dir_path / f'flowlines_{area_id}.tif').exists():
        pipeline_flowlines(area_dir_path, f'flowlines_{area_id}', dst_shape, main_crs, dst_transform, bbox, cfg)
        logger.debug(f'Flowlines raster completed successfully.')
    if not (area_dir_path / f'waterbody_{area_id}.tif').exists():
        pipeline_waterbody(area_dir_path, f'waterbody_{area_id}', dst_shape, main_crs, dst_transform, bbox, cfg)
        logger.debug(f'Waterbody raster completed successfully.')
    if not (area_dir_path / f'nlcd_{area_id}.tif').exists():
        # Use start year for NLCD
        start_year = int(start_date[:4])
        pipeline_NLCD(area_dir_path, f'nlcd_{area_id}', start_year, dst_shape, main_crs, dst_transform, cfg)
        logger.debug(f'NLCD raster completed successfully.')

    # validate raster shapes, CRS, transforms
    result = validate_event_rasters(area_dir_path, logger=logger)
    if not result.is_valid:
        logger.error(f'Raster validation failed for area {area_id}.')
        raise Exception(f'Raster validation failed for area {area_id}.')

    # Generate metadata file
    logger.info(f'Generating metadata file...')
    metadata = {
        "Download Date": CURRENT_DATE,
        "Sample ID": area_id,
        "Date Range": {
            "Start Date": start_date,
            "End Date": end_date
        },
        "CRS": main_crs,
        "State": state,
        "Bounding Box": {
            "minx": minx,
            "miny": miny,
            "maxx": maxx,
            "maxy": maxy
        },
        "S2 Products": s2_date_to_product,
        "S1 Products": s1_date_to_product,
        "Source": cfg.sampling.source
    }

    metadata_path = area_dir_path / 'metadata.json'
    with open(metadata_path, "w") as json_file:
        json.dump(metadata, json_file, indent=4)

    logger.info('Metadata and raster generation completed. Download finished.')
    return True

def get_bbox_from_shapefile(shapefile, crs='EPSG:4269'):
    """Get the bounding box in CRS from shapefile. Will add a 20% buffer to
    the height and width of the shape for the bbox

    Parameters
    ----------
    shapefile : str
        Path to shapefile.
    crs : str
        CRS string in form EPSG:XXXX.

    Returns
    -------
    bbox : tuple
        Bounding box in CRS.
    """
    # Read the shapefile
    gdf = gpd.read_file(shapefile)
    gdf_proj = gdf.to_crs(crs)

    # Get the total bounding box (minx, miny, maxx, maxy)
    minx, miny, maxx, maxy = gdf_proj.total_bounds

    # Compute width and height
    width = maxx - minx
    height = maxy - miny

    # Add 20% buffer on each side
    buffer_x = 0.2 * width
    buffer_y = 0.2 * height

    expanded_bounds = (
        minx - buffer_x,
        miny - buffer_y,
        maxx + buffer_x,
        maxy + buffer_y,
    )
    return expanded_bounds

def run_download_area(cfg: DictConfig) -> None:
    """Downloads S2 and S1 imagery + accompanying rasters for a specific study area within a date range.
    
    The script downloads all imagery within the specified date interval given a shapefile of ROI.
    For each date, it selects one S2 and one S1 product (if available) based on least missing data.
    To ensure shape consistency across different dates/images, a consistent CRS is enforced.
    
    Parameters
    ----------
    cfg : DictConfig
        Configuration object with required fields:
        - sampling.dir_path: Directory to save downloaded data
        - sampling.shapefile: Path to shapefile defining area of interest
        - sampling.start_date: Start date of interval (YYYY-MM-DD)
        - sampling.end_date: End date of interval (YYYY-MM-DD)
        - sampling.area_id: Identifier for area (optional, defaults to 'aoi')
        - sampling.product_ids_s2: List of S2 product IDs to filter for (optional)
        - sampling.product_ids_s1: List of S1 product IDs to filter for (optional)
        - sampling.crs: CRS to use for all rasters (optional, auto-detected if not provided)
        - sampling.include_cmap: Whether to include color maps for the rasters (optional, defaults to true)
        - sampling.source: mpc or aws (optional, defaults to mpc)
    """
    if cfg.sampling.dir_path is None:
        raise ValueError('sampling.dir_path config parameter is required to specify a directory for download.')
    
    if cfg.sampling.start_date is None or cfg.sampling.end_date is None:
        raise ValueError('sampling.start_date and sampling.end_date are required.')

    # Create directory if it doesn't exist
    Path(cfg.sampling.dir_path).mkdir(parents=True, exist_ok=True)
    
    # make sure dir_path ends with a slash
    if not cfg.sampling.dir_path.endswith('/'):
        cfg.sampling.dir_path += '/'

    # root logger
    rootLogger = setup_logging(cfg.sampling.dir_path, logger_name='main', log_level=logging.DEBUG, mode='w', include_console=False)

    # log sampling parameters used
    area_id = getattr(cfg.sampling, 'area_id', 'aoi')
    product_ids_s2 = getattr(cfg.sampling, 'product_ids_s2', [])
    product_ids_s1 = getattr(cfg.sampling, 'product_ids_s1', [])
    crs = getattr(cfg.sampling, 'crs', None)
    source = getattr(cfg.sampling, 'source', 'mpc')
    
    rootLogger.info(
        "Download area parameters used:\n"
        f"  Area ID: {area_id}\n"
        f"  Start date: {cfg.sampling.start_date}\n"
        f"  End date: {cfg.sampling.end_date}\n"
        f"  S2 Product IDs: {product_ids_s2 if product_ids_s2 else 'All'}\n"
        f"  S1 Product IDs: {product_ids_s1 if product_ids_s1 else 'All'}\n"
        f"  CRS: {cfg.sampling.crs if cfg.sampling.crs else 'Auto-detect'}\n"
        f"  Include color maps: {cfg.sampling.include_cmap}\n"
        f"  Source: {source}\n"
    )

    rootLogger.info("Initializing area download...")
    try:
        # get bbox from shapefile - use PRISM CRS for easy use of pipeline functions
        bbox = get_bbox_from_shapefile(cfg.sampling.shapefile, crs=PRISM_CRS)

        # get stac provider
        stac_provider = get_stac_provider(source.lower(),
                                        mpc_api_key=getattr(cfg, "mpc_api_key", None),
                                        aws_access_key_id=getattr(cfg, "aws_access_key_id", None),
                                        aws_secret_access_key=getattr(cfg, "aws_secret_access_key", None),
                                        logger=rootLogger)

        # download imagery
        max_attempts = 3
        for attempt in range(1, max_attempts + 1):
            try:
                download_area(
                    stac_provider=stac_provider,
                    bbox=bbox,
                    dir_path=cfg.sampling.dir_path,
                    area_id=area_id,
                    start_date=cfg.sampling.start_date,
                    end_date=cfg.sampling.end_date,
                    product_ids_s2=product_ids_s2,
                    product_ids_s1=product_ids_s1,
                    crs=crs,
                    cfg=cfg
                )
                break
            except (rasterio.errors.WarpOperationError, rasterio.errors.RasterioIOError, pystac_client.exceptions.APIError) as err:
                rootLogger.error(f"Connection error: {type(err)}")
                if attempt == max_attempts:
                    rootLogger.error(f'Maximum number of attempts reached, skipping download...')
                else:
                    rootLogger.info(f'Retrying ({attempt}/{max_attempts})...')
            except NoElevationError as err:
                rootLogger.error(f'Elevation file missing, skipping download...')
                break
            except Exception as err:
                raise err
    except Exception as err:
        rootLogger.exception("Unexpected error during area download")

@hydra.main(version_base=None, config_path="pkg://configs", config_name="config.yaml")
def main(cfg: DictConfig) -> None:
    run_download_area(cfg)

if __name__ == '__main__':
    main()