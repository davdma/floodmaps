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

from floodmaps.utils.sampling_utils import (
    PRISM_CRS,
    SEARCH_CRS,
    NLCD_CODE_TO_RGB,
    get_date_interval,
    setup_logging,
    db_scale,
    get_state,
    colormap_to_rgb,
    NoElevationError,
    crop_to_bounds
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
def pipeline_TCI(stac_provider, dir_path, save_as, dst_crs, item, bbox):
    """Generates TCI (True Color Image) raster of S2 multispectral file.

    Parameters
    ----------
    stac_provider : STACProvider
        STAC provider object.
    dir_path : str
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
    item_href = stac_provider.sign_asset_href(item.assets[visual_name].href)

    out_image, out_transform = crop_to_bounds(item_href, bbox, dst_crs, nodata=0, resampling=Resampling.bilinear)

    with rasterio.open(dir_path + save_as + '.tif', 'w', driver='Gtiff', count=3, height=out_image.shape[-2], width=out_image.shape[-1], crs=dst_crs, dtype=out_image.dtype, transform=out_transform, nodata=None) as dst:
        dst.write(out_image)

    return (out_image.shape[-2], out_image.shape[-1]), out_transform

def pipeline_SCL(stac_provider, dir_path, save_as, dst_shape, dst_crs, dst_transform, item, bbox, cfg):
    """Generates Scene Classification Layer raster of S2 multispectral file and resamples to 10m resolution.

    Parameters
    ----------
    stac_provider : STACProvider
        STAC provider object.
    dir_path : str
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
    scl_name = stac_provider.get_asset_names("s2")["SCL"]
    item_href = stac_provider.sign_asset_href(item.assets[scl_name].href)

    out_image, out_transform = crop_to_bounds(item_href, bbox, dst_crs, nodata=0, resampling=Resampling.nearest)
    clouds = np.isin(out_image[0], [8, 9, 10]).astype(np.uint8)

    # need to resample to grid of tci
    h, w = dst_shape[-2:]
    dest = np.zeros((h, w), dtype=clouds.dtype)
    reproject(
        clouds,
        dest,
        src_transform=out_transform,
        src_crs=dst_crs,
        dst_transform=dst_transform,
        dst_crs=dst_crs,
        resampling=Resampling.nearest)

    # only make cloud values (8, 9, 10) 1 everything else 0
    with rasterio.open(dir_path + save_as + '.tif', 'w', driver='Gtiff', count=1, height=h, width=w, crs=dst_crs,
                        dtype=clouds.dtype, transform=dst_transform) as dst:
        dst.write(dest, 1)

    if cfg.inference.include_cmap:
        rgb_clouds = np.zeros((3, h, w), dtype=np.uint8)
        rgb_clouds[0, :, :] = dest * 255
        rgb_clouds[1, :, :] = dest * 255
        rgb_clouds[2, :, :] = dest * 255

        with rasterio.open(dir_path + save_as + '_cmap.tif', 'w', driver='Gtiff', count=3, height=rgb_clouds.shape[-2], width=rgb_clouds.shape[-1],
                        crs=dst_crs, dtype=rgb_clouds.dtype, transform=dst_transform, nodata=None) as dst:
            dst.write(rgb_clouds)

def pipeline_RGB(stac_provider, dir_path, save_as, dst_crs, item, bbox):
    """Generates B02 (B), B03 (G), B04 (R) rasters of S2 multispectral file.

    Parameters
    ----------
    stac_provider : STACProvider
        STAC provider object.
    dir_path : str
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
    b02_item_href = stac_provider.sign_asset_href(item.assets[b02_name].href) # B
    b03_item_href = stac_provider.sign_asset_href(item.assets[b03_name].href) # G
    b04_item_href = stac_provider.sign_asset_href(item.assets[b04_name].href) # R

    # stack the three bands as rgb channels
    blue_image, out_transform = crop_to_bounds(b02_item_href, bbox, dst_crs, nodata=0, resampling=Resampling.bilinear)
    green_image, _ = crop_to_bounds(b03_item_href, bbox, dst_crs, nodata=0, resampling=Resampling.bilinear)
    red_image, _ = crop_to_bounds(b04_item_href, bbox, dst_crs, nodata=0, resampling=Resampling.bilinear)

    out_image = np.vstack((red_image, green_image, blue_image))

    with rasterio.open(dir_path + save_as + '.tif', 'w', driver='Gtiff', count=3, height=out_image.shape[-2], width=out_image.shape[-1], crs=dst_crs, dtype=out_image.dtype, transform=out_transform, nodata=0) as dst:
        dst.write(out_image)

def pipeline_B08(stac_provider, dir_path, save_as, dst_crs, item, bbox):
    """Generates NIR B8 band raster of S2 multispectral file.

    Parameters
    ----------
    stac_provider : STACProvider
        STAC provider object.
    dir_path : str
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
    item_href = stac_provider.sign_asset_href(item.assets[b08_name].href)

    out_image, out_transform = crop_to_bounds(item_href, bbox, dst_crs, nodata=0, resampling=Resampling.bilinear)

    with rasterio.open(dir_path + save_as + '.tif', 'w', driver='Gtiff', count=1, height=out_image.shape[-2], width=out_image.shape[-1], crs=dst_crs, dtype=out_image.dtype, transform=out_transform, nodata=0) as dst:
        dst.write(out_image)

def pipeline_NDWI(stac_provider, dir_path, save_as, dst_crs, item, bbox, cfg):
    """Generates NDWI raster from S2 multispectral files.

    Parameters
    ----------
    stac_provider : STACProvider
        STAC provider object.
    dir_path : str
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
    b03_name = stac_provider.get_asset_names("s2")["B03"]
    b08_name = stac_provider.get_asset_names("s2")["B08"]
    b03_item_href = stac_provider.sign_asset_href(item.assets[b03_name].href)

    out_image1, _ = crop_to_bounds(b03_item_href, bbox, dst_crs, nodata=0, resampling=Resampling.bilinear)
    green = out_image1[0].astype(np.int32)

    b08_item_href = stac_provider.sign_asset_href(item.assets[b08_name].href)

    out_image2, out_transform = crop_to_bounds(b08_item_href, bbox, dst_crs, nodata=0, resampling=Resampling.bilinear)
    nir = out_image2[0].astype(np.int32)

    # calculate ndwi
    ndwi = np.where((green + nir) != 0, (green - nir)/(green + nir), -999999)

    # save raw
    with rasterio.open(dir_path + save_as + '.tif', 'w', driver='Gtiff', count=1, height=ndwi.shape[-2], width=ndwi.shape[-1], crs=dst_crs, dtype=ndwi.dtype, transform=out_transform, nodata=-999999) as dst:
        dst.write(ndwi, 1)

    if cfg.inference.include_cmap:
        # before writing to file, we will make matplotlib colormap!
        ndwi_colored = colormap_to_rgb(ndwi, cmap='seismic_r', r=(-1.0, 1.0), no_data=-999999)

        # nodata should not be set for cmap files
        with rasterio.open(dir_path + save_as + '_cmap.tif', 'w', driver='Gtiff', count=3, height=ndwi_colored.shape[-2], width=ndwi_colored.shape[-1], crs=dst_crs, dtype=ndwi_colored.dtype, transform=out_transform, nodata=None) as dst:
            dst.write(ndwi_colored)

def pipeline_roads(dir_path, save_as, dst_shape, dst_crs, dst_transform, state, prism_bbox, cfg):
    """Generates raster with burned in geometries of roads given destination raster properties.

    Parameters
    ----------
    dir_path : str
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
    mem_ds = None
    raster_ds = None
    minx, miny, maxx, maxy = prism_bbox
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
            if cfg.inference.include_cmap:
                rgb_roads = np.zeros((3, rasterize_roads.shape[0], rasterize_roads.shape[1]), dtype=np.uint8)

                # Set values in the 3D array based on the binary_array
                rgb_roads[0, :, :] = rasterize_roads * 255
                rgb_roads[1, :, :] = rasterize_roads * 255
                rgb_roads[2, :, :] = rasterize_roads * 255
        else:
            # if no shapes to rasterize
            rasterize_roads = np.zeros(dst_shape, dtype=np.uint8)
            if cfg.inference.include_cmap:
                rgb_roads = np.zeros((3, *dst_shape), dtype=np.uint8)
    except Exception as e:
        raise e
    finally:
        if mem_ds:
            mem_ds = None
        if raster_ds:
            raster_ds = None

    with rasterio.open(dir_path + save_as + '.tif', 'w', driver='Gtiff', count=1, height=rasterize_roads.shape[-2], width=rasterize_roads.shape[-1],
                       crs=dst_crs, dtype=rasterize_roads.dtype, transform=dst_transform, nodata=0) as dst:
        dst.write(rasterize_roads, 1)

    if cfg.inference.include_cmap:
        with rasterio.open(dir_path + save_as + '_cmap.tif', 'w', driver='Gtiff', count=3, height=rgb_roads.shape[-2], width=rgb_roads.shape[-1],
                        crs=dst_crs, dtype=rgb_roads.dtype, transform=dst_transform, nodata=None) as dst:
            dst.write(rgb_roads)

def pipeline_dem(dir_path, save_as, dst_shape, dst_crs, dst_transform, bounds, cfg):
    """Generates Digital Elevation Map raster given destination raster properties and bounding box.

    Note to developers: slope raster generation removed in favor of calling np.gradient on dem tile during preprocessing.

    Parameters
    ----------
    dir_path : str
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
    with rasterio.open(dir_path + save_as + '.tif', 'w', driver='Gtiff', count=1, height=destination.shape[-2], width=destination.shape[-1], crs=dst_crs, dtype=destination.dtype, transform=dst_transform, nodata=no_data) as dst:
        dst.write(destination, 1)

    if cfg.inference.include_cmap:
        # for DEM use grayscale!
        dem_cmap = colormap_to_rgb(destination, cmap='gray', no_data=no_data)

        # save dem cmap
        with rasterio.open(dir_path + save_as + '_cmap.tif', 'w', driver='Gtiff', count=3, height=dem_cmap.shape[-2], width=dem_cmap.shape[-1], crs=dst_crs, dtype=dem_cmap.dtype, transform=dst_transform, nodata=None) as dst:
            dst.write(dem_cmap)

def pipeline_flowlines(dir_path, save_as, dst_shape, dst_crs, dst_transform, prism_bbox, cfg,
exact_fcodes=['46000', '46003', '46006', '46007', '55800', '33600', '33601', '33603', '33400', '42801', '42802', '42805', '42806', '42809']):
    """Generates raster with burned in geometries of flowlines given destination raster properties.

    NOTE: FCode for flowlines can be referenced here: https://files.hawaii.gov/dbedt/op/gis/data/NHD%20Complete%20FCode%20Attribute%20Value%20List.pdf

    Parameters
    ----------
    dir_path : str
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

        for code in GetHU4Codes(prism_bbox):
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
            if cfg.inference.include_cmap:
                rgb_flowlines = np.zeros((3, flowlines.shape[0], flowlines.shape[1]), dtype=np.uint8)

                # Set values in the 3D array based on the binary_array
                rgb_flowlines[0, :, :] = flowlines * 255
                rgb_flowlines[1, :, :] = flowlines * 255
                rgb_flowlines[2, :, :] = flowlines * 255
        else:
            # if no shapes to rasterize
            flowlines = np.zeros(dst_shape, dtype=np.uint8)
            if cfg.inference.include_cmap:
                rgb_flowlines = np.zeros((3, *dst_shape), dtype=np.uint8)
    except Exception as e:
        raise e
    finally:
        if mem_ds:
            mem_ds = None
        if raster_ds:
            raster_ds = None

    # flowlines raw
    with rasterio.open(dir_path + save_as + '.tif', 'w', driver='Gtiff', count=1, height=flowlines.shape[-2], width=flowlines.shape[-1],
                       crs=dst_crs, dtype=flowlines.dtype, transform=dst_transform, nodata=0) as dst:
        dst.write(flowlines, 1)

    # flowlines cmap
    if cfg.inference.include_cmap:
        with rasterio.open(dir_path + save_as + '_cmap.tif', 'w', driver='Gtiff', count=3, height=rgb_flowlines.shape[-2], width=rgb_flowlines.shape[-1], crs=dst_crs, dtype=rgb_flowlines.dtype, transform=dst_transform, nodata=None) as dst:
            dst.write(rgb_flowlines)

def pipeline_waterbody(dir_path, save_as, dst_shape, dst_crs, dst_transform, prism_bbox, cfg):
    """Generates raster with burned in geometries of waterbodies given destination raster properties.

    Parameters
    ----------
    dir_path : str
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

        for code in GetHU4Codes(prism_bbox):
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

            if cfg.inference.include_cmap:
                # Create RGB raster
                rgb_waterbody = np.zeros((3, waterbody.shape[0], waterbody.shape[1]), dtype=np.uint8)

                # Set values in the 3D array based on the binary_array
                rgb_waterbody[0, :, :] = waterbody * 255
                rgb_waterbody[1, :, :] = waterbody * 255
                rgb_waterbody[2, :, :] = waterbody * 255
        else:
            # if no shapes to rasterize
            waterbody = np.zeros(dst_shape, dtype=np.uint8)

            if cfg.inference.include_cmap:
                rgb_waterbody = np.zeros((3, *dst_shape), dtype=np.uint8)
    except Exception as e:
        raise e
    finally:
        if mem_ds:
            mem_ds = None
        if raster_ds:
            raster_ds = None

    # waterbody raw
    with rasterio.open(dir_path + save_as + '.tif', 'w', driver='Gtiff', count=1, height=waterbody.shape[-2], width=waterbody.shape[-1], crs=dst_crs, dtype=waterbody.dtype, transform=dst_transform, nodata=0) as dst:
        dst.write(waterbody, 1)

    # waterbody cmap
    if cfg.inference.include_cmap:
        with rasterio.open(dir_path + save_as + '_cmap.tif', 'w', driver='Gtiff', count=3, height=rgb_waterbody.shape[-2], width=rgb_waterbody.shape[-1], crs=dst_crs, dtype=rgb_waterbody.dtype, transform=dst_transform, nodata=None) as dst:
            dst.write(rgb_waterbody)

def pipeline_NLCD(dir_path, save_as, year, dst_shape, dst_crs, dst_transform, cfg):
    """Generates raster with NLCD land cover classes. Uses windowed reading of NLCD raster
    for speed (NLCD files are large).

    Parameters
    ----------
    dir_path : str
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
    with rasterio.open(dir_path + save_as + '.tif', 'w', driver='Gtiff', count=1, height=nlcd_arr.shape[-2], width=nlcd_arr.shape[-1], crs=dst_crs, dtype=nlcd_arr.dtype, transform=dst_transform, nodata=250) as dst:
        dst.write(nlcd_arr, 1)

    # create NLCD colormap
    if cfg.inference.include_cmap:
        H, W = nlcd_arr.shape
        rgb_img = np.zeros((H, W, 3), dtype=np.uint8)

        # vectorized mapping
        for code, rgb in NLCD_CODE_TO_RGB.items():
            mask = nlcd_arr == code
            rgb_img[mask] = rgb

        rgb_img = np.transpose(rgb_img, (2, 0, 1))

        with rasterio.open(dir_path + save_as + '_cmap.tif', 'w', driver='Gtiff', count=3, height=rgb_img.shape[-2], width=rgb_img.shape[-1], crs=dst_crs, dtype=rgb_img.dtype, transform=dst_transform, nodata=None) as dst:
            dst.write(rgb_img)


def pipeline_S1(stac_provider, dir_path, save_as, dst_crs, item, bbox, cfg):
    """Generates dB scale raster of SAR data in VV and VH polarizations.

    Parameters
    ----------
    stac_provider : STACProvider
        STAC provider object.
    dir_path : str
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
    vv_name = stac_provider.get_asset_names("s1")["vv"]
    vh_name = stac_provider.get_asset_names("s1")["vh"]
    item_hrefs_vv = stac_provider.sign_asset_href(item.assets[vv_name].href)
    item_hrefs_vh = stac_provider.sign_asset_href(item.assets[vh_name].href)

    out_image_vv, out_transform_vv = crop_to_bounds(item_hrefs_vv, bbox, dst_crs, nodata=0, resampling=Resampling.bilinear)
    out_image_vh, out_transform_vh = crop_to_bounds(item_hrefs_vh, bbox, dst_crs, nodata=0, resampling=Resampling.bilinear)

    with rasterio.open(dir_path + save_as + '_vv.tif', 'w', driver='Gtiff', count=1, height=out_image_vv.shape[-2], width=out_image_vv.shape[-1], crs=dst_crs, dtype=out_image_vv.dtype, transform=out_transform_vv, nodata=-9999) as dst:
        db_vv = db_scale(out_image_vv[0])
        dst.write(db_vv, 1)

    with rasterio.open(dir_path + save_as + '_vh.tif', 'w', driver='Gtiff', count=1, height=out_image_vh.shape[-2], width=out_image_vh.shape[-1], crs=dst_crs, dtype=out_image_vh.dtype, transform=out_transform_vh, nodata=-9999) as dst:
        db_vh = db_scale(out_image_vh[0])
        dst.write(db_vh, 1)

    # color maps
    if cfg.inference.include_cmap:
        img_vv_cmap = colormap_to_rgb(db_vv, cmap='gray', no_data=-9999)
        with rasterio.open(dir_path + save_as + '_vv_cmap.tif', 'w', driver='Gtiff', count=3, height=img_vv_cmap.shape[-2], width=img_vv_cmap.shape[-1], crs=dst_crs, dtype=np.uint8, transform=out_transform_vv, nodata=None) as dst:
            # get color map
            dst.write(img_vv_cmap)

        img_vh_cmap = colormap_to_rgb(db_vh, cmap='gray', no_data=-9999)
        with rasterio.open(dir_path + save_as + '_vh_cmap.tif', 'w', driver='Gtiff', count=3, height=img_vh_cmap.shape[-2], width=img_vh_cmap.shape[-1], crs=dst_crs, dtype=np.uint8, transform=out_transform_vh, nodata=None) as dst:
            dst.write(img_vh_cmap)

def download_area(stac_provider, bbox, cfg):
    """Downloads S2 and S1 imagery for a specific study area based on parameters and generates accompanying rasters.

    If a specific product ID is provided - it is assumed that the user only wants to download that specific product.
    Regardless of the size of the interval and the results, only that specific product will be downloaded.
    If no match found then the program exits.

    If no product ID is provided - it is assumed that the user wants to download all products within the
    specific interval, both S2 and S1 if available.

    Parameters
    ----------
    stac_provider : STACProvider
        STAC provider object.
    bbox : tuple
        Bounding box in PRISM CRS EPSG:4269.
    cfg : DictConfig
        Configuration object.

    Returns
    -------
    bool
        Return True if successful, False if unsuccessful event download and processing.
    """
    logger = logging.getLogger('main')
    minx, miny, maxx, maxy = bbox # assume bbox is in EPSG:4269
    # need to transform box from EPSG 4269 to EPSG 4326 for query
    conversion = transform(PRISM_CRS, SEARCH_CRS, (minx, maxx), (miny, maxy))
    search_bbox = (conversion[0][0], conversion[1][0], conversion[0][1], conversion[1][1])
    time_of_interest = get_date_interval(cfg.inference.event_date, cfg.inference.before, cfg.inference.after)
    eid = datetime.strptime(cfg.inference.event_date, '%Y-%m-%d').strftime('%Y%m%d')

    # STAC catalog search
    logger.info('Beginning catalog search...')
    items_s2 = stac_provider.search_s2(search_bbox, time_of_interest, query={"eo:cloud_cover": {"lt": 95}})
    items_s1 = stac_provider.search_s1(search_bbox, time_of_interest, query={"sar:instrument_mode": {"eq": "IW"}})

    logger.info('Filtering catalog search results...')
    if len(items_s2) == 0 and len(items_s1) == 0:
        logger.info(f'No S2 and S1 products found for date interval {time_of_interest}.')
        return False
    logger.info(f'Found {len(items_s2)} S2 products and {len(items_s1)} S1 products for date interval {time_of_interest}.')

    try:
        state = get_state(minx, miny, maxx, maxy)
        if state is None:
            raise Exception(f'State not found for {cfg.inference.event_date}, at {minx}, {miny}, {maxx}, {maxy}')
    except Exception as err:
        logger.exception(f'Error fetching state. Eid: {eid}.')
        return False

    # if product ID provided, look for match
    logger.info('Beginning download...')
    file_to_product = dict()
    if cfg.inference.product_id:
        logger.info(f'Checking for product ID: {cfg.inference.product_id}')
        product_id_item = None
        matched = None
        for item in items_s2:
            logger.info(f'Checking S2 product: {item.id}')
            if cfg.inference.product_id in item.id:
                logger.info(f'Matched S2 product: {item.id}')
                product_id_item = item
                matched = 's2'
                break
        for item in items_s1:
            logger.info(f'Checking S1 product: {item.id}')
            if cfg.inference.product_id in item.id:
                logger.info(f'Matched S1 product: {item.id}')
                product_id_item = item
                matched = 's1'
                break

        if not product_id_item:
            logger.exception(f'No product found with ID that matches {cfg.inference.product_id}.')
            return False

        main_crs = pe.ext(product_id_item).crs_string if cfg.inference.crs is None else cfg.inference.crs
        conversion = transform(PRISM_CRS, main_crs, (minx, maxx), (miny, maxy))
        cbbox = conversion[0][0], conversion[1][0], conversion[0][1], conversion[1][1]

        if matched == 's2':
            dt = product_id_item.datetime.strftime('%Y%m%d')
            dst_shape, dst_transform = pipeline_TCI(stac_provider, cfg.inference.dir_path, f'tci_{dt}_{eid}', main_crs, product_id_item, cbbox)
            logger.debug(f'TCI raster completed for {product_id_item.id} on {dt}.')
            pipeline_RGB(stac_provider, cfg.inference.dir_path, f'rgb_{dt}_{eid}', main_crs, product_id_item, cbbox)
            logger.debug(f'RGB raster completed for {product_id_item.id} on {dt}.')
            pipeline_B08(stac_provider, cfg.inference.dir_path, f'b08_{dt}_{eid}', main_crs, product_id_item, cbbox)
            logger.debug(f'B08 raster completed for {product_id_item.id} on {dt}.')
            pipeline_NDWI(stac_provider, cfg.inference.dir_path, f'ndwi_{dt}_{eid}', main_crs, product_id_item, cbbox, cfg)
            logger.debug(f'NDWI raster completed for {product_id_item.id} on {dt}.')
            pipeline_SCL(stac_provider, cfg.inference.dir_path, f'clouds_{dt}_{eid}', dst_shape, main_crs, dst_transform, product_id_item, cbbox, cfg)
            logger.debug(f'SCL raster completed for {product_id_item.id} on {dt}.')

            # record product used to generate rasters
            file_to_product[f'tci_{dt}_{eid}'] = product_id_item.id
            file_to_product[f'rgb_{dt}_{eid}'] = product_id_item.id
            file_to_product[f'b08_{dt}_{eid}'] = product_id_item.id
            file_to_product[f'ndwi_{dt}_{eid}'] = product_id_item.id
            file_to_product[f'clouds_{dt}_{eid}'] = product_id_item.id
        else:
            dt = product_id_item.datetime.strftime('%Y%m%d')
            pipeline_S1(stac_provider, cfg.inference.dir_path, f'sar_{dt}_{eid}', main_crs, product_id_item, cbbox, cfg)
            logger.debug(f'S1 raster completed for {product_id_item.id} on {dt}.')

            # record product used to generate rasters
            file_to_product[f'sar_{dt}_{eid}'] = product_id_item.id
    else:
        logger.info(f'Downloading {len(items_s2)} S2 products and {len(items_s1)} S1 products.')
        # try to download all in the interval
        all_items = items_s2 + items_s1
        main_crs = pe.ext(all_items[0]).crs_string if cfg.inference.crs is None else cfg.inference.crs

        conversion = transform(PRISM_CRS, main_crs, (minx, maxx), (miny, maxy))
        cbbox = conversion[0][0], conversion[1][0], conversion[0][1], conversion[1][1]

        for item in items_s2:
            dt = item.datetime.strftime('%Y%m%d')
            dst_shape, dst_transform = pipeline_TCI(stac_provider, cfg.inference.dir_path, f'tci_{dt}_{eid}', main_crs, item, cbbox)
            logger.debug(f'TCI raster completed for {item.id} on {dt}.')
            pipeline_RGB(stac_provider, cfg.inference.dir_path, f'rgb_{dt}_{eid}', main_crs, item, cbbox)
            logger.debug(f'RGB raster completed for {item.id} on {dt}.')
            pipeline_B08(stac_provider, cfg.inference.dir_path, f'b08_{dt}_{eid}', main_crs, item, cbbox)
            logger.debug(f'B08 raster completed for {item.id} on {dt}.')
            pipeline_NDWI(stac_provider, cfg.inference.dir_path, f'ndwi_{dt}_{eid}', main_crs, item, cbbox, cfg)
            logger.debug(f'NDWI raster completed for {item.id} on {dt}.')
            pipeline_SCL(stac_provider, cfg.inference.dir_path, f'clouds_{dt}_{eid}', dst_shape, main_crs, dst_transform, item, cbbox, cfg)
            logger.debug(f'SCL raster completed for {item.id} on {dt}.')

            # record product used to generate rasters
            file_to_product[f'tci_{dt}_{eid}'] = item.id
            file_to_product[f'rgb_{dt}_{eid}'] = item.id
            file_to_product[f'b08_{dt}_{eid}'] = item.id
            file_to_product[f'ndwi_{dt}_{eid}'] = item.id
            file_to_product[f'clouds_{dt}_{eid}'] = item.id

        for item in items_s1:
            dt = item.datetime.strftime('%Y%m%d')
            pipeline_S1(stac_provider, cfg.inference.dir_path, f'sar_{dt}_{eid}', main_crs, item, cbbox, cfg)
            logger.debug(f'S1 raster completed for {item.id} on {dt}.')

            # record product used to generate rasters
            file_to_product[f'sar_{dt}_{eid}'] = item.id

    # save all supplementary rasters in raw and rgb colormap
    logger.info('Processing supplementary rasters...')
    pipeline_roads(cfg.inference.dir_path, f'roads_{eid}', dst_shape, main_crs, dst_transform, state, bbox, cfg)
    logger.debug(f'Roads raster completed successfully.')
    pipeline_dem(cfg.inference.dir_path, f'dem_{eid}', dst_shape, main_crs, dst_transform, bbox, cfg)
    logger.debug(f'DEM raster completed successfully.')
    pipeline_flowlines(cfg.inference.dir_path, f'flowlines_{eid}', dst_shape, main_crs, dst_transform, bbox, cfg)
    logger.debug(f'Flowlines raster completed successfully.')
    pipeline_waterbody(cfg.inference.dir_path, f'waterbody_{eid}', dst_shape, main_crs, dst_transform, bbox, cfg)
    logger.debug(f'Waterbody raster completed successfully.')
    pipeline_NLCD(cfg.inference.dir_path, f'nlcd_{eid}', int(eid[:4]), dst_shape, main_crs, dst_transform, cfg)
    logger.debug(f'NLCD raster completed successfully.')

    # validate raster shapes, CRS, transforms
    result = validate_event_rasters(cfg.inference.dir_path, logger=logger)
    if not result.is_valid:
        logger.error(f'Raster validation failed for event {eid}. Removing directory and contents.')
        # shutil.rmtree(dir_path) - for now do not delete!
        raise Exception(f'Raster validation failed for event {eid}.')

    # lastly generate metadata file
    logger.info(f'Generating metadata file...')
    metadata = {
        "metadata": {
            "Download Date": cfg.inference.event_date,
            "CRS": main_crs,
            "State": state,
            "Bounding Box": {
                "minx": minx,
                "miny": miny,
                "maxx": maxx,
                "maxy": maxy
            },
            "Item IDs": file_to_product
        }
    }

    metadata_path = Path(cfg.inference.dir_path) / 'metadata.json'
    if metadata_path.exists():
        # if metadata file already exists, update the item IDs
        with open(metadata_path, "r") as json_file:
            existing_metadata = json.load(json_file)
            existing_metadata['metadata']['Item IDs'].update(metadata['metadata']['Item IDs'])
            metadata = existing_metadata

    with open(metadata_path, "w") as json_file:
        json.dump(metadata, json_file, indent=4)

    logger.info('Metadata and raster generation completed. Event finished.')
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
    """Downloads S2 or S1 imagery + accompanying rasters for a specific study area delineated by
    a shapefile. Best used with a specific S2 or S1 product ID in mind.
    
    The script downloads imagery given a shapefile of ROI. It is agnostic of event tiles.
    To ensure shape consistency across different dates/images, use the same CRS.
    
    Parameters
    ----------
    cfg : DictConfig
        Configuration object.
    """
    if cfg.inference.dir_path is None:
        raise ValueError('inference.dir_path config parameter is required to specify a directory for download.')

    # Create directory if it doesn't exist
    Path(cfg.inference.dir_path).mkdir(parents=True, exist_ok=True)
    
    # make sure dir_path ends with a slash
    if not cfg.inference.dir_path.endswith('/'):
        cfg.inference.dir_path += '/'

    # root logger
    rootLogger = setup_logging(cfg.inference.dir_path, logger_name='main', log_level=logging.DEBUG, mode='w', include_console=False)

    # log sampling parameters used
    rootLogger.info(
        "Download area parameters used:\n"
        f"  Event date: {cfg.inference.event_date}\n"
        f"  Days before flood event: {cfg.inference.before}\n"
        f"  Days after flood event: {cfg.inference.after}\n"
        f"  Product ID: {cfg.inference.product_id}\n"
        f"  CRS: {cfg.inference.crs}\n"
        f"  Include color maps: {cfg.inference.include_cmap}\n"
        f"  Source: {cfg.inference.source}\n"
    )

    rootLogger.info("Initializing area download...")
    try:
        # get bbox from shapefile - use PRISM CRS for easy use of pipeline functions
        bbox = get_bbox_from_shapefile(cfg.inference.shapefile, crs=PRISM_CRS)

        # get stac provider
        stac_provider = get_stac_provider(cfg.inference.source, logger=rootLogger)

        # download imagery
        max_attempts = 3
        for attempt in range(1, max_attempts + 1):
            try:
                download_area(stac_provider, bbox, cfg)
                break
            except (rasterio.errors.WarpOperationError, rasterio.errors.RasterioIOError, pystac_client.exceptions.APIError) as err:
                rootLogger.error(f"Connection error: {type(err)}")
                if attempt == max_attempts:
                    rootLogger.error(f'Maximum number of attempts reached, skipping event...')
                else:
                    rootLogger.info(f'Retrying ({attempt}/{max_attempts})...')
            except NoElevationError as err:
                rootLogger.error(f'Elevation file missing, skipping event...')
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