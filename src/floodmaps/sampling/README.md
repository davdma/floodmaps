# Sampling Flood Data
## Overview

<div align="center">
<img width="550" alt="datasetmethod" src="https://github.com/user-attachments/assets/3334f7e4-37b7-45a5-bb75-d300f19cfcc9" />
<p style="width: 60%; margin: 0 auto;"><em>Figure 1: Flow chart of flood imagery sampling pipeline.
</em></p>
</div>

Discretizing the US Sentinel-2 and Sentinel-1 flood dataset:
* One of the problems with aggregating a flood dataset is how to discretize it such that downloading and labeling can become systematic and efficient.
* To solve this, tiles much smaller than the individual satellite product extent was chosen as the unit of discretization. The dataset is broken up into 4km x 4km tiles (or cells) within a grid across the continental US provided by [PRISM](https://prism.oregonstate.edu/). Each PRISM tile can have its S2 and S1 capture sampled at a particular date, labeled, and used as a datapoint for training.

<p align="center" style="margin: 30px 0;">
   <img width="2442" height="925" alt="samplingprismcell" src="https://github.com/user-attachments/assets/e602e061-0e03-4ae9-9b04-f2ade3d6c66c" />
   <em>Figure 2: PRISM mesh grid displayed over satellite imagery in QGIS.</em>
   <br><br>
</p>

The method for data collection is as follows:
* Flood events are identified through high precipitation cells in the PRISM dataset from August 2016 onward. For cells that meet or exceed a specified precipitation threshold, the specific date, time, and georeferenced coordinates are then used to search for Sentinel-2 captures through STAC data providers (e.g. Microsoft Planetary Computer, Copernicus Data Space Ecosystem, or AWS).
* The resulting captures are stored in event directories, where each "event" we denote simply as a cell we queried on a specified date with a high precipitation measurement.
* Each event has an event id or `eid` in the format `YYYYMMDD_YCoord_XCoord` where `YCoord` and `XCoord` is the position of the cell on the grid defined by the PRISM dataset.
* It may also be the case that some flood events and corresponding captures of interest have low precipitation (e.g. lie downstream of high precipitation cells or simply not caused by precipitation) and get filtered out by this method. To ensure these are still included, an option `manual` is added to the scripts to specify cells for download regardless of precipitation. To learn more about this option go to [manual download](#manual-download).
* The manual files can be used to specify a list of event tiles for the script to download. For instance, in the `manual/` directory you will find `label.txt` files for obtaining all the event tiles associated with the hand labeled flood maps in the Texas and Illinois datasets. A list of carefully curated but unlabeled flood tiles are also provided.

## Currently Supported Channels
**S2 product channels:**
* RGB reflectances (B04, B03, B02)
* NIR band (B08)
* SWIR bands (B11, B12)
* Water indices: NDWI, MNDWI, AWEI_sh, AWEI_nsh (computed during preprocessing)
* True Color Image (TCI) visual channels
* Scene Classification Layer (SCL) for cloud/shadow masks

**S1 product channels:**
* VV polarization layer
* VH polarization layer

**Product independent channels:**
* DEM layer
* Slope (derived from DEM)
* Roads mask
* Flowlines mask
* Waterbody mask
* NLCD layer

## Setup: From LCRC
If working on LCRC as an argonne collaborator, you can skip downloading all the data files. Simply add my data file path to your config paths yaml with `data_dir: /lcrc/project/hydrosm/dma/data`.

## Setup: From Scratch
To use the data download scripts in a freshly cloned repo, first download the PRISM precipitation data file and other necessary data files for generating the layers in the sampled dataset. This can be done by running the following python scripts inside your directory.

Scripts for downloading PRISM and supplementary datasets:
* `get_prism.py` - run this first for downloading PRISM precipitation zip files and collating into NetCDF data file. Can later be used for updating precip data to latest available on PRISM.
* `get_supplementary.py` - run this for downloading TIGER roads, NHD, DEM, NLCD datasets needed for supplementary rasters.
    * For the DEM data you will need to filter for 1/3 arc second DEM and download the txt file of products via https://apps.nationalmap.gov/downloader/ and save as `neddownload.txt` inside of `sampling/` for the DEM downloading to work. The one in the repo can be used for now, but might eventually be out of date.
    * For the NHD data once you have downloaded the `.zip` files, it is important to have them unzipped for performance. This is done automatically in the download script.

Then make sure the file paths to the PRISM and supplementary data you have downloaded is correct in `configs/paths` yaml. Here is the default `configs/paths/default.yaml` file with the paths to the PRISM NetCDF file, meshgrid, region shapefiles (e.g. Illinois CESER boundary) among other data needed by the sampling scripts. If you just change the `base_dir` to point to your cloned repo then it should work by default:

```yaml
base_dir: /lcrc/project/hydrosm/dma
data_dir: ${.base_dir}/data
output_dir: ${.base_dir}/outputs
supplementary_dir: ${.data_dir}/supplementary

# supplementary files
prism_data: ${.supplementary_dir}/prism/prismprecip_20160801_20241130.nc
ceser_boundary: ${.supplementary_dir}/ceser/CESER_FM_potential_no-flow_boundary.shp
prism_meshgrid: ${.supplementary_dir}/prism/meshgrid/prism_4km_mesh.shp
nhd_wbd: ${.supplementary_dir}/nhd/WBD_National_GDB.gdb
elevation_dir: ${.supplementary_dir}/elevation/
nlcd_dir: ${.supplementary_dir}/nlcd/
roads_dir: ${.supplementary_dir}/roads/
nhd_dir: ${.supplementary_dir}/nhd/
prism_dir: ${.supplementary_dir}/prism/
```

For the Illinois data sampling, the CESER boundary shapefile is downloaded from the ANL box folder. The PRISM 4km mesh grid shapefile is downloaded from the [PRISM website link](https://prism.oregonstate.edu/downloads/data/prism_4km_mesh.zip).

## Setup: API Keys

Two options for satellite download API keys: set in `config.yaml` or pass via environment variable.

Microsoft Planetary Computer:
- Set in the `config.yaml` level with `mpc_api_key: ...`.
- OR Set environment variable `PC_SDK_SUBSCRIPTION_KEY`.

Copernicus Data Space Ecosystem (CDSE) and AWS:
- Require S3 credentials via environment variables: `AWS_ACCESS_KEY_ID` and `AWS_SECRET_ACCESS_KEY`.
- Note: AWS has known issues with offset handling; MPC or CDSE recommended for reliability.

## Sampling Scripts
There are several sampling scripts:
* `sample_s2.py` - this is for downloading tiles with the Sentinel-2 L2A layers alongside any ancillary data (DEM, roads, flowlines, waterbodies etc.). This should be run to produce the S2 dataset used for training the optical model.
* `sample_s2_s1.py` - this is for downloading tiles with both Sentinel-2 L2A and Sentinel-1 GRD layers that are temporally coincident. You will need to run this to produce the S1 dataset used for training the SAR model.
* `sample_s1.py` - this is for downloading only Sentinel-1 GRD layers on top of a pre-existing S2 dataset directory created from `sample_s2.py`. Use this to add coincident S1 SAR layers to an already existing S2 dataset.
* `sample_sar_multi.py` - this is for downloading multitemporal composite SAR data of the same cells represented in the S2 and S1 dataset. The difference being that it intentionally avoids the flood dates due to possible flood water inconsistencies in the temporal average.

<p align="center" style="margin: 30px 0;">
<img width="2552" height="1142" alt="samplingprismcell2" src="https://github.com/user-attachments/assets/a70a8aa6-2d81-426f-bbbc-dd35d33edc06" />
<em>Figure 3: Example of a downloaded S2 event folder with its TCI image visualized.</em>
   <br><br>
</p>

## Manual Download

Manual files allow the user to download any geographic tiles in the US at a point in time provided there are S2 and S1 captures that meet the parameter requirements such as the search interval (within the `before` and `after` sampling config arguments) and cloud cover threshold. The manual text files must have lines formatted in the form: `YYYY-MM-DD, YCoord, XCoord` or `YYYY-MM-DD, YCoord, XCoord, EPSG:XXXX`. Lines leading with the hash tag symbol `#` will be treated as comments and ignored. For example:

```text
# The script will try to download 5 separate event tiles
2016-03-11, 452, 845, EPSG:32616
2016-03-11, 452, 846, EPSG:32616
2017-08-28, 463, 695, EPSG:32615
2017-08-28, 482, 695, EPSG:32615
2015-11-09, 473, 940, EPSG:32616
```

Each line contains the event date, the cell Y index, the cell X index, and an optional CRS as a fourth argument. If the CRS is provided, it forces all the rasters for that specific event to be projected and aligned to that CRS (instead of the default behavior of using the first product CRS in alphabetical order). This is especially important to ensure that the resulting rasters are in the desired dimensions. A different CRS may lead to a different raster height and width and cause the shape to be incompatible with the associated label.

## For Developers
How the sampling scripts work:
1. Loads in a list of events from either filtering PRISM precipitation data or a manual text file.
2. For each event, look up products in the STAC data catalog that satisfy requirements (e.g. S2 must be within `x` days pre-event or `y` days post-event, S1 must be within `z` hours of an S2 product). S2 and S1 products are also filtered out based on percentage cloudy or null pixels (specified by the threshold `maxcoverpercentage`).
3. For all valid products, picks a product CRS (the first in alphabetical order) to crop and standardize the rest to the same projection and output shape. The chosen CRS determines the Affine transform and cell width and height from the bounding box.
4. For each available date, generate rasters + supplementary layers. If multiple S2 or S1 products are present on a single date, only one is used to represent that date.
5. Downloads and saves the data layers to the event folder `YYYYMMDD_YCoord_XCoord/` as `.tif` files. Colormaps or `cmap` files are also included for easy visualization of each layer in QGIS. Event metadata is saved in `metadata.json`.

Other items of note:
* Previously processed event indices (time, y, x) are tracked using a `history.pickle` file, so the scripts can be run iteratively to download more tiles within a pre-existing data directory. You will need to remove the `history.pickle` file to attempt downloading a tile a second time. You can also cap the maximum number of events you download at one time with `maxevents` argument.
* Each time the sampling scripts are run, a `main` and `events` log file is generated inside the specific dataset directory containing overall script and specific event processing debug messages respectively.

