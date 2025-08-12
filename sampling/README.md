# Sampling Flood Data
## Overview

<p align="center">
   <img width="550" alt="datasetmethod" src="https://github.com/user-attachments/assets/3334f7e4-37b7-45a5-bb75-d300f19cfcc9" />
   <br><br>
</p>

Discretizing the US Sentinel-2 and Sentinel-1 flood dataset:
* One of the problems with aggregating a flood dataset is how to discretize it such that downloading and labeling can become systematic and efficient.
* To solve this, tiles much smaller than the individual satellite product extent was chosen as the unit of discretization. The dataset is broken up into 4km x 4km tiles (or cells) within a grid across the continental US provided by [PRISM](https://prism.oregonstate.edu/). Each PRISM tile can have its S2 and S1 capture sampled at a particular date, labeled, and used as a datapoint for training.

<p align="center" style="margin: 30px 0;">
   <img width="2442" height="925" alt="samplingprismcell" src="https://github.com/user-attachments/assets/e602e061-0e03-4ae9-9b04-f2ade3d6c66c" />
   <em>PRISM mesh grid displayed over satellite imagery in QGIS.</em>
   <br><br>
</p>

The method for data collection is as follows:
* Flood events are identified through high precipitation cells in the PRISM dataset from August 2016 onward. For cells that meet or exceed a specified precipitation threshold, the specific date, time, and georeferenced coordinates are then used to search for Sentinel-2 captures through Copernicus data providers (e.g. Planetary Computer or AWS).
* The resulting captures are stored in event directories, where each "event" we denote simply as a cell we queried on a specified date with a high precipitation measurement.
* Each event has an event id or `eid` in the format `YYYYMMDD_YCoord_XCoord` where `YCoord` and `XCoord` is the position of the cell on the grid defined by the PRISM dataset.
* It may also be the case that some flood events and corresponding captures of interest have low precipitation (e.g. lie downstream of high precipitation cells) and get filtered out by this method. To ensure these are still included, an option `--manual` is added to the scripts to specify flooded cells without extreme precipitation for download.

## Currently Supported Channels
**S2 product channels:**
* RGB reflectances
* NIR band (B08)
* NDWI layer (calculated as `(B03 - B08) / (B03 + B08)`)
* True Color Image (TCI) visual channels
* Cloud mask layer

**S1 product channels:**
* VV polarization layer
* VH polarization layer

**Product independent channels:**
* DEM layer
* Roads mask
* Flowlines mask
* Waterbody mask
* NLCD layer

## Setup
To use the data download scripts and setup the `sampling/` directory in the cloned repo, first download the PRISM precipitation data file and other necessary data files for generating the layers in the sampled dataset. This can be done by running the following python scripts inside your directory.

Scripts for downloading PRISM and supplementary datasets:
* `get_prism.py` - run this first for downloading PRISM precipitation zip files and collating into NetCDF data file. Can later be used for updating precip data to latest available on PRISM.
* `get_supplementary.py` - for downloading TIGER roads, NHD, DEM, NLCD data.
    * For the DEM data you will need to filter for 1/3 arc second DEM and download the txt file of products via https://apps.nationalmap.gov/downloader/ and save as `neddownload.txt` inside of `sampling/` for the DEM downloading to work.

Then you will need to create a yaml configuration file (e.g. in `sampling/configs/`) to specify the file paths to the PRISM and supplementary data you have downloaded. An example `configs/sample_s2_s1.yaml` file with the paths to the PRISM NetCDF file, meshgrid, region shapefiles (e.g. Illinois CESER boundary) among other data needed by the sampling scripts:

```yaml
paths:
  prism_data: "PRISM/prismprecip_20160801_20241130.nc"
  ceser_boundary: "CESER/CESER_FM_potential_no-flow_boundary.shp"
  prism_meshgrid: "PRISM/meshgrid/prism_4km_mesh.shp"
  nhd_wbd: "NHD/WBD_National.zip"
  elevation_dir: "Elevation/"
  nlcd_dir: "NLCD/"
  roads_dir: "Roads/"
```

The CESER boundary shapefile is downloaded from the ANL box folder, while the PRISM 4km mesh grid shapefile is downloaded from the [PRISM website link](https://prism.oregonstate.edu/downloads/data/prism_4km_mesh.zip).

## Sampling Scripts
There are several sampling scripts:
* `sample_s2.py` - this is for downloading tiles with the Sentinel-2 L2A layers alongside any ancillary data (DEM, roads, flowlines, waterbodies etc.). This should be run to produce the S2 dataset used for training the optical model.
* `sample_s2_s1.py` - this is for downloading tiles with both Sentinel-2 L2A and Sentinel-1 GRD layers that are temporally coincident. You will need to run this to produce the S1 dataset used for training the SAR model.
* `sample_s1.py` - this is for downloading only Sentinel-1 GRD layers on top of a pre-existing S2 dataset directory created from `sample_s2.py`. Use this to add coincident S1 SAR layers to an already existing S2 dataset.
* `sample_sar_multi.py` - this is for downloading multitemporal composite SAR data of the same cells represented in the S2 and S1 dataset. The difference being that it intentionally avoids the flood dates due to possible flood water inconsistencies in the temporal average.

<p align="center" style="margin: 30px 0;">
<img width="2552" height="1142" alt="samplingprismcell2" src="https://github.com/user-attachments/assets/a70a8aa6-2d81-426f-bbbc-dd35d33edc06" />
<em>Example of a downloaded S2 event folder with its TCI image visualized.</em>
   <br><br>
</p>

## For Developers
How the sampling scripts work:
1. Loads in a list of event indices (time, y, x) from either filtering PRISM precipitation or a manual file.
2. For each event, look up products in the STAC data catalog that satisfy requirements (e.g. S2 must be within `x` days pre-event or `y` days post-event, S1 must be within `z` hours of an S2 product). S2 and S1 products are also filtered out based on percentage cloudy or null pixels (specified by the threshold `maxcoverpercentage`).
3. For all valid products, picks one product CRS to crop and standardize the rest to the same projection and output shape. The chosen CRS determines the Affine transform and cell width and height from the bounding box.
4. For each available date, generate rasters + supplementary layers. If multiple S2 or S1 products are present on a single date, only one is used to represent that date.
5. Downloads the data layers to the event folder `YYYYMMDD_YCoord_XCoord/` as `.tif` files. Colormaps or `cmap` files are also included for easy visualization of each layer in QGIS. Event metadata is saved in `metadata.json`.

Other items of note:
* Previously processed event indices are tracked using a `history.pickle` file, so the scripts can be run iteratively to download more tiles within a pre-existing data directory. You can cap the maximum number of events you download at one time with `maxevents` argument.
* Each time the sampling scripts are run, a `main` and `events` log file is generated inside the specific dataset directory containing overall script and specific event processing debug messages respectively.

