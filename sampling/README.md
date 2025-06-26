# Sampling Scripts
## Overview
The method for data collection is as follows:
* Flood events are identified through high precipitation cells in the PRISM dataset from August 2016 onward. For cells (on the grid of continental US) that meet or exceed a specified precipitation threshold, the specific date, time, and georeferenced coordinates are used to search for Sentinel-2 captures.
* The resulting captures are stored in event directories, where each "event" we denote simply as a cell we queried with a high precipitation event at a specified date.
* Each event has an event id or "eid" in the format `YYYYMMDD_YCoord_XCoord` where `YCoord` and `XCoord` is the position of the cell on the grid defined by the PRISM dataset.

Scripts for downloading PRISM and supplementary datasets:
* `get_prism.py` - for downloading OR updating PRISM precipitation files and data.
* `get_supplementary.py` - for downloading TIGER roads, NHD, DEM, NLCD data.
    * For the DEM data you will need to filter for 1/3 arc second DEM and download the txt file of products via https://apps.nationalmap.gov/downloader/ and save as `neddownload.txt`.

There are several sampling scripts:
* `sample_mpc.py` - this is for downloading the Sentinel-2 L2A Microsoft Planetary Computer dataset alongside any ancillary data (DEM, roads, flowlines, waterbodies etc.). This should be run to produce the S2 dataset used for training the optical model. NOTE: this is now deprecated so avoid using this script, use `sample_s2_s1.py` instead.
* `sample_s2_s1.py` - this is for downloading both Sentinel-2 L2A Microsoft Planetary Computer dataset and Sentinel-1 GRD Microsoft Planetary Computer dataset captures that are temporally coincident. You will most likely only need to run this to produce the SAR dataset used for training the SAR model.
* `sample_s1.py` - this is for downloading only Sentinel-1 GRD Microsoft Planetary Computer data into pre-existing S2 flood event directories. Use this to add SAR images to an already existing S2 dataset.
* `sample_sar_multi.py` - this is for downloading multitemporal composite SAR data of the same cells represented in the S2 and S1 dataset. The difference being that it intentionally avoids the flood dates due to flood water inconsistencies in the temporal average.

# Configuration System
## Overview

The sampling pipeline now uses a YAML-based configuration system to manage data file paths, making it more flexible and maintainable than hardcoded paths. The configuration system validates that all required paths exist.

## Configuration File

The default configuration file is located at `configs/sample_s2_s1.yaml`:

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

## Usage

```python
from config import DataConfig

# Use default config file
config = DataConfig()

# Access paths
prism_file = config.prism_file
elevation_dir = config.elevation_directory
```

### Custom Configuration

```python
# Use custom config file
config = DataConfig("path/to/custom_config.yaml")
```

```bash
# Specify custom config file to script
python sample_s2_s1.py 180 -b 7 -a 4 -c 10 -s 200 -d samples_180_7_4_10_ceser/ -w 24 --region ceser --config path/to/custom_config.yaml
```