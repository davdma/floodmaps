# AI Flood Detection

[![Python 3.11](https://img.shields.io/badge/python-3.11-blue.svg)](https://www.python.org/downloads/)
[![License: MIT](https://img.shields.io/badge/License-MIT-yellow.svg)](https://opensource.org/licenses/MIT)
[![PyTorch](https://img.shields.io/badge/PyTorch-2.0+-ee4c2c.svg)](https://pytorch.org/)
[![DOI](https://zenodo.org/badge/DOI/10.5281/zenodo.18528354.svg)](https://doi.org/10.5281/zenodo.18528354)
<!-- [![Hugging Face Dataset](https://img.shields.io/badge/🤗%20Dataset-floodmaps-blue)](https://huggingface.co/datasets/yourname/floodmaps) -->

### Background
Floods are a common and devastating natural hazard. There is a pressing need to replace traditional flood mapping methods with faster more accurate AI-assisted models that harness satellite imagery.

### Goals
The AI-assisted flood detection project aims to build an end-to-end computer vision workflow for flood detection capable of generating flood observation from multispectral and radar satellite imaging systems (specifically Copernicus).

![image](https://github.com/davdma/floodmaps/assets/42689743/0685799c-7ab7-4640-9ae4-759b797dd13f)

We aim to accomplish this task through the following process:
1. **Data Collection:** Collect and process remote sensing data (visual, infrared, synthetic aperture radar) following extreme precipitation and flood events for modeling.
2. **Manual Annotation:** Produce accurate ground truth data by manually labeling satellite images at pixel fidelity.
3. **Initial Model:** Train an initial multispectral model on a small labeled dataset to make quality predictions on unlabelled images, and then incrementally expand the dataset with human-corrected machine labels.
4. **Iterative Process:**  As label quantity and geographic diversity improves, the model training and prediction improves, which in turn leads to better quality machine labels and a larger dataset.
5. **Final Model(s):** Tune and benchmark high performing Sentinel-2 multispectral and Sentinel-1 SAR segmentation models using the cumulative human + machine labels. Inference the final models to detect flood water extent in satellite imagery.

## Data Pipeline
To collect and process satellite imagery, an automated data pipeline was implemented in the Python script `sample_s2.py` for Sentinel-2 only and `sample_s2_s1.py` for Sentinel-2 and Sentinel-1. The scripts download **4km x 4km geospatial "tiles"** which make up an indivisible unit within our dataset. The scripts are run on an Argonne computing cluster and submitted as slurm or pbsnodes jobs. For more information on setting up and using those scripts, consult the [sampling](src/floodmaps/sampling/README.md) folder documentation.

What it does:
* Queries extreme precipitation events from 2016-present using the [PRISM dataset](https://prism.oregonstate.edu/).
* Each "event" is demarcated by a date and corresponding cell in the 4km x 4km PRISM cell grid overlaid onto the Continental US.
* Downloads Copernicus Sentinel-2 RGB spectral bands, B8 near-infrared, B11 and B12 short-wave infrared (upsampled to 10m), and Sentinel-1 VV, VH bands from [Microsoft Planetary Computer](https://planetarycomputer.microsoft.com/dataset/sentinel-2-l2a) or [Copernicus](https://documentation.dataspace.copernicus.eu/APIs/STAC.html) as 4km x 4km geographic tiles with 10m resolution.
* Adds supplementary information to each tile - roads, waterbodies, slope, elevation (DEM), flowlines, land cover classes (NLCD), scene classification layer (SCL), as well as a Normalized Difference Water Index (NDWI) layer which is calculated from the green and B8 bands. These are generated from a variety of datasets including the TIGER Roads dataset, the National Hydrography Dataset and more.
* Additional water indices such as Modified Normalized Difference Water Index (MNDWI) and Automated Water Extraction Index (AWEI) are also added later during the preprocessing stage.

![image](https://github.com/davdma/floodmaps/assets/42689743/05168f81-c560-456e-9df3-87530d4b1def)
**Figure 1:** Sample of files collected and processed for each geographic tile. The figure omits SWIR, NLCD, SCL, MNDWI, AWEI layers which are also included in the data.

However, by themselves these raw satellite images lack the labels we need for modeling. So we must label water pixels manually.

## Flood Labeling

A fundamental goal in the project is to produce a US flood dataset with precise, high quality labels at 10m resolution. Most publicly available Sentinel-2 flood datasets suffer from coarse labeling that ignores the native 10m pixel resolution, resulting in wide-swath labels that miss fine-scale flood boundaries, narrow channels, and small flooded areas clearly visible in the imagery. To tackle this, an effective labeling workflow is developed with step by step instructions that uses GIMP (for pixel labelling) and QGIS (for raster projection and high resolution imagery). Context is incorporated both from high res Google Satellite imagery and supplementary rasters to assist in producing accurate **pixel-fidelity** hand labeled water masks from the collected data. This has been successfully used by other Argonne interns and staff for contributing high quality labels to the dataset.

**Labeling instruction doc can be viewed here:** [pdf](https://1drv.ms/b/c/1ded958179f3d5ad/EXI5Xlbf-j1Ik1BebYOHmMIBvSyrxDlMZ0A57EPvR7XTFg)

<div align="center">
<img width="75%" alt="labelgimp" src="https://github.com/davdma/floodmaps/assets/42689743/91799a7d-6fa8-4c04-b3c5-9f1a565b8e59" />
<p style="width: 60%; margin: 0 auto;"><em>Figure 2.1: Labeling flood images through GIMP software.
</em></p>
</div>

The True Color Image (TCI) provided by the S2 products can often be ambiguous as to whether a specific pixel contains water or not, e.g. a brown pixel could be wet soil, dark vegetation, dirt road, or still flood water. Hence, it is important for the labeler to have context when making judgments to achieve high quality labels with 10m pixel fidelity. This is done by zooming into high resolution imagery in and around the pixel of interest in QGIS and making comparisons between pre and post-flood images.

<div align="center">
<img width="75%" alt="labelwcontext" src="https://github.com/user-attachments/assets/4cdd6f7f-f246-4622-aead-2c1ff6404fc0" />
<p style="width: 60%; margin: 0 auto;"><em>Figure 2.2: Using high resolution imagery provided by NOAA to label flood images near Addicks Reservoir in Texas following Hurricane Harvey.
</em></p>
</div>

As a result, we produced **146+** 4km by 4km manually labeled S2 flood tiles spread across the midwestern states (Illinois, Wisconsin, Indiana, Iowa) and southern states (Texas, Mississippi, Louisiana, Florida, NC, SC), in addition to **7846** 4km by 4km weak labeled S1 flood tiles across the CONUS. An additional unlabeled **38252** S1 tiles consisting of multitemporal stacks were aggregated for training a despeckler CVAE model prior to downstream SAR flood mapping. The datasets and their properties are summarized in the table below.

## Dataset(s)

<div align="center">
<img width="70%" alt="dataset_table" src="https://github.com/user-attachments/assets/07afab63-fd37-42b5-abcc-43d14c4aca12" />
</div>

> **Note:** The S2 + PRISM dataset is available on [Zenodo](https://doi.org/10.5281/zenodo.18528354). See [Usage](#usage) for download and setup instructions.

## Benchmark(s)

<div align="center">
<img width="90%" alt="dataset_table" src="https://github.com/user-attachments/assets/b9f44f7d-1d20-4f23-bea1-94e48970e08f" />
</div>

## Use the Models
The weights of the top benchmarked UNet and UNet++ water detection models for S1 and S2 datasets are made available in the repo under [models](data/models). To use one of the models, copy the `.yaml` file into `configs/` and add it to the defaults list to enable.

To make a prediction on an area of interest, simply provide the shapefile to `download_aoi.py`. After the relevant rasters have been downloaded, run inference using the trained models with `inference_s2.py` for S2 model or `inference_sar.py` for S1 model.

Example `config.yaml`:
```yaml
defaults:
     - paths: default
     - inference: inference_s2  # Enable S2 inference config
     - s2_unetpp_all_best       # Enable the model config
     - _self_
```

Then run:

```bash
python -m floodmaps.inference.inference_s2
```

> **Note:** See [data/models/README.md](data/models/README.md) for recommended thresholds for each model.

<div align="center">
<img width="90%" alt="labelmapwtile" src="https://github.com/user-attachments/assets/89c28ee1-91d7-486f-ae5a-49faebf211ae" />
<p style="width: 60%; margin: 0 auto;"><em>Figure 3: Example predictions using the benchmarked S2 (left) and S1 (right) UNet++ model. The zoomed in area is 64 × 64 pixels.
</em></p>
</div>

# Usage

Dependencies are installed via the conda environments in `envs/`. First, clone the repository locally:

```bash
git clone https://github.com/davdma/floodmaps.git
```

Then install each of the conda environments:
```bash
conda env create -f envs/floodmaps-sampling.yml
conda env create -f envs/floodmaps-training.yml
conda env create -f envs/floodmaps-tuning.yml
```

Download the Sentinel-2 dataset and labels + PRISM NetCDF4 from Zenodo [![DOI](https://zenodo.org/badge/DOI/10.5281/zenodo.18528354.svg)](https://doi.org/10.5281/zenodo.18528354):

```bash
# extracts the rasters and labels into data/imagery/ and data/labels/
wget https://zenodo.org/records/18528354/files/s2-prism-dataset-v1.0.tar.gz
tar -xzvf dataset-v1.0.tar.gz -C data/
```

The project uses `hydra` to handle configurations. The `configs` folder contains the configuration groups and files for running the scripts. Scripts will first parse the input config from `configs/config.yaml`, so it is necessary to setup `config.yaml` correctly before running any script (note: there are some scripts that require additional `argparse` params). To setup configs correctly, set the directory paths first in `configs/paths/default.yaml`, then use interpolation of those paths across the config:

```yaml
# Setup base_dir to point to root of your cloned repo
base_dir: /lcrc/project/hydrosm/dma
data_dir: ${.base_dir}/data
output_dir: ${.base_dir}/outputs

# ... (additional configuration paths and settings)
```

```yaml
# In other config files use the interpolation syntax ${...} for paths
model:
    weights: "${paths.output_dir}/experiments/2025-07-24_unet_tv5yhqnb/unet_cls.pth"
```

On the cluster, make sure the right conda environment is active in the job submission script:

```bash
#!/bin/bash
#PBS -A hydrosm
#PBS -l select=1:ngpus=1:ncpus=16
#PBS -l walltime=2:00:00
#PBS -N inference_s2
#PBS -m bea
#PBS -o /lcrc/project/hydrosm/dma/outputs/logs/inference_s2.out
#PBS -e /lcrc/project/hydrosm/dma/outputs/logs/inference_s2.err

# Set up my environment
source ~/.bashrc
cd /lcrc/project/hydrosm/dma
conda activate floodmaps-training

# Run sampling script
python -m floodmaps.inference.inference_s2
```

By default `hydra` will save its specific log files and outputs next to the python script. To avoid cluttering the repo, it is highly recommended to either disable this logging completely or modify the hydra output directory location through this setting in `config.yaml`:

```yaml
hydra:
  run:
    dir: ${base_dir}/hydra
```




