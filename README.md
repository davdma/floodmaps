# AI Flood Detection
### Background
Floods are a common and devastating natural hazard. There is a pressing need to replace traditional flood mapping methods with faster more accurate AI-assisted models that harness satellite imagery.

### Goals
The AI-assisted flood detection project aims to build a computer vision workflow for flood detection capable of generating flood observation from radar, satellite and urban imaging systems.

![image](https://github.com/davdma/floodmaps/assets/42689743/0685799c-7ab7-4640-9ae4-759b797dd13f)

To accomplish this task, we want to:
1. Collect and process remote sensing data (visual, infrared, SAR) following extreme precipitation events for flood modeling.
2. Produce accurate ground truth data by manually labeling images.
3. Develop a water pixel detection algorithm to detect flood water extent after a flood event from satellite imagery.

## Data Pipeline
To collect and process satellite imagery, I have created an automated data pipeline implemented in the Python script `sample_mpc.py`. The script is run on the Argonne Bebop computing cluster and submitted as a job through the bash script `sample_job.sh`.

What it does:
* Queries extreme precipitation events from 2016-present using the [PRISM dataset](https://prism.oregonstate.edu/).
* Downloads Copernicus Sentinel-2 RGB and B8 near-infrared bands from [Microsoft Planetary Computer](https://planetarycomputer.microsoft.com/dataset/sentinel-2-l2a) as 4km x 4km geographic tiles with 10m resolution.
* Adds supplementary information to each tile - roads, flowlines, waterbodies, slope, elevation (DEM) - as well as calculating the Normalized Difference Water Index (NDWI).

However, by themselves these raw satellite images lack the labels we need for modeling. So we must label water pixels ourselves.

![image](https://github.com/davdma/floodmaps/assets/42689743/91799a7d-6fa8-4c04-b3c5-9f1a565b8e59)
**Figure 2:** Labeling flood images through GIMP software.

I developed a workflow with step by step instructions that uses Google Open Street View and the supplementary rasters for producing accurate hand labeled water masks from the collected data.

**Instructions can be viewed here:** [pdf](https://1drv.ms/b/s!Aq3V83mBle0dvhMcZAiCh04A59--?e=IdSswS)

## Model
* For our water pixel detection model, we want to test multiple different built-in architectures (UNet, AlexNet, ResNet etc.) to find what works and what doesn't.
* The model input consists of the RGB image, the NIR B8 band image, and the NDWI calculation.
* The labeled data is partitioned into 64 x 64 pixel tiles for input.
* The water pixel detection model is still in progress, but can be found in the notebook `unet.ipynb`.

Model tuning still in progress!
