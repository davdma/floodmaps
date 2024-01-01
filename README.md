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

### Code
* Sentinel-2 data pipeline code can be found in the script `sample_mpc.py`.
* The water pixel detection model is still in progress, but can be found in the notebook `unet.ipynb`.

## Data Pipeline
To collect and process satellite imagery, I have created an automated data pipeline written as the Python script `sample_mpc.py`. The script is run as a job through a bash script on the Argonne Bebop computing cluster.

Pipeline sequence:
* Queries extreme precipitation events between 2016-present using the [PRISM dataset](https://prism.oregonstate.edu/).
* Copernicus Sentinel-2 RGB bands and the B8 near infrared band.

However, by themselves these raw satellite images lack the labels we need for modeling. So we must label water pixels ourselves.

I developed a ground-truthing workflow with step by step instructions that uses Google Open Street View and the supplementary rasters for producing accurate hand labeled water masks from the collected data.
**Ground-Truthing instructions:** [pdf here](https://1drv.ms/b/s!Aq3V83mBle0dvhMcZAiCh04A59--?e=IdSswS)

## Model
