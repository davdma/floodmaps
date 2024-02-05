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

![image](https://github.com/davdma/floodmaps/assets/42689743/7c05362b-3bff-47ac-840d-5484ef0e0f03)
**Figure 1:** Files collected and processed for each geographic tile.

However, by themselves these raw satellite images lack the labels we need for modeling. So we must label water pixels ourselves.

![image](https://github.com/davdma/floodmaps/assets/42689743/91799a7d-6fa8-4c04-b3c5-9f1a565b8e59)
**Figure 2:** Labeling flood images through GIMP software.

I developed a workflow with step by step instructions that uses Google Open Street View and the supplementary rasters for producing accurate hand labeled water masks from the collected data.

**Instructions can be viewed here:** [pdf](https://1drv.ms/b/s!Aq3V83mBle0dvhMcZAiCh04A59--?e=IdSswS)

## Model
For our water pixel detection model, we want to test multiple different built-in architectures (UNet, AlexNet, ResNet etc.) to find what works and what doesn't. We first explore the UNet model:

![u-net-architecture](https://github.com/davdma/floodmaps/assets/42689743/d91c7627-52f4-4849-b5dc-86c2cc975c0d)
**Figure 3:** UNet architecture. For our model we also added dropouts after each max pooling layer to regularize the learning process and prevent overfitting.

Our model input consists of the RGB image, the NIR B8 band image, and the NDWI calculation. However, we cannot take the large images for use immediately. We must first take the labeled data and partition them into digestible 64 x 64 pixel tiles for input. First we tried breaking each larger image into patches by imposing a grid, but we found that the model failed to learn from the data this way. A much better way came from using random cropping - randomly sampling thousands of 64 x 64 patches from each image.

![image](https://github.com/davdma/floodmaps/assets/42689743/f33a5723-2a57-4efa-b0dd-fd737d3e2967)
**Figure 4:** Training and validation plots for UNet model using the random cropping sampling method. We see significant learning taking place, but the model still needs some more tuning.

<img src="https://github.com/davdma/floodmaps/assets/42689743/4a74c50b-34f3-4e47-b089-b27453800571" height="400">

**Figure 5:** Preliminary prediction results on validation set. Can observe that there is some underprediction is some areas, but this can be fixed with more tuning.

Note: The water pixel detection model is still being tuned, but can be found in the notebook `unet.ipynb`.
