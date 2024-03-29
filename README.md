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

Our model input consists of the RGB image, the NIR B8 band image, and the NDWI calculation, as well as the option of selecting additional channels such as the Digital Elevation Map (DEM), slope, roads and waterbodies. However, we cannot take the large images for use immediately. We must first take the labeled data and partition them into digestible 64 x 64 pixel tiles for input. First we tried breaking each larger image into patches by imposing a grid, but we found that the model failed to learn from the data this way. A much better way came from using random cropping - randomly sampling thousands of 64 x 64 patches from each image. This exploratory work can be found in the notebook `unet.ipynb`.

![classifierplots](https://github.com/davdma/floodmaps/assets/42689743/4fc0b400-cc8b-491e-a817-251994e22d73)
![classifierplots2](https://github.com/davdma/floodmaps/assets/42689743/e92a5ca8-4264-40a6-9eb8-bc44be7c9e31)
**Figure 4:** Training and validation plots for UNet model using the random cropping sampling method. We see significant learning taking place, but the model still needs some more tuning.

![allinputsresult](https://github.com/davdma/floodmaps/assets/42689743/d6259f20-82fa-4cfd-ba4b-37f429cf1b85)

**Figure 5:** Preliminary prediction results on validation set. Can observe that there is some overprediction is some areas, but this can be fixed with more tuning.

We then proceeded to add a SrGAN discriminator head to create a two part model. The discriminator would first take the input patch and determine whether the patch has water or not. If the discriminator detects water in the patch, it proceeds to run the patch through the UNet, otherwise it outputs a zero tensor. This two head model design allows us to skip unnecessary computation if the patch contains no water. The prediction results of this two head model on a large flood raster (the patches are run independently and then stitched together) generates some good results:

<p align="center">
  <img src="https://github.com/davdma/floodmaps/assets/42689743/78d029d1-2f32-4991-b62f-c5d6d6ca0167" height="500">
<p align="center">

**Figure 6:** Prediction results on a large flood tile. Using our initial model on unlabelled data allows us to automate our ground truthing process.

