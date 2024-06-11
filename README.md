# AI Flood Detection
### Background
Floods are a common and devastating natural hazard. There is a pressing need to replace traditional flood mapping methods with faster more accurate AI-assisted models that harness satellite imagery.

### Goals
The AI-assisted flood detection project aims to build a computer vision workflow for flood detection capable of generating flood observation from radar, satellite and urban imaging systems.

![image](https://github.com/davdma/floodmaps/assets/42689743/0685799c-7ab7-4640-9ae4-759b797dd13f)

We aim to accomplish this task through the following process:
1. **Data Collection:** Collect and process remote sensing data (visual, infrared, SAR) following extreme precipitation events for flood modeling.
2. **Manual Annotation:** Produce accurate ground truth data by manually labeling images.
3. **Initial Model:** Train an initial Sentinel-2 CNN model to make quality predictions on unlabelled images, and augment the dataset with new machine labels.
4. **Final Model:** Develop a Sentinel-1 SAR CNN model from our manual + machine labels to detect flood water extent after a flood event.

## Data Pipeline
To collect and process satellite imagery, I have created an automated data pipeline implemented in the Python script `sample_mpc.py` for Sentinel-2 only and `sample_mpc_v2.py` for Sentinel-2 and Sentinel-1. The scripts are run on an Argonne computing cluster and submitted as slurm or pbsnodes jobs.

What it does:
* Queries extreme precipitation events from 2016-present using the [PRISM dataset](https://prism.oregonstate.edu/).
* Downloads Copernicus Sentinel-2 RGB, B8 near-infrared and Sentinel-1 VV, VH bands from [Microsoft Planetary Computer](https://planetarycomputer.microsoft.com/dataset/sentinel-2-l2a) as 4km x 4km geographic tiles with 10m resolution.
* Adds supplementary information to each tile - roads, waterbodies, slope, elevation (DEM), as well as a Normalized Difference Water Index (NDWI) layer which is calculated from the green and B8 bands.

![image](https://github.com/davdma/floodmaps/assets/42689743/7c05362b-3bff-47ac-840d-5484ef0e0f03)
**Figure 1:** Files collected and processed for each geographic tile.

However, by themselves these raw satellite images lack the labels we need for modeling. So we must label water pixels ourselves.

![image](https://github.com/davdma/floodmaps/assets/42689743/91799a7d-6fa8-4c04-b3c5-9f1a565b8e59)
**Figure 2:** Labeling flood images through GIMP software.

I developed a workflow with step by step instructions that uses QGIS, Google Open Street View and the supplementary rasters for producing accurate hand labeled water masks from the collected data.

**Instructions can be viewed here:** [pdf](https://1drv.ms/b/s!Aq3V83mBle0dvhMcZAiCh04A59--?e=IdSswS)

## Model
For our water pixel detection model, we wanted to test multiple built-in architectures that have been used in the flood modelling literature extensively, most commonly UNet and UNet++. We first explore the UNet model:

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

## Quality Control

With our tuned UNet and UNet++ models, we visualized their predictions on unseen flood tiles in QGIS with georeferencing. Overlaid on high resolution satellite images, we were able to identify patterns and see how the model performs in practice. In the example below, we see that the model is a very powerful predictor of open water, and has a strong ability to distinguish fine outlines of water bodies. We also see some limitations of our model: due to the 10m resolution of the input channels, water bodies that are <10m have a higher likelihood of going undetected. Another limitation of note is cloud cover. In practice a percentage of the input satellite images will be obscured by clouds, making it hard for the model to see the waterbodies, and in the example below the model performs the worst in areas obscured by cloud cover (top right corner of waterbody - these clouds are not visible in the high res overlaid Google satellite image). 

![QGISUnetPrediction](https://github.com/davdma/floodmaps/assets/42689743/07f27d36-138f-4365-ab8f-b846c7204ce3)


