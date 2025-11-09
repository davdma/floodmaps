# AI Flood Detection
### Background
Floods are a common and devastating natural hazard. There is a pressing need to replace traditional flood mapping methods with faster more accurate AI-assisted models that harness satellite imagery.

### Goals
The AI-assisted flood detection project aims to build a computer vision workflow for flood detection capable of generating flood observation from radar, satellite and urban imaging systems.

![image](https://github.com/davdma/floodmaps/assets/42689743/0685799c-7ab7-4640-9ae4-759b797dd13f)

We aim to accomplish this task through the following process:
1. **Data Collection:** Collect and process remote sensing data (visual, infrared, SAR) following extreme precipitation events for flood modeling.
2. **Manual Annotation:** Produce accurate ground truth data by manually labeling images.
3. **Initial Model:** Train an initial Sentinel-2 CNN model on small human labeled dataset to make quality predictions on unlabelled images, and augment the dataset with the new machine labels. This enables an iterative process: as label quantity and quality improves, the model training and prediction improves, which in turn leads to better weak labels.
4. **Final Model(s):** Develop a separate Sentinel-2 and Sentinel-1 SAR CNN model using the cumulative manual + machine labels to detect flood water extent after a flood event.

## Data Pipeline
To collect and process satellite imagery, an automated data pipeline was implemented in the Python script `sample_s2.py` for Sentinel-2 only and `sample_s2_s1.py` for Sentinel-2 and Sentinel-1. The scripts download **4km x 4km geospatial "tiles"** which make up an indivisible unit within our dataset. The scripts are run on an Argonne computing cluster and submitted as slurm or pbsnodes jobs. For more information on setting up and using those scripts, consult the [sampling](sampling/README.md) folder documentation.

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

As a result, we were able to produce 111+ 4km by 4km manually labeled flood tiles spread across the midwestern states (Illinois, Wisconsin, Indiana, Iowa) and southern states (Texas, Mississippi, Louisiana, Florida, NC, SC). This does not include the hundreds of additional weak labels (generated by the best trained model on unlabeled data) used to augment the datasets for iterative model training and development.

<div align="center">
<img width="75%" alt="labelmapwtile" src="https://github.com/user-attachments/assets/dd5500cb-4f36-40f4-b7f9-e1f730689630" />
<p style="width: 60%; margin: 0 auto;"><em>Figure 2.3: Map of manually labeled tiles grouped into Illinois region and Texas region.
</em></p>
</div>

## Dataset Splitting and Stratification

To split the manually labeled S2 dataset into train, validation, and test set, particular care was made towards avoiding data leakage and maintaining similar distributions across the splits:

1. **Tile-level split:** While the 4km x 4km tiles were broken into smaller patches in the preprocessing stage, the dataset split was done on a tile level to avoid tile-specific information leaking from the training to target.
2. **Acquisition-level split:** In the dataset, many adjacent tiles were cropped from the same acquisition (i.e. S2 data product) which can span 110km in width. Tiles from the same acquisition were kept in the same split to avoid acquisition-specific information leaking from the training to target.
3. **Stratification of weather conditions:** Similar ratios of tiles with **haze, cloud cover, and cloud shadows** are maintained across the splits to ensure representativeness and robustness of the model across varying weather conditions.

<div align="center">
<img width="90%" alt="s2_labeled_split_lowres" src="https://github.com/user-attachments/assets/e31503d5-e702-4361-ae33-07dd61654e17" />
<p style="width: 60%; margin: 0 auto;"><em>Figure 3: S2 dataset split.
</em></p>
</div>

## Patching Strategy

Given that each of the 111+ labeled tiles produced are around 400 x 400 in size, we needed a patching strategy to break these tiles down into digestable patches (e.g. of size 64 x 64) that the model can train on. An initial exploratory [notebook](notebooks/unet.ipynb) found that discretely patching the tile (zero overlap) performed much worse than a random sampling strategy where 500 64 x 64 patches were randomly chosen from the larger tile to be added to the dataset.

However, the random sampling strategy also has its pitfalls. Since patches sampled must fall within bounds, random sampling has a bias towards the interior of the tile over the edges. Furthermore, when choosing the number of patches to sample it was hard not to oversample and create redundancy (too many similar patches) in the dataset. After running benchmarks on different strategies, we ultimately found that a **strided patch sampling strategy** outperformed random patch sampling with a **stride of 16** being the most optimal.

<div align="center">
<img width="75%" alt="sampling_benchmarks" src="https://github.com/user-attachments/assets/fb13a9fe-b3a7-4fbc-bae4-787c8111227e" />
<p style="width: 60%; margin: 0 auto;"><em>Figure 2.4: Benchmarked metrics of UNet trained on the same S2 dataset across different patching strategies. For random sampling, `N` is the number of patches sampled from each tile with uniform distribution across coordinates. For strided sampling, `S` is the stride of individual patches across the tile.
</em></p>
</div>

## S2 Model
For our water pixel detection model, we tested multiple built-in architectures that have been used in the flood modelling literature extensively, most commonly UNet and UNet++. Our model input consists of the RGB spectral bands, the NIR B8 band, the SWIR B11-12 bands, water indices like NDWI, MNDWI, AWEI, as well as the option of selecting additional channels such as the Digital Elevation Map (DEM), slope, roads, waterbodies, and flowlines.

<div align="center">
<img width="75%" alt="labelmapwtile" src="https://github.com/davdma/floodmaps/assets/42689743/78d029d1-2f32-4991-b62f-c5d6d6ca0167" />
<p style="width: 60%; margin: 0 auto;"><em>Figure 6: Prediction results on a large flood tile. Using our initial model on unlabelled data allows us to automate our ground truthing process.
</em></p>
</div>

With the initial tuned UNet and UNet++ models, we visualized their predictions on unseen flood tiles in QGIS with georeferencing. Overlaid on high resolution satellite images, we were able to identify patterns and see how the model performs in practice. In the example below, we see that the model is a very powerful predictor of open water, and has a strong ability to distinguish fine outlines of water bodies. We also see some limitations of our model: due to the 10m resolution of the input channels, water bodies that are <10m have a higher likelihood of going undetected. Another limitation of note is cloud cover. In practice a percentage of the input satellite images will be obscured by clouds, making it hard for the model to see the waterbodies, and in the example below the model performs the worst in areas obscured by cloud cover (top right corner of waterbody - these clouds are not visible in the high res overlaid Google satellite image). 

![QGISUnetPrediction](https://github.com/davdma/floodmaps/assets/42689743/07f27d36-138f-4365-ab8f-b846c7204ce3)
**Figure 7:** Prediction results overlaid on high resolution satellite imagery in QGIS.

## SAR S1 Model

Using our best initial S2 model, we made predictions on unlabelled data with coincident S1 captures (within 12 hours) to build a much larger dataset of 400+ flood event tiles across different geographic regions of the United States (including areas impacted by Hurricane Harvey). The temporal coincidence meant that our machine labels would most accurately reflect the flood water present in the SAR data. With the augmented dataset, we trained and tested different model architectures, and explored the potential advantage of an **autodespeckler** attachment to a regular segmentation model. The autodespeckler is an autoencoder that takes as input the SAR channels of each patch, and extracts the seminal features from the SAR data for the classifier to use, with the aim of reducing the impact of SAR speckle on prediction quality.

![sarworkflow](https://github.com/davdma/floodmaps/assets/42689743/2fdf3016-cc61-4e41-8118-bc3bf460ffa7)
**Figure 8:** The process for developing the final SAR flood prediction model.

## Results

Each model is tuned using Bayesian optimization with the deephyper package, and subsequently benchmarked on the test set with the tuned hyperparameters.
**Our best standalone classifier was a UNet++ SAR model that achieved a F1 score of 92.79, an accuracy of 97.14, precision of 93.82 and recall of 91.78.** The autodespeckler tuning is still TBD. Benchmarking results will also be posted soon.

![sarunet++model](https://github.com/davdma/floodmaps/assets/42689743/6e279d68-4597-4755-914c-532ca61d7206)

# The Autodespeckler

The autodespeckler attachement was added to tackle the speckle noise present in SAR input data that degrades its quality and interpretability for the SAR flood mapping models. Using a Conditional-VAE architecture, we trained the model on multitemporal composites of SAR images. The result was a decoder that was able to generate synthetic "clean" SAR images from noisy SAR input.

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




