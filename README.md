# AI Flood Detection
### Background
Floods are a common and devastating natural hazard. There is a pressing need to replace traditional flood mapping methods with faster more accurate AI-assisted models that harness satellite imagery.

### Goals
The AI-assisted flood detection project aims to build a computer vision workflow for flood detection capable of generating flood observation from radar, satellite and urban imaging systems.

To accomplish this task, we want to:
1. Collect and process remote sensing data (visual, infrared, SAR) following extreme precipitation events for flood modeling.
2. Produce accurate ground truth data by manually labeling images.
3. Develop a water pixel detection algorithm to detect flood water extent after a flood event.

## Data Pipeline

**Ground-Truthing instructions:** [pdf here](https://1drv.ms/b/s!Aq3V83mBle0dvhMcZAiCh04A59--?e=IdSswS)

## Model

### Code
* Sentinel-2 data pipeline code can be found in the script `sample_mpc.py`.
* The water pixel detection model is still in progress, but can be found in the notebook `unet.ipynb`.
