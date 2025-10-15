# Weak Labeling

To create weak labelled datasets for iterative model training, provide an already trained model
config (with specified weights path) to the `weak_labels.py` script. It will make predictions
on a dataset of interest, which are saved as `pred_*.tif` files inside the event directories.
This can then be used in training.

The `weak_labels_tci.py` script allows for weak labeling using the old TCI
input model (e.g. `data/models/s2_unet_v1.pth`), but is not compatible with the new data channels with RGB spectral bands.

# Inference

To generate a flood record for an area of interest across a time window use `sampling/download_aoi.py`.
The script takes a **time frame** and **shapefile** as input, and downloads all tiles across the time frame into one directory, along with necessary supplementary rasters. If a **product id** is provided, it will only download that product within the time frame.

The inference scripts `inference_s2.py` and `inference_sar.py` can then be used with a trained model in the repo to generate a segmentation mask for the samples. Make sure to pass the path to the directory containing the downloaded
products.
