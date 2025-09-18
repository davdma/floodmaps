# Inference

For creating floodmaps of areas of interest outside of the dataset, provide a path
to the shapefile of a geometry, e.g. Scott Air Force Base boundary in Illinois, 
to the `sampling/download_area.py` script - this will download the satellite imagery
and necessary supplementary rasters for the area.

The inference scripts can then be used with a trained model in the repo to generate
a segmentation mask for the area.

# Weak Labeling

To create weak labelled datasets for iterative model training, provide an already trained model
config (with specified weights path) to the `weak_labels.py` script. It will make predictions
on a dataset of interest, which are saved as `pred_*.tif` files. This can then be used in training.