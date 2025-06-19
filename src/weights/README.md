# Best Model Weights

### S2 Multispectral Flood Mapping Model
* `s2_unet_v1.pth` stores the weights for the benchmarked UNet model with 98.8% accuracy, 93.8% f1, 95.9% precision and 91.9% recall on the test set.
* The model does not use the DEM channel, so the channel setting is `1111101111` with dropout = `0.2987776077544917`.

### S1 SAR Flood Mapping Model
* `s1_unetpp_v1.pth` stores the weights for the benchmarked UNet++ SAR model with 96.5% accuracy, 91.2% f1, 93.2% precision and 89.8% recall on the test set.
* The model uses all channels with channel setting `1111111` with dropout = `0.0531091802785671` and deep_supervision = `True`.