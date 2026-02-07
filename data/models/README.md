# Model Weights

The `.pth` weights and corresponding model hyperparameters in `.yaml` files for models described in the publication are provided.

### S2 Multispectral Flood Mapping Model
* `s2_unet_all_best.pth` - benchmarked UNet model using all channels (except DEM).
    * Use threshold `t=0.5` for inference. Achieves **89.77\% F1, 91.69\% Recall, 87.94\% Precision** on test set.
* `s2_unetpp_all_best.pth` - benchmarked UNet++ model using all channels (except DEM).
    * Use calibrated threshold `t=0.75` for inference. Achieves **89.98\% F1, 91.12\% Recall, and 88.86\% Precision** on test set.

### S1 SAR Flood Mapping Model
* `s1_unetpp_all_best.pth` - benchmarked UNet++ SAR model using all channels (except DEM).
    * Use calibrated threshold `t=0.85` for inference. Achieves **75.32\% F1, 63.61\% Recall, and 92.35\% Precision** on test set.
* `s1_unetpp_all_cvae_best.pth` - benchmarked UNet++ SAR model using all channels (except DEM), specifically trained for CVAE despeckled SAR images.
    * Use calibrated threshold `t=0.87` for inference. Achieves **76.06\% F1, 64.82\% Recall, and 92.02\% Precision** on test set.

### S1 Despeckling Model
* `cvae.pth` - benchmarked SAR CVAE despeckler for VV and VH channels with **25.92-27.09 PSNR, 0.63-0.66 SSIM and 188.7-379.5 ENL** on test set.