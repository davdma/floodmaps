# Config File Structure
This document explains how to use [Hydra](https://hydra.cc/) for managing configurations for sampling data, running experiments, inference, tuning and more in the repo.

### Basic Usage
The main entry point to the configuration is the `config.yaml` file at the base of the `configs` directory. This is what the python scripts by default parses at runtime. Configs specific to sampling, preprocessing, inferencing etc. are organized inside `configs/` subdirectories and represent **config groups**.

A way to think about this is that `config.yaml` serves as the **composition root** that assembles different configuration pieces (i.e. files from different groups) together into the final `DictConfig` object given to the script.

For instance a `config.yaml` for running the `inference_s2.py` script might look like:
```yaml
defaults:
  - paths: default          # Config group: paths/default.yaml
  - inference: inference_s2  # Config group: inference/inference_s2.yaml  
  - s2_unet_infer_v2        # Base config: s2_unet_infer_v2.yaml
  - _self_                # Where to place this config's content

# This config's own parameters
hydra:
  run:
    dir: ${paths.base_dir}/hydra
```

As a sample of what the `DictConfig` object looks like from this, here is the output of `print(OmegaConf.to_yaml(cfg))` to the above `config.yaml` input:

```text
paths:
  base_dir: /lcrc/project/hydrosm/dma
  data_dir: ${.base_dir}/data
  output_dir: ${.base_dir}/outputs
  # ...
inference:
  format: tif
  replace: true
  data_dir: areas/safb_20240718/
seed: 831002
save: true
save_path: null
data:
  size: 64
  method: strided
  stride: 16
  channels: '1111111111011111'
  use_weak: false
  # ...
model:
  classifier: unet
  weights: ${paths.output_dir}/experiments/tmp_infer_s2_model/unet_cls.pth
  unet:
    dropout: 0.24
  unetpp:
    dropout: null
    deep_supervision: null
```

In this case you have `paths/default.yaml` used to specify paths across the repo, `inference/inference_s2.yaml` for inference params and `s2_unet_infer_v2.yaml` used to define the model that we want to use for inference. Since `paths` and `inference` are both config groups, their attributes are nested under `paths:` and `inference:` in the output config object. Observe that model params like `seed`, `save`, `save_path`, `data`, `model` that came from `s2_unet_infer_v2.yaml` are not indented under a shared `models:` group. See the reason for this below.

Developer note: for maximum flexibility, the config files are not type checked or validated by any schema (beyond some basic asserts). For this reason, follow the structure of the provided templates to avoid running into errors at runtime.

### Model Configs

**Important:** a choice was made to keep model configs at the same level as `config.yaml` instead of inside a config group or subdirectory like `configs/models/` due to how often they are needed and used in code. Consequently, when specifying model params, the config file e.g. `s2_unet.yaml` must at the same level as `config.yaml` and specified like this:

```yaml
# inside config.yaml
defaults:
  - s2_unet                 # Unet model training params
  - _self_
```

### API Keys

Two options for satellite download API keys: set in `config.yaml` or pass via environment variable.

Microsoft Planetary Computer:
- Set in the `config.yaml` level with `mpc_api_key: ...`.
- OR Set environment variable `PC_SDK_SUBSCRIPTION_KEY`.

AWS:
- Currently S3 requester pays bucket requires manual setup.

### The `__self__` Directive

`_self_` controls merge order in the defaults list. Put it at the end so your main config overrides everything else:

```yaml
defaults:
  - paths: default
  - sampling: sample_s2
  - _self_              # Main config has final say

sampling:
  threshold: 150  # This overrides threshold from sample_s2.yaml
```

### Variable Interpolation

Use `${...}` syntax to reference other config values:

```yaml
# From the paths/default.yaml
paths:
  base_dir: /lcrc/project/hydrosm/dma
  data_dir: ${.base_dir}/data
  output_dir: ${.base_dir}/outputs

# In your main config
model:
    weights: ${paths.output_dir}/experiments/best_model/unet_cls.pth
```

### Training
For training the config structure can differ slightly whether you are training S2 or S1 models.

For S2 training to submit to `train_s2.py` the structure looks like:
```yaml
seed: 831002
save: true
save_path:

data:
    size: 64 # pixel width of dataset patches
    method: "strided" # ["random", "strided"]
    samples: # Number of patches per tile (for "random" method)
    stride: 16 # Stride for sliding window (for "strided" method)
    channels: "1111111111011111" # Binary string selecting 16 available input channels
                                 # (R, G, B, B08, SWIR1, SWIR2, NDWI, MNDWI, AWEI_sh, AWEI_nsh, DEM, SlopeY, SlopeX, Water, Roads, Flowlines)
    random_flip: true
    mmap: false # Use memory-mapped file loading for large training datasets
    use_weak: false # Use s2_weak (machine labels) vs s2 (manual labels)
    suffix: "all_clouds_v2-4" # Optional suffix for preprocessing variant datasets

model:
    classifier: "unet++" # ['unet', 'unet++']
    weights: # path to pretrained weights (for inference/finetuning)
    unet:
        dropout:
    unetpp:
        dropout: 0.31
        deep_supervision: false

train:
    epochs: 300
    batch_size: 2048
    loss: BCEDiceLoss # ['BCELoss', 'BCEDiceLoss', 'TverskyLoss', 'FocalTverskyLoss']
    use_pos_weight: true   # enable positive class weighting for BCE/BCEDice
    pos_weight:            # null -> auto-compute from train labels; set float to override
    pos_weight_clip: 10.0  # clip auto-computed weight to [1, clip]
    tversky:
      alpha: 0.3
    clip: 1.0
    lr: 0.008
    optimizer: Adam # ['Adam', 'SGD', 'AdamW']
    weight_decay:
    LR_scheduler: Constant # ['Constant', 'ReduceLROnPlateau', 'CosAnnealingLR']
    LR_patience: 10
    LR_T_max:
    early_stopping: true
    patience: 20
    num_workers: 4
    checkpoint:
        load_chkpt: false
        load_chkpt_path:
        save_chkpt: false
        save_chkpt_path:
        save_chkpt_interval: 20

eval:
    mode: test # ['val', 'test']

wandb:
    project: S2_Training
    group:
    num_sample_predictions: 40
    percent_wet_patches: 0.8

logging:
    grad_norm_freq: 10
```

For S1 SAR training to submit to `train_sar.py` (single GPU) or `train_sar_ddp.py` (multi-GPU):
```yaml
seed: 33752
save: true
save_path:

data:
    size: 68 # pixel width of dataset patches
    window: 64 # pixel width of model input/output (smaller due to shift-invariant loss)
    method: "strided" # ["random", "strided"]
    samples:
    stride: 68
    channels: "11011111" # Binary string selecting 8 available input channels
                         # (VV, VH, DEM, SlopeY, SlopeX, Water, Roads, Flowlines)
    random_flip: true
    mmap: true
    suffix: "all"

model:
    classifier: "unet++" # ['unet', 'unet++']
    weights:
    unet:
        dropout:
    unetpp:
        dropout: 0.23
        deep_supervision: false
    autodespeckler: # optional CVAE despeckler attachment
        ad_config:
        ad_weights:
        freeze:
        freeze_epochs:

train:
    epochs: 400
    batch_size: 1024
    loss: BCELoss # ['BCELoss', 'BCEDiceLoss', 'TverskyLoss', 'FocalTverskyLoss']
    use_pos_weight: true
    pos_weight:
    pos_weight_clip: 30.0
    shift_invariant: true # align labels to predictions for shift-robust training
    tversky:
      alpha: 0.3
    focal_tversky:
      gamma: 1.33
    clip: 1.0
    lr: 0.0097
    optimizer: Adam # ['Adam', 'SGD', 'AdamW']
    weight_decay:
    LR_scheduler: Constant # ['Constant', 'ReduceLROnPlateau', 'CosAnnealingLR']
    LR_patience: 10
    LR_T_max:
    early_stopping: true
    patience: 30
    num_workers: 4
    keep_contiguous_in_mem: true # keep mmap'd data contiguous for faster access
    checkpoint:
        load_chkpt: false
        load_chkpt_path:
        save_chkpt: false
        save_chkpt_path:
        save_chkpt_interval: 20

eval:
    mode: test # ['val', 'test']

wandb:
    project: S1_Training
    group:
    num_sample_predictions: 40
    percent_wet_patches: 0.8

logging:
    grad_norm_freq: 10
```

### Tuning
For tuning you can provide a config file with the hyperparameter fields you want to tune blank. In the tuning script you can then override those hyperparameters. This makes it easy to set all the non tuned hyperparameters for the model before you start tuning.

For example for tuning we can use this snippet:
```yaml
model:
    classifier: "unet" # ['unet', 'unet++']
    weights:
    unet:
        dropout: # keep this blank so we can override in the tuning script
    unetpp:
        dropout:
        deep_supervision:
```

Here we instantiate the UNet model and keep the `dropout` hyperparameter blank for the tuning job to specify.