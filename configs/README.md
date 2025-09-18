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
  - __self__                # Where to place this config's content

# This config's own parameters
hydra:
  run:
    dir: ${paths.base_dir}/hydra
```

In this case you have `paths/default.yaml` used to specify paths across the repo, `inference/inference_s2.yaml` for inference params and `s2_unet_infer_v2.yaml` used to define the model that we want to use for inference.

Developer note: for maximum flexibility, the config files are not type checked or validated by any schema (beyond some basic asserts). For this reason, follow the structure to avoid running into errors at runtime.

### Model Configs

**Important:** a choice was made to keep model param config files at the same level as `config.yaml` instead of a config group or subdirectory like `configs/training/` due to how often they are needed and used in code. Consequently, when specifying model params, the config file e.g. `s2_unet.yaml` must at the same level as `config.yaml` and specified like this:

```yaml
# inside config.yaml
defaults:
  - s2_unet                 # UNET model training params
  - __self__
```

### The `__self__` Directive

`__self__` controls merge order in the defaults list. Put it at the end so your main config overrides everything else:

```yaml
defaults:
  - paths: default
  - sampling: sample_s2
  - __self__              # Main config has final say

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
save: True
save_path:
data:
    size: 64 # pixel width of dataset patches
    samples: 1000 # 1000
    channels: "11111011111" # (R, G, B, B08, NDWI, DEM, SlopeY, SlopeX, Water, Roads, Flowlines)
    use_weak: False # Use s2_weak (machine labels) vs s2 (manual labels)
    suffix: "" # Optional suffix for preprocessing variant datasets
    random_flip: True
model:
    classifier: "unet" # ['unet', 'unet++']
    discriminator: # ['classifier1', 'classifier2', 'classifier3']
    weights:
    unet:
        dropout: 0.2392593256577808
    unetpp:
        dropout:
        deep_supervision:
    discriminator_weights:
train:
    epochs: 400
    batch_size: 4608
    loss: BCELoss # ['BCELoss', 'BCEDiceLoss', 'TverskyLoss']
    tversky:
      alpha:
    clip: 1.0
    lr: 0.0057555553918297
    optimizer: Adam # ['Adam', 'SGD']
    LR_scheduler: Constant # ['Constant', 'ReduceLROnPlateau', 'CosAnnealingLR']
    LR_patience:
    LR_T_max: 200
    early_stopping: True
    patience: 20
    num_workers: 7
    checkpoint:
        load_chkpt: False # whether to train from a checkpoint specified in load_chkpt_path
        load_chkpt_path:
        save_chkpt: True # whether to save checkpoints
        save_chkpt_path:
        save_chkpt_interval: 20 # save checkpoint every N epochs
eval:
    mode: val # ['val', 'test'],
wandb:
    project: debug
    group:
    num_sample_predictions: 40
logging:
    grad_norm_freq: 10
```

For S1 training to submit to `train_s1.py` the structure looks like:
```yaml
seed: 831002
save: False
save_path:
data:
    size: 68 # pixel width of dataset patches
    window: 64 # pixel width of model input/output
    samples: 1000 # [250, 500, 1000]
    channels: "11011111" # (VV, VH, DEM, SlopeY, SlopeX, Water, Roads, Flowlines)
    use_lee: False
    suffix: "" # Optional suffix for preprocessing variant datasets
    random_flip: True
model:
    classifier: "unet++" # ['unet', 'unet++']
    weights:
    unet:
        dropout:
    unetpp:
        dropout:
        deep_supervision:
    autodespeckler:
        ad_config:
        ad_weights:
        freeze:
        freeze_epochs:
train:
    epochs: 200
    batch_size: 256
    loss: BCELoss # ['BCELoss', 'BCEDiceLoss', 'TverskyLoss']
    shift_invariant: True
    balance_coeff:
    tversky:
      alpha:
    clip: 1.0
    lr: 0.0005913
    optimizer: Adam # ['Adam', 'SGD']
    LR_scheduler: Constant # ['Constant', 'ReduceLROnPlateau', 'CosAnnealingLR']
    LR_patience:
    LR_T_max:
    early_stopping: True
    patience: 10
    subset: 0.15
    num_workers: 5
    checkpoint:
        load_chkpt: False # whether to train from a checkpoint specified in load_chkpt_path
        load_chkpt_path:
        save_chkpt: True # whether to save checkpoints
        save_chkpt_path:
        save_chkpt_interval: 20 # save checkpoint every N epochs
eval:
    mode: val # ['val', 'test'],
wandb:
    project:
    group:
    num_sample_predictions: 40
logging:
    grad_norm_freq: 10
```

### Tuning
For tuning you can provide a config file with the hyperparameter fields you want to tune blank. In the tuning script you can then override those hyperparameters. This makes it easy to set all the non tuned hyperparameters for the model before you start tuning.

For example for tuning we can use this snippet:
```yaml
model:
    classifier: "unet" # ['unet', 'unet++']
    discriminator:
    weights:
    unet:
        dropout: # keep this blank so we can override in the tuning script
    unetpp:
        dropout:
        deep_supervision:
    discriminator_weights:
```

Here we instantiate the UNet model and keep the `dropout` hyperparameter blank for the tuning job to specify.