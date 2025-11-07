# Training Flood Detection Models

## Overview

This directory contains the complete machine learning pipeline for training flood detection models on satellite imagery. The pipeline supports both Sentinel-2 (optical) and Sentinel-1 (SAR) data with manual and machine-generated labels.

## Directory Structure
```
floodmaps/
├── analysis/             # Permutation importance, model analyses
├── benchmarking/         # Multi-trial statistical evaluation
├── inference/            # Inference floodmaps from areas of interest
├── models/               # Model architectures and utilities
├── preprocess/           # Data preprocessing scripts
├── sampling/             # Data sampling scripts
├── training/             # Training scripts and utilities
├── tuning/               # Hyperparameter optimization scripts
└── utils/                # Shared utilities and helpers
```

## Pipeline

### Prerequisites
1. **Environment**: Python environment with `torch`, `wandb`, `hydra`, and other dependencies installed via `envs/` conda environment yml files
2. **Wandb Account**: For experiment tracking and visualization
3. **API Key(s)**: Needed if using the repo to download new satellite data using Microsoft PlanetarySTAC (recommended) or Copernicus STAC or AWS S3 bucket

### Workflow
```
sampling/ → preprocess/ → training/ → tuning/ → benchmarking/
   ↓            ↓           ↓           ↓           ↓
Raw Tiles →  Patches    →  Models  →  Config →   Metrics
```

1. **Downloading Tiles**: For sampling raw satellite data, see the [sampling doc](sampling/README.md)
2. **Preprocessing Patches**: For preprocessing satellite patches from sampled tiles, see the [preprocessing doc](preprocess/README.md)
3. **Training Models**: For training models from preprocessed patches, see the [training doc](training/README.md)
4. **Tuning Models**: For tuning models with bayesian hyperparameter search, see the [tuning doc](tuning/README.md)
5. **Benchmarking Models**: For benchmarking models, see the [benchmarking doc](benchmarking/README.md) 

### Configuration Management

`hydra`-based configuration files control inputs to all scripts in the directory. See [configs README](../../configs/README.md) for complete configuration reference.

### Supported Datasets

In the default config, the preprocessing script will store the output `.npy` files in the following filepaths:
- **S2 Manual**: `data/preprocess/s2/` - Multispectral patches with human-annotated ground truth (small scale)
- **S2 Weak**: `data/preprocess/s2_weak/` - Multispectral patches with machine-generated labels (large scale)  
- **S1 Weak**: `data/preprocess/s1_weak/` - SAR patches with machine-generated labels

### Supported Model Architectures
- **U-Net**: Standard encoder-decoder architecture
- **U-Net++**: Enhanced U-Net with nested skip connections
- **SAR Autodespeckler**: Optional SAR denoising attachment (C-VAE)