# Training Flood Detection Models

## Overview

This directory contains the complete machine learning pipeline for training flood detection models on satellite imagery. The pipeline supports both Sentinel-2 (optical) and Sentinel-1 (SAR) data with manual and machine-generated labels.

## Quick Start

### Prerequisites
1. **Downloaded Tiles**: For sampling raw satellite data, see the [sampling pipeline](../sampling/README.md)
1. **Preprocessed Patches**: Preprocessed satellite patches from [preprocessing pipeline](preprocess/README.md)
2. **Environment**: Python environment with PyTorch, wandb, and other dependencies installed via `envs/` conda environment yml files
3. **Wandb Account**: For experiment tracking and visualization

### Basic Training Commands

```bash
# Train S2 model
python training/train_s2.py --config_file configs/s2_template.yaml

# Train SAR model
python training/train_sar.py --config_file configs/sar_template.yaml
```

## Pipeline Architecture

### 1. Data Flow
```
sampling/ → preprocess/ → training/ → benchmarking/
   ↓            ↓           ↓           ↓
Raw Tiles →   Patches   →  Models   →  Metrics
```

### 2. Supported Datasets

Preprocessing script will store the output `.npy` files in the following filepaths:
- **S2 Manual**: `src/data/s2/` - Human-annotated ground truth (small scale)
- **S2 Weak**: `src/data/s2_weak/` - Machine-generated labels (large scale)  
- **S1 Weak**: `src/data/s1_weak/` - SAR with machine-generated labels

### 3. Supported Model Architectures
- **U-Net**: Standard encoder-decoder architecture
- **U-Net++**: Enhanced U-Net with nested skip connections
- **SAR Autodespeckler**: Optional denoising attachment (CNN, DAE, VAE, CVAE)

## Configuration Management

YAML-based configuration files control all training parameters. See [configs README](configs/README.md) for complete configuration reference.

## Benchmarking

### S2 Benchmark: [`benchmarking/benchmark_s2.py`](benchmarking/benchmark_s2.py)
Run multiple training trials with different random seeds for statistical evaluation.

```bash
python benchmarking/benchmark_s2.py \
    --config_file configs/s2_template.yaml \
    --trials 10 \
    --output_file results/s2_benchmark.csv
```

### SAR Benchmark: [`benchmarking/benchmark_sar.py`](benchmarking/benchmark_sar.py)
Comprehensive SAR model evaluation with autodespeckler tracking.

```bash
python benchmarking/benchmark_sar.py \
    --config_file configs/sar_template.yaml \
    --trials 5 \
    --output_file results/sar_benchmark.csv
```

**Features:**
- Statistical analysis across multiple trials
- Resumable benchmarking via checkpoints
- Comprehensive hyperparameter logging
- CSV export with mean/std performance metrics

## Experiment Tracking

### Wandb Integration
All training runs automatically log to [Weights & Biases](https://wandb.ai) for visualization and comparison.

**Logged Metrics:**
- Training/validation loss and accuracy
- Precision, recall, F1-score per epoch
- Learning rate scheduling
- Model parameters and memory usage

**Sample Predictions:**
- Input channels (RGB, SAR, ancillary data)
- Ground truth and model predictions
- False positive/negative analysis
- Reference imagery (TCI, NLCD)

## Common Workflows

### 1. Training a New Model
```bash
# 1. Ensure preprocessed data exists
ls src/data/s2_weak/samples_64_1000/

# 2. Create/modify config file
cp configs/s2_template.yaml configs/my_experiment.yaml

# 3. Train model
python training/train_s2.py --config_file configs/my_experiment.yaml

# 4. Monitor in wandb dashboard
```

### 2. Hyperparameter Tuning
```bash
# Use tuning scripts for systematic parameter search
python tuning/tuning_unet.py --config_file configs/s2_unet_tuning.yaml
```

### 3. Model Comparison
```bash
# Benchmark multiple models
python benchmarking/benchmark_s2.py --config_file configs/model_a.yaml --trials 10
python benchmarking/benchmark_s2.py --config_file configs/model_b.yaml --trials 10

# Compare results in wandb or CSV outputs
```

## Directory Structure
```
src/
├── benchmarking/         # Multi-trial statistical evaluation
├── configs/              # Training configuration files
├── data/                 # Preprocessed training patches
├── models/               # Model architectures and utilities
├── preprocess/           # Data preprocessing scripts
├── results/              # Saved experiments and benchmarks
├── training/             # Training scripts and utilities
├── tuning/               # Hyperparameter optimization scripts
└── utils/                # Shared utilities and helpers
```
