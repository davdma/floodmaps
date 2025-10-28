# Benchmarking Models

## Overview

The benchmarking scripts run model experiments with multiple random seeds to evaluate performance variability and compute robust statistics. This is essential for comparing different model configurations on the test set.

## Benchmarking Configuration

The benchmarking config group requires the following parameters:

```yaml
save_dir: ${paths.benchmarks_dir}/compare_sampling  # Directory for all benchmark outputs
trials: 10                                           # Total number of trials per benchmark
max_evals: 5                                         # Max evaluations per script run (for walltime limits)
config_id: s2_unet_stride8                          # Unique identifier for this benchmark
description: "UNet with sliding window stride=8"    # Optional description
seed: 263932                                         # Master random seed for reproducibility
```

## Output Files

Each benchmark run produces three files in `save_dir`:

1. **`{config_id}_trials.csv`**: Individual trial results
   - Columns: `trial`, `seed`, plus all metrics (core, NLCD groups, SCL groups)
   - Used for checkpointing and detailed analysis
   - Automatically saved after each trial completion

2. **`{config_id}_config.yaml`**: Full configuration
   - Saved once at the start of benchmarking
   - Allows exact reproduction of the benchmark

3. **`{config_id}_summary.csv`**: Summary statistics
   - Created when all trials complete
   - Columns: metadata (config_id, description, seed, project, group, classifier, channels, trials)
   - Followed by: `mean_{metric}` and `std_{metric}` for all metrics
   - One row per benchmark configuration

## Workflow

### 1. Configure Your Benchmark

Create or modify a config file (e.g., `configs/benchmarking/my_benchmark.yaml`), then specify in your `configs/config.yaml`:

```yaml
defaults:
  - paths: default
  - sampling:
  - preprocess:
  - inference:
  - tuning:
  - benchmarking: my_benchmark
  - s2_unet_cfg # model experiment config to benchmark
  - _self_
```

### 2. Run the Benchmark

```bash
# Run benchmarking script
python -m floodmaps.benchmarking.benchmark_s2
```

If interrupted, just run the script again. It automatically resumes from the last completed trial.

### 4. Compare Multiple Benchmarks

After running multiple benchmark configurations, concatenate all summary CSVs (containing metric mean, stds) for easy comparison. A utility script is provided.

```bash
# Concatenate all *_summary.csv files in a directory
python -m floodmaps.benchmarking.concatenate_summaries \
    --input-dir outputs/benchmarks/sampling_comparison \
    --output outputs/benchmarks/all_sampling_comparison.csv
```

This creates a single CSV with one row per benchmark configuration, making it easy to compare results.

## Metrics Tracked

The benchmark automatically captures all available metrics:

### Core Metrics
- Accuracy
- Precision
- Recall
- F1 Score
- IoU (Jaccard Index)

### NLCD Group Metrics
Per land cover class group (urban, forest, cultivated, etc.):
- Accuracy
- Precision
- Recall
- F1 Score
- IoU

### SCL Group Metrics
Per scene classification group (vegetation, water, cloud, etc.):
- Accuracy
- Precision
- Recall
- F1 Score
- IoU

All metrics are automatically flattened and included in the trials and summary CSVs with descriptive column names like `nlcd_urban_f1` or `scl_water_iou`.