# Benchmarking Models

Run model experiments with multiple random seeds to evaluate performance variability and compute robust statistics.

## Scripts
- `benchmark_s2.py` - Sentinel-2 optical model benchmarking
- `benchmark_sar.py` - Sentinel-1 SAR model benchmarking
- `benchmark_sar_ddp.py` - Distributed SAR benchmarking (multi-GPU)
- `benchmark_despeckling.py` - CVAE despeckler benchmarking
- `concatenate_summaries.py` - Combine summary CSVs for comparison

## Configuration

The benchmarking config group (`configs/benchmarking/`) requires:

```yaml
save_dir: ${paths.benchmarks_dir}/my_benchmark  # Output directory
trials: 10                                       # Number of trials
max_evals: 5                                     # Max evals per run (for walltime limits)
config_id: s2_unet_stride8                      # Unique identifier
description: "UNet with stride=8"               # Optional description
seed: 263932                                     # Master seed
```

## Usage

1. Create a benchmark config in `configs/benchmarking/` and enable it in `config.yaml`:

```yaml
defaults:
  - paths: default
  - benchmarking: my_benchmark
  - s2_unet_cfg  # model config to benchmark
  - _self_
```

2. Run the benchmark:

```bash
python -m floodmaps.benchmarking.benchmark_s2
```

If interrupted, rerun to resume from the last completed trial.

## Output Files

Each benchmark produces in `save_dir`:
- `{config_id}_trials.csv` - Individual trial results
- `{config_id}_config.yaml` - Full configuration snapshot
- `{config_id}_summary.csv` - Mean/std statistics (created when all trials complete)

## Comparing Benchmarks

After running multiple benchmarks, concatenate summaries:

```bash
python -m floodmaps.benchmarking.concatenate_summaries \
    --input-dir outputs/benchmarks/my_comparison \
    --output outputs/benchmarks/all_results.csv
```
