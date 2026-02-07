from random import Random
import pandas as pd
import numpy as np
import hydra
from omegaconf import DictConfig, OmegaConf
from pathlib import Path
from typing import Dict, Any, Optional
import torch
import torch.multiprocessing as mp

from floodmaps.training.train_sar_ddp import run_experiment_s1 as run_experiment_s1_ddp, find_free_port

# To avoid fork vs spawn context conflicts
mp.set_start_method("spawn", force=True)

def flatten_group_metrics(metrics_dict: Dict[str, Any], prefix: str) -> Dict[str, Any]:
    """Flatten nested group metrics into a single-level dictionary.
    
    Takes group_metrics and group_auprc from nlcd_metrics or scl_metrics and creates
    flattened keys like 'nlcd_urban_acc', 'scl_water_prec', 'nlcd_urban_auprc', etc.
    
    Parameters
    ----------
    metrics_dict : dict
        Dictionary containing 'group_metrics' and optionally 'group_auprc' with nested structure
    prefix : str
        Prefix for flattened keys (e.g., 'nlcd' or 'scl')
    
    Returns
    -------
    dict
        Flattened dictionary with keys like '{prefix}_{group}_{metric}'
    
    Examples
    --------
    >>> nlcd_metrics = {'group_metrics': {'urban': {'acc': 0.95, 'f1': 0.90}}, 'group_auprc': {'urban': 0.85}}
    >>> flatten_group_metrics(nlcd_metrics, 'nlcd')
    {'nlcd_urban_acc': 0.95, 'nlcd_urban_f1': 0.90, 'nlcd_urban_auprc': 0.85}
    """
    flat_dict = {}
    # Flatten group_metrics (acc, prec, rec, f1, iou)
    if 'group_metrics' in metrics_dict:
        for group_name, group_metrics in metrics_dict['group_metrics'].items():
            # Sanitize group name for use in column names
            safe_group = group_name.replace(' ', '_').replace('/', '_')
            for metric_name, metric_value in group_metrics.items():
                key = f"{prefix}_{safe_group}_{metric_name}"
                flat_dict[key] = metric_value
    
    # Flatten group_auprc
    if 'group_auprc' in metrics_dict:
        for group_name, auprc_value in metrics_dict['group_auprc'].items():
            safe_group = group_name.replace(' ', '_').replace('/', '_')
            flat_dict[f"{prefix}_{safe_group}_auprc"] = auprc_value
    
    return flat_dict


def extract_trial_metrics(fmetrics, split: str, partition: str) -> Dict[str, Any]:
    """Extract and flatten all metrics from a trial run.
    
    Extracts core metrics, NLCD group metrics, SCL group metrics, and shift
    distribution, flattening them into a single dictionary.
    
    Parameters
    ----------
    fmetrics : Metrics
        Metrics object from training run
    split : str
        Data split ('test' or 'val')
    partition : str
        Partition type for SAR metrics ('shift_invariant', 'non_shift_invariant', or 'aligned')
    
    Returns
    -------
    dict
        Flattened dictionary of all metrics
    """
    all_metrics = fmetrics.get_metrics(split=split, partition=partition)
    
    flat_metrics = {}
    
    # Extract core metrics (e.g., 'test accuracy', 'test precision', etc.)
    if 'core_metrics' in all_metrics:
        flat_metrics.update(all_metrics['core_metrics'])
    
    # Extract and flatten NLCD group metrics
    if 'nlcd_metrics' in all_metrics:
        flat_metrics.update(flatten_group_metrics(all_metrics['nlcd_metrics'], 'nlcd'))
    
    # Extract and flatten SCL group metrics
    if 'scl_metrics' in all_metrics:
        flat_metrics.update(flatten_group_metrics(all_metrics['scl_metrics'], 'scl'))
    
    # Extract shift distribution (only present for shift_invariant partition)
    if 'shift_distribution' in all_metrics:
        flat_metrics.update(all_metrics['shift_distribution'])
    
    return flat_metrics


def load_trials_checkpoint(trials_path: Path) -> pd.DataFrame:
    """Load trials checkpoint CSV or return empty DataFrame.
    
    Parameters
    ----------
    trials_path : Path
        Path to trials CSV file
    
    Returns
    -------
    pd.DataFrame
        DataFrame with completed trial results, or empty DataFrame if no checkpoint exists
    """
    if trials_path.exists():
        try:
            df = pd.read_csv(trials_path)
            print(f"Loaded checkpoint with {len(df)} completed trials")
            return df
        except Exception as e:
            print(f"Warning: Could not load checkpoint CSV: {e}")
            return pd.DataFrame()
    return pd.DataFrame()


def save_trials_checkpoint(trials_path: Path, trials_df: pd.DataFrame) -> None:
    """Save trials checkpoint to CSV.
    
    Parameters
    ----------
    trials_path : Path
        Path to save trials CSV
    trials_df : pd.DataFrame
        DataFrame with trial results
    """
    trials_df.to_csv(trials_path, index=False)


def save_config(config_path: Path, cfg: DictConfig) -> None:
    """Save configuration to YAML file (once per benchmark).
    
    Parameters
    ----------
    config_path : Path
        Path to save config YAML
    cfg : DictConfig
        Configuration object
    """
    if not config_path.exists():
        with open(config_path, 'w') as f:
            OmegaConf.save(cfg, f)
        print(f"Configuration saved to {config_path}")


def compute_summary_statistics(trials_df: pd.DataFrame, 
                               exclude_cols: list = None,
                               prefix: str = '') -> Dict[str, Any]:
    """Compute mean and std for all numeric metric columns.
    
    Parameters
    ----------
    trials_df : pd.DataFrame
        DataFrame with trial results
    exclude_cols : list, optional
        Columns to exclude from statistics (e.g., metadata columns)
    
    Returns
    -------
    dict
        Dictionary with mean and std for each metric
    """
    if exclude_cols is None:
        exclude_cols = ['trial', 'seed']
    
    stats = {}
    numeric_cols = trials_df.select_dtypes(include=[np.number]).columns
    metric_cols = [col for col in numeric_cols if col not in exclude_cols]
    
    for col in metric_cols:
        # Drop NaN/None values before computing statistics
        valid_values = trials_df[col].dropna()
        
        if len(valid_values) > 0:
            stats[f'mean_{prefix}_{col}'] = float(valid_values.mean())
            stats[f'std_{prefix}_{col}'] = float(valid_values.std())
        else:
            # All values were None/NaN
            stats[f'mean_{prefix}_{col}'] = None
            stats[f'std_{prefix}_{col}'] = None
    
    return stats


def save_summary(summary_path: Path, summary: Dict[str, Any]) -> None:
    """Save summary statistics to CSV.
    
    Parameters
    ----------
    summary_path : Path
        Path to save summary CSV
    summary : dict
        Summary statistics dictionary
    """
    summary_df = pd.DataFrame([summary])
    summary_df.to_csv(summary_path, index=False)
    print(f"Summary saved to {summary_path}")


def run_experiment_s1_ddp_wrapper(rank, world_size, free_port, cfg, ad_cfg, result_queue):
    fmetrics = run_experiment_s1_ddp(rank, world_size, free_port, cfg, ad_cfg)

    # Only rank 0 reports the result
    if rank == 0 and fmetrics is not None:
        result_queue.put(fmetrics)


def benchmark_sar(cfg: DictConfig) -> None:
    """Benchmarks SAR classifier model on test set with multiple random seeds,
    uses the DDP script.
    
    Runs n trials with different random seeds, collects all metrics for shift
    and non-shift invariant cases(core metrics, NLCD group metrics, SCL group metrics),
    and computes summary statistics. Supports checkpointing for resuming interrupted runs.
    
    The benchmark saves three files:
    1. {config_id}_shift_trials.csv - Individual trial results with all shift-invariant metrics
    2. {config_id}_non_shift_trials.csv - Individual trial results with all non-shift-invariant metrics
    3. {config_id}_config.yaml - Configuration used for this benchmark
    4. {config_id}_summary.csv - Mean and std statistics across all trials for shift-invariant and non-shift-invariant cases
    
    Configuration Parameters
    ------------------------
    cfg.eval.mode : str
        Must be 'test' for benchmarking
    cfg.benchmarking.save_dir : str
        Directory to save benchmark results
    cfg.benchmarking.config_id : str
        Unique identifier for this benchmark configuration
    cfg.benchmarking.description : str, optional
        Description of what's unique about this benchmark run
    cfg.benchmarking.trials : int
        Total number of trials to run
    cfg.benchmarking.max_evals : int
        Max evaluations per script run (for handling job time limits)
    cfg.benchmarking.seed : int
        Master random seed for reproducibility
    
    Raises
    ------
    AssertionError
        If eval.mode is not 'test'
    
    Notes
    -----
    The script automatically resumes from the last completed trial if interrupted.
    Progress is saved after each trial to ensure no work is lost.
    
    SAR-specific: Both shift invariant and non-shift invariant metrics are benchmarked.
    """
    assert cfg.eval.mode == 'test', 'Benchmarking must be run on test set.'
    
    # Get autodespeckler config if present
    ad_cfg = getattr(cfg, 'ad', None)
    
    # Setup paths
    save_dir = Path(cfg.benchmarking.save_dir)
    save_dir.mkdir(parents=True, exist_ok=True)
    config_id = cfg.benchmarking.config_id
    shift_trials_path = save_dir / f"{config_id}_shift_trials.csv"
    non_shift_trials_path = save_dir / f"{config_id}_non_shift_trials.csv"
    aligned_trials_path = save_dir / f"{config_id}_aligned_trials.csv"
    config_path = save_dir / f"{config_id}_config.yaml"
    summary_path = save_dir / f"{config_id}_summary.csv"
    
    # Check if aligned metrics are available (shift ablation enabled)
    is_shift_ablation_available = getattr(cfg.data, 'shift_ablation', False)
    
    # Save config (only once per benchmark)
    save_config(config_path, cfg)
    
    # Load or initialize trials DataFrame
    shift_trials_df = load_trials_checkpoint(shift_trials_path)
    non_shift_trials_df = load_trials_checkpoint(non_shift_trials_path)
    aligned_trials_df = load_trials_checkpoint(aligned_trials_path) if is_shift_ablation_available else pd.DataFrame()
    count = len(shift_trials_df)
    
    if count > 0:
        print(f"Resuming from trial {count + 1}/{cfg.benchmarking.trials}")
    else:
        print(f"Starting new benchmark: {count}/{cfg.benchmarking.trials} trials")
    
    # Generate all seeds upfront for reproducibility
    rng = Random(cfg.benchmarking.seed)
    seeds = rng.sample(range(0, 100000), cfg.benchmarking.trials)
    
    # Calculate how many trials to run this session
    remaining_trials = cfg.benchmarking.trials - count
    cur_evals = min(cfg.benchmarking.max_evals, remaining_trials)
    
    # Run trials
    for i in range(cur_evals):
        trial_num = count + i
        trial_seed = seeds[trial_num]
        
        print(f"\n{'='*70}")
        print(f"Running trial {trial_num + 1}/{cfg.benchmarking.trials} (seed={trial_seed})")
        print(f"{'='*70}")
        
        try:
            cfg.seed = trial_seed
            # resolve cfgs before pickling
            resolved_cfg = OmegaConf.to_container(cfg, resolve=True)
            resolved_ad_cfg = OmegaConf.to_container(ad_cfg, resolve=True) if ad_cfg is not None else None
            world_size = torch.cuda.device_count()
            free_port = find_free_port()
            result_queue = mp.SimpleQueue()
            mp.spawn(run_experiment_s1_ddp_wrapper, args=(world_size, free_port, resolved_cfg, resolved_ad_cfg, result_queue), nprocs=world_size)

            # Wait for all processes to finish and get the results
            fmetrics = result_queue.get()
            
            # Extract all metrics (automatically handles any structure changes)
            # SAR uses partitions, so we pass the partition parameter
            shift_trial_metrics = extract_trial_metrics(fmetrics, split=cfg.eval.mode, partition="shift_invariant")
            non_shift_trial_metrics = extract_trial_metrics(fmetrics, split=cfg.eval.mode, partition="non_shift_invariant")
            
            # Add metadata
            shift_trial_results = {
                'trial': trial_num,
                'seed': trial_seed,
                **shift_trial_metrics
            }
            non_shift_trial_results = {
                'trial': trial_num,
                'seed': trial_seed,
                **non_shift_trial_metrics
            }
            
            # Append to DataFrame
            shift_trials_df = pd.concat(
                [shift_trials_df, pd.DataFrame([shift_trial_results])], 
                ignore_index=True
            )
            non_shift_trials_df = pd.concat(
                [non_shift_trials_df, pd.DataFrame([non_shift_trial_results])], 
                ignore_index=True
            )
            
            # Extract and save aligned metrics if shift ablation is enabled
            if is_shift_ablation_available:
                aligned_trial_metrics = extract_trial_metrics(fmetrics, split=cfg.eval.mode, partition="aligned")
                aligned_trial_results = {
                    'trial': trial_num,
                    'seed': trial_seed,
                    **aligned_trial_metrics
                }
                aligned_trials_df = pd.concat(
                    [aligned_trials_df, pd.DataFrame([aligned_trial_results])],
                    ignore_index=True
                )
                save_trials_checkpoint(aligned_trials_path, aligned_trials_df)
            
            # Save checkpoint after each trial
            save_trials_checkpoint(shift_trials_path, shift_trials_df)
            save_trials_checkpoint(non_shift_trials_path, non_shift_trials_df)
            print(f"Trial {trial_num + 1} completed. Shift metrics saved to {shift_trials_path}, non-shift metrics saved to {non_shift_trials_path}")
            if is_shift_ablation_available:
                print(f"  Aligned metrics saved to {aligned_trials_path}")
        except Exception as err:
            err.add_note(f'Happened on benchmark trial number {trial_num + 1}.')
            print(f'\nERROR: Trial {trial_num + 1} failed. Checkpoint saved.')
            save_trials_checkpoint(shift_trials_path, shift_trials_df)
            save_trials_checkpoint(non_shift_trials_path, non_shift_trials_df)
            if is_shift_ablation_available:
                save_trials_checkpoint(aligned_trials_path, aligned_trials_df)
            raise err
    
    count += cur_evals
    
    # Compute and save final summary if all trials complete
    if count == cfg.benchmarking.trials:
        print(f"\n{'='*70}")
        print(f"All {cfg.benchmarking.trials} trials completed!")
        print(f"{'='*70}\n")
        
        # Compute summary statistics
        shift_stats = compute_summary_statistics(shift_trials_df, prefix="shift")
        non_shift_stats = compute_summary_statistics(non_shift_trials_df, prefix="non_shift")
        
        # Build summary with metadata columns first
        summary = {
            'config_id': cfg.benchmarking.config_id,
            'description': cfg.benchmarking.get('description', ''),
            'seed': cfg.benchmarking.seed,
            'project': cfg.wandb.project,
            'group': cfg.wandb.group,
            'classifier': cfg.model.classifier,
            'channels': cfg.data.channels,
            'shift_invariant': cfg.train.shift_invariant,
            'shift_ablation': is_shift_ablation_available,
            'trials': cfg.benchmarking.trials,
            **shift_stats,
            **non_shift_stats
        }
        
        # Add aligned metrics to summary if available
        if is_shift_ablation_available and len(aligned_trials_df) > 0:
            aligned_stats = compute_summary_statistics(aligned_trials_df, prefix="aligned")
            summary.update(aligned_stats)
        
        # Save summary
        save_summary(summary_path, summary)
        
        print(f"Benchmark complete!")
        print(f"  - Shift trials: {shift_trials_path}")
        print(f"  - Non-shift trials: {non_shift_trials_path}")
        if is_shift_ablation_available:
            print(f"  - Aligned trials: {aligned_trials_path}")
        print(f"  - Config: {config_path}")
        print(f"  - Summary: {summary_path}")
    else:
        print(f"\nProgress: {count}/{cfg.benchmarking.trials} trials completed")
        print(f"Checkpoint saved to: {shift_trials_path}")
        print(f"Checkpoint saved to: {non_shift_trials_path}")
        if is_shift_ablation_available:
            print(f"Checkpoint saved to: {aligned_trials_path}")
        print(f"Run again to continue from trial {count + 1}")


@hydra.main(version_base=None, config_path="pkg://configs", config_name="config.yaml")
def main(cfg: DictConfig) -> None:
    benchmark_sar(cfg)


if __name__ == "__main__":
    main()
