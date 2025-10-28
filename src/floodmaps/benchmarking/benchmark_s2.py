from random import Random
import pandas as pd
import numpy as np
import hydra
from omegaconf import DictConfig, OmegaConf
from pathlib import Path
from typing import Dict, Any, Optional

from floodmaps.training.train_s2 import run_experiment_s2


def flatten_group_metrics(metrics_dict: Dict[str, Any], prefix: str) -> Dict[str, Any]:
    """Flatten nested group metrics into a single-level dictionary.
    
    Takes group_metrics from nlcd_metrics or scl_metrics and creates flattened
    keys like 'nlcd_urban_acc', 'scl_water_prec', etc.
    
    Parameters
    ----------
    metrics_dict : dict
        Dictionary containing 'group_metrics' with nested structure
    prefix : str
        Prefix for flattened keys (e.g., 'nlcd' or 'scl')
    
    Returns
    -------
    dict
        Flattened dictionary with keys like '{prefix}_{group}_{metric}'
    
    Examples
    --------
    >>> nlcd_metrics = {'group_metrics': {'urban': {'acc': 0.95, 'f1': 0.90}}}
    >>> flatten_group_metrics(nlcd_metrics, 'nlcd')
    {'nlcd_urban_acc': 0.95, 'nlcd_urban_f1': 0.90}
    """
    flat_dict = {}
    if 'group_metrics' in metrics_dict:
        for group_name, group_metrics in metrics_dict['group_metrics'].items():
            # Sanitize group name for use in column names
            safe_group = group_name.replace(' ', '_').replace('/', '_')
            for metric_name, metric_value in group_metrics.items():
                key = f"{prefix}_{safe_group}_{metric_name}"
                flat_dict[key] = metric_value
    return flat_dict


def extract_trial_metrics(fmetrics, split: str) -> Dict[str, Any]:
    """Extract and flatten all metrics from a trial run.
    
    Extracts core metrics, NLCD group metrics, and SCL group metrics,
    flattening them into a single dictionary.
    
    Parameters
    ----------
    fmetrics : Metrics
        Metrics object from training run
    split : str
        Data split ('test' or 'val')
    
    Returns
    -------
    dict
        Flattened dictionary of all metrics
    """
    all_metrics = fmetrics.get_metrics(split=split)
    
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
                               exclude_cols: list = None) -> Dict[str, Any]:
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
            stats[f'mean_{col}'] = float(valid_values.mean())
            stats[f'std_{col}'] = float(valid_values.std())
        else:
            # All values were None/NaN
            stats[f'mean_{col}'] = None
            stats[f'std_{col}'] = None
    
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


def benchmark_s2(cfg: DictConfig) -> None:
    """Benchmarks S2 classifier model on test set with multiple random seeds.
    
    Runs n trials with different random seeds, collects all metrics (core metrics,
    NLCD group metrics, SCL group metrics), and computes summary statistics.
    Supports checkpointing for resuming interrupted runs.
    
    The benchmark saves three files:
    1. {config_id}_trials.csv - Individual trial results with all metrics
    2. {config_id}_config.yaml - Configuration used for this benchmark
    3. {config_id}_summary.csv - Mean and std statistics across all trials
    
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
    """
    assert cfg.eval.mode == 'test', 'Benchmarking must be run on test set.'
    
    # Setup paths
    save_dir = Path(cfg.benchmarking.save_dir)
    save_dir.mkdir(parents=True, exist_ok=True)
    config_id = cfg.benchmarking.config_id
    trials_path = save_dir / f"{config_id}_trials.csv"
    config_path = save_dir / f"{config_id}_config.yaml"
    summary_path = save_dir / f"{config_id}_summary.csv"
    
    # Save config (only once per benchmark)
    save_config(config_path, cfg)
    
    # Load or initialize trials DataFrame
    trials_df = load_trials_checkpoint(trials_path)
    count = len(trials_df)
    
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
            fmetrics = run_experiment_s2(cfg)
            
            # Extract all metrics (automatically handles any structure changes)
            trial_metrics = extract_trial_metrics(fmetrics, split=cfg.eval.mode)
            
            # Add metadata
            trial_results = {
                'trial': trial_num,
                'seed': trial_seed,
                **trial_metrics
            }
            
            # Append to DataFrame
            trials_df = pd.concat(
                [trials_df, pd.DataFrame([trial_results])], 
                ignore_index=True
            )
            
            # Save checkpoint after each trial
            save_trials_checkpoint(trials_path, trials_df)
            print(f"Trial {trial_num + 1} completed and saved to {trials_path}")
            
        except Exception as err:
            err.add_note(f'Happened on benchmark trial number {trial_num + 1}.')
            print(f'\nERROR: Trial {trial_num + 1} failed. Checkpoint saved.')
            save_trials_checkpoint(trials_path, trials_df)
            raise err
    
    count += cur_evals
    
    # Compute and save final summary if all trials complete
    if count == cfg.benchmarking.trials:
        print(f"\n{'='*70}")
        print(f"All {cfg.benchmarking.trials} trials completed!")
        print(f"{'='*70}\n")
        
        # Compute summary statistics
        stats = compute_summary_statistics(trials_df)
        
        # Build summary with metadata columns first
        summary = {
            'config_id': cfg.benchmarking.config_id,
            'description': cfg.benchmarking.get('description', ''),
            'seed': cfg.benchmarking.seed,
            'project': cfg.wandb.project,
            'group': cfg.wandb.group,
            'classifier': cfg.model.classifier,
            'channels': cfg.data.channels,
            'trials': cfg.benchmarking.trials,
            **stats
        }
        
        # Save summary
        save_summary(summary_path, summary)
        
        print(f"Benchmark complete!")
        print(f"  - Trials: {trials_path}")
        print(f"  - Config: {config_path}")
        print(f"  - Summary: {summary_path}")
    else:
        print(f"\nProgress: {count}/{cfg.benchmarking.trials} trials completed")
        print(f"Checkpoint saved to: {trials_path}")
        print(f"Run again to continue from trial {count + 1}")


@hydra.main(version_base=None, config_path="pkg://configs", config_name="config.yaml")
def main(cfg: DictConfig) -> None:
    benchmark_s2(cfg)


if __name__ == "__main__":
    main()
