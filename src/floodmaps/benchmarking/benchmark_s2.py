from random import Random
import pandas as pd
import numpy as np
import argparse
import os
import sys
import pickle

from floodmaps.utils.config import Config
from floodmaps.training.train_s2 import run_experiment_s2

def load_metrics(save_chkpt_path):
    """Initialize or load in metrics list from prior benchmarking runs.
    
    Parameters
    ----------
    save_chkpt_path : str
        Path to save chkpt file.

    Returns
    -------
    vacc_list : list
        List of test accuracies.
    vpre_list : list
        List of test precisions.
    vrec_list : list
        List of test recalls.
    vf1_list : list
        List of test F1 scores.
    """
    if save_chkpt_path is not None and os.path.exists(save_chkpt_path):
        with open(save_chkpt_path, 'rb') as f:
            stats = pickle.load(f)

        vacc_list, vpre_list, vrec_list, vf1_list = stats
    else:
        vacc_list = []
        vpre_list = []
        vrec_list = []
        vf1_list = []
    
    return vacc_list, vpre_list, vrec_list, vf1_list

def save_metrics(save_chkpt_path, stats):
    """Save metrics list to pickle file.
    
    Parameters
    ----------
    save_chkpt_path : str
        Path to save chkpt file.
    stats : list
        List of test accuracies, precisions, recalls, and F1 scores in order [vacc_list, vpre_list, vrec_list, vf1_list].
    """
    with open(save_chkpt_path, 'wb') as f:
        pickle.dump(stats, f)

def main(config_file, trials, max_evals=10, save_file='./benchmarks/runs_s2.csv', save_chkpt_path='./benchmarks/chkpt_s2.pkl', seed=263932):
    """Given model parameters and number of trials n, runs identical experiment n times each with
    unique seeding. Then calculates the mean and std of metrics across all n experiments.
    The collected data can then be used to compare model performance.

    Note: run with conda environment 'floodmaps-tuning'.
    """
    cfg = Config(config_file)
    assert cfg.eval.mode == 'test', 'Benchmarking must be run on test set.'

    # load in prior runs if resuming benchmarking
    count = 0
    vacc_list, vpre_list, vrec_list, vf1_list = load_metrics(save_chkpt_path)
    count += len(vacc_list)

    # each trial uses different random seed
    rng = Random(seed)
    seeds = rng.sample(range(0, 100000), trials)

    # save results in pandas dataframe with config parameters, number of runs, mean of run metrics, std of run metrics
    # ensure only run up to 'trials' experiments
    cur_evals = max_evals if count + max_evals <= trials else trials - count
    for i, trial_seed in enumerate(seeds[count:count + cur_evals]):
        try:
            cfg.seed = trial_seed
            fmetrics = run_experiment_s2(cfg)
        except Exception as err:
            # quickly save results if there is a crash
            err.add_note(f'Happened on benchmark trial number {count+i+1}.')
            print(f'Run crashed on trial number {count+i+1}. Saving current results to chkpt.')
            save_metrics(save_chkpt_path, [vacc_list, vpre_list, vrec_list, vf1_list])
            raise err

        test_metrics = fmetrics.get_metrics(split='test')
        vacc_list.append(test_metrics['test accuracy'])
        vpre_list.append(test_metrics['test precision'])
        vrec_list.append(test_metrics['test recall'])
        vf1_list.append(test_metrics['test f1'])

    count += cur_evals
    # calculate mean and std when finished
    if count == trials:
        results = {
            'config_file': config_file,
            'group': cfg.wandb.group,
            'architecture': cfg.model.classifier,
            'channels': cfg.data.channels,
            'trials': trials,
            'mean_vacc': np.mean(vacc_list),
            'std_vacc': np.std(vacc_list),
            'mean_vpre': np.mean(vpre_list),
            'std_vpre': np.std(vpre_list),
            'mean_vrec': np.mean(vrec_list),
            'std_vrec': np.std(vrec_list),
            'mean_vf1': np.mean(vf1_list),
            'std_vf1': np.std(vf1_list)
        }
        results_df = pd.DataFrame([results])

        # save to pandas dataframe
        if os.path.exists(save_file):
            prev_df = pd.read_csv(save_file)
            results_df = pd.concat([prev_df, results_df], ignore_index=True)
        
        results_df.to_csv(save_file, index=False)

        # remove chkpt file once benchmarking finished
        if os.path.exists(save_chkpt_path):
            os.remove(save_chkpt_path)
    else:
        # save to chkpt file
        save_metrics(save_chkpt_path, [vacc_list, vpre_list, vrec_list, vf1_list])

if __name__ == "__main__":
    parser = argparse.ArgumentParser(prog='benchmark_models_s2', description='Benchmarks S2 classifier model on test set given parameters.')
    
    # YAML config file
    parser.add_argument("--config_file", default="configs/s2_template.yaml", help="Path to YAML config file (default: configs/s2_template.yaml)")
    parser.add_argument("--trials", default=10, type=int, help="Number of total trials per benchmark (default: 10)")
    parser.add_argument("--save_file", default="./results/benchmarks/rgb_runs_s2.csv", help="Path to save file (default: ./results/benchmarks/rgb_runs_s2.csv)")
    parser.add_argument("--save_chkpt_path", default="./results/benchmarks/rgb_chkpt_s2.pkl", help="Path to save chkpt file (default: ./results/benchmarks/rgb_chkpt_s2.pkl)")
    parser.add_argument("--seed", default=263932, type=int, help="Random seed (default: 263932)")
    parser.add_argument("--max_evals", default=5, type=int, help="Maximum number of evaluations per job, set less than trials if want to split up benchmarking into multiple jobs (default: 5)")
    _args = parser.parse_args()

    sys.exit(main(_args.config_file, _args.trials, save_file=_args.save_file, save_chkpt_path=_args.save_chkpt_path, seed=_args.seed, max_evals=_args.max_evals))
