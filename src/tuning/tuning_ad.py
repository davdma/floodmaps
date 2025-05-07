from deephyper.hpo import HpProblem
from deephyper.hpo import CBO
from deephyper.evaluator import Evaluator
from deephyper.evaluator.callback import SearchEarlyStopping
from training.train_sar import run_experiment_s1
from training.train_ad_head import run_experiment_ad
from utils.config import Config
from datetime import datetime
import pandas as pd
import argparse
import os
import sys
import socket
import json

def load_stopper_info(filepath):
    """Loads historical early stopper objective and count."""
    if os.path.exists(filepath):
        with open(filepath, 'r') as f:
            state = json.load(f)
        return state['_best_objective'], state['_n_lower']
    print('Stopper filepath not found.')
    return None, 0

def save_stopper_info(stopper, filepath):
    """Save optimization early stopper objective and count."""
    state = {'_best_objective': stopper._best_objective, '_n_lower': stopper._n_lower}
    with open(filepath, 'w') as f:
        json.dump(state, f)

def print_save_best_params(save_file, file_path):
    """Prints the parameters of the best tuning run so far."""
    df = pd.read_csv(save_file)
    best_idx = df["objective"].idxmax()
    best_row = df.loc[best_idx]

    # Filter for columns starting with "p:"
    p_vars = best_row.filter(like="p:").to_dict()
    print(p_vars)

    # save best params to file
    with open(file_path, 'w') as f:
        json.dump(p_vars, f)

# tuning autodespeckler + unet architecture
# need to fill in need fields for loading, freezing, watching weights
def run_s1(parameters):
    override = {
        'project': 'S',
        'group': '_',
        'loss': 's',
        'lr': parameters['learning_rate'],
        'latent_dim': parameters['latent_dim'],
        'beta_period': parameters['beta_period'],
        'beta_cycles': parameters['beta_cycles']
    }
    cfg = Config(config_file="configs/VAE_tuning.yaml", **override)
    ad_cfg = Config(config_file="configs/VAE_tuning.yaml", **override)
    fmetrics = run_experiment_s1(cfg, ad_cfg=ad_cfg)
    results = fmetrics.get_metrics(split='val', partition='shift_invariant')
    return results['final model val f1']

def tuning_s1(file_index, max_evals, experiment_name, early_stopping, random_state=930):
    timestamp = datetime.now().strftime('%Y-%m-%d %H:%M:%S')
    hostname = socket.gethostname()
    print("----------------------------------")
    print("Running tuning script on:", hostname)
    print("Current timestamp:", timestamp)
    print(f'Beginning tuning for s1 experiment {experiment_name} for {max_evals} max evals using random seed {random_state}.')
    search_dir = './tuning/s1/' + experiment_name

    if not os.path.exists(search_dir):
        os.makedirs(search_dir)

    # define the variable you want to optimize
    # We want to create model w dem and model wo dem
    problem = HpProblem()
    problem.add_hyperparameter((0.3, 0.45), "alpha")
    problem.add_hyperparameter((0.00001, 0.01), "learning_rate") # real parameter
    problem.add_hyperparameter((0.01, 0.30), "dropout")
    problem.add_hyperparameter(["BCELoss", "BCEDiceLoss", "TverskyLoss"], "loss")
    problem.add_hyperparameter([True, False], "deep_supervision")
    # optional autodespeckler CNN first
    # problem.add_hyperparameter([1, 2, 3, 4, 5], "AD_num_layers")
    # problem.add_hyperparameter([3, 5, 7], "AD_kernel_size")
    # problem.add_hyperparameter((0.05, 0.30), "AD_dropout")
    # problem.add_hyperparameter(["leaky_relu", "relu"], "AD_activation_func")

    # VAE
    problem.add_hyperparameter([128, 256, 512], "latent_dim")
    problem.add_hyperparameter([0, 0.001, 0.01, 0.05, 0.1, 1, 4, 10, 20], "VAE_beta")

    # load in early stopping metrics if available
    if early_stopping:
        early_stopper = SearchEarlyStopping(patience=10)
        if os.path.exists(search_dir + '/stopper.json'):
            _best_objective, _n_lower = load_stopper_info(search_dir + '/stopper.json')
            early_stopper._best_objective = _best_objective
            early_stopper._n_lower = _n_lower
        method_kwargs = {"callbacks": [early_stopper]}
    else:
        method_kwargs = dict()

    with Evaluator.create(run_s1, method="serial", method_kwargs=method_kwargs) as evaluator:
        search = CBO(problem, evaluator, surrogate_model="RF", log_dir=search_dir, random_state=random_state)

        if int(file_index) >= 1:
            # fit model from previous checkpointed search
            search.fit_surrogate(search_dir + '/all.csv')

            # execute search
            results = search.search(max_evals=max_evals, timeout=23*3600)
        else:
            results = search.search(max_evals=max_evals, timeout=23*3600)

        print('Saving stopper and eval results to file...')
        save_stopper_info(early_stopper, search_dir + '/stopper.json')
        # save results to collective file
        save_file = search_dir + '/all.csv'
        if os.path.exists(save_file):
            # note: need to reorder column headers if a bunch of new hyperparameters are added to match csv file
            existing_df = pd.read_csv(save_file, nrows=0)
            existing_columns = existing_df.columns.tolist()
            results = results[existing_columns]

            results.to_csv(save_file, mode='a', index=False, header=False)
        else:
            results.to_csv(save_file, index=False)

        # if search stopped print params of best run
        if early_stopping and early_stopper.search_stopped:
            print_best_params(save_file)

# tuning autodespeckler alone
def run_vae(parameters):
    override = {
        'project': 'SAR_AD_Tuning_Head_3',
        'group': 'VAE_L2',
        'loss': 'MSELoss',
        'lr': parameters['learning_rate'],
        'latent_dim': parameters['latent_dim'],
        'beta_period': parameters['beta_period'],
        'beta_cycles': parameters['beta_cycles']
    }
    cfg = Config(config_file="configs/VAE_tuning.yaml", **override)
    fmetrics = run_experiment_ad(cfg)
    floss = fmetrics.get_metrics(split='val')['loss']
    return 0 - floss

def run_cnn1(parameters):
    override = {
        'project': 'SAR_AD_Tuning_Head_3',
        'group': 'CNN1_L2',
        'loss': 'MSELoss',
        'lr': parameters['learning_rate'],
        'latent_dim': parameters['latent_dim'],
        'AD_dropout': parameters['AD_dropout'],
        'AD_activation_func': parameters['AD_activation_func']
    }
    cfg = Config(config_file="configs/CNN1_tuning.yaml", **override)
    fmetrics = run_experiment_ad(cfg)
    floss = fmetrics.get_metrics(split='val')['loss']
    return 0 - floss

def run_cnn2(parameters):
    override = {
        'project': 'SAR_AD_Tuning_Head_3',
        'group': 'CNN2_L2',
        'loss': 'MSELoss',
        'lr': parameters['learning_rate']
    }
    cfg = Config(config_file="configs/CNN1_tuning.yaml", **override)
    fmetrics = run_experiment_ad(cfg)
    floss = fmetrics.get_metrics(split='val')['loss']
    return 0 - floss

def run_dae(parameters):
    override = {
        'project': 'SAR_AD_Tuning_Head_3',
        'group': 'DAE_L1',
        'loss': 'L1Loss',
        'lr': parameters['learning_rate']
    }
    cfg = Config(config_file="configs/DAE_tuning.yaml", **override)
    fmetrics = run_experiment_ad(cfg)
    floss = fmetrics.get_metrics(split='val')['loss']
    return 0 - floss

def tuning_ad(file_index, max_evals, model, experiment_name, early_stopping, random_state=17630):
    timestamp = datetime.now().strftime('%Y-%m-%d %H:%M:%S')
    hostname = socket.gethostname()
    print("----------------------------------")
    print("Running tuning script on:", hostname)
    print("Current timestamp:", timestamp)
    print(f'Beginning tuning for {model} experiment {experiment_name} for {max_evals} max evals using random seed {random_state}.')
    search_dir = './results/tuning/ad_3/' + experiment_name

    if not os.path.exists(search_dir):
        os.makedirs(search_dir)

    problem = HpProblem()
    # VAE
    if model == 'vae':
        problem.add_hyperparameter((0.00001, 0.01), "learning_rate")
        problem.add_hyperparameter([256, 512, 768, 1024], "latent_dim")
        problem.add_hyperparameter([10, 15, 20, 25, 30, 35, 40], "beta_period")
        problem.add_hyperparameter([1, 2, 3, 4, 5, 6], "beta_cycles")
        obj_func = run_vae

    # CNN1
    if model == 'cnn1':
        problem.add_hyperparameter((0.00001, 0.01), "learning_rate")
        problem.add_hyperparameter([256, 512, 768, 1024], "latent_dim")
        problem.add_hyperparameter((0.05, 0.3), "AD_dropout")
        problem.add_hyperparameter(['relu', 'leaky_relu', 'softplus', 'mish', 'gelu', 'elu'], "AD_activation_func")
        obj_func = run_cnn1

    # DAE
    if model == 'dae':
        problem.add_hyperparameter((0.00001, 0.01), "learning_rate")
        problem.add_hyperparameter([2, 3, 5, 7], "AD_num_layers")
        problem.add_hyperparameter([3, 5, 7], "AD_kernel_size")
        problem.add_hyperparameter((0.05, 1.0), "noise_coeff") # change interval for dif noise funcs
        problem.add_hyperparameter(['leaky_relu', 'relu', 'softplus', 'mish', 'gelu', 'elu'], "AD_activation_func")
        problem.add_hyperparameter((0.0001, 0.30), "AD_dropout")
        obj_func = run_dae

    # CNN2
    if model == 'cnn2':
        problem.add_hyperparameter((0.00001, 0.01), "learning_rate")
        obj_func = run_cnn2

    # load in early stopping metrics if available
    if early_stopping:
        early_stopper = SearchEarlyStopping(patience=10)
        if os.path.exists(search_dir + '/stopper.json'):
            _best_objective, _n_lower = load_stopper_info(search_dir + '/stopper.json')
            early_stopper._best_objective = _best_objective
            early_stopper._n_lower = _n_lower
        method_kwargs = {"callbacks": [early_stopper]}
    else:
        method_kwargs = dict()

    # define the evaluator to distribute the computation
    with Evaluator.create(obj_func, method="process", method_kwargs=method_kwargs) as evaluator:
        print(f"Created new evaluator with {evaluator.num_workers} \
            worker{'s' if evaluator.num_workers > 1 else ''} and config: {method_kwargs}")
        search = CBO(problem, evaluator, surrogate_model="RF", log_dir=search_dir, random_state=random_state)

        if int(file_index) >= 1:
            # fit model from previous checkpointed search
            search.fit_surrogate(search_dir + '/all.csv')

            # execute search
            results = search.search(max_evals=max_evals, timeout=23*3600)
        else:
            results = search.search(max_evals=max_evals, timeout=23*3600)

        print('Saving stopper and eval results to file...')
        save_stopper_info(early_stopper, search_dir + '/stopper.json')
        # save results to collective file
        save_file = search_dir + '/all.csv'
        if os.path.exists(save_file):
            # note: need to reorder column headers if a bunch of new hyperparameters are added to match csv file
            existing_df = pd.read_csv(save_file, nrows=0)
            existing_columns = existing_df.columns.tolist()
            results = results[existing_columns]

            results.to_csv(save_file, mode='a', index=False, header=False)
        else:
            results.to_csv(save_file, index=False)

        # if search stopped print params of best run
        if early_stopping and early_stopper.search_stopped:
            print_save_best_params(save_file, search_dir + '/best.json')

if __name__ == "__main__":
    parser = argparse.ArgumentParser()
    parser.add_argument("-i", "--run_index", type=int, default=0,
                        help='0 if random trials, 1 if bayesian opt (requires 10+ previous runs)')
    parser.add_argument("-e", "--max_evals", type=int, default=1)
    parser.add_argument("-m", "--model", default='vae', choices=['vae', 'cnn1', 'cnn2', 'dae'])
    parser.add_argument("-n", "--experiment_name", type=str, default="ad_vae")
    parser.add_argument('--early_stopping', action='store_true', help='early stopping (default: False)')
    parser.add_argument("-r", "--random_state", type=int, default=123213)
    ### To implement:
    # parser.add_argument('--full_model', action='store_true', help='whether to tune full model or just ad head (default: False)')

    args = parser.parse_args()
    sys.exit(tuning_ad(args.run_index, args.max_evals, args.model, args.experiment_name, args.early_stopping,
                        random_state=args.random_state))
    ### To implement:
    # if args.full_model:
    #     sys.exit(tuning_s1(args.run_index, args.max_evals, args.experiment_name, args.early_stopping,
    #                        random_state=args.random_state))
