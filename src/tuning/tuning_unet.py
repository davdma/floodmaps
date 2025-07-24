from deephyper.hpo import HpProblem
from deephyper.hpo import CBO
from deephyper.evaluator import Evaluator
from deephyper.evaluator.callback import SearchEarlyStopping
from training.train_s2 import run_experiment_s2
from training.train_sar import run_experiment_s1
from utils.config import Config
from datetime import datetime
from tuning.tuning_utils import load_stopper_info, save_stopper_info, print_save_best_params, save_problem
import pandas as pd
import argparse
import os
import sys
import socket

def run_s2(parameters):
    override = {
        'project': 'Texas_S2_NoDEM_Tuning',
        'group': 'UNet',
        'loss': parameters['loss'],
        'lr': parameters['learning_rate'],
        'LR_scheduler': parameters['LR_scheduler'],
        'dropout': parameters['dropout']
    }
    cfg = Config(config_file="configs/s2_unet_tuning.yaml", **override)
    fmetrics = run_experiment_s2(cfg)
    results = fmetrics.get_metrics(split='val')
    return results['val f1']

def tuning_s2(file_index, max_evals, experiment_name, early_stopping, patience=10, random_state=230):
    timestamp = datetime.now().strftime('%Y-%m-%d %H:%M:%S')
    hostname = socket.gethostname()
    print("----------------------------------")
    print("Running tuning script on:", hostname)
    print("Current timestamp:", timestamp)
    print(f'Beginning tuning for s2 experiment {experiment_name} for {max_evals} max evals using random seed {random_state}.')
    search_dir = './results/tuning/s2/' + experiment_name

    if not os.path.exists(search_dir):
        os.makedirs(search_dir)

    # define the variable you want to optimize
    problem = HpProblem()
    problem.add_hyperparameter((0.00001, 0.01), "learning_rate") # real parameter
    problem.add_hyperparameter((0.05, 0.40), "dropout")
    problem.add_hyperparameter(["BCELoss", "BCEDiceLoss", "TverskyLoss"], "loss")
    problem.add_hyperparameter(['Constant', 'ReduceLROnPlateau'], 'LR_scheduler')
    # problem.add_hyperparameter((0.20, 0.40), "alpha") # For tversky keep constant alpha (tune later)
    # problem.add_hyperparameter([True, False], "deep_supervision")

    # save problem to json
    save_problem(problem, search_dir + '/problem.json')

    # load in early stopping metrics if available
    if early_stopping:
        early_stopper = SearchEarlyStopping(patience=patience)
        if os.path.exists(search_dir + '/stopper.json'):
            _best_objective, _n_lower = load_stopper_info(search_dir + '/stopper.json')
            early_stopper._best_objective = _best_objective
            early_stopper._n_lower = _n_lower
        method_kwargs = {"callbacks": [early_stopper]}
    else:
        method_kwargs = dict()

    # define the evaluator to distribute the computation
    with Evaluator.create(run_s2, method="process", method_kwargs=method_kwargs) as evaluator:
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

def run_s1(parameters):
    override = {
        'project': 'SARUNetDespecklerTuning',
        'group': 'CNN2_autodespeckler',
        'lr': parameters['learning_rate'],
        'dropout': parameters['dropout'],
        'alpha': parameters['alpha'],
        'beta': 1 - parameters['alpha'],
        'loss': parameters['loss']
    }
    cfg = Config(config_file="configs/test.yaml", **override)
    ad_cfg = Config(config_file="configs/test.yaml", **override)
    fmetrics = run_experiment_s1(cfg, ad_cfg=ad_cfg)
    results = fmetrics.get_metrics(split='val', partition='shift_invariant')
    return results['val f1']

def tuning_s1(file_index, max_evals, experiment_name, early_stopping, patience=10, random_state=230):
    timestamp = datetime.now().strftime('%Y-%m-%d %H:%M:%S')
    hostname = socket.gethostname()
    print("----------------------------------")
    print("Running tuning script on:", hostname)
    print("Current timestamp:", timestamp)
    print(f'Beginning tuning for s1 experiment {experiment_name} for {max_evals} max evals using random seed {random_state}.')
    search_dir = './results/tuning/s1/' + experiment_name

    if not os.path.exists(search_dir):
        os.makedirs(search_dir)

    # define the variable you want to optimize
    # We want to create model w dem and model wo dem
    problem = HpProblem()
    problem.add_hyperparameter((0.3, 0.45), "alpha")
    problem.add_hyperparameter((0.00001, 0.01), "learning_rate") # real parameter
    problem.add_hyperparameter((0.10, 0.30), "dropout")
    problem.add_hyperparameter(["BCELoss", "BCEDiceLoss", "TverskyLoss"], "loss")

    # optional autodespeckler CNN first
    problem.add_hyperparameter([1, 2, 3, 4, 5], "AD_num_layers")
    problem.add_hyperparameter([3, 5, 7], "AD_kernel_size")
    problem.add_hyperparameter((0.05, 0.30), "AD_dropout")
    problem.add_hyperparameter(["leaky_relu", "relu"], "AD_activation_func")

    # save problem to json
    save_problem(problem, search_dir + '/problem.json')

    # load in early stopping metrics if available
    if early_stopping:
        early_stopper = SearchEarlyStopping(patience=patience)
        if os.path.exists(search_dir + '/stopper.json'):
            _best_objective, _n_lower = load_stopper_info(search_dir + '/stopper.json')
            early_stopper._best_objective = _best_objective
            early_stopper._n_lower = _n_lower
        method_kwargs = {"callbacks": [early_stopper]}
    else:
        method_kwargs = dict()

    # define the evaluator to distribute the computation
    with Evaluator.create(run_s1, method="process", method_kwargs=method_kwargs) as evaluator:
        print(f"Created new evaluator with {evaluator.num_workers} \
            worker{'s' if evaluator.num_workers > 1 else ''} and config: {method_kwargs}")
        search = CBO(problem, evaluator, surrogate_model="RF", log_dir=search_dir, random_state=599023)

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
    parser.add_argument('-d', '--dataset', default='s2', choices=['s1', 's2'], help=f"dataset options: s1, s2 (default: s2)")
    parser.add_argument("-i", "--run_index", type=int, default=0, help='0 for random trials, 1 for bayesian opt (requires 10+ previous runs)')
    parser.add_argument("-e", "--max_evals", type=int, default=1)
    parser.add_argument("-n", "--experiment_name", type=str, default="unet_s2")
    parser.add_argument('--early_stopping', action='store_true', help='early stopping (default: False)')
    parser.add_argument("-r", "--random_state", type=int, default=123213)
    parser.add_argument("-p", "--patience", type=int, default=10)

    args = parser.parse_args()

    if args.dataset == 's2':
        sys.exit(tuning_s2(args.run_index, args.max_evals, args.experiment_name, args.early_stopping, patience=args.patience, random_state=args.random_state))
    elif args.dataset == 's1':
        sys.exit(tuning_s1(args.run_index, args.max_evals, args.experiment_name, args.early_stopping, patience=args.patience, random_state=args.random_state))