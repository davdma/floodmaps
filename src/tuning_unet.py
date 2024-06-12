from deephyper.problem import HpProblem
from deephyper.search.hps import CBO
from deephyper.evaluator import Evaluator
from deephyper.evaluator.callback import SearchEarlyStopping
from train_classifier import run_experiment_s2
from train_sar import run_experiment_s1
import pandas as pd
import argparse
import os
import sys

def run_s2(job):
    # channels = [bool(int(x)) for x in str(int(job.parameters['channels']))]
    config = {
        'method': "random",
        'size': 64, 
        'samples': 1000, 
        'channels': [bool(int(x)) for x in '1111101111'], # exclude DEM tuning for now
        'sample_dir': '../sampling/samples_200_5_4_35/',
        'label_dir': '../sampling/labels/',
        'project': 'FloodSamplesUNetNoDEM', # 'FloodSamplesUNetTuning2', 
        'group': None,
        'num_sample_predictions': 60,
        'epochs': 100, 
        'batch_size': job.parameters['batch_size'], 
        'num_workers': 0,
        'learning_rate': job.parameters['learning_rate'], 
        'early_stopping': True, 
        'patience': 10,
        'name': 'unet',
        'dropout': job.parameters['dropout'],
        'deep_supervision': False,
        'loss': "TverskyLoss",
        'alpha': job.parameters['alpha'],
        'beta': 1 - job.parameters['alpha'],
        'optimizer': "Adam"
    }
    final_vacc, final_vpre, final_vrec, final_vf1 = run_experiment_s2(config)
    return final_vf1

def run_s1(job):
    config = {
        'size': 68, 
        'window': 64,
        'samples': 1000, 
        'method': 'minibatch',
        'filter': 'raw',
        'channels': [bool(int(x)) for x in '1111111'],
        'project': 'SARUNetDespecklerTuning',
        'group': 'CNN_autodespeckler',
        'num_sample_predictions': 60,
        'mode': 'val',
        'epochs': 250, 
        'batch_size': 2048, 
        'subset': 1.0,
        'learning_rate': job.parameters['learning_rate'], 
        'early_stopping': True, 
        'patience': 10,
        'name': 'unet',
        'dropout': job.parameters['dropout'],
        'deep_supervision': False,
        'autodespeckler': 'CNN', # if not None need to specify additional autodespeckler args
        'latent_dim': job.parameters['latent_dim'], # optional
        'AD_dropout': job.parameters['AD_dropout'], # optional
        'num_workers': 10,
        'loss': job.parameters['loss'],
        'alpha': job.parameters['alpha'],
        'beta': 1 - job.parameters['alpha'],
        'optimizer': "Adam",
        'seed': 13200
    }
    final_vacc, final_vpre, final_vrec, final_vf1 = run_experiment_s1(config)
    return final_vf1

def tuning_s2(file_index, max_evals, experiment_name, early_stopping):
    search_dir = './tuning/s2/' + experiment_name

    if not os.path.exists(search_dir):
        os.makedirs(search_dir)

    # define the variable you want to optimize
    # assume channels is 111111111
    problem = HpProblem()
    # problem.add_hyperparameter([111111111, 111111100, 111110000], "channels") - all_old.csv
    problem.add_hyperparameter((0.20, 0.40), "alpha")
    problem.add_hyperparameter([16, 32, 64, 128, 256, 512, 1024], "batch_size")
    problem.add_hyperparameter((0.00001, 0.01), "learning_rate") # real parameter
    problem.add_hyperparameter((0.10, 0.30), "dropout")
    # problem.add_hyperparameter(["BCELoss", "BCEDiceLoss", "TverskyLoss"], "loss")

    # define the evaluator to distribute the computation
    method_kwargs = {"callbacks": [SearchEarlyStopping(patience=10)]} if early_stopping else dict()

    with Evaluator.create(run_s2, method="serial", method_kwargs=method_kwargs) as evaluator:
        search = CBO(problem, evaluator, surrogate_model="RF", log_dir=search_dir, random_state=102995)
        
        if int(file_index) >= 1:
            # fit model from previous checkpointed search
            search.fit_surrogate(search_dir + '/all.csv')

            # execute search
            results = search.search(max_evals=max_evals, timeout=23*3600)
        else:
            results = search.search(max_evals=max_evals, timeout=23*3600)

        # save results to collective file
        save_file = search_dir + '/all.csv'
        if os.path.exists(save_file):
            results.to_csv(save_file, mode='a', index=False, header=False)
        else:
            results.to_csv(save_file, index=False)

def tuning_s1(file_index, max_evals, experiment_name, early_stopping):
    search_dir = './tuning/s1/' + experiment_name

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
    problem.add_hyperparameter([50, 100, 150, 200], "latent_dim")
    problem.add_hyperparameter((0.05, 0.30), "AD_dropout")

    # define the evaluator to distribute the computation
    method_kwargs = {"callbacks": [SearchEarlyStopping(patience=10)]} if early_stopping else dict()

    with Evaluator.create(run_s1, method="serial", method_kwargs=method_kwargs) as evaluator:
        search = CBO(problem, evaluator, surrogate_model="RF", log_dir=search_dir, random_state=17203)
        
        if int(file_index) >= 1:
            # fit model from previous checkpointed search
            search.fit_surrogate(search_dir + '/all.csv')

            # execute search
            results = search.search(max_evals=max_evals, timeout=23*3600)
        else:
            results = search.search(max_evals=max_evals, timeout=23*3600)

        # save results to collective file
        save_file = search_dir + '/all.csv'
        if os.path.exists(save_file):
            results.to_csv(save_file, mode='a', index=False, header=False)
        else:
            results.to_csv(save_file, index=False)

if __name__ == "__main__":
    parser = argparse.ArgumentParser()
    parser.add_argument('-d', '--dataset', default='s2', choices=['s1', 's2'], help=f"dataset options: s1, s2 (default: s2)")
    parser.add_argument("-i", "--run_index", type=int, default=0)
    parser.add_argument("-e", "--max_evals", type=int, default=1)
    parser.add_argument("-n", "--experiment_name", type=str, default="unetclassifier")
    parser.add_argument('--early_stopping', action='store_true', help='early stopping (default: False)')

    args = parser.parse_args()

    if args.dataset == 's2':
        sys.exit(tuning_s2(args.run_index, args.max_evals, args.experiment_name, args.early_stopping))
    elif args.dataset == 's1':
        sys.exit(tuning_s1(args.run_index, args.max_evals, args.experiment_name, args.early_stopping))
