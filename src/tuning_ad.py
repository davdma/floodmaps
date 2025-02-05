from deephyper.problem import HpProblem
from deephyper.search.hps import CBO
from deephyper.evaluator import Evaluator
from deephyper.evaluator.callback import SearchEarlyStopping
from train_sar import run_experiment_s1
from train_ad_head import run_experiment_ad
from datetime import datetime
import pandas as pd
import argparse
import os
import sys
import socket

# tuning autodespeckler + unet architecture
# need to fill in need fields for loading, freezing, watching weights
def run_s1(job):
    config = {
        'size': 68, 
        'window': 64,
        'samples': 1000, 
        'method': 'minibatch',
        'filter': 'raw',
        'channels': [bool(int(x)) for x in '1111111'],
        'project': 'SAR_AD_Tuning',
        'group': 'VAE_autodespeckler',
        'num_sample_predictions': 60,
        'mode': 'val',
        'epochs': 250, 
        'batch_size': 256, 
        'subset': 0.5,
        'learning_rate': job.parameters['learning_rate'], 
        'early_stopping': True, 
        'patience': 10,
        'name': 'unet++',
        'load_classifier': None,
        'deep_supervision': job.parameters['deep_supervision'],
        'dropout': job.parameters['dropout'],
        'autodespeckler': 'VAE', # if not None need to specify additional autodespeckler args
        'load_autodespeckler': None,
        'freeze_autodespeckler': True,
        'latent_dim': job.parameters['latent_dim'],
        'noise_type': None,
        'noise_coeff': None,
        'AD_num_layers': None,
        'AD_kernel_size': None, # optional
        'AD_dropout': None, # optional
        'AD_activation_func': None,
        'VAE_beta': job.parameters['VAE_beta'],
        'num_workers': 10,
        'loss': job.parameters['loss'],
        'alpha': job.parameters['alpha'],
        'beta': 1 - job.parameters['alpha'],
        'optimizer': "Adam",
        'seed': 19935,
        'random_flip': False,
        'shift_invariant': True,
        'watch_weights_grad': True
    }
    final_vmetrics = run_experiment_s1(config)
    final_vacc, final_vpre, final_vrec, final_vf1 = final_vmetrics.get_val_metrics()
    return final_vf1

def tuning_s1(file_index, max_evals, experiment_name, early_stopping, random_state=930):
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

    # define the evaluator to distribute the computation
    method_kwargs = {"callbacks": [SearchEarlyStopping(patience=10)]} if early_stopping else dict()

    with Evaluator.create(run_s1, method="serial", method_kwargs=method_kwargs) as evaluator:
        search = CBO(problem, evaluator, surrogate_model="RF", log_dir=search_dir, random_state=random_state)
        
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
            # note: need to reorder column headers if a bunch of new hyperparameters are added to match csv file
            existing_df = pd.read_csv(save_file, nrows=0)
            existing_columns = existing_df.columns.tolist()
            results = results[existing_columns]
            
            results.to_csv(save_file, mode='a', index=False, header=False)
        else:
            results.to_csv(save_file, index=False)

# tuning autodespeckler alone
def run_ad(job):
    config = {
        'size': 64, 
        'samples': 500, 
        'kernel_size': 5,
        'project': 'SAR_AD_Tuning_Head_Only',
        'group': 'DAE_mult_mse',
        'num_sample_predictions': 40,
        'mode': 'val',
        'epochs': 840,
        'batch_size': 1024,
        'subset': 1.0,
        'learning_rate': job.parameters['learning_rate'], 
        'clip': 1.0,
        'early_stopping': True,
        'patience': 30,
        'LR_scheduler': job.parameters['LR_scheduler'],
        'LR_patience': 10,
        'LR_T_max': 200,
        'autodespeckler': 'DAE',
        'latent_dim': None,
        'noise_type': 'log_gamma',
        'noise_coeff': job.parameters['noise_coeff'],
        'AD_num_layers': job.parameters['AD_num_layers'],
        'AD_kernel_size': job.parameters['AD_kernel_size'],
        'AD_dropout': job.parameters['AD_dropout'],
        'AD_activation_func': job.parameters['AD_activation_func'],
        'VAE_beta': None,
        'use_lee': job.parameters['use_lee'],
        'random_flip': True,
        'num_workers': 10,
        'loss': 'MSELoss', # change loss for dif tuning
        'optimizer': "Adam",
        'seed': 9290,
        'save': False,
        'grad_norm_freq': 10
    }
    final_vmetrics = run_experiment_ad(config)
    final_vloss = final_vmetrics.get_val_metrics()
    return 0 - final_vloss

def tuning_ad(file_index, max_evals, experiment_name, early_stopping, random_state=17630):
    timestamp = datetime.now().strftime('%Y-%m-%d %H:%M:%S')
    hostname = socket.gethostname()
    print("----------------------------------")
    print("Running tuning script on:", hostname)
    print("Current timestamp:", timestamp)
    print(f'Beginning tuning for experiment {experiment_name} for {max_evals} max evals using random seed {random_state}.')
    search_dir = './tuning/ad/' + experiment_name

    if not os.path.exists(search_dir):
        os.makedirs(search_dir)

    # define the variable you want to optimize
    # We want to create model w dem and model wo dem
    problem = HpProblem()
    problem.add_hyperparameter([True, False], "use_lee")
    problem.add_hyperparameter((0.00001, 0.01), "learning_rate") # real parameter
    problem.add_hyperparameter(["ReduceLROnPlateau", "CosAnnealingLR"], "LR_scheduler")
    # DAE
    problem.add_hyperparameter([2, 3, 5, 7], "AD_num_layers")
    problem.add_hyperparameter([3, 5, 7], "AD_kernel_size")
    problem.add_hyperparameter((0.05, 1.0), "noise_coeff") # change interval for dif noise funcs
    problem.add_hyperparameter(['leaky_relu', 'relu', 'softplus', 'mish', 'gelu', 'elu'], "AD_activation_func")
    # problem.add_hyperparameter(["MSELoss", "PseudoHuberLoss", "HuberLoss", "LogCoshLoss"], "loss")
    problem.add_hyperparameter((0.0001, 0.30), "AD_dropout")

    # define the evaluator to distribute the computation
    method_kwargs = {"callbacks": [SearchEarlyStopping(patience=10)]} if early_stopping else dict()

    with Evaluator.create(run_ad, method="serial", method_kwargs=method_kwargs) as evaluator:
        search = CBO(problem, evaluator, surrogate_model="RF", log_dir=search_dir, random_state=random_state)
        
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
            # note: need to reorder column headers if a bunch of new hyperparameters are added to match csv file
            existing_df = pd.read_csv(save_file, nrows=0)
            existing_columns = existing_df.columns.tolist()
            results = results[existing_columns]
            
            results.to_csv(save_file, mode='a', index=False, header=False)
        else:
            results.to_csv(save_file, index=False)

if __name__ == "__main__":
    parser = argparse.ArgumentParser()
    parser.add_argument("-i", "--run_index", type=int, default=0)
    parser.add_argument("-e", "--max_evals", type=int, default=1)
    parser.add_argument("-r", "--random_state", type=int, default=123213)
    parser.add_argument("-n", "--experiment_name", type=str, default="ad_vae_unet")
    parser.add_argument('--early_stopping', action='store_true', help='early stopping (default: False)')
    parser.add_argument('--full_model', action='store_true', help='whether to tune full model or just ad head (default: False)')

    args = parser.parse_args()
    if args.full_model:
        sys.exit(tuning_s1(args.run_index, args.max_evals, args.experiment_name, args.early_stopping,
                           random_state=args.random_state))
    else:
        sys.exit(tuning_ad(args.run_index, args.max_evals, args.experiment_name, args.early_stopping,
                           random_state=args.random_state))
