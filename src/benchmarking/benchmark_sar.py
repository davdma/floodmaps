from training.train_sar import run_experiment_s1
from random import Random
import pandas as pd
import numpy as np
import argparse
import os
import sys
import pickle
import wandb

MODEL_NAMES = ['unet', 'unet++']
AUTODESPECKLER_NAMES = ['CNN1', 'CNN2', 'DAE', 'VAE']
NOISE_NAMES = ['normal', 'masking', 'log_gamma']
LOSS_NAMES = ['BCELoss', 'BCEDiceLoss', 'TverskyLoss']

def main(config, save_file='./benchmarks/runs_s1_flip.csv', save_chkpt_path='./benchmarks/chkpt_s1_flip.pkl', seed=323311):
    """Given model parameters and number of trials n, runs identical experiment n times each with
    unique seeding. Then calculates the mean and std of metrics across all n experiments.
    The collected data can then be used to compare model performance.

    Note: run with conda environment 'floodmapsdp'.
    """
    # load in prior runs if resuming benchmarking
    count = 0
    if save_chkpt_path is not None and os.path.exists(save_chkpt_path):
        with open(save_chkpt_path, 'rb') as f:
            stats = pickle.load(f)

        vacc_list, vpre_list, vrec_list, vf1_list = stats
        # vacc_list, vpre_list, vrec_list, vf1_list, tmp_vacc_list, tmp_vpre_list, tmp_vrec_list, tmp_vf1_list = stats
        count += len(vacc_list)
    else:
        vacc_list = []
        vpre_list = []
        vrec_list = []
        vf1_list = []

        # tmp for non shift metrics
        # tmp_vacc_list = []
        # tmp_vpre_list = []
        # tmp_vrec_list = []
        # tmp_vf1_list = []

    rng = Random(seed)
    seeds = rng.sample(range(0, 100000), config['trials'])
    # each trial uses different random seed
    # save in pandas dataframe with config parameters, number of runs, mean of run metrics, std of run metrics
    # test set
    config['mode'] = 'test'
    for i, seed in enumerate(seeds[count:count + config['max_evals']]):
        config['seed'] = seed
        try:
            final_vmetrics = run_experiment_s1(config)
            final_vacc, final_vpre, final_vrec, final_vf1 = final_vmetrics.get_shift_metrics()

            # for non shift metrics
            # tmp_final_vacc, tmp_final_vpre, tmp_final_vrec, tmp_final_vf1 = final_vmetrics.get_non_shift_metrics()
        except Exception as err:
            # quickly save results if there is a crash
            err.add_note(f'Happened on benchmark trial number {count+i+1} with seed {seed}.')
            print(f'Run crashed on trial number {count+i+1}. Saving current results to chkpt.')
            stats = [vacc_list, vpre_list, vrec_list, vf1_list]
                     # tmp_vacc_list, tmp_vpre_list, tmp_vrec_list, tmp_vf1_list

            # Save the lists to file using pickle
            with open(save_chkpt_path, 'wb') as f:
                pickle.dump(stats, f)
                
            raise err
            
        vacc_list.append(final_vacc)
        vpre_list.append(final_vpre)
        vrec_list.append(final_vrec)
        vf1_list.append(final_vf1)

        # tmp
        # tmp_vacc_list.append(tmp_final_vacc)
        # tmp_vpre_list.append(tmp_final_vpre)
        # tmp_vrec_list.append(tmp_final_vrec)
        # tmp_vf1_list.append(tmp_final_vf1)

    count += config['max_evals']
    # calculate mean and std when finished
    if count == config['trials']:
        unetpp = config['name'] == 'unet++'
        tversky = config['loss'] == 'TverskyLoss'
        auto = config['autodespeckler'] is not None
        results = {
            'group': config['group'],
            'trials': config['trials'],
            'method': config['method'],
            'filter': config['filter'],
            'channels': ''.join('1' if b else '0' for b in config['channels']),
            'patch_size': config['size'],
            'architecture': config['name'],
            'dropout': config['dropout'],
            'deep_supervision': config['deep_supervision'] if unetpp else None,
            'learning_rate': config['learning_rate'],
            'epochs': config['epochs'],
            'early_stopping': config['early_stopping'],
            'patience': config['patience'],
            'batch_size': config['batch_size'],
            'optimizer': config['optimizer'],
            'loss_fn': config['loss'],
            'shift_invariant': config['shift_invariant'],
            'random_flip': config['random_flip'],
            'alpha': config['alpha'] if tversky else None,
            'beta': config['beta'] if tversky else None,
            'autodespeckler': config['autodespeckler'],
            'noise_type': config['noise_type'] if auto else None,
            'latent_dim': config['latent_dim'] if auto else None,
            'AD_num_layers': config['AD_num_layers'] if auto else None,
            'AD_kernel_size': config['AD_kernel_size'] if auto else None,
            'AD_dropout': config['AD_dropout'] if auto else None,
            'AD_activation_func': config['AD_activation_func'] if auto else None,
            'subset': config['subset'],
            'mean_vacc': np.mean(vacc_list),
            'std_vacc': np.std(vacc_list),
            'mean_vpre': np.mean(vpre_list),
            'std_vpre': np.std(vpre_list),
            'mean_vrec': np.mean(vrec_list),
            'std_vrec': np.std(vrec_list),
            'mean_vf1': np.mean(vf1_list),
            'std_vf1': np.std(vf1_list),
            # 'n_mean_vacc': np.mean(tmp_vacc_list), # STORE TMP NON SHIFT
            # 'n_std_vacc': np.std(tmp_vacc_list),
            # 'n_mean_vpre': np.mean(tmp_vpre_list),
            # 'n_std_vpre': np.std(tmp_vpre_list),
            # 'n_mean_vrec': np.mean(tmp_vrec_list),
            # 'n_std_vrec': np.std(tmp_vrec_list),
            # 'n_mean_vf1': np.mean(tmp_vf1_list),
            # 'n_std_vf1': np.std(tmp_vf1_list),
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
        stats = [vacc_list, vpre_list, vrec_list, vf1_list]
                # tmp_vacc_list, tmp_vpre_list, tmp_vrec_list, tmp_vf1_list

        # Save the lists to file using pickle
        with open(save_chkpt_path, 'wb') as f:
            pickle.dump(stats, f)

if __name__ == "__main__":
    parser = argparse.ArgumentParser(prog='benchmark_models_sar', description='Benchmarks SAR classifier model on test set given parameters.')
    def bool_indices(s):
        if len(s) == 7 and all(c in '01' for c in s):
            try:
                return [bool(int(x)) for x in s]
            except ValueError:
                raise argparse.ArgumentTypeError("Invalid boolean string: '{}'".format(s))
        else:
            raise argparse.ArgumentTypeError("Boolean string must be of length 7 and have binary digits")

    # parameters for benchmarking
    parser.add_argument('-x', '--size', type=int, default=68, help='pixel width of dataset patches (default: 68)')
    parser.add_argument('-w', '--window', type=int, default=64, help='pixel width of model input/output (default: 64)')
    parser.add_argument('-n', '--samples', type=int, default=1000, help='number of patches sampled per image (default: 1000)')
    parser.add_argument('-m', '--method', default='minibatch', choices=['minibatch', 'individual'], help='sampling method (default: minibatch)')
    parser.add_argument('--filter', default='raw', choices=['lee', 'raw'], help=f"filters: enhanced lee, raw (default: raw)")
    parser.add_argument('-c', '--channels', type=bool_indices, default="1111111", help='string of 7 binary digits for selecting among the 10 available channels (VV, VH, DEM, SlopeY, SlopeX, Water, Roads) (default: 1111111)')
    parser.add_argument('-e', '--epochs', type=int, default=30, help='(default: 30)')
    parser.add_argument('-b', '--batch_size', type=int, default=32, help='(default: 32)')
    parser.add_argument('-s', '--subset', dest='subset', type=float, default=1.0, help='percentage of training dataset to use per epoch (default: 1.0)')
    parser.add_argument('-l', '--learning_rate', type=float, default=0.0001, help='(default: 0.0001)')
    parser.add_argument('--early_stopping', action='store_true', help='early stopping (default: False)')
    parser.add_argument('-p', '--patience', type=int, default=10, help='(default: 5)')
    parser.add_argument('--name', default='unet', choices=MODEL_NAMES,
                        help=f"models: {', '.join(MODEL_NAMES)} (default: unet)")
    parser.add_argument('--dropout', type=float, default=0.2, help=f"(default: 0.2)")
    parser.add_argument('--deep_supervision', action='store_true', help='(default: False)')
    parser.add_argument('--autodespeckler', default=None, choices=AUTODESPECKLER_NAMES,
                        help=f"models: {', '.join(AUTODESPECKLER_NAMES)} (default: None)")
    parser.add_argument('--noise_type', default=None, choices=NOISE_NAMES,
                        help=f"models: {', '.join(NOISE_NAMES)} (default: None)")
    parser.add_argument('--noise_coeff', type=float, default=None,  help=f"noise coefficient (default: 0.1)")
    parser.add_argument('--latent_dim', type=int, default=None, help='latent dimensions (default: 200)')
    parser.add_argument('--AD_num_layers', default=None, type=int, help='Autoencoder layers (default: 5)')
    parser.add_argument('--AD_kernel_size', default=None, type=int, help='Autoencoder kernel size (default: 3)')
    parser.add_argument('--AD_dropout', default=None, type=float, help=f"(default: 0.1)")
    parser.add_argument('--AD_activation_func', default=None, choices=['leaky_relu', 'relu'], help=f'activations: leaky_relu, relu (default: leaky_relu)')
    parser.add_argument('--num_workers', type=int, default=10, help='(default: 10)')
    parser.add_argument('--loss', default='BCELoss', choices=LOSS_NAMES,
                        help=f"loss: {', '.join(LOSS_NAMES)} (default: BCELoss)")
    parser.add_argument('--shift_invariant', action='store_true', help='(default: False)')
    parser.add_argument('--random_flip', action='store_true', help='Randomly flip training patches horizontally and vertically (default: False)')
    parser.add_argument('--alpha', type=float, default=0.3, help='Tversky Loss alpha value (default: 0.3)')
    parser.add_argument('--beta', type=float, default=0.7, help='Tversky Loss beta value (default: 0.7)')
    parser.add_argument('--optimizer', default='Adam', choices=['Adam', 'SGD'],
                        help=f"optimizer: {', '.join(['Adam', 'SGD'])} (default: Adam)")
    
    # benchmarking settings
    parser.add_argument('--project', default="SARClassifierBenchmark", help='Wandb project where run will be logged')
    parser.add_argument('--group', default='main', help='Group name for model experiments')
    parser.add_argument('--trials', type=int, default=20, help='number of trials (default: 20)')
    parser.add_argument('--max_evals', type=int, default=5, help='max trials per benchmarking run - should divide into number of trials (default: 5)')
    parser.add_argument('--num_sample_predictions', type=int, default=60, help='number of predictions to visualize (default: 40)')
    
    config = vars(parser.parse_args())

    sys.exit(main(config))
