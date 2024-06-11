from train_sar import run_experiment_s1
from random import Random
import pandas as pd
import argparse
import os
import sys

def main(config, save_file='benchmarks/runs.csv', seed=13032):
    """Given model parameters and number of trials n, runs identical experiment n times each with
    unique seeding. Then calculates the mean and std of metrics across all n experiments.
    The collected data can then be used to compare model performance using a t-test.

    Note: run with conda environment 'floodmapsdp'.
    """
    vacc_list = []
    vpre_list = []
    vrec_list = []
    vf1_list = []

    rng = Random(seed)
    seeds = rng.sample(range(0, 100000), config['trials'])

    # each trial uses different random seed
    # save in pandas dataframe with config parameters, number of runs, mean of run metrics, std of run metrics
    # test set
    config['mode'] = 'test'
    for seed in range(seeds):
        config['seed'] = seed
        final_vacc, final_vpre, final_vrec, final_vf1 = run_experiment_s1(config)
        vacc_list.append(final_vacc)
        vpre_list.append(final_vpre)
        vrec_list.append(final_vrec)
        vf1_list.append(final_vf1)

    # calculate mean and std
    results = {
        'group': config['group'],
        'trials': config['trials'],
        'method': config['method'],
        'filter': config['filter'],
        'channels': ''.join('1' if b else '0' for b in config['channels']),
        'patch_size': config['size'],
        'architecture': config['name'],
        'dropout': config['dropout'],
        'deep_supervision': config['deep_supervision'],
        'autodespeckler': config['autodespeckler'],
        'noise_type': config['noise_type'],
        'latent_dim': config['latent_dim'],
        'AD_dropout': config['AD_dropout'],
        'learning_rate': config['learning_rate'],
        'epochs': config['epochs'],
        'early_stopping': config['early_stopping'],
        'patience': config['patience'],
        'batch_size': config['batch_size'],
        'optimizer': config['optimizer'],
        'loss_fn': config['loss'],
        'alpha': config['alpha'],
        'beta': config['beta'],
        'subset': config['subset'],
        'training_size': len(train_set),
        'evaluation_size': len(val_set),
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
        results_df.to_csv(save_file, mode='a', index=False, header=False)
    else:
        results_df.to_csv(save_file, index=False)

if __name__ == "__main__":
    parser = argparse.ArgumentParser(prog='benchmark_models', description='Benchmarks SAR classifier model on test set given parameters.')
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
    parser.add_argument('-p', '--patience', type=int, default=5, help='(default: 5)')
    parser.add_argument('--name', default='unet', choices=MODEL_NAMES,
                        help=f"models: {', '.join(MODEL_NAMES)} (default: unet)")
    parser.add_argument('--autodespeckler', default=None, choices=AUTODESPECKLER_NAMES,
                        help=f"models: {', '.join(AUTODESPECKLER_NAMES)} (default: None)")
    parser.add_argument('--noise_type', default=None, choices=NOISE_NAMES,
                        help=f"models: {', '.join(NOISE_NAMES)} (default: None)")
    parser.add_argument('--latent_dim', type=int, default=200, help='latent dimensions (default: 200)')
    parser.add_argument('--AD_dropout', type=float, default=0.1, help=f"(default: 0.1)")
    parser.add_argument('--dropout', type=float, default=0.2, help=f"(default: 0.2)")
    parser.add_argument('--deep_supervision', action='store_true', help='(default: False)')
    parser.add_argument('--num_workers', type=int, default=10, help='(default: 10)')
    parser.add_argument('--loss', default='BCELoss', choices=LOSS_NAMES,
                        help=f"loss: {', '.join(LOSS_NAMES)} (default: BCELoss)")
    parser.add_argument('--alpha', type=float, default=0.3, help='Tversky Loss alpha value (default: 0.3)')
    parser.add_argument('--beta', type=float, default=0.7, help='Tversky Loss beta value (default: 0.7)')
    parser.add_argument('--optimizer', default='Adam', choices=['Adam', 'SGD'],
                        help=f"optimizer: {', '.join(['Adam', 'SGD'])} (default: Adam)")
    
    # benchmarking settings
    parser.add_argument('--project', default="SARClassifierBenchmark", help='Wandb project where run will be logged')
    parser.add_argument('--group', default='main', help='Group name for model experiments')
    parser.add_argument('--trials', type=int, default=20, help='number of trials (default: 20)')
    parser.add_argument('--num_sample_predictions', type=int, default=60, help='number of predictions to visualize (default: 40)')
    
    config = vars(parser.parse_args())

    sys.exit(main(config))
