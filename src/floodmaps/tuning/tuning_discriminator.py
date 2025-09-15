from deephyper.problem import HpProblem
from deephyper.search.hps import CBO
from deephyper.evaluator import Evaluator
from deephyper.evaluator.callback import SearchEarlyStopping
import pandas as pd
import argparse
import os
import math

from floodmaps.training.train_discriminator import run_experiment

def run(job):
    # channels = [bool(int(x)) for x in str(int(job.parameters['channels']))]
    config = {
        'method': "random", # random
        'size': 64,
        'samples': 1000,
        'name': "c1",
        'channels': [bool(int(x)) for x in '1111111111'],
        'sample_dir': '../sampling/samples_200_5_4_35/',
        'label_dir': '../sampling/labels/',
        'project': 'FloodSamplesDiscriminatorTuning2',
        'group': None,
        'num_sample_predictions': 60,
        'epochs': 100,
        "num_pixels": 1,
        'batch_size': job.parameters['batch_size'],
        'num_workers': 0,
        'learning_rate': job.parameters['learning_rate'] * math.sqrt(job.parameters['batch_size'] // 16), # job.parameters['learning_rate'],
        'early_stopping': True,
        'patience': 10,
        'loss': 'TverskyLoss', # job.parameters['loss'],
        'alpha': job.parameters['alpha'],
        'beta': 1 - job.parameters['alpha'],
        'optimizer': "Adam"
    }
    final_vacc, final_vpre, final_vrec, final_vf1 = run_experiment(config)
    return final_vrec

if __name__ == "__main__":
    parser = argparse.ArgumentParser()
    parser.add_argument("-i", "--run_index", type=int, default=0)
    parser.add_argument("-e", "--max_evals", type=int, default=1)
    parser.add_argument("-n", "--experiment_name", type=str, default="discriminator_recall")
    parser.add_argument('--early_stopping', action='store_true', help='early stopping (default: False)')

    args = parser.parse_args()
    file_index = args.run_index
    max_evals = args.max_evals
    experiment_name = args.experiment_name
    early_stopping = args.early_stopping
    search_dir = './tuning/' + experiment_name

    if not os.path.exists(search_dir):
        os.makedirs(search_dir)

    # define the variable you want to optimize
    # assume channels is 111111111
    problem = HpProblem()
    problem.add_hyperparameter((0.1, 0.3), "alpha")
    problem.add_hyperparameter([16, 32, 64, 128, 256, 512], "batch_size")
    problem.add_hyperparameter((0.00001, 0.001), "learning_rate") # real parameter
    # problem.add_hyperparameter(["BCELoss", "BCEDiceLoss", "TverskyLoss"], "loss")

    # define the evaluator to distribute the computation
    method_kwargs = {"callbacks": [SearchEarlyStopping(patience=10)]} if early_stopping else dict()

    with Evaluator.create(run, method="serial", method_kwargs=method_kwargs) as evaluator:
        search = CBO(problem, evaluator, surrogate_model="RF", log_dir=search_dir, random_state=20295)

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
