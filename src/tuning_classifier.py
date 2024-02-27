from deephyper.problem import HpProblem
from deephyper.search.hps import CBO
from deephyper.evaluator import Evaluator
from deephyper.evaluator.callback import SearchEarlyStopping
from train_classifier import run_experiment
import argparse
import os

def run(job):
    channels = [bool(int(x)) for x in job.parameters['channels']]
    config = {
        'method': job.parameters['method'], 
        'size': job.parameters['size'], 
        'samples': job.parameters['samples'], 
        'channels': channels},
        'sample_dir': '../samples_200_5_4_35/',
        'label_dir': '../labels/',
        'project': 'FSUNetTuning', 
        'epochs': job.parameters['epochs'], 
        'batch_size': job.parameters['batch_size'], 
        'learning_rate': job.parameters['learning_rate'], 
        'early_stopping': True, 
        'name': 'unet',
        'loss': job.parameters['loss'],
        'alpha': 0.3,
        'beta': 0.7,
        'optimizer': job.parameters['optimizer']
    }
    vf1 = run_experiment(config)
    return vf1

if __name__ == "__main__":
    parser = argparse.ArgumentParser()
    parser.add_argument("-i", "--run_index", type=int, default=0)
    parser.add_argument("-n", "--experiment-name", type=str, default="unetclassifier")

    args = parser.parse_args()
    file_index = args.run_index
    experiment_name = args.experiment_name
    search_dir = './tuning/' + experiment_name

    if not os.path.exists(search_dir):
        os.makedirs(search_dir)

    # define the variable you want to optimize
    problem = HpProblem()
    problem.add_hyperparameter(["random"], "method")
    problem.add_hyperparameter([64], "size")
    problem.add_hyperparameter([1000], "samples")
    problem.add_hyperparameter(["111111111", "111111100", "111110000"], "channels")

    problem.add_hyperparameter(["BCELoss", "BCEDiceLoss", "TverskyLoss"], "loss")
    # how do we optimize beta values only if tversky being used?
    problem.add_hyperparameter(["Adam", "SGD"], "optimizer")

    problem.add_hyperparameter([16, 32, 64, 128, 256, 512, 1024], "batch_size")
    problem.add_hyperparameter((0.00001, 0.01), "learning_rate") # real parameter
    problem.add_hyperparameter([50, 100], "epochs")

    # define the evaluator to distribute the computation
    method_kwargs = {"callbacks": [SearchEarlyStopping(patience=10)]}

    with Evaluator.create(run, method="process", method_kwargs=method_kwargs) as evaluator:
        search = CBO(problem, evaluator, surrogate_model="RF", log_dir=search_dir, random_state=42)
        
        if int(file_index) >= 1:
            # fit model from previous checkpointed search
            search.fit_surrogate(search_dir)

            # execute search
            results = search.search(timeout=12*3600)
        else:
            results = search.search(timeout=12*3600)