from deephyper.hpo import HpProblem
from deephyper.hpo import CBO
from deephyper.evaluator import Evaluator
from deephyper.evaluator.callback import SearchEarlyStopping
from datetime import datetime
import pandas as pd
import socket
from pathlib import Path
import hydra
from omegaconf import DictConfig

from floodmaps.training.train_s2 import run_experiment_s2
from floodmaps.training.train_sar import run_experiment_s1
from floodmaps.utils.tuning_utils import load_stopper_info, save_stopper_info, print_save_best_params, save_problem

def run_s2(parameters, cfg: DictConfig):
    # cfg.wandb.project = 'S2_NoDEM_Tuning'
    # cfg.wandb.group = 'UNet_Fixed_Dropout'
    cfg.train.loss = parameters['loss']
    cfg.train.lr = parameters['learning_rate']
    cfg.train.LR_scheduler = parameters['LR_scheduler']
    cfg.model.unet.dropout = parameters['dropout']
    fmetrics = run_experiment_s2(cfg)
    results = fmetrics.get_metrics(split='val')
    return results['core_metrics']['val f1']

def tuning_s2(cfg: DictConfig) -> None:
    timestamp = datetime.now().strftime('%Y-%m-%d %H:%M:%S')
    hostname = socket.gethostname()
    print("----------------------------------")
    print("Running tuning script on:", hostname)
    print("Current timestamp:", timestamp)
    print(f'Beginning tuning for s2 experiment {cfg.tuning.experiment_name} for {cfg.tuning.max_evals} max evals using random seed {cfg.tuning.random_state}.')
    use_weak = getattr(cfg.data, 'use_weak', False)
    dataset_type = 's2_weak' if use_weak else 's2'
    search_dir = Path(cfg.paths.tuning_dir) / dataset_type / cfg.tuning.experiment_name
    search_dir.mkdir(parents=True, exist_ok=True)

    # define the variable you want to optimize
    problem = HpProblem()
    problem.add_hyperparameter((0.00001, 0.01), "learning_rate") # real parameter
    problem.add_hyperparameter((0.05, 0.40), "dropout")
    problem.add_hyperparameter(["BCELoss", "BCEDiceLoss", "TverskyLoss"], "loss")
    problem.add_hyperparameter(['Constant', 'ReduceLROnPlateau'], 'LR_scheduler')
    # problem.add_hyperparameter((0.20, 0.40), "alpha") # For tversky keep constant alpha (tune later)
    # problem.add_hyperparameter([True, False], "deep_supervision")

    # save problem to json
    save_problem(problem, search_dir / 'problem.json')

    # pass in cfg object to run experiment
    method_kwargs = {"run_function_kwargs": {"cfg": cfg}}

    # load in early stopping metrics if available
    if cfg.tuning.early_stopping:
        early_stopper = SearchEarlyStopping(patience=cfg.tuning.patience)
        if (search_dir / 'stopper.json').exists():
            _best_objective, _n_lower = load_stopper_info(search_dir / 'stopper.json')
            early_stopper._best_objective = _best_objective
            early_stopper._n_lower = _n_lower
        method_kwargs.update({"callbacks": [early_stopper]})

    # define the evaluator to distribute the computation
    with Evaluator.create(run_s2, method="process", method_kwargs=method_kwargs) as evaluator:
        print(f"Created new evaluator with {evaluator.num_workers} \
            worker{'s' if evaluator.num_workers > 1 else ''} and config: {method_kwargs}")
        search = CBO(problem, evaluator, surrogate_model="RF", log_dir=search_dir, random_state=cfg.tuning.random_state)

        if int(cfg.tuning.run_index) >= 1:
            # fit model from previous checkpointed search
            search.fit_surrogate(str(search_dir / 'all.csv'))

            # execute search
            results = search.search(max_evals=cfg.tuning.max_evals, timeout=23*3600)
        else:
            results = search.search(max_evals=cfg.tuning.max_evals, timeout=23*3600)

        print('Saving stopper and eval results to file...')
        if cfg.tuning.early_stopping:
            save_stopper_info(early_stopper, search_dir / 'stopper.json')

        # save results to collective file
        save_file = search_dir / 'all.csv'
        if save_file.exists():
            # note: need to reorder column headers if a bunch of new hyperparameters are added to match csv file
            existing_df = pd.read_csv(save_file, nrows=0)
            existing_columns = existing_df.columns.tolist()
            results = results[existing_columns]

            results.to_csv(save_file, mode='a', index=False, header=False)
        else:
            results.to_csv(save_file, index=False)
        
        # if search stopped print params of best run
        if cfg.tuning.early_stopping and early_stopper.search_stopped:
            print_save_best_params(save_file, search_dir / 'best.json')

def run_s1(parameters, cfg: DictConfig):
    """For setting the params in the cfg object make sure they are predeclared
    either as None or as some value to avoid error"""
    cfg.wandb.project = 'S1_NoDEM_All_Tuning'
    cfg.wandb.group = 'UNet'
    cfg.train.loss = parameters['loss']
    cfg.train.lr = parameters['learning_rate']
    cfg.train.LR_scheduler = parameters['LR_scheduler']
    cfg.model.unet.dropout = parameters['dropout']
    ad_cfg = getattr(cfg, 'ad', None)
    fmetrics = run_experiment_s1(cfg, ad_cfg=ad_cfg)
    results = fmetrics.get_metrics(split='val', partition='shift_invariant')
    return results['core_metrics']['val f1']

def tuning_s1(cfg: DictConfig) -> None:
    timestamp = datetime.now().strftime('%Y-%m-%d %H:%M:%S')
    hostname = socket.gethostname()
    print("----------------------------------")
    print("Running tuning script on:", hostname)
    print("Current timestamp:", timestamp)
    print(f'Beginning tuning for s1 experiment {cfg.tuning.experiment_name} for {cfg.tuning.max_evals} max evals using random seed {cfg.tuning.random_state}.')
    search_dir = Path(cfg.paths.tuning_dir) / 's1_weak' / cfg.tuning.experiment_name
    search_dir.mkdir(parents=True, exist_ok=True)

    # define the variable you want to optimize
    # We want to create model w dem and model wo dem
    problem = HpProblem()
    # problem.add_hyperparameter((0.3, 0.45), "alpha")
    problem.add_hyperparameter((0.00001, 0.01), "learning_rate") # real parameter
    problem.add_hyperparameter((0.05, 0.40), "dropout")
    problem.add_hyperparameter(["BCELoss", "BCEDiceLoss", "TverskyLoss"], "loss")
    problem.add_hyperparameter(['Constant', 'ReduceLROnPlateau'], 'LR_scheduler')

    # optional autodespeckler CNN first
    # problem.add_hyperparameter([1, 2, 3, 4, 5], "AD_num_layers")
    # problem.add_hyperparameter([3, 5, 7], "AD_kernel_size")
    # problem.add_hyperparameter((0.05, 0.30), "AD_dropout")
    # problem.add_hyperparameter(["leaky_relu", "relu"], "AD_activation_func")

    # save problem to json
    save_problem(problem, search_dir / 'problem.json')

    # pass in cfg object to run experiment
    method_kwargs = {"run_function_kwargs": {"cfg": cfg}}

    # load in early stopping metrics if available
    if cfg.tuning.early_stopping:
        early_stopper = SearchEarlyStopping(patience=cfg.tuning.patience)
        if (search_dir / 'stopper.json').exists():
            _best_objective, _n_lower = load_stopper_info(search_dir / 'stopper.json')
            early_stopper._best_objective = _best_objective
            early_stopper._n_lower = _n_lower
        method_kwargs.update({"callbacks": [early_stopper]})

    # define the evaluator to distribute the computation
    with Evaluator.create(run_s1, method="process", method_kwargs=method_kwargs) as evaluator:
        print(f"Created new evaluator with {evaluator.num_workers} \
            worker{'s' if evaluator.num_workers > 1 else ''} and config: {method_kwargs}")
        search = CBO(problem, evaluator, surrogate_model="RF", log_dir=search_dir, random_state=cfg.tuning.random_state)

        if int(cfg.tuning.run_index) >= 1:
            # fit model from previous checkpointed search
            search.fit_surrogate(str(search_dir / 'all.csv'))

            # execute search
            results = search.search(max_evals=cfg.tuning.max_evals, timeout=23*3600)
        else:
            results = search.search(max_evals=cfg.tuning.max_evals, timeout=23*3600)

        print('Saving stopper and eval results to file...')
        if cfg.tuning.early_stopping:
            save_stopper_info(early_stopper, search_dir / 'stopper.json')

        # save results to collective file
        save_file = search_dir / 'all.csv'
        if save_file.exists():
            # note: need to reorder column headers if a bunch of new hyperparameters are added to match csv file
            existing_df = pd.read_csv(save_file, nrows=0)
            existing_columns = existing_df.columns.tolist()
            results = results[existing_columns]

            results.to_csv(save_file, mode='a', index=False, header=False)
        else:
            results.to_csv(save_file, index=False)
        
        # if search stopped print params of best run
        if cfg.tuning.early_stopping and early_stopper.search_stopped:
            print_save_best_params(save_file, search_dir / 'best.json')

@hydra.main(version_base=None, config_path="pkg://configs", config_name="config.yaml")
def main(cfg: DictConfig) -> None:
    if cfg.tuning.dataset == 's2':
        tuning_s2(cfg)
    elif cfg.tuning.dataset == 's1':
        tuning_s1(cfg)

if __name__ == "__main__":
    main()