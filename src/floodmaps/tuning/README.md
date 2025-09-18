# Model Tuning

The arguments for the benchmark config group:
```yaml
dataset: s2 # ['s1', 's2']
run_index: 0 # 0 for random trials, 1 for bayesian opt (requires 10+ previous runs)
max_evals: 1 # max number of evaluations per run
experiment_name: unet_s2 # name of tuning experiments (will be directory where results are saved)
early_stopping: False # early stopping for bayesian opt
patience: 10
random_state: 230
```

Bayesian optimization of model hyperparameters works through the [DeepHyper](https://deephyper.readthedocs.io/en/stable/index.html) package. Specifying parameters to tune must be done by adding hyperparameters to the `HpProblem` object within the script:

```python
# define the variable you want to optimize
problem = HpProblem()
problem.add_hyperparameter((0.00001, 0.01), "learning_rate") # real parameter
problem.add_hyperparameter((0.05, 0.40), "dropout")
problem.add_hyperparameter(["BCELoss", "BCEDiceLoss", "TverskyLoss"], "loss")
problem.add_hyperparameter(['Constant', 'ReduceLROnPlateau'], 'LR_scheduler')
```

Inside the `run_function` it is necessary then to store the chosen hyperparameter in the `cfg` object:

```python
def run_s2(parameters, cfg: DictConfig):
    cfg.wandb.project = 'Texas_S2_NoDEM_Tuning'
    cfg.wandb.group = 'UNet'
    cfg.train.loss = parameters['loss']
    cfg.train.lr = parameters['learning_rate']
    cfg.train.LR_scheduler = parameters['LR_scheduler']
    cfg.model.unet.dropout = parameters['dropout']
```

Note: The current tuning implementation is serial, so each experiment is done in sequence. This might result in long tuning times for large models.