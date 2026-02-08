# Model Tuning

Bayesian optimization of model hyperparameters using [DeepHyper](https://deephyper.readthedocs.io/en/stable/index.html).

## Scripts
- `tuning_unet.py` - Tune UNet models (S2 and S1/SAR)
- `tuning_unet_plus.py` - Tune UNet++ models (S2 and S1/SAR)

## Configuration

The tuning config group (`configs/tuning/`) requires:
```yaml
dataset: s2 # ['s1', 's2']
run_index: 0 # 0 for random trials, 1+ for bayesian opt (requires 10+ previous runs)
max_evals: 10 # max evaluations per run
experiment_name: unet_s2_tuning # directory name for results
early_stopping: true
patience: 30
random_state: 230
```

## Defining Hyperparameters

Add hyperparameters to the `HpProblem` object within the tuning script:

```python
problem = HpProblem()
problem.add_hyperparameter((0.00001, 0.01, "log-uniform"), "learning_rate")
problem.add_hyperparameter((0.05, 0.40), "dropout")
problem.add_hyperparameter(["BCELoss", "BCEDiceLoss", "TverskyLoss"], "loss")
problem.add_hyperparameter(['Constant', 'ReduceLROnPlateau'], 'LR_scheduler')
```

Then map them to the config in the run function:

```python
def run_s2(parameters, cfg: DictConfig):
    cfg.train.loss = parameters['loss']
    cfg.train.lr = parameters['learning_rate']
    cfg.train.LR_scheduler = parameters['LR_scheduler']
    cfg.model.unet.dropout = parameters['dropout']
    # ...
```

## Usage

```bash
python -m floodmaps.tuning.tuning_unet
```

Results are saved to `outputs/tuning/{dataset}/{experiment_name}/`.
