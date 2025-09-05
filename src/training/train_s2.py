import wandb
import torch
import logging
import argparse
import torch
from torch import nn
from torch.utils.data import DataLoader
from datetime import datetime
from torchvision import transforms
from torchmetrics import MetricCollection
from torchmetrics.classification import BinaryAccuracy, BinaryPrecision, BinaryRecall, BinaryF1Score, BinaryConfusionMatrix
from matplotlib.cm import ScalarMappable
from matplotlib.colors import Normalize
import random
from random import Random
from PIL import Image
import numpy as np
import sys
import pickle
from pathlib import Path
import json

from models.model import S2WaterDetector
from utils.config import Config
from utils.utils import DATA_DIR, RESULTS_DIR, get_model_params, Metrics, EarlyStopper, ChannelIndexer, nlcd_to_rgb
from utils.checkpoint import save_checkpoint, load_checkpoint
from utils.metrics import compute_nlcd_metrics

from training.loss import BCEDiceLoss, TverskyLoss
from training.dataset import FloodSampleS2Dataset
from training.optim import get_optimizer
from training.scheduler import get_scheduler

# TO IMPLEMENT: WITH DISCRIMINATOR, ADDITIONAL TRACKING OF DISCRIMINATOR OUTPUTS
# COULD ALSO CONSIDER REMOVING DISCRIMINATOR, ONLY ATTACHING FOR TEST SET EVALUATION

MODEL_NAMES = ['unet', 'unet++']
LOSS_NAMES = ['BCELoss', 'BCEDiceLoss', 'TverskyLoss']

def get_loss_fn(cfg):
    if cfg.train.loss == 'BCELoss':
        loss_fn = nn.BCEWithLogitsLoss()
    elif cfg.train.loss == 'BCEDiceLoss':
        loss_fn = BCEDiceLoss()
    elif cfg.train.loss == 'TverskyLoss':
        loss_fn = TverskyLoss(alpha=cfg.train.tversky.alpha, beta=1-cfg.train.tversky.alpha)
    else:
        raise Exception(f'Loss function not found. Must be one of {LOSS_NAMES}')

    return loss_fn

# get our optimizer and metrics
def train_loop(model, dataloader, device, optimizer, loss_fn, run, epoch):
    running_loss = torch.tensor(0.0, device=device)
    metric_collection = MetricCollection([
        BinaryAccuracy(threshold=0.5),
        BinaryPrecision(threshold=0.5),
        BinaryRecall(threshold=0.5),
        BinaryF1Score(threshold=0.5)
    ]).to(device)
    all_preds = []
    all_targets = []

    model.train()
    for X, y, _ in dataloader:
        X = X.to(device)
        y = y.to(device)

        logits = model(X)
        loss = loss_fn(logits, y.float())
        
        optimizer.zero_grad()
        loss.backward()
        optimizer.step()

        y_pred = nn.functional.sigmoid(logits).flatten() > 0.5
        target = y.flatten() > 0.5
        all_preds.append(y_pred)
        all_targets.append(target)
        running_loss += loss.detach()

    # calculate metrics
    all_preds = torch.cat(all_preds)
    all_targets = torch.cat(all_targets)

    metric_collection.update(all_preds, all_targets)
    metric_results = metric_collection.compute()
    epoch_loss = running_loss.item() / len(dataloader)

    # wandb tracking loss and metrics per epoch - track recons loss as well
    log_dict = {"train accuracy": metric_results['BinaryAccuracy'].item(),
                "train precision": metric_results['BinaryPrecision'].item(),
                "train recall": metric_results['BinaryRecall'].item(),
                "train f1": metric_results['BinaryF1Score'].item(),
                "train loss": epoch_loss}
    run.log(log_dict, step=epoch)
    metric_collection.reset()

    return epoch_loss

def test_loop(model, dataloader, device, loss_fn, run, epoch, typ='val'):
    """Evaluate model on validation or test set with optional wandb logging.
    
    Parameters
    ----------
    model : torch.nn.Module
        The model to evaluate
    dataloader : torch.utils.data.DataLoader
        DataLoader for the evaluation set
    device : torch.device
        Device to run evaluation on
    loss_fn : callable
        Loss function
    run : wandb.run
        Wandb run object for logging (can be None for test evaluation)
    epoch : int
        Current epoch number (used for wandb logging)
    typ : str, optional
        Type of evaluation: 'val' for validation (logs to wandb), 'test' for test (no logging)
    
    Returns
    -------
    tuple
        (epoch_loss, metrics_dict)
    """
    running_vloss = torch.tensor(0.0, device=device)
    metric_collection = MetricCollection([
        BinaryAccuracy(threshold=0.5),
        BinaryPrecision(threshold=0.5),
        BinaryRecall(threshold=0.5),
        BinaryF1Score(threshold=0.5),
        BinaryConfusionMatrix(threshold=0.5)
    ]).to(device)
    all_preds = []
    all_targets = []
    all_nlcd_classes = []
    
    model.eval()
    with torch.no_grad():
        for X, y, supplementary in dataloader:
            X = X.to(device)
            y = y.to(device)
            nlcd_classes = supplementary[3, :, :].to(device)

            logits = model(X)
            loss = loss_fn(logits, y.float())
            
            y_pred = nn.functional.sigmoid(logits).flatten() > 0.5
            target = y.flatten() > 0.5
            all_preds.append(y_pred)
            all_targets.append(target)
            all_nlcd_classes.append(nlcd_classes)
            running_vloss += loss.detach()

    # calculate metrics
    all_preds = torch.cat(all_preds)
    all_targets = torch.cat(all_targets)
    all_nlcd_classes = torch.cat(all_nlcd_classes)

    metric_collection.update(all_preds, all_targets)
    metric_results = metric_collection.compute()
    epoch_vloss = running_vloss.item() / len(dataloader)

    metrics_dict = {
        f"{typ} accuracy": metric_results['BinaryAccuracy'].item(),
        f"{typ} precision": metric_results['BinaryPrecision'].item(),
        f"{typ} recall": metric_results['BinaryRecall'].item(),
        f"{typ} f1": metric_results['BinaryF1Score'].item()
    }
    
    # Only log to wandb for validation (not for test evaluation)
    if typ == 'val' and run is not None:
        log_dict = metrics_dict.copy()
        log_dict.update({f'{typ} loss': epoch_vloss})
        run.log(log_dict, step=epoch)

    # calculate confusion matrix + NLCD class metrics
    confusion_matrix = metric_results['BinaryConfusionMatrix'].tolist()
    confusion_matrix_dict = {
        "tn": confusion_matrix[0, 0],
        "fp": confusion_matrix[0, 1],
        "fn": confusion_matrix[1, 0],
        "tp": confusion_matrix[1, 1]
    }
    metrics_dict.update({f"{typ} confusion matrix": confusion_matrix_dict})
    metrics_dict.update("nlcd metrics": compute_nlcd_metrics(all_preds, all_targets, all_nlcd_classes))

    metric_collection.reset()

    return epoch_vloss, metrics_dict

def train(model, train_loader, val_loader, test_loader, device, cfg, run):
    timestamp = datetime.now().strftime('%Y-%m-%d %H:%M:%S')
    logging.info(f'''Starting training:
        Date:            {timestamp}
        Epochs:          {cfg.train.epochs}
        Batch size:      {cfg.train.batch_size}
        Learning rate:   {cfg.train.lr}
        Training size:   {len(train_loader.dataset)}
        Validation size: {len(val_loader.dataset)}
        Test size:       {len(test_loader.dataset) if test_loader is not None else 'NA'}
        Device:          {device}
    ''')
    # log weights and gradients each epoch
    run.watch(model, log="all", log_freq=10)

    # loss function
    loss_fn = get_loss_fn(cfg)

    # optimizer and scheduler for reducing learning rate
    optimizer = get_optimizer(model, cfg)
    scheduler = get_scheduler(optimizer, cfg)
    early_stopper = EarlyStopper(patience=cfg.train.patience) if cfg.train.early_stopping else None

    # load checkpoint if it exists
    if cfg.train.checkpoint.load_chkpt:
        chkpt = load_checkpoint(cfg.train.checkpoint.chkpt_path, model, optimizer=optimizer, scheduler=scheduler, early_stopper=early_stopper)
        start_epoch = chkpt['epoch'] + 1
        if cfg.train.epochs < start_epoch:
            raise ValueError(f"Epochs specified in config ({cfg.train.epochs}) is less than the epoch at which the checkpoint was saved ({start_epoch}).")
    else:
        start_epoch = 0

    for epoch in range(start_epoch, cfg.train.epochs):
        try:
            # train loop
            avg_loss = train_loop(model, train_loader, device, optimizer, loss_fn, run, epoch)

            # at the end of each training epoch compute validation
            avg_vloss, val_set_metrics = test_loop(model, val_loader, device, loss_fn, run, epoch, typ='val')
        except Exception as err:
            raise RuntimeError(f'Error while training occurred at epoch {epoch}.') from err

        if cfg.train.early_stopping:
            early_stopper.step(avg_vloss, model)
            early_stopper.store_best_metrics(val_set_metrics)
            if early_stopper.is_stopped():
                break

        if scheduler is not None:
            if isinstance(scheduler, torch.optim.lr_scheduler.ReduceLROnPlateau):
                scheduler.step(avg_vloss)
            else:
                scheduler.step()

        if cfg.train.checkpoint.save_chkpt and epoch % cfg.train.checkpoint.save_chkpt_interval == 0:
            save_checkpoint(cfg.train.checkpoint.save_chkpt_path, model, optimizer, epoch, scheduler=scheduler, early_stopper=early_stopper)
            
        run.log({"learning_rate": scheduler.get_last_lr()[0] if scheduler is not None else cfg.train.lr}, step=epoch)

    # Save our model
    fmetrics = Metrics(use_partitions=False)
    cls_weights = None
    if cfg.train.early_stopping:
        model_weights = early_stopper.get_best_weights()
        best_val_metrics = early_stopper.get_best_metrics()
        # reset model to checkpoint for later sample prediction
        model.load_state_dict(model_weights)
        cls_weights = model.classifier.state_dict()
        disc_weights = (model.discriminator.state_dict()
                      if model.uses_discriminator() else None)
        fmetrics.save_metrics('val', loss=early_stopper.get_min_validation_loss(), **best_val_metrics)
        run.summary.update({f'final model {key}': value for key, value in best_val_metrics.items()})
    else:
        cls_weights = model.classifier.state_dict()
        disc_weights = (model.discriminator.state_dict()
                      if model.uses_discriminator() else None)
        fmetrics.save_metrics('val', loss=avg_vloss, **val_set_metrics)
        run.summary.update({f'final model {key}': value for key, value in val_set_metrics.items()})

    # for benchmarking purposes
    if cfg.eval.mode == 'test':
        test_loss, test_set_metrics = test_loop(model, test_loader, device, loss_fn, None, None, typ='test')
        fmetrics.save_metrics('test', loss=test_loss, **test_set_metrics)
        run.summary.update({f'final model {key}': value for key, value in test_set_metrics.items()})

    return cls_weights, disc_weights, fmetrics

def save_experiment(cls_weights, disc_weights, metrics, cfg, run):
    """Save experiment files to directory specified by config save_path."""
    path = Path(cfg.save_path)
    path.mkdir(parents=True, exist_ok=True)

    if cls_weights is not None:
        torch.save(cls_weights, path / f"{cfg.model.classifier}_cls.pth")
    if disc_weights is not None:
        torch.save(disc_weights, path / f"{cfg.model.discriminator}_disc.pth")

    # save config
    cfg.save2yaml(path / "config.yaml")

    # save metrics
    metrics.to_json(path / "metrics.json")

    # save wandb id info to json
    wandb_info = {
        "entity": run.entity,
        "project": run.project,
        "group": run.group,
        "run_id": run.id,
        "name": run.name,
        "url": run.url,
        "dir": run.dir
    }
    with open(path / f"wandb_info.json", "w") as f:
        json.dump(wandb_info, f, indent=4)

def sample_predictions(model, sample_set, mean, std, cfg, seed=24330):
    """Generate predictions on a subset of images in the validation set for wandb logging.
    
    TO DO: FIX CHANNEL INDEXING HARDCODING. Need more flexible way to handle channels."""
    if cfg.wandb.num_sample_predictions <= 0:
        return None

    columns = ["id", "tci", "nlcd"] # TCI, NLCD always included
    channels = [bool(int(x)) for x in cfg.data.channels]
    my_channels = ChannelIndexer(channels) # ndwi, dem, slope_y, slope_x, waterbody, roads, flowlines
    # initialize wandb table given the channel settings
    columns += my_channels.get_display_channels()
    columns += ["truth", "prediction", "false positive", "false negative"] # added residual binary map
    table = wandb.Table(columns=columns)
    
    if my_channels.has_ndwi():
        # initialize mappable objects
        ndwi_norm = Normalize(vmin=-1, vmax=1)
        ndwi_map = ScalarMappable(norm=ndwi_norm, cmap='seismic_r')
    if my_channels.has_dem():
        dem_map = ScalarMappable(norm=None, cmap='gray')
    if my_channels.has_slope_y():
        slope_y_map = ScalarMappable(norm=None, cmap='RdBu')
    if my_channels.has_slope_x():
        slope_x_map = ScalarMappable(norm=None, cmap='RdBu')

    # get map of each channel to index of resulting tensor
    n = 0
    channel_indices = [-1] * 11
    for i, channel in enumerate(channels):
        if channel:
            channel_indices[i] = n
            n += 1
    
    model.to('cpu')
    model.eval()
    rng = Random(seed)
    samples = rng.sample(range(0, len(sample_set)), cfg.wandb.num_sample_predictions)

    for id, k in enumerate(samples):
        # get all images to shape (H, W, C) with C = 1 or 3 (1 for grayscale, 3 for RGB)
        X, y, supplementary = sample_set[k]
        
        with torch.no_grad():
            logits = model(X.unsqueeze(0))
            
        y_pred = torch.where(nn.functional.sigmoid(logits) > 0.5, 1.0, 0.0).squeeze(0) # (1, x, y)

        # Compute false positives
        fp = torch.logical_and(y == 0, y_pred == 1).squeeze(0).byte().mul(255).numpy()
        
        # Compute false negatives
        fn = torch.logical_and(y == 1, y_pred == 0).squeeze(0).byte().mul(255).numpy()

        # Channels are descaled using linear variance scaling
        X = X.permute(1, 2, 0)
        X = std * X + mean

        row = [k]

        # tci reference
        tci = supplementary[:3, :, :].permute(1, 2, 0).mul(255).clamp(0, 255).byte().numpy()
        tci_img = Image.fromarray(tci, mode="RGB")
        row.append(wandb.Image(tci_img))

        # NLCD land cover visualization
        nlcd = supplementary[3, :, :].byte().numpy()
        nlcd_rgb = nlcd_to_rgb(nlcd)
        nlcd_img = Image.fromarray(nlcd_rgb, mode="RGB")
        row.append(wandb.Image(nlcd_img))

        if my_channels.has_ndwi():
            ndwi = X[:, :, channel_indices[4]]
            ndwi = ndwi_map.to_rgba(ndwi.numpy(), bytes=True)
            ndwi = np.clip(ndwi, 0, 255).astype(np.uint8)
            ndwi_img = Image.fromarray(ndwi, mode="RGBA")
            row.append(wandb.Image(ndwi_img))
        if my_channels.has_dem():
            dem = X[:, :, channel_indices[5]].numpy()
            dem_map.set_norm(Normalize(vmin=np.min(dem), vmax=np.max(dem)))
            dem = dem_map.to_rgba(dem, bytes=True)
            dem = np.clip(dem, 0, 255).astype(np.uint8)
            dem_img = Image.fromarray(dem, mode="RGBA")
            row.append(wandb.Image(dem_img))
        if my_channels.has_slope_y():
            slope_y = X[:, :, channel_indices[6]].numpy()
            slope_y_map.set_norm(Normalize(vmin=np.min(slope_y), vmax=np.max(slope_y)))
            slope_y = slope_y_map.to_rgba(slope_y, bytes=True)
            slope_y = np.clip(slope_y, 0, 255).astype(np.uint8)
            slope_y_img = Image.fromarray(slope_y, mode="RGBA")
            row.append(wandb.Image(slope_y_img))
        if my_channels.has_slope_x():
            slope_x = X[:, :, channel_indices[7]].numpy()
            slope_x_map.set_norm(Normalize(vmin=np.min(slope_x), vmax=np.max(slope_x)))
            slope_x = slope_x_map.to_rgba(slope_x, bytes=True)
            slope_x = np.clip(slope_x, 0, 255).astype(np.uint8)
            slope_x_img = Image.fromarray(slope_x, mode="RGBA")
            row.append(wandb.Image(slope_x_img))
        if my_channels.has_waterbody():
            waterbody = X[:, :, channel_indices[8]].mul(255).clamp(0, 255).byte().numpy()
            waterbody_img = Image.fromarray(waterbody, mode="L")
            row.append(wandb.Image(waterbody_img))
        if my_channels.has_roads():
            roads = X[:, :, channel_indices[9]].mul(255).clamp(0, 255).byte().numpy()
            roads_img = Image.fromarray(roads, mode="L")
            row.append(wandb.Image(roads_img))
        if my_channels.has_flowlines():
            flowlines = X[:, :, channel_indices[10]].mul(255).clamp(0, 255).byte().numpy()
            flowlines_img = Image.fromarray(flowlines, mode="L")
            row.append(wandb.Image(flowlines_img))

        y = y.squeeze(0).mul(255).clamp(0, 255).byte().numpy()
        y_pred = y_pred.squeeze(0).mul(255).clamp(0, 255).byte().numpy()
        
        truth_img = Image.fromarray(y, mode="L")
        pred_img = Image.fromarray(y_pred, mode="L")
        fp_img = Image.fromarray(fp, mode="L")
        fn_img = Image.fromarray(fn, mode="L")
        row += [wandb.Image(truth_img), wandb.Image(pred_img), wandb.Image(fp_img), wandb.Image(fn_img)]

        table.add_data(*row)

    return table

def run_experiment_s2(cfg):
    """Run a single S2 model experiment given the configuration parameters."""
    if not wandb.login():
        raise Exception("Failed to login to wandb.")

    # seeding
    np.random.seed(cfg['seed'])
    random.seed(cfg['seed'])
    torch.manual_seed(cfg['seed'])
    
    device = (
        "cuda"
        if torch.cuda.is_available()
        else "mps"
        if torch.backends.mps.is_available()
        else "cpu"
    )

    # setup model
    model = S2WaterDetector(cfg).to(device)

    # dataset and transforms
    print(f"Using {device} device")
    model_name = cfg.model.classifier
    size = cfg.data.size
    samples = cfg.data.samples
    suffix = getattr(cfg.data, 'suffix', '')
    # Use weak labeled dataset if specified, otherwise use manual labeled dataset
    use_weak = getattr(cfg.data, 'use_weak', False)
    dataset_type = 's2_weak' if use_weak else 's2'
    if suffix:
        sample_dir = DATA_DIR / dataset_type / f'samples_{size}_{samples}_{suffix}/'
    else:
        sample_dir = DATA_DIR / dataset_type / f'samples_{size}_{samples}/'

    # load in mean and std
    channels = [bool(int(x)) for x in cfg.data.channels]
    with open(sample_dir / f'mean_std_{size}_{samples}.pkl', 'rb') as f:
        train_mean, train_std = pickle.load(f)

        train_mean = torch.from_numpy(train_mean[channels])
        train_std = torch.from_numpy(train_std[channels])

    standardize = transforms.Compose([transforms.Normalize(train_mean, train_std)])

    # datasets
    train_set = FloodSampleS2Dataset(sample_dir, channels=channels,
                                        typ="train", transform=standardize, random_flip=cfg.data.random_flip,
                                        seed=cfg.seed+1)
    val_set = FloodSampleS2Dataset(sample_dir, channels=channels,
                                    typ="val", transform=standardize)
    test_set = FloodSampleS2Dataset(sample_dir, channels=channels,
                                        typ="test", transform=standardize) if cfg.eval.mode == 'test' else None

    # dataloaders
    train_loader = DataLoader(train_set,
                             batch_size=cfg.train.batch_size,
                             num_workers=cfg.train.num_workers,
                             persistent_workers=cfg.train.num_workers>0,
                             pin_memory=True,
                             shuffle=True,
                             drop_last=False)

    val_loader = DataLoader(val_set,
                            batch_size=cfg.train.batch_size,
                            num_workers=cfg.train.num_workers,
                            persistent_workers=cfg.train.num_workers>0,
                            pin_memory=True,
                            shuffle=True,
                            drop_last=False)

    test_loader = DataLoader(test_set,
                            batch_size=cfg.train.batch_size,
                            num_workers=cfg.train.num_workers,
                            persistent_workers=cfg.train.num_workers>0,
                            pin_memory=True,
                            shuffle=True,
                            drop_last=False) if cfg.eval.mode == 'test' else None

    # initialize wandb run
    total_params, trainable_params, param_size_in_mb = get_model_params(model)
    run = wandb.init(
        project=cfg.wandb.project,
        group=cfg.wandb.group,
        config={
            "dataset": "Sentinel2",
            **cfg.to_dict(),
            "training_size": len(train_set),
            "validation_size": len(val_set),
            "test_size": len(test_set) if cfg.eval.mode == 'test' else None,
            "val_percent": len(val_set) / (len(train_set) + len(val_set)),
            "total_parameters": total_params,
            "trainable_parameters": trainable_params,
            "parameter_size_mb": param_size_in_mb
        }
    )

    try:
        if cfg.save:
            if cfg.save_path is None:
                default_path = RESULTS_DIR / "experiments" / f"{datetime.today().strftime('%Y-%m-%d')}_{cfg.model.classifier}_{run.id}/"
                cfg.save_path = str(default_path)
                run.config.update({"save_path": cfg.save_path}, allow_val_change=True)
            print(f'Save path set to: {cfg.save_path}')
        
        # train and save results metrics
        cls_weights, disc_weights, fmetrics = train(model, train_loader, val_loader,
                                              test_loader, device, cfg, run)
        if cfg.save:
            save_experiment(cls_weights, disc_weights, fmetrics, cfg, run)

        # log predictions on validation set using wandb
        pred_table = sample_predictions(model, test_set if cfg.eval.mode == 'test' else val_set,
                                        train_mean, train_std, cfg)
        run.log({"model_val_predictions": pred_table})
    except Exception as e:
        print("An exception occurred during training!")

        # Send an alert in the W&B UI
        run.alert(
            title="Training crashed 🚨",
            text=f"Run failed due to: {e}"
        )

        # Log to wandb summary
        run.summary["error"] = str(e)

        # remove save directory if needed
        raise e
    finally:
        run.finish()

    return fmetrics

def validate_config(cfg):
    def validate_channels(s):
        return type(s) == str and len(s) == 11 and all(c in '01' for c in s)

    # Add checks
    assert cfg.save in [True, False], "Save must be a boolean"
    if cfg.train.loss == 'TverskyLoss':
        assert 0.0 <= cfg.train.tversky.alpha <= 1.0, "Tversky alpha must be in [0, 1]"
    assert cfg.train.batch_size is not None and cfg.train.batch_size > 0, "Batch size must be defined and positive"
    assert cfg.train.lr > 0, "Learning rate must be positive"
    assert cfg.train.loss in LOSS_NAMES, f"Loss must be one of {LOSS_NAMES}"
    assert cfg.train.optimizer in ['Adam', 'SGD'], f"Optimizer must be one of {['Adam', 'SGD']}"
    assert cfg.train.LR_scheduler in ['Constant', 'ReduceLROnPlateau', 'CosAnnealingLR'], f"LR scheduler must be one of {['Constant', 'ReduceLROnPlateau', 'CosAnnealingLR']}"
    assert cfg.train.early_stopping in [True, False], "Early stopping must be a boolean"
    assert not cfg.train.early_stopping or cfg.train.patience is not None, "Patience must be set if early stopping is enabled"
    assert cfg.model.classifier in MODEL_NAMES, f"Model must be one of {MODEL_NAMES}"
    assert cfg.data.random_flip in [True, False], "Random flip must be a boolean"
    assert cfg.eval.mode in ['val', 'test'], f"Evaluation mode must be one of {['val', 'test']}"
    assert cfg.wandb.project is not None, "Wandb project must be specified"
    assert validate_channels(cfg.data.channels), "Channels must be a binary string of length 11"

def main(cfg):
    validate_config(cfg)
    run_experiment_s2(cfg)

if __name__ == '__main__':
    parser = argparse.ArgumentParser(prog='train_classifier', description='Trains classifier model from patches. The classifier inputs a patch with n channels and outputs a binary patch with water pixels labeled 1.')

    # YAML config file
    parser.add_argument("--config_file", default="configs/s2_template.yaml", help="Path to YAML config file (default: configs/s2_template.yaml)")

    # wandb
    parser.add_argument('--project', help='Wandb project where run will be logged')
    parser.add_argument('--group', help='Optional group name for model experiments (default: None)')
    parser.add_argument('--num_sample_predictions', type=int, help='number of predictions to visualize (default: 40)')

    # evaluation
    parser.add_argument('--mode', choices=['val', 'test'], help=f"dataset used for evaluation metrics (default: val)")

    # ml
    parser.add_argument('-e', '--epochs', type=int)
    parser.add_argument('-b', '--batch_size', type=int)
    parser.add_argument('-l', '--lr', type=float)
    parser.add_argument('-p', '--patience', type=int)

    # model
    parser.add_argument('--classifier', choices=MODEL_NAMES, help=f"models: {', '.join(MODEL_NAMES)}")
    # unet
    parser.add_argument('--dropout', type=float)

    # data loading
    parser.add_argument('--num_workers', type=int)

    # loss
    parser.add_argument('--loss', choices=LOSS_NAMES,
                        help=f"loss: {', '.join(LOSS_NAMES)}")

    # optimizer
    parser.add_argument('--optimizer', choices=['Adam', 'SGD'],
                        help=f"optimizer: {', '.join(['Adam', 'SGD'])}")

    # reproducibility
    parser.add_argument('--seed', type=int)

    _args = parser.parse_args()
    cfg = Config(**_args.__dict__)
    sys.exit(main(cfg))
