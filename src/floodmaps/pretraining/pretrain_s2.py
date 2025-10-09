import wandb
import torch
import logging
from torch import nn
from torch.utils.data import DataLoader
from datetime import datetime
from torchmetrics import MetricCollection
from torchmetrics.classification import BinaryAccuracy, BinaryPrecision, BinaryRecall, BinaryF1Score, BinaryConfusionMatrix, BinaryJaccardIndex
from matplotlib.cm import ScalarMappable
from matplotlib.colors import Normalize
import random
from random import Random
from PIL import Image
import numpy as np
from pathlib import Path
import json
from omegaconf import DictConfig, OmegaConf
import hydra

from floodmaps.models.model import S2WaterDetector
from floodmaps.utils.utils import flatten_dict, get_model_params, Metrics, EarlyStopper
from floodmaps.utils.checkpoint import save_checkpoint, load_checkpoint

from floodmaps.training.loss import BCEDiceLoss, TverskyLoss
from floodmaps.training.optim import get_optimizer
from floodmaps.training.scheduler import get_scheduler

from floodmaps.pretraining.dataset  import WorldFloodsS2Dataset
from floodmaps.pretraining.utils import WFChannelIndexer, wf_get_samples_with_wet_percentage
import albumentations as A

# TO IMPLEMENT: WITH DISCRIMINATOR, ADDITIONAL TRACKING OF DISCRIMINATOR OUTPUTS
# COULD ALSO CONSIDER REMOVING DISCRIMINATOR, ONLY ATTACHING FOR TEST SET EVALUATION

MODEL_NAMES = ['unet', 'unet++']
LOSS_NAMES = ['BCELoss', 'BCEDiceLoss', 'TverskyLoss']

def get_loss_fn(cfg, device=None, pos_weight=None):
    if cfg.train.loss == 'BCELoss':
        if pos_weight is not None:
            pw = torch.tensor(float(pos_weight), device=device) if device is not None else torch.tensor(float(pos_weight))
            loss_fn = nn.BCEWithLogitsLoss(pos_weight=pw)
        else:
            loss_fn = nn.BCEWithLogitsLoss()
    elif cfg.train.loss == 'BCEDiceLoss':
        # pass pos_weight into BCEDiceLoss BCE component if enabled
        if pos_weight is not None:
            pw = torch.tensor(float(pos_weight), device=device) if device is not None else torch.tensor(float(pos_weight))
            loss_fn = BCEDiceLoss(pos_weight=pw)
        else:
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
        BinaryF1Score(threshold=0.5),
        BinaryJaccardIndex(threshold=0.5)
    ]).to(device)

    model.train()
    for X, y in dataloader:
        X = X.to(device)
        y = y.to(device)

        logits = model(X)
        loss = loss_fn(logits, y.float())
        
        optimizer.zero_grad()
        loss.backward()
        optimizer.step()

        y_pred = nn.functional.sigmoid(logits).flatten()
        target = y.flatten()
        metric_collection.update(y_pred, target)
        running_loss += loss.detach()

    # calculate metrics
    metric_results = metric_collection.compute()
    epoch_loss = running_loss.item() / len(dataloader)

    # wandb tracking loss and metrics per epoch - track recons loss as well
    log_dict = {"train accuracy": metric_results['BinaryAccuracy'].item(),
                "train precision": metric_results['BinaryPrecision'].item(),
                "train recall": metric_results['BinaryRecall'].item(),
                "train f1": metric_results['BinaryF1Score'].item(),
                "train IoU": metric_results['BinaryJaccardIndex'].item(),
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
        BinaryConfusionMatrix(threshold=0.5),
        BinaryJaccardIndex(threshold=0.5)
    ]).to(device)
    
    model.eval()
    with torch.no_grad():
        for X, y in dataloader:  # WorldFloods doesn't use supplementary data
            X = X.to(device)
            y = y.to(device)

            logits = model(X)
            loss = loss_fn(logits, y.float())
            
            y_pred = nn.functional.sigmoid(logits).flatten()
            target = y.flatten()
            metric_collection.update(y_pred, target)
            running_vloss += loss.detach()

    # calculate metrics
    metric_results = metric_collection.compute()
    epoch_vloss = running_vloss.item() / len(dataloader)

    core_metrics_dict = {
        f"{typ} accuracy": metric_results['BinaryAccuracy'].item(),
        f"{typ} precision": metric_results['BinaryPrecision'].item(),
        f"{typ} recall": metric_results['BinaryRecall'].item(),
        f"{typ} f1": metric_results['BinaryF1Score'].item(),
        f"{typ} IoU": metric_results['BinaryJaccardIndex'].item()
    }
    
    # Only log to wandb for validation (not for test evaluation)
    if typ == 'val' and run is not None:
        log_dict = core_metrics_dict.copy()
        log_dict.update({f'{typ} loss': epoch_vloss})
        run.log(log_dict, step=epoch)

    # calculate confusion matrix (skip NLCD metrics - not available in WorldFloods)
    confusion_matrix = metric_results['BinaryConfusionMatrix'].tolist()
    confusion_matrix_dict = {
        "tn": confusion_matrix[0][0],
        "fp": confusion_matrix[0][1],
        "fn": confusion_matrix[1][0],
        "tp": confusion_matrix[1][1]
    }
    metric_collection.reset()

    # DEBUG PRINT CONFUSION MATRIX:
    print(f"Confusion matrix epoch {epoch}: {confusion_matrix_dict}")

    # separate the core loggable metrics from the nested dictionaries
    # for easier management downstream
    metrics_dict = {
        'core metrics': core_metrics_dict,
        'confusion matrix': confusion_matrix_dict
    }

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

    # compute pos_weight if enabled for rebalancing
    pos_weight_val = None
    if getattr(cfg.train, 'use_pos_weight', False):
        if getattr(cfg.train, 'pos_weight', None) is not None:
            pos_weight_val = float(cfg.train.pos_weight)
        else:
            # Efficient vectorized computation over loaded training labels
            # label plane is last channel in dataset (kept in-memory by dataset class)
            logging.info("Computing pos_weight from training labels")
            label_np = train_loader.dataset.dataset[:, -1, :, :]
            pos = float(label_np.sum())
            total = float(label_np.size)
            neg = max(total - pos, 1.0)
            # raw pos_weight = neg/pos; clip to [1, clip]
            raw_pw = neg / max(pos, 1.0)
            clip_max = float(getattr(cfg.train, 'pos_weight_clip', 10.0))
            pos_weight_val = max(1.0, min(raw_pw, clip_max))
            logging.info(f"Computed pos_weight: {pos_weight_val}")
            logging.info(f"Neg: {neg}, Pos: {pos}, Total: {total}")
            logging.info(f"Percentage of positive pixels: {pos / total:.2%}")

    # loss function
    loss_fn = get_loss_fn(cfg, device=device, pos_weight=pos_weight_val)

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
        run.summary.update({f'final model {key}': value for key, value in best_val_metrics['core metrics'].items()})
    else:
        cls_weights = model.classifier.state_dict()
        disc_weights = (model.discriminator.state_dict()
                      if model.uses_discriminator() else None)
        fmetrics.save_metrics('val', loss=avg_vloss, **val_set_metrics)
        run.summary.update({f'final model {key}': value for key, value in val_set_metrics['core metrics'].items()})

    # for benchmarking purposes
    if cfg.eval.mode == 'test':
        test_loss, test_set_metrics = test_loop(model, test_loader, device, loss_fn, None, None, typ='test')
        fmetrics.save_metrics('test', loss=test_loss, **test_set_metrics)
        run.summary.update({f'final model {key}': value for key, value in test_set_metrics['core metrics'].items()})

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
    with open(path / "config.yaml", "w") as f:
        OmegaConf.save(cfg, f)

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

def sample_predictions(model, sample_set, cfg, percent_wet_patches=0.5, seed=24330):
    """Generate predictions on a subset of images in the validation set for wandb logging.
    
    TODO: FIX CHANNEL INDEXING HARDCODING. Need more flexible way to handle channels.
    
    Parameters
    ----------
    model : torch.nn.Module
        The model to evaluate
    sample_set : torch.utils.data.Dataset
        The dataset to sample predictions from
    cfg : DictConfig
        The configuration dictionary
    percent_wet_patches : float, optional
        The percentage of wet patches to visualize
    seed : int, optional
        The seed for the random number generator
    """
    if cfg.wandb.num_sample_predictions <= 0:
        return None

    columns = ["id"]
    channels = [bool(int(x)) for x in cfg.data.channels]
    my_channels = WFChannelIndexer(channels) # ndwi, dem, slope_y, slope_x, waterbody, roads, flowlines
    # initialize wandb table given the channel settings
    columns += my_channels.get_display_channels()
    columns += ["truth", "prediction", "false positive", "false negative"] # added residual binary map
    table = wandb.Table(columns=columns)

    def to_rgb_image_gamma_corrected(s2_rgb, gamma=1/2.2):
        """Stretch Sentinel-2 RGB using percentile normalization."""
        rgb = np.clip(s2_rgb, 0, 1)
        rgb = np.power(rgb, gamma)
        return (rgb * 255).astype(np.uint8)

    if my_channels.has_nir():
        # initialize mappable objects
        nir_norm = Normalize(vmin=0, vmax=1)
        nir_map = ScalarMappable(norm=nir_norm, cmap='gray')
    
    if my_channels.has_ndwi():
        # initialize mappable objects
        ndwi_norm = Normalize(vmin=-1, vmax=1)
        ndwi_map = ScalarMappable(norm=ndwi_norm, cmap='seismic_r')
    
    # Build mapping from original channel index to position in filtered tensor
    n = 0
    channel_indices = [-1] * 5
    for i, channel in enumerate(channels):
        if channel:
            channel_indices[i] = n
            n += 1
    
    model.to('cpu')
    model.eval()
    rng = Random(seed)
    
    # Get samples with specified percentage of wet patches
    samples = wf_get_samples_with_wet_percentage(
        sample_set,
        cfg.wandb.num_sample_predictions,
        cfg.train.batch_size,
        cfg.train.num_workers,
        percent_wet_patches,
        rng
    )

    for id, k in enumerate(samples):
        # get all images to shape (H, W, C) with C = 1 or 3 (1 for grayscale, 3 for RGB)
        X, y = sample_set[k]  # supplementary is dummy data for WorldFloods
        
        with torch.no_grad():
            logits = model(X.unsqueeze(0))
            
        y_pred = torch.where(nn.functional.sigmoid(logits) > 0.5, 1.0, 0.0).squeeze(0) # (1, x, y)

        # Compute false positives
        fp = torch.logical_and(y == 0, y_pred == 1).squeeze(0).byte().mul(255).numpy()
        
        # Compute false negatives
        fn = torch.logical_and(y == 1, y_pred == 0).squeeze(0).byte().mul(255).numpy()

        # Channels are descaled using linear variance scaling
        X = X.permute(1, 2, 0)
        row = [k]

        if my_channels.has_rgb():
            rgb_arr = to_rgb_image_gamma_corrected(X[:, :, :3].numpy())
            rgb_img = Image.fromarray(rgb_arr, mode="RGB")
            row.append(wandb.Image(rgb_img))

        if my_channels.has_nir():
            nir_idx = channel_indices[3]
            if nir_idx == -1:
                raise ValueError("NIR channel requested but not available in filtered tensor")
            nir = X[:, :, nir_idx]
            nir = nir_map.to_rgba(nir.numpy(), bytes=True)
            nir = np.clip(nir, 0, 255).astype(np.uint8)
            nir_img = Image.fromarray(nir, mode="RGBA")
            row.append(wandb.Image(nir_img))

        if my_channels.has_ndwi():
            ndwi_idx = channel_indices[4]
            if ndwi_idx == -1:
                raise ValueError("NDWI channel requested but not available in filtered tensor")
            ndwi = X[:, :, ndwi_idx]
            ndwi = ndwi_map.to_rgba(ndwi.numpy(), bytes=True)
            ndwi = np.clip(ndwi, 0, 255).astype(np.uint8)
            ndwi_img = Image.fromarray(ndwi, mode="RGBA")
            row.append(wandb.Image(ndwi_img))

        y = y.squeeze(0).mul(255).clamp(0, 255).byte().numpy()
        y_pred = y_pred.squeeze(0).mul(255).clamp(0, 255).byte().numpy()
        
        truth_img = Image.fromarray(y, mode="L")
        pred_img = Image.fromarray(y_pred, mode="L")
        fp_img = Image.fromarray(fp, mode="L")
        fn_img = Image.fromarray(fn, mode="L")
        row += [wandb.Image(truth_img), wandb.Image(pred_img), wandb.Image(fp_img), wandb.Image(fn_img)]

        table.add_data(*row)

    return table

def run_pretrain_s2(cfg):
    """Run a single S2 pretraining model experiment given the configuration parameters.
    
    Parameters
    ----------
    cfg : object
        Config object for the S2 pretraining model.

    Returns
    -------
    fmetrics : Metrics
        Metrics object containing the metrics for the experiment.
    """
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
    sample_dir = Path(cfg.paths.preprocess_dir) / 'worldfloodsv2' / '50_clouds' # temporary name

    channels = [bool(int(x)) for x in cfg.data.channels]
    train_transform = A.Compose([
        A.HorizontalFlip(p=0.5),
        A.VerticalFlip(p=0.5),
        A.RandomRotate90(p=0.5),
    ], seed=cfg.seed)

    # datasets
    train_set = WorldFloodsS2Dataset(sample_dir, channels=channels,
                                        typ="train", transform=train_transform)
    val_set = WorldFloodsS2Dataset(sample_dir, channels=channels,
                                    typ="val", transform=None)
    test_set = WorldFloodsS2Dataset(sample_dir, channels=channels,
                                        typ="test", transform=None) if cfg.eval.mode == 'test' else None

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
                            shuffle=False,
                            drop_last=False)

    test_loader = DataLoader(test_set,
                            batch_size=cfg.train.batch_size,
                            num_workers=cfg.train.num_workers,
                            persistent_workers=cfg.train.num_workers>0,
                            pin_memory=True,
                            shuffle=False,
                            drop_last=False) if cfg.eval.mode == 'test' else None

    # initialize wandb run
    total_params, trainable_params, param_size_in_mb = get_model_params(model)

    # convert config to flat dict for logging
    config_dict = flatten_dict(OmegaConf.to_container(cfg, resolve=True))
    run = wandb.init(
        project=cfg.wandb.project,
        group=cfg.wandb.group,
        config={
            "dataset": "WorldFloodsS2",
            **config_dict,
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
                cfg.save_path = str(Path(cfg.paths.experiment_dir) / f"{datetime.today().strftime('%Y-%m-%d')}_{cfg.model.classifier}_{run.id}/")
                run.config.update({"save_path": cfg.save_path}, allow_val_change=True)
            print(f'Save path set to: {cfg.save_path}')
        
        # train and save results metrics
        cls_weights, disc_weights, fmetrics = train(model, train_loader, val_loader,
                                              test_loader, device, cfg, run)
        if cfg.save:
            save_experiment(cls_weights, disc_weights, fmetrics, cfg, run)

        # log predictions on validation set using wandb
        percent_wet = cfg.wandb.get('percent_wet_patches', 0.5)  # Default to 0.5 if not specified
        pred_table = sample_predictions(model, test_set if cfg.eval.mode == 'test' else val_set, cfg, percent_wet_patches=percent_wet)
        run.log({f"model_{cfg.eval.mode}_predictions": pred_table})
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
        return type(s) == str and len(s) == 5 and all(c in '01' for c in s)

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
    assert cfg.eval.mode in ['val', 'test'], f"Evaluation mode must be one of {['val', 'test']}"
    assert cfg.wandb.project is not None, "Wandb project must be specified"
    assert validate_channels(cfg.data.channels), "Channels must be a binary string of length 5"

@hydra.main(version_base=None, config_path="pkg://configs", config_name="config.yaml")
def main(cfg: DictConfig):
    run_pretrain_s2(cfg)

if __name__ == '__main__':
    main()