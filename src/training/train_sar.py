import wandb
import torch
import logging
import argparse
import copy
from datetime import datetime
from torch import nn
from torch.utils.data import DataLoader
from torchvision import transforms
from torchmetrics import MetricCollection
from torchmetrics.classification import BinaryAccuracy, BinaryPrecision, BinaryRecall, BinaryF1Score
from matplotlib.cm import ScalarMappable
from matplotlib.colors import Normalize
import random
from random import Random
from PIL import Image
from glob import glob
import numpy as np
import sys
import pickle
import json
from pathlib import Path

from models.model import SARWaterDetector
from utils.config import Config
from utils.utils import (SRC_DIR, DATA_DIR, RESULTS_DIR, Metrics, EarlyStopper,
                         SARChannelIndexer, get_model_params)
from training.loss import LossConfig
from training.dataset import FloodSampleSARDataset
from training.optim import get_optimizer
from training.scheduler import get_scheduler

MODEL_NAMES = ['unet', 'unet++']
AUTODESPECKLER_NAMES = ['CNN1', 'CNN2', 'DAE', 'VAE']
NOISE_NAMES = ['normal', 'masking', 'log_gamma']
LOSS_NAMES = ['BCELoss', 'BCEDiceLoss', 'TverskyLoss']

# get our optimizer and metrics
def train_loop(model, dataloader, device, optimizer, minibatches, loss_config,
                ad_cfg, c, run, epoch):
    running_tot_loss = torch.tensor(0.0, device=device) # all loss components
    running_cls_loss = torch.tensor(0.0, device=device) # only classifier loss
    if ad_cfg is not None:
        running_recons_loss = torch.tensor(0.0, device=device)

        # for VAE monitoring
        if ad_cfg.model.autodespeckler == 'VAE':
            running_kld_loss = torch.tensor(0.0, device=device)
            all_mu = []
            all_log_var = []

    metric_collection = MetricCollection([
        BinaryAccuracy(threshold=0.5),
        BinaryPrecision(threshold=0.5),
        BinaryRecall(threshold=0.5),
        BinaryF1Score(threshold=0.5)
    ]).to(device)
    all_preds = []
    all_targets = []

    model.train()
    for batch_i, (X, y) in enumerate(dataloader):
        X = X.to(device)
        y = y.to(device)

        # crop central window
        X_c = X[:, :, c[0]:c[1], c[0]:c[1]]
        out_dict = model(X_c)

        # also pass SAR layers for reconstruction loss
        loss_dict = loss_config.compute_loss(out_dict, y.float(), typ='train')
        loss = loss_dict['total_loss']
        y_true = loss_dict['true_label']

        optimizer.zero_grad()
        loss.backward()
        optimizer.step()

        logits = out_dict['classifier_output']
        y_pred = nn.functional.sigmoid(logits).flatten() > 0.5
        target = y_true.flatten() > 0.5

        all_preds.append(y_pred)
        all_targets.append(target)
        running_tot_loss += loss.detach()
        running_cls_loss += loss_dict['classifier_loss'].detach()

        if ad_cfg is not None:
            running_recons_loss += loss_dict['recons_loss'].detach()

            # for VAE monitoring only
            if ad_cfg.model.autodespeckler == 'VAE':
                # Collect mu and log_var for the whole epoch
                all_mu.append(out_dict['mu'].detach().cpu())
                all_log_var.append(out_dict['log_var'].detach().cpu())
                running_kld_loss += loss_dict['kld_loss'].detach()

        if batch_i >= minibatches:
            break

    # calculate metrics
    all_preds = torch.cat(all_preds)
    all_targets = torch.cat(all_targets)

    metric_collection.update(all_preds, all_targets)
    metric_results = metric_collection.compute()
    epoch_loss = running_tot_loss.item() / minibatches

    # wandb tracking loss and metrics per epoch - track recons loss as well
    log_dict = {"train accuracy": metric_results['BinaryAccuracy'].item(),
                "train precision": metric_results['BinaryPrecision'].item(),
                "train recall": metric_results['BinaryRecall'].item(),
                "train f1": metric_results['BinaryF1Score'].item(),
                "train tot loss": epoch_loss,
                "train cls loss": running_cls_loss.item() / minibatches}

    # autodespeckler monitoring
    if ad_cfg is not None:
        log_dict['train_recons_loss'] = cfg.train.balance_coeff * running_recons_loss.item() / minibatches

        epoch_ad_loss = (
            running_recons_loss.item() + ad_cfg.model.vae.VAE_beta * running_kld_loss.item()
            if ad_cfg.model.autodespeckler == 'VAE' else running_recons_loss.item()
        )
        log_dict['train_ad_loss'] = cfg.train.balance_coeff * epoch_ad_loss / minibatches

        # ad loss percentage of total loss
        ad_loss_percentage = (cfg.train.balance_coeff * epoch_ad_loss / running_tot_loss)
        log_dict['ad_loss_percentage'] = ad_loss_percentage

        # VAE mu and log_var monitoring
        if ad_cfg.model.autodespeckler == 'VAE':
            all_mu = torch.cat(all_mu, dim=0).numpy()
            all_log_var = torch.cat(all_log_var, dim=0).numpy()

            # kld loss as percentage of total ad loss
            kld_loss_percentage = (ad_cfg.model.vae.VAE_beta
                                   * running_kld_loss.item()
                                   / epoch_ad_loss)
            log_dict.update({
                "train_mu_mean": all_mu.mean(),
                "train_mu_std": all_mu.std(),
                "train_log_var_mean": all_log_var.mean(),
                "train_log_var_std": all_log_var.std(),
                "train_kld_loss": cfg.train.balance_coeff * running_kld_loss.item() / minibatches,
                "train_kld_loss_percentage": kld_loss_percentage,
                "beta": ad_cfg.model.vae.VAE_beta
            })
    run.log(log_dict, step=epoch)
    metric_collection.reset()

    return epoch_loss

def test_loop(model, dataloader, device, loss_config, ad_cfg, c, run, epoch):
    running_tot_vloss = torch.tensor(0.0, device=device) # all loss components
    running_cls_vloss = torch.tensor(0.0, device=device) # only classifier loss
    if ad_cfg is not None:
        running_recons_vloss = torch.tensor(0.0, device=device)

        # for VAE monitoring
        if ad_cfg.model.autodespeckler == 'VAE':
            running_kld_vloss = torch.tensor(0.0, device=device)
            all_mu = []
            all_log_var = []

    num_batches = len(dataloader)
    metric_collection = MetricCollection([
        BinaryAccuracy(threshold=0.5),
        BinaryPrecision(threshold=0.5),
        BinaryRecall(threshold=0.5),
        BinaryF1Score(threshold=0.5)
    ]).to(device)
    all_preds = []
    all_targets = []

    model.eval()
    with torch.no_grad():
        for X, y in dataloader:
            X = X.to(device)
            y = y.to(device)

            X_c = X[:, :, c[0]:c[1], c[0]:c[1]]

            out_dict = model(X_c)
            loss_dict = loss_config.compute_loss(out_dict, y.float(), typ='val')
            loss = loss_dict['total_loss']
            y_true = loss_dict['true_label']

            logits = out_dict['classifier_output']
            y_pred = nn.functional.sigmoid(logits).flatten() > 0.5
            target = y_true.flatten() > 0.5

            all_preds.append(y_pred)
            all_targets.append(target)
            running_tot_vloss += loss.detach()
            running_cls_vloss += loss_dict['classifier_loss'].detach()

            if ad_cfg is not None:
                running_recons_vloss += loss_dict['recons_loss'].detach()

                # for VAE monitoring only
                if ad_cfg.model.autodespeckler == 'VAE':
                    # Collect mu and log_var for the whole epoch
                    all_mu.append(out_dict['mu'].detach().cpu())
                    all_log_var.append(out_dict['log_var'].detach().cpu())
                    running_kld_vloss += loss_dict['kld_loss'].detach()

    # calculate metrics
    all_preds = torch.cat(all_preds)
    all_targets = torch.cat(all_targets)

    metric_collection.update(all_preds, all_targets)
    metric_results = metric_collection.compute()
    epoch_vloss = running_tot_vloss.item() / num_batches
    epoch_cls_vloss = running_cls_vloss.item() / num_batches

    metrics_dict = {
        "val accuracy": metric_results['BinaryAccuracy'].item(),
        "val precision": metric_results['BinaryPrecision'].item(),
        "val recall": metric_results['BinaryRecall'].item(),
        "val f1": metric_results['BinaryF1Score'].item()
    }
    log_dict = metrics_dict.copy()
    log_dict.update({
        "val tot loss": epoch_vloss,
        "val cls loss": epoch_cls_vloss
    })

    # autodespeckler monitoring
    if ad_cfg is not None:
        log_dict['val_recons_loss'] = cfg.train.balance_coeff * running_recons_vloss.item() / num_batches

        epoch_ad_vloss = (
            running_recons_vloss.item() + ad_cfg.model.vae.VAE_beta * running_kld_vloss.item()
            if ad_cfg.model.autodespeckler == 'VAE' else running_recons_vloss.item()
        )
        log_dict['val_ad_loss'] = cfg.train.balance_coeff * epoch_ad_vloss / num_batches

        # ad loss percentage of total loss
        ad_loss_percentage = (cfg.train.balance_coeff * epoch_ad_vloss / running_tot_vloss)
        log_dict['val_ad_loss_percentage'] = ad_loss_percentage

        # VAE mu and log_var monitoring
        if ad_cfg.model.autodespeckler == 'VAE':
            all_mu = torch.cat(all_mu, dim=0).numpy()
            all_log_var = torch.cat(all_log_var, dim=0).numpy()

            # kld loss as percentage of total ad loss
            kld_loss_percentage = (ad_cfg.model.vae.VAE_beta
                                   * running_kld_vloss.item()
                                   / epoch_ad_vloss)
            log_dict.update({
                "val_mu_mean": all_mu.mean(),
                "val_mu_std": all_mu.std(),
                "val_log_var_mean": all_log_var.mean(),
                "val_log_var_std": all_log_var.std(),
                "val_kld_loss": cfg.train.balance_coeff * running_kld_vloss.item() / num_batches,
                "val_kld_loss_percentage": kld_loss_percentage
            })

    run.log(log_dict, step=epoch)
    metric_collection.reset()

    return epoch_vloss, epoch_cls_vloss, metrics_dict

def evaluate(model, dataloader, device, loss_config, ad_cfg, c):
    """Evaluate metrics on test set without logging."""
    running_tot_vloss = torch.tensor(0.0, device=device)

    num_batches = len(dataloader)
    metric_collection = MetricCollection([
        BinaryAccuracy(threshold=0.5),
        BinaryPrecision(threshold=0.5),
        BinaryRecall(threshold=0.5),
        BinaryF1Score(threshold=0.5)
    ]).to(device)
    all_preds = []
    all_targets = []

    model.eval()
    with torch.no_grad():
        for X, y in dataloader:
            X = X.to(device)
            y = y.to(device)

            X_c = X[:, :, c[0]:c[1], c[0]:c[1]]

            out_dict = model(X_c)
            loss_dict = loss_config.compute_loss(out_dict, y.float(), typ='val')
            loss = loss_dict['total_loss']
            y_true = loss_dict['true_label']

            logits = out_dict['classifier_output']
            y_pred = nn.functional.sigmoid(logits).flatten() > 0.5
            target = y_true.flatten() > 0.5

            all_preds.append(y_pred)
            all_targets.append(target)
            running_tot_vloss += loss.detach()

    # calculate metrics
    all_preds = torch.cat(all_preds)
    all_targets = torch.cat(all_targets)

    metric_collection.update(all_preds, all_targets)
    metric_results = metric_collection.compute()
    epoch_vloss = running_tot_vloss.item() / num_batches

    metrics_dict = {
        "test accuracy": metric_results['BinaryAccuracy'].item(),
        "test precision": metric_results['BinaryPrecision'].item(),
        "test recall": metric_results['BinaryRecall'].item(),
        "test f1": metric_results['BinaryF1Score'].item()
    }
    metric_collection.reset()

    return epoch_vloss, metrics_dict

def train(model, train_loader, val_loader, test_loader, device, loss_cfg, cfg, ad_cfg, run):
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

    # optimizer and scheduler for reducing learning rate
    optimizer = get_optimizer(model, cfg)
    scheduler = get_scheduler(optimizer, cfg)

    ignore_ad_loss = (ad_cfg is not None
                and cfg.model.autodespeckler.freeze
                and cfg.model.autodespeckler.freeze_epochs >= cfg.train.epochs)
    if cfg.train.early_stopping:
        early_stopper = EarlyStopper(patience=cfg.train.patience)

    minibatches = int(len(train_loader) * cfg.train.subset)
    center_1 = (cfg.data.size - cfg.data.window) // 2
    center_2 = center_1 + cfg.data.window
    c = (center_1, center_2)
    for epoch in range(cfg.train.epochs):
        try:
            if (ad_cfg is not None
                and cfg.model.autodespeckler.freeze
                and epoch == cfg.model.autodespeckler.freeze_epochs):
                print(f"Unfreezing backbone at epoch {epoch}")
                model.unfreeze_ad_weights()

            # train loop
            avg_loss = train_loop(model, train_loader, device, optimizer, minibatches,
                                  loss_cfg, ad_cfg, c, run, epoch)

            # at the end of each training epoch compute validation
            avg_vloss, avg_cls_vloss, val_set_metrics = test_loop(model, val_loader, device, loss_cfg,
                                                   ad_cfg, c, run, epoch)
        except Exception as err:
            raise RuntimeError(f'Error while training occurred at epoch {epoch}.') from err

        if cfg.train.early_stopping:
            # use only classifier component for early stopping if AD is frozen
            early_stopper.step(avg_cls_vloss if ignore_ad_loss else avg_vloss, model)
            early_stopper.store_best_metrics(val_set_metrics)
            if early_stopper.is_stopped():
                break

        if scheduler is not None:
            if isinstance(scheduler, torch.optim.lr_scheduler.ReduceLROnPlateau):
                scheduler.step(avg_vloss)
            else:
                scheduler.step()

        run.log({"learning_rate": scheduler.get_last_lr()[0] if scheduler is not None else cfg.train.lr}, step=epoch)

    # Save our model
    fmetrics = Metrics(use_partitions=True)
    cls_weights = None
    partition = 'shift_invariant' if cfg.train.shift_invariant else 'non_shift_invariant'
    if cfg.train.early_stopping:
        model_weights = early_stopper.get_best_weights()
        best_val_metrics = early_stopper.get_best_metrics()
        # reset model to checkpoint for later sample prediction
        model.load_state_dict(model_weights)
        cls_weights = model.classifier.state_dict()
        ad_weights = (model.autodespeckler.state_dict()
                      if model.uses_autodespeckler() else None)
        fmetrics.save_metrics('val', partition=partition,
                              loss=early_stopper.get_min_validation_loss(),
                              **best_val_metrics)
        run.summary.update({f'final model {key}': value for key, value in best_val_metrics.items()})
    else:
        cls_weights = model.classifier.state_dict()
        ad_weights = (model.autodespeckler.state_dict()
                      if model.uses_autodespeckler() else None)
        fmetrics.save_metrics('val', partition=partition,
                              loss=avg_vloss,
                              **val_set_metrics)
        run.summary.update({f'final model {key}': value for key, value in val_set_metrics.items()})

    # for benchmarking purposes
    if cfg.eval.mode == 'test':
        test_loss, test_set_metrics = evaluate(model, test_loader, device, loss_cfg, ad_cfg, c)
        fmetrics.save_metrics('test', partition=partition, loss=test_loss, **test_set_metrics)
        run.summary.update({f'final model {key}': value for key, value in test_set_metrics.items()})

    return cls_weights, ad_weights, fmetrics

def save_experiment(cls_weights, ad_weights, metrics, cfg, ad_cfg, run):
    """Save experiment files to directory specified by config save_path."""
    path = Path(cfg.save_path)
    path.mkdir(parents=True, exist_ok=True)

    if cls_weights is not None:
        torch.save(cls_weights, path / f"{cfg.model.classifier}_cls.pth")
    if ad_weights is not None:
        torch.save(ad_weights, path / f"{ad_cfg.model.autodespeckler}_ad.pth")

    # save config
    cfg.save2yaml(path / "config.yaml")
    if ad_cfg is not None:
        ad_cfg.save2yaml(path / "ad_config.yaml")

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

def sample_predictions(model, sample_set, mean, std, loss_config, cfg, seed=24330):
    """Generate predictions on a subset of images in the dataset for wandb logging."""
    if cfg.wandb.num_sample_predictions <= 0:
        return None

    loss_config.val_loss_fn.change_device('cpu')
    columns = ["id"]
    channels = [bool(int(x)) for x in cfg.data.channels]
    my_channels = SARChannelIndexer(channels)
    # initialize wandb table given the channel settings
    columns += my_channels.get_channel_names()
    if model.uses_autodespeckler():
        columns += ['despeckled_vv', 'despeckled_vh']
    columns += ["truth", "prediction", "false positive", "false negative"] # added residual binary map
    table = wandb.Table(columns=columns)

    if my_channels.has_vv():
        # initialize mappable objects
        vv_map = ScalarMappable(norm=None, cmap='gray')
    if my_channels.has_vh():
        vh_map = ScalarMappable(norm=None, cmap='gray')
    if my_channels.has_dem():
        dem_map = ScalarMappable(norm=None, cmap='gray')
    if my_channels.has_slope_y():
        slope_y_map = ScalarMappable(norm=None, cmap='RdBu')
    if my_channels.has_slope_x():
        slope_x_map = ScalarMappable(norm=None, cmap='RdBu')

    # get map of each channel to index of resulting tensor
    n = 0
    channel_indices = [-1] * 7
    for i, channel in enumerate(channels):
        if channel:
            channel_indices[i] = n
            n += 1

    model.to('cpu')
    model.eval()
    rng = Random(seed)
    samples = rng.sample(range(0, len(sample_set)), cfg.wandb.num_sample_predictions)

    center_1 = (cfg.data.size - cfg.data.window) // 2
    center_2 = center_1 + cfg.data.window
    for id, k in enumerate(samples):
        # get all images to shape (H, W, C) with C = 1 or 3 (1 for grayscale, 3 for RGB)
        X, y = sample_set[k]

        X_c = X[:, center_1:center_2, center_1:center_2]

        with torch.no_grad():
            out_dict = model(X_c.unsqueeze(0))
            logits = out_dict['classifier_output']
            despeckler_output = out_dict['despeckler_output'].squeeze(0) if model.uses_autodespeckler() else None
            y_shifted = loss_config.get_label_alignment(logits, y.unsqueeze(0).float()).squeeze(0)

        y_pred = torch.where(nn.functional.sigmoid(logits) > 0.5, 1.0, 0.0).squeeze(0) # (1, x, y)

        # Compute false positives
        fp = torch.logical_and(y_shifted == 0, y_pred == 1).squeeze(0).byte().mul(255).numpy()

        # Compute false negatives
        fn = torch.logical_and(y_shifted == 1, y_pred == 0).squeeze(0).byte().mul(255).numpy()

        # Channels are descaled using linear variance scaling
        X_c = X_c.permute(1, 2, 0)
        X_c = std * X_c + mean

        row = [k]
        if my_channels.has_vv():
            vv = X_c[:, :, channel_indices[0]].numpy()
            vv_map.set_norm(Normalize(vmin=np.min(vv), vmax=np.max(vv)))
            vv = vv_map.to_rgba(vv, bytes=True)
            vv = np.clip(vv, 0, 255).astype(np.uint8)
            vv_img = Image.fromarray(vv, mode="RGBA")
            row.append(wandb.Image(vv_img))
        if my_channels.has_vh():
            vh = X_c[:, :, channel_indices[1]].numpy()
            vh_map.set_norm(Normalize(vmin=np.min(vh), vmax=np.max(vh)))
            vh = vh_map.to_rgba(vh, bytes=True)
            vh = np.clip(vh, 0, 255).astype(np.uint8)
            vh_img = Image.fromarray(vh, mode="RGBA")
            row.append(wandb.Image(vh_img))
        if my_channels.has_dem():
            dem = X_c[:, :, channel_indices[2]].numpy()
            dem_map.set_norm(Normalize(vmin=np.min(dem), vmax=np.max(dem)))
            dem = dem_map.to_rgba(dem, bytes=True)
            dem = np.clip(dem, 0, 255).astype(np.uint8)
            dem_img = Image.fromarray(dem, mode="RGBA")
            row.append(wandb.Image(dem_img))
        if my_channels.has_slope_y():
            slope_y = X_c[:, :, channel_indices[3]].numpy()
            slope_y_map.set_norm(Normalize(vmin=np.min(slope_y), vmax=np.max(slope_y)))
            slope_y = slope_y_map.to_rgba(slope_y, bytes=True)
            slope_y = np.clip(slope_y, 0, 255).astype(np.uint8)
            slope_y_img = Image.fromarray(slope_y, mode="RGBA")
            row.append(wandb.Image(slope_y_img))
        if my_channels.has_slope_x():
            slope_x = X_c[:, :, channel_indices[4]].numpy()
            slope_x_map.set_norm(Normalize(vmin=np.min(slope_x), vmax=np.max(slope_x)))
            slope_x = slope_x_map.to_rgba(slope_x, bytes=True)
            slope_x = np.clip(slope_x, 0, 255).astype(np.uint8)
            slope_x_img = Image.fromarray(slope_x, mode="RGBA")
            row.append(wandb.Image(slope_x_img))
        if my_channels.has_waterbody():
            waterbody = X_c[:, :, channel_indices[5]].mul(255).clamp(0, 255).byte().numpy()
            waterbody_img = Image.fromarray(waterbody, mode="L")
            row.append(wandb.Image(waterbody_img))
        if my_channels.has_roads():
            roads = X_c[:, :, channel_indices[6]].mul(255).clamp(0, 255).byte().numpy()
            roads_img = Image.fromarray(roads, mode="L")
            row.append(wandb.Image(roads_img))
        if model.uses_autodespeckler():
            # add reconstruction VV and VH
            recons_vv = despeckler_output[0].numpy()
            vv_map.set_norm(Normalize(vmin=np.min(recons_vv), vmax=np.max(recons_vv)))
            recons_vv = vv_map.to_rgba(recons_vv, bytes=True)
            recons_vv = np.clip(recons_vv, 0, 255).astype(np.uint8)
            recons_vv_img = Image.fromarray(recons_vv, mode="RGBA")
            row.append(wandb.Image(recons_vv_img))

            recons_vh = despeckler_output[1].numpy()
            vh_map.set_norm(Normalize(vmin=np.min(recons_vh), vmax=np.max(recons_vh)))
            recons_vh = vh_map.to_rgba(recons_vh, bytes=True)
            recons_vh = np.clip(recons_vh, 0, 255).astype(np.uint8)
            recons_vh_img = Image.fromarray(recons_vh, mode="RGBA")
            row.append(wandb.Image(recons_vh_img))

        y_shifted = y_shifted.squeeze(0).mul(255).clamp(0, 255).byte().numpy()
        y_pred = y_pred.squeeze(0).mul(255).clamp(0, 255).byte().numpy()

        truth_img = Image.fromarray(y_shifted, mode="L")
        pred_img = Image.fromarray(y_pred, mode="L")
        fp_img = Image.fromarray(fp, mode="L")
        fn_img = Image.fromarray(fn, mode="L")
        row += [wandb.Image(truth_img), wandb.Image(pred_img), wandb.Image(fp_img), wandb.Image(fn_img)]

        table.add_data(*row)

    return table

def run_experiment_s1(cfg, ad_cfg=None):
    """Run a single S1 SAR model experiment given the configuration parameters.

    Parameters
    ----------
    cfg : object
        Config object for the SAR classifier.
    ad_cfg : object, optional
        Config object for an optional SAR autodespeckler attachment.
    """
    if not wandb.login():
        raise Exception("Failed to login to wandb.")

    # seeding
    np.random.seed(cfg.seed)
    random.seed(cfg.seed)
    torch.manual_seed(cfg.seed)

    device = (
        "cuda"
        if torch.cuda.is_available()
        else "mps"
        if torch.backends.mps.is_available()
        else "cpu"
    )

    # setup model
    model = SARWaterDetector(cfg, ad_cfg=ad_cfg).to(device)

    # freeze AD weights
    if ad_cfg is not None and cfg.model.autodespeckler.freeze:
        model.freeze_ad_weights()

    # dataset and transforms
    print(f"Using {device} device")
    model_name = cfg.model.classifier
    filter = 'lee' if cfg.data.use_lee else 'raw'
    size = cfg.data.size
    samples = cfg.data.samples
    sample_dir = DATA_DIR / 'sar' / f'samples_{size}_{samples}_{filter}/'

    # load in mean and std
    channels = [bool(int(x)) for x in cfg.data.channels]
    b_channels = sum(channels[-2:])
    with open(sample_dir / f'mean_std_{size}_{samples}_{filter}.pkl', 'rb') as f:
        train_mean, train_std = pickle.load(f)

        train_mean = torch.from_numpy(train_mean[channels])
        train_std = torch.from_numpy(train_std[channels])
        # make sure binary channels are 0 mean and 1 std
        if b_channels > 0:
            train_mean[-b_channels:] = 0
            train_std[-b_channels:] = 1
    standardize = transforms.Compose([transforms.Normalize(train_mean, train_std)])

    # datasets
    train_set = FloodSampleSARDataset(sample_dir, channels=channels,
                                        typ="train", transform=standardize, random_flip=cfg.data.random_flip,
                                        seed=cfg.seed+1)
    val_set = FloodSampleSARDataset(sample_dir, channels=channels,
                                    typ="val", transform=standardize)
    test_set = FloodSampleSARDataset(sample_dir, channels=channels,
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
            "dataset": "Sentinel1",
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
                default_path = f"results/experiments/{datetime.today().strftime('%Y-%m-%d')}_{cfg.model.classifier}_{run.id}/"
                cfg.save_path = default_path
                run.config.update({"save_path": cfg.save_path}, allow_val_change=True)
            print(f'Save path set to: {cfg.save_path}')

        # initialize loss functions - train loss function is optimized for gradient calculations
        loss_cfg = LossConfig(cfg, ad_cfg=ad_cfg, device=device)

        # train and save results metrics
        cls_weights, ad_weights, fmetrics = train(model, train_loader, val_loader,
                                              test_loader, device, loss_cfg,
                                              cfg, ad_cfg, run)
        if cfg.save:
            save_experiment(cls_weights, ad_weights, fmetrics, cfg, ad_cfg, run)

        # log predictions on validation set using wandb
        pred_table = sample_predictions(model, test_set if cfg.eval.mode == 'test' else val_set,
                                        train_mean, train_std, loss_cfg, cfg)
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
        return type(s) == str and len(s) == 7 and all(c in '01' for c in s)

    # Add checks
    assert cfg.train.lr > 0, "Learning rate must be positive"
    assert cfg.model.classifier in MODEL_NAMES, f"Model must be one of {MODEL_NAMES}"
    assert cfg.train.loss in LOSS_NAMES, f"Loss must be one of {LOSS_NAMES}"
    assert cfg.train.optimizer in ['Adam', 'SGD'], f"Optimizer must be one of {['Adam', 'SGD']}"
    assert cfg.train.LR_scheduler in ['Constant', 'ReduceLROnPlateau', 'CosAnnealingLR'], f"LR scheduler must be one of {['Constant', 'ReduceLROnPlateau', 'CosAnnealingLR']}"
    assert cfg.train.early_stopping in [True, False], "Early stopping must be a boolean"
    assert not cfg.train.early_stopping or cfg.train.patience is not None, "Patience must be set if early stopping is enabled"
    assert cfg.train.random_flip in [True, False], "Random flip must be a boolean"
    assert cfg.train.save in [True, False], "Save must be a boolean"
    assert cfg.train.batch_size is not None and cfg.train.batch_size > 0, "Batch size must be defined and positive"
    assert cfg.eval.mode in ['val', 'test'], f"Evaluation mode must be one of {['val', 'test']}"
    assert cfg.wandb.project is not None, "Wandb project must be specified"
    assert validate_channels(cfg.data.channels), "Channels must be a binary string of length 7"

def main(cfg, ad_cfg):
    validate_config(cfg) # validate ad_cfg?
    run_experiment_s1(cfg, ad_cfg=ad_cfg)

if __name__ == '__main__':
    parser = argparse.ArgumentParser(prog='train_sar_classifier', description='Trains SAR classifier model from patches. The classifier inputs a patch with n channels and outputs a binary patch with water pixels labeled 1.')

    # YAML config file
    parser.add_argument("--config_file", default="configs/classifier_default.yaml", help="Path to YAML config file (default: configs/classifier_default.yaml)")
        
    # wandb
    parser.add_argument('--project', help='Wandb project where run will be logged')
    parser.add_argument('--group', help='Optional group name for model experiments (default: None)')
    parser.add_argument('--num_sample_predictions', type=int, help='number of predictions to visualize (default: 40)')

    # evaluation
    parser.add_argument('--mode', choices=['val', 'test'], help=f"dataset used for evaluation metrics (default: val)")

    # ml
    parser.add_argument('-e', '--epochs', type=int)
    parser.add_argument('-b', '--batch_size', type=int)
    parser.add_argument('-s', '--subset', dest='subset', type=float, help='percentage of training dataset to use per epoch (default: 1.0)')
    parser.add_argument('-l', '--lr', type=float)
    parser.add_argument('-p', '--patience', type=int, help='early stopping patience')

    # model
    parser.add_argument('--classifier', choices=MODEL_NAMES,
                        help=f"models: {', '.join(MODEL_NAMES)}")
    # unet
    parser.add_argument('--dropout', type=float)

    # autodespeckler
    parser.add_argument('--autodespeckler', choices=AUTODESPECKLER_NAMES,
                        help=f"models: {', '.join(AUTODESPECKLER_NAMES)}")
    parser.add_argument('--noise_type', choices=NOISE_NAMES,
                        help=f"models: {', '.join(NOISE_NAMES)}")
    parser.add_argument('--noise_coeff', type=float, help='noise coefficient')
    parser.add_argument('--latent_dim', type=int, help='latent dimensions')
    parser.add_argument('--AD_num_layers', type=int, help='Autoencoder layers')
    parser.add_argument('--AD_kernel_size', type=int, help='Autoencoder kernel size')
    parser.add_argument('--AD_dropout', type=float, help='Autoencoder dropout')
    parser.add_argument('--AD_activation_func', choices=['leaky_relu', 'relu'], help=f'activations: leaky_relu, relu')
    parser.add_argument('--VAE_beta', type=float)

    # load weights
    parser.add_argument('--load_autodespeckler', help='File path to .pth')
    parser.add_argument('--freeze_autodespeckler', type=bool,help='Freeze autodespeckler weights during training')

    # data loading
    parser.add_argument('--num_workers', type=int)

    # loss
    parser.add_argument('--loss', choices=LOSS_NAMES,
                        help=f"loss: {', '.join(LOSS_NAMES)}")

    # optimizer
    parser.add_argument('--optimizer', choices=['Adam', 'SGD'],
                        help=f"optimizer: {', '.join(['Adam', 'SGD'])}")

    # reproducibility
    parser.add_argument('--seed', type=int, help='seeding')

    _args = parser.parse_args()
    cfg = Config(**_args.__dict__)
    ad_config_path = cfg.model.autodespeckler.ad_config
    ad_cfg = Config(config_file=ad_config_path) if ad_config_path is not None else None
    sys.exit(main(cfg, ad_cfg))
