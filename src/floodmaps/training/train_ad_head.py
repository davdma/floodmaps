import wandb
import logging
import argparse
from pathlib import Path
from datetime import datetime
from matplotlib.cm import ScalarMappable
from matplotlib.colors import Normalize
import matplotlib.pyplot as plt
import random
from random import Random
from PIL import Image
from glob import glob
import numpy as np
import sys
import pickle
import json

import torch
from torch import nn
from torch.utils.data import DataLoader
from torchvision import transforms

from models.model import build_autodespeckler
from training.dataset import DespecklerSARDataset
from training.optim import get_optimizer
from training.scheduler import get_scheduler
from training.loss import get_ad_loss
from utils.config import Config
from utils.utils import (DATA_DIR,ADEarlyStopper, Metrics, BetaScheduler, get_gradient_norm,
                   get_model_params, print_model_params_and_grads)
from utils.metrics import (denormalize, TV_loss, var_laplacian, ssi, get_random_batch,
                    enl, RIS, quality_m)

AUTODESPECKLER_NAMES = ['CNN1', 'CNN2', 'DAE', 'VAE']
NOISE_NAMES = ['normal', 'masking', 'log_gamma']
AD_LOSS_NAMES = ['L1Loss', 'MSELoss', 'PseudoHuberLoss', 'HuberLoss', 'LogCoshLoss', 'JSDLoss']
SCHEDULER_NAMES = ['Constant', 'ReduceLROnPlateau', 'CosAnnealingLR'] # 'CosWarmRestarts'

def compute_loss(out_dict, targets, loss_fn, cfg, beta_scheduler=None, debug=False):
    recons_loss = loss_fn(out_dict['despeckler_output'], targets)
    loss_dict = dict()
    if cfg.model.autodespeckler == 'VAE':
        # beta hyperparameter - KL regularization
        log_var = torch.clamp(out_dict['log_var'], min=-6, max=6)
        mu = out_dict['mu']
        # ensure KLD loss on same scale as recons_loss! Mean over batch vs. element! imbalance = unstable
        kld_loss = torch.mean(-0.5 * torch.sum(1 + log_var - mu ** 2 - log_var.exp(), dim = 1), dim = 0)

        # debug ratio between recons_loss and kld_loss
        if debug:
            if kld_loss > 0:
                balance_ratio = recons_loss.item() / kld_loss.item()
            else:
                balance_ratio = float('inf')  # Handle division by zero if KLD loss is 0
            print(f"Reconstruction Loss: {recons_loss.item()}, KLD Loss: {kld_loss.item()}, Balance Ratio: {balance_ratio}")

        loss_dict['recons_loss'] = recons_loss
        loss_dict['kld_loss'] = kld_loss
        beta = beta_scheduler.get_beta() if cfg.model.vae.beta_annealing else cfg.model.vae.VAE_beta
        recons_loss = recons_loss + beta * kld_loss # KLD weighting dropped for better regularization

        if torch.isnan(recons_loss).any() or torch.isinf(recons_loss).any():
            print(f'min mu: {mu.min().item()}')
            print(f'max mu: {mu.max().item()}')
            print(f'min log_var: {log_var.min().item()}')
            print(f'max log_var: {log_var.max().item()}')
            raise Exception('recons_loss + kld_loss is nan or inf')

    loss_dict['final_loss'] = recons_loss
    return loss_dict

def train_loop(model, dataloader, device, optimizer, minibatches, loss_fn, cfg, run, epoch, beta_scheduler=None):
    running_tot_loss = torch.tensor(0.0, device=device)
    if cfg.model.autodespeckler == 'VAE':
        running_recons_loss = torch.tensor(0.0, device=device)
        running_kld_loss = torch.tensor(0.0, device=device)
    epoch_gradient_norm = torch.tensor(0.0, device=device)
    batches_logged = 0

    # for VAE monitoring
    all_mu = []
    all_log_var = []

    model.train()
    for batch_i, X in enumerate(dataloader):
        X = X.to(device)

        sar_in = X[:, :2, :, :] if not cfg.data.use_lee else X[:, 2:, :, :]
        out_dict = model(sar_in)

        # also pass SAR layers for reconstruction loss
        loss_dict = compute_loss(out_dict, sar_in, loss_fn, cfg, beta_scheduler=beta_scheduler)
        loss = loss_dict['final_loss']

        if torch.isnan(loss).any():
            err_file_name=f"outputs/ad_param_err_train_{cfg.model.autodespeckler}.json"
            stats_dict = print_model_params_and_grads(model, file_name=err_file_name)
            raise ValueError(f"Loss became NaN during training loop in batch {batch_i}. \
                                The input SAR was NaN: {torch.isnan(sar_in).any()}")

        optimizer.zero_grad()
        loss.backward()
        # Compute gradient norm, scaled by batch size
        if batch_i % cfg.logging.grad_norm_freq == 0:
            scaled_grad_norm = get_gradient_norm(model) / cfg.train.batch_size
            epoch_gradient_norm += scaled_grad_norm
            batches_logged += 1

        nn.utils.clip_grad_norm_(model.parameters(), cfg.train.clip)
        optimizer.step()

        running_tot_loss += loss.detach()

        # for VAE monitoring only
        if cfg.model.autodespeckler == 'VAE':
            # Collect mu and log_var for the whole epoch
            all_mu.append(out_dict['mu'].detach().cpu())
            all_log_var.append(out_dict['log_var'].detach().cpu())
            running_recons_loss += loss_dict['recons_loss'].detach()
            running_kld_loss += loss_dict['kld_loss'].detach()

        if batch_i >= minibatches:
            break

    # calculate metrics
    epoch_tot_loss = running_tot_loss.item() / minibatches
    avg_epoch_gradient_norm = epoch_gradient_norm.item() / batches_logged
    log_dict = {"train loss": epoch_tot_loss,
               "train gradient norm": avg_epoch_gradient_norm}

    # VAE mu and log_var monitoring
    if cfg.model.autodespeckler == 'VAE':
        # full histogram monitoring
        beta = beta_scheduler.get_beta() if cfg.model.vae.beta_annealing else cfg.model.vae.VAE_beta
        all_mu = torch.cat(all_mu, dim=0).numpy()
        all_log_var = torch.cat(all_log_var, dim=0).numpy()
        ratio = (running_recons_loss.item()
                 / (beta * running_kld_loss.item())
                if beta > 0
                else 400)
        log_dict.update({"mu_mean": all_mu.mean(),
                    "mu_std": all_mu.std(),
                    "log_var_mean": all_log_var.mean(),
                    "log_var_std": all_log_var.std(),
                    "train_recons_loss": running_recons_loss.item() / minibatches,
                    "train_kld_loss": running_kld_loss.item() / minibatches,
                    "train_ratio": ratio,
                    "beta": beta})
    run.log(log_dict, step=epoch)

    return epoch_tot_loss

def test_loop(model, dataloader, device, loss_fn, cfg, run, epoch, beta_scheduler=None):
    running_tot_vloss = torch.tensor(0.0, device=device)
    num_batches = len(dataloader)

    model.eval()
    with torch.no_grad():
        for batch_i, X in enumerate(dataloader):
            X = X.to(device)
            sar_in = X[:, :2, :, :] if not cfg.data.use_lee else X[:, 2:, :, :]
            out_dict = model(sar_in)
            loss_dict = compute_loss(out_dict, sar_in, loss_fn, cfg, beta_scheduler=beta_scheduler)
            loss = loss_dict['final_loss']
            if torch.isnan(loss).any():
                err_file_name=f"outputs/ad_param_err_val_{cfg.model.autodespeckler}.json"
                stats_dict = print_model_params_and_grads(model, file_name=err_file_name)
                raise ValueError(f"Loss became NaN during validation loop in batch {batch_i}. \
                                The input SAR was NaN: {torch.isnan(sar_in).any()}")
            running_tot_vloss += loss.detach()

    epoch_tot_vloss = running_tot_vloss.item() / num_batches

    run.log({'val loss': epoch_tot_vloss}, step=epoch)
    return epoch_tot_vloss

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

    # optimizer and scheduler for reducing learning rate
    optimizer = get_optimizer(model, cfg)
    scheduler = get_scheduler(optimizer, cfg)
    if cfg.model.autodespeckler == 'VAE' and cfg.model.vae.beta_annealing:
        beta_scheduler = BetaScheduler(beta=cfg.model.vae.VAE_beta,
                                    period=cfg.model.vae.beta_period,
                                    n_cycle=cfg.model.vae.beta_cycles,
                                    ratio=cfg.model.vae.beta_proportion)
    else:
        beta_scheduler = None

    if cfg.train.early_stopping:
        if cfg.model.autodespeckler == 'VAE':
            early_stopper = ADEarlyStopper(patience=cfg.train.patience, beta_annealing=cfg.model.vae.beta_annealing,
                                        period=cfg.model.vae.beta_period, n_cycle=cfg.model.vae.beta_cycles,
                                        count_cycles=False)
        else:
            early_stopper = ADEarlyStopper(patience=cfg.train.patience, beta_annealing=False,
                                        period=None, n_cycle=None, count_cycles=False)

    run.define_metric("val reconstruction loss", summary="min")
    minibatches = int(len(train_loader) * cfg.train.subset)
    loss_fn = get_ad_loss(cfg).to(device)
    for epoch in range(cfg.train.epochs):
        try:
            # train loop
            avg_loss = train_loop(model, train_loader, device, optimizer,
                                  minibatches, loss_fn, cfg, run, epoch,
                                  beta_scheduler=beta_scheduler)

            # at the end of each training epoch compute validation
            avg_vloss = test_loop(model, val_loader, device, loss_fn, cfg, run,
                                  epoch, beta_scheduler=beta_scheduler)
        except Exception as err:
            raise RuntimeError(f'Error while training occurred at epoch {epoch}.') from err

        if cfg.train.early_stopping:
            early_stopper.step(avg_vloss, model, epoch)
            if early_stopper.is_stopped():
                break

        if scheduler is not None:
            if isinstance(scheduler, torch.optim.lr_scheduler.ReduceLROnPlateau):
                scheduler.step(avg_vloss)
            else:
                scheduler.step()

        if beta_scheduler is not None:
            beta_scheduler.step()

        run.log({"learning_rate": scheduler.get_last_lr()[0] if scheduler is not None else cfg.train.lr}, step=epoch)

    # retrieve best metrics and weights
    fmetrics = Metrics(use_partitions=False)
    model_weights = None
    if cfg.train.early_stopping:
        model_weights = early_stopper.get_best_weights()
        # reset model to checkpoint for later sample prediction
        model.load_state_dict(model_weights)
        fmetrics.save_metrics('val', loss=early_stopper.get_min_validation_loss())
    else:
        model_weights = model.state_dict()
        fmetrics.save_metrics('val', loss=avg_vloss)

    # for benchmarking purposes
    if cfg.eval.mode == 'test':
        test_loss = test_loop(test_loader, model, device, cfg, logging=False)
        fmetrics.save_metrics('test', loss=test_loss)

    run.summary[f"final_{cfg.eval.mode}_loss"] = fmetrics.get_metrics(split=cfg.eval.mode)['loss']
    return model_weights, fmetrics

def calculate_metrics(dataloader, dataset, train_mean, train_std, model, \
                      device, metrics, cfg, run, sample_size=100):
    train_mean = train_mean[:2].to(device) if not cfg.data.use_lee else train_mean[2:].to(device)
    train_std = train_std[:2].to(device) if not cfg.data.use_lee else train_std[2:].to(device)

    # TV Loss, Var of Laplacian, SSI
    tv_loss = torch.tensor(0.0, device=device, dtype=torch.float32)
    var_lap = torch.tensor(0.0, device=device, dtype=torch.float32)
    ssi_val = torch.tensor(0.0, device=device, dtype=torch.float32)
    for X in dataloader:
        X = X.to(device)
        X = X[:, :2, :, :] if not cfg.data.use_lee else X[:, 2:, :, :]
        out_dict = model(X)
        result = out_dict['despeckler_output']

        # convert back to dB scale
        with torch.no_grad():
            # figure out how to denormalize here
            db_sar_filt = denormalize(result, train_mean, train_std)
            db_sar_noisy = denormalize(X, train_mean, train_std)
            tv_loss += TV_loss(db_sar_filt, weight=1, per_pixel=False)
            var_lap += var_laplacian(db_sar_filt, per_pixel=False)
            ssi_val += ssi(db_sar_noisy, db_sar_filt, per_pixel=False)

    # ENL, Quality M, RIS
    # subset 100 to pass through inference
    batch = get_random_batch(dataset, batch_size=sample_size)
    batch = batch.to(device)
    batch = batch[:, :2, :, :] if not cfg.data.use_lee else batch[:, 2:, :, :]
    out_dict = model(batch)
    result = out_dict['despeckler_output']

    # convert back to dB scale
    with torch.no_grad():
        db_batch_filt = denormalize(result, train_mean, train_std).cpu().numpy()
        db_batch_noisy = denormalize(batch, train_mean, train_std).cpu().numpy()

    tot_enl = 0.0
    tot_ris = 0.0
    tot_m = 0.0
    for i in range(sample_size):
        # only calculate for VV
        db_filt_vv = db_batch_filt[i, 0]
        db_noisy_vv = db_batch_noisy[i, 0]

        tot_enl += enl(db_filt_vv, N=10)
        tot_ris += RIS(db_noisy_vv, db_filt_vv)
        tot_m += quality_m(db_noisy_vv, db_filt_vv, samples=10)

    metrics_dict = {"tv_loss": (tv_loss / len(dataloader)).item(),
                    "var_lap": (var_lap / len(dataloader)).item(),
                    "ssi_val": (ssi_val / len(dataloader)).item(),
                    "avg_enl": tot_enl / sample_size,
                    "avg_m": tot_m / sample_size,
                    "avg_ris": tot_ris / sample_size}
    metrics.save_metrics(cfg.eval.mode, **metrics_dict)

    # log as wandb summary statistics
    run.summary.update(metrics_dict)

def save_experiment(weights, metrics, cfg, run):
    """Save experiment files to directory specified by config save_path."""
    path = Path(cfg.save_path)
    path.mkdir(parents=True, exist_ok=True)

    if weights is not None:
        torch.save(weights, path / "model.pth")

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


def create_histogram_plot(input_values, output_values, min=-2, max=2):
    """
    Creates an overlaid histogram and returns a WandB Image.

    Note: resolution for wandb image is low (except for the matplotlib distribution plots).
    This is for efficiency sake. During final benchmarking, can use matplotlib fig with dpi=300
    for high res WandB image.
    """
    fig, ax = plt.subplots(figsize=(5, 4), dpi=200)
    ax.hist(input_values, bins=50, alpha=0.5, range=(min, max), label="SAR-in", color="blue")
    ax.hist(output_values, bins=50, alpha=0.5, range=(min, max), label="SAR-out", color="orange")
    ax.legend()
    ax.set_title("Overlayed SAR Intensity Distribution")

    # Convert the figure to a WandB image
    wandb_image = wandb.Image(fig)
    plt.clf()
    plt.close(fig)  # Close figure to free memory
    return wandb_image

def sample_predictions(model, sample_set, mean, std, cfg, histogram=True, hist_freq=1, seed=24330):
    """Generate predictions on a subset of images in the dataset for wandb logging."""
    if cfg.wandb.num_sample_predictions <= 0:
        return None

    # enable dem visualization in dataset
    sample_set.set_include_dem(True)
    columns = ["id", 'input_vv', 'input_vh', 'enhanced_lee_vv', 'enhanced_lee_vh', 'despeckled_vv', 'despeckled_vh', 'dem']

    # some display options
    if cfg.model.autodespeckler == 'DAE':
        columns.insert(3, 'noisy_vv')
        columns.insert(4, 'noisy_vh')
    if histogram:
        columns.extend(['vv_histogram', 'vh_histogram'])

    table = wandb.Table(columns=columns)
    vv_map = ScalarMappable(norm=None, cmap='gray')
    vh_map = ScalarMappable(norm=None, cmap='gray')
    dem_map = ScalarMappable(norm=None, cmap='gray')
    dae_vv_map = ScalarMappable(norm=None, cmap='gray') # tmp
    dae_vh_map = ScalarMappable(norm=None, cmap='gray') # tmp

    model.to('cpu')
    model.eval()
    rng = Random(seed)
    samples = rng.sample(range(0, len(sample_set)), cfg.wandb.num_sample_predictions)
    for index, k in enumerate(samples):
        # get all images to shape (H, W, C) with C = 1 or 3 (1 for grayscale, 3 for RGB)
        X = sample_set[k]

        with torch.no_grad():
            sar_in = X[:2, :, :] if not cfg.data.use_lee else X[2:4, :, :]
            out_dict = model(sar_in.unsqueeze(0))
            despeckler_output = out_dict['despeckler_output'].squeeze(0)
            despeckler_input = out_dict['despeckler_input'].squeeze(0) # tmp

        # Channels are descaled using linear variance scaling
        X = X.permute(1, 2, 0)
        # X = std * X + mean - removed as we choose to work with standardized data

        row = [k]
        # inputs, outputs and their ranges for converting to grayscale
        vv = X[:, :, 0].numpy()
        en_vv = X[:, :, 2].numpy()
        recons_vv = despeckler_output[0].numpy()
        min_vv = np.min([np.min(vv), np.min(en_vv), np.min(recons_vv)])
        max_vv = np.max([np.max(vv), np.max(en_vv), np.max(recons_vv)])
        vv_map.set_norm(Normalize(vmin=min_vv, vmax=max_vv))

        vh = X[:, :, 1].numpy()
        en_vh = X[:, :, 3].numpy()
        recons_vh = despeckler_output[1].numpy()
        min_vh = np.min([np.min(vh), np.min(en_vh), np.min(recons_vh)])
        max_vh = np.max([np.max(vh), np.max(en_vh), np.max(recons_vh)])
        vh_map.set_norm(Normalize(vmin=min_vh, vmax=max_vh))

        dem = X[:, :, 4].numpy()
        dem_map.set_norm(Normalize(vmin=np.min(dem), vmax=np.max(dem)))

        # tmp
        noisy_vv = despeckler_input[0].numpy()
        noisy_vh = despeckler_input[1].numpy()
        dae_vv_map.set_norm(Normalize(vmin=np.min(noisy_vv), vmax=np.max(noisy_vv)))
        dae_vh_map.set_norm(Normalize(vmin=np.min(noisy_vh), vmax=np.max(noisy_vh)))

        # VV input images
        vv = vv_map.to_rgba(vv, bytes=True)
        vv_img = Image.fromarray(vv, mode="RGBA")
        row.append(wandb.Image(vv_img))

        # VH input images
        vh = vh_map.to_rgba(vh, bytes=True)
        vh_img = Image.fromarray(vh, mode="RGBA")
        row.append(wandb.Image(vh_img))

        # DAE ONLY
        if cfg.model.autodespeckler == 'DAE':
            # noisy VV
            noisy_vv = dae_vv_map.to_rgba(noisy_vv, bytes=True)
            noisy_vv_img = Image.fromarray(noisy_vv, mode="RGBA")
            row.append(wandb.Image(noisy_vv_img))

            # noisy VH
            noisy_vh = dae_vh_map.to_rgba(noisy_vh, bytes=True)
            noisy_vh_img = Image.fromarray(noisy_vh, mode="RGBA")
            row.append(wandb.Image(noisy_vh_img))

        # Enhanced lee vv
        en_vv = vv_map.to_rgba(en_vv, bytes=True)
        en_vv_img = Image.fromarray(en_vv, mode="RGBA")
        row.append(wandb.Image(en_vv_img))

        # Enhanced lee vh
        en_vh = vh_map.to_rgba(en_vh, bytes=True)
        en_vh_img = Image.fromarray(en_vh, mode="RGBA")
        row.append(wandb.Image(en_vh_img))

        # reconstruction VV
        recons_vv = vv_map.to_rgba(recons_vv, bytes=True)
        recons_vv_img = Image.fromarray(recons_vv, mode="RGBA")
        row.append(wandb.Image(recons_vv_img))

        # reconstruction VH
        recons_vh = vh_map.to_rgba(recons_vh, bytes=True)
        recons_vh_img = Image.fromarray(recons_vh, mode="RGBA")
        row.append(wandb.Image(recons_vh_img))

        # dem
        dem = dem_map.to_rgba(dem, bytes=True)
        dem_img = Image.fromarray(dem, mode="RGBA")
        row.append(wandb.Image(dem_img))

        # compare intensity distributions
        if histogram:
            if index % hist_freq == 0:
                # log_histograms(sar_in.numpy(), despeckler_output.numpy(), k)
                input_values = sar_in[0].numpy().flatten()
                output_values = despeckler_output[0].numpy().flatten()
                row.append(create_histogram_plot(input_values, output_values, min=-2, max=2))

                input_values = sar_in[1].numpy().flatten()
                output_values = despeckler_output[1].numpy().flatten()
                row.append(create_histogram_plot(input_values, output_values, min=-2, max=2))
            else:
                # skip so add empty cell
                row.extend([None, None])

        table.add_data(*row)

    return table

def sample_examples(model, sample_set, cfg, idxs=[14440, 3639, 7866]):
    """Generate curated examples for model qualitative analysis. Select example cases via dataset indices."""
    # include dem and create necessary coordinate grids for dem plots
    sample_set.set_include_dem(True)
    x = np.linspace(0, 1, 64)  # Adjust the range as needed
    y = np.linspace(0, 1, 64)
    Xs, Ys = np.meshgrid(x, y)
    Ys_flipped = np.flipud(Ys) # invert y to match imshow

    vv_map = ScalarMappable(norm=None, cmap='gray')
    vh_map = ScalarMappable(norm=None, cmap='gray')
    model.to('cpu')
    model.eval()
    examples = []
    for k in idxs:
        # get all images to shape (H, W, C) with C = 1 or 3 (1 for grayscale, 3 for RGB)
        X = sample_set[k]

        with torch.no_grad():
            sar_in = X[:2, :, :] if not cfg.data.use_lee else X[2:4, :, :]
            out_dict = model(sar_in.unsqueeze(0))
            despeckler_output = out_dict['despeckler_output'].squeeze(0)
            despeckler_input = out_dict['despeckler_input'].squeeze(0) # tmp

        # Channels are descaled using linear variance scaling
        X = X.permute(1, 2, 0)

        # inputs, outputs and their ranges for converting to grayscale
        vv = X[:, :, 0].numpy()
        en_vv = X[:, :, 2].numpy()
        recons_vv = despeckler_output[0].numpy()
        min_vv = np.min([np.min(vv), np.min(en_vv), np.min(recons_vv)])
        max_vv = np.max([np.max(vv), np.max(en_vv), np.max(recons_vv)])
        vv_map.set_norm(Normalize(vmin=min_vv, vmax=max_vv))

        vh = X[:, :, 1].numpy()
        en_vh = X[:, :, 3].numpy()
        recons_vh = despeckler_output[1].numpy()
        min_vh = np.min([np.min(vh), np.min(en_vh), np.min(recons_vh)])
        max_vh = np.max([np.max(vh), np.max(en_vh), np.max(recons_vh)])
        vh_map.set_norm(Normalize(vmin=min_vh, vmax=max_vh))

        dem = X[:, :, 4].numpy()

        # Stitch the VV and VH images into vertical columns
        stitched_vv = np.vstack([vv, en_vv, recons_vv])
        stitched_vv = vv_map.to_rgba(stitched_vv, bytes=True)
        vv_img = Image.fromarray(stitched_vv, mode="RGBA")
        examples.append(wandb.Image(vv_img, caption=f"({k}VV) Top: In, Middle: Lee, Bottom: Out"))

        # VH input images
        stitched_vh = np.vstack([vh, en_vh, recons_vh])
        stitched_vh = vh_map.to_rgba(stitched_vh, bytes=True)
        vh_img = Image.fromarray(stitched_vh, mode="RGBA")
        examples.append(wandb.Image(vh_img, caption=f"({k}VH) Top: In, Middle: Lee, Bottom: Out"))

        # DEM column with overlaid contour plots
        fig, axes = plt.subplots(3, 1, figsize=(6, 18), dpi=200)
        im = axes[0].imshow(dem, cmap='gray', vmin=np.min(dem), vmax=np.max(dem))
        axes[0].set_title("DEM")
        axes[0].axis("off")
        fig.colorbar(im, ax=axes[0], fraction=0.046, pad=0.04, label="DEM Value")

        axes[1].imshow(dem, cmap='gray', extent=[0, 1, 0, 1], origin='upper')
        contour = axes[1].contour(Xs, Ys_flipped, dem, levels=10, cmap='terrain', alpha=1)
        axes[1].set_title("DEM Contour Plot 1")
        axes[1].set_aspect('equal')
        fig.colorbar(contour, ax=axes[1], fraction=0.046, pad=0.04, label="DEM Value")

        contour = axes[2].contourf(Xs, Ys_flipped, dem, levels=10, cmap='cividis', alpha=1)
        axes[2].set_title("DEM Contour Plot 2")
        axes[2].set_aspect('equal')
        fig.colorbar(contour, ax=axes[2], fraction=0.046, pad=0.04, label="DEM Value")
        axes[2].axis('off')
        plt.tight_layout()
        examples.append(wandb.Image(fig, caption=f"({k}) DEM and Contour Plots"))

    sample_set.set_include_dem(False)
    return examples

def run_experiment_ad(cfg):
    """Run a single autodespeckler model experiment given the configuration parameters."""
    if not wandb.login():
        raise Exception("Failed to login to wandb.")

    # seeding
    np.random.seed(cfg.seed)
    random.seed(cfg.seed)
    torch.manual_seed(cfg.seed)

    # device and model
    device = (
        "cuda"
        if torch.cuda.is_available()
        else "mps"
        if torch.backends.mps.is_available()
        else "cpu"
    )
    model = build_autodespeckler(cfg).to(device)

    # dataset and transforms
    print(f"Using {device} device")
    model_name = cfg.model.autodespeckler
    kernel_size = cfg.data.kernel_size
    size = cfg.data.size
    samples = cfg.data.samples
    sample_dir = DATA_DIR / 'ad' / f'samples_{kernel_size}_{size}_{samples}_dem/'

    # load in mean and std
    with open(sample_dir / f'mean_std_{kernel_size}_{size}_{samples}_dem.pkl', 'rb') as f:
        train_mean, train_std = pickle.load(f)
        train_mean = torch.from_numpy(train_mean)
        train_std = torch.from_numpy(train_std)
    standardize = transforms.Compose([transforms.Normalize(train_mean, train_std)])

    # datasets
    train_set = DespecklerSARDataset(sample_dir, typ="train", transform=standardize, random_flip=cfg.data.random_flip,
                                        seed=cfg.seed+1)
    val_set = DespecklerSARDataset(sample_dir, typ="val", transform=standardize)
    test_set = DespecklerSARDataset(sample_dir, typ="test", transform=standardize) if cfg.eval.mode == 'test' else None

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
        # setup save path
        if cfg.save:
            if cfg.save_path is None:
                default_path = f"experiments/{datetime.today().strftime('%Y-%m-%d')}_{cfg.model.autodespeckler}_{run.id}/"
                cfg.save_path = default_path
                run.config.update({"save_path": cfg.save_path}, allow_val_change=True)
            print(f'Save path set to: {cfg.save_path}')

        # train and save results metrics
        weights, fmetrics = train(model, train_loader, val_loader, test_loader, device, cfg, run)
        calculate_metrics(test_loader if cfg.eval.mode == 'test' else val_loader,
                      test_set if cfg.eval.mode == 'test' else val_set,
                      train_mean, train_std, model, device, fmetrics, cfg, run)
        if cfg.save:
            save_experiment(weights, fmetrics, cfg, run)

        # sample predictions for analysis
        pred_table = sample_predictions(model, test_set if cfg.eval.mode == 'test' else val_set,
                                        train_mean, train_std, cfg)

        # pick 3 full res examples for closer look
        examples = sample_examples(model, test_set if cfg.eval.mode == 'test' else val_set, cfg)
        run.log({"model_val_predictions": pred_table, "examples": examples})
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
    # Add checks
    assert cfg.train.lr > 0, "Learning rate must be positive"
    assert cfg.model.autodespeckler in AUTODESPECKLER_NAMES, f"Model must be one of {AUTODESPECKLER_NAMES}"
    assert cfg.train.loss in AD_LOSS_NAMES, f"Loss must be one of {AD_LOSS_NAMES}"
    assert cfg.train.optimizer in ['Adam', 'SGD'], f"Optimizer must be one of {['Adam', 'SGD']}"
    assert cfg.train.LR_scheduler in SCHEDULER_NAMES, f"LR scheduler must be one of {SCHEDULER_NAMES}"
    assert cfg.train.early_stopping in [True, False], "Early stopping must be a boolean"
    assert not cfg.train.early_stopping or cfg.train.patience is not None, "Patience must be set if early stopping is enabled"
    assert cfg.train.random_flip in [True, False], "Random flip must be a boolean"
    assert cfg.train.save in [True, False], "Save must be a boolean"
    assert cfg.train.batch_size is not None and cfg.train.batch_size > 0, "Batch size must be defined and positive"
    assert cfg.eval.mode in ['val', 'test'], f"Evaluation mode must be one of {['val', 'test']}"
    assert cfg.wandb.project is not None, "Wandb project must be specified"

def main(cfg):
    validate_config(cfg)
    run_experiment_ad(cfg)

if __name__ == '__main__':
    parser = argparse.ArgumentParser(prog='train_ad_head', description='Trains SAR autoencoder head by itself.')

    # YAML config file
    parser.add_argument("--config_file", default="configs/default.yaml", help="Path to YAML config file (default: configs/default.yaml)")

    # save model, config, and wandb run info to folder
    # parser.add_argument('--save', action='store_true', help='save model and configs to file (default: False)')
    parser.add_argument('--save_path', help='directory path for saving the model')

    # wandb
    parser.add_argument('--project', help='Wandb project where run will be logged')
    parser.add_argument('--group', help='Optional group name for model experiments')
    parser.add_argument('--num_sample_predictions', type=int, help='number of predictions to visualize')

    # ml
    parser.add_argument('-e', '--epochs', type=int)
    parser.add_argument('-b', '--batch_size', type=int)
    parser.add_argument('-l', '--lr', type=float)
    parser.add_argument('-p', '--patience', type=int, help='early stopping patience')
    parser.add_argument('--LR_scheduler', choices=SCHEDULER_NAMES,
                        help=f"LR schedulers: {', '.join(SCHEDULER_NAMES)}")

    # autodespeckler
    parser.add_argument('--autodespeckler', choices=AUTODESPECKLER_NAMES,
                        help=f"models: {', '.join(AUTODESPECKLER_NAMES)}")
    parser.add_argument('--noise_type', choices=NOISE_NAMES,
                        help=f"models: {', '.join(NOISE_NAMES)}")
    parser.add_argument('--noise_coeff', type=float, help="noise coefficient")
    parser.add_argument('--latent_dim', type=int, help='latent dimensions')
    parser.add_argument('--AD_num_layers', type=int, help='Autoencoder layers')
    parser.add_argument('--AD_kernel_size', type=int, help='Autoencoder kernel size')
    parser.add_argument('--AD_dropout', type=float, help='Autoencoder dropout')
    parser.add_argument('--AD_activation_func', choices=['leaky_relu', 'relu', 'softplus', 'mish', 'gelu', 'elu'], help='activations: leaky_relu, relu, softplus, mish, gelu, elu')

    # VAE Beta
    parser.add_argument('--VAE_beta', type=float, help="VAE beta for KL divergence term")
    parser.add_argument('--beta_period', type=int, help="Epoch period for beta annealing")
    parser.add_argument('--beta_cycles', type=int, help="M cycles for beta annealing ")
    parser.add_argument('--beta_proportion', type=float, help="R proportion used to increase beta within a cycle")

    # data
    parser.add_argument('--num_workers', type=int)

    # loss
    parser.add_argument('--loss', choices=AD_LOSS_NAMES,
                        help=f"loss: {', '.join(AD_LOSS_NAMES)}")
    parser.add_argument('--clip', type=float, help="Gradient clipping max norm")
    # print statistics of current gradient and adjust norm used to clip according to the statistics
    # heuristically use 1

    # optimizer
    parser.add_argument('--optimizer', choices=['Adam', 'SGD'],
                        help=f"optimizer: {', '.join(['Adam', 'SGD'])}")

    # reproducibility
    parser.add_argument('--seed', type=int, help='seeding')

    # Load base config
    _args = parser.parse_args()
    cfg = Config(**_args.__dict__)
    sys.exit(main(cfg))
