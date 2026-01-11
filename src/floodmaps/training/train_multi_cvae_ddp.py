import wandb
import logging
from pathlib import Path
from datetime import datetime
from matplotlib.cm import ScalarMappable
from matplotlib.colors import Normalize
import matplotlib.pyplot as plt
import random
from random import Random
from PIL import Image
import numpy as np
import pickle
import json
import hydra
from omegaconf import DictConfig, OmegaConf
from pytorch_msssim import ssim
import torch.multiprocessing as mp
import os

import torch
import torch.distributed as dist
from torch import nn
from torch.utils.data import DataLoader
from torch.nn.parallel import DistributedDataParallel as DDP
from torch.utils.data.distributed import DistributedSampler
from torchvision import transforms

from floodmaps.models.model import build_autodespeckler
from floodmaps.training.dataset import ShardedConditionalSARDataset, ConditionalSARDataset, HomogenousSARDataset
from floodmaps.training.optim import get_optimizer
from floodmaps.training.scheduler import get_scheduler
from floodmaps.training.loss import get_ad_loss
from floodmaps.utils.utils import (flatten_dict, ADEarlyStopper, Metrics, BetaScheduler, get_gradient_norm,
                   get_model_params, print_model_params_and_grads, find_free_port)
from floodmaps.utils.checkpoint import save_checkpoint, load_checkpoint
from floodmaps.utils.metrics import (denormalize, normalize, convert_to_amplitude, var_laplacian, enl, psnr, ActiveUnitsTracker)

AD_LOSS_NAMES = ['L1Loss', 'MSELoss', 'PseudoHuberLoss', 'HuberLoss', 'LogCoshLoss']
SCHEDULER_NAMES = ['Constant', 'ReduceLROnPlateau']
VV_DB_MIN, VV_DB_MAX = -30, 0
VH_DB_MIN, VH_DB_MAX = -30, -5
amplitude_min_vv = 10.0 ** (VV_DB_MIN / 20.0)
amplitude_max_vv = 10.0 ** (VV_DB_MAX / 20.0)
amplitude_min_vh = 10.0 ** (VH_DB_MIN / 20.0)
amplitude_max_vh = 10.0 ** (VH_DB_MAX / 20.0)

# Use TF32 Tensor Cores for better performance
torch.backends.cuda.matmul.allow_tf32 = True
torch.backends.cudnn.allow_tf32 = True

### Script for training CVAE with conditioning input:
### Separate from train_multi.py as it allows for two different validation losses
### One for reconstruction (using encoder z) and one for inference (using gaussian z in decoder)

### Monitor for posterior collapse by making sure:
# KL loss is not zero
# Median KL_i is not zero
# Frac(KL < 1e-3) is not zero
# Ablation delta is not zero
# Variance of mu across dims is not near zero (use Active Units)

def compute_loss(out_dict, targets, loss_fn, cfg, beta_scheduler=None, debug=False):
    recons_loss = loss_fn(out_dict['despeckler_output'], targets)
    loss_dict = dict()

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
    
    beta = beta_scheduler.get_beta() if cfg.model.cvae.beta_annealing else cfg.model.cvae.VAE_beta
    elbo_loss = recons_loss + beta * kld_loss # KLD weighting dropped for better regularization

    if torch.isnan(elbo_loss).any() or torch.isinf(elbo_loss).any():
        print(f'min mu: {mu.min().item()}')
        print(f'max mu: {mu.max().item()}')
        print(f'min log_var: {log_var.min().item()}')
        print(f'max log_var: {log_var.max().item()}')
        raise Exception('elbo_loss is nan or inf')

    loss_dict['elbo_loss'] = elbo_loss
    return loss_dict

def compute_inference_loss(out_dict, targets, loss_fn):
    recons_loss = loss_fn(out_dict['despeckler_output'], targets)
    loss_dict = {'recons_loss': recons_loss}
    return loss_dict

def train_loop(rank, world_size, model, dataloader, device, optimizer, loss_fn, cfg, run, epoch, beta_scheduler=None):
    """Training loop for CVAE with posterior collapse monitoring.
    
    NOTE: The training loss is not used besides weight updates, therefore
    it is not all reduced across ranks. The returned loss is only the average
    of the losses on the local rank.
    
    Posterior collapse metrics (see README.md for interpretation):
    - AU_strict^/AU_lenient^: Active units at thresholds 1e-2/1e-3
    - median_KL^: Median per-sample KL (rank-local, not gathered)
    - frac_KL_small^: Fraction of samples with KL < 1e-3 (all-reduced)
    - log_var_min/max: Raw log variance extrema (all-reduced)
    """
    running_tot_loss = torch.tensor(0.0, device=device)
    running_recons_loss = torch.tensor(0.0, device=device)
    running_kld_loss = torch.tensor(0.0, device=device)
    epoch_gradient_norm = torch.tensor(0.0, device=device)
    batches_logged = 0
    num_batches = len(dataloader)

    # Posterior collapse monitoring
    latent_dim = cfg.model.cvae.latent_dim
    au_tracker = ActiveUnitsTracker(latent_dim=latent_dim).to(device)
    kl_per_sample_list = []  # Accumulate per-sample KL for median (rank-local)
    kl_small_count = torch.tensor(0, device=device, dtype=torch.long)  # Count of samples with KL < 1e-3
    total_samples = torch.tensor(0, device=device, dtype=torch.long)
    log_var_min = torch.tensor(float('inf'), device=device)
    log_var_max = torch.tensor(float('-inf'), device=device)

    # AMP setup - use bfloat16 for better numerical stability
    use_amp = getattr(cfg.train, 'use_amp', False)
    amp_dtype = torch.bfloat16

    model.train()
    for batch_i, (X, y) in enumerate(dataloader):
        X = X.to(device)
        y = y.to(device)

        with torch.autocast(device_type=device.type, dtype=amp_dtype, enabled=use_amp):
            out_dict = model(X, y)

        # also pass SAR layers for reconstruction loss
        loss_dict = compute_loss(out_dict, y, loss_fn, cfg, beta_scheduler=beta_scheduler)
        loss = loss_dict['elbo_loss']

        if torch.isnan(loss).any():
            err_file_name=f"outputs/ad_param_err_train_{cfg.model.autodespeckler}.json"
            stats_dict = print_model_params_and_grads(model, file_name=err_file_name)
            raise ValueError(f"Loss became NaN during training loop in batch {batch_i}. \
                                The input SAR was NaN: {torch.isnan(X).any()}")

        optimizer.zero_grad()
        loss.backward()
        # Compute gradient norm, scaled by batch size
        if batch_i % cfg.logging.grad_norm_freq == 0:
            epoch_gradient_norm += get_gradient_norm(model)
            batches_logged += 1

        nn.utils.clip_grad_norm_(model.parameters(), cfg.train.clip)
        optimizer.step()

        running_tot_loss += loss.detach()
        running_recons_loss += loss_dict['recons_loss'].detach()
        running_kld_loss += loss_dict['kld_loss'].detach()

        # Posterior collapse monitoring
        mu = out_dict['mu'].detach()
        raw_log_var = out_dict['log_var'].detach()  # Before clamping
        log_var = torch.clamp(raw_log_var, min=-6, max=6)
        
        # Track log_var extrema (before clamping)
        log_var_min = torch.min(log_var_min, raw_log_var.min())
        log_var_max = torch.max(log_var_max, raw_log_var.max())
        
        # Active Units tracking (streaming Welford)
        au_tracker.update(mu)
        
        # Per-sample KL: KL_i = 0.5 * sum_d(mu_d^2 + exp(log_var_d) - 1 - log_var_d)
        kl_per_sample = 0.5 * (mu.pow(2) + log_var.exp() - 1 - log_var).sum(dim=1)  # (B,)
        kl_per_sample_list.append(kl_per_sample)
        
        # Count samples with KL < 1e-3
        kl_small_count += (kl_per_sample < 1e-3).sum()
        total_samples += kl_per_sample.shape[0]

    # calculate metrics
    epoch_tot_loss = running_tot_loss.item() / num_batches
    avg_epoch_gradient_norm = epoch_gradient_norm.item() / batches_logged if batches_logged > 0 else 0.0
    log_dict = {"train loss": epoch_tot_loss,
               "train gradient norm": avg_epoch_gradient_norm}

    beta = beta_scheduler.get_beta() if cfg.model.cvae.beta_annealing else cfg.model.cvae.VAE_beta
    ratio = (running_recons_loss.item()
                / (beta * running_kld_loss.item())
            if beta > 0
            else 400)
    log_dict.update({
                "train_recons_loss": running_recons_loss.item() / num_batches,
                "train_kld_loss": running_kld_loss.item() / num_batches,
                "train_ratio": ratio,
                "beta": beta}
            )

    # All-reduce AU Welford accumulators and compute
    dist.all_reduce(au_tracker.count, op=dist.ReduceOp.SUM)
    dist.all_reduce(au_tracker.mean_per_dim, op=dist.ReduceOp.SUM)
    au_tracker.mean_per_dim /= world_size  # Average the means
    dist.all_reduce(au_tracker.M2_per_dim, op=dist.ReduceOp.SUM)
    au_results = au_tracker.compute()
    
    # All-reduce frac(KL < 1e-3)
    dist.all_reduce(kl_small_count, op=dist.ReduceOp.SUM)
    dist.all_reduce(total_samples, op=dist.ReduceOp.SUM)
    frac_kl_small = kl_small_count.float() / total_samples.float()
    
    # All-reduce log_var min/max
    dist.all_reduce(log_var_min, op=dist.ReduceOp.MIN)
    dist.all_reduce(log_var_max, op=dist.ReduceOp.MAX)
    
    # Median KL computed only on rank 0's samples (no gather for efficiency)
    all_kl = torch.cat(kl_per_sample_list)
    median_kl = torch.median(all_kl).item()
    
    log_dict.update({
        "train_AU_strict^": au_results['AU_strict'],
        "train_AU_lenient^": au_results['AU_lenient'],
        "train_median_KL^": median_kl,
        "train_frac_KL_small^": frac_kl_small.item(),
        "train_log_var_min": log_var_min.item(),
        "train_log_var_max": log_var_max.item(),
    })
    au_tracker.reset()

    if rank == 0:
        run.log(log_dict, step=epoch)

    return epoch_tot_loss

def test_loop(rank, world_size, model, dataloader, device, loss_fn, cfg, run, epoch, beta_scheduler=None):
    """Validation loop with posterior collapse monitoring and ablation diagnostic.
    
    Validation losses are all reduced across ranks.
    
    Posterior collapse metrics (see README.md for interpretation):
    - AU_strict^/AU_lenient^: Active units at thresholds 1e-2/1e-3
    - median_KL^: Median per-sample KL (rank-local, not gathered)
    - frac_KL_small^: Fraction of samples with KL < 1e-3 (all-reduced)
    - log_var_min/max: Raw log variance extrema (all-reduced)
    - ablation_delta^: L(z=0) - L(z=mu), tests if decoder ignores z
    """
    running_tot_vloss = torch.tensor(0.0, device=device)
    running_tot_vloss_inf = torch.tensor(0.0, device=device) # recons loss from random sampling z then decoding
    running_recons_loss = torch.tensor(0.0, device=device)
    running_kld_loss = torch.tensor(0.0, device=device)
    num_batches = len(dataloader)

    # Posterior collapse monitoring
    latent_dim = cfg.model.cvae.latent_dim
    au_tracker = ActiveUnitsTracker(latent_dim=latent_dim).to(device)
    kl_per_sample_list = []  # Accumulate per-sample KL for median (rank-local)
    kl_small_count = torch.tensor(0, device=device, dtype=torch.long)
    total_samples = torch.tensor(0, device=device, dtype=torch.long)
    log_var_min = torch.tensor(float('inf'), device=device)
    log_var_max = torch.tensor(float('-inf'), device=device)
    ablation_delta = torch.tensor(0.0, device=device)  # For first batch ablation

    model.eval()
    with torch.no_grad():
        for batch_i, (X, y) in enumerate(dataloader):
            X = X.to(device)
            y = y.to(device)
            # evaluate reconstruction
            out_dict = model(X, y)

            loss_dict = compute_loss(out_dict, y, loss_fn, cfg, beta_scheduler=beta_scheduler)
            loss = loss_dict['elbo_loss']
            if torch.isnan(loss).any():
                err_file_name=f"outputs/ad_param_err_val_{cfg.model.autodespeckler}.json"
                stats_dict = print_model_params_and_grads(model, file_name=err_file_name)
                raise ValueError(f"Loss became NaN during validation loop in batch {batch_i}. \
                                The input SAR was NaN: {torch.isnan(X).any()}")
            running_tot_vloss += loss.detach()
            running_recons_loss += loss_dict['recons_loss'].detach()
            running_kld_loss += loss_dict['kld_loss'].detach()

            # Posterior collapse monitoring
            mu = out_dict['mu'].detach()
            raw_log_var = out_dict['log_var'].detach()
            log_var = torch.clamp(raw_log_var, min=-6, max=6)
            
            log_var_min = torch.min(log_var_min, raw_log_var.min())
            log_var_max = torch.max(log_var_max, raw_log_var.max())
            au_tracker.update(mu)
            
            kl_per_sample = 0.5 * (mu.pow(2) + log_var.exp() - 1 - log_var).sum(dim=1)
            kl_per_sample_list.append(kl_per_sample)
            kl_small_count += (kl_per_sample < 1e-3).sum()
            total_samples += kl_per_sample.shape[0]

            # Ablation diagnostic on first batch only
            if batch_i == 0:
                # z=mu path (deterministic, no sampling)
                out_mu = model.module.decode(mu, X)
                loss_mu = loss_fn(out_mu, y)
                # z=0 path (ablated latent)
                z_zero = torch.zeros_like(mu)
                out_zero = model.module.decode(z_zero, X)
                loss_zero = loss_fn(out_zero, y)
                ablation_delta = loss_zero - loss_mu

            # evaluate inference / generation
            inference_out_dict = model.module.inference(X)
            inference_loss_dict = compute_inference_loss(inference_out_dict, y, loss_fn)
            inference_loss = inference_loss_dict['recons_loss'] # for inference we only have reconstruction loss
            running_tot_vloss_inf += inference_loss.detach()

    # synchronize all validation losses across ranks
    dist.all_reduce(running_tot_vloss, op=dist.ReduceOp.SUM)
    dist.all_reduce(running_tot_vloss_inf, op=dist.ReduceOp.SUM)
    dist.all_reduce(running_recons_loss, op=dist.ReduceOp.SUM)
    dist.all_reduce(running_kld_loss, op=dist.ReduceOp.SUM)
    running_tot_vloss /= world_size
    running_tot_vloss_inf /= world_size
    running_recons_loss /= world_size
    running_kld_loss /= world_size

    epoch_tot_vloss = running_tot_vloss.item() / num_batches
    epoch_tot_vloss_inf = running_tot_vloss_inf.item() / num_batches

    # two different validation losses:
    # encoder-decoder uses full model
    # *inference* uses only decoder (generation only metric)
    log_dict = {'val loss (encoder-decoder)': epoch_tot_vloss,
                'val_recons_loss (*inference*)': epoch_tot_vloss_inf}

    log_dict.update({
                "val_recons_loss (encoder-decoder)": running_recons_loss.item() / num_batches,
                "val_kld_loss (encoder-decoder)": running_kld_loss.item() / num_batches
            })

    # All-reduce AU Welford accumulators and compute
    dist.all_reduce(au_tracker.count, op=dist.ReduceOp.SUM)
    dist.all_reduce(au_tracker.mean_per_dim, op=dist.ReduceOp.SUM)
    au_tracker.mean_per_dim /= world_size
    dist.all_reduce(au_tracker.M2_per_dim, op=dist.ReduceOp.SUM)
    au_results = au_tracker.compute()
    
    # All-reduce frac(KL < 1e-3)
    dist.all_reduce(kl_small_count, op=dist.ReduceOp.SUM)
    dist.all_reduce(total_samples, op=dist.ReduceOp.SUM)
    frac_kl_small = kl_small_count.float() / total_samples.float()
    
    # All-reduce log_var min/max
    dist.all_reduce(log_var_min, op=dist.ReduceOp.MIN)
    dist.all_reduce(log_var_max, op=dist.ReduceOp.MAX)
    
    # Average ablation delta across ranks
    dist.all_reduce(ablation_delta, op=dist.ReduceOp.SUM)
    ablation_delta /= world_size
    
    # Median KL computed only on rank 0's samples (no gather for efficiency)
    all_kl = torch.cat(kl_per_sample_list)
    median_kl = torch.median(all_kl).item()
    
    log_dict.update({
        "val_AU_strict^": au_results['AU_strict'],
        "val_AU_lenient^": au_results['AU_lenient'],
        "val_median_KL^": median_kl,
        "val_frac_KL_small^": frac_kl_small.item(),
        "val_log_var_min": log_var_min.item(),
        "val_log_var_max": log_var_max.item(),
        "val_ablation_delta^": ablation_delta.item(),
    })
    au_tracker.reset()

    if rank == 0:
        run.log(log_dict, step=epoch)

    metrics_dict = log_dict.copy()
    return epoch_tot_vloss, epoch_tot_vloss_inf, metrics_dict

def evaluate(rank, world_size, model, dataloader, device, loss_fn, cfg, beta_scheduler=None):
    """Evaluate metrics on test set with posterior collapse monitoring.
    
    All ranks compute on their data shard. Results are all-reduced.
    
    Posterior collapse metrics (see README.md for interpretation):
    - AU_strict^/AU_lenient^: Active units at thresholds 1e-2/1e-3
    - median_KL^: Median per-sample KL (rank-local, not gathered)
    - frac_KL_small^: Fraction of samples with KL < 1e-3 (all-reduced)
    - log_var_min/max: Raw log variance extrema (all-reduced)
    - ablation_delta^: L(z=0) - L(z=mu), tests if decoder ignores z
    """
    running_tot_vloss = torch.tensor(0.0, device=device)
    running_tot_vloss_inf = torch.tensor(0.0, device=device) # recons loss from random sampling z then decoding
    running_recons_loss = torch.tensor(0.0, device=device)
    running_kld_loss = torch.tensor(0.0, device=device)
    num_batches = len(dataloader)

    # Posterior collapse monitoring
    latent_dim = cfg.model.cvae.latent_dim
    au_tracker = ActiveUnitsTracker(latent_dim=latent_dim).to(device)
    kl_per_sample_list = []
    kl_small_count = torch.tensor(0, device=device, dtype=torch.long)
    total_samples = torch.tensor(0, device=device, dtype=torch.long)
    log_var_min = torch.tensor(float('inf'), device=device)
    log_var_max = torch.tensor(float('-inf'), device=device)
    ablation_delta = torch.tensor(0.0, device=device)

    model.eval()
    with torch.no_grad():
        for batch_i, (X, y) in enumerate(dataloader):
            X = X.to(device)
            y = y.to(device)
            # evaluate reconstruction
            out_dict = model(X, y)
            loss_dict = compute_loss(out_dict, y, loss_fn, cfg, beta_scheduler=beta_scheduler)
            loss = loss_dict['elbo_loss']
            if torch.isnan(loss).any():
                err_file_name=f"outputs/ad_param_err_val_{cfg.model.autodespeckler}.json"
                stats_dict = print_model_params_and_grads(model, file_name=err_file_name)
                raise ValueError(f"Loss became NaN during validation loop in batch {batch_i}. \
                                The input SAR was NaN: {torch.isnan(X).any()}")
            running_tot_vloss += loss.detach()
            running_recons_loss += loss_dict['recons_loss'].detach()
            running_kld_loss += loss_dict['kld_loss'].detach()

            # Posterior collapse monitoring
            mu = out_dict['mu'].detach()
            raw_log_var = out_dict['log_var'].detach()
            log_var = torch.clamp(raw_log_var, min=-6, max=6)
            
            log_var_min = torch.min(log_var_min, raw_log_var.min())
            log_var_max = torch.max(log_var_max, raw_log_var.max())
            au_tracker.update(mu)
            
            kl_per_sample = 0.5 * (mu.pow(2) + log_var.exp() - 1 - log_var).sum(dim=1)
            kl_per_sample_list.append(kl_per_sample)
            kl_small_count += (kl_per_sample < 1e-3).sum()
            total_samples += kl_per_sample.shape[0]

            # Ablation diagnostic on first batch only
            if batch_i == 0:
                out_mu = model.module.decode(mu, X)
                loss_mu = loss_fn(out_mu, y)
                z_zero = torch.zeros_like(mu)
                out_zero = model.module.decode(z_zero, X)
                loss_zero = loss_fn(out_zero, y)
                ablation_delta = loss_zero - loss_mu

            # evaluate inference / generation
            inference_out_dict = model.module.inference(X)
            inference_loss_dict = compute_inference_loss(inference_out_dict, y, loss_fn)
            inference_loss = inference_loss_dict['recons_loss'] # for inference we only have reconstruction loss
            running_tot_vloss_inf += inference_loss.detach()

    # synchronize all test losses across ranks
    dist.all_reduce(running_tot_vloss, op=dist.ReduceOp.SUM)
    dist.all_reduce(running_tot_vloss_inf, op=dist.ReduceOp.SUM)
    dist.all_reduce(running_recons_loss, op=dist.ReduceOp.SUM)
    dist.all_reduce(running_kld_loss, op=dist.ReduceOp.SUM)
    running_tot_vloss /= world_size
    running_tot_vloss_inf /= world_size
    running_recons_loss /= world_size
    running_kld_loss /= world_size

    epoch_tot_vloss = running_tot_vloss.item() / num_batches
    epoch_tot_vloss_inf = running_tot_vloss_inf.item() / num_batches

    # two different validation losses:
    # encoder-decoder uses full model
    # *inference* uses only decoder (generation only metric)
    log_dict = {'test loss (encoder-decoder)': epoch_tot_vloss,
                'test_recons_loss (*inference*)': epoch_tot_vloss_inf}

    log_dict.update({
                "test_recons_loss (encoder-decoder)": running_recons_loss.item() / num_batches,
                "test_kld_loss (encoder-decoder)": running_kld_loss.item() / num_batches
            })

    # All-reduce AU Welford accumulators and compute
    dist.all_reduce(au_tracker.count, op=dist.ReduceOp.SUM)
    dist.all_reduce(au_tracker.mean_per_dim, op=dist.ReduceOp.SUM)
    au_tracker.mean_per_dim /= world_size
    dist.all_reduce(au_tracker.M2_per_dim, op=dist.ReduceOp.SUM)
    au_results = au_tracker.compute()
    
    # All-reduce frac(KL < 1e-3)
    dist.all_reduce(kl_small_count, op=dist.ReduceOp.SUM)
    dist.all_reduce(total_samples, op=dist.ReduceOp.SUM)
    frac_kl_small = kl_small_count.float() / total_samples.float()
    
    # All-reduce log_var min/max
    dist.all_reduce(log_var_min, op=dist.ReduceOp.MIN)
    dist.all_reduce(log_var_max, op=dist.ReduceOp.MAX)
    
    # Average ablation delta across ranks
    dist.all_reduce(ablation_delta, op=dist.ReduceOp.SUM)
    ablation_delta /= world_size
    
    # Median KL computed only on rank 0's samples (no gather for efficiency)
    all_kl = torch.cat(kl_per_sample_list)
    median_kl = torch.median(all_kl).item()
    
    log_dict.update({
        "test_AU_strict^": au_results['AU_strict'],
        "test_AU_lenient^": au_results['AU_lenient'],
        "test_median_KL^": median_kl,
        "test_frac_KL_small^": frac_kl_small.item(),
        "test_log_var_min": log_var_min.item(),
        "test_log_var_max": log_var_max.item(),
        "test_ablation_delta^": ablation_delta.item(),
    })
    au_tracker.reset()

    metrics_dict = log_dict.copy()
    return epoch_tot_vloss, epoch_tot_vloss_inf, metrics_dict

def train(rank, world_size, model, train_loader, val_loader, test_loader, device, cfg, run):
    # log weights and gradients each epoch
    if rank == 0:
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
        run.watch(model.module, log="all", log_freq=cfg.logging.grad_norm_freq)

    # optimizer and scheduler for reducing learning rate
    optimizer = get_optimizer(model, cfg)
    scheduler = get_scheduler(optimizer, cfg)
    if cfg.model.cvae.beta_annealing:
        beta_scheduler = BetaScheduler(beta=cfg.model.cvae.VAE_beta,
                                    period=cfg.model.cvae.beta_period,
                                    n_cycle=cfg.model.cvae.beta_cycles,
                                    ratio=cfg.model.cvae.beta_proportion)
    else:
        beta_scheduler = None

    if cfg.train.early_stopping:
        early_stopper = ADEarlyStopper(patience=cfg.train.patience, beta_annealing=cfg.model.cvae.beta_annealing,
                                    period=cfg.model.cvae.beta_period, n_cycle=cfg.model.cvae.beta_cycles,
                                    count_cycles=False)
    else:
        early_stopper = None

    # load checkpoint if it exists
    # Load into model.module so checkpoints are portable (compatible with non-DDP checkpoints)
    if cfg.train.checkpoint.load_chkpt:
        chkpt = load_checkpoint(cfg.train.checkpoint.load_chkpt_path, model.module, 
                               optimizer=optimizer, scheduler=scheduler, 
                               early_stopper=early_stopper,
                               beta_scheduler=beta_scheduler)
        start_epoch = chkpt['epoch'] + 1
        if cfg.train.epochs < start_epoch:
            raise ValueError(f"Epochs specified in config ({cfg.train.epochs}) is less than the epoch at which the checkpoint was saved ({start_epoch}).")
    else:
        start_epoch = 0

    loss_fn = get_ad_loss(cfg).to(device)
    for epoch in range(start_epoch, cfg.train.epochs):
        # Set epoch for ShardedConditionalSARDataset to shuffle shards each epoch
        train_loader.dataset.set_epoch(epoch)
        try:
            # train loop
            avg_loss = train_loop(rank, world_size, model, train_loader, device, optimizer,
                                  loss_fn, cfg, run, epoch,
                                  beta_scheduler=beta_scheduler)

            # at the end of each training epoch compute validation
            avg_vloss, avg_vloss_inf, val_metrics_dict = test_loop(rank, world_size, model, val_loader, device, loss_fn, cfg, run,
                                  epoch, beta_scheduler=beta_scheduler)
        except Exception as err:
            raise RuntimeError(f'Error while training occurred at epoch {epoch}.') from err

        if cfg.train.early_stopping:
            # early stop on elbo loss (NOT inference loss)
            # Pass model.module so weights are saved without 'module.' prefix
            early_stopper.step(avg_vloss, model.module, epoch, metrics=val_metrics_dict)

            # Synchronize early stopping decision across all ranks
            should_stop = torch.tensor([1 if early_stopper.is_stopped() else 0], dtype=torch.int, device=device)
            dist.all_reduce(should_stop, op=dist.ReduceOp.MAX)
            if should_stop.item() == 1:
                break

        if scheduler is not None:
            if isinstance(scheduler, torch.optim.lr_scheduler.ReduceLROnPlateau):
                if cfg.model.cvae.beta_annealing:
                    # only step once beta annealing cycles are complete
                    if epoch >= cfg.model.cvae.beta_cycles * cfg.model.cvae.beta_period:
                        scheduler.step(avg_vloss)
                else:
                    scheduler.step(avg_vloss)
            else:
                scheduler.step()

        if beta_scheduler is not None:
            beta_scheduler.step()
        
        if rank == 0:
            # Save checkpoint on interval OR when validation improves (best epoch)
            # Use (epoch + 1) so interval=20 saves after epoch 19, 39, ... (i.e., every 20 epochs)
            should_save_chkpt = cfg.train.checkpoint.save_chkpt and (
                (epoch + 1) % cfg.train.checkpoint.save_chkpt_interval == 0 or
                (cfg.train.early_stopping and early_stopper.best)
            )
            if should_save_chkpt:
                # Save model.module so checkpoint is portable (no 'module.' prefix in keys)
                save_checkpoint(cfg.train.checkpoint.save_chkpt_path, model.module, optimizer, epoch,
                               scheduler=scheduler, early_stopper=early_stopper,
                               beta_scheduler=beta_scheduler)

            run.log({"learning_rate": scheduler.get_last_lr()[0] if scheduler is not None else cfg.train.lr}, step=epoch)

    # retrieve best metrics and weights
    # All ranks must load best weights for consistent distributed evaluation
    fmetrics = None
    model_weights = None
    if cfg.train.early_stopping:
        model_weights = early_stopper.get_best_weights()
        # All ranks load best weights so evaluate() and calculate_metrics() use consistent models
        model.module.load_state_dict(model_weights)

    if rank == 0:
        fmetrics = Metrics(use_partitions=False)
        if cfg.train.early_stopping:
            best_val_metrics = early_stopper.get_best_metrics()
            fmetrics.save_metrics('val', loss=early_stopper.get_min_validation_loss(), **best_val_metrics)
            run.summary[f"best_epoch"] = early_stopper.get_best_epoch()
            run.summary.update({'final model val loss (encoder-decoder)': best_val_metrics['val loss (encoder-decoder)']})
            run.summary.update({'final model val loss (inference)': best_val_metrics['val_recons_loss (*inference*)']})
        else:
            model_weights = model.module.state_dict()
            fmetrics.save_metrics('val', loss=avg_vloss, **val_metrics_dict)
            run.summary.update({'final model val loss (encoder-decoder)': avg_vloss})
            run.summary.update({'final model val loss (inference)': avg_vloss_inf})

    # for benchmarking purposes - all ranks participate in evaluate
    if cfg.eval.mode == 'test':
        # NOTE: will only use the most recent beta scheduler state for test loss (may be inaccurate)
        test_loss, _, test_metrics = evaluate(rank, world_size, model, test_loader, device, loss_fn, cfg, beta_scheduler=beta_scheduler)
        if rank == 0:
            fmetrics.save_metrics('test', loss=test_loss, **test_metrics)
            run.summary.update({'final model test loss (encoder-decoder)': test_metrics['test loss (encoder-decoder)']})
            run.summary.update({'final model test loss (inference)': test_metrics['test_recons_loss (*inference*)']})

    return model_weights, fmetrics

def calculate_metrics(rank, world_size, dataloader, homogenous_dataloader, train_mean, train_std, model, \
                      device, metrics, cfg, run):
    """Calculate SAR despeckling metrics for VV and VH:
    
    1. PSNR (peak signal-to-noise ratio)
    2. SSIM (structural similarity)
    3. ENL (speckle suppression)
    4. Variance of Laplacian (edge sharpness)

    NOTE: No-reference metric ENL is calculated on a small sample of homogenous
    regions picked from the val or test set.

    NOTE: PSNR and SSIM are computed on amplitude scale SAR data normalized to [0, 1].

    NOTE: Evaluation deterministic inference using z=0.

    Parameters
    ----------
    rank : int
        The rank of the current process.
    world_size : int
        The total number of processes.
    dataloader : torch.utils.data.DataLoader
        DataLoader for the dataset.
    homogenous_dataloader : torch.utils.data.DataLoader
        DataLoader for the homogenous regions.
    train_mean : torch.Tensor
        Mean of the training data.
    train_std : torch.Tensor
        Standard deviation of the training data.
    model : torch.nn.Module
        Model to evaluate.
    device : torch.device
        Device to evaluate on.
    metrics : Metrics
        Metrics object to save the metrics to.
    cfg : DictConfig
        Configuration object.
    run : wandb.Run
        WandB run object.

    Returns
    -------
    None
    """
    train_mean = train_mean.to(device)
    train_std = train_std.to(device)
    psnr_vv = torch.tensor(0.0, device=device, dtype=torch.float32)
    psnr_vh = torch.tensor(0.0, device=device, dtype=torch.float32)
    ssim_vv = torch.tensor(0.0, device=device, dtype=torch.float32)
    ssim_vh = torch.tensor(0.0, device=device, dtype=torch.float32)
    var_lap_vv = torch.tensor(0.0, device=device, dtype=torch.float32)
    var_lap_vh = torch.tensor(0.0, device=device, dtype=torch.float32)
    count = torch.tensor(0, device=device, dtype=torch.int64)

    model.eval()
    # Compute PSNR, SSIM, Var of Laplacian across entire dataset
    for X, y in dataloader:
        X = X.to(device)
        y = y.to(device)
        B, C, H, W = X.shape

        with torch.no_grad():
            out_dict = model.module.inference(X, deterministic=True)
            result = out_dict['despeckler_output']

            # convert back to dB scale
            # figure out how to denormalize here
            db_sar_filt = denormalize(result, train_mean, train_std)
            db_sar_clean = denormalize(y, train_mean, train_std)

            # convert dB to amplitude scale and normalize to [0, 1]
            amplitude_sar_filt_vv = normalize(convert_to_amplitude(db_sar_filt[:, 0]), vmin=amplitude_min_vv, vmax=amplitude_max_vv)
            amplitude_sar_clean_vv = normalize(convert_to_amplitude(db_sar_clean[:, 0]), vmin=amplitude_min_vv, vmax=amplitude_max_vv)
            amplitude_sar_filt_vh = normalize(convert_to_amplitude(db_sar_filt[:, 1]), vmin=amplitude_min_vh, vmax=amplitude_max_vh)
            amplitude_sar_clean_vh = normalize(convert_to_amplitude(db_sar_clean[:, 1]), vmin=amplitude_min_vh, vmax=amplitude_max_vh)

            # PSNR
            psnr_vv += psnr(amplitude_sar_filt_vv, amplitude_sar_clean_vv).sum()
            psnr_vh += psnr(amplitude_sar_filt_vh, amplitude_sar_clean_vh).sum()

            ### MAKE SURE TO SEND TO GPU FOR FAST SSIM!
            ssim_vv += ssim(amplitude_sar_filt_vv.unsqueeze(1),
                            amplitude_sar_clean_vv.unsqueeze(1),
                            data_range=1, size_average=False).sum()
            ssim_vh += ssim(amplitude_sar_filt_vh.unsqueeze(1),
                            amplitude_sar_clean_vh.unsqueeze(1),
                            data_range=1, size_average=False).sum()
            
            # Variance of Laplacian should be computed on dB scale sar data normalized to [0, 1]
            db_sar_filt_vv = normalize(db_sar_filt[:, 0], vmin=VV_DB_MIN, vmax=VV_DB_MAX)
            db_sar_filt_vh = normalize(db_sar_filt[:, 1], vmin=VH_DB_MIN, vmax=VH_DB_MAX)
            db_sar_filt_concat = torch.cat([db_sar_filt_vv.unsqueeze(1), db_sar_filt_vh.unsqueeze(1)], dim=1)
            batch_var_laplacian = var_laplacian(db_sar_filt_concat)
            var_lap_vv += batch_var_laplacian[:, 0].sum()
            var_lap_vh += batch_var_laplacian[:, 1].sum()

            count += B

    # ENL computed on separate homogenous regions set
    enl_vv = torch.tensor(0.0, device=device, dtype=torch.float32)
    enl_vh = torch.tensor(0.0, device=device, dtype=torch.float32)
    count_homogenous = torch.tensor(0, device=device, dtype=torch.int64)
    for X, y in homogenous_dataloader:
        X = X.to(device)
        y = y.to(device)
        B, C, H, W = X.shape

        with torch.no_grad():
            out_dict = model.module.inference(X, deterministic=True)
            result = out_dict['despeckler_output']

            # convert back to dB scale
            db_sar_filt = denormalize(result, train_mean, train_std)

            # convert to linear power scale
            power_sar_filt = torch.float_power(10, db_sar_filt / 10)

            # Calculate ENL here
            enl_vv += enl(power_sar_filt[:, 0]).sum()
            enl_vh += enl(power_sar_filt[:, 1]).sum()

            count_homogenous += B

    # synchronize all metrics across ranks
    dist.all_reduce(psnr_vv, op=dist.ReduceOp.SUM)
    dist.all_reduce(psnr_vh, op=dist.ReduceOp.SUM)
    dist.all_reduce(ssim_vv, op=dist.ReduceOp.SUM)
    dist.all_reduce(ssim_vh, op=dist.ReduceOp.SUM)
    dist.all_reduce(var_lap_vv, op=dist.ReduceOp.SUM)
    dist.all_reduce(var_lap_vh, op=dist.ReduceOp.SUM)
    dist.all_reduce(count, op=dist.ReduceOp.SUM)
    dist.all_reduce(enl_vv, op=dist.ReduceOp.SUM)
    dist.all_reduce(enl_vh, op=dist.ReduceOp.SUM)
    dist.all_reduce(count_homogenous, op=dist.ReduceOp.SUM)

    # Only rank 0 saves metrics and logs to wandb
    if rank == 0:
        count_val = count.item()
        count_homogenous_val = count_homogenous.item()
        metrics_dict = {
            f"{cfg.eval.mode} avg psnr_vv": psnr_vv.item() / count_val,
            f"{cfg.eval.mode} avg psnr_vh": psnr_vh.item() / count_val,
            f"{cfg.eval.mode} avg ssim_vv": ssim_vv.item() / count_val,
            f"{cfg.eval.mode} avg ssim_vh": ssim_vh.item() / count_val,
            f"{cfg.eval.mode} avg var_lap_vv": var_lap_vv.item() / count_val,
            f"{cfg.eval.mode} avg var_lap_vh": var_lap_vh.item() / count_val,
            f"{cfg.eval.mode} avg enl_vv": enl_vv.item() / count_homogenous_val,
            f"{cfg.eval.mode} avg enl_vh": enl_vh.item() / count_homogenous_val,
        }
        metrics.save_metrics(cfg.eval.mode, **metrics_dict)

        # log as wandb summary statistics
        run.summary.update({f'final model {key}': value for key, value in metrics_dict.items()})

def save_experiment(weights, metrics, cfg, run):
    """Save experiment files to directory specified by config save_path."""
    path = Path(cfg.save_path)
    path.mkdir(parents=True, exist_ok=True)

    if weights is not None:
        torch.save(weights, path / "CVAE_ad.pth")

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


def create_histogram_plot(composite_values, output_values, db_min, db_max, channel_name="SAR"):
    """
    Creates an overlaid histogram comparing composite (target) and despeckled output in dB scale.

    Parameters
    ----------
    composite_values : np.ndarray
        Flattened composite/target values in dB scale.
    output_values : np.ndarray
        Flattened despeckled output values in dB scale.
    db_min : float
        Minimum dB value for histogram range (e.g., -30 for VV/VH).
    db_max : float
        Maximum dB value for histogram range (e.g., 0 for VV, -5 for VH).
    channel_name : str
        Channel name for the title (e.g., "VV" or "VH").

    Note: resolution for wandb image is low (except for the matplotlib distribution plots).
    This is for efficiency sake. During final benchmarking, can use matplotlib fig with dpi=300
    for high res WandB image.
    """
    fig, ax = plt.subplots(figsize=(5, 4), dpi=200)
    ax.hist(composite_values, bins=50, alpha=0.5, range=(db_min, db_max), label="Composite", color="blue")
    ax.hist(output_values, bins=50, alpha=0.5, range=(db_min, db_max), label="Despeckled", color="orange")
    ax.legend()
    ax.set_xlabel("dB")
    ax.set_ylabel("Count")
    ax.set_title(f"{channel_name} dB Distribution")

    # Convert the figure to a WandB image
    wandb_image = wandb.Image(fig)
    plt.clf()
    plt.close(fig)  # Close figure to free memory
    return wandb_image

def sample_predictions(model, sample_set, mean, std, cfg, histogram=True, hist_freq=1, seed=24330):
    """Generate predictions on a subset of images in the dataset for wandb logging.
    
    Visualizes SAR VV and VH patches in dB grayscale in [-30, 0] and [-30, -5] for VV and VH respectively.
    All data is denormalized back to dB scale before visualization.
    
    Parameters
    ----------
    model : nn.Module
        The trained model.
    sample_set : Dataset
        Dataset to sample predictions from.
    mean : torch.Tensor
        Mean used for normalization (shape: [C]).
    std : torch.Tensor
        Std used for normalization (shape: [C]).
    cfg : DictConfig
        Configuration object.
    histogram : bool
        Whether to include histogram plots.
    hist_freq : int
        Frequency of histogram logging (1 = every sample).
    seed : int
        Random seed for reproducibility.
    """
    if cfg.wandb.num_sample_predictions <= 0:
        return None

    columns = ["id", 'input_vv', 'input_vh', 'despeckled_vv', 'despeckled_vh', 'composite_vv', 'composite_vh']

    if histogram:
        columns.extend(['vv_histogram', 'vh_histogram'])

    table = wandb.Table(columns=columns)
    
    # Set up colormaps with fixed dB ranges
    vv_map = ScalarMappable(norm=Normalize(vmin=VV_DB_MIN, vmax=VV_DB_MAX), cmap='gray')
    vh_map = ScalarMappable(norm=Normalize(vmin=VH_DB_MIN, vmax=VH_DB_MAX), cmap='gray')

    model.to('cpu')
    model.eval()
    rng = Random(seed)
    samples = rng.sample(range(0, len(sample_set)), cfg.wandb.num_sample_predictions)
    
    for index, k in enumerate(samples):
        X, y = sample_set[k]

        with torch.no_grad():
            out_dict = model.module.inference(X.unsqueeze(0))
            despeckler_output = out_dict['despeckler_output'].squeeze(0)

        # Denormalize all tensors back to dB scale
        # X: (C, H, W) -> denormalize -> (H, W, C) for visualization
        # y: (C, H, W) -> denormalize -> (H, W, C) for visualization
        # despeckler_output: (C, H, W) -> denormalize
        
        # Denormalize input (noisy SAR)
        X_db = X * std.view(-1, 1, 1) + mean.view(-1, 1, 1)
        X_db = X_db.permute(1, 2, 0).numpy()  # (H, W, C)
        
        # Denormalize composite/target
        y_db = y * std.view(-1, 1, 1) + mean.view(-1, 1, 1)
        y_db = y_db.permute(1, 2, 0).numpy()  # (H, W, C)
        
        # Denormalize despeckled output
        despeckled_db = despeckler_output * std.view(-1, 1, 1) + mean.view(-1, 1, 1)
        despeckled_db = despeckled_db.numpy()  # (C, H, W)

        row = [k]
        
        # Extract VV and VH channels (all in dB scale now)
        vv_input = X_db[:, :, 0]
        vv_despeckled = despeckled_db[0]
        vv_composite = y_db[:, :, 0]

        vh_input = X_db[:, :, 1]
        vh_despeckled = despeckled_db[1]
        vh_composite = y_db[:, :, 1]

        # VV input image
        vv_rgba = vv_map.to_rgba(vv_input, bytes=True)
        row.append(wandb.Image(Image.fromarray(vv_rgba, mode="RGBA")))

        # VH input image
        vh_rgba = vh_map.to_rgba(vh_input, bytes=True)
        row.append(wandb.Image(Image.fromarray(vh_rgba, mode="RGBA")))

        # Despeckled VV
        vv_despeckled_rgba = vv_map.to_rgba(vv_despeckled, bytes=True)
        row.append(wandb.Image(Image.fromarray(vv_despeckled_rgba, mode="RGBA")))

        # Despeckled VH
        vh_despeckled_rgba = vh_map.to_rgba(vh_despeckled, bytes=True)
        row.append(wandb.Image(Image.fromarray(vh_despeckled_rgba, mode="RGBA")))

        # Composite VV
        vv_composite_rgba = vv_map.to_rgba(vv_composite, bytes=True)
        row.append(wandb.Image(Image.fromarray(vv_composite_rgba, mode="RGBA")))

        # Composite VH
        vh_composite_rgba = vh_map.to_rgba(vh_composite, bytes=True)
        row.append(wandb.Image(Image.fromarray(vh_composite_rgba, mode="RGBA")))

        # Compare dB distributions (composite vs despeckled output)
        if histogram:
            if index % hist_freq == 0:
                # VV histogram in dB scale
                row.append(create_histogram_plot(
                    vv_composite.flatten(),
                    vv_despeckled.flatten(),
                    db_min=VV_DB_MIN,
                    db_max=VV_DB_MAX,
                    channel_name="VV"
                ))
                # VH histogram in dB scale
                row.append(create_histogram_plot(
                    vh_composite.flatten(),
                    vh_despeckled.flatten(),
                    db_min=VH_DB_MIN,
                    db_max=VH_DB_MAX,
                    channel_name="VH"
                ))
            else:
                row.extend([None, None])

        table.add_data(*row)

    return table

def sample_examples(model, sample_set, mean, std, cfg, idxs=[100, 200, 300]):
    """Generate curated examples for model qualitative analysis. Select example cases via dataset indices.
    
    Visualizes SAR VV and VH patches in dB grayscale with fixed ranges:
    - VV: [-30, 0] dB
    - VH: [-30, -5] dB
    
    Parameters
    ----------
    model : nn.Module
        The trained model.
    sample_set : Dataset
        Dataset to sample from.
    mean : torch.Tensor
        Mean used for normalization (shape: [C]).
    std : torch.Tensor
        Std used for normalization (shape: [C]).
    cfg : DictConfig
        Configuration object.
    idxs : list[int]
        Dataset indices for curated examples.
    """
    # Set up colormaps with fixed dB ranges
    vv_map = ScalarMappable(norm=Normalize(vmin=VV_DB_MIN, vmax=VV_DB_MAX), cmap='gray')
    vh_map = ScalarMappable(norm=Normalize(vmin=VH_DB_MIN, vmax=VH_DB_MAX), cmap='gray')
    
    model.to('cpu')
    model.eval()
    examples = []
    
    for k in idxs:
        X, y = sample_set[k]

        with torch.no_grad():
            out_dict = model.module.inference(X.unsqueeze(0))
            despeckler_output = out_dict['despeckler_output'].squeeze(0)

        # Denormalize all tensors back to dB scale
        # X: (C, H, W) -> denormalize
        X_db = X * std.view(-1, 1, 1) + mean.view(-1, 1, 1)
        X_db = X_db.permute(1, 2, 0).numpy()  # (H, W, C)
        
        # y: (C, H, W) -> denormalize
        y_db = y * std.view(-1, 1, 1) + mean.view(-1, 1, 1)
        y_db = y_db.permute(1, 2, 0).numpy()  # (H, W, C)
        
        # despeckler_output: (C, H, W) -> denormalize
        despeckled_db = despeckler_output * std.view(-1, 1, 1) + mean.view(-1, 1, 1)
        despeckled_db = despeckled_db.numpy()  # (C, H, W)

        # Extract VV and VH channels (all in dB scale)
        vv_input = X_db[:, :, 0]
        vv_despeckled = despeckled_db[0]
        vv_composite = y_db[:, :, 0]

        vh_input = X_db[:, :, 1]
        vh_despeckled = despeckled_db[1]
        vh_composite = y_db[:, :, 1]

        # Stitch VV images into vertical column (Input | Despeckled | Composite)
        stitched_vv = np.vstack([vv_input, vv_despeckled, vv_composite])
        stitched_vv_rgba = vv_map.to_rgba(stitched_vv, bytes=True)
        vv_img = Image.fromarray(stitched_vv_rgba, mode="RGBA")
        examples.append(wandb.Image(vv_img, caption=f"({k}VV) Top: In, Middle: Out, Bottom: Composite"))

        # Stitch VH images into vertical column (Input | Despeckled | Composite)
        stitched_vh = np.vstack([vh_input, vh_despeckled, vh_composite])
        stitched_vh_rgba = vh_map.to_rgba(stitched_vh, bytes=True)
        vh_img = Image.fromarray(stitched_vh_rgba, mode="RGBA")
        examples.append(wandb.Image(vh_img, caption=f"({k}VH) Top: In, Middle: Out, Bottom: Composite"))

    return examples

def init_distributed(rank, world_size, free_port):
    os.environ["MASTER_ADDR"] = "localhost"
    os.environ["MASTER_PORT"] = str(free_port)
    torch.cuda.set_device(rank)
    dist.init_process_group(backend="nccl", rank=rank, world_size=world_size)

def run_experiment_ad(rank, world_size, free_port, cfg):
    """Run a DDP S1 SAR autodespeckler model experiment given the configuration parameters.

    NOTE: For OmegaConf objects, pass in OmegaConf.to_container(cfg, resolve=True)
    for the cfg parameter.

    Parameters
    ----------
    rank : int
        The rank of the current process
    world_size : int
        The total number of processes
    free_port : int
        The free port for the DDP process
    cfg : object
        Config object for the SAR autodespeckler.
    
    Returns
    -------
    fmetrics : Metrics
        Metrics object containing the metrics for the experiment.
    """
    torch.set_num_threads(int(torch.get_num_threads() / world_size))
    cfg = OmegaConf.create(cfg)

    init_distributed(rank, world_size, free_port)

    if rank == 0 and not wandb.login():
        raise Exception("Failed to login to wandb.")

    # seeding with rank offset
    rank_seed = cfg.seed + rank
    np.random.seed(rank_seed)
    random.seed(rank_seed)
    torch.manual_seed(rank_seed)

    # device and model
    if torch.cuda.is_available():
        device = torch.device("cuda", rank)
    else:
        raise ValueError("CUDA is required for distributed training")

    # setup model with DDP
    model = DDP(build_autodespeckler(cfg).to(device), device_ids=[rank])

    # dataset and transforms
    print(f"Using {device} device")
    model_name = cfg.model.autodespeckler
    method = cfg.data.method
    size = cfg.data.size
    sample_param = cfg.data.samples if cfg.data.method == 'random' else cfg.data.stride
    suffix = getattr(cfg.data, 'suffix', '')
    if suffix:
        sample_dir = Path(cfg.paths.preprocess_dir) / 's1_multi' / f'{method}_{size}_{sample_param}_{suffix}/'
    else:
        sample_dir = Path(cfg.paths.preprocess_dir) / 's1_multi' / f'{method}_{size}_{sample_param}/'

    # load in mean and std
    with open(sample_dir / f'mean_std.pkl', 'rb') as f:
        train_mean, train_std = pickle.load(f)
        train_mean = torch.from_numpy(train_mean)
        train_std = torch.from_numpy(train_std)

    standardize = transforms.Compose([transforms.Normalize(train_mean, train_std)])

    # training set is large so must be sharded - partitioned by rank internally
    shards = sorted(sample_dir.glob("train_patches_[0-9]*.npy"))
    train_set = ShardedConditionalSARDataset(shards, cfg.data.block_size, rank, world_size,
                                        epoch=0, transform=standardize,
                                        random_flip=cfg.data.random_flip, seed=cfg.seed+1,
                                        mmap_mode='r' if cfg.data.mmap else None)
    # validation and test set are smaller, use DistributedSampler
    val_set = ConditionalSARDataset(sample_dir, typ="val", transform=standardize)
    test_set = ConditionalSARDataset(sample_dir, typ="test", transform=standardize) if cfg.eval.mode == 'test' else None

    # for homogenous SAR patch evaluation - also need DistributedSampler for distributed calculate_metrics
    val_homogenous_set = HomogenousSARDataset(sample_dir, typ="val", transform=standardize)
    test_homogenous_set = HomogenousSARDataset(sample_dir, typ="test", transform=standardize) if cfg.eval.mode == 'test' else None

    # dataloaders
    # Training set: ShardedConditionalSARDataset handles partitioning internally - no DistributedSampler needed
    train_loader = DataLoader(train_set,
                             batch_size=cfg.train.batch_size,
                             num_workers=cfg.train.num_workers,
                             persistent_workers=cfg.train.num_workers>0,
                             pin_memory=True,
                             multiprocessing_context='fork',
                             drop_last=False)

    # Val/test loaders use DistributedSampler with fork context to share data in memory
    val_loader = DataLoader(val_set,
                            batch_size=cfg.train.batch_size,
                            num_workers=cfg.train.num_workers,
                            persistent_workers=cfg.train.num_workers>0,
                            pin_memory=True,
                            sampler=DistributedSampler(val_set, shuffle=False, drop_last=False),
                            multiprocessing_context='fork',
                            shuffle=False)

    test_loader = DataLoader(test_set,
                            batch_size=cfg.train.batch_size,
                            num_workers=cfg.train.num_workers,
                            persistent_workers=cfg.train.num_workers>0,
                            pin_memory=True,
                            sampler=DistributedSampler(test_set, shuffle=False, drop_last=False),
                            multiprocessing_context='fork',
                            shuffle=False) if cfg.eval.mode == 'test' else None

    # Homogenous loaders also use DistributedSampler for distributed calculate_metrics
    val_homogenous_loader = DataLoader(val_homogenous_set,
                            batch_size=cfg.train.batch_size,
                            num_workers=0,
                            sampler=DistributedSampler(val_homogenous_set, shuffle=False, drop_last=False),
                            shuffle=False,
                            drop_last=False)
    
    test_homogenous_loader = DataLoader(test_homogenous_set,
                            batch_size=cfg.train.batch_size,
                            num_workers=0,
                            sampler=DistributedSampler(test_homogenous_set, shuffle=False, drop_last=False),
                            shuffle=False,
                            drop_last=False) if cfg.eval.mode == 'test' else None

    # initialize wandb run - only rank 0
    if rank == 0:
        total_params, trainable_params, param_size_in_mb = get_model_params(model)
        
        # convert config to flat dict for logging
        config_dict = flatten_dict(OmegaConf.to_container(cfg, resolve=True))
        run = wandb.init(
            project=cfg.wandb.project,
            group=cfg.wandb.group,
            config={
                "dataset": "Sentinel1-Multitemporal",
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
    else:
        run = None

    try:
        # setup save path - only rank 0
        if rank == 0 and cfg.save:
            if cfg.save_path is None:
                cfg.save_path = str(Path(cfg.paths.experiment_dir) / f"{datetime.today().strftime('%Y-%m-%d')}_{cfg.model.autodespeckler}_{run.id}/")
                run.config.update({"save_path": cfg.save_path}, allow_val_change=True)
            print(f'Save path set to: {cfg.save_path}')
        
        # synchronize ranks
        dist.barrier()

        # train and save results metrics - all ranks participate
        weights, fmetrics = train(rank, world_size, model, train_loader, val_loader, test_loader, device, cfg, run)

        # final val despeckling metrics - all ranks participate
        calculate_metrics(rank, world_size, val_loader, val_homogenous_loader,
                        train_mean, train_std, model, device, fmetrics, cfg, run)

        # final test despeckling metrics - all ranks participate
        if cfg.eval.mode == 'test':
            calculate_metrics(rank, world_size, test_loader, test_homogenous_loader,
                        train_mean, train_std, model, device, fmetrics, cfg, run)

        # Only rank 0: save experiment and sample predictions
        if rank == 0:
            if cfg.save:
                save_experiment(weights, fmetrics, cfg, run)

            # NOTE: no random sampling examples for train set as it is an iterable dataset
            # sample predictions for analysis
            pred_table = sample_predictions(model, test_set if cfg.eval.mode == 'test' else val_set,
                                            train_mean, train_std, cfg)

            # pick 3 full res examples for closer look
            examples = sample_examples(model, test_set if cfg.eval.mode == 'test' else val_set, 
                                       train_mean, train_std, cfg, idxs=[100, 200, 300])
            run.log({f"model_{cfg.eval.mode}_predictions": pred_table, "val_examples": examples})
    except Exception as e:
        print("An exception occurred during training!")

        # Send an alert in the W&B UI - only rank 0
        if rank == 0:
            run.alert(
                title="Training crashed 🚨",
                text=f"Run failed due to: {e}"
            )

            # Log to wandb summary
            run.summary["error"] = str(e)

        raise e
    finally:
        if rank == 0:
            run.finish()
        dist.destroy_process_group()

    return fmetrics

def validate_config(cfg):
    # Add checks
    assert cfg.model.autodespeckler == 'CVAE', "Model must be CVAE"
    assert cfg.save in [True, False], "Save must be a boolean"
    assert cfg.train.batch_size is not None and cfg.train.batch_size > 0, "Batch size must be defined and positive"
    assert cfg.train.lr > 0, "Learning rate must be positive"
    assert cfg.train.loss in AD_LOSS_NAMES, f"Loss must be one of {AD_LOSS_NAMES}"
    assert cfg.train.optimizer in ['Adam', 'SGD'], f"Optimizer must be one of {['Adam', 'SGD']}"
    assert cfg.train.LR_scheduler in SCHEDULER_NAMES, f"LR scheduler must be one of {SCHEDULER_NAMES}"
    assert cfg.train.early_stopping in [True, False], "Early stopping must be a boolean"
    assert not cfg.train.early_stopping or cfg.train.patience is not None, "Patience must be set if early stopping is enabled"
    assert cfg.data.random_flip in [True, False], "Random flip must be a boolean"
    assert cfg.eval.mode in ['val', 'test'], f"Evaluation mode must be one of {['val', 'test']}"
    assert cfg.wandb.project is not None, "Wandb project must be specified"

@hydra.main(version_base=None, config_path="pkg://configs", config_name="config.yaml")
def main(cfg: DictConfig):
    validate_config(cfg)
    # resolve cfgs before pickling
    resolved_cfg = OmegaConf.to_container(cfg, resolve=True)

    world_size = torch.cuda.device_count()
    free_port = find_free_port()
    print(f"world_size = {world_size}")
    print(f"Found free port: {free_port}")
    mp.spawn(run_experiment_ad, args=(world_size, free_port, resolved_cfg), nprocs=world_size)

if __name__ == '__main__':
    main()
