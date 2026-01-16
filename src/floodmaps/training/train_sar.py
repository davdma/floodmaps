import wandb
import torch
import logging
from datetime import datetime
from torch import nn
from torch.utils.data import DataLoader
from torchvision import transforms
from torchmetrics import MetricCollection
from torchmetrics.classification import (BinaryAccuracy, BinaryPrecision,
                            BinaryRecall, BinaryF1Score, BinaryConfusionMatrix,
                            BinaryJaccardIndex, BinaryAveragePrecision)
from matplotlib.cm import ScalarMappable
from matplotlib.colors import Normalize
import random
from random import Random
from PIL import Image
import numpy as np
import pickle
import json
from pathlib import Path
import hydra
from omegaconf import DictConfig, OmegaConf

from floodmaps.models.model import SARWaterDetector
from floodmaps.utils.utils import (flatten_dict, Metrics, EarlyStopper,
                         SARChannelIndexer, get_model_params, nlcd_to_rgb,
                         get_samples_with_wet_percentage, scl_to_rgb, compute_pos_weight,
                         align_patches_with_shifts)
from floodmaps.utils.checkpoint import save_checkpoint, load_checkpoint
from floodmaps.utils.metrics import (PerClassConfusionMatrix, compute_confmat_dict,
                        NLCD_CLASSES, NLCD_GROUPS, SCL_CLASSES, SCL_GROUPS, RunningMeanVar)

from floodmaps.training.loss import SARLossConfig
from floodmaps.training.dataset import FloodSampleSARDataset
from floodmaps.training.optim import get_optimizer
from floodmaps.training.scheduler import get_scheduler

MODEL_NAMES = ['unet', 'unet++']
AUTODESPECKLER_NAMES = ['CNN1', 'CNN2', 'DAE', 'VAE']
NOISE_NAMES = ['normal', 'masking', 'log_gamma']
LOSS_NAMES = ['BCELoss', 'BCEDiceLoss', 'TverskyLoss', 'FocalTverskyLoss']

# get our optimizer and metrics
def train_loop(model, dataloader, device, optimizer, minibatches, loss_config,
                cfg, ad_cfg, c, run, epoch):
    running_tot_loss = torch.tensor(0.0, device=device) # all loss components
    running_cls_loss = torch.tensor(0.0, device=device) # only classifier loss
    if ad_cfg is not None:
        running_recons_loss = torch.tensor(0.0, device=device)

        # for VAE monitoring
        if ad_cfg.model.autodespeckler == 'VAE':
            running_kld_loss = torch.tensor(0.0, device=device)
            mu_mean_var = RunningMeanVar(unbiased=True).to(device)
            log_var_mean_var = RunningMeanVar(unbiased=True).to(device)

    metric_collection = MetricCollection([
        BinaryAccuracy(threshold=0.5),
        BinaryPrecision(threshold=0.5),
        BinaryRecall(threshold=0.5),
        BinaryF1Score(threshold=0.5),
        BinaryJaccardIndex(threshold=0.5)
    ]).to(device)

    model.train()
    for batch_i, (X, y, _) in enumerate(dataloader):
        if batch_i >= minibatches:
            break

        X = X.to(device)
        y = y.to(device)

        # crop central window
        X_c = X[:, :, c[0]:c[1], c[0]:c[1]]
        out_dict = model(X_c)

        # also pass SAR layers for reconstruction loss
        loss_dict = loss_config.compute_loss(out_dict, y.float(), typ='train',
                                            shift_invariant=cfg.train.shift_invariant)
        loss = loss_dict['total_loss']
        y_true = loss_dict['true_label']

        optimizer.zero_grad()
        loss.backward()
        optimizer.step()

        logits = out_dict['classifier_output']
        y_pred = nn.functional.sigmoid(logits).flatten() > 0.5
        target = y_true.flatten() > 0.5

        metric_collection.update(y_pred, target)
        running_tot_loss += loss.detach()
        running_cls_loss += loss_dict['classifier_loss'].detach()

        if ad_cfg is not None:
            running_recons_loss += loss_dict['recons_loss'].detach()

            # for VAE monitoring only
            if ad_cfg.model.autodespeckler == 'VAE':
                # Collect mu and log_var for the whole epoch
                mu_mean_var.update(out_dict['mu'])
                log_var_mean_var.update(out_dict['log_var'])
                running_kld_loss += loss_dict['kld_loss'].detach()

    metric_results = metric_collection.compute()
    epoch_loss = running_tot_loss.item() / minibatches

    # wandb tracking loss and metrics per epoch - track recons loss as well
    log_dict = {"train accuracy": metric_results['BinaryAccuracy'].item(),
                "train precision": metric_results['BinaryPrecision'].item(),
                "train recall": metric_results['BinaryRecall'].item(),
                "train f1": metric_results['BinaryF1Score'].item(),
                "train IoU": metric_results['BinaryJaccardIndex'].item(),
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
        ad_loss_percentage = (cfg.train.balance_coeff * epoch_ad_loss / running_tot_loss.item())
        log_dict['ad_loss_percentage'] = ad_loss_percentage

        # VAE mu and log_var monitoring
        if ad_cfg.model.autodespeckler == 'VAE':
            mu_mean_var_results = mu_mean_var.compute()
            log_var_mean_var_results = log_var_mean_var.compute()

            # kld loss as percentage of total ad loss
            kld_loss_percentage = (ad_cfg.model.vae.VAE_beta
                                   * running_kld_loss.item()
                                   / epoch_ad_loss)
            log_dict.update({
                "train_mu_mean": mu_mean_var_results['mean'].item(),
                "train_mu_std": mu_mean_var_results['var'].sqrt().item(),
                "train_log_var_mean": log_var_mean_var_results['mean'].item(),
                "train_log_var_std": log_var_mean_var_results['var'].sqrt().item(),
                "train_kld_loss": cfg.train.balance_coeff * running_kld_loss.item() / minibatches,
                "train_kld_loss_percentage": kld_loss_percentage,
                "beta": ad_cfg.model.vae.VAE_beta
            })
    run.log(log_dict, step=epoch)
    metric_collection.reset()
    if ad_cfg is not None and ad_cfg.model.autodespeckler == 'VAE':
        mu_mean_var.reset()
        log_var_mean_var.reset()

    return epoch_loss

def test_loop(model, dataloader, device, loss_config, cfg, ad_cfg, c, run, epoch):
    running_tot_vloss = torch.tensor(0.0, device=device) # all loss components
    running_cls_vloss = torch.tensor(0.0, device=device) # only classifier loss
    if ad_cfg is not None:
        running_recons_vloss = torch.tensor(0.0, device=device)

        # for VAE monitoring
        if ad_cfg.model.autodespeckler == 'VAE':
            running_kld_vloss = torch.tensor(0.0, device=device)
            mu_mean_var = RunningMeanVar(unbiased=True).to(device)
            log_var_mean_var = RunningMeanVar(unbiased=True).to(device)

    num_batches = len(dataloader)
    metric_collection = MetricCollection([
        BinaryAccuracy(threshold=0.5),
        BinaryPrecision(threshold=0.5),
        BinaryRecall(threshold=0.5),
        BinaryF1Score(threshold=0.5),
        BinaryConfusionMatrix(threshold=0.5),
        BinaryJaccardIndex(threshold=0.5),
        BinaryAveragePrecision(thresholds=None)
    ]).to(device)
    nlcd_metric_collection = PerClassConfusionMatrix(threshold=0.5, classes=NLCD_CLASSES).to(device)
    scl_metric_collection = PerClassConfusionMatrix(threshold=0.5, classes=SCL_CLASSES).to(device)

    window_size = cfg.data.window
    model.eval()
    with torch.no_grad():
        for X, y, supplementary in dataloader:
            X = X.to(device)
            y = y.to(device)
            # for nlcd data we can safely assume it is properly aligned to the SAR image
            nlcd_classes = supplementary[:, 3, c[0]:c[1], c[0]:c[1]].to(device)
            # SCL is S2 product so we align
            scl_classes_wide = supplementary[:, 4, :, :].to(device)

            X_c = X[:, :, c[0]:c[1], c[0]:c[1]]

            out_dict = model(X_c)
            loss_dict = loss_config.compute_loss(out_dict, y.float(), typ='val',
                                                shift_invariant=cfg.train.shift_invariant)
            loss = loss_dict['total_loss']
            y_true = loss_dict['true_label']
            
            # Align SCL using shift indices from loss computation
            row_shifts, col_shifts = loss_dict['shift_indices']
            scl_classes = align_patches_with_shifts(
                scl_classes_wide, row_shifts, col_shifts, window_size, window_size
            )

            logits = out_dict['classifier_output']
            y_pred = nn.functional.sigmoid(logits).flatten()
            target = y_true.flatten() > 0.5

            metric_collection.update(y_pred, target)
            nlcd_metric_collection.update(y_pred, target, nlcd_classes.flatten())
            scl_metric_collection.update(y_pred, target, scl_classes.flatten())
            running_tot_vloss += loss.detach()
            running_cls_vloss += loss_dict['classifier_loss'].detach()

            if ad_cfg is not None:
                running_recons_vloss += loss_dict['recons_loss'].detach()

                # for VAE monitoring only
                if ad_cfg.model.autodespeckler == 'VAE':
                    mu_mean_var.update(out_dict['mu'])
                    log_var_mean_var.update(out_dict['log_var'])
                    running_kld_vloss += loss_dict['kld_loss'].detach()

    metric_results = metric_collection.compute()
    nlcd_metrics_results = nlcd_metric_collection.compute()
    scl_metrics_results = scl_metric_collection.compute()
    epoch_vloss = running_tot_vloss.item() / num_batches
    epoch_cls_vloss = running_cls_vloss.item() / num_batches

    core_metrics_dict = {
        "val accuracy": metric_results['BinaryAccuracy'].item(),
        "val precision": metric_results['BinaryPrecision'].item(),
        "val recall": metric_results['BinaryRecall'].item(),
        "val f1": metric_results['BinaryF1Score'].item(),
        "val IoU": metric_results['BinaryJaccardIndex'].item(),
        "val AUPRC": metric_results['BinaryAveragePrecision'].item()
    }
    
    log_dict = core_metrics_dict.copy()
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
        ad_loss_percentage = (cfg.train.balance_coeff * epoch_ad_vloss / running_tot_vloss.item())
        log_dict['val_ad_loss_percentage'] = ad_loss_percentage

        # VAE mu and log_var monitoring
        if ad_cfg.model.autodespeckler == 'VAE':
            mu_mean_var_results = mu_mean_var.compute()
            log_var_mean_var_results = log_var_mean_var.compute()

            # kld loss as percentage of total ad loss
            kld_loss_percentage = (ad_cfg.model.vae.VAE_beta
                                   * running_kld_vloss.item()
                                   / epoch_ad_vloss)
            log_dict.update({
                "val_mu_mean": mu_mean_var_results['mean'].item(),
                "val_mu_std": mu_mean_var_results['var'].sqrt().item(),
                "val_log_var_mean": log_var_mean_var_results['mean'].item(),
                "val_log_var_std": log_var_mean_var_results['var'].sqrt().item(),
                "val_kld_loss": cfg.train.balance_coeff * running_kld_vloss.item() / num_batches,
                "val_kld_loss_percentage": kld_loss_percentage
            })

    run.log(log_dict, step=epoch)
    
    confusion_matrix = metric_results['BinaryConfusionMatrix'].tolist()
    confusion_matrix_dict = {
        "tn": confusion_matrix[0][0],
        "fp": confusion_matrix[0][1],
        "fn": confusion_matrix[1][0],
        "tp": confusion_matrix[1][1]
    }
    nlcd_metrics_dict = compute_confmat_dict(nlcd_metrics_results,
                                            nlcd_metric_collection.get_class_to_idx(),
                                            NLCD_CLASSES, groups=NLCD_GROUPS)
    scl_metrics_dict = compute_confmat_dict(scl_metrics_results,
                                            scl_metric_collection.get_class_to_idx(),
                                            SCL_CLASSES, groups=SCL_GROUPS)
    metric_collection.reset()
    nlcd_metric_collection.reset()
    scl_metric_collection.reset()
    if ad_cfg is not None and ad_cfg.model.autodespeckler == 'VAE':
        mu_mean_var.reset()
        log_var_mean_var.reset()

    metrics_dict = {
        'core_metrics': core_metrics_dict,
        'confusion_matrix': confusion_matrix_dict,
        'nlcd_metrics': nlcd_metrics_dict,
        'scl_metrics': scl_metrics_dict
    }

    return epoch_vloss, epoch_cls_vloss, metrics_dict

def evaluate(model, dataloader, test_aligned_loader, device, loss_config, cfg, ad_cfg, c):
    """Evaluate metrics on test set without logging.
    Computes both shift-invariant and non-shift-invariant loss and
    metrics.
    
    If shift ablation patches are available, aligned metrics are computed as well.
    """
    running_tot_shift_vloss = torch.tensor(0.0, device=device)
    running_tot_non_shift_vloss = torch.tensor(0.0, device=device)

    num_batches = len(dataloader)
    
    # Shift-invariant metric collections
    shift_metric_collection = MetricCollection([
        BinaryAccuracy(threshold=0.5),
        BinaryPrecision(threshold=0.5),
        BinaryRecall(threshold=0.5),
        BinaryF1Score(threshold=0.5),
        BinaryConfusionMatrix(threshold=0.5),
        BinaryJaccardIndex(threshold=0.5),
        BinaryAveragePrecision(thresholds=None)
    ]).to(device)
    shift_nlcd_metric_collection = PerClassConfusionMatrix(threshold=0.5, classes=NLCD_CLASSES).to(device)
    shift_scl_metric_collection = PerClassConfusionMatrix(threshold=0.5, classes=SCL_CLASSES).to(device)
    
    # Non-shift-invariant metric collections
    non_shift_metric_collection = MetricCollection([
        BinaryAccuracy(threshold=0.5),
        BinaryPrecision(threshold=0.5),
        BinaryRecall(threshold=0.5),
        BinaryF1Score(threshold=0.5),
        BinaryConfusionMatrix(threshold=0.5),
        BinaryJaccardIndex(threshold=0.5),
        BinaryAveragePrecision(thresholds=None)
    ]).to(device)
    non_shift_nlcd_metric_collection = PerClassConfusionMatrix(threshold=0.5, classes=NLCD_CLASSES).to(device)
    non_shift_scl_metric_collection = PerClassConfusionMatrix(threshold=0.5, classes=SCL_CLASSES).to(device)

    # Test aligned patch metrics
    if test_aligned_loader is not None:
        aligned_metric_collection = MetricCollection([
            BinaryAccuracy(threshold=0.5),
            BinaryPrecision(threshold=0.5),
            BinaryRecall(threshold=0.5),
            BinaryF1Score(threshold=0.5),
            BinaryConfusionMatrix(threshold=0.5),
            BinaryJaccardIndex(threshold=0.5),
            BinaryAveragePrecision(thresholds=None)
        ]).to(device)
        aligned_nlcd_metric_collection = PerClassConfusionMatrix(threshold=0.5, classes=NLCD_CLASSES).to(device)
        aligned_scl_metric_collection = PerClassConfusionMatrix(threshold=0.5, classes=SCL_CLASSES).to(device)

    window_size = cfg.data.window
    model.eval()
    with torch.no_grad():
        for X, y, supplementary in dataloader:
            X = X.to(device)
            y = y.to(device)
            nlcd_classes = supplementary[:, 3, c[0]:c[1], c[0]:c[1]].to(device)
            scl_classes_wide = supplementary[:, 4, :, :].to(device)

            X_c = X[:, :, c[0]:c[1], c[0]:c[1]]

            out_dict = model(X_c)
            logits = out_dict['classifier_output']
            y_pred_probs = nn.functional.sigmoid(logits).flatten()
            
            # Compute shift-invariant loss and metrics
            shift_loss_dict = loss_config.compute_loss(out_dict, y.float(), typ='test', shift_invariant=True)
            shift_loss = shift_loss_dict['total_loss']
            y_true_shift = shift_loss_dict['true_label']
            
            # Align SCL using shift indices from loss computation
            row_shifts, col_shifts = shift_loss_dict['shift_indices']
            scl_classes_shift = align_patches_with_shifts(
                scl_classes_wide, row_shifts, col_shifts, window_size, window_size
            )
            
            target_shift = y_true_shift.flatten() > 0.5
            shift_metric_collection.update(y_pred_probs, target_shift)
            shift_nlcd_metric_collection.update(y_pred_probs, target_shift, nlcd_classes.flatten())
            shift_scl_metric_collection.update(y_pred_probs, target_shift, scl_classes_shift.flatten())
            
            # Compute non-shift-invariant loss and metrics
            non_shift_loss_dict = loss_config.compute_loss(out_dict, y.float(), typ='test', shift_invariant=False)
            non_shift_loss = non_shift_loss_dict['total_loss']
            y_true_non_shift = non_shift_loss_dict['true_label']
            
            # Align SCL using center indices for non-shift
            row_shifts_ns, col_shifts_ns = non_shift_loss_dict['shift_indices']
            scl_classes_non_shift = align_patches_with_shifts(
                scl_classes_wide, row_shifts_ns, col_shifts_ns, window_size, window_size
            )
            
            target_non_shift = y_true_non_shift.flatten() > 0.5
            non_shift_metric_collection.update(y_pred_probs, target_non_shift)
            non_shift_nlcd_metric_collection.update(y_pred_probs, target_non_shift, nlcd_classes.flatten())
            non_shift_scl_metric_collection.update(y_pred_probs, target_non_shift, scl_classes_non_shift.flatten())
            
            running_tot_shift_vloss += shift_loss.detach()
            running_tot_non_shift_vloss += non_shift_loss.detach()
    
    # for shift ablation performance
    if test_aligned_loader is not None:
        with torch.no_grad():
            for X, y, supplementary in test_aligned_loader:
                X = X.to(device)
                y = y.to(device)
                nlcd_classes = supplementary[:, 3, :, :].to(device)
                scl_classes = supplementary[:, 4, :, :].to(device)

                out_dict = model(X)
                logits = out_dict['classifier_output']
                y_pred_probs = nn.functional.sigmoid(logits).flatten()
                target = y.flatten() > 0.5
                aligned_metric_collection.update(y_pred_probs, target)
                aligned_nlcd_metric_collection.update(y_pred_probs, target, nlcd_classes.flatten())
                aligned_scl_metric_collection.update(y_pred_probs, target, scl_classes.flatten())

    # Compute shift-invariant metrics
    shift_metric_results = shift_metric_collection.compute()
    shift_nlcd_metrics_results = shift_nlcd_metric_collection.compute()
    shift_scl_metrics_results = shift_scl_metric_collection.compute()
    
    # Compute non-shift-invariant metrics  
    non_shift_metric_results = non_shift_metric_collection.compute()
    non_shift_nlcd_metrics_results = non_shift_nlcd_metric_collection.compute()
    non_shift_scl_metrics_results = non_shift_scl_metric_collection.compute()

    # Compute aligned patch metrics
    if test_aligned_loader is not None:
        aligned_metric_results = aligned_metric_collection.compute()
        aligned_nlcd_metrics_results = aligned_nlcd_metric_collection.compute()
        aligned_scl_metrics_results = aligned_scl_metric_collection.compute()
    
    epoch_shift_vloss = running_tot_shift_vloss.item() / num_batches
    epoch_non_shift_vloss = running_tot_non_shift_vloss.item() / num_batches

    # Build shift-invariant core metrics dict
    shift_core_metrics_dict = {
        "test accuracy": shift_metric_results['BinaryAccuracy'].item(),
        "test precision": shift_metric_results['BinaryPrecision'].item(),
        "test recall": shift_metric_results['BinaryRecall'].item(),
        "test f1": shift_metric_results['BinaryF1Score'].item(),
        "test IoU": shift_metric_results['BinaryJaccardIndex'].item(),
        "test AUPRC": shift_metric_results['BinaryAveragePrecision'].item()
    }
    
    # Build non-shift-invariant core metrics dict
    non_shift_core_metrics_dict = {
        "test accuracy": non_shift_metric_results['BinaryAccuracy'].item(),
        "test precision": non_shift_metric_results['BinaryPrecision'].item(),
        "test recall": non_shift_metric_results['BinaryRecall'].item(),
        "test f1": non_shift_metric_results['BinaryF1Score'].item(),
        "test IoU": non_shift_metric_results['BinaryJaccardIndex'].item(),
        "test AUPRC": non_shift_metric_results['BinaryAveragePrecision'].item()
    }

    # Build aligned patch core metrics dict
    aligned_core_metrics_dict = {
        "test accuracy": aligned_metric_results['BinaryAccuracy'].item(),
        "test precision": aligned_metric_results['BinaryPrecision'].item(),
        "test recall": aligned_metric_results['BinaryRecall'].item(),
        "test f1": aligned_metric_results['BinaryF1Score'].item(),
        "test IoU": aligned_metric_results['BinaryJaccardIndex'].item(),
        "test AUPRC": aligned_metric_results['BinaryAveragePrecision'].item()
    } if test_aligned_loader is not None else {}
    
    # Build shift-invariant confusion matrix and per-class metrics
    shift_confusion_matrix = shift_metric_results['BinaryConfusionMatrix'].tolist()
    shift_confusion_matrix_dict = {
        "tn": shift_confusion_matrix[0][0],
        "fp": shift_confusion_matrix[0][1],
        "fn": shift_confusion_matrix[1][0],
        "tp": shift_confusion_matrix[1][1]
    }
    shift_nlcd_metrics_dict = compute_confmat_dict(shift_nlcd_metrics_results,
                                            shift_nlcd_metric_collection.get_class_to_idx(),
                                            NLCD_CLASSES, groups=NLCD_GROUPS)
    shift_scl_metrics_dict = compute_confmat_dict(shift_scl_metrics_results,
                                            shift_scl_metric_collection.get_class_to_idx(),
                                            SCL_CLASSES, groups=SCL_GROUPS)
    
    # Build non-shift-invariant confusion matrix and per-class metrics
    non_shift_confusion_matrix = non_shift_metric_results['BinaryConfusionMatrix'].tolist()
    non_shift_confusion_matrix_dict = {
        "tn": non_shift_confusion_matrix[0][0],
        "fp": non_shift_confusion_matrix[0][1],
        "fn": non_shift_confusion_matrix[1][0],
        "tp": non_shift_confusion_matrix[1][1]
    }
    non_shift_nlcd_metrics_dict = compute_confmat_dict(non_shift_nlcd_metrics_results,
                                            non_shift_nlcd_metric_collection.get_class_to_idx(),
                                            NLCD_CLASSES, groups=NLCD_GROUPS)
    non_shift_scl_metrics_dict = compute_confmat_dict(non_shift_scl_metrics_results,
                                            non_shift_scl_metric_collection.get_class_to_idx(),
                                            SCL_CLASSES, groups=SCL_GROUPS)

    # Build aligned patch confusion matrix and per-class metrics
    if test_aligned_loader is not None:
        aligned_confusion_matrix = aligned_metric_results['BinaryConfusionMatrix'].tolist()
        aligned_confusion_matrix_dict = {
            "tn": aligned_confusion_matrix[0][0],
            "fp": aligned_confusion_matrix[0][1],
            "fn": aligned_confusion_matrix[1][0],
            "tp": aligned_confusion_matrix[1][1]
        }
        aligned_nlcd_metrics_dict = compute_confmat_dict(aligned_nlcd_metrics_results,
                                                aligned_nlcd_metric_collection.get_class_to_idx(),
                                                NLCD_CLASSES, groups=NLCD_GROUPS)
        aligned_scl_metrics_dict = compute_confmat_dict(aligned_scl_metrics_results,
                                                aligned_scl_metric_collection.get_class_to_idx(),
                                                SCL_CLASSES, groups=SCL_GROUPS)
    
    # Reset all metric collections
    shift_metric_collection.reset()
    shift_nlcd_metric_collection.reset()
    shift_scl_metric_collection.reset()
    non_shift_metric_collection.reset()
    non_shift_nlcd_metric_collection.reset()
    non_shift_scl_metric_collection.reset()
    if test_aligned_loader is not None:
        aligned_metric_collection.reset()
        aligned_nlcd_metric_collection.reset()
        aligned_scl_metric_collection.reset()

    # Return both shift-invariant and non-shift-invariant metrics
    shift_metrics_dict = {
        'core_metrics': shift_core_metrics_dict,
        'confusion_matrix': shift_confusion_matrix_dict,
        'nlcd_metrics': shift_nlcd_metrics_dict,
        'scl_metrics': shift_scl_metrics_dict
    }
    non_shift_metrics_dict = {
        'core_metrics': non_shift_core_metrics_dict,
        'confusion_matrix': non_shift_confusion_matrix_dict,
        'nlcd_metrics': non_shift_nlcd_metrics_dict,
        'scl_metrics': non_shift_scl_metrics_dict
    }
    aligned_metrics_dict = {
        'core_metrics': aligned_core_metrics_dict,
        'confusion_matrix': aligned_confusion_matrix_dict,
        'nlcd_metrics': aligned_nlcd_metrics_dict,
        'scl_metrics': aligned_scl_metrics_dict
    } if test_aligned_loader is not None else {}

    return epoch_shift_vloss, epoch_non_shift_vloss, shift_metrics_dict, non_shift_metrics_dict, aligned_metrics_dict

def train(model, train_loader, val_loader, test_loader, test_aligned_loader, device, cfg, ad_cfg, run, cache_dir=None):
    timestamp = datetime.now().strftime('%Y-%m-%d %H:%M:%S')
    logging.info(f'''Starting training:
        Date:            {timestamp}
        Epochs:          {cfg.train.epochs}
        Batch size:      {cfg.train.batch_size}
        Learning rate:   {cfg.train.lr}
        Training size:   {len(train_loader.dataset)}
        Validation size: {len(val_loader.dataset)}
        Test size:       {len(test_loader.dataset) if test_loader is not None else 'NA'}
        Test aligned size: {len(test_aligned_loader.dataset) if test_aligned_loader is not None else 'NA'}
        Device:          {device}
    ''')
    # log weights and gradients each epoch
    run.watch(model, log="all", log_freq=cfg.logging.grad_norm_freq)

    # compute pos_weight if enabled for rebalancing
    pos_weight_val = None
    if getattr(cfg.train, 'use_pos_weight', False):
        if getattr(cfg.train, 'pos_weight', None) is not None:
            pos_weight_val = float(cfg.train.pos_weight)
        else:
            # Efficient vectorized computation over loaded training labels
            # label plane is last 6th channel in dataset
            label_np = train_loader.dataset.dataset[:, -6, :, :]
            clip_max = float(getattr(cfg.train, 'pos_weight_clip', 10.0))
            pos_weight_val = compute_pos_weight(label_np, pos_weight_clip=clip_max,
                                                cache_dir=cache_dir, dataset_name='train')
    cfg.train.pos_weight = pos_weight_val
    run.config.update({"train.pos_weight": pos_weight_val}, allow_val_change=True)
    
    # initialize loss functions - train loss function is optimized for gradient calculations
    loss_cfg = SARLossConfig(cfg, ad_cfg=ad_cfg, device=device)

    # optimizer and scheduler for reducing learning rate
    optimizer = get_optimizer(model, cfg)
    scheduler = get_scheduler(optimizer, cfg)
    early_stopper = EarlyStopper(patience=cfg.train.patience) if cfg.train.early_stopping else None

    # load checkpoint if it exists
    if cfg.train.checkpoint.load_chkpt:
        chkpt = load_checkpoint(cfg.train.checkpoint.load_chkpt_path, model, optimizer=optimizer, scheduler=scheduler, early_stopper=early_stopper)
        start_epoch = chkpt['epoch'] + 1
        if cfg.train.epochs < start_epoch:
            raise ValueError(f"Epochs specified in config ({cfg.train.epochs}) is less than the epoch at which the checkpoint was saved ({start_epoch}).")
    else:
        start_epoch = 0

    ignore_ad_loss = (ad_cfg is not None
                and cfg.model.autodespeckler.freeze
                and cfg.model.autodespeckler.freeze_epochs >= cfg.train.epochs)

    minibatches = int(len(train_loader) * cfg.train.subset)
    center_1 = (cfg.data.size - cfg.data.window) // 2
    center_2 = center_1 + cfg.data.window
    c = (center_1, center_2)
    for epoch in range(start_epoch, cfg.train.epochs):
        try:
            if (ad_cfg is not None
                and cfg.model.autodespeckler.freeze
                and epoch == cfg.model.autodespeckler.freeze_epochs):
                print(f"Unfreezing backbone at epoch {epoch}")
                model.unfreeze_ad_weights()

            # train loop
            avg_loss = train_loop(model, train_loader, device, optimizer, minibatches,
                                  loss_cfg, cfg, ad_cfg, c, run, epoch)

            # at the end of each training epoch compute validation
            avg_vloss, avg_cls_vloss, val_set_metrics = test_loop(model, val_loader, device,
                                    loss_cfg, cfg, ad_cfg, c, run, epoch)
        except Exception as err:
            raise RuntimeError(f'Error while training occurred at epoch {epoch}.') from err

        if cfg.train.early_stopping:
            # use only classifier component for early stopping if AD is frozen
            early_stopper.step(avg_cls_vloss if ignore_ad_loss else avg_vloss, model, epoch, metrics=val_set_metrics)
            if early_stopper.is_stopped():
                break

        if scheduler is not None:
            if isinstance(scheduler, torch.optim.lr_scheduler.ReduceLROnPlateau):
                scheduler.step(avg_vloss)
            else:
                scheduler.step()
        
        # Save checkpoint on interval OR when validation improves (best epoch)
        # Use (epoch + 1) so interval=20 saves after epoch 19, 39, ... (i.e., every 20 epochs)
        should_save_chkpt = cfg.train.checkpoint.save_chkpt and (
            (epoch + 1) % cfg.train.checkpoint.save_chkpt_interval == 0 or
            (cfg.train.early_stopping and early_stopper.best)
        )
        if should_save_chkpt:
            save_checkpoint(cfg.train.checkpoint.save_chkpt_path, model, optimizer, epoch, scheduler=scheduler, early_stopper=early_stopper)

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
        run.summary.update({f'final model {key}': value for key, value in best_val_metrics['core_metrics'].items()})
        run.summary[f"best_epoch"] = early_stopper.get_best_epoch()
    else:
        cls_weights = model.classifier.state_dict()
        ad_weights = (model.autodespeckler.state_dict()
                      if model.uses_autodespeckler() else None)
        fmetrics.save_metrics('val', partition=partition,
                              loss=avg_vloss,
                              **val_set_metrics)
        run.summary.update({f'final model {key}': value for key, value in val_set_metrics['core_metrics'].items()})

    # for benchmarking purposes
    if cfg.eval.mode == 'test':
        # benchmark both non shift and shift invariant metrics
        shift_test_loss, non_shift_test_loss, shift_test_set_metrics, non_shift_test_set_metrics, aligned_test_set_metrics = evaluate(model, test_loader, test_aligned_loader, device, loss_cfg, cfg, ad_cfg, c)
        fmetrics.save_metrics('test', partition="shift_invariant", loss=shift_test_loss, **shift_test_set_metrics)
        fmetrics.save_metrics('test', partition="non_shift_invariant", loss=non_shift_test_loss, **non_shift_test_set_metrics)
        run.summary.update({f'final model shift {key}': value for key, value in shift_test_set_metrics['core_metrics'].items()})
        run.summary.update({f'final model non shift {key}': value for key, value in non_shift_test_set_metrics['core_metrics'].items()})
        if test_aligned_loader is not None:
            fmetrics.save_metrics('test', partition="aligned", **aligned_test_set_metrics)
            run.summary.update({f'final model aligned {key}': value for key, value in aligned_test_set_metrics['core_metrics'].items()})

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
    with open(path / "config.yaml", "w") as f:
        OmegaConf.save(cfg, f)

    if ad_cfg is not None:
        with open(path / "ad_config.yaml", "w") as f:
            OmegaConf.save(ad_cfg, f)

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

def sample_predictions(model, sample_set, mean, std, cfg, ad_cfg, sample_dir, dataset_name, percent_wet_patches=0.5, seed=24330):
    """Generate predictions on a subset of images in the dataset for wandb logging.

    NOTE: For visualization the ground truth label is shifted if shift invariance used.
    Other channels like TCI, NLCD, SCL are not shifted and kept in the wider format.
    
    Parameters
    ----------
    model : torch.nn.Module
        The model to evaluate
    sample_set : torch.utils.data.Dataset
        The dataset to sample predictions from
    mean : torch.Tensor
        The mean of the dataset
    std : torch.Tensor
        The standard deviation of the dataset
    cfg : DictConfig
        The configuration dictionary
    ad_cfg : DictConfig
        The configuration dictionary for the autodespeckler
    sample_dir : Path or str
        Directory where samples are stored (used for caching wet/dry indices)
    dataset_name : str
        Name of the dataset ('val' or 'test') for cache file naming
    percent_wet_patches : float, optional
        The percentage of wet patches to visualize
    seed : int, optional
        The seed for the random number generator
    """
    if cfg.wandb.num_sample_predictions <= 0:
        return None

    loss_config = SARLossConfig(cfg, ad_cfg=ad_cfg, device='cpu')
    columns = ["id", "tci", "nlcd", "scl"] # TCI, NLCD, SCL always included
    channels = [bool(int(x)) for x in cfg.data.channels]
    my_channels = SARChannelIndexer(channels)
    # initialize wandb table given the channel settings
    columns += my_channels.get_display_channels()
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
    channel_indices = [-1] * 8
    for i, channel in enumerate(channels):
        if channel:
            channel_indices[i] = n
            n += 1

    model.to('cpu')
    model.eval()
    rng = Random(seed)
    
    # Get samples with specified percentage of wet patches (with caching)
    samples = get_samples_with_wet_percentage(sample_set,
        cfg.wandb.num_sample_predictions,
        cfg.train.batch_size,
        cfg.train.num_workers,
        percent_wet_patches,
        rng,
        cache_dir=sample_dir,
        dataset_name=dataset_name
    )

    center_1 = (cfg.data.size - cfg.data.window) // 2
    center_2 = center_1 + cfg.data.window
    window_size = cfg.data.window
    for id, k in enumerate(samples):
        # get all images to shape (H, W, C) with C = 1 or 3 (1 for grayscale, 3 for RGB)
        X, y, supplementary = sample_set[k]

        X_c = X[:, center_1:center_2, center_1:center_2]

        with torch.no_grad():
            out_dict = model(X_c.unsqueeze(0))
            logits = out_dict['classifier_output']
            despeckler_output = out_dict['despeckler_output'].squeeze(0) if model.uses_autodespeckler() else None
            y_shifted, shift_indices = loss_config.get_label_alignment(logits, y.unsqueeze(0).float(),
                                                                    shift_invariant=cfg.train.shift_invariant)
            y_shifted = y_shifted.squeeze(0)

        y_pred = torch.where(nn.functional.sigmoid(logits) > 0.5, 1.0, 0.0).squeeze(0) # (1, x, y)

        # Compute false positives
        fp = torch.logical_and(y_shifted == 0, y_pred == 1).squeeze(0).byte().mul(255).numpy()

        # Compute false negatives
        fn = torch.logical_and(y_shifted == 1, y_pred == 0).squeeze(0).byte().mul(255).numpy()

        # Channels are descaled using linear variance scaling
        X_c = X_c.permute(1, 2, 0)
        X_c = std * X_c + mean

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

        # SCL (Scene Classification Layer) visualization
        scl = supplementary[4, :, :].byte().numpy()
        scl_rgb = scl_to_rgb(scl)
        scl_img = Image.fromarray(scl_rgb, mode="RGB")
        row.append(wandb.Image(scl_img))

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
        if my_channels.has_flowlines():
            flowlines = X_c[:, :, channel_indices[7]].mul(255).clamp(0, 255).byte().numpy()
            flowlines_img = Image.fromarray(flowlines, mode="L")
            row.append(wandb.Image(flowlines_img))

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
    
    Returns
    -------
    fmetrics : Metrics
        Metrics object containing the metrics for the experiment.
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
    method = cfg.data.method
    size = cfg.data.size
    sample_param = cfg.data.samples if cfg.data.method == 'random' else cfg.data.stride
    suffix = getattr(cfg.data, 'suffix', '')
    if suffix:
        sample_dir = Path(cfg.paths.preprocess_dir) / 's1_weak' / f'{method}_{size}_{sample_param}_{suffix}/'
    else:
        sample_dir = Path(cfg.paths.preprocess_dir) / 's1_weak' / f'{method}_{size}_{sample_param}/'

    # load in mean and std
    channels = [bool(int(x)) for x in cfg.data.channels]
    with open(sample_dir / f'mean_std.pkl', 'rb') as f:
        train_mean, train_std = pickle.load(f)

        train_mean = torch.from_numpy(train_mean[channels])
        train_std = torch.from_numpy(train_std[channels])

    standardize = transforms.Compose([transforms.Normalize(train_mean, train_std)])

    # datasets (all can be memory mapped if desired)
    train_set = FloodSampleSARDataset(sample_dir, channels=channels,
                                        typ="train", transform=standardize, random_flip=cfg.data.random_flip,
                                        seed=cfg.seed+1, mmap_mode='r' if cfg.data.mmap else None,
                                        keep_contiguous_in_mem=getattr(cfg.train, 'keep_contiguous_in_mem', True))
    val_set = FloodSampleSARDataset(sample_dir, channels=channels,
                                    typ="val", transform=standardize)
    test_set = FloodSampleSARDataset(sample_dir, channels=channels,
                                        typ="test", transform=standardize) if cfg.eval.mode == 'test' else None

    # Only if manually aligned test patches available
    # Use cfg.data.shift_ablation = True
    # NOTE: See preprocess/shift_ablation.py
    is_shift_ablation_available = getattr(cfg.data, 'shift_ablation', False)
    if cfg.eval.mode == 'test' and is_shift_ablation_available:
        test_aligned_set = FloodSampleSARDataset(sample_dir, channels=channels,
                                            typ="test_shift_ablation", transform=standardize)
        # Verify shift ablation patch size matches expected window size
        _, _, H, W = test_aligned_set.dataset.shape
        expected_size = cfg.data.window
        if H != expected_size or W != expected_size:
            raise ValueError(
                f"Shift ablation patches have size {H}x{W} but model expects {expected_size}x{expected_size}. "
                f"Ensure cfg.preprocess.size matches cfg.data.window when preprocessing shift ablation patches."
            )
    else:
        test_aligned_set = None

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
    
    if cfg.eval.mode == 'test' and is_shift_ablation_available:
        # We expect small set of test patches, so num_workers = 0 is fine
        test_aligned_loader = DataLoader(test_aligned_set,
                            batch_size=cfg.train.batch_size,
                            num_workers=0,
                            pin_memory=True,
                            shuffle=False,
                            drop_last=False)
    else:
        test_aligned_loader = None

    # initialize wandb run
    total_params, trainable_params, param_size_in_mb = get_model_params(model)
    
    # convert config to flat dict for logging
    config_dict = flatten_dict(OmegaConf.to_container(cfg, resolve=True))
    run = wandb.init(
        project=cfg.wandb.project,
        group=cfg.wandb.group,
        config={
            "dataset": "Sentinel1",
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
        cls_weights, ad_weights, fmetrics = train(model, train_loader, val_loader,
                                              test_loader, test_aligned_loader, device, cfg, ad_cfg, run, cache_dir=sample_dir)

        if cfg.save:
            save_experiment(cls_weights, ad_weights, fmetrics, cfg, ad_cfg, run)

        # log predictions on validation set using wandb
        percent_wet = cfg.wandb.get('percent_wet_patches', 0.5)  # Default to 0.5 if not specified
        pred_table = sample_predictions(model, test_set if cfg.eval.mode == 'test' else val_set,
                                        train_mean, train_std, cfg, ad_cfg, sample_dir, cfg.eval.mode,
                                        percent_wet_patches=percent_wet)
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
        return type(s) == str and len(s) == 8 and all(c in '01' for c in s)

    # Add checks
    assert cfg.data.method in ['random', 'strided'], "Sampling method must be one of ['random', 'strided']"
    assert cfg.save in [True, False], "Save must be a boolean"
    if cfg.train.loss == 'TverskyLoss':
        assert 0.0 <= cfg.train.tversky.alpha <= 1.0, "Tversky alpha must be in [0, 1]"
    if cfg.train.loss == 'FocalTverskyLoss':
        assert 0.0 <= cfg.train.tversky.alpha <= 1.0, "Focal Tversky alpha must be in [0, 1]"
        assert 1 <= cfg.train.focal_tversky.gamma <= 3, "Focal Tversky gamma must be in [1, 3]"
    assert cfg.train.lr > 0, "Learning rate must be positive"
    assert cfg.train.loss in LOSS_NAMES, f"Loss must be one of {LOSS_NAMES}"
    assert cfg.train.optimizer in ['Adam', 'SGD', 'AdamW'], f"Optimizer must be one of {['Adam', 'SGD', 'AdamW']}"
    assert cfg.train.LR_scheduler in ['Constant', 'ReduceLROnPlateau', 'CosAnnealingLR'], f"LR scheduler must be one of {['Constant', 'ReduceLROnPlateau', 'CosAnnealingLR']}"
    assert cfg.train.early_stopping in [True, False], "Early stopping must be a boolean"
    assert not cfg.train.early_stopping or cfg.train.patience is not None, "Patience must be set if early stopping is enabled"
    assert cfg.train.batch_size is not None and cfg.train.batch_size > 0, "Batch size must be defined and positive"
    assert cfg.model.classifier in MODEL_NAMES, f"Model must be one of {MODEL_NAMES}"
    assert cfg.data.random_flip in [True, False], "Random flip must be a boolean"
    assert cfg.eval.mode in ['val', 'test'], f"Evaluation mode must be one of {['val', 'test']}"
    assert cfg.wandb.project is not None, "Wandb project must be specified"
    assert validate_channels(cfg.data.channels), "Channels must be a binary string of length 8"

@hydra.main(version_base=None, config_path="pkg://configs", config_name="config.yaml")
def main(cfg: DictConfig):
    validate_config(cfg)
    ad_cfg = getattr(cfg, 'ad', None)
    run_experiment_s1(cfg, ad_cfg=ad_cfg)

if __name__ == '__main__':
    main()
