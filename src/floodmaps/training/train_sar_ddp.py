import wandb
import torch
import logging
import math
from datetime import datetime
from torch import nn
from torch.utils.data import DataLoader
from torchvision import transforms
from torchmetrics import MetricCollection
from torchmetrics.classification import (BinaryAccuracy, BinaryPrecision,
                                        BinaryRecall, BinaryF1Score,
                                        BinaryConfusionMatrix, BinaryJaccardIndex,
                                        BinaryAveragePrecision)
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
import os
import torch.multiprocessing as mp
import torch.distributed as dist
from torch.nn.parallel import DistributedDataParallel as DDP
from torch.utils.data.distributed import DistributedSampler

from floodmaps.models.model import SARWaterDetector
from floodmaps.utils.utils import (flatten_dict, Metrics, EarlyStopper,
                         SARChannelIndexer, get_model_params, nlcd_to_rgb,
                         get_samples_with_wet_percentage, scl_to_rgb, compute_pos_weight,
                         align_patches_with_shifts, find_free_port)
from floodmaps.utils.checkpoint import save_checkpoint, load_checkpoint
from floodmaps.utils.metrics import (PerClassConfusionMatrix, PerGroupAUPRC,
                        compute_confmat_dict,
                        NLCD_CLASSES, NLCD_GROUPS, SCL_CLASSES, SCL_GROUPS)

from floodmaps.training.loss import SARLossConfig
from floodmaps.training.dataset import FloodSampleSARDataset
from floodmaps.training.optim import get_optimizer, get_optimizer_with_ad
from floodmaps.training.scheduler import get_scheduler

MODEL_NAMES = ['unet', 'unet++']
LOSS_NAMES = ['BCELoss', 'BCEDiceLoss', 'TverskyLoss', 'FocalTverskyLoss']

# SAR dB scale ranges for visualization (consistent with multitemporal training scripts)
VV_DB_MIN, VV_DB_MAX = -30, 0
VH_DB_MIN, VH_DB_MAX = -30, -5

def tensor_to_json_safe(val):
    """Convert tensor to JSON-safe Python value, replacing nan with None.
    
    JSON does not support NaN values, so we convert them to None which
    serializes to null. This is used for metrics that may be undefined
    when no samples exist for a particular class or group.
    """
    v = val.item()
    return None if math.isnan(v) else v

def get_ad_sample_dir(ad_cfg, paths_cfg):
    """Get the sample directory for the autodespeckler dataset (s1_multi).
    
    This function constructs the path to the autodespeckler's training dataset
    directory based on the ad_cfg.data parameters. Used to load the despeckler's
    normalization statistics (mean_std.pkl) for mixed normalization when using
    pretrained despeckler weights.
    
    Parameters
    ----------
    ad_cfg : DictConfig
        Autodespeckler configuration containing data.method, data.size, 
        data.samples/data.stride, and optional data.suffix.
    paths_cfg : DictConfig
        Paths configuration containing preprocess_dir.
    
    Returns
    -------
    Path
        Path to the autodespeckler sample directory (s1_multi).
    """
    method = ad_cfg.data.method
    size = ad_cfg.data.size
    sample_param = ad_cfg.data.samples if method == 'random' else ad_cfg.data.stride
    suffix = getattr(ad_cfg.data, 'suffix', '')
    if suffix:
        return Path(paths_cfg.preprocess_dir) / 's1_multi' / f'{method}_{size}_{sample_param}_{suffix}/'
    else:
        return Path(paths_cfg.preprocess_dir) / 's1_multi' / f'{method}_{size}_{sample_param}/'

# get our optimizer and metrics
def train_loop(rank, world_size, model, dataloader, device, optimizer, loss_config,
                cfg, ad_cfg, c, run, epoch):
    """NOTE: The training loss is not used besides weight updates, therefore
    it is not all reduced across ranks. The returned loss is only the average
    of the losses on the local rank."""
    running_loss = torch.tensor(0.0, device=device)

    metric_collection = MetricCollection([
        BinaryAccuracy(threshold=0.5),
        BinaryPrecision(threshold=0.5),
        BinaryRecall(threshold=0.5),
        BinaryF1Score(threshold=0.5),
        BinaryJaccardIndex(threshold=0.5)
    ]).to(device)

    model.train()
    for batch_i, (X, y, _) in enumerate(dataloader):

        X = X.to(device)
        y = y.to(device)

        # crop central window
        X_c = X[:, :, c[0]:c[1], c[0]:c[1]]
        out_dict = model(X_c)

        # also pass SAR layers for reconstruction loss
        loss_dict = loss_config.compute_loss(out_dict, y.float(), typ='train',
                                            shift_invariant=cfg.train.shift_invariant)
        loss = loss_dict['loss']
        y_true = loss_dict['true_label']

        optimizer.zero_grad()
        loss.backward()
        optimizer.step()

        logits = out_dict['classifier_output']
        y_pred = nn.functional.sigmoid(logits).flatten() > 0.5
        target = y_true.flatten() > 0.5

        metric_collection.update(y_pred, target)
        running_loss += loss.detach()

    metric_results = metric_collection.compute()
    epoch_loss = running_loss.item() / len(dataloader)

    # wandb tracking loss and metrics per epoch
    log_dict = {"train accuracy": metric_results['BinaryAccuracy'].item(),
                "train precision": metric_results['BinaryPrecision'].item(),
                "train recall": metric_results['BinaryRecall'].item(),
                "train f1": metric_results['BinaryF1Score'].item(),
                "train IoU": metric_results['BinaryJaccardIndex'].item(),
                "train loss": epoch_loss}

    if rank == 0:
        run.log(log_dict, step=epoch)

    metric_collection.reset()

    return epoch_loss

def test_loop(rank, world_size, model, dataloader, device, loss_config, cfg, ad_cfg, c, run, epoch):
    """Validation losses are all reduced across ranks."""
    running_vloss = torch.tensor(0.0, device=device)

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
            loss = loss_dict['loss']
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
            running_vloss += loss.detach()

    metric_results = metric_collection.compute()
    nlcd_metrics_results = nlcd_metric_collection.compute()
    scl_metrics_results = scl_metric_collection.compute()

    # synchronize all validation losses across ranks
    dist.all_reduce(running_vloss, op=dist.ReduceOp.SUM)
    running_vloss /= world_size

    epoch_vloss = running_vloss.item() / num_batches

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
        "val loss": epoch_vloss
    })

    if rank == 0:
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
        metrics_dict = {
            'core_metrics': core_metrics_dict,
            'confusion_matrix': confusion_matrix_dict,
            'nlcd_metrics': nlcd_metrics_dict,
            'scl_metrics': scl_metrics_dict
        }
    else:
        metrics_dict = {}

    metric_collection.reset()
    nlcd_metric_collection.reset()
    scl_metric_collection.reset()

    return epoch_vloss, metrics_dict

def evaluate(model, dataloader, test_aligned_loader, device, loss_config, cfg, ad_cfg, c):
    """Evaluate metrics on test set without logging.
    Computes both shift-invariant and non-shift-invariant loss and
    metrics. Should only be called by rank 0.
    
    If shift ablation patches are available, aligned metrics are computed as well.
    """
    running_tot_shift_vloss = torch.tensor(0.0, device=device)
    running_tot_non_shift_vloss = torch.tensor(0.0, device=device)

    # Shift distribution counting - compute shift window dimensions from config
    shift_range = cfg.data.size - cfg.data.window + 1  # 5 for 68/64
    # Use flat tensor for efficient bincount accumulation
    shift_counts = torch.zeros(shift_range * shift_range, dtype=torch.long, device=device)

    num_batches = len(dataloader)
    
    # Shift-invariant metric collections
    shift_metric_collection = MetricCollection([
        BinaryAccuracy(threshold=0.5, sync_on_compute=False),
        BinaryPrecision(threshold=0.5, sync_on_compute=False),
        BinaryRecall(threshold=0.5, sync_on_compute=False),
        BinaryF1Score(threshold=0.5, sync_on_compute=False),
        BinaryConfusionMatrix(threshold=0.5, sync_on_compute=False),
        BinaryJaccardIndex(threshold=0.5, sync_on_compute=False),
        BinaryAveragePrecision(thresholds=None, sync_on_compute=False)
    ]).to(device)
    shift_nlcd_metric_collection = PerClassConfusionMatrix(threshold=0.5, classes=NLCD_CLASSES, sync_on_compute=False).to(device)
    shift_scl_metric_collection = PerClassConfusionMatrix(threshold=0.5, classes=SCL_CLASSES, sync_on_compute=False).to(device)
    shift_nlcd_auprc = PerGroupAUPRC(groups=NLCD_GROUPS, sync_on_compute=False).to(device)
    shift_scl_auprc = PerGroupAUPRC(groups=SCL_GROUPS, sync_on_compute=False).to(device)
    
    # Non-shift-invariant metric collections
    non_shift_metric_collection = MetricCollection([
        BinaryAccuracy(threshold=0.5, sync_on_compute=False),
        BinaryPrecision(threshold=0.5, sync_on_compute=False),
        BinaryRecall(threshold=0.5, sync_on_compute=False),
        BinaryF1Score(threshold=0.5, sync_on_compute=False),
        BinaryConfusionMatrix(threshold=0.5, sync_on_compute=False),
        BinaryJaccardIndex(threshold=0.5, sync_on_compute=False),
        BinaryAveragePrecision(thresholds=None, sync_on_compute=False)
    ]).to(device)
    non_shift_nlcd_metric_collection = PerClassConfusionMatrix(threshold=0.5, classes=NLCD_CLASSES, sync_on_compute=False).to(device)
    non_shift_scl_metric_collection = PerClassConfusionMatrix(threshold=0.5, classes=SCL_CLASSES, sync_on_compute=False).to(device)
    non_shift_nlcd_auprc = PerGroupAUPRC(groups=NLCD_GROUPS, sync_on_compute=False).to(device)
    non_shift_scl_auprc = PerGroupAUPRC(groups=SCL_GROUPS, sync_on_compute=False).to(device)

    # Test aligned patch metrics
    if test_aligned_loader is not None:
        aligned_metric_collection = MetricCollection([
            BinaryAccuracy(threshold=0.5, sync_on_compute=False),
            BinaryPrecision(threshold=0.5, sync_on_compute=False),
            BinaryRecall(threshold=0.5, sync_on_compute=False),
            BinaryF1Score(threshold=0.5, sync_on_compute=False),
            BinaryConfusionMatrix(threshold=0.5, sync_on_compute=False),
            BinaryJaccardIndex(threshold=0.5, sync_on_compute=False),
            BinaryAveragePrecision(thresholds=None, sync_on_compute=False)
        ]).to(device)
        aligned_nlcd_metric_collection = PerClassConfusionMatrix(threshold=0.5, classes=NLCD_CLASSES, sync_on_compute=False).to(device)
        aligned_scl_metric_collection = PerClassConfusionMatrix(threshold=0.5, classes=SCL_CLASSES, sync_on_compute=False).to(device)
        aligned_nlcd_auprc = PerGroupAUPRC(groups=NLCD_GROUPS, sync_on_compute=False).to(device)
        aligned_scl_auprc = PerGroupAUPRC(groups=SCL_GROUPS, sync_on_compute=False).to(device)

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
            shift_loss = shift_loss_dict['loss']
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
            shift_nlcd_auprc.update(y_pred_probs, target_shift, nlcd_classes.flatten())
            shift_scl_auprc.update(y_pred_probs, target_shift, scl_classes_shift.flatten())
            
            # Vectorized shift counting using bincount
            flat_indices = row_shifts * shift_range + col_shifts
            shift_counts += torch.bincount(flat_indices, minlength=shift_range * shift_range)
            
            # Compute non-shift-invariant loss and metrics
            non_shift_loss_dict = loss_config.compute_loss(out_dict, y.float(), typ='test', shift_invariant=False)
            non_shift_loss = non_shift_loss_dict['loss']
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
            non_shift_nlcd_auprc.update(y_pred_probs, target_non_shift, nlcd_classes.flatten())
            non_shift_scl_auprc.update(y_pred_probs, target_non_shift, scl_classes_non_shift.flatten())
            
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
                aligned_nlcd_auprc.update(y_pred_probs, target, nlcd_classes.flatten())
                aligned_scl_auprc.update(y_pred_probs, target, scl_classes.flatten())
            

    # Compute shift-invariant metrics
    shift_metric_results = shift_metric_collection.compute()
    shift_nlcd_metrics_results = shift_nlcd_metric_collection.compute()
    shift_scl_metrics_results = shift_scl_metric_collection.compute()
    shift_nlcd_auprc_results = shift_nlcd_auprc.compute()
    shift_scl_auprc_results = shift_scl_auprc.compute()
    
    # Compute non-shift-invariant metrics  
    non_shift_metric_results = non_shift_metric_collection.compute()
    non_shift_nlcd_metrics_results = non_shift_nlcd_metric_collection.compute()
    non_shift_scl_metrics_results = non_shift_scl_metric_collection.compute()
    non_shift_nlcd_auprc_results = non_shift_nlcd_auprc.compute()
    non_shift_scl_auprc_results = non_shift_scl_auprc.compute()

    # Compute aligned patch metrics
    if test_aligned_loader is not None:
        aligned_metric_results = aligned_metric_collection.compute()
        aligned_nlcd_metrics_results = aligned_nlcd_metric_collection.compute()
        aligned_scl_metrics_results = aligned_scl_metric_collection.compute()
        aligned_nlcd_auprc_results = aligned_nlcd_auprc.compute()
        aligned_scl_auprc_results = aligned_scl_auprc.compute()
    
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
    shift_nlcd_metrics_dict['group_auprc'] = {
        name: tensor_to_json_safe(val) for name, val in shift_nlcd_auprc_results.items()
    }
    shift_scl_metrics_dict = compute_confmat_dict(shift_scl_metrics_results,
                                            shift_scl_metric_collection.get_class_to_idx(),
                                            SCL_CLASSES, groups=SCL_GROUPS)
    shift_scl_metrics_dict['group_auprc'] = {
        name: tensor_to_json_safe(val) for name, val in shift_scl_auprc_results.items()
    }
    
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
    non_shift_nlcd_metrics_dict['group_auprc'] = {
        name: tensor_to_json_safe(val) for name, val in non_shift_nlcd_auprc_results.items()
    }
    non_shift_scl_metrics_dict = compute_confmat_dict(non_shift_scl_metrics_results,
                                            non_shift_scl_metric_collection.get_class_to_idx(),
                                            SCL_CLASSES, groups=SCL_GROUPS)
    non_shift_scl_metrics_dict['group_auprc'] = {
        name: tensor_to_json_safe(val) for name, val in non_shift_scl_auprc_results.items()
    }

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
        aligned_nlcd_metrics_dict['group_auprc'] = {
            name: tensor_to_json_safe(val) for name, val in aligned_nlcd_auprc_results.items()
        }
        aligned_scl_metrics_dict = compute_confmat_dict(aligned_scl_metrics_results,
                                                aligned_scl_metric_collection.get_class_to_idx(),
                                                SCL_CLASSES, groups=SCL_GROUPS)
        aligned_scl_metrics_dict['group_auprc'] = {
            name: tensor_to_json_safe(val) for name, val in aligned_scl_auprc_results.items()
        }
    
    # Reset all metric collections
    shift_metric_collection.reset()
    shift_nlcd_metric_collection.reset()
    shift_scl_metric_collection.reset()
    shift_nlcd_auprc.reset()
    shift_scl_auprc.reset()
    non_shift_metric_collection.reset()
    non_shift_nlcd_metric_collection.reset()
    non_shift_scl_metric_collection.reset()
    non_shift_nlcd_auprc.reset()
    non_shift_scl_auprc.reset()
    if test_aligned_loader is not None:
        aligned_metric_collection.reset()
        aligned_nlcd_metric_collection.reset()
        aligned_scl_metric_collection.reset()
        aligned_nlcd_auprc.reset()
        aligned_scl_auprc.reset()


    # Build shift distribution dict from accumulated counts
    shift_counts_2d = shift_counts.reshape(shift_range, shift_range)
    shift_distribution = {}
    for i in range(shift_range):
        for j in range(shift_range):
            # shift_count_[row]_[col] format
            shift_distribution[f"shift_count_{i}_{j}"] = int(shift_counts_2d[i, j].item())
    shift_distribution["shift_range"] = shift_range

    # Return both shift-invariant and non-shift-invariant metrics
    shift_metrics_dict = {
        'core_metrics': shift_core_metrics_dict,
        'confusion_matrix': shift_confusion_matrix_dict,
        'nlcd_metrics': shift_nlcd_metrics_dict,
        'scl_metrics': shift_scl_metrics_dict,
        'shift_distribution': shift_distribution
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

def train(rank, world_size, model, train_loader, val_loader, test_loader, test_aligned_loader,
        device, cfg, ad_cfg, run, cache_dir=None):
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
            Test aligned size: {len(test_aligned_loader.dataset) if test_aligned_loader is not None else 'NA'}
            Device:          {device}
        ''')
        run.watch(model.module, log="all", log_freq=cfg.logging.grad_norm_freq)

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
    if rank == 0:
        run.config.update({"train.pos_weight": pos_weight_val}, allow_val_change=True)
    
    # initialize loss functions - train loss function is optimized for gradient calculations
    loss_cfg = SARLossConfig(cfg, ad_cfg=ad_cfg, device=device)

    # optimizer and scheduler for reducing learning rate
    # Use separate learning rates for classifier and autodespeckler if ad_cfg present
    if ad_cfg is not None:
        optimizer = get_optimizer_with_ad(model.module, cfg, ad_cfg)
    else:
        optimizer = get_optimizer(model, cfg)
    scheduler = get_scheduler(optimizer, cfg)
    early_stopper = EarlyStopper(patience=cfg.train.patience) if cfg.train.early_stopping else None

    # load checkpoint if it exists
    # Load into model.module so checkpoints are portable (compatible with non-DDP checkpoints)
    if cfg.train.checkpoint.load_chkpt:
        chkpt = load_checkpoint(cfg.train.checkpoint.load_chkpt_path, model.module, optimizer=optimizer, scheduler=scheduler, early_stopper=early_stopper)
        start_epoch = chkpt['epoch'] + 1
        if cfg.train.epochs < start_epoch:
            raise ValueError(f"Epochs specified in config ({cfg.train.epochs}) is less than the epoch at which the checkpoint was saved ({start_epoch}).")
    else:
        start_epoch = 0

    center_1 = (cfg.data.size - cfg.data.window) // 2
    center_2 = center_1 + cfg.data.window
    c = (center_1, center_2)

    for epoch in range(start_epoch, cfg.train.epochs):
        train_loader.sampler.set_epoch(epoch)
        try:
            # Handle unfreezing autodespeckler at specified epoch
            if (ad_cfg is not None
                and ad_cfg.train.freeze
                and epoch == max(start_epoch, ad_cfg.train.freeze_epochs)):
                if ad_cfg.train.unfreeze_decoder_only:
                    print(f"Unfreezing autodespeckler decoder only at epoch {epoch} from rank {rank}")
                    model.module.unfreeze_ad_decoder_weights()
                else:
                    print(f"Unfreezing entire autodespeckler at epoch {epoch} from rank {rank}")
                    model.module.unfreeze_ad_weights()

            # train loop
            avg_loss = train_loop(rank, world_size, model, train_loader, device, optimizer,
                                  loss_cfg, cfg, ad_cfg, c, run, epoch)

            # at the end of each training epoch compute validation
            avg_vloss, val_set_metrics = test_loop(rank, world_size, model, val_loader, device,
                                    loss_cfg, cfg, ad_cfg, c, run, epoch)
        except Exception as err:
            raise RuntimeError(f'Error while training occurred at epoch {epoch}.') from err

        if cfg.train.early_stopping:
            # Pass model.module to early_stopper so weights are saved without 'module.' prefix
            early_stopper.step(avg_vloss, model.module, epoch, metrics=val_set_metrics)

            # Synchronize early stopping decision across all ranks
            should_stop = torch.tensor([1 if early_stopper.is_stopped() else 0], dtype=torch.int, device=device)
            dist.all_reduce(should_stop, op=dist.ReduceOp.MAX)
            if should_stop.item() == 1:
                break

        if scheduler is not None:
            if isinstance(scheduler, torch.optim.lr_scheduler.ReduceLROnPlateau):
                scheduler.step(avg_vloss)
            else:
                scheduler.step()
        
        if rank == 0:
            # Save checkpoint on interval OR when validation improves (best epoch)
            # Use (epoch + 1) so interval=20 saves after epoch 19, 39, ... (i.e., every 20 epochs)
            should_save_chkpt = cfg.train.checkpoint.save_chkpt and (
                (epoch + 1) % cfg.train.checkpoint.save_chkpt_interval == 0 or
                (cfg.train.early_stopping and early_stopper.best)
            )
            if should_save_chkpt:
                # Save model.module so checkpoint is portable (no 'module.' prefix in keys)
                save_checkpoint(cfg.train.checkpoint.save_chkpt_path, model.module, optimizer, epoch, scheduler=scheduler, early_stopper=early_stopper)

            lr_log = {"learning_rate": scheduler.get_last_lr()[0] if scheduler is not None else cfg.train.lr}
            if ad_cfg is not None:
                # Log autodespeckler learning rate (second param group when using get_optimizer_with_ad)
                if scheduler is not None and len(scheduler.get_last_lr()) > 1:
                    lr_log["learning_rate_ad"] = scheduler.get_last_lr()[1]
                else:
                    lr_log["learning_rate_ad"] = ad_cfg.train.lr
            run.log(lr_log, step=epoch)

    # Save our model for main process
    cls_weights = None
    ad_weights = None
    fmetrics = None
    if rank == 0:
        fmetrics = Metrics(use_partitions=True)
        partition = 'shift_invariant' if cfg.train.shift_invariant else 'non_shift_invariant'
        if cfg.train.early_stopping:
            model_weights = early_stopper.get_best_weights()
            best_val_metrics = early_stopper.get_best_metrics()
            # reset model to checkpoint for later sample prediction
            # model_weights are from model.module (no 'module.' prefix)
            model.module.load_state_dict(model_weights)
            cls_weights = model.module.classifier.state_dict()
            ad_weights = (model.module.autodespeckler.state_dict()
                        if model.module.uses_autodespeckler() else None)
            fmetrics.save_metrics('val', partition=partition,
                                loss=early_stopper.get_min_validation_loss(),
                                **best_val_metrics)
            run.summary.update({f'final model {key}': value for key, value in best_val_metrics['core_metrics'].items()})
            run.summary[f"best_epoch"] = early_stopper.get_best_epoch()
        else:
            cls_weights = model.module.classifier.state_dict()
            ad_weights = (model.module.autodespeckler.state_dict()
                        if model.module.uses_autodespeckler() else None)
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
        # initialize mappable objects with fixed dB ranges for cross-comparability
        vv_map = ScalarMappable(norm=Normalize(vmin=VV_DB_MIN, vmax=VV_DB_MAX), cmap='gray')
    if my_channels.has_vh():
        vh_map = ScalarMappable(norm=Normalize(vmin=VH_DB_MIN, vmax=VH_DB_MAX), cmap='gray')
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
            # Use fixed dB range normalization (already set on vv_map)
            vv = vv_map.to_rgba(vv, bytes=True)
            vv = np.clip(vv, 0, 255).astype(np.uint8)
            vv_img = Image.fromarray(vv, mode="RGBA")
            row.append(wandb.Image(vv_img))
        if my_channels.has_vh():
            vh = X_c[:, :, channel_indices[1]].numpy()
            # Use fixed dB range normalization (already set on vh_map)
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
            # add despeckled VV and VH using fixed dB range normalization
            # Descale despeckler output from standardized space back to dB scale
            despeckled_descaled = std[:2].view(2, 1, 1) * despeckler_output + mean[:2].view(2, 1, 1)
            
            despeckled_vv = despeckled_descaled[0].numpy()
            # Use fixed dB range normalization (already set on vv_map)
            despeckled_vv = vv_map.to_rgba(despeckled_vv, bytes=True)
            despeckled_vv = np.clip(despeckled_vv, 0, 255).astype(np.uint8)
            despeckled_vv_img = Image.fromarray(despeckled_vv, mode="RGBA")
            row.append(wandb.Image(despeckled_vv_img))

            despeckled_vh = despeckled_descaled[1].numpy()
            # Use fixed dB range normalization (already set on vh_map)
            despeckled_vh = vh_map.to_rgba(despeckled_vh, bytes=True)
            despeckled_vh = np.clip(despeckled_vh, 0, 255).astype(np.uint8)
            despeckled_vh_img = Image.fromarray(despeckled_vh, mode="RGBA")
            row.append(wandb.Image(despeckled_vh_img))

        y_shifted = y_shifted.squeeze(0).mul(255).clamp(0, 255).byte().numpy()
        y_pred = y_pred.squeeze(0).mul(255).clamp(0, 255).byte().numpy()

        truth_img = Image.fromarray(y_shifted, mode="L")
        pred_img = Image.fromarray(y_pred, mode="L")
        fp_img = Image.fromarray(fp, mode="L")
        fn_img = Image.fromarray(fn, mode="L")
        row += [wandb.Image(truth_img), wandb.Image(pred_img), wandb.Image(fp_img), wandb.Image(fn_img)]

        table.add_data(*row)

    return table

def init_distributed(rank, world_size, free_port):
    os.environ["MASTER_ADDR"] = "localhost"
    os.environ["MASTER_PORT"] = str(free_port)
    torch.cuda.set_device(rank)
    dist.init_process_group(backend="nccl", rank=rank, world_size=world_size)

def run_experiment_s1(rank, world_size, free_port, cfg, ad_cfg=None):
    """Run a single node, multi-GPU S1 SAR model experiment given the configuration parameters.

    NOTE: For OmegaConf objects, pass in OmegaConf.to_container(cfg, resolve=True)
    for the cfg and ad_cfg parameters.

    Parameters
    ----------
    rank : int
        The rank of the current process
    world_size : int
        The total number of processes
    free_port : int
        The free port for the DDP process
    cfg : dict
        Config dictionary for the SAR classifier.
    ad_cfg : dict, optional
        Config dictionary for an optional SAR autodespeckler attachment.
        When ad_cfg.model.weights is provided, mixed normalization is used
        to ensure the pretrained despeckler receives correctly normalized inputs.
    
    Returns
    -------
    fmetrics : Metrics
        Metrics object containing the metrics for the experiment.
    """
    torch.set_num_threads(int(torch.get_num_threads() / world_size))
    cfg = OmegaConf.create(cfg)
    ad_cfg = OmegaConf.create(ad_cfg) if ad_cfg is not None else None

    init_distributed(rank, world_size, free_port)

    if rank == 0 and not wandb.login():
        raise Exception("Failed to login to wandb.")

    # seeding
    rank_seed = cfg.seed + rank
    np.random.seed(rank_seed)
    random.seed(rank_seed)
    torch.manual_seed(rank_seed)

    if torch.cuda.is_available():
        device = torch.device("cuda", rank)
    else:
        raise ValueError("CUDA is required for distributed training")

    # setup model
    sar_detector = SARWaterDetector(cfg, ad_cfg=ad_cfg).to(device)

    # For DDP, CVAE encoder which does not participate in training
    # must always be frozen (otherwise will get error)
    if ad_cfg is not None and ad_cfg.model.autodespeckler == "CVAE":
        sar_detector.freeze_ad_encoder_weights()

    # freeze AD weights based on settings
    static_graph = False
    find_unused_parameters = False
    if ad_cfg is not None:
        if ad_cfg.train.freeze:
            sar_detector.freeze_ad_weights()
            
            if ad_cfg.train.freeze_epochs <= cfg.train.epochs:
                # Decoder will be unfrozen later → graph changes
                find_unused_parameters = True
            else:
                # Permanently frozen → static graph
                static_graph = True
        else:
            # freeze=False: encoder frozen in __init__, decoder always trainable
            # Graph is static (no unfreezing will occur)
            static_graph = True
    
    model = DDP(sar_detector, device_ids=[rank], static_graph=static_graph, find_unused_parameters=find_unused_parameters)

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

    # Mixed normalization: When using pretrained autodespeckler weights, replace VV/VH
    # (channels 0-1) with despeckler's training statistics for correct normalization.
    # The despeckler was trained on s1_multi dataset with different statistics than
    # the classifier's s1_weak dataset. This ensures the despeckler receives inputs
    # in its expected normalized space.
    if ad_cfg is not None and getattr(ad_cfg.model, 'weights', None) is not None:
        ad_sample_dir = get_ad_sample_dir(ad_cfg, cfg.paths)
        with open(ad_sample_dir / 'mean_std.pkl', 'rb') as f:
            ad_mean, ad_std = pickle.load(f)
            ad_mean = torch.from_numpy(ad_mean)  # Shape: (2,) for VV, VH
            ad_std = torch.from_numpy(ad_std)
        # Replace VV/VH statistics (first 2 channels of selected channels)
        train_mean[:2] = ad_mean
        train_std[:2] = ad_std
        if rank == 0:
            print(f"[Mixed Normalization] Using despeckler VV/VH statistics from: {ad_sample_dir}")
            print(f"  VV: mean={ad_mean[0]:.4f}, std={ad_std[0]:.4f}")
            print(f"  VH: mean={ad_mean[1]:.4f}, std={ad_std[1]:.4f}")

    standardize = transforms.Compose([transforms.Normalize(train_mean, train_std)])

    # datasets (all can be memory mapped if desired)
    train_set = FloodSampleSARDataset(sample_dir, channels=channels,
                                        typ="train", transform=standardize, random_flip=cfg.data.random_flip,
                                        seed=cfg.seed+1, mmap_mode='r' if cfg.data.mmap else None,
                                        keep_contiguous_in_mem=getattr(cfg.train, 'keep_contiguous_in_mem', True))
    val_set = FloodSampleSARDataset(sample_dir, channels=channels,
                                    typ="val", transform=standardize)

    # only rank 0 loads the test set
    test_set = FloodSampleSARDataset(sample_dir, channels=channels,
                                        typ="test", transform=standardize) if (rank == 0 and cfg.eval.mode == 'test') else None

    # Only if manually aligned test patches available
    # Use cfg.data.shift_ablation = True
    # NOTE: See preprocess/shift_ablation.py
    is_shift_ablation_available = getattr(cfg.data, 'shift_ablation', False)
    if rank == 0 and cfg.eval.mode == 'test' and is_shift_ablation_available:
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
    # explicitly fork num workers in DDP to avoid memory bloat
    train_loader = DataLoader(train_set,
                             batch_size=cfg.train.batch_size,
                             num_workers=cfg.train.num_workers,
                             persistent_workers=cfg.train.num_workers>0,
                             pin_memory=True,
                             sampler=DistributedSampler(train_set, seed=cfg.seed, drop_last=True),
                             multiprocessing_context='fork',
                             shuffle=False)

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
                            shuffle=False,
                            multiprocessing_context='fork',
                            drop_last=False) if (rank == 0 and cfg.eval.mode == 'test') else None
    
    if rank == 0 and cfg.eval.mode == 'test' and is_shift_ablation_available:
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
    if rank == 0:
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
    else:
        run = None

    try:
        if rank == 0 and cfg.save:
            if cfg.save_path is None:
                cfg.save_path = str(Path(cfg.paths.experiment_dir) / f"{datetime.today().strftime('%Y-%m-%d')}_{cfg.model.classifier}_{run.id}/")
                run.config.update({"save_path": cfg.save_path}, allow_val_change=True)
            print(f'Save path set to: {cfg.save_path}')

        # train and save results metrics
        cls_weights, ad_weights, fmetrics = train(rank, world_size, model, train_loader, val_loader,
                                              test_loader, test_aligned_loader, device, cfg, ad_cfg, run, cache_dir=sample_dir)

        if rank == 0 and cfg.save:
            save_experiment(cls_weights, ad_weights, fmetrics, cfg, ad_cfg, run)

        # log predictions on validation set using wandb
        if rank == 0:
            percent_wet = cfg.wandb.get('percent_wet_patches', 0.5)  # Default to 0.5 if not specified
            # Pass model.module (underlying model without DDP wrapper)
            pred_table = sample_predictions(model.module, test_set if cfg.eval.mode == 'test' else val_set,
                                            train_mean, train_std, cfg, ad_cfg, sample_dir, cfg.eval.mode,
                                            percent_wet_patches=percent_wet)
            run.log({f"model_{cfg.eval.mode}_predictions": pred_table})
    except Exception as e:
        print("An exception occurred during training!")

        # Send an alert in the W&B UI
        if rank == 0:
            run.alert(
                title="Training crashed 🚨",
                text=f"Run failed due to: {e}"
            )

            # Log to wandb summary
            run.summary["error"] = str(e)

        # remove save directory if needed
        raise e
    finally:
        if rank == 0:
            run.finish()
        dist.destroy_process_group()

    return fmetrics

def validate_config(cfg, ad_cfg=None):
    def validate_channels(s):
        return type(s) == str and len(s) == 8 and all(c in '01' for c in s)

    # Add checks
    assert cfg.data.method in ['random', 'strided'], "Sampling method must be one of ['random', 'strided']"
    
    # VV and VH channels must always be active (required for autodespeckler)
    assert cfg.data.channels[0] == '1' and cfg.data.channels[1] == '1', \
        "VV (channel 0) and VH (channel 1) must always be enabled"
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
    ad_cfg = getattr(cfg, 'ad', None)
    validate_config(cfg, ad_cfg=ad_cfg)

    # resolve cfgs before pickling
    resolved_cfg = OmegaConf.to_container(cfg, resolve=True)
    resolved_ad_cfg = OmegaConf.to_container(ad_cfg, resolve=True) if ad_cfg is not None else None

    world_size = torch.cuda.device_count()
    free_port = find_free_port()
    print(f"world_size = {world_size}")
    print(f"Found free port: {free_port}")
    mp.spawn(run_experiment_s1, args=(world_size, free_port, resolved_cfg, resolved_ad_cfg), nprocs=world_size)

if __name__ == '__main__':
    main()
