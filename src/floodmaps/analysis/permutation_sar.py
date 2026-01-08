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

from floodmaps.models.model import SARWaterDetector
from floodmaps.utils.config import Config
from floodmaps.utils.utils import (SRC_DIR, RESULTS_DIR, SARChannelIndexer)
from floodmaps.training.loss import SARLossConfig
from floodmaps.training.dataset import FloodSampleSARDataset

LOSS_NAMES = ['BCELoss', 'BCEDiceLoss', 'TverskyLoss']

def evaluate(model, dataloader, channel, permute, name, device, loss_config, c):
    """Evaluate metrics on test set without logging.
    
    Parameters
    ----------
    channel: int
        Index of channel to permute.
    permute: bool
        To turn on permutation.
    name: str
        Name of channel to permute.
    """
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

            if permute:
                # Flatten spatial dimensions and permute the specified channel
                if name == 'vv':
                    # again vv and vh grouped together, hacky way to do this
                    bs, cs, h, w = X_c.shape
                    reshaped_channel_vv = X_c[:, channel].reshape(bs, -1)
                    reshaped_channel_vh = X_c[:, channel+1].reshape(bs, -1)
                    perm_indices = torch.randperm(64 * 64, device=device)
                    X_c[:, channel] = reshaped_channel_vv[:, perm_indices].reshape(bs, h, w)
                    X_c[:, channel+1] = reshaped_channel_vh[:, perm_indices].reshape(bs, h, w)
                else:
                    bs, cs, h, w = X_c.shape
                    reshaped_channel = X_c[:, channel].reshape(bs, -1)
                    perm_indices = torch.randperm(64 * 64, device=device)
                    X_c[:, channel] = reshaped_channel[:, perm_indices].reshape(bs, h, w)

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
        "loss": epoch_vloss,
        "accuracy": metric_results['BinaryAccuracy'].item(),
        "precision": metric_results['BinaryPrecision'].item(),
        "recall": metric_results['BinaryRecall'].item(),
        "f1": metric_results['BinaryF1Score'].item()
    }
    metric_collection.reset()

    return metrics_dict

def permutation_importance(cfg, ad_cfg=None, dir_path=None):
    """Evaluates permutation importance of each channel to the sar model.
    SAR VV and VH channels are treated as one singular channel. Results are
    saved to a json file in the save directory.

    Parameters
    ----------
    cfg : object
        Config object for the SAR classifier.
    ad_cfg : object, optional
        Config object for an optional SAR autodespeckler attachment.
    """
    # save directory
    if dir_path is None:
        raise ValueError("Save directory must be provided.")

    SAVE_DIR = Path(dir_path)
    SAVE_DIR.mkdir(parents=True, exist_ok=True)

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

    # load weights
    if cfg.model.weights is not None:
        model.load_classifier_weights(cfg.model.weights, device)
    if ad_cfg is not None and ad_cfg.model.weights is not None:
        model.load_autodespeckler_weights(ad_cfg.model.weights, device)
    model.eval()

    # dataset and transforms
    print(f"Using {device} device")
    model_name = cfg.model.classifier
    filter = 'lee' if cfg.data.use_lee else 'raw'
    size = cfg.data.size
    samples = cfg.data.samples
    sample_dir = SRC_DIR / f'data/sar/minibatch/{filter}/samples_{size}_{samples}/'

    # load in mean and std
    channels = [bool(int(x)) for x in cfg.data.channels]
    b_channels = sum(channels[-2:])
    with open(SRC_DIR / f'data/sar/stats/minibatch_{filter}_{size}_{samples}.pkl', 'rb') as f:
        train_mean, train_std = pickle.load(f)

        train_mean = torch.from_numpy(train_mean[channels])
        train_std = torch.from_numpy(train_std[channels])
        # make sure binary channels are 0 mean and 1 std
        if b_channels > 0:
            train_mean[-b_channels:] = 0
            train_std[-b_channels:] = 1
    standardize = transforms.Compose([transforms.Normalize(train_mean, train_std)])

    test_set = FloodSampleSARDataset(sample_dir, channels=channels,
                                        typ="test", transform=standardize)

    test_loader = DataLoader(test_set,
                            batch_size=cfg.train.batch_size,
                            num_workers=cfg.train.num_workers,
                            persistent_workers=cfg.train.num_workers>0,
                            pin_memory=True,
                            shuffle=True,
                            drop_last=False)

    loss_cfg = SARLossConfig(cfg, ad_cfg=ad_cfg, device=device)
    indexer = SARChannelIndexer(channels)
        
    # initialize loss functions - train loss function is optimized for gradient calculations
    loss_cfg = SARLossConfig(cfg, ad_cfg=ad_cfg, device=device)
    center_1 = (cfg.data.size - cfg.data.window) // 2
    center_2 = center_1 + cfg.data.window
    c = (center_1, center_2)

    # original loss and metrics
    print("Evaluating original model...")
    all_results = {}
    orig_dict = evaluate(model, test_loader, 0, False, None, device, loss_cfg, c)
    all_results['orig'] = orig_dict

    # map from index - channel name
    lst_map = indexer.get_map()
    for i, name in enumerate(lst_map):
        if name == 'vh':
            # hacky way to group vv and vh channels together
            continue
        print(f"Evaluating channel {name} permutation importance...")

        # permuted
        perm_dict = evaluate(model, test_loader, i, True, name, device, loss_cfg, c)
        all_results['sar' if name == 'vv' else name] = perm_dict

    # save all_results
    print("Saving results...")
    with open(SAVE_DIR / f'perm_sar_{model_name}.json', 'w') as f:
        json.dump(all_results, f)
    print("Done! Saved results to", SAVE_DIR / f'perm_sar_{model_name}.json')

    return 0

def main(cfg, ad_cfg, save_dir):
    permutation_importance(cfg, ad_cfg=ad_cfg, dir_path=save_dir)

if __name__ == '__main__':
    parser = argparse.ArgumentParser(prog='perm_sar', description='Evaluates permutation importance of channels given sar model.')

    # YAML config file
    parser.add_argument("--config_file", default="configs/classifier_default.yaml", help="Path to YAML config file (default: configs/classifier_default.yaml)")

    def bool_indices(s):
        if len(s) == 7 and all(c in '01' for c in s):
            try:
                return s
            except ValueError:
                raise argparse.ArgumentTypeError(f"Invalid boolean string: '{s}'")
        else:
            raise argparse.ArgumentTypeError("Boolean string must be of length 7 and have binary digits")

    # data loading
    parser.add_argument('--num_workers', type=int)

    # loss
    parser.add_argument('--loss', choices=LOSS_NAMES,
                        help=f"loss: {', '.join(LOSS_NAMES)}")

    # save directory
    parser.add_argument('--save_dir', type=str, help='save directory')

    # reproducibility
    parser.add_argument('--seed', type=int, help='seeding')

    _args = parser.parse_args()
    cfg = Config(**_args.__dict__)
    ad_config_path = cfg.model.autodespeckler.ad_config
    ad_cfg = Config(config_file=ad_config_path) if ad_config_path is not None else None
    sys.exit(main(cfg, ad_cfg, _args.save_dir))
