import torch
from torch import nn
from torch.utils.data import DataLoader
from torchvision import transforms
from torchmetrics import MetricCollection
from torchmetrics.classification import BinaryAccuracy, BinaryPrecision, BinaryRecall, BinaryF1Score, BinaryConfusionMatrix, BinaryJaccardIndex

import pickle
from pathlib import Path
import json
from omegaconf import DictConfig, OmegaConf
import hydra

from floodmaps.models.model import S2WaterDetector
from floodmaps.utils.metrics import compute_nlcd_metrics, compute_scl_metrics
from floodmaps.utils.utils import ChannelIndexer

from floodmaps.training.dataset import FloodSampleS2Dataset
from floodmaps.training.train_s2 import get_loss_fn

MODEL_NAMES = ['unet', 'unet++']
LOSS_NAMES = ['BCELoss', 'BCEDiceLoss', 'TverskyLoss']

def test_loop(model, dataloader, device, loss_fn, permute_channel=None, typ='test'):
    """Evaluate model on test set, optionally permuting a specified channel.
    
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
    permute_channel : int, optional
        Index of channel to permute.
    typ : str, optional
        Type of evaluation: 'val' for validation, 'test' for test
    
    Returns
    -------
    tuple
        metrics_dict
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
    all_preds = []
    all_targets = []
    all_nlcd_classes = []
    all_scl_classes = []
    
    model.eval()
    with torch.no_grad():
        for X, y, supplementary in dataloader:
            X = X.to(device)
            y = y.to(device)
            nlcd_classes = supplementary[:, 3, :, :].to(device)
            scl_classes = supplementary[:, 4, :, :].to(device)

            # permute the channel if specified
            if permute_channel is not None:
                assert isinstance(permute_channel, int), "Permute channel must be an integer"
                bs, cs, h, w = X.shape
                reshaped_channel = X[:, permute_channel].reshape(bs, -1)
                perm_indices = torch.randperm(h * w, device=device)
                X[:, permute_channel] = reshaped_channel[:, perm_indices].reshape(bs, h, w)

            logits = model(X)
            loss = loss_fn(logits, y.float())
            
            y_pred = nn.functional.sigmoid(logits).flatten() > 0.5
            target = y.flatten() > 0.5
            all_preds.append(y_pred)
            all_targets.append(target)
            all_nlcd_classes.append(nlcd_classes.flatten())
            all_scl_classes.append(scl_classes.flatten())
            running_vloss += loss.detach()

    # calculate metrics
    all_preds = torch.cat(all_preds)
    all_targets = torch.cat(all_targets)
    all_nlcd_classes = torch.cat(all_nlcd_classes)
    all_scl_classes = torch.cat(all_scl_classes)

    metric_collection.update(all_preds, all_targets)
    metric_results = metric_collection.compute()
    epoch_vloss = running_vloss.item() / len(dataloader)

    core_metrics_dict = {
        f"{typ} loss": epoch_vloss,
        f"{typ} accuracy": metric_results['BinaryAccuracy'].item(),
        f"{typ} precision": metric_results['BinaryPrecision'].item(),
        f"{typ} recall": metric_results['BinaryRecall'].item(),
        f"{typ} f1": metric_results['BinaryF1Score'].item(),
        f"{typ} IoU": metric_results['BinaryJaccardIndex'].item()
    }

    # calculate confusion matrix + NLCD class metrics + SCL class metrics
    confusion_matrix = metric_results['BinaryConfusionMatrix'].tolist()
    confusion_matrix_dict = {
        "tn": confusion_matrix[0][0],
        "fp": confusion_matrix[0][1],
        "fn": confusion_matrix[1][0],
        "tp": confusion_matrix[1][1]
    }
    nlcd_metrics_dict = compute_nlcd_metrics(all_preds, all_targets, all_nlcd_classes)
    scl_metrics_dict = compute_scl_metrics(all_preds, all_targets, all_scl_classes)
    metric_collection.reset()

    # separate the core loggable metrics from the nested dictionaries
    # for easier management downstream
    metrics_dict = {
        'core_metrics': core_metrics_dict,
        'confusion_matrix': confusion_matrix_dict,
        'nlcd_metrics': nlcd_metrics_dict,
        'scl_metrics': scl_metrics_dict
    }

    return metrics_dict

def run_permutation_importance(cfg):
    """Evaluates permutation importance of each channel to the S2 model.
    Results are saved to a json file in the save directory.

    Parameters
    ----------
    cfg : object
        Config object for the S2 classifier.
    """
    # save directory
    if cfg.analysis.save_dir is None:
        raise ValueError("Save directory must be provided.")

    save_dir = Path(cfg.analysis.save_dir)
    save_dir.mkdir(parents=True, exist_ok=True)
    config_path = save_dir / 'config.yaml'
    results_path = save_dir / 'perm_s2_results.json'
    OmegaConf.save(cfg, config_path)

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
    method = cfg.data.method
    size = cfg.data.size
    sample_param = cfg.data.samples if cfg.data.method == 'random' else cfg.data.stride
    suffix = getattr(cfg.data, 'suffix', '')
    # Use weak labeled dataset if specified, otherwise use manual labeled dataset
    use_weak = getattr(cfg.data, 'use_weak', False)
    dataset_type = 's2_weak' if use_weak else 's2'
    if suffix:
        sample_dir = Path(cfg.paths.preprocess_dir) / dataset_type / f'{method}_{size}_{sample_param}_{suffix}/'
    else:
        sample_dir = Path(cfg.paths.preprocess_dir) / dataset_type / f'{method}_{size}_{sample_param}/'

    # load in mean and std
    channels = [bool(int(x)) for x in cfg.data.channels]
    with open(sample_dir / f'mean_std.pkl', 'rb') as f:
        train_mean, train_std = pickle.load(f)

        train_mean = torch.from_numpy(train_mean[channels])
        train_std = torch.from_numpy(train_std[channels])

    standardize = transforms.Compose([transforms.Normalize(train_mean, train_std)])

    test_set = FloodSampleS2Dataset(sample_dir, channels=channels,
                                    typ="test", transform=standardize)

    test_loader = DataLoader(test_set,
                            batch_size=cfg.train.batch_size,
                            num_workers=cfg.train.num_workers,
                            persistent_workers=cfg.train.num_workers>0,
                            pin_memory=True,
                            shuffle=False,
                            drop_last=False)
    
    # compute pos_weight if enabled for rebalancing
    pos_weight_val = None
    if getattr(cfg.train, 'use_pos_weight', False):
        if getattr(cfg.train, 'pos_weight', None) is not None:
            pos_weight_val = float(cfg.train.pos_weight)
        else:
            # train set loaded just to calculate pos weight for loss function
            train_set = FloodSampleS2Dataset(sample_dir, channels=channels,
                                            typ="train", transform=standardize,
                                            mmap_mode='r' if cfg.data.mmap else None)
            train_loader = DataLoader(train_set,
                             batch_size=cfg.train.batch_size,
                             num_workers=cfg.train.num_workers,
                             persistent_workers=cfg.train.num_workers>0,
                             pin_memory=True,
                             shuffle=False,
                             drop_last=False)
            label_np = train_loader.dataset.dataset[:, -6, :, :]
            pos = float(label_np.sum())
            total = float(label_np.size)
            neg = max(total - pos, 1.0)
            # raw pos_weight = neg/pos; clip to [1, clip]
            raw_pw = neg / max(pos, 1.0)
            clip_max = float(getattr(cfg.train, 'pos_weight_clip', 10.0))
            pos_weight_val = max(1.0, min(raw_pw, clip_max))

    loss_fn = get_loss_fn(cfg, device=device, pos_weight=pos_weight_val)
        
    # normal loss and metrics
    print("Evaluating model without permutation...")
    all_results = {}
    no_perm_dict = test_loop(model, test_loader, device, loss_fn, permute_channel=None)
    all_results['no_permutation'] = no_perm_dict

    # permute each channel and evaluate loss and metrics
    channel_indexer = ChannelIndexer(channels)
    for name, idx in channel_indexer.get_name_to_index().items():
        # for each channel name and index evaluate
        print(f"Evaluating channel {name} permutation importance...")

        # permuted
        perm_metrics_dict = test_loop(model, test_loader, device, loss_fn, permute_channel=idx)
        all_results[name] = perm_metrics_dict

    # save all_results
    print("Saving results...")
    with open(results_path, 'w') as f:
        json.dump(all_results, f)
    print("Done! Saved results to", str(results_path))

    return 0

def validate_config(cfg):
    def validate_channels(s):
        return type(s) == str and len(s) == 16 and all(c in '01' for c in s)

    # Add checks
    assert cfg.data.method in ['random', 'strided'], "Sampling method must be one of ['random', 'strided']"
    assert cfg.save in [True, False], "Save must be a boolean"
    if cfg.train.loss == 'TverskyLoss':
        assert 0.0 <= cfg.train.tversky.alpha <= 1.0, "Tversky alpha must be in [0, 1]"
    assert cfg.train.batch_size is not None and cfg.train.batch_size > 0, "Batch size must be defined and positive"
    assert cfg.train.lr > 0, "Learning rate must be positive"
    assert cfg.train.loss in LOSS_NAMES, f"Loss must be one of {LOSS_NAMES}"
    assert cfg.train.optimizer in ['Adam', 'SGD', 'AdamW'], f"Optimizer must be one of {['Adam', 'SGD', 'AdamW']}"
    assert cfg.train.LR_scheduler in ['Constant', 'ReduceLROnPlateau', 'CosAnnealingLR'], f"LR scheduler must be one of {['Constant', 'ReduceLROnPlateau', 'CosAnnealingLR']}"
    assert cfg.train.early_stopping in [True, False], "Early stopping must be a boolean"
    assert not cfg.train.early_stopping or cfg.train.patience is not None, "Patience must be set if early stopping is enabled"
    assert cfg.model.classifier in MODEL_NAMES, f"Model must be one of {MODEL_NAMES}"
    assert cfg.data.random_flip in [True, False], "Random flip must be a boolean"
    assert cfg.eval.mode in ['val', 'test'], f"Evaluation mode must be one of {['val', 'test']}"
    assert cfg.wandb.project is not None, "Wandb project must be specified"
    assert validate_channels(cfg.data.channels), "Channels must be a binary string of length 16"

@hydra.main(version_base=None, config_path="pkg://configs", config_name="config.yaml")
def main(cfg: DictConfig):
    validate_config(cfg)
    run_permutation_importance(cfg)

if __name__ == '__main__':
    main()