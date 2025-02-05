import wandb
import torch
import logging
import argparse
import copy
import torch
from torch import nn
from torch.utils.data import DataLoader
from datetime import datetime
from dataset import FloodSampleSARDataset
from utils import EarlyStopper, SARChannelIndexer, SaveMetrics
from torchvision import transforms
from torchmetrics import MetricCollection
from torchmetrics.classification import BinaryAccuracy, BinaryPrecision, BinaryRecall, BinaryF1Score
from model import SARClassifier
from matplotlib.cm import ScalarMappable
from matplotlib.colors import Normalize
import random
from random import Random
from PIL import Image
from glob import glob
from loss import InvariantBCELoss, InvariantBCEDiceLoss, InvariantTverskyLoss, ShiftInvariantLoss, TrainShiftInvariantLoss, NonShiftInvariantLoss, BCEDiceLoss, TverskyLoss, LossConfig
import numpy as np
import sys
import pickle 

MODEL_NAMES = ['unet', 'unet++']
AUTODESPECKLER_NAMES = ['CNN1', 'CNN2', 'DAE', 'VAE']
NOISE_NAMES = ['normal', 'masking', 'log_gamma']
LOSS_NAMES = ['BCELoss', 'BCEDiceLoss', 'TverskyLoss']

# get our optimizer and metrics
def train_loop(dataloader, model, device, loss_config, optimizer, minibatches, c):
    running_loss = torch.tensor(0.0, device=device)
    running_recons_loss = torch.tensor(0.0, device=device)
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
        recons_loss = loss_dict['recons_loss']
        y_shifted = loss_dict['shifted_label']

        optimizer.zero_grad()
        loss.backward()
        optimizer.step()

        logits = out_dict['final_output']
        pred_y = nn.functional.sigmoid(logits).flatten() > 0.5
        target = y_shifted.flatten() > 0.5

        all_preds.append(pred_y)
        all_targets.append(target)
        running_loss += loss.detach()
        if loss_config.contains_reconstruction_loss():
            running_recons_loss += recons_loss.detach()

        if batch_i >= minibatches:
            break

    # calculate metrics
    all_preds = torch.cat(all_preds)
    all_targets = torch.cat(all_targets)

    metric_collection.update(all_preds, all_targets)
    metric_results = metric_collection.compute()
    epoch_acc = metric_results['BinaryAccuracy'].item()
    epoch_pre = metric_results['BinaryPrecision'].item()
    epoch_rec = metric_results['BinaryRecall'].item()
    epoch_f1 = metric_results['BinaryF1Score'].item()
    epoch_loss = running_loss.item() / minibatches

    # wandb tracking loss and metrics per epoch - track recons loss as well
    loss_log = {"train accuracy": epoch_acc, "train precision": epoch_pre, 
               "train recall": epoch_rec, "train f1": epoch_f1, "train loss": epoch_loss}
    if loss_config.contains_reconstruction_loss():
        loss_log['train reconstruction loss'] = running_recons_loss.item() / minibatches
    wandb.log(loss_log)
    metric_collection.reset()
    
    return epoch_loss

def test_loop(dataloader, model, device, loss_config, c, logging=True):
    running_vloss = torch.tensor(0.0, device=device)
    running_recons_vloss = torch.tensor(0.0, device=device)
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
            recons_vloss = loss_dict['recons_loss']
            y_shifted = loss_dict['shifted_label']
            
            running_vloss += loss.detach()
            if loss_config.contains_reconstruction_loss():
                running_recons_vloss += recons_vloss.detach()
            
            logits = out_dict['final_output']
            pred_y = nn.functional.sigmoid(logits).flatten() > 0.5
            target = y_shifted.flatten() > 0.5

            all_preds.append(pred_y)
            all_targets.append(target)

    # calculate metrics
    all_preds = torch.cat(all_preds)
    all_targets = torch.cat(all_targets)

    metric_collection.update(all_preds, all_targets)
    metric_results = metric_collection.compute()
    epoch_vacc = metric_results['BinaryAccuracy'].item()
    epoch_vpre = metric_results['BinaryPrecision'].item()
    epoch_vrec = metric_results['BinaryRecall'].item()
    epoch_vf1 = metric_results['BinaryF1Score'].item()
    epoch_vloss = running_vloss.item() / num_batches

    if logging:
        loss_log = {"val accuracy": epoch_vacc, "val precision": epoch_vpre, 
                    "val recall": epoch_vrec, "val f1": epoch_vf1, "val loss": epoch_vloss}
        if loss_config.contains_reconstruction_loss():
            loss_log['val reconstruction loss'] = running_recons_vloss.item() / num_batches
        wandb.log(loss_log)
        
    metric_collection.reset()
    epoch_vmetrics = (epoch_vacc, epoch_vpre, epoch_vrec, epoch_vf1)
    
    return epoch_vloss, epoch_vmetrics

def train(model, train_set, val_set, test_set, device, loss_config, config, save='model'):
    timestamp = datetime.now().strftime('%Y-%m-%d %H:%M:%S')
    logging.info(f'''Starting training:
        Date:            {timestamp}
        Epochs:          {config['epochs']}
        Batch size:      {config['batch_size']}
        Learning rate:   {config['learning_rate']}
        Training size:   {len(train_set)}
        Validation size: {len(val_set)}
        Device:          {device}
    ''')

    # log via wandb
    run = wandb.init(
        project=config['project'],
        group=config['group'],
        config={
        "dataset": "Sentinel1",
        "mode": config['mode'],
        "method": config['method'],
        "filter": config['filter'],
        "channels": ''.join('1' if b else '0' for b in config['channels']),
        "patch_size": config['size'],
        "window_size": config['window'],
        "architecture": config['name'],
        "load_classifier": config.get('load_classifier'),
        "dropout": config['dropout'],
        "deep_supervision": config['deep_supervision'] if config['name'] == 'unet++' else None,
        "autodespeckler": config['autodespeckler'],
        "load_autodespeckler": config.get('load_autodespeckler'),
        "freeze_autodespeckler": config['freeze_autodespeckler'],
        "latent_dim": config.get('latent_dim'),
        "AD_num_layers": config.get('AD_num_layers'),
        "AD_kernel_size": config.get('AD_kernel_size'),
        "AD_dropout": config.get('AD_dropout'),
        "AD_activation_func": config.get('AD_activation_func'),
        "noise_type": config.get('noise_type'),
        "noise_coeff": config.get('noise_coeff'),
        'VAE_beta': config.get('VAE_beta'),
        "learning_rate": config['learning_rate'],
        "epochs": config['epochs'],
        "early_stopping": config['early_stopping'],
        "patience": config['patience'],
        "batch_size": config['batch_size'],
        "num_workers": config['num_workers'],
        "optimizer": config['optimizer'],
        "loss_fn": config['loss'],
        "alpha": config.get('alpha') if config['loss'] == 'TverskyLoss' else None,
        "beta": config.get('beta') if config['loss'] == 'TverskyLoss' else None,
        "subset": config['subset'],
        "training_size": len(train_set),
        "validation_size": len(val_set),
        "test_size": len(test_set) if mode == 'test' else None,
        "val_percent": len(val_set) / (len(train_set) + len(val_set)),
        "save_file": save
        }
    )

    # log weights and gradients each epoch
    if config['watch_weights_grad']:
        wandb.watch(model, log="all", log_freq=20)
    
    # VAE only
    if config['autodespeckler'] == 'VAE':
        config['kld_weight'] = config['batch_size'] / len(train_set)
        
    # optimizer and scheduler for reducing learning rate
    if config['optimizer'] == 'Adam':
        optimizer = torch.optim.Adam(model.parameters(), lr=config['learning_rate'])
    elif config['optimizer'] == 'SGD':
        optimizer = torch.optim.SGD(model.parameters(), lr=config['learning_rate'])
    else:
        raise Exception('Optimizer not found.')
    
    scheduler = torch.optim.lr_scheduler.ReduceLROnPlateau(optimizer, patience=5)

    # make DataLoader
    train_loader = DataLoader(train_set,
                             batch_size=config['batch_size'],
                             num_workers=config['num_workers'],
                             persistent_workers=config['num_workers']>0,
                             pin_memory=True,
                             shuffle=True,
                             drop_last=False)
    
    val_loader = DataLoader(val_set,
                            batch_size=config['batch_size'],
                            num_workers=config['num_workers'],
                            persistent_workers=config['num_workers']>0,
                            pin_memory=True,
                            shuffle=True,
                            drop_last=False)

    test_loader = DataLoader(test_set,
                            batch_size=config['batch_size'],
                            num_workers=config['num_workers'],
                            persistent_workers=config['num_workers']>0,
                            pin_memory=True,
                            shuffle=True,
                            drop_last=False) if config['mode'] == 'test' else None

    # TRAIN AND TEST LOOP IS PER EPOCH!!!
    if config['early_stopping']:
        early_stopper = EarlyStopper(patience=config['patience'])

        # best model checkpoint
        min_model_weights = model.state_dict()
    
    # best summary params
    wandb.define_metric("val accuracy", summary="max")
    wandb.define_metric("val precision", summary="max")
    wandb.define_metric("val recall", summary="max")
    wandb.define_metric("val f1", summary="max")
    wandb.define_metric("val loss", summary="min")

    minibatches = int(len(train_loader) * config['subset'])
    center_1 = (config['size'] - config['window']) // 2
    center_2 = center_1 + config['window']
    c = (center_1, center_2)
    for epoch in range(config['epochs']):
        # train loop
        avg_loss = train_loop(train_loader, model, device, loss_config, optimizer, minibatches, c)

        # at the end of each training epoch compute validation
        avg_vloss, avg_vmetrics = test_loop(val_loader, model, device, loss_config, c)

        if config['early_stopping']:
            early_stopper.step(avg_vloss)
            if early_stopper.is_stopped():
                break
                
            if early_stopper.is_best_epoch():
                early_stopper.store_metric(avg_vmetrics)
                # Model weights are saved at the end of every epoch, if it's the best seen so far:
                min_model_weights = copy.deepcopy(model.state_dict())
            
        scheduler.step(avg_vloss)

    # Save our model
    final_vmetrics = SaveMetrics(shift_invariant=config['shift_invariant'])
    PATH = 'models/' + save + '.pth'
    if config['early_stopping']:
        torch.save(min_model_weights, PATH)

        # reset model to checkpoint for later sample prediction
        model.load_state_dict(min_model_weights)
        final_vmetrics.save_metrics(early_stopper.get_metric(), typ='val')
    else:
        torch.save(model.state_dict(), PATH)
        final_vmetrics.save_metrics(avg_vmetrics, typ='val')

    # for benchmarking purposes
    if config['mode'] == 'test':
        _, test_vmetrics = test_loop(test_loader, model, device, loss_config, c, logging=False)
        final_vmetrics.save_metrics(test_vmetrics, typ='test')

    return run, final_vmetrics

def sample_predictions(model, sample_set, mean, std, loss_config, config, seed=24330):
    """Generate predictions on a subset of images in the dataset for wandb logging."""
    if config['num_sample_predictions'] <= 0:
        return None

    loss_config.val_loss_fn.change_device('cpu')
    columns = ["id"]
    my_channels = SARChannelIndexer(config['channels'])
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
    for i, channel in enumerate(config['channels']):
        if channel:
            channel_indices[i] = n
            n += 1
    
    model.to('cpu')
    model.eval()
    rng = Random(seed)
    samples = rng.sample(range(0, len(sample_set)), config['num_sample_predictions'])

    center_1 = (config['size'] - config['window']) // 2
    center_2 = center_1 + config['window']
    for id, k in enumerate(samples):
        # get all images to shape (H, W, C) with C = 1 or 3 (1 for grayscale, 3 for RGB)
        X, y = sample_set[k]

        X_c = X[:, center_1:center_2, center_1:center_2]

        with torch.no_grad():
            out_dict = model(X_c.unsqueeze(0))
            logits = out_dict['final_output']
            despeckler_output = out_dict['despeckler_output'].squeeze(0) if model.uses_autodespeckler() else None 
            y_shifted = loss_config.get_label_alignment(logits, y.unsqueeze(0).float()).squeeze(0)
            
        pred_y = torch.where(nn.functional.sigmoid(logits) > 0.5, 1.0, 0.0).squeeze(0) # (1, x, y)

        # Compute false positives
        fp = torch.logical_and(y_shifted == 0, pred_y == 1).squeeze(0).byte().mul(255).numpy()
        
        # Compute false negatives
        fn = torch.logical_and(y_shifted == 1, pred_y == 0).squeeze(0).byte().mul(255).numpy()

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
        pred_y = pred_y.squeeze(0).mul(255).clamp(0, 255).byte().numpy()
        
        truth_img = Image.fromarray(y_shifted, mode="L")
        pred_img = Image.fromarray(pred_y, mode="L")
        fp_img = Image.fromarray(fp, mode="L")
        fn_img = Image.fromarray(fn, mode="L")
        row += [wandb.Image(truth_img), wandb.Image(pred_img), wandb.Image(fp_img), wandb.Image(fn_img)]

        table.add_data(*row)

    return table

def run_experiment_s1(config):
    """Run a single S1 SAR model experiment given the configuration parameters."""
    if wandb.login():
        # seeding
        np.random.seed(config['seed'])
        random.seed(config['seed'])
        torch.manual_seed(config['seed'])

        device = (
            "cuda"
            if torch.cuda.is_available()
            else "mps"
            if torch.backends.mps.is_available()
            else "cpu"
        )
        channels = config['channels']
        n_channels = sum(channels)
        model = SARClassifier(config, n_channels=n_channels).to(device)
        
        # load and freeze weights
        model.load_classifier_weights(config.get('load_classifier'), device)
        if model.uses_autodespeckler():
            model.load_autodespeckler_weights(config.get('load_autodespeckler'), device)
            if config['freeze_autodespeckler']:
                model.freeze_autodespeckler_weights()
        
        
        print(f"Using {device} device")
        model_name = config['name']
        method = config['method']
        filter = config['filter']
        size = config['size']
        samples = config['samples']
        sample_dir = f'data/sar/{method}/{filter}/samples_{size}_{samples}/'
        save_file = f"sar_{model_name}_model{len(glob(f'models/sar_{model_name}_model*.pth'))}"

        # load in mean and std
        b_channels = sum(channels[-2:])
        with open(f'data/sar/stats/{method}_{filter}_{size}_{samples}.pkl', 'rb') as f:
            train_mean, train_std = pickle.load(f)

            train_mean = torch.from_numpy(train_mean[channels])
            train_std = torch.from_numpy(train_std[channels])
            # make sure binary channels are 0 mean and 1 std
            if b_channels > 0:
                train_mean[-b_channels:] = 0
                train_std[-b_channels:] = 1

        # need to make sure normalize here works as intended
        standardize = transforms.Compose([transforms.Normalize(train_mean, train_std)])
        
        train_set = FloodSampleSARDataset(sample_dir, channels=config['channels'], 
                                          typ="train", transform=standardize, random_flip=config['random_flip'],
                                          seed=config['seed']+1)
        val_set = FloodSampleSARDataset(sample_dir, channels=config['channels'], 
                                        typ="val", transform=standardize)
        test_set = FloodSampleSARDataset(sample_dir, channels=config['channels'], 
                                         typ="test", transform=standardize) if config['mode'] == 'test' else None
        
        # initialize loss functions - train loss function is optimized for gradient calculations
        loss_config = LossConfig(config, device=device)
        run, final_vmetrics = train(model, train_set, val_set, test_set, device, loss_config, config, save=save_file)

        # summary metrics
        final_vacc, final_vpre, final_vrec, final_vf1 = final_vmetrics.get_val_metrics()
        run.summary["final_acc"] = final_vacc
        run.summary["final_pre"] = final_vpre
        run.summary["final_rec"] = final_vrec
        run.summary["final_f1"] = final_vf1
            
        # log predictions on validation set using wandb
        try:
            pred_table = sample_predictions(model, test_set if config['mode'] == 'test' else val_set, 
                                            train_mean, train_std, loss_config, config)
            run.log({"model_val_predictions": pred_table})
        finally:
            run.finish()

        # if want test metrics calculate model score on test set
        return final_vmetrics
    else:
        raise Exception("Failed to login to wandb.")

def main(config):
    run_experiment_s1(config)

if __name__ == '__main__':
    parser = argparse.ArgumentParser(prog='train_sar_classifier', description='Trains SAR classifier model from patches. The classifier inputs a patch with n channels and outputs a binary patch with water pixels labeled 1.')

    def bool_indices(s):
        if len(s) == 7 and all(c in '01' for c in s):
            try:
                return [bool(int(x)) for x in s]
            except ValueError:
                raise argparse.ArgumentTypeError("Invalid boolean string: '{}'".format(s))
        else:
            raise argparse.ArgumentTypeError("Boolean string must be of length 7 and have binary digits")

    # preprocessing
    parser.add_argument('-x', '--size', type=int, default=68, help='pixel width of dataset patches (default: 68)')
    parser.add_argument('-w', '--window', type=int, default=64, help='pixel width of model input/output (default: 64)')
    parser.add_argument('-n', '--samples', type=int, default=1000, help='number of patches sampled per image (default: 1000)')
    parser.add_argument('-m', '--method', default='minibatch', choices=['minibatch', 'individual'], help='sampling method (default: minibatch)')
    parser.add_argument('--filter', default='raw', choices=['lee', 'raw'], help=f"filters: enhanced lee, raw (default: raw)")
    parser.add_argument('-c', '--channels', type=bool_indices, default="1111111", help='string of 7 binary digits for selecting among the 10 available channels (VV, VH, DEM, SlopeY, SlopeX, Water, Roads) (default: 1111111)')

    # wandb
    parser.add_argument('--project', default="SARClassifier", help='Wandb project where run will be logged')
    parser.add_argument('--group', default=None, help='Optional group name for model experiments (default: None)')
    parser.add_argument('--num_sample_predictions', type=int, default=40, help='number of predictions to visualize (default: 40)')
    parser.add_argument('--watch_weights_grad', action='store_true', help='wandb weight and gradient monitoring (default: False)')

    # evaluation
    parser.add_argument('--mode', default='val', choices=['val', 'test'], help=f"dataset used for evaluation metrics (default: val)")
    
    # ml
    parser.add_argument('-e', '--epochs', type=int, default=30, help='(default: 30)')
    parser.add_argument('-b', '--batch_size', type=int, default=32, help='(default: 32)')
    parser.add_argument('-s', '--subset', dest='subset', type=float, default=1.0, help='percentage of training dataset to use per epoch (default: 1.0)')
    parser.add_argument('-l', '--learning_rate', type=float, default=0.0001, help='(default: 0.0001)')
    parser.add_argument('--early_stopping', action='store_true', help='early stopping (default: False)')
    parser.add_argument('-p', '--patience', type=int, default=5, help='(default: 5)')

    # model
    parser.add_argument('--name', default='unet', choices=MODEL_NAMES,
                        help=f"models: {', '.join(MODEL_NAMES)} (default: unet)")
    # unet
    parser.add_argument('--dropout', type=float, default=0.2, help=f"(default: 0.2)")
    # unet++
    parser.add_argument('--deep_supervision', action='store_true', help='(default: False)')
    
    # autodespeckler
    parser.add_argument('--autodespeckler', default=None, choices=AUTODESPECKLER_NAMES,
                        help=f"models: {', '.join(AUTODESPECKLER_NAMES)} (default: None)")
    parser.add_argument('--noise_type', default=None, choices=NOISE_NAMES,
                        help=f"models: {', '.join(NOISE_NAMES)} (default: None)")
    parser.add_argument('--noise_coeff', type=float, default=None,  help=f"noise coefficient (default: 0.1)")
    parser.add_argument('--latent_dim', default=None, type=int, help='latent dimensions (default: 200)')
    parser.add_argument('--AD_num_layers', default=None, type=int, help='Autoencoder layers (default: 5)')
    parser.add_argument('--AD_kernel_size', default=None, type=int, help='Autoencoder kernel size (default: 3)')
    parser.add_argument('--AD_dropout', default=None, type=float, help=f"(default: 0.1)")
    parser.add_argument('--AD_activation_func', default=None, choices=['leaky_relu', 'relu'], help=f'activations: leaky_relu, relu (default: leaky_relu)')
    parser.add_argument('--VAE_beta', default=1.0, type=float, help=f"(default: 1.0)")

    # load weights
    parser.add_argument('--load_classifier', default=None, help='File path to .pth')
    parser.add_argument('--load_autodespeckler', default=None, help='File path to .pth')
    parser.add_argument('--freeze_autodespeckler', default=False, help='Freeze autodespeckler weights during training (default: False)')

    # data augmentation
    parser.add_argument('--random_flip', action='store_true', help='Randomly flip training patches horizontally and vertically (default: False)')

    # data loading
    parser.add_argument('--num_workers', type=int, default=10, help='(default: 10)')
    
    # loss
    parser.add_argument('--loss', default='BCELoss', choices=LOSS_NAMES,
                        help=f"loss: {', '.join(LOSS_NAMES)} (default: BCELoss)")
    parser.add_argument('--shift_invariant', action='store_true', help='(default: False)')
    parser.add_argument('--alpha', type=float, default=0.3, help='Tversky Loss alpha value (default: 0.3)')
    parser.add_argument('--beta', type=float, default=0.7, help='Tversky Loss beta value (default: 0.7)')

    # optimizer
    parser.add_argument('--optimizer', default='Adam', choices=['Adam', 'SGD'],
                        help=f"optimizer: {', '.join(['Adam', 'SGD'])} (default: Adam)")

    # reproducibility
    parser.add_argument('--seed', type=int, default=831002, help='seed (default: 831002)')

    config = vars(parser.parse_args())
    sys.exit(main(config))
