import wandb
import torch
import logging
import argparse
import copy
import torch
from torch import nn
from torch.utils.data import DataLoader
from datetime import datetime
from dataset import FloodSampleDataset
from utils import trainMeanStd, EarlyStopper, ChannelIndexer
from torchvision import transforms
from torcheval.metrics import BinaryAccuracy, BinaryPrecision, BinaryRecall, BinaryF1Score
from architectures.unet import UNet
from architectures.unet_plus import NestedUNet
from matplotlib.cm import ScalarMappable
from matplotlib.colors import Normalize
import random
from random import Random
from PIL import Image
from glob import glob
from loss import BCEDiceLoss, TverskyLoss
import numpy as np
import sys

MODEL_NAMES = ['unet', 'unet++']
LOSS_NAMES = ['BCELoss', 'BCEDiceLoss', 'TverskyLoss']

# get our optimizer and metrics
def train_loop(dataloader, model, device, loss_fn, optimizer, epoch):
    running_loss = 0.0
    num_batches = len(dataloader)
    metric_acc = BinaryAccuracy(threshold=0.5, device=device)
    metric_pre = BinaryPrecision(threshold=0.5, device=device)
    metric_rec = BinaryRecall(threshold=0.5, device=device)
    metric_f1 = BinaryF1Score(threshold=0.5, device=device)

    model.train()
    for X, y in dataloader:
        X = X.to(device)
        y = y.to(device)

        logits = model(X)
        loss = loss_fn(logits, y.float())
        
        optimizer.zero_grad()
        loss.backward()
        optimizer.step()

        pred_y = nn.functional.sigmoid(logits).flatten()
        target = y.int().flatten()
        metric_acc.update(pred_y, target)
        metric_pre.update(pred_y, target)
        metric_rec.update(pred_y, target)
        metric_f1.update(pred_y, target)
        running_loss += loss.item()

    # wandb tracking loss and accuracy per epoch
    epoch_acc = metric_acc.compute().item()
    epoch_pre = metric_pre.compute().item()
    epoch_rec = metric_rec.compute().item()
    epoch_f1 = metric_f1.compute().item()
    epoch_loss = running_loss / num_batches
    wandb.log({"train accuracy": epoch_acc, "train precision": epoch_pre, 
               "train recall": epoch_rec, "train f1": epoch_f1, "train loss": epoch_loss}, step=epoch)
    
    return epoch_loss

def test_loop(dataloader, model, device, loss_fn, epoch, logging=True):
    running_vloss = 0.0
    num_batches = len(dataloader)
    metric_acc = BinaryAccuracy(threshold=0.5, device=device)
    metric_pre = BinaryPrecision(threshold=0.5, device=device)
    metric_rec = BinaryRecall(threshold=0.5, device=device)
    metric_f1 = BinaryF1Score(threshold=0.5, device=device)
    
    model.eval()
    with torch.no_grad():
        for X, y in dataloader:
            X = X.to(device)
            y = y.to(device)

            logits = model(X)
            loss = loss_fn(logits, y.float())
            
            running_vloss += loss.item()
            
            pred_y = nn.functional.sigmoid(logits).flatten()
            target = y.int().flatten()
            metric_acc.update(pred_y, target)
            metric_pre.update(pred_y, target)
            metric_rec.update(pred_y, target)
            metric_f1.update(pred_y, target)

    epoch_vacc = metric_acc.compute().item()
    epoch_vpre = metric_pre.compute().item()
    epoch_vrec = metric_rec.compute().item()
    epoch_vf1 = metric_f1.compute().item()
    epoch_vloss = running_vloss / num_batches
    if logging:
        wandb.log({"val accuracy": epoch_vacc, "val precision": epoch_vpre,
                   "val recall": epoch_vrec, "val f1": epoch_vf1, "val loss": epoch_vloss}, step=epoch)

    epoch_vmetrics = (epoch_vacc, epoch_vpre, epoch_vrec, epoch_vf1)
    return epoch_vloss, epoch_vmetrics

def train(train_set, val_set, test_set, model, device, config, save='model'):
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
        "dataset": "Sentinel2",
        "mode": config['mode'],
        "method": config['method'],
        "channels": ''.join('1' if b else '0' for b in config['channels']),
        "patch_size": config['size'],
        "samples": config['samples'],
        "architecture": config['name'],
        "dropout": config['dropout'],
        "deep_supervision": config['deep_supervision'] if config['name'] == 'unet++' else None,
        "learning_rate": config['learning_rate'],
        "epochs": config['epochs'],
        "early_stopping": config['early_stopping'],
        "patience": config['patience'],
        "batch_size": config['batch_size'],
        "num_workers": config['num_workers'],
        "optimizer": config['optimizer'],
        "loss_fn": config['loss'],
        "alpha": config['alpha'],
        "beta": config['beta'],
        "training_size": len(train_set),
        "validation_size": len(val_set),
        "test_size": len(test_set) if config['mode'] == 'test' else None,
        "val_percent": len(val_set) / (len(train_set) + len(val_set)),
        "save_file": save
        }
    )
    
    # initialize loss function
    if config['loss'] == 'BCELoss':
        loss_fn = nn.BCEWithLogitsLoss()
    elif config['loss'] == 'BCEDiceLoss':
        loss_fn = BCEDiceLoss()
    elif config['loss'] == 'TverskyLoss':
        loss_fn = TverskyLoss(alpha=config['alpha'], beta=config['beta'])
    else:
        raise Exception('Loss function not found.')

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
    for epoch in range(config['epochs']):
        # train loop
        avg_loss = train_loop(train_loader, model, device, loss_fn, optimizer, epoch)

        # at the end of each training epoch compute validation
        avg_vloss, avg_vmetrics = test_loop(val_loader, model, device, loss_fn, epoch)

        if config['early_stopping']:
            early_stopper.step(avg_vloss)
            if early_stopper.is_stopped():
                break
                
            if early_stopper.is_best_epoch():
                early_stopper.store_metric(avg_vmetrics)
                # Model weights are saved at the end of every epoch, if it's the best seen so far:
                min_model_weights = copy.deepcopy(model.state_dict())
            
        scheduler.step(avg_vloss)

        if epoch == 0:
            # allows loader to use cache after first epoch
            train_loader.dataset.set_use_cache(True)
            val_loader.dataset.set_use_cache(True)

    # Save our model
    PATH = 'models/' + save + '.pth'
    if config['early_stopping']:
        torch.save(min_model_weights, PATH)

        # reset model to checkpoint for later sample prediction
        model.load_state_dict(min_model_weights)
        final_vmetrics = early_stopper.get_metric()
    else:
        torch.save(model.state_dict(), PATH)
        final_vmetrics = avg_vmetrics

    if config['mode'] == 'test':
        # test mode
        _, final_vmetrics = test_loop(test_loader, model, device, loss_fn, logging=False)

    return run, final_vmetrics

def sample_predictions(model, val_set, mean, std, config, seed=24330):
    """Generate predictions on a subset of images in the validation set for wandb logging."""
    if config['num_sample_predictions'] <= 0:
        return None
        
    columns = ["id"]
    my_channels = ChannelIndexer(config['channels'])
    # initialize wandb table given the channel settings
    columns += my_channels.get_channel_names()
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
    channel_indices = [-1] * 10
    for i, channel in enumerate(config['channels']):
        if channel:
            channel_indices[i] = n
            n += 1
    
    model.to('cpu')
    model.eval()
    rng = Random(seed)
    samples = rng.sample(range(0, len(val_set)), config['num_sample_predictions'])

    for id, k in enumerate(samples):
        # get all images to shape (H, W, C) with C = 1 or 3 (1 for grayscale, 3 for RGB)
        X, y = val_set[k]
        
        logits = model(X.unsqueeze(0))
            
        pred_y = torch.where(nn.functional.sigmoid(logits) > 0.5, 1.0, 0.0).squeeze(0) # (1, x, y)

        # Compute false positives
        fp = torch.logical_and(y == 0, pred_y == 1).squeeze(0).byte().mul(255).numpy()
        
        # Compute false negatives
        fn = torch.logical_and(y == 1, pred_y == 0).squeeze(0).byte().mul(255).numpy()

        # Channels are descaled using linear variance scaling
        X = X.permute(1, 2, 0)
        X = std * X + mean

        row = [k]
        if my_channels.has_image():
            tci = X[:, :, :3].mul(255).clamp(0, 255).byte().numpy()
            tci_img = Image.fromarray(tci, mode="RGB")
            row.append(wandb.Image(tci_img))
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

        y = y.squeeze(0).mul(255).clamp(0, 255).byte().numpy()
        pred_y = pred_y.squeeze(0).mul(255).clamp(0, 255).byte().numpy()
        
        truth_img = Image.fromarray(y, mode="L")
        pred_img = Image.fromarray(pred_y, mode="L")
        fp_img = Image.fromarray(fp, mode="L")
        fn_img = Image.fromarray(fn, mode="L")
        row += [wandb.Image(truth_img), wandb.Image(pred_img), wandb.Image(fp_img), wandb.Image(fn_img)]

        table.add_data(*row)

    return table

def run_experiment_s2(config):
    """Run a single S2 model experiment given the configuration parameters."""
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
        n_channels = sum(config['channels'])
        if config['name'] == "unet":
            model = UNet(n_channels, dropout=config['dropout']).to(device)
        else:
            # unet++
            model = NestedUNet(n_channels, dropout=config['dropout'], deep_supervision=config['deep_supervision']).to(device)
        
        print(f"Using {device} device")
        method = config['method']
        size = config['size']
        samples = config['samples']
        dataset_dir = f'data/s2/{method}/'
        
        train_mean, train_std = trainMeanStd(channels=config['channels'], 
                                             sample_dir=config['sample_dir'], 
                                             label_dir=config['label_dir'])
        standardize = transforms.Compose([transforms.Normalize(train_mean, train_std)])
        
        train_set = FloodSampleDataset(dataset_dir, channels=config['channels'], 
                                       size=size, samples=samples, typ="train", transform=standardize)
        val_set = FloodSampleDataset(dataset_dir, channels=config['channels'], 
                                     size=size, samples=samples, typ="val", transform=standardize)
        test_set = FloodSampleDataset(dataset_dir, channels=config['channels'], 
                                      size=size, samples=samples, typ="test", transform=standardize) \
                                      if config['mode'] == 'test' else None

        model_name = config['name']
        run, (final_vacc, final_vpre, final_vrec, final_vf1) = train(train_set, val_set, test_set, model, device, config, save=f"{model_name}_model{len(glob(f'models/{model_name}_model*.pth'))}")

        # summary metrics
        run.summary["final_acc"] = final_vacc
        run.summary["final_pre"] = final_vpre
        run.summary["final_rec"] = final_vrec
        run.summary["final_f1"] = final_vf1
            
        # log predictions on validation set using wandb
        try:
            pred_table = sample_predictions(model, test_set if config['mode'] == 'test' else val_set, 
                                            train_mean, train_std, config)
            run.log({"model_val_predictions": pred_table})
        finally:
            run.finish()

        return final_vacc, final_vpre, final_vrec, final_vf1
    else:
        raise Exception("Failed to login to wandb.")

def main(config):
    run_experiment_s2(config)

if __name__ == '__main__':
    parser = argparse.ArgumentParser(prog='train_classifier', description='Trains classifier model from patches. The classifier inputs a patch with n channels and outputs a binary patch with water pixels labeled 1.')

    def bool_indices(s):
        if len(s) == 10 and all(c in '01' for c in s):
            try:
                return [bool(int(x)) for x in s]
            except ValueError:
                raise argparse.ArgumentTypeError("Invalid boolean string: '{}'".format(s))
        else:
            raise argparse.ArgumentTypeError("Boolean string must be of length 10 and have binary digits")
    
    parser.add_argument('-x', '--size', type=int, default=64, help='pixel width of patch (default: 64)')
    parser.add_argument('-n', '--samples', type=int, default=1000, help='number of samples per image (default: 1000)')
    parser.add_argument('-m', '--method', default='random', choices=['random'], help='sampling method (default: random)')
    parser.add_argument('-c', '--channels', type=bool_indices, default="1111111111", help='string of 10 binary digits for selecting among the 10 available channels (R, G, B, B08, NDWI, DEM, SlopeY, SlopeX, Water, Roads) (default: 1111111111)')
    parser.add_argument('--sdir', dest='sample_dir', default='../sampling/samples_200_5_4_35/', help='(default: ../sampling/samples_200_5_4_35/)')
    parser.add_argument('--ldir', dest='label_dir', default='../sampling/labels/', help='(default: ../sampling/labels/)')

    # wandb
    parser.add_argument('--project', default="FloodSamplesClassifier", help='Wandb project where run will be logged')
    parser.add_argument('--group', default=None, help='Optional group name for model experiments (default: None)')
    parser.add_argument('--num_sample_predictions', type=int, default=40, help='number of predictions to visualize (default: 40)')

    # evaluation
    parser.add_argument('--mode', default='val', choices=['val', 'test'], help=f"dataset used for evaluation metrics (default: val)")

    # ml
    parser.add_argument('-e', '--epochs', type=int, default=30, help='(default: 30)')
    parser.add_argument('-b', '--batch_size', type=int, default=32, help='(default: 32)')
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

    # data loading
    parser.add_argument('--num_workers', type=int, default=10, help='(default: 10)')
    
    # loss
    parser.add_argument('--loss', default='BCELoss', choices=LOSS_NAMES,
                        help=f"loss: {', '.join(LOSS_NAMES)} (default: BCELoss)")
    parser.add_argument('--alpha', type=float, default=0.3, help='Tversky Loss alpha value (default: 0.3)')
    parser.add_argument('--beta', type=float, default=0.7, help='Tversky Loss beta value (default: 0.7)')

    # optimizer
    parser.add_argument('--optimizer', default='Adam', choices=['Adam', 'SGD'],
                        help=f"optimizer: {', '.join(['Adam', 'SGD'])} (default: Adam)")

    # reproducibility
    parser.add_argument('--seed', type=int, default=831002, help='seed (default: 831002)')

    # config will be dict
    config = vars(parser.parse_args())

    sys.exit(main(config))
