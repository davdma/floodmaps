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
from utils import trainMeanStd, wet_label, EarlyStopper, ChannelIndexer
from torchvision import transforms
from torcheval.metrics import BinaryAccuracy, BinaryPrecision, BinaryRecall, BinaryF1Score
from architectures.discriminator import Classifier1, Classifier2, Classifier3
from matplotlib.cm import ScalarMappable
from matplotlib.colors import Normalize
from random import Random
from PIL import Image
from glob import glob
import loss
from loss import BCEDiceLoss, TverskyLoss
import numpy as np
import sys

MODEL_NAMES = ['c1', 'c2', 'c3']
LOSS_NAMES = ['BCELoss', 'BCEDiceLoss', 'TverskyLoss']

# get our optimizer and metrics
def train_loop(dataloader, model, device, loss_fn, optimizer, config):
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
        
        optimizer.zero_grad()
        pred_y = model(X)

        target = wet_label(y, config['size'], num_pixel=config['size']).flatten()
        loss = loss_fn(pred_y, target.float())
        loss.backward()
        optimizer.step()

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
               "train recall": epoch_rec, "train f1": epoch_f1, "train loss": epoch_loss})

    return epoch_loss

def test_loop(dataloader, model, device, loss_fn, config):
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
            
            pred_y = model(X)

            # determine whether the sample patch is considered wet or not
            target = wet_label(y, config['size'], num_pixel=config['size']).flatten()

            metric_acc.update(pred_y, target)
            metric_pre.update(pred_y, target)
            metric_rec.update(pred_y, target)
            metric_f1.update(pred_y, target)
            running_vloss += loss_fn(pred_y, target.float()).item()

    epoch_vacc = metric_acc.compute().item()
    epoch_vpre = metric_pre.compute().item()
    epoch_vrec = metric_rec.compute().item()
    epoch_vf1 = metric_f1.compute().item()
    epoch_vloss = running_vloss / num_batches
    wandb.log({"val accuracy": epoch_vacc, "val precision": epoch_vpre,
               "val recall": epoch_vrec, "val f1": epoch_vf1, "val loss": epoch_vloss})

    epoch_vmetrics = (epoch_vacc, epoch_vpre, epoch_vrec, epoch_vf1)
    return epoch_vloss, epoch_vmetrics

def train(train_set, val_set, model, device, config, save='discriminator'):
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
        "method": config['method'],
        "patch_size": config['size'],
        "samples": config['samples'],
        "architecture": config['name'],
        "num_pixels": config['num_pixels'],
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
                             shuffle=True,
                             drop_last=False)
    
    val_loader = DataLoader(val_set,
                            batch_size=config['batch_size'],
                            num_workers=config['num_workers'],
                            shuffle=True,
                            drop_last=False)

    if config['early_stopping']:
        early_stopper = EarlyStopper(patience=config['patience'])

        # best model checkpoint
        min_model_weights = model.state_dict()

    # TRAIN AND TEST LOOP IS PER EPOCH!!!
    for epoch in range(config['epochs']):
        # train loop
        avg_loss = train_loop(train_loader, model, device, loss_fn, optimizer, config)

        # at the end of each training epoch compute validation
        avg_vloss, avg_vmetrics = test_loop(val_loader, model, device, loss_fn, config)

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
        final_vrec = early_stopper.get_metric()
        torch.save(min_model_weights, PATH)

        # reset model to checkpoint for later sample prediction
        model.load_state_dict(min_model_weights)
        model.eval()
    else:
        final_vrec = avg_vrec
        torch.save(model.state_dict(), PATH)

    return run, final_vmetrics

def sample_predictions(model, val_set, mean, std, config, seed=24330):
    """Generate predictions on a subset of images in the validation set for wandb logging."""
    if config['num_sample_predictions'] <= 0:
        return None
        
    columns = ["id"]
    my_channels = ChannelIndexer(config['channels'])
    # initialize wandb table given the channel settings
    columns += my_channels.get_channel_names()
    columns += ["ground truth", "wet label", "predicted label"]
    table = wandb.Table(columns=columns)
    
    if my_channels.has_ndwi():
        # initialize mappable objects
        ndwi_norm = Normalize(vmin=-1, vmax=1)
        ndwi_map = ScalarMappable(norm=ndwi_norm, cmap='seismic_r')
    if my_channels.has_dem():
        dem_map = ScalarMappable(norm=None, cmap='gray')
    if my_channels.has_slope():
        slope_map = ScalarMappable(norm=None, cmap='jet')

    # get map of each channel to index of resulting tensor
    n = 0
    channel_indices = [-1] * 9
    for i, channel in enumerate(config['channels']):
        if channel:
            channel_indices[i] = n
            n += 1
    
    model.to('cpu')
    rng = Random(seed)
    samples = rng.sample(range(0, len(val_set)), config['num_sample_predictions'])

    for id, k in enumerate(samples):
        # get all images to shape (H, W, C) with C = 1 or 3 (1 for grayscale, 3 for RGB)
        X, y = val_set[k]

        truth = int(wet_label(y, config['size'], num_pixel=config['num_pixels']))
        prediction = model(X.unsqueeze(0))
        prediction = int(prediction > 0.5)

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
        if my_channels.has_slope():
            slope = X[:, :, channel_indices[6]].numpy()
            slope_map.set_norm(Normalize(vmin=np.min(slope), vmax=np.max(slope)))
            slope = slope_map.to_rgba(slope, bytes=True)
            slope = np.clip(slope, 0, 255).astype(np.uint8)
            slope_img = Image.fromarray(slope, mode="RGBA")
            row.append(wandb.Image(slope_img))
        if my_channels.has_waterbody():
            waterbody = X[:, :, channel_indices[7]].mul(255).clamp(0, 255).byte().numpy()
            waterbody_img = Image.fromarray(waterbody, mode="L")
            row.append(wandb.Image(waterbody_img))
        if my_channels.has_roads():
            roads = X[:, :, channel_indices[8]].mul(255).clamp(0, 255).byte().numpy()
            roads_img = Image.fromarray(roads, mode="L")
            row.append(wandb.Image(roads_img))

        y = y.squeeze(0).mul(255).clamp(0, 255).byte().numpy()
        truth_img = Image.fromarray(y, mode="L")
        row += [wandb.Image(truth_img), truth, prediction]

        table.add_data(*row)

    return table

def run_experiment(config):
    """Run a single S2 discriminator model experiment given the configuration parameters."""
    if wandb.login():
        device = (
            "cuda"
            if torch.cuda.is_available()
            else "mps"
            if torch.backends.mps.is_available()
            else "cpu"
        )
        n_channels = sum(config['channels'])
        # discriminator options
        # c1: 8 layer CNN
        # c2: 10 layer CNN
        # c3: SrGAN (not implemented)
        if config['name'] == "c1":
            discriminator = Classifier1(n_channels).to(device)
        elif config['name'] == "c2":
            discriminator = Classifier2(n_channels).to(device)
        elif config['name'] == "c3":
            discriminator = Classifier3(n_channels).to(device)
            # Initialize weights
            discriminator.apply(weights_init_normal)
            raise NotImplementedError("Classifier3 not ready for use.")
        else:
            raise Exception("Model not available.")
            
        print(f"Using {device} device")
        method = config['method']
        size = config['size']
        samples = config['samples']
        train_label_dir = f'data/s2/{method}/' + f'labels{size}_train_{samples}/'
        train_sample_dir = f'data/s2/{method}/' + f'samples{size}_train_{samples}/'
        test_label_dir = f'data/s2/{method}/' + f'labels{size}_test_{samples}/'
        test_sample_dir = f'data/s2/{method}/' + f'samples{size}_test_{samples}/'

        train_mean, train_std = trainMeanStd(channels=config['channels'], sample_dir=config['sample_dir'],
                                             label_dir=config['label_dir'])
        standardize = transforms.Compose([transforms.Normalize(train_mean, train_std)])

        train_set = FloodSampleDataset(train_sample_dir, train_label_dir, channels=config['channels'], typ="train",
                                       transform=standardize)
        val_set = FloodSampleDataset(test_sample_dir, test_label_dir, channels=config['channels'], typ="test",
                                     transform=standardize)

        try:
            run, (final_vacc, final_vpre, final_vrec, final_vf1) = train(train_set, val_set, discriminator, device, config,
                        save=f"discriminator{len(glob('models/discriminator*.pth'))}")

            # summary metrics
            run.summary["final_acc"] = final_vacc
            run.summary["final_pre"] = final_vpre
            run.summary["final_rec"] = final_vrec
            run.summary["final_f1"] = final_vf1
    
            # log predictions on validation set using wandb
            pred_table = sample_predictions(discriminator, val_set, train_mean, train_std, config)
            run.log({"model_val_predictions": pred_table})
        finally:
            run.finish()

        return final_vacc, final_vpre, final_vrec, final_vf1
    else:
        raise Exception("Failed to login to wandb.")

def validate_config(cfg):
    raise NotImplementedError("Not implemented.")

def main(cfg):
    validate_config(cfg)
    run_experiment(cfg)

if __name__ == '__main__':
    parser = argparse.ArgumentParser(prog='train_discriminator', description='Trains discriminator from patches. The discriminator inputs a patch with n channels and outputs a single value corresponding to whether the patch is likely to contain water or not.')

    def bool_indices(s):
        if len(s) == 9 and all(c in '01' for c in s):
            try:
                return [bool(int(x)) for x in s]
            except ValueError:
                raise argparse.ArgumentTypeError("Invalid boolean string: '{}'".format(s))
        else:
            raise argparse.ArgumentTypeError("Boolean string must be of length 9 and have binary digits")
            
    # dataset
    parser.add_argument('-x', '--size', type=int, default=64, help='pixel width of patch (default: 64)')
    parser.add_argument('-n', '--samples', type=int, default=1000, help='number of samples per image (default: 1000)')
    parser.add_argument('-m', '--method', default='random', choices=['random', 'relative'], help='sampling method (default: random)')
    parser.add_argument('-c', '--channels', type=bool_indices, default="111111111", help='string of 9 binary digits for selecting among the 9 available channels (R, G, B, B08, NDWI, DEM, Slope, Water, Roads) (default: 111111111)')
    parser.add_argument('--sdir', dest='sample_dir', default='../sampling/samples_200_5_4_35/', help='(default: ../sampling/samples_200_5_4_35/)')
    parser.add_argument('--ldir', dest='label_dir', default='../sampling/labels/', help='(default: ../sampling/labels/)')

    # wandb
    parser.add_argument('--project', default="FloodSamplesDiscriminator", help='Wandb project where run will be logged')
    parser.add_argument('--group', default=None, help='Optional group name for model experiments (default: None)')
    parser.add_argument('--num_sample_predictions', type=int, default=40, help='number of predictions to visualize (default: 40)')

    # ml
    parser.add_argument('-e', '--epochs', type=int, default=30, help='(default: 30)')
    parser.add_argument('-b', '--batch_size', type=int, default=32, help='(default: 32)')
    parser.add_argument('-l', '--learning_rate', type=float, default=0.01, help='(default: 0.01)')
    parser.add_argument('--early_stopping', action='store_true', help='early stopping (default: False)')
    parser.add_argument('-p', '--patience', type=int, default=5, help='(default: 5)')

    # model
    parser.add_argument('--name', default='c1', choices=MODEL_NAMES,
                        help=f"models: {', '.join(MODEL_NAMES)} (default: c1)")
    parser.add_argument('--num_pixels', type=int, default=100, help='(default: 100)')

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

    # config will be dict
    config = vars(parser.parse_args())

    sys.exit(main(config))
