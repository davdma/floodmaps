import wandb
import torch
import logging
import argparse
import torch
from torch import nn
from torch.utils.data import DataLoader
from datetime import datetime
from dataset import FloodSampleDataset
from utils import trainMeanStd, EarlyStopper
from torchvision import transforms
from torcheval.metrics import BinaryAccuracy, BinaryPrecision, BinaryRecall, BinaryF1Score
from architectures.unet import UNet
from matplotlib.cm import ScalarMappable
from matplotlib.colors import Normalize
from random import Random
from PIL import Image
from glob import glob
from loss import BCEDiceLoss, TverskyLoss
import numpy as np
import sys

MODEL_NAMES = ['unet']
LOSS_NAMES = ['BCELoss', 'BCEDiceLoss', 'TverskyLoss']

# get our optimizer and metrics
def train_loop(dataloader, model, device, loss_fn, optimizer):
    running_loss = 0.0
    num_batches = len(dataloader)
    metric_acc = BinaryAccuracy(threshold=0.5)
    metric_pre = BinaryPrecision(threshold=0.5)
    metric_rec = BinaryRecall(threshold=0.5)
    metric_f1 = BinaryF1Score(threshold=0.5)

    model.train()
    for X, y in dataloader:
        X = X.to(device)
        y = y.to(device)
        
        optimizer.zero_grad()
        logits = model(X)

        loss = loss_fn(logits, y.float())
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
    epoch_acc = metric_acc.compute()
    epoch_pre = metric_pre.compute()
    epoch_rec = metric_rec.compute()
    epoch_f1 = metric_f1.compute()
    epoch_loss = running_loss / num_batches
    wandb.log({"train accuracy": epoch_acc, "train precision": epoch_pre, 
               "train recall": epoch_rec, "train f1": epoch_f1, "train loss": epoch_loss})

    return epoch_loss

def test_loop(dataloader, model, device, loss_fn):
    running_vloss = 0.0
    num_batches = len(dataloader)
    metric_acc = BinaryAccuracy(threshold=0.5)
    metric_pre = BinaryPrecision(threshold=0.5)
    metric_rec = BinaryRecall(threshold=0.5)
    metric_f1 = BinaryF1Score(threshold=0.5)
    
    model.eval()
    with torch.no_grad():
        for X, y in dataloader:
            X = X.to(device)
            y = y.to(device)
            
            logits = model(X)
            running_vloss += loss_fn(logits, y.float()).item()
            
            pred_y = nn.functional.sigmoid(logits).flatten()
            target = y.int().flatten()
            metric_acc.update(pred_y, target)
            metric_pre.update(pred_y, target)
            metric_rec.update(pred_y, target)
            metric_f1.update(pred_y, target)

    epoch_vacc = metric_acc.compute()
    epoch_vpre = metric_pre.compute()
    epoch_vrec = metric_rec.compute()
    epoch_vf1 = metric_f1.compute()
    epoch_vloss = running_vloss / num_batches
    wandb.log({"val accuracy": epoch_vacc, "val precision": epoch_vpre,
               "val recall": epoch_vrec, "val f1": epoch_vf1, "val loss": epoch_vloss})

    return epoch_vloss, epoch_vf1

def train(train_set, val_set, model, device, config, save='model'):
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
        config={
        "dataset": "Sentinel2",
        "method": config['method'],
        "channels": ''.join('1' if b else '0' for b in config['channels']),
        "patch_size": config['size'],
        "samples": config['samples'],
        "architecture": config['name'],
        "learning_rate": config['learning_rate'],
        "epochs": config['epochs'],
        "batch_size": config['batch_size'],
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
                             num_workers=0,
                             shuffle=True,
                             drop_last=False)
    
    val_loader = DataLoader(val_set,
                            batch_size=config['batch_size'],
                            num_workers=0,
                            shuffle=True,
                            drop_last=False)

    # TRAIN AND TEST LOOP IS PER EPOCH!!!
    if config['early_stopping']:
        early_stopper = EarlyStopper(patience=5)
        
    for epoch in range(config['epochs']):
        # train loop
        avg_loss = train_loop(train_loader, model, device, loss_fn, optimizer)

        # at the end of each training epoch compute validation
        avg_vloss, avg_vf1 = test_loop(val_loader, model, device, loss_fn)

        if config['early_stopping'] and early_stopper.early_stop(avg_vloss):             
            break

        scheduler.step(avg_vloss)

    # Save our model
    PATH = 'models/' + save + '.pth'
    torch.save(model.state_dict(), PATH)

    return run, avg_vf1

def sample_predictions(model, val_set, mean, std, channels, seed=24000):
    columns = ["id"]
    if sum(channels[:3]) == 3:
        columns.append("image")
    if channels[4]:
        columns.append("ndwi")
        # initialize mappable objects
        ndwi_norm = Normalize(vmin=-1, vmax=1)
        ndwi_map = ScalarMappable(norm=ndwi_norm, cmap='seismic_r')
    if channels[5]:
        columns.append("dem")
        dem_map = ScalarMappable(norm=None, cmap='gray')
    if channels[6]:
        columns.append("slope")
        slope_map = ScalarMappable(norm=None, cmap='jet')
    if channels[7]:
        columns.append("waterbody")
    if channels[8]:
        columns.append("roads")
    columns += ["truth", "prediction"]
    table = wandb.Table(columns=columns)

    # get map of each channel to index of resulting tensor
    n = 0
    channel_indices = [-1] * 9
    for i, channel in enumerate(channels):
        if channel:
            channel_indices[i] = n
            n += 1
    
    model.to('cpu')
    rng = Random(seed)
    samples = rng.sample(range(0, len(val_set)), 40)

    for id, k in enumerate(samples):
        # get all images to shape (H, W, C) with C = 1 or 3 (1 for grayscale, 3 for RGB)
        X, y = val_set[k]

        logits = model(X.unsqueeze(0))
        pred_y = torch.where(nn.functional.sigmoid(logits) > 0.5, 1.0, 0.0).squeeze(0) # (1, x, y)

        X = X.permute(1, 2, 0)
        X = std * X + mean

        row = [k]
        if sum(channels[:3]) == 3:
            tci = X[:, :, :3].mul(255).clamp(0, 255).byte().numpy()
            tci_img = Image.fromarray(tci, mode="RGB")
            row.append(wandb.Image(tci_img))
        if channels[4]:
            ndwi = X[:, :, channel_indices[4]]
            ndwi = ndwi_map.to_rgba(ndwi.numpy(), bytes=True)
            ndwi = np.clip(ndwi, 0, 255).astype(np.uint8)
            ndwi_img = Image.fromarray(ndwi, mode="RGBA")
            row.append(wandb.Image(ndwi_img))
        if channels[5]:
            dem = X[:, :, channel_indices[5]].numpy()
            dem_map.set_norm(Normalize(vmin=np.min(dem), vmax=np.max(dem)))
            dem = dem_map.to_rgba(dem, bytes=True)
            dem = np.clip(dem, 0, 255).astype(np.uint8)
            dem_img = Image.fromarray(dem, mode="RGBA")
            row.append(wandb.Image(dem_img))
        if channels[6]:
            slope = X[:, :, channel_indices[6]].numpy()
            slope_map.set_norm(Normalize(vmin=np.min(slope), vmax=np.max(slope)))
            slope = slope_map.to_rgba(slope, bytes=True)
            slope = np.clip(slope, 0, 255).astype(np.uint8)
            slope_img = Image.fromarray(slope, mode="RGBA")
            row.append(wandb.Image(slope_img))
        if channels[7]:
            waterbody = X[:, :, channel_indices[7]].mul(255).clamp(0, 255).byte().numpy()
            waterbody_img = Image.fromarray(waterbody, mode="L")
            row.append(wandb.Image(waterbody_img))
        if channels[8]:
            roads = X[:, :, channel_indices[8]].mul(255).clamp(0, 255).byte().numpy()
            roads_img = Image.fromarray(roads, mode="L")
            row.append(wandb.Image(roads_img))

        y = y.squeeze(0).mul(255).clamp(0, 255).byte().numpy()
        pred_y = pred_y.squeeze(0).mul(255).clamp(0, 255).byte().numpy()
        
        truth_img = Image.fromarray(y, mode="L")
        pred_img = Image.fromarray(pred_y, mode="L")
        row += [wandb.Image(truth_img), wandb.Image(pred_img)]

        table.add_data(*row)

    return table

def run_experiment(config):
    if wandb.login():
        device = (
            "cuda"
            if torch.cuda.is_available()
            else "mps"
            if torch.backends.mps.is_available()
            else "cpu"
        )
        n_channels = sum(config['channels'])
        if config['name'] == "unet":
            model = UNet(n_channels, dropout=0.2).to(device)
        else:
            raise Exception("Model not available.")
        
        print(f"Using {device} device")
        method = config['method']
        size = config['size']
        samples = config['samples']
        train_label_dir = f'data/{method}/' + f'labels{size}_train_{samples}/'
        train_sample_dir = f'data/{method}/' + f'samples{size}_train_{samples}/'
        test_label_dir = f'data/{method}/' + f'labels{size}_test_{samples}/'
        test_sample_dir = f'data/{method}/' + f'samples{size}_test_{samples}/'
        
        train_set = FloodSampleDataset(train_sample_dir, train_label_dir, channels=config['channels'], typ="train")
        val_set = FloodSampleDataset(test_sample_dir, test_label_dir, channels=config['channels'], typ="test")

        train_mean, train_std = trainMeanStd(channels=config['channels'], sample_dir=config['sample_dir'], 
                                             label_dir=config['label_dir'])
        standardize = transforms.Compose([transforms.Normalize(train_mean, train_std)])

        train_set.transform = standardize
        val_set.transform = standardize

        try:
            run, vf1 = train(train_set, val_set, model, device, config, save=f"model{len(glob('models/model*.pth'))}")
            
            # log predictions on validation set using wandb
            pred_table = sample_predictions(model, val_set, train_mean, train_std, config['channels'])
            run.log({"model_val_predictions": pred_table})
        finally:
            run.finish()

        return vf1
    else:
        raise Exception("Failed to login to wandb.")

def main(config):
    run_experiment(config)

if __name__ == '__main__':
    parser = argparse.ArgumentParser(prog='train_classifier', description='Trains classifier model from patches.')

    def bool_indices(s):
        if len(s) == 9 and all(c in '01' for c in s):
            try:
                return [bool(int(x)) for x in s]
            except ValueError:
                raise argparse.ArgumentTypeError("Invalid boolean string: '{}'".format(s))
        else:
            raise argparse.ArgumentTypeError("Boolean string must be of length 9 and have binary digits")
    
    parser.add_argument('-x', '--size', type=int, default=64, help='pixel width of patch (default: 64)')
    parser.add_argument('-n', '--samples', type=int, default=1000, help='number of samples per image (default: 1000)')
    parser.add_argument('-m', '--method', default='random', choices=['random'], help='sampling method (default: random)')
    parser.add_argument('-c', '--channels', type=bool_indices, default="111111111", help='string of 9 binary digits for selecting among the 9 available channels (R, G, B, B08, NDWI, DEM, Slope, Water, Roads) (default: 111111111)')
    parser.add_argument('--sdir', dest='sample_dir', default='../samples_200_5_4_35/', help='(default: ../samples_200_5_4_35/)')
    parser.add_argument('--ldir', dest='label_dir', default='../labels/', help='(default: ../labels/)')

    # wandb
    parser.add_argument('--project', default="FloodSamplesClassifier", help='Wandb project where run will be logged')

    # ml
    parser.add_argument('-e', '--epochs', type=int, default=30, help='(default: 30)')
    parser.add_argument('-b', '--batch_size', type=int, default=32, help='(default: 32)')
    parser.add_argument('-l', '--learning_rate', type=float, default=0.0001, help='(default: 0.0001)')
    parser.add_argument('--early_stopping', action='store_true', help='early stopping (default: False)')

    # model
    parser.add_argument('--name', default='unet', choices=MODEL_NAMES,
                        help=f"models: {', '.join(MODEL_NAMES)} (default: unet)")
    
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
