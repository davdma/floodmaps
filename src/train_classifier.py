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
from torcheval.metrics import BinaryAccuracy
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
    metric = BinaryAccuracy(threshold=0.5)

    model.train()
    for X, y in dataloader:
        X = X.to(device)
        y = y.to(device)
        
        optimizer.zero_grad()
        logits = model(X)

        loss = loss_fn(logits, y.float())
        loss.backward()
        optimizer.step()

        metric.update(nn.functional.sigmoid(logits).flatten(), y.float().flatten())
        running_loss += loss.item()

    # wandb tracking loss and accuracy per epoch
    epoch_accuracy = metric.compute()
    epoch_loss = running_loss / num_batches
    wandb.log({"train accuracy": epoch_accuracy, "train loss": epoch_loss})

    return epoch_loss

def test_loop(dataloader, model, device, loss_fn):
    running_vloss = 0.0
    num_batches = len(dataloader)
    metric = BinaryAccuracy(threshold=0.5)
    
    model.eval()
    with torch.no_grad():
        for X, y in dataloader:
            X = X.to(device)
            y = y.to(device)
            
            logits = model(X)
            running_vloss += loss_fn(logits, y.float()).item()
            metric.update(nn.functional.sigmoid(logits).flatten(), y.float().flatten())

    epoch_vaccuracy = metric.compute()
    epoch_vloss = running_vloss / num_batches
    wandb.log({"val accuracy": epoch_vaccuracy, "val loss": epoch_vloss})

    return epoch_vloss

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
        "patch_size": config['size'],
        "samples": config['samples'],
        "architecture": config['name'],
        "learning_rate": config['learning_rate'],
        "epochs": config['epochs'],
        "batch_size": config['batch_size'],
        "optimizer": config['optimizer'],
        "loss_fn": config['loss'],
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
        loss_fn = TverskyLoss()
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
        avg_vloss = test_loop(val_loader, model, device, loss_fn)

        if config['early_stopping'] and early_stopper.early_stop(avg_vloss):             
            break

        scheduler.step(avg_vloss)

    # Save our model
    PATH = 'models/' + save + '.pth'
    torch.save(model.state_dict(), PATH)

    return run

def sample_predictions(model, val_set, table, mean, std, seed=24000):
    model.to('cpu')
    rng = Random(seed)
    samples = rng.sample(range(0, len(val_set)), 20)
    for id, k in enumerate(samples):
        # get all images to shape (H, W, C) with C = 1 or 3 (1 for grayscale, 3 for RGB)
        X, y = val_set[k]

        logits = model(X.unsqueeze(0))
        pred_y = torch.where(nn.functional.sigmoid(logits) > 0.5, 1.0, 0.0).squeeze(0) # (1, x, y)

        X = X.permute(1, 2, 0)
        X = std * X + mean
        
        tci = X[:, :, :3].mul(255).clamp(0, 255).byte().numpy()
        ndwi = X[:, :, 4]
        
        ndwi_cmap='seismic_r'
        ndwi_norm = Normalize(vmin=-1, vmax=1)
        ndwi_map = ScalarMappable(norm=ndwi_norm, cmap=ndwi_cmap)

        ndwi = ndwi_map.to_rgba(ndwi.numpy(), bytes=True)
        ndwi = np.clip(ndwi, 0, 255).astype(np.uint8)

        y = y.squeeze(0).mul(255).clamp(0, 255).byte().numpy()
        pred_y = pred_y.squeeze(0).mul(255).clamp(0, 255).byte().numpy()
        
        tci_img = Image.fromarray(tci, mode="RGB")
        ndwi_img = Image.fromarray(ndwi, mode="RGBA")
        truth_img = Image.fromarray(y, mode="L")
        pred_img = Image.fromarray(pred_y, mode="L")

        table.add_data(k, wandb.Image(tci_img), wandb.Image(ndwi_img), wandb.Image(truth_img), wandb.Image(pred_img))

def run_experiment(config):
    if wandb.login():
        device = (
            "cuda"
            if torch.cuda.is_available()
            else "mps"
            if torch.backends.mps.is_available()
            else "cpu"
        )
        if config['name'] == "unet":
            model = UNet(5, dropout=0.2).to(device)
        else:
            raise Exception("Model not available.")
        
        print(f"Using {device} device")
        model = model.to(device)

        method = config['method']
        size = config['size']
        samples = config['samples']
        train_label_dir = f'data/{method}/' + f'labels{size}_train_{samples}/'
        train_sample_dir = f'data/{method}/' + f'samples{size}_train_{samples}/'
        test_label_dir = f'data/{method}/' + f'labels{size}_test_{samples}/'
        test_sample_dir = f'data/{method}/' + f'samples{size}_test_{samples}/'
        
        train_set = FloodSampleDataset(train_sample_dir, train_label_dir, typ="train")
        val_set = FloodSampleDataset(test_sample_dir, test_label_dir, typ="test")

        train_mean, train_std = trainMeanStd(sample_dir=config['sample_dir'], label_dir=config['label_dir'])
        standardize = transforms.Compose([transforms.Normalize(train_mean, train_std)])

        train_set.transform = standardize
        val_set.transform = standardize
        run = train(train_set, val_set, model, device, config, save=f"model{len(glob('models/model*.pth'))}")

        # log predictions on validation set using wandb
        columns = ["id", "image", "ndwi", "truth", "prediction"]
        pred_table = wandb.Table(columns=columns)
        
        # attach images as pytorch tensors
        sample_predictions(model, val_set, pred_table, train_mean, train_std)
        run.log({"model_val_predictions": pred_table})
        run.finish()
    else:
        raise Exception("Failed to login to wandb.")

def main(config):
    run_experiment(config)

if __name__ == '__main__':
    parser = argparse.ArgumentParser(prog='train_classifier', description='Trains classifier model from patches.')
    # dataset
    parser.add_argument('-x', '--size', type=int, default=64, help='pixel width of patch (default: 64)')
    parser.add_argument('-n', '--samples', type=int, default=1000, help='number of samples per image (default: 1000)')
    parser.add_argument('-m', '--method', default='random', choices=['random'], help='sampling method (default: random)')
    parser.add_argument('--sdir', dest='sample_dir', default='../samples_200_5_4_35/', help='(default: ../samples_200_5_4_35/)')
    parser.add_argument('--ldir', dest='label_dir', default='../labels/', help='(default: ../labels/)')

    # wandb
    parser.add_argument('--project', default="FloodSamplesUNetRC", help='Wandb project where run will be logged')

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

    # optimizer
    parser.add_argument('--optimizer', default='Adam', choices=['Adam', 'SGD'],
                        help=f"optimizer: {', '.join(['Adam', 'SGD'])} (default: Adam)")

    # config will be dict
    config = vars(parser.parse_args())

    sys.exit(main(config))
