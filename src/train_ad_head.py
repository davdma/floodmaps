import wandb
import torch
import logging
import argparse
import copy
import math
import torch
from torch import nn
from torch.utils.data import DataLoader
from datetime import datetime
from dataset import DespecklerSARDataset
from loss import PseudoHuberLoss, LogCoshLoss
from utils import EarlyStopper, SaveMetrics, get_gradient_norm, get_model_params, print_model_params_and_grads
from torchvision import transforms
from architectures.autodespeckler import ConvAutoencoder1, ConvAutoencoder2, DenoiseAutoencoder, VarAutoencoder
from matplotlib.cm import ScalarMappable
from matplotlib.colors import Normalize
import matplotlib.pyplot as plt
import random
from random import Random
from PIL import Image
from glob import glob
import numpy as np
import sys
import pickle 

AUTODESPECKLER_NAMES = ['CNN1', 'CNN2', 'DAE', 'VAE']
NOISE_NAMES = ['normal', 'masking', 'log_gamma']
LOSS_NAMES = ['MSELoss', 'PseudoHuberLoss', 'HuberLoss', 'LogCoshLoss'] # pseudo huber - https://arxiv.org/pdf/2310.14189
SCHEDULER_NAMES = ['ReduceLROnPlateau', 'CosAnnealingLR'] # 'CosWarmRestarts'

def get_loss(config):
    if config['loss'] == 'MSELoss':
        return nn.MSELoss()
    elif config['loss'] == 'PseudoHuberLoss':
        return PseudoHuberLoss(c=0.03)
    elif config['loss'] == 'HuberLoss':
        return nn.HuberLoss()
    elif config['loss'] == 'LogCoshLoss':
        return LogCoshLoss()
    else:
        raise Exception(f"Loss must be one of: {', '.join(LOSS_NAMES)}")

def get_model(config):
    if config['autodespeckler'] == "CNN1":
        return ConvAutoencoder1(latent_dim=config['latent_dim'],
                                dropout=config['AD_dropout'],
                                activation_func=config['AD_activation_func'])
    elif config['autodespeckler'] == "CNN2":
        return ConvAutoencoder2(num_layers=config['AD_num_layers'], 
                                kernel_size=config['AD_kernel_size'], 
                                dropout=config['AD_dropout'], 
                                activation_func=config['AD_activation_func'])
    elif config['autodespeckler'] == "DAE":
        # need to modify with new AE architecture parameters
        return DenoiseAutoencoder(num_layers=config['AD_num_layers'],
                                  kernel_size=config['AD_kernel_size'],
                                  dropout=config['AD_dropout'],
                                  coeff=config['noise_coeff'],
                                  noise_type=config['noise_type'],
                                  activation_func=config['AD_activation_func'])
    elif config['autodespeckler'] == "VAE":
        # need to modify with new AE architecture parametersi
        return VarAutoencoder(latent_dim=config['latent_dim']) # more hyperparameters
    else:
        raise Exception('Invalid autodespeckler specified.')

def get_optimizer(model, config):
    if config['optimizer'] == 'Adam':
        optimizer = torch.optim.Adam(model.parameters(), lr=config['learning_rate'])
    elif config['optimizer'] == 'SGD':
        optimizer = torch.optim.SGD(model.parameters(), lr=config['learning_rate'])
    else:
        raise Exception('Optimizer not found.')

    return optimizer

def get_scheduler(optimizer, config):
    """Supports epoch stepping schedulers (batch step needs to be implemented)."""
    if config['LR_scheduler'] == 'ReduceLROnPlateau':
        scheduler = torch.optim.lr_scheduler.ReduceLROnPlateau(optimizer, patience=config['LR_patience'])
    elif config['LR_scheduler'] == 'CosAnnealingLR':
        scheduler = torch.optim.lr_scheduler.CosineAnnealingLR(optimizer, config['LR_T_max'], eta_min=0.000001)
    else:
        raise Exception('Scheduler not found.')
        
    #     config['scheduler_step'] = 'epoch'
    # elif config['LR_scheduler'] == 'CosWarmRestarts':
    #     scheduler = torch.optim.lr_scheduler.CosineAnnealingWarmRestarts(optimizer, config['LR_T_max'],
    #                                                                      T_mult=config['LR_T_mult'])
    #     config['scheduler_step'] = 'batch'
    # elif config['scheduler'] == 'OneCycle':
    #     scheduler = torch.optim.lr_scheduler.OneCycleLR(optimizer, max_lr, total_steps=None, epochs=None, steps_per_epoch=None, pct_start=0.3, anneal_strategy='cos', cycle_momentum=True, base_momentum=0.85, max_momentum=0.95, div_factor=25.0, final_div_factor=10000.0, three_phase=False)
    #     config['scheduler_step'] = 'batch'
    
    return scheduler

def compute_loss(out_dict, targets, loss_fn, config):
    despeckler_output = out_dict['despeckler_output']
    recons_loss = loss_fn(out_dict['despeckler_output'], targets)
    if config['autodespeckler'] == 'VAE':
        # beta hyperparameter
        log_var = torch.clamp(out_dict['log_var'], min=-6, max=6)
        mu = out_dict['mu']
        kld_loss = torch.mean(-0.5 * torch.sum(1 + log_var - mu ** 2 - log_var.exp(), dim = 1), dim = 0)
        recons_loss = recons_loss + config['VAE_beta'] * config['kld_weight'] * kld_loss
        
        if torch.isnan(recons_loss).any() or torch.isinf(recons_loss).any():
            print(f'min mu: {mu.min().item()}')
            print(f'max mu: {mu.max().item()}')
            print(f'min log_var: {log_var.min().item()}')
            print(f'max log_var: {log_var.max().item()}')
            raise Exception('recons_loss + kld_loss is nan or inf')

    return recons_loss

def train_loop(dataloader, model, device, optimizer, minibatches, loss_fn, config):
    running_recons_loss = torch.tensor(0.0, device=device)
    epoch_gradient_norm = torch.tensor(0.0, device=device)
    batches_logged = 0
    model.train()
    for batch_i, X in enumerate(dataloader):
        X = X.to(device)
        
        sar_in = X[:, :2, :, :] if not config['use_lee'] else X[:, 2:, :, :]
        out_dict = model(sar_in)

        # also pass SAR layers for reconstruction loss
        loss = compute_loss(out_dict, sar_in, loss_fn, config)
        if torch.isnan(loss).any():
            err_file_name=f"outputs/ad_param_err_train_{config['autodespeckler']}.json"
            stats_dict = print_model_params_and_grads(model, file_name=err_file_name)
            raise ValueError(f"Loss became NaN during training loop in batch {batch_i}. \
                                The input SAR was NaN: {torch.isnan(sar_in).any()}")
        
        optimizer.zero_grad()
        loss.backward()
        # Compute gradient norm, scaled by batch size
        if batch_i % config['grad_norm_freq'] == 0:
            scaled_grad_norm = get_gradient_norm(model) / config['batch_size']
            epoch_gradient_norm += scaled_grad_norm
            batches_logged += 1
            
        nn.utils.clip_grad_norm_(model.parameters(), config['clip'])
        optimizer.step()

        running_recons_loss += loss.detach()
        if batch_i >= minibatches:
            break
    
    # calculate metrics    
    epoch_recons_loss = running_recons_loss.item() / minibatches 
    avg_epoch_gradient_norm = epoch_gradient_norm.item() / batches_logged

    # wandb tracking loss and metrics per epoch - track recons loss as well
    wandb.log({"train reconstruction loss": epoch_recons_loss, 
               "train gradient norm": avg_epoch_gradient_norm})
    
    return epoch_recons_loss

def test_loop(dataloader, model, device, loss_fn, config, logging=True):
    running_recons_vloss = torch.tensor(0.0, device=device)
    num_batches = len(dataloader)
    
    model.eval()
    with torch.no_grad():
        for batch_i, X in enumerate(dataloader):
            X = X.to(device)
            sar_in = X[:, :2, :, :] if not config['use_lee'] else X[:, 2:, :, :]
            out_dict = model(sar_in)
            loss = compute_loss(out_dict, sar_in, loss_fn, config)
            if torch.isnan(loss).any():
                err_file_name=f"outputs/ad_param_err_val_{config['autodespeckler']}.json"
                stats_dict = print_model_params_and_grads(model, file_name=err_file_name)
                raise ValueError(f"Loss became NaN during validation loop in batch {batch_i}. \
                                The input SAR was NaN: {torch.isnan(sar_in).any()}")
            running_recons_vloss += loss.detach()

    epoch_recons_vloss = running_recons_vloss.item() / num_batches

    if logging:
        wandb.log({'val reconstruction loss': epoch_recons_vloss})
    
    return epoch_recons_vloss

def train(model, train_set, val_set, test_set, device, config, save_path=None):
    timestamp = datetime.now().strftime('%Y-%m-%d %H:%M:%S')
    logging.info(f'''Starting training:
        Date:            {timestamp}
        Epochs:          {config['epochs']}
        Batch size:      {config['batch_size']}
        Learning rate:   {config['learning_rate']}
        Training size:   {len(train_set)}
        Validation size: {len(val_set)}
        Test size:       {len(test_set) if test_set is not None else 'NA'}
        Device:          {device}
    ''')

    # log model parameters
    total_params, trainable_params, param_size_in_mb = get_model_params(model)
    
    # log via wandb
    run = wandb.init(
        project=config['project'],
        group=config['group'],
        config={
        "dataset": "Sentinel1",
        "use_lee": config['use_lee'],
        "mode": config['mode'],
        "patch_size": config['size'],
        "kernel_size": config['kernel_size'],
        "autodespeckler": config['autodespeckler'],
        "latent_dim": config.get('latent_dim'),
        "AD_num_layers": config.get('AD_num_layers'),
        "AD_kernel_size": config.get('AD_kernel_size'),
        "AD_dropout": config.get('AD_dropout'),
        "AD_activation_func": config.get('AD_activation_func'),
        "noise_type": config.get('noise_type'),
        "noise_coeff": config.get('noise_coeff'),
        'VAE_beta': config.get('VAE_beta'),
        "learning_rate": config['learning_rate'],
        "LR_scheduler": config['LR_scheduler'],
        "LR_patience": config['LR_patience'],
        "LR_T_max": config['LR_patience'],
        "epochs": config['epochs'],
        "early_stopping": config['early_stopping'],
        "patience": config['patience'],
        "batch_size": config['batch_size'],
        "random_flip": config['random_flip'],
        "num_workers": config['num_workers'],
        "optimizer": config['optimizer'],
        "loss_fn": config['loss'],
        "subset": config['subset'],
        "training_size": len(train_set),
        "validation_size": len(val_set),
        "test_size": len(test_set) if config['mode'] == 'test' else None,
        "val_percent": len(val_set) / (len(train_set) + len(val_set)),
        "clip": config['clip'],
        "grad_norm_freq": config['grad_norm_freq'],
        "seed": config['seed'],
        "total_parameters": total_params,
        "trainable_parameters": trainable_params,
        "parameter_size_mb": param_size_in_mb,
        "save_file": save_path
        }
    )
    
    # log weights and gradients each epoch
    wandb.watch(model, log="all", log_freq=10)

    # VAE only
    if config['autodespeckler'] == 'VAE':
        config['kld_weight'] = config['batch_size'] / len(train_set)
        
    # optimizer and scheduler for reducing learning rate
    optimizer = get_optimizer(model, config)
    scheduler = get_scheduler(optimizer, config)

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
    
    wandb.define_metric("val reconstruction loss", summary="min")
    minibatches = int(len(train_loader) * config['subset'])
    loss_fn = get_loss(config).to(device)
    for epoch in range(config['epochs']):
        try:
            # train loop
            avg_loss = train_loop(train_loader, model, device, optimizer, minibatches, loss_fn, config)
    
            # at the end of each training epoch compute validation
            avg_vloss = test_loop(val_loader, model, device, loss_fn, config)
        except Exception as err:
            raise RuntimeError(f'Error while training occurred at epoch {epoch}.') from err

        if config['early_stopping']:
            early_stopper.step(avg_vloss)
            if early_stopper.is_stopped():
                break
                
            if early_stopper.is_best_epoch():
                early_stopper.store_metric(avg_vloss)
                # Model weights are saved at the end of every epoch, if it's the best seen so far:
                min_model_weights = copy.deepcopy(model.state_dict())

        if isinstance(scheduler, torch.optim.lr_scheduler.ReduceLROnPlateau):
            scheduler.step(avg_vloss)
        else:
            scheduler.step()

        wandb.log({"learning_rate": scheduler.get_last_lr()[0], "epoch": epoch})

    # Save our model
    final_vmetrics = SaveMetrics()
    PATH = 'models/' + save_path + '.pth' if save_path is not None else None
    if config['early_stopping']:
        if config['save']:
            torch.save(min_model_weights, PATH) # save only if it is the best model so far

        # reset model to checkpoint for later sample prediction
        model.load_state_dict(min_model_weights)
        final_vmetrics.save_metrics(early_stopper.get_metric(), typ='val')
    else:
        if config['save']:
            torch.save(model.state_dict(), PATH)
        final_vmetrics.save_metrics(avg_vloss, typ='val')

    # for benchmarking purposes
    if config['mode'] == 'test':
        test_loss = test_loop(test_loader, model, device, config, logging=False)
        final_vmetrics.save_metrics(test_loss, typ='test')

    return run, final_vmetrics

# more interactive histogram?
# def log_histograms(input_image, output_image, id):
#     """
#     Logs histograms of the input and output SAR images to compare intensity distributions.
    
#     Args:
#         input_image (np.ndarray): Noisy SAR VV-VH (2, 64, 64) image (before despeckling).
#         output_image (np.ndarray): Denoised SAR VV-VH (2, 64, 64) image (after despeckling).
#         id (int): identifier for patch in dataset.
#     """
#     # Flatten images to compute histograms
#     input_values = input_image.flatten()
#     output_values = output_image.flatten()

#     # Log histograms in WandB
#     wandb.log({
#         f"SAR-in Intensity Distribution for Patch {id}": wandb.Histogram(input_values),
#         f"SAR-out Intensity Distribution for Patch {id}": wandb.Histogram(output_values),
#     })

def create_histogram_plot(input_values, output_values, min=-2, max=2):
    """
    Creates an overlaid histogram and returns a WandB Image.
    """
    fig, ax = plt.subplots(figsize=(5, 4), dpi=200)
    ax.hist(input_values, bins=50, alpha=0.5, range=(min, max), label="SAR-in", color="blue")
    ax.hist(output_values, bins=50, alpha=0.5, range=(min, max), label="SAR-out", color="orange")
    ax.legend()
    ax.set_title("Overlayed SAR Intensity Distribution")

    # Convert the figure to a WandB image
    wandb_image = wandb.Image(fig)
    plt.clf()
    plt.close(fig)  # Close figure to free memory
    return wandb_image

def sample_predictions(model, sample_set, mean, std, config, histogram=True, hist_freq=1, seed=24330):
    """Generate predictions on a subset of images in the dataset for wandb logging."""
    if config['num_sample_predictions'] <= 0:
        return None

    # enable dem visualization in dataset
    sample_set.set_include_dem(True)
    columns = ["id", 'input_vv', 'input_vh', 'enhanced_lee_vv', 'enhanced_lee_vh', 'despeckled_vv', 'despeckled_vh', 'dem'] 

    # some display options
    if config['autodespeckler'] == 'DAE':
        columns.insert(3, 'noisy_vv')
        columns.insert(4, 'noisy_vh')
    if histogram:
        columns.extend(['vv_histogram', 'vh_histogram'])
        
    table = wandb.Table(columns=columns)
    vv_map = ScalarMappable(norm=None, cmap='gray')
    vh_map = ScalarMappable(norm=None, cmap='gray')
    dem_map = ScalarMappable(norm=None, cmap='gray')
    dae_vv_map = ScalarMappable(norm=None, cmap='gray') # tmp
    dae_vh_map = ScalarMappable(norm=None, cmap='gray') # tmp

    model.to('cpu')
    model.eval()
    rng = Random(seed)
    samples = rng.sample(range(0, len(sample_set)), config['num_sample_predictions'])
    for index, k in enumerate(samples):
        # get all images to shape (H, W, C) with C = 1 or 3 (1 for grayscale, 3 for RGB)
        X = sample_set[k]

        with torch.no_grad():
            sar_in = X[:2, :, :] if not config['use_lee'] else X[2:4, :, :]
            out_dict = model(sar_in.unsqueeze(0))
            despeckler_output = out_dict['despeckler_output'].squeeze(0)
            despeckler_input = out_dict['despeckler_input'].squeeze(0) # tmp
            
        # Channels are descaled using linear variance scaling
        X = X.permute(1, 2, 0)
        # X = std * X + mean - removed as we choose to work with standardized data

        row = [k]
        # inputs, outputs and their ranges for converting to grayscale
        vv = X[:, :, 0].numpy()
        en_vv = X[:, :, 2].numpy()
        recons_vv = despeckler_output[0].numpy()
        min_vv = np.min([np.min(vv), np.min(en_vv), np.min(recons_vv)])
        max_vv = np.max([np.max(vv), np.max(en_vv), np.max(recons_vv)])
        vv_map.set_norm(Normalize(vmin=min_vv, vmax=max_vv))

        vh = X[:, :, 1].numpy()
        en_vh = X[:, :, 3].numpy()
        recons_vh = despeckler_output[1].numpy()
        min_vh = np.min([np.min(vh), np.min(en_vh), np.min(recons_vh)])
        max_vh = np.max([np.max(vh), np.max(en_vh), np.max(recons_vh)])
        vh_map.set_norm(Normalize(vmin=min_vh, vmax=max_vh))

        dem = X[:, :, 4].numpy()
        dem_map.set_norm(Normalize(vmin=np.min(dem), vmax=np.max(dem)))

        # tmp
        noisy_vv = despeckler_input[0].numpy()
        noisy_vh = despeckler_input[1].numpy()
        dae_vv_map.set_norm(Normalize(vmin=np.min(noisy_vv), vmax=np.max(noisy_vv)))
        dae_vh_map.set_norm(Normalize(vmin=np.min(noisy_vh), vmax=np.max(noisy_vh)))

        # VV input images
        vv = vv_map.to_rgba(vv, bytes=True)
        vv = np.clip(vv, 0, 255).astype(np.uint8)
        vv_img = Image.fromarray(vv, mode="RGBA")
        row.append(wandb.Image(vv_img))

        # VH input images
        vh = vh_map.to_rgba(vh, bytes=True)
        vh = np.clip(vh, 0, 255).astype(np.uint8)
        vh_img = Image.fromarray(vh, mode="RGBA")
        row.append(wandb.Image(vh_img))

        # DAE ONLY
        if config['autodespeckler'] == 'DAE':
            # noisy VV
            noisy_vv = dae_vv_map.to_rgba(noisy_vv, bytes=True)
            noisy_vv = np.clip(noisy_vv, 0, 255).astype(np.uint8)
            noisy_vv_img = Image.fromarray(noisy_vv, mode="RGBA")
            row.append(wandb.Image(noisy_vv_img))
    
            # noisy VH
            noisy_vh = dae_vh_map.to_rgba(noisy_vh, bytes=True)
            noisy_vh = np.clip(noisy_vh, 0, 255).astype(np.uint8)
            noisy_vh_img = Image.fromarray(noisy_vh, mode="RGBA")
            row.append(wandb.Image(noisy_vh_img))

        # Enhanced lee vv
        en_vv = vv_map.to_rgba(en_vv, bytes=True)
        en_vv = np.clip(en_vv, 0, 255).astype(np.uint8)
        en_vv_img = Image.fromarray(en_vv, mode="RGBA")
        row.append(wandb.Image(en_vv_img))

        # Enhanced lee vh
        en_vh = vh_map.to_rgba(en_vh, bytes=True)
        en_vh = np.clip(en_vh, 0, 255).astype(np.uint8)
        en_vh_img = Image.fromarray(en_vh, mode="RGBA")
        row.append(wandb.Image(en_vh_img))

        # reconstruction VV
        recons_vv = vv_map.to_rgba(recons_vv, bytes=True)
        recons_vv = np.clip(recons_vv, 0, 255).astype(np.uint8)
        recons_vv_img = Image.fromarray(recons_vv, mode="RGBA")
        row.append(wandb.Image(recons_vv_img))

        # reconstruction VH
        recons_vh = vh_map.to_rgba(recons_vh, bytes=True)
        recons_vh = np.clip(recons_vh, 0, 255).astype(np.uint8)
        recons_vh_img = Image.fromarray(recons_vh, mode="RGBA")
        row.append(wandb.Image(recons_vh_img))

        # dem
        dem = dem_map.to_rgba(dem, bytes=True)
        dem = np.clip(dem, 0, 255).astype(np.uint8)
        dem_img = Image.fromarray(dem, mode="RGBA")
        row.append(wandb.Image(dem_img))

        # compare intensity distributions
        if histogram:
            if index % hist_freq == 0:
                # log_histograms(sar_in.numpy(), despeckler_output.numpy(), k)
                input_values = sar_in[0].numpy().flatten()
                output_values = recons_vv.flatten()
                row.append(create_histogram_plot(input_values, output_values, min=-2, max=2))
    
                input_values = sar_in[1].numpy().flatten()
                output_values = recons_vh.flatten()
                row.append(create_histogram_plot(input_values, output_values, min=-2, max=2))
            else:
                # skip so add empty cell
                row.extend([None, None])
        
        table.add_data(*row)

    return table

def run_experiment_ad(config):
    """Run a single autodespeckler model experiment given the configuration parameters."""
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
        model = get_model(config).to(device)
        
        print(f"Using {device} device")
        model_name = config['autodespeckler']
        kernel_size = config['kernel_size']
        size = config['size']
        samples = config['samples']
        sample_dir = f'data/ad/samples_{kernel_size}_{size}_{samples}_dem/'
        save_file = f"ad/{model_name}_model{len(glob(f'models/ad/{model_name}_model*.pth'))}" if config['save'] else None

        # load in mean and std
        with open(f'data/ad/stats/{kernel_size}_{size}_{samples}_dem.pkl', 'rb') as f:
            train_mean, train_std = pickle.load(f)

            train_mean = torch.from_numpy(train_mean)
            train_std = torch.from_numpy(train_std)

        standardize = transforms.Compose([transforms.Normalize(train_mean, train_std)])
        train_set = DespecklerSARDataset(sample_dir, typ="train", transform=standardize, random_flip=config['random_flip'],
                                          seed=config['seed']+1)
        val_set = DespecklerSARDataset(sample_dir, typ="val", transform=standardize)
        test_set = DespecklerSARDataset(sample_dir, typ="test", transform=standardize) if config['mode'] == 'test' else None
        
        # initialize loss functions - train loss function is optimized for gradient calculations
        run, final_vmetrics = train(model, train_set, val_set, test_set, device, config, save_path=save_file)

        # summary metrics
        final_vloss = final_vmetrics.get_metrics(typ=config['mode'])
        run.summary[f"final_{config['mode']}_vloss"] = final_vloss
            
        # log predictions on validation set using wandb
        try:
            pred_table = sample_predictions(model, test_set if config['mode'] == 'test' else val_set, 
                                            train_mean, train_std, config)
            run.log({"model_val_predictions": pred_table})
        finally:
            run.finish()

        # if want test metrics calculate model score on test set
        return final_vmetrics
    else:
        raise Exception("Failed to login to wandb.")

def main(config):
    run_experiment_ad(config)

if __name__ == '__main__':
    parser = argparse.ArgumentParser(prog='train_ad_head', description='Trains SAR autoencoder head by itself.')

    # preprocessing
    parser.add_argument('-x', '--size', type=int, default=64, help='pixel width of dataset patches (default: 64)')
    parser.add_argument('-n', '--samples', type=int, default=500, help='number of patches sampled per image (default: 500)')
    parser.add_argument('--kernel_size', type=int, default=5, help='kernel size for enhanced lee (default: 5)')
    
    # wandb
    parser.add_argument('--project', default="SAR_AD_HEAD", help='Wandb project where run will be logged')
    parser.add_argument('--group', default=None, help='Optional group name for model experiments (default: None)')
    parser.add_argument('--num_sample_predictions', type=int, default=40, help='number of predictions to visualize (default: 40)')

    # evaluation
    parser.add_argument('--mode', default='val', choices=['val', 'test'], help=f"dataset used for evaluation metrics (default: val)")
    
    # ml
    parser.add_argument('-e', '--epochs', type=int, default=30, help='(default: 30)')
    parser.add_argument('-b', '--batch_size', type=int, default=32, help='(default: 32)')
    parser.add_argument('-s', '--subset', dest='subset', type=float, default=1.0, help='percentage of training dataset to use per epoch (default: 1.0)')
    parser.add_argument('-l', '--learning_rate', type=float, default=0.0001, help='(default: 0.0001)')
    parser.add_argument('--early_stopping', action='store_true', help='early stopping (default: False)')
    parser.add_argument('-p', '--patience', type=int, default=5, help='early stopping patience (default: 5)')
    parser.add_argument('--LR_scheduler', default='ReduceLROnPlateau', choices=SCHEDULER_NAMES,
                        help=f"LR schedulers: {', '.join(SCHEDULER_NAMES)} (default: ReduceLROnPlateau)")
    parser.add_argument('--LR_patience', type=int, default=5, help='Learning rate scheduler patience (default: 5)')
    parser.add_argument('--LR_T_max', type=int, default=200, help='Learning rate scheduler patience (default: 5)')

    # autodespeckler
    parser.add_argument('--autodespeckler', default='VAE', choices=AUTODESPECKLER_NAMES,
                        help=f"models: {', '.join(AUTODESPECKLER_NAMES)} (default: None)")
    parser.add_argument('--noise_type', default=None, choices=NOISE_NAMES,
                        help=f"models: {', '.join(NOISE_NAMES)} (default: None)")
    parser.add_argument('--noise_coeff', type=float, default=None,  help=f"noise coefficient (default: 0.1)")
    parser.add_argument('--latent_dim', default=None, type=int, help='latent dimensions (default: 200)')
    parser.add_argument('--AD_num_layers', default=None, type=int, help='Autoencoder layers (default: 5)')
    parser.add_argument('--AD_kernel_size', default=None, type=int, help='Autoencoder kernel size (default: 3)')
    parser.add_argument('--AD_dropout', default=None, type=float, help=f"(default: 0.1)")
    parser.add_argument('--AD_activation_func', default=None, choices=['leaky_relu', 'relu', 'softplus', 'mish', 'gelu', 'elu'], help=f'activations: leaky_relu, relu, softplus, mish, gelu, elu (default: leaky_relu)')
    parser.add_argument('--VAE_beta', default=1.0, type=float, help=f"(default: 1.0)")

    # data augmentation
    parser.add_argument('--use_lee', action='store_true', help='use enhanced lee filter on ad input (default: False)')
    parser.add_argument('--random_flip', action='store_true', help='Randomly flip training patches horizontally and vertically (default: False)')

    # data loading
    parser.add_argument('--num_workers', type=int, default=10, help='(default: 10)')
    
    # loss
    parser.add_argument('--loss', default='MSELoss', choices=LOSS_NAMES,
                        help=f"loss: {', '.join(LOSS_NAMES)} (default: MSELoss)")
    parser.add_argument('--clip', type=float, default=1.0, help=f"Gradient clipping max norm (default: 1.0)")
    # print statistics of current gradient and adjust norm used to clip according to the statistics
    # heuristically use 1
    
    # optimizer
    parser.add_argument('--optimizer', default='Adam', choices=['Adam', 'SGD'],
                        help=f"optimizer: {', '.join(['Adam', 'SGD'])} (default: Adam)")

    # reproducibility
    parser.add_argument('--seed', type=int, default=831002, help='seed (default: 831002)')

    # save model to file
    parser.add_argument('--save', action='store_true', help='save model to file (default: False)')

    # logging
    parser.add_argument('--grad_norm_freq', type=int, default=10, help=f"Grad norm logging batch frequency (default: 10)")

    config = vars(parser.parse_args())
    sys.exit(main(config))
