"""Benchmark despeckling methods on SAR test dataset.

Computes PSNR, SSIM, ENL, and Variance of Laplacian metrics for three 
despeckling modalities:
1. Raw SAR data (noisy patches as-is)
2. Enhanced Lee filtered data
3. Model filtered data (CVAE autodespeckler)

Results are saved to CSV with one row per modality.
"""
import pandas as pd
import numpy as np
import random
import pickle
import hydra
from omegaconf import DictConfig
from pathlib import Path
from pytorch_msssim import ssim
from tqdm import tqdm

import torch
from torch.utils.data import DataLoader
from torchvision import transforms

from floodmaps.models.model import build_autodespeckler
from floodmaps.training.dataset import ConditionalSARDataset, HomogenousSARDataset
from floodmaps.utils.utils import load_model_weights
from floodmaps.utils.metrics import denormalize, normalize, convert_to_amplitude, var_laplacian, enl, psnr
from floodmaps.utils.preprocess_utils import enhanced_lee_filter_batch

# dB scale bounds for VV and VH channels
VV_DB_MIN, VV_DB_MAX = -30, 0
VH_DB_MIN, VH_DB_MAX = -30, -5

# Amplitude scale bounds (derived from dB bounds)
amplitude_min_vv = 10.0 ** (VV_DB_MIN / 20.0)
amplitude_max_vv = 10.0 ** (VV_DB_MAX / 20.0)
amplitude_min_vh = 10.0 ** (VH_DB_MIN / 20.0)
amplitude_max_vh = 10.0 ** (VH_DB_MAX / 20.0)

# Modality names
MODALITIES = ['raw', 'enhanced_lee', 'model']


def calculate_metrics(dataloader, homogenous_dataloader, train_mean, train_std, model, device, cfg):
    """Calculate SAR despeckling metrics for VV and VH channels across 3 modalities.
    
    Metrics computed:
    1. PSNR (peak signal-to-noise ratio) - on amplitude scale normalized to [0, 1]
    2. SSIM (structural similarity) - on amplitude scale normalized to [0, 1]
    3. ENL (speckle suppression) - on linear power scale, homogenous patches only
    4. Variance of Laplacian (edge sharpness) - on dB scale normalized to [0, 1]

    Modalities:
    1. Raw: noisy input as-is (no filtering)
    2. Enhanced Lee: Enhanced Lee filter applied to noisy input
    3. Model: CVAE autodespeckler inference output

    Parameters
    ----------
    dataloader : torch.utils.data.DataLoader
        DataLoader for the test dataset.
    homogenous_dataloader : torch.utils.data.DataLoader
        DataLoader for the homogenous regions (for ENL).
    train_mean : torch.Tensor
        Mean of the training data for denormalization.
    train_std : torch.Tensor
        Standard deviation of the training data for denormalization.
    model : torch.nn.Module
        Trained autodespeckler model.
    device : torch.device
        Device to evaluate on.
    cfg : DictConfig
        Configuration object.

    Returns
    -------
    pd.DataFrame
        DataFrame with one row per modality and columns for each metric.
    """
    train_mean = train_mean.to(device)
    train_std = train_std.to(device)
    
    # Initialize accumulators for each modality
    accumulators = {mod: {
        'psnr_vv': 0.0, 'psnr_vh': 0.0,
        'ssim_vv': 0.0, 'ssim_vh': 0.0,
        'var_lap_vv': 0.0, 'var_lap_vh': 0.0,
        'enl_vv': 0.0, 'enl_vh': 0.0,
    } for mod in MODALITIES}
    count = 0
    count_homogenous = 0
    
    model.eval()
    
    # Main test loop (PSNR, SSIM, Var Laplacian)
    print("Computing PSNR, SSIM, Variance of Laplacian on test set...")
    for X, y in tqdm(dataloader, desc="Test batches"):
        X = X.to(device)
        y = y.to(device)
        B = X.shape[0]
        
        with torch.no_grad():
            # Denormalize inputs to dB scale
            db_noisy = denormalize(X, train_mean, train_std)  # (B, 2, H, W) in dB
            db_clean = denormalize(y, train_mean, train_std)  # (B, 2, H, W) in dB
            
            # Ground truth in amplitude scale normalized to [0, 1]
            amp_clean_vv = normalize(convert_to_amplitude(db_clean[:, 0]), vmin=amplitude_min_vv, vmax=amplitude_max_vv)
            amp_clean_vh = normalize(convert_to_amplitude(db_clean[:, 1]), vmin=amplitude_min_vh, vmax=amplitude_max_vh)
            
            # === Modality 1: Raw (noisy input as-is) ===
            db_raw = db_noisy
            amp_raw_vv = normalize(convert_to_amplitude(db_raw[:, 0]), vmin=amplitude_min_vv, vmax=amplitude_max_vv)
            amp_raw_vh = normalize(convert_to_amplitude(db_raw[:, 1]), vmin=amplitude_min_vh, vmax=amplitude_max_vh)
            
            # === Modality 2: Enhanced Lee Filter ===
            # Filter operates on CPU (numba), then convert result back to torch on device
            db_noisy_np = db_noisy.cpu().numpy()
            db_lee_np = enhanced_lee_filter_batch(db_noisy_np)
            db_lee = torch.from_numpy(db_lee_np).to(device, dtype=torch.float32)
            amp_lee_vv = normalize(convert_to_amplitude(db_lee[:, 0]), vmin=amplitude_min_vv, vmax=amplitude_max_vv)
            amp_lee_vh = normalize(convert_to_amplitude(db_lee[:, 1]), vmin=amplitude_min_vh, vmax=amplitude_max_vh)
            
            # === Modality 3: Model inference ===
            out_dict = model.inference(X, deterministic=True)
            db_model = denormalize(out_dict['despeckler_output'], train_mean, train_std)
            amp_model_vv = normalize(convert_to_amplitude(db_model[:, 0]), vmin=amplitude_min_vv, vmax=amplitude_max_vv)
            amp_model_vh = normalize(convert_to_amplitude(db_model[:, 1]), vmin=amplitude_min_vh, vmax=amplitude_max_vh)
            
            # Compute PSNR for each modality
            accumulators['raw']['psnr_vv'] += psnr(amp_raw_vv, amp_clean_vv).sum().item()
            accumulators['raw']['psnr_vh'] += psnr(amp_raw_vh, amp_clean_vh).sum().item()
            accumulators['enhanced_lee']['psnr_vv'] += psnr(amp_lee_vv, amp_clean_vv).sum().item()
            accumulators['enhanced_lee']['psnr_vh'] += psnr(amp_lee_vh, amp_clean_vh).sum().item()
            accumulators['model']['psnr_vv'] += psnr(amp_model_vv, amp_clean_vv).sum().item()
            accumulators['model']['psnr_vh'] += psnr(amp_model_vh, amp_clean_vh).sum().item()
            
            # Compute SSIM for each modality (requires (B, 1, H, W) format)
            accumulators['raw']['ssim_vv'] += ssim(amp_raw_vv.unsqueeze(1), amp_clean_vv.unsqueeze(1),
                                                    data_range=1, size_average=False).sum().item()
            accumulators['raw']['ssim_vh'] += ssim(amp_raw_vh.unsqueeze(1), amp_clean_vh.unsqueeze(1),
                                                    data_range=1, size_average=False).sum().item()
            accumulators['enhanced_lee']['ssim_vv'] += ssim(amp_lee_vv.unsqueeze(1), amp_clean_vv.unsqueeze(1),
                                                             data_range=1, size_average=False).sum().item()
            accumulators['enhanced_lee']['ssim_vh'] += ssim(amp_lee_vh.unsqueeze(1), amp_clean_vh.unsqueeze(1),
                                                             data_range=1, size_average=False).sum().item()
            accumulators['model']['ssim_vv'] += ssim(amp_model_vv.unsqueeze(1), amp_clean_vv.unsqueeze(1),
                                                      data_range=1, size_average=False).sum().item()
            accumulators['model']['ssim_vh'] += ssim(amp_model_vh.unsqueeze(1), amp_clean_vh.unsqueeze(1),
                                                      data_range=1, size_average=False).sum().item()
            
            # Compute Variance of Laplacian for each modality (on dB scale normalized to [0, 1])
            # Raw
            db_raw_vv_norm = normalize(db_raw[:, 0], vmin=VV_DB_MIN, vmax=VV_DB_MAX)
            db_raw_vh_norm = normalize(db_raw[:, 1], vmin=VH_DB_MIN, vmax=VH_DB_MAX)
            db_raw_concat = torch.cat([db_raw_vv_norm.unsqueeze(1), db_raw_vh_norm.unsqueeze(1)], dim=1)
            var_lap_raw = var_laplacian(db_raw_concat)
            accumulators['raw']['var_lap_vv'] += var_lap_raw[:, 0].sum().item()
            accumulators['raw']['var_lap_vh'] += var_lap_raw[:, 1].sum().item()
            
            # Enhanced Lee
            db_lee_vv_norm = normalize(db_lee[:, 0], vmin=VV_DB_MIN, vmax=VV_DB_MAX)
            db_lee_vh_norm = normalize(db_lee[:, 1], vmin=VH_DB_MIN, vmax=VH_DB_MAX)
            db_lee_concat = torch.cat([db_lee_vv_norm.unsqueeze(1), db_lee_vh_norm.unsqueeze(1)], dim=1)
            var_lap_lee = var_laplacian(db_lee_concat)
            accumulators['enhanced_lee']['var_lap_vv'] += var_lap_lee[:, 0].sum().item()
            accumulators['enhanced_lee']['var_lap_vh'] += var_lap_lee[:, 1].sum().item()
            
            # Model
            db_model_vv_norm = normalize(db_model[:, 0], vmin=VV_DB_MIN, vmax=VV_DB_MAX)
            db_model_vh_norm = normalize(db_model[:, 1], vmin=VH_DB_MIN, vmax=VH_DB_MAX)
            db_model_concat = torch.cat([db_model_vv_norm.unsqueeze(1), db_model_vh_norm.unsqueeze(1)], dim=1)
            var_lap_model = var_laplacian(db_model_concat)
            accumulators['model']['var_lap_vv'] += var_lap_model[:, 0].sum().item()
            accumulators['model']['var_lap_vh'] += var_lap_model[:, 1].sum().item()
            
            count += B
    
    # Homogenous loop (ENL only)
    print("Computing ENL on homogenous patches...")
    for X, y in tqdm(homogenous_dataloader, desc="Homogenous batches"):
        X = X.to(device)
        B = X.shape[0]
        
        with torch.no_grad():
            # Denormalize to dB scale
            db_noisy = denormalize(X, train_mean, train_std)
            
            # === Modality 1: Raw ===
            db_raw = db_noisy
            power_raw = torch.float_power(10, db_raw / 10)
            accumulators['raw']['enl_vv'] += enl(power_raw[:, 0]).sum().item()
            accumulators['raw']['enl_vh'] += enl(power_raw[:, 1]).sum().item()
            
            # === Modality 2: Enhanced Lee Filter ===
            db_noisy_np = db_noisy.cpu().numpy()
            db_lee_np = enhanced_lee_filter_batch(db_noisy_np)
            db_lee = torch.from_numpy(db_lee_np).to(device, dtype=torch.float32)
            power_lee = torch.float_power(10, db_lee / 10)
            accumulators['enhanced_lee']['enl_vv'] += enl(power_lee[:, 0]).sum().item()
            accumulators['enhanced_lee']['enl_vh'] += enl(power_lee[:, 1]).sum().item()
            
            # === Modality 3: Model inference ===
            out_dict = model.inference(X, deterministic=True)
            db_model = denormalize(out_dict['despeckler_output'], train_mean, train_std)
            power_model = torch.float_power(10, db_model / 10)
            accumulators['model']['enl_vv'] += enl(power_model[:, 0]).sum().item()
            accumulators['model']['enl_vh'] += enl(power_model[:, 1]).sum().item()
            
            count_homogenous += B
    
    # Build results DataFrame
    results = []
    for mod in MODALITIES:
        results.append({
            'modality': mod,
            'psnr_vv': accumulators[mod]['psnr_vv'] / count,
            'psnr_vh': accumulators[mod]['psnr_vh'] / count,
            'ssim_vv': accumulators[mod]['ssim_vv'] / count,
            'ssim_vh': accumulators[mod]['ssim_vh'] / count,
            'var_lap_vv': accumulators[mod]['var_lap_vv'] / count,
            'var_lap_vh': accumulators[mod]['var_lap_vh'] / count,
            'enl_vv': accumulators[mod]['enl_vv'] / count_homogenous,
            'enl_vh': accumulators[mod]['enl_vh'] / count_homogenous,
        })
    
    return pd.DataFrame(results)


def run_benchmark_ad(cfg):
    """Run despeckling benchmarks for S1 SAR autodespeckler model.

    Loads the model, test dataset, and homogenous patches dataset,
    computes metrics for 3 despeckling modalities, and saves results to CSV.

    Parameters
    ----------
    cfg : DictConfig
        Config object for the SAR autodespeckler benchmark.
    """
    # Seeding for reproducibility
    np.random.seed(cfg.seed)
    random.seed(cfg.seed)
    torch.manual_seed(cfg.seed)

    # Device setup (single GPU or CPU)
    if torch.cuda.is_available():
        device = torch.device("cuda", 0)
    else:
        device = torch.device("cpu")
        print("Warning: CUDA not available, running on CPU (will be slow)")

    print(f"Using {device} device")

    # Build and load model
    autodespeckler = build_autodespeckler(cfg).to(device)
    
    # Load pretrained weights
    if hasattr(cfg.model, 'weights') and cfg.model.weights is not None:
        load_model_weights(autodespeckler, cfg.model.weights, device, model_name="Autodespeckler")
    else:
        raise ValueError("Model weights must be specified in cfg.model.weights")

    # Dataset path construction
    method = cfg.data.method
    size = cfg.data.size
    sample_param = cfg.data.samples if cfg.data.method == 'random' else cfg.data.stride
    suffix = getattr(cfg.data, 'suffix', '')
    if suffix:
        sample_dir = Path(cfg.paths.preprocess_dir) / 's1_multi' / f'{method}_{size}_{sample_param}_{suffix}/'
    else:
        sample_dir = Path(cfg.paths.preprocess_dir) / 's1_multi' / f'{method}_{size}_{sample_param}/'

    print(f"Loading data from: {sample_dir}")

    # Load training mean and std for normalization
    with open(sample_dir / 'mean_std.pkl', 'rb') as f:
        train_mean, train_std = pickle.load(f)
        train_mean = torch.from_numpy(train_mean)
        train_std = torch.from_numpy(train_std)

    standardize = transforms.Compose([transforms.Normalize(train_mean, train_std)])

    # Create datasets
    test_set = ConditionalSARDataset(sample_dir, typ="test", transform=standardize)
    test_homogenous_set = HomogenousSARDataset(sample_dir, typ="test", transform=standardize)

    print(f"Test set size: {len(test_set)}")
    print(f"Homogenous set size: {len(test_homogenous_set)}")

    # Create dataloaders (no DistributedSampler - single process)
    test_loader = DataLoader(
        test_set,
        batch_size=cfg.train.batch_size,
        num_workers=cfg.train.num_workers,
        persistent_workers=cfg.train.num_workers > 0,
        pin_memory=True,
        shuffle=False
    )
    
    test_homogenous_loader = DataLoader(
        test_homogenous_set,
        batch_size=cfg.train.batch_size,
        num_workers=0,
        shuffle=False,
        drop_last=False
    )

    # Calculate metrics for all 3 modalities
    metrics_df = calculate_metrics(
        test_loader, 
        test_homogenous_loader,
        train_mean, 
        train_std, 
        autodespeckler, 
        device, 
        cfg
    )

    # Save results to CSV
    save_dir = Path(cfg.benchmarking.save_dir)
    save_dir.mkdir(parents=True, exist_ok=True)
    config_id = cfg.benchmarking.config_id
    csv_path = save_dir / f"{config_id}_despeckling_metrics.csv"
    
    metrics_df.to_csv(csv_path, index=False)
    print(f"\nResults saved to: {csv_path}")
    print("\nDespeckling Benchmark Results:")
    print(metrics_df.to_string(index=False))
    
    return metrics_df


def validate_config(cfg):
    """Validate configuration for despeckling benchmark."""
    assert cfg.model.autodespeckler == 'CVAE', "Model must be CVAE"
    assert cfg.model.weights is not None, "Model weights must be specified"
    assert cfg.eval.mode == "test", "Evaluation mode must be test"
    assert hasattr(cfg, 'benchmarking'), "benchmarking config section required"
    assert hasattr(cfg.benchmarking, 'save_dir'), "benchmarking.save_dir required"
    assert hasattr(cfg.benchmarking, 'config_id'), "benchmarking.config_id required"


@hydra.main(version_base=None, config_path="pkg://configs", config_name="config.yaml")
def main(cfg: DictConfig):
    validate_config(cfg)
    run_benchmark_ad(cfg)


if __name__ == '__main__':
    main()
