import torch
from torchvision import transforms
import pickle

from floodmaps.models.model import S2WaterDetector, SARWaterDetector
from floodmaps.utils.config import Config
from floodmaps.utils.utils import SRC_DIR, DATA_DIR

def get_standardize_s2(cfg):
    """Get standardization transform for S2 data."""
    pre_sample_dir = f'samples_{cfg.data.size}_{cfg.data.samples}'
    channels = [bool(int(x)) for x in cfg.data.channels]
    b_channels = sum(channels[-2:])
    with open(DATA_DIR / 's2' / pre_sample_dir / f'mean_std_{cfg.data.size}_{cfg.data.samples}.pkl', 'rb') as f:
        train_mean, train_std = pickle.load(f)

        train_mean = torch.from_numpy(train_mean[channels])
        train_std = torch.from_numpy(train_std[channels])
        # make sure binary channels are 0 mean and 1 std
        if b_channels > 0:
            train_mean[-b_channels:] = 0
            train_std[-b_channels:] = 1
    standardize = transforms.Compose([transforms.Normalize(train_mean, train_std)])
    return standardize

def predict_s2(image, device='cpu'):
    """Given S2 image and additional channels (or arbitrary height and width),
    use the trained S2 model to make flood extent predictions."""
    # load config
    cfg = Config(config_file='configs/s2_unet_v1.yaml')

    # load model
    s2_model = S2WaterDetector(cfg).to(device)
    s2_model.eval()

    # preprocess
    standardize = get_standardize_s2(cfg)

    # prediction done using sliding window with overlap
    patch_size = cfg.data.size
    overlap = patch_size // 4  # 25% overlap
    stride = patch_size - overlap
    
    # Get image dimensions
    _, _, H, W = image.shape
    
    # Initialize output prediction map
    pred_map = torch.zeros((1, H, W), dtype=torch.uint8, device=device)
    count_map = torch.zeros((H, W), dtype=torch.float32, device=device)
    
    # Sliding window prediction
    with torch.no_grad():
        hit_y_edge = False
        for y in range(0, H, stride):
            if hit_y_edge:
                break
            if y + patch_size >= H:
                y = H - patch_size
                hit_y_edge = True
            for x in range(0, W, stride):
                if hit_x_edge:
                    break
                if x + patch_size >= W:
                    x = W - patch_size
                    hit_x_edge = True

                # Extract patch
                patch = image[:, :, y:y+patch_size, x:x+patch_size]
                
                # Preprocess patch
                patch = standardize(patch)
                
                # Make prediction
                output = s2_model(patch)
                if isinstance(output, dict):
                    pred = output['classifier_output']
                else:
                    pred = output
                
                # Convert to binary prediction
                pred_binary = torch.where(torch.sigmoid(pred) > 0.5, 1.0, 0.0).byte()
                
                # Add to prediction map (average overlapping regions)
                pred_map[:, y:y+patch_size, x:x+patch_size] += pred_binary.squeeze(1)
                count_map[y:y+patch_size, x:x+patch_size] += 1.0
    
    # Average overlapping predictions
    pred_map = torch.where(count_map > 0, pred_map / count_map.unsqueeze(0), pred_map)
    
    # Convert to final binary prediction
    final_pred = torch.where(pred_map > 0.5, 1.0, 0.0).byte()
    
    return final_pred.squeeze(0)  # Return (H, W) prediction

def get_standardize_s1(cfg):
    """Get standardization transform for S1 data."""
    pre_sample_dir = f'samples_{cfg.data.size}_{cfg.data.samples}_{cfg.data.filter}'
    channels = [bool(int(x)) for x in cfg.data.channels]
    b_channels = sum(channels[-2:])
    with open(DATA_DIR / 'sar' / pre_sample_dir / f'mean_std_{cfg.data.size}_{cfg.data.samples}_{cfg.data.filter}.pkl', 'rb') as f:
        train_mean, train_std = pickle.load(f)

        train_mean = torch.from_numpy(train_mean[channels])
        train_std = torch.from_numpy(train_std[channels])
        # make sure binary channels are 0 mean and 1 std
        if b_channels > 0:
            train_mean[-b_channels:] = 0
            train_std[-b_channels:] = 1
    standardize = transforms.Compose([transforms.Normalize(train_mean, train_std)])
    return standardize

def predict_s1(image, device='cpu'):
    """Given SAR image and additional channels, use the trained SAR model
    to make flood extent predictions."""
    # load config
    cfg = Config(config_file='configs/s1_unetpp_v1.yaml')
    ad_config_path = cfg.model.autodespeckler.ad_config
    ad_cfg = Config(config_file=ad_config_path) if ad_config_path is not None else None

    # load model and weights
    s1_model = SARWaterDetector(cfg, ad_cfg=ad_cfg).to(device)
    s1_model.eval()

    # preprocess - load in mean and std
    standardize = get_standardize_s1(cfg)

    # prediction done using sliding window with overlap
    patch_size = cfg.data.size
    overlap = patch_size // 4  # 25% overlap
    stride = patch_size - overlap
    
    # Get image dimensions
    _, _, H, W = image.shape
    
    # Initialize output prediction map
    pred_map = torch.zeros((H, W), dtype=torch.uint8, device=device)
    count_map = torch.zeros((H, W), dtype=torch.float32, device=device)
    
    # Sliding window prediction
    with torch.no_grad():
        hit_y_edge = False
        for y in range(0, H, stride):
            if hit_y_edge:
                break
            if y + patch_size >= H:
                y = H - patch_size
                hit_y_edge = True
            for x in range(0, W, stride):
                if hit_x_edge:
                    break
                if x + patch_size >= W:
                    x = W - patch_size
                    hit_x_edge = True

                # Extract patch
                patch = image[:, :, y:y+patch_size, x:x+patch_size]
                
                # Preprocess patch
                patch = standardize(patch)
                
                # Make prediction
                output = s1_model(patch)
                if isinstance(output, dict):
                    pred = output['classifier_output']
                else:
                    pred = output
                
                # Convert to binary prediction
                pred_binary = torch.where(torch.sigmoid(pred) > 0.5, 1.0, 0.0).byte()
                
                # Add to prediction map (average overlapping regions)
                pred_map[y:y+patch_size, x:x+patch_size] += pred_binary.squeeze(1)
                count_map[y:y+patch_size, x:x+patch_size] += 1.0
    
    # Average overlapping predictions
    pred_map = torch.where(count_map > 0, pred_map / count_map, pred_map)
    
    # Convert to final binary prediction
    final_pred = torch.where(pred_map > 0.5, 1.0, 0.0).byte()
    
    return final_pred  # Return (H, W) prediction