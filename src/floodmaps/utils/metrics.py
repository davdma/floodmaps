from skimage.feature import graycomatrix, graycoprops
import torch
import numpy as np
import random
from torch.utils.data import DataLoader, Subset
import torch.nn.functional as F
from skimage.metrics import structural_similarity as ssim
from torchmetrics.metric import Metric
from torch import Tensor

# Define standard NLCD groups for flood mapping
NLCD_GROUPS = {
    'water': [11],
    'urban': [21, 22, 23, 24],
    'forest': [41, 42, 43],
    'cultivated': [81, 82],
    'shrubland': [51, 52],
    'wetlands': [90, 95],
    'barren': [31],
    'herbaceous': [71, 72, 73, 74],
    'ice/snow': [12]
}
NLCD_CLASSES = [11, 12, 21, 22, 23, 24, 31, 41, 42, 43, 51, 52, 71, 72, 73, 74, 81, 82, 90, 95]
SCL_GROUPS = {
    'topographic shadow': [2],
    'cloud shadow': [3],
    'vegetation': [4],
    'not vegetated': [5],
    'water': [6],
    'cloud': [8, 9],
    'cirrus': [10]
}
SCL_CLASSES = list(range(12))

class PerClassConfusionMatrix(Metric):
    """Compute the binary confusion matrix for each predefined class."""
    full_state_update = False

    def __init__(self, threshold: float = 0.5, classes: list[int] = None, **kwargs):
        super().__init__(**kwargs)
        if classes is None or not isinstance(classes, list) or len(classes) == 0:
            raise ValueError("Classes must be provided as a non-empty list of integers")
        if not all(isinstance(c, int) for c in classes):
            raise ValueError("Class values must be integers")
        if len(set(classes)) != len(classes):
            raise ValueError("Class values must be unique")

        C = len(classes)
        self.threshold = threshold
        self.classes = classes
        self.class_to_idx = {c: i for i, c in enumerate(classes)}
        self.add_state(f"confmat", torch.zeros(C, 2, 2, dtype=torch.long), dist_reduce_fx="sum")
    
    def update(self, preds: Tensor, targets: Tensor, classes: Tensor):
        preds = preds > self.threshold
        preds = preds.long()
        targets = targets.long()
        classes = classes.long()
        for i, c in enumerate(self.classes):
            mask = classes == c
            preds_by_class = preds[mask]
            targets_by_class = targets[mask]

            # compute confusion matrix
            confmat = fast_binary_confusion_matrix(preds_by_class, targets_by_class)
            self.confmat[i] += confmat
    
    def compute(self):
        return self.confmat
    
    def get_class_to_idx(self):
        return self.class_to_idx

class RunningMeanVar(Metric):
    """Computes the running mean and variance of a tensor using Welford's algorithm."""
    full_state_update = False

    def __init__(self, unbiased: bool = False, **kwargs):
        super().__init__(**kwargs)
        self.unbiased = unbiased

        # IMPORTANT: dist_reduce_fx=None so we can merge correctly ourselves in compute()
        self.add_state("count", default=torch.tensor(0.0), dist_reduce_fx=None)
        self.add_state("mean",  default=torch.tensor(0.0), dist_reduce_fx=None)
        self.add_state("M2",    default=torch.tensor(0.0), dist_reduce_fx=None)

    @torch.no_grad()
    def update(self, x: torch.Tensor):
        x = x.detach().float().reshape(-1)
        if x.numel() == 0:
            return

        b_count = torch.tensor(float(x.numel()), device=x.device)
        b_mean  = x.mean()
        b_M2    = ((x - b_mean) ** 2).sum()

        if self.count == 0:
            self.count, self.mean, self.M2 = b_count, b_mean, b_M2
            return

        delta = b_mean - self.mean
        new_count = self.count + b_count
        self.mean = self.mean + delta * (b_count / new_count)
        self.M2   = self.M2 + b_M2 + delta**2 * (self.count * b_count / new_count)
        self.count = new_count

    def compute(self):
        # After TorchMetrics sync, with dist_reduce_fx=None, each state may become a vector
        # over ranks (or remain scalar in non-DDP). We fold-merge if needed.
        count = self.count
        mean  = self.mean
        M2    = self.M2

        if count.ndim == 0:
            total_count, total_mean, total_M2 = count, mean, M2
        else:
            total_count = torch.tensor(0.0, device=count.device)
            total_mean  = torch.tensor(0.0, device=count.device)
            total_M2    = torch.tensor(0.0, device=count.device)

            for c, m, s in zip(count, mean, M2):
                if c == 0:
                    continue
                if total_count == 0:
                    total_count, total_mean, total_M2 = c, m, s
                    continue
                delta = m - total_mean
                new_count = total_count + c
                total_mean = total_mean + delta * (c / new_count)
                total_M2   = total_M2 + s + delta**2 * (total_count * c / new_count)
                total_count = new_count

        if total_count <= 0:
            return {"mean": torch.tensor(float("nan")), "var": torch.tensor(float("nan"))}

        if self.unbiased:
            denom = total_count - 1.0
            var = total_M2 / denom if denom > 0 else torch.tensor(float("nan"), device=total_M2.device)
        else:
            var = total_M2 / total_count

        return {"mean": total_mean, "var": var.clamp_min(0.0)}

def fast_binary_confusion_matrix(preds, targets):
    """Simplifies the implementation of pytorch lightning binary confusion matrix logic."""
    unique_mapping = (targets * 2 + preds).to(torch.long)
    bins = torch.bincount(unique_mapping, minlength=4)
    return bins.reshape(2, 2)

def _compute_group_metrics(TN, FP, FN, TP):
    """Compute accuracy, precision, recall, F1, and IoU from confusion matrix values.
    
    Parameters
    ----------
    TN, FP, FN, TP : int
        Confusion matrix values.
    
    Returns
    -------
    dict
        Dictionary with 'acc', 'prec', 'rec', 'f1', 'iou' keys.
    """
    total = TP + TN + FP + FN
    return {
        'acc': (TP + TN) / total if total > 0 else None,
        'prec': TP / (TP + FP) if (TP + FP) > 0 else None,
        'rec': TP / (TP + FN) if (TP + FN) > 0 else None,
        'f1': (2 * TP) / (2 * TP + FP + FN) if (2 * TP + FP + FN) > 0 else None,
        'iou': TP / (TP + FP + FN) if (TP + FP + FN) > 0 else None
    }

def compute_confmat_dict(confmat, class_to_idx, classes, groups=None):
    """Convert PerClassConfusionMatrix output to dictionary format.
    
    Parameters
    ----------
    confmat : Tensor
        Confusion matrix tensor of shape (C, 2, 2) from PerClassConfusionMatrix.compute().
    class_to_idx : dict
        Mapping from class value to index in confmat.
    classes : list[int]
        List of all class values to include in output.
    groups : dict, optional
        Dictionary mapping group names to lists of class values. Defaults to None.
    
    Returns
    -------
    output : dict
        Dictionary with 'confusion_matrix', 'group_confusion_matrix', and 'group_metrics'.
        
        The dictionary has the following structure:
        {
            'confusion_matrix': {
                '21': {'tn': 0, 'fp': 0, 'fn': 0, 'tp': 0},
                ...
            },
            'group_confusion_matrix': {
                'urban': {'tn': 0, 'fp': 0, 'fn': 0, 'tp': 0},
                ...
            },
            'group_metrics': {
                'urban': {'acc': 0.52, 'prec': 0.12, 'rec': 0.90, 'f1': 0.53, 'iou': 0.50},
                ...
            }

            Note: if metrics are undefined, they are set to None.
        }
    """
    output = {'confusion_matrix': {}, 'group_confusion_matrix': {}, 'group_metrics': {}}
    
    # Per-class confusion matrices
    for c in classes:
        if c in class_to_idx:
            idx = class_to_idx[c]
            cm = confmat[idx].tolist()
            output['confusion_matrix'][str(c)] = {
                'tn': cm[0][0], 'fp': cm[0][1],
                'fn': cm[1][0], 'tp': cm[1][1]
            }
        else:
            output['confusion_matrix'][str(c)] = {'tn': 0, 'fp': 0, 'fn': 0, 'tp': 0}
    
    # Group confusion matrices and metrics
    for group_name, group_classes in groups.items():
        # Sum confusion matrices for all classes in group
        group_cm = torch.zeros(2, 2, dtype=torch.long, device=confmat.device)
        for c in group_classes:
            if c in class_to_idx:
                idx = class_to_idx[c]
                group_cm += confmat[idx]
        
        TN, FP, FN, TP = group_cm[0, 0].item(), group_cm[0, 1].item(), group_cm[1, 0].item(), group_cm[1, 1].item()
        output['group_confusion_matrix'][group_name] = {'tn': TN, 'fp': FP, 'fn': FN, 'tp': TP}
        output['group_metrics'][group_name] = _compute_group_metrics(TN, FP, FN, TP)
    
    return output

def compute_nlcd_metrics(all_preds, all_targets, nlcd_classes, groups=NLCD_GROUPS):
    """Compute metrics for NLCD classes.

    Parameters
    ----------
    all_preds : tensor
        The predicted labels.
    all_targets : tensor
        The true labels.
    nlcd_classes : tensor
        The NLCD classes.
    groups : dict, optional
        The NLCD groups for computing acc, prec, rec, f1, IoU metrics by group.

    Returns
    -------
    output : dict
    
    The returned dictionary contains:
    - the confusion matrix for each NLCD class.
    - the confusion matrix for each NLCD group.
    - the acc, prec, rec, f1, IoU for each NLCD grouping (urban, vegetation, water, wetland etc.)
    
    The dictionary has the following structure:
    {
        'confusion_matrix': {
            '21': {'tn': 0, 'fp': 0, 'fn': 0, 'tp': 0},
            '22': {'tn': 0, 'fp': 0, 'fn': 0, 'tp': 0}},
            '23': {'tn': 0, 'fp': 0, 'fn': 0, 'tp': 0},
            ...
        },
        'group_confusion_matrix': {
            'urban': {'tn': 0, 'fp': 0, 'fn': 0, 'tp': 0},
            'vegetation': {'tn': 0, 'fp': 0, 'fn': 0, 'tp': 0},
            ...
        },
        'group_metrics': {
            'urban': {'acc': 0.52, 'prec': 0.12, 'rec': 0.90, 'f1': 0.53, 'iou': 0.50},
            'vegetation': {'acc': 0.90, 'prec': 0.98, 'rec': 0.82, 'f1': 0.90, 'iou': 0.4},
            ...
        }

        Note: if metrics are undefined, they are set to None.
    }
    """
    # convert boolean to integer for the binary confusion matrix computation
    all_preds = all_preds.long()
    all_targets = all_targets.long()
    nlcd_classes = nlcd_classes.long() # originally float32
    
    output = {'confusion_matrix': {}, 'group_confusion_matrix': {}, 'group_metrics': {}}
    for nlcd_class in NLCD_CLASSES:
        mask = nlcd_classes == nlcd_class
        pixel_count = int(mask.sum())
        if pixel_count == 0:
            output['confusion_matrix'][str(nlcd_class)] = {'tn': 0, 'fp': 0, 'fn': 0, 'tp': 0}
            continue
        
        preds_by_class = all_preds[mask]
        targets_by_class = all_targets[mask]

        # compute confusion matrix
        confusion_matrix = fast_binary_confusion_matrix(preds_by_class, targets_by_class).tolist()
        output['confusion_matrix'][str(nlcd_class)] = {
            'tn': confusion_matrix[0][0],
            'fp': confusion_matrix[0][1],
            'fn': confusion_matrix[1][0],
            'tp': confusion_matrix[1][1]
        }

    # compute group metrics
    for group, classes in groups.items():
        # compute confusion matrix for each group
        mask = torch.isin(nlcd_classes, torch.tensor(classes, device=nlcd_classes.device, dtype=nlcd_classes.dtype))
        pixel_count = int(mask.sum())
        if pixel_count == 0:
            output['group_confusion_matrix'][group] = {'tn': 0, 'fp': 0, 'fn': 0, 'tp': 0}
            output['group_metrics'][group] = {'acc': None, 'prec': None, 'rec': None, 'f1': None, 'iou': None}
            continue
        
        preds_by_group = all_preds[mask]
        targets_by_group = all_targets[mask]
        group_confusion_matrix = fast_binary_confusion_matrix(preds_by_group, targets_by_group).tolist()
        (TN, FP), (FN, TP) = group_confusion_matrix
        output['group_confusion_matrix'][group] = {
            'tn': TN,
            'fp': FP,
            'fn': FN,
            'tp': TP
        }
        
        output['group_metrics'][group] = {
            'acc': (TP + TN) / (TP + TN + FP + FN),
            'prec': TP / (TP + FP) if (TP + FP) > 0 else None,
            'rec': TP / (TP + FN) if (TP + FN) > 0 else None,
            'f1': (2 * TP) / (2 * TP + FP + FN) if (2 * TP + FP + FN) > 0 else None,
            'iou': TP / (TP + FP + FN) if (TP + FP + FN) > 0 else None
        }

    return output

def compute_scl_metrics(all_preds, all_targets, scl_classes, groups=SCL_GROUPS):
    """Compute metrics for SCL classes.

    Parameters
    ----------
    all_preds : tensor
        The predicted labels.
    all_targets : tensor
        The true labels.
    scl_classes : tensor
        The SCL classes.
    groups : dict, optional
        The SCL groups for computing acc, prec, rec, f1, IoU metrics by group.

    Returns
    -------
    output : dict
    
    The returned dictionary contains:
    - the confusion matrix for each SCL class.
    - the confusion matrix for each SCL group.
    - the acc, prec, rec, f1, IoU for each SCL grouping (topographic shadow, cloud shadows, vegetation, not vegetated, water, unclassified, cloud, cirrus etc.)
    
    The dictionary has the following structure:
    {
        'confusion_matrix': {
            '21': {'tn': 0, 'fp': 0, 'fn': 0, 'tp': 0},
            '22': {'tn': 0, 'fp': 0, 'fn': 0, 'tp': 0}},
            '23': {'tn': 0, 'fp': 0, 'fn': 0, 'tp': 0},
            ...
        },
        'group_confusion_matrix': {
            'topographic shadow': {'tn': 0, 'fp': 0, 'fn': 0, 'tp': 0},
            'vegetation': {'tn': 0, 'fp': 0, 'fn': 0, 'tp': 0},
            ...
        },
        'group_metrics': {
            'topographic shadow': {'acc': 0.52, 'prec': 0.12, 'rec': 0.90, 'f1': 0.53, 'iou': 0.50},
            'vegetation': {'acc': 0.90, 'prec': 0.98, 'rec': 0.82, 'f1': 0.90, 'iou': 0.4},
            ...
        }

        Note: if metrics are undefined, they are set to None.
    }
    """
    # convert boolean to integer for the binary confusion matrix computation
    all_preds = all_preds.long()
    all_targets = all_targets.long()
    scl_classes = scl_classes.long() # originally float32
    
    output = {'confusion_matrix': {}, 'group_confusion_matrix': {}, 'group_metrics': {}}
    for scl_class in SCL_CLASSES:
        mask = scl_classes == scl_class
        pixel_count = int(mask.sum())
        if pixel_count == 0:
            output['confusion_matrix'][str(scl_class)] = {'tn': 0, 'fp': 0, 'fn': 0, 'tp': 0}
            continue
        
        preds_by_class = all_preds[mask]
        targets_by_class = all_targets[mask]

        # compute confusion matrix
        confusion_matrix = fast_binary_confusion_matrix(preds_by_class, targets_by_class).tolist()
        output['confusion_matrix'][str(scl_class)] = {
            'tn': confusion_matrix[0][0],
            'fp': confusion_matrix[0][1],
            'fn': confusion_matrix[1][0],
            'tp': confusion_matrix[1][1]
        }

    # compute group metrics
    for group, classes in groups.items():
        # compute confusion matrix for each group
        mask = torch.isin(scl_classes, torch.tensor(classes, device=scl_classes.device, dtype=scl_classes.dtype))
        pixel_count = int(mask.sum())
        if pixel_count == 0:
            output['group_confusion_matrix'][group] = {'tn': 0, 'fp': 0, 'fn': 0, 'tp': 0}
            output['group_metrics'][group] = {'acc': None, 'prec': None, 'rec': None, 'f1': None, 'iou': None}
            continue
        
        preds_by_group = all_preds[mask]
        targets_by_group = all_targets[mask]
        group_confusion_matrix = fast_binary_confusion_matrix(preds_by_group, targets_by_group).tolist()
        (TN, FP), (FN, TP) = group_confusion_matrix
        output['group_confusion_matrix'][group] = {
            'tn': TN,
            'fp': FP,
            'fn': FN,
            'tp': TP
        }
        
        output['group_metrics'][group] = {
            'acc': (TP + TN) / (TP + TN + FP + FN),
            'prec': TP / (TP + FP) if (TP + FP) > 0 else None,
            'rec': TP / (TP + FN) if (TP + FN) > 0 else None,
            'f1': (2 * TP) / (2 * TP + FP + FN) if (2 * TP + FP + FN) > 0 else None,
            'iou': TP / (TP + FP + FN) if (TP + FP + FN) > 0 else None
        }

    return output

### Evaluation metrics for SAR despeckling
def denormalize(tensor, mean, std):
    """Reverts standardization back to the original dB scale.
    Parameters
    ----------
    tensor: shape (N, C, H, W)
    mean: shape (C,)
    std: shape (C,)
    """
    mean = mean.view(1, -1, 1, 1)
    std = std.view(1, -1, 1, 1)
    return tensor * std + mean

def TV_loss(img, weight=1, per_pixel=False):
    """TV Loss calculated on denormalized dB scale sar."""
    n_img, c_img, h_img, w_img = img.shape
    tv_h = torch.pow(img[..., 1:, :]-img[..., :-1, :], 2).sum()
    tv_w = torch.pow(img[..., :, 1:]-img[..., :, :-1], 2).sum()

    # per pixel or per patch
    num = n_img*c_img*h_img*w_img if per_pixel else n_img
    return weight*(tv_h+tv_w)/(num)

def var_laplacian(img, per_pixel=False):
    """Convolution of Laplace Operator as described in https://ieeexplore.ieee.org/document/7894491.
    The implementation convolves the laplace operator separately over each
    channel, and then sums their variance (and divides by 2).

    Parameters
    ----------
    img : tensor[b, 2, h, w]
        denormalized in dB sar data with 2 polarization channels.
    per_pixel : bool
        If true then accounts for patch size, otherwise averages by patch.
    """
    n_img, c_img, h_img, w_img = img.shape
    # Positive laplacian kernel for 2 channels
    laplacian_kernel = torch.tensor([[[[0, 1, 0],
                                        [1, -4, 1],
                                        [0, 1, 0]]]], dtype=img.dtype, device=img.device)
    # stack for c channels
    all_kernels = laplacian_kernel.expand(c_img, -1, -1, -1)

    # Apply Laplacian filter via 2D convolution
    laplacian = F.conv2d(img, all_kernels, padding=1, groups=c_img)

    # Compute variance of Laplacian over each image
    var_laplacian = laplacian.view(n_img, c_img, -1).var(dim=2).sum()

    num = n_img*c_img*h_img*w_img if per_pixel else n_img*c_img
    return var_laplacian / num

def ssi(noisy, filt, per_pixel=False):
    """Expects denormalized input.
    Speckle Suppression Index should be calculated on raw intensity rather
    than dB scale.

    Parameters
    ----------
    img : tensor[b, c, h, w]
        denormalized in dB sar data with c polarization channels.
    per_pixel : bool
        If true then accounts for patch size, otherwise averages by patch.
    """
    n_img, c_img, h_img, w_img = noisy.shape
    # first convert back to raw intensity
    r_noisy = torch.pow(10.0, noisy / 10).view(n_img, c_img, -1)
    r_filt = torch.pow(10.0, filt / 10).view(n_img, c_img, -1)
    std_noisy = r_noisy.std(dim=2)
    mean_noisy = r_noisy.mean(dim=2)
    std_filt = r_filt.std(dim=2)
    mean_filt = r_filt.mean(dim=2)

    channel_ssi = (std_filt / mean_filt) * (mean_noisy / std_noisy)
    num = n_img*c_img*h_img*w_img if per_pixel else n_img*c_img
    return channel_ssi.sum() / num

def get_random_batch(dataset, batch_size=50, seed=None):
    """Get a random batch from the dataset.
    Parameters
    ----------
    dataset : torch.utils.data.Dataset
        The dataset to get the batch from.
    batch_size : int
        The size of the batch to get.
    seed : int
        The seed to use for the random batch.
    """
    if seed is not None:
        random.seed(seed)

    indices = random.sample(range(len(dataset)), batch_size)
    subset = Subset(dataset, indices)
    loader = DataLoader(subset, batch_size=batch_size, shuffle=False)
    return next(iter(loader))

def permute_image(image):
    """Randomly permutes an image while keeping the pixel values the same."""
    permuted = image.flatten()  # Flatten to 1D array
    np.random.shuffle(permuted)  # Shuffle the values
    return permuted.reshape(image.shape)  # Reshape back to original

def convert_to_grayscale(image, levels=64):
    """Fast conversion of an image to grayscale with levels <= 256."""
    imin = image.min()
    imax = image.max()
    if imin == imax:
        return np.zeros(image.shape,dtype=np.uint8)
    else:
        return (np.round(np.clip((image - imin) / (imax - imin), 0, 1) * (levels-1))).astype(np.uint8)

def enl(img, N=5, window_size=8, levels=4):
    """ENL calculated for non-overlapping sliding windows of size x over the image.
    The N windows with the least variance have their ENLs averaged.
    We select homogenous regions via lowest regions as specified in textural
    analysis of https://oa.ee.tsinghua.edu.cn/~yangjian/xubin/pdf/manuscript_ENL.pdf

    Parameters
    ----------
    img : tensor[h, w]
        Denormalized in dB sar data of a chosen polarization channel.
    """
    # first convert to raw intensity, not dB scale
    h_img, w_img = img.shape
    intensity = np.power(10, img / 10)

    # sliding windows over 64 x 64
    axis_windows = h_img // window_size
    enls = np.empty(axis_windows**2)
    entropy = np.empty(axis_windows**2)
    for i in range(axis_windows):
        for j in range(axis_windows):
            window = intensity[i*window_size:(i+1)*window_size, j*window_size:(j+1)*window_size]
            std = window.std()
            mean = window.mean()
            enl_val = np.clip((mean / std)**2, 0, 9999) if std > 0 else 9999
            enls[j + i * axis_windows] = enl_val

            # calculate window entropy
            g = convert_to_grayscale(window, levels=levels)
            glcm = graycomatrix(g, distances=[1], angles=[0, np.pi/2], levels=levels, normed=True, symmetric=True)
            entropy[j + i * axis_windows] = graycoprops(glcm, 'entropy').sum()

    # avg of lowest std (highest homogeneity) enls
    sorted_indices = np.argsort(entropy)[:N]  # Select N smallest entropy windows
    avg_enl = np.mean(enls[sorted_indices])
    return avg_enl

def RIS(noisy, filt, levels=64):
    """Implemented as described in https://arxiv.org/pdf/1811.11872
    They use 4-connected sites, horizontal, and vertical directions.

    Parameters
    ----------
    noisy : tensor
        Noisy SAR image in dB scale
    filt : tensor
        Despeckled SAR image in dB scale
    levels : int
        Quantization level for ratio image (default: 64).
    """
    def ref_h0(glcm):
        """Homogeneity textual descriptor calculated assuming independence
        p(i, j) = p(i) * p(j)."""
        arr = glcm.squeeze(2).sum(axis=2) / 2
        w, h = arr.shape
        # calculate marginals
        p = np.empty(w)
        for i in range(w):
            p[i] = arr[i, :].sum()
        sum = 0
        for i in range(w):
            for j in range(h):
                sum += p[i] * p[j] / ((i - j)**2 + 1)
        return sum

    in_intensity = np.power(10, noisy / 10)
    out_intensity = np.power(10, filt / 10)
    ratio = in_intensity / out_intensity

    # haralick term - https://earlglynn.github.io/RNotes/package/EBImage/Haralick-Textural-Features.html
    img_O = convert_to_grayscale(ratio, levels=levels)
    distances = [1]
    angles = [0, np.pi/2]
    glcm = graycomatrix(img_O, distances=distances, angles=angles, levels=levels, normed=True, symmetric=True)

    # compute a homogeneity value for each direction and calculate the mean
    H = np.mean(graycoprops(glcm, 'homogeneity'))
    H_0 = ref_h0(glcm)
    ris = 100 * (H - H_0) / H_0
    return ris

def quality_m(noisy, filt, N=4, window_size=8, samples=10, levels=8):
    """Metric for filter quality as described in https://arxiv.org/pdf/1704.05952

    The paper uses n=8 automatically detected homogenous areas (500 x 500 sar images).
    r_enl_mu is usually between 2-11 for filters.

    Parameters
    ----------
    noisy : tensor
        Noisy SAR image in dB scale
    filt : tensor
        Despeckled SAR image in dB scale
    N : int
        Lowest N ENL diffs chosen as textureless area
    window_size : int
        Size of windows to calculate ENL and ratio for the r_enl_mu term.
    samples : int
        Number of random permutations of ratio image sampled to estimate
        homogeneity without structure.
    levels : int
        Quantization level for ratio image (default: 64).
    """
    in_intensity = np.power(10, noisy / 10)
    out_intensity = np.power(10, filt / 10)
    ratio = in_intensity / out_intensity
    # sliding windows over 64 x 64
    axis_windows = 64 // window_size
    enl_diffs = np.empty(axis_windows**2)
    enl_rel_diffs = np.empty(axis_windows**2)
    ratio_diffs = np.empty(axis_windows**2)
    for i in range(axis_windows):
        for j in range(axis_windows):
            noisy_window = in_intensity[i*window_size:(i+1)* window_size, j*window_size:(j+1)*window_size]
            ratio_window = ratio[i*window_size:(i+1)* window_size, j*window_size:(j+1)*window_size]

            noisy_var = noisy_window.var()
            noisy_mean = noisy_window.mean()
            enl_noisy = np.clip(noisy_mean**2 / noisy_var, 0, 9999) if noisy_var > 0 else 9999

            ratio_var = ratio_window.var()
            ratio_mean = ratio_window.mean()
            enl_ratio = np.clip(ratio_mean**2 / ratio_var, 0, 9999) if ratio_var > 0 else 9999

            # noisy enl, ratio enl, ratio mean
            tmp = abs(enl_noisy - enl_ratio)
            enl_diffs[j + i * axis_windows] = tmp
            enl_rel_diffs[j + i * axis_windows] = np.clip(tmp / enl_noisy, 0, 9999) if enl_noisy > 0 else 9999
            ratio_diffs[j + i * axis_windows] = abs(1 - ratio_mean)

    # choose lowest N ENL noisy - ratio diffs as textureless areas
    # empirically the ratio mean is usually ~1 so we do not need to worry but ENL error is not guaranteed
    sorted_indices = np.argsort(enl_diffs)[:N]
    r_enl_mu = (enl_rel_diffs[sorted_indices].sum() + ratio_diffs[sorted_indices].sum()) / 2

    # haralick term - https://earlglynn.github.io/RNotes/package/EBImage/Haralick-Textural-Features.html
    img_O = convert_to_grayscale(ratio, levels=levels)
    distances = [1, 2, 3] # focus on small scale textures and structure
    angles = [0, np.pi/4, np.pi/2, 3*np.pi/4]
    glcm = graycomatrix(img_O, distances=distances, angles=angles, levels=levels, normed=True, symmetric=True)
    h_O = np.mean(graycoprops(glcm, 'homogeneity'))
    h_g = np.empty(samples)
    for i in range(samples):
        img_g = permute_image(img_O)
        glcm = graycomatrix(img_g, distances=distances, angles=angles, levels=levels, normed=True, symmetric=True)
        h_g[i] = np.mean(graycoprops(glcm, 'homogeneity'))

    # may need to tweak scaling term 100
    h_term = 100 * abs(h_O - h_g.mean()) / h_O if h_O > 0 else 9999
    return r_enl_mu + h_term

def normalize(patch, vmin, vmax):
    """Normalize a patch to the range [0, 1]. Used for SSIM."""
    return np.clip((patch - vmin) / (vmax - vmin), 0, 1)

def psnr(noisy, ground_truth):
    """PSNR metric.
    
    Parameters
    ----------
    noisy : tensor
        Noisy SAR image in dB scale
    ground_truth : tensor
        Ground truth (multitemporal composite) SAR image in dB scale
    """
    mse = np.mean((noisy - ground_truth) ** 2)
    if mse == 0:
        return float('inf')
    max_pixel_value = 1.0
    psnr = 20 * np.log10(max_pixel_value / np.sqrt(mse))
    return psnr

def compute_ssim(noisy, ground_truth, data_range=1.0):
    """SSIM metric.
    
    Parameters
    ----------
    noisy : tensor
        Noisy SAR image in dB scale
    ground_truth : tensor
        Ground truth (multitemporal composite) SAR image in dB scale
    data_range : float
        Data range of the image
    """
    return ssim(ground_truth, noisy, data_range=data_range)



