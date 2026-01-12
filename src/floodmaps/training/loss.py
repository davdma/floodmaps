import torch
from torch import nn
import torch.nn.functional as F

ALPHA = 0.3
BETA = 0.7
GAMMA = 4/3
AD_LOSS_NAMES = ['MSELoss', 'L1Loss', 'PseudoHuberLoss', 'HuberLoss', 'LogCoshLoss', 'CharbonnierLoss']

class SARLossConfig():
    """Special loss handling object for training SAR flood mapping models. Includes
    logic for handling shift invariant loss calculation, adjusting the window
    inside the true label to align it properly with the SAR patch."""
    def __init__(self, cfg, ad_cfg=None, device='cpu'):
        """Sets up train, validation, test losses."""
        self.cfg = cfg
        self.ad_cfg = ad_cfg
        self.uses_autodespeckler = ad_cfg is not None

        # classifier logit output losses
        # shift losses (diff implementation for train, val, test loops for efficiency)
        self.shift_train_loss_fn, self.shift_val_loss_fn, self.shift_test_loss_fn = self.get_shift_losses(cfg, device)
        # non shift loss (same function used for train, val, test)
        self.non_shift_loss_fn = self.get_non_shift_loss(cfg, device)

        # autodespeckler reconstruction losses
        self.ad_loss_fn = get_ad_loss(ad_cfg).to(device) if self.uses_autodespeckler else None

    def compute_loss(self, out_dict, targets, typ='train', shift_invariant=True):
        """For autodespeckler architecture, will add reconstruction loss
        from output of despeckler to the final loss.
        
        Parameters
        ----------
        out_dict : dict
            Dictionary containing the output of the model.
        targets : torch.Tensor
            True labels
        typ : str
            Type of loss to compute necessary for shift invariant loss.
            Must be one of 'train', 'val', 'test'.
        shift_invariant : bool
            Whether to compute shift invariant loss.
        
        Returns
        -------
        loss_dict : dict
            Dictionary containing the computed loss.
        """
        # autodespeckler loss component - calculate reconstruction loss with respect to sar input
        loss_dict = dict()
        if self.uses_autodespeckler:
            recons_loss = self.ad_loss_fn(out_dict['despeckler_output'], out_dict['despeckler_input'])
            loss_dict['recons_loss'] = recons_loss

            if self.ad_cfg.model.autodespeckler == 'VAE':
                # beta hyperparameter
                log_var = torch.clamp(out_dict['log_var'], min=-6, max=6)
                mu = out_dict['mu']
                kld_loss = torch.mean(-0.5 * torch.sum(1 + log_var - mu ** 2 - log_var.exp(), dim = 1), dim = 0)
                loss_dict['kld_loss'] = kld_loss

                recons_loss = recons_loss + self.ad_cfg.model.vae.VAE_beta * kld_loss

                if torch.isnan(recons_loss).any() or torch.isinf(recons_loss).any():
                    print(f'min mu: {mu.min().item()}')
                    print(f'max mu: {mu.max().item()}')
                    print(f'min log_var: {log_var.min().item()}')
                    print(f'max log_var: {log_var.max().item()}')
                    raise Exception('recons_loss + kld_loss is nan or inf')

        # classifier loss component + true label (shifted or not) + shift indices
        if shift_invariant:
            if typ == 'train':
                main_loss, y_true, shift_indices = self.shift_train_loss_fn(out_dict['classifier_output'], targets)
            elif typ == 'val':
                main_loss, y_true, shift_indices = self.shift_val_loss_fn(out_dict['classifier_output'], targets)
            elif typ == 'test':
                main_loss, y_true, shift_indices = self.shift_test_loss_fn(out_dict['classifier_output'], targets)
            else:
                raise Exception('Invalid argument: typ not equal to one of train, val, test.')
        else:
            main_loss, y_true, shift_indices = self.non_shift_loss_fn(out_dict['classifier_output'], targets)

        total_loss = (
            self.cfg.train.balance_coeff * recons_loss + main_loss
            if self.uses_autodespeckler else main_loss
        )
        loss_dict['total_loss'] = total_loss
        loss_dict['classifier_loss'] = main_loss
        loss_dict['true_label'] = y_true
        loss_dict['shift_indices'] = shift_indices
        return loss_dict

    def get_label_alignment(self, inputs, targets, shift_invariant=True):
        """Get the window of the target label that aligns best with the
        prediction, along with the shift indices used.

        Parameters
        ----------
        inputs : torch.Tensor
            Predicted logits
        targets : torch.Tensor
            True labels
        shift_invariant : bool
            Whether to align using shift invariant loss. If false returns
            centered window with the same centered indices.
        
        Returns
        -------
        y_shifted : torch.Tensor
            Aligned label windows
        shift_indices : tuple of (torch.Tensor, torch.Tensor)
            (row_shifts, col_shifts) for each patch in batch
        """
        if shift_invariant:
            _, y_shifted, shift_indices = self.shift_val_loss_fn(inputs, targets)
        else:
            _, y_shifted, shift_indices = self.non_shift_loss_fn(inputs, targets)
        return y_shifted, shift_indices

    def get_shift_losses(self, cfg, device):
        """Chooses the type of loss used for training, validation, testing loop,
        and also based on whether losses used are shift invariant and
        best for memory performance."""
        loss_name = cfg.train.loss
        assert loss_name in ['BCELoss', 'BCEDiceLoss', 'TverskyLoss', 'FocalTverskyLoss'], 'Loss function not supported.'
        
        # Get pos_weight as float - loss classes handle float-to-tensor conversion internally
        pos_weight = getattr(cfg.train, 'pos_weight', None)
        
        if loss_name == 'BCELoss':
            train_loss_fn = TrainShiftInvariantLoss(InvariantBCELoss(pos_weight=pos_weight), device=device)
            val_loss_fn = test_loss_fn = ShiftInvariantLoss(InvariantBCELoss(pos_weight=pos_weight), device=device)
        elif loss_name == 'BCEDiceLoss':
            train_loss_fn = TrainShiftInvariantLoss(InvariantBCEDiceLoss(pos_weight=pos_weight), device=device)
            val_loss_fn = test_loss_fn = ShiftInvariantLoss(InvariantBCEDiceLoss(pos_weight=pos_weight), device=device)
        elif loss_name == 'TverskyLoss':
            train_loss_fn = TrainShiftInvariantLoss(InvariantTverskyLoss(alpha=cfg.train.tversky.alpha,
                                                                        beta=1-cfg.train.tversky.alpha),
                                                    device=device)
            val_loss_fn = test_loss_fn = ShiftInvariantLoss(InvariantTverskyLoss(alpha=cfg.train.tversky.alpha,
                                                    beta=1-cfg.train.tversky.alpha),
                                                    device=device)
        elif loss_name == 'FocalTverskyLoss':
            train_loss_fn = TrainShiftInvariantLoss(InvariantFocalTverskyLoss(alpha=cfg.train.tversky.alpha,
                                                        beta=1-cfg.train.tversky.alpha,
                                                        gamma=cfg.train.focal_tversky.gamma),
                                                    device=device)
            val_loss_fn = test_loss_fn = ShiftInvariantLoss(InvariantFocalTverskyLoss(alpha=cfg.train.tversky.alpha,
                                                        beta=1-cfg.train.tversky.alpha,
                                                        gamma=cfg.train.focal_tversky.gamma),
                                                    device=device)
        else:
            raise Exception('Loss function not found.')

        return train_loss_fn, val_loss_fn, test_loss_fn
    
    def get_non_shift_loss(self, cfg, device):
        """Returns the loss function used for non shift invariant loss, which
        is the same for train, val, test loops."""
        loss_name = cfg.train.loss
        assert loss_name in ['BCELoss', 'BCEDiceLoss', 'TverskyLoss', 'FocalTverskyLoss'], 'Loss function not supported.'
        
        # Get pos_weight as float - our custom loss classes handle conversion internally
        pos_weight = getattr(cfg.train, 'pos_weight', None)
        
        if loss_name == 'BCELoss':
            # nn.BCEWithLogitsLoss requires pos_weight to be a tensor, so convert here
            pos_weight_tensor = torch.tensor(float(pos_weight), device=device) if pos_weight is not None else None
            return NonShiftInvariantLoss(nn.BCEWithLogitsLoss(pos_weight=pos_weight_tensor),
                                        size=cfg.data.size,
                                        window=cfg.data.window,
                                        device=device)
        elif loss_name == 'BCEDiceLoss':
            return NonShiftInvariantLoss(BCEDiceLoss(pos_weight=pos_weight),
                                        size=cfg.data.size,
                                        window=cfg.data.window,
                                        device=device)
        elif loss_name == 'TverskyLoss':
            return NonShiftInvariantLoss(TverskyLoss(alpha=cfg.train.tversky.alpha,
                                            beta=1-cfg.train.tversky.alpha),
                                        size=cfg.data.size,
                                        window=cfg.data.window,
                                        device=device)
        elif loss_name == 'FocalTverskyLoss':
            return NonShiftInvariantLoss(FocalTverskyLoss(alpha=cfg.train.tversky.alpha,
                                            beta=1-cfg.train.tversky.alpha,
                                            gamma=cfg.train.focal_tversky.gamma),
                                        size=cfg.data.size,
                                        window=cfg.data.window,
                                        device=device)
        else:
            raise Exception('Loss function not found.')

    def contains_reconstruction_loss(self):
        return self.uses_autodespeckler

def get_ad_loss(cfg):
    """Note: important to consider scale of losses if summing different components.
    For our purposes and stability, better to make all losses calculated per patch instead of per pixel."""
    # SPECKLE OPTIMIZED LOSS? Note: compare between models on same scale
    # choose universal scale/metric - use dB scale and evaluate using RMSE, MSE, MAE, R^2 (when benchmarking)
    if cfg.train.loss == 'MSELoss':
        # scales mseloss to per sample (64 x 64 patch)
        return PatchMSELoss()
    elif cfg.train.loss == 'L1Loss':
        return PatchL1Loss()
    elif cfg.train.loss == 'PseudoHuberLoss':
         # pseudo huber - https://arxiv.org/pdf/2310.14189
         # use default c=0.3 instead of 0.03 due to mean 0 std 1 scale
        phuber_c = getattr(cfg.train, 'pseudo_huber_c', 0.3)
        return PseudoHuberLoss(c=phuber_c)
    elif cfg.train.loss == 'HuberLoss':
        return PatchHuberLoss()
    elif cfg.train.loss == 'LogCoshLoss':
        return LogCoshLoss()
    elif cfg.train.loss == 'CharbonnierLoss':
        return CharbonnierLoss(eps=1e-6)
    else:
        raise Exception(f"Loss must be one of: {', '.join(AD_LOSS_NAMES)}")

class PatchMSELoss(nn.Module):
    def __init__(self):
        super().__init__()

    def forward(self, inputs, targets):
        # return F.mse_loss(inputs, targets, reduction='sum') / inputs.size(0)
        # Compute element-wise MSE loss
        loss = (inputs - targets) ** 2
        # Sum over each patch, then average over the batch
        patch_loss = loss.view(loss.size(0), -1).sum(dim=1)  # Sum over patch elements
        return patch_loss.mean()  # Average over the batch

class PatchL1Loss(nn.Module):
    def __init__(self):
        super().__init__()

    def forward(self, inputs, targets):
        # return F.l1_loss(inputs, targets, reduction='sum') / inputs.size(0)
        # Compute element-wise L1 loss
        loss = torch.abs(inputs - targets)
        # Sum over each patch, then average over the batch
        patch_loss = loss.view(loss.size(0), -1).sum(dim=1)  # Sum over patch elements
        return patch_loss.mean()  # Average over the batch

class PatchHuberLoss(nn.Module):
    def __init__(self):
        super().__init__()

    def forward(self, inputs, targets):
        # return F.huber_loss(inputs, targets, reduction='sum') / inputs.size(0)
        diff = inputs - targets
        abs_diff = torch.abs(diff)
        loss = torch.where(abs_diff < 1, 0.5 * diff**2, abs_diff - 0.5)
        # Sum over each patch, then average over the batch
        patch_loss = loss.view(loss.size(0), -1).sum(dim=1)  # Sum over patch elements
        return patch_loss.mean()  # Average over the batch

# loss for AD
class PseudoHuberLoss(nn.Module):
    """Defined in the paper: https://arxiv.org/pdf/2310.14189."""
    def __init__(self, c=0.03):
        super().__init__()
        self.register_buffer('c', torch.tensor(c))

    def forward(self, inputs, targets):
        loss = torch.sqrt((inputs - targets) ** 2 + self.c ** 2) - self.c

        # Per patch loss
        patch_loss = loss.view(loss.size(0), -1).sum(dim=1)
        return patch_loss.mean()

# loss for AD
class LogCoshLoss(nn.Module):
    """Hypothetically can improve VAE performance: https://openreview.net/forum?id=rkglvsC9Ym."""
    def __init__(self):
        super().__init__()

    def forward(self, inputs, targets):
        loss = torch.log(torch.cosh(inputs - targets + 1e-12))  # Compute element-wise log-cosh loss

        # Sum over all pixels per patch
        patch_loss = loss.view(loss.size(0), -1).sum(dim=1)
        return patch_loss.mean()

# loss for AD - differentiable approximation of L1 loss
class CharbonnierLoss(nn.Module):
    """Charbonnier loss (differentiable variant of L1).
    
    Defined as: sqrt((x - y)^2 + eps^2)
    
    More robust to outliers than MSE while being differentiable everywhere
    unlike L1 loss. Commonly used in image restoration tasks.
    
    Parameters
    ----------
    eps : float
        Small constant for numerical stability. Default is 1e-6.
    """
    def __init__(self, eps=1e-6):
        super().__init__()
        self.register_buffer('eps', torch.tensor(eps))

    def forward(self, inputs, targets):
        diff = inputs - targets
        loss = torch.sqrt(diff ** 2 + self.eps ** 2)

        # Per patch loss
        patch_loss = loss.view(loss.size(0), -1).sum(dim=1)
        return patch_loss.mean()


# LSGAN losses for CVAE-GAN training
# Uses sum-over-spatial reduction to match ELBO loss scaling convention
class LSGANDiscriminatorLoss(nn.Module):
    """LSGAN discriminator loss with sum-over-spatial reduction.
    
    Computes:
        D_loss = 0.5 * (MSE(D(real), 1) + MSE(D(fake), 0))
    
    NOTE: Uses sum over spatial dimensions, mean over batch to match the ELBO loss
    scaling convention (sum-over-pixels for reconstruction, sum-over-latent for KL).
    
    LSGAN uses MSE instead of BCE for more stable training and less mode collapse.
    
    Parameters
    ----------
    real_label : float
        Target label for real samples (default 1.0)
    fake_label : float
        Target label for fake samples (default 0.0)
    """
    def __init__(self, real_label=1.0, fake_label=0.0):
        super().__init__()
        self.register_buffer('real_label', torch.tensor(real_label))
        self.register_buffer('fake_label', torch.tensor(fake_label))
    
    def forward(self, d_real, d_fake):
        """Compute discriminator loss.
        
        Parameters
        ----------
        d_real : torch.Tensor
            Discriminator output for real samples [B, 1, H, W]
        d_fake : torch.Tensor
            Discriminator output for fake samples [B, 1, H, W]
        
        Returns
        -------
        dict
            Dictionary with 'd_loss', 'd_loss_real', 'd_loss_fake'
        """
        # Element-wise MSE, sum over spatial, mean over batch
        target_real = torch.full_like(d_real, self.real_label)
        loss_real = (d_real - target_real) ** 2
        d_loss_real = loss_real.view(loss_real.size(0), -1).sum(dim=1).mean()
        
        target_fake = torch.full_like(d_fake, self.fake_label)
        loss_fake = (d_fake - target_fake) ** 2
        d_loss_fake = loss_fake.view(loss_fake.size(0), -1).sum(dim=1).mean()
        
        d_loss = 0.5 * (d_loss_real + d_loss_fake)
        
        return {
            'd_loss': d_loss,
            'd_loss_real': d_loss_real,
            'd_loss_fake': d_loss_fake
        }


class LSGANGeneratorLoss(nn.Module):
    """LSGAN generator adversarial loss with sum-over-spatial reduction.
    
    Computes:
        G_adv_loss = MSE(D(fake), 1)
    
    NOTE: Uses sum over spatial dimensions, mean over batch to match the ELBO loss
    scaling convention (sum-over-pixels for reconstruction, sum-over-latent for KL).
    
    Generator tries to fool discriminator by making fake samples look real.
    
    Parameters
    ----------
    real_label : float
        Target label for generator (wants D to output 1 for fakes)
    """
    def __init__(self, real_label=1.0):
        super().__init__()
        self.register_buffer('real_label', torch.tensor(real_label))
    
    def forward(self, d_fake):
        """Compute generator adversarial loss.
        
        Parameters
        ----------
        d_fake : torch.Tensor
            Discriminator output for fake samples [B, 1, H, W]
        
        Returns
        -------
        torch.Tensor
            Generator adversarial loss (scalar)
        """
        # Element-wise MSE, sum over spatial, mean over batch
        target = torch.full_like(d_fake, self.real_label)
        loss = (d_fake - target) ** 2
        per_sample = loss.view(loss.size(0), -1).sum(dim=1)
        return per_sample.mean()


class BCEDiceLoss(nn.Module):
    def __init__(self, pos_weight=None):
        """pos_weight can be a float or tensor - will be converted to tensor and registered as buffer."""
        super().__init__()
        # optional positive class weighting for the BCE component only
        if pos_weight is None:
            self.pos_weight = None
        else:
            self.register_buffer('pos_weight', torch.as_tensor(pos_weight))

    def forward(self, inputs, targets, smooth=1):
        # BCE Loss (use logits)
        logits = inputs.reshape(-1)

        # remove if your model contains a sigmoid or equivalent activation layer
        probs = F.sigmoid(inputs)

        # flatten label and prediction tensors
        probs = probs.reshape(-1)
        targets = targets.reshape(-1)

        intersection = (probs * targets).sum()
        dice_loss = 1 - (2. * intersection + smooth) / (probs.sum() + targets.sum() + smooth)
        BCE = F.binary_cross_entropy_with_logits(logits, targets, reduction='mean', pos_weight=self.pos_weight)
        BCEDice = BCE + dice_loss

        return BCEDice

class TverskyLoss(nn.Module):
    def __init__(self, alpha=ALPHA, beta=BETA):
        super().__init__()
        self.register_buffer('alpha', torch.as_tensor(alpha))
        self.register_buffer('beta', torch.as_tensor(beta))

    def forward(self, inputs, targets, smooth=1):

        # comment out if your model contains a sigmoid or equivalent activation layer
        inputs = F.sigmoid(inputs)

        # flatten label and prediction tensors
        inputs = inputs.reshape(-1)
        targets = targets.reshape(-1)

        #True Positives, False Positives & False Negatives
        TP = (inputs * targets).sum()
        FP = ((1-targets) * inputs).sum()
        FN = (targets * (1-inputs)).sum()

        Tversky = (TP + smooth) / (TP + self.alpha * FP + self.beta * FN + smooth)

        return 1 - Tversky

class FocalTverskyLoss(nn.Module):
    """Implements FTL as defined in https://arxiv.org/pdf/1810.07842"""
    def __init__(self, alpha=ALPHA, beta=BETA, gamma=GAMMA):
        super().__init__()
        assert 1 <= gamma <= 3, "Gamma must be in [1, 3]"
        self.register_buffer('alpha', torch.as_tensor(alpha))
        self.register_buffer('beta', torch.as_tensor(beta))
        self.register_buffer('gamma', torch.as_tensor(gamma))

    def forward(self, inputs, targets, smooth=1):

        # comment out if your model contains a sigmoid or equivalent activation layer
        inputs = F.sigmoid(inputs)

        # flatten label and prediction tensors
        inputs = inputs.reshape(-1)
        targets = targets.reshape(-1)

        #True Positives, False Positives & False Negatives
        TP = (inputs * targets).sum()
        FP = ((1-targets) * inputs).sum()
        FN = (targets * (1-inputs)).sum()

        Tversky = (TP + smooth) / (TP + self.alpha * FP + self.beta * FN + smooth)

        return (1 - Tversky) ** self.gamma

class InvariantBCELoss(nn.Module):
    """BCE loss with per-sample (patch) reduction for use with ShiftInvariantLoss.
    
    Unlike standard BCEWithLogitsLoss which reduces across the entire batch (returning a scalar),
    this loss reduces across pixels *within each sample*, returning a tensor of shape (batch_size,)
    with one loss value per sample. This per-sample output is required by ShiftInvariantLoss to
    find the optimal alignment shift independently for each sample in the batch.

    Parameters
    ----------
    weight : float, optional
        Manual rescaling weight for the loss.
    reduction : str, optional
        Reduction mode: 'mean' averages pixel losses per sample, 'sum' sums them,
        'none' returns unreduced per-pixel losses. Default is 'mean'.
    pos_weight : float, optional
        Weight for positive class to handle class imbalance.

    Returns
    -------
    torch.Tensor
        Loss tensor of shape (batch_size,) when reduction is 'mean' or 'sum',
        or (batch_size, num_pixels) when reduction is 'none'.
    """
    def __init__(self, weight=None, reduction='mean', pos_weight=None):
        super().__init__()
        self.reduction = reduction
        
        # Use register_buffer for tensor parameters (ensures device consistency)
        if weight is None:
            self.weight = None
        else:
            self.register_buffer('weight', torch.as_tensor(weight))
            
        if pos_weight is None:
            self.pos_weight = None
        else:
            self.register_buffer('pos_weight', torch.as_tensor(pos_weight))

    def forward(self, inputs, targets):
        # remove if your model contains a sigmoid or equivalent activation layer
        inputs = inputs.reshape(inputs.shape[0], -1)
        targets = targets.reshape(targets.shape[0], -1)

        # LOGSIGMOID NECESSARY FOR NUMERICAL STABILITY
        log_inputs = F.logsigmoid(inputs)
        log_minus_one_inputs = F.logsigmoid(-inputs)

        pos_term = targets * log_inputs
        neg_term = (1 - targets) * log_minus_one_inputs

        if self.pos_weight is not None:
            pos_term = self.pos_weight * pos_term

        loss = -(pos_term + neg_term)

        if self.weight is not None:
            loss = self.weight * loss

        # Apply the specified reduction method
        if self.reduction == 'mean':
            return loss.mean(axis=1)
        elif self.reduction == 'sum':
            return loss.sum(axis=1)
        else:
            return loss

class InvariantBCEDiceLoss(nn.Module):
    """Combined BCE + Dice loss with per-sample (patch) reduction for use with ShiftInvariantLoss.
    
    Unlike standard loss functions which reduce across the entire batch (returning a scalar),
    this loss reduces across pixels *within each sample*, returning a tensor of shape (batch_size,)
    with one loss value per sample. This per-sample output is required by ShiftInvariantLoss to
    find the optimal alignment shift independently for each sample in the batch.

    The loss combines BCE loss (via InvariantBCELoss) with Dice loss, both computed per-sample.

    Parameters
    ----------
    weight : float, optional
        Manual rescaling weight for the BCE component.
    pos_weight : float, optional
        Weight for positive class in BCE component to handle class imbalance.

    Returns
    -------
    torch.Tensor
        Loss tensor of shape (batch_size,) containing BCE + Dice loss for each sample.
    """
    def __init__(self, weight=None, pos_weight=None):
        super().__init__()
        self.BCE = InvariantBCELoss(weight=weight, reduction='mean', pos_weight=pos_weight)

    def forward(self, inputs, targets, smooth=1):
        # model should not contain a sigmoid or equivalent activation layer
        BCE = self.BCE(inputs, targets)
        inputs = F.sigmoid(inputs)

        # flatten label and prediction tensors
        inputs = inputs.reshape(inputs.shape[0], -1)
        targets = targets.reshape(targets.shape[0], -1)

        intersection = (inputs * targets).sum(axis=1)
        dice_loss = 1 - (2. * intersection + smooth) / (inputs.sum(axis=1) + targets.sum(axis=1) + smooth)
        BCEDice = BCE + dice_loss

        return BCEDice

class InvariantTverskyLoss(nn.Module):
    """Tversky loss with per-sample (patch) reduction for use with ShiftInvariantLoss.
    
    Unlike standard loss functions which reduce across the entire batch (returning a scalar),
    this loss reduces across pixels *within each sample*, returning a tensor of shape (batch_size,)
    with one loss value per sample. This per-sample output is required by ShiftInvariantLoss to
    find the optimal alignment shift independently for each sample in the batch.

    Tversky loss generalizes Dice loss by allowing asymmetric weighting of false positives
    and false negatives via alpha and beta parameters.

    Parameters
    ----------
    alpha : float, optional
        Weight for false positives. Default is ALPHA constant.
    beta : float, optional
        Weight for false negatives. Default is BETA constant.

    Returns
    -------
    torch.Tensor
        Loss tensor of shape (batch_size,) containing Tversky loss for each sample.
    """
    def __init__(self, alpha=ALPHA, beta=BETA):
        super().__init__()
        self.register_buffer('alpha', torch.tensor(alpha))
        self.register_buffer('beta', torch.tensor(beta))

    def forward(self, inputs, targets, smooth=1):
        # comment out if your model contains a sigmoid or equivalent activation layer
        inputs = F.sigmoid(inputs)

        # flatten label and prediction tensors but sample wise
        inputs = inputs.reshape(inputs.shape[0], -1)
        targets = targets.reshape(targets.shape[0], -1)

        #True Positives, False Positives & False Negatives
        TP = (inputs * targets).sum(axis=1) # sum across samples
        FP = ((1-targets) * inputs).sum(axis=1)
        FN = (targets * (1-inputs)).sum(axis=1)

        # want this to be list of size samples
        Tversky = (TP + smooth) / (TP + self.alpha * FP + self.beta * FN + smooth)

        return 1 - Tversky

class InvariantFocalTverskyLoss(nn.Module):
    """Focal Tversky loss with per-sample (patch) reduction for use with ShiftInvariantLoss.
    
    Unlike standard loss functions which reduce across the entire batch (returning a scalar),
    this loss reduces across pixels *within each sample*, returning a tensor of shape (batch_size,)
    with one loss value per sample. This per-sample output is required by ShiftInvariantLoss to
    find the optimal alignment shift independently for each sample in the batch.

    Parameters
    ----------
    alpha : float, optional
        Weight for false positives. Default is ALPHA constant.
    beta : float, optional
        Weight for false negatives. Default is BETA constant.
    gamma : float, optional
        Focusing parameter in [1, 3]. Default is GAMMA constant.

    Returns
    -------
    torch.Tensor
        Loss tensor of shape (batch_size,) containing Focal Tversky loss for each sample.
    """
    def __init__(self, alpha=ALPHA, beta=BETA, gamma=GAMMA):
        super().__init__()
        assert 1 <= gamma <= 3, "Gamma must be in [1, 3]"
        self.register_buffer('alpha', torch.tensor(alpha))
        self.register_buffer('beta', torch.tensor(beta))
        self.register_buffer('gamma', torch.tensor(gamma))

    def forward(self, inputs, targets, smooth=1):
        # comment out if your model contains a sigmoid or equivalent activation layer
        inputs = F.sigmoid(inputs)

        # flatten label and prediction tensors but sample wise
        inputs = inputs.reshape(inputs.shape[0], -1)
        targets = targets.reshape(targets.shape[0], -1)

        #True Positives, False Positives & False Negatives
        TP = (inputs * targets).sum(axis=1) # sum across samples
        FP = ((1-targets) * inputs).sum(axis=1)
        FN = (targets * (1-inputs)).sum(axis=1)

        # want this to be list of size samples
        Tversky = (TP + smooth) / (TP + self.alpha * FP + self.beta * FN + smooth)

        return (1 - Tversky) ** self.gamma

class ShiftInvariantLoss(nn.Module):
    """Implementation of a shift invariant loss function. Given a batch size B of input patches of dimension N x N
    and corresponding target patches of dimension M x M, where M > N, the loss function calculates the minimum
    loss from aligning each input patch with all possible N x N windows inside of its M x M target patch.
    The total loss is the sum of all B minimum losses.

    For each patch, the corresponding N x N window in the target that produces the minimum loss is also returned,
    along with the shift indices used to align each patch.

    To use, must specify compatible loss function (Invariant BCE Loss, Invariant BCE Dice Loss, Invariant Tversky Loss)
    at initialization.

    Parameters
    ----------
    loss : obj
        Instance of InvariantBCELoss, InvariantBCEDiceLoss, InvariantTverskyLoss, InvariantFocalTverskyLoss
    device : str
    
    Returns
    -------
    total_loss : torch.Tensor
        Sum of minimum losses across all patches
    adjusted_labels : torch.Tensor
        Aligned label windows for each patch in batch
    shift_indices : tuple of (torch.Tensor, torch.Tensor)
        (row_shifts, col_shifts) - indices for each patch in batch
    """
    def __init__(self, loss, device='cpu'):
        super().__init__()
        self.loss = loss.to(device)
        self.device = device

    def forward(self, inputs, targets):
        """Input must be smaller window than the target."""
        # for each shift we want to calculate all errors across patches for that shift
        # then we use min and argmin to calculate actual loss
        shift1 = targets.shape[-2] - inputs.shape[-2] + 1
        shift2 = targets.shape[-1] - inputs.shape[-1] + 1
        B = inputs.shape[0]

        # Compute center flat index for tie-breaking
        center_i = shift1 // 2
        center_j = shift2 // 2
        center_flat = center_i * shift2 + center_j

        # this is so we can take min across axis of patches
        patch_shift_err = torch.empty((B, shift1 * shift2), dtype=torch.float32, device=self.device)
        candidate_shifts = torch.empty((shift1 * shift2, *inputs.shape), dtype=torch.float32, device=self.device)

        # first calculate all potential shift arrays
        for i in range(shift1):
            for j in range(shift2):
                window = targets[:, :, i:i+inputs.shape[-2], j:j+inputs.shape[-1]]
                candidate_shifts[i * shift2 + j] = window
                patch_shift_err[:, i * shift2 + j] = self.loss(inputs, window)

        min_loss, min_ij = torch.min(patch_shift_err, dim=1)
        
        # Prefer center index when it is tied for minimum loss
        center_loss = patch_shift_err[:, center_flat]
        center_is_min = center_loss == min_loss
        min_ij = torch.where(center_is_min, 
                             torch.tensor(center_flat, device=self.device), 
                             min_ij)

        total_loss = min_loss.sum()
        adjusted_labels = candidate_shifts[min_ij, torch.arange(B, device=self.device)]
        
        # Convert flat indices to 2D shift coordinates
        row_shifts = min_ij // shift2
        col_shifts = min_ij % shift2
        
        return total_loss, adjusted_labels, (row_shifts, col_shifts)

    def change_device(self, device):
        self.loss = self.loss.to(device)
        self.device = device

class TrainShiftInvariantLoss(nn.Module):
    """Implementation of a shift invariant loss function optimized for training loop by only calculating
    gradients when necessary.

    Parameters
    ----------
    loss : obj
        Instance of InvariantBCELoss, InvariantBCEDiceLoss, InvariantTverskyLoss, InvariantFocalTverskyLoss
    device : str
    
    Returns
    -------
    total_loss : torch.Tensor
        Sum of minimum losses across all patches
    adjusted_labels : torch.Tensor
        Aligned label windows for each patch in batch
    shift_indices : tuple of (torch.Tensor, torch.Tensor)
        (row_shifts, col_shifts) - indices for each patch in batch
    """
    def __init__(self, loss, device='cpu'):
        super().__init__()
        self.loss = loss.to(device)
        self.device = device

    def forward(self, inputs, targets):
        """Input must be smaller window than the target."""
        # for each shift we want to calculate all errors across patches for that shift
        # then we use min and argmin to calculate actual loss
        shift1 = targets.shape[-2] - inputs.shape[-2] + 1
        shift2 = targets.shape[-1] - inputs.shape[-1] + 1
        B = inputs.shape[0]

        # Compute center flat index for tie-breaking
        center_i = shift1 // 2
        center_j = shift2 // 2
        center_flat = center_i * shift2 + center_j

        # this is so we can take min across axis of patches
        with torch.no_grad():
            patch_shift_err = torch.empty((B, shift1 * shift2), dtype=torch.float32, device=self.device)
            candidate_shifts = torch.empty((shift1 * shift2, *inputs.shape), dtype=torch.float32, device=self.device)

            # first calculate all potential shift arrays
            for i in range(shift1):
                for j in range(shift2):
                    window = targets[:, :, i:i+inputs.shape[-2], j:j+inputs.shape[-1]]
                    candidate_shifts[i * shift2 + j] = window
                    patch_shift_err[:, i * shift2 + j] = self.loss(inputs, window)

            min_loss, min_ij = torch.min(patch_shift_err, dim=1)
            
            # Prefer center index when it is tied for minimum loss
            center_loss = patch_shift_err[:, center_flat]
            center_is_min = center_loss == min_loss
            min_ij = torch.where(center_is_min, 
                                 torch.tensor(center_flat, device=self.device), 
                                 min_ij)

        adjusted_labels = candidate_shifts[min_ij, torch.arange(B, device=self.device)]
        total_loss = self.loss(inputs, adjusted_labels).sum()
        
        # Convert flat indices to 2D shift coordinates
        row_shifts = min_ij // shift2
        col_shifts = min_ij % shift2
        
        return total_loss, adjusted_labels, (row_shifts, col_shifts)

    def change_device(self, device):
        self.loss = self.loss.to(device)
        self.device = device

class NonShiftInvariantLoss(nn.Module):
    """Wrapper for regular non shifting loss. Used when target label is larger than predicted label.

    Parameters
    ----------
    loss : obj
        Instance of BCELoss, BCEDiceLoss, TverskyLoss, FocalTverskyLoss
    device : str
    
    Returns
    -------
    loss : torch.Tensor
        Computed loss value
    unadjusted_labels : torch.Tensor
        Center-cropped label windows
    shift_indices : tuple of (torch.Tensor, torch.Tensor)
        (row_shifts, col_shifts) - fixed center indices for all batch items
    """
    def __init__(self, loss, size=68, window=64, device='cpu'):
        super().__init__()
        self.loss = loss.to(device)
        self.device = device
        center_1 = (size - window) // 2
        self.c = (center_1, center_1 + window)
        self.center_offset = center_1  # Store for index return

    def forward(self, inputs, targets):
        """Calculates loss using the central window alignment."""
        # crop central window
        unadjusted_labels = targets[:, :, self.c[0]:self.c[1], self.c[0]:self.c[1]]
        loss = self.loss(inputs, unadjusted_labels)
        
        # Return fixed center indices for all batch items
        B = inputs.shape[0]
        row_shifts = torch.full((B,), self.center_offset, dtype=torch.long, device=self.device)
        col_shifts = torch.full((B,), self.center_offset, dtype=torch.long, device=self.device)
        
        return loss, unadjusted_labels, (row_shifts, col_shifts)

    def change_device(self, device):
        self.loss = self.loss.to(device)
        self.device = device
