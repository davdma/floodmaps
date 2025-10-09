import torch
from torch import nn
import torch.nn.functional as F

ALPHA = 0.3
BETA = 0.7

class LossConfig():
    """Special loss handling object for training SAR flood mapping models. Includes
    logic for handling shift invariant loss calculation, adjusting the window
    inside the true label to align it properly with the SAR patch."""
    def __init__(self, cfg, ad_cfg=None, device='cpu'):
        """Sets up train, validation, test losses."""
        self.cfg = cfg
        self.ad_cfg = ad_cfg
        self.uses_autodespeckler = ad_cfg is not None

        # classifier logit output losses
        self.train_loss_fn, self.val_loss_fn, self.test_loss_fn = self.get_losses(cfg, device)

        # autodespeckler reconstruction losses
        self.ad_loss_fn = get_ad_loss(ad_cfg).to(device) if self.uses_autodespeckler else None

    def compute_loss(self, out_dict, targets, typ='train'):
        """For autodespeckler architecture, will add reconstruction loss
        from output of despeckler to the final loss."""
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

        # classifier loss component + true label (shifted or not)
        if typ == 'train':
            main_loss, y_true = self.train_loss_fn(out_dict['classifier_output'], targets)
        elif typ == 'val':
            main_loss, y_true = self.val_loss_fn(out_dict['classifier_output'], targets)
        elif typ == 'test':
            main_loss, y_true = self.test_loss_fn(out_dict['classifier_output'], targets)
        else:
            raise Exception('Invalid argument: typ not equal to one of train, val, test.')

        total_loss = (
            self.cfg.train.balance_coeff * recons_loss + main_loss
            if self.uses_autodespeckler else main_loss
        )
        loss_dict['total_loss'] = total_loss
        loss_dict['classifier_loss'] = main_loss
        loss_dict['true_label'] = y_true
        return loss_dict

    def get_label_alignment(self, inputs, targets):
        """Get the window of the target label that aligns best with the
        prediction."""
        _, y_shifted = self.val_loss_fn(inputs, targets)
        return y_shifted

    def get_losses(self, cfg, device):
        """Chooses the type of loss used for training, validation, testing loop,
        and also based on whether losses used are shift invariant and
        best for memory performance."""
        if cfg.train.loss == 'BCELoss':
            if cfg.train.shift_invariant:
                train_loss_fn = TrainShiftInvariantLoss(InvariantBCELoss(), device=device)
                val_loss_fn = ShiftInvariantLoss(InvariantBCELoss(), device=device)
                test_loss_fn = val_loss_fn
            else:
                # non shift wrapper
                train_loss_fn = NonShiftInvariantLoss(nn.BCEWithLogitsLoss(),
                                                    size=cfg.data.size,
                                                    window=cfg.data.window,
                                                    device=device)
                val_loss_fn = train_loss_fn
                test_loss_fn = train_loss_fn
        elif cfg.train.loss == 'BCEDiceLoss':
            if cfg.train.shift_invariant:
                train_loss_fn = TrainShiftInvariantLoss(InvariantBCEDiceLoss(), device=device)
                val_loss_fn = ShiftInvariantLoss(InvariantBCEDiceLoss(), device=device)
                test_loss_fn = val_loss_fn
            else:
                train_loss_fn = NonShiftInvariantLoss(BCEDiceLoss(),
                                                    size=cfg.data.size,
                                                    window=cfg.data.window,
                                                    device=device)
                val_loss_fn = train_loss_fn
                test_loss_fn = train_loss_fn
        elif cfg.train.loss == 'TverskyLoss':
            if cfg.train.shift_invariant:
                train_loss_fn = TrainShiftInvariantLoss(
                    InvariantTverskyLoss(alpha=cfg.train.tversky.alpha,
                                         beta=1-cfg.train.tversky.alpha),
                                         device=device)
                val_loss_fn = ShiftInvariantLoss(
                    InvariantTverskyLoss(alpha=cfg.train.tversky.alpha,
                                         beta=1-cfg.train.tversky.alpha),
                                         device=device)
                test_loss_fn = val_loss_fn
            else:
                train_loss_fn = NonShiftInvariantLoss(
                    TverskyLoss(alpha=cfg.train.tversky.alpha,
                                beta=1-cfg.train.tversky.alpha),
                                size=cfg.data.size,
                                window=cfg.data.window,
                                device=device)
                val_loss_fn = train_loss_fn
                test_loss_fn = train_loss_fn
        else:
            raise Exception('Loss function not found.')

        return train_loss_fn, val_loss_fn, test_loss_fn

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
        return PseudoHuberLoss(c=0.03)
    elif cfg.train.loss == 'HuberLoss':
        return PatchHuberLoss()
    elif cfg.train.loss == 'LogCoshLoss':
        return LogCoshLoss()
    elif cfg.train.loss == 'JSDLoss':
        return JSD()
    else:
        raise Exception(f"Loss must be one of: {', '.join(LOSS_NAMES)}")

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

# JSD divergence https://arxiv.org/pdf/1511.01844
class JSD(nn.Module):
    """Calculate and sum JSD Divergence across both VV and VH channels.
    Since it uses KL Divergence, need the input and target to be probability distributions
    i.e. they sum to one. Thus need to use softmax and also target should be log.
    Input should also be distribution in log space!"""
    def __init__(self):
        super().__init__()
        self.kl = nn.KLDivLoss(reduction='batchmean', log_target=True)

    def forward(self, p: torch.tensor, q: torch.tensor):
        # assume that normalized SAR data target and model output is passed in
        # need to call softmax on both to turn into data distribution!
        b, c, h, w = p.shape  # (batch, 2, 64, 64)

        # Normalize each SAR channel independently over all pixels (64x64)
        p = F.softmax(p.view(b, c, -1), dim=2)
        q = F.softmax(q.view(b, c, -1), dim=2)

        p_vv = p[:, 0, :]
        p_vh = p[:, 1, :]
        q_vv = q[:, 0, :]
        q_vh = q[:, 1, :]

        # Compute the mean distribution
        m1 = (0.5 * (p_vv + q_vv)).log()
        m2 = (0.5 * (p_vh + q_vh)).log()

        # Compute Jensen-Shannon Divergence
        jsd_vv = 0.5 * (self.kl(m1, p_vv.log()) + self.kl(m1, q_vv.log()))
        jsd_vh = 0.5 * (self.kl(m2, p_vh.log()) + self.kl(m2, q_vh.log()))
        return jsd_vv + jsd_vh

class BCEDiceLoss(nn.Module):
    def __init__(self, weight=None, size_average=True, pos_weight=None):
        super().__init__()
        # optional positive class weighting for the BCE component only
        self.pos_weight = pos_weight

    def forward(self, inputs, targets, smooth=1):
        # BCE Loss (use logits)
        logits = inputs.view(-1)

        # remove if your model contains a sigmoid or equivalent activation layer
        probs = F.sigmoid(inputs)

        # flatten label and prediction tensors
        probs = probs.view(-1)
        targets = targets.view(-1)

        intersection = (probs * targets).sum()
        dice_loss = 1 - (2. * intersection + smooth) / (probs.sum() + targets.sum() + smooth)
        BCE = F.binary_cross_entropy_with_logits(logits, targets, reduction='mean', pos_weight=self.pos_weight)
        BCEDice = BCE + dice_loss

        return BCEDice

class TverskyLoss(nn.Module):
    def __init__(self, alpha=ALPHA, beta=BETA, weight=None, size_average=True):
        super().__init__()
        self.alpha = alpha
        self.beta = beta

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

class InvariantBCELoss(nn.Module):
    """Passed into ShiftInvariantLoss at initialization for optimized shift-invariant BCE loss calculations.
    Will calculate and return a torch tensor of BCE losses of each patch in the batch.
    """
    def __init__(self, weight=None, reduction='mean'):
        super().__init__()
        self.weight = weight
        self.reduction = reduction

    def forward(self, inputs, targets):
        # remove if your model contains a sigmoid or equivalent activation layer
        inputs = inputs.reshape(inputs.shape[0], -1)
        targets = targets.reshape(targets.shape[0], -1)

        # LOGSIGMOID NECESSARY FOR NUMERICAL STABILITY
        log_inputs = F.logsigmoid(inputs)
        log_minus_one_inputs = F.logsigmoid(-inputs)

        if self.weight is not None:
            loss = -self.weight * (targets * log_inputs + (1 - targets) * log_minus_one_inputs)
        else:
            loss = -(targets * log_inputs + (1 - targets) * log_minus_one_inputs)

        # Apply the specified reduction method
        if self.reduction == 'mean':
            return loss.mean(axis=1)
        elif self.reduction == 'sum':
            return loss.sum(axis=1)
        else:
            return loss

class InvariantBCEDiceLoss(nn.Module):
    """Passed into ShiftInvariantLoss at initialization for optimized shift-invariant BCE Dice loss calculations.
    Will calculate and return a torch tensor of BCE Dice losses of each patch in the batch.
    """
    def __init__(self, weight=None):
        super().__init__()
        self.BCE = InvariantBCELoss(weight=weight, reduction='mean')

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
    """Passed into ShiftInvariantLoss at initialization for optimized shift-invariant Tversky loss calculations.
    Will calculate and return a torch tensor of Tversky losses of each patch in the batch.
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

class ShiftInvariantLoss(nn.Module):
    """Implementation of a shift invariant loss function. Given a batch size B of input patches of dimension N x N
    and corresponding target patches of dimension M x M, where M > N, the loss function calculates the minimum
    loss from aligning each input patch with all possible N x N windows inside of its M x M target patch.
    The total loss is the sum of all B minimum losses.

    For each patch, the corresponding N x N window in the target that produces the minimum loss is also returned.

    To use, must specify compatible loss function (Invariant BCE Loss, Invariant BCE Dice Loss, Invariant Tversky Loss)
    at initialization.

    Parameters
    ----------
    loss : obj
        Instance of InvariantBCELoss, InvariantBCEDiceLoss, InvariantTverskyLoss
    device : str
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

        # this is so we can take min across axis of patches
        patch_shift_err = torch.empty((inputs.shape[0], shift1 * shift2), dtype=torch.float32, device=self.device)
        candidate_shifts = torch.empty((shift1 * shift2, *inputs.shape), dtype=torch.float32, device=self.device)

        # first calculate all potential shift arrays
        for i in range(shift1):
            for j in range(shift2):
                window = targets[:, :, i:i+inputs.shape[-2], j:j+inputs.shape[-1]]
                candidate_shifts[i * shift2 + j] = window
                patch_shift_err[:, i * shift2 + j] = self.loss(inputs, window)

        # CAN GET BACK INDICES AND MIN AT SAME TIME!!!
        min_loss, min_ij = torch.min(patch_shift_err, dim=1)
        total_loss = min_loss.sum()
        adjusted_labels = candidate_shifts[min_ij, torch.arange(len(min_ij), device=self.device)]
        # also return adjusted labels for each patch!!!
        return total_loss, adjusted_labels

    def change_device(self, device):
        self.loss = self.loss.to(device)
        self.device = device

class TrainShiftInvariantLoss(nn.Module):
    """Implementation of a shift invariant loss function optimized for training loop by only calculating
    gradients when necessary.

    Parameters
    ----------
    loss : obj
        Instance of InvariantBCELoss, InvariantBCEDiceLoss, InvariantTverskyLoss
    device : str
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

        # this is so we can take min across axis of patches
        with torch.no_grad():
            patch_shift_err = torch.empty((inputs.shape[0], shift1 * shift2), dtype=torch.float32, device=self.device)
            candidate_shifts = torch.empty((shift1 * shift2, *inputs.shape), dtype=torch.float32, device=self.device)

            # first calculate all potential shift arrays
            for i in range(shift1):
                for j in range(shift2):
                    window = targets[:, :, i:i+inputs.shape[-2], j:j+inputs.shape[-1]]
                    candidate_shifts[i * shift2 + j] = window
                    patch_shift_err[:, i * shift2 + j] = self.loss(inputs, window)

            min_ij = torch.argmin(patch_shift_err, dim=1)

        adjusted_labels = candidate_shifts[min_ij, torch.arange(len(min_ij), device=self.device)]
        total_loss = self.loss(inputs, adjusted_labels).sum()
        return total_loss, adjusted_labels

    def change_device(self, device):
        self.loss = self.loss.to(device)
        self.device = device

class NonShiftInvariantLoss(nn.Module):
    """Wrapper for regular non shifting loss. Used when target label is larger than predicted label.

    Parameters
    ----------
    loss : obj
        Instance of BCELoss, BCEDiceLoss, TverskyLoss
    device : str
    """
    def __init__(self, loss, size=68, window=64, device='cpu'):
        super().__init__()
        self.loss = loss.to(device)
        self.device = device
        center_1 = (size - window) // 2
        self.c = (center_1, center_1 + window)

    def forward(self, inputs, targets):
        """Calculates loss using the current alignment."""
        # crop central window
        unadjusted_labels = targets[:, :, self.c[0]:self.c[1], self.c[0]:self.c[1]]
        loss = self.loss(inputs, unadjusted_labels)
        return loss, unadjusted_labels

    def change_device(self, device):
        self.loss = self.loss.to(device)
        self.device = device
