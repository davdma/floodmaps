import torch
from torch import nn
import torch.nn.functional as F

ALPHA = 0.3
BETA = 0.7

class BCEDiceLoss(nn.Module):
    def __init__(self, weight=None, size_average=True):
        super().__init__()

    def forward(self, inputs, targets, smooth=1):
        
        # remove if your model contains a sigmoid or equivalent activation layer
        inputs = F.sigmoid(inputs)
        
        # flatten label and prediction tensors
        inputs = inputs.view(-1)
        targets = targets.view(-1)
        
        intersection = (inputs * targets).sum()                            
        dice_loss = 1 - (2. * intersection + smooth) / (inputs.sum() + targets.sum() + smooth)  
        BCE = F.binary_cross_entropy(inputs, targets, reduction='mean')
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