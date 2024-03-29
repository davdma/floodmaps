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
        inputs = inputs.view(-1)
        targets = targets.view(-1)
        
        #True Positives, False Positives & False Negatives
        TP = (inputs * targets).sum()    
        FP = ((1-targets) * inputs).sum()
        FN = (targets * (1-inputs)).sum()
       
        Tversky = (TP + smooth) / (TP + self.alpha * FP + self.beta * FN + smooth)  
        
        return 1 - Tversky
