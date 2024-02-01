import torch
from torch import nn

class WaterPixelDetector(nn.Module):
    def __init__(self, model, discriminator=None):
        super().__init__()
        self.model = model
        self.discriminator = discriminator

    def forward(x):
        if self.discriminator is not None:
            logit = self.discriminator(x)
            if logit > 0.5:
                pred = self.model(x)
            else:
                pred = torch.zeros(x.shape)
        else:
            pred = self.model(x)

        return pred
    