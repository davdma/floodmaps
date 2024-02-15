import torch
from torch import nn

class WaterPixelDetector(nn.Module):
    def __init__(self, model, n_channels=5, size=64, discriminator=None):
        super().__init__()
        self.model = model
        self.discriminator = discriminator
        self.n_channels = n_channels
        self.size = size

    def forward(self, x):
        # assume x = [1, n_channels, SIZE, SIZE]
        assert x.shape[0] == 1 and x.shape[1] == self.n_channels and x.shape[2] == self.size and x.shape[3] == self.size, "invalid shape"
        if self.discriminator is not None:
            wet = self.discriminator(x)
            if wet > 0.5:
                logits = self.model(x)
                pred = torch.where(nn.functional.sigmoid(logits) > 0.5, 1.0, 0.0).byte()
            else:
                pred = torch.zeros((1, 1, self.size, self.size), dtype=torch.uint8)
        else:
            logits = self.model(x)
            pred = torch.where(nn.functional.sigmoid(logits) > 0.5, 1.0, 0.0).byte() # [1, 1, SIZE, SIZE]

        return pred
    
