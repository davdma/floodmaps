import torch
from torch import nn

class DoubleConv2d(nn.Module):
    """Convolution layer followed by batch normalization and ReLU, done twice."""

    def __init__(self, in_channels, out_channels):
        super().__init__()
        self.double_conv = nn.Sequential(
            nn.Conv2d(in_channels, out_channels, kernel_size=3, padding='same', bias=False),
            nn.BatchNorm2d(out_channels),
            nn.ReLU(inplace=True),
            nn.Conv2d(out_channels, out_channels, kernel_size=3, padding='same', bias=False),
            nn.BatchNorm2d(out_channels),
            nn.ReLU(inplace=True)
        )

    def forward(self, x):
        return self.double_conv(x)

class UpConv2d(nn.Module):
    def __init__(self, in_channels, out_channels, conv, dropout=0.5):
        super().__init__()
        self.up = nn.ConvTranspose2d(in_channels, in_channels // 2, kernel_size=2, stride=2)
        self.drop = nn.Dropout(p=dropout)
        self.conv = conv(in_channels, out_channels) # can pass in DoubleConv2d

    def forward(self, x1, x2):
        x1 = self.up(x1)
        x3 = torch.cat([x1, x2], dim=1)
        x4 = self.drop(x3)
        return self.conv(x4)