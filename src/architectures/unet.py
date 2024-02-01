import torch
from torch import nn
from .blocks import DoubleConv2d, UpConv2d

class UNet(nn.Module):
    def __init__(self, n_channels, dropout=0.5):
        super().__init__()
        self.n_channels = n_channels

        self.conv1 = (DoubleConv2d(n_channels, 16))
        self.pool1 = nn.MaxPool2d(2)
        self.drop1 = nn.Dropout(p=dropout)

        self.conv2 = (DoubleConv2d(16, 32))
        self.pool2 = nn.MaxPool2d(2)
        self.drop2 = nn.Dropout(p=dropout)

        self.conv3 = (DoubleConv2d(32, 64))
        self.pool3 = nn.MaxPool2d(2)
        self.drop3 = nn.Dropout(p=dropout)

        self.conv4 = (DoubleConv2d(64, 128))
        self.pool4 = nn.MaxPool2d(2)
        self.drop4 = nn.Dropout(p=dropout)

        self.convm = (DoubleConv2d(128, 256))

        self.upconv4 = UpConv2d(256, 128, dropout=dropout)
        self.upconv3 = UpConv2d(128, 64, dropout=dropout)
        self.upconv2 = UpConv2d(64, 32, dropout=dropout)
        self.upconv1 = UpConv2d(32, 16, dropout=dropout)

        # output layer
        self.out = nn.Conv2d(16, 1, kernel_size=1)

    def forward(self, x):
        x1 = self.conv1(x)
        out = self.pool1(x1)
        out = self.drop1(out)
        x2 = self.conv2(out)
        out = self.pool2(x2)
        out = self.drop2(out)
        x3 = self.conv3(out)
        out = self.pool3(x3)
        out = self.drop3(out)
        x4 = self.conv4(out)
        out = self.pool4(x4)
        out = self.drop4(out)
        x5 = self.convm(out)

        x = self.upconv4(x5, x4)
        x = self.upconv3(x, x3)
        x = self.upconv2(x, x2)
        x = self.upconv1(x, x1)
        logits = self.out(x)
        
        return logits
