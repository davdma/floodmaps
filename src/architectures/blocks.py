import torch
from torch import nn
from torch.distributions import Gamma

class VGGBlock(nn.Module):
    def __init__(self, in_channels, middle_channels, out_channels):
        super().__init__()
        self.relu = nn.ReLU(inplace=True)
        self.conv1 = nn.Conv2d(in_channels, middle_channels, 3, padding=1, bias=False)
        self.bn1 = nn.BatchNorm2d(middle_channels)
        self.conv2 = nn.Conv2d(middle_channels, out_channels, 3, padding=1, bias=False)
        self.bn2 = nn.BatchNorm2d(out_channels)

    def forward(self, x):
        out = self.conv1(x)
        out = self.bn1(out)
        out = self.relu(out)

        out = self.conv2(out)
        out = self.bn2(out)
        out = self.relu(out)

        return out
    
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
    def __init__(self, in_channels, out_channels, conv=DoubleConv2d, dropout=0.5):
        super().__init__()
        self.up = nn.ConvTranspose2d(in_channels, in_channels // 2, kernel_size=2, stride=2)
        self.drop = nn.Dropout(p=dropout)
        self.conv = conv(in_channels, out_channels) # can pass in DoubleConv2d

    def forward(self, x1, x2):
        x1 = self.up(x1)
        x3 = torch.cat([x1, x2], dim=1)
        x4 = self.drop(x3)
        return self.conv(x4)

class DiscriminatorBlock1(nn.Module):
    def __init__(self, in_channels, out_channels, first_block=False):
        super().__init__()
        layers = []
        layers.append(nn.Conv2d(in_channels, out_channels, kernel_size=3, stride=1, padding=1))
        if not first_block:
            layers.append(nn.BatchNorm2d(out_channels))
        layers.append(nn.LeakyReLU(0.2, inplace=True))
        layers.append(nn.Conv2d(out_channels, out_channels, kernel_size=3, stride=2, padding=1))
        layers.append(nn.BatchNorm2d(out_channels))
        layers.append(nn.LeakyReLU(0.2, inplace=True))
        self.layers = nn.Sequential(*layers)

    def forward(self, x):
        return self.layers(x)

class DiscriminatorBlock2(nn.Module):
    def __init__(self, in_channels, out_channels, normalize=True):
        """Returns downsampling layers of each discriminator block"""
        super().__init__()
        layers = [nn.Conv2d(in_channels, out_channels, kernel_size=4, stride=2, padding=1)]
        if normalize:
            layers.append(nn.InstanceNorm2d(out_channels))
        layers.append(nn.LeakyReLU(0.2, inplace=True))
        self.layers = nn.Sequential(*layers)

    def forward(self, x):
        return self.layers(x)

# For DAE
class GaussianNoiseLayer(nn.Module):
    def __init__(self, mean=0.0, std=1.0):
        super().__init__()
        self.mean = mean
        self.std = std

    def forward(self, x):
        if self.training:
            noise = torch.normal(self.mean, self.std, x.shape, device=x.device).clip(0, 1)
            return x + noise
        else:
            return x

class LogGammaNoiseLayer(nn.Module):
    def __init__(self, looks=5, coeff=0.5, min_gamma_value=1e-10):
        super().__init__()
        self.looks = looks
        self.coeff = coeff
        self.min_gamma_value = min_gamma_value

    def forward(self, x):
        if self.training:
            gamma = Gamma(concentration=torch.ones(x.shape, device=x.device) * self.looks, 
                          rate=torch.ones(x.shape, device=x.device) * (1/self.looks)).sample()
            # ensure not zero before taking log!
            gamma = torch.clamp(gamma, min=self.min_gamma_value)
            log_gamma = 10 * torch.log10(gamma)
            # normalize
            mean = log_gamma.mean()
            std = log_gamma.std()
            normalized_log_gamma = (log_gamma - mean) / std
            return x + self.coeff * normalized_log_gamma
        else:
            return x

class MaskingNoiseLayer(nn.Module):
    def __init__(self, coeff=0.8):
        super().__init__()
        self.coeff = coeff

    def forward(self, x):
        if self.training:
            m = self.coeff * torch.ones(x.shape, device=x.device)
            return x * torch.bernoulli(m)
        else:
            return x
