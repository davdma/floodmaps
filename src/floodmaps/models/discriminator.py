import torch
from torch import nn
from torch.nn.utils import spectral_norm
from .blocks import DiscriminatorBlock1, DiscriminatorBlock2


class ConditionalPatchGAN(nn.Module):
    """34-RF PatchGAN with spectral normalization for CVAE-GAN.
    
    Conditional discriminator that takes concatenated (noisy SAR, output) pairs.
    For 64x64 input: outputs 14x14 patch logits.
    
    Architecture:
        a) conv(k=4,s=2,p=1) -> 32x32, LeakyReLU, no norm
        b) conv(k=4,s=2,p=1) -> 16x16, LeakyReLU, spectral norm
        c) conv(k=4,s=1,p=1) -> 15x15, LeakyReLU, spectral norm
        d) conv(k=4,s=1,p=1) -> 14x14, spectral norm (output 1 channel logits)
    
    Parameters
    ----------
    in_channels : int
        Number of input channels (default 4: 2 noisy SAR + 2 despeckled output)
    features : list of int
        Feature dimensions for each layer (default [64, 128, 256])
    """
    def __init__(self, in_channels=4, features=None):
        super().__init__()
        if features is None:
            features = [64, 128, 256]
        
        # a) conv(k=4,s=2,p=1) -> 32x32, no norm
        self.conv_a = nn.Sequential(
            nn.Conv2d(in_channels, features[0], kernel_size=4, stride=2, padding=1, bias=True),
            nn.LeakyReLU(0.2, inplace=True)
        )
        
        # b) conv(k=4,s=2,p=1) -> 16x16, spectral norm
        self.conv_b = nn.Sequential(
            spectral_norm(nn.Conv2d(features[0], features[1], kernel_size=4, stride=2, padding=1, bias=False)),
            nn.LeakyReLU(0.2, inplace=True)
        )
        
        # c) conv(k=4,s=1,p=1) -> 15x15, spectral norm
        self.conv_c = nn.Sequential(
            spectral_norm(nn.Conv2d(features[1], features[2], kernel_size=4, stride=1, padding=1, bias=False)),
            nn.LeakyReLU(0.2, inplace=True)
        )
        
        # d) conv(k=4,s=1,p=1) -> 14x14, spectral norm, output 1 channel
        self.conv_d = spectral_norm(nn.Conv2d(features[2], 1, kernel_size=4, stride=1, padding=1, bias=False))
        
        # Initialize weights
        self._init_weights()
    
    def _init_weights(self):
        """Initialize weights with normal distribution (standard for GANs)."""
        for m in self.modules():
            if isinstance(m, nn.Conv2d):
                nn.init.normal_(m.weight.data, 0.0, 0.02)
                if m.bias is not None:
                    nn.init.constant_(m.bias.data, 0.0)
    
    def forward(self, x):
        """Forward pass.
        
        Parameters
        ----------
        x : torch.Tensor
            Concatenated input [B, in_channels, H, W]
            Typically concat([noisy_sar, output], dim=1)
        
        Returns
        -------
        torch.Tensor
            Patch logits [B, 1, H', W'] (14x14 for 64x64 input)
        """
        x = self.conv_a(x)
        x = self.conv_b(x)
        x = self.conv_c(x)
        x = self.conv_d(x)
        return x


def weights_init_normal(m):
    classname = m.__class__.__name__
    if classname.find("Conv") != -1:
        torch.nn.init.normal_(m.weight.data, 0.0, 0.02)
        if hasattr(m, "bias") and m.bias is not None:
            torch.nn.init.constant_(m.bias.data, 0.0)
    elif classname.find("BatchNorm2d") != -1:
        torch.nn.init.normal_(m.weight.data, 1.0, 0.02)
        torch.nn.init.constant_(m.bias.data, 0.0)

class Classifier1(nn.Module):
    def __init__(self, n_channels):
        super().__init__()
        self.n_channels = n_channels
        
        self.dblock1 = DiscriminatorBlock1(n_channels, 64, first_block=True)
        self.dblock2 = DiscriminatorBlock1(64, 64, first_block=False)
        self.dblock3 = DiscriminatorBlock1(64, 128, first_block=False)
        self.dblock4 = DiscriminatorBlock1(128, 128, first_block=False)
        self.dblock5 = DiscriminatorBlock1(128, 256, first_block=False)
        self.dblock6 = DiscriminatorBlock1(256, 256, first_block=False)
        self.dblock7 = DiscriminatorBlock1(256, 512, first_block=False)
        self.out = nn.Conv2d(512, 1, kernel_size=3, stride=1, padding=1)

    def forward(self, x):
        x = self.dblock1(x)
        x = self.dblock2(x)
        x = self.dblock3(x)
        x = self.dblock4(x)
        x = self.dblock5(x)
        x = self.dblock6(x)
        x = self.dblock7(x)
        return self.out(x).flatten()

class Classifier2(nn.Module):
    def __init__(self, n_channels):
        super().__init__()
        self.n_channels = n_channels
        self.net = nn.Sequential(
            nn.Conv2d(n_channels, 64, kernel_size=3, padding=1),
            nn.LeakyReLU(0.2),

            nn.Conv2d(64, 64, kernel_size=3, stride=2, padding=1),
            nn.BatchNorm2d(64),
            nn.LeakyReLU(0.2),

            nn.Conv2d(64, 128, kernel_size=3, stride=1, padding=1),
            nn.BatchNorm2d(128),
            nn.LeakyReLU(0.2),

            nn.Conv2d(128, 128, kernel_size=3, stride=2, padding=1),
            nn.BatchNorm2d(128),
            nn.LeakyReLU(0.2),

            nn.Conv2d(128, 256, kernel_size=3, stride=1, padding=1),
            nn.BatchNorm2d(256),
            nn.LeakyReLU(0.2),

            nn.Conv2d(256, 256, kernel_size=3, stride=2, padding=1),
            nn.BatchNorm2d(256),
            nn.LeakyReLU(0.2),

            nn.Conv2d(256, 512, kernel_size=3, stride=1, padding=1),
            nn.BatchNorm2d(512),
            nn.LeakyReLU(0.2),

            nn.Conv2d(512, 512, kernel_size=3, stride=2, padding=1),
            nn.BatchNorm2d(512),
            nn.LeakyReLU(0.2),

            nn.AdaptiveAvgPool2d(1),
            nn.Conv2d(512, 1024, kernel_size=1, stride=1, padding=0),
            nn.LeakyReLU(0.2),
            nn.Conv2d(1024, 1, kernel_size=1, stride=1, padding=0)
        )

    def forward(self, x):
        return self.net(x).flatten()

class Classifier3(nn.Module):
    def __init__(self, n_channels):
        super().__init__()
        self.n_channels = n_channels
        # Calculate output shape of image discriminator (PatchGAN)
        # output_shape = (1, height // 2 ** 4, width // 2 ** 4)

        self.dblock1 = DiscriminatorBlock2(n_channels, 64, normalize=False)
        self.dblock2 = DiscriminatorBlock2(64, 128)
        self.dblock3 = DiscriminatorBlock2(128, 256)
        self.dblock4 = DiscriminatorBlock2(256, 512)
        self.out = nn.Sequential(
            nn.ZeroPad2d((1, 0, 1, 0)),
            nn.Conv2d(512, 1, kernel_size=4, padding=1)
        )

    def forward(self, x):
        x = self.dblock1(x)
        x = self.dblock2(x)
        x = self.dblock3(x)
        x = self.dblock4(x)
        return self.out(x).flatten()