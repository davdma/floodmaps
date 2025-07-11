import torch
from torch import nn
from .blocks import DiscriminatorBlock1, DiscriminatorBlock2

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