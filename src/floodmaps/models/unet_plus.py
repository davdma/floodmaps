import torch
from torch import nn
from .blocks import DoubleConv2d

class NestedUNet(nn.Module):
    """UNet++ (Nested UNet) implementation.
    
    Based on: UNet++: A Nested U-Net Architecture for Medical Image Segmentation
    (Zhou et al., 2018) https://arxiv.org/abs/1807.10165
    
    Parameters
    ----------
    n_channels: int
        Number of input channels
    num_classes: int
        Number of output classes (default: 1 for binary segmentation)
    dropout: float
        Dropout probability (default: 0.5 to match UNet)
    deep_supervision: bool
        If True, uses deep supervision with averaged outputs
    """
    def __init__(self, n_channels, num_classes=1, dropout=0.5, deep_supervision=False,
            nb_filter=[16, 32, 64, 128, 256], **kwargs):
        super().__init__()
        self.num_classes = num_classes
        self.n_channels = n_channels
        self.deep_supervision = deep_supervision
        self.nb_filter = nb_filter

        # Encoder pathway (left side, column 0)
        self.conv0_0 = DoubleConv2d(n_channels, nb_filter[0])
        self.pool0 = nn.MaxPool2d(2)
        
        self.conv1_0 = DoubleConv2d(nb_filter[0], nb_filter[1])
        self.pool1 = nn.MaxPool2d(2)
        
        self.conv2_0 = DoubleConv2d(nb_filter[1], nb_filter[2])
        self.pool2 = nn.MaxPool2d(2)
        self.drop2 = nn.Dropout2d(p=dropout)
        
        self.conv3_0 = DoubleConv2d(nb_filter[2], nb_filter[3])
        self.pool3 = nn.MaxPool2d(2)
        self.drop3 = nn.Dropout2d(p=dropout)
        
        self.conv4_0 = DoubleConv2d(nb_filter[3], nb_filter[4])

        # Nested skip pathway connections with learnable upsampling
        # Column 1
        self.up1_0 = nn.ConvTranspose2d(nb_filter[1], nb_filter[1]//2, kernel_size=2, stride=2)
        self.conv0_1 = DoubleConv2d(nb_filter[0]+nb_filter[1]//2, nb_filter[0])
        
        self.up2_0 = nn.ConvTranspose2d(nb_filter[2], nb_filter[2]//2, kernel_size=2, stride=2)
        self.conv1_1 = DoubleConv2d(nb_filter[1]+nb_filter[2]//2, nb_filter[1])
        
        self.up3_0 = nn.ConvTranspose2d(nb_filter[3], nb_filter[3]//2, kernel_size=2, stride=2)
        self.conv2_1 = DoubleConv2d(nb_filter[2]+nb_filter[3]//2, nb_filter[2])
        
        self.up4_0 = nn.ConvTranspose2d(nb_filter[4], nb_filter[4]//2, kernel_size=2, stride=2)
        self.conv3_1 = DoubleConv2d(nb_filter[3]+nb_filter[4]//2, nb_filter[3])

        # Column 2
        self.up1_1 = nn.ConvTranspose2d(nb_filter[1], nb_filter[1]//2, kernel_size=2, stride=2)
        self.conv0_2 = DoubleConv2d(nb_filter[0]*2+nb_filter[1]//2, nb_filter[0])
        
        self.up2_1 = nn.ConvTranspose2d(nb_filter[2], nb_filter[2]//2, kernel_size=2, stride=2)
        self.conv1_2 = DoubleConv2d(nb_filter[1]*2+nb_filter[2]//2, nb_filter[1])
        
        self.up3_1 = nn.ConvTranspose2d(nb_filter[3], nb_filter[3]//2, kernel_size=2, stride=2)
        self.conv2_2 = DoubleConv2d(nb_filter[2]*2+nb_filter[3]//2, nb_filter[2])

        # Column 3
        self.up1_2 = nn.ConvTranspose2d(nb_filter[1], nb_filter[1]//2, kernel_size=2, stride=2)
        self.conv0_3 = DoubleConv2d(nb_filter[0]*3+nb_filter[1]//2, nb_filter[0])
        
        self.up2_2 = nn.ConvTranspose2d(nb_filter[2], nb_filter[2]//2, kernel_size=2, stride=2)
        self.conv1_3 = DoubleConv2d(nb_filter[1]*3+nb_filter[2]//2, nb_filter[1])

        # Column 4
        self.up1_3 = nn.ConvTranspose2d(nb_filter[1], nb_filter[1]//2, kernel_size=2, stride=2)
        self.conv0_4 = DoubleConv2d(nb_filter[0]*4+nb_filter[1]//2, nb_filter[0])

        # Output layers
        if self.deep_supervision:
            self.final1 = nn.Conv2d(nb_filter[0], num_classes, kernel_size=1)
            self.final2 = nn.Conv2d(nb_filter[0], num_classes, kernel_size=1)
            self.final3 = nn.Conv2d(nb_filter[0], num_classes, kernel_size=1)
            self.final4 = nn.Conv2d(nb_filter[0], num_classes, kernel_size=1)
        else:
            self.final = nn.Conv2d(nb_filter[0], num_classes, kernel_size=1)


    def forward(self, input):
        # Encoder pathway (column 0 - left side)
        x0_0 = self.conv0_0(input)
        x1_0 = self.conv1_0(self.pool0(x0_0))
        x2_0 = self.conv2_0(self.pool1(x1_0))
        x3_0 = self.conv3_0(self.drop2(self.pool2(x2_0)))
        x4_0 = self.conv4_0(self.drop3(self.pool3(x3_0)))

        # Column 1 - first nested skip connections
        x0_1 = self.conv0_1(torch.cat([x0_0, self.up1_0(x1_0)], 1))
        x1_1 = self.conv1_1(torch.cat([x1_0, self.up2_0(x2_0)], 1))
        x2_1 = self.conv2_1(torch.cat([x2_0, self.up3_0(x3_0)], 1))
        x3_1 = self.conv3_1(torch.cat([x3_0, self.up4_0(x4_0)], 1))

        # Column 2 - second nested skip connections
        x0_2 = self.conv0_2(torch.cat([x0_0, x0_1, self.up1_1(x1_1)], 1))
        x1_2 = self.conv1_2(torch.cat([x1_0, x1_1, self.up2_1(x2_1)], 1))
        x2_2 = self.conv2_2(torch.cat([x2_0, x2_1, self.up3_1(x3_1)], 1))

        # Column 3 - third nested skip connections
        x0_3 = self.conv0_3(torch.cat([x0_0, x0_1, x0_2, self.up1_2(x1_2)], 1))
        x1_3 = self.conv1_3(torch.cat([x1_0, x1_1, x1_2, self.up2_2(x2_2)], 1))

        # Column 4 - final nested skip connection
        x0_4 = self.conv0_4(torch.cat([x0_0, x0_1, x0_2, x0_3, self.up1_3(x1_3)], 1))

        # Output
        if self.deep_supervision:
            output1 = self.final1(x0_1)
            output2 = self.final2(x0_2)
            output3 = self.final3(x0_3)
            output4 = self.final4(x0_4)
            return (output1 + output2 + output3 + output4) / 4
        else:
            output = self.final(x0_4)
            return output