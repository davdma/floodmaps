from torch import nn
from .blocks import DoubleConv2d, UpConv2d

class UNet(nn.Module):
    """UNet implementation.
    
    Parameters
    ----------
    n_channels: int
        Number of input channels
    num_classes: int
        Number of output classes (default: 1 for binary segmentation)
    dropout: float
        Dropout probability (default: 0.5 to match UNet)
    nb_filter: list
        List of filter sizes for the UNet blocks (default: [16, 32, 64, 128, 256])
    """
    def __init__(self, n_channels, num_classes=1, dropout=0.5, nb_filter=[16, 32, 64, 128, 256], **kwargs):
        super().__init__()
        self.num_classes = num_classes
        self.n_channels = n_channels
        self.nb_filter = nb_filter

        self.conv1 = DoubleConv2d(n_channels, nb_filter[0])
        self.pool1 = nn.MaxPool2d(2)

        self.conv2 = DoubleConv2d(nb_filter[0], nb_filter[1])
        self.pool2 = nn.MaxPool2d(2)

        self.conv3 = DoubleConv2d(nb_filter[1], nb_filter[2])
        self.pool3 = nn.MaxPool2d(2)
        self.drop3 = nn.Dropout2d(p=dropout)

        self.conv4 = DoubleConv2d(nb_filter[2], nb_filter[3])
        self.pool4 = nn.MaxPool2d(2)
        self.drop4 = nn.Dropout2d(p=dropout)

        self.convm = DoubleConv2d(nb_filter[3], nb_filter[4])

        self.upconv4 = UpConv2d(nb_filter[4], nb_filter[3])
        self.upconv3 = UpConv2d(nb_filter[3], nb_filter[2])
        self.upconv2 = UpConv2d(nb_filter[2], nb_filter[1])
        self.upconv1 = UpConv2d(nb_filter[1], nb_filter[0])

        # output layer
        self.out = nn.Conv2d(16, num_classes, kernel_size=1)

    def forward(self, x):
        x1 = self.conv1(x)
        x2 = self.conv2(self.pool1(x1))
        x3 = self.conv3(self.pool2(x2))
        x4 = self.conv4(self.drop3(self.pool3(x3)))
        x5 = self.convm(self.drop4(self.pool4(x4)))

        x = self.upconv4(x5, x4)
        x = self.upconv3(x, x3)
        x = self.upconv2(x, x2)
        x = self.upconv1(x, x1)
        logits = self.out(x)
        
        return logits
