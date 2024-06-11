import torch
from torch import nn
from architectures.unet import UNet
from architectures.unet_plus import NestedUNet
from architectures.autodespeckler import ConvAutoencoder, DenoiseAutoencoder, VarAutoencoder

class WaterPixelDetector(nn.Module):
    """General S2 water pixel detection model with classifier and optional discriminator.

    Parameters
    ----------
    model : obj
        UNet or UNet++ classifer or any water segmentation model.
    n_channels : int
        Number of input channels used.
    size : int
        Size of input patches.
    discriminator : obj, optional
        Discriminator predicts whether a patch contains water first. If water is present in the
        patch, the classifier is then used to label the areas of water presence.
    """
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

class SARClassifier(nn.Module):
    """General S1 water pixel detection model with classifier and optional autodespeckler.
    The autodespeckler is an autoencoder for the SAR channels that aimed to extract salient features 
    from speckled SAR data for improving labeling performance.

    Parameters
    ----------
    config : dict
        Dictionary containing model parameters.
    n_channels : int
        Number of input channels used.
    """
    def __init__(self, config, n_channels=7):
        super().__init__()
        # classifier
        if config['name'] == 'unet':
            self.classifier = UNet(n_channels, dropout=config['dropout'])
        elif config['name'] == 'unet++':
            self.classifier = NestedUNet(n_channels, dropout=config['dropout'], deep_supervision=config['deep_supervision'])
        else:
            raise Exception('Classifier not specified')

        # autodespeckler
        if config['autodespeckler'] == "CNN":
            self.autodespeckler = ConvAutoencoder(latent_dim=config['latent_dim'], dropout=config['AD_dropout'])
        elif config['autodespeckler'] == "DAE":
            self.autodespeckler = DenoiseAutoencoder(latent_dim=config['latent_dim'], dropout=config['AD_dropout'], 
                                                     std=config['normal_noise_std'], coeff=config['masking_noise_coeff'],
                                                     noise_type=config['noise_type'])
        elif config['autodespeckler'] == "VAE":
            self.autodespeckler = VarAutoencoder(latent_dim=config['latent_dim'])
        else:
            self.autodespeckler = None

    def forward(self, x):
        if self.autodespeckler is not None:
            sar = x[:, :2, :, :]
            despeckled = self.autodespeckler(sar)
            out = self.classifier(torch.cat((despeckled, x[:, 2:, :, :]), 1))
        else:
            out = self.classifier(x)
        return out
