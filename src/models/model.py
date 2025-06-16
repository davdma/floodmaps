import torch
from torch import nn
from models.unet import UNet
from models.unet_plus import NestedUNet
from models.autodespeckler import ConvAutoencoder1, ConvAutoencoder2, DenoiseAutoencoder, VarAutoencoder, CVAE
from utils.utils import load_model_weights

def build_autodespeckler(cfg):
    """Factory function for SAR autodespeckler model construction.

    Parameters
    ----------
    cfg : obj
        SAR autodespeckler config instance specified in config.py.
    """
    if cfg.model.autodespeckler == "CNN1":
        return ConvAutoencoder1(latent_dim=cfg.model.cnn1.latent_dim,
                                dropout=cfg.model.cnn1.AD_dropout,
                                activation_func=cfg.model.cnn1.AD_activation_func)
    elif cfg.model.autodespeckler == "CNN2":
        return ConvAutoencoder2(num_layers=cfg.model.cnn2.AD_num_layers,
                                kernel_size=cfg.model.cnn2.AD_kernel_size,
                                dropout=cfg.model.cnn2.AD_dropout,
                                activation_func=cfg.model.cnn2.AD_activation_func)
    elif cfg.model.autodespeckler == "DAE":
        # need to modify with new AE architecture parameters
        return DenoiseAutoencoder(num_layers=cfg.model.dae.AD_num_layers,
                                  kernel_size=cfg.model.dae.AD_kernel_size,
                                  dropout=cfg.model.dae.AD_dropout,
                                  coeff=cfg.model.dae.noise_coeff,
                                  noise_type=cfg.model.dae.noise_type,
                                  activation_func=cfg.model.dae.AD_activation_func)
    elif cfg.model.autodespeckler == "VAE":
        # need to modify with new AE architecture parameters
        return VarAutoencoder(latent_dim=cfg.model.vae.latent_dim) # more hyperparameters
    elif cfg.model.autodespeckler == "CVAE":
        # conditional VAE
        return CVAE(in_channels=2, out_channels=2, latent_dim=cfg.model.cvae.latent_dim,
                    unet_features=cfg.model.cvae.features)
    else:
        raise Exception('Invalid autodespeckler config.')

def build_sar_classifier(cfg):
    """Factory function for SAR classifier model construction.

    Parameters
    ----------
    cfg : obj
        SAR classifier config instance specified in config.py.
    """
    channels = [bool(int(x)) for x in cfg.data.channels]
    n_channels = sum(channels)
    if cfg.model.classifier == 'unet':
        return UNet(n_channels, dropout=cfg.model.unet.dropout)
    elif cfg.model.classifier == 'unet++':
        return NestedUNet(n_channels, dropout=cfg.model.unetpp.dropout,
                          deep_supervision=cfg.model.unetpp.deep_supervision)
    else:
        raise Exception('Invalid classifier config.')

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

class SARPixelDetector(nn.Module):
    """General S1 water pixel detection model with classifier and optional autodespeckler.
    The autodespeckler is an autoencoder for the SAR channels that aimed to extract salient features
    from speckled SAR data for improving labeling performance.

    Parameters
    ----------
    cfg: obj
        SAR classifier config instance as defined in config.py.
    ad_cfg: obj
        SAR autodespeckler config instance as defined in config.py.
    """
    def __init__(self, cfg, ad_cfg=None):
        super().__init__()
        self.cfg = cfg
        self.ad_cfg = ad_cfg
        self.classifier = build_sar_classifier(cfg)
        self.autodespeckler = (build_autodespeckler(ad_cfg)
                                if ad_cfg is not None else None)

    def get_classifier(self):
        return self.classifier

    def get_autodespeckler(self):
        return self.autodespeckler

    def uses_autodespeckler(self):
        return self.autodespeckler is not None

    def load_autodespeckler_weights(self, weight_path, device):
        """
        Load weights for the autodespeckler from a .pth file.

        Parameters
        ----------
        weight_path : str
            Path to the .pth file containing the autodespeckler weights.
        device: torch.device
        """
        try:
            load_model_weights(self.autodespeckler, weight_path, device,
                                model_name=f"{self.ad_cfg.model.autodespeckler} autodespeckler")
        except RuntimeError as e:
            print(f"Error loading autodespeckler weights: {e}")
            print("Attempting to load weights from old model without autodespeckler prefix...")
            state_dict = torch.load(weight_path, map_location=device)
            new_state_dict = {k.replace("autodespeckler.", ""): v for k, v in state_dict.items()}
            self.autodespeckler.load_state_dict(new_state_dict)
            print(f"{self.ad_cfg.model.autodespeckler} autodespeckler weights loaded successfully.")

    def load_classifier_weights(self, weight_path, device):
        """
        Load weights for the sar classifier from a .pth file.

        Parameters
        ----------
        weight_path : str
            Path to the .pth file containing the autodespeckler weights.
        device: torch.device
        """
        try:
            load_model_weights(self.classifier, weight_path, device,
                               model_name=f"{self.cfg.model.classifier} classifier")
        except RuntimeError as e:
            print(f"Error loading classifier weights: {e}")
            print("Attempting to load weights from old model without classifier prefix...")
            state_dict = torch.load(weight_path, map_location=device)
            new_state_dict = {k.replace("classifier.", ""): v for k, v in state_dict.items()}
            self.classifier.load_state_dict(new_state_dict, strict=False)
            print(f"{self.cfg.model.classifier} classifier weights loaded successfully.")

    def freeze_ad_weights(self):
        """Freeze the weights of the autodespeckler during training."""
        for param in self.autodespeckler.parameters():
            param.requires_grad = False

    def unfreeze_ad_weights(self):
        """Unfreeze the weights of the autodespeckler during training."""
        for param in self.autodespeckler.parameters():
            param.requires_grad = True

    def forward(self, x):
        """Returns dictionary containing outputs. If autodespeckler architecture used then
        output from the autodespeckler head is also included."""
        if self.uses_autodespeckler():
            sar = x[:, :2, :, :]
            despeckler_dict = self.autodespeckler(sar)
            logits = self.classifier(torch.cat((despeckler_dict['despeckler_output'], x[:, 2:, :, :]), 1))
            despeckler_dict['classifier_output'] = logits
            return despeckler_dict
        else:
            logits = self.classifier(x)
            out = {'classifier_output': logits}
            return out
