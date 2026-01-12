import torch
from torch import nn

from floodmaps.models.unet import UNet
from floodmaps.models.unet_plus import NestedUNet
from floodmaps.models.discriminator import Classifier1, Classifier2, Classifier3, ConditionalPatchGAN
from floodmaps.models.autodespeckler import VarAutoencoder, CVAE, CVAE_no_cond, ResidualDespeckler
from floodmaps.utils.utils import load_model_weights

def build_autodespeckler(cfg):
    """Factory function for SAR autodespeckler model construction.

    Parameters
    ----------
    cfg : obj
        SAR autodespeckler config instance specified in config.py.
        
    Supported autodespeckler types:
        - 'VAE': Variational Autoencoder
        - 'CVAE': Conditional Variational Autoencoder
        - 'unet': Residual U-Net for deterministic despeckling
        - 'unet++': Residual U-Net++ (NestedUNet) for deterministic despeckling
    """
    if cfg.model.autodespeckler == "VAE":
        # need to modify with new AE architecture parameters
        return VarAutoencoder(latent_dim=cfg.model.vae.latent_dim) # more hyperparameters
    elif cfg.model.autodespeckler == "CVAE":
        if cfg.model.cvae.no_cond:
            # TEMP: conditional VAE without conditioning signal in the encoder
            print("Using CVAE without conditioning signal.")
            return CVAE_no_cond(in_channels=2, out_channels=2, latent_dim=cfg.model.cvae.latent_dim,
                                unet_features=cfg.model.cvae.features, dropout=cfg.model.cvae.decoder_dropout)
        # conditional VAE
        return CVAE(in_channels=2, out_channels=2, latent_dim=cfg.model.cvae.latent_dim,
                    unet_features=cfg.model.cvae.features, dropout=cfg.model.cvae.decoder_dropout)
    elif cfg.model.autodespeckler == "unet":
        # Residual U-Net for deterministic despeckling
        # Input: 2 channels (VV, VH), Output: 2 channels (residual)
        backbone = UNet(n_channels=2, num_classes=2, dropout=cfg.model.unet.dropout)
        return ResidualDespeckler(backbone)
    elif cfg.model.autodespeckler == "unet++":
        # Residual U-Net++ (NestedUNet) for deterministic despeckling
        backbone = NestedUNet(n_channels=2, num_classes=2, dropout=cfg.model.unetpp.dropout,
                              deep_supervision=cfg.model.unetpp.deep_supervision)
        return ResidualDespeckler(backbone)
    else:
        raise Exception('Invalid autodespeckler config.')

def build_patchgan_discriminator(cfg):
    """Factory function for PatchGAN discriminator construction for CVAE-GAN.
    
    Builds a conditional PatchGAN discriminator with spectral normalization.
    Input is concatenated (noisy SAR, output) pairs (4 channels by default).
    
    Parameters
    ----------
    cfg : obj
        Config with model.discriminator.features (list of ints, default [64, 128, 256])
    
    Returns
    -------
    ConditionalPatchGAN
        PatchGAN discriminator instance
    """
    # Get features from config, default to [64, 128, 256]
    features = getattr(cfg.model.discriminator, 'features', [64, 128, 256])
    # Input: 2 channels noisy SAR + 2 channels output = 4 channels
    in_channels = 4
    return ConditionalPatchGAN(in_channels=in_channels, features=features)

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
        return UNet(n_channels, dropout=cfg.model.unet.dropout).to('cpu')
    elif cfg.model.classifier == 'unet++':
        return NestedUNet(n_channels, dropout=cfg.model.unetpp.dropout,
                          deep_supervision=cfg.model.unetpp.deep_supervision).to('cpu')
    else:
        raise Exception('Invalid classifier config.')

def build_multispectral_classifier(cfg):
    """Factory function for S2 classifier model construction.

    Parameters
    ----------
    cfg : obj
        S2 model config instance specified in config.py.
    """
    n_channels = sum([bool(int(x)) for x in cfg.data.channels])
    if cfg.model.classifier == "unet":
        return UNet(n_channels, dropout=cfg.model.unet.dropout).to('cpu')
    elif cfg.model.classifier == "unet++":
        return NestedUNet(n_channels, dropout=cfg.model.unetpp.dropout,
                        deep_supervision=cfg.model.unetpp.deep_supervision).to('cpu')
    else:
        raise Exception("model unknown")

def build_multispectral_discriminator(cfg):
    """Factory function for S2 discriminator model construction. Discriminators
    are an optional component to reduce computational cost for dry patches.

    Note: currently discriminators are defunct. New discriminators will be implemented in the future.

    Parameters
    ----------
    cfg : obj
        S2 model config instance specified in config.py.
    """
    n_channels = sum([bool(int(x)) for x in cfg.data.channels])
    if cfg.model.discriminator is None:
        return None
    elif cfg.model.discriminator == "classifier1":
        return Classifier1(n_channels).to('cpu')
    elif cfg.model.discriminator == "classifier2":
        return Classifier2(n_channels).to('cpu')
    elif cfg.model.discriminator == "classifier3":
        return Classifier3(n_channels).to('cpu')
    else:
        raise Exception("discriminator unknown")

class S2WaterDetector(nn.Module):
    """General S2 water pixel detection model with classifier and optional discriminator.
    The discriminator is an optional component to reduce computational cost for dry patches
    by only running the classifier on patches containing water.

    Parameters
    ----------
    cfg: obj
        S2 classifier config instance as defined in config.py.
    """
    def __init__(self, cfg): # model, n_channels=5, size=64, discriminator=None):
        super().__init__()
        self.cfg = cfg
        self.size = cfg.data.size
        self.n_channels = sum([bool(int(x)) for x in cfg.data.channels])
        self.classifier = build_multispectral_classifier(cfg)
        self.discriminator = build_multispectral_discriminator(cfg)

        # load weights
        if cfg.model.weights is not None:
            self.load_classifier_weights(cfg.model.weights)
        if cfg.model.discriminator is not None and cfg.model.discriminator.weights is not None:
            self.load_discriminator_weights(cfg.model.discriminator.weights)

    def get_classifier(self):
        return self.classifier

    def get_discriminator(self):
        return self.discriminator

    def uses_discriminator(self):
        return self.discriminator is not None

    def load_discriminator_weights(self, weight_path):
        """
        Load weights for the discriminator from a .pth file.

        Parameters
        ----------
        weight_path : str
            Path to the .pth file containing the discriminator weights.
        device: torch.device
        """
        try:
            load_model_weights(self.discriminator, weight_path, device='cpu',
                                model_name=f"{self.cfg.model.discriminator} discriminator")
        except RuntimeError as e:
            print(f"Error loading discriminator weights: {e}")
            print("Attempting to load weights from old model without discriminator prefix...")
            state_dict = torch.load(weight_path, map_location='cpu')
            new_state_dict = {k.replace("discriminator.", ""): v for k, v in state_dict.items()}
            self.discriminator.load_state_dict(new_state_dict)
            print(f"{self.cfg.model.discriminator} discriminator weights loaded successfully.")

    def load_classifier_weights(self, weight_path):
        """
        Load weights for the sar classifier from a .pth file.

        Parameters
        ----------
        weight_path : str
            Path to the .pth file containing the autodespeckler weights.
        device: torch.device
        """
        try:
            load_model_weights(self.classifier, weight_path, device='cpu',
                               model_name=f"{self.cfg.model.classifier} classifier")
        except RuntimeError as e:
            print(f"Error loading classifier weights: {e}")
            print("Attempting to load weights from old model without classifier prefix...")
            state_dict = torch.load(weight_path, map_location='cpu')
            new_state_dict = {k.replace("classifier.", ""): v for k, v in state_dict.items()}
            self.classifier.load_state_dict(new_state_dict, strict=False)
            print(f"{self.cfg.model.classifier} classifier weights loaded successfully.")

    def forward(self, x):
        # x: [B, n_channels, SIZE, SIZE]
        assert x.shape[1] == self.n_channels and x.shape[2] == self.size and x.shape[3] == self.size, "invalid shape"
        B = x.shape[0]
        device = x.device
        if self.discriminator is not None:
            discriminator_logits = self.discriminator(x).view(-1)  # [B]
            wet_probs = torch.sigmoid(discriminator_logits)  # [B]
            wet_mask = wet_probs > 0.5  # [B] boolean
            # Use large negative logits for dry patches to ensure sigmoid ≈ 0
            logits = torch.full((B, 1, self.size, self.size), -10.0, dtype=torch.float32, device=device)
            if wet_mask.any():
                classifier_logits = self.classifier(x[wet_mask])  # [N_wet, 1, SIZE, SIZE]
                logits[wet_mask] = classifier_logits
        else:
            logits = self.classifier(x)
        return logits

class SARWaterDetector(nn.Module):
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

        # load weights
        if cfg.model.weights is not None:
            self.load_classifier_weights(cfg.model.weights)
        if ad_cfg is not None and ad_cfg.model.weights is not None:
            self.load_autodespeckler_weights(ad_cfg.model.weights)

    def get_classifier(self):
        return self.classifier

    def get_autodespeckler(self):
        return self.autodespeckler

    def uses_autodespeckler(self):
        return self.autodespeckler is not None

    def load_autodespeckler_weights(self, weight_path):
        """
        Load weights for the autodespeckler from a .pth file.

        Parameters
        ----------
        weight_path : str
            Path to the .pth file containing the autodespeckler weights.
        """
        try:
            load_model_weights(self.autodespeckler, weight_path, device='cpu',
                                model_name=f"{self.ad_cfg.model.autodespeckler} autodespeckler")
        except RuntimeError as e:
            print(f"Error loading autodespeckler weights: {e}")
            print("Attempting to load weights from old model without autodespeckler prefix...")
            state_dict = torch.load(weight_path, map_location='cpu')
            new_state_dict = {k.replace("autodespeckler.", ""): v for k, v in state_dict.items()}
            self.autodespeckler.load_state_dict(new_state_dict)
            print(f"{self.ad_cfg.model.autodespeckler} autodespeckler weights loaded successfully.")

    def load_classifier_weights(self, weight_path):
        """
        Load weights for the sar classifier from a .pth file.

        Parameters
        ----------
        weight_path : str
            Path to the .pth file containing the classifier weights.
        """
        try:
            load_model_weights(self.classifier, weight_path, device='cpu',
                               model_name=f"{self.cfg.model.classifier} classifier")
        except RuntimeError as e:
            print(f"Error loading classifier weights: {e}")
            print("Attempting to load weights from old model without classifier prefix...")
            state_dict = torch.load(weight_path, map_location='cpu')
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
