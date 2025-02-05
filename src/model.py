import torch
from torch import nn
from architectures.unet import UNet
from architectures.unet_plus import NestedUNet
from architectures.autodespeckler import ConvAutoencoder1, ConvAutoencoder2, DenoiseAutoencoder, VarAutoencoder

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
        self.config = config
        self.classifier = self.get_classifier(config, n_channels)
        self.autodespeckler = self.get_autodespeckler(config)

    def get_classifier(self, config, n_channels):
        if config['name'] == 'unet':
            return UNet(n_channels, dropout=config['dropout'])
        elif config['name'] == 'unet++':
            return NestedUNet(n_channels, dropout=config['dropout'], deep_supervision=config['deep_supervision'])
        else:
            raise Exception('Classifier not specified')

    def get_autodespeckler(self, config):
        if config['autodespeckler'] == "CNN1":
            return ConvAutoencoder1(latent_dim=config['latent_dim'], dropout=config['AD_dropout'])
        elif config['autodespeckler'] == "CNN2":
            # activation function            
            return ConvAutoencoder2(num_layers=config['AD_num_layers'], 
                                    kernel_size=config['AD_kernel_size'], 
                                    dropout=config['AD_dropout'], 
                                    activation_func=config['AD_activation_func'])
        elif config['autodespeckler'] == "DAE":
            # need to modify with new AE architecture parameters
            return DenoiseAutoencoder(num_layers=config['AD_num_layers'], 
                                      kernel_size=config['AD_kernel_size'],
                                      dropout=config['AD_dropout'],
                                      coeff=config['noise_coeff'],
                                      noise_type=config['noise_type'],
                                      activation_func=config['AD_activation_func'])
        elif config['autodespeckler'] == "VAE":
            # need to modify with new AE architecture parametersi
            return VarAutoencoder(latent_dim=config['latent_dim']) # more hyperparameters
        else:
            return None

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
        if weight_path is None:
            return
            
        if not self.uses_autodespeckler():
            raise ValueError("Autodespeckler is not initialized in this model.")
        
        state_dict = torch.load(weight_path, map_location=device)
        try:
            self.autodespeckler.load_state_dict(state_dict)
            print("Autodespeckler weights loaded successfully.")
        except RuntimeError as e:
            print(f"Error loading autodespeckler weights: {e}")
            raise e

    def load_classifier_weights(self, weight_path, device):
        """
        Load weights for the sar classifier from a .pth file.
        
        Parameters
        ----------
        weight_path : str
            Path to the .pth file containing the autodespeckler weights.
        device: torch.device
        """
        if weight_path is None:
            return
        
        state_dict = torch.load(weight_path, map_location=device)
        try:
            self.classifier.load_state_dict(state_dict)
            print("Classifier weights loaded successfully.")
        except RuntimeError as e:
            print(f"Error loading classifier weights: {e}")
            raise e

    def freeze_autodespeckler_weights(self):
        """Freeze the weights of the autodespeckler during training."""
        for param in self.autodespeckler.parameters():
            param.requires_grad = False

    def unfreeze_autodespeckler_weights(self):
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
            despeckler_dict['final_output'] = logits
            return despeckler_dict
        else:
            logits = self.classifier(x)
            out = {'final_output': logits}
            return out
