import torch
from torch import nn

# VAE (Variational AutoEncoder)
# source: https://github.com/AntixK/PyTorch-VAE/blob/master/models/vanilla_vae.py
class VarAutoencoder(nn.Module):
    def __init__(self, in_channels=2, out_channels=2, latent_dim=200, hidden_dims=None):
        super().__init__()
        self.latent_dim = latent_dim

        modules = []
        # hidden_dims is list of ints
        if hidden_dims is None:
            hidden_dims = [32, 64, 128, 256, 512]

        # Build Encoder
        for h_dim in hidden_dims:
            modules.append(
                nn.Sequential(
                    nn.Conv2d(in_channels, out_channels=h_dim,
                              kernel_size=3, stride=2, padding=1),
                    nn.BatchNorm2d(h_dim),
                    nn.LeakyReLU())
            )
            in_channels = h_dim

        self.encoder = nn.Sequential(*modules)
        self.fc_mu = nn.Linear(hidden_dims[-1]*4, latent_dim)
        self.fc_var = nn.Linear(hidden_dims[-1]*4, latent_dim)


        # Build Decoder
        modules = []

        self.decoder_input = nn.Linear(latent_dim, hidden_dims[-1] * 4)

        hidden_dims.reverse()

        for i in range(len(hidden_dims) - 1):
            modules.append(
                nn.Sequential(
                    nn.ConvTranspose2d(hidden_dims[i],
                                       hidden_dims[i + 1],
                                       kernel_size=3,
                                       stride=2,
                                       padding=1,
                                       output_padding=1),
                    nn.BatchNorm2d(hidden_dims[i + 1]),
                    nn.LeakyReLU())
            )

        self.decoder = nn.Sequential(*modules)
        self.final_layer = nn.Sequential(
                            nn.ConvTranspose2d(hidden_dims[-1],
                                               hidden_dims[-1],
                                               kernel_size=3,
                                               stride=2,
                                               padding=1,
                                               output_padding=1),
                            nn.BatchNorm2d(hidden_dims[-1]),
                            nn.LeakyReLU(),
                            nn.Conv2d(hidden_dims[-1], out_channels=out_channels, # originally 3?
                                      kernel_size=3, padding= 1))

    def encode(self, input):
        """
        Encodes the input by passing through the encoder network
        and returns the latent codes.
        :param input: (Tensor) Input tensor to encoder [N x C x H x W]
        :return: (Tensor) List of latent codes
        """
        result = self.encoder(input)
        result = torch.flatten(result, start_dim=1)

        # Split the result into mu and var components
        # of the latent Gaussian distribution
        mu = self.fc_mu(result)
        log_var = self.fc_var(result)

        return [mu, log_var]

    def decode(self, z):
        """
        Maps the given latent codes
        onto the image space.
        :param z: (Tensor) [B x D]
        :return: (Tensor) [B x C x H x W]
        """
        result = self.decoder_input(z)
        result = result.view(-1, 512, 2, 2)
        result = self.decoder(result)
        result = self.final_layer(result)
        return result

    def reparameterize(self, mu, log_var):
        """
        Reparameterization trick to sample from N(mu, var) from
        N(0,1).
        :param mu: (Tensor) Mean of the latent Gaussian [B x D]
        :param logvar: (Tensor) Standard deviation of the latent Gaussian [B x D]
        :return: (Tensor) [B x D]
        """
        std = torch.exp(0.5 * torch.clamp(log_var, min=-10, max=10)) # clamp at -3 maybe?
        eps = torch.randn_like(std)
        return eps * std + mu

    def forward(self, x):
        mu, log_var = self.encode(x)
        z = self.reparameterize(mu, log_var)
        return {'despeckler_output': self.decode(z), 'despeckler_input': x, 'mu': mu, 'log_var': log_var}

class UNet(nn.Module):
    """UNet for embedding into the CVAE decoder.
    
    Dropout has been added to downsampling, bottleneck, and upsampling to reduce
    overfitting. Set dropout to 0.0 for regular UNet without dropout."""
    def __init__(self, in_channels, out_channels, features=[64, 128, 256, 512], dropout=0.0):
        super().__init__()
        self.downs = nn.ModuleList()
        self.ups = nn.ModuleList()
        self.ups_conv = nn.ModuleList()
        self.pool = nn.MaxPool2d(kernel_size=2, stride=2)
        self.dropout = dropout

        # Down part of UNet
        for feature in features:
            block = [
                nn.Conv2d(in_channels, feature, kernel_size=3, padding=1),
                nn.BatchNorm2d(feature),
                nn.ReLU(inplace=True),
                nn.Conv2d(feature, feature, kernel_size=3, padding=1),
                nn.BatchNorm2d(feature),
                nn.ReLU(inplace=True),
            ]
            if dropout > 0:
                block.append(nn.Dropout2d(dropout))
            self.downs.append(nn.Sequential(*block))
            in_channels = feature

        bottleneck_block = [
            nn.Conv2d(features[-1], features[-1]*2, kernel_size=3, padding=1),
            nn.BatchNorm2d(features[-1]*2),
            nn.ReLU(inplace=True),
            nn.Conv2d(features[-1]*2, features[-1]*2, kernel_size=3, padding=1),
            nn.BatchNorm2d(features[-1]*2),
            nn.ReLU(inplace=True),
        ]
        if dropout > 0:
            bottleneck_block.append(nn.Dropout2d(dropout))
        self.bottleneck = nn.Sequential(*bottleneck_block)

        # Up part of UNet
        for feature in reversed(features):
            up_block = [
                nn.ConvTranspose2d(2*feature, feature, kernel_size=2, stride=2),
                nn.BatchNorm2d(feature),
                nn.ReLU(inplace=True),
                nn.Conv2d(feature, feature, kernel_size=3, padding=1),
                nn.BatchNorm2d(feature),
                nn.ReLU(inplace=True),
            ]
            if dropout > 0:
                up_block.append(nn.Dropout2d(dropout))
            self.ups.append(nn.Sequential(*up_block))
            ups_conv_block = [
                nn.Conv2d(2*feature, feature, kernel_size=3, padding=1),
                nn.BatchNorm2d(feature),
                nn.ReLU(inplace=True),
                nn.Conv2d(feature, feature, kernel_size=3, padding=1),
                nn.BatchNorm2d(feature),
                nn.ReLU(inplace=True)
            ]
            if dropout > 0:
                ups_conv_block.append(nn.Dropout2d(dropout))
            self.ups_conv.append(nn.Sequential(*ups_conv_block))

        self.final_conv = nn.Conv2d(features[0], out_channels, kernel_size=1)
    
    def forward(self, x):
        skip_connections = []
        for down in self.downs:
            x = down(x)
            skip_connections.append(x)
            x = self.pool(x)
        x = self.bottleneck(x)
        skip_connections = skip_connections[::-1]
        for idx in range(len(self.ups)):
            x = self.ups[idx](x)
            x = torch.cat([skip_connections[idx], x], dim=1)
            x = self.ups_conv[idx](x)
        return self.final_conv(x)

# Conditional VAE q(z | y, x) -> p(y | z, x)
# x used as conditioning signal
class CVAE(nn.Module):
    """Conditional VAE adds x as conditioning signal in addition to latent
    codes in order to predict the despeckled y.
    
    Parameters
    ----------
    in_channels: (int) Number of channels in the speckled image.
    out_channels: (int) Number of channels in the despeckled or clean image.
    latent_dim: (int) Dimension of the latent space.
    hidden_dims: (list of ints) List of hidden dimensions for the encoder.
    """
    def __init__(self, in_channels=2, out_channels=2, latent_dim=200,
                 unet_features=[64, 128, 256, 512], dropout=0.0, hidden_dims=None):
        super().__init__()
        self.latent_dim = latent_dim

        modules = []
        # hidden_dims is list of ints
        if hidden_dims is None:
            hidden_dims = [32, 64, 128, 256, 512]

        # Build Encoder
        next_channels = in_channels + out_channels
        for h_dim in hidden_dims:
            modules.append(
                nn.Sequential(
                    nn.Conv2d(next_channels, out_channels=h_dim,
                              kernel_size=3, stride=2, padding=1),
                    nn.BatchNorm2d(h_dim),
                    nn.LeakyReLU())
            )
            next_channels = h_dim

        self.encoder = nn.Sequential(*modules)
        self.fc_mu = nn.Linear(hidden_dims[-1]*4, latent_dim)
        self.fc_var = nn.Linear(hidden_dims[-1]*4, latent_dim)


        # Build Decoder
        modules = []

        self.decoder_input = nn.Linear(latent_dim, hidden_dims[-1] * 4)

        hidden_dims.reverse()

        # first upsample z then concatenate with x
        for i in range(len(hidden_dims) - 1):
            modules.append(
                nn.Sequential(
                    nn.ConvTranspose2d(hidden_dims[i],
                                       hidden_dims[i + 1],
                                       kernel_size=3,
                                       stride=2,
                                       padding=1,
                                       output_padding=1),
                    nn.BatchNorm2d(hidden_dims[i + 1]),
                    nn.LeakyReLU()
                )
            )
        modules.append(
            nn.Sequential(
                nn.ConvTranspose2d(hidden_dims[-1],
                    hidden_dims[-1],
                    kernel_size=3,
                    stride=2,
                    padding=1,
                    output_padding=1),
                nn.BatchNorm2d(hidden_dims[-1]),
                nn.LeakyReLU()
            )
        )

        self.upsample_z = nn.Sequential(*modules)

        # decoder here is just a unet architecture that takes the
        # concatenated z and x (in_channels + hidden_dims[-1]) and outputs
        # the despeckled y (out_channels)
        self.final_decoder = UNet(
            in_channels=hidden_dims[-1] + in_channels,  # upsampled_z + x
            out_channels=out_channels,
            features=unet_features,
            dropout=dropout
        )

    def encode(self, x, y):
        """
        Encodes the input by concatenating x and y, then passing the channels
        through the encoder network and returns the latent codes z.

        Parameters
        ----------
        x: (Tensor) Speckled image [N x C x H x W]
        y: (Tensor) Despeckled or clean image [N x C x H x W]
        return: (Tensor) List of latent codes z [N x D]
        """
        result = self.encoder(torch.cat((x, y), dim=1))
        result = torch.flatten(result, start_dim=1)

        # Split the result into mu and var components
        # of the latent Gaussian distribution
        mu = self.fc_mu(result)
        log_var = self.fc_var(result)

        return [mu, log_var]

    def decode(self, z, x):
        """
        Maps the given latent codes and conditioning signal x
        onto the image space.

        Parameters
        ----------
        z: (Tensor) [B x D]
        x: (Tensor) [B x C x H x W]
        return: (Tensor) [B x C x H x W]
        """
        result = self.decoder_input(z)
        result = result.view(-1, 512, 2, 2)
        upsampled_z = self.upsample_z(result)
        concatenated = torch.cat([upsampled_z, x], dim=1)
        output = self.final_decoder(concatenated)
        return output

    def reparameterize(self, mu, log_var):
        """
        Reparameterization trick to sample from N(mu, var) from
        N(0,1).
        :param mu: (Tensor) Mean of the latent Gaussian [B x D]
        :param logvar: (Tensor) Standard deviation of the latent Gaussian [B x D]
        :return: (Tensor) [B x D]
        """
        std = torch.exp(0.5 * torch.clamp(log_var, min=-10, max=10)) # clamp at -3 maybe?
        eps = torch.randn_like(std)
        return eps * std + mu

    def forward(self, x, y):
        # x is speckled, y is despeckled or clean image
        mu, log_var = self.encode(x, y)
        z = self.reparameterize(mu, log_var)
        return {'despeckler_output': self.decode(z, x), 'despeckler_input': x, 'mu': mu, 'log_var': log_var}
    
    def inference(self, x, deterministic=False):
        """
        Inference mode for CVAE. Deterministic inference uses z=0, while
        stochastic inference samples z from N(0,1). For N mean stochastic inference
        feed in N samples of x and then average the outputs along the batch dimension.
        
        Parameters
        ----------
        x: (Tensor) Speckled image [N x C x H x W]
        deterministic: (bool) Whether to use deterministic inference
        
        Returns
        -------
        dict
            Dictionary containing the despeckled output, input, and latent codes
        """
        if deterministic:
            z = torch.zeros(x.shape[0], self.latent_dim).to(x.device)
        else:
            z = torch.randn(x.shape[0], self.latent_dim).to(x.device)
        return {'despeckler_output': self.decode(z, x), 'despeckler_input': x}

# TEMP: For ablation study we remove conditioning signal from encoder
class CVAE_no_cond(nn.Module):
    """CVAE without conditioning signal in the encoder.
    
    Parameters
    ----------
    in_channels: (int) Number of channels in the speckled image.
    out_channels: (int) Number of channels in the despeckled or clean image.
    latent_dim: (int) Dimension of the latent space.
    hidden_dims: (list of ints) List of hidden dimensions for the encoder.
    """
    def __init__(self, in_channels=2, out_channels=2, latent_dim=200,
                 unet_features=[64, 128, 256, 512], dropout=0.0, hidden_dims=None):
        super().__init__()
        self.latent_dim = latent_dim

        modules = []
        # hidden_dims is list of ints
        if hidden_dims is None:
            hidden_dims = [32, 64, 128, 256, 512]

        # Build Encoder
        next_channels = in_channels
        for h_dim in hidden_dims:
            modules.append(
                nn.Sequential(
                    nn.Conv2d(next_channels, out_channels=h_dim,
                              kernel_size=3, stride=2, padding=1),
                    nn.BatchNorm2d(h_dim),
                    nn.LeakyReLU())
            )
            next_channels = h_dim

        self.encoder = nn.Sequential(*modules)
        self.fc_mu = nn.Linear(hidden_dims[-1]*4, latent_dim)
        self.fc_var = nn.Linear(hidden_dims[-1]*4, latent_dim)


        # Build Decoder
        modules = []

        self.decoder_input = nn.Linear(latent_dim, hidden_dims[-1] * 4)

        hidden_dims.reverse()

        # first upsample z then concatenate with x
        for i in range(len(hidden_dims) - 1):
            modules.append(
                nn.Sequential(
                    nn.ConvTranspose2d(hidden_dims[i],
                                       hidden_dims[i + 1],
                                       kernel_size=3,
                                       stride=2,
                                       padding=1,
                                       output_padding=1),
                    nn.BatchNorm2d(hidden_dims[i + 1]),
                    nn.LeakyReLU()
                )
            )
        modules.append(
            nn.Sequential(
                nn.ConvTranspose2d(hidden_dims[-1],
                    hidden_dims[-1],
                    kernel_size=3,
                    stride=2,
                    padding=1,
                    output_padding=1),
                nn.BatchNorm2d(hidden_dims[-1]),
                nn.LeakyReLU()
            )
        )

        self.upsample_z = nn.Sequential(*modules)

        # decoder here is just a unet architecture that takes the
        # concatenated z and x (in_channels + hidden_dims[-1]) and outputs
        # the despeckled y (out_channels)
        self.final_decoder = UNet(
            in_channels=hidden_dims[-1] + in_channels,  # upsampled_z + x
            out_channels=out_channels,
            features=unet_features,
            dropout=dropout
        )

    def encode(self, y):
        """
        Encodes the input by concatenating x and y, then passing the channels
        through the encoder network and returns the latent codes z.

        Parameters
        ----------
        y: (Tensor) Despeckled or clean image [N x C x H x W]
        return: (Tensor) List of latent codes z [N x D]
        """
        result = self.encoder(y)
        result = torch.flatten(result, start_dim=1)

        # Split the result into mu and var components
        # of the latent Gaussian distribution
        mu = self.fc_mu(result)
        log_var = self.fc_var(result)

        return [mu, log_var]

    def decode(self, z, x):
        """
        Maps the given latent codes and conditioning signal x
        onto the image space.

        Parameters
        ----------
        z: (Tensor) [B x D]
        x: (Tensor) [B x C x H x W]
        return: (Tensor) [B x C x H x W]
        """
        result = self.decoder_input(z)
        result = result.view(-1, 512, 2, 2)
        upsampled_z = self.upsample_z(result)
        concatenated = torch.cat([upsampled_z, x], dim=1)
        output = self.final_decoder(concatenated)
        return output

    def reparameterize(self, mu, log_var):
        """
        Reparameterization trick to sample from N(mu, var) from
        N(0,1).
        :param mu: (Tensor) Mean of the latent Gaussian [B x D]
        :param logvar: (Tensor) Standard deviation of the latent Gaussian [B x D]
        :return: (Tensor) [B x D]
        """
        std = torch.exp(0.5 * torch.clamp(log_var, min=-10, max=10)) # clamp at -3 maybe?
        eps = torch.randn_like(std)
        return eps * std + mu

    def forward(self, x, y):
        # x is speckled, y is despeckled or clean image
        mu, log_var = self.encode(y)
        z = self.reparameterize(mu, log_var)
        return {'despeckler_output': self.decode(z, x), 'despeckler_input': x, 'mu': mu, 'log_var': log_var}
    
    def inference(self, x, deterministic=False):
        """
        Inference mode for CVAE. Deterministic inference uses z=0, while
        stochastic inference samples z from N(0,1). For N mean stochastic inference
        feed in N samples of x and then average the outputs along the batch dimension.
        
        Parameters
        ----------
        x: (Tensor) Speckled image [N x C x H x W]
        deterministic: (bool) Whether to use deterministic inference

        Returns
        -------
        dict
            Dictionary containing the despeckled output, input, and latent codes
        """
        if deterministic:
            z = torch.zeros(x.shape[0], self.latent_dim).to(x.device)
        else:
            z = torch.randn(x.shape[0], self.latent_dim).to(x.device)
        return {'despeckler_output': self.decode(z, x), 'despeckler_input': x}


class ResidualDespeckler(nn.Module):
    """Residual learning wrapper for U-Net/U-Net++ despeckling.
    
    Predicts residual rhat and outputs yhat = x - rhat.
    This is a deterministic model (no VAE components) that learns
    to predict the speckle noise pattern to be subtracted from the input.
    
    Parameters
    ----------
    backbone : nn.Module
        The backbone network (U-Net or U-Net++) that predicts the residual.
        Should accept input of shape [B, C, H, W] and output [B, C, H, W].
    
    Returns
    -------
    dict
        Dictionary containing:
        - 'despeckler_output': The despeckled image (x - residual)
        - 'despeckler_input': The original speckled input x
        - 'residual': The predicted residual/noise pattern
    """
    def __init__(self, backbone):
        super().__init__()
        self.backbone = backbone
    
    def forward(self, x, y=None):
        """Forward pass for residual despeckling.
        
        Parameters
        ----------
        x : torch.Tensor
            Speckled input image [B, C, H, W]
        y : torch.Tensor, optional
            Target clean image (ignored, for interface compatibility with CVAE)
        
        Returns
        -------
        dict
            Output dictionary with despeckler_output, despeckler_input, and residual
        """
        residual = self.backbone(x)
        despeckled = x - residual
        return {'despeckler_output': despeckled, 'despeckler_input': x, 'residual': residual}
    
    def inference(self, x):
        """Inference mode (identical to forward for deterministic model).
        
        Parameters
        ----------
        x : torch.Tensor
            Speckled input image [B, C, H, W]
        
        Returns
        -------
        dict
            Output dictionary with despeckler_output, despeckler_input, and residual
        """
        return self.forward(x)