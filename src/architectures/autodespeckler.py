import torch
from torch import nn
from .blocks import GaussianNoiseLayer, MaskingNoiseLayer, LogGammaNoiseLayer

# CAE (Convolutional AutoEncoder)
class ConvAutoencoder(nn.Module):
    def __init__(self, in_channels=2, out_channels=2, latent_dim=200, dropout=0.1):
        super().__init__()
        self.in_channels = in_channels
        self.out_channels = out_channels
        self.latent_dim = latent_dim
        
        # subset VV and VH and concatenation of other channels in the patches?
        self.encoder = nn.Sequential(
            nn.Conv2d(in_channels, out_channels, 3, padding=1), # (64, 64)
            nn.BatchNorm2d(out_channels),
            nn.ReLU(inplace=True),
            nn.Conv2d(out_channels, out_channels, 3, padding=1), 
            nn.BatchNorm2d(out_channels),
            nn.ReLU(inplace=True),
            nn.Dropout(p=dropout),
            nn.Conv2d(out_channels, 2*out_channels, 3, padding=1, stride=2), # (32, 32)
            nn.BatchNorm2d(2*out_channels),
            nn.ReLU(inplace=True),
            nn.Conv2d(2*out_channels, 2*out_channels, 3, padding=1),
            nn.BatchNorm2d(2*out_channels),
            nn.ReLU(inplace=True),
            nn.Dropout(p=dropout),
            nn.Conv2d(2*out_channels, 4*out_channels, 3, padding=1, stride=2), # (16, 16)
            nn.BatchNorm2d(4*out_channels),
            nn.ReLU(inplace=True),
            nn.Conv2d(4*out_channels, 4*out_channels, 3, padding=1),
            nn.BatchNorm2d(4*out_channels),
            nn.ReLU(inplace=True),
            nn.Dropout(p=dropout),
            nn.Flatten(),
            nn.Linear(4*out_channels*16*16, latent_dim),
            nn.ReLU(inplace=True)
        )
        self.linear = nn.Sequential(
            nn.Linear(latent_dim, 4*out_channels*16*16),
            nn.ReLU(inplace=True)
        )
        self.decoder = nn.Sequential(
            nn.ConvTranspose2d(4*out_channels, 4*out_channels, 3, padding=1), # (16, 16)
            nn.ReLU(inplace=True),
            nn.ConvTranspose2d(4*out_channels, 2*out_channels, 3, padding=1, 
                               stride=2, output_padding=1), # (32, 32)
            nn.ReLU(inplace=True),
            nn.ConvTranspose2d(2*out_channels, 2*out_channels, 3, padding=1),
            nn.ReLU(inplace=True),
            nn.ConvTranspose2d(2*out_channels, out_channels, 3, padding=1, 
                               stride=2, output_padding=1), # (64, 64)
            nn.ReLU(inplace=True),
            nn.ConvTranspose2d(out_channels, out_channels, 3, padding=1),
            nn.ReLU(inplace=True),
            nn.ConvTranspose2d(out_channels, in_channels, 3, padding=1)
        )

    def forward(self, x):
        encoded = self.encoder(x)
        middle = self.linear(encoded)
        middle = middle.view(-1, 4*self.out_channels, 16, 16)
        decoded = self.decoder(middle)
        return decoded

# DAE (Denoising AutoEncoder)
class DenoiseAutoencoder(nn.Module):
    def __init__(self, in_channels=2, out_channels=2, latent_dim=200, dropout=0.1, std=0.1, coeff=0.8, noise_type='normal'):
        super().__init__()
        if noise_type == 'normal':
            self.noise_layer = GaussianNoiseLayer(mean=0.0, std=std)
        elif noise_type == 'masking':
            self.noise_layer = MaskingNoiseLayer(coeff=coeff)
        elif noise_type == 'log_gamma':
            self.noise_layer = LogGammaNoiseLayer()
        else:
            raise Exception('Noise not correctly specified')
            
        self.autoencoder = ConvAutoencoder(in_channels=in_channels, out_channels=out_channels, latent_dim=latent_dim, dropout=dropout)
        
    def forward(self, x):
        x = self.noise_layer(x)
        x = self.autoencoder(x)
        return x

# VAE (Variational AutoEncoder)
# source: https://github.com/AntixK/PyTorch-VAE/blob/master/models/vanilla_vae.py
class VarAutoencoder(nn.Module):
    def __init__(self, in_channels=2, latent_dim=200, hidden_dims=None):
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
                                       stride = 2,
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
                            nn.Conv2d(hidden_dims[-1], out_channels= 3,
                                      kernel_size= 3, padding= 1),
                            nn.Tanh())

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

    def reparameterize(self, mu, logvar):
        """
        Reparameterization trick to sample from N(mu, var) from
        N(0,1).
        :param mu: (Tensor) Mean of the latent Gaussian [B x D]
        :param logvar: (Tensor) Standard deviation of the latent Gaussian [B x D]
        :return: (Tensor) [B x D]
        """
        std = torch.exp(0.5 * logvar)
        eps = torch.randn_like(std)
        return eps * std + mu

    def forward(self, input):
        mu, log_var = self.encode(input)
        z = self.reparameterize(mu, log_var)
        return self.decode(z)