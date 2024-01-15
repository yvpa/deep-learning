import torch.nn as nn

from autoencoder.decoder import Decoder
from autoencoder.encoder import Encoder


class AutoEncoder(nn.Module):
    def __init__(self, n_channels, embedding_dim):
        super().__init__()

        self.encoder = Encoder(n_channels, embedding_dim)
        self.decoder = Decoder(n_channels, embedding_dim)

    def encode(self, x):
        return self.encoder(x)

    def decode(self, x):
        return self.decoder(x)
