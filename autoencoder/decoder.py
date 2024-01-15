import torch.nn as nn


class Decoder(nn.Module):
    def __init__(self, n_channels, embedding_dim):
        super().__init__()
        self.model = nn.Sequential(
            nn.ConvTranspose2d(embedding_dim, n_channels,
                               kernel_size=(4, 4), stride=(1, 1)),
            nn.ReLU(inplace=True),
            nn.ConvTranspose2d(n_channels, n_channels,
                               kernel_size=(3, 3), stride=(2, 2)),
            nn.ReLU(inplace=True),
            nn.ConvTranspose2d(n_channels, n_channels,
                               kernel_size=(4, 4), stride=(2, 2)),
            nn.ReLU(inplace=True),
            nn.ConvTranspose2d(n_channels, n_channels,
                               kernel_size=(5, 5), stride=(1, 1)),
            nn.ReLU(inplace=True),
            nn.ConvTranspose2d(
                n_channels, 1, kernel_size=(5, 5), stride=(1, 1)),
        )

    def forward(self, x):
        return self.model(x)
