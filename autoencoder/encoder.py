import torch.nn as nn


class Encoder(nn.Module):
    def __init__(self, n_channels, embedding_dim):
        super().__init__()
        self.model = nn.Sequential(
            nn.Conv2d(1, n_channels, kernel_size=(5, 5), stride=(1, 1)),
            nn.ReLU(inplace=True),
            nn.Conv2d(n_channels, n_channels,
                      kernel_size=(5, 5), stride=(1, 1)),
            nn.ReLU(inplace=True),
            nn.Conv2d(n_channels, n_channels,
                      kernel_size=(4, 4), stride=(2, 2)),
            nn.ReLU(inplace=True),
            nn.Conv2d(n_channels, n_channels,
                      kernel_size=(3, 3), stride=(2, 2)),
            nn.ReLU(inplace=True),
            nn.Conv2d(n_channels, embedding_dim,
                      kernel_size=(4, 4), stride=(1, 1)),
        )

    def forward(self, x):
        return self.model(x)
