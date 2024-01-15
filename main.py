import torch.nn as nn
import torch.optim as optim

from autoencoder.autoencoder import AutoEncoder
from utils import load_data


def train_autoencoder(train_X, epochs, batch_size):
    model = AutoEncoder(32, 8)
    optimizer = optim.Adam(model.parameters(), lr=1e-3)
    criterion = nn.MSELoss()
    for epoch in range(epochs):
        for input in train_X.split(batch_size):
            input = input.view((batch_size, 1, 28, 28))
            z = model.encode(input)
            output = model.decode(z)
            loss = criterion(output, input)
            optimizer.zero_grad()
            loss.backward()
            optimizer.step()
        print(f'epoch: {epoch}, loss: {loss}')
    return model


if __name__ == '__main__':
    train_X, train_y, test_X, test_y = load_data(use_cifar=False)

    EPOCHS = 50
    BATCH_SIZE = 100
    model = train_autoencoder(train_X, EPOCHS, BATCH_SIZE)
