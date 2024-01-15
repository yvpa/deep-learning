import torch
from torch import Tensor
from torchvision import datasets


def convert_to_one_hot_labels(X, y):

    tmp = X.new_zeros(y.size(0), y.max() + 1)
    tmp.scatter_(1, y.view(-1, 1), 1.0)
    return tmp


def load_data(use_cifar: bool = False, size: str = None, one_hot_labels: bool = False, normalize: bool = False, flatten: bool = True, data_dir: str = './datasets', seed=42) -> (Tensor, Tensor, Tensor, Tensor):
    torch.manual_seed(seed)
    if use_cifar:
        cifar_train_set = datasets.CIFAR10(
            data_dir + '/cifar10/', train=True, download=True)
        cifar_test_set = datasets.CIFAR10(
            data_dir + '/cifar10/', train=False, download=True)

        train_X = torch.from_numpy(cifar_train_set.data)
        train_X = train_X.transpose(3, 1).transpose(2, 3).float()
        train_y = torch.tensor(cifar_train_set.targets, dtype=torch.int64)

        test_X = torch.from_numpy(cifar_test_set.data).float()
        test_X = test_X.transpose(3, 1).transpose(2, 3).float()
        test_y = torch.tensor(cifar_test_set.targets, dtype=torch.int64)
    else:
        mnist_train_set = datasets.MNIST(
            data_dir + '/mnist/', train=True, download=True)
        mnist_test_set = datasets.MNIST(
            data_dir + '/mnist/', train=False, download=True)

        train_X = mnist_train_set.data.view(-1, 1, 28, 28).float()
        train_y = mnist_train_set.targets
        test_X = mnist_test_set.data.view(-1, 1, 28, 28).float()
        test_y = mnist_test_set.targets

    if flatten:
        train_X = train_X.clone().reshape(train_X.size(0), -1)
        test_X = test_X.clone().reshape(test_X.size(0), -1)

    if size == 'full':
        pass
    elif size == 'tiny':
        train_X = train_X.narrow(0, 0, 500)
        train_y = train_y.narrow(0, 0, 500)
        test_X = test_X.narrow(0, 0, 100)
        test_y = test_y.narrow(0, 0, 100)
    else:
        train_X = train_X.narrow(0, 0, 1000)
        train_y = train_y.narrow(0, 0, 1000)
        test_X = test_X.narrow(0, 0, 1000)
        test_y = test_y.narrow(0, 0, 1000)

    if one_hot_labels:
        train_y = convert_to_one_hot_labels(train_X, train_y)
        test_y = convert_to_one_hot_labels(test_X, test_y)

    if normalize:
        mu, std = train_X.mean(), train_X.std()
        train_X.sub_(mu).div_(std)
        test_X.sub_(mu).div_(std)

    return train_X, train_y, test_X, test_y


def mnist_to_pairs(n_sample: int, X, y):

    X = torch.functional.F.avg_pool2d(X, kernel_size=2)
    a = torch.randperm(X.size(0))
    a = a[:2 * n_sample].view(n_sample, 2)
    X = torch.cat((X[a[:, 0]], X[a[:, 1]]), 1)
    classes = y[a]
    y = (classes[:, 0] <= classes[:, 1]).long()
    return X, y, classes


def generate_pair_sets(n_sample: int, data_dir: str = './datasets'):

    train_set = datasets.MNIST(data_dir + '/mnist/', train=True, download=True)
    train_X = train_set.data.view(-1, 1, 28, 28).float()
    train_y = train_set.targets

    test_set = datasets.MNIST(data_dir + '/mnist/', train=False, download=True)
    test_X = test_set.data.view(-1, 1, 28, 28).float()
    test_y = test_set.targets

    return mnist_to_pairs(n_sample, train_X, train_y) + mnist_to_pairs(n_sample, test_X, test_y)
