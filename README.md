# Deep Learning

Requirements:
    
    Python 3.7+ with tensorflow backend

    Default packages:
    sys
    pickle
    processing
	multithreading
    time

    Imports:
    pandas
    numpy    
    matplotlib
    keras
    tensorflow

Preparing environment:

    pip install pandas numpy matplotlib keras networkx tensorflow scikit-learn

Dataset:

    MNIST dataset can be downloaded manually and put into the datasets/MNIST folder.
    Link: http://yann.lecun.com/exdb/mnist/
    CIFAR dataset can be downloaded 

## Configurations

### Install CUDA support on WSL2

wget https://developer.download.nvidia.com/compute/cuda/repos/wsl-ubuntu/x86_64/cuda-wsl-ubuntu.pin
sudo mv cuda-wsl-ubuntu.pin /etc/apt/preferences.d/cuda-repository-pin-600
wget https://developer.download.nvidia.com/compute/cuda/12.0.0/local_installers/cuda-repo-wsl-ubuntu-12-0-local_12.0.0-1_amd64.deb
sudo dpkg -i cuda-repo-wsl-ubuntu-12-0-local_12.0.0-1_amd64.deb
sudo cp /var/cuda-repo-wsl-ubuntu-12-0-local/cuda-*-keyring.gpg /usr/share/keyrings/
sudo apt-get update
sudo apt-get -y install cuda
