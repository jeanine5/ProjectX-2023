from torchvision.datasets import CIFAR10, MNIST
import torchvision.transforms as transforms
from torch.utils.data import DataLoader
from EcoNAS.EA.NSGA import *


def load_data(dataset: str):
    """
    Loads the data for the given dataset
    :param dataset: The dataset to load
    :return: Returns the train and test datasets
    """
    transform = transforms.ToTensor()

    if dataset == 'MNIST':
        train_dataset = MNIST(root='../data', train=True, transform=transform, download=True)
        test_dataset = MNIST(root='../data', train=False, transform=transform)

    elif dataset == 'CIFAR10':
        train_dataset = CIFAR10(root='../data', train=True, transform=transform, download=True)
        test_dataset = CIFAR10(root='../data', train=False, transform=transform)

    batch_size = 128
    train_loader = DataLoader(dataset=train_dataset, batch_size=batch_size, shuffle=True)
    test_loader = DataLoader(dataset=test_dataset, batch_size=batch_size, shuffle=False)

    return train_loader, test_loader


def calculate_average_accuracy(archs: list):
    """
    Calculates the average accuracy of the given architectures
    :param archs: The list of architectures
    :return: Returns the average accuracy
    """
    avg_acc = 0
    for arch in archs:
        avg_acc += arch.objectives['accuracy']

    return avg_acc / len(archs)


def calculate_average_interpretability(archs: list):
    """
    Calculates the average interpretability of the given architectures
    :param archs: The list of architectures
    :return: Returns the average interpretability
    """
    avg_interpretability = 0
    for arch in archs:
        avg_interpretability += arch.objectives['interpretability']

    return avg_interpretability / len(archs)

def calculate_average_flops(archs: list):
    """
    Calculates the average FLOPs of the given architectures
    :param archs: The list of architectures
    :return: Returns the average FLOPs
    """
    avg_flops = 0
    for arch in archs:
        avg_flops += arch.objectives['flops']

    return avg_flops / len(archs)
