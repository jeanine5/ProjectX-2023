"""

"""
from torchvision.datasets import MNIST
import torchvision.transforms as transforms

from torch.utils.data import DataLoader

from EcoNAS.EA.NSGA import *


transform = transforms.ToTensor()

train_dataset = MNIST(root='../data',
                      train=True,
                      transform=transform,
                      download=True)

test_dataset = MNIST(root='../data',
                     train=False,
                     transform=transform)

batch_size = 128

train_loader = DataLoader(dataset=train_dataset, batch_size=batch_size, shuffle=True)

test_loader = DataLoader(dataset=test_dataset, batch_size=batch_size, shuffle=False)


nsga = NSGA_II(25, 10, 0.5, 0.5)

architectures = nsga.evolve(20, 200, train_loader, test_loader)

for arch in architectures:
    train_loss, train_acc = arch.train(train_loader, 12)
    acc_loss, acc, interpretable, flops = arch.evaluate_all_objectives(test_loader)
    print(f'Architecture: {arch.hidden_sizes}')
    print(f'Train Loss: {train_loss}, Train Accuracy: {train_acc}, Test Loss: {acc_loss}, Test Accuracy: {acc}')
    print(f'Interpretability: {interpretable}')
    print(f'FLOPs: {flops}')
    print("")
