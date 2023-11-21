"""
This contains the code for the Evolutionary Search algorithms used for our Artificial Neural Networks
"""
import random

import torch
import torch.nn as nn
import torch.nn.functional as F
import torch.optim as optim
from torch.utils.data import DataLoader, TensorDataset


class NeuralNetwork(nn.Module):
    def __init__(self, input_size, hidden_sizes, output_size, activation):
        super(NeuralNetwork, self).__init__()
        self.input_size = input_size

        self.hidden_layers = nn.ModuleList()
        for hidden_size in hidden_sizes:
            hidden_layer = nn.Linear(input_size, hidden_size)
            self.hidden_layers.append(hidden_layer)
            self.input_size = hidden_size

        self.output_layer = nn.Linear(hidden_sizes, output_size)
        self.activation = activation

    def forward(self, x):
        batch_size = len(x)
        x = x.view(batch_size, self.input_size)

        for layer in self.hidden_layers:
            h = layer(x)
            x = F.relu(h)

        output = self.output_layer(x)

        return output


class NeuralArchitecture:
    def __init__(self, input_size, hidden_sizes, output_size, activation):
        self.model = NeuralNetwork(input_size, hidden_sizes, output_size, activation)
        self.input_size = input_size
        self.hidden_sizes = hidden_sizes
        self.output_size = output_size
        self.activation = activation
        self.validation_score = 0.0
        self.interpretability_score = 0.0
        self.energy_score = 0.0

    @classmethod
    def random_initialization(cls, input_size, max_hidden_layers, max_hidden_size, output_size):
        """

        :param input_size:
        :param max_hidden_layers:
        :param max_hidden_size:
        :param output_size:
        :return:
        """
        num_hidden_layers = random.randint(1, max_hidden_layers)
        hidden_sizes = [random.randint(1, max_hidden_size) for _ in range(num_hidden_layers)]
        activation = nn.ReLU()

        return cls(input_size, hidden_sizes, output_size, activation)

    def calculate_interpretability(self, arch_b, loader, threshold):
        """

        :param arch_b: another architecture to compare with
        :param loader: the sample to test interpretability with
        :param threshold: interpreability threshold (delta)
        :return:
        """

        arch_a_output = self.model(loader)
        arch_b_output = arch_b.model(loader)

        dist = torch.norm(arch_a_output, arch_b_output)

        if dist < threshold:
            self.interpretability_score = dist.item()

    def accuracy(self, outputs, labels):
        """

        :param outputs:
        :param labels:
        :return:
        """
        predictions = outputs.argmax(-1)
        correct = torch.sum(labels == predictions).item()
        return correct / len(labels)

    def evaluate(self, loader):
        """

        :param loader:
        :return:
        """
        # loss function
        criterion = nn.CrossEntropyLoss()

        self.model.eval()
        acc = 0
        loss = 0
        n_samples = 0

        with torch.no_grad():
            for inputs, targets in loader:
                outputs = self.model(inputs)
                loss += criterion(outputs, targets).item() * len(targets)

                acc += self.accuracy(outputs, targets) * len(targets)
                n_samples += len(targets)

        return loss / n_samples, acc / n_samples

    def train(self, loader, epochs=8, lr=0.001):
        """

        :param loader:
        :param epochs:
        :param lr:
        :return:
        """
        criterion = nn.CrossEntropyLoss()
        optimizer = optim.Adam(self.model.parameters(), lr=lr)

        train_loss = 0
        train_acc = 0
        n_samples = 0

        for epoch in range(epochs):
            self.model.train()
            for inputs, targets in loader:
                optimizer.zero_grad()
                outputs = self.model(inputs)
                loss = criterion(outputs, targets)
                loss.backward()
                optimizer.step()

                train_loss += loss.detach().item() * len(targets)
                train_acc += self.accuracy(outputs, targets) * len(targets)
                n_samples += len(targets)

        return train_loss / n_samples, train_acc / n_samples


def initialize_cand_pool(population_size, input_size, hidden_layers, hidden_size, output_size, val_loader):
    """

    :param population_size:
    :param input_size:
    :param hidden_layers:
    :param hidden_size:
    :param output_size:
    :param val_loader:
    :return:
    """
    cand_pool = []
    # init_arch = NeuralNetwork(input_size, hidden_sizes, output_size, activation=nn.ReLU())

    for _ in range(population_size):
        architecture = NeuralArchitecture.random_initialization(input_size, hidden_layers, hidden_size, output_size)
        _, accuracy = architecture.evaluate(val_loader)
        # get interpretability
        architecture.validation_score = accuracy

        cand_pool.append(architecture)

    return cand_pool


def tournament_selection(population, tournament_size):
    """

    :param population:
    :param tournament_size:
    :return:
    """

    # select s parent models from population P
    selected_parents = random.sample(population, tournament_size)

    # determine two parents w/ best fitness vals
    parent_a = max(selected_parents, key=lambda x: x.validation_score)
    selected_parents.remove(parent_a)
    parent_b = max(selected_parents, key=lambda x: x.validation_score)

    return parent_a, parent_b


def crossover(parent1: NeuralArchitecture, parent2: NeuralArchitecture):
    """

    :param parent1:
    :param parent2:
    :return:
    """
    # Randomly choose a crossover point based on the number of hidden layers
    crossover_point = random.randint(1, min(len(parent1.hidden_sizes), len(parent2.hidden_sizes)) - 1)
    offspring_hidden_sizes = (
            parent1.hidden_sizes[:crossover_point] + parent2.hidden_sizes[crossover_point:]
    )

    offspring = NeuralArchitecture(
        parent1.input_size,
        offspring_hidden_sizes,
        parent1.output_size,
        parent1.activation
    )

    return offspring


def remove_lowest_scoring(population):
    """

    :param population:
    :return:
    """
