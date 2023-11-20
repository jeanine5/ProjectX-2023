"""
This contains the code for the Evolutionary Search algorithms used for our Artificial Neural Networks
"""
import torch
import torch.nn as nn
import torch.nn.functional as F
import torch.optim as optim
from torch.utils.data import DataLoader, TensorDataset


class NeuralNetwork(nn.Module):
    def __init__(self, input_size, hidden_sizes, output_size, activation):
        super(NeuralNetwork, self).__init__()
        self.input_size = input_size
        self.hidden_layer = nn.Linear(input_size, hidden_sizes)
        self.output_layer = nn.Linear(hidden_sizes, output_size)
        self.activation = activation

    def forward(self, x):
        batch_size = len(x)
        x = x.view(batch_size, self.input_size)
        hidden = self.hidden_layer(x)
        activation = F.relu(hidden)
        output = self.output_layer(activation)
        return output


class NeuralArchitecture:
    def __init__(self, input_size, hidden_sizes, output_size, activation, layers, hyperparameters):
        self.model = NeuralNetwork(input_size, hidden_sizes, output_size, activation)
        self.input_size = input_size
        self.hidden_sizes = hidden_sizes
        self.output_size = output_size
        self.layers = layers
        self.hyperparameters = hyperparameters
        self.activation = activation
        self.validation_score = None
        self.interpretability_score = 0.0
        self.energy_score = None

    @classmethod
    def random_initialization(cls, input_size, hidden_sizes, output_size):

        layers = torch.randint(1, 10, size=(1,)).item()
        activation = nn.ReLU()
        hyperparameters = torch.randn(layers * 3)

        return cls(input_size, hidden_sizes, output_size, activation, layers, hyperparameters)

    def mutate(self, architecture):
        """
        Mutates the given architecture with a certain probability
        :param architecture: The current artificial neural network to be mutated
        :return: mutated_architecture
        """

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

        self.validation_score = loss / n_samples

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


def initialize_cand_pool(population_size, input_size, hidden_sizes, output_size, activation):
    """

    :param population_size:
    :param input_size:
    :param hidden_sizes:
    :param output_size:
    :param activation:
    :return:
    """

    cand_pool = []

    for _ in range(population_size):
        architecture = NeuralArchitecture.random_initialization(input_size, hidden_sizes, output_size)
        cand_pool.append(architecture)

    return cand_pool


def mutate(architecture: NeuralArchitecture, loader, threshold):
    """

    :param architecture:
    :param loader:
    :param threshold:
    :return:
    """
    mutated_layers = torch.clamp(architecture.layers + torch.randint(-1, 2, size=(1,)), 1)
    mutated_hyperparameters = architecture.hyperparameters + torch.randn(mutated_layers.item() * 3)

    mutated_architecture = NeuralArchitecture(architecture.input_size, architecture.hidden_sizes,
                                              architecture.output_size,
                                              architecture.activation, mutated_layers, mutated_hyperparameters)

    mutated_architecture.calculate_interpretability(architecture, loader, threshold)

    if (mutated_architecture.validation_score > architecture.validation_score and
            mutated_architecture.interpretability_score > architecture.interpretability_score):

        return mutated_architecture
    else:
        return architecture


def replace_lowest_scoring(candidate_pool):
    ...
