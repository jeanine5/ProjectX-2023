"""
This contains the code for the Evolutionary Search algorithms used for our Artificial Neural Networks
"""
import math
import random

import numpy as np

import torch
import torch.nn as nn
import torch.nn.functional as F
import torch.optim as optim
import pymoo
from pymoo.optimize import minimize

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
        self.objectives = {
            'accuracy': 0.0,
            'interpretability': 0.0,
            'energy': 0.0
        }
        self.rank = None
        self.sp = set()


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

    def introspectability_metric(self, loader):
        """

        :param loader:
        :return:
        """

        # Set the model to evaluation mode
        self.model.eval()

        # Dictionary to store activations for each class
        class_activations = {}

        # Loop through the data loader
        with torch.no_grad():
            for inputs, targets in loader:
                outputs = self.model(inputs)

                # Convert targets and outputs to numpy arrays
                targets_np = targets.numpy()
                outputs_np = F.relu(outputs).numpy()  # Assuming ReLU activation

                # Loop through each sample
                for i in range(len(targets_np)):
                    target_class = targets_np[i]

                    # If the class is not in the dictionary, create an empty list
                    if target_class not in class_activations:
                        class_activations[target_class] = []

                    # Append the activations to the corresponding class list
                    class_activations[target_class].append(outputs_np[i])

        # Calculate mean activations for each class
        mean_activations = {cls: np.mean(np.array(acts), axis=0) for cls, acts in class_activations.items()}

        # Calculate introspectability using cosine distance
        introspectability = 0.0
        num_classes = len(mean_activations)

        for cls1 in mean_activations:
            for cls2 in mean_activations:
                if cls1 < cls2:
                    # Cosine distance between mean activations of two classes
                    cosine_distance = F.cosine_similarity(
                        torch.FloatTensor(mean_activations[cls1]),
                        torch.FloatTensor(mean_activations[cls2]),
                        dim=0
                    ).item()

                    introspectability += cosine_distance

        # Normalize by the number of unique class pairs
        introspectability /= num_classes * (num_classes - 1) / 2

        return introspectability


    def energy_metric(self):
        ...

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

        interpretability = self.introspectability_metric(loader)
        energy = ...

        self.objectives['accuracy'] = acc / n_samples
        self.objectives['interpretability'] = interpretability
        self.objectives['energy'] = energy

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
