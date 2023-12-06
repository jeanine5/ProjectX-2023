"""
This contains the code for the Evolutionary Search algorithms used for our Artificial Neural Networks
"""

import numpy as np

import torch
import torch.nn as nn
import torch.nn.functional as F
import torch.optim as optim
from thop import profile


from torchvision.datasets import MNIST
import torchvision.transforms as transforms

from torch.utils.data import DataLoader, TensorDataset


class NeuralNetwork(nn.Module):
    def __init__(self, hidden_sizes):
        super().__init__()

        input_size = 784
        self.hidden_layers = nn.ModuleList()
        for hidden_size in hidden_sizes:
            hidden_layer = nn.Linear(input_size, hidden_size)
            self.hidden_layers.append(hidden_layer)
            input_size = hidden_size

        self.output_layer = nn.Linear(input_size, 10)
        self.activation = nn.ReLU()

    def forward(self, x):
        batch_size = len(x)
        x = x.view(batch_size, 784)

        for layer in self.hidden_layers:
            h = layer(x)
            x = F.relu(h)

        output = self.output_layer(x)

        return output


class NeuralArchitecture:
    def __init__(self, hidden_sizes):
        self.model = NeuralNetwork(hidden_sizes)
        self.hidden_sizes = hidden_sizes
        self.activation = nn.ReLU()
        self.objectives = {
            'accuracy': 0.0,
            'interpretability': 0.0,
            'energy': 0.0
        }
        self.nondominated_rank = 0
        self.crowding_distance = 0.0

    def introspectability_metric(self, loader):
        """

        :param loader:
        :return:
        """

        self.model.eval()

        class_activations = {}

        with torch.no_grad():
            for inputs, targets in loader:
                outputs = self.model(inputs)

                # convert targets and outputs to numpy arrays
                targets_np = targets.numpy()
                outputs_np = F.relu(outputs).numpy()  # Assuming ReLU activation

                for i in range(len(targets_np)):
                    target_class = targets_np[i]

                    if target_class not in class_activations:
                        class_activations[target_class] = []

                    class_activations[target_class].append(outputs_np[i])

        # calculate mean activations for each class
        mean_activations = {cls: np.mean(np.array(acts), axis=0) for cls, acts in class_activations.items()}

        # cosine distance
        introspectability = 0.0
        num_classes = len(mean_activations)

        for cls1 in mean_activations:
            for cls2 in mean_activations:
                if cls1 < cls2:
                    # cosine distance between mean activations of two classes
                    cosine_distance = F.cosine_similarity(
                        torch.FloatTensor(mean_activations[cls1]),
                        torch.FloatTensor(mean_activations[cls2]),
                        dim=0
                    ).item()

                    introspectability += cosine_distance

        # normalize by the number of unique class pairs
        introspectability /= num_classes * (num_classes - 1) / 2

        return introspectability

    def flops_estimation(self, input_size=(1, 1, 28, 28)):
        """

        :param input_size:
        :return:
        """

        self.model.eval()

        # Create dummy input tensor
        dummy_input = torch.randn(input_size)

        # Use thop.profile to compute FLOPs
        flops, params = profile(self.model, inputs=(dummy_input,))

        self.objectives['energy'] = flops

        return flops

    def evaluate_interpretability(self, loader):
        """

        :param loader:
        :return:
        """

        interpretability = self.introspectability_metric(loader)
        self.objectives['interpretability'] = interpretability

        return interpretability

    def accuracy(self, outputs, labels):
        """

        :param outputs:
        :param labels:
        :return:
        """
        predictions = outputs.argmax(-1)
        correct = torch.sum(labels == predictions).item()
        return correct / len(labels)

    def evaluate_accuracy(self, loader):
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

        self.objectives['accuracy'] = acc / n_samples

        return loss / n_samples, acc / n_samples

    def evaluate_all_objectives(self, loader):
        """

        :param loader:
        :return:
        """

        acc_loss, acc = self.evaluate_accuracy(loader)
        interpretable = self.evaluate_interpretability(loader)
        flops = self.flops_estimation()

        return acc_loss, acc, interpretable, flops

    def train(self, loader, epochs):
        """

        :param epochs:
        :param loader:
        :return:
        """
        criterion = nn.CrossEntropyLoss()
        lr = 1e-4  # The learning rate is a hyperparameter
        optimizer = optim.Adam(self.model.parameters(), lr=lr)

        train_loss = 0
        train_acc = 0

        train_losses = []
        train_accuracies = []
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

                train_losses.append(train_loss / n_samples)
                train_accuracies.append(train_acc / n_samples)

        return train_losses[-1], train_accuracies[-1]

    def clone(self):
        """

        :return:
        """
        return NeuralArchitecture(self.hidden_sizes.copy())