"""
This contains the code for the neural network architecture. The neural network is a fully connected neural network
with a variable number of hidden layers. The number of hidden layers and the number of hidden units per layer are
hyperparameters. The neural network is used for the evolutionary search algorithms. Note, ther are no convolutional
layers in this neural network.
"""

import numpy as np
import torch
import torch.nn as nn
import torch.nn.functional as F
import torch.optim as optim
from thop import profile


class NeuralNetwork(nn.Module):
    """
    A fully connected neural network with a variable number of hidden layers
    If testing with the MNIST dataset, the input size is 784 (28x28), and the output size is 10 (10 classes)
    If testing with the CIFAR-10 dataset, the input size is 3072 (32x32x3), and the output size is 10 (10 classes)
    If testing with the Statlog dataset, the input size is 21, and the output size is 2 (2 classes)
    """
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
    """
    A wrapper class for the NeuralNetwork class. This class is used for the evolutionary search algorithms
    """
    def __init__(self, hidden_sizes):
        self.model = NeuralNetwork(hidden_sizes)
        self.hidden_sizes = hidden_sizes
        self.activation = nn.ReLU()
        self.objectives = {
            'accuracy': 0.0,
            'interpretability': 0.0,
            'flops': 0.0
        }
        self.nondominated_rank = 0
        self.crowding_distance = 0.0

    def introspectability_metric(self, loader):
        """
        Metric for evaluating the interpretability of a neural network. This is based on the paper
        https://arxiv.org/pdf/2112.08645.pdf
        :param loader: Data loader for the dataset
        :return: Returns the introspectability of the neural network
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
        Estimates the number of FLOPs for the neural network
        :param input_size: The input size of the neural network. For MNIST, this is (1, 1, 28, 28), for CIFAR-10, this
        is (1, 3, 32, 32).
        :return: Returns the number of FLOPs
        """

        self.model.eval()

        # Create dummy input tensor
        dummy_input = torch.randn(input_size)

        # Use thop.profile to compute FLOPs
        flops, params = profile(self.model, inputs=(dummy_input,))

        self.objectives['flops'] = flops

        return flops

    def evaluate_interpretability(self, loader):
        """
        Evaluates the interpretability of the neural network
        :param loader: Data loader for the dataset
        :return: Returns the interpretability of the neural network
        """

        interpretability = self.introspectability_metric(loader)
        self.objectives['interpretability'] = interpretability

        return interpretability

    def accuracy(self, outputs, labels):
        """
        This function calculates the accuracy of the neural network. It is used for training and evaluating the
        neural network.
        :param outputs: The outputs of the neural network. Data type is torch.Tensor
        :param labels: The labels of the data. Data type is torch.Tensor
        :return: Returns the accuracy of the neural network
        """
        predictions = outputs.argmax(-1)
        correct = torch.sum(labels == predictions).item()
        return correct / len(labels)

    def evaluate_accuracy(self, loader):
        """
        Evaluates the accuracy of the neural network
        :param loader: Data loader for the dataset
        :return: Returns the accuracy of the neural network
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
        Evaluates all the objectives of the neural network at once
        :param loader: Data loader for the dataset
        :return: Returns the loss, accuracy, interpretability, and FLOPs of the neural network
        """

        acc_loss, acc = self.evaluate_accuracy(loader)
        interpretable = self.evaluate_interpretability(loader)
        flops = self.flops_estimation()

        return acc_loss, acc, interpretable, flops

    def train(self, loader, epochs):
        """
        Trains the neural network. Optimizer is Adam, learning rate is 1e-4, and loss function is CrossEntropyLoss
        (can change if needed)
        :param epochs: Number of epochs (rounds) to train the neural network for
        :param loader: Data loader for the dataset
        :return: Returns the loss and accuracy of the neural network
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
        return NeuralArchitecture(self.hidden_sizes.copy())