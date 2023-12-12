"""

"""
import random
import csv

from EcoNAS.EA.Architectures import *


class BenchmarkDataset:
    def __init__(self):
        self.results = []

    def generate_architectures(self, max_hidden_layers, max_hidden_size):
        archs = []

        for _ in range(25):
            num_hidden_layers = random.randint(3, max_hidden_layers)
            hidden_sizes = [random.randint(10, max_hidden_size) for _ in range(num_hidden_layers)]
            arch = NeuralArchitecture(hidden_sizes)

            archs.append(arch)

        return archs

    def evaluate_architectures(self, architectures, train_loader, test_loader, epochs=8):
        """
        Train and evaluate a list of architectures for minimal epochs
        """
        i = 1
        for arch in architectures:
            print(f'Arch {i}')
            arch.train(train_loader, epochs)
            acc_loss, acc, interpretable, flops = arch.evaluate_all_objectives(test_loader)

            # store results
            result = {
                'hidden_layers': len(arch.hidden_sizes),
                'hidden_sizes_mean': sum(arch.hidden_sizes) / len(arch.hidden_sizes),
                'accuracy': acc,
                'interpretability': interpretable,
                'flops': flops
            }
            self.results.append(result)
            i += 1

    def store_results_to_csv(self, filename):
        """
        Store the benchmark results in a CSV file
        """
        with open(filename, 'a', newline='') as csvfile:
            fieldnames = ['hidden_layers', 'hidden_sizes_mean', 'accuracy', 'interpretability', 'flops']
            writer = csv.DictWriter(csvfile, fieldnames=fieldnames)

            csvfile.seek(0, 2)
            is_empty = csvfile.tell() == 0

            if is_empty:
                writer.writeheader()

            for result in self.results:
                writer.writerow(result)


