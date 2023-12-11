"""
The main function of the EcoNAS algorithm. This is the function that is called when the user wants to run the
algorithm.
"""
import math
from load_and_calculate import *


def run_algorithm(dataset: str):
    """

    :param dataset:
    :return:
    """
    train_loader, test_loader = load_data(dataset)

    # modify this before you run the algorithm below
    nsga = NSGA_II(population_size=200, generations=18, crossover_factor=0.75, mutation_factor=0.25)
    architectures = nsga.evolve(hidden_layers=10, hidden_size=128, train_loader=train_loader, test_loader=test_loader)

    avg_loss = []

    min_acc = math.inf
    max_acc = -math.inf

    min_interpretability = math.inf
    max_interpretability = -math.inf

    min_flops = math.inf
    max_flops = -math.inf

    for arch in architectures:
        acc, interpretability, flops = arch.objectives['accuracy'], arch.objectives['interpretability'], arch.objectives['flops']

        min_acc = min(min_acc, acc)
        max_acc = max(max_acc, acc)

        min_interpretability = min(min_interpretability, interpretability)
        max_interpretability = max(max_interpretability, interpretability)

        min_flops = min(min_flops, flops)
        max_flops = max(max_flops, flops)

    print(
        f'Average Accuracy: {calculate_average_accuracy(architectures)}, Min Accuracy: {min_acc}, Max Accuracy: {max_acc}')
    print(
        f'Average Interpretability: {calculate_average_interpretability(architectures)}, Min Interpretability: {min_interpretability}, Max Interpretability: {max_interpretability}')
    print(f'Average FLOPs: {calculate_average_flops(architectures)}, Min FLOPs: {min_flops}, Max FLOPs: {max_flops}')


# RUN THE ALGORITHM HERE
run_algorithm('MNIST')
