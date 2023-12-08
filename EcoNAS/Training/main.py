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
    nsga = NSGA_II(population_size=15, generations=3, crossover_factor=0.5, mutation_factor=0.5)
    architectures = nsga.evolve(hidden_layers=20, hidden_size=200, train_loader=train_loader, test_loader=test_loader)

    avg_loss = []

    min_acc = math.inf
    max_acc = -math.inf

    min_interpretability = math.inf
    max_interpretability = -math.inf

    min_flops = math.inf
    max_flops = -math.inf

    for arch in architectures:
        arch.train(train_loader, 12)
        acc_loss, acc, interpretability, flops = arch.evaluate_all_objectives(test_loader)
        avg_loss.append(acc_loss)

        if acc < min_acc:
            min_acc = acc
        if acc > max_acc:
            max_acc = acc

        if interpretability < min_interpretability:
            min_interpretability = interpretability
        if interpretability > max_interpretability:
            max_interpretability = interpretability

        if flops < min_flops:
            min_flops = flops
        if flops > max_flops:
            max_flops = flops

    print(f'Average Loss: {sum(avg_loss) / len(avg_loss)}')
    print(
        f'Average Accuracy: {calculate_average_accuracy(architectures)}, Min Accuracy: {min_acc}, Max Accuracy: {max_acc}')
    print(
        f'Average Interpretability: {calculate_average_interpretability(architectures)}, Min Interpretability: {min_interpretability}, Max Interpretability: {max_interpretability}')
    print(f'Average FLOPs: {calculate_average_flops(architectures)}, Min FLOPs: {min_flops}, Max FLOPs: {max_flops}')


# RUN THE ALGORITHM HERE
run_algorithm('MNIST')
