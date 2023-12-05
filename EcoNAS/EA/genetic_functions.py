"""
This contains the code for the multi and single objective optimization functions
"""
import random

from EcoNAS.EA.Architectures import *


def binary_tournament_selection(population, tournament_size=2):
    """

    :param population:
    :param tournament_size:
    :return:
    """

    # select 2 parent models from population P
    selected_parents = random.sample(population, tournament_size)

    # determine two parents w/ best fitness val
    return min(selected_parents, key=lambda arch: arch.nondominated_rank)


def crossover(parent1, parent2, crossover_rate):
    """

    :param parent1:
    :param parent2:
    :param crossover_rate:
    :return:
    """
    if random.uniform(0, 1) < crossover_rate:
        return two_point_crossover(parent1, parent2)
    else:
        return one_point_crossover(parent1, parent2)


def one_point_crossover(parent1: NeuralArchitecture, parent2: NeuralArchitecture):
    """

    :param parent1:
    :param parent2:
    :return:
    """
    # Randomly choose a crossover point based on the number of hidden layers
    crossover_point = random.randint(1, min(len(parent1.hidden_sizes), len(parent2.hidden_sizes)))
    offspring_hidden_sizes = (
            parent1.hidden_sizes[:crossover_point] + parent2.hidden_sizes[crossover_point:]
    )

    offspring = NeuralArchitecture(
        offspring_hidden_sizes
    )

    return offspring


def two_point_crossover(parent1: NeuralArchitecture, parent2: NeuralArchitecture):
    """

    :param parent1:
    :param parent2:
    :return:
    """
    len_parent1 = len(parent1.hidden_sizes)
    len_parent2 = len(parent2.hidden_sizes)

    if len_parent1 <= 2 or len_parent2 <= 2:
        return parent1 if len_parent1 > 0 else parent2

    # randomly choose a crossover point based on the number of hidden layers
    crossover_points = sorted(random.sample(range(1, min(len_parent1, len_parent2)), 2))

    child_hidden_sizes = (
            parent1.hidden_sizes[:crossover_points[0]] +
            parent2.hidden_sizes[crossover_points[0]:crossover_points[1]] +
            parent1.hidden_sizes[crossover_points[1]:]
    )

    offspring = NeuralArchitecture(
        child_hidden_sizes
    )

    return offspring


def mutate(offspring: NeuralArchitecture, mutation_factor):
    """

    :param offspring:
    :param mutation_factor:
    :return:
    """
    if random.uniform(0, 1) < mutation_factor:
        mutated_offspring = mutate_add_remove_hidden_layer(offspring)
    else:
        mutated_offspring = mutate_random_hidden_sizes(offspring)

    return mutated_offspring


def mutate_random_hidden_sizes(architecture: NeuralArchitecture):
    """

    :param architecture:
    :return:
    """
    mutated_architecture = architecture.clone()
    max_hidden_size = max(mutated_architecture.hidden_sizes)
    for i in range(len(mutated_architecture.hidden_sizes)):
        mutated_architecture.hidden_sizes[i] = random.randint(10, max_hidden_size)
    return mutated_architecture


def mutate_add_remove_hidden_layer(architecture: NeuralArchitecture):
    """

    :param architecture:
    :return:
    """
    mutated_architecture = architecture.clone()
    if len(mutated_architecture.hidden_sizes) > 1:
        index_to_remove = random.randint(0, len(mutated_architecture.hidden_sizes) - 1)
        del mutated_architecture.hidden_sizes[index_to_remove]
    else:
        mutated_architecture.hidden_sizes.append(random.randint(10, max(mutated_architecture.hidden_sizes)))
    return mutated_architecture


def generate_offspring(population, crossover_rate, mutation_rate, train_loader, test_loader):
    """

    :param population:
    :param crossover_rate:
    :param mutation_rate::
    :param train_loader:
    :param test_loader:
    :return:
    """
    offspring_pop = []

    for _ in range(len(population)):
        parent_1 = binary_tournament_selection(population)
        parent_2 = binary_tournament_selection(population)

        offspring = crossover(parent_1, parent_2, crossover_rate)

        mutated_offspring = mutate(offspring, mutation_rate)

        mutated_offspring.train(train_loader, 8)
        mutated_offspring.evaluate_all_objectives(test_loader)

        offspring_pop.append(mutated_offspring)

    return offspring_pop


def remove_lowest_scoring(population):
    """

    :param population:
    :return:
    """