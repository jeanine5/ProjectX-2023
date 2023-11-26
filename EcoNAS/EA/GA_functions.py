"""
This contains the code for the multi and single objective optimization functions
"""

import random

from Architectures import *

import numpy as np

import torch
import torch.nn as nn
import torch.nn.functional as F
import torch.optim as optim
from torch.utils.data import DataLoader, TensorDataset


def set_non_dominated(population: list[NeuralArchitecture]):
    """

    :param population:
    :return:
    """
    for i in range(len(population)):
        non_dom_count = 1
        for j in range(len(population)):
            if i != j:
                dominates = is_pareto_dominant(population[i].objectives, population[j].objectives)
                if not dominates:
                    non_dom_count += 1
        population[i].nondominated_rank = non_dom_count


def is_pareto_dominant(p, q):
    """

    :param p:
    :param q:
    :return:
    """
    larger_or_equal = p >= q
    strict_dom = p > q
    if np.all(larger_or_equal) and np.any(strict_dom):
        return True
    return False


def fast_non_dominating_sort(population):
    """
    Calculates the Pareto fronts.
    :param population:
    :return:
    """

    fronts = []
    dom_count = np.zeros(len(population), dtype=int)
    dom_set = [set() for _ in range(len(population))]

    for i, p in enumerate(population):
        for j, q in enumerate(population):
            if i != j:
                if is_pareto_dominant(p, q):
                    dom_set[i].add(j)
                elif is_pareto_dominant(q, p):
                    dom_count[i] += 1

    ndf = []
    dl = []
    dc = dom_count.copy()
    ndr = np.zeros(len(population), dtype=int)

    while np.any(dc == 0):
        curr_front = np.where(dc == 0)[0]
        fronts.append(curr_front)

        for s in curr_front:
            dc[s] = -1
            dom_by_curr_set = dom_set[s]
            for dominated_by_current in dom_by_curr_set:
                dc[dominated_by_current] -= 1

    for i, front in enumerate(fronts):
        for s in front:
            ndf.append(np.array([s], dtype=int))
            dl.append(front)
            ndr[s] = i + 1

    return np.array(ndf), np.array(dl), dc, ndr


def crowding_distance_assignment(pop_by_obj, front: list):
    """

    :param pop_by_obj:
    :param fronts:
    :return:
    """
    num_objectives = pop_by_obj.shape[1]
    num_individuals = pop_by_obj.shape[0]

    normalized_fitnesses = np.zeros_like(pop_by_obj)  # each objective contribution have same mag to the crowding metric
    for objective_i in range(num_objectives):
        min_val = np.min(pop_by_obj[:, objective_i])
        max_val = np.max(pop_by_obj[:, objective_i])
        val_range = max_val - min_val
        normalized_fitnesses[:, objective_i] = (pop_by_obj[:, objective_i] - min_val) / val_range

    pop_by_obj = normalized_fitnesses
    crowding_metrics = np.zeros(num_individuals)

    for objective_i in range(num_objectives):

        sorted_front = sorted(front, key=lambda x: pop_by_obj[x, objective_i])

        crowding_metrics[sorted_front[0]] = np.inf
        crowding_metrics[sorted_front[-1]] = np.inf
        if len(sorted_front) > 2:
            for i in range(2, len(sorted_front) - 1):
                crowding_metrics[sorted_front[i]] += pop_by_obj[sorted_front[i + 1], objective_i] - pop_by_obj[
                    sorted_front[i - 1], objective_i]

    return crowding_metrics


def binary_tournament_selection(population, tournament_size=2):
    """

    :param population:
    :param tournament_size:
    :return:
    """

    # select s parent models from population P
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


def two_point_crossover(parent1: NeuralArchitecture, parent2: NeuralArchitecture):
    """

    :param parent1:
    :param parent2:
    :return:
    """
    # Randomly choose a crossover point based on the number of hidden layers
    crossover_points = sorted(random.sample(range(1, len(parent1.hidden_sizes)), 2))

    child_hidden_sizes = (
            parent1.hidden_sizes[:crossover_points[0]] + parent2.hidden_sizes[crossover_points[0]:crossover_points[1]] +
            parent1.hidden_sizes[crossover_points[1]:]
    )
    offspring = NeuralArchitecture(
        parent1.input_size,
        child_hidden_sizes,
        parent1.output_size,
        parent1.activation
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


def mutate_random_hidden_sizes(architecture):
    """

    :param architecture:
    :return:
    """
    mutated_architecture = architecture.clone()
    max_hidden_size = mutated_architecture.hidden_sizes
    for i in range(len(mutated_architecture.hidden_sizes)):
        mutated_architecture.hidden_sizes[i] = random.randint(1, max_hidden_size)
    return mutated_architecture


def mutate_add_remove_hidden_layer(architecture):
    """

    :param architecture:
    :return:
    """
    mutated_architecture = architecture.clone()
    if len(mutated_architecture.hidden_sizes) > 1:
        index_to_remove = random.randint(0, len(mutated_architecture.hidden_sizes) - 1)
        del mutated_architecture.hidden_sizes[index_to_remove]
    else:
        mutated_architecture.hidden_sizes.append(random.randint(1, mutated_architecture.max_hidden_size))
    return mutated_architecture


def generate_offspring(population, crossover_rate, mutation_rate):
    """

    :param population:
    :param crossover_rate:
    :param mutation_rate:
    :return:
    """
    offspring_pop = []

    for _ in range(len(population)):
        parent_1 = binary_tournament_selection(population)
        parent_2 = binary_tournament_selection(population)

        offspring = crossover(parent_1, parent_2, crossover_rate)

        mutated_offspring = mutate(offspring, mutation_rate)

        offspring_pop.append(mutated_offspring)

    return offspring_pop


def remove_lowest_scoring(population):
    """

    :param population:
    :return:
    """
