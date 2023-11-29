"""
This contains the code for the multi and single objective optimization functions
"""
import functools
import random

from EcoNAS.EA.Architectures import *

import numpy as np

import torch
import torch.nn as nn
import torch.nn.functional as F
import torch.optim as optim
from torch.utils.data import DataLoader, TensorDataset


def get_corr_archs(front, architectures):
    """

    :param front:
    :param architectures:
    :return:
    """
    corr_archs = []
    front_size = np.size(front)
    for idx in front:
        corr_archs.append(architectures[idx])

    return corr_archs


def crowded_comparison_operator(ind1, ind2):
    """

    :param ind1:
    :param ind2:
    :return:
    """
    if ind1.nondominated_rank < ind2.nondominated_rank:
        return True
    elif ind1.nondominated_rank == ind2.nondominated_rank and ind1.crowding_distance > ind2.crowding_distance:
        return True
    else:
        return False


def set_non_dominated(population: list[NeuralArchitecture]):
    """

    :param population:
    :return:
    """

    pbo = np.array([[ind.objectives['accuracy'], ind.objectives['interpretability'],
                     ind.objectives['energy']] for ind in population])

    for i in range(len(population)):
        non_dom_count = 1
        for j in range(len(population)):
            if i != j:
                dominates = is_pareto_dominant(pbo[i], pbo[j])
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
    return np.all(larger_or_equal) and np.any(strict_dom)


def fast_non_dominating_sort(population):
    """
    Calculates the Pareto fronts.
    :param population:
    :return:
    """

    domination_sets = []
    domination_counts = []
    for fitnesses_1 in population:
        current_dimination_set = set()
        domination_counts.append(0)
        for i, fitnesses_2 in enumerate(population):
            if is_pareto_dominant(fitnesses_1, fitnesses_2):
                current_dimination_set.add(i)
            elif is_pareto_dominant(fitnesses_2, fitnesses_1):
                domination_counts[-1] += 1

        domination_sets.append(current_dimination_set)

    domination_counts = np.array(domination_counts)
    fronts = []
    while True:
        current_front = np.where(domination_counts == 0)[0]
        if len(current_front) == 0:
            break
        fronts.append(current_front)

        for individual in current_front:
            domination_counts[
                individual] = -1
            dominated_by_current_set = domination_sets[individual]
            for dominated_by_current in dominated_by_current_set:
                domination_counts[dominated_by_current] -= 1

    return fronts


def crowding_distance_assignment(pop_by_obj, front: list):
    """

    :param pop_by_obj:
    :param front:
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


def fronts_to_nondomination_rank(fronts):
    """

    :param fronts:
    :return:
    """
    non_domination_rank_dict = {}
    for i, front in enumerate(fronts):
        for x in front:
            non_domination_rank_dict[x] = i
    return non_domination_rank_dict


def nondominated_sort(nondomination_rank_dict, crowding):
    """

    :param nondomination_rank_dict:
    :param crowding:
    :return:
    """
    num_individuals = len(crowding)
    indicies = list(range(num_individuals))

    def nondominated_compare(a, b):

        if nondomination_rank_dict[a] > nondomination_rank_dict[b]:  # domination rank, smaller better
            return -1
        elif nondomination_rank_dict[a] < nondomination_rank_dict[b]:
            return 1
        else:
            if crowding[a] < crowding[b]:  # crowding metrics, larger better
                return -1
            elif crowding[a] > crowding[b]:
                return 1
            else:
                return 0

    non_dominated_sorted_indices = sorted(indicies, key=functools.cmp_to_key(nondominated_compare),
                                          reverse=True)  # decreasing order, the best is the first
    return non_dominated_sorted_indices


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
        mutated_architecture.hidden_sizes[i] = random.randint(1, max_hidden_size)
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
        mutated_architecture.hidden_sizes.append(random.randint(1, max(mutated_architecture.hidden_sizes)))
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

        mutated_offspring.train(train_loader)
        mutated_offspring.evaluate_all_objectives(test_loader)

        offspring_pop.append(mutated_offspring)

    return offspring_pop


def remove_lowest_scoring(population):
    """

    :param population:
    :return:
    """
