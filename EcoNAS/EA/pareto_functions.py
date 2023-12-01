"""

"""

import functools

from EcoNAS.EA.Architectures import *

import numpy as np


def get_corr_archs(front, architectures):
    """

    :param front:
    :param architectures:
    :return:
    """
    corr_archs = []
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
    first_two_objectives_dominate = np.all(p[:2] <= q[:2]) and np.any(p[:2] < q[:2])
    third_objective_minimization = p[2] <= q[2]

    return first_two_objectives_dominate and third_objective_minimization


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

    # Normalise each objectives, so they are in the range [0,1]
    # This is necessary, so each objective's contribution have the same magnitude to the crowding metric.
    normalized_fitnesses = np.zeros_like(pop_by_obj)
    for objective_i in range(num_objectives):
        min_val = np.min(pop_by_obj[:, objective_i])
        max_val = np.max(pop_by_obj[:, objective_i])
        val_range = max_val - min_val
        normalized_fitnesses[:, objective_i] = (pop_by_obj[:, objective_i] - min_val) / val_range

    fitnesses = normalized_fitnesses
    crowding_metrics = np.zeros(num_individuals)

    for objective_i in range(num_objectives):

        sorted_front = sorted(front, key=lambda x: fitnesses[x, objective_i])

        crowding_metrics[sorted_front[0]] = np.inf
        crowding_metrics[sorted_front[-1]] = np.inf
        if len(sorted_front) > 2:
            for i in range(1, len(sorted_front) - 1):
                crowding_metrics[sorted_front[i]] += fitnesses[sorted_front[i + 1], objective_i] - fitnesses[
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
