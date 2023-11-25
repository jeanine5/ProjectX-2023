"""
This contains the code for the multi and single objective optimization functions
"""

import random

from Architectures import *

import torch
import torch.nn as nn
import torch.nn.functional as F
import torch.optim as optim
from torch.utils.data import DataLoader, TensorDataset



def is_pareto_dominant(p, q):
    """

    :param p:
    :param q:
    :return:
    """

    return all(pi < qi for pi, qi in zip(p.objectives, q.objectives))


def fast_non_dominating_sort(population: list[NeuralArchitecture]):
    """

    :param population:
    :return:
    """

    fronts = []
    dom_count = {p: 0 for p in population}
    dom_set = {p: set() for p in population}

    for p in population:
        p.sp = set()
        p.np = 0
        for q in population:
            if is_pareto_dominant(p, q):
                p.sp.add(q)
            elif is_pareto_dominant(q, p):
                p.np += 1
                dom_set[q].add(p)

        if p.np == 0:
            p.rank = 1
            fronts.append([p])

    i = 1
    while fronts[i - 1]:
        next_front = []
        for p in fronts[i - 1]:
            for q in p.sp:
                dom_count[q] -= 1
                if dom_count[q] == 0:
                    q.rank = i + 1
                    next_front.append(q)

        i += 1
        fronts.append(next_front)

    return fronts


class NSGA_II:
    def __init__(self, population_size, generations, crossover_factor, mutation_factor):
        self.population_size = population_size
        self.generations = generations
        self. crossover_factor = crossover_factor
        self.mutation_factor = mutation_factor

    def evolve(self, population, input_size, hidden_layers, hidden_size, output_size, val_loader):
        """

        :param population:
        :param input_size:
        :param hidden_layers:
        :param hidden_size:
        :param output_size:
        :param val_loader:
        :return:
        """

        for generation in range(self.generations):
            archs = []
            # step 1: generate initial population
            for _ in range(self.population_size):
                num_hidden_layers = random.randint(1, hidden_layers)
                hidden_sizes = [random.randint(1, hidden_size) for _ in range(num_hidden_layers)]
                activation = nn.ReLU()
                arch = NeuralArchitecture(input_size, hidden_sizes, output_size, activation)

                arch.evaluate(val_loader)  # step 2: evalute the objective function values

                archs.append(arch)




            # crowd distance assignment


def tournament_selection(population, tournament_size):
    """

    :param population:
    :param tournament_size:
    :return:
    """

    # select s parent models from population P
    selected_parents = random.sample(population, tournament_size)

    # determine two parents w/ best fitness vals
    parent_a = max(selected_parents, key=lambda x: x.validation_score)
    selected_parents.remove(parent_a)
    parent_b = max(selected_parents, key=lambda x: x.validation_score)

    return parent_a, parent_b


def crossover(parent1: NeuralArchitecture, parent2: NeuralArchitecture):
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


def remove_lowest_scoring(population):
    """

    :param population:
    :return:
    """
