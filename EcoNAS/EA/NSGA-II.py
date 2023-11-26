import random

from Architectures import *
from GA_functions import *

import numpy as np


class NSGA_II:
    def __init__(self, population_size, generations, crossover_factor, mutation_factor):
        self.population_size = population_size
        self.generations = generations
        self.crossover_factor = crossover_factor
        self.mutation_factor = mutation_factor

    def initial_population(self, input_size, hidden_layers, hidden_size, output_size):
        """

        :param input_size:
        :param hidden_layers:
        :param hidden_size:
        :param output_size:
        :return:
        """
        archs = []
        for _ in range(self.population_size):
            num_hidden_layers = random.randint(1, hidden_layers)
            hidden_sizes = [random.randint(1, hidden_size) for _ in range(num_hidden_layers)]
            activation = nn.ReLU()
            arch = NeuralArchitecture(input_size, hidden_sizes, output_size, activation)

            archs.append(arch)

        return archs

    def evolve(self, input_size, hidden_layers, hidden_size, output_size, val_loader):
        """

        :param population:
        :param input_size:
        :param hidden_layers:
        :param hidden_size:
        :param output_size:
        :param val_loader:
        :return:
        """

        # step 1: generate initial population
        archs = self.initial_population(input_size, hidden_layers, hidden_size, output_size)

        # step 2 : evaluate the objective functions for each arch
        for a in archs:
            a.evaluate(val_loader)

        # step 3: set the non-dominated ranks for the population and sort the architectures by rank
        set_non_dominated(archs)  # fitness vals
        archs.sort(key=lambda arch: arch.nondominated_rank)

        # step 4: create an offspring population Q0 of size N
        offspring_pop = generate_offspring(archs, self.crossover_factor, self.mutation_factor)
        set_non_dominated(offspring_pop)

        # step 5: start algorithm's counter
        for generation in range(self.generations):
            # next step...
            # step 6: combine parent and offspring population
            combined_population = archs + offspring_pop

            population_by_objectives = np.array([[ind.objectives['accuracy'], ind.objectives['interpretability'],
                                                  ind.objectives['energy']] for ind in combined_population])

            # step 7:
            non_dom_fronts, dom_list, dom_count, non_dom_ranks = fast_non_dominating_sort(population_by_objectives)

            # step 8: initialize new parent list and non-dominated front counter
            new_parents, i = [], 0

            # step 9: calculate crowding-distance in Fi until the parent population is filled
            while len(new_parents) + len(non_dom_fronts[i]) <= len(combined_population) // 2:
                crowding_distance_assignment(population_by_objectives, non_dom_fronts[i])
                new_parents += non_dom_fronts[i]
                i += 1
