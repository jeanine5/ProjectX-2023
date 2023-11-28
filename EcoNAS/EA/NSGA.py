import random

from EcoNAS.EA.GA_functions import *

import numpy as np


class NSGA_II:
    def __init__(self, population_size, generations, crossover_factor, mutation_factor):
        self.population_size = population_size
        self.generations = generations
        self.crossover_factor = crossover_factor
        self.mutation_factor = mutation_factor

    def initial_population(self, max_hidden_layers, max_hidden_size):
        """

        :param max_hidden_layers:
        :param max_hidden_size:
        :return:
        """
        archs = []
        for _ in range(self.population_size):
            num_hidden_layers = random.randint(1, max_hidden_layers)
            hidden_sizes = [random.randint(1, max_hidden_size) for _ in range(num_hidden_layers)]
            arch = NeuralArchitecture(hidden_sizes)

            archs.append(arch)

        return archs

    def evolve(self, hidden_layers, hidden_size, optimizer, train_loader, test_loader):
        """

        :param hidden_layers:
        :param hidden_size:
        :param optimizer:
        :param train_loader:
        :param test_loader:
        :return:
        """

        # step 1: generate initial population
        archs = self.initial_population(hidden_layers, hidden_size)
        N = len(archs)

        # step 2 : evaluate the objective functions for each arch
        for a in archs:
            a.train(train_loader, optimizer)
            a.evaluate_accuracy(test_loader)

        # step 3: set the non-dominated ranks for the population and sort the architectures by rank
        set_non_dominated(archs)  # fitness vals
        archs.sort(key=lambda arch: arch.nondominated_rank)

        # step 4: create an offspring population Q0 of size N
        offspring_pop = generate_offspring(archs, self.crossover_factor, self.mutation_factor)
        set_non_dominated(offspring_pop)

        # step 5: start algorithm's counter
        for generation in range(self.generations):
            # step 6: combine parent and offspring population
            combined_population = archs + offspring_pop  # of size 2N

            population_by_objectives = np.array([[ind.objectives['accuracy'], ind.objectives['interpretability'],
                                                  ind.objectives['energy']] for ind in combined_population])

            # step 7:
            non_dom_fronts = fast_non_dominating_sort(population_by_objectives)

            non_domination_rank_dict = fronts_to_nondomination_rank(non_dom_fronts)

            # step 8: initialize new parent list and non-dominated front counter
            archs, i = [], 0

            # step 9: calculate crowding-distance in Fi until the parent population is filled
            while len(archs) + len(non_dom_fronts[i]) <= N:
                crowding_distance_assignment(population_by_objectives, non_dom_fronts[i])
                archs += non_dom_fronts[i]
                i += 1

            # step 8: sort front by crowding comparison operator
            sorted_indicies = nondominated_sort(non_domination_rank_dict, crowding_metrics)
            last_front = non_dom_fronts[i]
            #TO-DO

            # step 9: set new parent population
            archs = archs + non_dom_fronts[i][1: N - len(archs)]

            # step 10: generate new offspring population
            offspring_pop = generate_offspring(archs, self.crossover_factor, self.mutation_factor)
