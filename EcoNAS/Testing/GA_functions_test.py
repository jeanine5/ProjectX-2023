"""
Test the functionality of the genetic algorithm functions
"""

import numpy as np
import matplotlib.pyplot as plt
import functools

from EcoNAS.EA.NSGA import *

transform = transforms.ToTensor()  # Here we simply make sure that the images are transformed into Tensors

# We load both the training set and the test set. The data we'll be directly downloaded and stored in the data folder
train_dataset = MNIST(root='../data',
                      train=True,
                      transform=transform,
                      download=True)

test_dataset = MNIST(root='../data',
                     train=False,
                     transform=transform)

batch_size = 128  # The batch-size is a hyperparameter that defines the number of images processed in parallel by the NN.
# Choosing a batch-size too small will result in a slow training
# Choosing a batch-size too large will result in out-of-memory error
train_loader = DataLoader(dataset=train_dataset, batch_size=batch_size, shuffle=True)

test_loader = DataLoader(dataset=test_dataset, batch_size=batch_size, shuffle=False)


nsga = NSGA_II(15, 3, 0.5, 0.5)

archs = nsga.initial_population(2, 200)


for a in archs:
    flops = a.flops_estimation()
    train_loss, train_acc = a.train(train_loader)
    inter = a.evaluate_interpretability(train_loader)
    loss, acc = a.evaluate_accuracy(test_loader)
    print(f'Interpretability: {inter}')
    print(f'Energy: {flops}')
    print(f'Loss: {loss}, Train Accuracy: {train_acc}, Test Accuracy: {acc}')
    print("")

offspring_pop = generate_offspring(archs, 0.5, 0.5, train_loader, test_loader)

combined_population = archs + offspring_pop

population_by_objectives = np.array([[ind.objectives['accuracy'], ind.objectives['interpretability'],
                                                  ind.objectives['energy']] for ind in combined_population])

print(f'obj: {population_by_objectives}')

non_dom_fronts = fast_non_dominating_sort(population_by_objectives)
print(f'Non-dominating fronts: {non_dom_fronts}')
print(f'best front: {np.size(non_dom_fronts[0])}')

non_domination_rank_dict = fronts_to_nondomination_rank(non_dom_fronts)
print(f'Non-dominating fronts rank dict: {non_domination_rank_dict}')

keys = []

for key in non_domination_rank_dict:
    keys.append(key)
print(f'values: {keys}')
