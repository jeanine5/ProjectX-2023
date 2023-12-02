"""

"""

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


nsga = NSGA_II(15, 6, 0.75, 0.25)

architectures = nsga.evolve(20, 500, train_loader, test_loader)

for arch in architectures:
    train_loss, train_acc = arch.train(train_loader, 12)
    acc_loss, acc, interpretable, flops = arch.evaluate_all_objectives(test_loader)
    print(f'Architecture: {arch.hidden_sizes}')
    print(f'Train Loss: {train_loss}, Train Accuracy: {train_acc}, Test Loss: {acc_loss}, Test Accuracy: {acc}')
    print(f'Interpretability: {interpretable}')
    print(f'FLOPs: {flops}')
    print("")
