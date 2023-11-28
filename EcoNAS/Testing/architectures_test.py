"""
Test the functionality of the architectures
"""

from EcoNAS.EA.Architectures import *
from torchvision.datasets import MNIST
import torchvision.transforms as transforms
import matplotlib.pyplot as plt

# We can use transform to apply transformation to the dataset
transform = transforms.ToTensor()  # Here we simply make sure that the images are transformed into Tensors

# We load both the training set and the test set. The data we'll be directly downloaded and stored in the data folder
train_dataset = MNIST(root='../data',
                      train=True,
                      transform=transform,
                      download=True)

test_dataset = MNIST(root='../data',
                     train=False,
                     transform=transform)

train_data = iter(train_dataset)
image, label = next(train_data)
print(f"The training set has {len(train_dataset)} images of size {image.size()}")
print(f"The label of this image is: {label}")
plt.imshow(image.squeeze(), cmap="gray")
plt.axis('off')
plt.show()

batch_size = 128  # The batch-size is a hyperparameter that defines the number of images processed in parallel by the NN.
# Choosing a batch-size too small will result in a slow training
# Choosing a batch-size too large will result in out-of-memory error
train_loader = DataLoader(dataset=train_dataset, batch_size=batch_size, shuffle=True)

test_loader = DataLoader(dataset=test_dataset, batch_size=batch_size, shuffle=False)

model = NeuralArchitecture([125])

# We define the objective function we want to use
loss_function = nn.CrossEntropyLoss()  # For multiclass classification a popular choice for the objective function is the cross-entropy loss.

# We define the optimizer we want to use
lr = 1e-4  # The learning rate is a hyperparameter.
optimizer = optim.Adam(model.model.parameters(), lr=lr)
# optim.SGD(model.parameters(), lr=lr) # Here we're going to use stochastic gradient descent (SGD).

train_losses = []
test_losses = []
train_accuracy = []
test_accuracy = []

num_epochs = 12  # The number of epochs is a hyperparameter. The largest the number of epochs the longer we will train the neural network.

for epoch in range(num_epochs):
    train_loss, train_acc = model.train(train_loader, optimizer)
    test_loss, test_acc = model.evaluate_accuracy(test_loader)

    train_losses.append(train_loss)
    train_accuracy.append(train_acc)
    test_losses.append(test_loss)
    test_accuracy.append(test_acc)

    print(
        f'Epoch: {epoch + 1}/{num_epochs}, Train loss: {train_loss:.2e}, Train accuracy: {train_acc:.2%}, Test accuracy: {test_acc:.2%}')
