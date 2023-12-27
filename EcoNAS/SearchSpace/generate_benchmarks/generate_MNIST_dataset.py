"""

"""

from torchvision.datasets import MNIST
import torchvision.transforms as transforms
from torch.utils.data import DataLoader

from EcoNAS.SearchSpace.search_spaces.image_search_spaces import ImageBenchmarkDataset

transform = transforms.ToTensor()

train_dataset = MNIST(root='../data', train=True, transform=transform, download=True)
test_dataset = MNIST(root='../data', train=False, transform=transform)

batch_size = 128
train_loader = DataLoader(dataset=train_dataset, batch_size=batch_size, shuffle=True)
test_loader = DataLoader(dataset=test_dataset, batch_size=batch_size, shuffle=False)

benchmark_dataset = ImageBenchmarkDataset()
architectures = benchmark_dataset.generate_architectures(max_hidden_layers=20, max_hidden_size=200, data_name='MNIST')
benchmark_dataset.evaluate_architectures(architectures, train_loader, test_loader)
benchmark_dataset.store_results_to_csv(filename='../Benchmarks/mnist_benchmark_results_2.csv')
