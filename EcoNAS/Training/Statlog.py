"""

"""

import torchvision.transforms as transforms
import pandas as pd
from torch.utils.data import DataLoader, random_split
from EcoNAS.EA.NSGA import *


# Define the column names based on your dataset
column_names = ['Age', 'Sex', 'ChestPainType', 'RestingBloodPressure', 'SerumCholesterol',
                'FastingBloodSugar', 'RestingECG', 'MaxHeartRate', 'ExerciseInducedAngina',
                'STDepression', 'SlopeOfPeakExerciseSTSegment', 'NumMajorVessels', 'ThalliumTestResult',
                'ClassLabel']

# Specify the full path to your dataset file
dataset_path = 'EcoNAS/data/statlog_heart/heart.dat'

# Load the dataset into a Pandas DataFrame
df = pd.read_csv(dataset_path, header=None, names=column_names, delimiter=' ')

# Split the dataset into features and target
X = df.iloc[:, :-1]  # Features
y = df.iloc[:, -1]   # Target

train_ratio = 0.8
dataset_size = len(df)
train_size = int(train_ratio * dataset_size)
test_size = dataset_size - train_size

train_dataset, test_dataset = random_split(df, [train_size, test_size])

batch_size = 128
train_loader = DataLoader(train_dataset, batch_size=batch_size, shuffle=True)
test_loader = DataLoader(test_dataset, batch_size=batch_size, shuffle=False)

nsga = NSGA_II(25, 8, 0.5, 0.5)

architectures = nsga.evolve(10, 100, train_loader, test_loader)

for arch in architectures:
    train_loss, train_acc = arch.train(train_loader, 15)
    acc_loss, acc, interpretable, flops = arch.evaluate_all_objectives(test_loader)
    print(f'Architecture: {arch.hidden_sizes}')
    print(f'Train Loss: {train_loss}, Train Accuracy: {train_acc}, Test Loss: {acc_loss}, Test Accuracy: {acc}')
    print(f'Interpretability: {interpretable}')
    print(f'FLOPs: {flops}')
    print("")