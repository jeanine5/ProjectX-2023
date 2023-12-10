"""

"""

import pandas as pd
from sklearn.model_selection import train_test_split
from sklearn.linear_model import LinearRegression
from sklearn.metrics import mean_squared_error
import ast


class RegressionModelTrainer:
    def __init__(self, csv_filename):
        self.dataset = pd.read_csv(csv_filename)
        self.features = self.dataset[['hidden_layers', 'hidden_sizes_mean']]
        self.targets = self.dataset[['accuracy', 'interpretability', 'flops']]
        self.X_train, self.X_test, self.y_train, self.y_test = train_test_split(
            self.features, self.targets, test_size=0.2, random_state=42)
        self.regression_model = LinearRegression()

    def train_regression_model(self):
        """
        Train the linear regression model after preprocessing the data
        """
        self.regression_model.fit(self.X_train, self.y_train)

    def evaluate_model(self):
        """
        Evaluate the performance of the regression model on the test set
        """
        y_pred = self.regression_model.predict(self.X_test)
        mse = mean_squared_error(self.y_test, y_pred)
        print(f"Mean Squared Error: {mse}")

    def predict_performance(self, new_architecture):
        """
        Make predictions for the performance of a new architecture
        """
        new_architecture_df = pd.DataFrame(new_architecture, columns=['hidden_layers', 'hidden_sizes_mean'])
        predicted_performance = self.regression_model.predict(new_architecture_df)
        return predicted_performance


# Example usage:

regression_trainer = RegressionModelTrainer('../Benchmark/benchmark_results.csv')

# Train the regression model
regression_trainer.train_regression_model()

# Evaluate the model
regression_trainer.evaluate_model()

# Example of making predictions for a new architecture
new_architecture = {'hidden_layers': [3], 'hidden_sizes': [[64, 128]]}
predicted_performance = regression_trainer.predict_performance(new_architecture)
print("Predicted Performance for the New Architecture:")
print(predicted_performance)




