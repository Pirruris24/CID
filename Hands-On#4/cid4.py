import matplotlib.pyplot as plt
from math import sqrt
import random

class DataPoint:
    def __init__(self, height, weight, size):
        self.height = height
        self.weight = weight
        self.size = size

class KNN:
    def __init__(self, dataset):
        self.dataset = dataset

    def euclidean_distance(self, p1, p2):
        return sqrt((p1.height - p2.height) ** 2 + (p1.weight - p2.weight) ** 2)

    def find_nearest_neighbors(self, new_point, k):
        distances = [(point, self.euclidean_distance(point, new_point)) for point in self.dataset]
        sorted_distances = sorted(distances, key=lambda x: x[1])
        return [point[0] for point in sorted_distances[:k]]

    def predict_size(self, neighbors):
        sizes = [neighbor.size for neighbor in neighbors]
        size_counts = {size: sizes.count(size) for size in set(sizes)}
        return max(size_counts, key=size_counts.get)

class Plotter:
    def __init__(self, dataset):
        self.dataset = dataset

    def plot_dataset(self, new_data_point, predicted_size, perceptron=None):
        heights = [point.height for point in self.dataset]
        weights = [point.weight for point in self.dataset]
        sizes = [point.size for point in self.dataset]

        heights_M = [heights[i] for i in range(len(heights)) if sizes[i] == 'M']
        weights_M = [weights[i] for i in range(len(weights)) if sizes[i] == 'M']
        heights_L = [heights[i] for i in range(len(heights)) if sizes[i] == 'L']
        weights_L = [weights[i] for i in range(len(weights)) if sizes[i] == 'L']

        plt.scatter(heights_M, weights_M, c='blue', label='M')
        plt.scatter(heights_L, weights_L, c='red', label='L')

        plt.scatter(new_data_point.height, new_data_point.weight, c='green', marker='x', label=f'Predicted ({predicted_size})')

        plt.xlabel('Height (cm)')
        plt.ylabel('Weight (kg)')
        plt.title('T-Shirt Size Prediction using kNN and Perceptron')
        plt.legend()

        # Plot the decision boundary created by the perceptron
        if perceptron is not None:
            w1, w2 = perceptron.weights
            b = perceptron.bias
            x_values = [min(heights), max(heights)]
            y_values = [(-w1 / w2) * x - (b / w2) for x in x_values]
            plt.plot(x_values, y_values, linestyle='--', color='black', label='Perceptron Decision Boundary')

        plt.show()


class Perceptron:
    def __init__(self, dataset):
        self.dataset = dataset
        self.weights = [random.random() for _ in range(2)]  # Initialize random weights
        self.bias = random.random()  # Initialize random bias

    def predict(self, point):
        activation = self.bias + self.weights[0] * point.height + self.weights[1] * point.weight
        return 1 if activation >= 0 else -1

    def train(self, epochs):
        for _ in range(epochs):
            for point in self.dataset:
                target = 1 if point.size == 'M' else -1
                prediction = self.predict(point)
                if target != prediction:
                    error = target - prediction
                    self.weights[0] += error * point.height
                    self.weights[1] += error * point.weight
                    self.bias += error

    def accuracy(self, test_dataset):
        correct = 0
        for point in test_dataset:
            if self.predict(point) == (1 if point.size == 'M' else -1):
                correct += 1
        return correct / len(test_dataset) * 100

# Define the dataset
dataset = [
    DataPoint(158, 58, 'M'),
    DataPoint(158, 59, 'M'),
    DataPoint(158, 63, 'M'),
    DataPoint(160, 59, 'M'),
    DataPoint(160, 60, 'M'),
    DataPoint(163, 60, 'M'),
    DataPoint(163, 61, 'M'),
    DataPoint(160, 64, 'L'),
    DataPoint(163, 64, 'L'),
    DataPoint(165, 61, 'L'),
    DataPoint(165, 62, 'L'),
    DataPoint(165, 65, 'L'),
    DataPoint(168, 62, 'L'),
    DataPoint(168, 63, 'L'),
    DataPoint(168, 66, 'L'),
    DataPoint(170, 63, 'L'),
    DataPoint(170, 64, 'L'),
    DataPoint(170, 68, 'L')
]

# Initialize KNN, Plotter, and Perceptron objects
knn = KNN(dataset)
plotter = Plotter(dataset)
perceptron = Perceptron(dataset)

# Train the perceptron
perceptron.train(epochs=100)

# Take user input for prediction values
new_height = float(input("Enter the height in cm: "))
new_weight = float(input("Enter the weight in kg: "))

# New data point to predict
new_data_point = DataPoint(new_height, new_weight, '')

# Predict the size using kNN
nearest_neighbors = knn.find_nearest_neighbors(new_data_point, k=3)
predicted_size_knn = knn.predict_size(nearest_neighbors)

# Predict the size using Perceptron
predicted_size_perceptron = 'M' if perceptron.predict(new_data_point) == 1 else 'L'

print(f"Predicted size using kNN: {predicted_size_knn}")
print(f"Predicted size using Perceptron: {predicted_size_perceptron}")

# Plot the dataset with the predicted point
plotter.plot_dataset(new_data_point, predicted_size_knn)
