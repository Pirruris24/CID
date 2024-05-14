import numpy as np

class Dataset:
    def __init__(self, X, y):
        self.X = np.array(X)
        self.y = np.array(y)
        self.n = len(X)

class RegressionCalculator:
    def __init__(self, dataset):
        self.dataset = dataset

    def calculate_inverse(self, matrix):
        return np.linalg.inv(matrix)

    def linear_regression(self):
        X_design_linear = np.column_stack([np.ones_like(self.dataset.X), self.dataset.X])  # Design matrix for linear regression
        print(X_design_linear)
        X_design_T_X_linear = np.dot(X_design_linear.T, X_design_linear)
        print(X_design_T_X_linear)
        inverse_X_design_T_X_linear = self.calculate_inverse(X_design_T_X_linear)
        print(inverse_X_design_T_X_linear)
        X_design_T_y_linear = np.dot(X_design_linear.T, self.dataset.y)
        result_linear = np.dot(inverse_X_design_T_X_linear, X_design_T_y_linear)
        return result_linear

    def quadratic_regression(self):
        X_design_quadratic = np.column_stack([np.ones_like(self.dataset.X), self.dataset.X, self.dataset.X**2])  # Design matrix for quadratic regression
        X_design_T_X_quadratic = np.dot(X_design_quadratic.T, X_design_quadratic)
        inverse_X_design_T_X_quadratic = self.calculate_inverse(X_design_T_X_quadratic)
        X_design_T_y_quadratic = np.dot(X_design_quadratic.T, self.dataset.y)
        result_quadratic = np.dot(inverse_X_design_T_X_quadratic, X_design_T_y_quadratic)
        return result_quadratic

    def cubic_regression(self):
        X_design_cubic = np.column_stack([np.ones_like(self.dataset.X), self.dataset.X, self.dataset.X**2, self.dataset.X**3])  # Design matrix for cubic regression
        coefficients_cubic, _, _, _ = np.linalg.lstsq(X_design_cubic, self.dataset.y, rcond=None)
        return coefficients_cubic

# DATASET
X = [108, 115, 106, 97, 95, 91, 97, 83, 83, 78, 54, 67, 56, 53, 61, 115, 81, 78, 30, 45, 99, 32, 25, 28, 90, 89]
y = [95, 96, 95, 97, 93, 94, 95, 93, 92, 86, 73, 80, 65, 69, 77, 96, 87, 89, 60, 63, 95, 61, 55, 56, 94, 93]
dataset = Dataset(X, y)

# Create RegressionCalculator instance
regression_calculator = RegressionCalculator(dataset)

# Linear Regression
result_linear = regression_calculator.linear_regression()
B0_linear, B1_linear = result_linear[0], result_linear[1]
print("Linear Regression Coefficients:")
print("B0 =", B0_linear)
print("B1 =", B1_linear)
print(f"Expression: y = {B0_linear} + {B1_linear} * x")

# Quadratic Regression
result_quadratic = regression_calculator.quadratic_regression()
B0_quadratic, B1_quadratic, B2_quadratic = result_quadratic[0], result_quadratic[1], result_quadratic[2]
print("\nQuadratic Regression Coefficients:")
print("B0 =", B0_quadratic)
print("B1 =", B1_quadratic)
print("B2 =", B2_quadratic)
print(f"Expression: y = {B0_quadratic} + {B1_quadratic} * x + {B2_quadratic} * x^2")


# Cubic Regression
coefficients_cubic = regression_calculator.cubic_regression()
B0_cubic, B1_cubic, B2_cubic, B3_cubic = coefficients_cubic[0], coefficients_cubic[1], coefficients_cubic[2], coefficients_cubic[3]
print("\nCubic Regression Coefficients:")
print("B0 =", B0_cubic)
print("B1 =", B1_cubic)
print("B2 =", B2_cubic)
print("B3 =", B3_cubic)
print(f"Expression: y = {B0_cubic} + {B1_cubic} * x + {B2_cubic} * x^2 + {B3_cubic} * x^3")



# Make predictions from main program
while True:
    prediction_input = input("\nEnter a value of x to predict y (type 'done' to exit): ")
    if prediction_input.lower() == 'done':
        break
    try:
        x_value = float(prediction_input)
        # Predict y based on the linear regression equation
        y_predicted_linear = B0_linear + B1_linear * x_value
        print(f"Predicted y for x = {x_value} (Linear Regression): {y_predicted_linear}")

        # Predict y based on the quadratic regression equation
        y_predicted_quadratic = B0_quadratic + B1_quadratic * x_value + B2_quadratic * x_value**2
        print(f"Predicted y for x = {x_value} (Quadratic Regression): {y_predicted_quadratic}")

        # Predict y based on the cubic regression equation
        y_predicted_cubic = B0_cubic + B1_cubic * x_value + B2_cubic * x_value**2 + B3_cubic * x_value**3
        print(f"Predicted y for x = {x_value} (Cubic Regression): {y_predicted_cubic}")

    except ValueError:
        print("Invalid input. Please enter a valid number or 'done' to exit.")


