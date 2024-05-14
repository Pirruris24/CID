class Dataset:
    def __init__(self, X, y):
        self.X = X
        self.y = y
        self.n = len(X)

class Calculations:
    def __init__(self, dataset):
        self.dataset = dataset

    def calculate_linear_regression(self):
        X_design = [[1, x] for x in self.dataset.X]
        X_design_T = [[row[i] for row in X_design] for i in range(2)]
        X_design_T_X_design = [[sum(X_row[i] * X_row[j] for X_row in X_design) for j in range(2)] for i in range(2)]

        # Calculate the inverse of X_design_T_X_design
        inverse_X_design_T_X_design = self.calculate_inverse(X_design_T_X_design)
        
        X_design_T_y = [sum(a*b for a, b in zip(row_X_T, self.dataset.y)) for row_X_T in X_design_T]
        result = [sum(a*b for a, b in zip(row_inv, X_design_T_y)) for row_inv in inverse_X_design_T_X_design]
        B0 = result[0]
        B1 = result[1]
        return B0, B1

    def calculate_quadratic_regression(self):
        X_design = [[1, x, x**2] for x in self.dataset.X]
        X_design_T = [[row[i] for row in X_design] for i in range(3)]
        X_design_T_X_design = [[sum(X_row[i] * X_row[j] for X_row in X_design) for j in range(3)] for i in range(3)]

        # Calculate the inverse of X_design_T_X_design
        inverse_X_design_T_X_design = self.calculate_inverse(X_design_T_X_design)
        
        X_design_T_y = [sum(a*b for a, b in zip(row_X_T, self.dataset.y)) for row_X_T in X_design_T]
        result = [sum(a*b for a, b in zip(row_inv, X_design_T_y)) for row_inv in inverse_X_design_T_X_design]
        B0 = result[0]
        B1 = result[1]
        B2 = result[2]
        return B0, B1, B2

    def calculate_cubic_regression(self):
        X_design = [[1, x, x**2, x**3] for x in self.dataset.X]
        X_design_T = [[row[i] for row in X_design] for i in range(4)]
        X_design_T_X_design = [[sum(X_row[i] * X_row[j] for X_row in X_design) for j in range(4)] for i in range(4)]

        # Calculate the inverse of X_design_T_X_design
        inverse_X_design_T_X_design = self.calculate_inverse(X_design_T_X_design)
        
        X_design_T_y = [sum(a*b for a, b in zip(row_X_T, self.dataset.y)) for row_X_T in X_design_T]
        result = [sum(a*b for a, b in zip(row_inv, X_design_T_y)) for row_inv in inverse_X_design_T_X_design]
        B0 = result[0]
        B1 = result[1]
        B2 = result[2]
        B3 = result[3]
        return B0, B1, B2, B3

    def calculate_inverse(self, matrix):
        n = len(matrix)
        identity = [[1 if i == j else 0 for j in range(n)] for i in range(n)]
        augmented_matrix = [row + identity[i] for i, row in enumerate(matrix)]

        # Perform Gauss-Jordan elimination
        for i in range(n):
            pivot = augmented_matrix[i][i]
            for j in range(i + 1, n):
                ratio = augmented_matrix[j][i] / pivot
                augmented_matrix[j] = [a - ratio * b for a, b in zip(augmented_matrix[j], augmented_matrix[i])]

        # Backward substitution
        for i in range(n - 1, -1, -1):
            pivot = augmented_matrix[i][i]
            augmented_matrix[i] = [a / pivot for a in augmented_matrix[i]]
            for j in range(i - 1, -1, -1):
                ratio = augmented_matrix[j][i]
                augmented_matrix[j] = [a - ratio * b for a, b in zip(augmented_matrix[j], augmented_matrix[i])]

        # Extract the inverse matrix
        inverse_matrix = [row[n:] for row in augmented_matrix]

        return inverse_matrix

    def predict_linear_regression(self, x_value, B0, B1):
        return B0 + B1 * x_value

    def predict_quadratic_regression(self, x_value, B0, B1, B2):
        return B0 + B1 * x_value + B2 * x_value**2

    def predict_cubic_regression(self, x_value, B0, B1, B2, B3):
        return B0 + B1 * x_value + B2 * x_value**2 + B3 * x_value**3

    def calculate_correlation_and_determination(self, B0, B1, B2=None, B3=None):
        if B2 is None:
            y_predicted = [self.predict_linear_regression(x, B0, B1) for x in self.dataset.X]
        elif B3 is None:
            y_predicted = [self.predict_quadratic_regression(x, B0, B1, B2) for x in self.dataset.X]
        else:
            y_predicted = [self.predict_cubic_regression(x, B0, B1, B2, B3) for x in self.dataset.X]

        y_actual = self.dataset.y

        # Calculate correlation coefficient (r)
        mean_actual = sum(y_actual) / len(y_actual)
        mean_predicted = sum(y_predicted) / len(y_predicted)

        numerator = sum((y_i - mean_actual) * (y_hat_i - mean_predicted) for y_i, y_hat_i in zip(y_actual, y_predicted))
        denominator_actual = sum((y_i - mean_actual)**2 for y_i in y_actual)
        denominator_predicted = sum((y_hat_i - mean_predicted)**2 for y_hat_i in y_predicted)

        correlation = numerator / (denominator_actual * denominator_predicted)**0.5

        # Calculate coefficient of determination (R-squared)
        determination = correlation**2

        return correlation, determination

# DATASET
X = [108, 115, 106, 97, 95, 91, 97, 83, 83, 78, 54, 67, 56, 53, 61, 115, 81, 78, 30, 45, 99, 32, 25, 28, 90, 89]
y = [95, 96, 95, 97, 93, 94, 95, 93, 92, 86, 73, 80, 65, 69, 77, 96, 87, 89, 60, 63, 95, 61, 55, 56, 94, 93]

dataset = Dataset(X, y)
calculations = Calculations(dataset)

B0_linear, B1_linear = calculations.calculate_linear_regression()
correlation_linear, determination_linear = calculations.calculate_correlation_and_determination(B0_linear, B1_linear)
print(f"Linear Regression: y = {B0_linear} + {B1_linear} * x")
print(f"Correlation Coefficient (r): {correlation_linear}")
print(f"Coefficient of Determination (R-squared): {determination_linear}")


B0_quadratic, B1_quadratic, B2_quadratic = calculations.calculate_quadratic_regression()
print(f"\nQuadratic Regression: y = {B0_quadratic} + {B1_quadratic} * x + {B2_quadratic} * x^2")
correlation_quadratic, determination_quadratic = calculations.calculate_correlation_and_determination(B0_quadratic, B1_quadratic, B2_quadratic)
print(f"Correlation Coefficient (r): {correlation_quadratic}")
print(f"Coefficient of Determination (R-squared): {determination_quadratic}")


B0_cubic, B1_cubic, B2_cubic, B3_cubic = calculations.calculate_cubic_regression()
print(f"\nCubic Regression: y = {B0_cubic} + {B1_cubic} * x + {B2_cubic} * x^2 + {B3_cubic} * x^3")
correlation_cubic, determination_cubic = calculations.calculate_correlation_and_determination(B0_cubic, B1_cubic, B2_cubic, B3_cubic)
print(f"Correlation Coefficient (r): {correlation_cubic}")
print(f"Coefficient of Determination (R-squared): {determination_cubic}")

# Make predictions from main program
while True:
    prediction_input = input("\nEnter a value of x to predict y (type 'done' to exit): ")
    if prediction_input.lower() == 'done':
        break
    try:
        x_value = float(prediction_input)
        y_predicted_linear = calculations.predict_linear_regression(x_value, B0_linear, B1_linear)
        y_predicted_quadratic = calculations.predict_quadratic_regression(x_value, B0_quadratic, B1_quadratic, B2_quadratic)
        y_predicted_cubic = calculations.predict_cubic_regression(x_value, B0_cubic, B1_cubic, B2_cubic, B3_cubic)

        print(f"Predicted y for x = {x_value} (Linear Regression): {y_predicted_linear}")
        print(f"Predicted y for x = {x_value} (Quadratic Regression): {y_predicted_quadratic}")
        print(f"Predicted y for x = {x_value} (Cubic Regression): {y_predicted_cubic}")

    except ValueError:
        print("Invalid input. Please enter a valid number or 'done' to exit.")
