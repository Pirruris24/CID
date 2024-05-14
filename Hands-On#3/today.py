# DATASET
X = [108, 115, 106, 97, 95, 91, 97, 83, 83, 78, 54, 67, 56, 53, 61, 115, 81, 78, 30, 45, 99, 32, 25, 28, 90, 89]
y = [95, 96, 95, 97, 93, 94, 95, 93, 92, 86, 73, 80, 65, 69, 77, 96, 87, 89, 60, 63, 95, 61, 55, 56, 94, 93]

n = len(X)


def calculate_inverse(matrix):
    n = len(matrix)
    identity = [[1 if i == j else 0 for j in range(n)] for i in range(n)]
    augmented_matrix = [row + identity[i] for i, row in enumerate(matrix)]
    
    # Perform Gauss-Jordan elimination
    for i in range(n):
        pivot = augmented_matrix[i][i]
        if pivot == 0:
            # Assign one to expected value
            augmented_matrix[i][i] = 1
            continue
        
        for j in range(i+1, n):
            ratio = augmented_matrix[j][i] / pivot
            augmented_matrix[j] = [a - ratio*b for a, b in zip(augmented_matrix[j], augmented_matrix[i])]
    
    # Backward substitution
    for i in range(n-1, -1, -1):
        pivot = augmented_matrix[i][i]
        if pivot == 0:
            # Assign one to expected value
            augmented_matrix[i][i] = 1
            continue
        
        augmented_matrix[i] = [a / pivot for a in augmented_matrix[i]]
        for j in range(i-1, -1, -1):
            ratio = augmented_matrix[j][i]
            augmented_matrix[j] = [a - ratio*b for a, b in zip(augmented_matrix[j], augmented_matrix[i])]
    
    # Extract the inverse matrix
    inverse_matrix = [row[n:] for row in augmented_matrix]
    
    return inverse_matrix



print("LINEAR CALCULATIONS")

X_design = [[1, x] for x in X]  # Degree 1 polynomial regression

X_design_T = [[row[i] for row in X_design] for i in range(2)] 

X_design_T_X_design = [[sum(X_row[i] * X_row[j] for i in range(len(X_row))) for j in range(2)] for X_row in X_design]

inverse_X_design_T_X_design = calculate_inverse(X_design_T_X_design)

X_design_T_y = [sum(X_row[i] * y[i] for i in range(len(X_row))) for X_row in X_design_T]

result = [sum(row_inv[i] * X_design_T_y[i] for i in range(len(row_inv))) for row_inv in inverse_X_design_T_X_design]

B0 = result[0]
B1 = result[1]

print("B0 =", B0)
print("B1 =", B1)


print("\nQUADRATIC CALCULATIONS")

X_design = [[1, x, x**2] for x in X]  

X_design_T = [[row[i] for row in X_design] for i in range(3)] 

X_design_T_X_design = [[sum(X_row[i] * X_row[j] for i in range(len(X_row))) for j in range(3)] for X_row in X_design_T]

inverse_X_design_T_X_design = calculate_inverse(X_design_T_X_design)

X_design_T_y = [sum(X_row[i] * y[i] for i in range(len(X_row))) for X_row in X_design_T]

result = [sum(row_inv[i] * X_design_T_y[i] for i in range(len(row_inv))) for row_inv in inverse_X_design_T_X_design]

B0 = result[0]
B1 = result[1]
B2 = result[2]

print("B0 =", B0)
print("B1 =", B1)
print("B2 =", B2)


print("\nCUBIC CALCULATIONS")

X_design = [[1, x, x**2, x**3] for x in X] 

X_design_T = [[row[i] for row in X_design] for i in range(4)]  

X_design_T_X_design = [[sum(X_row[i] * X_row[j] for i in range(len(X_row))) for j in range(4)] for X_row in X_design_T]

inverse_X_design_T_X_design = calculate_inverse(X_design_T_X_design)

X_design_T_y = [sum(X_row[i] * y[i] for i in range(len(X_row))) for X_row in X_design_T]

result = [sum(row_inv[i] * X_design_T_y[i] for i in range(len(row_inv))) for row_inv in inverse_X_design_T_X_design]

B0 = result[0]
B1 = result[1]
B2 = result[2]
B3 = result[3]

print("B0 =", B0)
print("B1 =", B1)
print("B2 =", B2)
print("B3 =", B3)
