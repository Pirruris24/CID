# DATASET
X = [108, 115, 106, 97, 95, 91, 97, 83, 83, 78, 54, 67, 56, 53, 61, 115, 81, 78, 30, 45, 99, 32, 25, 28, 90, 89]
y = [95, 96, 95, 97, 93, 94, 95, 93, 92, 86, 73, 80, 65, 69, 77, 96, 87, 89, 60, 63, 95, 61, 55, 56, 94, 93]

#degree = 2

n = len(X)


print("LINEAR CALCULATIONS")

X_design = [[1, x] for x in X]  # Degree 2 polynomial regression
#X_design = []
#for i in range(n):
 #   row = [X[i]**d for d in range(degree + 1)]  # Include 0th degree term as well

 #   X_design.append(row)

X_design_T = [[row[i] for row in X_design] for i in range(2)]


print("X =")
for row in X_design:
    print(row)


print("X_T =")
for row in X_design_T:
    print(row)

#X_design_T_X_design = [[sum(a*b for a, b in zip(X_row, Y_col)) for Y_col in X_design] for X_row in X_design_T]
X_design_T_X_design = [[sum(X_row[i] * X_row[j] for X_row in X_design) for j in range(2)] for i in range(2)]



print("X_T * X =")
for row in X_design_T_X_design:
    print(row)

# Calculate the inverse of X_design_T_X_design using Gauss-Jordan elimination
def calculate_inverse(matrix):
    n = len(matrix)
    identity = [[1 if i == j else 0 for j in range(n)] for i in range(n)]
    augmented_matrix = [row + identity[i] for i, row in enumerate(matrix)]
    
    # Perform Gauss-Jordan elimination
    for i in range(n):
        pivot = augmented_matrix[i][i]
        for j in range(i+1, n):
            ratio = augmented_matrix[j][i] / pivot
            augmented_matrix[j] = [a - ratio*b for a, b in zip(augmented_matrix[j], augmented_matrix[i])]
    
    # Backward substitution
    for i in range(n-1, -1, -1):
        pivot = augmented_matrix[i][i]
        augmented_matrix[i] = [a / pivot for a in augmented_matrix[i]]
        for j in range(i-1, -1, -1):
            ratio = augmented_matrix[j][i]
            augmented_matrix[j] = [a - ratio*b for a, b in zip(augmented_matrix[j], augmented_matrix[i])]
    
    # Extract the inverse matrix
    inverse_matrix = [row[n:] for row in augmented_matrix]
    
    return inverse_matrix

# Calculate the inverse of X_design_T_X_design
inverse_X_design_T_X_design = calculate_inverse(X_design_T_X_design)


print("Inverse of X_T * X =")
for row in inverse_X_design_T_X_design:
    print(row)


X_design_T_y = [sum(a*b for a, b in zip(row_X_T, y)) for row_X_T in X_design_T]


print("X^T * y =", X_design_T_y)


result = [sum(a*b for a, b in zip(row_inv, X_design_T_y)) for row_inv in inverse_X_design_T_X_design]


print("Cofficients =", result)



B0=result[0]
B1=result[1]


print("B0=",B0)
print("B1=",B1)
print(f"Expression: y = {B0} + {B1} * x")

print("QUADRATIC CALCULATIONS")

X_design = [[1, x, x**2] for x in X]  

X_design_T = [[row[i] for row in X_design] for i in range(3)] 


print("X =")
for row in X_design:
    print(row)


print("X_T =")
for row in X_design_T:
    print(row)


X_design_T_X_design = [[sum(X_row[i] * X_row[j] for X_row in X_design) for j in range(3)] for i in range(3)]


print("X_T * X =")
for row in X_design_T_X_design:
    print(row)



inverse_X_design_T_X_design = calculate_inverse(X_design_T_X_design)

print("Inverse of X_T * X =")
for row in inverse_X_design_T_X_design:
    print(row)


X_design_T_y = [sum(a*b for a, b in zip(row_X_T, y)) for row_X_T in X_design_T]

print("X^T * y =", X_design_T_y)

result = [sum(a*b for a, b in zip(row_inv, X_design_T_y)) for row_inv in inverse_X_design_T_X_design]


print("Cofficients =", result)

B0=result[0]
B1=result[1]
B2=result[2]

print("B0=",B0)
print("B1=",B1)
print("B2=",B2)
print(f"Expression: y = {B0} + {B1} * x + {B2} * x^2")

print("CUBIC CALCULATIONS")

# Define the design matrix X_design for cubic regression
X_design = [[1, x, x**2, x**3] for x in X]

# Calculate X_design_T_X_design
X_design_T = [[row[i] for row in X_design] for i in range(4)]
X_design_T_X_design = [[sum(X_row[i] * X_row[j] for X_row in X_design) for j in range(4)] for i in range(4)]

print("X_T * X =")
for row in X_design_T_X_design:
    print(row)

# Calculate the inverse of X_design_T_X_design using the calculate_inverse function
inverse_X_design_T_X_design = calculate_inverse(X_design_T_X_design)

print("Inverse of X_T * X =")
for row in inverse_X_design_T_X_design:
    print(row)

# Calculate X_design_T_y
X_design_T_y = [sum(a*b for a, b in zip(row_X_T, y)) for row_X_T in X_design_T]
print("X^T * y =", X_design_T_y)

# Calculate the coefficients using matrix multiplication
result = [sum(a*b for a, b in zip(row_inv, X_design_T_y)) for row_inv in inverse_X_design_T_X_design]

print("Coefficients =", result)

# Assign the coefficients to B0, B1, B2, and B3
B0 = result[0]
B1 = result[1]
B2 = result[2]
B3 = result[3]

print("B0 =", B0)
print("B1 =", B1)
print("B2 =", B2)
print("B3 =", B3)

print(f"Expression: y = {B0} + {B1} * x + {B2} * x^2 + {B3} * x^3")