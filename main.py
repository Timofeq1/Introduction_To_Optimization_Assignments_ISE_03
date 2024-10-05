import numpy as np


# Define the function
def simplex_method(C, A, b, eps=1e-9):
    # Basic input checks to handle invalid problem formulations
    if not isinstance(C, list) or not isinstance(A, list) or not isinstance(b, list):
        print("The method is not applicable!")
        return None

    if len(C) == 0 or len(A) == 0 or len(b) == 0:
        print("The method is not applicable!")
        return None

    n = len(C)  # Number of variables
    m = len(b)  # Number of constraints

    # Check if the number of columns in A matches the number of variables (n)
    for row in A:
        if len(row) != n:
            print("The method is not applicable!")
            return None

    # Check if the number of rows in A matches the number of constraints (m)
    if len(A) != m:
        print("The method is not applicable!")
        return None

    # Step 1: Print the optimization problem
    print("Objective function:")
    print(f"max z = {' + '.join([f'{C[i]} * x{i + 1}' for i in range(n)])}")

    print("\nSubject to constraints:")
    for i in range(m):
        constraint = ' + '.join([f"{A[i][j]} * x{j + 1}" for j in range(n)])
        print(f"{constraint} <= {b[i]}")

    # Step 2: Initialize the tableau by introducing slack variables
    tableau = np.hstack([A, np.eye(m)])  # Add slack variables
    tableau = np.vstack([tableau, np.hstack([-np.array(C), np.zeros(m)])])  # Objective row, convert C to np.array
    b = np.array(b)
    tableau = np.column_stack([tableau, np.hstack([b, 0])])  # RHS column

    # Step 3: Simplex algorithm
    def get_pivot_column(tableau):
        last_row = tableau[-1, :-1]
        pivot_col = np.argmin(last_row)  # Most negative coefficient in the objective row
        return pivot_col if last_row[pivot_col] < -eps else -1

    def get_pivot_row(tableau, pivot_col):
        rhs = tableau[:-1, -1]
        pivot_column = tableau[:-1, pivot_col]
        ratio = np.where(pivot_column > 0, rhs / pivot_column, np.inf)  # Only positive pivot entries
        pivot_row = np.argmin(ratio)
        return pivot_row if ratio[pivot_row] != np.inf else -1

    def perform_pivot_operation(tableau, pivot_row, pivot_col):
        tableau[pivot_row] /= tableau[pivot_row, pivot_col]  # Make pivot element 1
        for i in range(tableau.shape[0]):
            if i != pivot_row:
                tableau[i] -= tableau[i, pivot_col] * tableau[pivot_row]  # Make other elements in pivot column 0

    # Main loop
    while True:
        pivot_col = get_pivot_column(tableau)
        if pivot_col == -1:
            # Optimal solution found
            solution = np.zeros(n)
            for i in range(n):
                col = tableau[:, i]
                if np.count_nonzero(col[:-1]) == 1 and np.count_nonzero(col[-1]) == 0:
                    solution[i] = tableau[np.where(col[:-1] == 1)[0][0], -1]
            return {
                "solver_state": "solved",
                "x*": solution,
                "z": tableau[-1, -1]
            }

        pivot_row = get_pivot_row(tableau, pivot_col)
        if pivot_row == -1:
            # Problem is unbounded
            return {
                "solver_state": "unbounded",
                "x*": None,
                "z": None
            }

        # Perform the pivot operation
        perform_pivot_operation(tableau, pivot_row, pivot_col)


# Function to print the result in the desired format
def print_result(result):
    if result['solver_state'] == "unbounded":
        # Only print the first line if the problem is unbounded
        print(f"- solver state: {result['solver_state']}")
    elif result['solver_state'] == "solved":
        print(f"- solver state: {result['solver_state']}")
        print(f"- x*: {', '.join(map(str, result['x*']))}")
        print(f"- z: {result['z']}")
    else:
        print("The method is not applicable!")


# Example usage
C = [3, 10]  # Coefficients of the objective function
A = [[2, 0], [1, 2]]  # Coefficients of the constraint functions
b = [6, 6]  # Right-hand side values

result = simplex_method(C, A, b)

# If the result is None, the method was not applicable
if result is None:
    print("The method is not applicable!")
else:
    # Print the result in the specified format
    print_result(result)
