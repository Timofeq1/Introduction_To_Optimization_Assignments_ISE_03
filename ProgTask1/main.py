from dataclasses import dataclass
from typing import Literal, Union


@dataclass
class LPP:
    c: list[float]  # Coefficients of the objective function
    a: list[list[float]]  # Coefficients of the constraints
    b: list[float]  # Right-hand side of the constraints
    type: Literal["max", "min"]  # Type of the problem, either maximization or minimization
    e: float = 1e-9  # Approximation accuracy


@dataclass
class LPPSolution:
    state: Literal["solved", "unbounded"]  # State of the solver
    x: list[float]  # Optimal values of the variables
    z: Union[int, float]  # Optimal value of the objective function


def get_min(a: list[float]) -> float:
    ans = a[0]
    for i in range(1, len(a)):
        if a[i] < ans:
            ans = a[i]
    return ans


def solve(llp: LPP) -> LPPSolution:
    num_vars = len(llp.c)
    num_constraints = len(llp.b)

    tableau: list[list[Union[float, int]]] = [llp.c + [0] * (num_constraints + 1)]
    for i in range(num_constraints):
        s_array = [0 if j != i else 1 for j in range(num_constraints)]
        tableau.append(llp.a[i] + s_array + [llp.b[i]])

    if llp.type == "max":
        for i in range(num_vars):
            tableau[0][i] *= -1

    basic_vars = list(range(num_vars, num_vars + num_constraints))

    while True:
        if all(coef >= 0 for coef in tableau[0][:-1]):
            break

        pivot_column = tableau[0].index(get_min(tableau[0][:-1]))
        pivot_row = None
        min_ratio = None
        for i in range(1, len(tableau)):
            if tableau[i][pivot_column] <= 0:
                continue
            ratio = tableau[i][-1] / tableau[i][pivot_column]
            if min_ratio is None or ratio < min_ratio:
                min_ratio = ratio
                pivot_row = i

        if pivot_row is None:
            return LPPSolution(
                state="unbounded",
                x=[],
                z=float('inf')
            )

        basic_vars[pivot_row - 1] = pivot_column

        pivot = tableau[pivot_row][pivot_column]
        for i in range(len(tableau[pivot_row])):
            tableau[pivot_row][i] /= pivot
        for i in range(len(tableau)):
            if i == pivot_row:
                continue
            factor = tableau[i][pivot_column]
            for j in range(len(tableau[i])):
                tableau[i][j] -= factor * tableau[pivot_row][j]

    x = [0.0] * (num_vars + num_constraints)
    for i in range(1, len(tableau)):
        var_index = basic_vars[i - 1]
        x[var_index] = tableau[i][-1]

    return LPPSolution(
        z=tableau[0][-1] * (-1 if llp.type == "min" else 1),
        x=x[:num_vars],
        state='solved'
    )


examples = [
    # Example from Lab 3 Problem 1
    LPP(
        c=[9, 10, 16],
        a=[
            [18, 15, 12],
            [6, 4, 8],
            [5, 3, 3]
        ],
        b=[360, 192, 180],
        type="max"
    ),
    # Example from Lab 3 Problem 3
    LPP(
        c=[-2, 2, -6],
        a=[
            [2, 1, -2],
            [1, 2, 4],
            [1, -1, 2]
        ],
        b=[24, 23, 10],
        type="min"
    ),
    # Unbounded Example
    LPP(
        c=[2, 1],
        a=[[1, -1]],
        b=[1],
        type="max"
    ),
    # Self-made Example
    LPP(
        c=[4, 3, 2, 1],
        a=[
            [1, 1, 1, 1],
            [2, 1, 3, 0],
            [3, 4, 0, 1]
        ],
        b=[40, -10, 30],
        type="max"
    ),
    # Example from Test 1 Demo Problem 2
    LPP(
        c=[-4, -3],
        a=[
            [3, 2],
            [3, 8],
        ],
        b=[-12, 24],
        type="min"
    ),
]

if __name__ == '__main__':
    for example in examples:
        result = solve(example)
        print(result)
        if result.state == 'unbounded':
            print("The method is not applicable!\n")
            continue
        print("Decision variables: ", result.x)
        print(f"{'Maximum' if example.type == 'max' else 'Minimum'} value: {result.z}\n")


