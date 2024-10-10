from dataclasses import dataclass
from typing import Literal, Union


@dataclass
class LPP:
    c: list[float]  # list of coefficients of objective function
    a: list[list[float]]  # matrix of coefficients of constraint function
    b: list[float]  # list of right-hand side values of constraints
    type: Literal["max", "min"]  # type of optimization problem
    e: float = 1e-9  # approximation accuracy


@dataclass
class LPPSolution:
    state: Literal["solved", "unbounded"]
    x: list[float]
    z: Union[int, float]


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

        pivot = tableau[pivot_row][pivot_column]
        for i in range(len(tableau[pivot_row])):
            tableau[pivot_row][i] /= pivot
        for i in range(len(tableau)):
            if i == pivot_row:
                continue
            factor = tableau[i][pivot_column]
            for j in range(len(tableau[i])):
                tableau[i][j] -= factor * tableau[pivot_row][j]

        basic_vars[pivot_row - 1] = pivot_column

    x = [0] * num_vars
    for i in range(num_constraints):
        if basic_vars[i] < num_vars:
            x[basic_vars[i]] = tableau[i + 1][-1]

    return LPPSolution(
        z=tableau[0][-1] * (-1 if llp.type == "min" else 1),
        x=x,
        state='solved'
    )


examples = [
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
    LPP(
        c=[2, 3, 0, -1, 0, 0],
        a=[
            [2, -1, 0, -2, 1, 0],
            [3, 2, 1, -3, 0, 0],
            [-1, 3, 0, 4, 0, 1]
        ],
        b=[16, 18, 24],
        type="max"
    ),
    LPP(
        c=[-2, 2, -6, 0],
        a=[
            [2, 1, -2, 0],
            [1, 2, 4, 0],
            [1, -1, 2, 0]
        ],
        b=[24, 23, 10],
        type="min"
    ),
    LPP(
        c=[2, 1],
        a=[[1, -1]],
        b=[1],
        type="max"
    ),
    LPP(
        c=[4, 3, 2, 1],
        a=[
            [1, 1, 1, 1],
            [2, 1, 3, 0],
            [3, 4, 0, 1]
        ],
        b=[40, -10, 30],
        type="max"
    )
]

if __name__ == '__main__':
    for example in examples:
        print(solve(example))
        print()
