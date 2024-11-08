from dataclasses import dataclass
from typing import Literal, Union
import numpy as np


@dataclass
class LPP:
    c: list[float]  # Coefficients of the objective function
    a: list[list[float]]  # Coefficients of the constraints
    x: list[float]  #
    type: Literal[
        "max", "min"
    ]  # Type of the problem, either maximization or minimization
    e: float = 0.0001  # Approximation accuracy


@dataclass
class LPPSolution:
    state: Literal["solved", "unbounded"]  # State of the solver
    x: list[float]  # Optimal values of the variables
    z: Union[int, float]  # Optimal value of the objective function


class InteriorPointSolver:
    def __init__(self, llp: LPP, alpha: float = 0.5):
        self.llp = llp
        self.c = np.array(llp.c, float)
        self.A = np.array(llp.a, float)
        self.x = np.array(llp.x, float)
        self.alpha = alpha

    def solve(self):
        it = 0
        while True:
            v = self.x
            D = np.diag(self.x)

            AA = np.dot(self.A, D)
            cc = np.dot(D, self.c)

            I = np.eye(len(self.c))

            F = np.dot(AA, np.transpose(AA))
            FI = np.linalg.inv(F)
            H = np.dot(np.transpose(AA), FI)

            P = np.subtract(I, np.dot(H, AA))

            cp = np.dot(P, cc)

            nu = np.absolute(np.min(cp))
            y = np.add(np.ones(len(self.c), float), (self.alpha / nu) * cp)

            yy = np.dot(D, y)
            self.x = yy

            if it in range(6):
                print(f"Iteration {it}: x = {self.x}")
                it += 1

            if np.linalg.norm(np.subtract(yy, v), ord=2) < self.llp.e:
                print(f"Optimal solution found: x = {self.x}")
                print(f"Z: {np.dot(self.x, self.c)}\n\n")
                break


if __name__ == "__main__":
    # Example from Lab 6 Problem 2
    llp = LPP(
        c=[9, 10, 16, 0, 0, 0],
        a=[[18, 15, 12, 1, 0, 0], [6, 4, 8, 0, 1, 0], [5, 3, 3, 0, 0, 1]],
        x=[1, 1, 1, 315, 174, 169],
        type="max",
    )
    solver = InteriorPointSolver(llp)
    solver.solve()

    # Example from Lab 3 Problem 1
    llp = LPP(
        c=[9, 10, 16, 0, 0, 0],
        a=[[18, 15, 12, 1, 0, 0], [6, 4, 8, 0, 1, 0], [5, 3, 3, 0, 0, 1]],
        x=[1, 1, 1, 315, 174, 169],
        type="max",
    )
    solver = InteriorPointSolver(llp)
    solver.solve()

    # Example from Lab 3 Problem 3
    llp = LPP(
        c=[-2, 2, -6, 0, 0, 0],
        a=[[2, 1, -2, 1, 0, 0], [1, 2, 4, 0, 1, 0], [1, -1, 2, 0, 0, 1]],
        x=[1, 1, 1, 23, 11, 8],
        type="max",
    )
    solver = InteriorPointSolver(llp)
    solver.solve()

    # Self-made Example
    llp = LPP(
        c=[4, 3, 2, 1, 0, 0, 0],
        a=[
            [1, 1, 1, 1, 1, 0, 0],
            [2, 1, 3, 0, 0, 1, 0],
            [3, 4, 0, 1, 0, 0, 1],
        ],
        x=[1, 1, 1, 1, 36, -16, 17],
        type="max",
    )
    solver = InteriorPointSolver(llp)
    solver.solve()

    # Example from Test 1 Demo Problem 2
    llp = LPP(
        c=[-4, -3, 0, 0],
        a=[
            [3, 2, 1, 0],
            [3, 8, 0, 1],
        ],
        x=[1, 1, -17, 13],
        type="min",
    )
    solver = InteriorPointSolver(llp)
    solver.solve()
