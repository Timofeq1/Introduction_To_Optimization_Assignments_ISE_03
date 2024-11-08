from dataclasses import dataclass
from copy import deepcopy

INF = 10 ** 5


@dataclass
class TransportationProblem:
    supply: list[int]  # Supply of each factory S
    demand: list[int]  # Demand of each warehouse D
    costs: list[list[int]]  # Costs of transportation C

    def __repr__(self):
        column_width = 8
        demand_labels = [f"D{j + 1}" for j in range(len(self.demand))]
        supply_labels = [f"S{i + 1}" for i in range(len(self.supply))]
        header = (
                f'{"":<{column_width}}'
                + "".join([f"{label:^{column_width}}" for label in demand_labels])
                + f'{"Supply":^{column_width}}'
        )
        separator = "-" * len(header)
        lines = [header, separator]
        for i, supply_label in enumerate(supply_labels):
            row = f"{supply_label:<{column_width}}"
            for cost in self.costs[i]:
                row += f"{cost:>{column_width}}"
            row += f"{self.supply[i]:>{column_width}}"
            lines.append(row)
        lines.append(separator)
        demand_line = (
                f'{"Demand":<{column_width}}'
                + "".join([f"{d:>{column_width}}" for d in self.demand])
                + " " * column_width
        )
        lines.append(demand_line)

        return "\n".join(lines)


class ProblemIsNotBalanced(Exception):
    """Raised when the problem is not balanced"""
    pass


@dataclass
class TransportationSolution:
    path: list[int]
    z: int

    def __repr__(self):
        return f"\nPath: {self.path}\nZ: {self.z}"


class NorthWestCorner:
    def __init__(self, tp: TransportationProblem):
        self.tp = tp
        self.supply = deepcopy(tp.supply)
        self.demand = deepcopy(tp.demand)
        self.costs = deepcopy(tp.costs)
        self.ans = 0
        self.path = []
        if sum(self.supply) != sum(self.demand):
            raise ProblemIsNotBalanced

    def solve(self) -> TransportationSolution:
        row, col = 0, 0

        while row != len(self.costs) and col != len(self.costs[0]):
            if self.supply[row] < self.demand[col]:
                self.ans += self.supply[row] * self.costs[row][col]
                self.demand[col] -= self.supply[row]
                self.path.append((self.supply[row], self.costs[row][col]))
                self.supply[row] = 0
                row += 1
            else:
                self.ans += self.demand[col] * self.costs[row][col]
                self.supply[row] -= self.demand[col]
                self.path.append((self.demand[col], self.costs[row][col]))
                self.demand[col] = 0
                col += 1

        return TransportationSolution(path=self.path, z=self.ans)


class VogelApproximation:
    def __init__(self, tp: TransportationProblem):
        self.tp = tp
        self.supply: list[int] = deepcopy(tp.supply)
        self.demand: list[int] = deepcopy(tp.demand)
        self.costs: list[list[int]] = deepcopy(tp.costs)
        self.ans = 0
        self.path = []
        if sum(self.supply) != sum(self.demand):
            raise ProblemIsNotBalanced

    def get_mins(self, arr: list[int]) -> tuple[int, int]:
        """
        Find 2 smallest values in given array

        :return: tuple of 2 smallest values
        """
        min1, min2 = INF, INF
        for val in arr:
            if val < min1:
                min1, min2 = val, min1
            elif val < min2:
                min2 = val
        return min1, min2

    def get_col(self, index: int) -> list[int]:
        """
        Get column of the matrix

        :param index: index of the column
        :return: list of values in the column
        """
        return [row[index] for row in self.costs]

    def get_diffs(self) -> tuple[list[int], list[int]]:
        """
        Find row and column differences

        :return: tuple of lists of differences for rows and columns respectively
        """

        row_diffs = []
        col_diffs = []

        for i in range(len(self.costs)):
            mins = self.get_mins(self.costs[i])
            row_diffs.append(abs(mins[0] - mins[1]))

        for i in range(len(self.costs[0])):
            col = self.get_col(i)
            mins = self.get_mins(col)
            col_diffs.append(abs(mins[0] - mins[1]))

        return row_diffs, col_diffs

    def solve(self) -> TransportationSolution:
        ans = 0
        while max(self.supply) > 0 and max(self.demand) > 0:
            row, col = self.get_diffs()
            max1, max2 = max(row), max(col)

            if max1 >= max2:
                for i, diff in enumerate(row):
                    if diff == max1:
                        min1 = min(self.costs[i])
                        for j, cost in enumerate(self.costs[i]):
                            if cost == min1:
                                min2 = min(self.supply[i], self.demand[j])
                                ans += min1 * min2
                                self.supply[i] -= min2
                                self.demand[j] -= min2
                                self.path.append((min2, self.costs[i][j]))
                                if self.demand[j] == 0:
                                    for k in range(len(self.costs)):
                                        self.costs[k][j] = INF
                                else:
                                    self.costs[i] = [INF] * len(self.costs[i])
                                break
                        break
            else:
                for i, diff in enumerate(col):
                    if diff == max2:
                        min1 = min(self.get_col(i))
                        for j in range(len(self.costs)):
                            cost = self.costs[j][i]
                            if cost == min1:
                                min2 = min(self.supply[j], self.demand[i])
                                ans += min1 * min2
                                self.supply[j] -= min2
                                self.demand[i] -= min2
                                self.path.append((min2, self.costs[j][i]))
                                if self.demand[i] == 0:
                                    for k in range(len(self.costs)):
                                        self.costs[k][i] = INF
                                else:
                                    self.costs[j] = [INF] * len(self.costs[j])
                                break
                        break

        return TransportationSolution(path=self.path, z=ans)


class RussellApproximation:
    def __init__(self, tp: TransportationProblem):
        self.tp = tp
        self.supply = deepcopy(tp.supply)
        self.demand = deepcopy(tp.demand)
        self.costs = deepcopy(tp.costs)
        self.ans = 0
        self.path = []
        if sum(self.supply) != sum(self.demand):
            raise ProblemIsNotBalanced

    def solve(self) -> TransportationSolution:
        while sum(self.supply) > 0 and sum(self.demand) > 0:
            reduced_costs = []
            min_i, min_j = 0, 0
            min_cost = 0

            for i in range(len(self.supply)):
                reduced_costs.append([])
                for j in range(len(self.demand)):
                    if self.supply[i] > 0 and self.demand[j] > 0:
                        reduced_costs[i].append(
                            self.costs[i][j]
                            - max(self.costs[i])
                            - max([self.costs[k][j] for k in range(len(self.supply))])
                        )
                        if reduced_costs[i][j] < min_cost:
                            min_cost = reduced_costs[i][j]
                            min_i, min_j = i, j
            reduce = min(self.supply[min_i], self.demand[min_j])
            self.path.append((reduce, self.costs[min_i][min_j]))
            self.supply[min_i] -= reduce
            self.demand[min_j] -= reduce
            self.ans += reduce * self.costs[min_i][min_j]
            if self.supply[min_i] == 0:
                self.supply.pop(min_i)
                self.costs.pop(min_i)
            if self.demand[min_j] == 0:
                self.demand.pop(min_j)
                for i in range(len(self.costs)):
                    self.costs[i].pop(min_j)

        return TransportationSolution(path=self.path, z=self.ans)


def main():
    examples = [
        # Example from Lab 7 Problem 2
        TransportationProblem(
            supply=[160, 140, 170],
            demand=[120, 50, 190, 110],
            costs=[
                [7, 8, 1, 2],
                [4, 5, 9, 8],
                [9, 2, 3, 6]
            ]
        ),
        # Example from Lab 7 Problem 5
        TransportationProblem(
            supply=[50, 30, 10],
            demand=[30, 30, 10, 20],
            costs=[
                [1, 2, 4, 1],
                [2, 3, 1, 5],
                [3, 2, 4, 4]
            ]
        ),
        # Self-made Example (not balanced)
        TransportationProblem(
            supply=[100, 200, 300],
            demand=[150, 250, 200, 100],
            costs=[
                [2, 3, 1, 4],
                [4, 2, 3, 1],
                [3, 1, 2, 3]
            ]
        ),

    ]

    for example in examples:
        print(example, end='\n\n')
        try:
            for solver in [
                NorthWestCorner(example),
                VogelApproximation(example),
                RussellApproximation(example),
            ]:
                print(f"Solver {solver.__class__.__name__}: {solver.solve()}")
        except ProblemIsNotBalanced:
            print("Problem is not balanced")
        print()


if __name__ == "__main__":
    main()
