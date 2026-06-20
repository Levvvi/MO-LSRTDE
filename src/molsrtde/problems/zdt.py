"""ZDT benchmark problems."""

from __future__ import annotations

import numpy as np

from molsrtde.problems.base import Problem


class ZDT1(Problem):
    """ZDT1 two-objective benchmark problem."""

    def __init__(self, n_var: int = 30) -> None:
        self.name = "zdt1"
        self.n_var = int(n_var)
        self.n_obj = 2
        self.xl = np.zeros(self.n_var, dtype=float)
        self.xu = np.ones(self.n_var, dtype=float)

    def evaluate(self, x: np.ndarray) -> np.ndarray:
        x = np.asarray(x, dtype=float)
        if x.ndim == 1:
            x = x[None, :]
        f1 = x[:, 0]
        g = 1.0 + 9.0 * np.mean(x[:, 1:], axis=1)
        h = 1.0 - np.sqrt(f1 / g)
        return np.column_stack([f1, g * h])

    def pareto_front(self, n_points: int = 100) -> np.ndarray:
        f1 = np.linspace(0.0, 1.0, int(n_points))
        return np.column_stack([f1, 1.0 - np.sqrt(f1)])


class ZDT2(ZDT1):
    """ZDT2 two-objective benchmark problem."""

    def __init__(self, n_var: int = 30) -> None:
        super().__init__(n_var=n_var)
        self.name = "zdt2"

    def evaluate(self, x: np.ndarray) -> np.ndarray:
        x = np.asarray(x, dtype=float)
        if x.ndim == 1:
            x = x[None, :]
        f1 = x[:, 0]
        g = 1.0 + 9.0 * np.mean(x[:, 1:], axis=1)
        h = 1.0 - (f1 / g) ** 2
        return np.column_stack([f1, g * h])

    def pareto_front(self, n_points: int = 100) -> np.ndarray:
        f1 = np.linspace(0.0, 1.0, int(n_points))
        return np.column_stack([f1, 1.0 - f1**2])


class ZDT3(ZDT1):
    """ZDT3 disconnected-front benchmark problem."""

    def __init__(self, n_var: int = 30) -> None:
        super().__init__(n_var=n_var)
        self.name = "zdt3"

    def evaluate(self, x: np.ndarray) -> np.ndarray:
        x = np.asarray(x, dtype=float)
        if x.ndim == 1:
            x = x[None, :]
        f1 = x[:, 0]
        g = 1.0 + 9.0 * np.mean(x[:, 1:], axis=1)
        ratio = f1 / g
        h = 1.0 - np.sqrt(ratio) - ratio * np.sin(10.0 * np.pi * f1)
        return np.column_stack([f1, g * h])

    def pareto_front(self, n_points: int = 100) -> np.ndarray:
        intervals = np.array(
            [
                [0.0, 0.0830015349],
                [0.1822287280, 0.2577623634],
                [0.4093136748, 0.4538821041],
                [0.6183967944, 0.6525117038],
                [0.8233317983, 0.8518328654],
            ],
            dtype=float,
        )
        counts = np.full(len(intervals), max(1, int(n_points) // len(intervals)), dtype=int)
        counts[: int(n_points) - counts.sum()] += 1
        pieces = [np.linspace(start, end, count) for (start, end), count in zip(intervals, counts, strict=True)]
        f1 = np.concatenate(pieces)
        f2 = 1.0 - np.sqrt(f1) - f1 * np.sin(10.0 * np.pi * f1)
        return np.column_stack([f1, f2])
