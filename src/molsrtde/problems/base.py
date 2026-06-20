"""Problem interfaces for MO-LSRTDE."""

from __future__ import annotations

from abc import ABC, abstractmethod

import numpy as np


class Problem(ABC):
    """Minimal batch-evaluable minimization problem interface."""

    name: str
    n_var: int
    n_obj: int
    xl: np.ndarray
    xu: np.ndarray

    @abstractmethod
    def evaluate(self, x: np.ndarray) -> np.ndarray | tuple[np.ndarray, np.ndarray]:
        """Evaluate one or more decision vectors."""

    def pareto_front(self, n_points: int = 100) -> np.ndarray | None:
        """Return an approximate Pareto front when available."""
        del n_points
        return None

