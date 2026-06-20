"""Reusable plotting hooks for MO-LSRTDE experiments.

Plotting from the legacy workspace remains experimental. Public plotting
functions should be added here only after their inputs and outputs are covered
by tests or documented reproduction commands.
"""

from __future__ import annotations

import numpy as np


def pareto_scatter_data(f: np.ndarray) -> tuple[np.ndarray, np.ndarray]:
    """Return x/y arrays for a two-objective Pareto scatter plot."""
    values = np.asarray(f, dtype=float)
    if values.ndim != 2 or values.shape[1] < 2:
        raise ValueError("Pareto scatter data requires an objective matrix with at least two columns.")
    return values[:, 0], values[:, 1]

