"""Result containers for MO-LSRTDE runs."""

from __future__ import annotations

from dataclasses import dataclass, field
from typing import Any

import numpy as np


@dataclass(slots=True)
class MOLSRTDEResult:
    """Observable output returned by `MOLSRTDE.run`."""

    X: np.ndarray
    F: np.ndarray
    n_evals: int
    history: dict[str, Any] = field(default_factory=dict)

    def __post_init__(self) -> None:
        self.X = np.asarray(self.X, dtype=float)
        self.F = np.asarray(self.F, dtype=float)
        self.n_evals = int(self.n_evals)
