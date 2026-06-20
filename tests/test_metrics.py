from __future__ import annotations

import numpy as np
import pytest

from molsrtde.metrics import gd, hypervolume, igd, igd_plus, non_dominated_mask, spread


def test_hypervolume_known_value() -> None:
    f = np.array([[0.2, 0.8], [0.8, 0.2]])
    assert hypervolume(f, np.array([1.0, 1.0])) == pytest.approx(0.28)


def test_distance_metrics_known_values() -> None:
    f = np.array([[0.0, 1.0], [1.0, 0.0]])
    pf = np.array([[0.0, 1.0], [1.0, 0.0]])

    assert igd(f, pf) == 0.0
    assert gd(f, pf) == 0.0
    assert igd_plus(f, pf) == 0.0
    assert spread(f) == 0.0


def test_non_dominated_mask_filters_dominated_points() -> None:
    f = np.array([[0.0, 0.0], [0.5, 0.5], [1.0, 0.0]])
    mask = non_dominated_mask(f)

    assert mask.tolist() == [True, False, False]
