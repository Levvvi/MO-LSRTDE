"""Dependency-light multi-objective performance metrics."""

from __future__ import annotations

import numpy as np


def _as_2d(points: np.ndarray) -> np.ndarray:
    arr = np.asarray(points, dtype=float)
    if arr.ndim == 1:
        arr = arr[None, :]
    return arr


def non_dominated_mask(f: np.ndarray) -> np.ndarray:
    """Return a boolean mask for nondominated minimization points."""
    f = _as_2d(f)
    n_points = f.shape[0]
    keep = np.ones(n_points, dtype=bool)
    for i in range(n_points):
        if not keep[i]:
            continue
        dominated_by_i = np.all(f[i] <= f, axis=1) & np.any(f[i] < f, axis=1)
        dominated_by_i[i] = False
        keep[dominated_by_i] = False
        if np.any(np.all(f <= f[i], axis=1) & np.any(f < f[i], axis=1)):
            keep[i] = False
    return keep


def hypervolume(f: np.ndarray, ref_point: np.ndarray) -> float:
    """Compute exact dominated hypervolume for two minimization objectives."""
    f = _as_2d(f)
    ref = np.asarray(ref_point, dtype=float).ravel()
    if f.shape[1] != 2 or ref.size != 2:
        raise NotImplementedError("This lightweight hypervolume implementation supports exactly two objectives.")

    finite = np.all(np.isfinite(f), axis=1)
    within_ref = np.all(f < ref, axis=1)
    points = f[finite & within_ref]
    if points.size == 0:
        return 0.0

    points = points[non_dominated_mask(points)]
    points = points[np.argsort(points[:, 0])]

    volume = 0.0
    previous_y = ref[1]
    for x_val, y_val in points:
        height = previous_y - y_val
        width = ref[0] - x_val
        if height > 0.0 and width > 0.0:
            volume += width * height
            previous_y = y_val
    return float(volume)


def igd(f: np.ndarray, pareto_front: np.ndarray) -> float:
    """Compute inverted generational distance."""
    f = _as_2d(f)
    pf = _as_2d(pareto_front)
    distances = np.linalg.norm(pf[:, None, :] - f[None, :, :], axis=2)
    return float(np.mean(np.min(distances, axis=1)))


def gd(f: np.ndarray, pareto_front: np.ndarray) -> float:
    """Compute generational distance."""
    f = _as_2d(f)
    pf = _as_2d(pareto_front)
    distances = np.linalg.norm(f[:, None, :] - pf[None, :, :], axis=2)
    return float(np.mean(np.min(distances, axis=1)))


def igd_plus(f: np.ndarray, pareto_front: np.ndarray) -> float:
    """Compute IGD+ for minimization."""
    f = _as_2d(f)
    pf = _as_2d(pareto_front)
    diff = np.maximum(f[None, :, :] - pf[:, None, :], 0.0)
    distances = np.linalg.norm(diff, axis=2)
    return float(np.mean(np.min(distances, axis=1)))


def spread(f: np.ndarray) -> float:
    """Return spacing-style spread based on nearest-neighbor distances."""
    f = _as_2d(f)
    if f.shape[0] < 2:
        return 0.0
    distances = np.linalg.norm(f[:, None, :] - f[None, :, :], axis=2)
    np.fill_diagonal(distances, np.inf)
    nearest = np.min(distances, axis=1)
    return float(np.std(nearest, ddof=0))

