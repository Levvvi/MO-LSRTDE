"""Thin-film anti-reflection benchmark from the legacy workspace."""

from __future__ import annotations

from dataclasses import dataclass

import numpy as np

from molsrtde.problems.base import Problem


@dataclass(frozen=True)
class ThinFilmARConfig:
    """Configuration for the thin-film anti-reflection problem."""

    lambda_min_nm: float = 400.0
    lambda_max_nm: float = 700.0
    n_lambda: int = 61
    n0: float = 1.0
    nL: float = 1.45
    nH: float = 2.10
    ns: float = 1.52
    n_layers: int = 20
    pattern: str = "HL"
    dmin_nm: float = 20.0
    dmax_nm: float = 200.0


class ThinFilmARProblem(Problem):
    """Multi-objective thin-film anti-reflection design problem."""

    def __init__(self, cfg: ThinFilmARConfig | None = None) -> None:
        self.cfg = cfg or ThinFilmARConfig()
        self.n_var = int(self.cfg.n_layers)
        self.n_obj = 3
        self.xl = np.full(self.n_var, float(self.cfg.dmin_nm), dtype=float)
        self.xu = np.full(self.n_var, float(self.cfg.dmax_nm), dtype=float)
        self.lambdas_nm = np.linspace(
            self.cfg.lambda_min_nm,
            self.cfg.lambda_max_nm,
            int(self.cfg.n_lambda),
            dtype=float,
        )
        self.k0 = 2.0 * np.pi / self.lambdas_nm
        self.n_stack = self._build_stack_indices()
        self.name = f"thinfilm_ar_L{self.cfg.n_layers}"

    def _build_stack_indices(self) -> np.ndarray:
        pattern = (self.cfg.pattern or "HL").upper()
        if not set(pattern).issubset({"H", "L"}):
            raise ValueError(f"pattern must only contain H and L, got: {self.cfg.pattern}")

        sequence = (pattern * (self.cfg.n_layers // len(pattern) + 1))[: self.cfg.n_layers]
        indices = np.empty(self.cfg.n_layers, dtype=float)
        for i, layer in enumerate(sequence):
            indices[i] = self.cfg.nH if layer == "H" else self.cfg.nL
        return indices

    @staticmethod
    def _tmm_transmittance_normal_incidence(
        k0: np.ndarray,
        n0: float,
        ns: float,
        n_layers: np.ndarray,
        d_nm: np.ndarray,
    ) -> tuple[np.ndarray, np.ndarray]:
        eta0 = float(n0)
        etas = float(ns)

        n_pop, n_layers_count = d_nm.shape
        n_lambda = k0.shape[0]

        m11 = np.ones((n_pop, n_lambda), dtype=np.complex128)
        m12 = np.zeros((n_pop, n_lambda), dtype=np.complex128)
        m21 = np.zeros((n_pop, n_lambda), dtype=np.complex128)
        m22 = np.ones((n_pop, n_lambda), dtype=np.complex128)

        for i in range(n_layers_count):
            n_i = float(n_layers[i])
            delta = (k0[None, :] * n_i) * d_nm[:, i][:, None]
            cosine = np.cos(delta)
            sine = np.sin(delta)

            layer11 = cosine
            layer22 = cosine
            layer12 = 1j * sine / n_i
            layer21 = 1j * n_i * sine

            new11 = m11 * layer11 + m12 * layer21
            new12 = m11 * layer12 + m12 * layer22
            new21 = m21 * layer11 + m22 * layer21
            new22 = m21 * layer12 + m22 * layer22
            m11, m12, m21, m22 = new11, new12, new21, new22

        denom = eta0 * m11 + eta0 * etas * m12 + m21 + etas * m22
        trans_amp = (2.0 * eta0) / denom
        refl_amp = (eta0 * m11 + eta0 * etas * m12 - m21 - etas * m22) / denom

        transmittance = (etas / eta0) * (np.abs(trans_amp) ** 2)
        reflectance = np.abs(refl_amp) ** 2
        return transmittance.real, reflectance.real

    def evaluate(self, x: np.ndarray) -> np.ndarray:
        x = np.asarray(x, dtype=float)
        if x.ndim == 1:
            x = x[None, :]
        if x.shape[1] != self.n_var:
            raise ValueError(f"X should have shape (N, {self.n_var}), got {x.shape}")

        thicknesses = np.clip(x, self.xl, self.xu)
        transmittance, _ = self._tmm_transmittance_normal_incidence(
            k0=self.k0,
            n0=self.cfg.n0,
            ns=self.cfg.ns,
            n_layers=self.n_stack,
            d_nm=thicknesses,
        )

        mean_t = np.mean(transmittance, axis=1)
        std_t = np.std(transmittance, axis=1)
        sum_d = np.sum(thicknesses, axis=1)
        return np.column_stack([1.0 - mean_t, std_t, sum_d]).astype(float)

    def sanity_check(self, n_samples: int = 3, seed: int = 0) -> dict[str, float]:
        """Return an energy-conservation sanity check for random designs."""
        rng = np.random.default_rng(seed)
        x = rng.uniform(self.xl, self.xu, size=(n_samples, self.n_var))
        t_values, r_values = self._tmm_transmittance_normal_incidence(
            self.k0,
            self.cfg.n0,
            self.cfg.ns,
            self.n_stack,
            np.clip(x, self.xl, self.xu),
        )
        err = np.max(np.abs((t_values + r_values) - 1.0))
        return {
            "max_energy_error": float(err),
            "mean_T_example": float(np.mean(t_values)),
            "mean_R_example": float(np.mean(r_values)),
        }


def thinfilm_default_ideal_nadir(n_layers: int, dmin: float, dmax: float) -> tuple[np.ndarray, np.ndarray]:
    """Return simple ideal/nadir estimates used by legacy thin-film experiments."""
    ideal = np.array([0.0, 0.0, float(n_layers) * float(dmin)], dtype=float)
    nadir = np.array([1.0, 0.5, float(n_layers) * float(dmax)], dtype=float)
    return ideal, nadir

