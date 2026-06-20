"""Benchmark problem interfaces and implementations."""

from __future__ import annotations

from molsrtde.problems.base import Problem
from molsrtde.problems.thinfilm import ThinFilmARConfig, ThinFilmARProblem, thinfilm_default_ideal_nadir
from molsrtde.problems.zdt import ZDT1, ZDT2, ZDT3

__all__ = [
    "Problem",
    "ThinFilmARConfig",
    "ThinFilmARProblem",
    "ZDT1",
    "ZDT2",
    "ZDT3",
    "load_problem",
    "thinfilm_default_ideal_nadir",
]


def load_problem(name: str, **kwargs) -> Problem:
    """Load a packaged benchmark problem by name."""
    key = str(name).strip().lower()
    if key == "zdt1":
        return ZDT1(**kwargs)
    if key == "zdt2":
        return ZDT2(**kwargs)
    if key == "zdt3":
        return ZDT3(**kwargs)
    if key in {"thinfilm", "thinfilm_ar", "thin_film_ar", "tf_ar"}:
        config = ThinFilmARConfig(
            lambda_min_nm=float(kwargs.get("lambda_min_nm", kwargs.get("lambda_min", 400.0))),
            lambda_max_nm=float(kwargs.get("lambda_max_nm", kwargs.get("lambda_max", 700.0))),
            n_lambda=int(kwargs.get("n_lambda", 61)),
            n0=float(kwargs.get("n0", 1.0)),
            nL=float(kwargs.get("nL", 1.45)),
            nH=float(kwargs.get("nH", 2.10)),
            ns=float(kwargs.get("ns", 1.52)),
            n_layers=int(kwargs.get("n_layers", kwargs.get("n_var", 20))),
            pattern=str(kwargs.get("pattern", "HL")),
            dmin_nm=float(kwargs.get("dmin_nm", kwargs.get("dmin", 20.0))),
            dmax_nm=float(kwargs.get("dmax_nm", kwargs.get("dmax", 200.0))),
        )
        return ThinFilmARProblem(config)
    raise ValueError(f"Unknown packaged problem: {name}")
