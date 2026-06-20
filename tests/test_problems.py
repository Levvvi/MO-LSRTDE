from __future__ import annotations

import numpy as np
import pytest

from molsrtde.problems import ZDT1, ZDT2, ZDT3, ThinFilmARProblem, load_problem


def test_load_zdt_problem_and_pareto_front() -> None:
    problem = load_problem("zdt1", n_var=5)
    x = np.zeros((2, problem.n_var))
    f = problem.evaluate(x)
    pf = problem.pareto_front(10)

    assert isinstance(problem, ZDT1)
    assert f.shape == (2, 2)
    assert pf is not None
    assert pf.shape == (10, 2)


def test_zdt_variants_evaluate() -> None:
    for cls in (ZDT2, ZDT3):
        problem = cls(n_var=5)
        x = np.full((3, problem.n_var), 0.5)
        f = problem.evaluate(x)
        pf = problem.pareto_front(12)
        assert f.shape == (3, 2)
        assert pf.shape[1] == 2


def test_thinfilm_problem_evaluate_and_sanity_check() -> None:
    problem = load_problem("thinfilm_ar", n_layers=2, n_lambda=5)
    assert isinstance(problem, ThinFilmARProblem)

    x = np.tile((problem.xl + problem.xu) / 2.0, (2, 1))
    f = problem.evaluate(x)
    sanity = problem.sanity_check(n_samples=2, seed=0)

    assert f.shape == (2, 3)
    assert sanity["max_energy_error"] < 1e-10


def test_unknown_problem_raises() -> None:
    with pytest.raises(ValueError):
        load_problem("does-not-exist")

