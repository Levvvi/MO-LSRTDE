from __future__ import annotations

import numpy as np

from molsrtde.algorithm import (
    MOLSRTDE,
    binomial_crossover,
    crowding_distance,
    dominates,
    fast_nondominated_sort,
    mirror_repair,
    mo_lsrtde,
    multi_objective_LSRTDE,
    multi_objective_LSRTDE_ZDT3_DTLZ3,
    polynomial_mutation,
    simple_gdr_restart,
)
from molsrtde.metrics import non_dominated_mask
from molsrtde.problems import ZDT1


def test_package_import() -> None:
    import molsrtde

    assert molsrtde.__version__
    assert molsrtde.MOLSRTDE is MOLSRTDE


def test_smoke_run_result_shape_and_bounds() -> None:
    problem = ZDT1(n_var=5)
    result = MOLSRTDE(pop_size=10, max_evals=30, seed=0).run(problem)

    assert result.X.ndim == 2
    assert result.F.ndim == 2
    assert result.X.shape[0] == result.F.shape[0]
    assert result.X.shape[1] == problem.n_var
    assert result.F.shape[1] == problem.n_obj
    assert np.all(problem.xl - 1e-12 <= result.X)
    assert np.all(problem.xu + 1e-12 >= result.X)


def test_seed_determinism_for_smoke_case() -> None:
    problem = ZDT1(n_var=5)
    first = MOLSRTDE(pop_size=10, max_evals=30, seed=7).run(problem)
    second = MOLSRTDE(pop_size=10, max_evals=30, seed=7).run(problem)

    np.testing.assert_allclose(first.X, second.X)
    np.testing.assert_allclose(first.F, second.F)
    assert first.n_evals == second.n_evals


def test_max_evals_budget_sanity_without_restart() -> None:
    problem = ZDT1(n_var=5)
    result = MOLSRTDE(pop_size=10, max_evals=30, seed=0, restart_interval=100).run(problem)

    assert result.n_evals <= 30
    assert result.history["actual_evals"] == result.n_evals


def test_result_is_nondominated_archive() -> None:
    problem = ZDT1(n_var=5)
    result = MOLSRTDE(pop_size=10, max_evals=40, seed=1, restart_interval=100).run(problem)

    assert np.all(non_dominated_mask(result.F))


def test_legacy_function_wrappers_return_arrays() -> None:
    problem = ZDT1(n_var=5)
    for func in (mo_lsrtde, multi_objective_LSRTDE, multi_objective_LSRTDE_ZDT3_DTLZ3):
        x, f = func(problem, pop_size=10, max_evals=30, seed=0)
        assert x.shape[0] == f.shape[0]
        assert x.shape[1] == problem.n_var
        assert f.shape[1] == problem.n_obj


def test_nondominated_sort_and_dominance_helpers() -> None:
    f = np.array([[0.0, 1.0], [1.0, 0.0], [0.5, 0.5], [1.0, 1.0]])
    g = np.zeros((4, 0))
    fronts = fast_nondominated_sort(f, g)

    assert set(fronts[0].tolist()) == {0, 1, 2}
    assert fronts[1].tolist() == [3]
    assert dominates(f[2], np.array([]), f[3], np.array([]))
    assert not dominates(f[3], np.array([]), f[2], np.array([]))


def test_crowding_distance_edges_are_infinite() -> None:
    f = np.array([[0.0, 1.0], [0.5, 0.5], [1.0, 0.0]])
    distance = crowding_distance(f)

    assert np.isinf(distance[0])
    assert np.isinf(distance[2])
    assert distance[1] > 0


def test_variation_and_repair_helpers() -> None:
    rng = np.random.RandomState(0)
    x = np.array([0.2, 0.8])
    mutant = np.array([0.9, 0.1])

    trial = binomial_crossover(x, mutant, cr=1.0, rng=rng)
    np.testing.assert_allclose(trial, mutant)

    unchanged = polynomial_mutation(x, np.zeros(2), np.ones(2), rng, p_mut=0.0)
    np.testing.assert_allclose(unchanged, x)

    repaired = mirror_repair(np.array([[-0.2, 1.2]]), np.zeros(2), np.ones(2))
    assert np.all(repaired >= 0.0)
    assert np.all(repaired <= 1.0)


def test_restart_helper_replaces_at_least_one_point() -> None:
    problem = ZDT1(n_var=3)
    rng = np.random.RandomState(3)
    pop = rng.uniform(problem.xl, problem.xu, size=(6, problem.n_var))
    f = problem.evaluate(pop)
    g = np.zeros((pop.shape[0], 0))

    new_pop, new_f, new_g, mask, evals = simple_gdr_restart(
        pop,
        f,
        g,
        problem,
        problem.xl,
        problem.xu,
        rng,
        restart_frac=0.25,
    )

    assert evals >= 1
    assert mask.any()
    assert new_pop.shape == pop.shape
    assert new_f.shape == f.shape
    assert new_g.shape == g.shape
