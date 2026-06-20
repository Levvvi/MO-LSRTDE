"""Core MO-LSRTDE algorithm implementation."""

from __future__ import annotations

from contextlib import suppress
from typing import Any

import numpy as np

from molsrtde.result import MOLSRTDEResult


def _eval_fg(problem: Any, x: np.ndarray, n_jobs: int = -1) -> tuple[np.ndarray, np.ndarray]:
    """Evaluate a problem and normalize output to `(F, G)` arrays."""
    del n_jobs
    x = np.asarray(x, dtype=float)
    if x.ndim == 1:
        x = x[None, :]

    out = problem.evaluate(x)
    if isinstance(out, tuple):
        f, g = out
    elif isinstance(out, dict):
        f = out.get("F")
        g = out.get("G", out.get("CV", None))
    else:
        f = out
        g = None

    f = np.asarray(f, dtype=float)
    if f.ndim == 1:
        f = f[:, None]

    if g is None:
        g = np.zeros((f.shape[0], 0), dtype=float)
    else:
        g = np.asarray(g, dtype=float)
        if g.ndim == 1:
            g = g[:, None]

    return f, g


def _constraint_violation(g: np.ndarray) -> np.ndarray:
    g = np.asarray(g, dtype=float)
    if g.size == 0:
        return np.zeros((g.shape[0] if g.ndim > 0 else 1,), dtype=float)
    if g.ndim == 1:
        g = g[:, None]
    return np.sum(np.maximum(g, 0.0), axis=1)


def dominates(
    fa: np.ndarray,
    ga: np.ndarray,
    fb: np.ndarray,
    gb: np.ndarray,
    eps: float = 0.0,
) -> bool:
    """Return whether point `a` Deb-dominates point `b` for minimization."""
    cva = _constraint_violation(ga.reshape(1, -1))[0]
    cvb = _constraint_violation(gb.reshape(1, -1))[0]
    if cva < cvb:
        return True
    if cva > cvb:
        return False

    if eps > 0.0:
        not_worse = np.all(fa <= fb + eps)
        strictly_better = np.any(fa < fb - eps)
    else:
        not_worse = np.all(fa <= fb)
        strictly_better = np.any(fa < fb)
    return bool(not_worse and strictly_better)


def fast_nondominated_sort(
    f: np.ndarray,
    g: np.ndarray | None,
    epsilon: float = 0.0,
) -> list[np.ndarray]:
    """Vectorized Deb-style nondominated sorting with constraint handling."""
    del epsilon
    f = np.asarray(f, dtype=float)

    if g is None:
        g = np.zeros((f.shape[0], 0), dtype=float)
    else:
        g = np.asarray(g, dtype=float)
        if g.ndim == 1:
            g = g[:, None]

    n_points = f.shape[0]
    if n_points == 0:
        return []

    cv = _constraint_violation(g)
    cvi = cv[:, None]
    cvj = cv[None, :]

    tol = 1e-12
    less_eq = f[:, None, :] <= f[None, :, :] + tol
    less = f[:, None, :] < f[None, :, :] - tol
    not_worse = np.all(less_eq, axis=2)
    strictly_better = np.any(less, axis=2)

    dom = (cvi < cvj) | ((cvi == cvj) & not_worse & strictly_better)
    np.fill_diagonal(dom, False)

    n_dom = dom.sum(axis=0).astype(int)
    remaining = np.ones(n_points, dtype=bool)
    fronts: list[np.ndarray] = []

    while True:
        front = np.where((n_dom == 0) & remaining)[0]
        if front.size == 0:
            break
        fronts.append(front)
        remaining[front] = False
        dominated_by_front = dom[front, :]
        n_dom = n_dom - dominated_by_front.sum(axis=0)
        n_dom[n_dom < 0] = 0

    leftover = np.where(remaining)[0]
    if leftover.size > 0:
        fronts.append(leftover)

    return [np.asarray(front, dtype=int) for front in fronts]


def crowding_distance(f: np.ndarray) -> np.ndarray:
    """Return NSGA-II crowding distance for an objective matrix."""
    f = np.asarray(f, dtype=float)
    n_points, n_obj = f.shape
    if n_points == 0:
        return np.array([], dtype=float)
    if n_points == 1:
        return np.array([np.inf])
    if n_points == 2:
        return np.array([np.inf, np.inf])

    distance = np.zeros(n_points, dtype=float)
    for obj_index in range(n_obj):
        order = np.argsort(f[:, obj_index])
        values = f[order, obj_index]
        f_min, f_max = values[0], values[-1]
        if f_max - f_min <= 1e-32:
            continue
        distance[order[0]] = distance[order[-1]] = np.inf
        for i in range(1, n_points - 1):
            distance[order[i]] += (values[i + 1] - values[i - 1]) / (f_max - f_min)
    return distance


def mirror_repair(
    x: np.ndarray,
    xl: np.ndarray,
    xu: np.ndarray,
    rng: np.random.RandomState | None = None,
) -> np.ndarray:
    """Repair out-of-bounds decision vectors by reflection and clipping."""
    if rng is None:
        rng = np.random.RandomState()
    del rng

    x = np.asarray(x, dtype=float)
    if x.ndim == 1:
        x = x[None, :]

    xl = np.asarray(xl, dtype=float).ravel()
    xu = np.asarray(xu, dtype=float).ravel()

    _, n_var = x.shape
    if xl.size == 1:
        xl = np.full(n_var, xl.item(), dtype=float)
    if xu.size == 1:
        xu = np.full(n_var, xu.item(), dtype=float)

    repaired = x.copy()
    for _ in range(2):
        for var_index in range(n_var):
            below = repaired[:, var_index] < xl[var_index]
            repaired[below, var_index] = 2.0 * xl[var_index] - repaired[below, var_index]
            above = repaired[:, var_index] > xu[var_index]
            repaired[above, var_index] = 2.0 * xu[var_index] - repaired[above, var_index]

    return np.clip(repaired, xl, xu)


def binomial_crossover(
    x: np.ndarray,
    v: np.ndarray,
    cr: float,
    rng: np.random.RandomState,
) -> np.ndarray:
    """Standard DE binomial crossover."""
    x = np.asarray(x, dtype=float)
    v = np.asarray(v, dtype=float)
    n_var = x.shape[0]

    trial = np.empty(n_var, dtype=float)
    j_rand = rng.randint(n_var)
    for var_index in range(n_var):
        if rng.rand() < cr or var_index == j_rand:
            trial[var_index] = v[var_index]
        else:
            trial[var_index] = x[var_index]
    return trial


def polynomial_mutation(
    y: np.ndarray,
    xl: np.ndarray,
    xu: np.ndarray,
    rng: np.random.RandomState,
    eta_m: float = 20.0,
    p_mut: float | None = None,
) -> np.ndarray:
    """NSGA-II polynomial mutation for a single decision vector."""
    y = np.asarray(y, dtype=float).copy()
    xl = np.asarray(xl, dtype=float)
    xu = np.asarray(xu, dtype=float)
    n_var = y.shape[0]

    if p_mut is None:
        p_mut = 1.0 / float(max(n_var, 1))

    for var_index in range(n_var):
        if rng.rand() >= p_mut:
            continue
        lower, upper = float(xl[var_index]), float(xu[var_index])
        width = upper - lower
        if width <= 0.0:
            y[var_index] = lower
            continue

        delta1 = (y[var_index] - lower) / width
        delta2 = (upper - y[var_index]) / width
        delta1 = float(np.clip(delta1, 0.0, 1.0))
        delta2 = float(np.clip(delta2, 0.0, 1.0))
        rand = rng.rand()
        mut_pow = 1.0 / (eta_m + 1.0)

        if rand <= 0.5:
            xy = float(np.clip(1.0 - delta1, 1e-14, 1.0))
            val = 2.0 * rand + (1.0 - 2.0 * rand) * (xy ** (eta_m + 1.0))
            deltaq = (max(val, 0.0) ** mut_pow) - 1.0
        else:
            xy = float(np.clip(1.0 - delta2, 1e-14, 1.0))
            val = 2.0 * (1.0 - rand) + 2.0 * (rand - 0.5) * (xy ** (eta_m + 1.0))
            deltaq = 1.0 - (max(val, 0.0) ** mut_pow)

        y[var_index] = y[var_index] + deltaq * width

    return np.minimum(np.maximum(y, xl), xu)


def simple_gdr_restart(
    pop: np.ndarray,
    f_pop: np.ndarray,
    g_pop: np.ndarray,
    problem: Any,
    xl: np.ndarray,
    xu: np.ndarray,
    rng: np.random.RandomState,
    restart_frac: float = 0.20,
    n_jobs: int = 1,
) -> tuple[np.ndarray, np.ndarray, np.ndarray, np.ndarray, int]:
    """Replace crowded dominated points with random restarts."""
    pop = np.asarray(pop, dtype=float)
    f_pop = np.asarray(f_pop, dtype=float)
    g_pop = np.asarray(g_pop, dtype=float)

    pop_size, n_var = pop.shape
    if pop_size == 0:
        return pop, f_pop, g_pop, np.zeros(0, dtype=bool), 0

    n_new = int(max(1, round(restart_frac * pop_size)))
    fronts = fast_nondominated_sort(f_pop, g_pop, epsilon=0.0)
    n_fronts = len(fronts)

    ranks = np.full(pop_size, n_fronts, dtype=int)
    for rank, front in enumerate(fronts):
        ranks[front] = rank

    crowding = np.zeros(pop_size, dtype=float)
    for front in fronts:
        if front.size == 0:
            continue
        crowding[front] = crowding_distance(f_pop[front])

    dominated_idx = np.where(ranks > 0)[0]
    candidates = dominated_idx if dominated_idx.size >= n_new else np.arange(pop_size, dtype=int)
    if candidates.size == 0:
        return pop, f_pop, g_pop, np.zeros(pop_size, dtype=bool), 0

    order_local = np.argsort(crowding[candidates])
    replace_count = min(n_new, candidates.size)
    replace_idx = candidates[order_local[:replace_count]]

    xl = np.asarray(xl, dtype=float).ravel()
    xu = np.asarray(xu, dtype=float).ravel()
    if xl.size == 1:
        xl = np.full(n_var, xl.item(), dtype=float)
    if xu.size == 1:
        xu = np.full(n_var, xu.item(), dtype=float)

    new_x = rng.uniform(xl, xu, size=(replace_count, n_var))
    new_f, new_g = _eval_fg(problem, new_x, n_jobs=n_jobs)
    evals_used = new_x.shape[0]

    pop_new = pop.copy()
    f_pop_new = f_pop.copy()
    g_pop_new = g_pop.copy()
    for k, idx in enumerate(replace_idx):
        pop_new[idx] = new_x[k]
        f_pop_new[idx] = new_f[k]
        g_pop_new[idx] = new_g[k]

    protected = np.zeros(pop_size, dtype=bool)
    protected[replace_idx] = True
    return pop_new, f_pop_new, g_pop_new, protected, evals_used


def _run_mo_lsrtde_result(
    problem: Any,
    pop_size: int = 100,
    max_evals: int = 100_000,
    seed: int = 42,
    epsilon_init: float = 0.0,
    sigma_share_init: float = 0.10,
    restart_interval: int = 60,
    stagn_tol: int = 15,
    logger: Any | None = None,
    inner_eval_jobs: int = 1,
) -> MOLSRTDEResult:
    del sigma_share_init, stagn_tol
    rng = np.random.RandomState(seed)
    n_var = int(problem.n_var)
    pop_size = int(pop_size)
    max_evals = int(max_evals)

    xl = np.asarray(problem.xl, dtype=float).ravel()
    xu = np.asarray(problem.xu, dtype=float).ravel()
    if xl.size == 1:
        xl = np.full(n_var, xl.item(), dtype=float)
    if xu.size == 1:
        xu = np.full(n_var, xu.item(), dtype=float)

    pop = rng.uniform(xl, xu, size=(pop_size, n_var))
    f_pop, g_pop = _eval_fg(problem, pop, n_jobs=inner_eval_jobs)
    evals = pop.shape[0]

    fronts_init = fast_nondominated_sort(f_pop, g_pop, epsilon=epsilon_init)
    best_front = fronts_init[0]
    archive = pop[best_front].copy()
    f_archive = f_pop[best_front].copy()
    g_archive = g_pop[best_front].copy()

    memory_size = 6
    mem_f = np.full(memory_size, 0.6, dtype=float)
    mem_cr = np.full(memory_size, 0.9, dtype=float)
    mem_pos = 0

    protect_gen_span = 10
    protect_until = np.zeros(pop_size, dtype=int)

    generation = 0
    all_idx = np.arange(pop_size, dtype=int)

    if logger is not None:
        with suppress(Exception):
            logger.maybe_update(evals, f_archive)

    while evals < max_evals:
        generation += 1

        memory_index = rng.randint(0, memory_size, size=pop_size)
        f_scale = np.empty(pop_size, dtype=float)
        cr_values = np.empty(pop_size, dtype=float)
        for i in range(pop_size):
            scale = -1.0
            while scale <= 0.0:
                scale = mem_f[memory_index[i]] + 0.1 * rng.standard_cauchy()
            f_scale[i] = min(scale, 1.0)
            cr_values[i] = float(np.clip(mem_cr[memory_index[i]] + 0.1 * rng.randn(), 0.05, 1.0))

        fronts = fast_nondominated_sort(f_pop, g_pop, epsilon=epsilon_init)
        best_front = fronts[0]
        if best_front.size < 2:
            order = np.argsort(np.sum(f_pop, axis=1))
            k = max(2, int(0.2 * pop_size))
            pbest_all = order[:k]
        else:
            k = max(2, int(0.3 * best_front.size))
            k = min(k, best_front.size)
            pbest_all = rng.choice(best_front, size=k, replace=False)

        offspring = np.empty_like(pop)
        for i in range(pop_size):
            x_i = pop[i]
            p_idx = int(rng.choice(pbest_all))
            x_pbest = pop[p_idx]

            candidate_idx = all_idx[all_idx != i]
            r1, r2 = rng.choice(candidate_idx, size=2, replace=False)
            x_r1, x_r2 = pop[r1], pop[r2]

            mutant = x_i + f_scale[i] * (x_pbest - x_i) + f_scale[i] * (x_r1 - x_r2)
            trial = binomial_crossover(x_i, mutant, cr_values[i], rng)
            trial = polynomial_mutation(trial, xl, xu, rng, eta_m=20.0, p_mut=None)
            offspring[i] = trial

        offspring = mirror_repair(offspring, xl, xu, rng=rng)
        f_offspring, g_offspring = _eval_fg(problem, offspring, n_jobs=inner_eval_jobs)
        evals += offspring.shape[0]
        if evals >= max_evals:
            break

        x_all = np.vstack([pop, offspring])
        f_all = np.vstack([f_pop, f_offspring])
        g_all = np.vstack([g_pop, g_offspring])

        fronts_all = fast_nondominated_sort(f_all, g_all, epsilon=epsilon_init)
        selected: list[int] = []
        for front in fronts_all:
            if len(selected) + front.size <= pop_size:
                selected.extend(front.tolist())
            else:
                remain = pop_size - len(selected)
                if remain <= 0:
                    break
                crowding = crowding_distance(f_all[front])
                order_front = np.argsort(-crowding)
                selected.extend(front[order_front[:remain]].tolist())
                break

        selected_idx = np.asarray(selected, dtype=int)
        selected_set = set(selected_idx.tolist())

        pop_new = x_all[selected_idx]
        f_new = f_all[selected_idx]
        g_new = g_all[selected_idx]

        succ_f: list[float] = []
        succ_cr: list[float] = []
        for i in range(pop_size):
            parent_idx = i
            offspring_idx = i + pop_size
            if (parent_idx not in selected_set) and (offspring_idx in selected_set):
                succ_f.append(float(f_scale[i]))
                succ_cr.append(float(cr_values[i]))

        if succ_f:
            success_f = np.asarray(succ_f, dtype=float)
            success_cr = np.asarray(succ_cr, dtype=float)
            weights = success_f / max(1e-32, float(np.sum(success_f)))
            mem_f[mem_pos] = float(np.sum(weights * success_f))
            mem_cr[mem_pos] = float(np.sum(weights * success_cr))
            mem_pos = (mem_pos + 1) % memory_size

        new_protect = np.zeros(pop_size, dtype=int)
        for new_idx, src_idx in enumerate(selected_idx):
            if src_idx < pop_size:
                new_protect[new_idx] = protect_until[src_idx]
            else:
                parent_idx = src_idx - pop_size
                if 0 <= parent_idx < pop_size:
                    new_protect[new_idx] = protect_until[parent_idx]
        protect_until = new_protect

        pop, f_pop, g_pop = pop_new, f_new, g_new

        archive_all = np.vstack([archive, pop])
        f_archive_all = np.vstack([f_archive, f_pop])
        g_archive_all = np.vstack([g_archive, g_pop])

        fronts_archive = fast_nondominated_sort(f_archive_all, g_archive_all, epsilon=epsilon_init)
        nd_idx = fronts_archive[0]
        archive = archive_all[nd_idx]
        f_archive = f_archive_all[nd_idx]
        g_archive = g_archive_all[nd_idx]

        if archive.shape[0] > pop_size:
            crowding_archive = crowding_distance(f_archive)
            order_archive = np.argsort(-crowding_archive)
            keep = order_archive[:pop_size]
            archive = archive[keep]
            f_archive = f_archive[keep]
            g_archive = g_archive[keep]

        if logger is not None:
            with suppress(Exception):
                logger.maybe_update(evals, f_archive)

        if (generation % max(1, int(restart_interval)) == 0) and (evals < max_evals):
            pop, f_pop, g_pop, protect_mask, restart_evals = simple_gdr_restart(
                pop,
                f_pop,
                g_pop,
                problem=problem,
                xl=xl,
                xu=xu,
                rng=rng,
                restart_frac=0.25,
                n_jobs=inner_eval_jobs,
            )
            evals += restart_evals
            protect_until = np.where(protect_mask, generation + protect_gen_span, protect_until)

            archive_all = np.vstack([archive, pop])
            f_archive_all = np.vstack([f_archive, f_pop])
            g_archive_all = np.vstack([g_archive, g_pop])
            fronts_archive = fast_nondominated_sort(f_archive_all, g_archive_all, epsilon=epsilon_init)
            nd_idx = fronts_archive[0]
            archive = archive_all[nd_idx]
            f_archive = f_archive_all[nd_idx]
            g_archive = g_archive_all[nd_idx]

            if archive.shape[0] > pop_size:
                crowding_archive = crowding_distance(f_archive)
                order_archive = np.argsort(-crowding_archive)
                keep = order_archive[:pop_size]
                archive = archive[keep]
                f_archive = f_archive[keep]
                g_archive = g_archive[keep]

    history = {
        "generations": generation,
        "seed": seed,
        "pop_size": pop_size,
        "max_evals": max_evals,
        "actual_evals": evals,
    }
    return MOLSRTDEResult(X=archive, F=f_archive, n_evals=evals, history=history)


def mo_lsrtde(
    problem: Any,
    pop_size: int = 100,
    max_evals: int = 100_000,
    seed: int = 42,
    epsilon_init: float = 0.0,
    sigma_share_init: float = 0.10,
    restart_interval: int = 60,
    stagn_tol: int = 15,
    logger: Any | None = None,
    inner_eval_jobs: int = 1,
) -> tuple[np.ndarray, np.ndarray]:
    """Compatibility function returning `(X, F)` like the legacy module."""
    result = _run_mo_lsrtde_result(
        problem=problem,
        pop_size=pop_size,
        max_evals=max_evals,
        seed=seed,
        epsilon_init=epsilon_init,
        sigma_share_init=sigma_share_init,
        restart_interval=restart_interval,
        stagn_tol=stagn_tol,
        logger=logger,
        inner_eval_jobs=inner_eval_jobs,
    )
    return result.X, result.F


def multi_objective_LSRTDE(
    problem: Any,
    pop_size: int = 100,
    max_evals: int = 100_000,
    seed: int = 42,
    epsilon_init: float = 0.0,
    sigma_share_init: float = 0.10,
    restart_interval: int = 60,
    stagn_tol: int = 15,
    logger: Any | None = None,
) -> tuple[np.ndarray, np.ndarray]:
    """Backward-compatible legacy entry point."""
    return mo_lsrtde(
        problem=problem,
        pop_size=pop_size,
        max_evals=max_evals,
        seed=seed,
        epsilon_init=epsilon_init,
        sigma_share_init=sigma_share_init,
        restart_interval=restart_interval,
        stagn_tol=stagn_tol,
        logger=logger,
        inner_eval_jobs=1,
    )


def multi_objective_LSRTDE_ZDT3_DTLZ3(
    problem: Any,
    pop_size: int = 100,
    max_evals: int = 100_000,
    seed: int = 42,
    epsilon_init: float = 0.0,
    sigma_share_init: float = 0.10,
    restart_interval: int = 60,
    stagn_tol: int = 15,
    logger: Any | None = None,
) -> tuple[np.ndarray, np.ndarray]:
    """Backward-compatible alias for old experiment configurations."""
    return multi_objective_LSRTDE(
        problem=problem,
        pop_size=pop_size,
        max_evals=max_evals,
        seed=seed,
        epsilon_init=epsilon_init,
        sigma_share_init=sigma_share_init,
        restart_interval=restart_interval,
        stagn_tol=stagn_tol,
        logger=logger,
    )


class MOLSRTDE:
    """Main MO-LSRTDE optimizer.

    Parameters mirror the legacy `mo_lsrtde` function while exposing a class
    API suitable for package users and tests.
    """

    def __init__(
        self,
        pop_size: int = 100,
        max_evals: int = 25_000,
        seed: int | None = None,
        epsilon_init: float = 0.0,
        sigma_share_init: float = 0.10,
        restart_interval: int = 60,
        stagn_tol: int = 15,
        inner_eval_jobs: int = 1,
    ) -> None:
        self.pop_size = int(pop_size)
        self.max_evals = int(max_evals)
        self.seed = 42 if seed is None else int(seed)
        self.epsilon_init = float(epsilon_init)
        self.sigma_share_init = float(sigma_share_init)
        self.restart_interval = int(restart_interval)
        self.stagn_tol = int(stagn_tol)
        self.inner_eval_jobs = int(inner_eval_jobs)

    def run(self, problem: Any) -> MOLSRTDEResult:
        """Run MO-LSRTDE on a problem instance and return a result object."""
        return _run_mo_lsrtde_result(
            problem=problem,
            pop_size=self.pop_size,
            max_evals=self.max_evals,
            seed=self.seed,
            epsilon_init=self.epsilon_init,
            sigma_share_init=self.sigma_share_init,
            restart_interval=self.restart_interval,
            stagn_tol=self.stagn_tol,
            inner_eval_jobs=self.inner_eval_jobs,
        )
