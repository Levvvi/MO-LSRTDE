"""Experiment orchestration for packaged MO-LSRTDE runs."""

from __future__ import annotations

import json
from pathlib import Path
from typing import Any

import numpy as np

from molsrtde.algorithm import MOLSRTDE
from molsrtde.metrics import hypervolume, igd, spread
from molsrtde.problems import load_problem


def load_config(path: str | Path) -> dict[str, Any]:
    """Load a JSON-compatible YAML config, using PyYAML when available."""
    config_path = Path(path)
    text = config_path.read_text(encoding="utf-8")

    try:
        import yaml  # type: ignore[import-untyped]
    except Exception:
        try:
            return json.loads(text)
        except json.JSONDecodeError as exc:
            raise RuntimeError(
                "PyYAML is not installed and the config is not valid JSON-compatible YAML."
            ) from exc
    loaded = yaml.safe_load(text)
    if not isinstance(loaded, dict):
        raise ValueError(f"Config must load to a mapping: {config_path}")
    return loaded


def _problem_from_spec(spec: dict[str, Any]) -> Any:
    problem_spec = spec.get("problem", spec.get("problem_name", "zdt1"))
    if isinstance(problem_spec, str):
        return load_problem(problem_spec)
    if isinstance(problem_spec, dict):
        name = problem_spec.get("name", "zdt1")
        params = problem_spec.get("params", {})
        return load_problem(name, **params)
    raise TypeError(f"Unsupported problem spec: {problem_spec!r}")


def _optimizer_from_spec(spec: dict[str, Any]) -> MOLSRTDE:
    algorithm_spec = spec.get("algorithm", {})
    if isinstance(algorithm_spec, str):
        if algorithm_spec.upper() != "MO-LSRTDE":
            raise ValueError(f"Only MO-LSRTDE is supported by the packaged smoke runner: {algorithm_spec}")
        params: dict[str, Any] = {}
    elif isinstance(algorithm_spec, dict):
        name = str(algorithm_spec.get("name", "MO-LSRTDE")).upper()
        if name != "MO-LSRTDE":
            raise ValueError(f"Only MO-LSRTDE is supported by the packaged smoke runner: {name}")
        params = dict(algorithm_spec.get("params", {}))
    else:
        params = {}

    for key in ("pop_size", "max_evals", "seed", "restart_interval"):
        if key in spec and key not in params:
            params[key] = spec[key]
    return MOLSRTDE(**params)


def _metric_summary(result_f: np.ndarray, problem: Any) -> dict[str, float]:
    metrics = {"spread": spread(result_f)}
    pareto_front = None
    if hasattr(problem, "pareto_front"):
        pareto_front = problem.pareto_front(n_points=200)
    if pareto_front is not None:
        metrics["igd"] = igd(result_f, pareto_front)
    if result_f.shape[1] == 2:
        if pareto_front is not None:
            ref = np.maximum(np.max(pareto_front, axis=0), np.max(result_f, axis=0)) + 0.1
        else:
            ref = np.max(result_f, axis=0) + 0.1
        metrics["hypervolume"] = hypervolume(result_f, ref)
    return {key: float(value) for key, value in metrics.items()}


def run_config(config: dict[str, Any], *, base_dir: str | Path | None = None) -> list[dict[str, Any]]:
    """Run all entries in an experiment config and return compact summaries."""
    root = Path(base_dir or ".").resolve()
    output_dir = root / str(config.get("output_dir", "results/smoke"))
    validation_summary = config.get("validation_summary", "validation/smoke_summary.json")
    runs = config.get("runs", [])
    if not isinstance(runs, list) or not runs:
        raise ValueError("Config must contain a non-empty 'runs' list.")

    summaries: list[dict[str, Any]] = []
    for run_index, run_spec in enumerate(runs):
        if not isinstance(run_spec, dict):
            raise TypeError(f"Run spec must be a mapping, got: {run_spec!r}")
        problem = _problem_from_spec(run_spec)
        optimizer = _optimizer_from_spec(run_spec)
        result = optimizer.run(problem)
        metrics = _metric_summary(result.F, problem)

        problem_name = getattr(problem, "name", run_spec.get("problem_name", "problem"))
        run_id = int(run_spec.get("run_id", run_index))
        run_dir = output_dir / problem_name / "MO-LSRTDE" / f"run_{run_id}"
        run_dir.mkdir(parents=True, exist_ok=True)
        np.savetxt(run_dir / "final_X.csv", result.X, delimiter=",")
        np.savetxt(run_dir / "final_F.csv", result.F, delimiter=",")
        (run_dir / "metrics.json").write_text(json.dumps(metrics, indent=2), encoding="utf-8")

        summaries.append(
            {
                "problem": problem_name,
                "algorithm": "MO-LSRTDE",
                "run_id": run_id,
                "seed": optimizer.seed,
                "pop_size": optimizer.pop_size,
                "max_evals": optimizer.max_evals,
                "n_evals": result.n_evals,
                "n_solutions": int(result.F.shape[0]),
                "metrics": metrics,
                "output_dir": str(run_dir.relative_to(root)),
            }
        )

    validation_path = root / str(validation_summary)
    validation_path.parent.mkdir(parents=True, exist_ok=True)
    validation_path.write_text(json.dumps(summaries, indent=2), encoding="utf-8")
    return summaries
