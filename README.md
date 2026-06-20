# MO-LSRTDE

[![CI](https://github.com/Levvvi/MO-LSRTDE/actions/workflows/ci.yml/badge.svg)](https://github.com/Levvvi/MO-LSRTDE/actions/workflows/ci.yml)

MO-LSRTDE is a Python package for a multi-objective local-search/restart differential evolution optimizer for real-valued multi-objective minimization. The current package exposes the migrated core optimizer, dependency-light ZDT and thin-film benchmark problems, metrics, tests, and a small reproducible experiment runner.

## Installation From Source

```powershell
git clone https://github.com/Levvvi/MO-LSRTDE.git
cd MO-LSRTDE
python -m pip install -e ".[dev,analysis,baselines]"
```

For a minimal local install without development tools:

```powershell
python -m pip install -e .
```

## Minimal Python Quickstart

```python
from molsrtde.algorithm import MOLSRTDE
from molsrtde.problems import ZDT1

problem = ZDT1(n_var=5)
result = MOLSRTDE(pop_size=10, max_evals=30, seed=0).run(problem)
print(result.F.shape)
```

## CLI Experiment Example

```powershell
python scripts/run_experiment.py --config configs/smoke_test.yaml
```

The smoke config writes raw outputs under `results/smoke/` and a compact validation summary under `validation/smoke_summary.json`.

## What's Inside

- `MOLSRTDE`: class API for the migrated optimizer
- Legacy-compatible function wrappers: `mo_lsrtde`, `multi_objective_LSRTDE`, and `multi_objective_LSRTDE_ZDT3_DTLZ3`
- Packaged problems: ZDT1, ZDT2, ZDT3, and the thin-film anti-reflection problem
- Metrics: hypervolume for two objectives, IGD, IGD+, GD, spread, and nondominated masks
- Config-driven smoke runner outside the installable package
- Inventory manifests documenting preserved legacy files and artifact policy decisions

## Project Layout

```text
MO-LSRTDE/
  src/molsrtde/                 # installable package
  experiments/molsrtde_experiments/
  tests/
  configs/
  scripts/run_experiment.py
  examples/quickstart.py
  results/.gitkeep              # raw outputs ignored under results/
  validation/
  docs/
  inventory/
```

## Testing and Coverage

```powershell
ruff check src/molsrtde tests scripts/run_experiment.py
pytest -q
pytest --cov=src/molsrtde --cov-report=term-missing --cov-fail-under=85
python -m build
```

The current local restructuring validation reaches the configured 85% coverage gate.

## Reproducibility

See `REPRODUCIBILITY.md` for environment setup, smoke-run commands, output locations, and excluded artifacts.

## License

This project is licensed under the MIT License. See `LICENSE`.

