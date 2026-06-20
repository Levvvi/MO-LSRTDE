# Reproducibility

This repository is structured so the installable package, experiment runner, and smoke validation can be reproduced from source.

## Environment Setup

Windows PowerShell:

```powershell
python -m venv .venv
.\.venv\Scripts\Activate.ps1
python -m pip install --upgrade pip
python -m pip install -e ".[dev,analysis,baselines]"
```

Unix shell:

```bash
python -m venv .venv
source .venv/bin/activate
python -m pip install --upgrade pip
python -m pip install -e ".[dev,analysis,baselines]"
```

For a minimal package-only install:

```bash
python -m pip install -e .
```

## Validation Commands

```bash
ruff check src/molsrtde tests scripts/run_experiment.py
pytest -q
pytest --cov=src/molsrtde --cov-report=term-missing --cov-fail-under=85
python -m build
python examples/quickstart.py
python scripts/run_experiment.py --config configs/smoke_test.yaml
```

## Smoke Experiment

The smoke config is `configs/smoke_test.yaml`. It runs a cheap deterministic ZDT1 case:

- problem: `zdt1`
- variables: `5`
- population: `10`
- max evaluations: `30`
- seed: `0`

Raw outputs are written to:

```text
results/smoke/zdt1/MO-LSRTDE/run_0/
```

The compact validation summary is written to:

```text
validation/smoke_summary.json
```

## Regenerating Full Results

Full legacy benchmark outputs are intentionally not committed. Use the packaged configs as templates:

```bash
python scripts/run_experiment.py --config configs/benchmark_zdt.yaml
```

Legacy full-result folders are preserved under ignored paths such as `results/legacy/` for local reference only. Additional benchmark families from the legacy workspace should be migrated into `src/molsrtde/problems/` or `experiments/` only after their behavior is protected by tests.

## Excluded From Git

The following are intentionally excluded from public tracking:

- raw result folders under `results/`
- legacy raw benchmark outputs under `results/legacy/`
- archived legacy code and unknown files under `archive/`
- private files under `private/`
- external datasets under `external/`
- virtual environments, caches, build outputs, and IDE state
- Word documents, publisher PDFs, manuscript drafts, and other publication-risk artifacts
- large spreadsheet artifacts unless deliberately converted to small reproducible CSV tables

See `docs/DATA_AND_ARTIFACT_POLICY.md` for the detailed policy.

