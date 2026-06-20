# MO-LSRTDE Restructure Inventory Summary

This public summary is redacted. Complete file-level manifests were produced before file moves, used during restructuring, and preserved locally under `private/inventory/`, which is ignored by Git because those manifests contain private/manuscript paths and local workspace metadata.

## Tickets Completed

- Ticket 00: initialized Git and created `chore/restructure-molsrtde`.
- Ticket 01: created and ran `tools/audit_inventory.py`; complete local manifests were generated before moves.
- Ticket 02: created the standard package skeleton and project metadata.
- Ticket 03: migrated the MO-LSRTDE core into `src/molsrtde` with the `MOLSRTDE` class API.
- Ticket 04: added packaged problems and metrics.
- Ticket 05: added config-driven experiment runner outside the installed package.
- Ticket 06: applied data/artifact policy and preserved risky or raw artifacts under ignored local paths.
- Ticket 07: added pytest regression tests.
- Ticket 08: added Ruff, coverage, build config, and GitHub Actions CI.
- Ticket 09: updated README, reproducibility docs, and quickstart.
- Ticket 10: documented paper reproduction status and notebook inspection.
- Ticket 11: validated, staged public-safe files, and committed locally.
- Ticket 12: pushed a PR-compatible branch and opened draft PR #1.
- Ticket 13: added known limitations and began post-upload verification.

## Public Inventory Aggregates

- Total files scanned locally: 49,785
- Total bytes scanned locally: 1,047,877,618
- Generated raw result files: 34,586, ignored
- Environment files: 14,872, ignored
- Paper/manuscript files: 9, ignored and preserved locally
- Unknown files: 8, ignored and preserved locally for review
- Public package source files: 15
- Public tests: 5

See `inventory/files.csv` and `inventory/files.json` for redacted aggregate counts.

## Validation

Final local validation passed:

- `python -m pip install -e ".[dev,analysis,baselines]"`
- `ruff check src\molsrtde tests scripts\run_experiment.py`
- `pytest -q`
- `pytest --cov=src/molsrtde --cov-report=term-missing --cov-fail-under=85`
- `python -m build`
- `python scripts\run_experiment.py --config configs\smoke_test.yaml`

Coverage reached the configured 85% gate at 86.39% locally.

## Remaining Risk

- Full legacy RE/CRE benchmark and plotting workflows are preserved locally but not stabilized in the package.
- Current generated figures and tables are ignored until their reproduction paths are documented and reviewed.
- Build succeeds with non-blocking setuptools license metadata deprecation warnings.
- CI was queued after the initial PR push and should be checked before marking the PR ready for review.

