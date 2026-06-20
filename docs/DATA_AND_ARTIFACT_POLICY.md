# Data and Artifact Policy

This repository tracks source code, tests, configs, reproducibility notes, and small reviewable validation artifacts.

## Tracked by Default

- Package source under `src/molsrtde/`
- Experiment orchestration under `experiments/`
- Config files under `configs/`
- Tests under `tests/`
- Reproducibility and policy documentation under `docs/`
- Compact validation summaries under `validation/`
- Small public figures or CSV tables only after review

## Regenerated or Ignored

- Raw benchmark outputs are written under `results/` and ignored, except `results/.gitkeep`.
- Legacy raw result folders are preserved under `results/legacy/` and ignored.
- Build outputs, virtual environments, Python caches, IDE state, and temporary audit outputs are ignored.
- Large or binary spreadsheet outputs are not tracked by default.

## Archived and Not Public by Default

- Manuscript drafts, Word documents, publisher PDFs, reviewer-response files, and copyright-risk paper artifacts are preserved under `archive/papers/` and ignored.
- Legacy root scripts and old mixed research code are preserved under `archive/legacy_code/` and ignored unless they are deliberately migrated with tests.
- Unknown files are preserved under `archive/legacy_unclassified/` and ignored until a human classifies them.
- Generated legacy audit outputs are preserved under `archive/legacy_audit/` or `archive/legacy_codex_output/` and ignored.
- Complete file-level inventory manifests are preserved under `private/inventory/` and ignored because they include private/manuscript paths and local workspace metadata. Public inventory files under `inventory/` are redacted aggregates.

## Reproducibility Rule

Public files should be either source inputs needed to reproduce a result or compact outputs that are small, rights-safe, and documented. Large raw outputs and private or publication-risk artifacts must be regenerated locally or restored from private storage, not committed to the public repository.
