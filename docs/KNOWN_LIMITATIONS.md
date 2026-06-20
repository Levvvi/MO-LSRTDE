# Known Limitations

## Algorithm Extensions

- Legacy RE/CRE adapters, baseline orchestration, and full paper-analysis utilities are preserved locally but not stabilized in the installable package.
- The public package currently exposes the migrated MO-LSRTDE core, ZDT smoke benchmarks, the thin-film problem, and dependency-light metrics.

## Benchmarks Not Yet Migrated

- DTLZ, WFG, RE, and CRE families from the legacy workspace need dedicated package modules and tests before they should be treated as supported APIs.
- Optional `pymoo` baselines are declared as an extra, but baseline experiment jobs are not part of the core smoke runner yet.

## Tests Still Missing

- Tests cover imports, smoke behavior, result shapes, bounds, determinism, budget sanity, nondominated archives, metrics, problem loading, and config loading.
- They do not yet prove numerical equivalence against every historical run or archived full benchmark table.

## Excluded Results and Artifacts

- Large raw result folders are intentionally excluded from Git and preserved locally under ignored paths.
- Current generated figures and tables are ignored until their regeneration path is documented and reviewed.
- Manuscript drafts, Word/PDF files, and publication-risk paper artifacts are intentionally excluded from the public branch.
- Complete file-level inventory manifests are local-only under ignored `private/inventory/`; public inventory files are redacted aggregates.

