# Paper Reproduction Status

The old workspace contained manuscript drafts, generated figures, compact tables, raw result folders, and legacy plotting/statistical scripts. This restructuring keeps those materials preserved but separates them from the installable package.

## Current Public Candidates

Small figures and CSV tables remain under:

- `figures/`
- `figures/legacy/`
- `figures/selected_for_manuscript/`
- `tables/`

These files are review candidates, not benchmark claims. They should be committed only if they are rights-safe, small, and tied to a documented regeneration path.

## Archived Material

The following are preserved but ignored by Git:

- manuscript and paper drafts: `archive/papers/`
- legacy plotting and post-processing code: `archive/legacy_code/visualization.py`
- legacy thin-film post-processing code: `archive/legacy_code/postprocess_thinfilm_thickness_bars.py`
- old full experiment runner: `archive/legacy_code/run_experiments.py`
- raw benchmark outputs: `results/legacy/`

## Notebook Status

No Jupyter notebooks were found during Ticket 10 inspection, so no output stripping was required.

## Regeneration Status

The current reproducible public path is the smoke runner:

```bash
python scripts/run_experiment.py --config configs/smoke_test.yaml
```

Full paper figures and statistical tables are not yet regenerated end-to-end from the new package layout. Before making paper claims in the README, migrate the needed legacy plotting/statistical code into `experiments/molsrtde_experiments/`, add tests or smoke checks for the inputs and outputs, and document the exact command sequence here.

