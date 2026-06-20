from __future__ import annotations

from pathlib import Path

from experiments.molsrtde_experiments.analysis import load_validation_summary
from experiments.molsrtde_experiments.benchmark import load_config, run_config


def test_config_loading() -> None:
    config = load_config(Path("configs/smoke_test.yaml"))

    assert config["runs"][0]["problem"]["name"] == "zdt1"
    assert config["runs"][0]["algorithm"]["name"] == "MO-LSRTDE"


def test_run_config_writes_outputs_under_requested_base_dir(tmp_path: Path) -> None:
    config = load_config(Path("configs/smoke_test.yaml"))
    summaries = run_config(config, base_dir=tmp_path)

    assert len(summaries) == 1
    output_dir = tmp_path / summaries[0]["output_dir"]
    assert (output_dir / "final_X.csv").exists()
    assert (output_dir / "final_F.csv").exists()
    assert (output_dir / "metrics.json").exists()

    summary_path = tmp_path / "validation" / "smoke_summary.json"
    loaded = load_validation_summary(summary_path)
    assert loaded[0]["n_evals"] == summaries[0]["n_evals"]
