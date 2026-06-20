"""Run MO-LSRTDE experiments from a config file."""

from __future__ import annotations

import argparse
import sys
from pathlib import Path

ROOT = Path(__file__).resolve().parents[1]
SRC = ROOT / "src"
if str(ROOT) not in sys.path:
    sys.path.insert(0, str(ROOT))
if str(SRC) not in sys.path:
    sys.path.insert(0, str(SRC))

from experiments.molsrtde_experiments.benchmark import load_config, run_config  # noqa: E402


def parse_args() -> argparse.Namespace:
    parser = argparse.ArgumentParser(description="Run a MO-LSRTDE experiment config.")
    parser.add_argument("--config", required=True, help="Path to a JSON-compatible YAML config file.")
    return parser.parse_args()


def main() -> None:
    args = parse_args()
    config = load_config(args.config)
    summaries = run_config(config, base_dir=ROOT)
    print(f"completed_runs={len(summaries)}")
    for summary in summaries:
        print(
            "{problem} run={run_id} seed={seed} n_evals={n_evals} output={output_dir}".format(
                **summary
            )
        )


if __name__ == "__main__":
    main()
