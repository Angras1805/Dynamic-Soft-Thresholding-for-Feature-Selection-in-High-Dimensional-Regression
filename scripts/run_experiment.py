from __future__ import annotations

import argparse
import json
from pathlib import Path

from src.config import ExperimentConfig
from src.experiments import run_all_experiments


def parse_args() -> argparse.Namespace:
    p = argparse.ArgumentParser(description="Run sparse regression benchmarks.")
    p.add_argument("--data_dir", type=str, required=True, help="Directory containing train.csv")
    p.add_argument("--out_dir", type=str, required=True, help="Output directory for results")
    return p.parse_args()


def main() -> None:
    args = parse_args()
    cfg = ExperimentConfig()

    out_dir = Path(args.out_dir)
    out_dir.mkdir(parents=True, exist_ok=True)

    results = run_all_experiments(
        data_dir=Path(args.data_dir),
        out_dir=out_dir,
        cfg=cfg,
    )

    (out_dir / "run_config.json").write_text(json.dumps(cfg.__dict__, indent=2), encoding="utf-8")
    (out_dir / "results.csv").write_text(results.to_csv(index=False), encoding="utf-8")


if __name__ == "__main__":
    main()

