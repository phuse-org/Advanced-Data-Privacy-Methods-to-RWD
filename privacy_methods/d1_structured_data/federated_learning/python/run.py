"""Benchmark runner scaffold.

This script provides the CLI skeleton for running this benchmark.
It currently writes stub outputs; the assigned group should replace the
body of `main()` with real benchmark logic.

Expected behavior once implemented:
- Accept a dataset name or local path
- Run the baseline and privacy-preserving method(s)
- Write:
    ../results/<dataset>/<run_id>/metrics.json
    ../results/<dataset>/<run_id>/params.json
"""

from __future__ import annotations

import argparse
import json
from pathlib import Path
from datetime import datetime

METHOD_DIR = Path(__file__).resolve().parent.parent


def main() -> None:
    p = argparse.ArgumentParser()
    p.add_argument("--dataset", default="(local)", help="Dataset name (see datasets/datasets.yaml)")
    p.add_argument("--data-path", default="", help="Optional path to local data under data/")
    p.add_argument("--seed", type=int, default=0)
    args = p.parse_args()

    run_id = datetime.utcnow().strftime("%Y%m%dT%H%M%SZ")
    out = METHOD_DIR / "results" / args.dataset / run_id
    out.mkdir(parents=True, exist_ok=True)

    params = vars(args)
    metrics = {"status": "stub", "note": "Replace with real benchmark logic. See README.md for planned methods and metrics."}

    (out / "params.json").write_text(json.dumps(params, indent=2))
    (out / "metrics.json").write_text(json.dumps(metrics, indent=2))
    print(f"Wrote stub outputs to: {out}")


if __name__ == "__main__":
    main()
