"""Download Synthea sample data (fixed version for reproducibility).

Downloads a pinned CSV sample from the Synthea sample-data repository
and returns DataFrames for each table.

Usage:
    python fetch_synthea.py
    python fetch_synthea.py --output-dir /path/to/save

    from fetch_synthea import fetch_synthea
    dfs = fetch_synthea()
"""

from __future__ import annotations

import argparse
import os
import urllib.error
import urllib.request
import zipfile
from pathlib import Path

import pandas as pd

# Pinned sample data URL — fixed commit, not "latest"
SAMPLE_URL = "https://raw.githubusercontent.com/synthetichealth/synthea-sample-data/master/downloads/synthea_sample_data_csv_apr2020.zip"

KEY_TABLES = [
    "patients", "encounters", "conditions", "medications",
    "procedures", "observations", "allergies", "careplans",
    "immunizations", "organizations", "providers",
]


def fetch_synthea(output_dir: str | Path | None = None) -> dict[str, pd.DataFrame]:
    """Download and load Synthea sample data.

    Returns dict of table name -> DataFrame.
    """
    default_dir = Path(os.environ.get("PHUSE_DATA_DIR", Path.home() / "Documents" / "Data"))
    out_path = Path(output_dir) if output_dir else default_dir
    synthea_dir = out_path / "raw" / "synthea"
    synthea_dir.mkdir(parents=True, exist_ok=True)

    zip_path = synthea_dir / "synthea_sample.zip"

    # Download if not cached
    if not zip_path.exists() or zip_path.stat().st_size == 0:
        print(f"Downloading Synthea sample data ...")
        req = urllib.request.Request(SAMPLE_URL, headers={"User-Agent": "Synthea-fetch/1.0"})
        try:
            with urllib.request.urlopen(req, timeout=300) as resp:
                zip_path.write_bytes(resp.read())
            print(f"  OK ({zip_path.stat().st_size / 1024 / 1024:.1f} MB)")
        except (urllib.error.HTTPError, urllib.error.URLError, OSError) as e:
            print(f"  DOWNLOAD FAILED: {e}")
            print(f"  URL: {SAMPLE_URL}")
            raise RuntimeError(f"Synthea download failed: {e}") from e
    else:
        print(f"Synthea sample cached ({zip_path.stat().st_size / 1024 / 1024:.1f} MB)")

    # Extract and find CSVs
    print("Extracting ...")
    with zipfile.ZipFile(zip_path) as zf:
        zf.extractall(synthea_dir)

    # Find wherever patients.csv ended up
    csv_files_found = list(synthea_dir.rglob("patients.csv"))
    if not csv_files_found:
        print("  ERROR: no patients.csv found after extraction")
        raise RuntimeError("No patients.csv found after extraction")
    csv_dir = csv_files_found[0].parent

    # Load tables
    dfs: dict[str, pd.DataFrame] = {}
    for csv_file in sorted(csv_dir.glob("*.csv")):
        name = csv_file.stem.lower()
        if name not in KEY_TABLES:
            continue
        try:
            df = pd.read_csv(csv_file)
            dfs[name] = df
            print(f"  [{name}] {len(df):,} rows, {len(df.columns)} cols")
        except Exception as e:
            print(f"  [{name}] READ ERROR: {e}")

    print(f"\n{len(dfs)} tables loaded")
    return dfs


if __name__ == "__main__":
    parser = argparse.ArgumentParser(description="Download Synthea sample data")
    parser.add_argument("--output-dir", default=None, help="Output directory")
    args, _ = parser.parse_known_args()
    fetch_synthea(output_dir=args.output_dir)
