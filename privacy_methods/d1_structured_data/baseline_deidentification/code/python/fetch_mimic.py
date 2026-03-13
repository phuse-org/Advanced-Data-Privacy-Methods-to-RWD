"""Download MIMIC-IV data from PhysioNet (requires credentialed access).

Guides users through PhysioNet credentialing, then automates the download
of hospital and ICU module tables via wget or the PhysioNet API.

Usage (CLI):
    # Print credentialing instructions
    python fetch_mimic.py

    # Download after credentialing (will prompt for password)
    python fetch_mimic.py --username YOUR_PHYSIONET_USERNAME

    # Download specific modules only
    python fetch_mimic.py --username USER --modules hosp
    python fetch_mimic.py --username USER --modules icu

    # Download specific tables
    python fetch_mimic.py --username USER --tables patients admissions labevents icustays

Usage (Jupyter):
    from fetch_mimic import fetch_mimic
    dfs = fetch_mimic(username="YOUR_USERNAME", password="YOUR_PASSWORD")

Data source: https://physionet.org/content/mimiciv/3.1/
"""

from __future__ import annotations

import argparse
import csv
import getpass
import gzip
import io
import os
import subprocess
import urllib.error
import urllib.request
from base64 import b64encode
from pathlib import Path

import pandas as pd

# ---------------------------------------------------------------------------
# MIMIC-IV v3.1 table definitions
# ---------------------------------------------------------------------------

MIMIC_VERSION = "3.1"
BASE_URL = f"https://physionet.org/files/mimiciv/{MIMIC_VERSION}"

MODULES: dict[str, dict[str, str]] = {
    "hosp": {
        "patients":           "Patient demographics (age, gender, anchor year)",
        "admissions":         "Hospital admissions and discharges",
        "transfers":          "Intra-hospital unit transfers",
        "diagnoses_icd":      "Billed ICD diagnosis codes",
        "procedures_icd":     "Billed ICD procedure codes",
        "labevents":          "Laboratory test results",
        "prescriptions":      "Medication prescriptions",
        "services":           "Hospital service assignments",
        "drgcodes":           "Diagnosis-Related Group codes",
        "omr":                "Online Medical Record (height, weight, BP, BMI)",
        "d_labitems":         "Dictionary of lab item IDs",
        "d_icd_diagnoses":    "Dictionary of ICD diagnosis codes",
        "d_icd_procedures":   "Dictionary of ICD procedure codes",
    },
    "icu": {
        "icustays":           "ICU stay tracking",
        "chartevents":        "Charted observations (vitals, scores)",
        "inputevents":        "IV fluid and medication inputs",
        "outputevents":       "Patient outputs (urine, drainage)",
        "procedureevents":    "ICU procedures",
        "d_items":            "Dictionary of ICU item IDs",
    },
}


def _print_credentialing_instructions() -> None:
    """Print step-by-step credentialing guide."""
    print("""
=== MIMIC-IV Credentialing Instructions ===

MIMIC-IV requires PhysioNet credentialing (free, but takes a few days).

Step 1: Create a PhysioNet account
  -> https://physionet.org/register/

Step 2: Complete CITI "Data or Specimens Only Research" training
  -> https://about.citiprogram.org/
  -> Affiliate with "Massachusetts Institute of Technology Affiliates"
  -> Complete the course and download the Completion Report (not certificate)

Step 3: Submit credentialing application
  -> https://physionet.org/settings/credentialing/
  -> Upload your CITI Completion Report
  -> Use an institutional email for faster approval

Step 4: Sign the MIMIC-IV Data Use Agreement
  -> https://physionet.org/content/mimiciv/3.1/
  -> Scroll to bottom, sign the DUA
  -> Access is granted immediately after signing

Step 5: Re-run this script:
  python fetch_mimic.py --username YOUR_PHYSIONET_USERNAME

Key restrictions:
  - NO data sharing (each person needs their own access)
  - NO re-identification attempts
  - Report any PHI to PHI-report@physionet.org
  - Cite the dataset in publications
""")


def _download_table(
    module: str,
    table: str,
    username: str,
    password: str,
    dest_dir: Path,
) -> Path | None:
    """Download a single gzipped CSV table from PhysioNet."""
    filename = f"{table}.csv.gz"
    url = f"{BASE_URL}/{module}/{filename}"
    dest = dest_dir / filename

    if dest.exists() and dest.stat().st_size > 0:
        print(f"    cached ({dest.stat().st_size / 1024 / 1024:.1f} MB)")
        return dest

    credentials = b64encode(f"{username}:{password}".encode()).decode()
    headers = {
        "Authorization": f"Basic {credentials}",
        "User-Agent": "MIMIC-fetch/1.0",
    }

    req = urllib.request.Request(url, headers=headers)
    try:
        with urllib.request.urlopen(req, timeout=600) as resp:
            dest.write_bytes(resp.read())
        print(f"    OK ({dest.stat().st_size / 1024 / 1024:.1f} MB)")
        return dest
    except urllib.error.HTTPError as e:
        if e.code == 401:
            print("    FAILED — Invalid credentials. Check username/password.")
        elif e.code == 403:
            print("    FAILED — Access denied. Have you signed the DUA?")
            print("    -> https://physionet.org/content/mimiciv/3.1/")
        else:
            print(f"    FAILED — HTTP {e.code}: {e.reason}")
        return None
    except (urllib.error.URLError, OSError) as e:
        print(f"    FAILED — {e}")
        return None


def fetch_mimic(
    username: str | None = None,
    password: str | None = None,
    modules: list[str] | None = None,
    tables: list[str] | None = None,
    output_dir: str | Path | None = None,
) -> dict[str, pd.DataFrame]:
    """Download MIMIC-IV tables and return DataFrames.

    Args:
        username: PhysioNet username. If None, prints setup instructions.
        password: PhysioNet password. If None, prompts interactively.
        modules: List of modules to download ("hosp", "icu"). None = both.
        tables: Specific table names. Overrides modules if provided.
        output_dir: Where to save files. Defaults to ~/Documents/Data
                    (override with PHUSE_DATA_DIR env var).

    Returns:
        Dict mapping table names to DataFrames.
    """
    if not username:
        _print_credentialing_instructions()
        return {}

    if not password:
        password = getpass.getpass("PhysioNet password: ")

    default_dir = Path(os.environ.get("PHUSE_DATA_DIR", Path.home() / "Documents" / "Data"))
    out_path = Path(output_dir) if output_dir else default_dir

    # Build download list
    to_download: list[tuple[str, str, str]] = []

    if tables:
        # Find which module each table belongs to
        for t in tables:
            found = False
            for mod_name, mod_tables in MODULES.items():
                if t in mod_tables:
                    to_download.append((mod_name, t, mod_tables[t]))
                    found = True
                    break
            if not found:
                print(f"  warning: unknown table '{t}', skipping")
    else:
        selected_modules = modules or list(MODULES.keys())
        for mod_name in selected_modules:
            if mod_name not in MODULES:
                print(f"  warning: unknown module '{mod_name}', skipping")
                continue
            for tbl_name, tbl_desc in MODULES[mod_name].items():
                to_download.append((mod_name, tbl_name, tbl_desc))

    print(f"MIMIC-IV v{MIMIC_VERSION} — downloading {len(to_download)} tables\n")

    dfs: dict[str, pd.DataFrame] = {}

    for mod_name, tbl_name, tbl_desc in to_download:
        cache_dir = out_path / "raw" / "mimic-iv" / mod_name
        cache_dir.mkdir(parents=True, exist_ok=True)

        print(f"  [{mod_name}/{tbl_name}] {tbl_desc}")
        gz_path = _download_table(mod_name, tbl_name, username, password, cache_dir)

        if gz_path is None:
            continue

        try:
            dfs[tbl_name] = pd.read_csv(gz_path, compression="gzip", nrows=None)
            rows = len(dfs[tbl_name])
            cols = len(dfs[tbl_name].columns)
            print(f"    {rows:,} rows, {cols} cols")
        except Exception as e:
            print(f"    READ ERROR — {e}")

    if dfs:
        summary_path = out_path / "mimic_iv_summary.txt"
        with open(summary_path, "w") as f:
            f.write(f"MIMIC-IV v{MIMIC_VERSION} Download Summary\n")
            f.write("=" * 40 + "\n\n")
            for name, df in dfs.items():
                f.write(f"{name}: {len(df):,} rows x {len(df.columns)} cols\n")
                f.write(f"  columns: {', '.join(df.columns[:10])}")
                if len(df.columns) > 10:
                    f.write(f" ... (+{len(df.columns)-10} more)")
                f.write("\n\n")
        print(f"\nSummary written to {summary_path}")

    print(f"\nDownloaded {len(dfs)} tables to {out_path / 'raw' / 'mimic-iv'}")
    return dfs


if __name__ == "__main__":
    parser = argparse.ArgumentParser(description="Download MIMIC-IV data from PhysioNet")
    parser.add_argument("--username", default=None, help="PhysioNet username")
    parser.add_argument("--modules", nargs="*", default=None, choices=["hosp", "icu"],
                        help="Modules to download (default: both)")
    parser.add_argument("--tables", nargs="*", default=None,
                        help="Specific tables (e.g., patients admissions icustays)")
    parser.add_argument("--output-dir", default=None, help="Output directory")
    args, _ = parser.parse_known_args()

    fetch_mimic(username=args.username, modules=args.modules,
                tables=args.tables, output_dir=args.output_dir)
