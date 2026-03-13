"""Profile Synthea CSV tables and build a patient-level analytic mart.

Classifies columns as direct identifiers, quasi-identifiers, or safe,
then builds a flat patient-level table suitable for all four D1 methods:
  - Baseline de-identification
  - Differential privacy
  - Synthetic data generation
  - Federated learning (site-partitioned via ORGANIZATION)

Usage:
    python analyze_synthea.py
    python analyze_synthea.py --output-dir /path/to/data

    from analyze_synthea import profile_tables, build_analytic_mart
    dfs = fetch_synthea()
    report = profile_tables(dfs)
    mart = build_analytic_mart(dfs)
"""

from __future__ import annotations

import argparse
import os
import sys
from pathlib import Path

import numpy as np
import pandas as pd

# Ensure sibling modules are importable regardless of working directory
_THIS_DIR = str(Path(__file__).resolve().parent)
if _THIS_DIR not in sys.path:
    sys.path.insert(0, _THIS_DIR)

from fetch_synthea import fetch_synthea

# ── Identifier classification ────────────────────────────────────────────────

DIRECT_IDENTIFIERS = {
    "SSN", "DRIVERS", "PASSPORT", "FIRST", "LAST", "MAIDEN",
    "PREFIX", "SUFFIX", "Id", "PATIENT", "PAYER",
}

QUASI_IDENTIFIERS = {
    "BIRTHDATE", "DEATHDATE", "RACE", "ETHNICITY", "GENDER",
    "BIRTHPLACE", "ADDRESS", "CITY", "STATE", "COUNTY", "FIPS",
    "ZIP", "LAT", "LON", "MARITAL",
}

SENSITIVE_VALUES = {
    "HEALTHCARE_EXPENSES", "HEALTHCARE_COVERAGE", "INCOME",
}


def classify_column(col: str) -> str:
    """Return 'direct', 'quasi', 'sensitive', or 'safe'."""
    upper = col.upper()
    if upper in {c.upper() for c in DIRECT_IDENTIFIERS}:
        return "direct"
    if upper in {c.upper() for c in QUASI_IDENTIFIERS}:
        return "quasi"
    if upper in {c.upper() for c in SENSITIVE_VALUES}:
        return "sensitive"
    return "safe"


# ── Table profiling ──────────────────────────────────────────────────────────

def profile_tables(dfs: dict[str, pd.DataFrame]) -> pd.DataFrame:
    """Profile every column in every table.

    Returns a DataFrame with one row per (table, column):
        table, column, dtype, n_unique, pct_null, classification
    """
    rows = []
    for table, df in sorted(dfs.items()):
        for col in df.columns:
            rows.append({
                "table": table,
                "column": col,
                "dtype": str(df[col].dtype),
                "n_unique": df[col].nunique(),
                "pct_null": round(df[col].isna().mean() * 100, 1),
                "classification": classify_column(col),
            })
    report = pd.DataFrame(rows)
    return report


def print_profile_summary(report: pd.DataFrame) -> None:
    """Print a concise profile summary grouped by classification."""
    print("\n=== Column Classification Summary ===\n")
    for cls in ["direct", "quasi", "sensitive", "safe"]:
        subset = report[report["classification"] == cls]
        if subset.empty:
            continue
        print(f"  [{cls.upper()}] {len(subset)} columns")
        for _, r in subset.iterrows():
            print(f"    {r['table']}.{r['column']}  "
                  f"(unique={r['n_unique']}, null={r['pct_null']}%)")
        print()

    # Highlight patients table specifically
    patients = report[report["table"] == "patients"]
    if not patients.empty:
        n_direct = (patients["classification"] == "direct").sum()
        n_quasi = (patients["classification"] == "quasi").sum()
        print(f"  patients.csv: {n_direct} direct IDs, "
              f"{n_quasi} quasi-IDs out of {len(patients)} columns\n")


# ── Analytic mart ────────────────────────────────────────────────────────────

def build_analytic_mart(dfs: dict[str, pd.DataFrame]) -> pd.DataFrame:
    """Build a flat patient-level analytic table.

    Joins patients with aggregated encounter, condition, medication,
    procedure, and observation features, plus organization (site) linkage
    for federated learning partitioning.

    Returns one row per patient.
    """
    patients = dfs["patients"].copy()

    # Parse dates
    for col in ["BIRTHDATE", "DEATHDATE"]:
        if col in patients.columns:
            patients[col] = pd.to_datetime(patients[col], errors="coerce")

    # Age at last encounter or today
    encounters = dfs.get("encounters", pd.DataFrame())
    if not encounters.empty:
        encounters["START"] = pd.to_datetime(encounters["START"], errors="coerce")
        encounters["STOP"] = pd.to_datetime(encounters["STOP"], errors="coerce")

        # Encounter counts and span
        # Use STOP for last_encounter so observed follow-up is not shortened
        enc_agg = encounters.groupby("PATIENT").agg(
            n_encounters=("START", "count"),
            first_encounter=("START", "min"),
            last_encounter=("STOP", "max"),
            n_encounter_types=("ENCOUNTERCLASS", "nunique"),
        ).reset_index().rename(columns={"PATIENT": "Id"})
        patients = patients.merge(enc_agg, on="Id", how="left")

        # Primary organization (site) — most frequent org per patient
        if "ORGANIZATION" in encounters.columns:
            primary_org = (
                encounters.groupby("PATIENT")["ORGANIZATION"]
                .agg(lambda x: x.mode().iloc[0] if not x.mode().empty else np.nan)
                .reset_index()
                .rename(columns={"PATIENT": "Id", "ORGANIZATION": "primary_organization"})
            )
            patients = patients.merge(primary_org, on="Id", how="left")

    # Condition counts
    conditions = dfs.get("conditions", pd.DataFrame())
    if not conditions.empty:
        cond_agg = conditions.groupby("PATIENT").agg(
            n_conditions=("CODE", "count"),
            n_unique_conditions=("CODE", "nunique"),
        ).reset_index().rename(columns={"PATIENT": "Id"})
        patients = patients.merge(cond_agg, on="Id", how="left")

    # Medication counts
    medications = dfs.get("medications", pd.DataFrame())
    if not medications.empty:
        med_agg = medications.groupby("PATIENT").agg(
            n_medications=("CODE", "count"),
            n_unique_medications=("CODE", "nunique"),
        ).reset_index().rename(columns={"PATIENT": "Id"})
        patients = patients.merge(med_agg, on="Id", how="left")

    # Procedure counts
    procedures = dfs.get("procedures", pd.DataFrame())
    if not procedures.empty:
        proc_agg = procedures.groupby("PATIENT").agg(
            n_procedures=("CODE", "count"),
            n_unique_procedures=("CODE", "nunique"),
        ).reset_index().rename(columns={"PATIENT": "Id"})
        patients = patients.merge(proc_agg, on="Id", how="left")

    # Observation counts
    observations = dfs.get("observations", pd.DataFrame())
    if not observations.empty:
        obs_agg = observations.groupby("PATIENT").agg(
            n_observations=("CODE", "count"),
            n_unique_observations=("CODE", "nunique"),
        ).reset_index().rename(columns={"PATIENT": "Id"})
        patients = patients.merge(obs_agg, on="Id", how="left")

    # Fill NaN counts with 0
    count_cols = [c for c in patients.columns if c.startswith("n_")]
    patients[count_cols] = patients[count_cols].fillna(0).astype(int)

    return patients


def print_mart_summary(mart: pd.DataFrame) -> None:
    """Print a summary of the analytic mart."""
    print(f"\n=== Analytic Mart ===\n")
    print(f"  {len(mart)} patients, {len(mart.columns)} features\n")

    # Site distribution for federated learning
    if "primary_organization" in mart.columns:
        site_counts = mart["primary_organization"].value_counts()
        print(f"  {len(site_counts)} unique sites (organizations)")
        print(f"  Site size: min={site_counts.min()}, "
              f"median={int(site_counts.median())}, max={site_counts.max()}\n")

    # Feature overview
    print("  Feature summary:")
    for col in ["n_encounters", "n_conditions", "n_medications",
                "n_procedures", "n_observations"]:
        if col in mart.columns:
            print(f"    {col}: mean={mart[col].mean():.1f}, "
                  f"median={mart[col].median():.0f}, "
                  f"max={mart[col].max()}")
    print()


# ── Main ─────────────────────────────────────────────────────────────────────

def main(output_dir: str | None = None) -> tuple[pd.DataFrame, pd.DataFrame]:
    """Run full profiling and mart build pipeline."""
    dfs = fetch_synthea(output_dir=output_dir)

    # Profile
    report = profile_tables(dfs)
    print_profile_summary(report)

    # Build mart
    mart = build_analytic_mart(dfs)
    print_mart_summary(mart)

    # Save outputs
    default_dir = Path(os.environ.get("PHUSE_DATA_DIR", Path.home() / "Documents" / "Data"))
    out_path = Path(output_dir) if output_dir else default_dir
    results_dir = out_path / "results" / "baseline_deidentification"
    results_dir.mkdir(parents=True, exist_ok=True)

    report.to_csv(results_dir / "column_profile.csv", index=False)
    mart.to_csv(results_dir / "analytic_mart.csv", index=False)
    print(f"Saved: {results_dir / 'column_profile.csv'}")
    print(f"Saved: {results_dir / 'analytic_mart.csv'}")

    return report, mart


if __name__ == "__main__":
    parser = argparse.ArgumentParser(description="Profile Synthea data and build analytic mart")
    parser.add_argument("--output-dir", default=None, help="Data directory")
    args, _ = parser.parse_known_args()
    main(output_dir=args.output_dir)
