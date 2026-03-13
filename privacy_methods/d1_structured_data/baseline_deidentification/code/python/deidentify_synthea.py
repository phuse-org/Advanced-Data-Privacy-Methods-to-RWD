"""Baseline de-identification methods for Synthea patient data.

Applies five standard techniques:
  1. Suppression    — remove direct identifiers entirely
  2. Generalization — reduce quasi-identifier precision (zip→3-digit, age bands,
                      drop LAT/LON, coarsen dates to year, bucket counts/expenses)
  3. Pseudonymization — replace patient/org IDs with consistent pseudonyms
  4. Date shifting   — shift all dates by a per-patient random offset
  5. Bucketing       — bin high-cardinality numeric fields into ranges

Usage:
    python deidentify_synthea.py
    python deidentify_synthea.py --output-dir /path/to/data

    from deidentify_synthea import deidentify_mart
    from analyze_synthea import build_analytic_mart
    deid = deidentify_mart(mart)
"""

from __future__ import annotations

import argparse
import hashlib
import os
import sys
from pathlib import Path

import numpy as np
import pandas as pd

# Ensure sibling modules are importable regardless of working directory
_THIS_DIR = str(Path(__file__).resolve().parent)
if _THIS_DIR not in sys.path:
    sys.path.insert(0, _THIS_DIR)

from analyze_synthea import build_analytic_mart, print_mart_summary
from fetch_synthea import fetch_synthea

# Columns to suppress entirely (direct identifiers)
SUPPRESS_COLS = [
    "SSN", "DRIVERS", "PASSPORT", "FIRST", "LAST", "MAIDEN",
    "PREFIX", "SUFFIX", "ADDRESS", "BIRTHPLACE",
]

# Columns to pseudonymize (replace with consistent hash-based pseudonym)
PSEUDO_COLS = ["Id", "primary_organization"]

# Date columns eligible for shifting
DATE_COLS = ["BIRTHDATE", "DEATHDATE", "first_encounter", "last_encounter"]

# Utilization count columns to bucket into ranges
COUNT_BUCKET_COLS = [
    "n_encounters", "n_conditions", "n_medications", "n_procedures",
    "n_observations", "n_unique_conditions", "n_unique_medications",
    "n_unique_procedures", "n_unique_observations", "n_encounter_types",
]

# Financial columns to bucket into quantile bands
EXPENSE_COLS = ["HEALTHCARE_EXPENSES", "HEALTHCARE_COVERAGE"]

# Count bucketing scheme: (upper bound exclusive, label)
COUNT_BINS = [0, 1, 6, 16, 31, 51, 101, 10000]
COUNT_LABELS = ["0", "1-5", "6-15", "16-30", "31-50", "51-100", "100+"]


# ── Suppression ──────────────────────────────────────────────────────────────

def suppress(df: pd.DataFrame, columns: list[str] | None = None) -> pd.DataFrame:
    """Drop direct-identifier columns."""
    cols = columns or SUPPRESS_COLS
    to_drop = [c for c in cols if c in df.columns]
    result = df.drop(columns=to_drop)
    print(f"  [suppress] dropped {len(to_drop)} columns: {to_drop}")
    return result


# ── Generalization ───────────────────────────────────────────────────────────

def generalize(df: pd.DataFrame) -> pd.DataFrame:
    """Reduce precision of quasi-identifiers."""
    result = df.copy()

    # ZIP → 3-digit prefix
    if "ZIP" in result.columns:
        result["ZIP"] = result["ZIP"].astype(str).str[:3] + "**"
        print("  [generalize] ZIP → 3-digit prefix")

    # Drop LAT/LON entirely — even 1-decimal (~11 km) is too specific
    # when combined with other QIs
    drop_coords = [c for c in ["LAT", "LON"] if c in result.columns]
    if drop_coords:
        result = result.drop(columns=drop_coords)
        print(f"  [generalize] dropped coordinates: {drop_coords}")

    # BIRTHDATE → 5-year age band, then drop fine-grained dates
    # Use max encounter date as reference for reproducibility
    if "BIRTHDATE" in result.columns:
        bd = pd.to_datetime(result["BIRTHDATE"], errors="coerce")
        if "last_encounter" in result.columns:
            ref = pd.to_datetime(result["last_encounter"], errors="coerce").max()
            if hasattr(ref, "tz") and ref.tz is not None:
                ref = ref.tz_localize(None)
        else:
            ref = pd.Timestamp("2020-04-01")
        if pd.isna(ref):
            ref = pd.Timestamp("2020-04-01")
        age = ((ref - bd).dt.days / 365.25).astype(float)
        result["AGE_BAND"] = pd.cut(
            age,
            bins=list(range(0, 110, 5)) + [200],
            labels=[f"{i}-{i+4}" for i in range(0, 105, 5)] + ["105+"],
            right=False,
        )
        drop_dates = [c for c in ["BIRTHDATE", "DEATHDATE"] if c in result.columns]
        result = result.drop(columns=drop_dates)
        print("  [generalize] BIRTHDATE → AGE_BAND (5-year bands)")
        if drop_dates:
            print(f"  [generalize] dropped date quasi-identifiers: {drop_dates}")

    # Encounter dates → year only (preserves temporal trends, removes day-level specificity)
    for col in ["first_encounter", "last_encounter"]:
        if col in result.columns:
            dt = pd.to_datetime(result[col], errors="coerce")
            result[col] = dt.dt.year
            print(f"  [generalize] {col} → year only")

    # CITY → STATE only (drop CITY, COUNTY, FIPS)
    drop_geo = [c for c in ["CITY", "COUNTY", "FIPS"] if c in result.columns]
    if drop_geo:
        result = result.drop(columns=drop_geo)
        print(f"  [generalize] dropped fine-grained geo: {drop_geo}")

    return result


# ── Bucketing ────────────────────────────────────────────────────────────────

def bucket_counts(df: pd.DataFrame, columns: list[str] | None = None) -> pd.DataFrame:
    """Bin high-cardinality count columns into ranges.

    Reduces re-identification risk from exact utilization counts while
    preserving ordinal relationships for analysis.
    """
    result = df.copy()
    cols = columns or COUNT_BUCKET_COLS

    bucketed = []
    for col in cols:
        if col not in result.columns:
            continue
        result[col] = pd.cut(
            result[col].astype(float),
            bins=COUNT_BINS,
            labels=COUNT_LABELS,
            right=False,
            include_lowest=True,
        ).astype(str)
        bucketed.append(col)

    if bucketed:
        print(f"  [bucket] {len(bucketed)} count columns → range bins: {COUNT_LABELS}")

    return result


def bucket_expenses(df: pd.DataFrame, columns: list[str] | None = None,
                    n_quantiles: int = 10) -> pd.DataFrame:
    """Bin financial columns into quantile bands (deciles by default)."""
    result = df.copy()
    cols = columns or EXPENSE_COLS

    bucketed = []
    for col in cols:
        if col not in result.columns:
            continue
        result[col] = pd.qcut(
            result[col].astype(float),
            q=n_quantiles,
            labels=[f"Q{i+1}" for i in range(n_quantiles)],
            duplicates="drop",
        ).astype(str)
        bucketed.append(col)

    if bucketed:
        print(f"  [bucket] {len(bucketed)} expense columns → {n_quantiles} quantile bands")

    return result


# ── Pseudonymization ─────────────────────────────────────────────────────────

def _pseudonymize_value(val: str, salt: str = "phuse2025") -> str:
    """Generate a deterministic pseudonym from a value."""
    if pd.isna(val):
        return val
    h = hashlib.sha256(f"{salt}:{val}".encode()).hexdigest()[:12]
    return f"PSE-{h}"


def pseudonymize(df: pd.DataFrame, columns: list[str] | None = None) -> pd.DataFrame:
    """Replace ID columns with consistent pseudonyms."""
    result = df.copy()
    cols = columns or PSEUDO_COLS
    for col in cols:
        if col not in result.columns:
            continue
        result[col] = result[col].apply(_pseudonymize_value)
        print(f"  [pseudonymize] {col} → PSE-{{hash}}")
    return result


# ── Date shifting ────────────────────────────────────────────────────────────

def date_shift(
    df: pd.DataFrame,
    id_col: str = "Id",
    date_cols: list[str] | None = None,
    max_shift_days: int = 365,
    seed: int = 42,
) -> pd.DataFrame:
    """Shift dates by a per-patient random offset (±max_shift_days).

    The same offset is applied to all dates for a given patient,
    preserving relative time intervals within a patient.
    """
    result = df.copy()
    cols = date_cols or [c for c in DATE_COLS if c in result.columns]

    rng = np.random.default_rng(seed)
    patient_ids = result[id_col].unique()
    shifts = {
        pid: pd.Timedelta(days=int(rng.integers(-max_shift_days, max_shift_days + 1)))
        for pid in patient_ids
    }
    shift_series = result[id_col].map(shifts)

    for col in cols:
        if col not in result.columns:
            continue
        result[col] = pd.to_datetime(result[col], errors="coerce") + shift_series
        print(f"  [date_shift] {col} shifted ±{max_shift_days} days per patient")

    return result


# ── Full pipeline ────────────────────────────────────────────────────────────

def _compute_survival_columns(df: pd.DataFrame) -> pd.DataFrame:
    """Compute survival_time_years and event_observed from date columns.

    Must be called *after* date_shift (so intervals are preserved) but
    *before* generalize drops BIRTHDATE / DEATHDATE.
    """
    result = df.copy()

    if "BIRTHDATE" not in result.columns:
        return result  # nothing to compute

    def _to_naive_dt(s: pd.Series) -> pd.Series:
        s = pd.to_datetime(s, errors="coerce")
        if hasattr(s.dt, "tz") and s.dt.tz is not None:
            s = s.dt.tz_localize(None)
        return s

    birth = _to_naive_dt(result["BIRTHDATE"])
    if birth.isna().all():
        return result

    death = (_to_naive_dt(result["DEATHDATE"])
             if "DEATHDATE" in result.columns
             else pd.Series(pd.NaT, index=result.index))
    last_enc = (_to_naive_dt(result["last_encounter"])
                if "last_encounter" in result.columns
                else pd.Series(pd.NaT, index=result.index))

    result["event_observed"] = death.notna().astype(int)
    end_date = death.fillna(last_enc)
    result["survival_time_years"] = (end_date - birth).dt.days / 365.25

    # Clamp negative durations (shouldn't happen, but safety)
    result.loc[result["survival_time_years"] < 0, "survival_time_years"] = np.nan

    print("  [survival] computed survival_time_years + event_observed from shifted dates")
    return result


def deidentify_mart(mart: pd.DataFrame) -> pd.DataFrame:
    """Apply all de-identification steps to the analytic mart.

    Order: date_shift → survival columns → suppress → generalize →
           bucket → pseudonymize
    (date shift first because it needs the original Id before pseudonymization;
     survival columns computed from shifted dates before generalize drops them;
     bucketing after generalization so we bucket the right columns)
    """
    print("\n=== De-identification Pipeline ===\n")

    result = date_shift(mart)
    result = _compute_survival_columns(result)
    result = suppress(result)
    result = generalize(result)
    result = bucket_counts(result)
    result = bucket_expenses(result)
    result = pseudonymize(result)

    print(f"\n  Result: {len(result)} patients, {len(result.columns)} features")
    return result


# Columns that should NOT appear in a public release file
# (unique/near-unique identifiers even after pseudonymization)
_EVAL_ONLY_COLS = [
    "Id", "primary_organization",
    "survival_time_years", "event_observed",
]


def split_public_eval(deid: pd.DataFrame) -> tuple[pd.DataFrame, pd.DataFrame]:
    """Split the de-identified mart into public-release and eval-only frames.

    Public: drops pseudonymized IDs and continuous survival columns
            (which are highly identifying).
    Eval:   full de-identified mart for internal utility evaluation.
    """
    drop = [c for c in _EVAL_ONLY_COLS if c in deid.columns]
    public = deid.drop(columns=drop)
    return public, deid


# ── Main ─────────────────────────────────────────────────────────────────────

def main(output_dir: str | None = None) -> pd.DataFrame:
    """Run de-identification on the Synthea analytic mart."""
    dfs = fetch_synthea(output_dir=output_dir)
    mart = build_analytic_mart(dfs)

    deid = deidentify_mart(mart)
    public, eval_df = split_public_eval(deid)

    # Save
    default_dir = Path(os.environ.get("PHUSE_DATA_DIR", Path.home() / "Documents" / "Data"))
    out_path = Path(output_dir) if output_dir else default_dir
    results_dir = out_path / "results" / "baseline_deidentification"
    results_dir.mkdir(parents=True, exist_ok=True)

    public.to_csv(results_dir / "deid_mart_public.csv", index=False)
    eval_df.to_csv(results_dir / "deid_mart_eval.csv", index=False)
    print(f"\nSaved: {results_dir / 'deid_mart_public.csv'}")
    print(f"Saved: {results_dir / 'deid_mart_eval.csv'}")

    return deid


if __name__ == "__main__":
    parser = argparse.ArgumentParser(description="De-identify Synthea analytic mart")
    parser.add_argument("--output-dir", default=None, help="Data directory")
    args, _ = parser.parse_known_args()
    main(output_dir=args.output_dir)
