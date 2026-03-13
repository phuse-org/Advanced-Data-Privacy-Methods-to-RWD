"""Evaluate de-identification quality: privacy risk and data utility.

Metrics:
  Privacy:
    - k-anonymity check on quasi-identifier equivalence classes
    - Direct identifier leakage (should be zero after suppression)
    - Unique-record risk (fraction of patients in equivalence class of size 1)

  Utility:
    - Marginal distribution similarity (Jensen-Shannon divergence)
    - Aggregate statistic preservation (mean/median comparisons)
    - Correlation structure preservation (Pearson correlation delta)

Usage:
    python evaluate_deidentification.py
    python evaluate_deidentification.py --output-dir /path/to/data

    from evaluate_deidentification import evaluate
    results = evaluate(original_mart, deid_mart)
"""

from __future__ import annotations

import argparse
import os
import sys
from pathlib import Path

import numpy as np
import pandas as pd
from scipy.spatial.distance import jensenshannon

# Ensure sibling modules are importable regardless of working directory
_THIS_DIR = str(Path(__file__).resolve().parent)
if _THIS_DIR not in sys.path:
    sys.path.insert(0, _THIS_DIR)

from analyze_synthea import build_analytic_mart, profile_tables
from deidentify_synthea import deidentify_mart, split_public_eval
from fetch_synthea import fetch_synthea

# Core demographic quasi-identifiers for k-anonymity
QI_COLUMNS = ["AGE_BAND", "GENDER", "RACE", "ETHNICITY", "STATE", "ZIP", "MARITAL"]

# Extended QI set: all plausible linkable fields in the released file
RELEASE_QI_COLUMNS = QI_COLUMNS + [
    "first_encounter", "last_encounter",
    "n_encounters", "n_conditions", "n_medications", "n_procedures",
    "primary_organization",
]


# ── Privacy metrics ──────────────────────────────────────────────────────────

def check_direct_id_leakage(deid: pd.DataFrame) -> dict:
    """Check if any direct identifiers survived de-identification."""
    direct_ids = ["SSN", "DRIVERS", "PASSPORT", "FIRST", "LAST",
                  "MAIDEN", "ADDRESS", "BIRTHPLACE"]
    leaked = [c for c in direct_ids if c in deid.columns]
    return {
        "direct_ids_remaining": leaked,
        "leakage_count": len(leaked),
        "passed": len(leaked) == 0,
    }


def compute_k_anonymity(deid: pd.DataFrame, qi_cols: list[str] | None = None) -> dict:
    """Compute k-anonymity on quasi-identifier equivalence classes.

    Returns both EC-level and patient-level metrics:
      EC-level:   what fraction of *distinct QI combinations* are singletons
      Patient-level: what fraction of *patients* sit in small ECs
    """
    cols = qi_cols or QI_COLUMNS
    available = [c for c in cols if c in deid.columns]

    if not available:
        return {"error": "No quasi-identifier columns found", "k_min": 0}

    # Group by QI columns, count equivalence class sizes.
    # observed=True avoids phantom size-0 groups from unused categorical levels
    # (e.g. empty AGE_BAND bins produced by pd.cut).
    ec_sizes = (
        deid
        .groupby(available, dropna=False, observed=True)
        .size()
        .rename("k")
    )

    # EC-level metrics
    k_min = int(ec_sizes.min())
    k_median_ec = float(ec_sizes.median())
    k_max = int(ec_sizes.max())
    n_ecs = len(ec_sizes)
    pct_unique_ec = round(float((ec_sizes == 1).mean() * 100), 1)

    # Patient-level metrics: merge EC size back onto each patient row
    tmp = deid.merge(ec_sizes, left_on=available, right_index=True, how="left")
    pct_patients_small = round(float((tmp["k"] < 5).mean() * 100), 1)
    pct_unique_patients = round(float((tmp["k"] == 1).mean() * 100), 1)
    k_median_patient = float(tmp["k"].median())

    return {
        "qi_columns": available,
        "n_equivalence_classes": n_ecs,
        "k_min": k_min,
        "k_median_ec": round(k_median_ec, 1),
        "k_median_patient": round(k_median_patient, 1),
        "k_max": k_max,
        "pct_unique_ec": pct_unique_ec,
        "pct_unique_patients": pct_unique_patients,
        "pct_patients_in_small_ec": pct_patients_small,
    }


# ── Utility metrics ──────────────────────────────────────────────────────────

def compare_marginals(
    original: pd.DataFrame, deid: pd.DataFrame, columns: list[str] | None = None
) -> pd.DataFrame:
    """Compare marginal distributions using Jensen-Shannon divergence.

    Lower JSD = better utility preservation (0 = identical, 1 = maximally different).
    """
    cols = columns or [c for c in original.columns if c in deid.columns]
    rows = []

    for col in cols:
        if col not in deid.columns:
            continue
        orig_vals = original[col].dropna()
        deid_vals = deid[col].dropna()

        # Both must be numeric to use histogram binning
        both_numeric = (pd.api.types.is_numeric_dtype(orig_vals)
                        and pd.api.types.is_numeric_dtype(deid_vals))
        if both_numeric:
            combined = pd.concat([orig_vals, deid_vals])
            bins = np.linspace(combined.min(), combined.max(), 21)
            p = np.histogram(orig_vals, bins=bins)[0].astype(float)
            q = np.histogram(deid_vals, bins=bins)[0].astype(float)
        else:
            # Categorical: align value counts
            all_cats = sorted(set(orig_vals.unique()) | set(deid_vals.unique()),
                             key=str)
            orig_counts = orig_vals.value_counts().to_dict()
            deid_counts = deid_vals.value_counts().to_dict()
            p = np.array([orig_counts.get(c, 0) for c in all_cats], dtype=float)
            q = np.array([deid_counts.get(c, 0) for c in all_cats], dtype=float)

        # Normalize to probability distributions
        p_sum, q_sum = p.sum(), q.sum()
        if p_sum == 0 or q_sum == 0:
            continue
        p = p / p_sum
        q = q / q_sum

        jsd = float(jensenshannon(p, q))
        rows.append({"column": col, "jsd": round(jsd, 4), "type": str(orig_vals.dtype)})

    return pd.DataFrame(rows).sort_values("jsd", ascending=False)


def compare_aggregates(original: pd.DataFrame, deid: pd.DataFrame) -> pd.DataFrame:
    """Compare mean/median for numeric columns between original and de-identified."""
    numeric_cols = [
        c for c in original.select_dtypes(include="number").columns
        if c in deid.columns and c in deid.select_dtypes(include="number").columns
    ]

    rows = []
    for col in numeric_cols:
        orig_mean = original[col].mean()
        deid_mean = deid[col].mean()
        orig_median = original[col].median()
        deid_median = deid[col].median()
        rows.append({
            "column": col,
            "orig_mean": round(orig_mean, 2),
            "deid_mean": round(deid_mean, 2),
            "mean_pct_change": round(abs(deid_mean - orig_mean) / max(abs(orig_mean), 1e-9) * 100, 1),
            "orig_median": round(orig_median, 2),
            "deid_median": round(deid_median, 2),
        })

    return pd.DataFrame(rows)


def compare_correlations(original: pd.DataFrame, deid: pd.DataFrame) -> dict:
    """Compare correlation matrices between original and de-identified data.

    Returns mean absolute difference in pairwise correlations.
    """
    numeric_cols = [
        c for c in original.select_dtypes(include="number").columns
        if c in deid.select_dtypes(include="number").columns
    ]

    if len(numeric_cols) < 2:
        return {"error": "Not enough numeric columns for correlation comparison"}

    orig_corr = original[numeric_cols].corr()
    deid_corr = deid[numeric_cols].corr()

    # Align matrices
    common = orig_corr.index.intersection(deid_corr.index)
    diff = (orig_corr.loc[common, common] - deid_corr.loc[common, common]).abs()

    # Upper triangle only (exclude diagonal)
    mask = np.triu(np.ones(diff.shape, dtype=bool), k=1)
    upper_diffs = diff.values[mask]

    return {
        "n_pairs": len(upper_diffs),
        "mean_abs_corr_diff": round(float(np.mean(upper_diffs)), 4),
        "max_abs_corr_diff": round(float(np.max(upper_diffs)), 4),
        "median_abs_corr_diff": round(float(np.median(upper_diffs)), 4),
    }


# ── Full evaluation ──────────────────────────────────────────────────────────

def evaluate(original: pd.DataFrame, deid: pd.DataFrame,
             public: pd.DataFrame | None = None) -> dict:
    """Run full evaluation and print results.

    Args:
        original: Original analytic mart.
        deid: Full de-identified mart (eval file, may include Id/org/survival).
        public: Public-release mart (Id/org/survival dropped). If None,
                release QI check runs on deid instead.
    """
    results = {}

    # Privacy
    print("\n=== Privacy Evaluation ===\n")

    leakage = check_direct_id_leakage(deid)
    results["direct_id_leakage"] = leakage
    status = "PASS" if leakage["passed"] else "FAIL"
    print(f"  Direct ID leakage: {status} ({leakage['leakage_count']} remaining)")

    # Core demographic QIs (on eval file)
    k_anon = compute_k_anonymity(deid, qi_cols=QI_COLUMNS)
    results["k_anonymity_core"] = k_anon
    if "error" not in k_anon:
        print(f"  k-anonymity — core demographic QIs ({', '.join(k_anon['qi_columns'])}):")
        print(f"    k_min={k_anon['k_min']}, k_median={k_anon['k_median_ec']}, "
              f"k_max={k_anon['k_max']}")
        print(f"    {k_anon['n_equivalence_classes']} equivalence classes, "
              f"{k_anon['pct_unique_ec']}% are singletons")
        print(f"    {k_anon['pct_unique_patients']}% of patients unique (k=1), "
              f"{k_anon['pct_patients_in_small_ec']}% in small EC (k<5)")

    # Extended release QIs — run on the public file (not eval)
    # so the result reflects actual release risk
    release_frame = public if public is not None else deid
    release_label = "public file" if public is not None else "eval file (conservative)"
    # Filter RELEASE_QI_COLUMNS to those actually present in the release frame
    release_qis = [c for c in RELEASE_QI_COLUMNS if c in release_frame.columns]
    k_anon_release = compute_k_anonymity(release_frame, qi_cols=release_qis)
    results["k_anonymity_release"] = k_anon_release
    if "error" not in k_anon_release:
        print(f"\n  k-anonymity — release QIs on {release_label}:")
        print(f"    QIs used: {', '.join(k_anon_release['qi_columns'])}")
        print(f"    k_min={k_anon_release['k_min']}, "
              f"k_median={k_anon_release['k_median_ec']}, "
              f"k_max={k_anon_release['k_max']}")
        print(f"    {k_anon_release['pct_unique_patients']}% of patients unique (k=1), "
              f"{k_anon_release['pct_patients_in_small_ec']}% in small EC (k<5)")

    # For backwards compat, keep "k_anonymity" pointing to core
    results["k_anonymity"] = k_anon

    # Utility
    print("\n=== Utility Evaluation ===\n")

    # Marginal distribution comparison (Jensen-Shannon divergence)
    marginals = compare_marginals(original, deid)
    results["marginal_comparison"] = marginals
    if not marginals.empty:
        print("  Marginal distribution similarity (Jensen-Shannon divergence):")
        for _, row in marginals.head(10).iterrows():
            print(f"    {row['column']}: JSD={row['jsd']}")
        if len(marginals) > 10:
            print(f"    ... and {len(marginals) - 10} more columns")

    agg = compare_aggregates(original, deid)
    results["aggregate_comparison"] = agg
    if not agg.empty:
        print("\n  Aggregate statistic changes:")
        for _, row in agg.iterrows():
            print(f"    {row['column']}: mean {row['orig_mean']} → {row['deid_mean']} "
                  f"({row['mean_pct_change']}% change)")

    corr = compare_correlations(original, deid)
    results["correlation_preservation"] = corr
    if "error" not in corr:
        print(f"\n  Correlation preservation:")
        print(f"    Mean |Δr| = {corr['mean_abs_corr_diff']} "
              f"(max = {corr['max_abs_corr_diff']})")

    return results


# ── Main ─────────────────────────────────────────────────────────────────────

def main(output_dir: str | None = None) -> dict:
    """Run full pipeline: fetch → mart → deidentify → evaluate."""
    dfs = fetch_synthea(output_dir=output_dir)
    mart = build_analytic_mart(dfs)
    deid = deidentify_mart(mart)
    public, _ = split_public_eval(deid)
    results = evaluate(mart, deid, public=public)

    # Save evaluation summary
    default_dir = Path(os.environ.get("PHUSE_DATA_DIR", Path.home() / "Documents" / "Data"))
    out_path = Path(output_dir) if output_dir else default_dir
    results_dir = out_path / "results" / "baseline_deidentification"
    results_dir.mkdir(parents=True, exist_ok=True)

    # Save aggregate comparison
    if isinstance(results.get("aggregate_comparison"), pd.DataFrame):
        results["aggregate_comparison"].to_csv(
            results_dir / "utility_aggregates.csv", index=False
        )

    # Save marginal comparison
    if isinstance(results.get("marginal_comparison"), pd.DataFrame):
        results["marginal_comparison"].to_csv(
            results_dir / "utility_marginals.csv", index=False
        )

    # Save k-anonymity summaries (core + release)
    for key, fname in [("k_anonymity_core", "k_anonymity_core.csv"),
                       ("k_anonymity_release", "k_anonymity_release.csv")]:
        k = results.get(key, {})
        if k and "error" not in k:
            pd.DataFrame([k]).to_csv(results_dir / fname, index=False)

    print(f"\nResults saved to: {results_dir}")
    return results


if __name__ == "__main__":
    parser = argparse.ArgumentParser(description="Evaluate de-identification quality")
    parser.add_argument("--output-dir", default=None, help="Data directory")
    args, _ = parser.parse_known_args()
    main(output_dir=args.output_dir)
