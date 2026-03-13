"""Task-based utility analyses: survival, regression, and predictive modeling.

Runs each analysis on both original and de-identified data, then compares
results to quantify how much analytic utility the de-identification preserves.

Analyses:
  1. Survival   — Kaplan-Meier curves for encounter span by age group
                   (uses lifelines.KaplanMeierFitter)
  2. Regression — Linear model: healthcare expenses ~ demographics + utilization
  3. Predictive — Logistic regression: high-utilizer classification (AUC, accuracy)

Usage:
    python run_utility_analyses.py
    python run_utility_analyses.py --output-dir /path/to/data

    from run_utility_analyses import run_all_analyses
    results = run_all_analyses(original_mart, deid_mart)
"""

from __future__ import annotations

import argparse
import os
import sys
import warnings
from pathlib import Path

import matplotlib
matplotlib.use("Agg")  # non-interactive backend for saving figures
import matplotlib.pyplot as plt
import numpy as np
import pandas as pd
from lifelines import KaplanMeierFitter
from lifelines.statistics import logrank_test
from sklearn.linear_model import LinearRegression, LogisticRegression
from sklearn.metrics import roc_auc_score, accuracy_score, r2_score
from sklearn.model_selection import cross_val_predict, StratifiedKFold, KFold
from sklearn.preprocessing import LabelEncoder

# Ensure sibling modules are importable regardless of working directory
_THIS_DIR = str(Path(__file__).resolve().parent)
if _THIS_DIR not in sys.path:
    sys.path.insert(0, _THIS_DIR)

from analyze_synthea import build_analytic_mart
from deidentify_synthea import deidentify_mart, COUNT_LABELS
from fetch_synthea import fetch_synthea

warnings.filterwarnings("ignore", category=UserWarning)


# ── Ordered-category mappings ────────────────────────────────────────────────
# Explicit ordinal maps for bucketed columns so models see correct ordering
# (LabelEncoder sorts lexicographically: "100+" < "1-5" < "16-30" — wrong)

COUNT_ORDER = {label: i for i, label in enumerate(COUNT_LABELS)}
# COUNT_LABELS = ["0", "1-5", "6-15", "16-30", "31-50", "51-100", "100+"]

EXPENSE_ORDER = {f"Q{i}": i for i in range(1, 11)}

AGE_ORDER = {f"{i}-{i+4}": i // 5 for i in range(0, 105, 5)}
AGE_ORDER["105+"] = 21

# Columns that should use ordered mappings (not one-hot)
_ORDERED_MAPS = {
    "AGE_BAND": AGE_ORDER,
    "n_encounters": COUNT_ORDER,
    "n_conditions": COUNT_ORDER,
    "n_medications": COUNT_ORDER,
    "n_procedures": COUNT_ORDER,
    "n_observations": COUNT_ORDER,
    "n_unique_conditions": COUNT_ORDER,
    "n_unique_medications": COUNT_ORDER,
    "n_unique_procedures": COUNT_ORDER,
    "n_unique_observations": COUNT_ORDER,
    "n_encounter_types": COUNT_ORDER,
    "HEALTHCARE_EXPENSES": EXPENSE_ORDER,
    "HEALTHCARE_COVERAGE": EXPENSE_ORDER,
}

# Unordered categoricals → one-hot encode
_ONEHOT_COLS = {"GENDER", "RACE", "ETHNICITY", "STATE", "MARITAL"}


# ── Helpers ──────────────────────────────────────────────────────────────────

def _encode_features(df: pd.DataFrame, feature_cols: list[str]) -> pd.DataFrame:
    """Encode features for modeling.

    - Numeric columns pass through as-is.
    - Ordered buckets (count bins, expense deciles, age bands) use explicit
      ordinal mappings so the model sees correct ordering.
    - Unordered categoricals (GENDER, RACE, etc.) are one-hot encoded.
    """
    encoded = pd.DataFrame(index=df.index)
    for col in feature_cols:
        if col not in df.columns:
            continue
        series = df[col]

        if pd.api.types.is_numeric_dtype(series):
            encoded[col] = series.astype(float)
        elif col in _ORDERED_MAPS:
            mapping = _ORDERED_MAPS[col]
            encoded[col] = series.astype(str).map(mapping).astype(float)
        elif col in _ONEHOT_COLS:
            dummies = pd.get_dummies(
                series.astype(str).fillna("__missing__"),
                prefix=col, drop_first=True,
            ).astype(float)
            encoded = pd.concat([encoded, dummies], axis=1)
        else:
            # Fallback: LabelEncoder for anything unexpected
            le = LabelEncoder()
            vals = series.astype(str).fillna("__missing__")
            encoded[col] = le.fit_transform(vals)
    return encoded


def _encode_ordinal_target(series: pd.Series) -> pd.Series:
    """Map a bucketed target column to ordinal values using known mappings."""
    s = series.astype(str)
    for mapping in [EXPENSE_ORDER, COUNT_ORDER, AGE_ORDER]:
        mapped = s.map(mapping)
        if mapped.notna().sum() > len(s) * 0.5:
            return mapped.astype(float)
    # Fallback
    le = LabelEncoder()
    return pd.Series(le.fit_transform(s.fillna("0")), index=series.index).astype(float)


def _safe_feature_cols(df: pd.DataFrame, candidates: list[str]) -> list[str]:
    """Return candidate columns that exist in df."""
    return [c for c in candidates if c in df.columns]


def _get_results_dir(output_dir: str | None = None) -> Path:
    """Return the results directory path."""
    default_dir = Path(os.environ.get("PHUSE_DATA_DIR", Path.home() / "Documents" / "Data"))
    out_path = Path(output_dir) if output_dir else default_dir
    results_dir = out_path / "results" / "baseline_deidentification"
    results_dir.mkdir(parents=True, exist_ok=True)
    return results_dir


# ── 1. Survival Analysis ────────────────────────────────────────────────────

def prepare_survival_data(
    df: pd.DataFrame,
    birth_col: str = "BIRTHDATE",
    death_col: str = "DEATHDATE",
    censor_col: str = "last_encounter",
) -> pd.DataFrame:
    """Build a survival-analysis frame with duration_years and event_observed.

    For the original mart: computes from raw date columns.
    For the de-id mart: uses pre-computed survival_time_years + event_observed
    (created in the de-id pipeline before dates are dropped).
    """
    # If survival columns were pre-computed (de-id path), use them directly
    if "survival_time_years" in df.columns and "event_observed" in df.columns:
        out = df[["survival_time_years", "event_observed"]].copy()
        out = out.rename(columns={"survival_time_years": "duration_years"})
        out = out.loc[out["duration_years"].notna() & (out["duration_years"] >= 0)]
        return out

    # Otherwise compute from date columns (original path)
    def _to_naive_dt(s: pd.Series) -> pd.Series:
        s = pd.to_datetime(s, errors="coerce")
        if hasattr(s.dt, "tz") and s.dt.tz is not None:
            s = s.dt.tz_localize(None)
        return s

    x = df.copy()

    if birth_col not in x.columns:
        return pd.DataFrame(columns=["duration_years", "event_observed"])

    birth = _to_naive_dt(x[birth_col])
    if birth.isna().all():
        return pd.DataFrame(columns=["duration_years", "event_observed"])

    death = _to_naive_dt(x[death_col]) if death_col in x.columns else pd.Series(pd.NaT, index=x.index)
    censor = _to_naive_dt(x[censor_col]) if censor_col in x.columns else pd.Series(pd.NaT, index=x.index)

    x["event_observed"] = death.notna().astype(int)
    end_date = death.fillna(censor)
    x["duration_years"] = (end_date - birth).dt.days / 365.25

    out = x.loc[
        x["duration_years"].notna() & (x["duration_years"] >= 0),
        ["duration_years", "event_observed"],
    ].copy()
    return out


def survival_analysis(original: pd.DataFrame, deid: pd.DataFrame,
                      results_dir: Path | None = None) -> dict:
    """Compare Kaplan-Meier survival curves (original vs de-identified).

    Uses birth-to-death (or birth-to-last-encounter if censored) as duration.
    The de-id pipeline pre-computes survival_time_years from shifted dates
    before dropping BIRTHDATE/DEATHDATE, so the KM curve is faithfully preserved.
    """
    print("\n── Survival Analysis (lifelines KaplanMeierFitter) ──\n")

    results = {}
    km_fitters = {}
    surv_data = {}

    for label, df in [("original", original), ("deid", deid)]:
        sdf = prepare_survival_data(df)
        if sdf.empty:
            print(f"  [{label}] Cannot compute survival duration — skipping")
            continue

        surv_data[label] = sdf

        kmf = KaplanMeierFitter(label=label.capitalize())
        kmf.fit(
            durations=sdf["duration_years"],
            event_observed=sdf["event_observed"],
        )
        km_fitters[label] = kmf

        median_surv = kmf.median_survival_time_
        results[f"median_survival_{label}"] = (
            round(float(median_surv), 1) if np.isfinite(median_surv) else None
        )
        print(f"  [{label}] n={len(sdf)}, "
              f"median survival: {results[f'median_survival_{label}'] or '>max'} years")

        # Stratified by AGE_BAND (top 3 bands by size)
        if "AGE_BAND" in df.columns:
            valid_idx = sdf.index
            band_counts = df.loc[valid_idx, "AGE_BAND"].value_counts()
            top_bands = band_counts.head(3).index.tolist()
            for band in top_bands:
                band_idx = valid_idx[df.loc[valid_idx, "AGE_BAND"] == band]
                if len(band_idx) < 10:
                    continue
                kmf_band = KaplanMeierFitter(label=f"{label} {band}")
                kmf_band.fit(
                    durations=sdf.loc[band_idx, "duration_years"],
                    event_observed=sdf.loc[band_idx, "event_observed"],
                )
                bmed = kmf_band.median_survival_time_
                bmed_val = round(float(bmed), 1) if np.isfinite(bmed) else None
                results[f"median_survival_{label}_{band}"] = bmed_val
                print(f"    AGE_BAND={band}: n={len(band_idx)}, "
                      f"median={bmed_val or '>max'} years")

    # Log-rank test between original and deid
    if "original" in surv_data and "deid" in surv_data:
        lr = logrank_test(
            surv_data["original"]["duration_years"],
            surv_data["deid"]["duration_years"],
            event_observed_A=surv_data["original"]["event_observed"],
            event_observed_B=surv_data["deid"]["event_observed"],
        )
        results["logrank_p_value"] = round(lr.p_value, 4)
        results["logrank_statistic"] = round(float(lr.test_statistic), 2)
        print(f"\n  Log-rank test: statistic={results['logrank_statistic']}, "
              f"p={results['logrank_p_value']}")

    # Compare median survival
    orig_med = results.get("median_survival_original")
    deid_med = results.get("median_survival_deid")
    if orig_med is not None and deid_med is not None:
        diff = abs(orig_med - deid_med)
        results["median_survival_diff_years"] = round(diff, 2)
        print(f"  Median survival difference: {round(diff, 2)} years")
    else:
        print("\n  (Cannot compare — one or both datasets lack survival data)")

    # Save KM plot
    if km_fitters and results_dir:
        fig, axes = plt.subplots(1, 2, figsize=(14, 5))

        # Panel 1: Overall comparison (original vs de-identified)
        ax = axes[0]
        for kmf in km_fitters.values():
            kmf.plot_survival_function(ax=ax, ci_show=True)
        ax.set_title("Kaplan-Meier: Original vs De-identified")
        ax.set_xlabel("Time (years)")
        ax.set_ylabel("Survival probability")
        ax.legend()
        ax.grid(True, alpha=0.3)

        # Panel 2: Stratified by AGE_BAND (original, top 5 bands)
        ax2 = axes[1]
        orig_sdf = surv_data.get("original")
        if orig_sdf is not None:
            valid_idx = orig_sdf.index

            # Derive age bands from BIRTHDATE if AGE_BAND not present
            if "AGE_BAND" in original.columns:
                age_bands = original.loc[valid_idx, "AGE_BAND"]
            elif "BIRTHDATE" in original.columns:
                bd = pd.to_datetime(original.loc[valid_idx, "BIRTHDATE"], errors="coerce")
                last_enc = original.get("last_encounter")
                if last_enc is not None:
                    ref = pd.to_datetime(last_enc, errors="coerce").max()
                    if hasattr(ref, "tz") and ref.tz is not None:
                        ref = ref.tz_localize(None)
                else:
                    ref = pd.Timestamp("2020-04-01")
                if pd.isna(ref):
                    ref = pd.Timestamp("2020-04-01")
                if hasattr(bd.dt, "tz") and bd.dt.tz is not None:
                    bd = bd.dt.tz_localize(None)
                age = ((ref - bd).dt.days / 365.25)
                age_bands = pd.cut(
                    age,
                    bins=list(range(0, 110, 5)) + [200],
                    labels=[f"{i}-{i+4}" for i in range(0, 105, 5)] + ["105+"],
                    right=False,
                )
            else:
                age_bands = None

            if age_bands is not None:
                band_counts = age_bands.value_counts()
                for band in band_counts.head(5).index:
                    band_idx = valid_idx[age_bands == band]
                    if len(band_idx) < 10:
                        continue
                    kmf_b = KaplanMeierFitter(label=str(band))
                    kmf_b.fit(
                        durations=orig_sdf.loc[band_idx, "duration_years"],
                        event_observed=orig_sdf.loc[band_idx, "event_observed"],
                    )
                    kmf_b.plot_survival_function(ax=ax2, ci_show=False)
                ax2.set_title("KM by Age Band (Original)")
                ax2.set_xlabel("Time (years)")
                ax2.set_ylabel("Survival probability")
                ax2.legend(title="AGE_BAND", fontsize=8)
                ax2.grid(True, alpha=0.3)

        plt.tight_layout()
        fig_path = results_dir / "km_survival_curves.png"
        fig.savefig(fig_path, dpi=150, bbox_inches="tight")
        plt.close(fig)
        print(f"  KM plot saved: {fig_path}")

    return results


# ── 2. Regression Analysis ──────────────────────────────────────────────────

def regression_analysis(original: pd.DataFrame, deid: pd.DataFrame) -> dict:
    """Linear regression: healthcare expenses ~ demographics + utilization.

    Both original and de-identified targets are bucketed into the same
    10 quantile bands so R² measures the same ordinal prediction task.
    """
    print("\n── Regression Analysis ──\n")

    target = "HEALTHCARE_EXPENSES"
    feature_candidates = [
        "AGE_BAND", "GENDER", "RACE", "ETHNICITY", "STATE", "MARITAL",
        "n_encounters", "n_conditions", "n_medications", "n_procedures",
        "n_observations", "n_encounter_types",
    ]

    results = {}

    # Compute shared quantile boundaries from the original continuous target
    # so both datasets get the same ordinal scale (0–9).
    orig_target = original.get(target)
    if orig_target is not None and pd.api.types.is_numeric_dtype(orig_target):
        _, bin_edges = pd.qcut(
            orig_target.astype(float), q=10, labels=False,
            retbins=True, duplicates="drop",
        )
    else:
        bin_edges = None

    for label, df in [("original", original), ("deid", deid)]:
        if target not in df.columns:
            print(f"  [{label}] Target '{target}' not found — skipping")
            continue

        # Get target — convert to ordinal decile in both cases
        y_raw = df[target]
        if pd.api.types.is_numeric_dtype(y_raw) and bin_edges is not None:
            # Bucket the continuous target with the same edges
            y = pd.cut(
                y_raw.astype(float), bins=bin_edges,
                labels=False, include_lowest=True,
            ).astype(float)
        elif not pd.api.types.is_numeric_dtype(y_raw):
            # Already bucketed (de-id path) — use ordered mapping
            y = _encode_ordinal_target(y_raw)
        else:
            y = y_raw.astype(float)

        features = _safe_feature_cols(df, feature_candidates)
        X = _encode_features(df, features)

        # Drop rows with NaN in target or features
        valid = X.notna().all(axis=1) & y.notna()
        X_clean = X[valid]
        y_clean = y[valid].values
        encoded_col_names = X_clean.columns.tolist()
        X_clean = X_clean.values

        if len(X_clean) < 50:
            print(f"  [{label}] Too few valid rows ({len(X_clean)}) — skipping")
            continue

        # 5-fold cross-validated predictions
        model = LinearRegression()
        cv = KFold(n_splits=5, shuffle=True, random_state=42)
        y_pred = cross_val_predict(model, X_clean, y_clean, cv=cv)
        r2 = r2_score(y_clean, y_pred)

        # Fit on full data for coefficients
        model.fit(X_clean, y_clean)
        coefs = dict(zip(encoded_col_names, model.coef_))

        results[f"r2_{label}"] = round(r2, 4)
        results[f"coefs_{label}"] = coefs
        results[f"n_{label}"] = len(X_clean)

        print(f"  [{label}] n={len(X_clean)}, R² (5-fold CV) = {r2:.4f}")
        # Top 3 predictors by absolute coefficient
        sorted_coefs = sorted(coefs.items(), key=lambda x: abs(x[1]), reverse=True)
        for name, val in sorted_coefs[:3]:
            print(f"    {name}: β = {val:.2f}")

    # Compare
    r2_orig = results.get("r2_original")
    r2_deid = results.get("r2_deid")
    if r2_orig is not None and r2_deid is not None:
        results["r2_diff"] = round(abs(r2_orig - r2_deid), 4)
        print(f"\n  R² difference: {results['r2_diff']} "
              f"(original={r2_orig:.4f}, deid={r2_deid:.4f})")

    return results


# ── 3. Predictive Modeling ──────────────────────────────────────────────────

def predictive_modeling(original: pd.DataFrame, deid: pd.DataFrame) -> dict:
    """Logistic regression: predict high-utilizer patients.

    Target: n_encounters > threshold → 1, else → 0
    The threshold is the original dataset's median, applied to both datasets
    so the label definition is identical and prevalence shifts reflect only
    information loss from de-identification, not a changed threshold.
    """
    print("\n── Predictive Modeling ──\n")

    feature_candidates = [
        "AGE_BAND", "GENDER", "RACE", "ETHNICITY", "STATE", "MARITAL",
        "n_conditions", "n_medications", "n_procedures",
        "n_unique_conditions", "n_unique_medications",
    ]

    results = {}

    # Compute shared target labels from the original data.
    # We define "high utilizer" using the original's binary split, then
    # transfer that same labeling to de-id patients by index alignment.
    # This ensures identical prevalence is possible and any shift is real.
    orig_enc = original.get("n_encounters")
    if orig_enc is not None and pd.api.types.is_numeric_dtype(orig_enc):
        orig_median = float(orig_enc.median())
        shared_labels = (orig_enc > orig_median).astype(int)
    else:
        shared_labels = None
        orig_median = None

    for label, df in [("original", original), ("deid", deid)]:
        # Build target: high utilizer based on n_encounters
        if "n_encounters" not in df.columns:
            print(f"  [{label}] 'n_encounters' not found — skipping")
            continue

        if label == "original" and shared_labels is not None:
            # Use the labels we already computed
            y = shared_labels
        elif label == "deid" and shared_labels is not None:
            # Transfer labels from original by index (same patients, same order)
            y = shared_labels
        else:
            enc = df["n_encounters"]
            if not pd.api.types.is_numeric_dtype(enc):
                enc = enc.astype(str).map(COUNT_ORDER).astype(float)
            y = (enc > enc.median()).astype(int)

        features = _safe_feature_cols(df, feature_candidates)
        # Remove the target-related column from features
        features = [f for f in features if f != "n_encounters"]
        X = _encode_features(df, features)

        valid = X.notna().all(axis=1) & y.notna()
        X_clean = X[valid].values
        y_clean = y[valid].values

        if len(X_clean) < 50 or y_clean.sum() < 10 or (1 - y_clean).sum() < 10:
            print(f"  [{label}] Insufficient data for classification — skipping")
            continue

        # Stratified 5-fold CV
        model = LogisticRegression(max_iter=1000, random_state=42)
        cv = StratifiedKFold(n_splits=5, shuffle=True, random_state=42)
        y_prob = cross_val_predict(model, X_clean, y_clean, cv=cv, method="predict_proba")[:, 1]
        y_pred = (y_prob >= 0.5).astype(int)

        auc = roc_auc_score(y_clean, y_prob)
        acc = accuracy_score(y_clean, y_pred)

        results[f"auc_{label}"] = round(auc, 4)
        results[f"accuracy_{label}"] = round(acc, 4)
        results[f"n_{label}"] = len(X_clean)
        results[f"prevalence_{label}"] = round(y_clean.mean(), 3)

        print(f"  [{label}] n={len(X_clean)}, prevalence={y_clean.mean():.3f}")
        print(f"    AUC = {auc:.4f}, Accuracy = {acc:.4f}")

    # Compare
    auc_orig = results.get("auc_original")
    auc_deid = results.get("auc_deid")
    if auc_orig is not None and auc_deid is not None:
        results["auc_diff"] = round(abs(auc_orig - auc_deid), 4)
        results["accuracy_diff"] = round(
            abs(results["accuracy_original"] - results["accuracy_deid"]), 4
        )
        print(f"\n  AUC difference: {results['auc_diff']} "
              f"(original={auc_orig:.4f}, deid={auc_deid:.4f})")
        print(f"  Accuracy difference: {results['accuracy_diff']} "
              f"(original={results['accuracy_original']:.4f}, "
              f"deid={results['accuracy_deid']:.4f})")

    return results


# ── Run all ──────────────────────────────────────────────────────────────────

def run_all_analyses(original: pd.DataFrame, deid: pd.DataFrame,
                     results_dir: Path | None = None) -> dict:
    """Run all three utility analyses and return combined results."""
    print("\n" + "=" * 60)
    print("  TASK-BASED UTILITY ANALYSES")
    print("=" * 60)

    results = {}
    results["survival"] = survival_analysis(original, deid, results_dir=results_dir)
    results["regression"] = regression_analysis(original, deid)
    results["predictive"] = predictive_modeling(original, deid)

    # Summary
    print("\n" + "=" * 60)
    print("  SUMMARY")
    print("=" * 60 + "\n")

    rows = []
    surv = results["survival"]
    if "median_survival_diff_years" in surv:
        rows.append({
            "analysis": "Survival (KM median)",
            "original": surv.get("median_survival_original"),
            "deid": surv.get("median_survival_deid"),
            "diff": surv["median_survival_diff_years"],
        })

    reg = results["regression"]
    if "r2_diff" in reg:
        rows.append({
            "analysis": "Regression (R² CV)",
            "original": reg.get("r2_original"),
            "deid": reg.get("r2_deid"),
            "diff": reg["r2_diff"],
        })

    pred = results["predictive"]
    if "auc_diff" in pred:
        rows.append({
            "analysis": "Classification (AUC)",
            "original": pred.get("auc_original"),
            "deid": pred.get("auc_deid"),
            "diff": pred["auc_diff"],
        })

    if rows:
        summary = pd.DataFrame(rows)
        print(summary.to_string(index=False))
        results["summary"] = summary
    else:
        print("  No analyses completed successfully.")

    return results


# ── Main ─────────────────────────────────────────────────────────────────────

def main(output_dir: str | None = None) -> dict:
    """Run full pipeline: fetch → mart → deidentify → analyses."""
    print("Loading Synthea data...")
    dfs = fetch_synthea(output_dir=output_dir)
    mart = build_analytic_mart(dfs)

    print("Running de-identification...")
    deid = deidentify_mart(mart)

    # Results directory
    results_dir = _get_results_dir(output_dir)

    # Run analyses
    results = run_all_analyses(mart, deid, results_dir=results_dir)

    # Save summary
    if "summary" in results:
        results["summary"].to_csv(results_dir / "utility_analyses_summary.csv", index=False)
        print(f"\nSummary saved to: {results_dir / 'utility_analyses_summary.csv'}")

    return results


if __name__ == "__main__":
    parser = argparse.ArgumentParser(description="Run utility analyses on Synthea data")
    parser.add_argument("--output-dir", default=None, help="Data directory")
    args, _ = parser.parse_known_args()
    main(output_dir=args.output_dir)
