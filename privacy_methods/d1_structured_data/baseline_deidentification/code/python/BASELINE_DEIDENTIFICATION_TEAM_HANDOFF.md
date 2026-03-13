# PHUSE D1 Structured Data — Baseline De-identification: Review & Team Handoff

**Date**: 2026-03-13
**Status**: Baseline prototype — ready to present as a skeleton to proceed

---

## Executive Summary

This is a **reproducible baseline de-identification benchmark on Synthea synthetic patient data**. It demonstrates the full end-to-end workflow — ingest, profile, build analytic mart, de-identify, evaluate privacy, and evaluate utility — and surfaces an important finding:

> **Removing direct identifiers is not enough.** Even after standard generalization, the combination of demographics + bucketed utilization counts + year-level encounter dates is enough to uniquely identify **99.3% of patients** in the public release file.

### What this is ready to be

- An end-to-end reference workflow for the D1 working group
- A first benchmark on a structured healthcare dataset
- A demonstration that direct-ID removal is insufficient
- A starting point for extending the same framework to NHANES

### What this is not yet

- A privacy-safe public release
- A completed benchmark suite
- A final utility evaluation framework
- A validated production de-identification system

---

## Dataset

### Dataset Type

This baseline uses the **Synthea sample CSV data** — a synthetic, longitudinal, multi-table, EHR-like structured dataset. It is a strong prototype dataset for developing the pipeline because it contains person-level records, longitudinal dates, direct and quasi-identifiers, utilization patterns, site information, and multiple clinically meaningful linked tables.

### Raw Data (11 Synthea CSV Tables)

| Table | Rows | Columns | Purpose in Pipeline |
|---|---:|---:|---|
| patients | 1,171 | 25 | Base person-level table |
| encounters | 53,346 | 15 | Encounter counts, first/last dates, site linkage |
| conditions | 8,376 | 6 | Condition counts |
| medications | 42,989 | 13 | Medication counts |
| procedures | 34,981 | 8 | Procedure counts |
| observations | 299,697 | 8 | Observation counts |
| allergies | 597 | 6 | Profiled only |
| careplans | 3,483 | 9 | Profiled only |
| immunizations | 15,478 | 6 | Profiled only |
| organizations | 1,119 | 11 | Profiled only / site context |
| providers | 5,855 | 12 | Profiled only |

### Why This Dataset Is Suitable for a Baseline

- Direct IDs exist and can be removed
- Quasi-identifiers exist and can be generalized
- Dates and geography create linkage risk
- Site linkage creates a realistic federated-learning partitioning variable
- Downstream analyses can be run on a patient-level mart

---

## Pipeline Overview

```
fetch_synthea.py → analyze_synthea.py → deidentify_synthea.py → evaluate_deidentification.py → run_utility_analyses.py
```

| Step | Script | What It Does |
|------|--------|-------------|
| 1. Fetch | `fetch_synthea.py` | Downloads and caches the Synthea CSV sample (~8.6 MB, 11 tables) |
| 2. Profile & Mart | `analyze_synthea.py` | Classifies columns (direct/quasi/sensitive/safe), builds a flat patient-level analytic mart (1,171 patients, 38 features) |
| 3. De-identify | `deidentify_synthea.py` | Applies 5 techniques: date shifting, suppression, generalization, bucketing, pseudonymization |
| 4. Privacy Eval | `evaluate_deidentification.py` | Checks direct-ID leakage, k-anonymity (core + release QIs), marginal distributions (JSD), aggregates, correlations |
| 5. Utility Eval | `run_utility_analyses.py` | Kaplan-Meier survival, linear regression (R²), logistic classification (AUC) |

---

## Step-by-Step: What the Pipeline Does

### 1. Profiling and Column Classification

`analyze_synthea.py` profiles every column across the source tables and classifies them as:

- **Direct identifiers**: `Id`, `SSN`, `DRIVERS`, `PASSPORT`, `FIRST`, `LAST`, `MAIDEN`, `PATIENT`
- **Quasi-identifiers**: `BIRTHDATE`, `DEATHDATE`, `RACE`, `ETHNICITY`, `GENDER`, `ZIP`, `CITY`, `COUNTY`, `LAT`, `LON`, `MARITAL`
- **Sensitive**: `HEALTHCARE_EXPENSES`, `HEALTHCARE_COVERAGE`
- **Safe**: everything else

This makes assumptions explicit rather than burying them in code.

### 2. Analytic Mart Creation

`analyze_synthea.py` builds a flat patient-level analytic mart: **1,171 rows × 38 columns**.

The mart aggregates longitudinal events into features:
- `n_encounters`, `first_encounter`, `last_encounter`, `n_encounter_types`
- `n_conditions`, `n_unique_conditions`
- `n_medications`, `n_unique_medications`
- `n_procedures`, `n_unique_procedures`
- `n_observations`, `n_unique_observations`
- `primary_organization` (most frequent org per patient — site linkage for federated learning)

### 3. Baseline De-identification

`deidentify_synthea.py` applies the following transformations **in order**:

1. **Date shifting** — All dates (`BIRTHDATE`, `DEATHDATE`, `first_encounter`, `last_encounter`) shifted by the same random offset per patient (±365 days). Preserves within-patient intervals.
2. **Survival-variable creation** — `survival_time_years` + `event_observed` computed from shifted dates **before** dropping raw dates.
3. **Suppression** — Drops 10 direct identifiers: `SSN`, `DRIVERS`, `PASSPORT`, `FIRST`, `LAST`, `MAIDEN`, `PREFIX`, `SUFFIX`, `ADDRESS`, `BIRTHPLACE`
4. **Generalization**:
   - `ZIP` → 3-digit prefix (e.g. `101**`)
   - Drops coordinates (`LAT`, `LON`)
   - `BIRTHDATE` → `AGE_BAND` (5-year bands, reference = max encounter date)
   - Drops `BIRTHDATE` and `DEATHDATE`
   - `first_encounter` and `last_encounter` → year only
   - Drops fine-grained geography (`CITY`, `COUNTY`, `FIPS`)
5. **Bucketing**:
   - 10 utilization counts → ordinal bins: `0`, `1-5`, `6-15`, `16-30`, `31-50`, `51-100`, `100+`
   - 2 expense columns → quantile bands (`Q1`–`Q10`)
6. **Pseudonymization** — `Id` and `primary_organization` → deterministic `PSE-{hash}` values
7. **Split into public vs evaluation file**:
   - `deid_mart_public.csv` (1,171 × 21): drops `Id`, `primary_organization`, `survival_time_years`, `event_observed`
   - `deid_mart_eval.csv` (1,171 × 25): retains everything for internal evaluation

### 4. Privacy Evaluation

`evaluate_deidentification.py` runs three checks:

#### Direct-ID Leakage: **PASS** (0 remaining)

All direct identifiers successfully removed.

#### Core Demographic QI k-anonymity

QIs used: `AGE_BAND`, `GENDER`, `RACE`, `ETHNICITY`, `STATE`, `ZIP`, `MARITAL`

| Metric | Value |
|--------|-------|
| Equivalence classes | 719 |
| k_min | 1 |
| k_max | 17 |
| % of ECs that are singletons | 81.5% |
| % of patients unique (k=1) | 50.0% |
| % of patients in small EC (k<5) | 71.2% |

**Interpretation**: Direct IDs were removed correctly, but baseline generalization is not strong enough for release-quality anonymity.

#### Release QI k-anonymity (on actual public file)

QIs used: core demographics + `first_encounter`, `last_encounter`, `n_encounters`, `n_conditions`, `n_medications`, `n_procedures`

| Metric | Value |
|--------|-------|
| Equivalence classes | 1,167 |
| k_min | 1 |
| k_max | 2 |
| % of patients unique (k=1) | **99.3%** |
| % of patients in small EC (k<5) | **100.0%** |

**Key finding**: The public release file is **not** privacy-safe. Adding utilization counts and year-level encounter dates makes nearly all patients re-identifiable. This is the most important privacy message — it motivates stronger generalization in future iterations.

### 5. Utility Evaluation

Two layers of utility evaluation:

#### Layer A: Generic Utility Files (placeholders)

- `utility_marginals.csv` — Jensen-Shannon divergence per column. Currently not very informative because it compares original continuous variables against bucketed/categorical de-identified versions.
- `utility_aggregates.csv` — Mean/median comparisons. Currently sparse because after bucketing there are few shared numeric columns.

These are kept as placeholders but should **not** be emphasized in presentations.

#### Layer B: Task-Based Utility Analyses (the strong part)

| Analysis | Original | De-identified | Difference | Notes |
|----------|----------|---------------|------------|-------|
| **Survival (KM median)** | 84.1 years | 84.1 years | **0.0** | Perfectly preserved — date shifting keeps intervals intact |
| **Regression (R² CV)** | 0.4113 | 0.5194 | 0.1081 | Both on same ordinal decile scale; illustrative, not final |
| **Classification (AUC)** | 0.9730 | 0.9642 | **0.0088** | Same target labels applied to both; prevalence identical (0.486) |

---

## The Three Analytical Methods — Detailed Explanation for Replication

Each method answers a specific question about how de-identification affects a downstream analysis that a researcher would actually run on this Synthea patient cohort.

### Method 1: Kaplan-Meier Survival Analysis — "Does patient lifespan estimation survive date shifting?"

**Clinical question**: In this cohort of 1,171 synthetic patients (89 deceased, 1,082 censored), what is the overall birth-to-death survival curve, and does it change after all dates are shifted by a random per-patient offset of ±365 days?

**What the data looks like**:

| Column | Original mart | De-identified mart |
|--------|--------------|-------------------|
| Birth date | `BIRTHDATE` (raw, e.g. `1950-02-18`) | Dropped — converted to `AGE_BAND` |
| Death date | `DEATHDATE` (raw or NaT for living patients) | Dropped |
| Censor date | `last_encounter` = max(`STOP`) across all encounters | Dropped — converted to year only |
| Duration | Not stored — computed at analysis time | `survival_time_years` (pre-computed from shifted dates, e.g. `68.4`) |
| Event flag | Not stored — derived from `DEATHDATE` presence | `event_observed` (1 = deceased, 0 = censored) |

**Why pre-computation matters**: The de-id pipeline drops `BIRTHDATE` and `DEATHDATE` during generalization (step 4). If `survival_time_years` were computed *after* dropping, it would be impossible. So the pipeline computes it in step 2, from the *shifted* dates, before they are removed. Because the shift is the same per patient, `(shifted_death - shifted_birth) == (original_death - original_birth)`, so duration is perfectly preserved.

**Implementation** (`run_utility_analyses.py`):
```python
# For original: compute from raw dates
duration = (DEATHDATE.fillna(last_encounter) - BIRTHDATE).dt.days / 365.25
event    = DEATHDATE.notna().astype(int)

# For de-id: use pre-computed fields
duration = df["survival_time_years"]
event    = df["event_observed"]

# Both: fit KM
kmf = KaplanMeierFitter()
kmf.fit(durations=duration, event_observed=event)
```

**Results on this Synthea cohort**:
- Both curves yield **median survival = 84.1 years**
- Log-rank test: statistic = 0.0, p = 1.0 (curves are statistically identical)
- The KM plot (`km_survival_curves.png`, left panel) shows the two curves overlapping perfectly

**Why this is the strongest utility proof point**: Date shifting is one of the few de-identification techniques that provably preserves a specific analysis type. This Synthea cohort demonstrates that clearly because 89 out of 1,171 patients have a death event, giving enough signal for a meaningful KM curve that is then perfectly reproduced.

**Right panel — age-stratified KM**: The original mart has `BIRTHDATE` but not `AGE_BAND`, so the pipeline derives 5-year age bands using the same logic as the de-id step (reference date = max encounter date). The top 5 age bands by patient count are plotted separately, showing younger cohorts have higher survival probability — a basic sanity check.

**NHANES limitation**: This Synthea cohort has longitudinal encounter data spanning years per patient. NHANES is cross-sectional — each respondent is measured once per cycle. A KM analysis on NHANES only makes sense if you use the CDC's linked mortality follow-up files (NDI linkage). Without mortality linkage, replace this method with a different utility task (e.g. weighted subgroup prevalence).

**Replication steps**:
1. Run `analyze_synthea.py` → builds `analytic_mart.csv` (1,171 patients with `BIRTHDATE`, `DEATHDATE`, `last_encounter`)
2. Run `deidentify_synthea.py` → computes `survival_time_years` and `event_observed` from ±365-day shifted dates, then drops raw dates
3. Run `run_utility_analyses.py` → calls `prepare_survival_data()` for each mart, fits `KaplanMeierFitter`, runs `logrank_test`, saves `km_survival_curves.png`
4. Check: median survival difference should be 0.0, log-rank p ≈ 1.0

---

### Method 2: Regression Analysis — "Can we still predict healthcare spending from demographics + utilization?"

**Clinical question**: In this cohort, can a linear model predict a patient's total healthcare expense decile from their age band, demographics (gender, race, ethnicity, state, marital status), and utilization counts (encounters, conditions, medications, procedures, observations)? Does the predictive accuracy change after generalization and bucketing?

**What the data looks like**:

| Column | Original mart | De-identified mart | Encoding |
|--------|--------------|-------------------|----------|
| `HEALTHCARE_EXPENSES` (target) | Continuous dollar amount (e.g. `$142,356.71`) | Bucketed to `Q1`–`Q10` decile labels | Both → ordinal 0–9 using shared quantile edges from original |
| `AGE_BAND` | **Not present** — only raw `BIRTHDATE` | `50-54`, `60-64`, etc. (5-year bands) | Ordinal: `0-4`→0, `5-9`→1, ..., `105+`→21 |
| `GENDER` | `M` / `F` | `M` / `F` (unchanged) | One-hot: drop first → 1 dummy column |
| `RACE` | `white`, `black`, `asian`, `native`, `other` (7 values) | Same | One-hot: 6 dummy columns |
| `ETHNICITY` | `nonhispanic`, `hispanic` | Same | One-hot: 1 dummy column |
| `STATE` | 2-letter codes (51 values) | Same | One-hot: 50 dummy columns |
| `MARITAL` | `M`, `S`, `D`, `W`, `nan` | Same | One-hot |
| `n_encounters` | Raw integer (e.g. `47`) | Bucketed: `31-50` | Ordinal: `0`→0, `1-5`→1, `6-15`→2, `16-30`→3, `31-50`→4, `51-100`→5, `100+`→6 |
| `n_conditions` | Raw integer | Bucketed | Same ordinal map as above |
| `n_medications` | Raw integer | Bucketed | Same ordinal map |
| ... | (6 more count columns) | Bucketed | Same ordinal map |

**Why encoding matters**: If you use `LabelEncoder` on the count bins, it sorts lexicographically: `"100+" < "1-5" < "16-30"` — which destroys the ordering. The pipeline uses explicit ordinal maps (`{\"0\": 0, \"1-5\": 1, \"6-15\": 2, ...}`) so the model sees the correct monotonic ordering.

**Implementation** (`run_utility_analyses.py`):
```python
# Make targets comparable: both on 0-9 ordinal scale
_, bin_edges = pd.qcut(original["HEALTHCARE_EXPENSES"], q=10, retbins=True)
y_orig = pd.cut(original["HEALTHCARE_EXPENSES"], bins=bin_edges, labels=False)  # 0-9
y_deid = deid["HEALTHCARE_EXPENSES"].map({"Q1": 0, "Q2": 1, ..., "Q10": 9})    # 0-9

# Encode features, run 5-fold CV
model = LinearRegression()
cv = KFold(n_splits=5, shuffle=True, random_state=42)
y_pred = cross_val_predict(model, X, y, cv=cv)
r2 = r2_score(y, y_pred)
```

**Results on this Synthea cohort**:
- Original: R² = 0.4113 (5-fold CV on ~1,171 patients)
- De-identified: R² = 0.5194
- Difference: 0.1081

**Why the de-id R² is *higher***: This is counter-intuitive. The likely cause is the feature-set asymmetry: the de-id model has `AGE_BAND` as a feature (a strong predictor of healthcare cost), while the original model does not (because `AGE_BAND` hasn't been derived for the original mart yet — only raw `BIRTHDATE` exists, which is not in the feature list). The de-id model also has coarser count bins, which may reduce overfitting noise.

**Known limitation — not apples-to-apples**: The original mart has `BIRTHDATE` (continuous date) but no `AGE_BAND`. The de-id mart has `AGE_BAND` but no `BIRTHDATE`. So the regression models use different feature sets. This should be fixed by deriving `AGE_BAND` for the original mart too. Until then, interpret this result as **illustrative of the pipeline working end-to-end**, not as a formal utility loss measurement.

**Replication steps**:
1. Both targets must be on the same ordinal scale — use `pd.qcut` quantile edges from original, map de-id `Q1`–`Q10` to 0–9
2. Features: `AGE_BAND`, `GENDER`, `RACE`, `ETHNICITY`, `STATE`, `MARITAL`, + 6 utilization counts
3. Encode: ordinal maps for bucketed columns, `pd.get_dummies(drop_first=True)` for unordered categoricals
4. Run `LinearRegression` with 5-fold `KFold(shuffle=True, random_state=42)`
5. Report `r2_score` on cross-validated predictions

---

### Method 3: Logistic Classification — "Can we still identify high-utilizer patients?"

**Clinical question**: In this cohort, can a logistic model predict which patients are "high utilizers" (above-median encounter count) from their demographics and clinical burden? Does the classifier's discriminative power (AUC) change after de-identification?

**What the data looks like**:

| Column | Original mart | De-identified mart | Role |
|--------|--------------|-------------------|------|
| `n_encounters` | Raw integer (e.g. `47`) | Bucketed: `31-50` | **Target source** — used to define the binary label, then excluded from features |
| Binary label | `n_encounters > median` → 1 (high utilizer), else 0 | **Same labels transferred from original by row index** | Target variable |
| `n_conditions` | Raw integer | Bucketed (ordinal) | Feature |
| `n_medications` | Raw integer | Bucketed (ordinal) | Feature |
| `n_procedures` | Raw integer | Bucketed (ordinal) | Feature |
| `n_unique_conditions` | Raw integer | Bucketed (ordinal) | Feature |
| `n_unique_medications` | Raw integer | Bucketed (ordinal) | Feature |
| Demographics | `GENDER`, `RACE`, `ETHNICITY`, `STATE`, `MARITAL` | Same | Features (one-hot encoded) |
| `AGE_BAND` | **Not present** | 5-year bands | Feature (ordinal) |

**Why the target is transferred, not recomputed**: In an earlier version, the de-id classifier used `n_encounters > median(n_encounters_deid)` as its target. But the de-id `n_encounters` is bucketed (`"31-50"`), which after ordinal mapping becomes 0–6. The median of 0–6 is very different from the median of raw counts (typically ~27). This caused a prevalence shift (original = 0.486, de-id = 0.424) that made AUC comparison meaningless. The fix: compute the binary labels once from the original raw counts, then reuse the exact same labels for both models. Now both have prevalence = 0.486.

**Implementation** (`run_utility_analyses.py`):
```python
# Define labels from original (same for both)
orig_median = original["n_encounters"].median()  # ~27
shared_labels = (original["n_encounters"] > orig_median).astype(int)

# Use shared_labels for both original and deid models
# Features for deid: AGE_BAND(ordinal) + demographics(one-hot) + condition/med/proc counts(ordinal)
model = LogisticRegression(max_iter=1000, random_state=42)
cv = StratifiedKFold(n_splits=5, shuffle=True, random_state=42)
y_prob = cross_val_predict(model, X, y, cv=cv, method="predict_proba")[:, 1]
auc = roc_auc_score(y, y_prob)
```

**Results on this Synthea cohort**:
- Original: AUC = 0.9730, Accuracy = 0.9402, Prevalence = 0.486
- De-identified: AUC = 0.9642, Accuracy = 0.9419, Prevalence = 0.486
- AUC drop: **0.0088** (less than 1%)

**Why AUC is so high**: The AUC is high (>0.97) because `n_conditions`, `n_medications`, and `n_procedures` are strongly correlated with encounter count. Even after bucketing to 7 ordinal bins, these utilization features retain enough discriminative power to identify high utilizers. This is expected — the clinical question is straightforward and the features are strong proxies.

**Why the AUC drop is small**: Bucketing from raw integers to 7 ordinal bins loses some granularity but preserves the rank ordering that logistic regression needs. The 0.88% AUC drop shows that for this classification task, the information loss from generalization is minimal.

**Known limitation — same feature-set asymmetry as regression**: `AGE_BAND` is available only in the de-id model. This should be harmonized before calling it a formal benchmark.

**Replication steps**:
1. Compute `shared_labels = (original["n_encounters"] > original["n_encounters"].median()).astype(int)`
2. Use these labels for both original and de-id models (do **not** recompute threshold on de-id data)
3. Features: demographics (one-hot) + clinical burden counts (ordinal) — exclude `n_encounters` itself
4. Run `LogisticRegression(max_iter=1000)` with `StratifiedKFold(n_splits=5, shuffle=True, random_state=42)`
5. Report `roc_auc_score` on cross-validated predicted probabilities

---

## What Is Solid

- Pipeline flow is clear, reproducible, and easy to follow
- Column classification (direct/quasi/sensitive/safe) is correct and explicit
- k-anonymity calculation uses `observed=True` (no phantom empty bins)
- Two privacy views: core demographic QIs and full release QIs on the actual public file
- KM survival uses `lifelines.KaplanMeierFitter` with proper birth-to-death duration
- Survival duration computed from shifted dates before dropping raw dates
- Modeling uses one-hot encoding for unordered categoricals and explicit ordinal maps for bucketed columns
- Classifier target uses a shared threshold (original's median) so prevalence doesn't artificially shift
- Regression target bucketed with same quantile edges for both datasets
- Coefficient labels correctly reflect one-hot encoded column names
- Public/eval file separation drops highly-identifying columns from public release

---

## Known Issues and Limitations

### 1. Generic utility files are placeholders

`utility_aggregates.csv` is effectively empty and `utility_marginals.csv` is dominated by type-changed columns (all showing JSD ≈ 0.83). Keep them in the repo but don't feature them in summaries until improved.

### 2. Feature-set asymmetry in regression and classification

`AGE_BAND` exists in the de-identified mart but **not** in the original mart. This means the two models don't use exactly the same feature set. The results are illustrative but not benchmark-grade until this is fixed.

**Fix**: Derive `AGE_BAND` for the original mart too, or centralize feature engineering into one common preprocessing helper.

### 3. The public release file is not privacy-safe

99.3% uniqueness is far too high for any real release. This is by design — the baseline surfaces the problem, which motivates stronger techniques.

### 4. Synthea URL not commit-pinned

`fetch_synthea.py` points to a `master/...apr2020.zip` path rather than a specific commit hash. Small but real reproducibility issue.

### 5. Pseudonymization salt is hard-coded

Fine for a benchmark prototype. Move to configuration or environment variables before anything beyond prototype use.

### 6. Coefficient interpretation

Raw β values from one-hot encoded features are less intuitive. Do not emphasize coefficient interpretation in presentations.

---

## Recommended Next Steps

### Immediate (Synthea baseline)

1. **Make utility models fully apples-to-apples** — derive `AGE_BAND` on the original mart too; centralize feature engineering for both original and de-id
2. **Refine utility summary files** — replace or redesign aggregate/marginal summaries; compare like-with-like after type harmonization
3. **Tighten privacy** — coarser encounter-year handling, further utilization coarsening, consider removing more release fields, add richer privacy metrics (l-diversity, t-closeness)
4. **Clean reproducibility** — pin Synthea source to specific commit hash, move pseudonym salt to config

### NHANES Adaptation

#### Key Differences from Synthea

| Aspect | Synthea | NHANES |
|--------|---------|--------|
| Structure | Longitudinal, multi-table, EHR-like | Cross-sectional, merged by `SEQN` |
| Person key | `Id` / `PATIENT` | `SEQN` |
| Temporal data | Encounters, dates, follow-up | Usually no native longitudinal follow-up |
| Domain modules | Encounters, conditions, medications, etc. | Demographics, exams, labs, questionnaires |
| Missingness | Sparse event tables | Subsampling / module eligibility |
| Weights | None | Survey design variables (weights, strata, PSU) |

The **pipeline pattern** transfers well, but the **exact mart and utility tasks** must be adapted.

#### What Already Helps

The repo includes `fetch_nhanes.py` which downloads selected NHANES XPT files, loads them with `pandas.read_sas()`, merges on `SEQN`, and writes a merged CSV.

#### What Should Change

**A. Create NHANES-specific scripts**
- `analyze_nhanes.py` and `deidentify_nhanes.py` (or refactor to a shared engine with dataset-specific config)
- Do not force Synthea mart logic onto NHANES — it assumes columns like `BIRTHDATE`, `first_encounter`, `primary_organization`

**B. Define NHANES-specific quasi-identifiers**
- Age / age band, sex, race/ethnicity
- Marital status, education / income bands if included
- Examination date or cycle
- Highly distinctive lab combinations if released

**C. Handle survey-design variables intentionally**
- Sample weights, strata, PSU/cluster identifiers, subsample weights
- These should not be accidentally dropped or treated as ordinary covariates

**D. Re-specify utility tasks**

| Task | Synthea Baseline | NHANES Suggestion |
|------|------------------|-------------------|
| Survival (KM) | Birth-to-death with date shifting | Only if mortality linkage is included; otherwise replace with subgroup prevalence / weighted mean comparison |
| Regression | `HEALTHCARE_EXPENSES` → decile target | BMI, systolic BP, HbA1c, cholesterol, or ordinal risk category |
| Classification | High-utilizer (encounter count) | Diabetes, hypertension, obesity, smoking status, elevated HbA1c |

#### Suggested NHANES Benchmark Design

1. **Analytic mart**: One respondent per row, keyed by `SEQN`, with demographics, body measures, selected labs, questionnaire variables, survey design variables, and a declared QI list
2. **Privacy baseline**: Suppression, age banding, categorical coarsening, lab value bucketing, k-anonymity on declared QI set, release-file vs eval-file separation
3. **Utility baseline**: Regression + binary classification + subgroup prevalence comparison; use KM only if mortality linkage is deliberately included

---

## How to Run

```bash
# Install dependencies
pip install lifelines scikit-learn matplotlib pandas numpy scipy

# Run the full pipeline
python fetch_synthea.py
python analyze_synthea.py
python deidentify_synthea.py
python evaluate_deidentification.py
python run_utility_analyses.py

# Or specify a custom output directory
python run_utility_analyses.py --output-dir /path/to/data
```

### Output Files

All saved to `{output_dir}/results/baseline_deidentification/`:

| File | Description |
|------|-------------|
| `column_profile.csv` | Column classification for all 11 Synthea tables |
| `analytic_mart.csv` | Original patient-level analytic mart (1,171 rows, 38 cols) |
| `deid_mart_public.csv` | Public-release de-identified mart (IDs/org/survival dropped) |
| `deid_mart_eval.csv` | Full de-identified mart for internal evaluation |
| `km_survival_curves.png` | Two-panel KM survival plot (original vs de-id + age stratification) |
| `utility_analyses_summary.csv` | Summary: survival, regression, classification metrics |
| `utility_aggregates.csv` | Mean/median comparison for numeric columns |
| `utility_marginals.csv` | Jensen-Shannon divergence for all shared columns |
| `k_anonymity_core.csv` | k-anonymity on core demographic QIs |
| `k_anonymity_release.csv` | k-anonymity on full release QIs (public file) |

---

## Presentation Talking Points

### Point 1: We have a working baseline pipeline on structured healthcare data

This is not a toy notebook fragment. It is a structured pipeline with separate scripts for data fetch, profiling, mart construction, de-identification, privacy evaluation, and utility evaluation — all reproducible.

### Point 2: The baseline shows direct-ID removal is not enough

The current public release file still has **99.3% unique patients** under release-QI k-anonymity. Direct identifiers can be removed successfully, but linkage risk remains extreme from the combination of demographics + utilization patterns + encounter dates.

### Point 3: Some utility can be preserved

The KM survival curve is **essentially unchanged** after patient-level date shifting (median survival difference = 0.0 years, log-rank p = 1.0). The classification task retains most discriminative power (AUC drop < 0.01).

### Closing

**Next step**: Adapt the same framework to NHANES using NHANES-specific quasi-identifiers and utility tasks.

---

## Files to Circulate

**Recommended package**:
- Code: all Python scripts in `privacy_methods/d1_structured_data/baseline_deidentification/python/`
- Results: all CSV and PNG files in `results/baseline_deidentification/`
- This handoff document: `BASELINE_DEIDENTIFICATION_TEAM_HANDOFF.md`
