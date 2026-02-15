# Longitudinal / Repeated Measures — Python Implementation

## Required Libraries

```bash
pip install numpy pandas statsmodels formulaic matplotlib seaborn scipy
```

- **numpy / pandas**: Data manipulation and numerical computation.
- **statsmodels**: `MixedLM` for linear mixed-effects models, `GEE` for generalized estimating equations.
- **formulaic**: Formula interface for model specification (optional, enhances formula syntax).
- **matplotlib / seaborn**: Plotting and visualization.
- **scipy**: Statistical distributions for inference.

## Example Dataset

We simulate a clinical trial with 300 patients randomized to Active or Placebo, measured at baseline and weeks 2, 4, 8, 12, and 16. The outcome is a continuous symptom severity score.

```python
import numpy as np
import pandas as pd

np.random.seed(42)
n_subj = 300
visits = [0, 2, 4, 8, 12, 16]
n_visits = len(visits)

# Subject-level data
treatment = np.repeat(['Placebo', 'Active'], n_subj // 2)
age = np.round(np.random.normal(50, 12, n_subj)).astype(int)
baseline_score = np.round(np.random.normal(60, 10, n_subj), 1)

# Random effects (intercept, slope)
re_cov = np.array([[25, -0.8], [-0.8, 0.04]])
re = np.random.multivariate_normal([0, 0], re_cov, n_subj)

# Build long-format data
rows = []
for i in range(n_subj):
    for j, week in enumerate(visits):
        trt_eff = -0.5 * week if treatment[i] == 'Active' else 0
        score = (baseline_score[i] + re[i, 0] +
                 (-0.2 + re[i, 1]) * week + trt_eff +
                 np.random.normal(0, 3))
        rows.append({
            'subject': i + 1,
            'treatment': treatment[i],
            'age': age[i],
            'baseline_score': baseline_score[i],
            'visit_week': week,
            'score': round(score, 1)
        })

long_data = pd.DataFrame(rows)

# Introduce MAR dropout (~20%)
for s in range(1, n_subj + 1):
    subj_mask = long_data['subject'] == s
    subj_idx = long_data[subj_mask].index.tolist()
    for v_pos in range(1, n_visits):
        prev_score = long_data.loc[subj_idx[v_pos - 1], 'score']
        is_placebo = int(long_data.loc[subj_idx[0], 'treatment'] == 'Placebo')
        drop_prob = 1 / (1 + np.exp(-(-4 + 0.02 * prev_score + 0.3 * is_placebo)))
        if np.random.random() < drop_prob:
            for v_drop in range(v_pos, n_visits):
                long_data.loc[subj_idx[v_drop], 'score'] = np.nan
            break

# Create visit factor
long_data['visit_factor'] = long_data['visit_week'].astype(str)

print(f"Total observations: {len(long_data)}")
print(f"Missing: {long_data['score'].isna().sum()}")
print(f"Missing rate: {long_data['score'].isna().mean():.3f}")
```

## Complete Worked Example

### Step 1: Exploratory Visualization

```python
import matplotlib.pyplot as plt
import seaborn as sns

fig, ax = plt.subplots(figsize=(10, 6))

# Individual trajectories (thin gray lines)
for subj in long_data['subject'].unique():
    subj_data = long_data[long_data['subject'] == subj]
    ax.plot(subj_data['visit_week'], subj_data['score'],
            color='gray', alpha=0.08, linewidth=0.5)

# Group means
for trt, color in [('Placebo', 'coral'), ('Active', 'steelblue')]:
    grp = long_data[long_data['treatment'] == trt]
    means = grp.groupby('visit_week')['score'].mean()
    sems = grp.groupby('visit_week')['score'].sem()
    ax.plot(means.index, means.values, 'o-', color=color,
            linewidth=2, markersize=6, label=trt)
    ax.fill_between(means.index,
                    (means - 1.96 * sems).values,
                    (means + 1.96 * sems).values,
                    color=color, alpha=0.15)

ax.set_xlabel('Week', fontsize=12)
ax.set_ylabel('Symptom Score', fontsize=12)
ax.set_title('Individual Trajectories and Group Means (95% CI)', fontsize=14)
ax.legend(fontsize=11)
plt.tight_layout()
plt.savefig('spaghetti_plot.png', dpi=150, bbox_inches='tight')
plt.show()
```

**Output interpretation**: Individual gray trajectories show the heterogeneity between patients. Group mean lines with shaded 95% CI bands reveal the average treatment trajectory. A widening gap between Active and Placebo over time indicates an increasing treatment benefit.

### Step 2: Linear Mixed-Effects Model

```python
import statsmodels.formula.api as smf

# Random intercept and slope model
# Use only non-missing data
analysis_data = long_data.dropna(subset=['score']).copy()

lmm = smf.mixedlm(
    "score ~ C(treatment) * visit_week + baseline_score",
    data=analysis_data,
    groups=analysis_data['subject'],
    re_formula="~ visit_week"
)

lmm_fit = lmm.fit(reml=True)
print(lmm_fit.summary())
```

**Output interpretation**: The key fixed effects are:
- `C(treatment)[T.Active]`: Treatment difference at baseline (should be near zero for a randomized trial).
- `visit_week`: Slope for placebo group (rate of change per week).
- `C(treatment)[T.Active]:visit_week`: Difference in slopes (the primary estimand). A significant negative value means the Active group improves faster.
- Random effects: `Group Var` is the random intercept variance; `visit_week Var` is the random slope variance; `Group x visit_week Cov` is their covariance.

```python
# Extract random effects variance components
print("\n=== Variance Components ===")
print(f"Random intercept variance: {lmm_fit.cov_re.iloc[0, 0]:.3f}")
print(f"Random slope variance: {lmm_fit.cov_re.iloc[1, 1]:.6f}")
print(f"Residual variance: {lmm_fit.scale:.3f}")

# ICC
icc = lmm_fit.cov_re.iloc[0, 0] / (lmm_fit.cov_re.iloc[0, 0] + lmm_fit.scale)
print(f"ICC (intercept): {icc:.3f}")
```

### Step 3: GEE (Generalized Estimating Equations)

```python
import statsmodels.api as sm
from statsmodels.genmod.generalized_estimating_equations import GEE
from statsmodels.genmod.cov_struct import Exchangeable, Autoregressive
from statsmodels.genmod.families import Gaussian

# Prepare data (GEE requires sorted by subject and no missing in predictors)
gee_data = analysis_data.sort_values(['subject', 'visit_week']).copy()
gee_data['treatment_active'] = (gee_data['treatment'] == 'Active').astype(int)
gee_data['trt_time'] = gee_data['treatment_active'] * gee_data['visit_week']

# GEE with exchangeable working correlation
formula = "score ~ treatment_active + visit_week + trt_time + baseline_score"

gee_exch = GEE.from_formula(
    formula,
    groups="subject",
    data=gee_data,
    family=Gaussian(),
    cov_struct=Exchangeable()
)
gee_exch_fit = gee_exch.fit()
print("=== GEE with Exchangeable Correlation ===")
print(gee_exch_fit.summary())

# GEE with AR(1) working correlation
gee_ar1 = GEE.from_formula(
    formula,
    groups="subject",
    data=gee_data,
    family=Gaussian(),
    cov_struct=Autoregressive()
)
gee_ar1_fit = gee_ar1.fit()
print("\n=== GEE with AR(1) Correlation ===")
print(gee_ar1_fit.summary())
```

**Output interpretation**: GEE provides population-averaged effects with robust (sandwich) standard errors. The `trt_time` coefficient gives the additional rate of change per week for the Active group versus Placebo. GEE standard errors are consistent even if the working correlation structure is misspecified, though efficiency improves with a better-specified structure. Note the estimated working correlation parameter.

### Step 4: MMRM-Style Analysis

```python
# MMRM in Python: Use MixedLM with visit as categorical (fixed effect)
# and unstructured within-subject variance
# Note: statsmodels MixedLM does not support unstructured residual covariance
# directly. We approximate MMRM using visit-specific random effects.

# Post-baseline data only
post_bl = analysis_data[analysis_data['visit_week'] > 0].copy()
post_bl['visit_cat'] = pd.Categorical(post_bl['visit_week'])

# Create dummy variables for visits
visit_dummies = pd.get_dummies(post_bl['visit_week'], prefix='v', dtype=float)
visit_dummies.columns = [f'v{int(c.split("_")[1])}' for c in visit_dummies.columns]
post_bl = pd.concat([post_bl, visit_dummies], axis=1)

# Treatment by visit interaction
post_bl['trt_active'] = (post_bl['treatment'] == 'Active').astype(float)
for v in [2, 4, 8, 12, 16]:
    post_bl[f'trt_v{v}'] = post_bl['trt_active'] * post_bl[f'v{v}']

# Fit a model with visit as categorical and treatment-by-visit interaction
# Random intercept per subject (approximation to MMRM)
mmrm_formula = ("score ~ trt_active + v4 + v8 + v12 + v16 + "
                "trt_v4 + trt_v8 + trt_v12 + trt_v16 + baseline_score")

mmrm_model = smf.mixedlm(
    mmrm_formula,
    data=post_bl,
    groups=post_bl['subject'],
    re_formula="~1"
)
mmrm_fit = mmrm_model.fit(reml=True)
print("=== MMRM-Style Analysis ===")
print(mmrm_fit.summary())

# Treatment effect at each visit
print("\n=== Treatment Effect at Each Visit ===")
trt_base = mmrm_fit.params['trt_active']  # at visit 2 (reference)
for v in [2, 4, 8, 12, 16]:
    if v == 2:
        effect = trt_base
        se = mmrm_fit.bse['trt_active']
    else:
        effect = trt_base + mmrm_fit.params[f'trt_v{v}']
        # Approximate SE using variance of sum
        var_sum = (mmrm_fit.bse['trt_active']**2 +
                   mmrm_fit.bse[f'trt_v{v}']**2)
        se = np.sqrt(var_sum)

    ci_low = effect - 1.96 * se
    ci_high = effect + 1.96 * se
    z = effect / se
    p = 2 * (1 - stats.norm.cdf(abs(z)))
    print(f"  Week {v:2d}: Effect = {effect:7.3f}, "
          f"95% CI = ({ci_low:.3f}, {ci_high:.3f}), p = {p:.4f}")
```

**Output interpretation**: The MMRM-style analysis treats visit as categorical, allowing the treatment effect to vary freely across visits without assuming a linear time trend. The primary estimand is the treatment effect at week 16 (the sum of `trt_active` and `trt_v16`). This is the most commonly reported analysis in clinical trial publications and regulatory submissions.

Note: For a production-quality MMRM with unstructured covariance, consider using the `mmrm` R package via `rpy2` or the `pymmrm` package if available.

## Advanced Example

### Comparison of Multiple Covariance Structures

```python
from scipy import stats

# Fit models with different random effects structures
# Model 1: Random intercept only
m1 = smf.mixedlm(
    "score ~ C(treatment) * visit_week + baseline_score",
    data=analysis_data, groups=analysis_data['subject'],
    re_formula="~1"
).fit(reml=True)

# Model 2: Random intercept + slope
m2 = smf.mixedlm(
    "score ~ C(treatment) * visit_week + baseline_score",
    data=analysis_data, groups=analysis_data['subject'],
    re_formula="~visit_week"
).fit(reml=True)

print("=== Model Comparison ===")
print(f"{'Model':<30} {'AIC':>10} {'BIC':>10} {'Log-lik':>12}")
print("-" * 65)
print(f"{'Random intercept':<30} {m1.aic:>10.1f} {m1.bic:>10.1f} "
      f"{m1.llf:>12.1f}")
print(f"{'Random intercept + slope':<30} {m2.aic:>10.1f} {m2.bic:>10.1f} "
      f"{m2.llf:>12.1f}")

# Likelihood ratio test
lr_stat = 2 * (m2.llf - m1.llf)
lr_df = 2  # additional parameters: slope variance + covariance
lr_p = 1 - stats.chi2.cdf(lr_stat, lr_df)
print(f"\nLR test: chi2 = {lr_stat:.2f}, df = {lr_df}, p = {lr_p:.6f}")
```

**Output interpretation**: Lower AIC/BIC indicates better model fit. The likelihood ratio test compares nested models. A significant p-value means the more complex model (with random slopes) fits significantly better. Including random slopes captures individual differences in the rate of change over time.

### Predicted Individual Trajectories

```python
# Extract random effects for each subject
re_df = pd.DataFrame(m2.random_effects).T
re_df.columns = ['intercept_re', 'slope_re']
re_df['subject'] = re_df.index.astype(int)

# Merge with subject data
subj_info = long_data[['subject', 'treatment', 'baseline_score']].drop_duplicates()
re_df = re_df.merge(subj_info, on='subject')

# Plot predicted trajectories for a sample of subjects
fig, axes = plt.subplots(1, 2, figsize=(14, 6))
sample_size = 20

for idx, trt in enumerate(['Placebo', 'Active']):
    ax = axes[idx]
    trt_subj = re_df[re_df['treatment'] == trt].head(sample_size)

    for _, row in trt_subj.iterrows():
        weeks = np.array(visits)
        # Fixed effects + random effects
        fe_intercept = m2.fe_params['Intercept']
        fe_trt = m2.fe_params.get('C(treatment)[T.Active]', 0) if trt == 'Active' else 0
        fe_slope = m2.fe_params['visit_week']
        fe_trt_slope = m2.fe_params.get('C(treatment)[T.Active]:visit_week', 0) if trt == 'Active' else 0
        fe_bl = m2.fe_params['baseline_score']

        pred = (fe_intercept + fe_trt + row['intercept_re'] +
                (fe_slope + fe_trt_slope + row['slope_re']) * weeks +
                fe_bl * row['baseline_score'])

        ax.plot(weeks, pred, alpha=0.4,
                color='steelblue' if trt == 'Active' else 'coral')

    ax.set_title(f'{trt} Group (n={sample_size} shown)', fontsize=13)
    ax.set_xlabel('Week', fontsize=11)
    ax.set_ylabel('Predicted Score', fontsize=11)

plt.suptitle('Model-Predicted Individual Trajectories', fontsize=14, y=1.02)
plt.tight_layout()
plt.savefig('predicted_trajectories.png', dpi=150, bbox_inches='tight')
plt.show()
```

**Output interpretation**: The predicted trajectories show how individual patients (with their random effects) are expected to change over time. The spread of lines reflects between-subject variability. A steeper downward trend in the Active group indicates the treatment benefit.

## Visualization

### Treatment Effect Over Time

```python
from scipy import stats

# Calculate treatment effect at each post-baseline visit
visit_effects = []
for week in [2, 4, 8, 12, 16]:
    trt_mask = (analysis_data['visit_week'] == week)
    active = analysis_data[(analysis_data['treatment'] == 'Active') & trt_mask]['score'].dropna()
    placebo = analysis_data[(analysis_data['treatment'] == 'Placebo') & trt_mask]['score'].dropna()

    diff = active.mean() - placebo.mean()
    se = np.sqrt(active.var() / len(active) + placebo.var() / len(placebo))
    visit_effects.append({
        'week': week, 'difference': diff,
        'ci_lower': diff - 1.96 * se, 'ci_upper': diff + 1.96 * se
    })

eff_df = pd.DataFrame(visit_effects)

fig, ax = plt.subplots(figsize=(8, 6))
ax.errorbar(eff_df['week'], eff_df['difference'],
            yerr=[eff_df['difference'] - eff_df['ci_lower'],
                  eff_df['ci_upper'] - eff_df['difference']],
            fmt='o-', color='steelblue', capsize=5, linewidth=2,
            markersize=8, capthick=2)
ax.axhline(y=0, color='gray', linestyle='--', linewidth=1)
ax.set_xlabel('Week', fontsize=12)
ax.set_ylabel('Treatment Difference (Active - Placebo)', fontsize=12)
ax.set_title('Treatment Effect Over Time', fontsize=14)
plt.tight_layout()
plt.savefig('treatment_effect_time.png', dpi=150, bbox_inches='tight')
plt.show()
```

### Residual Diagnostics

```python
fig, axes = plt.subplots(1, 3, figsize=(15, 5))

residuals = m2.resid
fitted_vals = m2.fittedvalues

# Residuals vs fitted
axes[0].scatter(fitted_vals, residuals, alpha=0.2, color='steelblue', s=10)
axes[0].axhline(y=0, color='red', linestyle='--')
axes[0].set_xlabel('Fitted Values')
axes[0].set_ylabel('Residuals')
axes[0].set_title('Residuals vs Fitted')

# Q-Q plot
stats.probplot(residuals, dist="norm", plot=axes[1])
axes[1].set_title('Normal Q-Q Plot')
axes[1].get_lines()[0].set_color('steelblue')
axes[1].get_lines()[1].set_color('coral')

# Histogram of residuals
axes[2].hist(residuals, bins=40, density=True, color='steelblue',
             edgecolor='white', alpha=0.7)
x_range = np.linspace(residuals.min(), residuals.max(), 100)
axes[2].plot(x_range, stats.norm.pdf(x_range, residuals.mean(), residuals.std()),
             'r-', linewidth=2, label='Normal density')
axes[2].set_xlabel('Residuals')
axes[2].set_ylabel('Density')
axes[2].set_title('Distribution of Residuals')
axes[2].legend()

plt.suptitle('Residual Diagnostics for LMM', fontsize=14, y=1.02)
plt.tight_layout()
plt.savefig('residual_diagnostics.png', dpi=150, bbox_inches='tight')
plt.show()
```

**Output interpretation**: The residual vs fitted plot should show a random cloud centered at zero. Systematic patterns (fan shape, curvature) indicate heteroscedasticity or model misspecification. The Q-Q plot assesses normality — most points should lie on the reference line. The histogram provides a visual check for skewness or heavy tails.

## Tips and Best Practices

1. **Use `statsmodels.MixedLM`** for LMMs in Python. It supports random intercepts, slopes, and REML estimation. For MMRM with unstructured covariance, consider interfacing with R via `rpy2`.
2. **Always sort data by subject and time** before fitting GEE models — the working correlation depends on observation ordering.
3. **Include baseline as a covariate**, not as a response variable. This is the standard ANCOVA approach and improves power.
4. **Check convergence**: If `MixedLM.fit()` fails, try different optimization methods (`method='powell'` or `method='nm'`).
5. **For GEE in Python**, the robust (sandwich) standard errors are returned by default, ensuring valid inference even with misspecified working correlation.
6. **Be cautious with p-values**: `statsmodels` MixedLM uses Wald z-tests, which can be anti-conservative for small samples. Satterthwaite/Kenward-Roger adjustments require R.
7. **Visualize before modeling**: Spaghetti plots reveal non-linear trends, outliers, and the extent of between-subject variability.
8. **For clinical trial submissions**, the R `mmrm` package is strongly recommended over Python implementations, as it is validated and widely accepted by regulators.
9. **Report the covariance structure** used, the estimation method (REML vs ML), and the specific treatment contrast at the primary time point.
10. **Handle missing data carefully**: LMMs and MMRM use all available data via likelihood-based estimation under MAR. GEE requires MCAR or weighted extensions for MAR.
