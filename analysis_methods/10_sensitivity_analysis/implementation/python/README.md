# Sensitivity Analysis — Python Implementation

## Required Libraries

```bash
pip install numpy pandas scipy statsmodels matplotlib seaborn
```

- **numpy / pandas**: Core data manipulation and numerical computation.
- **scipy**: Statistical distributions for inference and optimization.
- **statsmodels**: Regression models for primary analysis.
- **matplotlib / seaborn**: Visualization.

Note: Dedicated sensitivity analysis packages for Python are limited compared to R. This implementation provides custom functions for E-value computation, quantitative bias analysis, tipping-point analysis, and sensitivity arrays.

## Example Dataset

We simulate an observational study of statin use and cardiovascular (CV) event risk in 5000 older adults, with an unmeasured confounder (smoking) creating residual confounding.

```python
import numpy as np
import pandas as pd
from scipy import stats
import statsmodels.formula.api as smf

np.random.seed(42)
n = 5000

obs_data = pd.DataFrame({
    'statin_use': np.random.binomial(1, 0.35, n),
    'age': np.round(np.random.normal(65, 10, n)).astype(int),
    'male': np.random.binomial(1, 0.55, n),
    'diabetes': np.random.binomial(1, 0.25, n),
    'hypertension': np.random.binomial(1, 0.40, n),
    'smoker': np.random.binomial(1, 0.20, n),  # unmeasured
})

# Simulate CV events (~12%)
log_odds = (-3.0 + 0.03 * obs_data['age'] + 0.4 * obs_data['male'] +
            0.6 * obs_data['diabetes'] + 0.5 * obs_data['hypertension'] +
            0.8 * obs_data['smoker'] - 0.5 * obs_data['statin_use'])
obs_data['cv_event'] = np.random.binomial(1, 1 / (1 + np.exp(-log_odds)))

# Primary analysis: adjusted logistic regression (omitting smoker)
model = smf.logit(
    'cv_event ~ statin_use + age + male + diabetes + hypertension',
    data=obs_data
).fit(disp=0)

or_statin = np.exp(model.params['statin_use'])
ci_statin = np.exp(model.conf_int().loc['statin_use'])

print("=== Primary Analysis ===")
print(f"Adjusted OR for statin use: {or_statin:.4f}")
print(f"95% CI: ({ci_statin[0]:.4f}, {ci_statin[1]:.4f})")
print(f"p-value: {model.pvalues['statin_use']:.6f}")
```

## Complete Worked Example

### Step 1: E-Value Computation

```python
def compute_evalue_or(or_est, or_lo=None, or_hi=None, rare=True):
    """
    Compute E-value for an odds ratio.

    Parameters:
    -----------
    or_est : float - Point estimate of the odds ratio
    or_lo  : float - Lower bound of 95% CI
    or_hi  : float - Upper bound of 95% CI
    rare   : bool  - If True, use OR directly; if False, convert to RR
    """
    if rare:
        rr = or_est
        rr_lo = or_lo if or_lo is not None else None
    else:
        # Approximate conversion: use square root transformation
        # RR ≈ OR / (1 - p0 + p0 * OR) where p0 is baseline risk
        # Simplified: for moderate outcomes, use sqrt(OR) as approximation
        rr = np.sqrt(or_est) if or_est >= 1 else 1 / np.sqrt(1 / or_est)
        rr_lo = np.sqrt(or_lo) if or_lo is not None and or_lo >= 1 else None
        if or_lo is not None and or_lo < 1:
            rr_lo = 1 / np.sqrt(1 / or_lo)

    # E-value formula for RR
    def evalue_rr(rr_val):
        if rr_val is None:
            return None
        if rr_val >= 1:
            return rr_val + np.sqrt(rr_val * (rr_val - 1))
        else:
            rr_inv = 1 / rr_val
            return rr_inv + np.sqrt(rr_inv * (rr_inv - 1))

    eval_point = evalue_rr(rr)
    eval_ci = evalue_rr(rr_lo) if rr_lo is not None else None

    return {
        'rr_approx': rr,
        'evalue_point': eval_point,
        'evalue_ci': eval_ci
    }

# Compute E-values
# For a protective effect (OR < 1), we analyze 1/OR
evalue_result = compute_evalue_or(
    or_est=or_statin,
    or_lo=ci_statin[0],
    or_hi=ci_statin[1],
    rare=False
)

print("=== E-Value Analysis ===")
print(f"Approximate RR: {evalue_result['rr_approx']:.4f}")
print(f"E-value (point estimate): {evalue_result['evalue_point']:.2f}")
print(f"E-value (CI bound): {evalue_result['evalue_ci']:.2f}")
print()
print("Interpretation:")
print(f"  An unmeasured confounder would need to be associated with both")
print(f"  statin use and CV events by a risk ratio of at least "
      f"{evalue_result['evalue_point']:.2f}")
print(f"  (each association) to fully explain away the observed effect.")
print(f"  To move the CI to include the null, an RR of at least "
      f"{evalue_result['evalue_ci']:.2f} is needed.")
```

**Output interpretation**: The E-value provides a single-number summary of sensitivity to unmeasured confounding. Compare the E-value against known confounder strengths. For example, if the E-value is 2.5 and the strongest known unmeasured confounder (smoking) has RRs of about 2.0 and 1.4 with the outcome and exposure respectively, the product (2.8) exceeds the E-value, suggesting that an unmeasured confounder of this magnitude could explain the finding.

### Step 2: Quantitative Bias Analysis — Unmeasured Confounding

```python
def qba_unmeasured_confounding(or_obs, prev_exposed, prev_unexposed, rr_confounder):
    """
    Quantitative bias analysis for a single unmeasured confounder.

    Parameters:
    -----------
    or_obs         : float - Observed odds ratio
    prev_exposed   : float - Prevalence of confounder in exposed group
    prev_unexposed : float - Prevalence of confounder in unexposed group
    rr_confounder  : float - RR (or OR) of confounder with outcome
    """
    # Bias factor
    bias_factor = ((rr_confounder * prev_exposed + (1 - prev_exposed)) /
                   (rr_confounder * prev_unexposed + (1 - prev_unexposed)))

    or_adjusted = or_obs / bias_factor
    pct_change = (or_adjusted - or_obs) / or_obs * 100

    return {
        'or_observed': or_obs,
        'bias_factor': bias_factor,
        'or_adjusted': or_adjusted,
        'pct_change': pct_change
    }

# Scenario: Smoking as unmeasured confounder
# Prevalence in statin users: 15%, in non-users: 25%
# OR of smoking with CV events: 2.0
qba_result = qba_unmeasured_confounding(
    or_obs=or_statin,
    prev_exposed=0.15,      # smoking prevalence in statin users
    prev_unexposed=0.25,    # smoking prevalence in non-users
    rr_confounder=2.0       # smoking-CV event OR
)

print("=== QBA: Unmeasured Confounding (Smoking Scenario) ===")
print(f"Observed OR: {qba_result['or_observed']:.4f}")
print(f"Bias factor: {qba_result['bias_factor']:.4f}")
print(f"Adjusted OR: {qba_result['or_adjusted']:.4f}")
print(f"Change: {qba_result['pct_change']:.1f}%")
print()
if qba_result['or_adjusted'] < 1:
    print("Result: Protective effect persists after adjustment.")
else:
    print("Result: Protective effect is explained away by confounding.")
```

**Output interpretation**: The bias factor quantifies how much the confounder shifts the observed association. If the adjusted OR remains below 1.0 (for a protective effect), the result is robust to this particular confounding scenario. Multiple scenarios with different bias parameters should be explored.

### Step 3: Misclassification Bias Analysis

```python
def qba_outcome_misclassification(a, b, c, d, se_exp, se_unexp,
                                   sp_exp, sp_unexp):
    """
    Correct for outcome misclassification in a 2x2 table.

    Parameters:
    -----------
    a, b, c, d    : int - Cells of the 2x2 table
                    a=exposed+disease, b=exposed+no disease
                    c=unexposed+disease, d=unexposed+no disease
    se_exp/unexp  : float - Sensitivity in exposed/unexposed
    sp_exp/unexp  : float - Specificity in exposed/unexposed
    """
    n1 = a + b  # total exposed
    n0 = c + d  # total unexposed

    # Observed proportions
    p1_obs = a / n1
    p0_obs = c / n0

    # Corrected proportions
    p1_true = (p1_obs - (1 - sp_exp)) / (se_exp - (1 - sp_exp))
    p0_true = (p0_obs - (1 - sp_unexp)) / (se_unexp - (1 - sp_unexp))

    # Corrected cell counts
    a_corr = p1_true * n1
    b_corr = (1 - p1_true) * n1
    c_corr = p0_true * n0
    d_corr = (1 - p0_true) * n0

    or_obs = (a * d) / (b * c)
    or_corr = (a_corr * d_corr) / (b_corr * c_corr)

    return {
        'or_observed': or_obs,
        'or_corrected': or_corr,
        'p1_observed': p1_obs,
        'p1_corrected': p1_true,
        'p0_observed': p0_obs,
        'p0_corrected': p0_true
    }

# 2x2 table from the data
tab = pd.crosstab(obs_data['statin_use'], obs_data['cv_event'])
a = tab.loc[1, 1]  # statin + event
b = tab.loc[1, 0]  # statin + no event
c = tab.loc[0, 1]  # no statin + event
d = tab.loc[0, 0]  # no statin + no event

misclass = qba_outcome_misclassification(
    a, b, c, d,
    se_exp=0.85, se_unexp=0.85,    # 85% sensitivity
    sp_exp=0.98, sp_unexp=0.98     # 98% specificity
)

print("=== QBA: Outcome Misclassification ===")
print(f"Crude OR (before adjustment): {misclass['or_observed']:.4f}")
print(f"Corrected OR: {misclass['or_corrected']:.4f}")
print(f"Observed event rate (exposed): {misclass['p1_observed']:.4f}")
print(f"Corrected event rate (exposed): {misclass['p1_corrected']:.4f}")
print()
print("Interpretation: Non-differential outcome misclassification")
print("typically biases the OR toward the null. The corrected OR")
print("should show a stronger association than the crude OR.")
```

**Output interpretation**: Non-differential misclassification of the outcome (same sensitivity/specificity in both groups) biases the odds ratio toward 1.0. The corrected OR is further from 1.0, meaning the true effect is likely stronger than observed. Differential misclassification can bias in either direction.

### Step 4: Probabilistic Bias Analysis

```python
def probabilistic_bias_analysis(or_obs, n_sim=50000):
    """
    Probabilistic bias analysis for unmeasured confounding.
    Draws bias parameters from distributions and computes adjusted ORs.
    """
    np.random.seed(42)

    # Prior distributions for bias parameters (trapezoidal via SciPy)
    from scipy.stats import trapezoid
    # trapezoid(c, d, loc, scale): c and d define the flat portion as fractions of scale
    prev_exposed = trapezoid.rvs(c=1/3, d=2/3, loc=0.10, scale=0.15, size=n_sim)
    prev_unexposed = trapezoid.rvs(c=1/3, d=2/3, loc=0.20, scale=0.15, size=n_sim)
    rr_confounder = trapezoid.rvs(c=0.3, d=0.7, loc=1.5, scale=1.0, size=n_sim)

    # Compute bias factor for each simulation
    bias_factors = ((rr_confounder * prev_exposed + (1 - prev_exposed)) /
                    (rr_confounder * prev_unexposed + (1 - prev_unexposed)))

    # Adjusted OR
    or_adjusted = or_obs / bias_factors

    results = {
        'median': np.median(or_adjusted),
        'mean': np.mean(or_adjusted),
        'ci_2.5': np.percentile(or_adjusted, 2.5),
        'ci_97.5': np.percentile(or_adjusted, 97.5),
        'prob_above_1': np.mean(or_adjusted > 1.0),
        'distribution': or_adjusted
    }
    return results

pba = probabilistic_bias_analysis(or_statin)

print("=== Probabilistic Bias Analysis ===")
print(f"Observed OR: {or_statin:.4f}")
print(f"Median adjusted OR: {pba['median']:.4f}")
print(f"Mean adjusted OR: {pba['mean']:.4f}")
print(f"95% Simulation Interval: ({pba['ci_2.5']:.4f}, {pba['ci_97.5']:.4f})")
print(f"P(adjusted OR > 1.0): {pba['prob_above_1']:.4f}")
```

**Output interpretation**: The simulation interval reflects both the uncertainty in the bias parameters and the variability in the adjusted estimate. The probability of the adjusted OR exceeding 1.0 indicates how likely the protective effect would vanish under the specified range of confounding scenarios. A low probability (e.g., < 5%) suggests the result is robust.

## Advanced Example

### Tipping-Point Analysis for Missing Data

```python
# Simulate a clinical trial with missing outcomes
np.random.seed(123)
n_trial = 400
trial = pd.DataFrame({
    'treatment': np.repeat(['Placebo', 'Active'], n_trial // 2),
    'baseline': np.round(np.random.normal(50, 10, n_trial), 1),
})

# Simulate outcomes
trt_eff = np.where(trial['treatment'] == 'Active', -5.0, 0.0)
trial['outcome'] = trial['baseline'] + trt_eff + np.random.normal(0, 8, n_trial)

# Introduce MAR missingness (~25%)
miss_prob = 1 / (1 + np.exp(-(-2 + 0.03 * (trial['baseline'] - 50) +
                                0.5 * (trial['treatment'] == 'Placebo').astype(int))))
trial['observed'] = np.random.random(n_trial) > miss_prob
trial.loc[~trial['observed'], 'outcome'] = np.nan

print(f"Missing outcomes: {trial['outcome'].isna().sum()} / {n_trial}")

# Impute missing values using mean imputation (simplified for illustration)
# In practice, use MICE as shown in the missing data methods chapter
mean_active = trial.loc[(trial['treatment'] == 'Active') &
                         trial['outcome'].notna(), 'outcome'].mean()
mean_placebo = trial.loc[(trial['treatment'] == 'Placebo') &
                          trial['outcome'].notna(), 'outcome'].mean()

# Tipping-point analysis
deltas = np.arange(0, 12.1, 0.5)
tp_results = []

for delta in deltas:
    trial_tp = trial.copy()

    # Impute missing active-arm outcomes shifted by delta (worse)
    active_missing = trial_tp['outcome'].isna() & (trial_tp['treatment'] == 'Active')
    placebo_missing = trial_tp['outcome'].isna() & (trial_tp['treatment'] == 'Placebo')

    trial_tp.loc[active_missing, 'outcome'] = mean_active + delta
    trial_tp.loc[placebo_missing, 'outcome'] = mean_placebo

    # ANCOVA
    model_tp = smf.ols('outcome ~ C(treatment) + baseline', data=trial_tp).fit()
    trt_param = 'C(treatment)[T.Active]' if 'C(treatment)[T.Active]' in model_tp.params.index else 'C(treatment)[T.Placebo]'

    if 'C(treatment)[T.Active]' in model_tp.params.index:
        est = model_tp.params['C(treatment)[T.Active]']
        pval = model_tp.pvalues['C(treatment)[T.Active]']
        ci = model_tp.conf_int().loc['C(treatment)[T.Active]']
    else:
        est = -model_tp.params['C(treatment)[T.Placebo]']
        pval = model_tp.pvalues['C(treatment)[T.Placebo]']
        ci = -model_tp.conf_int().loc['C(treatment)[T.Placebo]'][::-1]

    tp_results.append({
        'delta': delta,
        'estimate': est,
        'ci_lower': ci.iloc[0] if hasattr(ci, 'iloc') else ci[0],
        'ci_upper': ci.iloc[1] if hasattr(ci, 'iloc') else ci[1],
        'p_value': pval
    })

tp_df = pd.DataFrame(tp_results)
print("\n=== Tipping Point Analysis ===")
print(tp_df.to_string(index=False, float_format='{:.4f}'.format))

# Find tipping point
tp_row = tp_df[tp_df['p_value'] > 0.05]
if len(tp_row) > 0:
    tp_value = tp_row.iloc[0]['delta']
    print(f"\nTipping point: delta = {tp_value:.1f}")
    print("The treatment effect becomes non-significant when imputed")
    print(f"active-arm values are shifted by {tp_value:.1f} units.")
else:
    print("\nNo tipping point found — result is robust across all tested deltas.")
```

**Output interpretation**: The tipping point is the delta shift applied to imputed values for active-arm dropouts that renders the treatment effect non-significant. A small tipping point (e.g., 1-2 units on the outcome scale) indicates fragility. A large tipping point (e.g., >8 units) indicates robustness. Consider whether the tipping-point delta is clinically plausible given the outcome scale and expected disease progression.

### Sensitivity Array (Grid Approach)

```python
import matplotlib.pyplot as plt

# Grid of confounder parameters
rr_eu = np.arange(1.0, 3.25, 0.25)
rr_ud = np.arange(1.0, 3.25, 0.25)
p0 = 0.25  # confounder prevalence in unexposed

# Compute adjusted OR for each combination
grid_or = np.zeros((len(rr_ud), len(rr_eu)))

for i, rud in enumerate(rr_ud):
    for j, reu in enumerate(rr_eu):
        p1 = p0 / (p0 + (1 - p0) / reu)
        bias = (rud * p1 + (1 - p1)) / (rud * p0 + (1 - p0))
        grid_or[i, j] = or_statin / bias

fig, ax = plt.subplots(figsize=(10, 8))
im = ax.imshow(grid_or, cmap='RdBu_r', origin='lower', aspect='auto',
               extent=[rr_eu[0]-0.125, rr_eu[-1]+0.125,
                       rr_ud[0]-0.125, rr_ud[-1]+0.125],
               vmin=0.4, vmax=1.2)

# Add contour at OR = 1.0 (null effect)
ax.contour(rr_eu, rr_ud, grid_or, levels=[1.0], colors='black',
           linewidths=2, linestyles='--')

# Add text annotations
for i, rud in enumerate(rr_ud):
    for j, reu in enumerate(rr_eu):
        color = 'white' if abs(grid_or[i, j] - 0.8) > 0.3 else 'black'
        ax.text(reu, rud, f'{grid_or[i, j]:.2f}', ha='center', va='center',
                fontsize=7, color=color)

# Mark known confounder benchmarks
ax.plot(1.4, 2.0, 'v', color='darkgreen', markersize=12, zorder=5)
ax.annotate('Smoking', xy=(1.4, 2.0), xytext=(1.55, 2.3),
            fontsize=10, color='darkgreen',
            arrowprops=dict(arrowstyle='->', color='darkgreen'))

plt.colorbar(im, ax=ax, label='Adjusted OR')
ax.set_xlabel('RR (Confounder-Exposure)', fontsize=12)
ax.set_ylabel('RR (Confounder-Outcome)', fontsize=12)
ax.set_title('Sensitivity Array: Adjusted OR by Confounder Strength\n'
             'Dashed line = null effect (OR = 1.0)', fontsize=13)
plt.tight_layout()
plt.savefig('sensitivity_array.png', dpi=150, bbox_inches='tight')
plt.show()
```

**Output interpretation**: The heatmap shows the bias-adjusted OR for each combination of confounder-exposure and confounder-outcome associations. Blue cells indicate a remaining protective effect; red cells indicate the effect is reversed. The dashed black contour marks OR = 1.0 (null). Combinations above/right of this line would fully explain away the observed association. The smoking benchmark helps calibrate: if it falls above the null line, a smoking-like confounder could explain the finding.

## Visualization

### Probabilistic Bias Analysis Distribution

```python
fig, axes = plt.subplots(1, 2, figsize=(14, 5))

# Histogram of adjusted ORs
axes[0].hist(pba['distribution'], bins=80, density=True,
             color='steelblue', edgecolor='white', alpha=0.7)
axes[0].axvline(x=1.0, color='red', linestyle='--', lw=2, label='Null (OR=1)')
axes[0].axvline(x=pba['median'], color='darkblue', linestyle='-', lw=2,
                label=f"Median = {pba['median']:.3f}")
axes[0].axvline(x=pba['ci_2.5'], color='darkblue', linestyle=':', lw=1.5)
axes[0].axvline(x=pba['ci_97.5'], color='darkblue', linestyle=':', lw=1.5)
axes[0].set_xlabel('Bias-Adjusted OR', fontsize=12)
axes[0].set_ylabel('Density', fontsize=12)
axes[0].set_title('Probabilistic Bias Analysis', fontsize=13)
axes[0].legend(fontsize=10)

# Tipping-point plot
axes[1].plot(tp_df['delta'], tp_df['estimate'], 'o-', color='steelblue',
             lw=2, markersize=5)
axes[1].fill_between(tp_df['delta'],
                     tp_df['ci_lower'], tp_df['ci_upper'],
                     alpha=0.15, color='steelblue')
axes[1].axhline(y=0, color='red', linestyle='--', lw=1.5)
if len(tp_row) > 0:
    axes[1].axvline(x=tp_value, color='red', linestyle=':', lw=1.5,
                    alpha=0.8)
    axes[1].annotate(f'Tipping point\n(delta={tp_value:.1f})',
                     xy=(tp_value, 0), xytext=(tp_value + 1, 1.5),
                     fontsize=10, color='red',
                     arrowprops=dict(arrowstyle='->', color='red'))
axes[1].set_xlabel('Delta (shift to imputed active-arm values)', fontsize=12)
axes[1].set_ylabel('Treatment Effect', fontsize=12)
axes[1].set_title('Tipping Point Analysis', fontsize=13)

plt.suptitle('Sensitivity Analysis Summary', fontsize=14, y=1.02)
plt.tight_layout()
plt.savefig('sensitivity_summary.png', dpi=150, bbox_inches='tight')
plt.show()
```

### E-Value Visualization

```python
fig, ax = plt.subplots(figsize=(8, 7))

# Plot the E-value curve
# For a protective effect, work with 1/RR
rr_inv = 1 / evalue_result['rr_approx']
eval_pt = evalue_result['evalue_point']

# Create the curve: combinations of RR_EU and RR_UD that produce the E-value
rr_range = np.linspace(1.0, eval_pt * 1.3, 200)
# The curve satisfies: RR_EU * RR_UD / (RR_EU + RR_UD - 1) >= RR_observed
# Rearranging: RR_UD >= RR_obs * (RR_EU - 1 + 1) / (RR_EU - RR_obs + RR_obs)
# Simplified boundary: for each RR_EU, find minimum RR_UD

rr_ud_boundary = []
for reu in rr_range:
    # Solve: reu * rud / (reu + rud - 1) = rr_inv
    # rud = rr_inv * (reu - 1) / (reu - rr_inv)
    if reu > rr_inv:
        rud = rr_inv * (reu - 1) / (reu - rr_inv)
        rr_ud_boundary.append(rud)
    else:
        rr_ud_boundary.append(np.nan)

rr_ud_boundary = np.array(rr_ud_boundary)
valid = ~np.isnan(rr_ud_boundary) & (rr_ud_boundary > 0) & (rr_ud_boundary < 10)

ax.fill_between(rr_range[valid], rr_ud_boundary[valid], 10,
                alpha=0.1, color='coral', label='Would explain away effect')
ax.plot(rr_range[valid], rr_ud_boundary[valid], '-', color='coral', lw=2)

# E-value point
ax.plot(eval_pt, eval_pt, 'D', color='red', markersize=12, zorder=5,
        label=f'E-value = {eval_pt:.2f}')

# Benchmark confounders
ax.plot(1.4, 2.0, '^', color='darkgreen', markersize=12, zorder=5,
        label='Smoking benchmark')
ax.plot(1.3, 1.8, 's', color='purple', markersize=10, zorder=5,
        label='Diabetes benchmark')

ax.set_xlabel('RR (Confounder-Exposure)', fontsize=12)
ax.set_ylabel('RR (Confounder-Outcome)', fontsize=12)
ax.set_title('E-Value Plot with Benchmark Confounders', fontsize=14)
ax.legend(loc='upper right', fontsize=10)
ax.set_xlim([1, eval_pt * 1.2])
ax.set_ylim([1, eval_pt * 1.2])
plt.tight_layout()
plt.savefig('evalue_plot.png', dpi=150, bbox_inches='tight')
plt.show()
```

**Output interpretation**: The E-value plot shows the boundary curve separating confounder parameter combinations that could (shaded region) and could not explain away the observed association. The diamond marks the E-value itself (the minimum point on the boundary where both RR parameters are equal). Benchmark confounders plotted on this space show whether known confounders of similar strength fall within the "could explain away" region.

## Tips and Best Practices

1. **Always compute E-values** for observational study results. They are simple to implement and provide an intuitive summary of robustness to unmeasured confounding.
2. **Calibrate against benchmarks**: An E-value without context is less informative. Always compare against the strength of measured or known confounders.
3. **Use probabilistic bias analysis** rather than single-scenario QBA when possible. It better reflects uncertainty in bias parameter values.
4. **Specify bias parameter distributions** from external evidence (validation studies, published literature, expert opinion), not from the study data itself.
5. **Tipping-point analysis** is the standard sensitivity analysis for clinical trials with missing data. Always present alongside primary multiple imputation results.
6. **Consider multiple biases**: Unmeasured confounding, misclassification, and selection bias may all be present simultaneously. Sequential or joint analysis is ideal.
7. **Pre-specify sensitivity analyses** in the study protocol to avoid post-hoc selection of favorable scenarios.
8. **For Python users**, consider using R packages via `rpy2` for more mature implementations (EValue, episensr, sensemakr, rbounds).
9. **Report sensitivity analyses prominently** in the main manuscript, not just supplementary material.
10. **Negative controls** provide empirical evidence about residual bias. When feasible, include both negative control exposures and negative control outcomes alongside the primary analysis.
