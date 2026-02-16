# Missing Data Methods — Python Implementation

## Required Libraries

```bash
pip install numpy pandas scikit-learn miceforest missingno matplotlib seaborn statsmodels
```

- **numpy / pandas**: Core data manipulation.
- **scikit-learn**: `SimpleImputer`, `IterativeImputer`, `KNNImputer` for single and iterative imputation.
- **miceforest**: MICE implementation using LightGBM (fast, handles mixed types).
- **missingno**: Missing data visualization.
- **matplotlib / seaborn**: Plotting.
- **statsmodels**: Regression models for analysis of imputed datasets.

## Example Dataset

We simulate a clinical trial with 500 patients randomized to treatment or placebo, measuring HbA1c change from baseline to week 16. Approximately 25% of patients have missing week-16 outcomes under a MAR mechanism.

```python
import numpy as np
import pandas as pd

np.random.seed(42)
n = 500

trial_data = pd.DataFrame({
    'id': range(1, n + 1),
    'treatment': np.repeat(['Placebo', 'Active'], n // 2),
    'age': np.round(np.random.normal(55, 10, n)).astype(int),
    'sex': np.random.choice(['Male', 'Female'], n),
    'bmi': np.round(np.random.normal(30, 5, n), 1),
    'hba1c_bl': np.round(np.random.normal(8.5, 1.2, n), 1),
})

# Simulate longitudinal HbA1c outcomes
trt_effect = np.where(trial_data['treatment'] == 'Active', -0.8, 0.0)
noise = np.random.normal(0, 0.3, (n, 4))
increments = np.column_stack([trt_effect / 4] * 4) + noise
trajectories = trial_data['hba1c_bl'].values[:, None] + np.cumsum(increments, axis=1)

trial_data['hba1c_w4'] = np.round(trajectories[:, 0], 1)
trial_data['hba1c_w8'] = np.round(trajectories[:, 1], 1)
trial_data['hba1c_w12'] = np.round(trajectories[:, 2], 1)
trial_data['hba1c_w16'] = np.round(trajectories[:, 3], 1)

# Introduce MAR missingness
miss_prob_w16 = 1 / (1 + np.exp(-(-2 + 0.3 * (trial_data['hba1c_bl'] - 8.5) +
                                    0.5 * (trial_data['treatment'] == 'Placebo').astype(int))))
missing_w16 = np.random.random(n) < miss_prob_w16
trial_data.loc[missing_w16, 'hba1c_w16'] = np.nan

miss_prob_w12 = 1 / (1 + np.exp(-(-3 + 0.2 * (trial_data['hba1c_bl'] - 8.5))))
missing_w12 = np.random.random(n) < miss_prob_w12
trial_data.loc[missing_w12, 'hba1c_w12'] = np.nan
trial_data.loc[missing_w12, 'hba1c_w16'] = np.nan  # monotone dropout

print(f"Missing at week 12: {trial_data['hba1c_w12'].isna().sum()} / {n}")
print(f"Missing at week 16: {trial_data['hba1c_w16'].isna().sum()} / {n}")
```

## Complete Worked Example

### Step 1: Visualize Missing Data Patterns

```python
import missingno as msno
import matplotlib.pyplot as plt

# Missing data matrix
fig, ax = plt.subplots(figsize=(10, 6))
msno.matrix(trial_data[['hba1c_bl', 'hba1c_w4', 'hba1c_w8',
                          'hba1c_w12', 'hba1c_w16']],
            ax=ax, sparkline=False, color=(0.27, 0.51, 0.71))
ax.set_title('Missing Data Pattern Matrix', fontsize=14)
plt.tight_layout()
plt.savefig('missing_matrix.png', dpi=150, bbox_inches='tight')
plt.show()

# Bar chart of missingness by variable
fig, ax = plt.subplots(figsize=(8, 5))
msno.bar(trial_data[['hba1c_bl', 'hba1c_w4', 'hba1c_w8',
                      'hba1c_w12', 'hba1c_w16']],
         ax=ax, color='steelblue')
ax.set_title('Data Completeness by Visit', fontsize=14)
plt.tight_layout()
plt.savefig('missing_bar.png', dpi=150, bbox_inches='tight')
plt.show()

# Heatmap of nullity correlations
fig, ax = plt.subplots(figsize=(8, 6))
msno.heatmap(trial_data[['hba1c_w4', 'hba1c_w8', 'hba1c_w12', 'hba1c_w16']],
             ax=ax, cmap='RdBu')
ax.set_title('Missingness Correlation Heatmap', fontsize=14)
plt.tight_layout()
plt.savefig('missing_heatmap.png', dpi=150, bbox_inches='tight')
plt.show()
```

**Output interpretation**: The matrix plot shows each row (patient) and column (variable), with white indicating missing values. The pattern should reveal monotone dropout (missing from a visit onward). The bar chart shows the proportion of data present per visit. The heatmap reveals whether variables tend to be missing together (high correlation = monotone dropout pattern).

### Step 2: Complete Case Analysis (Baseline Comparison)

```python
import statsmodels.formula.api as smf

# Complete case analysis
trial_cc = trial_data.dropna(subset=['hba1c_w16']).copy()
trial_cc['change_w16'] = trial_cc['hba1c_w16'] - trial_cc['hba1c_bl']

cc_model = smf.ols('change_w16 ~ C(treatment) + hba1c_bl', data=trial_cc).fit()
print("=== Complete Case Analysis ===")
print(f"N used: {len(trial_cc)} / {n}")
print(cc_model.summary().tables[1])
```

**Output interpretation**: The complete case analysis uses only patients with observed week-16 data. Under MAR, this can produce biased treatment effect estimates because the reasons for dropout are related to baseline severity and treatment arm. The reduced sample size also decreases power.

### Step 3: Multiple Imputation with miceforest

```python
import miceforest as mf

# Prepare data for imputation
imp_cols = ['treatment', 'age', 'sex', 'bmi', 'hba1c_bl',
            'hba1c_w4', 'hba1c_w8', 'hba1c_w12', 'hba1c_w16']

imp_data = trial_data[imp_cols].copy()
# Encode categoricals for miceforest
imp_data['treatment_num'] = (imp_data['treatment'] == 'Active').astype(int)
imp_data['sex_num'] = (imp_data['sex'] == 'Male').astype(int)
imp_data_numeric = imp_data.drop(columns=['treatment', 'sex'])

# Create kernel and run imputation
kernel = mf.ImputationKernel(
    imp_data_numeric,
    num_datasets=25,
    random_state=123
)

# Run MICE iterations
kernel.mice(iterations=20, verbose=False)

print("Imputation complete: 25 datasets, 20 iterations each")
```

**Output interpretation**: miceforest uses LightGBM as the imputation model, which handles non-linear relationships and interactions automatically. Twenty-five imputed datasets are created, each reflecting a different plausible completion of the missing data.

### Step 4: Analyze Imputed Datasets and Pool Results (Rubin's Rules)

```python
from scipy import stats

estimates = []
variances = []

for d in range(25):
    completed = kernel.complete_data(dataset=d)

    # Merge back treatment labels
    completed['treatment'] = imp_data['treatment'].values
    completed['change_w16'] = completed['hba1c_w16'] - completed['hba1c_bl']

    model = smf.ols('change_w16 ~ C(treatment) + hba1c_bl',
                    data=completed).fit()

    # Extract treatment effect (Active vs Placebo)
    trt_idx = [i for i, name in enumerate(model.params.index)
               if 'Active' in name][0]
    estimates.append(model.params.iloc[trt_idx])
    variances.append(model.bse.iloc[trt_idx] ** 2)

# Rubin's rules
m = 25
theta_bar = np.mean(estimates)
W_bar = np.mean(variances)
B = np.var(estimates, ddof=1)
T_total = W_bar + (1 + 1/m) * B
se_total = np.sqrt(T_total)

# Degrees of freedom (Barnard-Rubin adjustment)
r = (1 + 1/m) * B / W_bar
nu_old = (m - 1) * (1 + 1/r) ** 2
nu_obs = (len(trial_cc) - 3) * (1 + W_bar / ((1 + 1/m) * B)) / 2  # adjusted
nu = (nu_old * nu_obs) / (nu_old + nu_obs)

# Confidence interval and p-value
t_crit = stats.t.ppf(0.975, df=nu)
ci_lower = theta_bar - t_crit * se_total
ci_upper = theta_bar + t_crit * se_total
t_stat = theta_bar / se_total
p_value = 2 * stats.t.sf(np.abs(t_stat), df=nu)

# Fraction of missing information
fmi = (B + B/m) / T_total

print("=== Multiple Imputation Results (Rubin's Rules) ===")
print(f"Pooled treatment effect: {theta_bar:.4f}")
print(f"Standard error: {se_total:.4f}")
print(f"95% CI: ({ci_lower:.4f}, {ci_upper:.4f})")
print(f"t-statistic: {t_stat:.4f}, df = {nu:.1f}")
print(f"p-value: {p_value:.6f}")
print(f"Fraction of missing information (FMI): {fmi:.3f}")
print(f"Between-imputation variance: {B:.6f}")
print(f"Within-imputation variance: {W_bar:.6f}")
```

**Output interpretation**: The pooled treatment effect averages the 25 complete-data estimates. The total variance (T) combines within-imputation variance (W, sampling uncertainty) and between-imputation variance (B, missing-data uncertainty). FMI indicates how much the missing data inflates uncertainty. FMI above 0.3 suggests missing data has a meaningful impact on the results. The degrees of freedom may be small when FMI is high, leading to wider confidence intervals.

### Step 5: Compare Complete Case vs MI Results

```python
print("\n=== Comparison ===")
print(f"{'Method':<25} {'Estimate':>10} {'SE':>10} {'95% CI':>25} {'p-value':>10}")
print("-" * 80)

# CC results
cc_est = cc_model.params['C(treatment)[T.Active]']
cc_se = cc_model.bse['C(treatment)[T.Active]']
cc_ci = cc_model.conf_int().loc['C(treatment)[T.Active]']
cc_p = cc_model.pvalues['C(treatment)[T.Active]']
print(f"{'Complete Case':<25} {cc_est:>10.4f} {cc_se:>10.4f} "
      f"{'({:.4f}, {:.4f})'.format(cc_ci[0], cc_ci[1]):>25} {cc_p:>10.6f}")

print(f"{'Multiple Imputation':<25} {theta_bar:>10.4f} {se_total:>10.4f} "
      f"{'({:.4f}, {:.4f})'.format(ci_lower, ci_upper):>25} {p_value:>10.6f}")
```

**Output interpretation**: Under MAR, the MI estimate is expected to be less biased than the complete case estimate. The MI standard error may be larger (reflecting uncertainty from missing data) or smaller (because it uses all available data). A meaningful difference between the two estimates suggests that the complete case analysis is biased.

## Advanced Example

### Tipping-Point Sensitivity Analysis

```python
deltas = np.arange(0, 1.6, 0.1)
tipping_results = []

for delta in deltas:
    delta_estimates = []
    delta_variances = []

    for d in range(25):
        completed = kernel.complete_data(dataset=d).copy()
        completed['treatment'] = imp_data['treatment'].values
        completed['change_w16'] = completed['hba1c_w16'] - completed['hba1c_bl']

        # Add delta to imputed values in the active arm only
        was_missing = trial_data['hba1c_w16'].isna().values
        is_active = (completed['treatment'] == 'Active').values
        adjust_mask = was_missing & is_active
        completed.loc[adjust_mask, 'change_w16'] += delta

        model = smf.ols('change_w16 ~ C(treatment) + hba1c_bl',
                        data=completed).fit()

        trt_idx = [i for i, name in enumerate(model.params.index)
                   if 'Active' in name][0]
        delta_estimates.append(model.params.iloc[trt_idx])
        delta_variances.append(model.bse.iloc[trt_idx] ** 2)

    # Pool with Rubin's rules
    theta_d = np.mean(delta_estimates)
    W_d = np.mean(delta_variances)
    B_d = np.var(delta_estimates, ddof=1)
    T_d = W_d + (1 + 1/m) * B_d
    se_d = np.sqrt(T_d)
    r_d = (1 + 1/m) * B_d / W_d
    nu_d = (m - 1) * (1 + 1/r_d) ** 2 if r_d > 0 else 1000
    t_stat_d = theta_d / se_d
    p_d = 2 * stats.t.sf(np.abs(t_stat_d), df=nu_d)

    tipping_results.append({
        'delta': delta,
        'estimate': theta_d,
        'se': se_d,
        'ci_lower': theta_d - 1.96 * se_d,
        'ci_upper': theta_d + 1.96 * se_d,
        'p_value': p_d
    })

tp_df = pd.DataFrame(tipping_results)
print("=== Tipping Point Analysis ===")
print(tp_df.to_string(index=False, float_format='{:.4f}'.format))

# Find tipping point
tp_row = tp_df[tp_df['p_value'] > 0.05].iloc[0] if (tp_df['p_value'] > 0.05).any() else None
if tp_row is not None:
    print(f"\nTipping point: delta = {tp_row['delta']:.1f}")
    print("At this shift, the treatment effect is no longer statistically significant.")
else:
    print("\nNo tipping point found within the tested range — results are robust.")
```

**Output interpretation**: The tipping-point analysis adds a penalty (delta) to the imputed HbA1c changes for active-arm dropouts, assuming they do worse than MAR predicts. The tipping point is the smallest delta that nullifies the treatment effect. A tipping point of 0.5 or less would be clinically plausible and concerning; a tipping point above 1.0 suggests robustness.

## Visualization

### Tipping-Point Plot

```python
fig, ax = plt.subplots(figsize=(9, 6))

ax.plot(tp_df['delta'], tp_df['estimate'], 'o-', color='steelblue',
        lw=2, markersize=6, label='Treatment effect')
ax.fill_between(tp_df['delta'], tp_df['ci_lower'], tp_df['ci_upper'],
                alpha=0.15, color='steelblue')
ax.axhline(y=0, color='red', linestyle='--', lw=1, label='No effect')

if tp_row is not None:
    ax.axvline(x=tp_row['delta'], color='red', linestyle=':',
               lw=1.5, alpha=0.8)
    ax.annotate(f"Tipping point\n(delta = {tp_row['delta']:.1f})",
                xy=(tp_row['delta'], tp_row['estimate']),
                xytext=(tp_row['delta'] + 0.2, tp_row['estimate'] + 0.1),
                fontsize=10, color='red',
                arrowprops=dict(arrowstyle='->', color='red'))

ax.set_xlabel('Delta (added to imputed active-arm values)', fontsize=12)
ax.set_ylabel('Pooled Treatment Effect (HbA1c change)', fontsize=12)
ax.set_title('Tipping Point Sensitivity Analysis', fontsize=14)
ax.legend(loc='upper left', fontsize=10)
plt.tight_layout()
plt.savefig('tipping_point.png', dpi=150, bbox_inches='tight')
plt.show()
```

### Distribution of Imputed vs Observed Values

```python
fig, axes = plt.subplots(1, 2, figsize=(12, 5))

for idx, trt in enumerate(['Placebo', 'Active']):
    ax = axes[idx]
    trt_mask = trial_data['treatment'] == trt
    observed = trial_data.loc[trt_mask & trial_data['hba1c_w16'].notna(), 'hba1c_w16']

    # Get imputed values from first imputation
    completed_0 = kernel.complete_data(dataset=0)
    imputed_vals = completed_0.loc[trt_mask & trial_data['hba1c_w16'].isna(), 'hba1c_w16']

    ax.hist(observed, bins=20, density=True, alpha=0.6,
            color='steelblue', label='Observed', edgecolor='white')
    if len(imputed_vals) > 0:
        ax.hist(imputed_vals, bins=15, density=True, alpha=0.6,
                color='coral', label='Imputed', edgecolor='white')
    ax.set_title(f'{trt} Arm: Week 16 HbA1c', fontsize=12)
    ax.set_xlabel('HbA1c')
    ax.set_ylabel('Density')
    ax.legend()

plt.suptitle('Observed vs Imputed Distributions', fontsize=14, y=1.02)
plt.tight_layout()
plt.savefig('imputed_distributions.png', dpi=150, bbox_inches='tight')
plt.show()
```

**Output interpretation**: Comparing observed and imputed distributions reveals whether the imputation model generates plausible values. Under MAR, the imputed distribution may differ from the observed (e.g., imputed dropouts may have worse values). Very similar distributions in a setting with differential dropout may indicate an imputation model that is not capturing the MAR mechanism well.

## Tips and Best Practices

1. **Visualize missing data first**: Use `missingno` to identify patterns, rates, and potential mechanisms before choosing a method.
2. **Use miceforest for flexible MICE**: It handles non-linear relationships via LightGBM without manual model specification.
3. **Include auxiliary variables**: Add variables correlated with the outcome or missingness to strengthen the MAR assumption.
4. **Use sufficient imputations**: m = 25 or more for clinical trial analyses. With FMI above 0.5, use m = 50+.
5. **Pool results correctly**: Always use Rubin's rules. Do not average imputed datasets and analyze once — this underestimates variance.
6. **Never impute the dependent variable** in a predictive model deployment (only during model development for training).
7. **Validate against R**: Python MI tools are less mature; cross-check key results against the `mice` R package.
8. **Perform sensitivity analyses**: Tipping-point analysis is practical and interpretable. Always explore MNAR scenarios.
9. **Report transparently**: State the missingness rate, assumed mechanism, imputation method, number of imputations, convergence diagnostics, and FMI.
10. **Align with the estimand**: Ensure the missing data method matches the intercurrent event strategy specified in the statistical analysis plan (ICH E9(R1)).
