# Causal Inference — Python Implementation

## Required Libraries

```bash
pip install causalinference dowhy econml scikit-learn pandas numpy matplotlib seaborn statsmodels
```

```python
import numpy as np
import pandas as pd
import matplotlib.pyplot as plt
import seaborn as sns
from sklearn.linear_model import LogisticRegression, LinearRegression
from sklearn.ensemble import GradientBoostingClassifier, GradientBoostingRegressor
import statsmodels.api as sm
import warnings
warnings.filterwarnings('ignore')
```

## Example Dataset

We simulate an observational study comparing two treatments for hypertension, with confounding by indication. The true ATE is -8 mmHg.

```python
np.random.seed(42)
n = 1000

# Simulate confounders
age = np.round(np.random.normal(60, 10, n)).astype(int)
bmi = np.round(np.random.normal(28, 5, n), 1)
diabetes = np.random.binomial(1, 0.3, n)
smoking = np.random.binomial(1, 0.25, n)
baseline_sbp = np.round(np.random.normal(150, 15, n))
creatinine = np.round(np.random.normal(1.1, 0.3, n), 2)

# Treatment depends on confounders
from scipy.special import expit
lp_treat = -1.5 + 0.03 * age + 0.05 * bmi + 0.6 * diabetes + \
           0.4 * smoking + 0.02 * baseline_sbp - 0.5 * creatinine
prob_treat = expit(lp_treat)
treatment = np.random.binomial(1, prob_treat)

# Outcome with true ATE = -8
sbp_6m = (140 - 8 * treatment + 0.15 * age + 0.3 * bmi +
          3 * diabetes + 2 * smoking + 0.2 * baseline_sbp -
          2 * creatinine + np.random.normal(0, 8, n))
sbp_6m = np.round(sbp_6m, 1)

df = pd.DataFrame({
    'age': age, 'bmi': bmi, 'diabetes': diabetes, 'smoking': smoking,
    'baseline_sbp': baseline_sbp, 'creatinine': creatinine,
    'treatment': treatment, 'sbp_6m': sbp_6m
})

covariates = ['age', 'bmi', 'diabetes', 'smoking', 'baseline_sbp', 'creatinine']

print(f"Sample size: {len(df)}")
print(f"Treated: {treatment.sum()}, Control: {(1-treatment).sum()}")
print(f"True ATE: -8 mmHg")
```

## Complete Worked Example

### Step 1: Assess Baseline Imbalance

```python
def standardized_mean_diff(df, covariates, treatment_col):
    """Compute standardized mean differences."""
    treated = df[df[treatment_col] == 1]
    control = df[df[treatment_col] == 0]

    smds = {}
    for cov in covariates:
        d = treated[cov].mean() - control[cov].mean()
        pooled_sd = np.sqrt((treated[cov].var() + control[cov].var()) / 2)
        smds[cov] = d / pooled_sd if pooled_sd > 0 else 0
    return pd.Series(smds)

smd_before = standardized_mean_diff(df, covariates, 'treatment')
print("Standardized Mean Differences (unadjusted):")
for cov, smd in smd_before.items():
    flag = " ***" if abs(smd) > 0.1 else ""
    print(f"  {cov:15s}: {smd:+.4f}{flag}")
```

**Interpretation**: SMDs greater than 0.1 in absolute value (flagged with ***) indicate meaningful imbalance. These confounders need to be addressed through propensity score methods.

### Step 2: Naive Comparison

```python
naive_diff = df.loc[df['treatment'] == 1, 'sbp_6m'].mean() - \
             df.loc[df['treatment'] == 0, 'sbp_6m'].mean()
print(f"Naive difference: {naive_diff:.2f} mmHg")
print(f"True ATE: -8.00 mmHg")
print(f"Bias: {naive_diff - (-8):.2f} mmHg")
```

**Interpretation**: The naive difference is biased because confounders that influence both treatment assignment and outcome are not accounted for.

### Step 3: Propensity Score Estimation

```python
# Estimate propensity score
ps_model = LogisticRegression(max_iter=1000, C=1.0)
X = df[covariates].values
ps_model.fit(X, df['treatment'].values)
df['ps'] = ps_model.predict_proba(X)[:, 1]

# Visualize overlap
fig, ax = plt.subplots(figsize=(10, 6))
for trt, label, color in [(0, 'Control', '#E41A1C'), (1, 'Treated', '#377EB8')]:
    mask = df['treatment'] == trt
    ax.hist(df.loc[mask, 'ps'], bins=40, alpha=0.5, label=label,
            color=color, density=True, edgecolor='white')
ax.set_xlabel('Propensity Score', fontsize=12)
ax.set_ylabel('Density', fontsize=12)
ax.set_title('Propensity Score Distribution by Treatment Group', fontsize=14)
ax.legend(fontsize=11)
ax.grid(True, alpha=0.3)
plt.tight_layout()
plt.savefig('ps_overlap.png', dpi=150)
plt.show()
```

**Interpretation**: Good overlap between the propensity score distributions is required for valid causal inference. Regions with no overlap indicate positivity violations.

### Step 4: Inverse Probability of Treatment Weighting (IPTW)

```python
# Compute IPTW weights (ATE)
df['iptw'] = df['treatment'] / df['ps'] + (1 - df['treatment']) / (1 - df['ps'])

# Stabilized weights
p_treat = df['treatment'].mean()
df['iptw_stabilized'] = (df['treatment'] * p_treat / df['ps'] +
                          (1 - df['treatment']) * (1 - p_treat) / (1 - df['ps']))

# Check weight distribution
print("IPTW weight summary:")
print(f"  Mean: {df['iptw'].mean():.2f}")
print(f"  Max:  {df['iptw'].max():.2f}")
print(f"  Min:  {df['iptw'].min():.2f}")
print(f"  Stabilized mean: {df['iptw_stabilized'].mean():.2f}")

# Weighted ATE estimate
treated = df[df['treatment'] == 1]
control = df[df['treatment'] == 0]
ate_iptw = (np.average(treated['sbp_6m'], weights=treated['iptw']) -
            np.average(control['sbp_6m'], weights=control['iptw']))
print(f"\nATE estimate (IPTW): {ate_iptw:.2f} mmHg")

# Check balance after weighting
def weighted_smd(df, covariates, treatment_col, weight_col):
    treated = df[df[treatment_col] == 1]
    control = df[df[treatment_col] == 0]
    smds = {}
    for cov in covariates:
        m1 = np.average(treated[cov], weights=treated[weight_col])
        m0 = np.average(control[cov], weights=control[weight_col])
        pooled_sd = np.sqrt((treated[cov].var() + control[cov].var()) / 2)
        smds[cov] = (m1 - m0) / pooled_sd if pooled_sd > 0 else 0
    return pd.Series(smds)

smd_after = weighted_smd(df, covariates, 'treatment', 'iptw')
print("\nSMDs after IPTW:")
for cov, smd in smd_after.items():
    flag = " ***" if abs(smd) > 0.1 else ""
    print(f"  {cov:15s}: {smd:+.4f}{flag}")
```

**Interpretation**: IPTW creates a pseudo-population where confounders are balanced across treatment groups. The weighted mean difference estimates the ATE. Stabilized weights maintain the original sample size and reduce variance. All SMDs should be below 0.1 after weighting.

### Step 5: G-Computation

```python
# Fit outcome model with treatment-confounder interactions
from sklearn.linear_model import LinearRegression

formula_vars = covariates + ['treatment']
X_outcome = df[formula_vars].copy()
for cov in covariates:
    X_outcome[f'treatment_x_{cov}'] = df['treatment'] * df[cov]

outcome_model = LinearRegression()
outcome_model.fit(X_outcome, df['sbp_6m'])

# Predict under treatment and control for everyone
X_treat = X_outcome.copy()
X_treat['treatment'] = 1
for cov in covariates:
    X_treat[f'treatment_x_{cov}'] = 1 * df[cov]

X_control = X_outcome.copy()
X_control['treatment'] = 0
for cov in covariates:
    X_control[f'treatment_x_{cov}'] = 0 * df[cov]

pred_treat = outcome_model.predict(X_treat)
pred_control = outcome_model.predict(X_control)

ate_gcomp = np.mean(pred_treat) - np.mean(pred_control)
print(f"ATE estimate (G-computation): {ate_gcomp:.2f} mmHg")

# Bootstrap confidence interval
n_boot = 1000
ate_boot = np.zeros(n_boot)
for b in range(n_boot):
    idx = np.random.choice(len(df), len(df), replace=True)
    boot_df = df.iloc[idx].copy()
    X_b = boot_df[formula_vars].copy()
    for cov in covariates:
        X_b[f'treatment_x_{cov}'] = boot_df['treatment'] * boot_df[cov]
    mod_b = LinearRegression().fit(X_b, boot_df['sbp_6m'])

    X_t = X_b.copy(); X_t['treatment'] = 1
    X_c = X_b.copy(); X_c['treatment'] = 0
    for cov in covariates:
        X_t[f'treatment_x_{cov}'] = 1 * boot_df[cov]
        X_c[f'treatment_x_{cov}'] = 0 * boot_df[cov]
    ate_boot[b] = np.mean(mod_b.predict(X_t)) - np.mean(mod_b.predict(X_c))

ci_low, ci_high = np.percentile(ate_boot, [2.5, 97.5])
print(f"95% CI: ({ci_low:.2f}, {ci_high:.2f})")
```

**Interpretation**: G-computation standardizes the treatment effect over the covariate distribution. It requires correct outcome model specification. Bootstrap provides valid confidence intervals.

### Step 6: Doubly Robust Estimation (AIPW)

```python
def aipw_estimator(Y, A, X, ps_model_class=LogisticRegression,
                    outcome_model_class=LinearRegression):
    """
    Augmented Inverse Probability Weighting (AIPW) estimator for ATE.
    """
    n = len(Y)

    # Fit propensity score model
    ps_mod = ps_model_class(max_iter=1000)
    ps_mod.fit(X, A)
    e = ps_mod.predict_proba(X)[:, 1]
    e = np.clip(e, 0.01, 0.99)  # Truncate extreme probabilities

    # Fit outcome models
    X_treat = X[A == 1]
    X_control = X[A == 0]

    mu1_mod = outcome_model_class()
    mu1_mod.fit(X_treat, Y[A == 1])
    mu1 = mu1_mod.predict(X)

    mu0_mod = outcome_model_class()
    mu0_mod.fit(X_control, Y[A == 0])
    mu0 = mu0_mod.predict(X)

    # AIPW estimator
    aipw_1 = A * (Y - mu1) / e + mu1
    aipw_0 = (1 - A) * (Y - mu0) / (1 - e) + mu0

    ate = np.mean(aipw_1) - np.mean(aipw_0)
    se = np.sqrt(np.var(aipw_1 - aipw_0) / n)

    return ate, se

ate_aipw, se_aipw = aipw_estimator(
    df['sbp_6m'].values,
    df['treatment'].values,
    df[covariates].values
)

print(f"ATE estimate (AIPW): {ate_aipw:.2f} mmHg")
print(f"Standard error: {se_aipw:.2f}")
print(f"95% CI: ({ate_aipw - 1.96*se_aipw:.2f}, {ate_aipw + 1.96*se_aipw:.2f})")
```

**Interpretation**: The AIPW estimator is doubly robust: it is consistent if either the propensity score model or the outcome model is correctly specified. The influence-function-based standard error provides valid inference.

### Step 7: Using DoWhy for Causal Analysis

```python
import dowhy
from dowhy import CausalModel

# Define causal model
model = CausalModel(
    data=df,
    treatment='treatment',
    outcome='sbp_6m',
    common_causes=covariates
)

# Identify causal effect
identified_estimand = model.identify_effect(proceed_when_unidentifiable=True)
print(identified_estimand)

# Estimate using different methods
# Method 1: Propensity score matching
estimate_matching = model.estimate_effect(
    identified_estimand,
    method_name="backdoor.propensity_score_matching"
)
print(f"\nDoWhy PS Matching: {estimate_matching.value:.2f}")

# Method 2: IPTW
estimate_iptw = model.estimate_effect(
    identified_estimand,
    method_name="backdoor.propensity_score_weighting"
)
print(f"DoWhy IPTW: {estimate_iptw.value:.2f}")

# Method 3: Linear regression
estimate_lr = model.estimate_effect(
    identified_estimand,
    method_name="backdoor.linear_regression"
)
print(f"DoWhy Linear Regression: {estimate_lr.value:.2f}")

# Refutation: Add random common cause
refutation = model.refute_estimate(
    identified_estimand,
    estimate_iptw,
    method_name="random_common_cause"
)
print(f"\nRefutation (random common cause):\n{refutation}")
```

**Interpretation**: DoWhy provides a structured causal inference workflow: model, identify, estimate, refute. The refutation step tests the robustness of the estimate to potential violations of assumptions (e.g., adding a random common cause should not substantially change the estimate).

## Advanced Example

### EconML: Heterogeneous Treatment Effects

```python
from econml.dml import LinearDML
from sklearn.ensemble import GradientBoostingRegressor, GradientBoostingClassifier

# Fit Double Machine Learning model
dml = LinearDML(
    model_y=GradientBoostingRegressor(n_estimators=100, max_depth=3),
    model_t=GradientBoostingClassifier(n_estimators=100, max_depth=3),
    cv=5,
    random_state=42
)

X_effect = df[['age', 'diabetes']].values  # Effect modifiers
W = df[covariates].values  # Controls

dml.fit(df['sbp_6m'].values, df['treatment'].values, X=X_effect, W=W)

# Average treatment effect
ate_dml = dml.ate(X_effect)
ci_dml = dml.ate_interval(X_effect, alpha=0.05)
print(f"ATE (DML): {ate_dml:.2f} mmHg")
print(f"95% CI: ({ci_dml[0]:.2f}, {ci_dml[1]:.2f})")

# Heterogeneous effects by age
age_grid = np.column_stack([np.arange(35, 85), np.zeros(50)])
cate = dml.effect(age_grid)
cate_ci = dml.effect_interval(age_grid, alpha=0.05)

fig, ax = plt.subplots(figsize=(10, 6))
ax.plot(np.arange(35, 85), cate, color='navy', linewidth=2, label='CATE')
ax.fill_between(np.arange(35, 85), cate_ci[0], cate_ci[1],
                alpha=0.2, color='navy', label='95% CI')
ax.axhline(y=-8, color='red', linestyle='--', label='True ATE = -8')
ax.set_xlabel('Age', fontsize=12)
ax.set_ylabel('Conditional Average Treatment Effect (mmHg)', fontsize=12)
ax.set_title('Heterogeneous Treatment Effect by Age (DML)', fontsize=14)
ax.legend(fontsize=11)
ax.grid(True, alpha=0.3)
plt.tight_layout()
plt.savefig('cate_by_age.png', dpi=150)
plt.show()
```

**Interpretation**: Double Machine Learning estimates heterogeneous treatment effects (CATEs) while using flexible ML models for nuisance parameters. The CATE plot reveals how the treatment effect varies across subgroups, which is valuable for precision medicine.

### Summary of All Estimates

```python
results = pd.DataFrame({
    'Method': ['Naive', 'IPTW', 'G-Computation', 'AIPW', 'DoWhy IPTW', 'DML'],
    'Estimate': [naive_diff, ate_iptw, ate_gcomp, ate_aipw,
                 estimate_iptw.value, ate_dml],
    'True_ATE': [-8] * 6
})
results['Bias'] = results['Estimate'] - results['True_ATE']
results['Estimate'] = results['Estimate'].round(2)
results['Bias'] = results['Bias'].round(2)
print(results.to_string(index=False))
```

## Visualization

### Love Plot: Balance Before and After Weighting

```python
fig, ax = plt.subplots(figsize=(8, 6))

y_pos = np.arange(len(covariates))
ax.scatter(smd_before.abs().values, y_pos, color='red', s=80,
           marker='o', label='Before', zorder=3)
ax.scatter(smd_after.abs().values, y_pos, color='blue', s=80,
           marker='^', label='After IPTW', zorder=3)
ax.axvline(x=0.1, color='gray', linestyle='--', linewidth=1, label='Threshold (0.1)')
ax.set_yticks(y_pos)
ax.set_yticklabels(covariates)
ax.set_xlabel('Absolute Standardized Mean Difference', fontsize=12)
ax.set_title('Covariate Balance: Love Plot', fontsize=14)
ax.legend(fontsize=10)
ax.grid(True, alpha=0.3, axis='x')
plt.tight_layout()
plt.savefig('love_plot.png', dpi=150)
plt.show()
```

### Treatment Effect Comparison

```python
fig, ax = plt.subplots(figsize=(9, 5))

methods = results['Method'].values[1:]  # Exclude naive
estimates = results['Estimate'].values[1:]

ax.barh(methods, estimates, color='steelblue', alpha=0.7, edgecolor='navy')
ax.axvline(x=-8, color='red', linestyle='--', linewidth=2, label='True ATE = -8')
ax.set_xlabel('Estimated Treatment Effect (mmHg)', fontsize=12)
ax.set_title('Causal Effect Estimates Across Methods', fontsize=14)
ax.legend(fontsize=11)
ax.grid(True, alpha=0.3, axis='x')
plt.tight_layout()
plt.savefig('method_comparison.png', dpi=150)
plt.show()
```

## Tips and Best Practices

1. **Start with a DAG**: Use `dowhy` or `dagitty` (via `pydagitty`) to encode and verify your causal assumptions before analysis.

2. **Check overlap rigorously**: Plot propensity score distributions. If extreme weights exist (max weight > 20), consider trimming or overlap weighting.

3. **Use cross-fitting for ML-based estimators**: EconML and similar packages use cross-fitting by default to prevent overfitting bias when using flexible models for nuisance parameters.

4. **Truncate propensity scores**: Clip extreme values (e.g., `np.clip(ps, 0.01, 0.99)`) to prevent explosive weights.

5. **Conduct sensitivity analysis**: Use the E-value to quantify how strong unmeasured confounding would need to be:

```python
def e_value(hr):
    return hr + np.sqrt(hr * (hr - 1))
```

6. **Report balance diagnostics**: Always show covariate balance before and after adjustment. Balance is more important than the propensity score model's predictive accuracy.

7. **Consider multiple estimators**: If IPTW, G-computation, and AIPW all agree, the result is more credible. Disagreement suggests model misspecification.

8. **Use DoWhy's refutation methods**: Random common cause, placebo treatment, and data subset refutations test the robustness of your findings.

9. **Be explicit about the estimand**: Clearly state whether you are targeting ATE, ATT, or another estimand. Different methods target different estimands by default.

10. **Document all analytic decisions**: Causal inference involves many analyst choices (confounder set, model specification, trimming thresholds). Pre-registration and protocol writing help prevent p-hacking.
