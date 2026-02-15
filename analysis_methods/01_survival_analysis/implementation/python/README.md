# Survival Analysis — Python Implementation

## Required Libraries

```bash
pip install lifelines scikit-survival matplotlib pandas numpy seaborn
```

```python
import numpy as np
import pandas as pd
import matplotlib.pyplot as plt
import seaborn as sns
from lifelines import KaplanMeierFitter, CoxPHFitter, WeibullAFTFitter, LogNormalAFTFitter
from lifelines import LogLogisticAFTFitter, WeibullFitter, ExponentialFitter
from lifelines.statistics import logrank_test, multivariate_logrank_test
from lifelines.plotting import plot_lifetimes
from sksurv.linear_model import CoxPHSurvivalAnalysis
from sksurv.metrics import concordance_index_censored
```

## Example Dataset

We use the NCCTG lung cancer dataset, equivalent to R's `survival::lung`. The dataset contains 228 patients with advanced lung cancer with survival times, censoring indicators, and clinical covariates.

```python
from lifelines.datasets import load_lung

lung = load_lung()
print(lung.head())
print(lung.describe())
print(f"Shape: {lung.shape}")
```

## Complete Worked Example

### Step 1: Load and Prepare Data

```python
from lifelines.datasets import load_lung

lung = load_lung()

# Recode status: lifelines expects 1 = event, 0 = censored
# In the original dataset, status 2 = dead, 1 = censored
lung['event'] = (lung['status'] == 2).astype(int)

# Recode sex for labeling
lung['sex_label'] = lung['sex'].map({1: 'Male', 2: 'Female'})

# Check missing values
print("Missing values:\n", lung.isnull().sum())

# Drop rows with missing covariates for modeling
lung_clean = lung.dropna(subset=['age', 'sex', 'ph.ecog', 'ph.karno', 'wt.loss']).copy()

print(f"\nSample size: {len(lung_clean)}")
print(f"Number of events: {lung_clean['event'].sum()}")
print(f"Censoring rate: {1 - lung_clean['event'].mean():.3f}")
```

**Interpretation**: After removing records with missing covariates, we have a complete dataset ready for analysis. The event rate and censoring rate are essential to understand the information content of the data.

### Step 2: Kaplan-Meier Estimation

```python
from lifelines import KaplanMeierFitter

kmf = KaplanMeierFitter()

# Fit overall KM
kmf.fit(lung_clean['time'], event_observed=lung_clean['event'], label='Overall')

# Print summary at specific time points
print("Survival probabilities at key time points:")
for t in [90, 180, 365, 730]:
    prob = kmf.predict(t)
    print(f"  S({t} days) = {prob:.3f}")

# Median survival
print(f"\nMedian survival time: {kmf.median_survival_time_:.1f} days")
print(f"95% CI for median: {kmf.confidence_interval_median_survival_time_}")
```

**Interpretation**: The KM estimator gives survival probabilities at landmark time points. The median survival time is the point where the survival curve crosses 0.5. For the lung dataset, median survival is approximately 310 days.

### Step 3: Kaplan-Meier Curves by Group

```python
fig, ax = plt.subplots(figsize=(10, 7))

for sex_val, sex_label in [(1, 'Male'), (2, 'Female')]:
    mask = lung_clean['sex'] == sex_val
    kmf_group = KaplanMeierFitter()
    kmf_group.fit(lung_clean.loc[mask, 'time'],
                  event_observed=lung_clean.loc[mask, 'event'],
                  label=sex_label)
    kmf_group.plot_survival_function(ax=ax, ci_show=True)

ax.set_xlabel('Time (days)', fontsize=12)
ax.set_ylabel('Survival Probability', fontsize=12)
ax.set_title('Kaplan-Meier Survival Curves by Sex', fontsize=14)
ax.legend(fontsize=11)
ax.grid(True, alpha=0.3)
plt.tight_layout()
plt.savefig('km_curves_by_sex.png', dpi=150)
plt.show()
```

**Interpretation**: The KM plot shows stratified survival curves. The visual separation of curves suggests a potential survival difference between males and females, which the log-rank test will formally assess.

### Step 4: Log-Rank Test

```python
from lifelines.statistics import logrank_test

male = lung_clean[lung_clean['sex'] == 1]
female = lung_clean[lung_clean['sex'] == 2]

result = logrank_test(male['time'], female['time'],
                      event_observed_A=male['event'],
                      event_observed_B=female['event'])

result.print_summary()
print(f"\nTest statistic: {result.test_statistic:.4f}")
print(f"P-value: {result.p_value:.4f}")
```

**Interpretation**: The log-rank test evaluates the null hypothesis that the two survival curves are identical. A small p-value (< 0.05) provides evidence of differing survival experiences. The test is most appropriate when the proportional hazards assumption holds.

### Step 5: Cox Proportional Hazards Model

```python
from lifelines import CoxPHFitter

# Prepare data for Cox model
cox_data = lung_clean[['time', 'event', 'age', 'sex', 'ph.ecog', 'ph.karno', 'wt.loss']].copy()

# Fit Cox PH model
cph = CoxPHFitter()
cph.fit(cox_data, duration_col='time', event_col='event')

# Print full summary
cph.print_summary()

# Hazard ratios
print("\nHazard Ratios:")
print(np.exp(cph.params_))

# Concordance index
print(f"\nConcordance index: {cph.concordance_index_:.4f}")
```

**Interpretation**: The Cox model summary shows coefficients (log HR), hazard ratios (exp(coef)), p-values, and confidence intervals. A hazard ratio < 1 indicates lower hazard (protective effect), and HR > 1 indicates higher hazard (risk factor). The concordance index measures the model's ability to correctly rank survival times.

### Step 6: Check Proportional Hazards Assumption

```python
# Test proportional hazards assumption
cph.check_assumptions(cox_data, p_value_threshold=0.05, show_plots=True)
```

**Interpretation**: The `check_assumptions()` method tests the PH assumption for each covariate using Schoenfeld residuals. It produces both statistical tests and diagnostic plots. A significant result (p < 0.05) for a covariate suggests its effect varies over time. The plots show scaled Schoenfeld residuals against time; a trend indicates PH violation.

### Step 7: Survival and Hazard Predictions

```python
# Predict survival function for specific profiles
new_patient = pd.DataFrame({
    'age': [65],
    'sex': [2],
    'ph.ecog': [1],
    'ph.karno': [80],
    'wt.loss': [5]
})

# Survival function prediction
surv_func = cph.predict_survival_function(new_patient)
surv_func.plot()
plt.xlabel('Time (days)')
plt.ylabel('Survival Probability')
plt.title('Predicted Survival Function for a 65-year-old Female (ECOG 1)')
plt.grid(True, alpha=0.3)
plt.show()

# Partial hazard (relative risk)
print(f"Partial hazard (relative risk): {cph.predict_partial_hazard(new_patient).values[0]:.4f}")

# Predicted median survival
print(f"Predicted median survival: {cph.predict_median(new_patient).values[0]:.1f} days")
```

**Interpretation**: Individual survival function predictions allow personalized risk assessment. The partial hazard gives the relative risk compared to the baseline, while the predicted median gives the expected time at which 50% survival is reached.

## Advanced Example

### Accelerated Failure Time Models

```python
from lifelines import WeibullAFTFitter, LogNormalAFTFitter, LogLogisticAFTFitter

aft_data = lung_clean[['time', 'event', 'age', 'sex', 'ph.ecog']].copy()

# Weibull AFT
waft = WeibullAFTFitter()
waft.fit(aft_data, duration_col='time', event_col='event')
print("=== Weibull AFT ===")
waft.print_summary()

# Log-normal AFT
lnaft = LogNormalAFTFitter()
lnaft.fit(aft_data, duration_col='time', event_col='event')
print("\n=== Log-Normal AFT ===")
lnaft.print_summary()

# Log-logistic AFT
llaft = LogLogisticAFTFitter()
llaft.fit(aft_data, duration_col='time', event_col='event')
print("\n=== Log-Logistic AFT ===")
llaft.print_summary()

# Compare models by AIC
models = {'Weibull': waft, 'Log-Normal': lnaft, 'Log-Logistic': llaft}
comparison = pd.DataFrame({
    'Model': list(models.keys()),
    'AIC': [m.AIC_ for m in models.values()],
    'BIC': [m.BIC_ for m in models.values()]
}).sort_values('AIC')
print("\nModel Comparison:")
print(comparison.to_string(index=False))
```

**Interpretation**: AFT models parametrically model survival times. Coefficients are interpreted as time ratios (acceleration factors). A coefficient of 0.3 on log scale means the covariate multiplies survival time by exp(0.3) = 1.35. AIC/BIC comparison selects the best-fitting distribution.

### Concordance Index Comparison with scikit-survival

```python
from sksurv.linear_model import CoxPHSurvivalAnalysis
from sksurv.metrics import concordance_index_censored

# Prepare structured array for sksurv
y_sksurv = np.array(
    [(bool(e), t) for e, t in zip(lung_clean['event'], lung_clean['time'])],
    dtype=[('event', bool), ('time', float)]
)

X = lung_clean[['age', 'sex', 'ph.ecog', 'ph.karno', 'wt.loss']].values

# Fit Cox PH via scikit-survival
skcox = CoxPHSurvivalAnalysis(alpha=0.01)
skcox.fit(X, y_sksurv)

# Concordance index
pred_risk = skcox.predict(X)
c_index = concordance_index_censored(
    lung_clean['event'].astype(bool).values,
    lung_clean['time'].values,
    pred_risk
)
print(f"Concordance index: {c_index[0]:.4f}")
print(f"Concordant pairs: {c_index[1]}")
print(f"Discordant pairs: {c_index[2]}")
```

**Interpretation**: scikit-survival provides a machine-learning-oriented interface for survival analysis. The concordance index measures how well the model's predicted risk scores agree with observed survival ordering. A C-index of 0.5 is random; values above 0.7 indicate good discrimination.

## Visualization

### Survival Function with Confidence Bands and Risk Table

```python
fig, axes = plt.subplots(2, 1, figsize=(10, 9), gridspec_kw={'height_ratios': [3, 1]})

# KM curves
for sex_val, sex_label, color in [(1, 'Male', '#E7B800'), (2, 'Female', '#2E9FDF')]:
    mask = lung_clean['sex'] == sex_val
    kmf_g = KaplanMeierFitter()
    kmf_g.fit(lung_clean.loc[mask, 'time'],
              event_observed=lung_clean.loc[mask, 'event'],
              label=sex_label)
    kmf_g.plot_survival_function(ax=axes[0], ci_show=True, color=color)

axes[0].set_xlabel('')
axes[0].set_ylabel('Survival Probability', fontsize=12)
axes[0].set_title('Kaplan-Meier Survival Estimates', fontsize=14)
axes[0].legend(fontsize=11)
axes[0].grid(True, alpha=0.3)

# Risk table
time_points = np.arange(0, 1001, 100)
for sex_val, sex_label in [(1, 'Male'), (2, 'Female')]:
    mask = lung_clean['sex'] == sex_val
    at_risk = []
    for t in time_points:
        n = ((lung_clean.loc[mask, 'time'] >= t)).sum()
        at_risk.append(n)
    axes[1].text(0, sex_val * 0.3, f'{sex_label}: ' + '  '.join(str(n) for n in at_risk),
                 fontsize=9, family='monospace')

axes[1].set_xlim(axes[0].get_xlim())
axes[1].set_ylim(0, 1.2)
axes[1].set_xlabel('Time (days)', fontsize=12)
axes[1].set_title('Number at Risk', fontsize=10)
axes[1].set_yticks([])
axes[1].grid(False)

plt.tight_layout()
plt.savefig('km_with_risk_table.png', dpi=150)
plt.show()
```

### Hazard Ratio Forest Plot

```python
# Extract coefficients for forest plot
summary_df = cph.summary

fig, ax = plt.subplots(figsize=(8, 5))

covariates = summary_df.index.tolist()
hrs = summary_df['exp(coef)'].values
lower = summary_df['exp(coef) lower 95%'].values
upper = summary_df['exp(coef) upper 95%'].values

y_pos = range(len(covariates))
ax.errorbar(hrs, y_pos, xerr=[hrs - lower, upper - hrs],
            fmt='o', color='navy', capsize=4, markersize=6)
ax.axvline(x=1, color='red', linestyle='--', linewidth=1, label='HR = 1')
ax.set_yticks(list(y_pos))
ax.set_yticklabels(covariates)
ax.set_xlabel('Hazard Ratio (95% CI)', fontsize=12)
ax.set_title('Forest Plot: Cox PH Model', fontsize=14)
ax.grid(True, alpha=0.3, axis='x')
plt.tight_layout()
plt.savefig('forest_plot.png', dpi=150)
plt.show()
```

### Baseline Hazard and Survival Curves

```python
fig, axes = plt.subplots(1, 2, figsize=(14, 5))

# Baseline survival
cph.plot_partial_effects_on_outcome(
    covariates='sex',
    values=[1, 2],
    ax=axes[0],
    plot_baseline=True
)
axes[0].set_title('Effect of Sex on Survival')
axes[0].legend(['Male', 'Female'])

# Baseline cumulative hazard
cph.baseline_cumulative_hazard_.plot(ax=axes[1])
axes[1].set_xlabel('Time (days)')
axes[1].set_ylabel('Cumulative Hazard')
axes[1].set_title('Baseline Cumulative Hazard')

plt.tight_layout()
plt.savefig('cox_diagnostics.png', dpi=150)
plt.show()
```

## Tips and Best Practices

1. **Data preparation**: lifelines expects a flat DataFrame with duration and event columns. Ensure the event column is binary (0/1) with 1 indicating the event occurred.

2. **Handling ties**: lifelines uses the Efron approximation by default in `CoxPHFitter`, which is generally preferred over the Breslow approximation when many tied event times exist.

3. **Convergence warnings**: If the Cox model does not converge, consider removing collinear covariates, standardizing continuous variables, or increasing `step_size` in the fitter.

4. **Penalization**: For high-dimensional data, use `CoxPHFitter(penalizer=0.1, l1_ratio=0.0)` to add L2 (Ridge) regularization. Set `l1_ratio=1.0` for L1 (Lasso).

5. **Large datasets**: For very large datasets, scikit-survival can be more efficient due to its NumPy-based backend. Consider `sksurv.ensemble.RandomSurvivalForest` for non-linear relationships.

6. **Reporting**: Always report the number of subjects, number of events, median follow-up time, and the concordance index alongside model results.

7. **Validation**: Use `lifelines.utils.k_fold_cross_validation` for internal validation:

```python
from lifelines.utils import k_fold_cross_validation
scores = k_fold_cross_validation(CoxPHFitter(), cox_data,
                                  duration_col='time', event_col='event', k=5)
print(f"Cross-validated C-index: {np.mean(scores):.4f} +/- {np.std(scores):.4f}")
```

8. **Avoid common pitfalls**:
   - Do not use survival status as a predictor variable.
   - Do not exclude censored observations; they contribute valid information.
   - Be cautious interpreting hazard ratios when the PH assumption is violated.
   - Do not confuse median survival time with mean survival time.

9. **Time-varying covariates**: Use the `add_covariate_to_timeline()` utility in lifelines to create an episodic dataset suitable for time-varying Cox models.

10. **Competing risks**: If subjects can experience multiple types of events, standard KM overestimates the event-specific cumulative incidence. Use competing risks methods (see the competing risks module).
