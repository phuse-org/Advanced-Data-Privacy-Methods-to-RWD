# Quasi-Experimental Methods — Python Implementation

## Required Libraries

```bash
pip install statsmodels numpy pandas matplotlib scipy rdrobust seaborn
```

```python
import numpy as np
import pandas as pd
import matplotlib.pyplot as plt
import statsmodels.api as sm
from statsmodels.formula.api import ols
from scipy import stats
```

## Example Dataset

### Dataset 1: ITS — FDA Warning and Opioid Prescribing

```python
np.random.seed(123)

n_months = 48
time = np.arange(1, n_months + 1)
intervention = (time > 24).astype(int)
time_after = np.where(time > 24, time - 24, 0)

# Pre: upward trend; Post: level drop + declining trend
prescriptions = (450 + 2 * time - 40 * intervention - 3 * time_after +
                 np.random.normal(0, 12, n_months))

its_df = pd.DataFrame({
    'month': time, 'prescriptions': prescriptions,
    'intervention': intervention, 'time_after': time_after
})
```

### Dataset 2: DiD — Hospital Sepsis Bundle

```python
np.random.seed(456)
n_hospitals = 200
n_periods = 10

hospital_ids = np.repeat(np.arange(n_hospitals), n_periods)
periods = np.tile(np.arange(1, n_periods + 1), n_hospitals)
treated_group = (hospital_ids < 100).astype(int)
post = (periods > 5).astype(int)
hospital_fe = np.repeat(np.random.normal(0, 5, n_hospitals), n_periods)

mortality = (20 + hospital_fe + 0.5 * periods -
             3 * treated_group * post + np.random.normal(0, 2, len(hospital_ids)))

did_df = pd.DataFrame({
    'hospital': hospital_ids, 'period': periods,
    'treated': treated_group, 'post': post, 'mortality': mortality
})
```

### Dataset 3: RD — Statin Prescription at LDL Threshold

```python
np.random.seed(789)
n_patients = 2000

ldl = np.random.uniform(140, 240, n_patients)
above = (ldl >= 190).astype(int)
prob_statin = np.where(ldl >= 190, 0.75, 0.20)
statin = np.random.binomial(1, prob_statin)
cv_events = np.random.poisson(np.exp(0.5 + 0.01 * (ldl - 190) - 0.4 * statin))

rd_df = pd.DataFrame({
    'ldl': ldl, 'above': above, 'statin': statin, 'cv_events': cv_events
})
```

## Complete Worked Example

### Part A: Interrupted Time Series

```python
# Fit segmented regression with OLS
its_model = ols('prescriptions ~ month + intervention + time_after', data=its_df).fit()

# Newey-West standard errors for autocorrelation
its_robust = its_model.get_robustcov_results(cov_type='HAC', maxlags=3)
print(its_robust.summary())

print("\n--- ITS Results ---")
print(f"Pre-intervention trend: {its_model.params['month']:.2f} per month")
print(f"Immediate level change: {its_model.params['intervention']:.2f}")
print(f"Post-intervention trend change: {its_model.params['time_after']:.2f}")

# Interpretation:
# - 'month': The pre-intervention monthly change in prescriptions.
# - 'intervention': The immediate drop at the time of the FDA warning.
#   A large negative value indicates the warning had an immediate impact.
# - 'time_after': Additional monthly decline beyond the pre-existing trend.
#   Negative means the decline accelerated after the warning.
```

#### ITS Visualization

```python
fig, ax = plt.subplots(figsize=(10, 6))

# Observed data
ax.scatter(its_df['month'], its_df['prescriptions'], color='grey', alpha=0.6,
           s=30, label='Observed', zorder=3)

# Fitted values
predicted = its_model.predict(its_df)
ax.plot(its_df['month'], predicted, color='steelblue', linewidth=2, label='Fitted model')

# Counterfactual (pre-intervention trend extrapolated)
counterfactual = its_model.params['Intercept'] + its_model.params['month'] * its_df['month']
ax.plot(its_df['month'][its_df['month'] > 24],
        counterfactual[its_df['month'] > 24],
        color='red', linestyle='--', linewidth=1.5, label='Counterfactual')

ax.axvline(x=24, color='black', linestyle=':', alpha=0.7)
ax.annotate('FDA Warning', xy=(24, ax.get_ylim()[1] * 0.95),
            fontsize=10, fontstyle='italic')
ax.set_xlabel('Month')
ax.set_ylabel('Opioid Prescriptions per 100,000')
ax.set_title('Interrupted Time Series: Impact of FDA Warning on Opioid Prescribing')
ax.legend()
plt.tight_layout()
plt.show()
```

### Part B: Difference-in-Differences

```python
# DiD regression
did_model = ols('mortality ~ treated + post + treated:post', data=did_df).fit(
    cov_type='cluster', cov_kwds={'groups': did_df['hospital']}
)
print(did_model.summary())

did_coef = did_model.params['treated:post']
did_se = did_model.bse['treated:post']
did_ci = did_model.conf_int().loc['treated:post']

print(f"\n--- DiD Results ---")
print(f"ATT estimate: {did_coef:.3f} (SE: {did_se:.3f})")
print(f"95% CI: [{did_ci[0]:.3f}, {did_ci[1]:.3f}]")
print(f"p-value: {did_model.pvalues['treated:post']:.4f}")
# Expected: approximately -3.0

# Parallel trends check
trends = did_df.groupby(['period', 'treated'])['mortality'].mean().reset_index()

fig, ax = plt.subplots(figsize=(8, 5))
for grp, label, color in [(0, 'Control', 'grey'), (1, 'Treated', 'steelblue')]:
    mask = trends['treated'] == grp
    ax.plot(trends.loc[mask, 'period'], trends.loc[mask, 'mortality'],
            'o-', color=color, label=label, linewidth=2, markersize=5)

ax.axvline(x=5.5, linestyle='--', color='black', alpha=0.5)
ax.annotate('Intervention', xy=(5.5, ax.get_ylim()[1] * 0.95), fontsize=10)
ax.set_xlabel('Period')
ax.set_ylabel('30-Day Mortality (%)')
ax.set_title('Difference-in-Differences: Sepsis Bundle and Hospital Mortality')
ax.legend()
plt.tight_layout()
plt.show()
```

### Part C: Regression Discontinuity

```python
from rdrobust import rdrobust, rdplot

# Sharp RD: effect of crossing LDL=190 on statin prescription (first stage)
rd_first = rdrobust(rd_df['statin'].values, rd_df['ldl'].values, c=190)
print("\n--- First Stage RD ---")
print(rd_first.summary())

# Reduced form: effect on cardiovascular events
rd_reduced = rdrobust(rd_df['cv_events'].values, rd_df['ldl'].values, c=190)
print("\n--- Reduced Form RD ---")
print(rd_reduced.summary())

# Fuzzy RD: effect of statin on CV events using threshold as instrument
rd_fuzzy = rdrobust(rd_df['cv_events'].values, rd_df['ldl'].values, c=190,
                     fuzzy=rd_df['statin'].values)
print("\n--- Fuzzy RD (IV Estimate) ---")
print(rd_fuzzy.summary())
```

#### RD Visualization

```python
fig, axes = plt.subplots(1, 2, figsize=(14, 5))

# First stage plot
ax = axes[0]
bins = np.linspace(140, 240, 41)
bin_centers = (bins[:-1] + bins[1:]) / 2
statin_means = [rd_df.loc[(rd_df['ldl'] >= bins[i]) & (rd_df['ldl'] < bins[i+1]),
                'statin'].mean() for i in range(len(bins)-1)]
ax.scatter(bin_centers, statin_means, s=20, color='steelblue', alpha=0.7)
ax.axvline(x=190, color='red', linestyle='--', linewidth=1.5)
ax.set_xlabel('LDL Cholesterol (mg/dL)')
ax.set_ylabel('P(Statin Prescribed)')
ax.set_title('First Stage: Statin Rx at LDL=190 Cutoff')

# Outcome plot
ax = axes[1]
cv_means = [rd_df.loc[(rd_df['ldl'] >= bins[i]) & (rd_df['ldl'] < bins[i+1]),
            'cv_events'].mean() for i in range(len(bins)-1)]
ax.scatter(bin_centers, cv_means, s=20, color='darkorange', alpha=0.7)
ax.axvline(x=190, color='red', linestyle='--', linewidth=1.5)
ax.set_xlabel('LDL Cholesterol (mg/dL)')
ax.set_ylabel('CV Events (5-year)')
ax.set_title('Reduced Form: CV Events at LDL=190 Cutoff')

plt.tight_layout()
plt.show()
```

## Advanced Example

### Manual DiD with Event Study Plot

```python
# Create event time variable (time relative to treatment at period 5)
did_df['event_time'] = did_df['period'] - 5
# Exclude the period just before treatment (reference period)
event_times = sorted(did_df['event_time'].unique())
event_times.remove(0)  # reference period

# Run event study regression
formula_parts = ['mortality ~ treated']
for t in event_times:
    col_name = f'et_{t}'.replace('-', 'neg')
    did_df[col_name] = ((did_df['event_time'] == t) & (did_df['treated'] == 1)).astype(int)
    formula_parts.append(col_name)
formula_parts.append('C(period)')
formula = ' + '.join(formula_parts)

es_model = ols(formula, data=did_df).fit(
    cov_type='cluster', cov_kwds={'groups': did_df['hospital']}
)

# Extract event study coefficients
es_coefs = []
for t in event_times:
    col_name = f'et_{t}'.replace('-', 'neg')
    es_coefs.append({
        'event_time': t,
        'coef': es_model.params[col_name],
        'se': es_model.bse[col_name],
        'lower': es_model.params[col_name] - 1.96 * es_model.bse[col_name],
        'upper': es_model.params[col_name] + 1.96 * es_model.bse[col_name]
    })
# Add reference period
es_coefs.append({'event_time': 0, 'coef': 0, 'se': 0, 'lower': 0, 'upper': 0})
es_df = pd.DataFrame(es_coefs).sort_values('event_time')

fig, ax = plt.subplots(figsize=(10, 5))
ax.errorbar(es_df['event_time'], es_df['coef'],
            yerr=1.96 * es_df['se'], fmt='o-', color='steelblue',
            capsize=3, markersize=5, linewidth=1.5)
ax.axhline(y=0, linestyle='--', color='grey', alpha=0.7)
ax.axvline(x=0, linestyle=':', color='black', alpha=0.5)
ax.set_xlabel('Event Time (periods relative to treatment)')
ax.set_ylabel('Coefficient')
ax.set_title('Event Study: Dynamic Treatment Effects')
plt.tight_layout()
plt.show()

# Interpretation: Pre-treatment coefficients should be near zero (supporting
# parallel trends). Post-treatment coefficients capture the dynamic effect.
```

### Synthetic Control (Manual Implementation)

```python
from scipy.optimize import minimize

# Simulated data: one treated region + 5 control regions, 20 time periods
np.random.seed(202)
n_time = 20
treatment_time = 10

# Generate panel data
regions = {}
base_trends = np.random.normal(0, 0.5, n_time).cumsum()
for i in range(6):
    region_effect = np.random.normal(0, 2, n_time)
    regions[f'region_{i}'] = 50 + base_trends + region_effect

# Add treatment effect to region_0 after treatment_time
regions['region_0'][treatment_time:] -= 5

sc_df = pd.DataFrame(regions)
sc_df['time'] = np.arange(n_time)

# Construct synthetic control for region_0
Y_treated_pre = sc_df.loc[:treatment_time-1, 'region_0'].values
Y_donors_pre = sc_df.loc[:treatment_time-1,
                          [f'region_{i}' for i in range(1, 6)]].values

def sc_objective(w):
    synthetic = Y_donors_pre @ w
    return np.sum((Y_treated_pre - synthetic) ** 2)

# Constraints: weights sum to 1, all non-negative
n_donors = 5
constraints = {'type': 'eq', 'fun': lambda w: np.sum(w) - 1}
bounds = [(0, 1)] * n_donors
w0 = np.ones(n_donors) / n_donors

result = minimize(sc_objective, w0, bounds=bounds, constraints=constraints, method='SLSQP')
optimal_weights = result.x

print("Synthetic Control Weights:")
for i, w in enumerate(optimal_weights):
    print(f"  Region {i+1}: {w:.3f}")

# Compute synthetic control for all periods
Y_donors_all = sc_df[[f'region_{i}' for i in range(1, 6)]].values
synthetic_control = Y_donors_all @ optimal_weights
treatment_effect = sc_df['region_0'].values - synthetic_control

fig, axes = plt.subplots(1, 2, figsize=(14, 5))
ax = axes[0]
ax.plot(sc_df['time'], sc_df['region_0'], 'o-', color='steelblue',
        label='Treated Region', linewidth=2)
ax.plot(sc_df['time'], synthetic_control, 's--', color='red',
        label='Synthetic Control', linewidth=2)
ax.axvline(x=treatment_time - 0.5, linestyle=':', color='black')
ax.set_xlabel('Time Period')
ax.set_ylabel('Outcome')
ax.set_title('Synthetic Control vs Treated Unit')
ax.legend()

ax = axes[1]
ax.plot(sc_df['time'], treatment_effect, 'o-', color='darkorange', linewidth=2)
ax.axhline(y=0, linestyle='--', color='grey')
ax.axvline(x=treatment_time - 0.5, linestyle=':', color='black')
ax.set_xlabel('Time Period')
ax.set_ylabel('Treatment Effect (Gap)')
ax.set_title('Estimated Treatment Effect')

plt.tight_layout()
plt.show()

print(f"\nAverage post-intervention effect: {treatment_effect[treatment_time:].mean():.2f}")
```

## Visualization

```python
fig, axes = plt.subplots(2, 2, figsize=(14, 10))

# ITS
ax = axes[0, 0]
ax.scatter(its_df['month'], its_df['prescriptions'], color='grey', s=15, alpha=0.6)
ax.plot(its_df['month'], its_model.predict(its_df), color='steelblue', linewidth=2)
ax.axvline(x=24, linestyle=':', color='black')
ax.set_title('ITS: Opioid Prescribing')
ax.set_xlabel('Month')
ax.set_ylabel('Prescriptions per 100k')

# DiD
ax = axes[0, 1]
for grp, color in [(0, 'grey'), (1, 'steelblue')]:
    mask = trends['treated'] == grp
    ax.plot(trends.loc[mask, 'period'], trends.loc[mask, 'mortality'],
            'o-', color=color, linewidth=2)
ax.axvline(x=5.5, linestyle='--', color='black', alpha=0.5)
ax.set_title('DiD: Sepsis Mortality')
ax.set_xlabel('Period')
ax.set_ylabel('Mortality (%)')

# RD first stage
ax = axes[1, 0]
ax.scatter(bin_centers, statin_means, s=15, color='steelblue', alpha=0.7)
ax.axvline(x=190, color='red', linestyle='--')
ax.set_title('RD First Stage: Statin Rx')
ax.set_xlabel('LDL (mg/dL)')
ax.set_ylabel('P(Statin)')

# Synthetic control gap
ax = axes[1, 1]
ax.plot(sc_df['time'], treatment_effect, 'o-', color='darkorange', linewidth=2)
ax.axhline(y=0, linestyle='--', color='grey')
ax.axvline(x=treatment_time - 0.5, linestyle=':', color='black')
ax.set_title('Synthetic Control: Treatment Gap')
ax.set_xlabel('Time')
ax.set_ylabel('Effect')

plt.suptitle('Quasi-Experimental Methods: Summary', fontsize=14, fontweight='bold')
plt.tight_layout()
plt.show()
```

## Tips and Best Practices

1. **ITS seasonality**: If the outcome has seasonal patterns (e.g., flu-related hospitalizations),
   include harmonic terms (sine/cosine) or seasonal dummies in the ITS model.

2. **Always visualize raw data**: Before fitting any model, plot the time series, group trends,
   or scatter around the cutoff. Visualizations can reveal violations of assumptions that
   statistical tests may miss.

3. **Placebo tests strengthen credibility**: Run your analysis at fake intervention times (ITS),
   on outcomes that should not be affected (DiD), or at fake cutoffs (RD). Null results from
   these placebo tests strengthen the causal interpretation.

4. **RD bandwidth sensitivity is essential**: Report the main result at the optimal bandwidth
   and then show how the estimate changes at 0.5x, 0.75x, 1.25x, and 2x the bandwidth.

5. **Cluster standard errors appropriately**: In DiD, cluster at the level of treatment
   assignment. In multi-site ITS, cluster at the site level.

6. **Consider pre-trends tests in DiD**: Regress the outcome on treatment-group-by-time
   interactions in the pre-period. Joint significance of these interactions signals a
   parallel trends violation.

7. **Synthetic control diagnostics**: Report the pre-treatment RMSPE (root mean squared
   prediction error). A large RMSPE means the synthetic control is a poor match. Also
   run placebo-in-space tests (apply the method to each donor unit).

8. **Be transparent about limitations**: Every quasi-experimental method relies on assumptions
   that cannot be fully verified. Discuss potential threats to validity honestly.
