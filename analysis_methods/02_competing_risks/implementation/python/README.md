# Competing Risks — Python Implementation

## Required Libraries

```bash
pip install lifelines scikit-survival matplotlib pandas numpy seaborn
```

```python
import numpy as np
import pandas as pd
import matplotlib.pyplot as plt
import seaborn as sns
from lifelines import KaplanMeierFitter, CoxPHFitter
```

## Example Dataset

We simulate a bone marrow transplant dataset where patients can experience relapse (event 1), treatment-related mortality/TRM (event 2), or remain censored (event 0). This is a common competing risks scenario in transplant medicine.

```python
np.random.seed(42)
n = 500

bmt = pd.DataFrame({
    'id': range(1, n + 1),
    'time': np.round(np.random.exponential(scale=333, size=n), 1),
    'age': np.round(np.random.normal(45, 12, size=n)).astype(int),
    'donor_type': np.random.choice(['Matched', 'Mismatched'], size=n, p=[0.6, 0.4]),
    'disease_stage': np.random.choice(['Early', 'Intermediate', 'Advanced'], size=n, p=[0.3, 0.4, 0.3]),
    'gvhd': np.random.choice([0, 1], size=n, p=[0.6, 0.4])
})

# Assign event types based on covariates
lp_relapse = (-2.5 + 0.01 * bmt['age']
              + 0.3 * (bmt['disease_stage'] == 'Advanced').astype(int)
              + 0.2 * (bmt['disease_stage'] == 'Intermediate').astype(int)
              - 0.2 * bmt['gvhd'])
lp_trm = (-3.0 + 0.02 * bmt['age']
           + 0.4 * (bmt['donor_type'] == 'Mismatched').astype(int)
           + 0.3 * bmt['gvhd'])

from scipy.special import expit
p_relapse = expit(lp_relapse)
p_trm = expit(lp_trm) * 0.6
p_censor = 1 - p_relapse - p_trm
p_censor = np.maximum(p_censor, 0.05)

probs = np.column_stack([p_censor, p_relapse, p_trm])
probs = probs / probs.sum(axis=1, keepdims=True)

bmt['event_type'] = [np.random.choice([0, 1, 2], p=p) for p in probs]
bmt['time'] = np.minimum(bmt['time'], 1000)
bmt.loc[bmt['time'] == 0, 'time'] = 0.5

event_labels = {0: 'Censored', 1: 'Relapse', 2: 'TRM'}
bmt['event_label'] = bmt['event_type'].map(event_labels)

print("Event distribution:")
print(bmt['event_label'].value_counts())
print(f"\nMedian follow-up: {bmt['time'].median():.1f} days")
```

## Complete Worked Example

### Step 1: Non-Parametric Cumulative Incidence Function (CIF)

The Aalen-Johansen estimator provides non-parametric CIF estimates. We implement this manually since lifelines focuses on single-event analysis.

```python
def aalen_johansen_cif(time, event, event_of_interest):
    """
    Compute the Aalen-Johansen estimate of the cumulative incidence function.

    Parameters
    ----------
    time : array-like, event/censoring times
    event : array-like, event type (0 = censored)
    event_of_interest : int, the event type to compute CIF for

    Returns
    -------
    times : array, unique event times
    cif : array, cumulative incidence estimates
    """
    df = pd.DataFrame({'time': time, 'event': event}).sort_values('time')
    unique_times = np.sort(df.loc[df['event'] > 0, 'time'].unique())

    n_total = len(df)
    cif = np.zeros(len(unique_times))
    surv = 1.0  # Overall survival (Kaplan-Meier type)

    for i, t in enumerate(unique_times):
        at_risk = (df['time'] >= t).sum()
        if at_risk == 0:
            cif[i] = cif[i - 1] if i > 0 else 0
            continue

        d_interest = ((df['time'] == t) & (df['event'] == event_of_interest)).sum()
        d_total = ((df['time'] == t) & (df['event'] > 0)).sum()

        # CIF increment: cause-specific hazard * overall survival
        cif[i] = (cif[i - 1] if i > 0 else 0) + surv * (d_interest / at_risk)

        # Update overall survival
        surv *= (1 - d_total / at_risk)

    return unique_times, cif


# Compute CIF for relapse (event 1) and TRM (event 2)
times_r, cif_r = aalen_johansen_cif(bmt['time'], bmt['event_type'], event_of_interest=1)
times_t, cif_t = aalen_johansen_cif(bmt['time'], bmt['event_type'], event_of_interest=2)

# CIF at key time points
for target_t in [100, 200, 365, 730]:
    idx_r = np.searchsorted(times_r, target_t, side='right') - 1
    idx_t = np.searchsorted(times_t, target_t, side='right') - 1
    cif_r_val = cif_r[max(idx_r, 0)] if idx_r >= 0 else 0
    cif_t_val = cif_t[max(idx_t, 0)] if idx_t >= 0 else 0
    print(f"  t = {target_t} days: Relapse CIF = {cif_r_val:.3f}, TRM CIF = {cif_t_val:.3f}")
```

**Interpretation**: The Aalen-Johansen CIF estimates the probability of each event type accounting for the competing event. At each time point, the CIFs for all event types plus the event-free probability sum to 1.

### Step 2: Plot Cumulative Incidence Functions

```python
fig, ax = plt.subplots(figsize=(10, 7))

ax.step(times_r, cif_r, where='post', color='firebrick', linewidth=2, label='Relapse')
ax.step(times_t, cif_t, where='post', color='steelblue', linewidth=2, label='TRM')
ax.set_xlabel('Time (days)', fontsize=12)
ax.set_ylabel('Cumulative Incidence', fontsize=12)
ax.set_title('Cumulative Incidence Functions (Aalen-Johansen)', fontsize=14)
ax.legend(fontsize=11)
ax.grid(True, alpha=0.3)
ax.set_xlim(0, 1000)
ax.set_ylim(0, 0.7)
plt.tight_layout()
plt.savefig('cif_plot.png', dpi=150)
plt.show()
```

### Step 3: CIF by Group

```python
fig, axes = plt.subplots(1, 2, figsize=(14, 6))

for ax_idx, (event_code, event_name) in enumerate([(1, 'Relapse'), (2, 'TRM')]):
    ax = axes[ax_idx]
    for donor in ['Matched', 'Mismatched']:
        mask = bmt['donor_type'] == donor
        t, c = aalen_johansen_cif(bmt.loc[mask, 'time'],
                                   bmt.loc[mask, 'event_type'],
                                   event_of_interest=event_code)
        ax.step(t, c, where='post', linewidth=2, label=donor)

    ax.set_xlabel('Time (days)', fontsize=11)
    ax.set_ylabel('Cumulative Incidence', fontsize=11)
    ax.set_title(f'CIF of {event_name} by Donor Type', fontsize=13)
    ax.legend(fontsize=10)
    ax.grid(True, alpha=0.3)
    ax.set_xlim(0, 1000)

plt.tight_layout()
plt.savefig('cif_by_donor.png', dpi=150)
plt.show()
```

**Interpretation**: Comparing CIF curves by group visually demonstrates whether the competing event incidence differs. For a formal test, we implement Gray's test below.

### Step 4: Gray's Test for CIF Equality

```python
def grays_test(time, event, group, event_of_interest):
    """
    Simplified Gray's test comparing CIF between two groups.
    Uses a chi-squared approximation based on weighted differences.
    For production use, prefer the R cmprsk package or specialized software.
    """
    groups = sorted(pd.Series(group).unique())
    assert len(groups) == 2, "Gray's test implementation supports exactly 2 groups"

    mask1 = group == groups[0]
    mask2 = group == groups[1]

    t1, cif1 = aalen_johansen_cif(time[mask1], event[mask1], event_of_interest)
    t2, cif2 = aalen_johansen_cif(time[mask2], event[mask2], event_of_interest)

    # Evaluate at common time points
    all_times = np.sort(np.unique(np.concatenate([t1, t2])))
    cif1_interp = np.interp(all_times, t1, cif1, left=0)
    cif2_interp = np.interp(all_times, t2, cif2, left=0)

    n1 = mask1.sum()
    n2 = mask2.sum()
    n = n1 + n2

    weighted_diff = np.sum(cif1_interp - cif2_interp) / len(all_times)
    se = np.sqrt(1 / n1 + 1 / n2) * 0.5  # Approximate SE
    test_stat = (weighted_diff / se) ** 2

    from scipy.stats import chi2
    p_value = 1 - chi2.cdf(test_stat, df=1)

    return test_stat, p_value


stat, pval = grays_test(
    bmt['time'].values, bmt['event_type'].values,
    bmt['donor_type'].values, event_of_interest=1
)
print(f"Gray's test for Relapse CIF by Donor Type:")
print(f"  Test statistic: {stat:.4f}")
print(f"  P-value: {pval:.4f}")
```

**Interpretation**: Gray's test evaluates whether the CIF differs between groups. A small p-value indicates a statistically significant difference in the cumulative incidence of the event of interest across groups, accounting for competing risks.

### Step 5: Cause-Specific Cox Regression

```python
from lifelines import CoxPHFitter

# Prepare data for cause-specific models
bmt_model = bmt.copy()
bmt_model['donor_mismatched'] = (bmt_model['donor_type'] == 'Mismatched').astype(int)
bmt_model['stage_intermediate'] = (bmt_model['disease_stage'] == 'Intermediate').astype(int)
bmt_model['stage_advanced'] = (bmt_model['disease_stage'] == 'Advanced').astype(int)

covariates = ['age', 'donor_mismatched', 'stage_intermediate', 'stage_advanced', 'gvhd']

# Cause-specific model for RELAPSE (censor TRM events)
relapse_data = bmt_model[['time'] + covariates].copy()
relapse_data['event'] = (bmt_model['event_type'] == 1).astype(int)

cph_relapse = CoxPHFitter()
cph_relapse.fit(relapse_data, duration_col='time', event_col='event')
print("=== Cause-Specific Cox Model: Relapse ===")
cph_relapse.print_summary()

# Cause-specific model for TRM (censor relapse events)
trm_data = bmt_model[['time'] + covariates].copy()
trm_data['event'] = (bmt_model['event_type'] == 2).astype(int)

cph_trm = CoxPHFitter()
cph_trm.fit(trm_data, duration_col='time', event_col='event')
print("\n=== Cause-Specific Cox Model: TRM ===")
cph_trm.print_summary()
```

**Interpretation**: In cause-specific models, subjects experiencing the competing event are treated as censored. The hazard ratios represent the effect of each covariate on the rate of the specific event among those still at risk. Compare across event types to understand differential covariate effects.

### Step 6: Comparing Cause-Specific Hazard Ratios

```python
# Side-by-side comparison
comparison = pd.DataFrame({
    'Covariate': covariates,
    'Relapse_HR': np.exp(cph_relapse.params_[covariates]).values,
    'Relapse_p': cph_relapse.summary.loc[covariates, 'p'].values,
    'TRM_HR': np.exp(cph_trm.params_[covariates]).values,
    'TRM_p': cph_trm.summary.loc[covariates, 'p'].values
})
comparison['Relapse_HR'] = comparison['Relapse_HR'].round(3)
comparison['TRM_HR'] = comparison['TRM_HR'].round(3)
comparison['Relapse_p'] = comparison['Relapse_p'].round(4)
comparison['TRM_p'] = comparison['TRM_p'].round(4)

print("Cause-Specific Hazard Ratios Comparison:")
print(comparison.to_string(index=False))
```

**Interpretation**: This comparison reveals which covariates affect relapse vs. TRM differently. For example, mismatched donors may increase TRM but have less effect on relapse, while advanced disease stage may primarily increase relapse risk.

## Advanced Example

### Predicted Cumulative Incidence from Cause-Specific Models

```python
def predicted_cif_from_cs(cph_models, new_data, time_grid, event_codes):
    """
    Compute predicted CIF from cause-specific Cox models.

    The CIF for event k is obtained by integrating the cause-specific hazard
    for event k weighted by the overall survival from all causes.
    """
    # Get baseline cumulative hazards for each cause
    surv_funcs = {}
    for code, model in zip(event_codes, cph_models):
        sf = model.predict_survival_function(new_data)
        surv_funcs[code] = sf.values.flatten()

    # Overall survival = product of cause-specific survivals
    overall_surv = np.ones(len(time_grid))
    for code in event_codes:
        cs_surv_interp = np.interp(time_grid, sf.index.values, surv_funcs[code])
        overall_surv *= cs_surv_interp

    # Approximate CIF using discrete increments
    cifs = {}
    for code, model in zip(event_codes, cph_models):
        cs_surv = np.interp(time_grid, sf.index.values, surv_funcs[code])
        cs_haz = -np.diff(np.log(np.maximum(cs_surv, 1e-10)), prepend=0)
        cif = np.cumsum(cs_haz * np.concatenate([[1], overall_surv[:-1]]))
        cifs[code] = cif

    return cifs


# Predict for a specific patient profile
new_patient = pd.DataFrame({
    'age': [55],
    'donor_mismatched': [1],
    'stage_intermediate': [0],
    'stage_advanced': [1],
    'gvhd': [0]
})

time_grid = np.linspace(0.5, 1000, 500)
cifs_pred = predicted_cif_from_cs(
    [cph_relapse, cph_trm], new_patient, time_grid, [1, 2]
)

fig, ax = plt.subplots(figsize=(10, 6))
ax.plot(time_grid, cifs_pred[1], color='firebrick', linewidth=2, label='Relapse')
ax.plot(time_grid, cifs_pred[2], color='steelblue', linewidth=2, label='TRM')
ax.set_xlabel('Time (days)', fontsize=12)
ax.set_ylabel('Predicted Cumulative Incidence', fontsize=12)
ax.set_title('Predicted CIF: 55yo, Mismatched Donor, Advanced Stage, No GVHD', fontsize=13)
ax.legend(fontsize=11)
ax.grid(True, alpha=0.3)
plt.tight_layout()
plt.savefig('predicted_cif.png', dpi=150)
plt.show()
```

**Interpretation**: Individual-level predicted CIFs from cause-specific models allow personalized risk communication. The stacked CIF can show how total event probability is divided between competing outcomes for a given patient profile.

### Using scikit-survival for Competing Risks

```python
from sksurv.functions import StepFunction
from sksurv.nonparametric import kaplan_meier_estimator

# scikit-survival approach for cause-specific analysis
from sksurv.linear_model import CoxPHSurvivalAnalysis

# Prepare structured arrays
for event_name, event_code in [('Relapse', 1), ('TRM', 2)]:
    cs_event = (bmt_model['event_type'] == event_code).astype(bool)
    y = np.array(
        [(e, t) for e, t in zip(cs_event, bmt_model['time'])],
        dtype=[('event', bool), ('time', float)]
    )
    X = bmt_model[covariates].values

    model = CoxPHSurvivalAnalysis(alpha=0.01)
    model.fit(X, y)

    from sksurv.metrics import concordance_index_censored
    pred = model.predict(X)
    c_idx = concordance_index_censored(cs_event.values, bmt_model['time'].values, pred)

    print(f"\n{event_name} (Cause-Specific Cox via scikit-survival):")
    print(f"  Coefficients: {dict(zip(covariates, np.round(model.coef_, 4)))}")
    print(f"  C-index: {c_idx[0]:.4f}")
```

**Interpretation**: scikit-survival provides an alternative implementation that is well-suited for integration with machine learning workflows and cross-validation pipelines.

## Visualization

### Stacked CIF Plot

```python
fig, ax = plt.subplots(figsize=(10, 7))

ax.fill_between(times_r, 0, cif_r, step='post', alpha=0.5, color='firebrick', label='Relapse')

# Stack TRM on top of relapse
cif_t_interp = np.interp(times_r, times_t, cif_t, left=0)
ax.fill_between(times_r, cif_r, cif_r + cif_t_interp, step='post',
                alpha=0.5, color='steelblue', label='TRM')

ax.set_xlabel('Time (days)', fontsize=12)
ax.set_ylabel('Cumulative Incidence', fontsize=12)
ax.set_title('Stacked Cumulative Incidence Functions', fontsize=14)
ax.legend(fontsize=11, loc='upper left')
ax.grid(True, alpha=0.3)
ax.set_xlim(0, 1000)
ax.set_ylim(0, 1)
plt.tight_layout()
plt.savefig('stacked_cif.png', dpi=150)
plt.show()
```

### Forest Plot: Cause-Specific HRs for Both Events

```python
fig, ax = plt.subplots(figsize=(10, 6))

hr_relapse = np.exp(cph_relapse.params_[covariates]).values
hr_trm = np.exp(cph_trm.params_[covariates]).values

ci_relapse_lo = np.exp(cph_relapse.summary.loc[covariates, 'coef lower 95%']).values
ci_relapse_hi = np.exp(cph_relapse.summary.loc[covariates, 'coef upper 95%']).values
ci_trm_lo = np.exp(cph_trm.summary.loc[covariates, 'coef lower 95%']).values
ci_trm_hi = np.exp(cph_trm.summary.loc[covariates, 'coef upper 95%']).values

y = np.arange(len(covariates))
offset = 0.15

ax.errorbar(hr_relapse, y - offset, xerr=[hr_relapse - ci_relapse_lo, ci_relapse_hi - hr_relapse],
            fmt='o', color='firebrick', capsize=4, markersize=7, label='Relapse')
ax.errorbar(hr_trm, y + offset, xerr=[hr_trm - ci_trm_lo, ci_trm_hi - hr_trm],
            fmt='s', color='steelblue', capsize=4, markersize=7, label='TRM')

ax.axvline(x=1, color='gray', linestyle='--', linewidth=1)
ax.set_yticks(y)
ax.set_yticklabels(covariates)
ax.set_xlabel('Cause-Specific Hazard Ratio (95% CI)', fontsize=12)
ax.set_title('Cause-Specific Hazard Ratios: Relapse vs. TRM', fontsize=14)
ax.legend(fontsize=11)
ax.grid(True, alpha=0.3, axis='x')
plt.tight_layout()
plt.savefig('forest_competing.png', dpi=150)
plt.show()
```

## Tips and Best Practices

1. **Avoid 1-KM for event-specific incidence**: When competing events are present, use the Aalen-Johansen CIF estimator. The KM complement overestimates the true probability of the event of interest.

2. **Report results for all event types**: Even if one event type is the primary interest, showing cause-specific results for all events provides a complete picture and helps reviewers assess the analysis.

3. **Python ecosystem limitations**: Native competing risks support in Python is less mature than in R. For formal Fine-Gray modeling, consider calling R from Python using `rpy2`:

```python
import rpy2.robjects as ro
from rpy2.robjects.packages import importr
cmprsk = importr('cmprsk')
```

4. **Validate CIF estimates**: The sum of all CIFs plus the event-free probability should equal 1 at every time point. Check this as a quality control step.

5. **Sample size considerations**: Each event type needs sufficient events for stable estimation. With multiple competing events, the effective sample size for each event type is smaller than the total sample.

6. **Time-varying effects**: If the cause-specific hazard ratio changes over time, consider using time-varying coefficients or restricted cubic splines in the Cox model.

7. **Sensitivity analysis**: Perform sensitivity analyses to assess the impact of treating competing events differently (e.g., as censored vs. as a separate event type).

8. **Clinical interpretation**: Present CIF curves alongside hazard ratio results. CIF provides the absolute risk that patients and clinicians understand, while hazard ratios provide the relative effect size.

9. **Handling rare competing events**: When one event type is very rare, the CIF and 1-KM will be very similar. Competing risks methods are most important when competing events are common.

10. **Consider scikit-survival for ML integration**: When building prediction models with cross-validation, scikit-survival's cause-specific Cox implementation integrates well with scikit-learn pipelines.
