# Study Design Frameworks — Python Implementation

## Required Libraries

```bash
pip install numpy pandas scipy statsmodels lifelines matplotlib seaborn scikit-learn
```

```python
import numpy as np
import pandas as pd
from scipy import stats
from scipy.special import expit  # logistic function
import statsmodels.api as sm
import statsmodels.formula.api as smf
from lifelines import CoxPHFitter, KaplanMeierFitter
from sklearn.linear_model import LogisticRegression
from sklearn.neighbors import NearestNeighbors
import matplotlib.pyplot as plt
import seaborn as sns
```

## Example Dataset

Simulated observational cohort comparing two oral anticoagulants (Drug A vs. Drug B) for
stroke prevention in atrial fibrillation patients, structured as a new-user design.

```python
np.random.seed(2024)
n = 2000

# Baseline confounders
age          = np.round(np.random.normal(72, 10, n))
female       = np.random.binomial(1, 0.42, n)
chads_vasc   = np.clip(np.random.poisson(3, n), 0, 9)
prior_stroke = np.random.binomial(1, 0.15, n)
ckd          = np.random.binomial(1, 0.20, n)
hf           = np.random.binomial(1, 0.25, n)

# Confounded treatment assignment
lp_trt = -0.5 + 0.15 * chads_vasc + 0.3 * ckd - 0.01 * age
prob_a = expit(lp_trt)
drug_a = np.random.binomial(1, prob_a)

# Time-to-event outcome
lp_out = -5 + 0.05 * age + 0.3 * prior_stroke + 0.2 * chads_vasc + \
         0.15 * ckd + 0.2 * hf - 0.25 * drug_a
rate = np.exp(lp_out)
time_event  = np.random.exponential(1 / rate)
time_censor = np.random.uniform(0.5, 5, n)
time  = np.minimum(time_event, time_censor)
event = (time_event <= time_censor).astype(int)

cohort = pd.DataFrame({
    'id': range(1, n + 1), 'age': age, 'female': female,
    'chads_vasc': chads_vasc, 'prior_stroke': prior_stroke,
    'ckd': ckd, 'hf': hf, 'drug_a': drug_a, 'time': time, 'event': event
})

print(f"Drug A: {drug_a.sum()} | Drug B: {(1 - drug_a).sum()}")
print(f"Events: {event.sum()} ({100 * event.mean():.1f}%)")
print(cohort.describe().round(2))
```

## Complete Worked Example

### Step 1 — Target Trial Specification

```python
target_trial = {
    'Eligibility':       'Adults >= 18 with new AF diagnosis, no prior anticoagulant',
    'Treatment':         'Initiation of Drug A vs. Drug B within 30 days of AF',
    'Assignment':        'Emulated via propensity score adjustment',
    'Start of follow-up':'Date of first anticoagulant dispensing (time zero)',
    'Outcome':           'First ischaemic stroke (ICD-10 I63.x)',
    'Causal contrast':   'ITT (as-initiated) and per-protocol (IPCW)',
    'Analysis':          'Cox PH with PS matching or IPTW'
}

print("Target Trial Specification:")
print("-" * 60)
for comp, desc in target_trial.items():
    print(f"  {comp:20s}  {desc}")
```

### Step 2 — New-User Cohort Construction

```python
# In real-world data, the steps would be:
# 1. Identify first-ever anticoagulant dispensing per patient
# 2. Require >= 365 days of prior continuous enrolment (washout)
# 3. Apply eligibility criteria at time zero
# 4. Exclude patients with prior outcome

new_users = cohort[cohort['age'] >= 18].copy()
print(f"New-user cohort: n = {len(new_users)}")

# Verify no outcomes at time zero (by construction in simulation)
assert (new_users['time'] > 0).all(), "Outcomes at time zero detected!"
```

### Step 3 — Propensity Score Estimation

```python
covariates = ['age', 'female', 'chads_vasc', 'prior_stroke', 'ckd', 'hf']

# Logistic regression for propensity score
ps_model = smf.logit(
    'drug_a ~ age + female + chads_vasc + prior_stroke + ckd + hf',
    data=new_users
).fit(disp=0)
new_users['ps'] = ps_model.predict()

print(ps_model.summary())

# Check overlap
fig, ax = plt.subplots(figsize=(8, 5))
for trt, label, color in [(1, 'Drug A', 'firebrick'), (0, 'Drug B', 'steelblue')]:
    subset = new_users.loc[new_users['drug_a'] == trt, 'ps']
    ax.hist(subset, bins=50, alpha=0.5, label=label, color=color, density=True)
ax.set_xlabel('Propensity Score')
ax.set_ylabel('Density')
ax.set_title('Propensity Score Overlap')
ax.legend()
plt.tight_layout()
plt.savefig('ps_overlap.png', dpi=150)
plt.show()

# Trim non-overlap
trimmed = new_users[(new_users['ps'] > 0.05) & (new_users['ps'] < 0.95)].copy()
print(f"After trimming: n = {len(trimmed)}")
```

### Step 4 — Propensity Score Matching (1:1 Nearest Neighbour)

```python
def ps_match(data, treatment_col='drug_a', ps_col='ps', caliper=0.05):
    """1:1 nearest-neighbour propensity score matching with caliper."""
    treated = data[data[treatment_col] == 1].copy()
    control = data[data[treatment_col] == 0].copy()

    nn = NearestNeighbors(n_neighbors=1, metric='euclidean')
    nn.fit(control[[ps_col]].values)

    distances, indices = nn.kneighbors(treated[[ps_col]].values)

    # Apply caliper
    within_caliper = distances.flatten() <= caliper
    matched_treated = treated.iloc[within_caliper].copy()
    matched_control = control.iloc[indices.flatten()[within_caliper]].copy()

    matched_treated['match_id'] = range(len(matched_treated))
    matched_control['match_id'] = range(len(matched_control))

    return pd.concat([matched_treated, matched_control], ignore_index=True)

matched = ps_match(trimmed, caliper=0.05)
print(f"Matched cohort: n = {len(matched)}")
```

### Step 5 — Covariate Balance Assessment

```python
def compute_smd(data, treatment_col, covariates):
    """Compute standardised mean differences."""
    treated = data[data[treatment_col] == 1]
    control = data[data[treatment_col] == 0]
    smds = {}
    for cov in covariates:
        mean_t = treated[cov].mean()
        mean_c = control[cov].mean()
        sd_pooled = np.sqrt((treated[cov].var() + control[cov].var()) / 2)
        smds[cov] = (mean_t - mean_c) / sd_pooled if sd_pooled > 0 else 0
    return smds

smd_before = compute_smd(trimmed, 'drug_a', covariates)
smd_after  = compute_smd(matched, 'drug_a', covariates)

balance_df = pd.DataFrame({
    'Covariate': covariates,
    'SMD_Before': [smd_before[c] for c in covariates],
    'SMD_After':  [smd_after[c] for c in covariates]
})
print("Covariate Balance (SMD):")
print(balance_df.to_string(index=False))

# Love plot
fig, ax = plt.subplots(figsize=(7, 5))
y_pos = range(len(covariates))
ax.scatter(balance_df['SMD_Before'].abs(), y_pos, marker='x', s=80,
           color='red', label='Before matching', zorder=3)
ax.scatter(balance_df['SMD_After'].abs(), y_pos, marker='o', s=80,
           color='blue', label='After matching', zorder=3)
ax.axvline(x=0.1, color='grey', linestyle='--', alpha=0.7)
ax.set_yticks(y_pos)
ax.set_yticklabels(covariates)
ax.set_xlabel('Absolute SMD')
ax.set_title('Covariate Balance: Love Plot')
ax.legend()
plt.tight_layout()
plt.savefig('love_plot.png', dpi=150)
plt.show()
```

### Step 6 — Outcome Analysis

```python
# Cox model on matched cohort
cox_data = matched[['drug_a', 'time', 'event'] + covariates].copy()
cph = CoxPHFitter()
cph.fit(cox_data, duration_col='time', event_col='event',
        formula='drug_a + age + chads_vasc + ckd')
cph.print_summary()

hr = np.exp(cph.params_['drug_a'])
ci = np.exp(cph.confidence_intervals_.loc['drug_a'].values)
print(f"\nHR (Drug A vs B): {hr:.3f} (95% CI: {ci[0]:.3f} - {ci[1]:.3f})")

# Kaplan-Meier by treatment
fig, ax = plt.subplots(figsize=(8, 6))
kmf = KaplanMeierFitter()
for trt, label, color in [(0, 'Drug B', 'steelblue'), (1, 'Drug A', 'firebrick')]:
    mask = matched['drug_a'] == trt
    kmf.fit(matched.loc[mask, 'time'], matched.loc[mask, 'event'], label=label)
    kmf.plot_survival_function(ax=ax, color=color, linewidth=2)

ax.set_xlabel('Years')
ax.set_ylabel('Stroke-Free Survival')
ax.set_title('Kaplan-Meier: Matched Cohort')
plt.tight_layout()
plt.savefig('km_matched.png', dpi=150)
plt.show()
```

## Advanced Example

### IPTW Analysis

```python
# Compute stabilised IPTW
p_trt = new_users['drug_a'].mean()
new_users['iptw'] = np.where(
    new_users['drug_a'] == 1,
    p_trt / new_users['ps'],
    (1 - p_trt) / (1 - new_users['ps'])
)

print("Stabilised IPTW summary:")
print(new_users['iptw'].describe().round(3))

# Truncate extreme weights at 99th percentile
cap = new_users['iptw'].quantile(0.99)
new_users['iptw_trunc'] = new_users['iptw'].clip(upper=cap)

# Weighted Cox model (lifelines supports weights)
cox_iptw_data = new_users[['drug_a', 'time', 'event', 'iptw_trunc']].copy()
cph_w = CoxPHFitter()
cph_w.fit(cox_iptw_data, duration_col='time', event_col='event',
          weights_col='iptw_trunc', formula='drug_a')
cph_w.print_summary()

hr_w = np.exp(cph_w.params_['drug_a'])
print(f"IPTW HR: {hr_w:.3f}")
```

### Demonstrating Immortal Time Bias

```python
# WRONG: add pre-treatment time to follow-up
new_users['time_to_trt'] = np.random.uniform(0, 0.5, len(new_users))
new_users['time_wrong'] = new_users['time'] + new_users['time_to_trt']

# Biased analysis (misaligned time zero)
cox_wrong = CoxPHFitter()
wrong_data = new_users[['drug_a', 'time_wrong', 'event']].copy()
cox_wrong.fit(wrong_data, duration_col='time_wrong', event_col='event',
              formula='drug_a')
hr_wrong = np.exp(cox_wrong.params_['drug_a'])

# Correct analysis (aligned time zero)
cox_right = CoxPHFitter()
right_data = new_users[['drug_a', 'time', 'event']].copy()
cox_right.fit(right_data, duration_col='time', event_col='event',
              formula='drug_a')
hr_right = np.exp(cox_right.params_['drug_a'])

print(f"HR with immortal time bias (WRONG):  {hr_wrong:.3f}")
print(f"HR with aligned time zero (CORRECT): {hr_right:.3f}")
print("Note: The biased HR is artificially more protective.")
```

### Per-Protocol Analysis with IPCW

```python
# Simulate treatment discontinuation
new_users['disc_time'] = np.random.exponential(1 / 0.3, len(new_users))
new_users['adhered']   = (new_users['disc_time'] >= new_users['time']).astype(int)

# Per-protocol censoring
new_users['time_pp']  = np.minimum(new_users['time'], new_users['disc_time'])
new_users['event_pp'] = np.where(
    new_users['disc_time'] < new_users['time'], 0, new_users['event']
)

# IPCW model
ipcw_model = smf.logit(
    'adhered ~ drug_a + age + chads_vasc + ckd',
    data=new_users
).fit(disp=0)
p_adhere = ipcw_model.predict()
new_users['ipcw'] = 1 / p_adhere
new_users['ipcw'] = new_users['ipcw'].clip(upper=new_users['ipcw'].quantile(0.99))

# Per-protocol Cox
pp_data = new_users[['drug_a', 'time_pp', 'event_pp', 'ipcw'] + covariates].copy()
cph_pp = CoxPHFitter()
cph_pp.fit(pp_data, duration_col='time_pp', event_col='event_pp',
           weights_col='ipcw', formula='drug_a + age + chads_vasc + ckd')
print("Per-Protocol Analysis (IPCW):")
cph_pp.print_summary()
```

## Visualization

```python
# 1. Study design timeline figure
fig, ax = plt.subplots(figsize=(10, 4))
ax.barh(['Lookback\n(confounders)'], [3], left=[0], height=0.5,
        color='lightyellow', edgecolor='orange')
ax.barh(['Follow-up\n(outcomes)'], [7], left=[3], height=0.5,
        color='lightblue', edgecolor='steelblue')
ax.barh(['On treatment'], [4], left=[3], height=0.5,
        color='lightgreen', edgecolor='darkgreen')
ax.axvline(x=3, color='red', linewidth=2, linestyle='--', label='Time Zero')
ax.set_xlabel('Study Timeline')
ax.set_title('Target Trial Emulation: Timeline')
ax.legend(loc='upper right')
plt.tight_layout()
plt.savefig('timeline.png', dpi=150)
plt.show()

# 2. Comparison of analyses
results = pd.DataFrame({
    'Analysis': ['Unadjusted', 'PS-Matched', 'IPTW', 'Immortal Time (WRONG)'],
    'HR': [np.exp(CoxPHFitter().fit(
               new_users[['drug_a', 'time', 'event']],
               duration_col='time', event_col='event',
               formula='drug_a').params_['drug_a']),
           hr, hr_w, hr_wrong]
})

fig, ax = plt.subplots(figsize=(7, 4))
colors = ['grey', 'steelblue', 'teal', 'red']
ax.barh(results['Analysis'], results['HR'], color=colors, edgecolor='black')
ax.axvline(x=1.0, color='black', linestyle='--', linewidth=1)
ax.set_xlabel('Hazard Ratio (Drug A vs Drug B)')
ax.set_title('Impact of Analysis Method on HR Estimate')
plt.tight_layout()
plt.savefig('hr_comparison.png', dpi=150)
plt.show()
```

## Tips and Best Practices

1. **Specify the target trial before analysing data.** Document all seven components.
   This is the single most important step to avoid bias in observational studies.

2. **Enforce the new-user design.** Exclude prevalent users. Require a washout period
   (e.g., 365 days without the drug) to ensure first-time use.

3. **Align time zero with treatment initiation.** Never count pre-treatment person-time
   as exposed. This is the most common source of immortal time bias.

4. **Check and report propensity score overlap.** If positivity is violated (some
   patients have near-zero probability of one treatment), trim or restrict the cohort.

5. **Target SMD < 0.1 for all covariates** after matching or weighting. Use the love
   plot for transparent reporting.

6. **Truncate extreme IPTW weights** (e.g., at the 99th percentile or at a fixed
   value like 10). Extreme weights inflate variance and can bias results.

7. **For per-protocol analyses,** use IPCW with time-varying covariates to account for
   informative censoring due to treatment discontinuation.

8. **Report the E-value** to quantify the minimum strength of unmeasured confounding
   that could explain the observed result. This helps readers assess robustness.
