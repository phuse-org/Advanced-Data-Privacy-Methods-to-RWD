# Healthcare Cost and Resource Analysis — Python Implementation

## Required Libraries

```bash
pip install numpy pandas scipy statsmodels matplotlib seaborn
```

```python
import numpy as np
import pandas as pd
import statsmodels.api as sm
import statsmodels.formula.api as smf
from scipy import stats
import matplotlib.pyplot as plt
import seaborn as sns
```

## Example Dataset

Simulated data from a randomised trial comparing a new diabetes medication to standard
of care. Total 12-month medical expenditure is the cost endpoint; QALYs measure
effectiveness.

```python
np.random.seed(123)
n = 800

age       = np.round(np.random.normal(60, 12, n))
female    = np.random.binomial(1, 0.52, n)
bmi       = np.random.normal(30, 5, n)
treatment = np.repeat([0, 1], n // 2)

# Zero-cost probability
prob_zero = 1 / (1 + np.exp(2 - 0.01 * age + 0.3 * treatment - 0.02 * bmi))
any_cost  = np.random.binomial(1, 1 - prob_zero)

# Positive costs from gamma
shape = 2.0
scale = np.exp(7.5 + 0.01 * age + 0.3 * treatment + 0.005 * bmi) / shape
pos_cost = np.random.gamma(shape, scale, n)
total_cost = np.where(any_cost == 1, pos_cost, 0.0)

# QALYs
qaly = np.random.normal(0.75 + 0.05 * treatment - 0.002 * age, 0.1)

df = pd.DataFrame({
    'treatment': treatment, 'age': age, 'female': female, 'bmi': bmi,
    'total_cost': total_cost, 'qaly': qaly, 'any_cost': (total_cost > 0).astype(int)
})

print(df[['total_cost', 'qaly']].describe())
print(f"Proportion zero cost: {(total_cost == 0).mean():.3f}")
```

## Complete Worked Example

### Step 1 — Exploratory Analysis

```python
fig, axes = plt.subplots(1, 2, figsize=(12, 5))

for trt, label, color in [(0, 'Control', 'steelblue'), (1, 'Treatment', 'firebrick')]:
    subset = df[df['treatment'] == trt]['total_cost']
    axes[0].hist(subset, bins=50, alpha=0.6, label=label, color=color)
    axes[1].hist(subset[subset > 0], bins=50, alpha=0.6, label=label, color=color)

axes[0].set_title('All Costs (Including Zeros)')
axes[0].set_xlabel('Total Cost ($)')
axes[0].legend()
axes[1].set_title('Positive Costs Only')
axes[1].set_xlabel('Total Cost ($)')
axes[1].legend()
plt.tight_layout()
plt.savefig('cost_distribution.png', dpi=150)
plt.show()

# Summary statistics by group
print(df.groupby('treatment').agg(
    n=('total_cost', 'count'),
    mean_cost=('total_cost', 'mean'),
    median_cost=('total_cost', 'median'),
    sd_cost=('total_cost', 'std'),
    pct_zero=('total_cost', lambda x: (x == 0).mean())
).round(2))
```

### Step 2 — GLM with Gamma Family and Log Link

```python
# Fit on positive costs
pos_df = df[df['total_cost'] > 0].copy()

glm_gamma = smf.glm(
    'total_cost ~ treatment + age + bmi + female',
    data=pos_df,
    family=sm.families.Gamma(link=sm.families.links.Log())
).fit()

print(glm_gamma.summary())

# Treatment effect on cost ratio scale
cost_ratio = np.exp(glm_gamma.params['treatment'])
print(f"\nCost ratio (treatment vs control): {cost_ratio:.4f}")
print("Interpretation: treatment multiplies expected positive costs by "
      f"{cost_ratio:.2f}")

# Modified Park test
glm_gauss = smf.glm(
    'total_cost ~ treatment + age + bmi + female',
    data=pos_df,
    family=sm.families.Gaussian(link=sm.families.links.Log())
).fit()
resid_sq = np.log(glm_gauss.resid_response ** 2)
pred_log = np.log(glm_gauss.fittedvalues)
park = sm.OLS(resid_sq, sm.add_constant(pred_log)).fit()
print(f"\nPark test slope: {park.params[1]:.3f}")
print("(slope ~2 supports gamma family)")
```

### Step 3 — Two-Part (Hurdle) Model

```python
# Part 1: logistic regression for P(cost > 0)
part1 = smf.logit('any_cost ~ treatment + age + bmi + female', data=df).fit()
print("=== Part 1: Logistic (any cost) ===")
print(part1.summary())

# Part 2: gamma GLM on positive costs
part2 = smf.glm(
    'total_cost ~ treatment + age + bmi + female',
    data=pos_df,
    family=sm.families.Gamma(link=sm.families.links.Log())
).fit()
print("\n=== Part 2: Gamma GLM (positive costs) ===")
print(part2.summary())

def predict_two_part(new_data, part1, part2):
    """Predict unconditional mean cost from two-part model."""
    p_pos = part1.predict(new_data)
    cond_mean = part2.predict(new_data)
    return p_pos * cond_mean

# Predict for a typical patient
nd_trt  = pd.DataFrame({'treatment': [1], 'age': [60], 'bmi': [30], 'female': [0]})
nd_ctrl = pd.DataFrame({'treatment': [0], 'age': [60], 'bmi': [30], 'female': [0]})

cost_trt  = predict_two_part(nd_trt, part1, part2).values[0]
cost_ctrl = predict_two_part(nd_ctrl, part1, part2).values[0]
print(f"\nPredicted cost (treatment): ${cost_trt:,.2f}")
print(f"Predicted cost (control):   ${cost_ctrl:,.2f}")
print(f"Incremental cost:           ${cost_trt - cost_ctrl:,.2f}")
```

### Step 4 — Bootstrap CI for Mean Cost Difference

```python
def bootstrap_cost_diff(df, n_boot=2000, seed=456):
    """Bootstrap the mean cost difference between treatment groups."""
    rng = np.random.default_rng(seed)
    diffs = np.empty(n_boot)
    trt = df[df['treatment'] == 1]['total_cost'].values
    ctrl = df[df['treatment'] == 0]['total_cost'].values
    for i in range(n_boot):
        trt_boot  = rng.choice(trt, size=len(trt), replace=True)
        ctrl_boot = rng.choice(ctrl, size=len(ctrl), replace=True)
        diffs[i] = trt_boot.mean() - ctrl_boot.mean()
    return diffs

boot_diffs = bootstrap_cost_diff(df)
ci = np.percentile(boot_diffs, [2.5, 97.5])
print(f"Bootstrap mean cost difference: ${np.mean(boot_diffs):,.2f}")
print(f"95% Percentile CI: [${ci[0]:,.2f}, ${ci[1]:,.2f}]")
```

### Step 5 — Cost-Effectiveness Analysis

```python
# Point estimates
trt_mask = df['treatment'] == 1
ctrl_mask = df['treatment'] == 0

delta_c = df.loc[trt_mask, 'total_cost'].mean() - df.loc[ctrl_mask, 'total_cost'].mean()
delta_e = df.loc[trt_mask, 'qaly'].mean() - df.loc[ctrl_mask, 'qaly'].mean()
icer = delta_c / delta_e if delta_e != 0 else np.inf

print(f"Incremental Cost:  ${delta_c:,.2f}")
print(f"Incremental QALY:  {delta_e:.4f}")
print(f"ICER ($/QALY):     ${icer:,.2f}")

# Net monetary benefit
wtp = 50_000
nmb = wtp * delta_e - delta_c
print(f"\nNMB at WTP=$50,000: ${nmb:,.2f}")
print(f"Cost-effective: {'Yes' if nmb > 0 else 'No'}")
```

## Advanced Example

### Cost-Effectiveness Acceptability Curve

```python
def bootstrap_ce(df, n_boot=5000, seed=789):
    """Bootstrap joint (delta_C, delta_E) distribution."""
    rng = np.random.default_rng(seed)
    results = np.empty((n_boot, 2))
    n_total = len(df)
    for i in range(n_boot):
        sample = df.sample(n=n_total, replace=True, random_state=rng.integers(1e8))
        trt  = sample[sample['treatment'] == 1]
        ctrl = sample[sample['treatment'] == 0]
        results[i, 0] = trt['total_cost'].mean() - ctrl['total_cost'].mean()
        results[i, 1] = trt['qaly'].mean() - ctrl['qaly'].mean()
    return results

boot_ce = bootstrap_ce(df)
delta_c_boot = boot_ce[:, 0]
delta_e_boot = boot_ce[:, 1]

# CEAC
wtp_grid = np.arange(0, 200_001, 1000)
prob_ce = np.array([np.mean(w * delta_e_boot - delta_c_boot > 0) for w in wtp_grid])

fig, ax = plt.subplots(figsize=(8, 5))
ax.plot(wtp_grid, prob_ce, color='darkgreen', linewidth=2)
ax.axhline(y=0.5, linestyle='--', color='grey', alpha=0.5)
ax.set_xlabel('Willingness-to-Pay Threshold ($/QALY)')
ax.set_ylabel('Probability Cost-Effective')
ax.set_title('Cost-Effectiveness Acceptability Curve')
ax.set_xlim(0, 200_000)
ax.set_ylim(0, 1)
plt.tight_layout()
plt.savefig('ceac.png', dpi=150)
plt.show()
```

## Visualization

```python
# 1. Cost-Effectiveness Plane
fig, ax = plt.subplots(figsize=(8, 6))
ax.scatter(delta_e_boot, delta_c_boot, alpha=0.05, s=10, color='steelblue')
ax.axhline(y=0, linestyle='--', color='black', linewidth=0.8)
ax.axvline(x=0, linestyle='--', color='black', linewidth=0.8)

# WTP threshold line
e_range = np.array([delta_e_boot.min(), delta_e_boot.max()])
ax.plot(e_range, 50_000 * e_range, 'r--', linewidth=1.5, label='WTP = $50,000/QALY')

ax.set_xlabel('Incremental QALY')
ax.set_ylabel('Incremental Cost ($)')
ax.set_title('Cost-Effectiveness Plane')
ax.legend()
plt.tight_layout()
plt.savefig('ce_plane.png', dpi=150)
plt.show()

# 2. NMB distribution
nmb_boot = 50_000 * delta_e_boot - delta_c_boot

fig, ax = plt.subplots(figsize=(8, 5))
ax.hist(nmb_boot, bins=80, color='teal', alpha=0.7, edgecolor='white')
ax.axvline(x=0, color='red', linestyle='--', linewidth=2)
ax.set_xlabel('Net Monetary Benefit ($)')
ax.set_ylabel('Frequency')
ax.set_title(f'Distribution of NMB (WTP = $50,000/QALY)\n'
             f'P(NMB > 0) = {np.mean(nmb_boot > 0):.3f}')
plt.tight_layout()
plt.savefig('nmb_distribution.png', dpi=150)
plt.show()

# 3. QQ-plot to assess gamma fit for positive costs
from scipy.stats import gamma as gamma_dist

pos_costs = df.loc[df['total_cost'] > 0, 'total_cost'].values
fit_alpha, fit_loc, fit_beta = gamma_dist.fit(pos_costs, floc=0)

fig, ax = plt.subplots(figsize=(6, 6))
stats.probplot(pos_costs, dist=gamma_dist, sparams=(fit_alpha, fit_loc, fit_beta),
               plot=ax)
ax.set_title('QQ-Plot: Positive Costs vs Fitted Gamma')
plt.tight_layout()
plt.savefig('gamma_qq_plot.png', dpi=150)
plt.show()
```

## Tips and Best Practices

1. **Use `statsmodels` GLM with Gamma + Log link** as the default for positive cost
   modelling. It avoids retransformation bias and handles heteroscedasticity.

2. **Implement the two-part model manually** (logistic + gamma GLM) when there are
   structural zeros. Python does not have a built-in hurdle model for continuous data.

3. **Always bootstrap cost differences and ICERs.** Normal-theory CIs for skewed cost
   data have poor coverage. Use at least 2000 replicates.

4. **Present the full decision-analysis picture:** the CE plane, CEAC, and NMB
   distribution. A point ICER alone is insufficient.

5. **Use the modified Park test** (`log(residuals^2) ~ log(fitted)`) to choose the
   GLM variance family before committing to gamma.

6. **For censored costs** (patients lost to follow-up), implement Lin's partitioned
   estimator or IPCW. Ignoring censoring biases costs downward.

7. **Discount future costs and QALYs** when the time horizon exceeds one year. Apply the
   rate recommended by the relevant HTA body (e.g., 3% for US, 3.5% for UK NICE).

8. **Conduct probabilistic sensitivity analysis (PSA)** by drawing parameters from their
   sampling distributions and re-computing the ICER for each draw. The CEAC summarises
   the PSA results.
