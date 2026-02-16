# Meta-Analysis — Python Implementation

## Required Libraries

```bash
pip install numpy scipy pandas statsmodels matplotlib forestplot PythonMeta
```

- **numpy / scipy**: Core numerical computation and statistical distributions.
- **pandas**: Data manipulation and tabular summaries.
- **statsmodels**: Regression-based approaches for meta-analysis.
- **matplotlib**: Publication-quality plots.
- **forestplot**: Dedicated forest plot library.
- **PythonMeta**: Meta-analysis specific package (fixed/random effects, heterogeneity).

## Example Dataset

We use data from 13 randomized controlled trials comparing a new antihypertensive drug versus placebo, reporting the mean difference (MD) in systolic blood pressure (SBP) reduction and its standard error.

```python
import numpy as np
import pandas as pd

dat = pd.DataFrame({
    'study': [f'Trial_{i}' for i in range(1, 14)],
    'year':  [2005, 2007, 2008, 2009, 2010, 2011, 2012, 2013, 2014, 2015, 2016, 2018, 2020],
    'md':    [-8.2, -5.1, -7.8, -6.3, -9.1, -4.5, -7.0, -8.8, -6.0, -10.2, -5.5, -7.3, -6.9],
    'se':    [1.5, 2.0, 1.8, 1.2, 2.5, 1.0, 1.6, 1.4, 2.2, 1.9, 1.3, 1.7, 1.1],
    'dose':  [10, 10, 20, 20, 10, 20, 10, 20, 10, 20, 10, 20, 20],
})
```

## Complete Worked Example

### Step 1: Fixed-Effect Meta-Analysis (Inverse-Variance Method)

```python
import numpy as np
from scipy import stats

yi = dat['md'].values
sei = dat['se'].values
vi = sei ** 2

# Fixed-effect weights
w_fe = 1.0 / vi
pooled_fe = np.sum(w_fe * yi) / np.sum(w_fe)
se_fe = np.sqrt(1.0 / np.sum(w_fe))
ci_fe_lower = pooled_fe - 1.96 * se_fe
ci_fe_upper = pooled_fe + 1.96 * se_fe
z_fe = pooled_fe / se_fe
p_fe = 2 * stats.norm.sf(np.abs(z_fe))

print("=== Fixed-Effect Model ===")
print(f"Pooled MD: {pooled_fe:.3f}")
print(f"95% CI: ({ci_fe_lower:.3f}, {ci_fe_upper:.3f})")
print(f"Z = {z_fe:.3f}, p = {p_fe:.6f}")
```

**Output interpretation**: The fixed-effect pooled estimate represents the common true effect assuming all trials estimate the same underlying parameter. With antihypertensive trials, we expect a negative MD indicating blood pressure reduction. The narrow CI reflects the combined sample size.

### Step 2: Heterogeneity Assessment

```python
# Cochran's Q statistic
Q = np.sum(w_fe * (yi - pooled_fe) ** 2)
df_Q = len(yi) - 1
p_Q = 1 - stats.chi2.cdf(Q, df_Q)

# I-squared
I2 = max(0, (Q - df_Q) / Q) * 100

# Tau-squared (DerSimonian-Laird)
C = np.sum(w_fe) - np.sum(w_fe ** 2) / np.sum(w_fe)
tau2_DL = max(0, (Q - df_Q) / C)

print("=== Heterogeneity ===")
print(f"Q = {Q:.3f}, df = {df_Q}, p = {p_Q:.4f}")
print(f"I-squared = {I2:.1f}%")
print(f"Tau-squared (DL) = {tau2_DL:.4f}")
```

**Output interpretation**: A significant Q-test (p < 0.10) and I-squared above 50% indicate substantial heterogeneity. Tau-squared gives the between-study variance on the effect-size scale. When heterogeneity is present, the random-effects model is preferred.

### Step 3: Random-Effects Meta-Analysis

```python
# Random-effects weights using DL tau-squared
w_re = 1.0 / (vi + tau2_DL)
pooled_re = np.sum(w_re * yi) / np.sum(w_re)
se_re = np.sqrt(1.0 / np.sum(w_re))
ci_re_lower = pooled_re - 1.96 * se_re
ci_re_upper = pooled_re + 1.96 * se_re
z_re = pooled_re / se_re
p_re = 2 * stats.norm.sf(np.abs(z_re))

print("=== Random-Effects Model (DL) ===")
print(f"Pooled MD: {pooled_re:.3f}")
print(f"95% CI: ({ci_re_lower:.3f}, {ci_re_upper:.3f})")
print(f"Z = {z_re:.3f}, p = {p_re:.6f}")

# Prediction interval
k = len(yi)
t_crit = stats.t.ppf(0.975, df=k - 2)
pi_lower = pooled_re - t_crit * np.sqrt(tau2_DL + se_re ** 2)
pi_upper = pooled_re + t_crit * np.sqrt(tau2_DL + se_re ** 2)
print(f"95% Prediction Interval: ({pi_lower:.3f}, {pi_upper:.3f})")
```

**Output interpretation**: The random-effects estimate will typically have a wider confidence interval than the fixed-effect estimate. The prediction interval is even wider and shows where the true effect in a new study would likely fall. If the prediction interval includes zero, the treatment may not be effective in all settings.

### Step 4: REML Estimation Using Statsmodels

```python
import warnings
warnings.filterwarnings('ignore')

from scipy.optimize import minimize_scalar

def neg_reml_loglik(tau2, yi, vi):
    """Negative REML log-likelihood for random-effects meta-analysis."""
    wi_star = 1.0 / (vi + tau2)
    mu_hat = np.sum(wi_star * yi) / np.sum(wi_star)
    ll = -0.5 * np.sum(np.log(vi + tau2))
    ll -= 0.5 * np.sum(wi_star * (yi - mu_hat) ** 2)
    ll -= 0.5 * np.log(np.sum(wi_star))
    return -ll

result = minimize_scalar(neg_reml_loglik, bounds=(0, 10), method='bounded',
                         args=(yi, vi))
tau2_reml = result.x

w_reml = 1.0 / (vi + tau2_reml)
pooled_reml = np.sum(w_reml * yi) / np.sum(w_reml)
se_reml = np.sqrt(1.0 / np.sum(w_reml))

print("=== REML Estimation ===")
print(f"Tau-squared (REML): {tau2_reml:.4f}")
print(f"Pooled MD (REML): {pooled_reml:.3f}")
print(f"95% CI: ({pooled_reml - 1.96*se_reml:.3f}, {pooled_reml + 1.96*se_reml:.3f})")
```

**Output interpretation**: REML typically produces slightly larger tau-squared estimates than DerSimonian-Laird, leading to wider confidence intervals with better coverage properties. This is preferred when the number of studies is small (fewer than ~20).

### Step 5: Publication Bias Assessment

```python
# Egger's regression test
from scipy import stats as sp_stats

precision = 1.0 / sei
std_effect = yi / sei

slope, intercept, r_value, p_egger, std_err = sp_stats.linregress(precision, std_effect)

print("=== Egger's Test for Publication Bias ===")
print(f"Intercept: {intercept:.3f}")
print(f"p-value: {p_egger:.4f}")
if p_egger < 0.10:
    print("Evidence of funnel plot asymmetry (potential publication bias)")
else:
    print("No significant asymmetry detected")
```

**Output interpretation**: A significant intercept in Egger's test (p < 0.10) suggests funnel plot asymmetry, potentially due to publication bias. However, this test has low power with fewer than 10 studies and can be confounded by genuine heterogeneity.

## Advanced Example

### Meta-Regression

```python
import statsmodels.api as sm

# Meta-regression: effect of dose on treatment effect
# Weighted least squares with random-effects weights
X = sm.add_constant(dat['dose'].values)
W = np.diag(w_re)

wls_model = sm.WLS(yi, X, weights=w_re)
wls_result = wls_model.fit()

print("=== Meta-Regression: Effect of Dose ===")
print(wls_result.summary())
print(f"\nIntercept: {wls_result.params[0]:.3f} (higher dose -> baseline effect)")
print(f"Dose coefficient: {wls_result.params[1]:.3f} per mg increase")
if wls_result.pvalues[1] < 0.05:
    print("Dose significantly moderates the treatment effect")
else:
    print("No significant dose-response relationship detected")
```

**Output interpretation**: The dose coefficient indicates whether increasing dose is associated with larger (more negative) SBP reductions. A significant negative coefficient would suggest a dose-response relationship. The R-squared from this model (compared to the base model) indicates the proportion of heterogeneity explained by dose.

### Leave-One-Out Sensitivity Analysis

```python
loo_results = []
for i in range(k):
    yi_loo = np.delete(yi, i)
    vi_loo = np.delete(vi, i)
    w_loo = 1.0 / (vi_loo + tau2_DL)
    pooled_loo = np.sum(w_loo * yi_loo) / np.sum(w_loo)
    se_loo = np.sqrt(1.0 / np.sum(w_loo))
    loo_results.append({
        'omitted': dat['study'].iloc[i],
        'pooled_md': pooled_loo,
        'ci_lower': pooled_loo - 1.96 * se_loo,
        'ci_upper': pooled_loo + 1.96 * se_loo
    })

loo_df = pd.DataFrame(loo_results)
print("=== Leave-One-Out Analysis ===")
print(loo_df.to_string(index=False))
```

**Output interpretation**: If omitting a single study substantially changes the pooled estimate, that study is influential. Consistent estimates across all omissions suggest robustness.

## Visualization

### Forest Plot

```python
import matplotlib.pyplot as plt
import matplotlib.patches as mpatches

fig, ax = plt.subplots(figsize=(10, 8))

studies = dat['study'].values
y_positions = np.arange(len(studies), 0, -1)

# Plot each study
for i, (study, y_pos) in enumerate(zip(studies, y_positions)):
    ci_lower_i = yi[i] - 1.96 * sei[i]
    ci_upper_i = yi[i] + 1.96 * sei[i]
    weight_i = w_re[i] / np.sum(w_re) * 100

    # CI line
    ax.plot([ci_lower_i, ci_upper_i], [y_pos, y_pos], 'b-', linewidth=1)
    # Point estimate (size proportional to weight)
    ax.plot(yi[i], y_pos, 's', color='darkblue',
            markersize=3 + weight_i * 0.8, zorder=5)
    # Study label
    ax.text(-14, y_pos, f"{study}", va='center', ha='left', fontsize=9)
    # Effect and CI text
    ax.text(2, y_pos, f"{yi[i]:.1f} [{ci_lower_i:.1f}, {ci_upper_i:.1f}]",
            va='center', ha='left', fontsize=8)

# Pooled estimate diamond
diamond_y = 0
diamond_hw = 0.3
ax.fill([ci_re_lower, pooled_re, ci_re_upper, pooled_re],
        [diamond_y, diamond_y + diamond_hw, diamond_y, diamond_y - diamond_hw],
        color='steelblue', edgecolor='darkblue', linewidth=1.5)
ax.text(-14, diamond_y, "Pooled (RE)", va='center', ha='left',
        fontsize=9, fontweight='bold')
ax.text(2, diamond_y,
        f"{pooled_re:.1f} [{ci_re_lower:.1f}, {ci_re_upper:.1f}]",
        va='center', ha='left', fontsize=8, fontweight='bold')

# Reference line at zero
ax.axvline(x=0, color='black', linestyle='--', linewidth=0.8, alpha=0.5)

ax.set_xlabel("Mean Difference in SBP (mmHg)", fontsize=11)
ax.set_title("Forest Plot: Antihypertensive Drug vs Placebo", fontsize=13)
ax.set_yticks([])
ax.set_xlim(-15, 6)
ax.spines['top'].set_visible(False)
ax.spines['right'].set_visible(False)
ax.spines['left'].set_visible(False)

plt.tight_layout()
plt.savefig("forest_plot.png", dpi=150, bbox_inches='tight')
plt.show()
```

**Output interpretation**: The forest plot displays each trial as a square with horizontal confidence interval bars. Larger squares indicate greater weight. The diamond at the bottom represents the pooled random-effects estimate. All studies showing negative mean differences indicate consistent blood pressure reduction.

### Funnel Plot

```python
fig, ax = plt.subplots(figsize=(8, 6))

ax.scatter(yi, sei, s=50, color='darkblue', edgecolors='black', zorder=5)

# Pseudo-confidence region
se_range = np.linspace(0.01, max(sei) * 1.3, 100)
ax.fill_betweenx(se_range,
                 pooled_re - 1.96 * se_range,
                 pooled_re + 1.96 * se_range,
                 alpha=0.1, color='steelblue', label='95% pseudo-CI')
ax.axvline(x=pooled_re, color='red', linestyle='--', linewidth=1,
           label=f'Pooled MD = {pooled_re:.2f}')

ax.set_xlabel("Mean Difference (mmHg)", fontsize=11)
ax.set_ylabel("Standard Error", fontsize=11)
ax.set_title("Funnel Plot", fontsize=13)
ax.invert_yaxis()
ax.legend(loc='lower right')

plt.tight_layout()
plt.savefig("funnel_plot.png", dpi=150, bbox_inches='tight')
plt.show()
```

**Output interpretation**: In a symmetric funnel plot, studies scatter evenly around the pooled estimate, with smaller studies (larger SE) showing more spread. Asymmetry — particularly a gap in the lower-right region — may suggest publication bias, where small studies with non-significant results were not published.

### Cumulative Meta-Analysis

```python
# Sort by year and compute cumulative pooled estimates
sort_idx = np.argsort(dat['year'].values)
yi_sorted = yi[sort_idx]
vi_sorted = vi[sort_idx]
studies_sorted = dat['study'].values[sort_idx]
years_sorted = dat['year'].values[sort_idx]

cum_pooled = []
cum_lower = []
cum_upper = []

for j in range(1, k + 1):
    y_j = yi_sorted[:j]
    v_j = vi_sorted[:j]
    w_j = 1.0 / (v_j + tau2_DL)
    p_j = np.sum(w_j * y_j) / np.sum(w_j)
    se_j = np.sqrt(1.0 / np.sum(w_j))
    cum_pooled.append(p_j)
    cum_lower.append(p_j - 1.96 * se_j)
    cum_upper.append(p_j + 1.96 * se_j)

fig, ax = plt.subplots(figsize=(10, 6))
y_pos = np.arange(k, 0, -1)

for i in range(k):
    ax.plot([cum_lower[i], cum_upper[i]], [y_pos[i], y_pos[i]], 'b-', linewidth=1)
    ax.plot(cum_pooled[i], y_pos[i], 's', color='steelblue', markersize=6)
    ax.text(-14, y_pos[i],
            f"+ {studies_sorted[i]} ({years_sorted[i]})",
            va='center', ha='left', fontsize=8)

ax.axvline(x=0, color='black', linestyle='--', linewidth=0.8, alpha=0.5)
ax.set_xlabel("Cumulative Pooled MD (mmHg)", fontsize=11)
ax.set_title("Cumulative Meta-Analysis (by Year)", fontsize=13)
ax.set_yticks([])
ax.spines['top'].set_visible(False)
ax.spines['right'].set_visible(False)
ax.spines['left'].set_visible(False)
plt.tight_layout()
plt.savefig("cumulative_meta_analysis.png", dpi=150, bbox_inches='tight')
plt.show()
```

**Output interpretation**: The cumulative meta-analysis shows how the pooled estimate evolves as each study is added chronologically. If the estimate stabilizes over time, this suggests the evidence has converged. A dramatic shift from a late study warrants investigation.

## Tips and Best Practices

1. **Use REML over DerSimonian-Laird** for tau-squared estimation when possible. REML has better statistical properties, especially with fewer than 20 studies.
2. **Always report heterogeneity metrics**: present Q, I-squared, tau-squared, and the prediction interval together for a complete picture.
3. **Do not dichotomize I-squared**: treat it as a continuous measure; rigid cutoffs (25/50/75%) are guidelines, not rules.
4. **Validate custom implementations** against established R packages (metafor) using the same dataset before relying on results.
5. **For publication bias**, Egger's test is unreliable with fewer than 10 studies. Consider contour-enhanced funnel plots and selection model approaches.
6. **Pre-register your analysis plan** including planned subgroup and sensitivity analyses to avoid post-hoc data dredging.
7. **For network meta-analysis** in Python, consider using R via `rpy2` to access the `netmeta` or `gemtc` packages, as Python NMA tools are less mature.
8. **Check for outliers** using leave-one-out analysis and examine whether any single study drives the conclusion.
9. **Present both fixed-effect and random-effects results** when heterogeneity is present to show the impact of the modeling assumption.
10. **Follow PRISMA reporting guidelines** for transparency and reproducibility.
