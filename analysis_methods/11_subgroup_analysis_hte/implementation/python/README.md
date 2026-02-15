# Subgroup Analysis and Treatment Effect Heterogeneity (HTE) — Python Implementation

## Required Libraries

```bash
pip install econml causalml scikit-learn shap matplotlib numpy pandas seaborn
```

```python
import numpy as np
import pandas as pd
import matplotlib.pyplot as plt
import seaborn as sns
from sklearn.ensemble import RandomForestRegressor, GradientBoostingRegressor
from sklearn.model_selection import cross_val_predict
```

## Example Dataset

We simulate a cardiovascular trial where treatment effect on LDL reduction depends on baseline
LDL, diabetes status, and age.

```python
np.random.seed(42)
n = 1000

# Patient characteristics
age = np.random.normal(62, 10, n)
ldl_baseline = np.random.normal(150, 35, n)
diabetes = np.random.binomial(1, 0.3, n)
female = np.random.binomial(1, 0.45, n)
sbp = np.random.normal(140, 18, n)

# Treatment assignment (1:1 randomized)
treatment = np.random.binomial(1, 0.5, n)

# True CATE: benefit increases with higher LDL and diabetes
true_cate = -5 - 0.1 * (ldl_baseline - 150) - 4 * diabetes

# Outcome: change in LDL at 12 weeks
noise = np.random.normal(0, 12, n)
y = 0.3 * age + 0.4 * ldl_baseline + 5 * diabetes + treatment * true_cate + noise

df = pd.DataFrame({
    'y': y, 'treatment': treatment, 'age': age,
    'ldl_baseline': ldl_baseline, 'diabetes': diabetes,
    'female': female, 'sbp': sbp, 'true_cate': true_cate
})

X = df[['age', 'ldl_baseline', 'diabetes', 'female', 'sbp']].values
feature_names = ['age', 'ldl_baseline', 'diabetes', 'female', 'sbp']
print(df.head())
```

## Complete Worked Example

### Step 1: Classical Subgroup Analysis

```python
import statsmodels.api as sm
from statsmodels.formula.api import ols

# Interaction model for diabetes
model = ols('y ~ treatment * diabetes + age + ldl_baseline + female + sbp', data=df).fit()
print(model.summary())

# The treatment:diabetes coefficient shows the additional effect of treatment
# in diabetic vs non-diabetic patients.
# Expected: significant negative coefficient (diabetics benefit more).
print(f"\nInteraction coefficient: {model.params['treatment:diabetes']:.3f}")
print(f"Interaction p-value: {model.pvalues['treatment:diabetes']:.4f}")
```

### Step 2: Subgroup Forest Plot

```python
from scipy import stats

def subgroup_effect(data, subgroup_col, level):
    """Estimate treatment effect within a subgroup."""
    sub = data[data[subgroup_col] == level]
    treated = sub[sub['treatment'] == 1]['y']
    control = sub[sub['treatment'] == 0]['y']
    effect = treated.mean() - control.mean()
    se = np.sqrt(treated.var()/len(treated) + control.var()/len(control))
    return effect, se, len(sub)

# Define subgroups
df['age_group'] = np.where(df['age'] >= 65, 'Age >= 65', 'Age < 65')
df['ldl_group'] = np.where(df['ldl_baseline'] >= 160, 'LDL >= 160', 'LDL < 160')
df['diabetes_group'] = np.where(df['diabetes'] == 1, 'Diabetic', 'Non-diabetic')
df['sex_group'] = np.where(df['female'] == 1, 'Female', 'Male')

subgroups = [
    ('age_group', ['Age < 65', 'Age >= 65']),
    ('ldl_group', ['LDL < 160', 'LDL >= 160']),
    ('diabetes_group', ['Non-diabetic', 'Diabetic']),
    ('sex_group', ['Female', 'Male']),
]

results = []
for col, levels in subgroups:
    for level in levels:
        eff, se, n_sub = subgroup_effect(df, col, level)
        results.append({
            'subgroup': f'{level} (n={n_sub})',
            'effect': eff, 'se': se,
            'lower': eff - 1.96 * se, 'upper': eff + 1.96 * se
        })

# Add overall
treated_all = df[df['treatment'] == 1]['y']
control_all = df[df['treatment'] == 0]['y']
eff_all = treated_all.mean() - control_all.mean()
se_all = np.sqrt(treated_all.var()/len(treated_all) + control_all.var()/len(control_all))
results.insert(0, {
    'subgroup': f'Overall (n={len(df)})',
    'effect': eff_all, 'se': se_all,
    'lower': eff_all - 1.96 * se_all, 'upper': eff_all + 1.96 * se_all
})

res_df = pd.DataFrame(results)

# Forest plot
fig, ax = plt.subplots(figsize=(8, 6))
y_pos = range(len(res_df))
ax.errorbar(res_df['effect'], y_pos, xerr=1.96 * res_df['se'],
            fmt='o', color='steelblue', capsize=4, markersize=6)
ax.axvline(x=0, linestyle='--', color='grey', alpha=0.7)
ax.set_yticks(y_pos)
ax.set_yticklabels(res_df['subgroup'])
ax.set_xlabel('Treatment Effect (LDL Change)')
ax.set_title('Subgroup Forest Plot')
ax.invert_yaxis()
plt.tight_layout()
plt.show()
```

### Step 3: Causal Forest with EconML

```python
from econml.dml import CausalForestDML

# Fit causal forest
causal_forest = CausalForestDML(
    model_y=GradientBoostingRegressor(n_estimators=100, max_depth=4),
    model_t=GradientBoostingRegressor(n_estimators=100, max_depth=4),
    n_estimators=2000,
    min_samples_leaf=5,
    random_state=42
)

causal_forest.fit(Y=df['y'].values, T=df['treatment'].values, X=X)

# Estimate CATE for each patient
cate_pred = causal_forest.effect(X)
cate_intervals = causal_forest.effect_interval(X, alpha=0.05)

df['cate_hat'] = cate_pred
df['cate_lower'] = cate_intervals[0]
df['cate_upper'] = cate_intervals[1]

print("CATE Summary Statistics:")
print(df['cate_hat'].describe())

# Compare estimated vs true CATE
correlation = np.corrcoef(df['true_cate'], df['cate_hat'])[0, 1]
print(f"\nCorrelation between true and estimated CATE: {correlation:.3f}")
```

### Step 4: Feature Importance with SHAP

```python
import shap

# SHAP values for the causal forest (returns numpy array in EconML >= 0.14)
shap_values = causal_forest.shap_values(X)

# SHAP summary plot
shap.summary_plot(
    shap_values,
    X,
    feature_names=feature_names,
    show=True
)
# Interpretation: Features with wide SHAP value spread have the greatest
# influence on treatment effect heterogeneity. Expect ldl_baseline and
# diabetes to have the highest importance.
```

## Advanced Example

### T-Learner with CausalML

```python
from causalml.inference.meta import BaseTRegressor

# T-Learner using gradient boosting
t_learner = BaseTRegressor(
    learner=GradientBoostingRegressor(n_estimators=200, max_depth=4, random_state=42)
)

cate_t = t_learner.fit_predict(
    X=X,
    treatment=df['treatment'].values,
    y=df['y'].values
)

df['cate_t_learner'] = cate_t.flatten()
print("T-Learner CATE Summary:")
print(df['cate_t_learner'].describe())
```

### X-Learner

```python
from causalml.inference.meta import BaseXRegressor

x_learner = BaseXRegressor(
    learner=GradientBoostingRegressor(n_estimators=200, max_depth=4, random_state=42)
)

cate_x = x_learner.fit_predict(
    X=X,
    treatment=df['treatment'].values,
    y=df['y'].values
)

df['cate_x_learner'] = cate_x.flatten()
print("X-Learner CATE Summary:")
print(df['cate_x_learner'].describe())
```

### Compare All Meta-Learner Estimates

```python
methods = {
    'True CATE': df['true_cate'],
    'Causal Forest': df['cate_hat'],
    'T-Learner': df['cate_t_learner'],
    'X-Learner': df['cate_x_learner']
}

print("\nCorrelation with True CATE:")
for name, values in methods.items():
    if name != 'True CATE':
        r = np.corrcoef(df['true_cate'], values)[0, 1]
        print(f"  {name}: {r:.3f}")
```

## Visualization

```python
fig, axes = plt.subplots(2, 2, figsize=(14, 10))

# Plot 1: Distribution of estimated CATEs across methods
ax = axes[0, 0]
ax.hist(df['true_cate'], bins=40, alpha=0.5, label='True', color='black')
ax.hist(df['cate_hat'], bins=40, alpha=0.5, label='Causal Forest', color='steelblue')
ax.axvline(df['cate_hat'].mean(), color='red', linestyle='--', label='Mean est.')
ax.set_xlabel('CATE')
ax.set_ylabel('Count')
ax.set_title('Distribution of Treatment Effects')
ax.legend()

# Plot 2: Estimated vs True CATE scatter
ax = axes[0, 1]
ax.scatter(df['true_cate'], df['cate_hat'], alpha=0.3, s=10, color='steelblue')
lims = [df['true_cate'].min() - 2, df['true_cate'].max() + 2]
ax.plot(lims, lims, '--', color='red', label='Perfect calibration')
ax.set_xlabel('True CATE')
ax.set_ylabel('Estimated CATE (Causal Forest)')
ax.set_title('Calibration: Estimated vs True CATE')
ax.legend()

# Plot 3: CATE by LDL baseline, stratified by diabetes
ax = axes[1, 0]
for diab_val, label, color in [(0, 'No diabetes', 'blue'), (1, 'Diabetes', 'red')]:
    mask = df['diabetes'] == diab_val
    ax.scatter(df.loc[mask, 'ldl_baseline'], df.loc[mask, 'cate_hat'],
               alpha=0.3, s=10, color=color, label=label)
ax.set_xlabel('Baseline LDL (mg/dL)')
ax.set_ylabel('Estimated CATE')
ax.set_title('CATE by Baseline LDL and Diabetes')
ax.legend()

# Plot 4: CATE confidence intervals for a random subset
ax = axes[1, 1]
sample_idx = np.random.choice(len(df), 50, replace=False)
sample_idx = sample_idx[np.argsort(df['cate_hat'].values[sample_idx])]
y_pos = range(len(sample_idx))
ax.errorbar(
    df['cate_hat'].values[sample_idx], y_pos,
    xerr=[
        df['cate_hat'].values[sample_idx] - df['cate_lower'].values[sample_idx],
        df['cate_upper'].values[sample_idx] - df['cate_hat'].values[sample_idx]
    ],
    fmt='o', markersize=3, capsize=2, color='steelblue'
)
ax.axvline(0, linestyle='--', color='grey')
ax.set_xlabel('Estimated CATE (95% CI)')
ax.set_ylabel('Patient (sorted)')
ax.set_title('Individual CATE Estimates with Uncertainty')

plt.tight_layout()
plt.show()
```

## Tips and Best Practices

1. **Choose the right meta-learner**: T-learners work well when treatment and control outcome
   surfaces differ substantially. S-learners can miss heterogeneity when effects are small
   relative to prognostic variation. X-learners excel with imbalanced treatment groups.

2. **Cross-fit to avoid overfitting**: EconML's `CausalForestDML` uses cross-fitting (double
   machine learning) by default. This is important for valid inference.

3. **Validate with known DGP**: Before applying to real data, test your pipeline on simulated
   data where the true CATE is known. This calibration step builds confidence in the method.

4. **Interpret SHAP carefully**: SHAP values for CATE tell you which features drive heterogeneity
   in treatment effects, not which features predict the outcome. A variable can be highly
   prognostic (predicting Y) but not predictive (modifying treatment effect).

5. **Report confidence intervals**: Always report uncertainty in CATE estimates. Point estimates
   without intervals can be misleading, especially for individual-level predictions.

6. **Use appropriate sample sizes**: CATE estimation requires much larger samples than ATE
   estimation. With fewer than 500 observations, classical subgroup analysis with a few
   pre-specified subgroups may outperform ML methods.

7. **Consider the clinical context**: Not all statistically significant heterogeneity is
   clinically meaningful. A CATE difference of 1 mg/dL in LDL may be statistically detectable
   in a large trial but clinically irrelevant.

8. **Replicate before acting**: Exploratory HTE findings from causal forests or meta-learners
   should be treated as hypothesis-generating. Independent validation is essential before
   changing clinical practice or regulatory labeling.
