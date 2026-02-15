# Causal Mediation Analysis — Python Implementation

## Required Libraries

```bash
pip install numpy pandas statsmodels pingouin matplotlib seaborn dowhy
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

We simulate a clinical trial: an anti-inflammatory drug (treatment) reduces joint
damage (outcome), partly by lowering CRP (mediator).

```python
np.random.seed(42)
n = 500

age       = np.random.normal(55, 10, n)
female    = np.random.binomial(1, 0.45, n)
treatment = np.random.binomial(1, 0.5, n)

crp_reduction = (0.8 * treatment + 0.02 * age - 0.1 * female
                 + 0.3 * treatment * female + np.random.normal(0, 1, n))

joint_damage = (-1.5 * treatment - 0.6 * crp_reduction
                + 0.2 * treatment * crp_reduction
                + 0.03 * age + 0.1 * female + np.random.normal(0, 1.2, n))

df = pd.DataFrame({
    'treatment': treatment, 'crp_reduction': crp_reduction,
    'joint_damage': joint_damage, 'age': age, 'female': female
})
print(df.head())
```

## Complete Worked Example

### Step 1 — Baron and Kenny Approach (manual with statsmodels)

```python
# Path c: total effect
total = smf.ols('joint_damage ~ treatment + age + female', data=df).fit()
print("=== Total Effect (c) ===")
print(f"  Treatment coef: {total.params['treatment']:.4f}")
print(f"  p-value:        {total.pvalues['treatment']:.4f}")

# Path a: treatment -> mediator
med_model = smf.ols('crp_reduction ~ treatment + age + female', data=df).fit()
print("\n=== Path a (Treatment -> Mediator) ===")
print(f"  Treatment coef: {med_model.params['treatment']:.4f}")

# Paths b and c': mediator + treatment -> outcome
out_model = smf.ols('joint_damage ~ treatment + crp_reduction + age + female',
                     data=df).fit()
print("\n=== Path b and c' ===")
print(f"  CRP coef (b):       {out_model.params['crp_reduction']:.4f}")
print(f"  Treatment coef (c'): {out_model.params['treatment']:.4f}")

# Indirect effect
a = med_model.params['treatment']
b = out_model.params['crp_reduction']
indirect = a * b
direct   = out_model.params['treatment']
print(f"\n  Indirect effect (a*b): {indirect:.4f}")
print(f"  Direct effect (c'):    {direct:.4f}")
print(f"  Total (c' + a*b):      {indirect + direct:.4f}")
```

### Step 2 — Bootstrap Confidence Interval for Indirect Effect

```python
def bootstrap_indirect(df, n_boot=2000):
    """Bootstrap the indirect effect a*b."""
    effects = np.empty(n_boot)
    for i in range(n_boot):
        sample = df.sample(n=len(df), replace=True)
        med = smf.ols('crp_reduction ~ treatment + age + female',
                       data=sample).fit()
        out = smf.ols('joint_damage ~ treatment + crp_reduction + age + female',
                       data=sample).fit()
        effects[i] = med.params['treatment'] * out.params['crp_reduction']
    return effects

boot_ie = bootstrap_indirect(df, n_boot=2000)
ci_lower, ci_upper = np.percentile(boot_ie, [2.5, 97.5])
print(f"Bootstrap indirect effect: {np.mean(boot_ie):.4f}")
print(f"95% CI: [{ci_lower:.4f}, {ci_upper:.4f}]")
# If the CI excludes zero, the indirect effect is significant at alpha=0.05.
```

### Step 3 — Counterfactual Mediation (simulation-based, like Imai et al.)

```python
def counterfactual_mediation(df, n_sim=1000):
    """
    Simulation-based counterfactual mediation analysis.
    Estimates NDE and NIE allowing treatment-mediator interaction.
    """
    # Fit models WITH interaction
    med_fit = smf.ols('crp_reduction ~ treatment * female + age', data=df).fit()
    out_fit = smf.ols('joint_damage ~ treatment * crp_reduction + age + female',
                       data=df).fit()

    nde_list, nie_list = [], []

    for _ in range(n_sim):
        # Simulate mediator under treatment and control
        df_t1 = df.copy(); df_t1['treatment'] = 1
        df_t0 = df.copy(); df_t0['treatment'] = 0

        m1 = med_fit.predict(df_t1) + np.random.normal(0, med_fit.mse_resid**0.5, len(df))
        m0 = med_fit.predict(df_t0) + np.random.normal(0, med_fit.mse_resid**0.5, len(df))

        # Y(1, M(0)) — direct effect world
        df_nde = df.copy(); df_nde['treatment'] = 1; df_nde['crp_reduction'] = m0
        y1_m0 = out_fit.predict(df_nde)

        # Y(0, M(0))
        df_00 = df.copy(); df_00['treatment'] = 0; df_00['crp_reduction'] = m0
        y0_m0 = out_fit.predict(df_00)

        # Y(1, M(1))
        df_11 = df.copy(); df_11['treatment'] = 1; df_11['crp_reduction'] = m1
        y1_m1 = out_fit.predict(df_11)

        nde_list.append(np.mean(y1_m0 - y0_m0))
        nie_list.append(np.mean(y1_m1 - y1_m0))

    nde = np.mean(nde_list)
    nie = np.mean(nie_list)
    te  = nde + nie
    prop_mediated = nie / te if te != 0 else np.nan

    return {
        'NDE': nde, 'NDE_CI': np.percentile(nde_list, [2.5, 97.5]),
        'NIE': nie, 'NIE_CI': np.percentile(nie_list, [2.5, 97.5]),
        'TE':  te,  'Prop_Mediated': prop_mediated
    }

results = counterfactual_mediation(df)
print("=== Counterfactual Mediation Results ===")
for k, v in results.items():
    print(f"  {k}: {v}")
```

### Step 4 — Pingouin Mediation (simplified)

```python
import pingouin as pg

# Pingouin provides a quick Baron-Kenny-style mediation with Sobel and bootstrap
med_pg = pg.mediation_analysis(
    data=df, x='treatment', m='crp_reduction', y='joint_damage',
    covar=['age', 'female'], alpha=0.05, seed=42
)
print(med_pg.to_string(index=False))
# Columns: path, coef, se, pval, CI[2.5%], CI[97.5%]
```

### Step 5 — Sensitivity Analysis (rho-based)

```python
def sensitivity_analysis(df, rho_range=np.arange(-0.9, 0.91, 0.05)):
    """
    Sensitivity of the indirect effect to correlation rho between
    mediator and outcome residuals (unmeasured confounding).
    """
    med_fit = smf.ols('crp_reduction ~ treatment + age + female', data=df).fit()
    out_fit = smf.ols('joint_damage ~ treatment + crp_reduction + age + female',
                       data=df).fit()

    a = med_fit.params['treatment']
    sigma_m = np.sqrt(med_fit.mse_resid)
    sigma_y = np.sqrt(out_fit.mse_resid)

    adjusted_ie = []
    for rho in rho_range:
        bias = rho * sigma_m * sigma_y / sigma_m  # simplified bias term
        adjusted = a * (out_fit.params['crp_reduction'] - rho * sigma_y / sigma_m)
        adjusted_ie.append(adjusted)

    return pd.DataFrame({'rho': rho_range, 'adjusted_IE': adjusted_ie})

sens_df = sensitivity_analysis(df)
print(sens_df.to_string(index=False))
```

## Advanced Example

### Causal Mediation with DoWhy

```python
import dowhy
from dowhy import CausalModel

# Build causal graph
model = CausalModel(
    data=df,
    treatment='treatment',
    outcome='joint_damage',
    graph="""digraph {
        treatment -> crp_reduction;
        treatment -> joint_damage;
        crp_reduction -> joint_damage;
        age -> crp_reduction;
        age -> joint_damage;
        female -> crp_reduction;
        female -> joint_damage;
    }"""
)

# Identify NDE
identified_nde = model.identify_effect(
    estimand_type="nonparametric-nde",
    proceed_when_unidentifiable=True
)
print(identified_nde)

# Identify NIE
identified_nie = model.identify_effect(
    estimand_type="nonparametric-nie",
    proceed_when_unidentifiable=True
)
print(identified_nie)

# Estimate (using linear regression estimator)
nde_est = model.estimate_effect(
    identified_nde,
    method_name="mediation.two_stage_regression",
    method_params={"first_stage_model": smf.ols, "second_stage_model": smf.ols}
)
print(f"NDE estimate: {nde_est.value:.4f}")
```

## Visualization

```python
# 1. Forest plot of mediation effects
fig, ax = plt.subplots(figsize=(8, 4))
effects = ['NIE (ACME)', 'NDE (ADE)', 'Total Effect']
estimates = [results['NIE'], results['NDE'], results['TE']]
ci_lo = [results['NIE_CI'][0], results['NDE_CI'][0],
         results['NIE_CI'][0] + results['NDE_CI'][0]]
ci_hi = [results['NIE_CI'][1], results['NDE_CI'][1],
         results['NIE_CI'][1] + results['NDE_CI'][1]]

y_pos = range(len(effects))
ax.errorbar(estimates, y_pos, xerr=[np.array(estimates)-np.array(ci_lo),
            np.array(ci_hi)-np.array(estimates)], fmt='o', capsize=5, color='steelblue')
ax.axvline(x=0, color='red', linestyle='--', alpha=0.7)
ax.set_yticks(y_pos)
ax.set_yticklabels(effects)
ax.set_xlabel('Effect Estimate')
ax.set_title('Causal Mediation Analysis Results')
plt.tight_layout()
plt.savefig('mediation_forest_plot.png', dpi=150)
plt.show()

# 2. Sensitivity analysis plot
fig, ax = plt.subplots(figsize=(7, 5))
ax.plot(sens_df['rho'], sens_df['adjusted_IE'], 'b-', linewidth=2)
ax.axhline(y=0, color='red', linestyle='--')
ax.axvline(x=0, color='grey', linestyle=':', alpha=0.5)
ax.set_xlabel('Sensitivity Parameter (rho)')
ax.set_ylabel('Adjusted Indirect Effect')
ax.set_title('Sensitivity of Indirect Effect to Unmeasured Confounding')
ax.fill_between(sens_df['rho'], sens_df['adjusted_IE'],
                alpha=0.1, color='blue')
plt.tight_layout()
plt.savefig('sensitivity_plot.png', dpi=150)
plt.show()

# 3. Path diagram (text-based visualisation)
print("""
  Path Diagram
  ============
  Treatment --[a]--> CRP Reduction --[b]--> Joint Damage
       |                                        ^
       +------------[c' (direct)]---------------+

  a  = {:.3f}
  b  = {:.3f}
  c' = {:.3f}
  a*b (indirect) = {:.3f}
""".format(
    med_model.params['treatment'],
    out_model.params['crp_reduction'],
    out_model.params['treatment'],
    med_model.params['treatment'] * out_model.params['crp_reduction']
))
```

## Tips and Best Practices

1. **Always bootstrap the indirect effect.** The Sobel test assumes normality of the
   product a*b, which is violated in practice. Use at least 2000 bootstrap replicates.

2. **Include the treatment-mediator interaction term** in the outcome model. If omitted
   and interaction exists, both NDE and NIE estimates are biased.

3. **Report and interpret the sensitivity analysis.** State the rho value at which the
   indirect effect becomes non-significant. This is the most important robustness check.

4. **Use DoWhy's graph-based approach** when you have a complex DAG with multiple
   confounders and mediators. It enforces identification before estimation.

5. **For binary mediators or outcomes,** replace `smf.ols` with `smf.logit` or
   `smf.probit` and use the simulation-based approach for effect decomposition.

6. **Pingouin is convenient for quick checks** but does not support interaction terms or
   counterfactual decomposition. Use the custom implementation for rigorous analysis.

7. **Pre-specify the mediation hypothesis** in the analysis plan. Post-hoc mediation
   analyses should be clearly labelled as exploratory.

8. **Check temporal ordering.** The mediator must be measured after treatment and before
   the outcome. Cross-sectional data cannot support causal mediation claims.
