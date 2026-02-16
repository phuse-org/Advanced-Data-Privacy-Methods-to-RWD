# Bayesian Methods — Python Implementation

## Required Libraries

```bash
pip install pymc arviz bambi pandas numpy matplotlib seaborn scipy
```

```python
import numpy as np
import pandas as pd
import matplotlib.pyplot as plt
import seaborn as sns
import pymc as pm
import arviz as az
import warnings
warnings.filterwarnings('ignore')

# Set ArviZ style
az.style.use("arviz-darkgrid")
```

## Example Dataset

We simulate a multi-center clinical trial where patients receive treatment or placebo and the primary outcome is LDL cholesterol reduction. The hierarchical structure (patients nested within centers) motivates Bayesian multilevel modeling.

```python
np.random.seed(42)
n_centers = 12
n_per_center = np.random.poisson(40, n_centers)
n_total = n_per_center.sum()

# Center-level random effects
center_effects = np.random.normal(0, 3, n_centers)

# Patient-level data
center_id = np.repeat(np.arange(n_centers), n_per_center)
treatment = np.random.binomial(1, 0.5, n_total)
age = np.round(np.random.normal(58, 10, n_total)).astype(int)
sex = np.random.binomial(1, 0.45, n_total)  # 0=Female, 1=Male
baseline_ldl = np.round(np.random.normal(160, 30, n_total), 1)

# Outcome: LDL reduction (true treatment effect = 12)
ldl_reduction = np.round(
    15 + 12 * treatment + 0.1 * age - 2 * sex +
    0.05 * baseline_ldl + center_effects[center_id] +
    np.random.normal(0, 8, n_total), 1
)

# Binary outcome
meaningful_reduction = (ldl_reduction > 20).astype(int)

trial = pd.DataFrame({
    'center_id': center_id,
    'treatment': treatment,
    'age': age,
    'sex': sex,
    'baseline_ldl': baseline_ldl,
    'ldl_reduction': ldl_reduction,
    'meaningful_reduction': meaningful_reduction
})

print(f"Total patients: {n_total}")
print(f"Centers: {n_centers}")
print(f"Treatment rate: {treatment.mean():.2f}")
print(f"Mean reduction (treatment): {ldl_reduction[treatment==1].mean():.1f}")
print(f"Mean reduction (placebo): {ldl_reduction[treatment==0].mean():.1f}")
```

## Complete Worked Example

### Step 1: Bayesian Linear Regression with PyMC

```python
import pymc as pm

with pm.Model() as linear_model:
    # Priors
    intercept = pm.Normal('intercept', mu=0, sigma=20)
    b_treatment = pm.Normal('b_treatment', mu=0, sigma=10)
    b_age = pm.Normal('b_age', mu=0, sigma=10)
    b_sex = pm.Normal('b_sex', mu=0, sigma=10)
    b_baseline = pm.Normal('b_baseline', mu=0, sigma=10)
    sigma = pm.HalfStudentT('sigma', nu=3, sigma=10)

    # Linear predictor
    mu = (intercept + b_treatment * trial['treatment'].values +
          b_age * trial['age'].values +
          b_sex * trial['sex'].values +
          b_baseline * trial['baseline_ldl'].values)

    # Likelihood
    y_obs = pm.Normal('y_obs', mu=mu, sigma=sigma,
                       observed=trial['ldl_reduction'].values)

    # Sample from posterior
    trace = pm.sample(3000, tune=1000, chains=4, cores=4,
                       random_seed=42, target_accept=0.9)

# Summary
print(az.summary(trace, var_names=['intercept', 'b_treatment', 'b_age',
                                     'b_sex', 'b_baseline', 'sigma'],
                  hdi_prob=0.95))
```

**Interpretation**: The summary shows posterior means, standard deviations, and 95% highest density intervals (HDI) for each parameter. The treatment effect posterior mean should be near the true value of 12. Rhat near 1.00 and ESS > 400 confirm convergence. The HDI provides a direct probability statement: "There is a 95% probability that the treatment effect lies in this interval."

### Step 2: MCMC Diagnostics

```python
import arviz as az

# Trace plots
az.plot_trace(trace, var_names=['b_treatment', 'b_age', 'sigma'],
              compact=True, figsize=(12, 8))
plt.tight_layout()
plt.savefig('trace_plots.png', dpi=150)
plt.show()

# Rank plots (more sensitive than trace plots)
az.plot_rank(trace, var_names=['b_treatment', 'b_age', 'sigma'],
             figsize=(12, 6))
plt.tight_layout()
plt.savefig('rank_plots.png', dpi=150)
plt.show()

# Autocorrelation
az.plot_autocorr(trace, var_names=['b_treatment', 'b_age'],
                  figsize=(12, 4))
plt.tight_layout()
plt.show()

# R-hat and ESS summary
diag = az.summary(trace, var_names=['b_treatment', 'b_age', 'b_sex',
                                      'b_baseline', 'sigma'],
                   kind='diagnostics')
print("\nConvergence Diagnostics:")
print(diag)
```

**Interpretation**: Trace plots should show well-mixed chains exploring the same region. Rank plots should show uniform histograms across chains. Autocorrelation should decay rapidly. All Rhat values should be below 1.01, and ESS (both bulk and tail) should exceed 400.

### Step 3: Posterior Analysis

```python
# Extract treatment effect posterior
treatment_posterior = trace.posterior['b_treatment'].values.flatten()

# Posterior summary
print(f"Posterior mean: {treatment_posterior.mean():.2f}")
print(f"Posterior median: {np.median(treatment_posterior):.2f}")
print(f"Posterior SD: {treatment_posterior.std():.2f}")
print(f"95% HDI: ({np.percentile(treatment_posterior, 2.5):.2f}, "
      f"{np.percentile(treatment_posterior, 97.5):.2f})")

# Direct probability statements
print(f"\nP(treatment effect > 0): {(treatment_posterior > 0).mean():.4f}")
print(f"P(treatment effect > 5): {(treatment_posterior > 5).mean():.4f}")
print(f"P(treatment effect > 10): {(treatment_posterior > 10).mean():.4f}")
print(f"P(treatment effect > 15): {(treatment_posterior > 15).mean():.4f}")

# Posterior density plot
fig, ax = plt.subplots(figsize=(10, 6))
az.plot_posterior(trace, var_names=['b_treatment'], ax=ax,
                   hdi_prob=0.95, ref_val=0)
ax.axvline(x=12, color='red', linestyle='--', linewidth=1.5, label='True effect = 12')
ax.legend(fontsize=11)
ax.set_title('Posterior Distribution: Treatment Effect on LDL Reduction', fontsize=14)
plt.tight_layout()
plt.savefig('posterior_treatment.png', dpi=150)
plt.show()
```

**Interpretation**: The direct probability statements are the hallmark of Bayesian analysis. We can report that "there is a 99.8% posterior probability that the treatment reduces LDL" and "a 75% probability that the reduction exceeds 10 mg/dL." These are directly interpretable by clinicians.

### Step 4: Bayesian Hierarchical Model

```python
center_idx = trial['center_id'].values
n_centers_data = len(np.unique(center_idx))

with pm.Model() as hierarchical_model:
    # Hyperpriors
    mu_center = pm.Normal('mu_center', mu=0, sigma=20)
    sigma_center = pm.HalfStudentT('sigma_center', nu=3, sigma=5)

    # Center random effects
    center_offset = pm.Normal('center_offset', mu=0, sigma=1,
                               shape=n_centers_data)
    center_intercept = mu_center + sigma_center * center_offset

    # Fixed effect priors
    b_treatment = pm.Normal('b_treatment', mu=0, sigma=10)
    b_age = pm.Normal('b_age', mu=0, sigma=10)
    b_sex = pm.Normal('b_sex', mu=0, sigma=10)
    b_baseline = pm.Normal('b_baseline', mu=0, sigma=10)
    sigma = pm.HalfStudentT('sigma', nu=3, sigma=10)

    # Linear predictor
    mu = (center_intercept[center_idx] +
          b_treatment * trial['treatment'].values +
          b_age * trial['age'].values +
          b_sex * trial['sex'].values +
          b_baseline * trial['baseline_ldl'].values)

    # Likelihood
    y_obs = pm.Normal('y_obs', mu=mu, sigma=sigma,
                       observed=trial['ldl_reduction'].values)

    # Sample
    trace_hier = pm.sample(3000, tune=1000, chains=4, cores=4,
                            random_seed=42, target_accept=0.95)

print(az.summary(trace_hier,
                  var_names=['mu_center', 'sigma_center', 'b_treatment',
                             'b_age', 'b_sex', 'b_baseline', 'sigma'],
                  hdi_prob=0.95))
```

**Interpretation**: The hierarchical model estimates center-specific intercepts that are partially pooled toward the grand mean (mu_center). The between-center standard deviation (sigma_center) quantifies the variability across centers. The non-centered parameterization (using center_offset) improves MCMC sampling efficiency.

### Step 5: Bayesian Logistic Regression

```python
with pm.Model() as logistic_model:
    # Priors
    intercept = pm.Normal('intercept', mu=0, sigma=5)
    b_treatment = pm.Normal('b_treatment', mu=0, sigma=2.5)
    b_age = pm.Normal('b_age', mu=0, sigma=2.5)
    b_sex = pm.Normal('b_sex', mu=0, sigma=2.5)
    b_baseline = pm.Normal('b_baseline', mu=0, sigma=2.5)

    # Linear predictor
    logit_p = (intercept +
               b_treatment * trial['treatment'].values +
               b_age * trial['age'].values +
               b_sex * trial['sex'].values +
               b_baseline * trial['baseline_ldl'].values)

    # Likelihood
    y_obs = pm.Bernoulli('y_obs', logit_p=logit_p,
                          observed=trial['meaningful_reduction'].values)

    # Sample
    trace_logit = pm.sample(3000, tune=1000, chains=4, cores=4,
                             random_seed=42, target_accept=0.9)

# Odds ratios
or_summary = az.summary(trace_logit,
                          var_names=['b_treatment', 'b_age', 'b_sex', 'b_baseline'],
                          hdi_prob=0.95)
or_summary['OR'] = np.exp(or_summary['mean'])
or_summary['OR_lower'] = np.exp(or_summary['hdi_2.5%'])
or_summary['OR_upper'] = np.exp(or_summary['hdi_97.5%'])
print("\nPosterior Odds Ratios:")
print(or_summary[['OR', 'OR_lower', 'OR_upper']].round(3))
```

**Interpretation**: The posterior odds ratios with 95% credible intervals quantify the effect of each predictor on the probability of meaningful LDL reduction. A treatment OR with a CrI entirely above 1 provides strong evidence that treatment increases the probability of meaningful reduction.

### Step 6: Prior Sensitivity Analysis

```python
results_sensitivity = {}

for prior_name, prior_sd in [('Weakly informative (SD=10)', 10),
                               ('Skeptical (SD=3)', 3),
                               ('Strong (SD=1)', 1)]:
    with pm.Model():
        intercept = pm.Normal('intercept', mu=0, sigma=20)
        b_treatment = pm.Normal('b_treatment', mu=0, sigma=prior_sd)
        b_age = pm.Normal('b_age', mu=0, sigma=10)
        b_sex = pm.Normal('b_sex', mu=0, sigma=10)
        b_baseline = pm.Normal('b_baseline', mu=0, sigma=10)
        sigma = pm.HalfStudentT('sigma', nu=3, sigma=10)

        mu = (intercept + b_treatment * trial['treatment'].values +
              b_age * trial['age'].values +
              b_sex * trial['sex'].values +
              b_baseline * trial['baseline_ldl'].values)

        y_obs = pm.Normal('y_obs', mu=mu, sigma=sigma,
                           observed=trial['ldl_reduction'].values)

        trace_sens = pm.sample(2000, tune=1000, chains=4, cores=4,
                                random_seed=42, progressbar=False)

    post = trace_sens.posterior['b_treatment'].values.flatten()
    results_sensitivity[prior_name] = {
        'mean': post.mean(),
        'sd': post.std(),
        'hdi_low': np.percentile(post, 2.5),
        'hdi_high': np.percentile(post, 97.5)
    }

sensitivity_df = pd.DataFrame(results_sensitivity).T.round(2)
print("\nPrior Sensitivity Analysis (Treatment Effect):")
print(sensitivity_df)
```

**Interpretation**: If the posterior is similar across different prior specifications, the data are informative and the conclusions are robust. If the posterior changes substantially with the prior, the data are less informative and the prior choice requires careful justification.

## Advanced Example

### Using Bambi for Formula-Based Bayesian Models

```python
import bambi as bmb

# Bambi provides an R-like formula interface for Bayesian models
model_bambi = bmb.Model(
    'ldl_reduction ~ treatment + age + sex + baseline_ldl + (1|center_id)',
    data=trial,
    family='gaussian'
)

# Set priors (optional; bambi has sensible defaults)
model_bambi.build()
print(model_bambi)

# Fit
results_bambi = model_bambi.fit(draws=3000, tune=1000, chains=4, cores=4,
                                 random_seed=42, target_accept=0.95)

# Summary
print(az.summary(results_bambi, hdi_prob=0.95))
```

**Interpretation**: Bambi simplifies Bayesian modeling in Python by providing a formula interface similar to R's `brms`. The `(1|center_id)` syntax specifies random intercepts for centers. This makes hierarchical Bayesian modeling accessible without writing the full PyMC model specification.

### Model Comparison with LOO-CV

```python
# Compute LOO for both models
with linear_model:
    pm.compute_log_likelihood(trace)
loo_linear = az.loo(trace)
print("Simple Linear Model:")
print(loo_linear)

with hierarchical_model:
    pm.compute_log_likelihood(trace_hier)
loo_hier = az.loo(trace_hier)
print("\nHierarchical Model:")
print(loo_hier)

# Compare
comparison = az.compare({'Linear': trace, 'Hierarchical': trace_hier})
print("\nModel Comparison:")
print(comparison)

# Visualization
az.plot_compare(comparison, figsize=(10, 4))
plt.title('Model Comparison (LOO-CV)')
plt.tight_layout()
plt.savefig('model_comparison.png', dpi=150)
plt.show()
```

**Interpretation**: LOO-CV comparison ranks models by their expected out-of-sample predictive accuracy (elpd_loo). Higher elpd is better. The weight column gives the relative probability that each model provides the best predictions. Pareto k diagnostics flag observations that are influential.

### Posterior Predictive Checks

```python
# Generate posterior predictive samples
with hierarchical_model:
    ppc = pm.sample_posterior_predictive(trace_hier, random_seed=42)

# Plot posterior predictive check
fig, axes = plt.subplots(1, 2, figsize=(14, 5))

# Density overlay (PyMC v4+/v5 returns InferenceData directly)
az.plot_ppc(ppc, kind='kde', num_pp_samples=50, ax=axes[0])
axes[0].set_title('Posterior Predictive Check: KDE Overlay')
axes[0].set_xlabel('LDL Reduction')

# Cumulative
az.plot_ppc(ppc, kind='cumulative', num_pp_samples=50, ax=axes[1])
axes[1].set_title('Posterior Predictive Check: Cumulative')
axes[1].set_xlabel('LDL Reduction')

plt.tight_layout()
plt.savefig('ppc.png', dpi=150)
plt.show()
```

**Interpretation**: Posterior predictive checks compare simulated data from the model to observed data. The observed data (dark line) should look like a typical replicate (light lines). Systematic departures indicate model misfit that needs to be addressed.

## Visualization

### Posterior Distribution Forest Plot

```python
fig, ax = plt.subplots(figsize=(10, 5))
az.plot_forest(trace_hier,
               var_names=['b_treatment', 'b_age', 'b_sex', 'b_baseline'],
               combined=True, hdi_prob=0.95, ax=ax)
ax.axvline(x=0, color='red', linestyle='--', linewidth=1)
ax.set_title('Posterior Distributions: Fixed Effects (95% HDI)', fontsize=14)
plt.tight_layout()
plt.savefig('forest_bayesian.png', dpi=150)
plt.show()
```

### Prior vs. Posterior Comparison

```python
fig, axes = plt.subplots(1, 3, figsize=(15, 4))

params = [('b_treatment', 'Treatment Effect', 10),
          ('b_age', 'Age Effect', 10),
          ('sigma', 'Residual SD', 10)]

for ax, (param, title, prior_sd) in zip(axes, params):
    posterior = trace.posterior[param].values.flatten()
    if param == 'sigma':
        prior = np.abs(np.random.standard_t(3, 100000)) * prior_sd
        prior = prior[prior < 40]
    else:
        prior = np.random.normal(0, prior_sd, 100000)

    ax.hist(prior, bins=80, density=True, alpha=0.4, color='gray', label='Prior')
    ax.hist(posterior, bins=80, density=True, alpha=0.6, color='steelblue', label='Posterior')
    ax.set_title(title, fontsize=12)
    ax.set_xlabel('Value')
    ax.legend(fontsize=9)
    ax.grid(True, alpha=0.3)

plt.suptitle('Prior vs. Posterior Distributions', fontsize=14, y=1.02)
plt.tight_layout()
plt.savefig('prior_vs_posterior.png', dpi=150)
plt.show()
```

### Shrinkage Plot

```python
# Compare raw center means to hierarchical (shrunk) estimates
center_raw_means = trial.groupby('center_id')['ldl_reduction'].agg(['mean', 'count'])
center_raw_means.columns = ['raw_mean', 'n']

center_offsets = trace_hier.posterior['center_offset'].mean(dim=['chain', 'draw']).values
mu_center = trace_hier.posterior['mu_center'].mean(dim=['chain', 'draw']).values
sigma_center = trace_hier.posterior['sigma_center'].mean(dim=['chain', 'draw']).values
center_shrunk = mu_center + sigma_center * center_offsets

fig, ax = plt.subplots(figsize=(8, 8))
ax.scatter(center_raw_means['raw_mean'], center_shrunk,
           s=center_raw_means['n'] * 3, alpha=0.7, color='steelblue',
           edgecolor='navy', linewidth=0.5)
lims = [min(center_raw_means['raw_mean'].min(), center_shrunk.min()) - 2,
        max(center_raw_means['raw_mean'].max(), center_shrunk.max()) + 2]
ax.plot(lims, lims, 'k--', alpha=0.5, label='No shrinkage')
ax.axhline(y=mu_center, color='red', linestyle=':', label=f'Grand mean = {mu_center:.1f}')

for i in range(len(center_raw_means)):
    ax.annotate('', xy=(center_raw_means['raw_mean'].iloc[i], center_shrunk[i]),
                xytext=(center_raw_means['raw_mean'].iloc[i],
                        center_raw_means['raw_mean'].iloc[i]),
                arrowprops=dict(arrowstyle='->', color='gray', lw=1))

ax.set_xlabel('Raw Center Mean', fontsize=12)
ax.set_ylabel('Posterior Center Mean (Shrunk)', fontsize=12)
ax.set_title('Bayesian Shrinkage of Center Effects', fontsize=14)
ax.legend(fontsize=10)
ax.grid(True, alpha=0.3)
ax.set_aspect('equal')
plt.tight_layout()
plt.savefig('shrinkage_plot.png', dpi=150)
plt.show()
```

**Interpretation**: The arrows show how each center's raw mean is pulled toward the grand mean (red line). Smaller centers (smaller dots) show more shrinkage. This partial pooling improves estimates for data-sparse centers by borrowing information from the overall distribution.

## Tips and Best Practices

1. **Always check convergence**: Examine Rhat (< 1.01), ESS (> 400 for both bulk and tail), trace plots, and rank histograms. Never interpret a model that has not converged.

2. **Use weakly informative priors**: Completely flat priors can cause sampling problems. PyMC and bambi set reasonable defaults, but always review them with `model.check_prior()` or manual inspection.

3. **Use the non-centered parameterization for hierarchical models**: Writing `center_offset ~ Normal(0, 1)` and then `center_effect = mu + sigma * center_offset` improves MCMC sampling efficiency for hierarchical models.

4. **Increase `target_accept` for divergences**: If PyMC reports divergent transitions, increase `target_accept` (e.g., 0.95 or 0.99). This makes the sampler more careful at the cost of speed.

5. **Use ArviZ for all diagnostics and plots**: ArviZ provides a unified interface for posterior analysis, diagnostics, and model comparison regardless of the inference engine.

6. **Conduct posterior predictive checks**: Use `pm.sample_posterior_predictive()` and `az.plot_ppc()` to verify that the model can reproduce the observed data patterns.

7. **Use bambi for standard models**: When fitting standard regression-type models, bambi provides a much simpler interface than raw PyMC while retaining full Bayesian inference.

8. **Report probability statements**: The key advantage of Bayesian analysis is direct probability statements. Report P(effect > clinically relevant threshold) alongside credible intervals.

9. **Save traces for reproducibility**: Use `az.to_netcdf(trace, 'model_trace.nc')` to save posterior samples for later analysis without re-running MCMC.

10. **Consider computational trade-offs**: For quick exploration, use variational inference (`pm.fit(method='advi')`). For final results, always use full MCMC sampling. Variational inference is fast but can underestimate posterior uncertainty.
