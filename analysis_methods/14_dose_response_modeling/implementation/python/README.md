# Dose-Response Modeling — Python Implementation

## Required Libraries

```bash
pip install numpy scipy pandas matplotlib seaborn statsmodels
```

```python
import numpy as np
import pandas as pd
import matplotlib.pyplot as plt
from scipy.optimize import curve_fit, minimize
from scipy.stats import t as t_dist
import warnings
warnings.filterwarnings('ignore')
```

## Example Dataset

We simulate the same Phase II anti-inflammatory dose-finding trial as the R example: six
dose groups (including placebo) with a sigmoid Emax true dose-response.

```python
np.random.seed(42)

doses = np.array([0, 25, 50, 100, 200, 400])
n_per_dose = 50
n_total = len(doses) * n_per_dose

# True model parameters
TRUE_E0 = 0
TRUE_EMAX = -15
TRUE_ED50 = 80
TRUE_HILL = 1.5

def true_sigmoid_emax(d, e0, emax, ed50, h):
    """Sigmoid Emax (Hill) model."""
    return e0 + emax * d**h / (ed50**h + d**h)

dose_vec = np.repeat(doses, n_per_dose)
true_mean = true_sigmoid_emax(dose_vec, TRUE_E0, TRUE_EMAX, TRUE_ED50, TRUE_HILL)
sigma = 8
response = true_mean + np.random.normal(0, sigma, n_total)

df = pd.DataFrame({'dose': dose_vec, 'response': response})

# Summary statistics by dose
summary = df.groupby('dose').agg(
    n=('response', 'size'),
    mean=('response', 'mean'),
    sd=('response', 'std'),
    se=('response', lambda x: x.std() / np.sqrt(len(x)))
).reset_index()

print("Dose Group Summary:")
print(summary.round(3).to_string(index=False))
```

## Complete Worked Example

### Step 1: Visualize Raw Data

```python
fig, ax = plt.subplots(figsize=(9, 6))
ax.scatter(df['dose'] + np.random.normal(0, 3, len(df)), df['response'],
           alpha=0.2, s=10, color='grey')
ax.errorbar(summary['dose'], summary['mean'], yerr=1.96 * summary['se'],
            fmt='o', color='red', markersize=8, capsize=5, linewidth=2,
            label='Group mean +/- 95% CI')
ax.set_xlabel('Dose (mg)')
ax.set_ylabel('Change in Inflammation Score')
ax.set_title('Phase II Dose-Finding: Raw Data with Group Means')
ax.legend()
plt.tight_layout()
plt.show()
```

### Step 2: Define Candidate Dose-Response Models

```python
def linear_model(d, e0, slope):
    """Linear dose-response."""
    return e0 + slope * d

def emax_model(d, e0, emax, ed50):
    """Standard Emax model (Hill = 1)."""
    return e0 + emax * d / (ed50 + d)

def sigmoid_emax_model(d, e0, emax, ed50, h):
    """Sigmoid Emax (Hill) model."""
    return e0 + emax * d**h / (ed50**h + d**h)

def log_linear_model(d, e0, slope):
    """Log-linear model."""
    return e0 + slope * np.log(d + 1)

def quadratic_model(d, e0, b1, b2):
    """Quadratic dose-response (allows non-monotone)."""
    return e0 + b1 * d + b2 * d**2

def exponential_model(d, e0, e1, delta):
    """Exponential model."""
    return e0 + e1 * (np.exp(d / delta) - 1)

# Dictionary of models with initial parameter guesses and bounds
candidate_models = {
    'Linear': {
        'func': linear_model,
        'p0': [0, -0.03],
        'bounds': ([-10, -1], [10, 0])
    },
    'Emax': {
        'func': emax_model,
        'p0': [0, -15, 100],
        'bounds': ([-10, -30, 1], [10, 0, 500])
    },
    'Sigmoid Emax': {
        'func': sigmoid_emax_model,
        'p0': [0, -15, 100, 1.5],
        'bounds': ([-10, -30, 1, 0.1], [10, 0, 500, 5])
    },
    'Log-Linear': {
        'func': log_linear_model,
        'p0': [0, -3],
        'bounds': ([-10, -10], [10, 0])
    },
    'Quadratic': {
        'func': quadratic_model,
        'p0': [0, -0.05, 0.0001],
        'bounds': ([-10, -0.5, -0.001], [10, 0.5, 0.001])
    }
}
```

### Step 3: Fit All Candidate Models

```python
results = {}
dose_data = df['dose'].values
resp_data = df['response'].values

for name, spec in candidate_models.items():
    try:
        popt, pcov = curve_fit(
            spec['func'], dose_data, resp_data,
            p0=spec['p0'], bounds=spec['bounds'],
            maxfev=10000
        )
        # Calculate residuals and AIC
        model_fn = spec['func']
        y_pred = model_fn(dose_data, *popt)
        residuals = resp_data - y_pred
        rss = np.sum(residuals**2)
        n = len(resp_data)
        k = len(popt)
        aic = n * np.log(rss / n) + 2 * k
        bic = n * np.log(rss / n) + k * np.log(n)

        results[name] = {
            'params': popt,
            'cov': pcov,
            'aic': aic,
            'bic': bic,
            'rss': rss,
            'n_params': k,
            'func': spec['func']
        }
        print(f"\n{name} model fitted successfully:")
        print(f"  Parameters: {dict(zip(spec['func'].__code__.co_varnames[1:k+1], popt.round(3)))}")
        print(f"  AIC: {aic:.1f}, BIC: {bic:.1f}")

    except RuntimeError as e:
        print(f"\n{name} model failed to converge: {e}")

# Model comparison table
comparison = pd.DataFrame({
    'Model': list(results.keys()),
    'AIC': [r['aic'] for r in results.values()],
    'BIC': [r['bic'] for r in results.values()],
    'n_params': [r['n_params'] for r in results.values()]
}).sort_values('AIC')

comparison['deltaAIC'] = comparison['AIC'] - comparison['AIC'].min()
comparison['AIC_weight'] = np.exp(-0.5 * comparison['deltaAIC'])
comparison['AIC_weight'] /= comparison['AIC_weight'].sum()

print("\n--- Model Comparison ---")
print(comparison.round(3).to_string(index=False))

# Interpretation: The model with the lowest AIC is the best fit.
# Sigmoid Emax should rank first since it matches the true data-generating process.
```

### Step 4: Optimal Model Results

```python
best_model_name = comparison.iloc[0]['Model']
best = results[best_model_name]

print(f"\n--- Best Model: {best_model_name} ---")
param_names = best['func'].__code__.co_varnames[1:best['n_params']+1]

# Parameter estimates with standard errors
se = np.sqrt(np.diag(best['cov']))
for pname, pval, pse in zip(param_names, best['params'], se):
    print(f"  {pname}: {pval:.3f} (SE: {pse:.3f})")

# Target dose estimation
if best_model_name == 'Sigmoid Emax':
    e0, emax, ed50, h = best['params']
    # ED80: dose at which response = 80% of Emax
    # Solve: emax * d^h / (ed50^h + d^h) = 0.8 * emax
    # d^h / (ed50^h + d^h) = 0.8
    # d^h = 0.8 * ed50^h / 0.2 = 4 * ed50^h
    # d = ed50 * 4^(1/h)
    ed80 = ed50 * (0.8 / 0.2) ** (1 / h)
    ed50_est = ed50
    print(f"\n  Target doses:")
    print(f"    ED50: {ed50_est:.1f} mg")
    print(f"    ED80 (80% of Emax): {ed80:.1f} mg")
    print(f"\n  True values: ED50={TRUE_ED50}, Emax={TRUE_EMAX}, Hill={TRUE_HILL}")
```

## Advanced Example

### Model Averaging

```python
# Compute model-averaged predictions
dose_grid = np.linspace(0, 400, 200)
aic_weights = comparison.set_index('Model')['AIC_weight']

averaged_pred = np.zeros(len(dose_grid))
model_preds = {}

for name, res in results.items():
    model_fn = res['func']
    pred = model_fn(dose_grid, *res['params'])
    model_preds[name] = pred
    averaged_pred += aic_weights[name] * pred

fig, ax = plt.subplots(figsize=(10, 6))
for name, pred in model_preds.items():
    ax.plot(dose_grid, pred, '--', alpha=0.5, linewidth=1,
            label=f'{name} (w={aic_weights[name]:.3f})')
ax.plot(dose_grid, averaged_pred, color='red', linewidth=2.5,
        label='Model Average')
ax.errorbar(summary['dose'], summary['mean'], yerr=1.96 * summary['se'],
            fmt='ko', capsize=4, markersize=6, label='Observed')
ax.set_xlabel('Dose (mg)')
ax.set_ylabel('Change in Inflammation Score')
ax.set_title('Dose-Response: Model Averaging with AIC Weights')
ax.legend(fontsize=8)
plt.tight_layout()
plt.show()
```

### MCP Step: Contrast Tests for Dose-Response Signal Detection

```python
def compute_optimal_contrast(model_func, params, doses, n_per_dose):
    """Compute optimal contrast for a candidate model."""
    mu = model_func(np.array(doses), *params)
    mu_centered = mu - mu.mean()
    # Standardize to unit length
    contrast = mu_centered / np.sqrt(np.sum(mu_centered**2 / n_per_dose))
    return contrast

def contrast_test(contrast, group_means, group_vars, group_ns):
    """Perform a contrast test and return test statistic and p-value."""
    test_stat = np.sum(contrast * group_means)
    var_stat = np.sum(contrast**2 * group_vars / group_ns)
    se = np.sqrt(var_stat)
    t_stat = test_stat / se
    df = np.sum(group_ns) - len(group_ns)
    p_value = t_dist.sf(t_stat, df)  # one-sided
    return t_stat, p_value

# Group-level statistics
group_means = summary['mean'].values
group_vars = summary['sd'].values**2
group_ns = summary['n'].values

# Define candidate model shapes for contrast computation
# Using guesstimates (not fitted parameters)
contrast_models = {
    'Linear': (linear_model, [0, -0.03]),
    'Emax (ED50=50)': (emax_model, [0, -12, 50]),
    'Emax (ED50=150)': (emax_model, [0, -12, 150]),
    'Sigmoid Emax': (sigmoid_emax_model, [0, -12, 80, 2]),
}

print("\n--- MCP Step: Contrast Tests ---")
test_results = []
for name, (func, params) in contrast_models.items():
    contrast = compute_optimal_contrast(func, params, doses.tolist(),
                                         n_per_dose * np.ones(len(doses)))
    t_stat, p_val = contrast_test(contrast, group_means, group_vars, group_ns)
    test_results.append({'Model': name, 'T-statistic': t_stat, 'p-value': p_val})
    print(f"  {name}: T = {t_stat:.3f}, p = {p_val:.6f}")

# Multiplicity adjustment (max-T approach: compare against max of correlated tests)
# Simplified: use Bonferroni for illustration
n_models = len(test_results)
print(f"\nBonferroni-adjusted p-values (m={n_models}):")
for res in test_results:
    adj_p = min(res['p-value'] * n_models, 1.0)
    print(f"  {res['Model']}: adjusted p = {adj_p:.6f} -> "
          f"{'SIGNIFICANT' if adj_p < 0.025 else 'Not significant'}")

# Interpretation: If any adjusted p-value < 0.025 (one-sided), a dose-response
# signal is confirmed. This provides strong control of FWER across candidate models.
```

### Bootstrap Confidence Intervals for Dose-Response Curve

```python
np.random.seed(99)
n_boot = 500
boot_preds = np.full((n_boot, len(dose_grid)), np.nan)

for b in range(n_boot):
    boot_idx = np.random.choice(len(df), len(df), replace=True)
    boot_dose = dose_data[boot_idx]
    boot_resp = resp_data[boot_idx]
    try:
        popt_b, _ = curve_fit(
            sigmoid_emax_model, boot_dose, boot_resp,
            p0=results['Sigmoid Emax']['params'],
            bounds=([-10, -30, 1, 0.1], [10, 0, 500, 5]),
            maxfev=5000
        )
        boot_preds[b] = sigmoid_emax_model(dose_grid, *popt_b)
    except RuntimeError:
        pass

ci_lower = np.nanpercentile(boot_preds, 2.5, axis=0)
ci_upper = np.nanpercentile(boot_preds, 97.5, axis=0)
pred_best = sigmoid_emax_model(dose_grid, *results['Sigmoid Emax']['params'])
```

## Visualization

```python
fig, axes = plt.subplots(2, 2, figsize=(14, 10))

# Plot 1: Best model fit with bootstrap CI
ax = axes[0, 0]
ax.fill_between(dose_grid, ci_lower, ci_upper, alpha=0.2, color='steelblue',
                label='95% Bootstrap CI')
ax.plot(dose_grid, pred_best, color='steelblue', linewidth=2, label='Sigmoid Emax fit')
ax.plot(dose_grid, true_sigmoid_emax(dose_grid, TRUE_E0, TRUE_EMAX, TRUE_ED50, TRUE_HILL),
        '--', color='green', linewidth=1.5, label='True model')
ax.errorbar(summary['dose'], summary['mean'], yerr=1.96 * summary['se'],
            fmt='ro', capsize=4, markersize=6, label='Observed')
ax.set_xlabel('Dose (mg)')
ax.set_ylabel('Response')
ax.set_title('Sigmoid Emax: Fitted vs True with 95% CI')
ax.legend(fontsize=8)

# Plot 2: Residual analysis
ax = axes[0, 1]
fitted_vals = sigmoid_emax_model(dose_data, *results['Sigmoid Emax']['params'])
resids = resp_data - fitted_vals
bp = ax.boxplot([resids[dose_data == d] for d in doses],
                labels=[str(d) for d in doses], patch_artist=True)
for patch in bp['boxes']:
    patch.set_facecolor('lightblue')
ax.axhline(y=0, linestyle='--', color='red')
ax.set_xlabel('Dose (mg)')
ax.set_ylabel('Residual')
ax.set_title('Residuals by Dose Group')

# Plot 3: AIC weights
ax = axes[1, 0]
sorted_comp = comparison.sort_values('AIC_weight', ascending=True)
ax.barh(sorted_comp['Model'], sorted_comp['AIC_weight'], color='darkorange', alpha=0.8)
for i, (_, row) in enumerate(sorted_comp.iterrows()):
    ax.text(row['AIC_weight'] + 0.01, i, f"{row['AIC_weight']:.3f}", va='center')
ax.set_xlabel('AIC Weight')
ax.set_title('Model Comparison: AIC Weights')

# Plot 4: Target dose identification
ax = axes[1, 1]
ax.plot(dose_grid, pred_best, color='steelblue', linewidth=2)
if best_model_name == 'Sigmoid Emax':
    emax_val = results['Sigmoid Emax']['params'][1]
    target_80 = 0.8 * emax_val
    ax.axhline(y=target_80, linestyle='--', color='darkgreen', alpha=0.7)
    ax.axvline(x=ed80, linestyle='--', color='darkgreen', alpha=0.7)
    ax.annotate(f'ED80 = {ed80:.0f} mg', xy=(ed80, target_80),
                xytext=(ed80 + 30, target_80 + 2),
                arrowprops=dict(arrowstyle='->', color='darkgreen'),
                fontsize=10, color='darkgreen')
ax.errorbar(summary['dose'], summary['mean'], yerr=1.96 * summary['se'],
            fmt='ro', capsize=4, markersize=6)
ax.set_xlabel('Dose (mg)')
ax.set_ylabel('Response')
ax.set_title('Target Dose Identification')

plt.tight_layout()
plt.show()
```

## Tips and Best Practices

1. **Provide good starting values**: `curve_fit` is sensitive to initial parameter guesses for
   nonlinear models. Use the observed data to inform starting values (e.g., set `e0` to the
   placebo group mean, `emax` to the difference between highest dose and placebo).

2. **Use bounded optimization**: Always set parameter bounds to prevent nonsensical estimates
   (e.g., negative ED50, negative Hill coefficient). The `bounds` argument in `curve_fit`
   enables box constraints.

3. **Check convergence**: If `curve_fit` raises `RuntimeError`, try different starting values,
   increase `maxfev`, or simplify the model. Never trust results from a model that did not
   converge.

4. **Compare models rigorously**: Use AIC (or BIC) for model selection. Do not rely solely on
   visual fit. Report delta-AIC and model weights.

5. **Model averaging is robust**: When several models fit comparably well (delta-AIC < 4),
   model averaging provides more reliable predictions than selecting a single model.

6. **Bootstrap for uncertainty**: For target dose estimation (ED50, ED80), bootstrap confidence
   intervals are more reliable than delta-method approximations, especially for highly nonlinear
   models.

7. **Simulate before fitting real data**: Test your fitting pipeline on simulated data with
   known parameters. This verifies that the algorithm can recover the true parameters and
   that your model set is appropriate.

8. **Regulatory alignment**: If the analysis will support regulatory submissions, follow the
   MCP-Mod framework. Pre-specify candidate models and their guesstimates. Use one-sided
   alpha = 0.025. Document the analysis comprehensively.
