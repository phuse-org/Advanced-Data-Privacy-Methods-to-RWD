# Adaptive Trial Design — Python Implementation

## Required Libraries

```bash
pip install numpy scipy matplotlib pandas
```

```python
import numpy as np
from scipy import stats
from scipy.optimize import brentq
import matplotlib.pyplot as plt
import pandas as pd
```

## Example Dataset

We design a group sequential trial comparing a new anticoagulant to standard heparin
for VTE prevention after knee replacement. Primary endpoint: proportion with VTE at
30 days.

```python
# Design parameters
alpha = 0.025         # one-sided
beta  = 0.20          # 80% power
p_ctrl = 0.15         # control VTE rate
p_trt  = 0.08         # expected treatment VTE rate
K = 3                 # number of analyses
info_frac = np.array([1/3, 2/3, 1.0])

# Fixed-design sample size per arm (normal approximation)
delta = p_ctrl - p_trt
p_bar = (p_ctrl + p_trt) / 2
z_alpha = stats.norm.ppf(1 - alpha)
z_beta  = stats.norm.ppf(1 - beta)
n_fixed = int(np.ceil(
    (z_alpha * np.sqrt(2 * p_bar * (1 - p_bar)) +
     z_beta * np.sqrt(p_ctrl*(1-p_ctrl) + p_trt*(1-p_trt)))**2 / delta**2
))
print(f"Fixed-design sample size per arm: {n_fixed}")
```

## Complete Worked Example

### Step 1 — Alpha Spending Functions

```python
def alpha_spend_obf(t, alpha=0.025):
    """Lan-DeMets O'Brien-Fleming alpha spending function."""
    if t <= 0:
        return 0.0
    if t >= 1:
        return alpha
    return 2 - 2 * stats.norm.cdf(stats.norm.ppf(1 - alpha / 2) / np.sqrt(t))

def alpha_spend_pocock(t, alpha=0.025):
    """Lan-DeMets Pocock alpha spending function."""
    if t <= 0:
        return 0.0
    if t >= 1:
        return alpha
    return alpha * np.log(1 + (np.e - 1) * t)

# Incremental alpha at each look
alpha_incr_obf = []
for i, tf in enumerate(info_frac):
    prev = alpha_spend_obf(info_frac[i-1]) if i > 0 else 0
    alpha_incr_obf.append(alpha_spend_obf(tf) - prev)

print("OBF incremental alpha:", [f"{a:.6f}" for a in alpha_incr_obf])
# Most alpha is reserved for the final analysis
```

### Step 2 — Boundary Computation (Recursive)

```python
def compute_boundaries_obf(info_frac, alpha=0.025, n_grid=2000):
    """
    Compute O'Brien-Fleming-type group sequential boundaries using
    numerical integration over the canonical joint distribution.
    """
    K = len(info_frac)
    boundaries = np.zeros(K)

    # At look 1: simple normal quantile
    alpha1 = alpha_spend_obf(info_frac[0], alpha)
    boundaries[0] = stats.norm.ppf(1 - alpha1)

    # For subsequent looks, use the recursive probability
    for k in range(1, K):
        alpha_k = alpha_spend_obf(info_frac[k], alpha) - \
                  alpha_spend_obf(info_frac[k-1], alpha)

        # Binary search for boundary
        def target(u_k):
            # P(Z_1 < u_1, ..., Z_{k-1} < u_{k-1}, Z_k >= u_k | H0)
            # Simplified using independent increments approximation
            # For the first interim, this is exact
            corr = np.sqrt(info_frac[k-1] / info_frac[k])
            # P(Z_k >= u_k AND Z_{k-1} < u_{k-1})
            # = P(Z_k >= u_k) - P(Z_k >= u_k AND Z_{k-1} >= u_{k-1})
            p_exceed = 1 - stats.norm.cdf(u_k)
            # Joint probability via bivariate normal
            p_joint = 1 - stats.norm.cdf(u_k) - stats.norm.cdf(boundaries[k-1]) + \
                      stats.multivariate_normal.cdf(
                          [boundaries[k-1], u_k],
                          mean=[0, 0],
                          cov=[[1, corr], [corr, 1]]
                      )
            return p_exceed - p_joint - alpha_k

        boundaries[k] = brentq(target, 0.5, 5.0)

    return boundaries

bounds_obf = compute_boundaries_obf(info_frac, alpha)
print("\nO'Brien-Fleming Boundaries (Z-scale):")
for i, (tf, b) in enumerate(zip(info_frac, bounds_obf)):
    print(f"  Look {i+1} (t={tf:.2f}): Z >= {b:.4f}")
```

### Step 3 — Simplified Boundary Approximation

```python
def obf_boundaries_approx(info_frac, alpha=0.025):
    """
    Quick O'Brien-Fleming boundary approximation:
    u_k = c / sqrt(t_k) where c = z_{alpha/2} for a two-sided test.
    For one-sided, c = z_alpha.
    """
    z_crit = stats.norm.ppf(1 - alpha)
    return z_crit / np.sqrt(info_frac)

def pocock_boundaries_approx(K, alpha=0.025):
    """
    Pocock boundaries are constant across looks.
    Approximation: use the same critical value at each look.
    The exact value requires solving a K-dimensional integral.
    """
    # Approximate: adjust for K correlated tests
    # Use Bonferroni as starting point, then refine
    z_bonf = stats.norm.ppf(1 - alpha / K)
    return np.repeat(z_bonf, K)

bounds_obf_approx = obf_boundaries_approx(info_frac)
bounds_pocock_approx = pocock_boundaries_approx(K)

comparison = pd.DataFrame({
    'Look': [1, 2, 3],
    'Info_Fraction': info_frac,
    'OBF_Boundary': bounds_obf_approx.round(4),
    'Pocock_Boundary': bounds_pocock_approx.round(4)
})
print("\nBoundary Comparison:")
print(comparison.to_string(index=False))
```

### Step 4 — Interim Analysis Decision

```python
np.random.seed(101)

# Maximum sample size (inflate by ~3% for OBF)
inflation = 1.03
max_n = int(np.ceil(n_fixed * inflation))
n_at_look = np.ceil(max_n * info_frac).astype(int)

# Simulate interim data at look 1
n_ia1 = n_at_look[0]
events_ctrl = np.random.binomial(n_ia1, p_ctrl)
events_trt  = np.random.binomial(n_ia1, p_trt)

p_hat_ctrl = events_ctrl / n_ia1
p_hat_trt  = events_trt / n_ia1
p_pool = (events_ctrl + events_trt) / (2 * n_ia1)

z_ia1 = (p_hat_ctrl - p_hat_trt) / np.sqrt(p_pool * (1 - p_pool) * 2 / n_ia1)

print(f"Interim Analysis 1 (n per arm = {n_ia1}):")
print(f"  Control event rate: {p_hat_ctrl:.3f}")
print(f"  Treatment event rate: {p_hat_trt:.3f}")
print(f"  Z-statistic: {z_ia1:.4f}")
print(f"  Efficacy boundary: {bounds_obf_approx[0]:.4f}")

if z_ia1 >= bounds_obf_approx[0]:
    print("  --> STOP for EFFICACY")
elif z_ia1 <= 0:
    print("  --> Consider STOPPING for FUTILITY (trend in wrong direction)")
else:
    print("  --> CONTINUE to next analysis")
```

### Step 5 — Conditional Power

```python
def conditional_power(z_current, info_current, info_final, boundary_final, theta=None):
    """
    Compute conditional power at the final analysis given current data.

    Parameters:
        z_current: observed Z-statistic at current look
        info_current: information fraction at current look
        info_final: information fraction at final (1.0)
        boundary_final: efficacy boundary at final analysis
        theta: assumed drift parameter (None = current trend)
    """
    if theta is None:
        theta = z_current / np.sqrt(info_current)

    # Z_final ~ N(theta*sqrt(I_final), 1)
    # Given Z_current, the remaining Z increment is independent
    drift_remaining = theta * (np.sqrt(info_final) - np.sqrt(info_current))
    var_remaining = info_final - info_current
    # Approximate: P(Z_final >= boundary | Z_current)
    z_needed = (boundary_final * np.sqrt(info_final) - z_current * np.sqrt(info_current))
    z_needed /= np.sqrt(info_final - info_current)

    cp = 1 - stats.norm.cdf(z_needed - drift_remaining / np.sqrt(var_remaining))
    return cp

cp_h1 = conditional_power(z_ia1, info_frac[0], 1.0, bounds_obf_approx[2],
                           theta=stats.norm.ppf(1 - alpha) + stats.norm.ppf(1 - beta))
cp_trend = conditional_power(z_ia1, info_frac[0], 1.0, bounds_obf_approx[2])

print(f"\nConditional power (design H1):  {cp_h1:.4f}")
print(f"Conditional power (current trend): {cp_trend:.4f}")
```

## Advanced Example

### Operating Characteristics via Simulation

```python
def simulate_trial(n_sim, p_true_ctrl, p_true_trt, max_n, info_frac, boundaries,
                   seed=42):
    """Simulate n_sim group sequential trials and report operating characteristics."""
    rng = np.random.default_rng(seed)
    n_at_look = np.ceil(max_n * info_frac).astype(int)
    K = len(info_frac)

    stopped_eff = np.zeros(K)
    stopped_fut = np.zeros(K)
    total_n_used = np.zeros(n_sim)

    for s in range(n_sim):
        y_ctrl = rng.binomial(1, p_true_ctrl, max_n)
        y_trt  = rng.binomial(1, p_true_trt, max_n)

        for look in range(K):
            ni = n_at_look[look]
            pc = y_ctrl[:ni].mean()
            pt = y_trt[:ni].mean()
            pp = (y_ctrl[:ni].sum() + y_trt[:ni].sum()) / (2 * ni)
            if pp * (1 - pp) == 0:
                z = 0
            else:
                z = (pc - pt) / np.sqrt(pp * (1 - pp) * 2 / ni)

            if z >= boundaries[look]:
                stopped_eff[look] += 1
                total_n_used[s] = 2 * ni
                break
            if look < K - 1 and z <= 0:  # simple futility: wrong direction
                stopped_fut[look] += 1
                total_n_used[s] = 2 * ni
                break
            if look == K - 1:
                total_n_used[s] = 2 * ni

    return {
        'eff_stop_prob': stopped_eff / n_sim,
        'fut_stop_prob': stopped_fut / n_sim,
        'overall_power': stopped_eff.sum() / n_sim,
        'expected_N': total_n_used.mean(),
        'max_N': 2 * max_n
    }

# Under H1
oc_h1 = simulate_trial(10000, p_ctrl, p_trt, max_n, info_frac, bounds_obf_approx)
print("Operating Characteristics under H1:")
for k, v in oc_h1.items():
    print(f"  {k}: {v}")

# Under H0
oc_h0 = simulate_trial(10000, p_ctrl, p_ctrl, max_n, info_frac, bounds_obf_approx)
print("\nOperating Characteristics under H0:")
for k, v in oc_h0.items():
    print(f"  {k}: {v}")
print(f"  Type I error: {oc_h0['overall_power']:.4f}")
```

### Power Curve Across Effect Sizes

```python
p_trt_range = np.arange(0.04, 0.16, 0.01)
power_vals = []

for pt in p_trt_range:
    oc = simulate_trial(5000, p_ctrl, pt, max_n, info_frac, bounds_obf_approx,
                        seed=int(pt * 10000))
    power_vals.append(oc['overall_power'])

power_df = pd.DataFrame({'p_treatment': p_trt_range, 'power': power_vals})
print(power_df.to_string(index=False))
```

## Visualization

```python
# 1. Boundary plot
fig, ax = plt.subplots(figsize=(8, 5))
ax.plot(info_frac, bounds_obf_approx, 'bo-', linewidth=2, markersize=8,
        label='OBF Efficacy')
ax.plot(info_frac, bounds_pocock_approx, 'rs--', linewidth=2, markersize=8,
        label='Pocock Efficacy')
ax.axhline(y=stats.norm.ppf(1 - alpha), color='grey', linestyle=':', alpha=0.5,
           label=f'Fixed design (z={stats.norm.ppf(1-alpha):.2f})')
ax.set_xlabel('Information Fraction')
ax.set_ylabel('Z-value Boundary')
ax.set_title('Group Sequential Efficacy Boundaries')
ax.legend()
ax.grid(True, alpha=0.3)
plt.tight_layout()
plt.savefig('boundaries.png', dpi=150)
plt.show()

# 2. Alpha spending functions
t_grid = np.linspace(0, 1, 200)
alpha_obf_curve = [alpha_spend_obf(t) for t in t_grid]
alpha_poc_curve = [alpha_spend_pocock(t) for t in t_grid]

fig, ax = plt.subplots(figsize=(8, 5))
ax.plot(t_grid, alpha_obf_curve, 'b-', linewidth=2, label="O'Brien-Fleming")
ax.plot(t_grid, alpha_poc_curve, 'r--', linewidth=2, label='Pocock')
ax.plot([0, 1], [0, alpha], 'g:', linewidth=1, label='Linear')
ax.set_xlabel('Information Fraction')
ax.set_ylabel('Cumulative Alpha Spent')
ax.set_title('Alpha Spending Functions')
ax.legend()
ax.grid(True, alpha=0.3)
plt.tight_layout()
plt.savefig('alpha_spending.png', dpi=150)
plt.show()

# 3. Power curve
fig, ax = plt.subplots(figsize=(8, 5))
ax.plot(p_trt_range, power_vals, 'b-o', linewidth=2)
ax.axhline(y=0.8, color='red', linestyle='--', alpha=0.7, label='80% power')
ax.axvline(x=p_trt, color='green', linestyle=':', alpha=0.7,
           label=f'Design alternative (p={p_trt})')
ax.set_xlabel('True Treatment VTE Rate')
ax.set_ylabel('Power')
ax.set_title('Power Curve for Group Sequential Design')
ax.legend()
ax.grid(True, alpha=0.3)
plt.tight_layout()
plt.savefig('power_curve.png', dpi=150)
plt.show()
```

## Tips and Best Practices

1. **Use the R packages `gsDesign` or `rpact` for production designs.** The Python
   implementations above are educational; for regulatory submissions, use validated
   software.

2. **The O'Brien-Fleming spending function** is the standard choice for confirmatory
   trials because it minimises sample-size inflation and preserves the final-analysis
   critical value near 1.96.

3. **Non-binding futility boundaries** are preferred. They do not inflate Type I error
   if the DSMB decides to continue despite crossing the futility boundary.

4. **Verify boundaries and operating characteristics by simulation.** Analytic formulas
   use the canonical joint distribution, which should be validated numerically.

5. **Account for the maximum sample size inflation** when planning logistics and budget.
   OBF inflation is typically 3%; Pocock can be 15-20%.

6. **Report conditional power at each interim** to guide DSMB decisions. Distinguish
   between conditional power under the design alternative and under the observed trend.

7. **Bias-adjust the treatment effect estimate** when stopping early for efficacy.
   Naive estimates are biased upward (the "winner's curse").

8. **Pre-specify the adaptation rules in the protocol.** Any deviation from the
   pre-specified plan requires a protocol amendment and may compromise the design's
   statistical validity.
