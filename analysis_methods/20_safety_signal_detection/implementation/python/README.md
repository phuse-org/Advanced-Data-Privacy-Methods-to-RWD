# Safety Signal Detection — Python Implementation

## Required Libraries

```bash
pip install numpy pandas scipy matplotlib seaborn
```

```python
import numpy as np
import pandas as pd
from scipy import stats
from scipy.special import gammaln
import matplotlib.pyplot as plt
import seaborn as sns
```

## Example Dataset

Simulated FAERS-like spontaneous reporting database: 50,000 reports across 100 drugs
and 200 adverse events, with three embedded known signals.

```python
np.random.seed(2024)
n_reports = 50000
n_drugs   = 100
n_events  = 200

drug_probs = np.concatenate([np.repeat(3.0, 10), np.repeat(1.0, 90)])
drug_probs /= drug_probs.sum()
event_probs = np.concatenate([np.repeat(2.0, 20), np.repeat(1.0, 180)])
event_probs /= event_probs.sum()

drug_names  = [f"Drug_{i:03d}" for i in range(1, n_drugs + 1)]
event_names = [f"AE_{i:03d}" for i in range(1, n_events + 1)]

drug_ids  = np.random.choice(drug_names, n_reports, p=drug_probs)
event_ids = np.random.choice(event_names, n_reports, p=event_probs)

# Inject known signals
rng = np.random.default_rng(2024)
idx_s1 = rng.choice(n_reports, 150, replace=False)
drug_ids[idx_s1]  = "Drug_001"
event_ids[idx_s1] = "AE_001"

remaining = np.setdiff1d(np.arange(n_reports), idx_s1)
idx_s2 = rng.choice(remaining, 80, replace=False)
drug_ids[idx_s2]  = "Drug_005"
event_ids[idx_s2] = "AE_010"

remaining2 = np.setdiff1d(remaining, idx_s2)
idx_s3 = rng.choice(remaining2, 40, replace=False)
drug_ids[idx_s3]  = "Drug_010"
event_ids[idx_s3] = "AE_020"

reports = pd.DataFrame({'drug': drug_ids, 'event': event_ids})
print(f"Total reports: {len(reports)}")
print(f"Unique drugs:  {reports['drug'].nunique()}")
print(f"Unique events: {reports['event'].nunique()}")
```

## Complete Worked Example

### Step 1 — Build 2x2 Table and Compute PRR/ROR

```python
def compute_disproportionality(reports, target_drug, target_event):
    """Compute PRR, ROR, and chi-squared for a drug-event pair."""
    is_drug  = reports['drug'] == target_drug
    is_event = reports['event'] == target_event
    N = len(reports)

    a = int((is_drug & is_event).sum())
    b = int((is_drug & ~is_event).sum())
    c = int((~is_drug & is_event).sum())
    d = int((~is_drug & ~is_event).sum())

    # Expected count under independence
    E = (a + b) * (a + c) / N

    # PRR
    prr = (a / (a + b)) / (c / (c + d)) if (a + b) > 0 and (c + d) > 0 else np.nan
    se_ln_prr = np.sqrt(1/max(a, 0.5) - 1/(a+b) + 1/max(c, 0.5) - 1/(c+d))
    prr_ci = np.exp(np.log(max(prr, 1e-10)) + np.array([-1, 1]) * 1.96 * se_ln_prr)

    # Chi-squared
    chi2 = ((a*d - b*c)**2 * N) / ((a+b) * (c+d) * (a+c) * (b+d)) \
           if all(v > 0 for v in [(a+b), (c+d), (a+c), (b+d)]) else 0

    # ROR
    ror = (a * d) / (b * c) if b > 0 and c > 0 else np.nan
    se_ln_ror = np.sqrt(1/max(a,0.5) + 1/max(b,0.5) + 1/max(c,0.5) + 1/max(d,0.5))
    ror_ci = np.exp(np.log(max(ror, 1e-10)) + np.array([-1, 1]) * 1.96 * se_ln_ror)

    return {
        'drug': target_drug, 'event': target_event,
        'a': a, 'E': round(E, 2),
        'PRR': round(prr, 3), 'PRR_lower': round(prr_ci[0], 3),
        'PRR_upper': round(prr_ci[1], 3), 'chi2': round(chi2, 2),
        'ROR': round(ror, 3), 'ROR_lower': round(ror_ci[0], 3),
        'ROR_upper': round(ror_ci[1], 3),
        'signal_PRR': prr > 2 and chi2 > 4 and a >= 3,
        'signal_ROR': ror_ci[0] > 1 if not np.isnan(ror) else False
    }

# Evaluate known signals and a non-signal
pairs = [
    ("Drug_001", "AE_001"),   # strong signal
    ("Drug_005", "AE_010"),   # moderate signal
    ("Drug_010", "AE_020"),   # weak signal
    ("Drug_050", "AE_100"),   # expected non-signal
]

results = pd.DataFrame([compute_disproportionality(reports, d, e) for d, e in pairs])
print("Disproportionality Analysis Results:")
print(results.to_string(index=False))
```

### Step 2 — Information Component (BCPNN-style)

```python
def compute_ic_all(reports, alpha_prior=0.5):
    """Compute IC for all drug-event pairs in the database."""
    # Build contingency table
    contingency = pd.crosstab(reports['drug'], reports['event'])
    N = contingency.values.sum()

    # Marginals
    n_drug  = contingency.sum(axis=1)
    n_event = contingency.sum(axis=0)

    results = []
    for drug in contingency.index:
        for event in contingency.columns:
            observed = contingency.loc[drug, event]
            expected = n_drug[drug] * n_event[event] / N

            # Shrunk IC (gamma-Poisson posterior)
            ic = np.log2((observed + alpha_prior) / (expected + alpha_prior))

            # Approximate variance and lower credible interval
            ic_var = 1 / (observed + alpha_prior)
            ic025 = ic - 1.96 * np.sqrt(ic_var)

            results.append({
                'drug': drug, 'event': event,
                'observed': observed, 'expected': round(expected, 2),
                'IC': round(ic, 3), 'IC025': round(ic025, 3),
                'signal_IC': ic025 > 0
            })

    return pd.DataFrame(results)

ic_results = compute_ic_all(reports)

# Show known signals
known = ic_results[
    ((ic_results['drug'] == 'Drug_001') & (ic_results['event'] == 'AE_001')) |
    ((ic_results['drug'] == 'Drug_005') & (ic_results['event'] == 'AE_010')) |
    ((ic_results['drug'] == 'Drug_010') & (ic_results['event'] == 'AE_020'))
].sort_values('IC', ascending=False)

print("\nIC Results for Known Signals:")
print(known.to_string(index=False))

# Top signals across database
top_signals = ic_results[ic_results['observed'] >= 3].nlargest(15, 'IC')
print("\nTop 15 Signals by IC:")
print(top_signals[['drug', 'event', 'observed', 'expected', 'IC', 'IC025',
                    'signal_IC']].to_string(index=False))
```

### Step 3 — Simplified EBGM (Empirical Bayes Geometric Mean)

```python
def compute_ebgm(reports, prior_alpha1=0.2, prior_beta1=0.1,
                 prior_alpha2=2.0, prior_beta2=4.0, prior_w=0.1):
    """
    Simplified EBGM using a two-component gamma mixture prior.
    Full MGPS would fit the mixture via EM; here we use fixed priors.
    """
    contingency = pd.crosstab(reports['drug'], reports['event'])
    N = contingency.values.sum()
    n_drug  = contingency.sum(axis=1)
    n_event = contingency.sum(axis=0)

    results = []
    for drug in contingency.index:
        for event in contingency.columns:
            obs = contingency.loc[drug, event]
            exp = n_drug[drug] * n_event[event] / N
            if exp == 0:
                continue

            # Posterior weights for two-component mixture
            def log_marginal(alpha, beta):
                return (gammaln(alpha + obs) - gammaln(alpha) +
                        alpha * np.log(beta) -
                        (alpha + obs) * np.log(beta + exp))

            lm1 = log_marginal(prior_alpha1, prior_beta1)
            lm2 = log_marginal(prior_alpha2, prior_beta2)

            log_w1 = np.log(prior_w) + lm1
            log_w2 = np.log(1 - prior_w) + lm2
            max_lw = max(log_w1, log_w2)
            w1_post = np.exp(log_w1 - max_lw) / (np.exp(log_w1 - max_lw) +
                                                   np.exp(log_w2 - max_lw))

            # Posterior mean (EBGM) from each component
            ebgm1 = (prior_alpha1 + obs) / (prior_beta1 + exp)
            ebgm2 = (prior_alpha2 + obs) / (prior_beta2 + exp)
            ebgm = w1_post * ebgm1 + (1 - w1_post) * ebgm2

            # EB05 (approximate 5th percentile — simplified)
            # Use the dominant component's gamma distribution
            if w1_post > 0.5:
                eb05 = stats.gamma.ppf(0.05, prior_alpha1 + obs,
                                       scale=1/(prior_beta1 + exp))
            else:
                eb05 = stats.gamma.ppf(0.05, prior_alpha2 + obs,
                                       scale=1/(prior_beta2 + exp))

            results.append({
                'drug': drug, 'event': event,
                'observed': obs, 'expected': round(exp, 2),
                'EBGM': round(ebgm, 3), 'EB05': round(eb05, 3),
                'signal_EBGM': eb05 >= 2
            })

    return pd.DataFrame(results)

ebgm_results = compute_ebgm(reports)
top_ebgm = ebgm_results[ebgm_results['observed'] >= 3].nlargest(15, 'EBGM')
print("Top 15 Signals by EBGM:")
print(top_ebgm.to_string(index=False))
```

### Step 4 — Hy's Law / eDISH Plot

```python
np.random.seed(456)
n_patients = 300
treatment = np.repeat(['Active', 'Placebo'], n_patients // 2)

# ALT (xULN)
alt_uln = np.where(
    treatment == 'Active',
    np.random.lognormal(0, 0.6, n_patients // 2),
    np.random.lognormal(-0.1, 0.4, n_patients // 2)
)

# Bilirubin (xULN)
bili_uln = np.where(
    treatment == 'Active',
    np.random.lognormal(-0.3, 0.5, n_patients // 2),
    np.random.lognormal(-0.4, 0.3, n_patients // 2)
)

# Inject Hy's Law cases
hy_idx = np.random.choice(np.where(treatment == 'Active')[0], 4, replace=False)
alt_uln[hy_idx]  = np.random.uniform(3.5, 12, 4)
bili_uln[hy_idx] = np.random.uniform(2.5, 6, 4)

liver = pd.DataFrame({
    'treatment': treatment, 'alt_uln': alt_uln, 'bili_uln': bili_uln
})

# eDISH plot
fig, ax = plt.subplots(figsize=(9, 7))
colors = {'Active': 'firebrick', 'Placebo': 'steelblue'}

for trt in ['Placebo', 'Active']:
    subset = liver[liver['treatment'] == trt]
    ax.scatter(subset['alt_uln'], subset['bili_uln'],
               alpha=0.5, s=30, color=colors[trt], label=trt)

# Reference lines
ax.axvline(x=3, color='darkred', linestyle='--', linewidth=1.5, alpha=0.7)
ax.axhline(y=2, color='darkred', linestyle='--', linewidth=1.5, alpha=0.7)

# Quadrant labels
ax.text(8, 4, "Hy's Law\nQuadrant", fontsize=10, fontweight='bold',
        color='darkred', ha='center')
ax.text(8, 0.6, "Temple's\nCorollary", fontsize=9, color='grey', ha='center')
ax.text(0.5, 4, "Cholestatic", fontsize=9, color='grey', ha='center')
ax.text(0.5, 0.6, "Normal", fontsize=9, color='grey', ha='center')

ax.set_xscale('log')
ax.set_yscale('log')
ax.set_xlim(0.3, 20)
ax.set_ylim(0.3, 10)
ax.set_xlabel('Peak ALT (xULN)', fontsize=12)
ax.set_ylabel('Peak Bilirubin (xULN)', fontsize=12)
ax.set_title('eDISH Plot: Drug-Induced Liver Injury Assessment', fontsize=13)
ax.legend()
plt.tight_layout()
plt.savefig('edish_plot.png', dpi=150)
plt.show()

# Count Hy's Law cases
hys = liver[(liver['alt_uln'] >= 3) & (liver['bili_uln'] >= 2)]
print("\nHy's Law cases by treatment:")
print(hys['treatment'].value_counts())
```

## Advanced Example

### Database-Wide Signal Screening

```python
def screen_all_signals(reports, min_count=3):
    """Run PRR, ROR, and IC on all drug-event pairs with sufficient counts."""
    contingency = pd.crosstab(reports['drug'], reports['event'])
    N = contingency.values.sum()
    n_drug = contingency.sum(axis=1)
    n_event = contingency.sum(axis=0)

    signals = []
    for drug in contingency.index:
        for event in contingency.columns:
            a = contingency.loc[drug, event]
            if a < min_count:
                continue
            b = n_drug[drug] - a
            c = n_event[event] - a
            d = N - a - b - c

            E = n_drug[drug] * n_event[event] / N
            prr = (a / (a + b)) / (c / (c + d)) if (c + d) > 0 else np.nan
            ror = (a * d) / (b * c) if b > 0 and c > 0 else np.nan

            ic = np.log2((a + 0.5) / (E + 0.5))
            ic025 = ic - 1.96 / np.sqrt(a + 0.5)

            signals.append({
                'drug': drug, 'event': event, 'count': a, 'expected': round(E, 2),
                'PRR': round(prr, 2) if not np.isnan(prr) else None,
                'ROR': round(ror, 2) if not np.isnan(ror) else None,
                'IC': round(ic, 2), 'IC025': round(ic025, 2)
            })

    df = pd.DataFrame(signals)
    # Flag concordant signals (at least 2 of 3 methods agree)
    df['flag_PRR'] = df['PRR'].apply(lambda x: x is not None and x > 2)
    df['flag_ROR'] = df['ROR'].apply(lambda x: x is not None and x > 2)
    df['flag_IC']  = df['IC025'] > 0
    df['n_flags']  = df[['flag_PRR', 'flag_ROR', 'flag_IC']].sum(axis=1)
    return df.sort_values('n_flags', ascending=False)

all_signals = screen_all_signals(reports)
concordant = all_signals[all_signals['n_flags'] >= 2]
print(f"Total pairs screened: {len(all_signals)}")
print(f"Concordant signals (>=2 methods): {len(concordant)}")
print(concordant.head(20).to_string(index=False))
```

## Visualization

```python
# 1. Volcano plot
fig, ax = plt.subplots(figsize=(10, 6))
screened = all_signals[all_signals['count'] >= 3].copy()
screened['neg_log_p'] = -np.log10(stats.norm.sf(screened['IC'] / (1/np.sqrt(screened['count']+0.5))))

ax.scatter(screened['IC'], screened['neg_log_p'], alpha=0.2, s=10, color='grey')

# Highlight known signals
for drug, event, color in [('Drug_001', 'AE_001', 'red'),
                            ('Drug_005', 'AE_010', 'orange'),
                            ('Drug_010', 'AE_020', 'green')]:
    row = screened[(screened['drug'] == drug) & (screened['event'] == event)]
    if len(row) > 0:
        ax.scatter(row['IC'], row['neg_log_p'], color=color, s=100, zorder=5,
                   edgecolors='black', label=f"{drug}-{event}")

ax.axvline(x=0, linestyle='--', color='black', alpha=0.3)
ax.set_xlabel('Information Component (IC)')
ax.set_ylabel('-log10(p-value)')
ax.set_title('Volcano Plot: Signal Detection Across All Drug-Event Pairs')
ax.legend()
plt.tight_layout()
plt.savefig('volcano_plot.png', dpi=150)
plt.show()

# 2. PRR vs ROR scatter
fig, ax = plt.subplots(figsize=(7, 7))
valid = all_signals.dropna(subset=['PRR', 'ROR'])
valid = valid[(valid['PRR'] < 50) & (valid['ROR'] < 50)]
ax.scatter(valid['PRR'], valid['ROR'], alpha=0.3, s=15, color='steelblue')
ax.plot([0, 50], [0, 50], 'r--', alpha=0.5, label='PRR = ROR')
ax.axhline(y=2, color='grey', linestyle=':', alpha=0.5)
ax.axvline(x=2, color='grey', linestyle=':', alpha=0.5)
ax.set_xlabel('PRR')
ax.set_ylabel('ROR')
ax.set_title('PRR vs ROR Comparison')
ax.legend()
plt.tight_layout()
plt.savefig('prr_vs_ror.png', dpi=150)
plt.show()

# 3. Heatmap of top signals
top = all_signals.nlargest(40, 'IC')
heat = top.pivot_table(index='drug', columns='event', values='IC', fill_value=0)

fig, ax = plt.subplots(figsize=(14, 8))
sns.heatmap(heat, cmap='RdBu_r', center=0, annot=True, fmt='.1f',
            ax=ax, linewidths=0.5)
ax.set_title('IC Heatmap: Top Drug-Event Pairs')
plt.tight_layout()
plt.savefig('ic_heatmap.png', dpi=150)
plt.show()
```

## Tips and Best Practices

1. **Use multiple methods and look for concordance.** Flag drug-event pairs where PRR,
   ROR, and IC all exceed their thresholds. Discordant signals warrant scrutiny.

2. **Apply Bayesian shrinkage (EBGM, BCPNN)** for sparse data. Raw PRR/ROR are
   unstable when cell counts are small; shrinkage toward the null reduces false
   positives.

3. **Disproportionality is not risk estimation.** PRR and ROR do not estimate relative
   risk because spontaneous reports lack a denominator. They indicate reporting
   imbalance, not causation.

4. **Always generate the eDISH plot** for liver safety in clinical programs. FDA
   expects it in NDA/BLA submissions. Place it in the clinical overview and ISS.

5. **Use negative control outcomes** (events known not to be caused by the drug) to
   calibrate your signal detection method and estimate the false-positive rate.

6. **Account for masking (competition bias).** A very commonly reported drug can
   suppress signals for other drugs sharing the same event. Stratified or restricted
   analyses can help.

7. **Implement the full signal management workflow:** detection, validation,
   prioritisation, evaluation, and action. Document each step for regulatory inspection.

8. **For sequential surveillance (e.g., vaccine safety),** use MaxSPRT or conditional
   sequential sampling to control the Type I error across multiple calendar-time looks.
