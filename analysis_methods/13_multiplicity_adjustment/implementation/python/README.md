# Multiplicity Adjustment — Python Implementation

## Required Libraries

```bash
pip install statsmodels scipy numpy pandas matplotlib networkx
```

```python
import numpy as np
import pandas as pd
import matplotlib.pyplot as plt
from scipy import stats
from statsmodels.stats.multitest import multipletests
```

## Example Dataset

We consider the same Phase III chronic pain trial: two doses (high and low) tested against
placebo on two endpoints (pain VAS and physical function HAQ-DI), yielding 4 hypotheses.

```python
np.random.seed(42)
n_per_arm = 150

# Simulate outcomes
placebo_pain = np.random.normal(55, 20, n_per_arm)
placebo_func = np.random.normal(1.4, 0.5, n_per_arm)

low_pain = np.random.normal(48, 20, n_per_arm)
low_func = np.random.normal(1.2, 0.5, n_per_arm)

high_pain = np.random.normal(44, 20, n_per_arm)
high_func = np.random.normal(1.05, 0.5, n_per_arm)

# Compute raw p-values (one-sided t-tests)
_, p_h1 = stats.ttest_ind(high_pain, placebo_pain)
_, p_h2 = stats.ttest_ind(high_func, placebo_func)
_, p_h3 = stats.ttest_ind(low_pain, placebo_pain)
_, p_h4 = stats.ttest_ind(low_func, placebo_func)

# Convert to one-sided (treatment is expected to reduce outcomes)
p_h1 = p_h1 / 2 if np.mean(high_pain) < np.mean(placebo_pain) else 1 - p_h1 / 2
p_h2 = p_h2 / 2 if np.mean(high_func) < np.mean(placebo_func) else 1 - p_h2 / 2
p_h3 = p_h3 / 2 if np.mean(low_pain) < np.mean(placebo_pain) else 1 - p_h3 / 2
p_h4 = p_h4 / 2 if np.mean(low_func) < np.mean(placebo_func) else 1 - p_h4 / 2

raw_pvalues = np.array([p_h1, p_h2, p_h3, p_h4])
hyp_names = ['H1: HighDose Pain', 'H2: HighDose Function',
             'H3: LowDose Pain', 'H4: LowDose Function']

print("Raw p-values:")
for name, p in zip(hyp_names, raw_pvalues):
    print(f"  {name}: {p:.6f}")
```

## Complete Worked Example

### Step 1: Standard p-Value Adjustments

```python
# statsmodels multipletests provides multiple correction methods
methods = {
    'Bonferroni': 'bonferroni',
    'Holm': 'holm',
    'Hochberg': 'simes-hochberg',
    'BH (FDR)': 'fdr_bh'
}

results = {'Hypothesis': hyp_names, 'Raw': raw_pvalues}

for display_name, method_code in methods.items():
    reject, p_adj, _, _ = multipletests(raw_pvalues, alpha=0.05, method=method_code)
    results[display_name] = p_adj

results_df = pd.DataFrame(results)
print("\n--- Adjusted p-values ---")
print(results_df.round(6).to_string(index=False))

# Decisions at alpha = 0.05
print("\nDecisions at alpha = 0.05:")
for display_name, method_code in methods.items():
    reject, _, _, _ = multipletests(raw_pvalues, alpha=0.05, method=method_code)
    rejected = [h for h, r in zip(hyp_names, reject) if r]
    print(f"  {display_name}: Reject {', '.join(rejected) if rejected else 'none'}")

# Interpretation:
# - Bonferroni is the most conservative (fewest rejections).
# - Holm is always at least as powerful as Bonferroni.
# - Hochberg may reject more than Holm (under PRDS dependence).
# - BH controls FDR (not FWER) and is the most liberal.
```

### Step 2: Graphical MCP (Manual Implementation)

```python
import networkx as nx

class GraphicalMCP:
    """Implementation of the Bretz et al. (2009) graphical testing procedure."""

    def __init__(self, hypotheses, weights, transitions, alpha=0.05):
        """
        Parameters
        ----------
        hypotheses : list of str
            Hypothesis names.
        weights : np.array
            Initial alpha allocation (sums to 1).
        transitions : np.array
            Transition matrix (m x m). transitions[i,j] is the fraction
            of alpha from H_i passed to H_j upon rejection.
        alpha : float
            Overall significance level.
        """
        self.hypotheses = list(hypotheses)
        self.weights = np.array(weights, dtype=float)
        self.transitions = np.array(transitions, dtype=float)
        self.alpha = alpha
        self.m = len(hypotheses)

    def test(self, pvalues):
        """Run the graphical testing procedure."""
        pvalues = np.array(pvalues)
        rejected = np.zeros(self.m, dtype=bool)
        w = self.weights.copy()
        G = self.transitions.copy()
        steps = []

        while True:
            # Find hypotheses that can be rejected
            testable = ~rejected
            can_reject = np.where(testable & (pvalues <= w * self.alpha))[0]

            if len(can_reject) == 0:
                break

            # Reject the most significant among testable
            # (choose smallest p / threshold ratio for determinism)
            ratios = np.full(self.m, np.inf)
            ratios[can_reject] = pvalues[can_reject] / (w[can_reject] * self.alpha)
            idx = np.argmin(ratios)

            rejected[idx] = True
            steps.append({
                'step': len(steps) + 1,
                'rejected': self.hypotheses[idx],
                'alpha_used': w[idx] * self.alpha,
                'p_value': pvalues[idx]
            })

            # Update weights and transitions (Algorithm 1 from Bretz et al.)
            freed_alpha = w[idx]
            w_new = w.copy()
            G_new = G.copy()

            for j in range(self.m):
                if not rejected[j]:
                    w_new[j] = w[j] + freed_alpha * G[idx, j]

            # Update transition matrix
            for i in range(self.m):
                if rejected[i]:
                    continue
                for j in range(self.m):
                    if rejected[j] or i == j:
                        G_new[i, j] = 0
                    else:
                        denom = 1 - G[i, idx] * G[idx, i]
                        if denom > 1e-12:
                            G_new[i, j] = (G[i, j] + G[i, idx] * G[idx, j]) / denom
                        else:
                            G_new[i, j] = 0

            w = w_new
            G = G_new
            w[rejected] = 0

        return rejected, steps

# Define the procedure for our trial
weights = np.array([0.5, 0.5, 0.0, 0.0])
transitions = np.array([
    [0, 0.5, 0.5, 0],
    [0.5, 0, 0, 0.5],
    [0, 0, 0, 1],
    [0, 0, 1, 0]
])

gmcp = GraphicalMCP(hyp_names, weights, transitions, alpha=0.05)
rejected, steps = gmcp.test(raw_pvalues)

print("\n--- Graphical MCP Results ---")
for step in steps:
    print(f"  Step {step['step']}: Reject {step['rejected']} "
          f"(p={step['p_value']:.6f} <= alpha={step['alpha_used']:.4f})")

print("\nFinal decisions:")
for name, rej in zip(hyp_names, rejected):
    print(f"  {name}: {'REJECTED' if rej else 'Not rejected'}")
```

### Step 3: Visualize the Graphical Procedure

```python
def plot_graphical_mcp(hypotheses, weights, transitions, alpha=0.05):
    """Visualize the graphical MCP procedure as a directed graph."""
    G = nx.DiGraph()
    m = len(hypotheses)
    short_names = [f'H{i+1}' for i in range(m)]

    for i in range(m):
        G.add_node(short_names[i],
                    label=f'{short_names[i]}\nw={weights[i]*alpha:.4f}')

    for i in range(m):
        for j in range(m):
            if transitions[i, j] > 0:
                G.add_edge(short_names[i], short_names[j],
                           weight=transitions[i, j])

    pos = {
        'H1': (0, 1), 'H2': (2, 1),
        'H3': (0, 0), 'H4': (2, 0)
    }

    fig, ax = plt.subplots(figsize=(8, 6))

    colors = ['#4DBEEE' if i < 2 else '#EDB120' for i in range(m)]
    nx.draw_networkx_nodes(G, pos, node_color=colors, node_size=2000, ax=ax)

    labels = {n: f'{n}\nalpha={weights[i]*alpha:.4f}' for i, n in enumerate(short_names)}
    nx.draw_networkx_labels(G, pos, labels, font_size=9, ax=ax)

    edge_labels = {}
    for i in range(m):
        for j in range(m):
            if transitions[i, j] > 0:
                edge_labels[(short_names[i], short_names[j])] = f'{transitions[i,j]:.1f}'

    nx.draw_networkx_edges(G, pos, arrows=True, arrowsize=20,
                            connectionstyle='arc3,rad=0.1', ax=ax)
    nx.draw_networkx_edge_labels(G, pos, edge_labels, font_size=8, ax=ax)

    ax.set_title('Graphical MCP: Multi-Endpoint Pain Trial\n'
                 '(Blue=High Dose, Yellow=Low Dose)', fontsize=12)
    ax.axis('off')
    plt.tight_layout()
    plt.show()

plot_graphical_mcp(hyp_names, weights, transitions, alpha=0.05)
```

### Step 4: Fixed-Sequence Testing

```python
print("\n--- Fixed-Sequence Testing ---")
print("Order: H1 -> H2 -> H3 -> H4")
alpha_level = 0.05
for i in range(4):
    reject = raw_pvalues[i] <= alpha_level
    status = "REJECT" if reject else "FAIL TO REJECT (STOP)"
    print(f"  Step {i+1}: Test {hyp_names[i]} at alpha={alpha_level:.4f}, "
          f"p={raw_pvalues[i]:.6f} -> {status}")
    if not reject:
        break
```

## Advanced Example

### Gatekeeping for Primary and Secondary Endpoints

```python
# Primary: overall survival (OS)
# Secondary: PFS, ORR, QoL
p_primary = 0.003
p_secondary = np.array([0.012, 0.048, 0.09])
sec_names = ['PFS', 'ORR', 'QoL']

print("--- Serial Gatekeeping ---")
print(f"Primary (OS): p = {p_primary:.4f} -> "
      f"{'REJECT' if p_primary <= 0.05 else 'FAIL'} at alpha = 0.05")

if p_primary <= 0.05:
    print("Gate open: testing secondary endpoints with Hochberg correction")
    reject_sec, p_adj_sec, _, _ = multipletests(p_secondary, alpha=0.05,
                                                 method='simes-hochberg')
    for name, p_raw, p_adj, rej in zip(sec_names, p_secondary, p_adj_sec, reject_sec):
        print(f"  {name}: raw p={p_raw:.4f}, adjusted p={p_adj:.4f} -> "
              f"{'REJECT' if rej else 'FAIL'}")
else:
    print("Gate closed: secondary endpoints not tested")
```

### Large-Scale FDR Control (Biomarker Screening)

```python
np.random.seed(555)
m_total = 1000
m_true = 50

# Null p-values: uniform; alternative p-values: small (Beta distribution)
p_null = np.random.uniform(0, 1, m_total - m_true)
p_alt = np.random.beta(1, 20, m_true)
all_p = np.concatenate([p_alt, p_null])
truth = np.array(['alt'] * m_true + ['null'] * (m_total - m_true))

# BH procedure
reject_bh, p_adj_bh, _, _ = multipletests(all_p, alpha=0.05, method='fdr_bh')

discoveries = reject_bh.sum()
true_discoveries = ((reject_bh) & (truth == 'alt')).sum()
false_discoveries = ((reject_bh) & (truth == 'null')).sum()
fdp = false_discoveries / max(discoveries, 1)

print(f"\n--- FDR Control ({m_total} tests, {m_true} true signals) ---")
print(f"Total discoveries: {discoveries}")
print(f"True discoveries: {true_discoveries}")
print(f"False discoveries: {false_discoveries}")
print(f"Observed FDP: {fdp:.3f} (target FDR: 0.05)")
print(f"Sensitivity (power): {true_discoveries/m_true:.3f}")

# Compare with Bonferroni
reject_bonf, _, _, _ = multipletests(all_p, alpha=0.05, method='bonferroni')
print(f"\nBonferroni discoveries: {reject_bonf.sum()} "
      f"(true: {((reject_bonf) & (truth == 'alt')).sum()})")
# Interpretation: BH discovers many more true signals while controlling FDP.
# Bonferroni is much more conservative, missing many true associations.
```

## Visualization

```python
fig, axes = plt.subplots(2, 2, figsize=(14, 10))

# Plot 1: Comparison of adjusted p-values across methods
ax = axes[0, 0]
method_names = list(methods.keys())
x = np.arange(len(hyp_names))
width = 0.18
for i, (display_name, method_code) in enumerate(methods.items()):
    _, p_adj, _, _ = multipletests(raw_pvalues, alpha=0.05, method=method_code)
    ax.bar(x + i * width, p_adj, width, label=display_name, alpha=0.8)
ax.axhline(y=0.05, linestyle='--', color='red', label='alpha=0.05')
ax.set_xticks(x + 1.5 * width)
ax.set_xticklabels(['H1', 'H2', 'H3', 'H4'])
ax.set_ylabel('Adjusted p-value')
ax.set_title('Adjusted p-values by Method')
ax.legend(fontsize=8)

# Plot 2: Power comparison across number of tests
ax = axes[0, 1]
m_range = np.arange(1, 21)
bonf_threshold = 0.05 / m_range
holm_first_threshold = 0.05 / m_range  # most conservative step
bh_first_threshold = 0.05 * 1 / m_range
ax.plot(m_range, bonf_threshold, 'o-', label='Bonferroni', markersize=4)
ax.plot(m_range, bh_first_threshold, 's-', label='BH (1st step)', markersize=4)
ax.axhline(y=0.05, linestyle='--', color='grey', alpha=0.5, label='Nominal alpha')
ax.set_xlabel('Number of Tests (m)')
ax.set_ylabel('Threshold for Smallest p-value')
ax.set_title('How Thresholds Shrink with More Tests')
ax.legend(fontsize=8)

# Plot 3: BH procedure visualization for biomarker screening
ax = axes[1, 0]
sorted_p = np.sort(all_p)
m_total_val = len(sorted_p)
bh_line = 0.05 * np.arange(1, m_total_val + 1) / m_total_val
ax.plot(range(1, 101), sorted_p[:100], 'o', markersize=2, color='steelblue',
        label='Sorted p-values')
ax.plot(range(1, 101), bh_line[:100], '--', color='red', label='BH threshold')
ax.set_xlabel('Rank')
ax.set_ylabel('p-value')
ax.set_title('BH Procedure: Sorted p-values vs Threshold (first 100)')
ax.legend()

# Plot 4: Manhattan-style plot for biomarker screening
ax = axes[1, 1]
neg_log_p = -np.log10(all_p)
colors = np.where(reject_bh, 'red', 'grey')
ax.scatter(range(m_total), neg_log_p, c=colors, s=5, alpha=0.5)
ax.axhline(y=-np.log10(0.05), linestyle='--', color='grey', alpha=0.7,
           label='Unadjusted alpha')
ax.axhline(y=-np.log10(0.05 / m_total), linestyle='--', color='blue', alpha=0.7,
           label='Bonferroni threshold')
ax.set_xlabel('Biomarker Index')
ax.set_ylabel('-log10(p)')
ax.set_title('Biomarker Screening: Red = BH Significant')
ax.legend(fontsize=8)

plt.tight_layout()
plt.show()
```

## Tips and Best Practices

1. **Use `multipletests` from statsmodels**: This is the standard Python function for p-value
   adjustment. It supports Bonferroni, Holm, Hochberg, BH, BY, and other methods.

2. **Validate against known results**: When implementing graphical MCP manually, verify against
   the R `gMCP` package on the same dataset. The graphical algorithm has subtle edge cases
   in transition matrix updates.

3. **FDR is not FWER**: The BH procedure controls FDR, not FWER. Do not use BH for the
   primary analysis of a confirmatory clinical trial. Regulators expect FWER control.

4. **Interpret adjusted p-values correctly**: An adjusted p-value is the smallest FWER (or
   FDR) level at which the hypothesis would be rejected. It is NOT the probability that the
   hypothesis is true.

5. **Consider the dependence structure**: If your test statistics are positively correlated
   (common in clinical trials with shared subjects), Hochberg and BH are valid and more
   powerful than Bonferroni/Holm.

6. **Document and visualize**: Always produce a graph diagram of the testing procedure. Use
   networkx or a similar library to create publication-quality figures.

7. **Alpha allocation is a design decision**: How you split alpha across hypotheses reflects
   clinical priorities. This should be discussed with the clinical team and pre-specified.

8. **For regulatory submissions**: Follow ICH E9 guidance on multiplicity. The testing procedure
   must control strong FWER at the one-sided 0.025 or two-sided 0.05 level.
