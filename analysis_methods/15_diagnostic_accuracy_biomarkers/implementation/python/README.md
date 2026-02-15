# Diagnostic Test Accuracy and Biomarker Evaluation — Python Implementation

## Required Libraries

```bash
pip install scikit-learn numpy pandas matplotlib scipy seaborn
```

```python
import numpy as np
import pandas as pd
import matplotlib.pyplot as plt
from sklearn.metrics import (roc_curve, auc, confusion_matrix,
                              classification_report, roc_auc_score)
from scipy import stats
```

## Example Dataset

We simulate an emergency department cohort evaluating a novel cardiac biomarker (hs-cTnX)
for diagnosing acute myocardial infarction (AMI) among chest pain patients.

```python
np.random.seed(42)
n = 800
prevalence = 0.25

# Disease status
ami = np.random.binomial(1, prevalence, n)

# Biomarker: log-normal distribution, higher in AMI patients
biomarker = np.where(
    ami == 1,
    np.exp(np.random.normal(3.5, 0.8, n)),   # AMI: median ~33 ng/L
    np.exp(np.random.normal(2.0, 0.9, n))    # No AMI: median ~7 ng/L
)

# Clinical covariates
age = np.random.normal(np.where(ami == 1, 68, 58), 12)
male = np.random.binomial(1, np.where(ami == 1, 0.65, 0.50))

df = pd.DataFrame({
    'ami': ami, 'biomarker': biomarker, 'age': age, 'male': male
})

print(f"Dataset: {n} patients, {ami.sum()} AMI cases ({100*ami.mean():.1f}% prevalence)")
print(f"Biomarker median - AMI: {np.median(biomarker[ami==1]):.1f}, "
      f"No AMI: {np.median(biomarker[ami==0]):.1f} ng/L")
```

## Complete Worked Example

### Step 1: ROC Curve Analysis

```python
# Compute ROC curve
fpr, tpr, thresholds = roc_curve(df['ami'], df['biomarker'])
roc_auc = auc(fpr, tpr)

print(f"\n--- ROC Analysis ---")
print(f"AUC: {roc_auc:.3f}")

# Bootstrap confidence interval for AUC
n_boot = 2000
boot_aucs = []
for _ in range(n_boot):
    idx = np.random.choice(n, n, replace=True)
    if len(np.unique(df['ami'].values[idx])) < 2:
        continue
    boot_auc = roc_auc_score(df['ami'].values[idx], df['biomarker'].values[idx])
    boot_aucs.append(boot_auc)

ci_lower = np.percentile(boot_aucs, 2.5)
ci_upper = np.percentile(boot_aucs, 97.5)
print(f"95% Bootstrap CI: [{ci_lower:.3f}, {ci_upper:.3f}]")

# Interpretation: AUC of ~0.85 indicates good discrimination.
# The biomarker can distinguish AMI from non-AMI patients well.
```

### Step 2: Plot ROC Curve

```python
fig, ax = plt.subplots(figsize=(7, 7))
ax.plot(fpr, tpr, color='steelblue', linewidth=2,
        label=f'hs-cTnX (AUC = {roc_auc:.3f})')
ax.plot([0, 1], [0, 1], '--', color='grey', alpha=0.7, label='Random classifier')
ax.fill_between(fpr, tpr, alpha=0.1, color='steelblue')
ax.set_xlabel('1 - Specificity (False Positive Rate)', fontsize=12)
ax.set_ylabel('Sensitivity (True Positive Rate)', fontsize=12)
ax.set_title('ROC Curve: hs-cTnX for AMI Diagnosis', fontsize=14)
ax.legend(fontsize=11)
ax.set_aspect('equal')
plt.tight_layout()
plt.show()
```

### Step 3: Optimal Cutpoint Selection

```python
def youden_index(fpr, tpr, thresholds):
    """Find optimal cutpoint using Youden's J statistic."""
    j_scores = tpr - fpr  # equivalent to Se + Sp - 1
    best_idx = np.argmax(j_scores)
    return {
        'threshold': thresholds[best_idx],
        'sensitivity': tpr[best_idx],
        'specificity': 1 - fpr[best_idx],
        'youden_j': j_scores[best_idx]
    }

def closest_topleft(fpr, tpr, thresholds):
    """Find cutpoint closest to the (0, 1) corner."""
    distances = np.sqrt(fpr**2 + (1 - tpr)**2)
    best_idx = np.argmin(distances)
    return {
        'threshold': thresholds[best_idx],
        'sensitivity': tpr[best_idx],
        'specificity': 1 - fpr[best_idx],
        'distance': distances[best_idx]
    }

# Youden method
youden = youden_index(fpr, tpr, thresholds)
print(f"\n--- Optimal Cutpoint (Youden Index) ---")
print(f"Threshold: {youden['threshold']:.2f} ng/L")
print(f"Sensitivity: {youden['sensitivity']:.3f}")
print(f"Specificity: {youden['specificity']:.3f}")
print(f"Youden J: {youden['youden_j']:.3f}")

# Closest to (0,1)
closest = closest_topleft(fpr, tpr, thresholds)
print(f"\n--- Optimal Cutpoint (Closest to Top-Left) ---")
print(f"Threshold: {closest['threshold']:.2f} ng/L")
print(f"Sensitivity: {closest['sensitivity']:.3f}")
print(f"Specificity: {closest['specificity']:.3f}")
```

### Step 4: Full Performance Metrics at Optimal Cutpoint

```python
optimal_cut = youden['threshold']
predicted = (df['biomarker'] >= optimal_cut).astype(int)

# Confusion matrix
tn, fp, fn, tp = confusion_matrix(df['ami'], predicted).ravel()

sensitivity = tp / (tp + fn)
specificity = tn / (tn + fp)
ppv = tp / (tp + fp)
npv = tn / (tn + fn)
lr_pos = sensitivity / (1 - specificity) if specificity < 1 else np.inf
lr_neg = (1 - sensitivity) / specificity if specificity > 0 else np.inf
accuracy = (tp + tn) / n

print(f"\n--- Performance at Cutpoint = {optimal_cut:.2f} ng/L ---")
print(f"Sensitivity: {sensitivity:.3f}")
print(f"Specificity: {specificity:.3f}")
print(f"PPV: {ppv:.3f}")
print(f"NPV: {npv:.3f}")
print(f"LR+: {lr_pos:.2f}")
print(f"LR-: {lr_neg:.3f}")
print(f"Accuracy: {accuracy:.3f}")

print(f"\nConfusion Matrix:")
print(f"             Predicted +  Predicted -")
print(f"  Disease +     {tp:>5}        {fn:>5}")
print(f"  Disease -     {fp:>5}        {tn:>5}")

print(f"\nNote: PPV={ppv:.3f} at prevalence={prevalence:.0%}. "
      f"In a population with 5% prevalence, PPV would be "
      f"{(sensitivity*0.05)/(sensitivity*0.05+(1-specificity)*0.95):.3f}")
```

### Step 5: Decision Curve Analysis (Custom Implementation)

```python
def decision_curve_analysis(y_true, y_prob, thresholds):
    """
    Compute net benefit for a prediction model across threshold probabilities.

    Net Benefit = TP/n - FP/n * (p_t / (1 - p_t))

    Parameters
    ----------
    y_true : array-like, binary outcome (0/1)
    y_prob : array-like, predicted probabilities
    thresholds : array-like, threshold probabilities to evaluate

    Returns
    -------
    DataFrame with net benefit for the model, treat-all, and treat-none.
    """
    n = len(y_true)
    results = []

    for pt in thresholds:
        # Model
        pred_pos = (y_prob >= pt).astype(int)
        tp = np.sum((pred_pos == 1) & (y_true == 1))
        fp = np.sum((pred_pos == 1) & (y_true == 0))
        nb_model = tp / n - fp / n * (pt / (1 - pt))

        # Treat all
        nb_all = np.mean(y_true) - (1 - np.mean(y_true)) * (pt / (1 - pt))

        # Treat none
        nb_none = 0

        results.append({
            'threshold': pt,
            'model': nb_model,
            'treat_all': nb_all,
            'treat_none': nb_none
        })

    return pd.DataFrame(results)

# Build a logistic regression model
from sklearn.linear_model import LogisticRegression

# Model 1: Biomarker only
X_bio = np.log(df['biomarker']).values.reshape(-1, 1)
model_bio = LogisticRegression(random_state=42).fit(X_bio, df['ami'])
prob_bio = model_bio.predict_proba(X_bio)[:, 1]

# Model 2: Biomarker + clinical variables
X_full = df[['biomarker', 'age', 'male']].copy()
X_full['biomarker'] = np.log(X_full['biomarker'])
model_full = LogisticRegression(random_state=42).fit(X_full, df['ami'])
prob_full = model_full.predict_proba(X_full)[:, 1]

# Compute DCA
thresholds = np.arange(0.01, 0.60, 0.01)
dca_bio = decision_curve_analysis(df['ami'].values, prob_bio, thresholds)
dca_full = decision_curve_analysis(df['ami'].values, prob_full, thresholds)

# Plot DCA
fig, ax = plt.subplots(figsize=(9, 6))
ax.plot(dca_bio['threshold'], dca_bio['model'], color='steelblue', linewidth=2,
        label='Biomarker Only')
ax.plot(dca_full['threshold'], dca_full['model'], color='darkorange', linewidth=2,
        label='Biomarker + Clinical')
ax.plot(dca_bio['threshold'], dca_bio['treat_all'], color='grey', linewidth=1,
        linestyle='--', label='Treat All')
ax.axhline(y=0, color='black', linewidth=0.8, label='Treat None')
ax.set_xlabel('Threshold Probability', fontsize=12)
ax.set_ylabel('Net Benefit', fontsize=12)
ax.set_title('Decision Curve Analysis: Biomarker Models for AMI', fontsize=14)
ax.legend(fontsize=10)
ax.set_ylim(-0.05, max(dca_bio['model'].max(), dca_full['model'].max()) + 0.05)
plt.tight_layout()
plt.show()

# Interpretation: The model with the highest net benefit at a given threshold
# is the best strategy at that clinical decision point. If neither model
# exceeds "Treat None" and "Treat All", the biomarker adds no clinical value.
# For AMI, relevant thresholds might be 5-15% (where missing a case is costly).
```

## Advanced Example

### Comparing Two Biomarkers (AUC Comparison with DeLong-like Test)

```python
# Simulate a second, inferior biomarker
np.random.seed(100)
biomarker2 = np.where(
    df['ami'] == 1,
    np.exp(np.random.normal(3.0, 1.0, n)),
    np.exp(np.random.normal(2.2, 0.9, n))
)

fpr2, tpr2, thresh2 = roc_curve(df['ami'], biomarker2)
auc2 = auc(fpr2, tpr2)
print(f"Biomarker 1 AUC: {roc_auc:.3f}")
print(f"Biomarker 2 AUC: {auc2:.3f}")

# Bootstrap test for AUC difference
n_boot_comp = 2000
auc_diffs = []
for _ in range(n_boot_comp):
    idx = np.random.choice(n, n, replace=True)
    if len(np.unique(df['ami'].values[idx])) < 2:
        continue
    a1 = roc_auc_score(df['ami'].values[idx], df['biomarker'].values[idx])
    a2 = roc_auc_score(df['ami'].values[idx], biomarker2[idx])
    auc_diffs.append(a1 - a2)

auc_diffs = np.array(auc_diffs)
p_value = 2 * min(np.mean(auc_diffs < 0), np.mean(auc_diffs > 0))

print(f"AUC difference: {roc_auc - auc2:.3f}")
print(f"Bootstrap 95% CI for difference: [{np.percentile(auc_diffs, 2.5):.3f}, "
      f"{np.percentile(auc_diffs, 97.5):.3f}]")
print(f"Bootstrap p-value: {p_value:.4f}")
```

### PPV and NPV Across Prevalence Levels

```python
prev_range = np.linspace(0.01, 0.50, 100)
ppv_by_prev = (sensitivity * prev_range) / (
    sensitivity * prev_range + (1 - specificity) * (1 - prev_range))
npv_by_prev = (specificity * (1 - prev_range)) / (
    (1 - sensitivity) * prev_range + specificity * (1 - prev_range))

fig, ax = plt.subplots(figsize=(8, 5))
ax.plot(prev_range, ppv_by_prev, color='steelblue', linewidth=2, label='PPV')
ax.plot(prev_range, npv_by_prev, color='darkorange', linewidth=2, label='NPV')
ax.axvline(x=prevalence, linestyle='--', color='grey', alpha=0.7,
           label=f'Study prevalence ({prevalence:.0%})')
ax.set_xlabel('Disease Prevalence', fontsize=12)
ax.set_ylabel('Predictive Value', fontsize=12)
ax.set_title('PPV and NPV as a Function of Prevalence', fontsize=14)
ax.legend(fontsize=10)
plt.tight_layout()
plt.show()
```

## Visualization

```python
fig, axes = plt.subplots(2, 2, figsize=(14, 10))

# Plot 1: ROC comparison
ax = axes[0, 0]
ax.plot(fpr, tpr, color='steelblue', linewidth=2,
        label=f'Biomarker 1 (AUC={roc_auc:.3f})')
ax.plot(fpr2, tpr2, color='darkorange', linewidth=2,
        label=f'Biomarker 2 (AUC={auc2:.3f})')
ax.plot([0, 1], [0, 1], '--', color='grey')
ax.set_xlabel('1 - Specificity')
ax.set_ylabel('Sensitivity')
ax.set_title('ROC Comparison')
ax.legend(fontsize=9)
ax.set_aspect('equal')

# Plot 2: Biomarker distributions with cutpoint
ax = axes[0, 1]
ax.hist(np.log(df.loc[df['ami']==0, 'biomarker']), bins=40, alpha=0.5,
        color='steelblue', density=True, label='No AMI')
ax.hist(np.log(df.loc[df['ami']==1, 'biomarker']), bins=40, alpha=0.5,
        color='red', density=True, label='AMI')
ax.axvline(x=np.log(optimal_cut), color='black', linestyle='--', linewidth=2)
ax.annotate(f'Cutpoint\n{optimal_cut:.1f} ng/L',
            xy=(np.log(optimal_cut), ax.get_ylim()[1] * 0.8),
            fontsize=9, ha='center')
ax.set_xlabel('log(Biomarker)')
ax.set_ylabel('Density')
ax.set_title('Biomarker Distribution by Disease Status')
ax.legend()

# Plot 3: Sensitivity and Specificity vs threshold
ax = axes[1, 0]
plot_thresh = thresholds[1:-1]  # exclude 0 and inf
plot_se = tpr[1:-1]
plot_sp = 1 - fpr[1:-1]
ax.plot(plot_thresh, plot_se, color='steelblue', linewidth=1.5, label='Sensitivity')
ax.plot(plot_thresh, plot_sp, color='darkorange', linewidth=1.5, label='Specificity')
ax.axvline(x=optimal_cut, linestyle='--', color='grey')
ax.set_xlim(0, np.percentile(df['biomarker'], 95))
ax.set_xlabel('Biomarker Threshold (ng/L)')
ax.set_ylabel('Value')
ax.set_title('Sensitivity & Specificity vs Threshold')
ax.legend()

# Plot 4: DCA
ax = axes[1, 1]
ax.plot(dca_bio['threshold'], dca_bio['model'], color='steelblue', linewidth=2,
        label='Biomarker Only')
ax.plot(dca_full['threshold'], dca_full['model'], color='darkorange', linewidth=2,
        label='Biomarker + Clinical')
ax.plot(dca_bio['threshold'], dca_bio['treat_all'], '--', color='grey',
        label='Treat All')
ax.axhline(y=0, color='black', linewidth=0.8, label='Treat None')
ax.set_xlabel('Threshold Probability')
ax.set_ylabel('Net Benefit')
ax.set_title('Decision Curve Analysis')
ax.legend(fontsize=8)
ax.set_ylim(-0.05, 0.3)

plt.tight_layout()
plt.show()
```

## Tips and Best Practices

1. **Use sklearn.metrics for standard ROC/AUC**: The `roc_curve` and `roc_auc_score` functions
   are well-tested and efficient. For confidence intervals, use bootstrap (as shown above).

2. **Always validate cutpoints**: An optimal cutpoint from the training data will be overly
   optimistic. Use cross-validation or an independent test set to get realistic performance
   estimates.

3. **DCA over AUC for clinical decisions**: AUC measures overall discrimination but does not
   address whether using the test improves patient outcomes. Decision curve analysis directly
   evaluates clinical utility at clinically relevant thresholds.

4. **Account for prevalence**: PPV and NPV change dramatically with prevalence. Always report
   the study prevalence and show how predictive values change across plausible prevalence
   values for the target population.

5. **Log-transform skewed biomarkers**: Many biomarkers (troponin, CRP, BNP) have right-skewed
   distributions. Use log-transformation before building logistic regression models.

6. **Report likelihood ratios**: LR+ and LR- are prevalence-independent and easily
   communicated to clinicians. An LR+ > 10 is strong evidence for ruling in disease;
   an LR- < 0.1 is strong evidence for ruling out disease.

7. **Consider the intended clinical role**: For a screening test (high sensitivity needed),
   select a lower threshold. For a confirmatory test (high specificity needed), select a
   higher threshold. The Youden index gives equal weight to both and may not be appropriate
   in all clinical contexts.

8. **Follow STARD reporting guidelines**: For any diagnostic accuracy study, report patient
   enrollment, reference standard, test execution, and results following the STARD checklist.
   This ensures transparency and reproducibility.
