# Machine Learning for Clinical Prediction — Python Implementation

## Required Libraries

```bash
pip install numpy pandas scikit-learn xgboost lightgbm shap matplotlib seaborn
```

- **numpy / pandas**: Data manipulation and numerical operations.
- **scikit-learn**: Comprehensive ML toolkit (preprocessing, models, validation, metrics).
- **xgboost**: Gradient boosting framework.
- **lightgbm**: Fast gradient boosting with histogram-based learning.
- **shap**: SHAP value computation and visualization for model interpretability.
- **matplotlib / seaborn**: Plotting and visualization.

## Example Dataset

We simulate an ICU mortality prediction dataset with 2000 patients and 10 clinical features.

```python
import numpy as np
import pandas as pd

np.random.seed(42)
n = 2000

icu_data = pd.DataFrame({
    'age': np.round(np.random.normal(65, 15, n)).astype(int),
    'sex': np.random.choice(['Male', 'Female'], n),
    'sofa_score': np.random.poisson(5, n),
    'apache_ii': np.round(np.random.normal(18, 7, n)).astype(int),
    'heart_rate': np.round(np.random.normal(90, 20, n)).astype(int),
    'sbp': np.round(np.random.normal(120, 25, n)).astype(int),
    'creatinine': np.round(np.random.lognormal(0.3, 0.6, n), 2),
    'wbc': np.round(np.random.lognormal(2.2, 0.5, n), 1),
    'lactate': np.round(np.random.lognormal(0.5, 0.7, n), 2),
    'mech_vent': np.random.choice([0, 1], n, p=[0.6, 0.4]),
})

# Simulate mortality (~20% event rate)
log_odds = (-4 + 0.03 * icu_data['age'] + 0.15 * icu_data['sofa_score'] +
            0.08 * icu_data['apache_ii'] + 0.5 * icu_data['lactate'] -
            0.01 * icu_data['sbp'] + 0.3 * icu_data['mech_vent'])
prob = 1 / (1 + np.exp(-log_odds))
icu_data['mortality'] = np.random.binomial(1, prob)

print(f"Dataset shape: {icu_data.shape}")
print(f"Mortality rate: {icu_data['mortality'].mean():.3f}")
print(icu_data.head())
```

## Complete Worked Example

### Step 1: Data Preparation and Splitting

```python
from sklearn.model_selection import train_test_split
from sklearn.preprocessing import StandardScaler, LabelEncoder

# Encode categorical variable
le = LabelEncoder()
icu_data['sex_encoded'] = le.fit_transform(icu_data['sex'])

# Define features and target
feature_cols = ['age', 'sex_encoded', 'sofa_score', 'apache_ii',
                'heart_rate', 'sbp', 'creatinine', 'wbc',
                'lactate', 'mech_vent']
X = icu_data[feature_cols].values
y = icu_data['mortality'].values

# Stratified train-test split
X_train, X_test, y_train, y_test = train_test_split(
    X, y, test_size=0.25, random_state=123, stratify=y
)

# Scale features (for logistic regression and SVM)
scaler = StandardScaler()
X_train_scaled = scaler.fit_transform(X_train)
X_test_scaled = scaler.transform(X_test)

print(f"Training set: {X_train.shape[0]} patients, "
      f"event rate: {y_train.mean():.3f}")
print(f"Test set: {X_test.shape[0]} patients, "
      f"event rate: {y_test.mean():.3f}")
```

**Output interpretation**: Stratified splitting ensures proportional representation of mortality events in both sets. Feature scaling is applied to training data and the same transformation is applied to test data to prevent information leakage.

### Step 2: Model Training with Cross-Validation

```python
from sklearn.linear_model import LogisticRegression
from sklearn.ensemble import RandomForestClassifier
from xgboost import XGBClassifier
from lightgbm import LGBMClassifier
from sklearn.model_selection import cross_val_score, StratifiedKFold

# Define models
models = {
    'Logistic Regression': LogisticRegression(
        C=1.0, penalty='l2', max_iter=1000, random_state=42
    ),
    'Random Forest': RandomForestClassifier(
        n_estimators=500, max_depth=6, min_samples_leaf=10,
        max_features='sqrt', random_state=42, n_jobs=-1
    ),
    'XGBoost': XGBClassifier(
        n_estimators=500, max_depth=4, learning_rate=0.05,
        subsample=0.8, colsample_bytree=0.8, min_child_weight=10,
        eval_metric='logloss', random_state=42
    ),
    'LightGBM': LGBMClassifier(
        n_estimators=500, max_depth=4, learning_rate=0.05,
        subsample=0.8, colsample_bytree=0.8, min_child_samples=10,
        random_state=42, verbose=-1
    ),
}

# 10-fold stratified cross-validation
cv = StratifiedKFold(n_splits=10, shuffle=True, random_state=42)

print("=== Cross-Validation AUC ===")
cv_results = {}
for name, model in models.items():
    X_input = X_train_scaled if name == 'Logistic Regression' else X_train
    scores = cross_val_score(model, X_input, y_train,
                             cv=cv, scoring='roc_auc', n_jobs=-1)
    cv_results[name] = scores
    print(f"{name:25s}: AUC = {scores.mean():.4f} (+/- {scores.std():.4f})")
```

**Output interpretation**: Cross-validated AUC estimates generalization performance. The mean and standard deviation across folds indicate both the expected performance and its stability. Models with similar AUC but less variance are preferred. If logistic regression performs comparably to ML methods, it is the better choice for interpretability.

### Step 3: Train Final Model and Evaluate on Test Set

```python
from sklearn.metrics import (roc_auc_score, roc_curve, brier_score_loss,
                              classification_report, confusion_matrix)

# Train final XGBoost model on full training set
final_model = XGBClassifier(
    n_estimators=500, max_depth=4, learning_rate=0.05,
    subsample=0.8, colsample_bytree=0.8, min_child_weight=10,
    eval_metric='logloss', random_state=42, use_label_encoder=False
)
final_model.fit(X_train, y_train)

# Predict probabilities on test set
y_prob = final_model.predict_proba(X_test)[:, 1]
y_pred = final_model.predict(X_test)

# Performance metrics
auc = roc_auc_score(y_test, y_prob)
brier = brier_score_loss(y_test, y_prob)

print(f"=== Test Set Performance (XGBoost) ===")
print(f"AUC: {auc:.4f}")
print(f"Brier Score: {brier:.4f}")
print(f"\nClassification Report (default threshold 0.5):")
print(classification_report(y_test, y_pred,
                            target_names=['Survived', 'Died']))
```

**Output interpretation**: The test set AUC provides an unbiased assessment of discrimination. Brier score combines discrimination and calibration (lower is better; the null model Brier score equals prevalence * (1 - prevalence)). The classification report shows precision, recall, and F1 at the default 0.5 threshold, which may not be optimal.

### Step 4: ROC Curve

```python
import matplotlib.pyplot as plt

fpr, tpr, thresholds = roc_curve(y_test, y_prob)

fig, ax = plt.subplots(figsize=(8, 6))
ax.plot(fpr, tpr, color='steelblue', lw=2,
        label=f'XGBoost (AUC = {auc:.3f})')
ax.plot([0, 1], [0, 1], 'k--', lw=1, alpha=0.5, label='Reference')
ax.fill_between(fpr, tpr, alpha=0.1, color='steelblue')
ax.set_xlabel('1 - Specificity (False Positive Rate)', fontsize=12)
ax.set_ylabel('Sensitivity (True Positive Rate)', fontsize=12)
ax.set_title('ROC Curve: ICU Mortality Prediction', fontsize=14)
ax.legend(loc='lower right', fontsize=11)
ax.set_xlim([-0.02, 1.02])
ax.set_ylim([-0.02, 1.02])
plt.tight_layout()
plt.savefig('roc_curve.png', dpi=150, bbox_inches='tight')
plt.show()
```

**Output interpretation**: The ROC curve plots sensitivity against 1-specificity across all classification thresholds. The area under this curve (AUC) summarizes discriminative ability. A curve hugging the top-left corner indicates excellent discrimination.

### Step 5: Calibration Plot

```python
from sklearn.calibration import calibration_curve

prob_true, prob_pred = calibration_curve(y_test, y_prob, n_bins=10,
                                         strategy='uniform')

fig, ax = plt.subplots(figsize=(8, 6))
ax.plot(prob_pred, prob_true, 's-', color='steelblue', lw=2,
        markersize=8, label='XGBoost')
ax.plot([0, 1], [0, 1], 'k--', lw=1, alpha=0.5, label='Perfect calibration')
ax.set_xlabel('Mean Predicted Probability', fontsize=12)
ax.set_ylabel('Observed Proportion', fontsize=12)
ax.set_title('Calibration Plot: ICU Mortality Prediction', fontsize=14)
ax.legend(loc='upper left', fontsize=11)
ax.set_xlim([-0.02, 1.02])
ax.set_ylim([-0.02, 1.02])
plt.tight_layout()
plt.savefig('calibration_plot.png', dpi=150, bbox_inches='tight')
plt.show()
```

**Output interpretation**: Points on the diagonal line indicate perfect calibration. If the curve lies above the diagonal, the model underestimates risk (predicted probabilities are too low); if below, it overestimates. Most tree-based models require recalibration (Platt scaling or isotonic regression) for well-calibrated probabilities.

## Advanced Example

### SHAP Values for Model Interpretability

```python
import shap

# Compute SHAP values
explainer = shap.TreeExplainer(final_model)
shap_values = explainer.shap_values(X_test)

# Summary (beeswarm) plot
fig, ax = plt.subplots(figsize=(10, 7))
shap.summary_plot(shap_values, X_test, feature_names=feature_cols,
                  show=False)
plt.title('SHAP Summary Plot: Feature Contributions to Mortality Risk')
plt.tight_layout()
plt.savefig('shap_summary.png', dpi=150, bbox_inches='tight')
plt.show()

# Waterfall plot for a single patient (first test patient)
shap.plots.waterfall(shap.Explanation(
    values=shap_values[0],
    base_values=explainer.expected_value,
    data=X_test[0],
    feature_names=feature_cols
))
```

**Output interpretation**: The SHAP summary plot ranks features by importance (top = most important). Each dot represents one patient. Color indicates the feature value (red = high, blue = low). Dots to the right increase the predicted mortality probability. For example, high lactate (red dots on the right) increases mortality risk. The waterfall plot decomposes one patient's prediction into individual feature contributions.

### SHAP Dependence Plot

```python
# Dependence plot for lactate (expected top feature)
shap.dependence_plot('lactate', shap_values, X_test,
                     feature_names=feature_cols,
                     interaction_index='sofa_score')
```

**Output interpretation**: The dependence plot shows how the SHAP value for lactate varies with its actual value. The color indicates the interaction with SOFA score. A strong positive slope means that increasing lactate monotonically increases mortality risk. Interactions appear as color separation at similar lactate levels.

### Decision Curve Analysis

```python
def decision_curve_analysis(y_true, y_probs_dict, thresholds):
    """Compute net benefit for each model across threshold probabilities."""
    n = len(y_true)
    results = []
    for t in thresholds:
        row = {'threshold': t}
        # Treat all strategy
        row['Treat All'] = y_true.mean() - (1 - y_true.mean()) * t / (1 - t)
        # Treat none
        row['Treat None'] = 0.0
        for name, probs in y_probs_dict.items():
            tp = np.sum((probs >= t) & (y_true == 1)) / n
            fp = np.sum((probs >= t) & (y_true == 0)) / n
            row[name] = tp - fp * t / (1 - t)
        results.append(row)
    return pd.DataFrame(results)

# Train logistic regression for comparison
lr_model = LogisticRegression(C=1.0, max_iter=1000, random_state=42)
lr_model.fit(X_train_scaled, y_train)
y_prob_lr = lr_model.predict_proba(X_test_scaled)[:, 1]

thresholds = np.arange(0.01, 0.61, 0.01)
dca_df = decision_curve_analysis(
    y_test,
    {'XGBoost': y_prob, 'Logistic Regression': y_prob_lr},
    thresholds
)

fig, ax = plt.subplots(figsize=(9, 6))
ax.plot(dca_df['threshold'], dca_df['Treat All'], 'k-', lw=1,
        label='Treat All', alpha=0.7)
ax.axhline(y=0, color='gray', linestyle='-', lw=1, label='Treat None')
ax.plot(dca_df['threshold'], dca_df['XGBoost'], '-', color='steelblue',
        lw=2, label='XGBoost')
ax.plot(dca_df['threshold'], dca_df['Logistic Regression'], '-',
        color='coral', lw=2, label='Logistic Regression')
ax.set_xlabel('Threshold Probability', fontsize=12)
ax.set_ylabel('Net Benefit', fontsize=12)
ax.set_title('Decision Curve Analysis: Clinical Utility', fontsize=14)
ax.legend(loc='upper right', fontsize=10)
ax.set_xlim([0, 0.6])
ax.set_ylim([-0.05, max(dca_df['Treat All'].max(), 0.3)])
plt.tight_layout()
plt.savefig('dca_plot.png', dpi=150, bbox_inches='tight')
plt.show()
```

**Output interpretation**: Decision curve analysis shows net benefit across threshold probabilities. The model with the highest net benefit at a clinically relevant threshold (e.g., 10-30% for ICU mortality) offers the best tradeoff between identifying true positives and avoiding false positives. If a model's net benefit never exceeds "Treat All" or "Treat None," it has no clinical utility.

## Visualization

### Feature Importance Comparison

```python
# XGBoost built-in importance (gain)
xgb_importance = final_model.feature_importances_
sorted_idx = np.argsort(xgb_importance)

fig, ax = plt.subplots(figsize=(8, 6))
ax.barh(range(len(sorted_idx)), xgb_importance[sorted_idx],
        color='steelblue', edgecolor='darkblue')
ax.set_yticks(range(len(sorted_idx)))
ax.set_yticklabels([feature_cols[i] for i in sorted_idx])
ax.set_xlabel('Feature Importance (Gain)', fontsize=12)
ax.set_title('XGBoost Feature Importance', fontsize=14)
plt.tight_layout()
plt.savefig('feature_importance.png', dpi=150, bbox_inches='tight')
plt.show()
```

### Threshold Optimization

```python
from sklearn.metrics import precision_recall_curve, f1_score

# Find optimal threshold using Youden's J
j_scores = tpr - fpr
optimal_idx = np.argmax(j_scores)
optimal_threshold = thresholds[optimal_idx] if optimal_idx < len(thresholds) else 0.5

print(f"Optimal threshold (Youden's J): {optimal_threshold:.3f}")
print(f"Sensitivity at optimal: {tpr[optimal_idx]:.3f}")
print(f"Specificity at optimal: {1 - fpr[optimal_idx]:.3f}")

# Confusion matrix at optimal threshold
from sklearn.metrics import ConfusionMatrixDisplay

y_pred_optimal = (y_prob >= optimal_threshold).astype(int)
fig, ax = plt.subplots(figsize=(6, 5))
ConfusionMatrixDisplay.from_predictions(
    y_test, y_pred_optimal,
    display_labels=['Survived', 'Died'],
    cmap='Blues', ax=ax
)
ax.set_title(f'Confusion Matrix (threshold = {optimal_threshold:.3f})')
plt.tight_layout()
plt.savefig('confusion_matrix.png', dpi=150, bbox_inches='tight')
plt.show()
```

**Output interpretation**: Youden's J index finds the threshold that maximizes the sum of sensitivity and specificity. In clinical practice, the threshold should reflect the clinical consequences: for a high-stakes outcome like mortality, higher sensitivity (lower threshold) may be preferred even at the cost of more false positives.

## Tips and Best Practices

1. **Start simple**: Always include logistic regression as a benchmark. It frequently matches or exceeds ML methods on clinical tabular data.
2. **Prevent data leakage**: Scaling, imputation, and feature selection must be done within the cross-validation loop, not before splitting.
3. **Use stratified splits** for imbalanced outcomes to ensure representative event rates in each fold.
4. **Report calibration**: AUC alone is insufficient. Use `CalibratedClassifierCV` from scikit-learn to recalibrate tree-based models.
5. **Prefer SHAP over built-in importance**: Built-in feature importance (gain/split count) can be misleading; SHAP provides consistent attributions.
6. **Handle class imbalance thoughtfully**: Use `scale_pos_weight` in XGBoost or `class_weight='balanced'` in scikit-learn rather than oversampling.
7. **Perform external validation** before clinical deployment. Internal CV routinely overestimates performance.
8. **Monitor for dataset shift**: Clinical practice evolves. Retrain models periodically and track AUC and calibration over time windows.
9. **Follow TRIPOD guidelines** for reporting: describe the study population, sample size, event rate, feature selection, validation strategy, and all performance metrics.
10. **Consider clinical workflow**: Ensure all input features are routinely available at the point of care when the prediction would be made.
