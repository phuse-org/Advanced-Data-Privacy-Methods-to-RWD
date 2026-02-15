# Regression and GLMs — Python Implementation

## Required Libraries

```bash
pip install statsmodels scikit-learn pygam pandas numpy matplotlib seaborn scipy
```

```python
import numpy as np
import pandas as pd
import matplotlib.pyplot as plt
import seaborn as sns
import statsmodels.api as sm
import statsmodels.formula.api as smf
from sklearn.linear_model import LogisticRegression, LinearRegression, LassoCV, RidgeCV, ElasticNetCV
from sklearn.preprocessing import StandardScaler
from sklearn.model_selection import cross_val_score
from sklearn.metrics import roc_auc_score, roc_curve, mean_squared_error
import warnings
warnings.filterwarnings('ignore')
```

## Example Dataset

We simulate a clinical cohort dataset with continuous (HbA1c), binary (poor diabetes control), and count (hospitalizations) outcomes.

```python
np.random.seed(42)
n = 800

health = pd.DataFrame({
    'age': np.round(np.random.normal(55, 12, n)).astype(int),
    'bmi': np.round(np.random.normal(29, 5, n), 1),
    'exercise_hours': np.round(np.maximum(np.random.normal(3, 2, n), 0), 1),
    'medication': np.random.binomial(1, 0.5, n),
    'smoking': np.random.choice(['Never', 'Former', 'Current'], n, p=[0.4, 0.35, 0.25]),
    'sex': np.random.choice(['Male', 'Female'], n),
    'baseline_hba1c': np.round(np.random.normal(7.5, 1.2, n), 1),
    'num_comorbidities': np.random.poisson(2, n)
})

# Continuous outcome
health['hba1c_12m'] = np.round(
    6.5 + 0.01 * health['age'] + 0.05 * health['bmi'] -
    0.15 * health['exercise_hours'] - 0.8 * health['medication'] +
    0.3 * (health['smoking'] == 'Current').astype(int) +
    0.15 * (health['smoking'] == 'Former').astype(int) +
    0.3 * health['baseline_hba1c'] + 0.1 * health['num_comorbidities'] +
    np.random.normal(0, 0.5, n), 1)

# Binary outcome
health['poor_control'] = (health['hba1c_12m'] > 8).astype(int)

# Count outcome
health['hospitalizations'] = np.random.poisson(
    np.exp(-1.5 + 0.02 * health['age'] + 0.03 * health['bmi'] -
           0.1 * health['exercise_hours'] + 0.15 * health['num_comorbidities']))

print(health.describe())
print(f"\nPoor control rate: {health['poor_control'].mean():.3f}")
print(f"Mean hospitalizations: {health['hospitalizations'].mean():.2f}")
```

## Complete Worked Example

### Step 1: Linear Regression with statsmodels

```python
import statsmodels.formula.api as smf

# Fit OLS model using formula interface
lm_fit = smf.ols(
    'hba1c_12m ~ age + bmi + exercise_hours + medication + '
    'C(smoking, Treatment("Never")) + C(sex) + baseline_hba1c + num_comorbidities',
    data=health
).fit()

print(lm_fit.summary())

# Confidence intervals
print("\nCoefficient 95% CIs:")
print(lm_fit.conf_int().round(4))
```

**Interpretation**: The OLS summary provides coefficients, standard errors, t-statistics, and p-values. The coefficient for `medication` (-0.8) indicates that, holding other variables constant, medication use is associated with a 0.8-unit reduction in HbA1c. R-squared indicates the proportion of outcome variance explained. The F-statistic tests whether the model as a whole is significant.

### Step 2: Linear Model Diagnostics

```python
# Residual diagnostics
fig, axes = plt.subplots(2, 2, figsize=(12, 10))

# Residuals vs Fitted
residuals = lm_fit.resid
fitted_vals = lm_fit.fittedvalues
axes[0, 0].scatter(fitted_vals, residuals, alpha=0.3, s=10)
axes[0, 0].axhline(y=0, color='red', linestyle='--')
axes[0, 0].set_xlabel('Fitted Values')
axes[0, 0].set_ylabel('Residuals')
axes[0, 0].set_title('Residuals vs Fitted')

# Q-Q plot
from scipy import stats
stats.probplot(residuals, plot=axes[0, 1])
axes[0, 1].set_title('Normal Q-Q Plot')

# Scale-Location
sqrt_abs_resid = np.sqrt(np.abs(residuals))
axes[1, 0].scatter(fitted_vals, sqrt_abs_resid, alpha=0.3, s=10)
axes[1, 0].set_xlabel('Fitted Values')
axes[1, 0].set_ylabel('sqrt(|Residuals|)')
axes[1, 0].set_title('Scale-Location')

# Cook's distance
influence = lm_fit.get_influence()
cooks_d = influence.cooks_distance[0]
axes[1, 1].stem(range(len(cooks_d)), cooks_d, markerfmt=',', basefmt=' ')
axes[1, 1].axhline(y=4/n, color='red', linestyle='--', label=f'Threshold (4/n)')
axes[1, 1].set_xlabel('Observation')
axes[1, 1].set_ylabel("Cook's Distance")
axes[1, 1].set_title("Cook's Distance")
axes[1, 1].legend()

plt.tight_layout()
plt.savefig('lm_diagnostics.png', dpi=150)
plt.show()

# Variance Inflation Factors
from statsmodels.stats.outliers_influence import variance_inflation_factor

X_vif = health[['age', 'bmi', 'exercise_hours', 'medication',
                 'baseline_hba1c', 'num_comorbidities']].copy()
X_vif = sm.add_constant(X_vif)
vif_data = pd.DataFrame({
    'Variable': X_vif.columns[1:],
    'VIF': [variance_inflation_factor(X_vif.values, i+1) for i in range(X_vif.shape[1]-1)]
})
print("\nVariance Inflation Factors:")
print(vif_data.to_string(index=False))
```

**Interpretation**: The diagnostic plots check key assumptions. Random scatter in Residuals vs Fitted supports linearity and homoscedasticity. Points on the Q-Q line support normality. Cook's distance above 4/n flags potentially influential observations. VIF below 5 indicates acceptable collinearity.

### Step 3: Logistic Regression

```python
# Fit logistic regression with statsmodels
logit_fit = smf.logit(
    'poor_control ~ age + bmi + exercise_hours + medication + '
    'C(smoking, Treatment("Never")) + C(sex) + baseline_hba1c + num_comorbidities',
    data=health
).fit()

print(logit_fit.summary())

# Odds ratios with 95% CI
params = logit_fit.params
conf = logit_fit.conf_int()
or_table = pd.DataFrame({
    'OR': np.exp(params),
    'OR_lower': np.exp(conf[0]),
    'OR_upper': np.exp(conf[1]),
    'p_value': logit_fit.pvalues
}).round(3)
print("\nOdds Ratios:")
print(or_table)
```

**Interpretation**: Exponentiated coefficients are odds ratios. An OR of 1.5 for BMI means each unit increase in BMI is associated with 50% higher odds of poor diabetes control. ORs with confidence intervals that cross 1.0 are not statistically significant at the 0.05 level.

### Step 4: ROC Curve and Model Evaluation

```python
from sklearn.metrics import roc_auc_score, roc_curve

# Predicted probabilities
pred_prob = logit_fit.predict()

# ROC curve
fpr, tpr, thresholds = roc_curve(health['poor_control'], pred_prob)
auc = roc_auc_score(health['poor_control'], pred_prob)

fig, ax = plt.subplots(figsize=(8, 6))
ax.plot(fpr, tpr, color='navy', linewidth=2, label=f'AUC = {auc:.3f}')
ax.plot([0, 1], [0, 1], color='gray', linestyle='--', linewidth=1)
ax.set_xlabel('False Positive Rate', fontsize=12)
ax.set_ylabel('True Positive Rate', fontsize=12)
ax.set_title('ROC Curve: Logistic Regression for Poor Diabetes Control', fontsize=14)
ax.legend(fontsize=12)
ax.grid(True, alpha=0.3)
plt.tight_layout()
plt.savefig('roc_curve.png', dpi=150)
plt.show()
```

**Interpretation**: The ROC curve plots sensitivity vs. 1-specificity at all thresholds. The AUC (area under the curve) summarizes discrimination. AUC = 0.5 is random; AUC > 0.7 is acceptable; AUC > 0.8 is good.

### Step 5: Poisson Regression

```python
# Fit Poisson model
pois_fit = smf.glm(
    'hospitalizations ~ age + bmi + exercise_hours + num_comorbidities + '
    'medication + C(smoking, Treatment("Never"))',
    data=health,
    family=sm.families.Poisson()
).fit()

print(pois_fit.summary())

# Rate ratios
rr_table = pd.DataFrame({
    'RR': np.exp(pois_fit.params),
    'RR_lower': np.exp(pois_fit.conf_int()[0]),
    'RR_upper': np.exp(pois_fit.conf_int()[1]),
    'p_value': pois_fit.pvalues
}).round(3)
print("\nRate Ratios:")
print(rr_table)

# Check overdispersion
pearson_chi2 = pois_fit.pearson_chi2
dispersion = pearson_chi2 / pois_fit.df_resid
print(f"\nPearson dispersion: {dispersion:.3f}")
print(f"Overdispersed: {dispersion > 1.5}")
```

**Interpretation**: Rate ratios from Poisson regression represent the multiplicative change in the event rate per unit change in the predictor. An RR of 1.03 for age means each additional year increases the hospitalization rate by 3%. The Pearson dispersion statistic near 1 supports the Poisson assumption; values substantially above 1 indicate overdispersion.

### Step 6: Negative Binomial Regression

```python
# Fit negative binomial model
nb_fit = smf.glm(
    'hospitalizations ~ age + bmi + exercise_hours + num_comorbidities + '
    'medication + C(smoking, Treatment("Never"))',
    data=health,
    family=sm.families.NegativeBinomial(alpha=1.0)
).fit()

print(nb_fit.summary())

# Compare AIC
print(f"\nPoisson AIC: {pois_fit.aic:.1f}")
print(f"NB AIC: {nb_fit.aic:.1f}")
print(f"Preferred: {'Negative Binomial' if nb_fit.aic < pois_fit.aic else 'Poisson'}")
```

**Interpretation**: If the negative binomial AIC is substantially lower than the Poisson AIC, overdispersion is present and the NB model is preferred. The NB model adds a dispersion parameter that accounts for extra-Poisson variability.

## Advanced Example

### Penalized Regression (LASSO, Ridge, Elastic Net)

```python
from sklearn.linear_model import LassoCV, RidgeCV, ElasticNetCV
from sklearn.preprocessing import StandardScaler

# Prepare features with dummy coding
X_pen = pd.get_dummies(health[['age', 'bmi', 'exercise_hours', 'medication',
                                'smoking', 'sex', 'baseline_hba1c', 'num_comorbidities']],
                        drop_first=True)
y_pen = health['hba1c_12m'].values

# Standardize
scaler = StandardScaler()
X_scaled = scaler.fit_transform(X_pen)

# LASSO
lasso = LassoCV(cv=10, random_state=42, max_iter=10000)
lasso.fit(X_scaled, y_pen)
print(f"LASSO: optimal alpha = {lasso.alpha_:.5f}")
print(f"LASSO: non-zero coefficients = {np.sum(lasso.coef_ != 0)} / {len(lasso.coef_)}")
print(f"LASSO: CV R-squared = {lasso.score(X_scaled, y_pen):.4f}")

# Ridge
ridge = RidgeCV(cv=10)
ridge.fit(X_scaled, y_pen)
print(f"\nRidge: optimal alpha = {ridge.alpha_:.5f}")
print(f"Ridge: CV R-squared = {ridge.score(X_scaled, y_pen):.4f}")

# Elastic Net
enet = ElasticNetCV(cv=10, l1_ratio=[0.1, 0.3, 0.5, 0.7, 0.9], random_state=42, max_iter=10000)
enet.fit(X_scaled, y_pen)
print(f"\nElastic Net: optimal alpha = {enet.alpha_:.5f}, l1_ratio = {enet.l1_ratio_:.2f}")
print(f"Elastic Net: CV R-squared = {enet.score(X_scaled, y_pen):.4f}")

# Coefficient comparison
coef_comparison = pd.DataFrame({
    'Feature': X_pen.columns,
    'LASSO': np.round(lasso.coef_, 4),
    'Ridge': np.round(ridge.coef_, 4),
    'ElasticNet': np.round(enet.coef_, 4)
})
print("\nCoefficient Comparison (standardized):")
print(coef_comparison.to_string(index=False))
```

**Interpretation**: LASSO performs variable selection by setting some coefficients to zero. Features with non-zero LASSO coefficients are the most important predictors. Ridge retains all variables but shrinks coefficients. Elastic Net provides a middle ground. Cross-validated R-squared estimates out-of-sample performance.

### LASSO Regularization Path

```python
from sklearn.linear_model import lasso_path

alphas, coefs, _ = lasso_path(X_scaled, y_pen, alphas=np.logspace(-4, 0, 100))

fig, ax = plt.subplots(figsize=(10, 6))
for i, feature in enumerate(X_pen.columns):
    ax.plot(np.log10(alphas), coefs[i, :], linewidth=1.5, label=feature)
ax.axvline(x=np.log10(lasso.alpha_), color='black', linestyle='--', label='CV optimal')
ax.set_xlabel('log10(alpha)', fontsize=12)
ax.set_ylabel('Coefficient', fontsize=12)
ax.set_title('LASSO Regularization Path', fontsize=14)
ax.legend(fontsize=8, loc='upper right', ncol=2)
ax.grid(True, alpha=0.3)
plt.tight_layout()
plt.savefig('lasso_path.png', dpi=150)
plt.show()
```

### Generalized Additive Models (GAMs)

```python
from pygam import LinearGAM, LogisticGAM, s, f, l

# Prepare data
X_gam = health[['age', 'bmi', 'exercise_hours', 'medication',
                 'baseline_hba1c', 'num_comorbidities']].values
y_gam = health['hba1c_12m'].values

# Fit GAM with smooth terms for continuous, linear for binary
gam = LinearGAM(
    s(0) + s(1) + s(2) + l(3) + s(4) + l(5)
).gridsearch(X_gam, y_gam)

print(gam.summary())

# Plot partial dependence
feature_names = ['age', 'bmi', 'exercise_hours', 'medication',
                 'baseline_hba1c', 'num_comorbidities']

fig, axes = plt.subplots(2, 3, figsize=(15, 9))
for i, ax in enumerate(axes.flatten()):
    if i < len(feature_names):
        XX = gam.generate_X_grid(term=i)
        pdep, confi = gam.partial_dependence(term=i, X=XX, width=0.95)
        ax.plot(XX[:, i], pdep, color='navy', linewidth=2)
        ax.fill_between(XX[:, i], confi[:, 0], confi[:, 1], alpha=0.2, color='navy')
        ax.set_xlabel(feature_names[i], fontsize=11)
        ax.set_ylabel('Partial Effect', fontsize=11)
        ax.set_title(f'Effect of {feature_names[i]}', fontsize=12)
        ax.grid(True, alpha=0.3)

plt.suptitle('GAM Partial Dependence Plots', fontsize=14, y=1.02)
plt.tight_layout()
plt.savefig('gam_partial.png', dpi=150)
plt.show()

# Compare GAM vs OLS
from sklearn.metrics import r2_score
gam_pred = gam.predict(X_gam)
ols_pred = lm_fit.fittedvalues.values

print(f"\nR-squared comparison:")
print(f"  OLS: {r2_score(y_gam, ols_pred):.4f}")
print(f"  GAM: {r2_score(y_gam, gam_pred):.4f}")
```

**Interpretation**: The GAM partial dependence plots show the estimated smooth relationship between each predictor and the outcome. Non-linear shapes (curves, thresholds) indicate that the linear assumption is violated and the GAM captures important structure. The shaded bands are 95% confidence intervals.

## Visualization

### Coefficient Plot (Linear Model)

```python
params = lm_fit.params[1:]  # Exclude intercept
ci = lm_fit.conf_int().iloc[1:]

fig, ax = plt.subplots(figsize=(8, 6))
y_pos = range(len(params))
ax.errorbar(params.values, y_pos, xerr=[params.values - ci.iloc[:, 0].values,
            ci.iloc[:, 1].values - params.values],
            fmt='o', color='navy', capsize=4, markersize=6)
ax.axvline(x=0, color='red', linestyle='--', linewidth=1)
ax.set_yticks(list(y_pos))
ax.set_yticklabels(params.index, fontsize=10)
ax.set_xlabel('Coefficient (95% CI)', fontsize=12)
ax.set_title('Linear Regression Coefficients: HbA1c at 12 Months', fontsize=14)
ax.grid(True, alpha=0.3, axis='x')
plt.tight_layout()
plt.savefig('coef_plot_lm.png', dpi=150)
plt.show()
```

### Predicted vs Actual (OLS)

```python
fig, ax = plt.subplots(figsize=(8, 8))
ax.scatter(lm_fit.fittedvalues, health['hba1c_12m'], alpha=0.3, s=15, color='steelblue')
lims = [health['hba1c_12m'].min() - 0.5, health['hba1c_12m'].max() + 0.5]
ax.plot(lims, lims, 'r--', linewidth=1.5)
ax.set_xlabel('Predicted HbA1c', fontsize=12)
ax.set_ylabel('Observed HbA1c', fontsize=12)
ax.set_title('Predicted vs. Observed (Linear Model)', fontsize=14)
ax.set_xlim(lims)
ax.set_ylim(lims)
ax.set_aspect('equal')
ax.grid(True, alpha=0.3)
plt.tight_layout()
plt.savefig('pred_vs_actual.png', dpi=150)
plt.show()
```

## Tips and Best Practices

1. **Use `statsmodels` for inference, `scikit-learn` for prediction**: statsmodels provides p-values, confidence intervals, and diagnostic tests. scikit-learn excels at cross-validation and regularization.

2. **Always check residual diagnostics**: Plot residuals vs. fitted values, Q-Q plots, and Cook's distance. Patterns indicate model misspecification.

3. **Interpret odds ratios carefully**: ORs from logistic regression are non-collapsible. An adjusted OR differs from a marginal OR even with no confounding. Consider marginal effects for communication:

```python
marginal_effects = logit_fit.get_margeff()
print(marginal_effects.summary())
```

4. **Handle categorical variables explicitly**: Use `C()` in formulas or `pd.get_dummies()` for design matrices. Always specify the reference category.

5. **Standardize for penalized regression**: LASSO and Ridge penalties are scale-dependent. Always standardize predictors before fitting, or use built-in standardization.

6. **Use cross-validation for model selection**: Never select lambda (regularization parameter) based on in-sample fit. Always use cross-validation (LassoCV, RidgeCV).

7. **Report complete results**: Include sample size, number of events (for logistic), all coefficients, confidence intervals, and model fit metrics (R-squared, AIC, AUC).

8. **Check for separation in logistic regression**: If a predictor perfectly predicts the outcome, maximum likelihood estimates diverge. Use Firth's penalized likelihood:

```python
# statsmodels does not natively support Firth's method.
# Consider using R via rpy2 or the 'firthlogist' package.
```

9. **Use GAMs when non-linearity is suspected**: GAMs are a natural extension of GLMs that can reveal non-linear dose-response relationships without pre-specifying the functional form.

10. **Validate externally when possible**: Internal cross-validation is good, but external validation on an independent dataset provides the strongest evidence of model generalizability.
