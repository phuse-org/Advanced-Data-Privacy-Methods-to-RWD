# Regression and Generalized Linear Models — Theory

## Introduction

Regression analysis is the most widely used statistical framework in health research. It models the relationship between an outcome variable and one or more predictor variables, enabling estimation, inference, and prediction. Generalized linear models (GLMs) extend classical linear regression to accommodate non-normal outcome distributions — binary (disease present/absent), count (number of hospitalizations), and other types common in clinical and epidemiological data.

Understanding regression and GLMs is foundational for nearly every quantitative analysis in health sciences, from clinical trial analysis to observational epidemiology to health services research.

## Mathematical Foundation

### Linear Regression (Ordinary Least Squares)

The linear model specifies the conditional mean of a continuous outcome Y given predictors X:

```
Y_i = beta_0 + beta_1 * X_i1 + beta_2 * X_i2 + ... + beta_p * X_ip + epsilon_i
```

where epsilon_i ~ N(0, sigma^2) are independent errors. In matrix form: Y = X * beta + epsilon.

The OLS estimator minimizes the sum of squared residuals:

```
beta_hat = (X'X)^{-1} X'Y
```

Properties under standard assumptions:
- **Unbiased**: E[beta_hat] = beta
- **Minimum variance**: Among linear unbiased estimators (Gauss-Markov theorem)
- **Variance**: Var(beta_hat) = sigma^2 * (X'X)^{-1}

**R-squared** measures the proportion of variance explained:
```
R^2 = 1 - RSS/TSS = 1 - sum(y_i - y_hat_i)^2 / sum(y_i - y_bar)^2
```

**Adjusted R-squared** penalizes for the number of predictors:
```
R^2_adj = 1 - (1 - R^2) * (n - 1) / (n - p - 1)
```

### Generalized Linear Models (GLMs)

GLMs (Nelder and Wedderburn, 1972) consist of three components:

1. **Random component**: Y_i follows a distribution from the exponential family (Normal, Binomial, Poisson, Gamma, etc.)
2. **Systematic component**: A linear predictor eta_i = X_i * beta
3. **Link function**: g(mu_i) = eta_i, connecting the mean of Y to the linear predictor

```
g(E[Y_i | X_i]) = beta_0 + beta_1 * X_i1 + ... + beta_p * X_ip
```

Parameters are estimated via **maximum likelihood** using iteratively reweighted least squares (IRLS).

### Logistic Regression

For binary outcomes Y in {0, 1}, the logistic model uses the logit link:

```
log(p_i / (1 - p_i)) = beta_0 + beta_1 * X_i1 + ... + beta_p * X_ip
```

where p_i = P(Y_i = 1 | X_i). The exponentiated coefficients are **odds ratios**:

```
OR = exp(beta_j) = odds(Y=1 | X_j + 1) / odds(Y=1 | X_j)
```

An OR > 1 indicates increased odds; OR < 1 indicates decreased odds.

**Ordinal logistic regression** extends to ordinal outcomes using the proportional odds model:
```
log(P(Y <= k) / P(Y > k)) = alpha_k - X * beta
```

**Multinomial logistic regression** handles nominal categorical outcomes with J > 2 categories by modeling J-1 log-odds ratios relative to a reference category.

### Poisson Regression

For count outcomes Y >= 0, the Poisson model uses the log link:

```
log(E[Y_i | X_i]) = beta_0 + beta_1 * X_i1 + ... + beta_p * X_ip
```

Exponentiated coefficients are **rate ratios** (or incidence rate ratios):
```
RR = exp(beta_j)
```

An **offset** term accounts for varying exposure time:
```
log(E[Y_i]) = log(t_i) + X_i * beta
```

### Negative Binomial Regression

When count data exhibit overdispersion (variance > mean), the negative binomial model adds a dispersion parameter:

```
Var(Y_i) = mu_i + mu_i^2 / theta
```

where theta is the dispersion parameter. As theta approaches infinity, the model reduces to Poisson.

### Generalized Additive Models (GAMs)

GAMs extend GLMs by allowing smooth, non-linear relationships:

```
g(E[Y_i]) = beta_0 + f_1(X_i1) + f_2(X_i2) + ... + beta_p * X_ip
```

where f_j are smooth functions estimated using splines (thin-plate, cubic, P-splines). The smoothing parameter is selected by generalized cross-validation (GCV) or restricted maximum likelihood (REML).

### Penalized Regression

When predictors are numerous or correlated, penalized methods add a regularization term to the loss function:

**Ridge (L2)**:
```
beta_hat = argmin { sum(y_i - X_i*beta)^2 + lambda * sum(beta_j^2) }
```

**LASSO (L1)** (Tibshirani, 1996):
```
beta_hat = argmin { sum(y_i - X_i*beta)^2 + lambda * sum(|beta_j|) }
```

LASSO performs variable selection by shrinking some coefficients exactly to zero.

**Elastic Net** combines L1 and L2:
```
beta_hat = argmin { sum(y_i - X_i*beta)^2 + lambda * [alpha * sum(|beta_j|) + (1-alpha) * sum(beta_j^2)] }
```

The tuning parameter lambda controls the strength of regularization and is typically chosen by cross-validation.

## Key Concepts

### Model Selection

- **AIC** (Akaike Information Criterion): AIC = -2 * logL + 2p. Balances fit and complexity; lower is better.
- **BIC** (Bayesian Information Criterion): BIC = -2 * logL + p * log(n). Penalizes complexity more heavily than AIC.
- **Likelihood Ratio Test**: Compares nested models. Test statistic: -2 * (logL_reduced - logL_full) ~ chi^2(df).

### Diagnostics

- **Residual analysis**: Examine patterns in residuals vs. fitted values. Non-random patterns suggest model misspecification.
- **Influence diagnostics**: Cook's distance identifies influential observations. DFBETAS measures influence on individual coefficients.
- **Multicollinearity**: Variance Inflation Factor (VIF) > 10 suggests problematic collinearity.
- **Goodness of fit**: Hosmer-Lemeshow test for logistic regression; deviance and Pearson chi-squared for GLMs.

### Coefficient Interpretation

- **Linear model**: beta_j = change in E[Y] for a 1-unit increase in X_j, holding other predictors constant.
- **Logistic model**: exp(beta_j) = odds ratio for a 1-unit increase in X_j.
- **Poisson model**: exp(beta_j) = rate ratio for a 1-unit increase in X_j.
- **Log-transformed outcome**: beta_j is approximately the percentage change in Y for a 1-unit increase in X_j (when small).

## Assumptions

### Linear Regression
1. Linearity of the relationship between X and E[Y].
2. Independence of errors.
3. Homoscedasticity (constant variance of errors).
4. Normality of errors (for inference; not needed for estimation).
5. No perfect multicollinearity.

### GLMs
1. Correct specification of the distribution family and link function.
2. Independence of observations.
3. Correct specification of the mean-variance relationship.
4. No severe multicollinearity.

### Logistic Regression (Additional)
5. Linearity of continuous predictors on the logit scale.
6. No extreme separation or quasi-separation.

### Poisson Regression (Additional)
5. Mean equals variance (equidispersion). If violated, use negative binomial or quasi-Poisson.

## Variants and Extensions

- **Robust regression**: M-estimators and Huber regression for outlier-resistant estimation.
- **Quantile regression**: Models conditional quantiles (e.g., median) instead of the mean.
- **Mixed-effects models**: Add random effects for clustered or longitudinal data.
- **Zero-inflated models**: Handle excess zeros in count data (ZIP, ZINB).
- **Beta regression**: For outcomes bounded between 0 and 1 (proportions, rates).
- **Gamma regression**: For positive continuous outcomes with right-skewed distributions.

## When to Use This Method

- **Linear regression**: Continuous outcome with approximately normal errors.
- **Logistic regression**: Binary outcome (disease yes/no, death yes/no).
- **Poisson/NB regression**: Count outcome (number of events, hospital visits).
- **GAMs**: When non-linear relationships are suspected but no parametric form is known.
- **Penalized regression**: High-dimensional data (many predictors relative to sample size).
- **Ordinal regression**: Ordered categorical outcome (mild/moderate/severe).

## Strengths and Limitations

### Strengths
- Well-understood theoretical properties and extensive diagnostic tools.
- Interpretable coefficients with direct clinical meaning (ORs, RRs).
- Flexible framework accommodating many outcome types.
- Computationally efficient with closed-form or fast iterative solutions.
- Foundation for more advanced methods (mixed models, survival analysis, causal inference).

### Limitations
- Parametric assumptions may not hold in practice.
- Susceptible to multicollinearity, influential observations, and model misspecification.
- Logistic regression odds ratios are non-collapsible and may not approximate risk ratios.
- Penalized methods sacrifice unbiasedness for lower variance.
- Cannot establish causation without additional assumptions.

## Key References

1. McCullagh P, Nelder JA. *Generalized Linear Models.* 2nd ed. Chapman & Hall; 1989.
2. Hosmer DW, Lemeshow S, Sturdivant RX. *Applied Logistic Regression.* 3rd ed. Wiley; 2013.
3. Wood SN. *Generalized Additive Models: An Introduction with R.* 2nd ed. Chapman & Hall/CRC; 2017.
4. Hastie T, Tibshirani R, Friedman J. *The Elements of Statistical Learning.* 2nd ed. Springer; 2009.
5. Tibshirani R. Regression shrinkage and selection via the lasso. *J R Stat Soc Series B.* 1996;58(1):267-288.
6. Agresti A. *Categorical Data Analysis.* 3rd ed. Wiley; 2013.
7. Harrell FE. *Regression Modeling Strategies.* 2nd ed. Springer; 2015.
