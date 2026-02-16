# Regression and GLMs — R Implementation

## Required Packages

```r
install.packages(c("stats", "MASS", "glmnet", "mgcv", "broom", "ggplot2",
                    "dplyr", "car", "performance", "see"))

library(MASS)
library(glmnet)
library(mgcv)
library(broom)
library(ggplot2)
library(dplyr)
library(car)
```

## Example Dataset

We simulate a health dataset representing a clinical cohort study with multiple outcomes: a continuous outcome (HbA1c), a binary outcome (diabetes control status), and a count outcome (number of hospitalizations). This allows demonstrating linear, logistic, and Poisson regression on the same cohort.

```r
set.seed(42)
n <- 800

health_data <- data.frame(
  age = round(rnorm(n, 55, 12)),
  bmi = round(rnorm(n, 29, 5), 1),
  exercise_hours = round(pmax(rnorm(n, 3, 2), 0), 1),
  medication = rbinom(n, 1, 0.5),
  smoking = factor(sample(c("Never", "Former", "Current"), n,
                           replace = TRUE, prob = c(0.4, 0.35, 0.25))),
  sex = factor(sample(c("Male", "Female"), n, replace = TRUE)),
  baseline_hba1c = round(rnorm(n, 7.5, 1.2), 1),
  num_comorbidities = rpois(n, 2)
)

# Continuous outcome: HbA1c at 12 months
health_data$hba1c_12m <- with(health_data, {
  6.5 + 0.01 * age + 0.05 * bmi - 0.15 * exercise_hours -
    0.8 * medication + 0.3 * (smoking == "Current") +
    0.15 * (smoking == "Former") + 0.3 * baseline_hba1c +
    0.1 * num_comorbidities + rnorm(n, 0, 0.5)
}) %>% round(1)

# Binary outcome: Poor diabetes control (HbA1c > 8)
health_data$poor_control <- as.integer(health_data$hba1c_12m > 8)

# Count outcome: Number of hospitalizations in 12 months
health_data$hospitalizations <- rpois(n, lambda = exp(
  -1.5 + 0.02 * health_data$age + 0.03 * health_data$bmi -
    0.1 * health_data$exercise_hours + 0.15 * health_data$num_comorbidities
))

str(health_data)
summary(health_data)
```

## Complete Worked Example

### Step 1: Linear Regression (OLS)

```r
# Fit linear model for HbA1c at 12 months
lm_fit <- lm(hba1c_12m ~ age + bmi + exercise_hours + medication +
                smoking + sex + baseline_hba1c + num_comorbidities,
              data = health_data)

summary(lm_fit)

# Tidy output with broom
tidy(lm_fit, conf.int = TRUE) %>%
  mutate(across(where(is.numeric), ~ round(., 4))) %>%
  print()

# Model fit statistics
glance(lm_fit)
```

**Interpretation**: The coefficient for `medication` represents the average change in HbA1c associated with being on medication, holding all other variables constant. A coefficient of -0.8 means medication is associated with a 0.8 unit lower HbA1c. The R-squared indicates the proportion of variance explained. The F-statistic tests the overall model significance.

### Step 2: Linear Model Diagnostics

```r
# Diagnostic plots
par(mfrow = c(2, 2))
plot(lm_fit)
par(mfrow = c(1, 1))

# Variance inflation factors
vif_values <- car::vif(lm_fit)
print(vif_values)
cat("\nAll VIF < 5:", all(vif_values < 5), "\n")

# Normality of residuals
shapiro.test(residuals(lm_fit)[1:min(5000, n)])

# Breusch-Pagan test for heteroscedasticity
car::ncvTest(lm_fit)
```

**Interpretation**: The four diagnostic plots check linearity (Residuals vs Fitted), normality (Q-Q plot), homoscedasticity (Scale-Location), and influential points (Cook's distance). VIF values above 5-10 indicate multicollinearity concerns.

### Step 3: Logistic Regression

```r
# Fit logistic regression for poor diabetes control
logit_fit <- glm(poor_control ~ age + bmi + exercise_hours + medication +
                   smoking + sex + baseline_hba1c + num_comorbidities,
                  family = binomial(link = "logit"), data = health_data)

summary(logit_fit)

# Odds ratios with 95% CI
or_table <- tidy(logit_fit, conf.int = TRUE, exponentiate = TRUE) %>%
  mutate(across(where(is.numeric), ~ round(., 3)))
print(or_table)

# Model fit
cat("AIC:", AIC(logit_fit), "\n")
cat("Null deviance:", logit_fit$null.deviance, "\n")
cat("Residual deviance:", logit_fit$deviance, "\n")

# Likelihood ratio test (overall model)
anova(logit_fit, test = "LRT")
```

**Interpretation**: Odds ratios are the exponentiated logistic regression coefficients. An OR of 1.5 for BMI (per unit) means each 1-unit increase in BMI is associated with 50% higher odds of poor diabetes control. The AIC is used for model comparison; the deviance test assesses overall model significance.

### Step 4: Logistic Regression Diagnostics

```r
# Hosmer-Lemeshow goodness-of-fit test
# install.packages("ResourceSelection")
library(ResourceSelection)
hoslem.test(logit_fit$y, fitted(logit_fit), g = 10)

# ROC curve and AUC
# install.packages("pROC")
library(pROC)
roc_obj <- roc(health_data$poor_control, fitted(logit_fit))
plot(roc_obj, main = "ROC Curve: Logistic Regression for Poor Control")
cat("AUC:", auc(roc_obj), "\n")

# Influential observations
plot(cooks.distance(logit_fit), type = "h",
     main = "Cook's Distance", ylab = "Cook's D")
abline(h = 4 / nrow(health_data), col = "red", lty = 2)
```

**Interpretation**: The Hosmer-Lemeshow test assesses calibration; a non-significant p-value indicates adequate fit. The AUC measures discrimination; AUC > 0.7 indicates acceptable discrimination. Cook's distance flags influential observations.

### Step 5: Poisson Regression

```r
# Fit Poisson model for hospitalizations
pois_fit <- glm(hospitalizations ~ age + bmi + exercise_hours +
                  num_comorbidities + medication + smoking,
                 family = poisson(link = "log"), data = health_data)

summary(pois_fit)

# Rate ratios
rr_table <- tidy(pois_fit, conf.int = TRUE, exponentiate = TRUE) %>%
  mutate(across(where(is.numeric), ~ round(., 3)))
print(rr_table)

# Check overdispersion
dispersion <- sum(residuals(pois_fit, type = "pearson")^2) / pois_fit$df.residual
cat("Dispersion statistic:", round(dispersion, 3), "\n")
cat("Overdispersed:", dispersion > 1.5, "\n")
```

**Interpretation**: Exponentiated Poisson coefficients are rate ratios. An RR of 1.03 for age means each additional year of age is associated with a 3% higher hospitalization rate. The dispersion statistic should be near 1; values substantially above 1 indicate overdispersion, suggesting a negative binomial model may be more appropriate.

### Step 6: Negative Binomial Regression

```r
library(MASS)

# Fit negative binomial if overdispersion detected
nb_fit <- glm.nb(hospitalizations ~ age + bmi + exercise_hours +
                    num_comorbidities + medication + smoking,
                  data = health_data)

summary(nb_fit)

# Compare Poisson vs NB
cat("Poisson AIC:", AIC(pois_fit), "\n")
cat("NB AIC:", AIC(nb_fit), "\n")

# Likelihood ratio test for overdispersion
lr_stat <- 2 * (logLik(nb_fit) - logLik(pois_fit))
cat("LR test statistic:", lr_stat, "\n")
cat("P-value:", pchisq(as.numeric(lr_stat), df = 1, lower.tail = FALSE) / 2, "\n")
```

**Interpretation**: The negative binomial model adds a dispersion parameter to handle overdispersion. If AIC is lower for the NB model and the LR test is significant, prefer the negative binomial. The theta parameter indicates the degree of overdispersion (larger theta = closer to Poisson).

## Advanced Example

### Penalized Regression (LASSO, Ridge, Elastic Net)

```r
library(glmnet)

# Prepare design matrix
X <- model.matrix(hba1c_12m ~ age + bmi + exercise_hours + medication +
                    smoking + sex + baseline_hba1c + num_comorbidities +
                    age:bmi + age:medication + bmi:exercise_hours,
                  data = health_data)[, -1]
y <- health_data$hba1c_12m

# LASSO (alpha = 1)
cv_lasso <- cv.glmnet(X, y, alpha = 1, nfolds = 10)
plot(cv_lasso, main = "LASSO: Cross-Validation")
cat("Optimal lambda (min):", cv_lasso$lambda.min, "\n")
cat("Optimal lambda (1se):", cv_lasso$lambda.1se, "\n")

# Coefficients at lambda.1se (more parsimonious)
coef_lasso <- coef(cv_lasso, s = "lambda.1se")
print(coef_lasso)

# Ridge (alpha = 0)
cv_ridge <- cv.glmnet(X, y, alpha = 0, nfolds = 10)

# Elastic Net (alpha = 0.5)
cv_enet <- cv.glmnet(X, y, alpha = 0.5, nfolds = 10)

# Compare models
cat("\nCross-validated MSE:\n")
cat("  LASSO:", min(cv_lasso$cvm), "\n")
cat("  Ridge:", min(cv_ridge$cvm), "\n")
cat("  Elastic Net:", min(cv_enet$cvm), "\n")
```

**Interpretation**: LASSO performs variable selection by setting some coefficients to zero. The `lambda.1se` rule selects the simplest model within one standard error of the minimum CV error. Non-zero coefficients identify the most important predictors. Ridge shrinks but retains all variables. Elastic Net offers a compromise.

### Generalized Additive Models (GAMs)

```r
library(mgcv)

# Fit GAM with smooth terms for continuous predictors
gam_fit <- gam(hba1c_12m ~ s(age) + s(bmi) + s(exercise_hours) +
                 medication + smoking + sex + s(baseline_hba1c) +
                 num_comorbidities,
                data = health_data, method = "REML")

summary(gam_fit)

# Check effective degrees of freedom
cat("\nEffective degrees of freedom for smooth terms:\n")
for (i in seq_along(gam_fit$smooth)) {
  cat(sprintf("  %s: %.2f\n", gam_fit$smooth[[i]]$label,
              sum(gam_fit$edf[gam_fit$smooth[[i]]$first.para:gam_fit$smooth[[i]]$last.para])))
}

# Compare GAM vs linear model
cat("\nAIC comparison:")
cat("\n  Linear model:", AIC(lm_fit))
cat("\n  GAM:", AIC(gam_fit), "\n")
```

**Interpretation**: The GAM summary shows effective degrees of freedom (edf) for each smooth term. edf near 1 suggests a linear relationship; higher edf indicates non-linearity. The p-values test whether each smooth term is significantly different from zero. An AIC comparison with the linear model indicates whether the additional flexibility of the GAM is justified.

### Visualizing GAM Smooth Terms

```r
# Plot smooth functions
par(mfrow = c(2, 2))
plot(gam_fit, shade = TRUE, shade.col = "lightblue",
     residuals = TRUE, pch = 1, cex = 0.5)
par(mfrow = c(1, 1))

# Check GAM diagnostics
par(mfrow = c(2, 2))
gam.check(gam_fit)
par(mfrow = c(1, 1))
```

**Interpretation**: The partial effect plots show the estimated smooth relationship between each predictor and the outcome. The shaded region is the 95% CI. The rug plot along the x-axis shows the data distribution. gam.check() provides basis dimension checks and residual diagnostics.

## Visualization

### Coefficient Plot (Linear Model)

```r
library(broom)

coef_data <- tidy(lm_fit, conf.int = TRUE) %>%
  filter(term != "(Intercept)")

ggplot(coef_data, aes(x = estimate, y = reorder(term, estimate))) +
  geom_point(size = 3, color = "navy") +
  geom_errorbarh(aes(xmin = conf.low, xmax = conf.high), height = 0.2, color = "navy") +
  geom_vline(xintercept = 0, linestyle = "dashed", color = "red") +
  labs(x = "Coefficient Estimate (95% CI)", y = "",
       title = "Linear Regression: Predictors of HbA1c at 12 Months") +
  theme_minimal(base_size = 12)
```

### Odds Ratio Plot (Logistic Regression)

```r
or_data <- tidy(logit_fit, conf.int = TRUE, exponentiate = TRUE) %>%
  filter(term != "(Intercept)")

ggplot(or_data, aes(x = estimate, y = reorder(term, estimate))) +
  geom_point(size = 3, color = "darkgreen") +
  geom_errorbarh(aes(xmin = conf.low, xmax = conf.high), height = 0.2, color = "darkgreen") +
  geom_vline(xintercept = 1, linestyle = "dashed", color = "red") +
  scale_x_log10() +
  labs(x = "Odds Ratio (95% CI, log scale)", y = "",
       title = "Logistic Regression: Predictors of Poor Diabetes Control") +
  theme_minimal(base_size = 12)
```

### LASSO Regularization Path

```r
plot(glmnet(X, y, alpha = 1), xvar = "lambda", label = TRUE)
title("LASSO Regularization Path")
abline(v = log(cv_lasso$lambda.1se), lty = 2, col = "blue")
```

### Predicted vs Actual Plot

```r
health_data$predicted <- fitted(lm_fit)

ggplot(health_data, aes(x = predicted, y = hba1c_12m)) +
  geom_point(alpha = 0.3, color = "steelblue") +
  geom_abline(slope = 1, intercept = 0, color = "red", linewidth = 1) +
  labs(x = "Predicted HbA1c", y = "Observed HbA1c",
       title = "Predicted vs. Observed HbA1c at 12 Months") +
  theme_minimal(base_size = 12) +
  coord_equal()
```

## Tips and Best Practices

1. **Check assumptions before interpreting results**: Run diagnostic plots for every model. Residual patterns, non-normality, and heteroscedasticity can invalidate inference.

2. **Use VIF to check multicollinearity**: VIF > 5-10 for any predictor is a warning sign. Consider removing redundant predictors or using penalized methods.

3. **Report effect sizes, not just p-values**: Present coefficients with confidence intervals. In logistic regression, always report odds ratios. In Poisson, report rate ratios.

4. **Choose the right family for count data**: Start with Poisson, check overdispersion, and switch to negative binomial if needed. For excess zeros, consider zero-inflated models.

5. **Use AIC/BIC for model comparison, not R-squared**: R-squared always increases with more predictors. AIC/BIC penalize model complexity appropriately.

6. **Center and scale continuous predictors for penalized regression**: `glmnet` standardizes internally by default, but for interpretation, keep the original scale in mind.

7. **Be cautious with stepwise selection**: Automated stepwise procedures inflate type I error and produce unstable models. Pre-specify models based on clinical knowledge, or use penalized methods.

8. **Report the complete model**: Do not selectively report only significant predictors. This introduces bias and inflates the appearance of strong effects.

9. **Use `broom` for tidy output**: The `tidy()`, `glance()`, and `augment()` functions produce consistent, analysis-ready data frames from model objects.

10. **Validate with out-of-sample prediction**: Use cross-validation or a holdout test set to assess prediction performance. In-sample metrics (R-squared, AIC) can be overly optimistic.
