# Subgroup Analysis and Treatment Effect Heterogeneity (HTE) — R Implementation

## Required Packages

```r
install.packages(c("grf", "survival", "ggplot2", "dplyr", "gridExtra"))
# For BART-based causal inference:
install.packages("bartCause")
# For personalized medicine:
install.packages("personalized")
```

## Example Dataset

We simulate a clinical trial for a cardiovascular drug with treatment effect heterogeneity
driven by age, baseline LDL cholesterol, and diabetes status.

```r
set.seed(42)
n <- 1000

# Patient characteristics
age <- rnorm(n, mean = 62, sd = 10)
ldl_baseline <- rnorm(n, mean = 150, sd = 35)
diabetes <- rbinom(n, 1, prob = 0.3)
female <- rbinom(n, 1, prob = 0.45)
sbp <- rnorm(n, mean = 140, sd = 18)

# Treatment assignment (1:1 randomization)
treatment <- rbinom(n, 1, prob = 0.5)

# True CATE: treatment benefit increases with higher LDL and in diabetics
true_cate <- -5 - 0.1 * (ldl_baseline - 150) - 4 * diabetes

# Outcome: change in LDL at 12 weeks (lower is better)
noise <- rnorm(n, 0, 12)
y <- 0.3 * age + 0.4 * ldl_baseline + 5 * diabetes + treatment * true_cate + noise

trial_data <- data.frame(
  y, treatment, age, ldl_baseline, diabetes, female, sbp
)
head(trial_data)
```

## Complete Worked Example

### Step 1: Classical Subgroup Analysis with Interaction Tests

```r
# Test treatment-by-diabetes interaction
model_interaction <- lm(y ~ treatment * diabetes + age + ldl_baseline + female + sbp,
                        data = trial_data)
summary(model_interaction)

# Interpretation: The treatment:diabetes coefficient estimates the additional
# treatment effect in diabetic patients compared to non-diabetics.
# A significant negative coefficient means diabetics benefit more from treatment.
```

### Step 2: Forest Plot for Subgroup Effects

```r
library(ggplot2)
library(dplyr)

# Define subgroups
trial_data$age_group <- ifelse(trial_data$age >= 65, "Age >= 65", "Age < 65")
trial_data$ldl_group <- ifelse(trial_data$ldl_baseline >= 160, "LDL >= 160", "LDL < 160")
trial_data$diabetes_group <- ifelse(trial_data$diabetes == 1, "Diabetic", "Non-diabetic")
trial_data$sex_group <- ifelse(trial_data$female == 1, "Female", "Male")

subgroup_vars <- c("age_group", "ldl_group", "diabetes_group", "sex_group")

# Estimate treatment effect in each subgroup
forest_data <- do.call(rbind, lapply(subgroup_vars, function(var) {
  levels <- unique(trial_data[[var]])
  do.call(rbind, lapply(levels, function(lev) {
    subset_data <- trial_data[trial_data[[var]] == lev, ]
    fit <- lm(y ~ treatment, data = subset_data)
    ci <- confint(fit, "treatment", level = 0.95)
    data.frame(
      subgroup = var,
      level = lev,
      estimate = coef(fit)["treatment"],
      lower = ci[1],
      upper = ci[2],
      n = nrow(subset_data)
    )
  }))
}))

# Overall effect
fit_overall <- lm(y ~ treatment, data = trial_data)
ci_overall <- confint(fit_overall, "treatment", level = 0.95)
overall_row <- data.frame(
  subgroup = "Overall", level = "All patients",
  estimate = coef(fit_overall)["treatment"],
  lower = ci_overall[1], upper = ci_overall[2], n = nrow(trial_data)
)
forest_data <- rbind(overall_row, forest_data)
forest_data$label <- paste0(forest_data$level, " (n=", forest_data$n, ")")
forest_data$label <- factor(forest_data$label, levels = rev(forest_data$label))

# Forest plot
ggplot(forest_data, aes(x = estimate, y = label)) +
  geom_point(size = 3) +
  geom_errorbarh(aes(xmin = lower, xmax = upper), height = 0.2) +
  geom_vline(xintercept = 0, linetype = "dashed", color = "grey50") +
  labs(x = "Treatment Effect (change in LDL)", y = "",
       title = "Subgroup Forest Plot: Treatment Effect on LDL Change") +
  theme_minimal(base_size = 12)
```

**Interpretation**: The forest plot displays point estimates and 95% confidence intervals for
the treatment effect within each subgroup. Overlapping intervals suggest no significant
heterogeneity. The key question is whether any subgroup's CI fails to overlap with the overall
estimate, suggesting differential benefit.

### Step 3: Causal Forest for CATE Estimation

```r
library(grf)

X <- as.matrix(trial_data[, c("age", "ldl_baseline", "diabetes", "female", "sbp")])
W <- trial_data$treatment
Y <- trial_data$y

# Fit causal forest
cf <- causal_forest(X, Y, W,
                    num.trees = 2000,
                    honesty = TRUE,
                    seed = 42)

# Estimate CATE for each individual
cate_estimates <- predict(cf, estimate.variance = TRUE)
trial_data$cate_hat <- cate_estimates$predictions
trial_data$cate_se <- sqrt(cate_estimates$variance.estimates)

# Summary of estimated CATEs
summary(trial_data$cate_hat)
# Expect negative values (treatment reduces LDL), with more negative
# values indicating greater treatment benefit.

# Variable importance: which covariates drive heterogeneity?
var_importance <- variable_importance(cf)
importance_df <- data.frame(
  variable = c("age", "ldl_baseline", "diabetes", "female", "sbp"),
  importance = as.numeric(var_importance)
)
importance_df <- importance_df[order(-importance_df$importance), ]
print(importance_df)
# Expect ldl_baseline and diabetes to rank highest, matching the true DGP.
```

### Step 4: Test for Overall Heterogeneity

```r
# Omnibus test: is there any treatment effect heterogeneity?
test_calibration(cf)

# Output interpretation:
# - "mean.forest.prediction" tests whether the average CATE differs from zero.
# - "differential.forest.prediction" tests whether the heterogeneity is real.
#   A significant coefficient (p < 0.05) indicates that the causal forest is
#   capturing genuine variation in treatment effects, not just noise.
```

## Advanced Example

### CATE Estimation with BART

```r
library(bartCause)

# Fit BART causal model
bart_fit <- bartc(
  response = y,
  treatment = treatment,
  confounders = cbind(age, ldl_baseline, diabetes, female, sbp),
  data = trial_data,
  n.samples = 500L,
  n.burn = 200L,
  n.chains = 4L,
  seed = 42,
  verbose = FALSE
)

# Summary of treatment effect
summary(bart_fit)

# Individual treatment effects with posterior uncertainty
ite_samples <- extract(bart_fit, type = "ite")  # n.samples x n matrix
ite_means <- colMeans(ite_samples)
ite_lower <- apply(ite_samples, 2, quantile, 0.025)
ite_upper <- apply(ite_samples, 2, quantile, 0.975)

trial_data$bart_ite <- ite_means
trial_data$bart_lower <- ite_lower
trial_data$bart_upper <- ite_upper

# Compare BART and causal forest CATE estimates
cor(trial_data$cate_hat, trial_data$bart_ite)
# High correlation indicates agreement between methods.
```

### Quartile-Based Heterogeneity Analysis (PATH Approach)

```r
# Estimate baseline risk (prognostic score) using control group only
control_data <- trial_data[trial_data$treatment == 0, ]
risk_model <- lm(y ~ age + ldl_baseline + diabetes + female + sbp, data = control_data)
trial_data$risk_score <- predict(risk_model, newdata = trial_data)

# Divide into risk quartiles
trial_data$risk_quartile <- cut(trial_data$risk_score,
                                 breaks = quantile(trial_data$risk_score, probs = 0:4/4),
                                 include.lowest = TRUE, labels = 1:4)

# Treatment effect by risk quartile
risk_effects <- trial_data %>%
  group_by(risk_quartile) %>%
  summarise(
    n = n(),
    effect = mean(y[treatment == 1]) - mean(y[treatment == 0]),
    se = sqrt(var(y[treatment == 1])/sum(treatment == 1) +
              var(y[treatment == 0])/sum(treatment == 0)),
    .groups = "drop"
  ) %>%
  mutate(lower = effect - 1.96 * se, upper = effect + 1.96 * se)

print(risk_effects)
# Interpretation: If the treatment effect (absolute scale) increases with
# baseline risk, this supports risk-based HTE.
```

## Visualization

```r
library(ggplot2)
library(gridExtra)

# Plot 1: Distribution of estimated CATEs
p1 <- ggplot(trial_data, aes(x = cate_hat)) +
  geom_histogram(bins = 40, fill = "steelblue", alpha = 0.7) +
  geom_vline(xintercept = mean(trial_data$cate_hat), color = "red", linetype = "dashed") +
  labs(x = "Estimated CATE", y = "Count",
       title = "Distribution of Individual Treatment Effects (Causal Forest)") +
  theme_minimal()

# Plot 2: CATE vs LDL baseline (key effect modifier)
p2 <- ggplot(trial_data, aes(x = ldl_baseline, y = cate_hat, color = factor(diabetes))) +
  geom_point(alpha = 0.4, size = 1.5) +
  geom_smooth(method = "loess", se = TRUE) +
  labs(x = "Baseline LDL (mg/dL)", y = "Estimated CATE",
       color = "Diabetes", title = "CATE by Baseline LDL and Diabetes Status") +
  theme_minimal()

# Plot 3: Variable importance
p3 <- ggplot(importance_df, aes(x = reorder(variable, importance), y = importance)) +
  geom_col(fill = "darkorange") +
  coord_flip() +
  labs(x = "", y = "Variable Importance",
       title = "Causal Forest Variable Importance") +
  theme_minimal()

# Plot 4: Treatment effect by risk quartile
p4 <- ggplot(risk_effects, aes(x = risk_quartile, y = effect)) +
  geom_point(size = 3) +
  geom_errorbar(aes(ymin = lower, ymax = upper), width = 0.15) +
  geom_hline(yintercept = 0, linetype = "dashed", color = "grey50") +
  labs(x = "Baseline Risk Quartile", y = "Treatment Effect",
       title = "Treatment Effect by Baseline Risk Quartile (PATH)") +
  theme_minimal()

grid.arrange(p1, p2, p3, p4, ncol = 2)
```

## Tips and Best Practices

1. **Pre-specify subgroups**: In confirmatory trials, all subgroup analyses should be listed in
   the statistical analysis plan before unblinding. Limit to 5-10 clinically motivated subgroups.

2. **Always test the interaction**: Comparing p-values within subgroups is not a valid test of
   heterogeneity. A treatment can be significant in one subgroup and not another even when the
   true effect is identical. Always test the treatment-by-subgroup interaction.

3. **Correct for multiplicity**: If testing many subgroups, apply Bonferroni or FDR correction
   to interaction tests.

4. **Use both relative and absolute scales**: Report treatment effects on both scales. Risk-based
   HTE on the absolute scale is often more clinically relevant.

5. **Validate with causal forests**: Use `test_calibration()` to verify the causal forest captures
   real heterogeneity. Check that the "differential.forest.prediction" is significant.

6. **Sample size matters**: Causal forests and BART need at least 500-1000 observations to estimate
   CATE reliably. For classical interaction tests, you need roughly 4 times the sample size needed
   for the main effect.

7. **Beware overfitting**: ML-based CATE estimators can overfit. Always use honest estimation
   (separate tree-building and estimation samples) and evaluate on held-out data.

8. **Replicate findings**: Exploratory HTE results should be replicated in an independent dataset.
   Never make definitive treatment recommendations based on a single exploratory HTE analysis.
