# Missing Data Methods — R Implementation

## Required Packages

```r
install.packages(c("mice", "Amelia", "naniar", "mitools", "VIM",
                   "finalfit", "missMethods"))

library(mice)
library(naniar)
library(VIM)
library(finalfit)
```

- **mice**: Multivariate Imputation by Chained Equations — the standard R package for MI.
- **naniar**: Tidy tools for exploring and visualizing missing data.
- **VIM**: Visualization and Imputation of Missing values.
- **Amelia**: Expectation-Maximization based multiple imputation (assumes multivariate normality).
- **mitools**: Tools for combining results from multiply imputed datasets.
- **finalfit**: Convenient clinical research reporting, including missing data summaries.

## Example Dataset

We simulate a clinical trial dataset with 500 patients randomized to treatment or placebo, with a continuous primary endpoint (change in HbA1c) measured at baseline and weeks 4, 8, 12, and 16. Approximately 25% of patients have missing week-16 outcomes, with missingness related to baseline HbA1c and treatment arm (MAR mechanism).

```r
set.seed(42)
n <- 500

trial_data <- data.frame(
  id        = 1:n,
  treatment = factor(rep(c("Placebo", "Active"), each = n/2)),
  age       = round(rnorm(n, 55, 10)),
  sex       = factor(sample(c("Male", "Female"), n, replace = TRUE)),
  bmi       = round(rnorm(n, 30, 5), 1),
  hba1c_bl  = round(rnorm(n, 8.5, 1.2), 1),
  hba1c_w4  = NA_real_,
  hba1c_w8  = NA_real_,
  hba1c_w12 = NA_real_,
  hba1c_w16 = NA_real_
)

# Simulate outcomes with treatment effect
for (i in 1:n) {
  trt_eff <- ifelse(trial_data$treatment[i] == "Active", -0.8, 0)
  trajectory <- trial_data$hba1c_bl[i] +
                cumsum(c(trt_eff/4, trt_eff/4, trt_eff/4, trt_eff/4)) +
                rnorm(4, 0, 0.3)
  trial_data$hba1c_w4[i]  <- round(trajectory[1], 1)
  trial_data$hba1c_w8[i]  <- round(trajectory[2], 1)
  trial_data$hba1c_w12[i] <- round(trajectory[3], 1)
  trial_data$hba1c_w16[i] <- round(trajectory[4], 1)
}

# Introduce MAR missingness (~25% at week 16, ~15% at week 12)
miss_prob_w16 <- plogis(-2 + 0.3 * (trial_data$hba1c_bl - 8.5) +
                         0.5 * (trial_data$treatment == "Placebo"))
trial_data$hba1c_w16[runif(n) < miss_prob_w16] <- NA

miss_prob_w12 <- plogis(-3 + 0.2 * (trial_data$hba1c_bl - 8.5))
trial_data$hba1c_w12[runif(n) < miss_prob_w12] <- NA
# If week 12 missing, week 16 must also be missing (monotone dropout)
trial_data$hba1c_w16[is.na(trial_data$hba1c_w12)] <- NA

cat("Missing at week 12:", sum(is.na(trial_data$hba1c_w12)), "/", n, "\n")
cat("Missing at week 16:", sum(is.na(trial_data$hba1c_w16)), "/", n, "\n")
```

## Complete Worked Example

### Step 1: Explore Missing Data Patterns

```r
library(naniar)

# Summary of missingness by variable
miss_var_summary(trial_data)

# Missingness by treatment group
trial_data %>%
  group_by(treatment) %>%
  miss_var_summary() %>%
  filter(variable %in% c("hba1c_w12", "hba1c_w16"))
```

**Output interpretation**: The summary shows the count and percentage of missing values per variable. If the placebo group has a higher rate of missing week-16 data, this supports a MAR mechanism (missingness related to treatment arm).

```r
# Visualize missingness pattern
library(VIM)
aggr(trial_data[, c("hba1c_bl", "hba1c_w4", "hba1c_w8",
                     "hba1c_w12", "hba1c_w16")],
     col = c("steelblue", "red"),
     numbers = TRUE, sortVars = TRUE,
     labels = c("Baseline", "Week 4", "Week 8", "Week 12", "Week 16"),
     cex.axis = 0.8,
     gap = 3,
     ylab = c("Proportion Missing", "Pattern"))
```

**Output interpretation**: The VIM aggregation plot shows (left) the proportion missing per variable and (right) the missingness patterns as a combination matrix. The most common pattern should be fully observed; the next should be monotone dropout (missing from a certain visit onward).

```r
# Upset plot for missingness patterns
gg_miss_upset(trial_data[, c("hba1c_w4", "hba1c_w8",
                              "hba1c_w12", "hba1c_w16")])
```

### Step 2: Complete Case Analysis (for comparison)

```r
# Complete case analysis: ANCOVA of week 16 change from baseline
trial_cc <- trial_data %>%
  filter(!is.na(hba1c_w16)) %>%
  mutate(change_w16 = hba1c_w16 - hba1c_bl)

cc_model <- lm(change_w16 ~ treatment + hba1c_bl, data = trial_cc)
summary(cc_model)
confint(cc_model)

cat("\nComplete cases used:", nrow(trial_cc), "of", n, "\n")
```

**Output interpretation**: The complete case analysis discards all patients with missing week-16 data. The treatment effect estimate may be biased because dropout is related to baseline severity and treatment. The effective sample size is reduced, widening confidence intervals.

### Step 3: Multiple Imputation with MICE

```r
library(mice)

# Set up imputation model
# Include all predictors that may be related to missingness or the outcome
imp_vars <- c("treatment", "age", "sex", "bmi", "hba1c_bl",
              "hba1c_w4", "hba1c_w8", "hba1c_w12", "hba1c_w16")

# Check default methods
ini <- mice(trial_data[, imp_vars], maxit = 0)
ini$method
ini$predictorMatrix

# Run MICE with PMM (predictive mean matching) for continuous variables
imp <- mice(trial_data[, imp_vars],
            m = 25,          # 25 imputations
            method = "pmm",  # predictive mean matching
            maxit = 20,      # iterations for convergence
            seed = 123,
            printFlag = FALSE)

# Check convergence
plot(imp, c("hba1c_w12", "hba1c_w16"))
```

**Output interpretation**: The convergence plots show the mean and standard deviation of imputed values across iterations for each imputation. Healthy convergence shows freely intermingled, random-looking streams without trends. If streams have not converged, increase `maxit`.

### Step 4: Analyze Imputed Datasets and Pool Results

```r
# Create change from baseline in each imputed dataset
imp_long <- complete(imp, action = "long", include = TRUE)
imp_long$change_w16 <- imp_long$hba1c_w16 - imp_long$hba1c_bl

# Convert back to mids object
imp_with_change <- as.mids(imp_long)

# Fit ANCOVA to each imputed dataset
fit_mi <- with(imp_with_change,
               lm(change_w16 ~ treatment + hba1c_bl))

# Pool results using Rubin's rules
pooled <- pool(fit_mi)
summary(pooled, conf.int = TRUE)

# Fraction of missing information
cat("\nFraction of missing information (FMI):\n")
print(pooled$pooled[, c("estimate", "fmi", "lambda")])
```

**Output interpretation**: The pooled treatment effect is the average across 25 imputed analyses. The total variance accounts for both within-imputation and between-imputation variability. The FMI indicates how much the missing data contributes to uncertainty. An FMI above 0.5 means the results are heavily influenced by the imputation model. Lambda represents the proportion of total variance attributable to missing data.

### Step 5: Compare Imputed vs Observed Distributions

```r
# Density plot comparing observed and imputed values
densityplot(imp, ~ hba1c_w16 | treatment,
            layout = c(2, 1),
            main = "Observed (blue) vs Imputed (red) HbA1c at Week 16")

# Strip plot
stripplot(imp, hba1c_w16 ~ treatment,
          main = "Imputed Values by Treatment Group")
```

**Output interpretation**: The density plot overlays the distributions of observed (blue) and imputed (red) values. Under MAR, the imputed distribution may differ from the observed distribution. Large discrepancies warrant investigation — they may indicate model misspecification or suggest that the MAR assumption is strained.

## Advanced Example

### Sensitivity Analysis: Tipping-Point Analysis

```r
# Tipping-point analysis: shift imputed values in the active treatment arm
# by delta to explore MNAR sensitivity
deltas <- seq(0, 1.5, by = 0.1)
tipping_results <- data.frame(delta = deltas,
                               estimate = NA,
                               lower = NA,
                               upper = NA,
                               p_value = NA)

for (d in seq_along(deltas)) {
  imp_tp <- complete(imp, action = "long", include = TRUE)
  imp_tp$change_w16 <- imp_tp$hba1c_w16 - imp_tp$hba1c_bl

  # Add delta only to imputed values in active arm
  imputed_active <- imp_tp$.imp > 0 &
                    is.na(trial_data$hba1c_w16[imp_tp$.id]) &
                    imp_tp$treatment == "Active"
  imp_tp$change_w16[imputed_active] <-
    imp_tp$change_w16[imputed_active] + deltas[d]

  imp_tp_mids <- as.mids(imp_tp)
  fit_tp <- with(imp_tp_mids,
                 lm(change_w16 ~ treatment + hba1c_bl))
  pooled_tp <- summary(pool(fit_tp), conf.int = TRUE)

  trt_row <- pooled_tp[pooled_tp$term == "treatmentActive", ]
  tipping_results$estimate[d] <- trt_row$estimate
  tipping_results$lower[d]    <- trt_row$`2.5 %`
  tipping_results$upper[d]    <- trt_row$`97.5 %`
  tipping_results$p_value[d]  <- trt_row$p.value
}

print(tipping_results)

# Find tipping point
tp_idx <- which(tipping_results$p_value > 0.05)[1]
cat("\nTipping point: delta =", tipping_results$delta[tp_idx], "\n")
cat("At this delta, treatment effect is no longer significant.\n")

# Plot tipping point analysis
plot(tipping_results$delta, tipping_results$estimate,
     type = "b", pch = 19, col = "steelblue",
     xlab = "Delta (added to imputed active arm)",
     ylab = "Treatment Effect (change in HbA1c)",
     main = "Tipping Point Analysis",
     ylim = range(c(tipping_results$lower, tipping_results$upper)))
polygon(c(tipping_results$delta, rev(tipping_results$delta)),
        c(tipping_results$lower, rev(tipping_results$upper)),
        col = adjustcolor("steelblue", 0.2), border = NA)
abline(h = 0, lty = 2, col = "red")
if (!is.na(tp_idx)) {
  abline(v = tipping_results$delta[tp_idx], lty = 3, col = "red")
  text(tipping_results$delta[tp_idx], max(tipping_results$upper),
       paste("Tipping point =", tipping_results$delta[tp_idx]),
       pos = 4, col = "red")
}
```

**Output interpretation**: The tipping-point analysis shows how the treatment effect changes as imputed values for treatment-arm dropouts are shifted toward worse outcomes (positive delta for HbA1c = less improvement). The tipping point is the delta value where the treatment effect first becomes non-significant (p > 0.05). A large tipping point indicates robustness to departures from MAR; a small tipping point suggests fragile conclusions.

### Reference-Based Imputation (Jump to Reference)

```r
# Jump-to-Reference (J2R) imputation
# After dropout, impute values as if patient were in placebo group
library(mice)

# Custom imputation function for J2R
# Conceptual approach: modify the imputation model for active arm dropouts
# to use the reference (placebo) group distribution

# First, fit separate models for each group
placebo_data <- trial_data[trial_data$treatment == "Placebo",
                           c("hba1c_bl", "hba1c_w4", "hba1c_w8",
                             "hba1c_w12", "hba1c_w16")]

# Impute placebo arm normally
imp_placebo <- mice(placebo_data, m = 25, method = "pmm",
                    maxit = 20, seed = 456, printFlag = FALSE)

# For active arm dropouts, impute using the placebo model
# This is a simplified illustration; in practice, use the rbmi package
# for proper reference-based imputation

cat("Note: For production-quality reference-based imputation,\n")
cat("use the 'rbmi' package (Gower-Page et al.) which implements\n")
cat("J2R, CIR, and copy reference methods with proper variance estimation.\n")
```

## Visualization

### Comprehensive Missingness Summary

```r
library(naniar)

# Missing data summary plot
gg_miss_var(trial_data, show_pct = TRUE) +
  labs(title = "Percentage Missing by Variable",
       y = "Variable", x = "% Missing")

# Missing data matrix (heatmap)
vis_miss(trial_data[, c("hba1c_bl", "hba1c_w4", "hba1c_w8",
                         "hba1c_w12", "hba1c_w16")],
         cluster = TRUE, sort_miss = TRUE) +
  labs(title = "Missing Data Pattern (clustered)")

# Relationship between missingness and observed data
ggplot(trial_data, aes(x = hba1c_bl,
                        fill = factor(is.na(hba1c_w16)))) +
  geom_density(alpha = 0.5) +
  scale_fill_manual(values = c("steelblue", "coral"),
                    labels = c("Observed", "Missing"),
                    name = "Week 16 Status") +
  labs(title = "Baseline HbA1c by Week-16 Missingness Status",
       x = "Baseline HbA1c", y = "Density") +
  facet_wrap(~ treatment)
```

**Output interpretation**: The density plot comparing baseline HbA1c between those with observed and missing week-16 outcomes reveals whether missingness is related to baseline severity. If the distributions differ, this supports a MAR mechanism (missingness depends on observed baseline values) and underscores the need for MI rather than complete case analysis.

### Flux Plot for Variable Selection

```r
# Flux plot to identify which variables to include in imputation model
flux_result <- flux(trial_data[, imp_vars])
print(flux_result)

# Variables with high outflux and low influx are good auxiliary variables
# Variables with high influx need to be imputed carefully
```

**Output interpretation**: The flux analysis identifies variables that are strong predictors of other variables (high outflux) and should be included in the imputation model, versus variables that depend on others (high influx) and need careful imputation.

## Tips and Best Practices

1. **Always visualize missingness patterns** before choosing a method. Look for monotone vs non-monotone patterns, and examine whether missingness is associated with observed variables.
2. **Include auxiliary variables** in the imputation model: variables that predict the outcome or missingness improve the plausibility of MAR and increase efficiency.
3. **Use predictive mean matching (PMM)** for continuous variables — it preserves the data distribution and avoids implausible values.
4. **Use at least m = 20 imputations**; more when the fraction of missing information is high. The old rule of m = 5 is outdated.
5. **Check convergence** of the MICE algorithm using trace plots. Non-convergent chains require more iterations.
6. **Always perform sensitivity analysis** — tipping-point analysis is practical and interpretable for regulators.
7. **Do not impute the outcome in a time-to-event analysis** — use appropriate censoring methods instead.
8. **Include the outcome variable and all analysis model covariates** in the imputation model; omitting the outcome leads to attenuated associations.
9. **For clinical trials**, align missing data handling with the estimand: treatment policy estimands may require reference-based imputation; hypothetical estimands may use MAR-based MI.
10. **Report the missing data rate, mechanism assessment, imputation method, number of imputations, and FMI** in all publications per STROBE/CONSORT guidelines.
