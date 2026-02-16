# Longitudinal / Repeated Measures — R Implementation

## Required Packages

```r
install.packages(c("lme4", "nlme", "geepack", "mmrm",
                   "JMbayes2", "lmerTest", "broom.mixed",
                   "ggplot2", "lattice"))

library(lme4)
library(nlme)
library(geepack)
library(mmrm)
library(lmerTest)
library(broom.mixed)
library(ggplot2)
```

- **lme4**: Modern implementation of LMMs and GLMMs (uses REML by default).
- **nlme**: Classic LMM package with flexible covariance structures (required for AR(1), CS, etc.).
- **geepack**: GEE implementation with robust standard errors.
- **mmrm**: Purpose-built MMRM package optimized for clinical trial analyses.
- **lmerTest**: Adds p-values (Satterthwaite, Kenward-Roger) to `lme4` models.
- **JMbayes2**: Bayesian joint models for longitudinal and survival data.
- **broom.mixed**: Tidy summaries for mixed-effects models.

## Example Dataset

We simulate a clinical trial with 300 patients randomized to Active or Placebo, measured at baseline and weeks 2, 4, 8, 12, and 16. The primary endpoint is a continuous symptom severity score (lower is better). Approximately 20% of patients drop out under a MAR mechanism.

```r
set.seed(42)
n_subj <- 300
visits <- c(0, 2, 4, 8, 12, 16)  # weeks
n_visits <- length(visits)

# Generate subject-level data
subj_data <- data.frame(
  subject = 1:n_subj,
  treatment = factor(rep(c("Placebo", "Active"), each = n_subj / 2)),
  age = round(rnorm(n_subj, 50, 12)),
  baseline_score = round(rnorm(n_subj, 60, 10), 1)
)

# Random effects: intercept and slope
re_cov <- matrix(c(25, -0.8, -0.8, 0.04), nrow = 2)
re <- MASS::mvrnorm(n_subj, mu = c(0, 0), Sigma = re_cov)

# Create long-format data
long_data <- expand.grid(subject = 1:n_subj, visit_week = visits)
long_data <- merge(long_data, subj_data, by = "subject")
long_data <- long_data[order(long_data$subject, long_data$visit_week), ]

# True trajectory: treatment effect increases over time
long_data$trt_effect <- ifelse(long_data$treatment == "Active",
                                -0.5 * long_data$visit_week, 0)
long_data$score <- long_data$baseline_score +
                   re[long_data$subject, 1] +
                   (-0.2 + re[long_data$subject, 2]) * long_data$visit_week +
                   long_data$trt_effect +
                   rnorm(nrow(long_data), 0, 3)
long_data$score <- round(long_data$score, 1)

# Set baseline score equal to actual baseline observation
long_data$baseline_score[long_data$visit_week == 0] <-
  long_data$score[long_data$visit_week == 0]

# Introduce MAR dropout (~20%)
for (s in 1:n_subj) {
  idx <- which(long_data$subject == s)
  for (v in 2:n_visits) {
    drop_prob <- plogis(-4 + 0.02 * long_data$score[idx[v - 1]] +
                         0.3 * (long_data$treatment[idx[1]] == "Placebo"))
    if (runif(1) < drop_prob) {
      long_data$score[idx[v:n_visits]] <- NA
      break
    }
  }
}

# Convert visit to factor for MMRM
long_data$visit_factor <- factor(long_data$visit_week)
long_data$subject <- factor(long_data$subject)

cat("Total observations:", nrow(long_data), "\n")
cat("Missing observations:", sum(is.na(long_data$score)), "\n")
cat("Proportion missing:", round(mean(is.na(long_data$score)), 3), "\n")
```

## Complete Worked Example

### Step 1: Exploratory Visualization

```r
library(ggplot2)

# Spaghetti plot with group means
ggplot(long_data, aes(x = visit_week, y = score, group = subject)) +
  geom_line(alpha = 0.1, color = "gray60") +
  stat_summary(aes(group = treatment, color = treatment),
               fun = mean, geom = "line", linewidth = 1.5) +
  stat_summary(aes(group = treatment, color = treatment),
               fun = mean, geom = "point", size = 3) +
  scale_color_manual(values = c("Active" = "steelblue", "Placebo" = "coral")) +
  labs(title = "Individual Trajectories and Group Means",
       x = "Week", y = "Symptom Score",
       color = "Treatment") +
  theme_minimal(base_size = 13)
```

**Output interpretation**: Individual trajectories (gray lines) show the variability between subjects. Group means (colored lines) reveal the average treatment effect over time. A widening gap between treatment and placebo over time indicates an increasing treatment benefit.

### Step 2: Linear Mixed-Effects Model (lme4)

```r
library(lme4)
library(lmerTest)

# Random intercept and slope model
lmm_fit <- lmer(score ~ treatment * visit_week + baseline_score +
                  (1 + visit_week | subject),
                data = long_data, REML = TRUE)

summary(lmm_fit)

# Type III tests with Satterthwaite degrees of freedom
anova(lmm_fit, type = 3, ddf = "Satterthwaite")

# Extract random effects variance components
VarCorr(lmm_fit)
```

**Output interpretation**: The fixed effects show the average intercept, slope, treatment difference at baseline, and treatment-by-time interaction (the key estimand: does the rate of change differ by treatment?). A significant negative interaction means the active group improves faster. The random effects show between-subject variability in intercepts and slopes. The correlation between random intercept and slope indicates whether patients with higher baseline scores change faster.

```r
# ICC calculation
vc <- as.data.frame(VarCorr(lmm_fit))
icc <- vc$vcov[1] / (vc$vcov[1] + vc$vcov[3])  # intercept / (intercept + residual)
cat("ICC:", round(icc, 3), "\n")
```

**Output interpretation**: A high ICC (e.g., > 0.5) indicates substantial within-subject correlation and confirms that repeated measures methods are necessary.

### Step 3: MMRM — The Clinical Trial Standard

```r
library(mmrm)

# MMRM with unstructured covariance (visit as categorical)
# Exclude baseline visit (use only post-baseline visits)
mmrm_data <- long_data[long_data$visit_week > 0, ]

mmrm_fit <- mmrm(
  formula = score ~ treatment * visit_factor + baseline_score +
            us(visit_factor | subject),
  data = mmrm_data,
  reml = TRUE
)

summary(mmrm_fit)

# LS means at each visit by treatment
library(emmeans)
emm <- emmeans(mmrm_fit, ~ treatment | visit_factor)
print(emm)

# Treatment contrast at each visit
contrasts <- pairs(emm, reverse = TRUE)
print(confint(contrasts))

# Primary estimand: treatment difference at week 16
summary(contrasts)[summary(contrasts)$visit_factor == "16", ]
```

**Output interpretation**: The MMRM output shows the estimated treatment difference at each visit. The primary interest is typically the contrast at the last visit (week 16). The unstructured covariance allows each visit pair to have a unique correlation, avoiding model misspecification. Under MAR, the MMRM provides valid likelihood-based inference using all available data (no need for imputation).

### Step 4: GEE (Generalized Estimating Equations)

```r
library(geepack)

# GEE with exchangeable working correlation
gee_fit <- geeglm(score ~ treatment * visit_week + baseline_score,
                  data = long_data,
                  id = subject,
                  family = gaussian,
                  corstr = "exchangeable",
                  std.err = "san.se")  # sandwich (robust) SE

summary(gee_fit)

# Compare with AR(1) working correlation
gee_ar1 <- geeglm(score ~ treatment * visit_week + baseline_score,
                   data = long_data,
                   id = subject,
                   family = gaussian,
                   corstr = "ar1",
                   std.err = "san.se")

summary(gee_ar1)
```

**Output interpretation**: GEE provides population-averaged treatment effects. The sandwich standard errors are robust to working correlation misspecification, so the choice of `corstr` affects efficiency but not consistency. The treatment-by-time interaction has the same interpretation as in the LMM for a linear model. Note that GEE requires MCAR for valid inference with missing data — use weighted GEE for MAR settings.

### Step 5: Different Covariance Structures with nlme

```r
library(nlme)

# Compound symmetry
fit_cs <- gls(score ~ treatment * visit_factor + baseline_score,
              data = mmrm_data,
              correlation = corCompSymm(form = ~ 1 | subject),
              na.action = na.omit)

# AR(1)
fit_ar1 <- gls(score ~ treatment * visit_factor + baseline_score,
               data = mmrm_data,
               correlation = corAR1(form = ~ visit_week | subject),
               na.action = na.omit)

# Unstructured
fit_un <- gls(score ~ treatment * visit_factor + baseline_score,
              data = mmrm_data,
              correlation = corSymm(form = ~ 1 | subject),
              weights = varIdent(form = ~ 1 | visit_factor),
              na.action = na.omit)

# Compare via AIC/BIC
cat("AIC comparison:\n")
cat("  CS:   ", AIC(fit_cs), "\n")
cat("  AR(1):", AIC(fit_ar1), "\n")
cat("  UN:   ", AIC(fit_un), "\n")
```

**Output interpretation**: AIC and BIC help select the most appropriate covariance structure. Lower values indicate better fit. In clinical trials, unstructured is preferred by convention (as recommended by FDA and EMA guidelines) because it avoids assumptions about the correlation pattern.

## Advanced Example

### Joint Model for Longitudinal and Survival Data

```r
library(JMbayes2)

# Add a survival outcome: time to study discontinuation
# (We'll create this from our dropout pattern)
surv_data <- data.frame(subject = 1:n_subj)
surv_data$treatment <- subj_data$treatment

# Determine event time and status
for (s in 1:n_subj) {
  subj_rows <- long_data[long_data$subject == s, ]
  last_obs <- max(subj_rows$visit_week[!is.na(subj_rows$score)])
  surv_data$time[s] <- last_obs
  surv_data$event[s] <- ifelse(last_obs < 16, 1, 0)  # 1 = dropout
}

# Fit the longitudinal submodel
lmm_jm <- lme(score ~ treatment * visit_week + baseline_score,
               random = ~ visit_week | subject,
               data = long_data[!is.na(long_data$score), ],
               control = lmeControl(opt = "optim"))

# Fit the survival submodel
library(survival)
surv_jm <- coxph(Surv(time, event) ~ treatment,
                 data = surv_data, x = TRUE)

# Fit the joint model
joint_fit <- jm(surv_jm, lmm_jm, time_var = "visit_week",
                n_iter = 3000L, n_burnin = 1000L)

summary(joint_fit)
```

**Output interpretation**: The joint model links the longitudinal trajectory to the dropout hazard. The association parameter ($\alpha$) indicates how the current value (or slope) of the longitudinal outcome affects dropout risk. A positive association means that worse scores (higher symptom severity) increase dropout hazard, confirming informative dropout. The treatment effect from the joint model is adjusted for this informative dropout, providing less biased estimates than standard MMRM under MNAR.

### Random Effects Visualization

```r
# Extract and plot random effects
re_estimates <- ranef(lmm_fit)$subject
re_df <- data.frame(
  intercept = re_estimates[, 1],
  slope = re_estimates[, 2],
  treatment = subj_data$treatment
)

ggplot(re_df, aes(x = intercept, y = slope, color = treatment)) +
  geom_point(alpha = 0.6, size = 2) +
  stat_ellipse(level = 0.95, linewidth = 1) +
  scale_color_manual(values = c("Active" = "steelblue", "Placebo" = "coral")) +
  labs(title = "Random Effects: Intercept vs Slope",
       x = "Random Intercept (baseline deviation)",
       y = "Random Slope (rate of change deviation)",
       color = "Treatment") +
  theme_minimal(base_size = 13)
```

**Output interpretation**: The scatter plot of random effects shows the individual-level variability. A negative correlation between intercept and slope means patients with higher baseline scores tend to improve more over time (regression to the mean). The 95% ellipses show the distribution of random effects by treatment group.

## Visualization

### Model-Predicted Trajectories

```r
# Predicted trajectories from MMRM
pred_data <- expand.grid(
  treatment = c("Placebo", "Active"),
  visit_factor = factor(c(2, 4, 8, 12, 16)),
  baseline_score = mean(long_data$baseline_score, na.rm = TRUE)
)

pred_data$predicted <- predict(mmrm_fit, newdata = pred_data, type = "response")

ggplot(pred_data, aes(x = as.numeric(as.character(visit_factor)),
                       y = predicted, color = treatment)) +
  geom_line(linewidth = 1.5) +
  geom_point(size = 3) +
  scale_color_manual(values = c("Active" = "steelblue", "Placebo" = "coral")) +
  labs(title = "MMRM-Predicted Mean Trajectories",
       x = "Week", y = "Predicted Symptom Score",
       color = "Treatment") +
  theme_minimal(base_size = 13)
```

### Forest Plot of Treatment Effects by Visit

```r
# Extract treatment contrasts at each visit
contrast_df <- as.data.frame(confint(contrasts))
contrast_df$visit <- as.numeric(as.character(contrast_df$visit_factor))

ggplot(contrast_df, aes(x = estimate, y = factor(visit))) +
  geom_vline(xintercept = 0, linetype = "dashed", color = "gray50") +
  geom_errorbarh(aes(xmin = lower.CL, xmax = upper.CL),
                 height = 0.2, color = "steelblue", linewidth = 1) +
  geom_point(size = 3, color = "darkblue") +
  labs(title = "Treatment Effect (Active - Placebo) at Each Visit",
       x = "Treatment Difference (95% CI)", y = "Week") +
  theme_minimal(base_size = 13)
```

**Output interpretation**: This forest plot shows the estimated treatment difference at each visit with 95% confidence intervals. An increasing magnitude over time with CIs excluding zero confirms a significant and growing treatment benefit. The primary endpoint is typically the estimate at the final visit (week 16).

### Residual Diagnostics

```r
# Residual diagnostics for the LMM
resid_data <- data.frame(
  fitted = fitted(lmm_fit),
  residual = residuals(lmm_fit),
  visit_week = long_data$visit_week[!is.na(long_data$score)]
)

p1 <- ggplot(resid_data, aes(x = fitted, y = residual)) +
  geom_point(alpha = 0.3, color = "steelblue") +
  geom_hline(yintercept = 0, linetype = "dashed") +
  geom_smooth(method = "loess", color = "coral", se = FALSE) +
  labs(title = "Residuals vs Fitted", x = "Fitted Values", y = "Residuals") +
  theme_minimal()

p2 <- ggplot(resid_data, aes(sample = residual)) +
  stat_qq(color = "steelblue", alpha = 0.5) +
  stat_qq_line(color = "coral") +
  labs(title = "Normal Q-Q Plot of Residuals") +
  theme_minimal()

library(patchwork)
p1 + p2
```

**Output interpretation**: The residual vs fitted plot should show random scatter around zero with constant variance. Systematic patterns suggest model misspecification (e.g., missing non-linear terms). The Q-Q plot assesses normality of residuals — deviations in the tails may indicate heavy-tailed distributions.

## Tips and Best Practices

1. **Use MMRM with unstructured covariance** as the primary analysis in confirmatory clinical trials. It is the standard recommended by FDA and EMA for continuous endpoints.
2. **Treat visit as a categorical variable** in MMRM to avoid assumptions about the functional form of change over time.
3. **Always include baseline as a covariate** (ANCOVA approach), not as a response. This improves power and reduces bias.
4. **Use REML for estimation** — it provides less biased variance component estimates than ML, especially with small samples.
5. **Use Kenward-Roger or Satterthwaite degrees of freedom** for inference in small samples. The default Wald-type inference from `lme4` can be anti-conservative.
6. **Check model convergence**: singular fits (boundary estimates) indicate overfitting of the random effects structure. Simplify if needed.
7. **GEE is appropriate for population-averaged inference** but requires MCAR for missing data. For clinical trials with dropout, MMRM is preferred.
8. **Consider joint models** when dropout may be informative (MNAR). This is especially relevant in oncology and chronic disease trials.
9. **Visualize individual trajectories** before modeling. Spaghetti plots reveal non-linear trends, outliers, and dropout patterns.
10. **Report the covariance structure, estimation method, degrees of freedom method, and the estimated treatment effect at the primary time point** per regulatory guidance.
