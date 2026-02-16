# Causal Inference — R Implementation

## Required Packages

```r
install.packages(c("MatchIt", "WeightIt", "cobalt", "twang", "AIPW",
                    "sandwich", "lmtest", "ggplot2", "dplyr", "tableone"))

library(MatchIt)
library(WeightIt)
library(cobalt)
library(AIPW)
library(sandwich)
library(lmtest)
library(ggplot2)
library(dplyr)
library(tableone)
```

## Example Dataset

We use a simulated clinical dataset representing an observational study comparing two treatments for hypertension. The dataset includes baseline confounders, treatment assignment (non-random), and a continuous outcome (systolic blood pressure at 6 months).

```r
set.seed(42)
n <- 1000

# Simulate confounders
age <- round(rnorm(n, 60, 10))
bmi <- round(rnorm(n, 28, 5), 1)
diabetes <- rbinom(n, 1, 0.3)
smoking <- rbinom(n, 1, 0.25)
baseline_sbp <- round(rnorm(n, 150, 15))
creatinine <- round(rnorm(n, 1.1, 0.3), 2)

# Treatment assignment depends on confounders (non-random)
lp_treatment <- -1.5 + 0.03 * age + 0.05 * bmi + 0.6 * diabetes +
  0.4 * smoking + 0.02 * baseline_sbp - 0.5 * creatinine
prob_treatment <- plogis(lp_treatment)
treatment <- rbinom(n, 1, prob_treatment)

# Outcome depends on treatment and confounders
sbp_6m <- 140 - 8 * treatment + 0.15 * age + 0.3 * bmi +
  3 * diabetes + 2 * smoking + 0.2 * baseline_sbp -
  2 * creatinine + rnorm(n, 0, 8)
sbp_6m <- round(sbp_6m, 1)

obs_data <- data.frame(
  age, bmi, diabetes, smoking, baseline_sbp, creatinine,
  treatment, sbp_6m
)

cat("Sample size:", nrow(obs_data), "\n")
cat("Treated:", sum(treatment), " Control:", sum(1 - treatment), "\n")
cat("True ATE: -8 mmHg\n")
```

## Complete Worked Example

### Step 1: Assess Baseline Imbalance

```r
library(tableone)

# Create Table 1
covariates <- c("age", "bmi", "diabetes", "smoking", "baseline_sbp", "creatinine")
tab1 <- CreateTableOne(vars = covariates, strata = "treatment", data = obs_data,
                        test = FALSE)
print(tab1, smd = TRUE)
```

**Interpretation**: The standardized mean differences (SMDs) show the degree of imbalance between treatment groups. SMDs > 0.1 indicate meaningful imbalance that should be addressed. In this simulated dataset, confounders are imbalanced because treatment assignment depends on them.

### Step 2: Naive (Unadjusted) Comparison

```r
# Unadjusted difference
naive_diff <- mean(obs_data$sbp_6m[obs_data$treatment == 1]) -
  mean(obs_data$sbp_6m[obs_data$treatment == 0])
cat("Naive difference:", round(naive_diff, 2), "mmHg\n")
cat("True ATE: -8 mmHg\n")
cat("Bias:", round(naive_diff - (-8), 2), "mmHg\n")
```

**Interpretation**: The naive comparison is biased because it does not account for confounding. Treated patients differ systematically from controls in ways that affect the outcome.

### Step 3: Propensity Score Estimation

```r
# Estimate propensity score using logistic regression
ps_model <- glm(treatment ~ age + bmi + diabetes + smoking + baseline_sbp + creatinine,
                 family = binomial, data = obs_data)
obs_data$ps <- predict(ps_model, type = "response")

# Check propensity score overlap
ggplot(obs_data, aes(x = ps, fill = factor(treatment))) +
  geom_density(alpha = 0.5) +
  labs(x = "Propensity Score", y = "Density", fill = "Treatment",
       title = "Propensity Score Distribution by Treatment Group") +
  scale_fill_manual(values = c("0" = "#E41A1C", "1" = "#377EB8"),
                     labels = c("Control", "Treated")) +
  theme_minimal()
```

**Interpretation**: Overlap of propensity score distributions is essential. If there are regions where one group has near-zero density, the positivity assumption is violated. Trimming or overlap weighting should be considered.

### Step 4: Propensity Score Matching with MatchIt

```r
library(MatchIt)

# 1:1 nearest-neighbor matching with caliper
m_out <- matchit(treatment ~ age + bmi + diabetes + smoking + baseline_sbp + creatinine,
                  data = obs_data,
                  method = "nearest",
                  distance = "glm",
                  caliper = 0.2,
                  ratio = 1,
                  replace = FALSE)

summary(m_out)

# Extract matched data
matched_data <- match.data(m_out)
cat("Matched sample size:", nrow(matched_data), "\n")

# Estimate ATT from matched data
att_matched <- lm(sbp_6m ~ treatment, data = matched_data, weights = weights)
coeftest(att_matched, vcov. = vcovCL, cluster = ~subclass)
cat("ATT estimate (matching):", round(coef(att_matched)["treatment"], 2), "mmHg\n")
```

**Interpretation**: Matching creates a pseudo-population where treated and control subjects are comparable on observed confounders. The treatment coefficient from the regression on matched data estimates the ATT. Using cluster-robust standard errors accounts for the matched-pair structure.

### Step 5: Assess Covariate Balance After Matching

```r
library(cobalt)

# Love plot showing balance before and after matching
love.plot(m_out,
          stats = c("m", "v"),
          thresholds = c(m = 0.1),
          abs = TRUE,
          var.order = "unadjusted",
          colors = c("red", "blue"),
          shapes = c("circle", "triangle"),
          title = "Covariate Balance: Before and After Matching")

# Balance table
bal.tab(m_out, un = TRUE, stats = c("m", "v"))
```

**Interpretation**: The Love plot shows standardized mean differences before and after matching. After successful matching, all SMDs should be below the threshold (typically 0.1). If imbalances persist, consider different matching specifications or alternative methods.

### Step 6: Inverse Probability of Treatment Weighting (IPTW)

```r
library(WeightIt)

# Estimate ATE weights
W <- weightit(treatment ~ age + bmi + diabetes + smoking + baseline_sbp + creatinine,
               data = obs_data,
               method = "ps",
               estimand = "ATE")

summary(W)

# Check balance
bal.tab(W, stats = c("m", "v"), thresholds = c(m = 0.1))

# Estimate ATE using weighted regression
obs_data$iptw <- W$weights
ate_iptw <- lm(sbp_6m ~ treatment, data = obs_data, weights = iptw)
coeftest(ate_iptw, vcov. = vcovHC, type = "HC3")
cat("ATE estimate (IPTW):", round(coef(ate_iptw)["treatment"], 2), "mmHg\n")
```

**Interpretation**: IPTW creates a pseudo-population where confounders are balanced across treatment groups. The weighted regression coefficient for treatment estimates the ATE. Robust standard errors (HC3) account for the weighting. Check for extreme weights; if present, use stabilized or trimmed weights.

### Step 7: Overlap Weighting

```r
# Overlap weighting targets ATO
W_overlap <- weightit(treatment ~ age + bmi + diabetes + smoking + baseline_sbp + creatinine,
                       data = obs_data,
                       method = "ps",
                       estimand = "ATO")

summary(W_overlap)
bal.tab(W_overlap, stats = c("m", "v"), thresholds = c(m = 0.1))

obs_data$ow <- W_overlap$weights
ate_ow <- lm(sbp_6m ~ treatment, data = obs_data, weights = ow)
coeftest(ate_ow, vcov. = vcovHC, type = "HC3")
cat("ATO estimate (Overlap Weighting):", round(coef(ate_ow)["treatment"], 2), "mmHg\n")
```

**Interpretation**: Overlap weighting naturally down-weights subjects with extreme propensity scores, addressing positivity issues without arbitrary trimming. The ATO estimand focuses on the subpopulation with the most overlap between treatment groups.

### Step 8: Doubly Robust Estimation (AIPW)

```r
library(AIPW)

# AIPW estimator
aipw_est <- AIPW$new(
  Y = obs_data$sbp_6m,
  A = obs_data$treatment,
  W = obs_data[, covariates],
  Q.SL.library = c("SL.glm", "SL.glm.interaction"),
  g.SL.library = c("SL.glm"),
  k_split = 5,
  verbose = FALSE
)

aipw_est$fit()
aipw_est$summary()
```

**Interpretation**: The AIPW estimator combines outcome modeling and propensity score weighting. It is consistent if either the outcome model or the propensity score model is correctly specified. The Super Learner ensemble can be used for both models to further protect against misspecification. The output includes ATE estimates with 95% confidence intervals.

## Advanced Example

### G-Computation

```r
# Outcome model
outcome_model <- lm(sbp_6m ~ treatment * (age + bmi + diabetes + smoking +
                                             baseline_sbp + creatinine),
                     data = obs_data)

# Predict under treatment and control for everyone
data_treat <- obs_data
data_treat$treatment <- 1
data_control <- obs_data
data_control$treatment <- 0

pred_treat <- predict(outcome_model, newdata = data_treat)
pred_control <- predict(outcome_model, newdata = data_control)

ate_gcomp <- mean(pred_treat) - mean(pred_control)
cat("ATE estimate (G-computation):", round(ate_gcomp, 2), "mmHg\n")

# Bootstrap for confidence interval
set.seed(123)
n_boot <- 1000
ate_boot <- numeric(n_boot)
for (b in 1:n_boot) {
  idx <- sample(1:n, n, replace = TRUE)
  boot_data <- obs_data[idx, ]
  mod_boot <- lm(sbp_6m ~ treatment * (age + bmi + diabetes + smoking +
                                          baseline_sbp + creatinine),
                  data = boot_data)
  d1 <- boot_data; d1$treatment <- 1
  d0 <- boot_data; d0$treatment <- 0
  ate_boot[b] <- mean(predict(mod_boot, d1)) - mean(predict(mod_boot, d0))
}
cat("95% CI:", round(quantile(ate_boot, c(0.025, 0.975)), 2), "\n")
```

**Interpretation**: G-computation estimates the ATE by standardizing over the confounder distribution. It relies on correct specification of the outcome model. Bootstrap provides non-parametric confidence intervals.

### Summary of All Estimates

```r
results <- data.frame(
  Method = c("Naive", "Matching (ATT)", "IPTW (ATE)", "Overlap Weighting (ATO)",
             "G-computation (ATE)", "AIPW (ATE)"),
  Estimate = round(c(naive_diff,
                      coef(att_matched)["treatment"],
                      coef(ate_iptw)["treatment"],
                      coef(ate_ow)["treatment"],
                      ate_gcomp,
                      aipw_est$estimates$risk_difference$Estimate), 2),
  True_ATE = -8
)
results$Bias <- results$Estimate - results$True_ATE
print(results)
```

**Interpretation**: Comparing estimates across methods provides a robustness check. If all well-specified methods give similar estimates close to the truth, confidence in the causal conclusion is strengthened. Discrepancies may indicate model misspecification or positivity issues.

## Visualization

### Propensity Score Balance Plot

```r
love.plot(W,
          stats = c("m"),
          abs = TRUE,
          thresholds = c(m = 0.1),
          colors = c("red", "blue"),
          title = "Covariate Balance: IPTW",
          var.order = "unadjusted")
```

### Weight Distribution

```r
ggplot(obs_data, aes(x = iptw, fill = factor(treatment))) +
  geom_histogram(bins = 50, alpha = 0.6, position = "identity") +
  labs(x = "IPTW Weight", y = "Count", fill = "Treatment",
       title = "Distribution of IPTW Weights") +
  scale_fill_manual(values = c("0" = "#E41A1C", "1" = "#377EB8")) +
  theme_minimal()
```

### Treatment Effect Comparison Plot

```r
results_plot <- results[results$Method != "Naive", ]

ggplot(results_plot, aes(x = Estimate, y = Method)) +
  geom_point(size = 4, color = "navy") +
  geom_vline(xintercept = -8, linetype = "dashed", color = "red", linewidth = 1) +
  annotate("text", x = -8.5, y = 0.6, label = "True ATE = -8", color = "red") +
  labs(x = "Estimated Treatment Effect (mmHg)", y = "",
       title = "Treatment Effect Estimates Across Methods") +
  theme_minimal(base_size = 12)
```

## Tips and Best Practices

1. **Start with a DAG**: Before any analysis, draw a DAG encoding your causal assumptions. This clarifies which variables to adjust for and which to avoid (colliders, mediators).

2. **Always check overlap**: Plot propensity score distributions by group. If there is poor overlap, use overlap weighting or trimming rather than forcing IPTW with extreme weights.

3. **Check balance, not just p-values**: Covariate balance (SMD < 0.1) is the goal of propensity score methods, not statistical significance of the propensity score model.

4. **Use doubly robust methods as default**: AIPW or TMLE provide robustness against single-model misspecification. Use Super Learner for the nuisance models.

5. **Report multiple estimands**: ATE, ATT, and ATO answer different questions. Be explicit about which estimand you are targeting and why.

6. **Conduct sensitivity analysis for unmeasured confounding**: Use methods like the E-value (VanderWeele and Ding, 2017) or Rosenbaum bounds to quantify how strong unmeasured confounding would need to be to explain away the observed effect.

7. **Avoid conditioning on post-treatment variables**: Do not adjust for variables that are affected by treatment (mediators or colliders) unless you are specifically doing mediation analysis.

8. **Bootstrap or sandwich variance**: Always use robust or bootstrapped standard errors with propensity score methods, as the weights induce correlation.

9. **Target trial emulation**: Frame your observational study as an emulation of a specific target trial. This clarifies the eligibility criteria, treatment definition, and time-zero.

10. **Cross-validate nuisance models**: When using machine learning for propensity score or outcome estimation, use cross-fitting (sample splitting) to avoid overfitting bias, as in AIPW with cross-validation.
