# Sensitivity Analysis — R Implementation

## Required Packages

```r
install.packages(c("EValue", "episensr", "sensemakr", "rbounds",
                   "ggplot2", "dplyr"))

library(EValue)
library(episensr)
library(sensemakr)
library(rbounds)
library(ggplot2)
library(dplyr)
```

- **EValue**: Computes E-values for unmeasured confounding sensitivity analysis.
- **episensr**: Comprehensive quantitative bias analysis (unmeasured confounding, misclassification, selection bias, probabilistic bias analysis).
- **sensemakr**: Sensitivity analysis for regression using the omitted variable bias framework (Cinelli and Hazlett).
- **rbounds**: Rosenbaum bounds for matched observational studies.

## Example Dataset

We use data from a simulated observational study of statin use and cardiovascular (CV) event risk in 5000 older adults, adjusted for age, sex, diabetes, and hypertension. The concern is unmeasured confounding by healthy user behavior (e.g., exercise, diet).

```r
set.seed(42)
n <- 5000

obs_data <- data.frame(
  id = 1:n,
  statin_use = rbinom(n, 1, 0.35),
  age = round(rnorm(n, 65, 10)),
  male = rbinom(n, 1, 0.55),
  diabetes = rbinom(n, 1, 0.25),
  hypertension = rbinom(n, 1, 0.40),
  smoker = rbinom(n, 1, 0.20)
)

# Simulate CV event (~12% rate)
log_odds <- -3.0 + 0.03 * obs_data$age + 0.4 * obs_data$male +
            0.6 * obs_data$diabetes + 0.5 * obs_data$hypertension +
            0.8 * obs_data$smoker - 0.5 * obs_data$statin_use
obs_data$cv_event <- rbinom(n, 1, plogis(log_odds))

# Fit adjusted logistic regression (omitting smoker — unmeasured confounder)
model <- glm(cv_event ~ statin_use + age + male + diabetes + hypertension,
             family = binomial, data = obs_data)
summary(model)

# Extract adjusted OR for statin use
or_statin <- exp(coef(model)["statin_use"])
ci_statin <- exp(confint.default(model)["statin_use", ])
cat("Adjusted OR for statins:", round(or_statin, 3), "\n")
cat("95% CI:", round(ci_statin, 3), "\n")
```

The adjusted odds ratio for statin use indicates a protective effect against CV events, but smoking status is unmeasured.

## Complete Worked Example

### Step 1: E-Value Computation

```r
library(EValue)

# Convert OR to approximate RR for a common outcome
# For rare outcomes, OR approximates RR directly
# For common outcomes, use the conversion

# E-value for the point estimate
eval_result <- evalues.OR(
  est = or_statin,
  lo = ci_statin[1],
  hi = ci_statin[2],
  rare = FALSE  # CV events ~12%, not rare
)

print(eval_result)

# Visualize the E-value
plot(eval_result,
     main = "E-Value for Statin-CV Event Association")
```

**Output interpretation**: The E-value for the point estimate tells you the minimum strength of association (on the RR scale) that an unmeasured confounder would need with both statin use and CV events to fully explain away the observed protective effect. The E-value for the confidence interval lower bound tells you what is needed to shift the CI to include the null. If the E-value is large (e.g., > 3), it means only a very strong confounder could explain the result. If it is small (e.g., < 1.5), even modest unmeasured confounding could nullify the finding.

```r
# Interpret against known confounders
# Smoking: RR with CV events ~2.0, RR with statin use ~0.7
# So smoking is plausible as an unmeasured confounder
cat("\nComparison with known confounder magnitudes:\n")
cat("Smoking-CV events RR: ~2.0\n")
cat("Smoking-statin use RR: ~1.4 (inverse of 0.7)\n")
cat("Product: ~2.8\n")
cat("This is ", ifelse(2.8 > eval_result[2, 1], "above", "below"),
    " the E-value threshold.\n")
```

### Step 2: Quantitative Bias Analysis — Unmeasured Confounding

```r
library(episensr)

# Create 2x2 table from the data
tab <- table(obs_data$statin_use, obs_data$cv_event)
print(tab)

# QBA for unmeasured confounding
# Assume unmeasured confounder (smoking):
# - Prevalence in statin users: 15%
# - Prevalence in non-statin users: 25%
# - OR with CV event: 2.0 (after adjusting for measured covariates)

conf_result <- confounders(
  tab,
  type = "OR",
  bias_parms = c(0.25, 0.15, 2.0)
  # c(prev_unexposed, prev_exposed, OR_confounder_outcome)
)

print(conf_result)
```

**Output interpretation**: The QBA output shows the crude OR, the adjusted OR after accounting for the hypothesized unmeasured confounder, and the percent change. If the adjusted OR is substantially closer to 1.0, the unmeasured confounder meaningfully weakens the finding. If the adjusted OR remains below 1.0 (protective), the result is robust to this specific confounding scenario.

### Step 3: QBA — Outcome Misclassification

```r
# Suppose CV events are ascertained from claims data with imperfect sensitivity
# Sensitivity = 85% (15% of true events are missed)
# Specificity = 98% (2% of non-events are falsely classified as events)
# Assume non-differential misclassification

misclass_result <- misclassification(
  tab,
  type = "outcome",
  bias_parms = c(0.85, 0.85, 0.98, 0.98)
  # c(Se_exposed, Se_unexposed, Sp_exposed, Sp_unexposed)
)

print(misclass_result)
```

**Output interpretation**: Non-differential outcome misclassification typically biases the odds ratio toward the null. The corrected estimate should be further from 1.0 (stronger protective effect). If sensitivity or specificity is differential by exposure status, the bias direction depends on the specific pattern.

### Step 4: Probabilistic Bias Analysis

```r
# Probabilistic bias analysis for unmeasured confounding
# Specify distributions for bias parameters
set.seed(789)

pba_result <- probsens.conf(
  tab,
  type = "OR",
  reps = 50000,
  prev.exp = list("trapezoidal", c(0.10, 0.15, 0.20, 0.25)),
  prev.nexp = list("trapezoidal", c(0.20, 0.25, 0.30, 0.35)),
  risk = list("trapezoidal", c(1.5, 1.8, 2.2, 2.5))
)

print(pba_result)
```

**Output interpretation**: Probabilistic bias analysis produces a distribution of bias-corrected estimates. The median adjusted OR and the 2.5th/97.5th percentiles form a "systematic error interval" that accounts for both random error and the specified bias. This is more informative than a single bias-corrected estimate because it reflects uncertainty in the bias parameters.

```r
# Plot the distribution of bias-adjusted estimates
plot(pba_result,
     main = "Probabilistic Bias Analysis: Adjusted OR Distribution")
```

### Step 5: Sensemakr — Omitted Variable Bias Framework

```r
library(sensemakr)

# Fit the linear probability model (for sensemakr; requires continuous outcome)
# Alternative: use the logistic model coefficients
lm_model <- lm(cv_event ~ statin_use + age + male + diabetes + hypertension,
               data = obs_data)

# Sensitivity analysis
sens <- sensemakr(
  model = lm_model,
  treatment = "statin_use",
  benchmark_covariates = c("diabetes", "hypertension"),
  kd = c(1, 2, 3),  # confounder strength as 1x, 2x, 3x the benchmark
  ky = c(1, 2, 3)
)

print(sens)

# Summary with formal bounds
summary(sens)

# Contour plot: R-squared of confounder with treatment and outcome
plot(sens,
     sensitivity.of = "estimate",
     main = "Sensitivity Contour Plot: Statin Use")

# Extreme scenario plot
plot(sens,
     sensitivity.of = "t-value",
     main = "Sensitivity of t-statistic to Confounding")
```

**Output interpretation**: The contour plot shows combinations of confounder partial R-squared values (with treatment and outcome) that would reduce the treatment effect to zero. Benchmark covariates (diabetes, hypertension) provide calibration: the plot shows where 1x, 2x, or 3x confounders of similar strength to these known covariates would fall. If these benchmark points are far from the zero-effect contour, the result is robust. The robustness value (RV) gives the minimum R-squared an omitted variable must have with both treatment and outcome to change the conclusion.

## Advanced Example

### Rosenbaum Bounds for Matched Studies

```r
library(rbounds)

# Simulate a matched cohort study (propensity score matched)
# 200 matched pairs, binary outcome
set.seed(321)
n_pairs <- 200

# Simulate discordant pair outcomes under a mild treatment effect
# In Rosenbaum's framework, we need the number of discordant pairs
# and the distribution of treated/control successes in those pairs

# Matched pair data (Wilcoxon signed rank test format)
# Difference in outcome for each pair
pair_diffs <- c(
  rep(1, 85),   # treated had event, control did not
  rep(-1, 55),  # control had event, treated did not
  rep(0, 60)    # concordant pairs (both event or both no event)
)

# McNemar test (unadjusted)
b <- 85  # discordant pairs where treated = 1, control = 0
c <- 55  # discordant pairs where treated = 0, control = 1
mcnemar_stat <- (b - c)^2 / (b + c)
mcnemar_p <- 1 - pchisq(mcnemar_stat, 1)
cat("McNemar test: chi2 =", mcnemar_stat, ", p =", mcnemar_p, "\n")

# Rosenbaum bounds for binary outcome (McNemar test)
# binarysens() assesses sensitivity for matched pair binary outcomes
binarysens(b, c, Gamma = 3, GammaInc = 0.1)
```

**Output interpretation**: The Rosenbaum bounds output shows how the McNemar test p-value changes as Gamma increases from 1 (no hidden bias) to the specified upper limit. The critical Gamma is the value at which the p-value first exceeds 0.05. For example, if the result remains significant up to Gamma = 2.0, the finding is insensitive to a hidden bias that would double the odds of treatment assignment within matched pairs.

```r
# For continuous outcomes: Hodges-Lehmann aligned rank test
# Simulate continuous outcome differences
pair_outcomes <- rnorm(n_pairs, mean = 2.5, sd = 5)

# Rosenbaum bounds for continuous outcomes
psens(pair_outcomes, Gamma = 3, GammaInc = 0.25)
```

### Array (Grid) Approach for Unmeasured Confounding

```r
# Create array of bias-adjusted ORs across a grid of confounder parameters
rr_eu_vals <- seq(1.0, 3.0, by = 0.25)  # confounder-exposure RR
rr_ud_vals <- seq(1.0, 3.0, by = 0.25)  # confounder-outcome RR
p0 <- 0.25  # confounder prevalence in unexposed

grid <- expand.grid(RR_EU = rr_eu_vals, RR_UD = rr_ud_vals)

# Compute bias factor and adjusted OR
grid$p1 <- p0 / (p0 + (1 - p0) / grid$RR_EU)  # approx prevalence in exposed
grid$bias_factor <- (grid$RR_UD * grid$p1 + (1 - grid$p1)) /
                    (grid$RR_UD * p0 + (1 - p0))
grid$adjusted_OR <- or_statin / grid$bias_factor

# Reshape for heatmap
library(ggplot2)
ggplot(grid, aes(x = RR_EU, y = RR_UD, fill = adjusted_OR)) +
  geom_tile() +
  geom_contour(aes(z = adjusted_OR), breaks = 1.0,
               color = "red", linewidth = 1.2) +
  scale_fill_gradient2(low = "steelblue", mid = "white", high = "coral",
                       midpoint = 1.0, name = "Adjusted OR") +
  geom_text(aes(label = round(adjusted_OR, 2)), size = 2.5) +
  labs(title = "Sensitivity Array: Adjusted OR by Confounder Strength",
       x = "RR (Confounder-Exposure)",
       y = "RR (Confounder-Outcome)") +
  theme_minimal(base_size = 12)
```

**Output interpretation**: The heatmap shows how the observed OR changes across different confounder strengths. The red contour line marks where the adjusted OR equals 1.0 (null effect). Combinations of confounder parameters above/right of this line would nullify the observed effect. Comparing with plausible confounder strengths (e.g., smoking RRs) indicates whether the result could be explained away.

## Visualization

### E-Value Sensitivity Plot

```r
# Custom E-value plot with benchmark confounders
rr_point <- 1 / or_statin  # Invert for harmful direction (for plotting)
rr_lower <- 1 / ci_statin[2]

eval_pt <- evalues.OR(est = or_statin, lo = ci_statin[1],
                      hi = ci_statin[2], rare = FALSE)

# Plot with annotations for known confounders
plot(eval_pt)

# Add benchmark confounders
# Diabetes: OR with CV events ~2.0, OR with statin use ~1.3
# Smoking: OR with CV events ~2.5, OR with statin use ~1.4
points(1.3, 2.0, pch = 17, col = "darkgreen", cex = 2)
text(1.3, 2.0, "Diabetes", pos = 4, col = "darkgreen", cex = 0.9)

points(1.4, 2.5, pch = 17, col = "darkred", cex = 2)
text(1.4, 2.5, "Smoking", pos = 4, col = "darkred", cex = 0.9)
```

**Output interpretation**: By plotting known confounder strengths as benchmarks, we can assess whether plausible unmeasured confounders fall within the region that would explain away the observed association. If benchmark confounders are inside the E-value curve, similar unmeasured confounders could nullify the result.

### Multi-Bias Sensitivity Summary

```r
# Summary table of all sensitivity analyses
sensitivity_summary <- data.frame(
  Analysis = c("Primary analysis (adjusted OR)",
               "E-value (point estimate)",
               "E-value (CI lower bound)",
               "QBA: unmeasured confounding (smoking scenario)",
               "QBA: outcome misclassification (Se=85%, Sp=98%)",
               "Probabilistic bias analysis (median adjusted OR)",
               "Rosenbaum Gamma (critical value)"),
  Result = c(
    paste0(round(or_statin, 3), " (", round(ci_statin[1], 3), "-",
           round(ci_statin[2], 3), ")"),
    round(eval_result[2, 1], 2),
    round(eval_result[3, 1], 2),
    round(conf_result$adj.measures[2, 1], 3),
    round(misclass_result$adj.measures[2, 1], 3),
    round(pba_result$adj.measures[1, 1], 3),
    ">2.0"
  ),
  Interpretation = c(
    "Protective effect after adjusting for measured confounders",
    "Confounder needs this minimum RR with both exposure and outcome",
    "Needed to shift CI to include null",
    "Effect after accounting for hypothesized smoking confounding",
    "Effect after correcting for imperfect outcome ascertainment",
    "Central estimate across range of bias parameter values",
    "Robust to hidden bias doubling treatment odds within pairs"
  )
)

print(sensitivity_summary, right = FALSE)
```

## Tips and Best Practices

1. **Always compute E-values** for observational studies. They are simple, widely understood, and provide an intuitive benchmark for unmeasured confounding.
2. **Calibrate E-values against known confounders**: An E-value without context is less informative. Compare against the observed strength of measured confounders.
3. **Use multiple sensitivity analysis approaches**: E-values, QBA, and sensemakr address different aspects of unmeasured confounding. Their concordance strengthens conclusions.
4. **Specify bias parameters from external data** when possible (validation studies, prior literature, expert elicitation), rather than arbitrary values.
5. **Probabilistic bias analysis** is preferred over deterministic QBA because it accounts for uncertainty in bias parameters.
6. **For matched studies**, Rosenbaum bounds are the natural sensitivity analysis. Report the critical Gamma value.
7. **Include negative controls** when feasible. A null result for a negative control exposure provides empirical evidence about residual confounding.
8. **For clinical trials with missing data**, always perform tipping-point analysis alongside the primary multiple imputation analysis.
9. **Pre-specify sensitivity analyses** in the protocol/SAP to avoid post-hoc selection.
10. **Report sensitivity analyses prominently** — not buried in a supplement. They are integral to the interpretation of causal findings.
