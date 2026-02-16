# Quasi-Experimental Methods — R Implementation

## Required Packages

```r
install.packages(c("lmtest", "sandwich", "ggplot2", "dplyr", "rdrobust",
                   "did", "augsynth", "tidyr"))
```

## Example Dataset

### Dataset 1: ITS — Effect of a Drug Safety Warning on Opioid Prescribing

```r
set.seed(123)

# Monthly opioid prescriptions per 100,000 population over 48 months
# Intervention (FDA boxed warning) occurs at month 24
n_months <- 48
time <- 1:n_months
intervention <- ifelse(time > 24, 1, 0)
time_after <- ifelse(time > 24, time - 24, 0)

# Pre-intervention: slight upward trend; post-intervention: drop + declining trend
prescriptions <- 450 + 2 * time - 40 * intervention - 3 * time_after +
  rnorm(n_months, 0, 12)

its_data <- data.frame(
  month = time,
  prescriptions = prescriptions,
  intervention = intervention,
  time_after = time_after
)
```

### Dataset 2: DiD — Hospital Adoption of a Sepsis Bundle

```r
set.seed(456)
n_hospitals <- 200
n_periods <- 10  # 5 pre, 5 post

did_data <- expand.grid(hospital = 1:n_hospitals, period = 1:n_periods)
did_data$treated_group <- ifelse(did_data$hospital <= 100, 1, 0)
did_data$post <- ifelse(did_data$period > 5, 1, 0)
did_data$hospital_fe <- rep(rnorm(n_hospitals, 0, 5), each = n_periods)
did_data$time_trend <- 0.5 * did_data$period

# Outcome: 30-day mortality rate (%)
# Treatment effect: -3 percentage points reduction
did_data$mortality <- 20 + did_data$hospital_fe + did_data$time_trend -
  3 * did_data$treated_group * did_data$post + rnorm(nrow(did_data), 0, 2)
```

### Dataset 3: RD — Statin Prescription at LDL Threshold

```r
set.seed(789)
n_patients <- 2000

# Running variable: LDL cholesterol, threshold at 190 mg/dL
ldl <- runif(n_patients, 140, 240)
above_threshold <- ifelse(ldl >= 190, 1, 0)

# Fuzzy RD: probability of statin prescription jumps at 190
prob_statin <- ifelse(ldl >= 190, 0.75, 0.20)
statin <- rbinom(n_patients, 1, prob_statin)

# Outcome: cardiovascular events in next 5 years (count)
cv_events <- rpois(n_patients, exp(0.5 + 0.01 * (ldl - 190) - 0.4 * statin))

rd_data <- data.frame(ldl, above_threshold, statin, cv_events)
```

## Complete Worked Example

### Part A: Interrupted Time Series Analysis

```r
library(lmtest)
library(sandwich)

# Fit segmented regression
its_model <- lm(prescriptions ~ month + intervention + time_after, data = its_data)

# Use Newey-West standard errors to account for autocorrelation
coeftest(its_model, vcov = NeweyWest(its_model, lag = 3))

# Interpretation:
# - month: pre-intervention trend (prescriptions per month increase)
# - intervention: immediate level change at the warning date
#   Expected: significant negative (sudden drop in prescribing)
# - time_after: change in slope after intervention
#   Expected: negative (continued decline beyond the level drop)

cat("\n--- ITS Results ---\n")
cat(sprintf("Pre-intervention trend: %.2f per month\n", coef(its_model)["month"]))
cat(sprintf("Immediate level change: %.2f\n", coef(its_model)["intervention"]))
cat(sprintf("Post-intervention trend change: %.2f per month\n", coef(its_model)["time_after"]))
```

#### ITS Visualization

```r
library(ggplot2)

its_data$predicted <- predict(its_model)
its_data$counterfactual <- coef(its_model)[1] + coef(its_model)[2] * its_data$month

ggplot(its_data, aes(x = month)) +
  geom_point(aes(y = prescriptions), color = "grey40", size = 2) +
  geom_line(aes(y = predicted), color = "steelblue", linewidth = 1) +
  geom_line(aes(y = counterfactual), color = "red", linetype = "dashed", linewidth = 0.8) +
  geom_vline(xintercept = 24, linetype = "dotted", color = "black") +
  annotate("text", x = 24, y = max(its_data$prescriptions),
           label = "FDA Warning", hjust = -0.1, fontface = "italic") +
  labs(x = "Month", y = "Opioid Prescriptions per 100,000",
       title = "Interrupted Time Series: Impact of FDA Boxed Warning on Opioid Prescribing",
       subtitle = "Blue = fitted model; Red dashed = counterfactual (no intervention)") +
  theme_minimal(base_size = 12)
```

### Part B: Difference-in-Differences

```r
library(dplyr)

# Simple DiD regression with hospital fixed effects
did_model <- lm(mortality ~ treated_group + post + treated_group:post +
                  factor(hospital), data = did_data)

# Extract DiD estimate
did_estimate <- coef(did_model)["treated_group:post"]
did_se <- sqrt(diag(vcov(did_model)))["treated_group:post"]

cat(sprintf("\n--- DiD Results ---\n"))
cat(sprintf("DiD estimate (ATT): %.3f (SE: %.3f)\n", did_estimate, did_se))
cat(sprintf("95%% CI: [%.3f, %.3f]\n",
            did_estimate - 1.96 * did_se, did_estimate + 1.96 * did_se))
# Expected: approximately -3.0 (true effect)

# Check parallel trends visually
trends <- did_data %>%
  group_by(period, treated_group) %>%
  summarise(mean_mortality = mean(mortality), .groups = "drop")

ggplot(trends, aes(x = period, y = mean_mortality,
                    color = factor(treated_group), group = factor(treated_group))) +
  geom_line(linewidth = 1) +
  geom_point(size = 2) +
  geom_vline(xintercept = 5.5, linetype = "dashed") +
  annotate("text", x = 5.5, y = max(trends$mean_mortality),
           label = "Intervention", hjust = -0.1) +
  scale_color_manual(values = c("0" = "grey50", "1" = "steelblue"),
                     labels = c("Control hospitals", "Treated hospitals")) +
  labs(x = "Period", y = "30-Day Mortality (%)", color = "",
       title = "Difference-in-Differences: Sepsis Bundle Impact on Mortality",
       subtitle = "Parallel trends in pre-period support the DiD assumption") +
  theme_minimal(base_size = 12)
```

### Part C: Regression Discontinuity

```r
library(rdrobust)

# Sharp RD: effect of crossing LDL threshold on statin prescription
rd_first_stage <- rdrobust(rd_data$statin, rd_data$ldl, c = 190)
summary(rd_first_stage)
# Expected: large positive jump in statin prescription probability at 190

# RD plot for statin use
rdplot(rd_data$statin, rd_data$ldl, c = 190,
       title = "First Stage: Statin Prescription Jumps at LDL = 190",
       x.label = "LDL Cholesterol (mg/dL)", y.label = "P(Statin Prescribed)")

# Reduced form: effect on cardiovascular events
rd_outcome <- rdrobust(rd_data$cv_events, rd_data$ldl, c = 190)
summary(rd_outcome)

# Fuzzy RD: IV estimate of statin effect on CV events
rd_fuzzy <- rdrobust(rd_data$cv_events, rd_data$ldl, c = 190,
                      fuzzy = rd_data$statin)
summary(rd_fuzzy)

cat("\n--- RD Results ---\n")
cat(sprintf("Fuzzy RD estimate (LATE): %.3f\n", rd_fuzzy$coef[1]))
cat(sprintf("95%% CI: [%.3f, %.3f]\n", rd_fuzzy$ci[1, 1], rd_fuzzy$ci[1, 2]))
```

## Advanced Example

### Staggered DiD with the `did` Package

```r
library(did)

# Create staggered adoption data
set.seed(101)
n_units <- 300
n_time <- 12

stag_data <- expand.grid(id = 1:n_units, time = 1:n_time)
# Groups adopt at different times: group 1 at t=5, group 2 at t=8, group 3 never
stag_data$group <- rep(sample(1:3, n_units, replace = TRUE), each = n_time)
stag_data$adoption_time <- ifelse(stag_data$group == 1, 5,
                                   ifelse(stag_data$group == 2, 8, 0))
stag_data$treated <- ifelse(stag_data$adoption_time > 0 &
                             stag_data$time >= stag_data$adoption_time, 1, 0)
stag_data$unit_fe <- rep(rnorm(n_units, 0, 3), each = n_time)
stag_data$outcome <- 10 + stag_data$unit_fe + 0.5 * stag_data$time -
  2.5 * stag_data$treated + rnorm(nrow(stag_data), 0, 1.5)

# Callaway and Sant'Anna (2021) estimator
cs_result <- att_gt(
  yname = "outcome",
  tname = "time",
  idname = "id",
  gname = "adoption_time",
  data = stag_data,
  control_group = "nevertreated"
)

summary(cs_result)

# Aggregate to an overall ATT
agg_result <- aggte(cs_result, type = "simple")
summary(agg_result)

# Event study plot
es_result <- aggte(cs_result, type = "dynamic")
ggdid(es_result) +
  labs(title = "Event Study: Staggered DiD (Callaway-Sant'Anna)",
       subtitle = "Pre-trend coefficients near zero support parallel trends") +
  theme_minimal()
```

## Visualization

```r
library(gridExtra)

# Combined visualization panel
p1 <- ggplot(its_data, aes(x = month)) +
  geom_point(aes(y = prescriptions), alpha = 0.6) +
  geom_line(aes(y = predicted), color = "steelblue", linewidth = 1) +
  geom_vline(xintercept = 24, linetype = "dotted") +
  labs(title = "ITS: Opioid Prescribing", x = "Month", y = "Prescriptions") +
  theme_minimal(base_size = 10)

p2 <- ggplot(trends, aes(x = period, y = mean_mortality,
                           color = factor(treated_group))) +
  geom_line(linewidth = 1) + geom_point() +
  geom_vline(xintercept = 5.5, linetype = "dashed") +
  scale_color_manual(values = c("grey50", "steelblue"), guide = "none") +
  labs(title = "DiD: Sepsis Mortality", x = "Period", y = "Mortality (%)") +
  theme_minimal(base_size = 10)

grid.arrange(p1, p2, ncol = 2)
```

## Tips and Best Practices

1. **ITS requires enough time points**: At least 8 pre-intervention and 8 post-intervention
   observations are recommended. Fewer points make it difficult to estimate trends reliably
   and to detect autocorrelation.

2. **Always check for autocorrelation in ITS**: Use the Durbin-Watson test or examine ACF/PACF
   plots. If present, use Newey-West SEs or fit an ARIMA model with intervention indicators.

3. **Test parallel trends for DiD**: Plot group-specific trends in the pre-period. Formally,
   test for differential pre-trends by interacting group with pre-treatment time indicators.
   If trends diverge, DiD is not credible.

4. **Use cluster-robust SEs in DiD**: When data are panel (repeated observations on units),
   cluster standard errors at the unit level to account for within-unit correlation.

5. **RD bandwidth sensitivity**: Always report results across multiple bandwidths. The main
   result should use the MSE-optimal bandwidth from `rdrobust`, but show robustness to
   halving and doubling the bandwidth.

6. **Run the McCrary density test**: In RD designs, test for manipulation of the running
   variable. A discontinuity in the density at the cutoff suggests subjects may have
   strategically sorted around the threshold.

7. **Beware of two-way fixed effects with staggered DiD**: The standard TWFE estimator can
   produce biased estimates when treatment timing varies. Use Callaway-Sant'Anna or other
   heterogeneity-robust DiD estimators.

8. **Combine methods for robustness**: A controlled ITS (ITS + DiD) is more credible than
   either method alone. Present multiple analyses to triangulate the causal effect.
