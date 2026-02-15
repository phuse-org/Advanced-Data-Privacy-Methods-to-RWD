# Healthcare Cost and Resource Analysis — R Implementation

## Required Packages

```r
install.packages(c("pscl", "boot", "MASS", "ggplot2", "dplyr", "BCEA"))

library(pscl)       # hurdle / zero-inflated models
library(boot)       # bootstrap inference
library(MASS)       # negative binomial, gamma
library(ggplot2)
library(dplyr)
library(BCEA)       # Bayesian cost-effectiveness analysis
```

## Example Dataset

We simulate healthcare cost data from a randomised trial comparing a new diabetes
medication to standard of care. Costs include total 12-month medical expenditure.

```r
set.seed(123)
n <- 800

# Covariates
age       <- round(rnorm(n, 60, 12))
female    <- rbinom(n, 1, 0.52)
bmi       <- rnorm(n, 30, 5)
treatment <- rep(c(0, 1), each = n / 2)

# Generate cost data (mixture of zeros and gamma-distributed positive costs)
# About 15% of patients have zero cost
prob_zero <- plogis(-2 + 0.01 * age - 0.3 * treatment + 0.02 * bmi)
any_cost  <- rbinom(n, 1, 1 - prob_zero)

# Positive costs from gamma distribution
shape  <- 2
rate   <- 2 / exp(7.5 + 0.01 * age + 0.3 * treatment + 0.005 * bmi)
pos_cost <- rgamma(n, shape = shape, rate = rate)

total_cost <- ifelse(any_cost == 1, pos_cost, 0)

# QALYs (effectiveness measure)
qaly <- rnorm(n, mean = 0.75 + 0.05 * treatment - 0.002 * age, sd = 0.1)

cost_df <- data.frame(
  treatment, age, female, bmi, total_cost, qaly,
  any_cost = as.integer(total_cost > 0)
)

summary(cost_df[, c("total_cost", "qaly")])
cat("Proportion zero cost:", mean(total_cost == 0), "\n")
```

## Complete Worked Example

### Step 1 — Exploratory Analysis

```r
# Distribution of costs
ggplot(cost_df, aes(x = total_cost, fill = factor(treatment))) +
  geom_histogram(bins = 50, alpha = 0.6, position = "identity") +
  scale_fill_manual(values = c("steelblue", "firebrick"),
                    labels = c("Control", "Treatment")) +
  labs(title = "Distribution of 12-Month Healthcare Costs",
       x = "Total Cost ($)", y = "Frequency", fill = "Group") +
  theme_minimal()

# Summary by group
cost_df %>%
  group_by(treatment) %>%
  summarise(
    n         = n(),
    mean_cost = mean(total_cost),
    median_cost = median(total_cost),
    sd_cost   = sd(total_cost),
    pct_zero  = mean(total_cost == 0)
  )
```

### Step 2 — GLM with Gamma Family and Log Link

```r
# Fit on positive costs only (common approach) or all data
# For all data, add a small constant or use two-part model

# Gamma GLM on positive costs
pos_data <- cost_df %>% filter(total_cost > 0)
glm_gamma <- glm(total_cost ~ treatment + age + bmi + female,
                 family = Gamma(link = "log"), data = pos_data)
summary(glm_gamma)

# Treatment effect on cost ratio scale
exp(coef(glm_gamma)["treatment"])
# Values > 1 mean treatment increases costs; < 1 means decreases

# Modified Park test to verify variance function
glm_check <- glm(total_cost ~ treatment + age + bmi + female,
                 family = gaussian(link = "log"), data = pos_data)
resid_sq  <- log(residuals(glm_check, type = "response")^2)
pred_log  <- log(fitted(glm_check))
park_test <- lm(resid_sq ~ pred_log)
cat("Park test slope:", coef(park_test)[2], "\n")
# Slope near 2 supports gamma family
```

### Step 3 — Two-Part (Hurdle) Model

```r
# Part 1: logistic regression for P(cost > 0)
part1 <- glm(any_cost ~ treatment + age + bmi + female,
             family = binomial(link = "logit"), data = cost_df)
summary(part1)

# Part 2: gamma GLM on positive costs
part2 <- glm(total_cost ~ treatment + age + bmi + female,
             family = Gamma(link = "log"), data = pos_data)
summary(part2)

# Predict unconditional mean cost
predict_two_part <- function(newdata, part1, part2) {
  p_pos     <- predict(part1, newdata = newdata, type = "response")
  cond_mean <- predict(part2, newdata = newdata, type = "response")
  p_pos * cond_mean
}

# Mean cost by treatment group
nd_trt  <- data.frame(treatment = 1, age = 60, bmi = 30, female = 0)
nd_ctrl <- data.frame(treatment = 0, age = 60, bmi = 30, female = 0)

cost_trt  <- predict_two_part(nd_trt, part1, part2)
cost_ctrl <- predict_two_part(nd_ctrl, part1, part2)
cat("Predicted cost (treatment):", round(cost_trt, 2), "\n")
cat("Predicted cost (control):  ", round(cost_ctrl, 2), "\n")
cat("Incremental cost:          ", round(cost_trt - cost_ctrl, 2), "\n")
```

### Step 4 — Bootstrap CI for Cost Difference

```r
boot_cost_diff <- function(data, indices) {
  d <- data[indices, ]
  mean(d$total_cost[d$treatment == 1]) - mean(d$total_cost[d$treatment == 0])
}

set.seed(456)
boot_res <- boot(cost_df, boot_cost_diff, R = 2000)
print(boot_res)

# Percentile CI
boot_ci <- boot.ci(boot_res, type = c("perc", "bca"))
print(boot_ci)

# Interpretation: if the CI for the cost difference includes 0, the
# between-group difference is not statistically significant.
```

### Step 5 — Basic Cost-Effectiveness Analysis

```r
# Compute incremental cost and incremental QALY
delta_c <- mean(cost_df$total_cost[cost_df$treatment == 1]) -
           mean(cost_df$total_cost[cost_df$treatment == 0])
delta_e <- mean(cost_df$qaly[cost_df$treatment == 1]) -
           mean(cost_df$qaly[cost_df$treatment == 0])

icer <- delta_c / delta_e
cat("Incremental Cost:  $", round(delta_c, 2), "\n")
cat("Incremental QALY:  ",  round(delta_e, 4), "\n")
cat("ICER ($/QALY):     $", round(icer, 2), "\n")

# Net monetary benefit at WTP = $50,000/QALY
wtp <- 50000
nmb <- wtp * delta_e - delta_c
cat("Net Monetary Benefit at WTP=$50k: $", round(nmb, 2), "\n")
# NMB > 0 means the treatment is cost-effective at this threshold.
```

## Advanced Example

### Cost-Effectiveness Acceptability Curve with BCEA

```r
# Bootstrap joint distribution of (Delta_C, Delta_E)
boot_ce <- function(data, indices) {
  d <- data[indices, ]
  dc <- mean(d$total_cost[d$treatment == 1]) - mean(d$total_cost[d$treatment == 0])
  de <- mean(d$qaly[d$treatment == 1]) - mean(d$qaly[d$treatment == 0])
  c(dc, de)
}

set.seed(789)
boot_ce_res <- boot(cost_df, boot_ce, R = 5000)

# Create matrices for BCEA
# BCEA expects: e = matrix(n.sim x n.interventions), c = same
# Simulate from bootstrap distribution
cost_mat <- cbind(
  boot_ce_res$t[, 1] * 0 + mean(cost_df$total_cost[cost_df$treatment == 0]),
  boot_ce_res$t[, 1] + mean(cost_df$total_cost[cost_df$treatment == 0])
)
eff_mat <- cbind(
  boot_ce_res$t[, 2] * 0 + mean(cost_df$qaly[cost_df$treatment == 0]),
  boot_ce_res$t[, 2] + mean(cost_df$qaly[cost_df$treatment == 0])
)

bcea_obj <- bcea(eff = eff_mat, cost = cost_mat,
                 ref = 1, interventions = c("Control", "Treatment"))

# CEAC
ceac.plot(bcea_obj, graph = "ggplot2") +
  labs(title = "Cost-Effectiveness Acceptability Curve") +
  theme_minimal()

# CE plane
ceplane.plot(bcea_obj, graph = "ggplot2") +
  labs(title = "Cost-Effectiveness Plane") +
  theme_minimal()
```

### Hurdle Model with pscl

```r
# pscl's hurdle() fits both parts simultaneously
# For continuous positive data, use a manual two-part; pscl is for count data.
# Here we show zero-inflated Poisson for count of healthcare visits:

cost_df$visits <- rpois(n, lambda = exp(1.5 + 0.2 * treatment + 0.01 * age))
cost_df$visits[cost_df$any_cost == 0] <- 0

hurdle_fit <- hurdle(visits ~ treatment + age + bmi + female,
                     dist = "poisson", zero.dist = "binomial", data = cost_df)
summary(hurdle_fit)

# The count part shows effect of treatment on visit rate among users
# The zero part shows effect of treatment on probability of any visit
```

## Visualization

```r
# 1. Cost distribution by group
ggplot(cost_df, aes(x = total_cost, fill = factor(treatment))) +
  geom_density(alpha = 0.5) +
  scale_fill_manual(values = c("#2166AC", "#B2182B"),
                    labels = c("Control", "Treatment")) +
  labs(title = "Cost Density by Treatment Group",
       x = "Total 12-Month Cost ($)", y = "Density", fill = "Group") +
  theme_minimal()

# 2. CE plane (manual)
ggplot(data.frame(delta_e = boot_ce_res$t[, 2],
                  delta_c = boot_ce_res$t[, 1]),
       aes(x = delta_e, y = delta_c)) +
  geom_point(alpha = 0.1, color = "steelblue") +
  geom_hline(yintercept = 0, linetype = "dashed") +
  geom_vline(xintercept = 0, linetype = "dashed") +
  geom_abline(slope = 50000, intercept = 0, color = "red",
              linetype = "dotted", linewidth = 1) +
  labs(title = "Cost-Effectiveness Plane",
       x = "Incremental QALY", y = "Incremental Cost ($)",
       caption = "Red line = WTP $50,000/QALY") +
  theme_minimal()

# 3. CEAC (manual)
wtp_grid <- seq(0, 200000, by = 1000)
prob_ce  <- sapply(wtp_grid, function(w) {
  nmb_boot <- w * boot_ce_res$t[, 2] - boot_ce_res$t[, 1]
  mean(nmb_boot > 0)
})

ggplot(data.frame(wtp = wtp_grid, prob = prob_ce), aes(x = wtp, y = prob)) +
  geom_line(color = "darkgreen", linewidth = 1) +
  geom_hline(yintercept = 0.5, linetype = "dashed", alpha = 0.5) +
  scale_x_continuous(labels = scales::dollar_format()) +
  labs(title = "Cost-Effectiveness Acceptability Curve",
       x = "Willingness-to-Pay Threshold ($/QALY)",
       y = "Probability Cost-Effective") +
  theme_minimal()
```

## Tips and Best Practices

1. **Use GLM (gamma + log link) as the default** for modelling positive costs. It avoids
   the retransformation problem of log-OLS and handles heteroscedasticity naturally.

2. **Apply the modified Park test** to select the GLM variance family. Do not assume
   gamma without checking.

3. **Use two-part models when there are structural zeros.** If the zero-cost group is
   clinically distinct (never-users vs. occasional users), model the two processes
   separately.

4. **Bootstrap cost differences and ICERs.** Skewed cost distributions make normal-theory
   CIs unreliable. Report percentile or BCa bootstrap CIs.

5. **Present the full CE plane and CEAC,** not just the point ICER. The ICER is
   undefined when Delta_E is near zero, and the CEAC communicates decision uncertainty.

6. **Discount future costs and QALYs** at the rate specified by the HTA agency
   (typically 3-3.5% per year).

7. **Conduct sensitivity analyses:** one-way, probabilistic (PSA), and scenario analyses
   are expected in HTA submissions.

8. **Handle censored costs appropriately.** If patients die or drop out before the end of
   follow-up, total costs are right-censored. Use Lin's method or IPCW to avoid bias.
