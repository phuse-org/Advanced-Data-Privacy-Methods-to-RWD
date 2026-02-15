# Adaptive Trial Design — R Implementation

## Required Packages

```r
install.packages(c("gsDesign", "rpact", "ggplot2"))

library(gsDesign)    # group sequential design and boundaries
library(rpact)       # comprehensive adaptive design toolkit
library(ggplot2)
```

## Example Dataset

We design a group sequential trial comparing a new anticoagulant to standard heparin
for prevention of venous thromboembolism (VTE) after knee replacement surgery. The
primary endpoint is the proportion of patients experiencing VTE within 30 days.

```r
# Design parameters
alpha       <- 0.025   # one-sided
beta        <- 0.20    # 80% power
p_control   <- 0.15    # VTE rate under heparin
p_treatment <- 0.08    # expected VTE rate under new drug
k           <- 3       # number of analyses (2 interim + 1 final)

# Fixed-sample size (no interim)
n_fixed <- power.prop.test(p1 = p_control, p2 = p_treatment,
                           sig.level = 2 * alpha, power = 1 - beta)$n
cat("Fixed-design sample size per arm:", ceiling(n_fixed), "\n")
```

## Complete Worked Example

### Step 1 — O'Brien-Fleming Group Sequential Design with gsDesign

```r
# O'Brien-Fleming boundaries
gs_obf <- gsDesign(
  k       = 3,
  test.type = 2,          # two-sided symmetric (use 1 for one-sided)
  alpha   = alpha,
  beta    = beta,
  sfu     = sfLDOF,        # Lan-DeMets O'Brien-Fleming spending function
  sfl     = sfLDOF,        # futility spending (binding)
  timing  = c(0.33, 0.67)  # information fractions
)

summary(gs_obf)

# Key output:
#   Upper boundaries (efficacy): very large at IA1, moderate at IA2, near 1.96 at final
#   Lower boundaries (futility): very negative early, rising toward 0

# Boundaries on Z-scale
cat("\n--- Efficacy Boundaries (Z-scale) ---\n")
print(round(gs_obf$upper$bound, 4))
cat("\n--- Futility Boundaries (Z-scale) ---\n")
print(round(gs_obf$lower$bound, 4))

# Maximum sample size (inflation factor relative to fixed design)
cat("\nInflation factor:", round(gs_obf$nFixSurv, 4), "\n")
cat("Max sample size per arm:", ceiling(n_fixed * gs_obf$nFixSurv), "\n")
```

### Step 2 — Pocock Boundaries for Comparison

```r
gs_pocock <- gsDesign(
  k       = 3,
  test.type = 2,
  alpha   = alpha,
  beta    = beta,
  sfu     = sfLDPocock,    # Pocock spending function
  timing  = c(0.33, 0.67)
)

# Compare boundaries
comparison <- data.frame(
  Analysis = 1:3,
  Info_Fraction = c(0.33, 0.67, 1.0),
  OBF_Upper = round(gs_obf$upper$bound, 3),
  Pocock_Upper = round(gs_pocock$upper$bound, 3),
  OBF_Lower = round(gs_obf$lower$bound, 3),
  Pocock_Lower = round(gs_pocock$lower$bound, 3)
)
print(comparison)
# Pocock: uniform boundaries (~2.29 at each look)
# OBF: decreasing boundaries (~3.47, 2.45, 2.00)
```

### Step 3 — Plot Boundaries

```r
plot(gs_obf, plottype = 1, main = "O'Brien-Fleming Boundaries")

# Custom ggplot
bnd_df <- data.frame(
  analysis = rep(1:3, 4),
  info_frac = rep(c(0.33, 0.67, 1.0), 4),
  boundary = c(gs_obf$upper$bound, gs_obf$lower$bound,
               gs_pocock$upper$bound, gs_pocock$lower$bound),
  type = rep(c("OBF Efficacy", "OBF Futility",
               "Pocock Efficacy", "Pocock Futility"), each = 3)
)

ggplot(bnd_df, aes(x = info_frac, y = boundary, color = type, linetype = type)) +
  geom_line(linewidth = 1) +
  geom_point(size = 3) +
  labs(title = "Group Sequential Boundaries: OBF vs Pocock",
       x = "Information Fraction", y = "Z-value Boundary", color = "", linetype = "") +
  theme_minimal() +
  theme(legend.position = "bottom")
```

### Step 4 — Interim Analysis Decision Rules

```r
# Simulate interim data at the first interim (33% information)
set.seed(101)
n_per_arm_ia1 <- ceiling(n_fixed * gs_obf$nFixSurv * 0.33)

events_ctrl <- rbinom(1, n_per_arm_ia1, p_control)
events_trt  <- rbinom(1, n_per_arm_ia1, p_treatment)

p_hat_ctrl <- events_ctrl / n_per_arm_ia1
p_hat_trt  <- events_trt / n_per_arm_ia1

# Z-statistic (pooled proportion test)
p_pool <- (events_ctrl + events_trt) / (2 * n_per_arm_ia1)
z_ia1  <- (p_hat_ctrl - p_hat_trt) /
          sqrt(p_pool * (1 - p_pool) * (2 / n_per_arm_ia1))

cat("Interim Analysis 1 Results:\n")
cat("  n per arm:", n_per_arm_ia1, "\n")
cat("  Control event rate:", round(p_hat_ctrl, 3), "\n")
cat("  Treatment event rate:", round(p_hat_trt, 3), "\n")
cat("  Z-statistic:", round(z_ia1, 4), "\n")
cat("  Efficacy boundary:", round(gs_obf$upper$bound[1], 4), "\n")
cat("  Futility boundary:", round(gs_obf$lower$bound[1], 4), "\n")

if (z_ia1 >= gs_obf$upper$bound[1]) {
  cat("  Decision: STOP for EFFICACY\n")
} else if (z_ia1 <= gs_obf$lower$bound[1]) {
  cat("  Decision: STOP for FUTILITY\n")
} else {
  cat("  Decision: CONTINUE to next analysis\n")
}
```

### Step 5 — Conditional Power at Interim

```r
# Conditional power under the original alternative
cp_orig <- gsCP(x = gs_obf, i = 1, zi = z_ia1, theta = NULL)
cat("\nConditional power (original H1):", round(cp_orig$upper$prob, 4), "\n")

# Conditional power under current trend
theta_hat <- z_ia1 / sqrt(n_per_arm_ia1)
cp_trend <- gsCP(x = gs_obf, i = 1, zi = z_ia1,
                 theta = theta_hat * sqrt(ceiling(n_fixed * gs_obf$nFixSurv)))
cat("Conditional power (current trend):", round(cp_trend$upper$prob, 4), "\n")
# If CP < 10-20%, consider futility stopping.
```

## Advanced Example

### Adaptive Design with rpact

```r
# rpact provides a more modern interface
design_rpact <- getDesignGroupSequential(
  sided                = 1,
  alpha                = 0.025,
  beta                 = 0.2,
  informationRates     = c(0.33, 0.67, 1.0),
  typeOfDesign         = "asOF",         # alpha-spending O'Brien-Fleming
  typeBetaSpending     = "bsOF",         # beta-spending O'Brien-Fleming
  bindingFutility      = FALSE
)

summary(design_rpact)

# Sample size for the proportions example
ss_rpact <- getSampleSizeRates(
  design          = design_rpact,
  pi1             = p_treatment,
  pi2             = p_control,
  allocationRatioPlanned = 1
)

summary(ss_rpact)

# Power simulation
sim_results <- getSimulationRates(
  design        = design_rpact,
  pi1           = seq(0.05, 0.15, 0.01),     # range of treatment rates
  pi2           = p_control,
  plannedSubjects = ss_rpact$maxNumberOfSubjects,
  maxNumberOfIterations = 10000,
  seed = 42
)

# Plot power curve
plot(sim_results, type = "power")
```

### Operating Characteristics via Simulation

```r
# Simulate 10,000 trials under the alternative hypothesis
set.seed(999)
n_sim <- 10000
max_n <- ceiling(n_fixed * gs_obf$nFixSurv)
n_at_look <- ceiling(max_n * c(0.33, 0.67, 1.0))

stopped_eff <- stopped_fut <- numeric(3)
final_reject <- 0

for (s in 1:n_sim) {
  # Generate full dataset
  y_ctrl <- rbinom(max_n, 1, p_control)
  y_trt  <- rbinom(max_n, 1, p_treatment)

  for (look in 1:3) {
    ni <- n_at_look[look]
    p1 <- mean(y_ctrl[1:ni])
    p2 <- mean(y_trt[1:ni])
    pp <- (sum(y_ctrl[1:ni]) + sum(y_trt[1:ni])) / (2 * ni)
    z  <- (p1 - p2) / sqrt(pp * (1 - pp) * 2 / ni)

    if (z >= gs_obf$upper$bound[look]) {
      stopped_eff[look] <- stopped_eff[look] + 1
      break
    }
    if (z <= gs_obf$lower$bound[look]) {
      stopped_fut[look] <- stopped_fut[look] + 1
      break
    }
  }
}

cat("Operating Characteristics (under H1):\n")
cat("  Efficacy stop at IA1:", stopped_eff[1] / n_sim, "\n")
cat("  Efficacy stop at IA2:", stopped_eff[2] / n_sim, "\n")
cat("  Efficacy stop at Final:", stopped_eff[3] / n_sim, "\n")
cat("  Overall power:", sum(stopped_eff) / n_sim, "\n")
cat("  Futility stop at IA1:", stopped_fut[1] / n_sim, "\n")
cat("  Futility stop at IA2:", stopped_fut[2] / n_sim, "\n")
cat("  Expected sample size (fraction):",
    sum(stopped_eff * c(0.33, 0.67, 1) + stopped_fut * c(0.33, 0.67, 1)) /
    n_sim, "\n")
```

## Visualization

```r
# 1. Boundary plot
plot(gs_obf, plottype = "Z", main = "Group Sequential Boundaries (Z-scale)")

# 2. Expected sample size under different effect sizes
plot(gs_obf, plottype = 5, main = "Expected Sample Size vs Effect Size")

# 3. Alpha spending function
t_grid <- seq(0, 1, by = 0.01)
alpha_obf <- sfLDOF(alpha, t_grid)$spend
alpha_poc <- sfLDPocock(alpha, t_grid)$spend

ggplot(data.frame(t = rep(t_grid, 2),
                  spend = c(alpha_obf, alpha_poc),
                  type = rep(c("O'Brien-Fleming", "Pocock"), each = length(t_grid))),
       aes(x = t, y = spend, color = type)) +
  geom_line(linewidth = 1.2) +
  labs(title = "Alpha Spending Functions",
       x = "Information Fraction", y = "Cumulative Alpha Spent",
       color = "Spending Function") +
  theme_minimal()
```

## Tips and Best Practices

1. **Pre-specify everything** in the protocol and SAP: number of interim looks,
   information fractions, spending functions, futility rules, and decision criteria.

2. **Use the Lan-DeMets alpha spending approach** rather than fixed boundaries. It
   tolerates deviations in the planned information fractions.

3. **Prefer O'Brien-Fleming spending** for confirmatory trials. It preserves most of
   the alpha for the final analysis and has minimal sample-size inflation (~3%).

4. **Non-binding futility** is generally preferred over binding futility because it does
   not inflate the Type I error if the DSMB overrides the futility stop.

5. **Verify operating characteristics by simulation** under the null, the alternative,
   and a range of intermediate effect sizes.

6. **Account for overrunning** — patients enrolled between the interim data cutoff and
   the stopping decision should be included in the analysis.

7. **Document the DSMB charter** with explicit stopping guidelines, access rules, and
   communication procedures.

8. **Be cautious about effect-size overestimation** in trials stopped early for
   efficacy. Report the bias-adjusted estimate (e.g., median unbiased estimate from
   `gsDesign::gsCI()`).
