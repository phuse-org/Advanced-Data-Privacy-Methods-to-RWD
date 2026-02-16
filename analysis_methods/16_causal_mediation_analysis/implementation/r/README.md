# Causal Mediation Analysis — R Implementation

## Required Packages

```r
install.packages(c("mediation", "medflex", "CMAverse", "survival", "ggplot2"))

library(mediation)
library(medflex)
library(CMAverse)
library(survival)
library(ggplot2)
```

## Example Dataset

We simulate a clinical trial where a new anti-inflammatory drug (treatment) reduces
joint damage (outcome) partly by lowering CRP, an inflammatory biomarker (mediator).

```r
set.seed(42)
n <- 500

# Confounders
age       <- rnorm(n, mean = 55, sd = 10)
female    <- rbinom(n, 1, 0.45)

# Treatment assignment (randomised)
treatment <- rbinom(n, 1, 0.5)

# Mediator: CRP reduction (continuous, higher = more reduction)
crp_reduction <- 0.8 * treatment + 0.02 * age - 0.1 * female +
                 0.3 * treatment * female + rnorm(n, 0, 1)

# Outcome: change in joint-damage score (lower = better)
joint_damage <- -1.5 * treatment - 0.6 * crp_reduction +
                0.2 * treatment * crp_reduction +
                0.03 * age + 0.1 * female + rnorm(n, 0, 1.2)

dat <- data.frame(treatment, crp_reduction, joint_damage, age, female)
head(dat)
```

## Complete Worked Example

### Step 1 — Traditional Baron and Kenny Approach

```r
# Path c  (total effect)
fit_total <- lm(joint_damage ~ treatment + age + female, data = dat)
summary(fit_total)
# Expect treatment coefficient around -2.0

# Path a  (treatment -> mediator)
fit_med <- lm(crp_reduction ~ treatment + age + female, data = dat)
summary(fit_med)
# Expect treatment coefficient around 0.8

# Path b and c' (mediator + treatment -> outcome)
fit_out <- lm(joint_damage ~ treatment + crp_reduction + age + female, data = dat)
summary(fit_out)
# c' (direct) should be attenuated relative to c

# Product of coefficients
a <- coef(fit_med)["treatment"]
b <- coef(fit_out)["crp_reduction"]
indirect_bk <- a * b
cat("Baron-Kenny indirect effect:", round(indirect_bk, 4), "\n")
cat("Baron-Kenny direct effect:",   round(coef(fit_out)["treatment"], 4), "\n")
```

### Step 2 — Counterfactual Mediation with the `mediation` Package

```r
# Fit mediator and outcome models (include interaction)
med_model <- lm(crp_reduction ~ treatment * female + age, data = dat)
out_model <- lm(joint_damage ~ treatment * crp_reduction + age + female, data = dat)

# Run causal mediation analysis
med_result <- mediate(
  med_model,
  out_model,
  treat     = "treatment",
  mediator  = "crp_reduction",
  boot      = TRUE,
  boot.ci.type = "perc",
  sims      = 1000
)

summary(med_result)
# Output includes:
#   ACME  = Average Causal Mediation Effect (NIE)
#   ADE   = Average Direct Effect (NDE)
#   Total = Total Effect
#   Prop. Mediated

# Interpretation: If ACME is significantly negative, CRP reduction
# mediates part of the treatment benefit on joint damage.
```

### Step 3 — Sensitivity Analysis (Imai's rho)

```r
sens <- medsens(med_result, rho.by = 0.05, effect.type = "indirect", sims = 500)
summary(sens)

# The output reports the value of rho at which the ACME becomes zero.
# A larger |rho| at the crossover indicates more robust mediation.

plot(sens, main = "Sensitivity of ACME to Unmeasured Confounding")
```

### Step 4 — VanderWeele 4-Way Decomposition with `CMAverse`

```r
res_4way <- cmest(
  data      = dat,
  model     = "rb",            # regression-based
  outcome   = "joint_damage",
  exposure  = "treatment",
  mediator  = "crp_reduction",
  basec     = c("age", "female"),
  EMint     = TRUE,            # exposure-mediator interaction
  mreg      = list("linear"),
  yreg      = "linear",
  astar     = 0,               # reference treatment level
  a         = 1,               # active treatment level
  mval      = list(0),         # controlled-direct-effect mediator value
  estimation = "imputation",
  inference  = "bootstrap",
  nboot      = 1000
)

summary(res_4way)
# Reports: CDE, INTref, INTmed, PIE, and proportion mediated / due to interaction.
# CDE   = effect with mediator fixed at mval
# PIE   = pure indirect effect
# INTmed = portion due to both mediation and interaction
# INTref = portion due to interaction alone
```

## Advanced Example

### Multiple Mediators — Joint Mediation

```r
# Add a second mediator: ESR (erythrocyte sedimentation rate reduction)
dat$esr_reduction <- 0.5 * treatment + 0.4 * crp_reduction +
                     0.01 * age + rnorm(n, 0, 0.8)

# Joint mediation: treat both mediators as a block
med1 <- lm(crp_reduction ~ treatment + age + female, data = dat)
med2 <- lm(esr_reduction ~ treatment + crp_reduction + age + female, data = dat)
out  <- lm(joint_damage ~ treatment + crp_reduction + esr_reduction + age + female,
           data = dat)

# Path-specific indirect through CRP only
med_crp <- mediate(med1, out, treat = "treatment",
                   mediator = "crp_reduction", sims = 500)
summary(med_crp)

# Note: sequential mediators (CRP -> ESR -> Outcome) require careful
# identification. CMAverse supports this via the multistate = TRUE option.
```

### Mediation with a Survival Outcome

```r
# Simulate time-to-event outcome
dat$time  <- rexp(n, rate = exp(-3 + 0.5 * treatment + 0.3 * crp_reduction))
dat$event <- rbinom(n, 1, 0.8)

library(survival)

med_model_surv <- lm(crp_reduction ~ treatment + age + female, data = dat)
out_model_surv <- survreg(
  Surv(time, event) ~ treatment + crp_reduction + age + female,
  data = dat, dist = "weibull"
)

# Note: the mediation package supports survreg objects for AFT models
med_surv <- mediate(
  med_model_surv, out_model_surv,
  treat    = "treatment",
  mediator = "crp_reduction",
  sims     = 500
)
summary(med_surv)
```

## Visualization

```r
# 1. Mediation effect plot
plot(med_result, main = "Causal Mediation Analysis Results")

# 2. Custom path diagram summary
results_df <- data.frame(
  Effect = c("ACME (NIE)", "ADE (NDE)", "Total Effect"),
  Estimate = c(med_result$d0, med_result$z0, med_result$tau.coef),
  Lower    = c(med_result$d0.ci[1], med_result$z0.ci[1], med_result$tau.ci[1]),
  Upper    = c(med_result$d0.ci[2], med_result$z0.ci[2], med_result$tau.ci[2])
)

ggplot(results_df, aes(x = Effect, y = Estimate)) +
  geom_point(size = 3) +
  geom_errorbar(aes(ymin = Lower, ymax = Upper), width = 0.2) +
  geom_hline(yintercept = 0, linetype = "dashed", color = "red") +
  coord_flip() +
  labs(title = "Mediation Effect Estimates with 95% CI",
       y = "Effect Estimate", x = "") +
  theme_minimal()

# 3. Sensitivity contour plot
plot(sens, sens.par = "rho", effect.type = "indirect",
     main = "Sensitivity Analysis: ACME vs rho")
```

## Tips and Best Practices

1. **Always include the treatment-mediator interaction** in the outcome model. Omitting
   it biases the NDE/NIE when interaction truly exists.

2. **Use bootstrap (not Sobel) for inference.** The product `a*b` is non-normal in
   finite samples; percentile bootstrap CIs have better coverage.

3. **Report the sensitivity analysis.** Mediation findings are only as credible as the
   sequential-ignorability assumption. Reporting the rho-at-zero value is essential.

4. **Consider the 4-way decomposition** when interaction is plausible. It separates
   mediation from interaction, preventing misattribution.

5. **Watch for post-treatment confounding.** If a variable is a confounder of M-Y but is
   itself affected by T, standard adjustment is biased. Consider inverse-probability
   weighting or g-estimation.

6. **Multiple mediators** require extra no-confounding assumptions between mediators.
   Joint mediation is simpler but less informative; path-specific effects are more
   informative but less identified.

7. **For binary outcomes,** use logistic mediator/outcome models and let `mediate()`
   handle the simulation-based computation of effects on the probability scale.

8. **Pre-register the mediation hypothesis.** Post-hoc mediation is exploratory and
   should be labelled as such.
