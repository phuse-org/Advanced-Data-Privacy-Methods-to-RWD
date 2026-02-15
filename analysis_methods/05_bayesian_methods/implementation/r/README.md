# Bayesian Methods — R Implementation

## Required Packages

```r
install.packages(c("brms", "rstanarm", "bayesplot", "loo", "tidybayes",
                    "ggplot2", "dplyr", "posterior"))

library(brms)
library(rstanarm)
library(bayesplot)
library(loo)
library(tidybayes)
library(ggplot2)
library(dplyr)
library(posterior)
```

Note: `brms` and `rstanarm` require a working C++ toolchain and RStan. On first install, run:
```r
install.packages("rstan", repos = c("https://mc-stan.org/r-packages/", getOption("repos")))
```

## Example Dataset

We use a simulated multi-center clinical trial dataset where patients receive either treatment or placebo and the outcome is change in a continuous biomarker (e.g., LDL cholesterol reduction). The hierarchical structure (patients within centers) makes this ideal for Bayesian multilevel modeling.

```r
set.seed(42)
n_centers <- 12
n_per_center <- rpois(n_centers, lambda = 40)
n_total <- sum(n_per_center)

# Center-level random effects
center_effects <- rnorm(n_centers, 0, 3)

# Generate patient-level data
trial_data <- data.frame(
  center_id = factor(rep(1:n_centers, n_per_center)),
  treatment = rbinom(n_total, 1, 0.5),
  age = round(rnorm(n_total, 58, 10)),
  sex = factor(rbinom(n_total, 1, 0.45), labels = c("Female", "Male")),
  baseline_ldl = round(rnorm(n_total, 160, 30), 1)
)

# Outcome: LDL reduction (positive = good)
center_idx <- as.numeric(trial_data$center_id)
trial_data$ldl_reduction <- round(
  15 + 12 * trial_data$treatment + 0.1 * trial_data$age -
    2 * (trial_data$sex == "Male") + 0.05 * trial_data$baseline_ldl +
    center_effects[center_idx] + rnorm(n_total, 0, 8), 1
)

# Binary outcome: clinically meaningful reduction (> 20 mg/dL)
trial_data$meaningful_reduction <- as.integer(trial_data$ldl_reduction > 20)

cat("Total patients:", n_total, "\n")
cat("Number of centers:", n_centers, "\n")
cat("Treatment rate:", mean(trial_data$treatment), "\n")
cat("Mean LDL reduction (treatment):", mean(trial_data$ldl_reduction[trial_data$treatment == 1]), "\n")
cat("Mean LDL reduction (placebo):", mean(trial_data$ldl_reduction[trial_data$treatment == 0]), "\n")
```

## Complete Worked Example

### Step 1: Bayesian Linear Regression with brms

```r
library(brms)

# Define priors
priors <- c(
  prior(normal(0, 20), class = "Intercept"),
  prior(normal(0, 10), class = "b"),
  prior(student_t(3, 0, 10), class = "sigma")
)

# Fit Bayesian linear model
bayes_lm <- brm(
  ldl_reduction ~ treatment + age + sex + baseline_ldl,
  data = trial_data,
  family = gaussian(),
  prior = priors,
  chains = 4,
  iter = 4000,
  warmup = 1000,
  seed = 42,
  cores = 4
)

summary(bayes_lm)
```

**Interpretation**: The summary provides posterior means, estimation errors, 95% credible intervals, Rhat, and ESS for each parameter. The treatment effect posterior mean should be close to the true value of 12. The 95% CI gives the range of plausible treatment effects given the data and priors. Rhat near 1.00 and ESS > 400 indicate good convergence.

### Step 2: MCMC Diagnostics

```r
library(bayesplot)

# Extract posterior draws
posterior_draws <- as_draws_df(bayes_lm)

# Trace plots
mcmc_trace(bayes_lm, pars = c("b_treatment", "b_age", "sigma"))

# Rank histograms (more sensitive than trace plots)
mcmc_rank_overlay(bayes_lm, pars = c("b_treatment", "b_age", "sigma"))

# Autocorrelation
mcmc_acf(bayes_lm, pars = c("b_treatment", "b_age"))

# Summary statistics with convergence diagnostics
summarise_draws(posterior_draws,
                mean, median, sd,
                ~quantile2(.x, probs = c(0.025, 0.975)),
                rhat, ess_bulk, ess_tail) %>%
  filter(variable %in% c("b_treatment", "b_age", "b_sexMale", "b_baseline_ldl", "sigma")) %>%
  print()
```

**Interpretation**: Well-behaved trace plots show overlapping chains that explore the same region ("hairy caterpillar" pattern). Rank histograms should be uniform. Autocorrelation should decay rapidly to zero. Rhat < 1.01 and ESS > 400 for both bulk and tail confirm adequate sampling.

### Step 3: Posterior Summaries and Visualization

```r
# Posterior density of treatment effect
mcmc_areas(bayes_lm, pars = "b_treatment",
           prob = 0.95, prob_outer = 0.99) +
  ggtitle("Posterior Distribution: Treatment Effect on LDL Reduction") +
  xlab("Treatment Effect (mg/dL)")

# Probability that treatment effect > 0
treatment_draws <- as_draws_df(bayes_lm)$b_treatment
cat("P(treatment effect > 0):", mean(treatment_draws > 0), "\n")
cat("P(treatment effect > 10):", mean(treatment_draws > 10), "\n")
cat("P(treatment effect > 15):", mean(treatment_draws > 15), "\n")
```

**Interpretation**: These direct probability statements are a key advantage of Bayesian analysis. We can say, for example, "there is a 99.5% posterior probability that the treatment reduces LDL by more than 0 mg/dL" and "a 78% probability that the reduction exceeds 10 mg/dL." These are intuitive for clinical decision-making.

### Step 4: Bayesian Hierarchical Model

```r
# Hierarchical model with random intercepts for centers
bayes_hier <- brm(
  ldl_reduction ~ treatment + age + sex + baseline_ldl + (1 | center_id),
  data = trial_data,
  family = gaussian(),
  prior = c(
    prior(normal(0, 20), class = "Intercept"),
    prior(normal(0, 10), class = "b"),
    prior(student_t(3, 0, 10), class = "sigma"),
    prior(student_t(3, 0, 5), class = "sd")
  ),
  chains = 4,
  iter = 4000,
  warmup = 1000,
  seed = 42,
  cores = 4
)

summary(bayes_hier)

# Random effects (center-specific intercepts)
ranef(bayes_hier)

# Caterpillar plot of random effects
library(tidybayes)
trial_data %>%
  distinct(center_id) %>%
  add_epred_draws(bayes_hier, re_formula = ~ (1 | center_id),
                  ndraws = 500) %>%
  ggplot(aes(x = .epred, y = center_id)) +
  stat_pointinterval(.width = c(0.66, 0.95)) +
  labs(x = "Predicted LDL Reduction", y = "Center",
       title = "Center-Specific Predictions (Hierarchical Model)") +
  theme_minimal()
```

**Interpretation**: The hierarchical model partially pools center-specific intercepts toward the grand mean. Centers with fewer patients or more extreme estimates are shrunk more toward the overall mean. This borrowing of strength is automatic in the Bayesian framework and improves estimation for small centers.

### Step 5: Bayesian Logistic Regression

```r
# Bayesian logistic regression for clinically meaningful reduction
bayes_logit <- brm(
  meaningful_reduction ~ treatment + age + sex + baseline_ldl,
  data = trial_data,
  family = bernoulli(link = "logit"),
  prior = c(
    prior(normal(0, 5), class = "Intercept"),
    prior(normal(0, 2.5), class = "b")
  ),
  chains = 4,
  iter = 4000,
  warmup = 1000,
  seed = 42,
  cores = 4
)

summary(bayes_logit)

# Posterior odds ratios
or_draws <- exp(as_draws_df(bayes_logit)[, c("b_treatment", "b_age", "b_sexMale", "b_baseline_ldl")])
cat("\nPosterior Odds Ratios (median and 95% CrI):\n")
apply(or_draws, 2, function(x) {
  c(median = median(x), `2.5%` = quantile(x, 0.025), `97.5%` = quantile(x, 0.975))
}) %>% round(3) %>% print()
```

**Interpretation**: The Bayesian logistic model provides posterior distributions for odds ratios. The credible interval for the treatment OR indicates the range of plausible odds ratios. A 95% CrI that excludes 1 indicates strong evidence of a treatment effect on the probability of meaningful LDL reduction.

### Step 6: Prior Sensitivity Analysis

```r
# Fit with different priors for the treatment effect
# Weakly informative
fit_weak <- brm(
  ldl_reduction ~ treatment + age + sex + baseline_ldl,
  data = trial_data, family = gaussian(),
  prior = prior(normal(0, 10), class = "b"),
  chains = 4, iter = 4000, warmup = 1000, seed = 42, cores = 4,
  silent = 2
)

# Skeptical prior (centered near zero, tight)
fit_skeptical <- brm(
  ldl_reduction ~ treatment + age + sex + baseline_ldl,
  data = trial_data, family = gaussian(),
  prior = prior(normal(0, 3), class = "b"),
  chains = 4, iter = 4000, warmup = 1000, seed = 42, cores = 4,
  silent = 2
)

# Enthusiastic prior (favoring treatment)
fit_enthusiastic <- brm(
  ldl_reduction ~ treatment + age + sex + baseline_ldl,
  data = trial_data, family = gaussian(),
  prior = prior(normal(10, 5), class = "b", coef = "treatment"),
  chains = 4, iter = 4000, warmup = 1000, seed = 42, cores = 4,
  silent = 2
)

# Compare treatment effect posteriors
comparison <- data.frame(
  Prior = c("Weakly Informative", "Skeptical", "Enthusiastic"),
  Posterior_Mean = c(
    fixef(fit_weak)["treatment", "Estimate"],
    fixef(fit_skeptical)["treatment", "Estimate"],
    fixef(fit_enthusiastic)["treatment", "Estimate"]
  ),
  Lower_95 = c(
    fixef(fit_weak)["treatment", "Q2.5"],
    fixef(fit_skeptical)["treatment", "Q2.5"],
    fixef(fit_enthusiastic)["treatment", "Q2.5"]
  ),
  Upper_95 = c(
    fixef(fit_weak)["treatment", "Q97.5"],
    fixef(fit_skeptical)["treatment", "Q97.5"],
    fixef(fit_enthusiastic)["treatment", "Q97.5"]
  )
)
print(round(comparison, 2))
```

**Interpretation**: Prior sensitivity analysis demonstrates how different prior assumptions affect the posterior. With sufficient data, the posteriors should be similar across reasonable priors (the data dominate). Large discrepancies indicate that the prior is influential, warranting careful justification.

## Advanced Example

### Model Comparison with LOO-CV

```r
library(loo)

# Compare models using LOO-CV
loo_lm <- loo(bayes_lm)
loo_hier <- loo(bayes_hier)

print(loo_lm)
print(loo_hier)

# Formal comparison
loo_compare(loo_lm, loo_hier)
```

**Interpretation**: LOO-CV compares models based on out-of-sample predictive performance. The model with the higher `elpd_loo` (expected log pointwise predictive density) is preferred. The `loo_compare` function provides the difference and standard error. A difference of more than 4 in elpd is considered meaningful.

### Posterior Predictive Check

```r
# Posterior predictive check for the hierarchical model
pp_check(bayes_hier, type = "dens_overlay", ndraws = 50) +
  ggtitle("Posterior Predictive Check: Hierarchical Model")

# Distribution of test statistic
pp_check(bayes_hier, type = "stat", stat = "mean", ndraws = 1000) +
  ggtitle("Posterior Predictive Check: Mean")

pp_check(bayes_hier, type = "stat", stat = "sd", ndraws = 1000) +
  ggtitle("Posterior Predictive Check: SD")

# Grouped PPC by treatment
pp_check(bayes_hier, type = "dens_overlay_grouped",
         group = "treatment", ndraws = 50) +
  ggtitle("Posterior Predictive Check by Treatment Group")
```

**Interpretation**: Posterior predictive checks (PPCs) compare the distribution of observed data to data simulated from the posterior. If the model is well-specified, the observed data should look like a typical replicate. Systematic discrepancies (e.g., wrong skewness, bimodality) indicate model misfit.

### Using rstanarm for Quick Bayesian Models

```r
library(rstanarm)

# rstanarm provides a simpler interface for standard models
rstan_fit <- stan_glm(
  ldl_reduction ~ treatment + age + sex + baseline_ldl,
  data = trial_data,
  family = gaussian(),
  prior = normal(0, 10),
  prior_intercept = normal(0, 20),
  seed = 42,
  chains = 4,
  iter = 4000
)

print(rstan_fit, digits = 3)
posterior_interval(rstan_fit, prob = 0.95)

# Prior summary
prior_summary(rstan_fit)
```

**Interpretation**: `rstanarm` uses the same Stan engine as `brms` but provides a more familiar interface mirroring `glm()` syntax. It is ideal for standard models where the `brms` formula syntax is not needed.

## Visualization

### Posterior Distributions

```r
# Posterior intervals for all fixed effects
mcmc_intervals(bayes_hier, pars = c("b_treatment", "b_age", "b_sexMale",
                                      "b_baseline_ldl"),
               prob = 0.8, prob_outer = 0.95) +
  ggtitle("Posterior Intervals: Fixed Effects")
```

### Prior vs. Posterior Comparison

```r
# Compare prior and posterior for treatment effect
prior_draws <- rnorm(10000, 0, 10)
posterior_treatment <- as_draws_df(bayes_hier)$b_treatment

ggplot() +
  geom_density(aes(x = prior_draws), fill = "gray80", alpha = 0.5) +
  geom_density(aes(x = posterior_treatment), fill = "steelblue", alpha = 0.5) +
  geom_vline(xintercept = 12, color = "red", linetype = "dashed", linewidth = 1) +
  annotate("text", x = 12, y = 0.01, label = "True effect = 12", color = "red") +
  labs(x = "Treatment Effect (mg/dL)", y = "Density",
       title = "Prior (gray) vs. Posterior (blue) for Treatment Effect") +
  theme_minimal(base_size = 12)
```

### Shrinkage Plot for Hierarchical Model

```r
# Show shrinkage of center-specific estimates toward the grand mean
center_raw <- trial_data %>%
  group_by(center_id) %>%
  summarise(raw_mean = mean(ldl_reduction), n = n())

center_posterior <- ranef(bayes_hier)$center_id[, , "Intercept"]
center_raw$posterior_mean <- fixef(bayes_hier)["Intercept", "Estimate"] +
  center_posterior[, "Estimate"]
center_raw$grand_mean <- fixef(bayes_hier)["Intercept", "Estimate"]

ggplot(center_raw, aes(x = raw_mean, y = posterior_mean)) +
  geom_point(aes(size = n), color = "steelblue") +
  geom_abline(slope = 1, intercept = 0, linestyle = "dashed", color = "gray50") +
  geom_hline(yintercept = mean(center_raw$grand_mean), color = "red", linetype = "dotted") +
  labs(x = "Raw Center Mean", y = "Posterior Center Mean (Shrunk)",
       title = "Bayesian Shrinkage: Center-Specific Estimates",
       size = "Sample Size") +
  theme_minimal(base_size = 12)
```

**Interpretation**: The shrinkage plot shows how center-specific estimates are pulled toward the grand mean (red line). Centers with smaller samples (smaller points) are shrunk more. Points above/below the diagonal line are shrunk toward the grand mean. This partial pooling is a key benefit of hierarchical Bayesian models.

## Tips and Best Practices

1. **Always run convergence diagnostics**: Check Rhat (< 1.01), ESS (> 400), trace plots, and rank histograms before interpreting results. Never skip this step.

2. **Use weakly informative priors as default**: `brms` and `rstanarm` set sensible defaults. Avoid completely flat priors, which can cause sampling difficulties.

3. **Conduct prior sensitivity analysis**: Fit the model with different priors and check if conclusions change. Report sensitivity results alongside the primary analysis.

4. **Use posterior predictive checks**: PPCs are the Bayesian analog of residual diagnostics. Use `pp_check()` to verify model adequacy.

5. **Prefer LOO-CV over DIC for model comparison**: WAIC and PSIS-LOO are more reliable than DIC. The `loo` package makes this easy.

6. **Report full posterior summaries**: Report posterior mean, median, 95% CrI, and the probability of clinically meaningful effects. Do not reduce Bayesian results to a single p-value equivalent.

7. **Use `brms` for flexibility, `rstanarm` for speed**: `brms` can fit virtually any model Stan can handle. `rstanarm` is faster for standard GLM-type models because it uses pre-compiled Stan code.

8. **Increase `adapt_delta` for divergent transitions**: If you see warnings about divergent transitions, increase `adapt_delta` (e.g., `control = list(adapt_delta = 0.99)`).

9. **Consider computational resources**: Bayesian models can be slow. Use `cores = 4` (or more) for parallel chains. For very large datasets, consider variational Bayes via `brm(..., algorithm = "meanfield")` as a quick approximation.

10. **Leverage hierarchical modeling**: Whenever data has a natural grouping structure (centers, patients, time points), use random effects. Bayesian methods handle hierarchical models more gracefully than frequentist methods, especially with few groups or unbalanced designs.
