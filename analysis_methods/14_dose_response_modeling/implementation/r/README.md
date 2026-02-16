# Dose-Response Modeling — R Implementation

## Required Packages

```r
install.packages(c("DoseFinding", "drda", "ggplot2", "dplyr", "nls2"))
```

## Example Dataset

We simulate a Phase II dose-finding trial for an anti-inflammatory drug. The primary endpoint
is change from baseline in a joint inflammation score (lower is better). Five dose groups
plus placebo are studied.

```r
set.seed(42)

# Dose levels (mg)
doses <- c(0, 25, 50, 100, 200, 400)
n_per_dose <- 50
n_total <- length(doses) * n_per_dose

# True dose-response: sigmoid Emax model
# E0 = 0 (placebo change), Emax = -15 (max reduction), ED50 = 80, Hill = 1.5
true_e0 <- 0
true_emax <- -15
true_ed50 <- 80
true_hill <- 1.5

true_response <- function(d) {
  true_e0 + true_emax * d^true_hill / (true_ed50^true_hill + d^true_hill)
}

# Generate individual-level data
dose_vec <- rep(doses, each = n_per_dose)
true_mean <- true_response(dose_vec)
sigma <- 8
response <- true_mean + rnorm(n_total, 0, sigma)

trial_data <- data.frame(
  dose = dose_vec,
  response = response,
  dose_factor = factor(dose_vec)
)

# Summary by dose group
dose_summary <- trial_data %>%
  group_by(dose) %>%
  summarise(
    n = n(),
    mean_response = mean(response),
    sd_response = sd(response),
    se = sd(response) / sqrt(n()),
    .groups = "drop"
  )
print(dose_summary)
```

## Complete Worked Example

### Step 1: Visualize Raw Dose-Response Data

```r
library(ggplot2)

ggplot(trial_data, aes(x = dose, y = response)) +
  geom_jitter(width = 5, alpha = 0.3, color = "grey50", size = 1) +
  geom_point(data = dose_summary, aes(y = mean_response),
             color = "red", size = 3) +
  geom_errorbar(data = dose_summary,
                aes(y = mean_response,
                    ymin = mean_response - 1.96 * se,
                    ymax = mean_response + 1.96 * se),
                width = 8, color = "red") +
  labs(x = "Dose (mg)", y = "Change in Inflammation Score",
       title = "Phase II Dose-Finding: Raw Data with Group Means") +
  theme_minimal(base_size = 12)
```

### Step 2: MCP-Mod Analysis with `DoseFinding`

```r
library(DoseFinding)

# Define candidate dose-response models
# Each model is specified with guesstimates of key parameters
models <- Mods(
  linear = NULL,
  emax = c(50),              # ED50 guesstimate
  sigEmax = c(80, 3),        # ED50, Hill coefficient
  quadratic = c(-0.002),     # beta2 guesstimate
  exponential = c(200),      # delta parameter
  doses = doses,
  placEff = 0,
  maxEff = -12               # expected maximum effect (change from placebo)
)

# Visualize candidate models
plot(models, main = "Candidate Dose-Response Models for MCP-Mod")

# Step 1: MCP — Test for a dose-response signal
# Compute optimal contrasts and perform multiplicity-adjusted tests
mcp_result <- MCTtest(
  dose = trial_data$dose,
  resp = trial_data$response,
  models = models,
  alpha = 0.025,       # one-sided alpha for regulatory setting
  alternative = "one.sided"
)
print(mcp_result)

# Interpretation: The test statistics and adjusted p-values are shown for each
# candidate model. If any adjusted p-value < 0.025, a dose-response signal is
# detected. The model with the largest test statistic is the best-fitting
# candidate shape.

cat("\n--- MCP Step Results ---\n")
cat("Dose-response signal detected: ")
cat(ifelse(any(attr(mcp_result, "pVal") < 0.025), "YES", "NO"), "\n")
```

### Step 3: Model Fitting (Mod Step)

```r
# Step 2: Mod — Fit the best model(s) and estimate the dose-response curve

# Fit Emax model using nonlinear least squares
emax_fit <- fitMod(
  dose = trial_data$dose,
  resp = trial_data$response,
  model = "sigEmax"
)
summary(emax_fit)

cat("\n--- Fitted Sigmoid Emax Parameters ---\n")
cat(sprintf("E0 (placebo): %.2f\n", coef(emax_fit)["e0"]))
cat(sprintf("Emax: %.2f\n", coef(emax_fit)["eMax"]))
cat(sprintf("ED50: %.2f mg\n", coef(emax_fit)["ed50"]))
cat(sprintf("Hill: %.2f\n", coef(emax_fit)["h"]))

# Compare with true values
cat(sprintf("\nTrue values: E0=%.1f, Emax=%.1f, ED50=%.1f, Hill=%.1f\n",
            true_e0, true_emax, true_ed50, true_hill))

# Predict the dose-response curve
dose_grid <- seq(0, 400, length.out = 200)
pred <- predict(emax_fit, predType = "full-model", doseSeq = dose_grid)

# Estimate target doses
# Dose for 50% of max effect
# Dose for 80% of max effect
cat("\nTarget dose estimates:\n")
td50 <- TD(emax_fit, Delta = abs(coef(emax_fit)["eMax"]) * 0.5, direction = "decreasing")
td80 <- TD(emax_fit, Delta = abs(coef(emax_fit)["eMax"]) * 0.8, direction = "decreasing")
cat(sprintf("  ED50 (50%% of Emax): %.1f mg\n", td50))
cat(sprintf("  ED80 (80%% of Emax): %.1f mg\n", td80))
```

### Step 4: Fit Multiple Models and Compare

```r
# Fit all candidate models
model_names <- c("linear", "emax", "sigEmax", "quadratic", "exponential")
fits <- list()
aic_values <- numeric()

for (mod_name in model_names) {
  tryCatch({
    fit <- fitMod(
      dose = trial_data$dose,
      resp = trial_data$response,
      model = mod_name
    )
    fits[[mod_name]] <- fit
    aic_values[mod_name] <- AIC(fit)
  }, error = function(e) {
    cat(sprintf("Model %s failed to converge: %s\n", mod_name, e$message))
  })
}

# Model comparison
aic_df <- data.frame(
  Model = names(aic_values),
  AIC = aic_values,
  deltaAIC = aic_values - min(aic_values)
)
aic_df$weight <- exp(-0.5 * aic_df$deltaAIC) / sum(exp(-0.5 * aic_df$deltaAIC))
aic_df <- aic_df[order(aic_df$AIC), ]

cat("\n--- Model Comparison ---\n")
print(aic_df, row.names = FALSE)
# Interpretation: The model with the lowest AIC (and highest weight) fits best.
# The sigmoid Emax should have the best fit since it matches the true DGP.
```

## Advanced Example

### Model Averaging

```r
# Compute model-averaged predictions
dose_grid <- seq(0, 400, length.out = 200)
pred_matrix <- matrix(NA, nrow = length(dose_grid), ncol = length(fits))
colnames(pred_matrix) <- names(fits)

for (mod_name in names(fits)) {
  pred_matrix[, mod_name] <- predict(fits[[mod_name]], predType = "full-model",
                                      doseSeq = dose_grid)
}

# Weighted average using AIC weights
weights <- aic_df$weight
names(weights) <- aic_df$Model
averaged_pred <- numeric(length(dose_grid))
for (mod_name in names(fits)) {
  averaged_pred <- averaged_pred + weights[mod_name] * pred_matrix[, mod_name]
}

# Plot all models plus the averaged prediction
plot_df <- data.frame(
  dose = rep(dose_grid, length(fits) + 1),
  prediction = c(as.vector(pred_matrix), averaged_pred),
  model = c(rep(names(fits), each = length(dose_grid)),
            rep("Model Average", length(dose_grid)))
)

ggplot() +
  geom_point(data = dose_summary, aes(x = dose, y = mean_response),
             color = "black", size = 3) +
  geom_errorbar(data = dose_summary,
                aes(x = dose, y = mean_response,
                    ymin = mean_response - 1.96 * se,
                    ymax = mean_response + 1.96 * se),
                width = 5) +
  geom_line(data = plot_df[plot_df$model != "Model Average", ],
            aes(x = dose, y = prediction, color = model), alpha = 0.5, linewidth = 0.7) +
  geom_line(data = plot_df[plot_df$model == "Model Average", ],
            aes(x = dose, y = prediction), color = "red", linewidth = 1.5) +
  labs(x = "Dose (mg)", y = "Change in Inflammation Score",
       title = "Dose-Response: Individual Models and Model Average",
       color = "Model") +
  theme_minimal(base_size = 12)
```

### Manual Emax Fit with `nls()`

```r
# For educational purposes: fit Emax model using base R nls()
emax_nls <- nls(
  response ~ e0 + emax * dose^h / (ed50^h + dose^h),
  data = trial_data,
  start = list(e0 = 0, emax = -12, ed50 = 100, h = 1),
  lower = c(-10, -30, 1, 0.1),
  upper = c(10, 0, 500, 5),
  algorithm = "port"
)

summary(emax_nls)
cat("\nParameter estimates from nls():\n")
print(coef(emax_nls))
# Should closely match DoseFinding results.

# Profile confidence intervals
ci <- confint(emax_nls, level = 0.95)
print(ci)
```

## Visualization

```r
library(ggplot2)
library(gridExtra)

# Plot 1: Fitted dose-response curve with confidence band
dose_grid <- seq(0, 400, length.out = 200)
pred_emax <- predict(emax_nls, newdata = data.frame(dose = dose_grid))

# Bootstrap confidence intervals
set.seed(99)
n_boot <- 500
boot_preds <- matrix(NA, nrow = n_boot, ncol = length(dose_grid))
for (b in 1:n_boot) {
  boot_idx <- sample(nrow(trial_data), replace = TRUE)
  boot_data <- trial_data[boot_idx, ]
  tryCatch({
    boot_fit <- nls(
      response ~ e0 + emax * dose^h / (ed50^h + dose^h),
      data = boot_data,
      start = coef(emax_nls),
      lower = c(-10, -30, 1, 0.1),
      upper = c(10, 0, 500, 5),
      algorithm = "port",
      control = nls.control(maxiter = 100, warnOnly = TRUE)
    )
    boot_preds[b, ] <- predict(boot_fit, newdata = data.frame(dose = dose_grid))
  }, error = function(e) {})
}

ci_lower <- apply(boot_preds, 2, quantile, 0.025, na.rm = TRUE)
ci_upper <- apply(boot_preds, 2, quantile, 0.975, na.rm = TRUE)

curve_df <- data.frame(dose = dose_grid, pred = pred_emax,
                        lower = ci_lower, upper = ci_upper)

p1 <- ggplot() +
  geom_ribbon(data = curve_df, aes(x = dose, ymin = lower, ymax = upper),
              fill = "steelblue", alpha = 0.2) +
  geom_line(data = curve_df, aes(x = dose, y = pred),
            color = "steelblue", linewidth = 1.2) +
  geom_point(data = dose_summary, aes(x = dose, y = mean_response),
             color = "red", size = 3) +
  geom_errorbar(data = dose_summary,
                aes(x = dose, y = mean_response,
                    ymin = mean_response - 1.96 * se,
                    ymax = mean_response + 1.96 * se),
                width = 8, color = "red") +
  labs(x = "Dose (mg)", y = "Change from Baseline",
       title = "Sigmoid Emax Fit with 95% Bootstrap CI") +
  theme_minimal(base_size = 11)

# Plot 2: Residuals
residuals_df <- data.frame(
  dose = trial_data$dose,
  residual = residuals(emax_nls)
)

p2 <- ggplot(residuals_df, aes(x = factor(dose), y = residual)) +
  geom_boxplot(fill = "lightblue", alpha = 0.7) +
  geom_hline(yintercept = 0, linetype = "dashed", color = "red") +
  labs(x = "Dose (mg)", y = "Residual",
       title = "Residuals by Dose Group") +
  theme_minimal(base_size = 11)

# Plot 3: AIC weights
p3 <- ggplot(aic_df, aes(x = reorder(Model, -weight), y = weight)) +
  geom_col(fill = "darkorange", alpha = 0.8) +
  geom_text(aes(label = sprintf("%.3f", weight)), vjust = -0.3) +
  labs(x = "", y = "AIC Weight",
       title = "Model Weights (AIC-based)") +
  theme_minimal(base_size = 11)

# Plot 4: Target dose identification
p4 <- ggplot() +
  geom_line(data = curve_df, aes(x = dose, y = pred),
            color = "steelblue", linewidth = 1.2) +
  geom_hline(yintercept = coef(emax_nls)["emax"] * 0.8,
             linetype = "dashed", color = "darkgreen") +
  geom_vline(xintercept = td80, linetype = "dashed", color = "darkgreen") +
  annotate("text", x = td80 + 20, y = -2,
           label = sprintf("ED80 = %.0f mg", td80), hjust = 0, color = "darkgreen") +
  labs(x = "Dose (mg)", y = "Predicted Response",
       title = "Target Dose Identification (80% of Emax)") +
  theme_minimal(base_size = 11)

grid.arrange(p1, p2, p3, p4, ncol = 2)
```

## Tips and Best Practices

1. **Pre-specify candidate models**: For MCP-Mod, define candidate models and their parameter
   guesstimates before seeing the data. A standard set (linear, Emax, sigmoid Emax,
   exponential, quadratic) covers most shapes encountered in practice.

2. **Use clinically informed guesstimates**: The ED50 guesstimate for MCP-Mod does not need
   to be exact. It determines the shape of the contrast vector. Explore a range of values
   to ensure robustness.

3. **Check convergence**: Nonlinear models can fail to converge. Always inspect convergence
   diagnostics, try multiple starting values, and consider bounded optimization (`algorithm = "port"`).

4. **Model averaging for prediction**: When multiple models fit similarly well, use AIC-weighted
   model averaging rather than selecting a single model. This accounts for model uncertainty.

5. **Do not extrapolate**: The fitted curve is reliable only within the range of studied doses.
   Predicting effects at doses higher than the maximum studied dose is unreliable.

6. **Consider the endpoint**: For binary endpoints (responder rates), use generalized models
   (logistic Emax). For time-to-event endpoints, adapt accordingly.

7. **Report target doses with uncertainty**: When estimating ED80 or other target doses, always
   provide confidence intervals (via bootstrap or delta method). A point estimate alone can
   be misleading.

8. **Regulatory context**: MCP-Mod has been qualified by both EMA (2014) and FDA as an
   acceptable method for Phase II dose-finding. Document the procedure following their
   guidance for regulatory submissions.
