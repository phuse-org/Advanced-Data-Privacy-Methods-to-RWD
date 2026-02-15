# Diagnostic Test Accuracy and Biomarker Evaluation — R Implementation

## Required Packages

```r
install.packages(c("pROC", "OptimalCutpoints", "dcurves", "mada",
                   "ggplot2", "dplyr"))
```

## Example Dataset

We simulate a clinical dataset evaluating a novel cardiac biomarker (hs-cTnX) for diagnosing
acute myocardial infarction (AMI) in patients presenting to the emergency department with
chest pain.

```r
set.seed(42)

n <- 800
prevalence <- 0.25  # 25% of chest pain patients have AMI

# Disease status
ami <- rbinom(n, 1, prevalence)

# Biomarker: higher in AMI patients (log-normal distributed)
biomarker <- ifelse(ami == 1,
                    exp(rnorm(n, mean = 3.5, sd = 0.8)),   # AMI: mean ~33 ng/L
                    exp(rnorm(n, mean = 2.0, sd = 0.9)))   # No AMI: mean ~7 ng/L

# Clinical covariates
age <- rnorm(n, mean = ifelse(ami == 1, 68, 58), sd = 12)
male <- rbinom(n, 1, prob = ifelse(ami == 1, 0.65, 0.50))

dx_data <- data.frame(
  ami = ami,
  biomarker = biomarker,
  age = age,
  male = male
)

cat(sprintf("Dataset: %d patients, %d AMI cases (%.1f%% prevalence)\n",
            n, sum(ami), 100 * mean(ami)))
cat(sprintf("Biomarker median - AMI: %.1f ng/L, No AMI: %.1f ng/L\n",
            median(biomarker[ami == 1]), median(biomarker[ami == 0])))
```

## Complete Worked Example

### Step 1: ROC Analysis with `pROC`

```r
library(pROC)

# Compute ROC curve
roc_obj <- roc(dx_data$ami, dx_data$biomarker,
               levels = c(0, 1), direction = "<",
               ci = TRUE, ci.method = "delong")

print(roc_obj)
cat(sprintf("\nAUC: %.3f (95%% CI: %.3f - %.3f)\n",
            auc(roc_obj), ci.auc(roc_obj)[1], ci.auc(roc_obj)[3]))

# Interpretation:
# AUC > 0.8 suggests good discrimination. The 95% CI from the DeLong method
# quantifies uncertainty. If the CI includes 0.5, the biomarker has no
# discriminatory ability.

# Partial AUC (high-specificity region: Sp >= 0.9)
pauc <- auc(roc_obj, partial.auc = c(1, 0.9), partial.auc.focus = "sp")
cat(sprintf("Partial AUC (Sp >= 0.9): %.4f\n", pauc))
```

### Step 2: Plot ROC Curve

```r
# Base R plot from pROC
plot(roc_obj, main = "ROC Curve: hs-cTnX for AMI Diagnosis",
     print.auc = TRUE, print.auc.y = 0.4,
     col = "steelblue", lwd = 2)

# ggplot version for publication
roc_df <- data.frame(
  sensitivity = roc_obj$sensitivities,
  specificity = roc_obj$specificities,
  threshold = roc_obj$thresholds
)

library(ggplot2)
ggplot(roc_df, aes(x = 1 - specificity, y = sensitivity)) +
  geom_line(color = "steelblue", linewidth = 1.2) +
  geom_abline(slope = 1, intercept = 0, linetype = "dashed", color = "grey50") +
  annotate("text", x = 0.6, y = 0.3,
           label = sprintf("AUC = %.3f", auc(roc_obj)), size = 5) +
  labs(x = "1 - Specificity (False Positive Rate)",
       y = "Sensitivity (True Positive Rate)",
       title = "ROC Curve: hs-cTnX for AMI Diagnosis") +
  theme_minimal(base_size = 12) +
  coord_equal()
```

### Step 3: Optimal Cutpoint Selection

```r
library(OptimalCutpoints)

# Youden index (maximizes Se + Sp - 1)
opt_youden <- optimal.cutpoints(
  X = "biomarker",
  status = "ami",
  tag.healthy = 0,
  methods = "Youden",
  data = dx_data
)
summary(opt_youden)

# Extract optimal cutpoint
youden_cut <- opt_youden$Youden$Global$optimal.cutoff$cutoff
youden_se <- opt_youden$Youden$Global$optimal.cutoff$Se
youden_sp <- opt_youden$Youden$Global$optimal.cutoff$Sp

cat(sprintf("\n--- Optimal Cutpoint (Youden) ---\n"))
cat(sprintf("Cutpoint: %.2f ng/L\n", youden_cut))
cat(sprintf("Sensitivity: %.3f\n", youden_se))
cat(sprintf("Specificity: %.3f\n", youden_sp))

# Also compute from pROC for verification
coords_youden <- coords(roc_obj, "best", best.method = "youden", ret = "all")
print(coords_youden)

# Additional cutpoint methods
# Closest to (0,1) corner
coords_closest <- coords(roc_obj, "best", best.method = "closest.topleft", ret = "all")
cat(sprintf("\nClosest-to-(0,1): cutpoint = %.2f, Se = %.3f, Sp = %.3f\n",
            coords_closest$threshold, coords_closest$sensitivity,
            coords_closest$specificity))
```

### Step 4: Performance Metrics at the Optimal Cutpoint

```r
# Apply the Youden cutpoint
predicted_pos <- ifelse(dx_data$biomarker >= youden_cut, 1, 0)

# Confusion matrix
tp <- sum(predicted_pos == 1 & dx_data$ami == 1)
fp <- sum(predicted_pos == 1 & dx_data$ami == 0)
tn <- sum(predicted_pos == 0 & dx_data$ami == 0)
fn <- sum(predicted_pos == 0 & dx_data$ami == 1)

se <- tp / (tp + fn)
sp <- tn / (tn + fp)
ppv <- tp / (tp + fp)
npv <- tn / (tn + fn)
lr_pos <- se / (1 - sp)
lr_neg <- (1 - se) / sp
accuracy <- (tp + tn) / n

cat("\n--- Performance at Youden Cutpoint ---\n")
cat(sprintf("Sensitivity: %.3f\n", se))
cat(sprintf("Specificity: %.3f\n", sp))
cat(sprintf("PPV: %.3f\n", ppv))
cat(sprintf("NPV: %.3f\n", npv))
cat(sprintf("LR+: %.2f\n", lr_pos))
cat(sprintf("LR-: %.3f\n", lr_neg))
cat(sprintf("Accuracy: %.3f\n", accuracy))
cat(sprintf("\nNote: PPV=%.3f reflects the study prevalence of %.1f%%.\n",
            ppv, 100 * prevalence))
cat("In a lower-prevalence population, PPV would be lower.\n")
```

### Step 5: Decision Curve Analysis

```r
library(dcurves)

# Fit a logistic regression model using the biomarker
dx_data$biomarker_log <- log(dx_data$biomarker)
logit_model <- glm(ami ~ biomarker_log, data = dx_data, family = binomial)
dx_data$pred_prob <- predict(logit_model, type = "response")

# A model using biomarker + clinical variables
logit_full <- glm(ami ~ biomarker_log + age + male, data = dx_data, family = binomial)
dx_data$pred_prob_full <- predict(logit_full, type = "response")

# Decision curve analysis
dca_result <- dca(
  ami ~ pred_prob + pred_prob_full,
  data = dx_data,
  thresholds = seq(0.01, 0.60, by = 0.01),
  label = list(pred_prob = "Biomarker Only", pred_prob_full = "Biomarker + Clinical")
)

plot(dca_result) +
  labs(title = "Decision Curve Analysis: Biomarker for AMI Diagnosis",
       x = "Threshold Probability",
       y = "Net Benefit") +
  theme_minimal(base_size = 12)

# Interpretation: The DCA plot shows net benefit vs threshold probability.
# - "Treat All" line: net benefit of treating everyone.
# - "Treat None" line: net benefit = 0 (horizontal at y=0).
# - A model is clinically useful if its net benefit exceeds both default
#   strategies across a range of clinically relevant thresholds.
# For AMI rule-out, thresholds of 0.05-0.15 are most relevant.
```

## Advanced Example

### Comparing Two Biomarkers (DeLong Test)

```r
# Simulate a second, inferior biomarker
dx_data$biomarker2 <- ifelse(dx_data$ami == 1,
                              exp(rnorm(n, mean = 3.0, sd = 1.0)),
                              exp(rnorm(n, mean = 2.2, sd = 0.9)))

roc_obj2 <- roc(dx_data$ami, dx_data$biomarker2, levels = c(0, 1), direction = "<")

# DeLong test for comparing AUCs
comparison <- roc.test(roc_obj, roc_obj2, method = "delong")
print(comparison)

cat(sprintf("\nBiomarker 1 AUC: %.3f\n", auc(roc_obj)))
cat(sprintf("Biomarker 2 AUC: %.3f\n", auc(roc_obj2)))
cat(sprintf("DeLong test p-value: %.4f\n", comparison$p.value))
cat(sprintf("Conclusion: %s\n",
            ifelse(comparison$p.value < 0.05,
                   "AUCs are significantly different",
                   "No significant difference in AUCs")))
```

### Basic DTA Meta-Analysis with `mada`

```r
library(mada)

# Simulated data from 10 studies evaluating the biomarker
set.seed(123)
n_studies <- 10
meta_data <- data.frame(
  TP = c(85, 92, 78, 88, 90, 82, 87, 93, 80, 86),
  FP = c(12, 8, 15, 10, 7, 14, 11, 6, 13, 9),
  FN = c(15, 8, 22, 12, 10, 18, 13, 7, 20, 14),
  TN = c(188, 192, 185, 190, 193, 186, 189, 194, 187, 191)
)

# Bivariate model (Reitsma et al.)
bivariate_fit <- reitsma(meta_data)
summary(bivariate_fit)

# Summary sensitivity and specificity
cat("\n--- DTA Meta-Analysis (Bivariate Model) ---\n")
summ <- summary(bivariate_fit)
cat(sprintf("Summary Sensitivity: %.3f (95%% CI: %.3f - %.3f)\n",
            summ$coefficients["tsens", "Estimate"],
            summ$coefficients["tsens", "Estimate"] - 1.96 * summ$coefficients["tsens", "Std. Error"],
            summ$coefficients["tsens", "Estimate"] + 1.96 * summ$coefficients["tsens", "Std. Error"]))

# SROC plot
plot(bivariate_fit, sroclwd = 2,
     main = "Summary ROC Curve (Bivariate Meta-Analysis)")
points(fpr(meta_data), sens(meta_data), pch = 19, col = "steelblue")

# Forest plots for sensitivity and specificity
par(mfrow = c(1, 2))
forest(madauni(meta_data, type = "sens"),
       main = "Forest Plot: Sensitivity")
forest(madauni(meta_data, type = "spec"),
       main = "Forest Plot: Specificity")
par(mfrow = c(1, 1))
```

## Visualization

```r
library(ggplot2)
library(gridExtra)

# Plot 1: ROC curves for two biomarkers
p1 <- ggroc(list(Biomarker1 = roc_obj, Biomarker2 = roc_obj2)) +
  geom_abline(slope = 1, intercept = 1, linetype = "dashed", color = "grey") +
  scale_color_manual(values = c("steelblue", "darkorange")) +
  annotate("text", x = 0.3, y = 0.25,
           label = sprintf("AUC1 = %.3f\nAUC2 = %.3f\np = %.3f",
                           auc(roc_obj), auc(roc_obj2), comparison$p.value),
           size = 3.5) +
  labs(title = "ROC Comparison: Two Biomarkers", color = "Biomarker") +
  theme_minimal(base_size = 11)

# Plot 2: Biomarker distribution by disease status
p2 <- ggplot(dx_data, aes(x = log(biomarker), fill = factor(ami))) +
  geom_density(alpha = 0.5) +
  geom_vline(xintercept = log(youden_cut), linetype = "dashed", color = "red") +
  annotate("text", x = log(youden_cut) + 0.2, y = 0.5,
           label = sprintf("Cutpoint\n%.1f ng/L", youden_cut),
           color = "red", size = 3) +
  scale_fill_manual(values = c("steelblue", "red"),
                    labels = c("No AMI", "AMI")) +
  labs(x = "log(Biomarker)", y = "Density", fill = "Status",
       title = "Biomarker Distribution by Disease Status") +
  theme_minimal(base_size = 11)

# Plot 3: Sensitivity and Specificity vs threshold
threshold_grid <- seq(min(dx_data$biomarker), max(dx_data$biomarker),
                       length.out = 200)
se_grid <- sapply(threshold_grid, function(t) {
  sum(dx_data$biomarker >= t & dx_data$ami == 1) / sum(dx_data$ami == 1)
})
sp_grid <- sapply(threshold_grid, function(t) {
  sum(dx_data$biomarker < t & dx_data$ami == 0) / sum(dx_data$ami == 0)
})

threshold_df <- data.frame(
  threshold = rep(threshold_grid, 2),
  value = c(se_grid, sp_grid),
  metric = rep(c("Sensitivity", "Specificity"), each = length(threshold_grid))
)

p3 <- ggplot(threshold_df, aes(x = threshold, y = value, color = metric)) +
  geom_line(linewidth = 1) +
  geom_vline(xintercept = youden_cut, linetype = "dashed", color = "grey40") +
  scale_color_manual(values = c("steelblue", "darkorange")) +
  scale_x_continuous(limits = c(0, 100)) +
  labs(x = "Biomarker Threshold (ng/L)", y = "Value",
       color = "", title = "Sensitivity and Specificity vs Threshold") +
  theme_minimal(base_size = 11)

# Plot 4: PPV/NPV vs prevalence
prev_grid <- seq(0.01, 0.50, by = 0.01)
ppv_grid <- (se * prev_grid) / (se * prev_grid + (1 - sp) * (1 - prev_grid))
npv_grid <- (sp * (1 - prev_grid)) / ((1 - se) * prev_grid + sp * (1 - prev_grid))

prev_df <- data.frame(
  prevalence = rep(prev_grid, 2),
  value = c(ppv_grid, npv_grid),
  metric = rep(c("PPV", "NPV"), each = length(prev_grid))
)

p4 <- ggplot(prev_df, aes(x = prevalence, y = value, color = metric)) +
  geom_line(linewidth = 1) +
  scale_color_manual(values = c("darkorange", "steelblue")) +
  geom_vline(xintercept = prevalence, linetype = "dashed", color = "grey40") +
  labs(x = "Disease Prevalence", y = "Predictive Value",
       color = "", title = "PPV and NPV vs Prevalence") +
  theme_minimal(base_size = 11)

grid.arrange(p1, p2, p3, p4, ncol = 2)
```

## Tips and Best Practices

1. **Always report confidence intervals**: For AUC, sensitivity, specificity, and predictive
   values. Use DeLong's method or bootstrap for AUC CIs.

2. **Prevalence matters for PPV/NPV**: Always report the study prevalence and note that PPV
   and NPV will differ in populations with different prevalence. Provide a PPV-vs-prevalence
   plot for transparency.

3. **Use DCA for clinical utility**: AUC tells you about discrimination, but DCA tells you
   whether the test actually improves clinical decisions. A high-AUC biomarker may still have
   no net benefit if the clinical context does not favor its use.

4. **Validate cutpoints externally**: An optimal cutpoint derived from one dataset will
   typically perform worse in a new population (optimism bias). Always validate in an
   independent cohort or use cross-validation.

5. **Follow STARD guidelines**: When reporting diagnostic accuracy studies, use the STARD
   checklist to ensure completeness and transparency.

6. **Consider the clinical role**: A rule-out test needs very high sensitivity (NPV). A
   rule-in test needs very high specificity (PPV). Choose the cutpoint based on the intended
   clinical role, not just the Youden index.

7. **Handle spectrum bias**: The study population should match the intended clinical use. A
   biomarker evaluated only in patients with obvious disease vs healthy controls will appear
   to perform much better than it does in the intended (harder) clinical scenario.

8. **Meta-analysis needs sufficient studies**: The bivariate model requires at least 4-5
   studies to estimate between-study heterogeneity reliably. With fewer studies, simpler
   pooling methods (e.g., Moses-Littenberg SROC) may be used.
