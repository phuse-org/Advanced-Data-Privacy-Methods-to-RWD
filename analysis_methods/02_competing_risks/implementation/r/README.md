# Competing Risks — R Implementation

## Required Packages

```r
install.packages(c("cmprsk", "tidycmprsk", "survival", "ggplot2", "dplyr", "tidyr"))

library(cmprsk)
library(tidycmprsk)
library(survival)
library(ggplot2)
library(dplyr)
library(tidyr)
```

## Example Dataset

We simulate a bone marrow transplant dataset representing a clinical scenario where patients can experience relapse (event of interest), treatment-related mortality (competing event), or remain in remission (censored). This mirrors real-world transplant studies where multiple outcomes compete.

```r
set.seed(42)
n <- 500

bmt_data <- data.frame(
  id = 1:n,
  time = round(rexp(n, rate = 0.003), 1),
  age = round(rnorm(n, mean = 45, sd = 12)),
  donor_type = sample(c("Matched", "Mismatched"), n, replace = TRUE, prob = c(0.6, 0.4)),
  disease_stage = sample(c("Early", "Intermediate", "Advanced"), n, replace = TRUE, prob = c(0.3, 0.4, 0.3)),
  gvhd = sample(0:1, n, replace = TRUE, prob = c(0.6, 0.4))
)

# Assign event types based on covariates
set.seed(123)
event_prob <- with(bmt_data, {
  lp_relapse <- -2.5 + 0.01 * age + 0.3 * (disease_stage == "Advanced") +
    0.2 * (disease_stage == "Intermediate") - 0.2 * gvhd
  lp_trm <- -3.0 + 0.02 * age + 0.4 * (donor_type == "Mismatched") + 0.3 * gvhd
  p_relapse <- plogis(lp_relapse)
  p_trm <- plogis(lp_trm) * 0.6
  p_censor <- 1 - p_relapse - p_trm
  cbind(p_censor, p_relapse, p_trm)
})

bmt_data$event_type <- apply(event_prob, 1, function(p) {
  p <- pmax(p, 0)
  p <- p / sum(p)
  sample(0:2, 1, prob = p)
})

# Cap follow-up at 1000 days
bmt_data$time <- pmin(bmt_data$time, 1000)
bmt_data$time[bmt_data$time == 0] <- 0.5

# Event labels
bmt_data$event_label <- factor(bmt_data$event_type,
                                levels = 0:2,
                                labels = c("Censored", "Relapse", "TRM"))

cat("Event distribution:\n")
table(bmt_data$event_label)
```

## Complete Worked Example

### Step 1: Explore the Data

```r
cat("Sample size:", nrow(bmt_data), "\n")
cat("Event distribution:\n")
print(table(bmt_data$event_label))
cat("\nMedian follow-up time:", median(bmt_data$time), "days\n")

# Summary by event type
bmt_data %>%
  group_by(event_label) %>%
  summarise(
    n = n(),
    mean_age = round(mean(age), 1),
    pct_mismatched = round(mean(donor_type == "Mismatched") * 100, 1),
    pct_advanced = round(mean(disease_stage == "Advanced") * 100, 1),
    median_time = round(median(time), 1)
  ) %>%
  print()
```

**Interpretation**: The data summary shows the distribution of events across the three categories. Understanding the event breakdown is essential before proceeding with competing risks analysis.

### Step 2: Non-Parametric Cumulative Incidence Function (CIF)

```r
library(cmprsk)

# Estimate CIF using cuminc()
cif_overall <- cuminc(ftime = bmt_data$time,
                       fstatus = bmt_data$event_type,
                       cencode = 0)

# Print CIF estimates at key time points
cat("CIF estimates at key time points:\n")
timepoints <- c(100, 200, 365, 500, 730)
for (tp in timepoints) {
  cif_vals <- timepoints(cif_overall, times = tp)
  cat(sprintf("  t = %d days: Relapse = %.3f, TRM = %.3f\n",
              tp, cif_vals[1, 1], cif_vals[1, 2]))
}

# Plot CIF
plot(cif_overall,
     xlab = "Time (days)",
     ylab = "Cumulative Incidence",
     main = "Cumulative Incidence Functions",
     col = c("red", "blue"),
     lty = 1,
     lwd = 2)
legend("topleft", legend = c("Relapse", "TRM"),
       col = c("red", "blue"), lty = 1, lwd = 2)
```

**Interpretation**: The CIF plot shows the estimated probability of each event type over time. Unlike 1-KM, the CIFs properly account for the competing nature of events: their sum at any time point, plus the probability of remaining event-free, equals 1.

### Step 3: CIF by Group with Gray's Test

```r
# CIF by donor type
cif_donor <- cuminc(ftime = bmt_data$time,
                     fstatus = bmt_data$event_type,
                     group = bmt_data$donor_type,
                     cencode = 0)

# Gray's test for equality of CIFs
cat("Gray's test results:\n")
print(cif_donor$Tests)

# Plot CIF by donor type
plot(cif_donor,
     xlab = "Time (days)",
     ylab = "Cumulative Incidence",
     main = "CIF by Donor Type",
     col = c("red", "darkred", "blue", "darkblue"),
     lty = c(1, 2, 1, 2),
     lwd = 2)
legend("topleft",
       legend = c("Relapse - Matched", "Relapse - Mismatched",
                   "TRM - Matched", "TRM - Mismatched"),
       col = c("red", "darkred", "blue", "darkblue"),
       lty = c(1, 2, 1, 2), lwd = 2, cex = 0.8)
```

**Interpretation**: Gray's test is the competing risks analog of the log-rank test. It compares CIFs between groups for each event type. A significant p-value indicates that the cumulative incidence of the specific event differs between groups.

### Step 4: Using tidycmprsk for Publication-Quality Plots

```r
library(tidycmprsk)

# Create CIF with tidycmprsk
cif_tidy <- cuminc(Surv(time, event_label) ~ donor_type, data = bmt_data)

# Summary
summary(cif_tidy, times = c(100, 365, 730))

# ggplot-based CIF plot
ggcuminc(cif_tidy, outcome = "Relapse") +
  labs(x = "Time (days)",
       y = "Cumulative Incidence of Relapse",
       title = "Cumulative Incidence of Relapse by Donor Type") +
  add_confidence_interval() +
  add_risktable() +
  theme_minimal() +
  scale_color_manual(values = c("#E41A1C", "#377EB8"))
```

**Interpretation**: The `tidycmprsk` package provides a tidy interface that integrates with ggplot2 for publication-quality CIF plots with risk tables and confidence bands.

### Step 5: Fine-Gray Subdistribution Hazard Model

```r
library(cmprsk)

# Prepare covariate matrix
covariates <- model.matrix(~ age + donor_type + disease_stage + gvhd,
                            data = bmt_data)[, -1]

# Fine-Gray model for relapse (event type 1)
fg_relapse <- crr(ftime = bmt_data$time,
                   fstatus = bmt_data$event_type,
                   cov1 = covariates,
                   failcode = 1,
                   cencode = 0)

summary(fg_relapse)

# Fine-Gray model for TRM (event type 2)
fg_trm <- crr(ftime = bmt_data$time,
               fstatus = bmt_data$event_type,
               cov1 = covariates,
               failcode = 2,
               cencode = 0)

summary(fg_trm)
```

**Interpretation**: The Fine-Gray model output provides subdistribution hazard ratios (sHR). For the relapse model, sHR > 1 indicates higher cumulative incidence of relapse. These coefficients directly model the CIF: a positive coefficient means the covariate increases the probability of the specific event by time t.

### Step 6: Cause-Specific Cox Models

```r
library(survival)

# Cause-specific model for relapse: censor TRM events
bmt_data$relapse_event <- ifelse(bmt_data$event_type == 1, 1, 0)
cox_relapse <- coxph(Surv(time, relapse_event) ~ age + donor_type + disease_stage + gvhd,
                      data = bmt_data)
summary(cox_relapse)

# Cause-specific model for TRM: censor relapse events
bmt_data$trm_event <- ifelse(bmt_data$event_type == 2, 1, 0)
cox_trm <- coxph(Surv(time, trm_event) ~ age + donor_type + disease_stage + gvhd,
                  data = bmt_data)
summary(cox_trm)

# Compare results side by side
comparison <- data.frame(
  Covariate = names(coef(cox_relapse)),
  CS_HR_Relapse = round(exp(coef(cox_relapse)), 3),
  CS_p_Relapse = round(summary(cox_relapse)$coefficients[, 5], 4),
  CS_HR_TRM = round(exp(coef(cox_trm)), 3),
  CS_p_TRM = round(summary(cox_trm)$coefficients[, 5], 4)
)
print(comparison)
```

**Interpretation**: Cause-specific Cox models estimate the rate of each event type among subjects still at risk of any event. Comparing cause-specific HRs across event types reveals whether a covariate affects different outcomes differently. For example, GVHD might reduce relapse risk (graft-vs-leukemia effect) but increase TRM.

## Advanced Example

### Comparing Cause-Specific and Fine-Gray Results

```r
# Consolidate results
cs_relapse_hr <- exp(coef(cox_relapse))
fg_relapse_hr <- exp(fg_relapse$coef)

comparison_full <- data.frame(
  Covariate = names(cs_relapse_hr),
  CS_HR = round(cs_relapse_hr, 3),
  FG_sHR = round(fg_relapse_hr, 3)
)
print(comparison_full)
```

**Interpretation**: Cause-specific and subdistribution hazard ratios will generally differ. The cause-specific HR reflects the direct effect on event rate. The subdistribution HR reflects the net effect on cumulative incidence, incorporating the indirect effect through competing events. When the competing event rate is low, the two approaches give similar results.

### Predicted CIF from Fine-Gray Model

```r
# Predict CIF for a new patient profile
new_patient <- matrix(c(50, 1, 0, 1, 0), nrow = 1)  # age=50, mismatched, intermediate, no GVHD
colnames(new_patient) <- colnames(covariates)

pred_cif <- predict(fg_relapse, cov1 = new_patient)

plot(pred_cif,
     xlab = "Time (days)",
     ylab = "Predicted CIF of Relapse",
     main = "Predicted Cumulative Incidence of Relapse\n(50yo, Mismatched Donor, Intermediate Stage, No GVHD)",
     lwd = 2, col = "red")
```

**Interpretation**: The predicted CIF gives the estimated probability of relapse over time for a patient with specific characteristics. This is directly useful for clinical risk communication.

### Stacked CIF Plot

```r
# Create stacked cumulative incidence plot
cif_data <- data.frame(
  time = c(cif_overall$`1 1`$time, cif_overall$`1 2`$time),
  incidence = c(cif_overall$`1 1`$est, cif_overall$`1 2`$est),
  event = rep(c("Relapse", "TRM"), c(length(cif_overall$`1 1`$time),
                                       length(cif_overall$`1 2`$time)))
)

ggplot() +
  geom_area(data = cif_data %>% filter(event == "TRM"),
            aes(x = time, y = incidence), fill = "steelblue", alpha = 0.7) +
  geom_area(data = cif_data %>% filter(event == "Relapse"),
            aes(x = time, y = incidence), fill = "firebrick", alpha = 0.7) +
  labs(x = "Time (days)", y = "Cumulative Incidence",
       title = "Stacked Cumulative Incidence Functions") +
  theme_minimal() +
  annotate("text", x = 600, y = 0.15, label = "Relapse", color = "firebrick", size = 5) +
  annotate("text", x = 600, y = 0.35, label = "TRM", color = "steelblue", size = 5)
```

## Visualization

### Publication-Quality CIF Plot with Risk Table

```r
library(tidycmprsk)

cif_pub <- cuminc(Surv(time, event_label) ~ donor_type, data = bmt_data)

p1 <- ggcuminc(cif_pub, outcome = "Relapse") +
  labs(x = "Time (days)",
       y = "Cumulative Incidence",
       title = "Relapse") +
  add_confidence_interval() +
  add_risktable() +
  theme_minimal(base_size = 12)

p2 <- ggcuminc(cif_pub, outcome = "TRM") +
  labs(x = "Time (days)",
       y = "Cumulative Incidence",
       title = "Treatment-Related Mortality") +
  add_confidence_interval() +
  add_risktable() +
  theme_minimal(base_size = 12)

# Display side by side (requires patchwork or gridExtra)
# install.packages("patchwork")
library(patchwork)
p1 + p2
```

### Forest Plot Comparing Cause-Specific and Fine-Gray Models

```r
# Build a combined forest plot
covariate_names <- names(coef(cox_relapse))
cs_hr <- exp(coef(cox_relapse))
cs_lower <- exp(confint(cox_relapse))[, 1]
cs_upper <- exp(confint(cox_relapse))[, 2]

fg_hr <- exp(fg_relapse$coef)
fg_lower <- exp(fg_relapse$coef - 1.96 * sqrt(diag(fg_relapse$var)))
fg_upper <- exp(fg_relapse$coef + 1.96 * sqrt(diag(fg_relapse$var)))

forest_data <- data.frame(
  covariate = rep(covariate_names, 2),
  hr = c(cs_hr, fg_hr),
  lower = c(cs_lower, fg_lower),
  upper = c(cs_upper, fg_upper),
  model = rep(c("Cause-Specific", "Fine-Gray"), each = length(covariate_names))
)

ggplot(forest_data, aes(x = hr, y = covariate, color = model)) +
  geom_point(position = position_dodge(width = 0.5), size = 3) +
  geom_errorbarh(aes(xmin = lower, xmax = upper),
                 position = position_dodge(width = 0.5), height = 0.2) +
  geom_vline(xintercept = 1, linestyle = "dashed", color = "gray40") +
  labs(x = "Hazard Ratio (95% CI)", y = "",
       title = "Cause-Specific vs. Fine-Gray Hazard Ratios for Relapse",
       color = "Model") +
  theme_minimal(base_size = 12) +
  scale_color_manual(values = c("Cause-Specific" = "#E41A1C", "Fine-Gray" = "#377EB8"))
```

## Tips and Best Practices

1. **Never use 1 - KM as cumulative incidence**: When competing risks exist, the Kaplan-Meier complement overestimates the event-specific probability. Always use CIF estimators.

2. **Report both cause-specific and subdistribution analyses**: They answer different questions. Cause-specific models address etiology; Fine-Gray models address prognosis. Both perspectives are informative.

3. **Check the proportional hazards assumption**: Use `cox.zph()` for cause-specific models. For Fine-Gray, examine the linearity of the CIF on the complementary log-log scale.

4. **Mind the sample size**: Each event type needs sufficient events for stable estimation. A minimum of 10 events per covariate per event type is a rough guideline.

5. **Be careful with covariate interpretation in Fine-Gray models**: A subdistribution HR > 1 means higher cumulative incidence, but the mechanism could be direct (higher event-specific rate) or indirect (lower competing event rate). Always interpret alongside cause-specific results.

6. **Gray's test is preferred over the log-rank test for CIF comparisons**: The log-rank test applied to event-specific data does not properly account for competing risks.

7. **Use tidycmprsk for modern R workflows**: It provides a tidy interface compatible with ggplot2 and the tidyverse, making competing risks analysis more accessible and reproducible.

8. **Consider multi-state models for complex settings**: When there are intermediate states (e.g., GVHD between transplant and relapse/death), multi-state models using the `mstate` or `survival` packages provide a richer framework.

9. **Handle multiple testing carefully**: When testing covariates across multiple event types, consider adjusting for multiplicity or clearly stating the exploratory nature of the analysis.

10. **Validate with clinical context**: Statistical significance of a competing risk does not always imply clinical importance. Discuss results with clinical investigators to ensure meaningful interpretation.
