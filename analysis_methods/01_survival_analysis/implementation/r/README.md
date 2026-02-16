# Survival Analysis — R Implementation

## Required Packages

```r
install.packages(c("survival", "survminer", "flexsurv", "ggplot2", "dplyr"))

library(survival)
library(survminer)
library(flexsurv)
library(ggplot2)
library(dplyr)
```

## Example Dataset

We use the `lung` dataset from the `survival` package. It contains data from the North Central Cancer Treatment Group (NCCTG) on 228 patients with advanced lung cancer. Key variables:

- `time`: Survival time in days
- `status`: Censoring status (1 = censored, 2 = dead)
- `age`: Age in years
- `sex`: 1 = Male, 2 = Female
- `ph.ecog`: ECOG performance score (0 = good, 5 = dead)
- `ph.karno`: Karnofsky performance score rated by physician
- `wt.loss`: Weight loss in the last six months

## Complete Worked Example

### Step 1: Load and Prepare Data

```r
library(survival)
library(survminer)

data(lung)
str(lung)

# Recode status: survival package expects 0 = censored, 1 = event
lung$status_event <- lung$status - 1  # Now 0 = censored, 1 = dead

# Recode sex as a factor
lung$sex <- factor(lung$sex, levels = c(1, 2), labels = c("Male", "Female"))

# Check for missing data
colSums(is.na(lung))

# Remove rows with missing ph.ecog for modeling
lung_clean <- lung %>% filter(!is.na(ph.ecog))

cat("Sample size:", nrow(lung_clean), "\n")
cat("Number of events:", sum(lung_clean$status_event), "\n")
cat("Censoring rate:", round(1 - mean(lung_clean$status_event), 3), "\n")
```

**Interpretation**: The lung dataset contains 228 patients. After cleaning, roughly 72% of patients experienced the event (death), making this a well-powered dataset for survival analysis.

### Step 2: Kaplan-Meier Estimation

```r
# Overall KM estimate
km_fit <- survfit(Surv(time, status_event) ~ 1, data = lung_clean)
summary(km_fit, times = c(90, 180, 365, 730))

# Print median survival with 95% CI
print(km_fit)
```

**Interpretation**: The `survfit` summary gives survival probabilities at 90, 180, 365, and 730 days. The median survival time is reported with its 95% confidence interval. For the lung dataset, median survival is approximately 310 days.

### Step 3: Kaplan-Meier Curves by Group

```r
# KM by sex
km_sex <- survfit(Surv(time, status_event) ~ sex, data = lung_clean)
print(km_sex)

# Plot with survminer
ggsurvplot(km_sex,
           data = lung_clean,
           pval = TRUE,              # Log-rank p-value
           conf.int = TRUE,          # 95% CI bands
           risk.table = TRUE,        # Number at risk table
           risk.table.col = "strata",
           xlab = "Time (days)",
           ylab = "Survival Probability",
           title = "Kaplan-Meier Survival by Sex",
           palette = c("#E7B800", "#2E9FDF"),
           ggtheme = theme_minimal(),
           legend.labs = c("Male", "Female"),
           break.time.by = 100)
```

**Interpretation**: The KM plot shows survival curves for males and females. If the p-value (from the log-rank test) is below 0.05, there is evidence of a statistically significant difference in survival between sexes. Females in the lung dataset tend to have longer survival.

### Step 4: Log-Rank Test

```r
# Log-rank test
logrank_test <- survdiff(Surv(time, status_event) ~ sex, data = lung_clean)
print(logrank_test)

# Extract p-value
pval <- 1 - pchisq(logrank_test$chisq, df = 1)
cat("Log-rank test p-value:", format.pval(pval), "\n")
```

**Interpretation**: The log-rank test compares overall survival distributions. A significant result (p < 0.05) indicates that the survival experience differs between groups. The test is most powerful when the proportional hazards assumption holds.

### Step 5: Cox Proportional Hazards Model

```r
# Univariable Cox model
cox_uni <- coxph(Surv(time, status_event) ~ sex, data = lung_clean)
summary(cox_uni)

# Multivariable Cox model
cox_multi <- coxph(Surv(time, status_event) ~ age + sex + ph.ecog + ph.karno + wt.loss,
                   data = lung_clean)
summary(cox_multi)

# Hazard ratios with 95% CI
exp(cbind(HR = coef(cox_multi), confint(cox_multi)))
```

**Interpretation**: The Cox model output includes coefficients (log hazard ratios), exponentiated coefficients (hazard ratios), standard errors, z-statistics, and p-values. A hazard ratio of 0.60 for female sex means females have 40% lower hazard of death compared to males. The concordance statistic (C-index) measures discriminative ability.

### Step 6: Check Proportional Hazards Assumption

```r
# Schoenfeld residuals test
ph_test <- cox.zph(cox_multi)
print(ph_test)

# Plot Schoenfeld residuals for each covariate
ggcoxzph(ph_test)

# A significant p-value for a covariate suggests PH violation
```

**Interpretation**: The `cox.zph()` function tests the PH assumption for each covariate and globally. A p-value < 0.05 for a covariate suggests its effect changes over time. The plots show scaled Schoenfeld residuals over time; a non-zero slope indicates PH violation. If violated, consider stratification, time-varying coefficients, or a different model.

### Step 7: Forest Plot

```r
# Forest plot of hazard ratios
ggforest(cox_multi, data = lung_clean)
```

**Interpretation**: The forest plot visualizes hazard ratios and 95% confidence intervals for each covariate. Points to the left of 1.0 indicate reduced hazard (protective); points to the right indicate increased hazard (risk factor).

## Advanced Example

### Parametric Model Comparison with flexsurv

```r
library(flexsurv)

# Fit multiple parametric models
fit_exp <- flexsurvreg(Surv(time, status_event) ~ sex + age + ph.ecog,
                       data = lung_clean, dist = "exp")
fit_weibull <- flexsurvreg(Surv(time, status_event) ~ sex + age + ph.ecog,
                           data = lung_clean, dist = "weibull")
fit_lnorm <- flexsurvreg(Surv(time, status_event) ~ sex + age + ph.ecog,
                          data = lung_clean, dist = "lnorm")
fit_llogis <- flexsurvreg(Surv(time, status_event) ~ sex + age + ph.ecog,
                           data = lung_clean, dist = "llogis")
fit_gengamma <- flexsurvreg(Surv(time, status_event) ~ sex + age + ph.ecog,
                             data = lung_clean, dist = "gengamma")

# Compare models by AIC
model_comparison <- data.frame(
  Model = c("Exponential", "Weibull", "Log-normal", "Log-logistic", "Gen. Gamma"),
  AIC = c(AIC(fit_exp), AIC(fit_weibull), AIC(fit_lnorm),
          AIC(fit_llogis), AIC(fit_gengamma)),
  BIC = c(BIC(fit_exp), BIC(fit_weibull), BIC(fit_lnorm),
          BIC(fit_llogis), BIC(fit_gengamma))
)
model_comparison <- model_comparison[order(model_comparison$AIC), ]
print(model_comparison)

# Best model summary
summary(fit_weibull)
```

**Interpretation**: AIC and BIC are used to compare parametric model fit. The model with the lowest AIC/BIC provides the best balance of fit and parsimony. The Weibull model often performs well and generalizes the exponential. The generalized gamma nests several distributions and can serve as a formal test.

### Restricted Mean Survival Time (RMST)

```r
# RMST comparison between groups
library(survival)

rmst_male <- survfit(Surv(time, status_event) ~ 1,
                     data = lung_clean[lung_clean$sex == "Male", ])
rmst_female <- survfit(Surv(time, status_event) ~ 1,
                       data = lung_clean[lung_clean$sex == "Female", ])

# Print RMST at tau = 365 days
print(rmst_male, rmean = 365)
print(rmst_female, rmean = 365)

# Alternatively, use the survRM2 package for formal comparison
# install.packages("survRM2")
library(survRM2)
rmst_result <- rmst2(lung_clean$time, lung_clean$status_event,
                     arm = as.numeric(lung_clean$sex) - 1, tau = 365)
print(rmst_result)
```

**Interpretation**: RMST gives the average survival time up to a specified time horizon (tau). The difference in RMST between groups represents the average additional days of survival for one group vs the other, which is clinically intuitive.

## Visualization

### Cumulative Hazard Plot

```r
# Nelson-Aalen cumulative hazard
ggsurvplot(km_sex,
           data = lung_clean,
           fun = "cumhaz",
           xlab = "Time (days)",
           ylab = "Cumulative Hazard",
           title = "Nelson-Aalen Cumulative Hazard by Sex",
           palette = c("#E7B800", "#2E9FDF"),
           legend.labs = c("Male", "Female"))
```

### Log-Log Survival Plot (PH Check)

```r
# Complementary log-log plot for PH assessment
ggsurvplot(km_sex,
           data = lung_clean,
           fun = "cloglog",
           xlab = "log(Time)",
           ylab = "log(-log(S(t)))",
           title = "Complementary Log-Log Plot (PH Assessment)",
           palette = c("#E7B800", "#2E9FDF"),
           legend.labs = c("Male", "Female"))
```

**Interpretation**: If the PH assumption holds, the cloglog curves for different groups should be approximately parallel. Non-parallel or crossing curves suggest the hazard ratio changes over time.

### Cox Model Diagnostic Plots

```r
# Martingale residuals vs fitted values (functional form check)
ggcoxdiagnostics(cox_multi, type = "martingale", ox.scale = "linear.predictions")

# Deviance residuals (outlier detection)
ggcoxdiagnostics(cox_multi, type = "deviance", ox.scale = "linear.predictions")
```

**Interpretation**: Martingale residuals should show no systematic pattern when plotted against linear predictions or individual covariates. Deviance residuals should be roughly symmetrically distributed around zero; values beyond +/- 2 may indicate influential observations.

## Tips and Best Practices

1. **Always report the number at risk**: KM plots without a risk table can be misleading when the number at risk becomes very small in the tail.

2. **Do not extrapolate KM curves**: The KM estimate is only valid within the range of observed data. For extrapolation, use parametric models with caution.

3. **Check the PH assumption before relying on Cox model results**: Use `cox.zph()` and graphical diagnostics. If PH is violated, consider stratification (`strata()` in the formula), time-varying coefficients, or alternative models.

4. **Handle ties appropriately**: The default Efron method in R is generally preferred over the Breslow method when there are many tied event times. Specify `ties = "efron"` (the default in `coxph()`).

5. **Be cautious with stepwise selection**: Automated variable selection can lead to unstable models. Prefer pre-specified models based on clinical knowledge.

6. **Report both KM estimates and model-based results**: KM provides non-parametric descriptive summaries; the Cox model provides adjusted inference. Together they tell a complete story.

7. **Account for clustering**: If data come from multiple centers or have repeated events per patient, use frailty terms or robust variance estimation:
   ```r
   coxph(Surv(time, status_event) ~ sex + age + cluster(inst), data = lung_clean)
   ```

8. **Consider competing risks**: If subjects can experience events other than the one of interest (e.g., death from other causes), standard KM overestimates the cumulative incidence. Use competing risks methods instead.

9. **Use the counting process notation for time-varying covariates**:
   ```r
   coxph(Surv(tstart, tstop, event) ~ x_timevarying + x_baseline, data = long_data)
   ```

10. **Validate with concordance index**: The C-index from `concordance()` or `summary(cox_model)$concordance` quantifies discrimination. Values above 0.7 indicate reasonable discrimination.
