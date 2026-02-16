# Study Design Frameworks — R Implementation

## Required Packages

```r
install.packages(c("survival", "MatchIt", "cobalt", "ipw", "ggplot2",
                   "dplyr", "lubridate", "TrialEmulation"))

library(survival)
library(MatchIt)
library(cobalt)
library(ipw)
library(ggplot2)
library(dplyr)
library(lubridate)
```

## Example Dataset

We simulate an observational cohort study comparing two oral anticoagulants (Drug A vs.
Drug B) for stroke prevention in patients with atrial fibrillation (AF). This emulates a
target trial of new users.

```r
set.seed(2024)
n <- 2000

# Baseline date
index_date <- as.Date("2018-01-01") + sample(0:730, n, replace = TRUE)

# Confounders
age           <- round(rnorm(n, 72, 10))
female        <- rbinom(n, 1, 0.42)
chads_vasc    <- pmin(pmax(rpois(n, 3), 0), 9)  # CHA2DS2-VASc score
prior_stroke  <- rbinom(n, 1, 0.15)
ckd           <- rbinom(n, 1, 0.20)
hf            <- rbinom(n, 1, 0.25)

# Treatment assignment (confounded by CHA2DS2-VASc and CKD)
lp_trt <- -0.5 + 0.15 * chads_vasc + 0.3 * ckd - 0.01 * age
prob_a <- plogis(lp_trt)
drug_a <- rbinom(n, 1, prob_a)

# Outcome: time to stroke (event) or censoring
lp_out <- -5 + 0.05 * age + 0.3 * prior_stroke + 0.2 * chads_vasc +
          0.15 * ckd + 0.2 * hf - 0.25 * drug_a
rate     <- exp(lp_out)
time_event <- rexp(n, rate)
time_censor <- runif(n, 0.5, 5)  # administrative censoring up to 5 years
time     <- pmin(time_event, time_censor)
event    <- as.integer(time_event <= time_censor)

cohort <- data.frame(
  id = 1:n, index_date, age, female, chads_vasc, prior_stroke, ckd, hf,
  drug_a, time, event
)

cat("Drug A:", sum(drug_a), "patients | Drug B:", sum(1 - drug_a), "patients\n")
cat("Events:", sum(event), "(", round(100 * mean(event), 1), "%)\n")
```

## Complete Worked Example

### Step 1 — Target Trial Specification

```r
# Document the target trial components (this would go in the protocol)
target_trial <- list(
  eligibility   = "Adults >= 18 with new AF diagnosis, no prior anticoagulant use",
  treatment     = "Initiation of Drug A vs. Drug B within 30 days of AF diagnosis",
  assignment    = "Random (emulated via propensity score adjustment)",
  start_followup = "Date of first anticoagulant dispensing (time zero)",
  outcome       = "First ischaemic stroke, assessed by ICD-10 codes I63.x",
  causal_contrast = "ITT (as-initiated) and per-protocol (IPCW-weighted)",
  analysis      = "Cox proportional hazards with PS matching or weighting"
)

cat("Target Trial Specification:\n")
for (comp in names(target_trial)) {
  cat(sprintf("  %-18s %s\n", paste0(comp, ":"), target_trial[[comp]]))
}
```

### Step 2 — New-User Cohort Construction

```r
# In real data, you would filter for:
# 1. First-ever anticoagulant prescription (new user)
# 2. Prior diagnosis of AF
# 3. No prior stroke outcome (clean window)
# 4. Minimum lookback period for confounder assessment

# Our simulated data is already new-user by construction.
# Demonstrate the filtering logic:

new_users <- cohort %>%
  filter(age >= 18) %>%           # eligibility
  # In real data: filter first-ever drug dispensing
  # In real data: require >= 365 days of prior enrolment (washout)
  mutate(
    time_zero = index_date,       # aligned with drug initiation
    followup_end = time_zero + days(round(time * 365.25))
  )

cat("New-user cohort: n =", nrow(new_users), "\n")
```

### Step 3 — Propensity Score Estimation

```r
ps_model <- glm(drug_a ~ age + female + chads_vasc + prior_stroke + ckd + hf,
                family = binomial, data = new_users)
new_users$ps <- predict(ps_model, type = "response")

# Check overlap
ggplot(new_users, aes(x = ps, fill = factor(drug_a))) +
  geom_density(alpha = 0.5) +
  scale_fill_manual(values = c("steelblue", "firebrick"),
                    labels = c("Drug B", "Drug A")) +
  labs(title = "Propensity Score Distribution by Treatment",
       x = "Propensity Score", y = "Density", fill = "Group") +
  theme_minimal()

# Trim non-overlap region
trimmed <- new_users %>% filter(ps > 0.05 & ps < 0.95)
cat("After trimming:", nrow(trimmed), "patients\n")
```

### Step 4 — Propensity Score Matching

```r
match_out <- matchit(
  drug_a ~ age + female + chads_vasc + prior_stroke + ckd + hf,
  data = trimmed, method = "nearest", distance = "glm",
  caliper = 0.1, ratio = 1
)
summary(match_out)

# Covariate balance
love.plot(match_out, thresholds = c(m = 0.1),
          var.order = "unadjusted",
          title = "Covariate Balance: Before and After Matching")

matched <- match.data(match_out)
cat("Matched cohort: n =", nrow(matched), "\n")
```

### Step 5 — Outcome Analysis on Matched Cohort

```r
# Cox model on matched data (cluster-robust SEs for matched pairs)
cox_matched <- coxph(Surv(time, event) ~ drug_a + strata(subclass),
                     data = matched, robust = TRUE)
summary(cox_matched)

# Hazard ratio for Drug A vs Drug B
hr <- exp(coef(cox_matched)["drug_a"])
ci <- exp(confint(cox_matched)["drug_a", ])
cat(sprintf("\nHR (Drug A vs B): %.3f (95%% CI: %.3f - %.3f)\n", hr, ci[1], ci[2]))

# Kaplan-Meier by treatment in matched cohort
km_fit <- survfit(Surv(time, event) ~ drug_a, data = matched)
plot(km_fit, col = c("steelblue", "firebrick"), lwd = 2,
     xlab = "Years", ylab = "Stroke-Free Survival",
     main = "Kaplan-Meier Curves (Matched Cohort)")
legend("bottomleft", c("Drug B", "Drug A"), col = c("steelblue", "firebrick"),
       lwd = 2)
```

### Step 6 — IPTW Analysis (Alternative to Matching)

```r
# Inverse probability of treatment weights
new_users$iptw <- ifelse(new_users$drug_a == 1,
                         1 / new_users$ps,
                         1 / (1 - new_users$ps))

# Stabilised weights
p_trt <- mean(new_users$drug_a)
new_users$siptw <- ifelse(new_users$drug_a == 1,
                          p_trt / new_users$ps,
                          (1 - p_trt) / (1 - new_users$ps))

cat("Stabilised weight summary:\n")
summary(new_users$siptw)

# Weighted Cox model
cox_iptw <- coxph(Surv(time, event) ~ drug_a, data = new_users,
                  weights = siptw, robust = TRUE)
summary(cox_iptw)

hr_iptw <- exp(coef(cox_iptw)["drug_a"])
ci_iptw <- exp(confint(cox_iptw)["drug_a", ])
cat(sprintf("IPTW HR: %.3f (95%% CI: %.3f - %.3f)\n",
            hr_iptw, ci_iptw[1], ci_iptw[2]))
```

## Advanced Example

### Handling Immortal Time Bias

```r
# Demonstrate immortal time bias with a WRONG analysis
# Suppose we misalign time zero: follow-up starts at AF diagnosis,
# but treatment starts later.

new_users$time_to_treatment <- runif(n, 0, 0.5)  # 0-6 months delay
new_users$time_wrong <- new_users$time + new_users$time_to_treatment

# WRONG: count pre-treatment time as exposed
cox_wrong <- coxph(Surv(time_wrong, event) ~ drug_a, data = new_users)
hr_wrong <- exp(coef(cox_wrong)["drug_a"])

# CORRECT: start follow-up at treatment initiation (our original analysis)
cox_correct <- coxph(Surv(time, event) ~ drug_a + age + chads_vasc + ckd,
                     data = new_users)
hr_correct <- exp(coef(cox_correct)["drug_a"])

cat("HR with immortal time bias (WRONG):", round(hr_wrong, 3), "\n")
cat("HR with aligned time zero (CORRECT):", round(hr_correct, 3), "\n")
# The biased HR will be more protective (further from 1 toward 0)
```

### Per-Protocol Analysis with IPCW

```r
# Simulate treatment discontinuation
new_users$disc_time <- rexp(n, 0.3)  # time to discontinuation
new_users$adhered   <- as.integer(new_users$disc_time >= new_users$time)

# Censor at discontinuation for per-protocol
new_users$time_pp   <- pmin(new_users$time, new_users$disc_time)
new_users$event_pp  <- ifelse(new_users$disc_time < new_users$time, 0, new_users$event)

# IPCW: model probability of remaining on treatment
# (simplified: in practice, use time-varying covariates)
ipcw_model <- glm(adhered ~ drug_a + age + chads_vasc + ckd,
                  family = binomial, data = new_users)
p_adhere <- predict(ipcw_model, type = "response")
new_users$ipcw <- 1 / p_adhere

cox_pp <- coxph(Surv(time_pp, event_pp) ~ drug_a + age + chads_vasc + ckd,
                data = new_users, weights = ipcw)
summary(cox_pp)
cat("Per-protocol HR (IPCW):", round(exp(coef(cox_pp)["drug_a"]), 3), "\n")
```

## Visualization

```r
# 1. Study design schematic
par(mar = c(2, 8, 3, 2))
plot(0, 0, type = "n", xlim = c(0, 10), ylim = c(0, 7),
     xlab = "", ylab = "", axes = FALSE, main = "Target Trial Emulation Timeline")
# Lookback
rect(0, 5.5, 3, 6.5, col = "lightyellow", border = "orange")
text(1.5, 6, "Lookback\n(confounders)", cex = 0.7)
# Time zero
abline(v = 3, col = "red", lwd = 2, lty = 2)
text(3, 7, "Time Zero\n(Drug initiation)", col = "red", cex = 0.8)
# Follow-up
rect(3, 3.5, 10, 4.5, col = "lightblue", border = "steelblue")
text(6.5, 4, "Follow-up (outcome ascertainment)", cex = 0.7)
# Treatment
rect(3, 1.5, 7, 2.5, col = "lightgreen", border = "darkgreen")
text(5, 2, "On treatment", cex = 0.7)
arrows(7, 2, 10, 2, col = "grey", lty = 3)
text(8.5, 2.3, "Discontinued", cex = 0.6, col = "grey")

# 2. Balance plot
love.plot(match_out, thresholds = c(m = 0.1), abs = TRUE,
          title = "Absolute Standardised Mean Differences")

# 3. KM plot with ggplot
library(survminer)
ggsurvplot(km_fit, data = matched,
           palette = c("steelblue", "firebrick"),
           legend.labs = c("Drug B", "Drug A"),
           risk.table = TRUE,
           title = "Stroke-Free Survival: Matched Cohort",
           xlab = "Years", ylab = "Survival Probability")
```

## Tips and Best Practices

1. **Always specify the target trial first.** Write out the seven components before
   touching the data. This forces transparent design decisions.

2. **Use the new-user (incident-user) design** as the default for comparative
   effectiveness studies. Prevalent-user analyses require strong justification.

3. **Align time zero with treatment initiation.** Misalignment is the most common source
   of immortal time bias. Verify by checking that no outcome events precede time zero.

4. **Check propensity score overlap.** If groups do not overlap (positivity violation),
   trim the non-overlap region or use matching instead of IPTW.

5. **Report standardised mean differences (SMDs) < 0.1** after matching or weighting.
   Use `cobalt::love.plot()` for clear visualisation.

6. **For the per-protocol estimand,** use IPCW with time-varying confounders. The
   naive as-treated analysis is biased by confounding that caused discontinuation.

7. **Conduct quantitative bias analysis** for unmeasured confounding (e.g., E-value).
   No observational study is immune to residual confounding.

8. **Pre-register the study protocol** (e.g., on ClinicalTrials.gov or ENCePP). Post-hoc
   design decisions undermine the credibility of RWE studies.
