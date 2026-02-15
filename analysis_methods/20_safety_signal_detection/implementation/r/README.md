# Safety Signal Detection — R Implementation

## Required Packages

```r
install.packages(c("PhViD", "ggplot2", "dplyr", "tidyr", "EmpiricalCalibration"))

library(PhViD)                # disproportionality analysis (PRR, ROR, BCPNN, GPS)
library(ggplot2)
library(dplyr)
library(tidyr)
library(EmpiricalCalibration) # empirical calibration of p-values
```

## Example Dataset

We simulate a FAERS-like spontaneous reporting database with 50,000 reports covering
100 drugs and 200 adverse events. Three known drug-event signals are embedded.

```r
set.seed(2024)
n_reports <- 50000
n_drugs   <- 100
n_events  <- 200

# Generate random drug-event pairs
drug_id  <- sample(paste0("Drug_", sprintf("%03d", 1:n_drugs)), n_reports, replace = TRUE,
                   prob = c(rep(3, 10), rep(1, 90)) / (10 * 3 + 90))
event_id <- sample(paste0("AE_", sprintf("%03d", 1:n_events)), n_reports, replace = TRUE,
                   prob = c(rep(2, 20), rep(1, 180)) / (20 * 2 + 180))

# Inject 3 known signals (elevated reporting)
# Signal 1: Drug_001 + AE_001 (strong signal)
idx_s1 <- sample(n_reports, 150)
drug_id[idx_s1]  <- "Drug_001"
event_id[idx_s1] <- "AE_001"

# Signal 2: Drug_005 + AE_010 (moderate signal)
idx_s2 <- sample(setdiff(1:n_reports, idx_s1), 80)
drug_id[idx_s2]  <- "Drug_005"
event_id[idx_s2] <- "AE_010"

# Signal 3: Drug_010 + AE_020 (weak signal)
idx_s3 <- sample(setdiff(1:n_reports, c(idx_s1, idx_s2)), 40)
drug_id[idx_s3]  <- "Drug_010"
event_id[idx_s3] <- "AE_020"

reports <- data.frame(report_id = 1:n_reports, drug = drug_id, event = event_id,
                      stringsAsFactors = FALSE)
cat("Total reports:", nrow(reports), "\n")
cat("Unique drugs:", length(unique(reports$drug)), "\n")
cat("Unique events:", length(unique(reports$event)), "\n")
```

## Complete Worked Example

### Step 1 — Build the 2x2 Table for a Single Drug-Event Pair

```r
build_2x2 <- function(data, target_drug, target_event) {
  a <- sum(data$drug == target_drug & data$event == target_event)
  b <- sum(data$drug == target_drug & data$event != target_event)
  c <- sum(data$drug != target_drug & data$event == target_event)
  d <- sum(data$drug != target_drug & data$event != target_event)
  matrix(c(a, b, c, d), nrow = 2, byrow = TRUE,
         dimnames = list(c("Drug+", "Drug-"), c("Event+", "Event-")))
}

tab <- build_2x2(reports, "Drug_001", "AE_001")
print(tab)
```

### Step 2 — Compute PRR and ROR

```r
compute_disproportionality <- function(data, target_drug, target_event) {
  a <- sum(data$drug == target_drug & data$event == target_event)
  b <- sum(data$drug == target_drug & data$event != target_event)
  c <- sum(data$drug != target_drug & data$event == target_event)
  d <- sum(data$drug != target_drug & data$event != target_event)

  # PRR
  prr <- (a / (a + b)) / (c / (c + d))
  se_ln_prr <- sqrt(1/a - 1/(a+b) + 1/c - 1/(c+d))
  prr_ci <- exp(log(prr) + c(-1, 1) * 1.96 * se_ln_prr)
  chi2 <- (a * d - b * c)^2 * (a + b + c + d) /
           ((a + b) * (c + d) * (a + c) * (b + d))

  # ROR
  ror <- (a * d) / (b * c)
  se_ln_ror <- sqrt(1/a + 1/b + 1/c + 1/d)
  ror_ci <- exp(log(ror) + c(-1, 1) * 1.96 * se_ln_ror)

  data.frame(
    drug = target_drug, event = target_event,
    a = a, E = (a + b) * (a + c) / (a + b + c + d),
    PRR = prr, PRR_lower = prr_ci[1], PRR_upper = prr_ci[2], chi2 = chi2,
    ROR = ror, ROR_lower = ror_ci[1], ROR_upper = ror_ci[2]
  )
}

# Test on known signals
signal_pairs <- list(
  c("Drug_001", "AE_001"),
  c("Drug_005", "AE_010"),
  c("Drug_010", "AE_020"),
  c("Drug_050", "AE_100")   # expected non-signal
)

results <- do.call(rbind, lapply(signal_pairs, function(p) {
  compute_disproportionality(reports, p[1], p[2])
}))
print(results, digits = 3)

# Evans criteria: PRR > 2, chi2 > 4, a >= 3
results$signal_PRR <- with(results, PRR > 2 & chi2 > 4 & a >= 3)
results$signal_ROR <- results$ROR_lower > 1
cat("\nSignal flags:\n")
print(results[, c("drug", "event", "signal_PRR", "signal_ROR")])
```

### Step 3 — BCPNN / Information Component (IC)

```r
compute_ic <- function(data) {
  # Create drug-event contingency table
  tab <- table(data$drug, data$event)
  N <- sum(tab)

  results <- expand.grid(drug = rownames(tab), event = colnames(tab),
                         stringsAsFactors = FALSE)
  results$observed <- as.vector(tab)
  results$n_drug   <- rowSums(tab)[results$drug]
  results$n_event  <- colSums(tab)[results$event]
  results$expected <- results$n_drug * results$n_event / N

  # IC = log2(observed / expected), with shrinkage
  # Simplified: add 0.5 to avoid log(0)
  results$IC <- log2((results$observed + 0.5) / (results$expected + 0.5))

  # Bayesian shrinkage (simplified posterior using gamma-Poisson)
  alpha_prior <- 0.5
  results$IC_shrunk <- log2((results$observed + alpha_prior) /
                            (results$expected + alpha_prior))

  # Approximate IC025 (lower credible interval)
  results$IC_var <- 1 / (results$observed + alpha_prior)
  results$IC025  <- results$IC_shrunk - 1.96 * sqrt(results$IC_var)

  results
}

ic_results <- compute_ic(reports)

# Filter for embedded signals
known_signals <- ic_results %>%
  filter((drug == "Drug_001" & event == "AE_001") |
         (drug == "Drug_005" & event == "AE_010") |
         (drug == "Drug_010" & event == "AE_020")) %>%
  arrange(desc(IC_shrunk))

cat("IC Results for Known Signals:\n")
print(known_signals[, c("drug", "event", "observed", "expected",
                         "IC_shrunk", "IC025")], digits = 3)
cat("Signal if IC025 > 0\n")
```

### Step 4 — Full Database Scan with PhViD

```r
# Convert to PhViD-compatible format (contingency table)
contingency <- table(reports$drug, reports$event)

# Run multiple disproportionality methods
# Note: PhViD expects the contingency table as input

# GPS (Gamma Poisson Shrinker) — similar to MGPS/EBGM
gps_result <- GPS(contingency, DECISION = 3)   # DECISION=3 uses EBGM lower bound
# BCPNN
bcpnn_result <- BCPNN(contingency, DECISION = 3)

# Top signals from GPS
cat("\n--- Top GPS/EBGM Signals ---\n")
head(gps_result$ALLSIGNALS, 20)

cat("\n--- Top BCPNN Signals ---\n")
head(bcpnn_result$ALLSIGNALS, 20)
```

## Advanced Example

### Hy's Law Evaluation — eDISH Plot

```r
# Simulate clinical trial liver safety data
set.seed(456)
n_patients <- 300
treatment  <- rep(c("Active", "Placebo"), each = n_patients / 2)

# ALT values (xULN) — some elevation in active arm
alt_uln <- ifelse(treatment == "Active",
                  rlnorm(n_patients / 2, meanlog = 0, sdlog = 0.6),
                  rlnorm(n_patients / 2, meanlog = -0.1, sdlog = 0.4))

# Bilirubin values (xULN)
bili_uln <- ifelse(treatment == "Active",
                   rlnorm(n_patients / 2, meanlog = -0.3, sdlog = 0.5),
                   rlnorm(n_patients / 2, meanlog = -0.4, sdlog = 0.3))

# Inject a few Hy's Law cases in active arm
hy_cases <- sample(which(treatment == "Active"), 4)
alt_uln[hy_cases]  <- runif(4, 3.5, 12)
bili_uln[hy_cases]  <- runif(4, 2.5, 6)

liver_df <- data.frame(treatment, alt_uln, bili_uln)

# eDISH plot
ggplot(liver_df, aes(x = alt_uln, y = bili_uln, color = treatment)) +
  geom_point(alpha = 0.6, size = 2) +
  geom_vline(xintercept = 3, linetype = "dashed", color = "darkred") +
  geom_hline(yintercept = 2, linetype = "dashed", color = "darkred") +
  scale_x_log10(limits = c(0.3, 20)) +
  scale_y_log10(limits = c(0.3, 10)) +
  scale_color_manual(values = c("firebrick", "steelblue")) +
  annotate("text", x = 8, y = 4, label = "Hy's Law\nQuadrant",
           color = "darkred", fontface = "bold", size = 3.5) +
  annotate("text", x = 8, y = 0.8, label = "Temple's\nCorollary",
           color = "grey40", size = 3) +
  annotate("text", x = 0.8, y = 4, label = "Cholestatic",
           color = "grey40", size = 3) +
  annotate("text", x = 0.8, y = 0.8, label = "Normal",
           color = "grey40", size = 3) +
  labs(title = "eDISH Plot: Drug-Induced Liver Injury Assessment",
       x = "Peak ALT (xULN)", y = "Peak Bilirubin (xULN)",
       color = "Treatment") +
  theme_minimal()

# Count Hy's Law cases
hys_law <- liver_df %>%
  filter(alt_uln >= 3 & bili_uln >= 2)
cat("\nHy's Law cases:\n")
print(table(hys_law$treatment))
```

### Empirical Calibration with Negative Controls

```r
# Simulate negative control estimates (true effect = 0)
set.seed(789)
n_neg_controls <- 50
neg_ctrl_logRR <- rnorm(n_neg_controls, mean = 0.05, sd = 0.3)  # slight bias
neg_ctrl_seLogRR <- runif(n_neg_controls, 0.1, 0.5)

# Fit null distribution
null_dist <- fitNull(neg_ctrl_logRR, neg_ctrl_seLogRR)
print(null_dist)

# Calibrate a test estimate
test_logRR <- 0.8
test_se    <- 0.25

cal_p <- calibrateP(null_dist, test_logRR, test_se)
cal_ci <- calibrateCi(null_dist, test_logRR, test_se)
cat("\nUncalibrated p-value:", 2 * pnorm(-abs(test_logRR / test_se)), "\n")
cat("Calibrated p-value:", cal_p, "\n")
cat("Calibrated 95% CI:", round(exp(cal_ci), 3), "\n")

# Plot null distribution
plotCalibrationEffect(neg_ctrl_logRR, neg_ctrl_seLogRR,
                      title = "Empirical Null Distribution")
```

## Visualization

```r
# 1. Volcano plot of all drug-event pairs
all_results <- compute_ic(reports)
all_results$neg_log_p <- -log10(pnorm(-all_results$IC_shrunk / sqrt(all_results$IC_var)))

ggplot(all_results %>% filter(observed >= 3),
       aes(x = IC_shrunk, y = neg_log_p)) +
  geom_point(alpha = 0.2, size = 1, color = "grey50") +
  geom_point(data = known_signals, aes(x = IC_shrunk, y = -log10(
    pnorm(-IC_shrunk / sqrt(IC_var)))),
    color = "red", size = 3) +
  geom_vline(xintercept = 0, linetype = "dashed") +
  labs(title = "Volcano Plot: Drug-Event Signal Detection",
       x = "Information Component (IC)",
       y = "-log10(p-value)") +
  theme_minimal()

# 2. Signal detection summary
signal_summary <- results[, c("drug", "event", "PRR", "ROR", "signal_PRR")]
print(signal_summary)

# 3. Heatmap of top signals
top_pairs <- ic_results %>%
  filter(observed >= 5) %>%
  arrange(desc(IC_shrunk)) %>%
  head(50)

top_drugs  <- unique(top_pairs$drug)
top_events <- unique(top_pairs$event)

heat_data <- ic_results %>%
  filter(drug %in% top_drugs & event %in% top_events)

ggplot(heat_data, aes(x = event, y = drug, fill = IC_shrunk)) +
  geom_tile() +
  scale_fill_gradient2(low = "blue", mid = "white", high = "red", midpoint = 0) +
  labs(title = "IC Heatmap: Top Drug-Event Pairs",
       fill = "IC") +
  theme_minimal() +
  theme(axis.text.x = element_text(angle = 45, hjust = 1, size = 6),
        axis.text.y = element_text(size = 6))
```

## Tips and Best Practices

1. **Use multiple disproportionality measures** (PRR, ROR, IC, EBGM) and look for
   concordance across methods. No single measure is optimal.

2. **Apply Bayesian shrinkage** (BCPNN or MGPS) rather than raw PRR/ROR. Shrinkage
   reduces false positives from sparse cells with low expected counts.

3. **Set appropriate thresholds.** Common defaults: PRR > 2 + chi2 > 4 + N >= 3;
   ROR lower CI > 1; IC025 > 0; EB05 >= 2. Adjust based on the screening objective.

4. **Use empirical calibration** (OHDSI approach) when applying disproportionality to
   observational databases. Negative control outcomes reveal systematic bias.

5. **Disproportionality is hypothesis-generating, not confirmatory.** Signals require
   clinical review, epidemiological investigation, and biological plausibility.

6. **For liver safety, always generate the eDISH plot** early in clinical development.
   FDA expects it in NDA submissions.

7. **Account for confounding by indication.** Drugs used in sicker patients will have
   inflated event reporting. The SCCS and case-crossover designs mitigate this.

8. **Maintain a signal management log** documenting detection, validation, evaluation,
   and resolution for each signal. This is a regulatory expectation.
