# Multiplicity Adjustment — R Implementation

## Required Packages

```r
install.packages(c("gMCP", "multcomp", "ggplot2", "dplyr"))
# gMCP: graphical multiple comparison procedures
# multcomp: general multiple comparisons
```

## Example Dataset

We consider a Phase III clinical trial comparing an active drug against placebo for a
chronic pain condition. The trial has:
- Two doses: low dose and high dose
- Two co-primary endpoints: pain intensity (VAS) and physical function (HAQ-DI)
- This creates 4 hypotheses: H1 (high dose, pain), H2 (high dose, function),
  H3 (low dose, pain), H4 (low dose, function)

```r
set.seed(42)
n_per_arm <- 150

# Simulate trial data
placebo_pain <- rnorm(n_per_arm, mean = 55, sd = 20)
placebo_func <- rnorm(n_per_arm, mean = 1.4, sd = 0.5)

low_pain <- rnorm(n_per_arm, mean = 48, sd = 20)      # moderate pain reduction
low_func <- rnorm(n_per_arm, mean = 1.2, sd = 0.5)    # moderate function improvement

high_pain <- rnorm(n_per_arm, mean = 44, sd = 20)     # larger pain reduction
high_func <- rnorm(n_per_arm, mean = 1.05, sd = 0.5)  # larger function improvement

# Two-sample t-tests for each hypothesis
test_h1 <- t.test(high_pain, placebo_pain, alternative = "less")  # high dose, pain
test_h2 <- t.test(high_func, placebo_func, alternative = "less")  # high dose, function
test_h3 <- t.test(low_pain, placebo_pain, alternative = "less")   # low dose, pain
test_h4 <- t.test(low_func, placebo_func, alternative = "less")   # low dose, function

raw_pvalues <- c(H1 = test_h1$p.value, H2 = test_h2$p.value,
                  H3 = test_h3$p.value, H4 = test_h4$p.value)
cat("Raw p-values:\n")
print(round(raw_pvalues, 6))
```

## Complete Worked Example

### Step 1: Simple p-Value Adjustments with `p.adjust()`

```r
# Bonferroni correction
p_bonf <- p.adjust(raw_pvalues, method = "bonferroni")

# Holm step-down
p_holm <- p.adjust(raw_pvalues, method = "holm")

# Hochberg step-up
p_hoch <- p.adjust(raw_pvalues, method = "hochberg")

# Benjamini-Hochberg FDR
p_bh <- p.adjust(raw_pvalues, method = "BH")

# Compare all methods
results <- data.frame(
  Hypothesis = names(raw_pvalues),
  Raw = round(raw_pvalues, 6),
  Bonferroni = round(p_bonf, 6),
  Holm = round(p_holm, 6),
  Hochberg = round(p_hoch, 6),
  BH_FDR = round(p_bh, 6)
)
print(results)

# Decision at alpha = 0.05
cat("\nDecisions at alpha = 0.05:\n")
for (method in c("Bonferroni", "Holm", "Hochberg", "BH_FDR")) {
  rejected <- results$Hypothesis[results[[method]] <= 0.05]
  cat(sprintf("  %s: Reject %s\n", method,
              ifelse(length(rejected) > 0, paste(rejected, collapse = ", "), "none")))
}

# Interpretation: Holm and Hochberg are uniformly more powerful than Bonferroni.
# BH (FDR) is the most liberal, rejecting the most hypotheses. For confirmatory
# trials, use Holm or Hochberg (FWER control). For exploratory screening, use BH.
```

### Step 2: Graphical MCP for the Multi-Endpoint Trial

```r
library(gMCP)

# Define the graphical testing procedure:
# - Start with full alpha on high dose (clinical priority)
# - H1 (high dose, pain) and H2 (high dose, function) each get alpha/2
# - If H1 rejected, its alpha passes to H3 (low dose, pain) and H2
# - If H2 rejected, its alpha passes to H4 (low dose, function) and H1
# - Low dose hypotheses pass alpha to each other

# Hypothesis names
hypotheses <- c("H1: HighDose Pain", "H2: HighDose Function",
                "H3: LowDose Pain", "H4: LowDose Function")

# Initial alpha weights (sum to 1, multiplied by alpha=0.05 internally)
weights <- c(0.5, 0.5, 0, 0)

# Transition matrix: g[i,j] = fraction of alpha from H_i going to H_j
# Rows must sum to 1 (or 0 if no outgoing edges)
transitions <- rbind(
  c(0, 0.5, 0.5, 0),     # H1 -> split between H2 and H3
  c(0.5, 0, 0, 0.5),     # H2 -> split between H1 and H4
  c(0, 0, 0, 1),         # H3 -> all to H4
  c(0, 0, 1, 0)          # H4 -> all to H3
)

# Create the graphical test
graph <- matrix2graph(transitions)
graph@nodeAttr$fill <- c("H1" = "#4DBEEE", "H2" = "#4DBEEE",
                          "H3" = "#EDB120", "H4" = "#EDB120")

# Set weights
graph@weights <- weights

# Perform the graphical test
alpha <- 0.05
gMCP_result <- gMCP(graph, pvalues = raw_pvalues, alpha = alpha)

cat("\n--- Graphical MCP Results ---\n")
print(gMCP_result)

# The result shows which hypotheses are rejected after the full
# alpha-recycling procedure. Rejected hypotheses are marked TRUE.
cat("\nRejected hypotheses:\n")
rejected <- gMCP_result@rejected
names(rejected) <- hypotheses
print(rejected)
```

### Step 3: Visualize the Graphical Procedure

```r
# Plot the initial graph
plot(graph, main = "Initial Graphical Testing Procedure")

# The graph shows:
# - Nodes labeled with hypothesis names and initial alpha allocations
# - Directed edges with transition weights
# - Blue nodes = high dose, yellow nodes = low dose

# Step-by-step illustration
cat("\n--- Step-by-Step Procedure ---\n")
ordered_p <- sort(raw_pvalues)
cat(sprintf("Ordered p-values: %s\n",
            paste(sprintf("%s=%.6f", names(ordered_p), ordered_p), collapse = ", ")))

# After each rejection, the graph is updated. Remaining hypotheses may
# receive additional alpha from rejected hypotheses, making them easier
# to reject in subsequent steps.
```

### Step 4: Fixed-Sequence Testing

```r
# Alternative: hierarchical testing with a pre-specified order
# Order: H1 -> H2 -> H3 -> H4 (test high dose first, then low dose)

cat("\n--- Fixed-Sequence Testing ---\n")
alpha_seq <- 0.05
for (i in 1:4) {
  hyp_name <- names(raw_pvalues)[i]
  p_val <- raw_pvalues[i]
  reject <- p_val <= alpha_seq
  cat(sprintf("Step %d: Test %s at alpha=%.4f, p=%.6f -> %s\n",
              i, hyp_name, alpha_seq, p_val,
              ifelse(reject, "REJECT", "FAIL TO REJECT (STOP)")))
  if (!reject) break
}
# Interpretation: Fixed-sequence is maximally powerful for H1 (tested at full alpha)
# but testing stops at the first failure, so later hypotheses may never be tested.
```

## Advanced Example

### Gatekeeping for Primary and Secondary Endpoints

```r
# Scenario: 1 primary endpoint (overall survival) and 3 secondary endpoints
# (PFS, ORR, QoL). Secondaries tested only if primary is significant.

set.seed(100)
p_primary <- 0.003   # OS: significant
p_secondary <- c(PFS = 0.012, ORR = 0.048, QoL = 0.09)

cat("--- Serial Gatekeeping ---\n")
cat(sprintf("Primary (OS): p = %.4f -> %s at alpha = 0.05\n",
            p_primary, ifelse(p_primary <= 0.05, "REJECT", "FAIL")))

if (p_primary <= 0.05) {
  cat("Gate open: proceed to secondary endpoints\n")
  # Apply Hochberg to secondary endpoints at full alpha
  p_sec_adj <- p.adjust(p_secondary, method = "hochberg")
  for (i in seq_along(p_secondary)) {
    cat(sprintf("  %s: raw p = %.4f, adjusted p = %.4f -> %s\n",
                names(p_secondary)[i], p_secondary[i], p_sec_adj[i],
                ifelse(p_sec_adj[i] <= 0.05, "REJECT", "FAIL")))
  }
} else {
  cat("Gate closed: secondary endpoints not tested\n")
}
```

### Large-Scale FDR Control (Biomarker Screening)

```r
# Simulate 1000 biomarker tests, 50 truly associated
set.seed(555)
m <- 1000
m_true <- 50

p_null <- runif(m - m_true)  # null p-values uniform
p_alt <- rbeta(m_true, 1, 20) # alternative p-values small

all_p <- c(p_alt, p_null)
true_status <- c(rep("alternative", m_true), rep("null", m - m_true))

# BH procedure
p_bh <- p.adjust(all_p, method = "BH")
discoveries_bh <- sum(p_bh <= 0.05)
false_discoveries_bh <- sum(p_bh <= 0.05 & true_status == "null")
fdp_bh <- false_discoveries_bh / max(discoveries_bh, 1)

cat(sprintf("\n--- FDR Control (1000 tests) ---\n"))
cat(sprintf("BH discoveries: %d (of %d true associations)\n", discoveries_bh, m_true))
cat(sprintf("False discoveries: %d\n", false_discoveries_bh))
cat(sprintf("False discovery proportion: %.3f (target FDR: 0.05)\n", fdp_bh))
```

## Visualization

```r
library(ggplot2)

# Comparison of adjustment methods across a range of raw p-values
p_grid <- seq(0.001, 0.1, by = 0.001)
m_tests <- 4

viz_data <- data.frame(
  raw_p = rep(p_grid, 4),
  method = rep(c("Bonferroni", "Holm", "Hochberg", "BH (FDR)"), each = length(p_grid)),
  adjusted_p = c(
    pmin(p_grid * m_tests, 1),                        # Bonferroni
    pmin(p_grid * m_tests, 1),                        # Holm (approx for smallest p)
    pmin(p_grid * (m_tests), 1),                      # Hochberg (approx)
    pmin(p_grid * m_tests / seq_along(p_grid) * length(p_grid), 1)  # BH approx
  )
)

# Simpler: just show the thresholds
threshold_data <- data.frame(
  method = c("Bonferroni", "Holm (step 1)", "Holm (step 4)", "BH (step 1)", "BH (step 4)"),
  threshold = c(0.05/4, 0.05/4, 0.05/1, 0.05*1/4, 0.05*4/4),
  label = c("0.0125", "0.0125", "0.0500", "0.0125", "0.0500")
)

ggplot(threshold_data, aes(x = reorder(method, threshold), y = threshold)) +
  geom_col(fill = "steelblue", width = 0.6) +
  geom_text(aes(label = label), vjust = -0.5) +
  geom_hline(yintercept = 0.05, linetype = "dashed", color = "red") +
  annotate("text", x = 0.5, y = 0.052, label = "Nominal alpha = 0.05", color = "red",
           hjust = 0, size = 3.5) +
  labs(x = "", y = "Rejection Threshold",
       title = "Comparison of Multiplicity-Adjusted Thresholds (m=4 tests)") +
  theme_minimal(base_size = 12) +
  coord_flip()

# Volcano-style plot for biomarker screening
bio_data <- data.frame(
  neg_log_p = -log10(all_p),
  bh_sig = p_bh <= 0.05,
  truth = true_status
)

ggplot(bio_data, aes(x = seq_along(all_p), y = neg_log_p, color = bh_sig)) +
  geom_point(alpha = 0.5, size = 1) +
  geom_hline(yintercept = -log10(0.05), linetype = "dashed", color = "grey") +
  scale_color_manual(values = c("FALSE" = "grey70", "TRUE" = "red"),
                     labels = c("Not significant", "BH significant")) +
  labs(x = "Biomarker Index", y = "-log10(p-value)", color = "",
       title = "BH FDR-Adjusted Significance in Biomarker Screening") +
  theme_minimal(base_size = 12)
```

## Tips and Best Practices

1. **Pre-specify everything**: The multiplicity strategy must be finalized in the SAP before
   database lock. Post-hoc changes to the testing hierarchy or alpha allocation undermine
   the integrity of the analysis.

2. **Match the method to the goal**: Use FWER control (Holm, graphical) for confirmatory
   analyses. Use FDR control (BH) for exploratory, high-dimensional screening. Never use
   FDR in a registration-enabling trial's primary analysis.

3. **Graphical approaches are recommended**: The FDA and EMA both recognize graphical MCP
   procedures. They are flexible, transparent, and can encode clinical priorities directly.
   The `gMCP` package makes implementation straightforward.

4. **Holm is always valid**: When in doubt, Holm controls FWER under any dependence structure
   and is uniformly more powerful than Bonferroni. It should be the default simple procedure.

5. **Alpha allocation reflects priorities**: In graphical procedures, allocate more initial
   alpha to the most important hypotheses. The transition matrix encodes what happens when
   a hypothesis is rejected.

6. **Do not adjust for pre-specified subgroups tested in a hierarchical manner**: If a trial
   protocol specifies testing the overall population first, then a biomarker subgroup only
   if the overall test fails, this is a valid fixed-sequence procedure that does not require
   additional Bonferroni-type correction.

7. **Document the procedure with a figure**: Always include a diagram of the graphical testing
   procedure in the SAP and the clinical study report. This makes the procedure transparent
   and reproducible.

8. **Consider the correlation structure**: If test statistics are highly correlated (e.g.,
   co-primary endpoints measured on the same patients), Bonferroni-based methods waste power.
   Parametric methods (Dunnett, Simes-based closed testing) can recapture this power.
