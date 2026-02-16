# Meta-Analysis — R Implementation

## Required Packages

```r
install.packages(c("meta", "metafor", "netmeta", "dmetar"))

library(meta)
library(metafor)
library(netmeta)
```

- **meta**: High-level functions for common meta-analyses, forest/funnel plots.
- **metafor**: Comprehensive, flexible meta-analytic modeling (REML, meta-regression, diagnostics).
- **netmeta**: Frequentist network meta-analysis.
- **dmetar**: Companion package for advanced diagnostics and helpers.

## Example Dataset

We use a dataset of 13 randomized controlled trials comparing a new antihypertensive drug versus placebo for reduction in systolic blood pressure (SBP, mmHg). Each trial reports the mean difference (MD) in SBP change from baseline and its standard error.

```r
dat <- data.frame(
  study = paste0("Trial_", 1:13),
  year  = c(2005, 2007, 2008, 2009, 2010, 2011, 2012, 2013, 2014, 2015, 2016, 2018, 2020),
  md    = c(-8.2, -5.1, -7.8, -6.3, -9.1, -4.5, -7.0, -8.8, -6.0, -10.2, -5.5, -7.3, -6.9),
  se    = c(1.5, 2.0, 1.8, 1.2, 2.5, 1.0, 1.6, 1.4, 2.2, 1.9, 1.3, 1.7, 1.1),
  n_treat   = c(120, 85, 100, 200, 60, 300, 150, 130, 75, 90, 180, 110, 250),
  n_control = c(118, 87, 102, 198, 62, 295, 148, 132, 73, 88, 182, 108, 248),
  dose  = c(10, 10, 20, 20, 10, 20, 10, 20, 10, 20, 10, 20, 20)
)
```

This represents 13 trials with mean differences ranging from -4.5 to -10.2 mmHg, with varying sample sizes and two dose levels (10mg and 20mg).

## Complete Worked Example

### Step 1: Load Data and Fit Fixed-Effect and Random-Effects Models

```r
library(metafor)

# Fit random-effects model using REML
res_re <- rma(yi = md, sei = se, data = dat, method = "REML")
summary(res_re)
```

**Output interpretation**: The `rma()` output shows the pooled mean difference, its 95% CI, and a z-test for significance. The `tau^2` estimate and `I^2` statistic quantify heterogeneity. If `I^2` exceeds ~50%, substantial heterogeneity is present.

```r
# Fit fixed-effect model for comparison
res_fe <- rma(yi = md, sei = se, data = dat, method = "FE")
summary(res_fe)
```

**Output interpretation**: The fixed-effect estimate will typically be similar in direction but may differ in magnitude and will have a narrower confidence interval. Compare with the random-effects estimate to understand the impact of heterogeneity.

### Step 2: Assess Heterogeneity

```r
# Heterogeneity statistics from the RE model
cat("Tau-squared:", res_re$tau2, "\n")
cat("I-squared:", res_re$I2, "%\n")
cat("H-squared:", res_re$H2, "\n")
cat("Q-statistic:", res_re$QE, "df =", res_re$k - 1,
    "p =", res_re$QEp, "\n")

# 95% prediction interval
pi <- predict(res_re)
cat("Prediction interval:", pi$pi.lb, "to", pi$pi.ub, "\n")
```

**Output interpretation**: The Q-test p-value below 0.10 suggests significant heterogeneity. `I^2` quantifies what proportion of variability is between-study. The prediction interval shows the range within which the true effect in a future study would likely fall — this is broader than the CI for the mean and is more clinically relevant.

### Step 3: Forest Plot

```r
library(meta)

m <- metagen(TE = md, seTE = se, studlab = study, data = dat,
             sm = "MD", fixed = TRUE, random = TRUE,
             title = "Antihypertensive Drug vs Placebo: SBP Reduction")

forest(m,
       sortvar = year,
       leftcols  = c("studlab", "TE", "seTE"),
       leftlabs  = c("Study", "MD", "SE"),
       rightcols = c("effect", "ci", "w.random"),
       rightlabs = c("MD", "95% CI", "Weight"),
       print.tau2 = TRUE,
       print.I2 = TRUE,
       print.pval.Q = TRUE,
       col.diamond = "steelblue",
       col.square = "darkblue")
```

**Output interpretation**: The forest plot shows each trial as a square (size proportional to weight) with horizontal CI bars. The diamond at the bottom represents the pooled effect. Visual inspection reveals whether studies are consistent and whether the pooled estimate is driven by a few large studies.

### Step 4: Funnel Plot and Publication Bias

```r
# Funnel plot
funnel(res_re, main = "Funnel Plot: SBP Meta-Analysis",
       xlab = "Mean Difference (mmHg)")

# Egger's regression test for asymmetry
regtest(res_re, model = "lm")

# Trim-and-fill analysis
tf <- trimfill(res_re)
summary(tf)
funnel(tf, main = "Trim-and-Fill Funnel Plot")
```

**Output interpretation**: If Egger's test p-value is < 0.10, funnel plot asymmetry is present, suggesting possible publication bias. The trim-and-fill method imputes "missing" studies (shown as filled points on the funnel plot) and recalculates the pooled estimate. A substantially attenuated adjusted estimate suggests that publication bias may inflate the observed effect.

### Step 5: Meta-Regression

```r
# Meta-regression: does dose level explain heterogeneity?
res_mr <- rma(yi = md, sei = se, mods = ~ dose, data = dat, method = "REML")
summary(res_mr)

# Proportion of heterogeneity explained
cat("R-squared:", res_mr$R2, "%\n")

# Bubble plot
regplot(res_mr, xlab = "Dose (mg)", ylab = "Mean Difference (mmHg)",
        main = "Meta-Regression: Effect by Dose Level")
```

**Output interpretation**: The moderator coefficient for `dose` indicates whether higher doses are associated with larger SBP reductions. `R^2` shows the proportion of between-study variance explained by dose. A significant moderator suggests that dose is a source of heterogeneity. The bubble plot visualizes this relationship, with circle sizes proportional to study precision.

## Advanced Example

### Network Meta-Analysis

Consider a scenario comparing three treatments (Drug A, Drug B, Placebo) across multiple trials.

```r
library(netmeta)

# Simulated NMA data: pairwise comparisons from trials
nma_dat <- data.frame(
  study  = c("S1","S2","S3","S4","S5","S6","S7","S8","S9","S10"),
  treat1 = c("DrugA","DrugA","DrugB","DrugA","DrugB","DrugA",
             "DrugB","DrugA","DrugB","DrugA"),
  treat2 = c("Placebo","Placebo","Placebo","DrugB","Placebo",
             "Placebo","DrugA","DrugB","Placebo","DrugB"),
  md     = c(-7.5, -8.1, -5.2, -2.8, -4.8, -9.0, 3.1, -3.0, -5.5, -2.5),
  se     = c(1.4, 1.6, 1.2, 1.5, 1.8, 1.1, 1.3, 1.7, 2.0, 1.4)
)

# Fit NMA
net <- netmeta(TE = md, seTE = se,
               treat1 = treat1, treat2 = treat2,
               studlab = study, data = nma_dat,
               sm = "MD", reference.group = "Placebo",
               comb.fixed = FALSE, comb.random = TRUE)

summary(net)

# Forest plot of NMA results
forest(net, reference.group = "Placebo",
       sortvar = TE, smlab = "Mean Difference vs Placebo")

# League table
netleague(net)

# Ranking (P-scores, analogous to SUCRA)
netrank(net, small.values = "desirable")

# Check consistency (node-splitting)
netsplit(net)
```

**Output interpretation**: The NMA summary provides treatment effect estimates for all pairwise comparisons. P-scores rank treatments (higher = better for beneficial outcomes; here we use `small.values = "desirable"` since larger SBP reduction is better). The `netsplit()` output tests for inconsistency at each comparison node — significant p-values indicate discrepancy between direct and indirect evidence.

## Visualization

### Comprehensive Forest Plot with Subgroups

```r
# Subgroup analysis by dose
m_sub <- update(m, subgroup = dat$dose, print.subgroup.name = TRUE)

forest(m_sub,
       sortvar = year,
       print.tau2 = TRUE,
       print.I2.ci = TRUE,
       test.subgroup = TRUE,
       col.diamond = "steelblue",
       col.diamond.lines = "darkblue",
       header = "Forest Plot with Dose Subgroups")
```

### Influence Diagnostics

```r
# Leave-one-out analysis
loo <- leave1out(res_re)
print(loo)

# Baujat plot: identifies studies contributing to heterogeneity
baujat(res_re, main = "Baujat Plot: Contribution to Heterogeneity")

# Influence diagnostics
inf <- influence(res_re)
plot(inf)
```

**Output interpretation**: The leave-one-out analysis shows how the pooled estimate changes when each study is removed. The Baujat plot identifies influential studies: studies in the upper-right contribute most to both heterogeneity and the overall result. The influence diagnostics panel includes Cook's distance, DFFITS, and other measures to flag outlier studies.

## Tips and Best Practices

1. **Always report both fixed-effect and random-effects results** when they diverge, as this signals meaningful heterogeneity.
2. **Use REML or Paule-Mandel** estimators for tau-squared rather than DerSimonian-Laird, especially with fewer than 20 studies.
3. **Apply the Hartung-Knapp-Sidik-Jonkman (HKSJ) adjustment** for the confidence interval of the pooled effect under random effects — it produces more appropriate coverage than the standard Wald CI.
4. **Always present a prediction interval** alongside the pooled estimate to convey the expected range of effects in new settings.
5. **Do not rely solely on the Q-test** for heterogeneity — it has low power with few studies and excessive power with many studies.
6. **Egger's test has limited power** with fewer than 10 studies; interpret with caution.
7. **For NMA**, always check the consistency assumption via node-splitting before interpreting rankings.
8. **Pre-specify subgroup and meta-regression analyses** to avoid data-dredging. At least 10 studies per covariate is a common rule of thumb.
9. **Register your systematic review protocol** (e.g., PROSPERO) to enhance transparency.
10. **Follow PRISMA guidelines** for reporting systematic reviews and meta-analyses.
