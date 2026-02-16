# Healthcare Cost and Resource Analysis — Theory

## Introduction

Healthcare cost data present unique statistical challenges. Medical expenditures are
typically right-skewed, contain a large proportion of zeros (patients who incur no cost
in a given period), exhibit heteroscedasticity, and are bounded below at zero. Naive
application of ordinary least squares on raw or log-transformed costs leads to biased or
inefficient estimates. This chapter covers the statistical models and economic evaluation
frameworks used in pharmacoeconomics, health technology assessment (HTA), and
comparative-effectiveness research.

## Mathematical Foundation

### The Two-Part (Hurdle) Model

The two-part model separately models (i) the probability of any cost and (ii) the
conditional amount given positive cost:

```
Part 1:  P(Y > 0 | X) = logit^{-1}(X * gamma)         [logistic regression]
Part 2:  E[Y | Y > 0, X] = g^{-1}(X * beta)            [GLM on positive costs]
```

The unconditional mean is:

```
E[Y | X] = P(Y > 0 | X) * E[Y | Y > 0, X]
```

This is preferred when the zero-cost population is structurally different from
the positive-cost population (e.g., patients who never seek care vs. those who do).

### Generalised Linear Models (GLM) for Costs

GLMs with a **log link** and **gamma family** are widely recommended for cost data
(Manning and Mullahy, 2001):

```
E[Y | X] = exp(X * beta)
Var(Y | X) = phi * [E[Y | X]]^2         [gamma variance function]
```

The log link ensures predicted costs are positive. The gamma variance function
accommodates heteroscedasticity proportional to the square of the mean, which is
typical in cost data.

**Choosing the variance function:** The modified Park test regresses the log of the
squared residual on the log of the predicted mean. The slope indicates:

| Slope | Family |
|-------|--------|
| ~0 | Gaussian |
| ~1 | Poisson |
| ~2 | Gamma |
| ~3 | Inverse Gaussian |

### Log-Transformed OLS and Retransformation

An alternative is `log(Y) = X*beta + e`, with retransformation to the original scale.
If errors are normal and homoscedastic:

```
E[Y | X] = exp(X*beta) * exp(sigma^2 / 2)         [Duan's smearing factor]
```

When errors are heteroscedastic, Duan's non-parametric smearing estimator is used:

```
E[Y | X] = exp(X*beta) * (1/n) * SUM(exp(e_hat_i))
```

This approach is fragile; GLMs are generally preferred.

### Tobit Model

For censored cost data (e.g., insurance caps, deductibles), the Tobit model treats
zero costs as left-censored observations of a latent continuous variable:

```
Y* = X*beta + e,   e ~ N(0, sigma^2)
Y  = max(0, Y*)
```

This is appropriate when zeros represent censoring, not a structural zero process.

### Zero-Inflated Models

When zeros arise from two distinct processes (structural zeros and sampling zeros),
zero-inflated gamma or zero-inflated lognormal models may be used. These combine a
point mass at zero with a continuous positive distribution.

## Key Concepts

### Cost-Effectiveness Analysis (CEA)

CEA compares an intervention to a comparator in terms of incremental costs and
incremental effectiveness:

```
ICER = (C_new - C_comparator) / (E_new - E_comparator) = Delta_C / Delta_E
```

where effectiveness is measured in natural units (life-years, QALYs, etc.).

The **willingness-to-pay (WTP) threshold** lambda determines whether the intervention
is cost-effective:

```
Cost-effective if ICER < lambda
Equivalently:    NMB = lambda * Delta_E - Delta_C > 0
```

where NMB is the **net monetary benefit**.

### The Cost-Effectiveness Plane

The CE plane plots (Delta_E, Delta_C) pairs. Four quadrants:

| Quadrant | Interpretation |
|----------|---------------|
| NE (Delta_E > 0, Delta_C > 0) | More effective, more costly (trade-off) |
| SE (Delta_E > 0, Delta_C < 0) | More effective, less costly (dominant) |
| NW (Delta_E < 0, Delta_C > 0) | Less effective, more costly (dominated) |
| SW (Delta_E < 0, Delta_C < 0) | Less effective, less costly (trade-off) |

### Cost-Effectiveness Acceptability Curve (CEAC)

The CEAC plots the probability that the intervention is cost-effective as a function
of WTP threshold lambda:

```
P(NMB > 0 | lambda) = P(lambda * Delta_E - Delta_C > 0)
```

Estimated from bootstrap replicates or Bayesian posterior draws.

### Bootstrap Inference for Cost Differences

Because cost distributions are skewed, normal-theory CIs for mean cost differences are
unreliable. Non-parametric bootstrap percentile or bias-corrected accelerated (BCa)
CIs are recommended.

## Assumptions

1. **GLM assumptions:** Correct specification of link function and variance function.
2. **Two-part model:** The zero-generating process is independent of the positive-cost
   process (conditional on covariates).
3. **Independence:** Observations are independent (or clustering is addressed).
4. **No informative censoring:** For cost data truncated by death or loss to follow-up,
   methods for censored cost data (Lin et al., 1997) are needed.
5. **Exchangeability:** For causal cost comparisons, treatment groups are exchangeable
   conditional on covariates.

## Variants and Extensions

### Markov Models for Health Economics

State-transition (Markov) models simulate disease progression through health states over
time. Each state has associated costs and utilities. The model produces expected total
costs and QALYs for each strategy.

- **Cycle length:** typically 1 month or 1 year.
- **Transition probabilities:** derived from clinical data.
- **Half-cycle correction:** adjusts for events occurring mid-cycle.
- **Discounting:** future costs and outcomes are discounted (e.g., 3% per year).

### Propensity-Score Weighted Cost Comparisons

In observational data, inverse probability of treatment weighting (IPTW) balances
confounders before comparing costs. Combined with GLM, this provides doubly-robust
cost estimates.

### Censored Cost Data

When patients die or are lost to follow-up before the end of the cost-accrual window,
total costs are censored. Methods include:

- **Lin's method (1997):** partitions the time horizon into intervals and uses
  Kaplan-Meier-weighted average costs.
- **IPCW:** inverse probability of censoring weighting.

## When to Use This Method

- Health technology assessment submissions (NICE, CADTH, PBAC, ICER).
- Clinical trial-based economic evaluations.
- Retrospective claims-data cost analyses.
- Budget impact modelling for formulary decisions.
- Comparative effectiveness research with cost endpoints.

## Strengths and Limitations

### Strengths
- GLMs handle skewness and heteroscedasticity without transformation.
- Two-part models correctly represent the zero-cost process.
- Bootstrap inference is distribution-free for cost differences.
- CEA integrates costs and effectiveness into a single decision metric.

### Limitations
- Model choice (GLM family/link) can materially affect results.
- Heavy right tails may require trimming or truncation sensitivity analyses.
- Observational cost comparisons are vulnerable to unmeasured confounding.
- Markov models require many assumptions about transition probabilities.

## Key References

1. Manning, W. G., & Mullahy, J. (2001). Estimating log models: to transform or not
   to transform? *Journal of Health Economics*, 20(4), 461-494.
2. Duan, N. (1983). Smearing estimate: a nonparametric retransformation method.
   *JASA*, 78(383), 605-610.
3. Drummond, M. F., et al. (2015). *Methods for the Economic Evaluation of Health
   Care Programmes*. 4th ed. Oxford University Press.
4. Lin, D. Y., et al. (1997). Estimating medical costs from incomplete follow-up data.
   *Biometrics*, 53(2), 419-434.
5. Barber, J. A., & Thompson, S. G. (2000). Analysis of cost data in randomized trials.
   *Statistical Methods in Medical Research*, 9(4), 303-325.
6. Briggs, A. H., et al. (2006). *Decision Modelling for Health Economic Evaluation*.
   Oxford University Press.
