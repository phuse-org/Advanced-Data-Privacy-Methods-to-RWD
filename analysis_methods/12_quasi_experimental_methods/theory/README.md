# Quasi-Experimental Methods — Theory

## Introduction

Quasi-experimental methods occupy a critical space between randomized experiments and purely
observational analyses. When randomization is infeasible, unethical, or when researchers must
evaluate interventions using retrospective data, these methods provide rigorous frameworks
for causal inference. In health research, common applications include evaluating the impact
of new drug policies, regulatory changes, clinical threshold-based treatment decisions, and
population-level health interventions.

The core challenge is that without randomization, treatment assignment may be confounded by
unobserved factors. Each quasi-experimental method exploits a different structural feature of
the data to address this confounding, effectively creating a "natural experiment."

## Mathematical Foundation

### Interrupted Time Series (ITS)

ITS analyzes the impact of an intervention on a time-series outcome by modeling the change in
level and trend at the intervention point. The basic segmented regression model is:

```
Y_t = beta_0 + beta_1 * time + beta_2 * intervention + beta_3 * time_after + epsilon_t
```

Where:
- `beta_0`: baseline level at time zero
- `beta_1`: pre-intervention trend (slope per time unit)
- `beta_2`: immediate change in level at the intervention point
- `beta_3`: change in trend after the intervention
- `time`: elapsed time from start
- `intervention`: indicator (0 before, 1 after)
- `time_after`: time since intervention (0 before intervention)

The counterfactual is the extrapolation of the pre-intervention trend. Autocorrelation in
the errors is typically handled via Newey-West standard errors or ARIMA modeling.

### Difference-in-Differences (DiD)

DiD compares the change over time in an outcome between a treatment group and a control group:

```
Y_it = alpha + beta_1 * Post_t + beta_2 * Treat_i + delta * (Post_t x Treat_i) + epsilon_it
```

Where:
- `delta` is the DiD estimate of the causal effect (the ATT)
- `Post_t`: indicator for the post-intervention period
- `Treat_i`: indicator for the treatment group

The key identifying assumption is **parallel trends**: absent the intervention, the treatment
and control groups would have followed the same trajectory. The ATT is:

```
delta = [E(Y|Treat,Post) - E(Y|Treat,Pre)] - [E(Y|Control,Post) - E(Y|Control,Pre)]
```

### Regression Discontinuity (RD)

RD exploits a known threshold (cutoff) in a continuous assignment variable (running variable)
that determines treatment:

**Sharp RD**: Treatment is a deterministic function of the running variable X:

```
D_i = 1{X_i >= c}
```

The causal effect is estimated at the cutoff:

```
tau_RD = lim_{x->c+} E[Y|X=x] - lim_{x->c-} E[Y|X=x]
```

**Fuzzy RD**: The probability of treatment jumps at the cutoff but does not change from 0 to 1.
This is an IV-type estimator:

```
tau_FuzzyRD = [lim E(Y|X=x+) - lim E(Y|X=x-)] / [lim E(D|X=x+) - lim E(D|X=x-)]
```

This estimates a local average treatment effect (LATE) at the cutoff.

### Synthetic Control Method

For comparative case studies (one treated unit, several control units), the synthetic control
constructs a weighted combination of control units that best matches the pre-intervention
trajectory of the treated unit:

```
Y_treated,post - sum(w_j * Y_j,post)
```

where weights w_j are chosen to minimize pre-intervention prediction error:

```
min_{W} ||X_1 - X_0 * W||  subject to  w_j >= 0, sum(w_j) = 1
```

## Key Concepts

### Parallel Trends Assumption (DiD)

This is the central identifying assumption for DiD. It states that trends in the outcome
would have been identical across groups in the absence of treatment. While untestable, it
can be assessed by examining pre-treatment trends. If pre-treatment trends differ, the DiD
estimate is biased.

### Bandwidth Selection (RD)

In RD designs, the treatment effect is estimated locally around the cutoff. The bandwidth
determines how much data is used. Narrower bandwidths reduce bias (observations far from
the cutoff are less comparable) but increase variance. Optimal bandwidth selection methods
(Imbens-Kalyanaraman, Calonico-Cattaneo-Titiunik) balance this bias-variance tradeoff.

### Controlled ITS

A controlled ITS adds a comparison group that was not exposed to the intervention, combining
the strengths of ITS (modeling temporal trends) and DiD (differencing out common shocks). It
is more robust than a single-group ITS because it controls for concurrent events.

## Assumptions

**ITS:**
1. The intervention is the only cause of the level/trend change at the time point.
2. The pre-intervention trend would have continued absent the intervention.
3. No other co-interventions occurred at the same time.

**DiD:**
1. Parallel trends in the absence of treatment.
2. No anticipation effects.
3. Stable composition of treatment and control groups.
4. SUTVA (no spillover between groups).

**RD:**
1. No manipulation of the running variable around the cutoff.
2. Potential outcomes are continuous at the cutoff.
3. Continuity of density of the running variable at the cutoff (tested via McCrary test).

**Synthetic Control:**
1. The donor pool is unaffected by the intervention.
2. Good pre-intervention fit implies good counterfactual prediction.
3. No spillover from the treated to control units.

## Variants and Extensions

- **Staggered DiD**: When units adopt treatment at different times. Recent methods (Callaway
  and Sant'Anna, Sun and Abraham) address bias in two-way fixed effects with staggered
  adoption.
- **Event study designs**: Plot treatment effects at each time period relative to adoption,
  providing visual evidence for parallel trends and dynamic effects.
- **RD with multiple cutoffs**: Pool estimates across multiple thresholds for increased power.
- **Augmented synthetic control (augsynth)**: Combines outcome modeling with the synthetic
  control weighting to reduce bias.
- **Geographic regression discontinuity**: Exploits spatial boundaries (e.g., state borders)
  as quasi-experimental variation.

## When to Use This Method

- **ITS**: Evaluating a policy that was implemented at a known time point affecting an entire
  population (e.g., a new drug safety regulation, ban on a medication).
- **DiD**: When you have a treated and untreated group observed before and after an intervention
  (e.g., comparing hospitals that adopted a new protocol vs those that did not).
- **RD**: When treatment is determined by a continuous score crossing a threshold (e.g.,
  prescribing statins when LDL exceeds 190 mg/dL, eligibility for a program based on a
  risk score).
- **Synthetic Control**: When a single unit (region, hospital system) receives an intervention
  and you need to construct a credible counterfactual from unaffected units.

## Strengths and Limitations

### Strengths
- Exploit natural variation, avoiding the ethical and practical challenges of randomization.
- ITS and DiD control for time-invariant confounders.
- RD provides estimates with high internal validity near the cutoff.
- Transparent assumptions that can often be partially tested.

### Limitations
- ITS requires sufficiently long pre- and post-intervention series (typically 8+ time points each).
- DiD is biased if parallel trends fail; this assumption is fundamentally untestable.
- RD estimates are local to the cutoff and may not generalize.
- Synthetic control requires a good donor pool and long pre-intervention period.
- All methods require that the quasi-experimental variation is not contaminated by co-interventions.

## Key References

1. Bernal JL, Cummins S, Gasparrini A. Interrupted time series regression for the evaluation
   of public health interventions. *International Journal of Epidemiology*, 2017;46(1):348-355.
2. Wing C, Simon K, Bello-Gomez RA. Designing difference in difference studies: best practices
   for public health policy research. *Annual Review of Public Health*, 2018;39:453-469.
3. Cattaneo MD, Idrobo N, Titiunik R. *A Practical Introduction to Regression Discontinuity
   Designs*. Cambridge University Press, 2020.
4. Abadie A, Diamond A, Hainmueller J. Synthetic control methods for comparative case studies.
   *Journal of the American Statistical Association*, 2010;105(490):493-505.
5. Callaway B, Sant'Anna PHC. Difference-in-differences with multiple time periods. *Journal
   of Econometrics*, 2021;225(2):200-230.
6. Calonico S, Cattaneo MD, Titiunik R. Robust nonparametric confidence intervals for
   regression-discontinuity designs. *Econometrica*, 2014;82(6):2295-2326.
