# Dose-Response Modeling — Theory

## Introduction

Dose-response modeling is fundamental to drug development. It characterizes how the magnitude
of a therapeutic effect (or an adverse effect) changes as the dose of a drug increases. This
relationship guides dose selection for Phase III trials, supports regulatory labeling, and
informs clinical dosing recommendations. The challenge lies in fitting appropriate mathematical
models to limited data from Phase II dose-finding trials, where typically 3-6 dose levels
plus placebo are studied.

Modern dose-response analysis has moved beyond simple pairwise comparisons toward model-based
approaches that borrow information across dose groups, provide smooth dose-response curves,
and enable interpolation to untested doses. The MCP-Mod (Multiple Comparisons Procedure -
Modelling) framework, now endorsed by the EMA and FDA as a fit-for-purpose methodology, is
the current standard for Phase II dose-finding.

## Mathematical Foundation

### Emax Model

The Emax model is the most common dose-response model in drug development:

```
E(Y|d) = E0 + Emax * d / (ED50 + d)
```

Where:
- `E0`: placebo response (response at dose = 0)
- `Emax`: maximum drug effect (asymptotic effect at infinite dose)
- `d`: dose level
- `ED50`: dose producing 50% of Emax (potency parameter)

### Sigmoid Emax (Hill) Model

The Hill equation adds a steepness parameter:

```
E(Y|d) = E0 + Emax * d^h / (ED50^h + d^h)
```

Where `h` (the Hill coefficient) controls the steepness of the curve. When h = 1, this
reduces to the standard Emax model. When h > 1, the curve is steeper (more switch-like).
When h < 1, the curve rises more gradually.

### Linear and Log-Linear Models

Simple parametric alternatives:

```
Linear:      E(Y|d) = E0 + beta * d
Log-linear:  E(Y|d) = E0 + beta * log(d + c)
```

where `c` is a small constant to handle dose = 0. These models are appropriate when the
dose range is narrow and curvature is minimal.

### Quadratic Model

```
E(Y|d) = E0 + beta_1 * d + beta_2 * d^2
```

This allows for a non-monotone response (e.g., efficacy increases then decreases at high
doses due to tolerability issues).

### Exponential Model

```
E(Y|d) = E0 + E1 * (exp(d / delta) - 1)
```

This captures a dose-response that is initially flat then rises steeply.

## Key Concepts

### MCP-Mod (Multiple Comparisons Procedure - Modelling)

MCP-Mod (Bretz et al., 2005) is a two-step procedure:

**Step 1 — MCP (Signal Detection):**
- Specify a set of candidate dose-response models (e.g., linear, Emax, sigmoid Emax).
- For each model, compute the optimal contrast test statistic.
- Apply a multiplicity-adjusted significance test to determine whether a dose-response
  signal exists.
- This step controls the FWER at level alpha.

**Step 2 — Mod (Dose-Response Estimation):**
- Among the significant models, select the best-fitting model(s) using AIC or other criteria.
- Fit the selected model to estimate the dose-response curve.
- Use the fitted curve to estimate the target dose (e.g., the minimum effective dose or
  the dose achieving a specified percentage of maximum effect).

The MCP step ensures type I error control (important for regulatory decisions), while the Mod
step provides efficient estimation by leveraging the dose-response shape.

### Contrast Tests

For a candidate model f(d, theta), the optimal contrast vector is:

```
c_opt = S^{-1} * mu / ||S^{-1} * mu||
```

where `mu` is the vector of mean responses under the candidate model (centered to remove
the intercept) and `S` is the covariance matrix. The contrast test statistic is:

```
T = sum(c_i * Y_bar_i) / sqrt(sum(c_i^2 * var_i / n_i))
```

### Model Averaging

Rather than selecting a single "best" model, model averaging combines predictions across
multiple candidate models, weighted by their posterior probability or AIC weights:

```
E_avg(Y|d) = sum(w_k * E_k(Y|d))
```

where `w_k` are model weights (e.g., AIC weights: w_k proportional to exp(-0.5 * delta_AIC_k)).
This accounts for model uncertainty and typically provides better predictions than any
single model.

### Benchmark Dose (BMD) Analysis

In toxicology and safety assessment, the BMD is the dose corresponding to a pre-specified
change in response (the benchmark response, BMR). For example, the BMD_10 is the dose at
which the response changes by 10% from the background rate. The BMDL (lower confidence
limit on the BMD) is used as a point of departure for risk assessment.

### Exposure-Response

In regulatory pharmacology, dose-response is increasingly supplemented by exposure-response
analysis, which relates drug concentrations (AUC, Cmax) rather than administered dose to
effects. This accounts for pharmacokinetic variability between patients.

```
E(Y|AUC) = E0 + Emax * AUC / (EC50 + AUC)
```

## Assumptions

1. **Monotonicity** (for Emax models): The response is assumed to increase (or decrease)
   monotonically with dose. Violated if the dose-response is U-shaped or bell-shaped.
2. **Correct functional form**: Nonlinear models assume a specific mathematical relationship.
   Misspecification can lead to biased dose estimates.
3. **Independence**: Observations are independent (within standard clinical trial designs).
4. **Normally distributed residuals** (for continuous endpoints): Required for standard
   nonlinear least squares. Generalized models extend to binary and count data.
5. **Fixed dose levels**: Standard analysis assumes dose is measured without error and
   patients receive exactly the assigned dose.

## Variants and Extensions

- **Bayesian dose-response modeling**: Places prior distributions on model parameters, enabling
  formal incorporation of prior information and posterior probability statements.
- **Generalized MCP-Mod**: Extends MCP-Mod to non-normal endpoints (binary, time-to-event)
  using generalized nonlinear models.
- **Longitudinal dose-response**: Models the dose-response relationship over time using mixed
  effects models (e.g., how quickly different doses reach plateau effect).
- **Combination dose-response**: Models the joint effect of two or more drugs combined at
  various dose levels, including interaction/synergy assessment.
- **Population PK/PD**: Full mechanistic modeling of drug absorption, distribution, and
  effect using compartmental models and NONMEM-type software.

## When to Use This Method

- **Phase II dose-finding trials**: MCP-Mod is the recommended approach for detecting and
  characterizing the dose-response relationship.
- **Dose selection for Phase III**: Use the fitted dose-response model to identify the
  dose(s) that provide a clinically meaningful effect with acceptable tolerability.
- **Regulatory labeling**: Dose-response data support the dosing recommendations in the
  prescribing information.
- **Toxicology / Risk assessment**: BMD analysis is the standard approach for establishing
  safe exposure levels.
- **Biologics and biosimilars**: Exposure-response modeling is increasingly used instead of
  or alongside traditional dose-response analysis.

## Strengths and Limitations

### Strengths
- Model-based approaches borrow information across dose groups, increasing efficiency.
- MCP-Mod provides type I error control while allowing flexible dose-response estimation.
- Model averaging handles uncertainty about the true dose-response shape.
- Smooth curves enable interpolation to untested doses.

### Limitations
- Requires pre-specification of candidate models (though standard sets exist).
- Nonlinear models can be sensitive to starting values and may not converge.
- Limited dose groups (typical 4-6 in Phase II) constrain the complexity of models that
  can be reliably fitted.
- Extrapolation beyond the studied dose range is unreliable.
- Exposure-response analysis requires pharmacokinetic data, adding complexity.

## Key References

1. Bretz F, Pinheiro JC, Branson M. Combining multiple comparisons and modeling techniques
   in dose-response studies. *Biometrics*, 2005;61(3):738-748.
2. Pinheiro JC, Bornkamp B, Glimm E, Bretz F. Model-based dose finding under model
   uncertainty using general parametric models. *Statistics in Medicine*, 2014;33(10):1646-1661.
3. Thomas N, Sweeney K, Somayaji V. Meta-analysis of clinical dose-response in a large
   drug development portfolio. *Statistics in Biopharmaceutical Research*, 2014;6(4):302-317.
4. European Medicines Agency. Qualification opinion of MCP-Mod as an efficient statistical
   methodology for model-based design and analysis of Phase II dose finding studies. EMA, 2014.
5. FDA. Adaptive Designs for Clinical Trials of Drugs and Biologics: Guidance for Industry. 2019.
6. Macdougall J. Analysis of dose-response studies — Emax model. In: Ting N, ed. *Dose
   Finding in Drug Development*. Springer, 2006.
7. Bornkamp B, Bretz F, Dmitrienko A, et al. Innovative approaches for designing and analyzing
   adaptive dose-ranging trials. *Journal of Biopharmaceutical Statistics*, 2007;17(6):965-995.
