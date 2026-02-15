# Causal Mediation Analysis — Theory

## Introduction

Causal mediation analysis decomposes the total effect of a treatment (or exposure) on an
outcome into an **indirect effect** operating through one or more mediators and a **direct
effect** that captures all remaining pathways. In clinical research the canonical question
is: "Does drug X improve the outcome partly *because* it changes biomarker M?"

The field has evolved from the classical Baron and Kenny (1986) regression-based approach
to a modern counterfactual (potential-outcomes) framework that accommodates
treatment-mediator interactions, nonlinear models, and survival endpoints.

## Mathematical Foundation

### Classical Baron and Kenny Approach

Three regressions are fit:

1. **Total effect:** `Y = c0 + c*T + e1`
2. **Mediator model:** `M = a0 + a*T + e2`
3. **Outcome model:** `Y = b0 + c'*T + b*M + e3`

- **Indirect effect (IE)** = `a * b` (product of coefficients) or equivalently `c - c'`.
- **Direct effect (DE)** = `c'`.
- **Total effect (TE)** = `c = c' + a*b`.

Significance of mediation was originally assessed by the Sobel test:

```
z = (a * b) / sqrt(b^2 * se_a^2 + a^2 * se_b^2)
```

Modern practice replaces this with bootstrap confidence intervals because the product
`a * b` is not normally distributed in finite samples.

### Counterfactual (Potential-Outcomes) Framework

Let `Y(t, m)` denote the potential outcome when treatment is set to `t` and mediator to
`m`, and `M(t)` the potential mediator value under treatment `t`. With binary treatment
(T = 1 vs. 0):

| Quantity | Definition |
|---|---|
| **Total Effect (TE)** | `E[Y(1, M(1))] - E[Y(0, M(0))]` |
| **Natural Direct Effect (NDE)** | `E[Y(1, M(0))] - E[Y(0, M(0))]` |
| **Natural Indirect Effect (NIE)** | `E[Y(1, M(1))] - E[Y(1, M(0))]` |
| **Controlled Direct Effect (CDE(m))** | `E[Y(1, m)] - E[Y(0, m)]` for a fixed `m` |

The decomposition `TE = NDE + NIE` holds on the difference scale.

### VanderWeele 4-Way Decomposition

When treatment-mediator interaction is present, VanderWeele (2014) decomposes TE into
four non-overlapping components:

1. **CDE** — controlled direct effect (neither mediation nor interaction).
2. **INTref** — reference interaction (interaction only).
3. **INTmed** — mediated interaction (both mediation and interaction).
4. **PIE** — pure indirect effect (mediation only).

```
TE = CDE + INTref + INTmed + PIE
```

This reveals whether the treatment works through the mediator, through interaction with
the mediator, or through a pure direct pathway.

## Key Concepts

### Sequential Ignorability Assumption

Identification of natural direct and indirect effects requires two untestable conditions
(Imai, Keele, and Tingley 2010):

1. **Conditional on confounders C, treatment T is independent of all potential outcomes
   and potential mediators.**
   `{Y(t', m), M(t)} _||_ T | C`

2. **Conditional on confounders C and observed treatment T, the mediator M is independent
   of potential outcomes.**
   `Y(t', m) _||_ M | T = t, C`

The second assumption is particularly demanding because it rules out unmeasured
confounders of the M-Y relationship that are themselves affected by T (post-treatment
confounding).

### Sensitivity Analysis for Unmeasured Confounding

Because sequential ignorability is untestable, Imai et al. propose a sensitivity
parameter rho — the correlation between the error terms of the mediator and outcome
models — and examine how the indirect effect changes as rho departs from zero.

- At `rho = 0`, sequential ignorability holds exactly.
- The analyst reports the value of rho at which the indirect effect crosses zero
  (the "sensitivity point").

### Exposure-Mediator Interaction

The counterfactual framework naturally handles interaction. In a linear model:

```
E[Y | T, M, C] = beta_0 + beta_1*T + beta_2*M + beta_3*T*M + beta_4*C
```

- `NDE = beta_1 + beta_3 * E[M(0) | C]`
- `NIE = (beta_2 + beta_3 * 1) * (E[M(1)|C] - E[M(0)|C])`

Without interaction (beta_3 = 0), the counterfactual NDE and NIE reduce to the Baron
and Kenny quantities.

## Assumptions

1. No unmeasured treatment-outcome confounding (given C).
2. No unmeasured mediator-outcome confounding (given T and C).
3. No confounders of the mediator-outcome relationship that are affected by treatment
   (the "cross-world" assumption).
4. Correct specification of the mediator and outcome models.
5. Temporal ordering: T precedes M precedes Y.
6. Consistency and SUTVA (stable unit treatment value assumption).

## Variants and Extensions

### Multiple Mediators

When several mediators operate simultaneously, options include:

- **Joint mediation:** treat the vector of mediators as a single block.
- **Path-specific effects:** decompose the indirect effect through each mediator
  separately (requires additional no-confounding assumptions across mediators).
- **Sequential mediation:** M1 -> M2 -> Y, with path-specific identification.

### Mediation with Survival Outcomes

For time-to-event outcomes, the NDE and NIE can be defined on the hazard-ratio or
survival-probability scale. Approaches include:

- Accelerated failure time models (natural effect decomposition on the log-time scale).
- Counterfactual hazard-based decomposition (VanderWeele 2011).
- Inverse-odds-ratio weighting (Tchetgen Tchetgen 2013).

### Mediation with Binary or Count Outcomes

When outcome or mediator models are nonlinear (logistic, Poisson), the product-of-
coefficients and difference-of-coefficients methods no longer coincide. The
counterfactual framework provides model-agnostic definitions, and estimation proceeds
via simulation (quasi-Bayesian) or analytical formulas for specific model combinations.

## When to Use This Method

- **Mechanism investigation:** Understanding *how* or *why* a treatment works.
- **Surrogate endpoint evaluation:** Assessing whether a biomarker captures the
  treatment effect.
- **Intervention design:** Deciding whether to target the mediator directly.
- **Regulatory submissions:** FDA and EMA increasingly request mediation evidence for
  surrogate endpoints.

## Strengths and Limitations

### Strengths
- Provides mechanistic insight beyond "does it work?"
- The counterfactual framework handles nonlinearities and interactions.
- Sensitivity analysis quantifies robustness to unmeasured confounding.
- The 4-way decomposition separates mediation from interaction.

### Limitations
- Sequential ignorability is strong and untestable.
- Post-treatment confounding is common and invalidates standard approaches.
- Multiple mediators require additional, often implausible, assumptions.
- Results can be sensitive to model specification.
- With observational data, causal claims require careful justification.

## Key References

1. Baron, R. M., & Kenny, D. A. (1986). The moderator-mediator variable distinction.
   *Journal of Personality and Social Psychology*, 51(6), 1173-1182.
2. Imai, K., Keele, L., & Tingley, D. (2010). A general approach to causal mediation
   analysis. *Psychological Methods*, 15(4), 309-334.
3. VanderWeele, T. J. (2014). A unification of mediation and interaction: a 4-way
   decomposition. *Epidemiology*, 25(5), 749-761.
4. VanderWeele, T. J. (2015). *Explanation in Causal Inference: Methods for Mediation
   and Interaction*. Oxford University Press.
5. Robins, J. M., & Greenland, S. (1992). Identifiability and exchangeability for
   direct and indirect effects. *Epidemiology*, 3(2), 143-155.
6. Tchetgen Tchetgen, E. J. (2013). Inverse odds ratio-weighted estimation for causal
   mediation analysis. *Statistics in Medicine*, 32(26), 4567-4580.
7. Shi, B., et al. (2021). CMAverse: a suite of functions for causal mediation analysis.
   *Epidemiology*, 32(5), e20-e22.
