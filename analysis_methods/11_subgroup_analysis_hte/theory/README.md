# Subgroup Analysis and Treatment Effect Heterogeneity (HTE) — Theory

## Introduction

Subgroup analysis and treatment effect heterogeneity (HTE) address a fundamental question in
clinical research: does the treatment work differently for different types of patients? While
a randomized controlled trial may demonstrate an overall average treatment effect, this average
can mask meaningful variation across patient populations. Understanding HTE is critical for
personalizing treatment decisions, identifying patients who benefit most (or may be harmed),
and informing regulatory and clinical guidelines.

Historically, subgroup analysis relied on testing treatment-by-covariate interactions within
pre-specified subgroups. Modern methods have expanded the toolkit dramatically, incorporating
machine learning approaches that can discover complex, multidimensional heterogeneity patterns
without relying solely on investigator-defined subgroups.

## Mathematical Foundation

### Average Treatment Effect (ATE)

The ATE is defined as:

```
ATE = E[Y(1) - Y(0)]
```

where Y(1) and Y(0) are potential outcomes under treatment and control, respectively.

### Conditional Average Treatment Effect (CATE)

The CATE conditions on covariates X:

```
CATE(x) = E[Y(1) - Y(0) | X = x] = tau(x)
```

This function tau(x) is the primary target of HTE analysis. Estimating tau(x) allows us
to understand how the treatment effect varies as a function of patient characteristics.

### Interaction Tests (Classical Subgroup Analysis)

In a regression framework for subgroup S (binary):

```
Y = beta_0 + beta_1 * T + beta_2 * S + beta_3 * (T x S) + epsilon
```

The coefficient beta_3 represents the difference in treatment effect between subgroups.
A test of H0: beta_3 = 0 assesses whether the treatment effect differs across levels of S.

### Scale Dependence

HTE depends on the scale of measurement. On the risk difference (RD) scale:

```
HTE_RD(x) = P(Y=1|T=1,X=x) - P(Y=1|T=0,X=x)
```

On the relative risk (RR) scale:

```
HTE_RR(x) = P(Y=1|T=1,X=x) / P(Y=1|T=0,X=x)
```

A treatment with a constant relative risk will show heterogeneity on the risk difference scale
if baseline risk varies. This scale dependence has deep implications for interpreting HTE.

## Key Concepts

### Prognostic vs Predictive Factors

- **Prognostic factors** affect outcome regardless of treatment. A patient with advanced disease
  has worse prognosis whether treated or not.
- **Predictive factors** modify the treatment effect specifically. A biomarker is predictive if
  the treatment works in biomarker-positive patients but not in biomarker-negative patients.

This distinction is central. Prognostic factors drive risk-based HTE on the absolute scale
but do not indicate differential treatment benefit on the relative scale.

### PATH Statement (Predictive Approaches to Treatment Effect Heterogeneity)

The PATH statement (Kent et al., 2020) provides a framework for credible HTE analysis:

1. Use a risk-modeling approach as the primary HTE analysis.
2. Evaluate treatment effect variation across risk strata.
3. Effect-modification analyses should be pre-specified and limited.
4. Present both relative and absolute treatment effects.

### Meta-Learners for CATE Estimation

- **T-Learner**: Fit separate outcome models for treated and control groups. CATE = mu_1(x) - mu_0(x).
- **S-Learner**: Fit a single model including treatment as a feature. CATE = mu(x, T=1) - mu(x, T=0).
- **X-Learner** (Kunzel et al., 2019): A two-stage approach that imputes individual treatment effects
  and then fits models on these pseudo-outcomes, weighted by propensity scores.
- **R-Learner** (Nie and Wager, 2021): Uses a Robinson decomposition to directly target CATE by
  minimizing a loss function involving residualized outcomes and residualized treatment.

### Causal Forests

Causal forests (Wager and Athey, 2018) extend random forests to estimate CATE. Key features:

- Honesty: Separate samples for building the tree structure and estimating leaf effects.
- Asymptotic normality: Valid confidence intervals for CATE estimates.
- Built-in variable importance for identifying key effect modifiers.

### BART for Causal Inference

Bayesian Additive Regression Trees (BART) can estimate CATE by fitting flexible nonparametric
models. The `bartCause` framework provides posterior distributions for individual treatment
effects, enabling uncertainty quantification.

### Virtual Twins

The Virtual Twins method (Foster et al., 2011):

1. Fit a flexible model (e.g., random forest) to estimate E[Y|X,T].
2. Predict individual treatment effects: d(x) = mu_hat(x,1) - mu_hat(x,0).
3. Use a simple tree on d(x) to identify subgroups with enhanced treatment effects.

## Assumptions

1. **Unconfoundedness / Ignorability**: In RCTs this holds by design. In observational studies,
   all confounders must be measured (strong assumption).
2. **SUTVA (Stable Unit Treatment Value Assumption)**: No interference between subjects and
   a single version of each treatment level.
3. **Overlap / Positivity**: All subgroups must have a non-trivial probability of receiving
   each treatment. Violations lead to extreme weights and unreliable estimates.
4. **Correct model specification** (for parametric approaches): Interaction tests assume the
   correct functional form for the interaction.

## Variants and Extensions

- **Optimal Treatment Regimes (OTR)**: Use estimated CATE to assign patients to the treatment
  that maximizes their expected outcome. Methods include outcome-weighted learning and
  Q-learning.
- **Bayesian subgroup identification**: Bayesian approaches such as BAFT (Bayesian Additive
  Fixed Trees) formally incorporate uncertainty.
- **Policy learning**: Directly optimize a treatment policy rather than first estimating CATE.
  The `policytree` package implements this approach.
- **Multi-arm treatment settings**: Extend CATE estimation to compare multiple treatments
  simultaneously using generalized causal forests.

## When to Use This Method

- **Confirmatory trials**: Pre-specified subgroup analyses with interaction tests. Use forest
  plots for visual presentation. Keep the number of subgroups small and pre-registered.
- **Exploratory analyses**: Causal forests, BART, or meta-learners for data-driven HTE discovery
  when sample size is large (n > 500 typically needed).
- **Risk-based HTE**: When you suspect the treatment effect varies by baseline risk rather
  than by any single biomarker. Use the PATH framework.
- **Precision medicine / Biomarker-driven trials**: When you have candidate predictive biomarkers
  and want to identify an enriched population.

## Strengths and Limitations

### Strengths
- Classical subgroup analysis is simple and well-understood by regulators.
- Modern ML methods can discover complex heterogeneity patterns.
- Causal forests provide valid confidence intervals.
- Risk-based HTE is robust and clinically interpretable.

### Limitations
- Classical subgroup analysis is underpowered (interaction tests need ~4x the sample size of main effects).
- Multiple subgroup tests inflate type I error without multiplicity correction.
- ML-based methods require large samples and careful validation.
- Results on different scales (RD vs RR) may lead to different conclusions.
- Exploratory HTE findings require independent replication.

## Key References

1. Kent DM, Paulus JK, van Klaveren D, et al. The Predictive Approaches to Treatment effect
   Heterogeneity (PATH) statement. *Annals of Internal Medicine*, 2020;172(1):35-45.
2. Wager S, Athey S. Estimation and inference of heterogeneous treatment effects using random
   forests. *Journal of the American Statistical Association*, 2018;113(523):1228-1242.
3. Kunzel SR, Sekhon JS, Bickel PJ, Yu B. Metalearners for estimating heterogeneous treatment
   effects using machine learning. *PNAS*, 2019;116(10):4156-4165.
4. Nie X, Wager S. Quasi-oracle estimation of heterogeneous treatment effects. *Biometrika*,
   2021;108(2):299-319.
5. Foster JC, Taylor JMG, Ruberg SJ. Subgroup identification from randomized clinical trial
   data. *Statistics in Medicine*, 2011;30(24):2867-2880.
6. Lipkovich I, Dmitrienko A, D'Agostino RB. Tutorial in biostatistics: data-driven subgroup
   identification and analysis in clinical trials. *Statistics in Medicine*, 2017;36(1):136-196.
7. Hill JL. Bayesian nonparametric modeling for causal inference. *Journal of Computational and
   Graphical Statistics*, 2011;20(1):217-240.
