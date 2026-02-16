# Causal Inference — Theory

## Introduction

Causal inference is the discipline of drawing conclusions about cause-and-effect relationships from data. In health research, the fundamental question is often: "What is the effect of treatment A compared to treatment B on patient outcome Y?" While randomized controlled trials (RCTs) are the gold standard for causal claims, observational data from electronic health records (EHRs), registries, and claims databases are increasingly used for causal questions where RCTs are infeasible, unethical, or insufficient.

Causal inference methods provide the statistical framework to estimate treatment effects from non-randomized data by addressing confounding, the systematic differences between treated and untreated groups that distort naive comparisons. These methods are essential in pharmacoepidemiology, comparative effectiveness research, health policy evaluation, and Mendelian randomization studies.

## Mathematical Foundation

### Potential Outcomes Framework (Rubin Causal Model)

For each subject i, define potential outcomes Y_i(1) under treatment and Y_i(0) under control. The **individual causal effect** is:

```
tau_i = Y_i(1) - Y_i(0)
```

The **fundamental problem of causal inference** is that we observe only one potential outcome for each subject. We define population-level estimands:

**Average Treatment Effect (ATE)**:
```
ATE = E[Y(1) - Y(0)] = E[Y(1)] - E[Y(0)]
```

**Average Treatment Effect on the Treated (ATT)**:
```
ATT = E[Y(1) - Y(0) | A = 1]
```

**Average Treatment Effect on the Untreated (ATU)**:
```
ATU = E[Y(1) - Y(0) | A = 0]
```

where A denotes treatment assignment.

### Directed Acyclic Graphs (DAGs) — Pearl's Framework

DAGs encode causal relationships as directed edges between variables. Key concepts:

- **Confounder**: A common cause of treatment and outcome (backdoor path).
- **Mediator**: A variable on the causal path from treatment to outcome.
- **Collider**: A common effect of two variables; conditioning on a collider opens a spurious association.

The **backdoor criterion** states that adjustment for a set Z of covariates identifies the causal effect if Z blocks all backdoor paths from A to Y and Z contains no descendants of A.

The **do-calculus** formalizes interventional distributions:
```
P(Y | do(A = a)) != P(Y | A = a)  in general
```

The interventional distribution removes confounding and represents the outcome distribution if we were to intervene and set treatment to a.

### Propensity Score

The propensity score (Rosenbaum and Rubin, 1983) is the probability of receiving treatment given observed confounders:

```
e(X) = P(A = 1 | X)
```

Under the assumption of no unmeasured confounding, the propensity score is a **balancing score**: conditional on e(X), treatment assignment is independent of potential outcomes. This reduces the dimensionality of confounding adjustment from p covariates to a single scalar.

### Inverse Probability of Treatment Weighting (IPTW)

IPTW creates a pseudo-population where treatment is independent of confounders by weighting each observation:

```
w_i = A_i / e(X_i) + (1 - A_i) / (1 - e(X_i))
```

The ATE is estimated by:
```
ATE_hat = (1/n) * sum [A_i * Y_i / e(X_i) - (1 - A_i) * Y_i / (1 - e(X_i))]
```

Stabilized weights reduce variance:
```
sw_i = A_i * P(A=1) / e(X_i) + (1 - A_i) * P(A=0) / (1 - e(X_i))
```

### G-Computation (Standardization)

G-computation (Robins, 1986) estimates causal effects by modeling the outcome as a function of treatment and confounders, then averaging predictions over the confounder distribution:

```
ATE = (1/n) * sum [E_hat(Y | A=1, X_i) - E_hat(Y | A=0, X_i)]
```

This approach requires correct specification of the outcome model.

### Doubly Robust Estimation

**Augmented Inverse Probability Weighting (AIPW)** combines IPTW and outcome modeling:

```
ATE_hat = (1/n) * sum { [A_i * Y_i / e(X_i) - (A_i - e(X_i)) / e(X_i) * m_1(X_i)]
                       - [(1-A_i) * Y_i / (1-e(X_i)) + (A_i - e(X_i)) / (1-e(X_i)) * m_0(X_i)] }
```

where m_a(X) = E(Y | A=a, X). The AIPW estimator is consistent if **either** the propensity score model or the outcome model is correctly specified (but not necessarily both). This "double robustness" provides protection against model misspecification.

**Targeted Maximum Likelihood Estimation (TMLE)** is an alternative doubly robust approach that:
1. Fits an initial outcome model.
2. Computes a clever covariate based on the propensity score.
3. Updates the outcome model using the clever covariate via a targeting step.
4. Computes the ATE from the updated model.

TMLE respects the parameter space (e.g., probabilities stay between 0 and 1) and has favorable finite-sample properties.

### Instrumental Variables (IV)

When unmeasured confounding is present, instrumental variables provide causal identification. An instrument Z must satisfy:
1. **Relevance**: Z is associated with treatment A.
2. **Exclusion restriction**: Z affects outcome Y only through A.
3. **Independence**: Z is independent of unmeasured confounders.

**Two-Stage Least Squares (2SLS)**:
- Stage 1: Regress A on Z and X to get predicted A_hat.
- Stage 2: Regress Y on A_hat and X.

The IV estimate identifies the **Local Average Treatment Effect (LATE)** for compliers.

**Mendelian Randomization** uses genetic variants as instruments for modifiable risk factors, leveraging the random allocation of alleles at conception to estimate causal effects of exposures (e.g., BMI, cholesterol levels) on disease outcomes.

## Key Concepts

### Identification Assumptions

Three core assumptions for causal identification from observational data:

1. **Exchangeability (No unmeasured confounding)**: Y(a) independent of A | X. Treatment groups are comparable after conditioning on X.
2. **Positivity**: 0 < P(A = 1 | X) < 1 for all X. Every subject has a non-zero probability of receiving either treatment.
3. **Consistency**: Y_i = Y_i(a) when A_i = a. The observed outcome equals the potential outcome under the received treatment.

### Target Trial Emulation

Hernan and Robins proposed that observational studies should emulate a hypothetical target trial. This framework requires specifying:
- Eligibility criteria
- Treatment strategies
- Assignment procedures
- Follow-up period
- Outcome
- Causal contrast (estimand)
- Statistical analysis plan

Each component of the target trial is mapped to the observational data, clarifying assumptions and reducing bias.

### Overlap and Positivity

Positivity violations occur when certain covariate patterns make treatment assignment nearly deterministic. Methods to address this include:
- **Trimming**: Excluding subjects with extreme propensity scores.
- **Overlap weighting**: Using weights proportional to the probability of being in the opposite treatment group: w_i = 1 - e(X_i) for treated, e(X_i) for controls.

## Assumptions

1. **No unmeasured confounding (exchangeability)**: All variables affecting both treatment and outcome are measured and included.
2. **Positivity**: Every combination of confounders has a positive probability of receiving each treatment.
3. **Consistency**: Well-defined treatments with no ambiguity about what "receiving treatment" means.
4. **No interference (SUTVA)**: One subject's treatment does not affect another's outcome.
5. **Correct model specification**: For parametric methods, the models for propensity score and/or outcome must be correctly specified (doubly robust methods relax this partially).

## Variants and Extensions

### Propensity Score Matching

Matching on the propensity score pairs treated and control subjects with similar treatment probabilities. Methods include nearest-neighbor matching (with or without replacement), caliper matching, optimal matching, and full matching.

### Propensity Score Stratification

Dividing subjects into strata (typically quintiles) based on the propensity score and estimating the treatment effect within each stratum. The overall effect is a weighted average across strata.

### Overlap Weighting

A recent alternative to IPTW that naturally handles positivity issues by down-weighting subjects with extreme propensity scores. It targets the **Average Treatment Effect for the Overlap population (ATO)**.

### Difference-in-Differences

For panel data, DiD exploits the parallel trends assumption to estimate causal effects when treatment timing varies across groups.

### Regression Discontinuity

When treatment assignment is determined by a cutoff on a continuous variable, regression discontinuity designs estimate the local causal effect at the cutoff.

## When to Use This Method

- **Use propensity score methods**: When treatment assignment depends on observed baseline confounders and you want to emulate a randomized comparison.
- **Use IPTW**: For ATE estimation with time-varying confounding or marginal structural models.
- **Use matching**: When intuitive pairwise comparison is desired and the matched sample is the target population.
- **Use doubly robust methods**: When you want protection against model misspecification (recommended as default).
- **Use IV/Mendelian randomization**: When unmeasured confounding is a serious concern and valid instruments are available.
- **Use target trial emulation**: As a framework for any observational causal study to clarify design choices.

## Strengths and Limitations

### Strengths
- Enables causal conclusions from observational data under stated assumptions.
- Multiple complementary approaches provide robustness through triangulation.
- Doubly robust methods protect against single-model misspecification.
- Well-established theoretical foundations (Rubin, Pearl, Robins).
- Growing regulatory acceptance for real-world evidence.

### Limitations
- No unmeasured confounding is untestable and often implausible in practice.
- Positivity violations can lead to extreme weights and unstable estimates.
- Results are sensitive to the choice of confounders (DAG specification).
- Instrumental variables require strong and often unverifiable exclusion restrictions.
- Complex methods may be difficult to communicate to clinical audiences.
- Causal interpretation requires domain expertise, not just statistical technique.

## Key References

1. Rubin DB. Estimating causal effects of treatments in randomized and nonrandomized studies. *J Educ Psychol.* 1974;66(5):688-701.
2. Rosenbaum PR, Rubin DB. The central role of the propensity score in observational studies for causal effects. *Biometrika.* 1983;70(1):41-55.
3. Pearl J. *Causality: Models, Reasoning, and Inference.* 2nd ed. Cambridge University Press; 2009.
4. Hernan MA, Robins JM. *Causal Inference: What If.* Chapman & Hall/CRC; 2020.
5. Bang H, Robins JM. Doubly robust estimation in missing data and causal inference models. *Biometrics.* 2005;61(4):962-973.
6. van der Laan MJ, Rose S. *Targeted Learning.* Springer; 2011.
7. Hernan MA, Robins JM. Using big data to emulate a target trial when a randomized trial is not available. *Am J Epidemiol.* 2016;183(8):758-764.
8. Davey Smith G, Ebrahim S. 'Mendelian randomization': can genetic epidemiology contribute to understanding environmental determinants of disease? *Int J Epidemiol.* 2003;32(1):1-22.
