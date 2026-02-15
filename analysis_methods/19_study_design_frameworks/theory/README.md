# Study Design Frameworks — Theory

## Introduction

Real-world evidence (RWE) studies using observational data are vulnerable to systematic
biases that do not arise in randomised trials. Study design frameworks provide structured
approaches to minimise these biases by emulating the features of a well-designed trial.
This chapter covers target trial emulation, new-user designs, time-related biases,
common data models, and the ICH E9(R1) estimand framework that formalises what a study
aims to estimate in the presence of intercurrent events.

## Mathematical Foundation

### Target Trial Emulation (Hernan and Robins)

The target trial emulation framework asks: "What randomised trial would we like to
conduct?" and then designs the observational analysis to emulate each component of
that trial. The seven components to specify are:

| Component | Target Trial | Observational Emulation |
|-----------|-------------|------------------------|
| **Eligibility** | Inclusion/exclusion criteria | Apply same criteria at time zero |
| **Treatment strategies** | Randomised arms | Defined treatment sequences |
| **Assignment** | Random | Adjust for confounding (IP weighting, etc.) |
| **Start of follow-up** | Randomisation date | Align time zero with treatment initiation |
| **Outcome** | Well-defined endpoint | Same definition |
| **Causal contrast** | ITT, per-protocol | ITT (as-started), per-protocol (IP-weighted) |
| **Analysis plan** | Pre-specified | Pre-specified (to the extent possible) |

The critical insight is that **misalignment of eligibility, treatment assignment, and
start of follow-up introduces immortal time bias and other biases that do not exist in
the target trial.**

### Per-Protocol Effect via Inverse Probability Weighting

To estimate the per-protocol effect (what happens if everyone adheres), we censor
patients at the time they deviate from their assigned strategy and apply inverse
probability of censoring weights (IPCW):

```
W_i(t) = 1 / P(Uncensored at t | Past treatment and covariates)
```

This creates a pseudo-population in which censoring due to non-adherence is
non-informative.

### New-User (Incident-User) Design

The new-user design restricts the study population to patients initiating (not
continuing) a treatment during the study period. This avoids:

1. **Prevalent-user bias:** Patients who have been on treatment for a long time are
   survivors; including them selects for tolerators and responders.
2. **Depletion of susceptibles:** Early adverse events have already occurred in
   prevalent users, diluting the apparent risk.
3. **Confounding by indication changes:** The reasons for starting treatment differ
   from the reasons for continuing.

The active-comparator new-user (ACNU) design further restricts to patients initiating
one of two active treatments, which:
- Ensures exchangeability at baseline (both groups have a reason to start treatment).
- Defines a clear time zero (date of treatment initiation).
- Allows a valid per-protocol analysis.

## Key Concepts

### Immortal Time Bias

Immortal time is a span of follow-up during which the outcome cannot occur by design.
It arises when:

- **Misclassified immortal time:** Time between cohort entry and treatment start is
  counted as exposed person-time (but the patient could not have died before starting
  treatment).
- **Time-zero misalignment:** Eligibility is assessed at one time, but follow-up starts
  at another.

The bias inflates the apparent protective effect of treatment because immortal time
artificially lengthens the treated group's denominator.

**Solution:** Align time zero with treatment initiation (new-user design) or use
time-varying treatment analysis.

### Other Time-Related Biases

| Bias | Description | Solution |
|------|-------------|----------|
| **Immortal time bias** | See above | New-user design, align time zero |
| **Time-window bias** | Differential opportunity to be classified as exposed | Match on available follow-up time |
| **Time-lag bias** | Different disease stages at treatment initiation | Restrict to comparable disease stage |
| **Immeasurable time bias** | Treatment data unavailable during hospitalisation | Exclude or account for hospital stays |

### ICH E9(R1) Estimand Framework

The ICH E9(R1) addendum (2019) defines an **estimand** as the precise description of
the treatment effect that a trial aims to estimate. It has five attributes:

1. **Population:** Who is the target population?
2. **Treatment:** What treatment strategies are compared?
3. **Endpoint:** What is the outcome variable?
4. **Population-level summary:** What summary measure (mean difference, hazard ratio)?
5. **Intercurrent events (ICEs):** How are events like treatment discontinuation,
   rescue medication, or death handled?

### Strategies for Intercurrent Events

| Strategy | Description | Estimand Question |
|----------|-------------|-------------------|
| **Treatment policy** | Include all data regardless of ICE | "What is the effect of being assigned to treatment?" |
| **Composite** | Incorporate ICE into the outcome | "What is the risk of outcome or ICE?" |
| **Hypothetical** | Estimate effect in a world where ICE does not occur | "What would happen if no one switched?" |
| **Principal stratum** | Restrict to patients who would (not) experience ICE | "What is the effect among adherers?" |
| **While on treatment** | Use data only while patient is on treatment | "What is the effect while on treatment?" |

## Assumptions

1. **No unmeasured confounding:** Exchangeability conditional on measured covariates.
2. **Positivity:** Every covariate stratum has a non-zero probability of each treatment.
3. **Consistency:** The observed outcome under the observed treatment equals the
   potential outcome under that treatment.
4. **Correct model specification:** For parametric adjustments (propensity scores, IPCW).
5. **No model misspecification in time-varying confounding adjustment** when using MSMs
   or g-estimation.

### OMOP Common Data Model

The Observational Medical Outcomes Partnership (OMOP) CDM standardises the structure
and vocabulary of health databases, enabling multi-site studies. Key tables include:

- `person`, `condition_occurrence`, `drug_exposure`, `procedure_occurrence`,
  `measurement`, `observation_period`, `visit_occurrence`.
- Standardised vocabularies (SNOMED, RxNorm, LOINC, ICD) allow cross-database queries.

OHDSI (Observational Health Data Sciences and Informatics) provides open-source tools
(ATLAS, CohortDiagnostics, CohortMethod) built on OMOP.

### Sentinel System

The FDA Sentinel System uses a distributed data network of electronic health records and
claims databases to conduct post-market safety surveillance. It uses a common data model
and executes standardised queries across sites without pooling patient-level data.

### STaRT-RWE Template

The STaRT-RWE (Structured Template for planning and Reporting on the implementation of
Real World Evidence studies) provides a checklist for designing and reporting RWE
studies. It covers:

- Research question and target trial specification.
- Data source and eligibility criteria.
- Exposure, comparator, and outcome definitions.
- Confounder identification and adjustment strategy.
- Sensitivity analyses (quantitative bias analysis).

## When to Use This Method

- Whenever designing an observational comparative effectiveness or safety study.
- When regulatory submissions require RWE (e.g., label extensions, post-market
  commitments).
- For multi-database studies using OMOP or Sentinel infrastructure.
- When defining estimands for clinical trial SAPs under ICH E9(R1).
- For health technology assessment submissions requiring RWE.

## Strengths and Limitations

### Strengths
- Target trial emulation provides a principled, transparent design framework.
- The new-user design eliminates prevalent-user bias and defines a clear time zero.
- The estimand framework forces explicit decisions about intercurrent events.
- OMOP/OHDSI enables reproducible, multi-site studies at scale.

### Limitations
- Unmeasured confounding is always possible in observational data.
- The per-protocol estimand requires IPCW, which is sensitive to model misspecification.
- Target trial emulation does not eliminate all bias — it structures the analysis to
  minimise known biases.
- Estimand specification can be complex for trials with multiple ICEs.

## Key References

1. Hernan, M. A., & Robins, J. M. (2016). Using big data to emulate a target trial when
   a randomized trial is not available. *American Journal of Epidemiology*, 183(8),
   758-764.
2. Ray, W. A. (2003). Evaluating medication effects outside of clinical trials: new-user
   designs. *American Journal of Epidemiology*, 158(9), 915-920.
3. Suissa, S. (2008). Immortal time bias in pharmacoepidemiology. *American Journal of
   Epidemiology*, 167(4), 492-499.
4. ICH E9(R1) (2019). Addendum on estimands and sensitivity analysis in clinical trials.
5. Hripcsak, G., et al. (2015). Observational Health Data Sciences and Informatics
   (OHDSI). *JAMIA*, 22(2), 260-263.
6. Wang, S. V., et al. (2021). STaRT-RWE: structured template for planning and reporting
   on the implementation of real-world evidence studies. *BMJ*, 372, m4856.
7. Lund, J. L., et al. (2015). The active comparator, new user study design in
   pharmacoepidemiology. *Pharmacoepidemiology and Drug Safety*, 24(5), 459-467.
