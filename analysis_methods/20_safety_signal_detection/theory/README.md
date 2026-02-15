# Safety Signal Detection — Theory

## Introduction

Safety signal detection is the systematic process of identifying previously unrecognised
causal associations between medications and adverse events. It is a cornerstone of
pharmacovigilance, operating across spontaneous reporting systems (FAERS, EudraVigilance,
VigiBase), electronic health records, claims databases, and clinical trial safety data.

A **safety signal** is defined (CIOMS VIII) as "information that arises from one or
multiple sources which suggests a new potentially causal association, or a new aspect of
a known association, between an intervention and an event." Signal detection methods
range from simple disproportionality measures to sophisticated Bayesian and sequential
surveillance techniques.

## Mathematical Foundation

### Disproportionality Analysis

Disproportionality methods compare the observed frequency of a drug-event combination
to what would be expected if drug and event were independent. The starting point is the
2x2 contingency table:

```
                   Event of Interest    All Other Events
Drug of Interest         a                    b
All Other Drugs          c                    d
```

where `a + b + c + d = N` (total reports in the database).

#### Proportional Reporting Ratio (PRR)

```
PRR = [a / (a + b)] / [c / (c + d)]
```

A PRR > 2 with chi-squared > 4 and a >= 3 is the traditional Evans signal threshold.

#### Reporting Odds Ratio (ROR)

```
ROR = (a * d) / (b * c)
```

The ROR is the odds ratio from the 2x2 table and can be accompanied by a 95% CI:

```
ln(ROR) +/- 1.96 * sqrt(1/a + 1/b + 1/c + 1/d)
```

A signal is flagged when the lower 95% CI of the ROR exceeds 1.

#### Bayesian Confidence Propagation Neural Network (BCPNN / IC)

The **Information Component (IC)** is the log2 of the ratio of observed to expected:

```
IC = log2(P(drug, event) / (P(drug) * P(event)))
```

The Bayesian version (DuMouchel and BCPNN) uses a Beta distribution prior to shrink
estimates, yielding `IC_025` (the lower 2.5% credible bound). A signal is flagged when
`IC_025 > 0`. The WHO Uppsala Monitoring Centre uses IC in VigiBase.

#### Multi-item Gamma Poisson Shrinker (MGPS / EBGM)

The MGPS (DuMouchel 1999) models the observed-to-expected ratio with a mixture of two
gamma distributions (empirical Bayes):

```
E[lambda | a, E] = (a + prior_mean) / (E + prior_weight)
```

The **Empirical Bayes Geometric Mean (EBGM)** is the posterior mean of the true
reporting ratio. The lower 5th percentile (**EB05**) serves as the signal threshold:

- `EB05 >= 2` is the FDA's traditional signal criterion in FAERS.

### Sequential Surveillance Methods

#### MaxSPRT (Maximised Sequential Probability Ratio Test)

For prospective safety surveillance (e.g., Vaccine Safety Datalink, Sentinel), the
MaxSPRT tests whether the observed number of events exceeds what is expected under H0
at each sequential look:

```
LR(t) = max_{lambda >= 1} [exp(-E(t)(lambda - 1)) * lambda^{C(t)}]
```

where `C(t)` is the cumulative count and `E(t)` is the expected count under H0. The
test rejects when `LR(t) >= CV`, where the critical value CV is chosen to control the
overall alpha.

MaxSPRT is exact for Poisson data and handles continuous or group sequential monitoring.

### Self-Controlled Designs

#### Self-Controlled Case Series (SCCS)

The SCCS compares event rates within risk windows (after exposure) to baseline windows
(before or long after exposure) within the same individual, inherently controlling for
all time-fixed confounders:

```
P(event in risk window) / P(event in baseline window) = exp(beta)
```

The relative incidence (RI) is estimated via conditional Poisson regression. The SCCS
is valid when:
- The event does not influence subsequent exposures.
- The event is recurrent or rare (so it does not appreciably alter the observation
  period).

#### Case-Crossover Design

Each case serves as their own control. Exposure during a hazard period (just before
the event) is compared to exposure during one or more control periods. Matched-pair
odds ratios are computed using conditional logistic regression. This is suited to
transient exposures and acute outcomes.

## Key Concepts

### Spontaneous Reporting Systems

| System | Geography | Organisation |
|--------|-----------|-------------|
| **FAERS** | United States | FDA |
| **EudraVigilance** | European Union | EMA |
| **VigiBase** | Global | WHO Uppsala Monitoring Centre |
| **JADER** | Japan | PMDA |
| **DAEN** | Australia | TGA |

Limitations of spontaneous reports: under-reporting (estimated 1-10% of AEs), reporting
biases (stimulated reporting, notoriety bias, Weber effect), no denominator (exposed
population size unknown), duplicate reports.

### Hy's Law and eDISH

**Hy's Law** (named after Hyman Zimmerman) identifies drugs with potential to cause
severe drug-induced liver injury (DILI). The criteria are:

1. ALT or AST >= 3x ULN (Upper Limit of Normal).
2. Total bilirubin >= 2x ULN.
3. No other explanation (e.g., biliary obstruction, viral hepatitis).

The **eDISH (evaluation of Drug-Induced Serious Hepatotoxicity)** plot is a scatter
plot of peak ALT/ULN (x-axis) vs. peak bilirubin/ULN (y-axis), with quadrants:

| Quadrant | ALT | Bilirubin | Interpretation |
|----------|-----|-----------|----------------|
| Normal | < 3x ULN | < 2x ULN | No liver signal |
| Temple's corollary | >= 3x ULN | < 2x ULN | Hepatocellular, usually reversible |
| Cholestatic | < 3x ULN | >= 2x ULN | Cholestatic or non-hepatic cause |
| **Hy's Law** | >= 3x ULN | >= 2x ULN | Potential severe DILI |

### OHDSI LEGEND Framework

The LEGEND (Large-scale Evidence Generation and Evaluation across a Network of
Databases) framework applies systematic, standardised methods to every drug-outcome
pair across multiple databases. Key features:

- Negative control outcomes (outcomes known not to be caused by the drug) calibrate
  Type I error and systematic bias.
- Positive control outcomes (with known effect sizes) assess sensitivity.
- Empirical calibration adjusts p-values and CIs for residual systematic error.
- Results are generated at scale (thousands of drug-outcome pairs) and made publicly
  available.

### Signal Management Process

The ICH E2E guideline describes the signal management lifecycle:

1. **Signal detection:** Disproportionality analysis, literature review, clinical trials.
2. **Signal validation:** Confirm the signal is real (not artefact).
3. **Signal prioritisation:** Rank by clinical seriousness and strength of evidence.
4. **Signal evaluation:** Detailed assessment (case review, epidemiological studies).
5. **Recommendation for action:** Label change, REMS, withdrawal, further study.
6. **Communication:** DHPC, safety update, Periodic Safety Update Report (PSUR).

## Assumptions

1. **Disproportionality methods assume** that the database is large and representative
   enough for independence-based expected counts to be meaningful.
2. **SCCS assumes** events do not influence subsequent exposures and observation periods.
3. **MaxSPRT assumes** Poisson-distributed events with known expected rate under H0.
4. **All methods assume** adequate data quality (accurate coding, no excessive
   duplication, reasonably complete reporting).

## When to Use This Method

- Post-market safety surveillance (regulatory requirement).
- Signal detection in spontaneous reporting databases (FAERS mining).
- Active surveillance using electronic health records or claims (Sentinel, OHDSI).
- Clinical trial safety monitoring (DILI assessment, DSMB reviews).
- Vaccine safety monitoring (MaxSPRT in the Vaccine Safety Datalink).

## Strengths and Limitations

### Strengths
- Disproportionality analysis is computationally simple and scalable.
- Bayesian shrinkage (BCPNN, MGPS) reduces false positives from sparse data.
- Self-controlled designs eliminate time-fixed confounding.
- MaxSPRT enables continuous monitoring with controlled Type I error.
- LEGEND framework provides systematic, transparent, and reproducible results.

### Limitations
- Spontaneous reports lack denominators; disproportionality does not estimate risk.
- Reporting biases (stimulated reporting, notoriety, Weber effect) inflate signals.
- Multiple comparisons: thousands of drug-event pairs increase false positives.
- Signals require clinical validation — statistical association is not causation.
- SCCS violations (event affecting exposure) can bias results.

## Key References

1. Evans, S. J. W., Waller, P. C., & Davis, S. (2001). Use of proportional reporting
   ratios (PRRs) for signal generation. *Pharmacoepidemiology and Drug Safety*, 10,
   483-486.
2. DuMouchel, W. (1999). Bayesian data mining in large frequency tables, with an
   application to the FDA spontaneous reporting system. *American Statistician*, 53(3),
   177-190.
3. Bate, A., et al. (1998). A Bayesian neural network method for adverse drug reaction
   signal generation. *European Journal of Clinical Pharmacology*, 54, 315-321.
4. Kulldorff, M., et al. (2011). A maximized sequential probability ratio test for drug
   and vaccine safety surveillance. *Sequential Analysis*, 30(1), 58-78.
5. Schuemie, M. J., et al. (2018). Empirical confidence interval calibration for
   population-level effect estimation studies. *PNAS*, 115(11), 2571-2577.
6. Senior, J. R. (2014). Evolution of the Food and Drug Administration approach to
   liver safety assessment. *Clinical Gastroenterology and Hepatology*, 12(9), 1445-1453.
7. Petersen, I., et al. (2016). Self controlled case series methods. *BMJ*, 354, i4515.
