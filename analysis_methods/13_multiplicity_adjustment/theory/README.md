# Multiplicity Adjustment — Theory

## Introduction

Multiplicity adjustment is a cornerstone of confirmatory clinical trial statistics. When a
trial tests multiple hypotheses simultaneously — multiple endpoints, multiple dose groups,
multiple subpopulations, or interim analyses — the probability of at least one false positive
finding increases dramatically beyond the nominal significance level. Without proper adjustment,
a trial testing 20 independent hypotheses at alpha = 0.05 has a 64% chance of at least one
false positive, even if no treatment effect exists.

Regulatory agencies (FDA, EMA) require multiplicity control in confirmatory trials to maintain
the integrity of the decision-making process. The statistical framework for multiplicity has
evolved from simple corrections (Bonferroni) to sophisticated graphical approaches that
reflect clinical priorities while preserving strong type I error control.

## Mathematical Foundation

### Family-Wise Error Rate (FWER)

The FWER is the probability of making at least one type I error among a family of hypotheses:

```
FWER = P(reject at least one true H0)
```

Strong FWER control ensures this probability does not exceed alpha for any configuration of
true and false null hypotheses. This is the standard required for confirmatory trials.

### False Discovery Rate (FDR)

The FDR is the expected proportion of false rejections among all rejections:

```
FDR = E[V / max(R, 1)]
```

where V = number of false rejections and R = total number of rejections. FDR control is less
conservative than FWER control and is appropriate for exploratory analyses with many tests
(e.g., genomics, biomarker screening).

### Bonferroni Correction

The simplest FWER-controlling procedure. For m hypotheses, reject H_i if p_i <= alpha/m.

```
alpha_adjusted = alpha / m
```

This controls FWER at level alpha regardless of the dependence structure among tests.
It is conservative because it does not exploit the correlation between test statistics.

### Holm Step-Down Procedure

A uniformly more powerful alternative to Bonferroni. Order p-values: p_(1) <= p_(2) <= ... <= p_(m).

```
Step 1: Reject H_(1) if p_(1) <= alpha/m
Step 2: Reject H_(2) if p_(2) <= alpha/(m-1)
...continue until the first non-rejection, then stop.
```

Holm controls FWER under arbitrary dependence and always rejects at least as many hypotheses
as Bonferroni.

### Hochberg Step-Up Procedure

The reverse of Holm. Start from the largest p-value:

```
Step 1: If p_(m) <= alpha, reject all.
Step 2: If p_(m-1) <= alpha/2, reject H_(1),...,H_(m-1).
...continue downward.
```

Hochberg is more powerful than Holm but requires non-negative dependence (PRDS condition)
among test statistics. This condition is met for one-sided tests based on multivariate
normal statistics (common in clinical trials).

### Benjamini-Hochberg (BH) Procedure for FDR

Order p-values and find the largest k such that:

```
p_(k) <= (k / m) * alpha
```

Reject all H_(1), ..., H_(k). This controls FDR at level alpha under independence or
positive regression dependence (PRDS).

## Key Concepts

### Gatekeeping Procedures

In clinical trials, hypotheses often have a hierarchical structure:

- Primary endpoints must be significant before secondary endpoints are tested.
- A higher dose must demonstrate efficacy before a lower dose is tested.

Serial gatekeeping enforces these constraints. The primary family is tested at full alpha.
Only if a primary hypothesis is rejected does alpha "pass through the gate" to the
secondary family.

### Graphical Approaches (Bretz et al., 2009)

The graphical approach provides a unified framework for multiplicity. It is defined by:

1. **Nodes**: Each hypothesis H_i with an initial alpha allocation w_i (where sum(w_i) = alpha).
2. **Edges**: Directed weighted edges g_ij representing the fraction of alpha from H_i
   that is propagated to H_j upon rejection of H_i.

The algorithm:
1. Test all hypotheses at their current alpha levels.
2. If H_i is rejected, redistribute its alpha to remaining hypotheses according to the
   transition weights.
3. Update the graph (remove H_i, recalculate transition weights).
4. Repeat until no more hypotheses can be rejected.

This framework encompasses Bonferroni, Holm, fixed-sequence, and fallback procedures as
special cases.

### Fixed-Sequence (Hierarchical) Testing

Hypotheses are tested in a pre-specified order. The first hypothesis is tested at full alpha.
If rejected, the second is tested at full alpha, and so on. Testing stops at the first
non-rejection.

```
Test H_1 at alpha -> if rejected -> Test H_2 at alpha -> if rejected -> ...
```

This is maximally powerful for earlier hypotheses but requires strong prior ordering.

### Alpha Recycling

When a hypothesis is rejected, its alpha can be "recycled" to other hypotheses. In the
graphical framework, this is formalized through the transition matrix. Alpha recycling
allows later hypotheses to be tested at levels greater than their initial allocation.

### Closed Testing Principle

A hypothesis H_i can be rejected at level alpha if and only if every intersection hypothesis
containing H_i is rejected by a valid local test at level alpha. The closed testing principle
guarantees strong FWER control. Many common procedures (Holm, Hochberg, graphical approaches)
can be derived as special cases of closed testing.

## Assumptions

1. **FWER methods**: Bonferroni and Holm require no assumptions about dependence. Hochberg
   requires PRDS (positive regression dependency on subsets).
2. **FDR methods**: BH procedure requires independence or PRDS. The Benjamini-Yekutieli
   procedure controls FDR under arbitrary dependence but is more conservative.
3. **Graphical approaches**: The shortcut algorithm assumes Bonferroni-based local tests.
   Extensions to Simes-based tests require PRDS.
4. **Pre-specification**: For confirmatory trials, the multiplicity strategy must be fully
   pre-specified in the statistical analysis plan before unblinding.

## Variants and Extensions

- **Fallback procedures**: A hybrid of fixed-sequence and Bonferroni, allowing later hypotheses
  to retain some alpha even if earlier ones are not rejected.
- **Parametric approaches**: When the joint distribution of test statistics is known (e.g.,
  multivariate normal from Dunnett's test), parametric methods are more powerful than
  Bonferroni-based approaches.
- **Adaptive graphical procedures**: Combine graphical approaches with closed testing using
  Simes' inequality for greater power.
- **Group-sequential and multiplicity**: In trials with interim analyses and multiple endpoints,
  alpha spending functions are combined with multiplicity adjustments.

## When to Use This Method

- **FWER control (Bonferroni, Holm, Hochberg, graphical)**: Required in confirmatory clinical
  trials for regulatory submissions. Use whenever strong control of false positives is needed.
- **FDR control (BH procedure)**: Appropriate for exploratory analyses, genomics, biomarker
  screening, or when testing hundreds to thousands of hypotheses.
- **Graphical approaches**: When the trial has a complex hypothesis structure with clinical
  priorities (e.g., primary and secondary endpoints, multiple doses, co-primary endpoints).
- **Fixed-sequence**: When there is a strong scientific ordering of hypotheses.

## Strengths and Limitations

### Strengths
- FWER control protects the integrity of confirmatory trials.
- Graphical approaches are intuitive, flexible, and easily communicated to clinical teams.
- FDR methods provide a good balance between power and error control in high-dimensional settings.
- Closed testing is a powerful general principle that maximizes rejections.

### Limitations
- FWER control is conservative, especially with many hypotheses.
- Complex multiplicity strategies can be difficult to explain and implement correctly.
- Pre-specification requirements limit flexibility in analysis.
- Scale of the problem matters: Bonferroni with 2 endpoints is mild; with 1000 endpoints it
  is extremely conservative.

## Key References

1. Bretz F, Maurer W, Brannath W, Posch M. A graphical approach to sequentially rejective
   multiple test procedures. *Statistics in Medicine*, 2009;28(4):586-604.
2. Dmitrienko A, Tamhane AC, Bretz F. *Multiple Testing Problems in Pharmaceutical
   Statistics*. Chapman and Hall/CRC, 2009.
3. Benjamini Y, Hochberg Y. Controlling the false discovery rate: a practical and powerful
   approach to multiple testing. *JRSS Series B*, 1995;57(1):289-300.
4. Hochberg Y. A sharper Bonferroni procedure for multiple tests of significance. *Biometrika*,
   1988;75(4):800-802.
5. Holm S. A simple sequentially rejective multiple test procedure. *Scandinavian Journal
   of Statistics*, 1979;6(2):65-70.
6. Bretz F, Posch M, Glimm E, et al. Graphical approaches for multiple comparison procedures
   using weighted Bonferroni, Simes, or parametric tests. *Biometrical Journal*, 2011;53(6):894-913.
7. Dmitrienko A, D'Agostino RB. Multiplicity considerations in clinical trials. *NEJM*,
   2018;378(22):2115-2122.
