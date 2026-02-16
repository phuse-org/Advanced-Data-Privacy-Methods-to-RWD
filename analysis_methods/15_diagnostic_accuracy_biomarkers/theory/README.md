# Diagnostic Test Accuracy and Biomarker Evaluation — Theory

## Introduction

Evaluating diagnostic tests and biomarkers is essential for clinical decision-making. A
diagnostic test classifies patients into disease-positive or disease-negative categories,
while a biomarker provides a continuous or ordinal measurement that can be used to predict
disease status, prognosis, or treatment response. Rigorous statistical methods are needed to
quantify how well these tools discriminate between disease states, to select optimal decision
thresholds, and to assess whether using the test improves clinical outcomes.

This field spans traditional measures (sensitivity, specificity), global discrimination
metrics (ROC curves, AUC), threshold optimization (Youden index), clinical utility assessment
(decision curve analysis), and meta-analytic synthesis of diagnostic accuracy studies.

## Mathematical Foundation

### Sensitivity and Specificity

For a binary test with a binary disease status:

```
Sensitivity (Se) = P(Test+ | Disease+) = TP / (TP + FN)
Specificity (Sp) = P(Test- | Disease-) = TN / (TN + FP)
```

Where TP = true positives, FP = false positives, TN = true negatives, FN = false negatives.

Sensitivity is the probability of correctly identifying diseased individuals (true positive
rate). Specificity is the probability of correctly identifying non-diseased individuals
(true negative rate, equal to 1 - false positive rate).

### Predictive Values

```
Positive Predictive Value (PPV) = P(Disease+ | Test+) = TP / (TP + FP)
Negative Predictive Value (NPV) = P(Disease- | Test-) = TN / (TN + FN)
```

PPV and NPV depend on disease prevalence. By Bayes' theorem:

```
PPV = (Se * prevalence) / (Se * prevalence + (1 - Sp) * (1 - prevalence))
NPV = (Sp * (1 - prevalence)) / ((1 - Se) * prevalence + Sp * (1 - prevalence))
```

A test with excellent Se and Sp will still have low PPV when prevalence is very low.

### Likelihood Ratios

```
LR+ = Se / (1 - Sp)     (likelihood ratio positive)
LR- = (1 - Se) / Sp     (likelihood ratio negative)
```

Likelihood ratios are prevalence-independent and indicate how much the odds of disease
change given a positive or negative test. An LR+ > 10 or LR- < 0.1 indicates a useful test.

### ROC Curve and AUC

The Receiver Operating Characteristic (ROC) curve plots sensitivity (y-axis) against
1 - specificity (x-axis) across all possible thresholds for a continuous biomarker:

```
ROC(t) = (Se(t), 1 - Sp(t))   for threshold t
```

The Area Under the ROC Curve (AUC) summarizes overall discrimination:

```
AUC = P(X_disease > X_healthy)
```

where X is the biomarker value. The AUC equals the probability that a randomly chosen
diseased individual has a higher biomarker value than a randomly chosen non-diseased
individual. AUC = 0.5 indicates no discrimination; AUC = 1.0 indicates perfect discrimination.

Rough guidelines: AUC 0.5-0.6 = fail, 0.6-0.7 = poor, 0.7-0.8 = fair, 0.8-0.9 = good,
0.9-1.0 = excellent.

### Partial AUC

The partial AUC restricts the AUC to a clinically relevant range of specificity or sensitivity.
For example, pAUC over specificity in [0.8, 1.0] focuses on the high-specificity region
where the test is most useful as a confirmatory test.

```
pAUC(a, b) = integral from a to b of Se(1 - Sp) d(1 - Sp)
```

## Key Concepts

### Optimal Cutpoint Selection

When a continuous biomarker is used as a diagnostic test, a threshold (cutpoint) must be
chosen. Common methods:

**Youden Index:**
```
J = max_t {Se(t) + Sp(t) - 1}
```
The threshold that maximizes J gives equal weight to sensitivity and specificity.

**Closest-to-(0,1) corner:**
```
d = min_t {sqrt((1 - Se(t))^2 + (1 - Sp(t))^2)}
```
The threshold closest to the perfect point (Se=1, Sp=1) on the ROC curve.

**Cost-based optimization:** Minimize expected misclassification cost:
```
Cost(t) = C_FN * (1-Se(t)) * prevalence + C_FP * (1-Sp(t)) * (1-prevalence)
```
where C_FN and C_FP are costs of false negatives and false positives.

### Decision Curve Analysis (DCA) and Net Benefit

DCA (Vickers and Elkin, 2006) evaluates the clinical utility of a diagnostic test or
prediction model by calculating the net benefit across a range of threshold probabilities:

```
Net Benefit(p_t) = (TP/n) - (FP/n) * (p_t / (1 - p_t))
```

where p_t is the threshold probability at which a patient would choose treatment. The term
p_t / (1 - p_t) represents the odds ratio weighting of false positives relative to true
positives.

DCA compares the model against two default strategies:
- **Treat all**: Treat every patient regardless of the test.
- **Treat none**: Treat no patient.

A useful test should have higher net benefit than both defaults across a clinically
meaningful range of threshold probabilities.

### Time-Dependent ROC

For survival outcomes, the discrimination ability of a biomarker may change over time. The
time-dependent ROC at time t defines:

```
Se(c, t) = P(X > c | T <= t)     (sensitivity: marker high among events by time t)
Sp(c, t) = P(X <= c | T > t)     (specificity: marker low among survivors at time t)
```

Several definitions exist (cumulative/dynamic, incident/dynamic) depending on how cases and
controls are defined over time.

### Bivariate Meta-Analysis of Diagnostic Test Accuracy

When synthesizing multiple DTA studies, the bivariate model (Reitsma et al., 2005) jointly
models logit-sensitivity and logit-specificity:

```
(logit(Se_i), logit(Sp_i)) ~ N2(mu, Sigma)
```

where mu = (mu_Se, mu_Sp) and Sigma captures the between-study heterogeneity and the
correlation between sensitivity and specificity (which arises from threshold variation
across studies). This model accounts for the trade-off between Se and Sp and produces a
summary ROC (SROC) curve.

## Assumptions

1. **Gold standard**: A definitive reference standard exists to classify disease status. If
   the reference standard is imperfect, bias occurs (imperfect gold standard bias).
2. **Independence of test and reference**: The test result should not influence the reference
   standard assessment (incorporation bias) or the decision to apply the reference standard
   (verification bias).
3. **Representative spectrum**: The study population should reflect the clinical setting where
   the test will be used (spectrum bias).
4. **Threshold consistency** (for meta-analysis): Studies should use comparable patient
   populations and test protocols, though threshold variation is modeled.

## Variants and Extensions

- **Net Reclassification Improvement (NRI)**: Measures how a new marker reclassifies patients
  compared to an existing model. Reported as category-based or continuous NRI.
- **Integrated Discrimination Improvement (IDI)**: The average improvement in predicted
  probability for events minus the average change for non-events.
- **Calibration assessment**: Beyond discrimination, a model should be well-calibrated
  (predicted probabilities match observed frequencies). Calibration plots and the
  Hosmer-Lemeshow test evaluate this.
- **Multi-class ROC**: Extensions for tests with more than two outcome categories.

## When to Use This Method

- **Evaluating a new biomarker**: ROC analysis, AUC, and optimal cutpoint selection.
- **Comparing biomarkers or models**: Compare AUC values (DeLong test), net reclassification.
- **Clinical utility assessment**: DCA to determine whether using the biomarker improves
  decisions compared to treat-all or treat-none strategies.
- **Systematic reviews of diagnostic tests**: Bivariate meta-analysis and SROC curves.
- **Regulatory submissions**: For companion diagnostics and in vitro diagnostics, sensitivity,
  specificity, and PPV/NPV are required. STARD guidelines govern reporting.

## Strengths and Limitations

### Strengths
- ROC/AUC provides a threshold-independent measure of discrimination.
- DCA directly addresses clinical utility, not just statistical performance.
- Bivariate meta-analysis properly handles the Se-Sp correlation.
- Methods are well-established with clear regulatory and reporting guidelines (STARD).

### Limitations
- AUC can be misleading for comparing models when ROC curves cross.
- PPV/NPV depend strongly on prevalence and may not generalize across settings.
- Optimal cutpoints from one study may not perform well in a new population.
- DCA requires specifying a range of threshold probabilities, which involves clinical judgment.
- Time-dependent ROC methods are sensitive to censoring patterns.

## Key References

1. Pepe MS. *The Statistical Evaluation of Medical Tests for Classification and Prediction*.
   Oxford University Press, 2003.
2. Vickers AJ, Elkin EB. Decision curve analysis: a novel method for evaluating prediction
   models. *Medical Decision Making*, 2006;26(6):565-574.
3. DeLong ER, DeLong DM, Clarke-Pearce DL. Comparing the areas under two or more correlated
   receiver operating characteristic curves. *Biometrics*, 1988;44(3):837-845.
4. Reitsma JB, Glas AS, Rutjes AW, et al. Bivariate analysis of sensitivity and specificity
   produces informative summary measures in diagnostic reviews. *JCCE*, 2005;58(10):982-990.
5. Youden WJ. Index for rating diagnostic tests. *Cancer*, 1950;3(1):32-35.
6. Bossuyt PM, Reitsma JB, Bruns DE, et al. STARD 2015: An updated list of essential items
   for reporting diagnostic accuracy studies. *BMJ*, 2015;351:h5527.
7. Heagerty PJ, Lumley T, Pepe MS. Time-dependent ROC curves for censored survival data.
   *Biometrics*, 2000;56(2):337-344.
