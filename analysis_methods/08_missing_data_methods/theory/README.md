# Missing Data Methods — Theory

## Introduction

Missing data is pervasive in clinical research. In clinical trials, patients discontinue treatment, miss visits, or have incomplete laboratory measurements. In observational studies, records are incomplete due to varied data collection practices. Ignoring missing data or handling it naively can lead to biased treatment effect estimates, loss of statistical power, and invalid inference. Understanding missing data mechanisms and employing principled methods is essential for credible clinical research.

The ICH E9(R1) addendum on estimands has elevated the importance of missing data handling by requiring researchers to specify precisely what treatment effect is being estimated and how intercurrent events (including treatment discontinuation and missing data) are addressed.

## Mathematical Foundation

### Missing Data Mechanisms (Rubin's Classification)

Let $Y$ denote the complete data, $Y_{obs}$ the observed portion, $Y_{mis}$ the missing portion, and $R$ the missingness indicator ($R_i = 1$ if observed, $R_i = 0$ if missing).

1. **Missing Completely at Random (MCAR)**: $P(R | Y_{obs}, Y_{mis}) = P(R)$. Missingness is unrelated to any data, observed or unobserved. Example: data lost due to a laboratory equipment failure.

2. **Missing at Random (MAR)**: $P(R | Y_{obs}, Y_{mis}) = P(R | Y_{obs})$. Missingness depends only on observed data, not on the missing values themselves. Example: sicker patients (observable) are more likely to drop out, but conditional on observed severity, missingness does not depend on the unobserved outcome.

3. **Missing Not at Random (MNAR)**: $P(R | Y_{obs}, Y_{mis})$ depends on $Y_{mis}$. Missingness depends on the unobserved values. Example: patients drop out because they are experiencing the very outcome being measured (e.g., worsening symptoms).

### Consequences of Ignoring Missing Data

- **Complete case analysis (CCA)**: Analyzes only fully observed cases. Unbiased under MCAR but biased under MAR/MNAR. Always loses power by discarding data.
- **Single imputation methods** (mean imputation, LOCF): Underestimate variability because they treat imputed values as known. LOCF assumes the last observed value persists, which is often clinically implausible.

### Multiple Imputation (MI)

Multiple imputation (Rubin, 1987) creates $m$ completed datasets, each with missing values replaced by draws from the predictive distribution of the missing data given the observed data. The procedure has three phases:

1. **Imputation phase**: Generate $m$ completed datasets $\{Y^{(1)}, \ldots, Y^{(m)}\}$ by drawing from $P(Y_{mis} | Y_{obs}, R)$ under an assumed model.

2. **Analysis phase**: Perform the planned analysis on each completed dataset, obtaining estimates $\hat{\theta}^{(j)}$ and variances $\hat{V}^{(j)}$ for $j = 1, \ldots, m$.

3. **Pooling phase (Rubin's rules)**: Combine results:
   - Pooled estimate: $\bar{\theta} = \frac{1}{m}\sum_{j=1}^{m}\hat{\theta}^{(j)}$
   - Within-imputation variance: $\bar{W} = \frac{1}{m}\sum_{j=1}^{m}\hat{V}^{(j)}$
   - Between-imputation variance: $B = \frac{1}{m-1}\sum_{j=1}^{m}(\hat{\theta}^{(j)} - \bar{\theta})^2$
   - Total variance: $T = \bar{W} + (1 + 1/m)B$
   - Degrees of freedom: $\nu = (m-1)\left(1 + \frac{\bar{W}}{(1+1/m)B}\right)^2$

The term $(1 + 1/m)B$ reflects the additional uncertainty due to missing data. With $m \geq 20$ imputations, results are typically stable.

### Fraction of Missing Information

The fraction of missing information (FMI) quantifies how much the missing data contributes to total uncertainty: $\gamma = \frac{(1+1/m)B}{T}$. High FMI (e.g., >0.5) indicates that missing data substantially affects inference and that results are sensitive to the imputation model.

## Key Concepts

### MICE / Fully Conditional Specification (FCS)

Multivariate Imputation by Chained Equations (MICE) imputes each variable iteratively, conditional on all other variables, using variable-specific models. For each variable $Y_j$ with missing values:

$$Y_j^{mis} \sim P(Y_j | Y_{-j}, R_j = 1)$$

The algorithm cycles through all incomplete variables until convergence. MICE is flexible because it allows different models for different variable types (linear regression for continuous, logistic for binary, polytomous for categorical, Poisson for counts).

### Predictive Mean Matching (PMM)

PMM imputes missing values by finding observed cases whose predicted values are closest to the predicted value for the missing case, then sampling one of their observed values. This preserves the distributional properties of the observed data and avoids implausible imputed values (e.g., negative blood pressure).

### Reference-Based Imputation

For clinical trials with the ICH E9(R1) estimand framework:

- **Jump to Reference (J2R)**: After discontinuation, imputed values follow the reference (e.g., placebo) group distribution.
- **Copy Reference**: Imputed values are drawn directly from the reference group distribution.
- **Copy Increment from Reference (CIR)**: Changes from the last observed value follow the reference group trajectory.

These methods are used to implement treatment policy estimands where the effect of interest reflects what would happen if patients reverted to the comparator after discontinuation.

### Pattern-Mixture Models

Pattern-mixture models stratify the analysis by missingness pattern and combine results across patterns. The distribution of the outcome is factored as:

$$P(Y, R) = P(Y | R) P(R)$$

This directly models how outcomes differ by missingness pattern and requires identifying assumptions about the unobserved data in each pattern.

### Selection Models

Selection models factor the joint distribution as:

$$P(Y, R) = P(R | Y) P(Y)$$

The outcome model $P(Y)$ is specified first, and the missingness model $P(R|Y)$ describes how the probability of being observed depends on the outcome. Heckman's selection model and shared-parameter models are examples.

### Inverse Probability of Censoring Weighting (IPCW)

IPCW weights each observed case by the inverse of their probability of being observed:

$$w_i = 1 / P(R_i = 1 | X_i)$$

This reweights the observed sample to represent the full population. IPCW is commonly used in survival analysis and in combination with estimand-based analyses.

## Assumptions

1. **MAR assumption**: Multiple imputation and MICE are valid under MAR. This is untestable from the data alone but can be made more plausible by including strong predictors of both the outcome and missingness in the imputation model.
2. **Congeniality**: The imputation model must be compatible with (at least as general as) the analysis model. Uncongenial models can produce biased pooled estimates.
3. **Sufficient imputations**: At least $m = 20$ imputations are recommended, and more when FMI is high.
4. **Correct imputation model specification**: The imputation model should include the outcome, all analysis variables, auxiliary variables predictive of missingness, and interactions/non-linearities present in the analysis model.

## Variants and Extensions

### Sensitivity Analysis for Missing Data

Since the MAR assumption is untestable, sensitivity analyses explore departures:

- **Tipping-point analysis**: Systematically shifts imputed values (e.g., adding delta to the imputed outcome in the treatment arm) until the treatment effect is no longer significant. The delta at which this occurs is the tipping point.
- **Delta adjustment**: Adds a fixed shift to imputed values: $Y_{mis}^{adj} = Y_{mis}^{MAR} + \delta$. Different deltas represent different MNAR scenarios.
- **Pattern-mixture sensitivity**: Varies the assumptions about the distribution of missing outcomes across different dropout patterns.

### Auxiliary Variables

Including auxiliary variables (predictors of the outcome or missingness that are not in the analysis model) in the imputation model improves the plausibility of MAR and increases efficiency. For example, including baseline severity in the imputation model even if it is not a covariate in the primary analysis.

### Connection to ICH E9(R1) Estimand Framework

The estimand framework requires specifying:
- **Population**: Who is included.
- **Variable**: What outcome is measured.
- **Intercurrent events**: How events like treatment discontinuation are handled.
- **Population-level summary**: What treatment effect measure is used.

Missing data handling is directly linked to the intercurrent event strategy. For a treatment policy estimand, all data (including post-discontinuation) is needed, making missing data methods critical. For a hypothetical estimand (what would have happened if all patients adhered), reference-based imputation or MMRM under MAR is often used.

## When to Use This Method

- **Always assess and address missing data** — it should never be ignored without justification.
- Use **complete case analysis** only when missingness is minimal (<5%) and MCAR is plausible.
- Use **multiple imputation** when MAR is plausible and the proportion of missing data is non-trivial.
- Use **reference-based imputation** in confirmatory clinical trials following the ICH E9(R1) framework.
- Use **sensitivity analysis** always — regardless of the primary missing data method.

## Strengths and Limitations

### Strengths
- Multiple imputation properly accounts for the uncertainty due to missing data.
- MICE is flexible and can handle mixed data types.
- Rubin's rules provide a principled framework for combining results.
- Reference-based methods align with regulatory estimand requirements.
- Sensitivity analyses quantify robustness to untestable assumptions.

### Limitations
- MAR is untestable; all MI results are conditional on this assumption.
- Imputation model misspecification can introduce bias.
- Computationally intensive for large datasets with many variables.
- Results can be sensitive to the choice of imputation method and included predictors.
- MNAR methods require strong, often unverifiable, assumptions.

## Key References

1. Rubin DB. *Multiple Imputation for Nonresponse in Surveys*. Wiley, 1987.
2. Van Buuren S. *Flexible Imputation of Missing Data*. 2nd ed. CRC Press, 2018.
3. Little RJA, Rubin DB. *Statistical Analysis with Missing Data*. 3rd ed. Wiley, 2019.
4. Carpenter JR, Kenward MG. *Multiple Imputation and its Application*. Wiley, 2013.
5. ICH E9(R1). Addendum on estimands and sensitivity analysis in clinical trials. 2019.
6. White IR, Royston P, Wood AM. Multiple imputation using chained equations: issues and guidance for practice. *Statistics in Medicine*. 2011;30(4):377-399.
7. Cro S, et al. Sensitivity analysis for clinical trials with missing continuous outcome data using controlled multiple imputation: a practical guide. *Statistics in Medicine*. 2020;39(21):2815-2842.
8. National Research Council. *The Prevention and Treatment of Missing Data in Clinical Trials*. National Academies Press, 2010.
