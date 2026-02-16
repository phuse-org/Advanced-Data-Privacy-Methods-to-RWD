# Sensitivity Analysis — Theory

## Introduction

Sensitivity analysis assesses the robustness of study findings to violations of assumptions, unmeasured biases, and analytic choices. In clinical and epidemiological research, causal inference from observational data rests on untestable assumptions — no unmeasured confounding, no misclassification, no selection bias. Sensitivity analysis quantifies how strong these biases would need to be to invalidate the observed conclusions.

Unlike primary statistical analyses that yield point estimates and p-values, sensitivity analysis asks: "How much bias would be needed to explain away our result?" If the required bias is implausibly large, the finding is considered robust. If a small plausible bias would nullify the result, the finding is fragile and warrants caution.

Sensitivity analysis is now routinely required by leading journals (BMJ, JAMA, Lancet) and is recommended by the STROBE guidelines for observational studies and the ICH E9(R1) addendum for clinical trials.

## Mathematical Foundation

### E-Value

The E-value (VanderWeele and Ding, 2017) quantifies the minimum strength of association that an unmeasured confounder would need to have with both the treatment and the outcome, conditional on measured covariates, to fully explain away the observed treatment-outcome association.

For a risk ratio (RR):

$$E\text{-value} = RR + \sqrt{RR \times (RR - 1)}$$

For the lower confidence limit (RR_lower):

$$E\text{-value}_{CI} = RR_{lower} + \sqrt{RR_{lower} \times (RR_{lower} - 1)}$$

Interpretation: An E-value of 3.5 means an unmeasured confounder would need to be associated with both the treatment and the outcome by a risk ratio of at least 3.5 each (above and beyond measured confounders) to explain away the observed effect. If no plausible confounder has associations of this magnitude, the finding is robust.

For other effect measures (OR, HR, mean difference), the E-value formula is applied after converting to the RR scale using appropriate approximations.

### Quantitative Bias Analysis (QBA)

QBA systematically corrects effect estimates for known or hypothesized biases. The three main types are:

#### 1. Unmeasured Confounding

Using the bias factor approach (Rosenbaum and Rubin, 1983), the adjusted effect is:

$$RR_{adj} = \frac{RR_{obs}}{B}$$

where the bias factor $B$ depends on:
- $RR_{UD}$: Association between the unmeasured confounder $U$ and the outcome $D$.
- $RR_{EU}$: Association between the exposure $E$ and the confounder $U$ (prevalence difference).

$$B = \frac{RR_{UD} \times p_1 + (1 - p_1)}{RR_{UD} \times p_0 + (1 - p_0)}$$

where $p_1$ and $p_0$ are the prevalence of the confounder in exposed and unexposed groups.

#### 2. Misclassification Bias

For outcome misclassification with sensitivity $Se$ and specificity $Sp$:

$$P_{true} = \frac{P_{obs} - (1 - Sp)}{Se - (1 - Sp)}$$

The corrected cell counts can be derived and the bias-adjusted effect estimate recalculated. For non-differential misclassification of a binary exposure, the observed OR is biased toward the null.

#### 3. Selection Bias

Selection bias arises when the probability of being included in the study depends on both the exposure and the outcome. The bias-corrected estimate is:

$$OR_{adj} = \frac{OR_{obs}}{B_{selection}}$$

where $B_{selection}$ is a function of the selection probabilities in each exposure-outcome stratum.

### Probabilistic Bias Analysis

Rather than using single bias parameter values, probabilistic bias analysis (PBA) specifies probability distributions for bias parameters and produces a distribution of bias-adjusted estimates through Monte Carlo simulation:

1. Draw bias parameters from their prior distributions.
2. Compute the bias-corrected estimate.
3. Repeat thousands of times.
4. Summarize the distribution of adjusted estimates (median, 2.5th and 97.5th percentiles).

This accounts for uncertainty in the bias parameters themselves.

### Rosenbaum Bounds

For matched observational studies (e.g., propensity score matched), Rosenbaum bounds test sensitivity to hidden bias by parameterizing the odds of treatment assignment for matched pairs:

$$\frac{1}{\Gamma} \leq \frac{P(T_i = 1 | X_i)}{P(T_j = 1 | X_j)} \leq \Gamma$$

where $\Gamma \geq 1$ quantifies the degree of departure from random assignment within matched pairs. The analysis asks: at what value of $\Gamma$ does the treatment effect become non-significant? A large $\Gamma$ (e.g., > 2) indicates robustness.

### Tipping-Point Analysis

For clinical trials with missing data, tipping-point analysis systematically shifts imputed values for dropouts by an amount $\delta$ until the treatment effect crosses the significance threshold:

$$\hat{\theta}(\delta) = 0 \text{ or } p(\delta) = 0.05$$

The tipping point is the smallest $\delta$ that nullifies the result. If this $\delta$ is clinically implausible, the result is robust to departures from the MAR assumption.

## Key Concepts

### Rule-Out Approach

The rule-out approach specifies a single plausible unmeasured confounder scenario and checks whether it can explain the observed result. For example: "If an unmeasured confounder with prevalence 30% in exposed and 10% in unexposed had a risk ratio of 2.0 with the outcome, the adjusted RR would be..."

### Array Approach

The array approach displays bias-adjusted estimates across a grid of confounder parameters ($RR_{EU}$ and $RR_{UD}$), showing how the observed effect changes for various confounder strengths. This generates a two-dimensional sensitivity matrix.

### Negative Controls

Negative controls are exposures or outcomes known to have no causal relationship with the study outcome or exposure, respectively:

- **Negative control exposure**: An exposure that cannot plausibly affect the outcome but would be subject to the same confounding. If this "control" exposure appears associated with the outcome, residual confounding is likely.
- **Negative control outcome**: An outcome that cannot plausibly be affected by the exposure. If the exposure appears associated with this control outcome, confounding or bias is present.

Example: In a study of statins and cancer, influenza vaccination (negative control exposure) should not affect cancer risk. If it does, unmeasured confounding (e.g., healthy user bias) is indicated.

### Falsification Testing

Falsification testing extends the negative control idea by testing multiple known null hypotheses. If the analytical approach correctly finds null results for known null associations, confidence in the observed positive result increases. This is sometimes called the "negative control outcome battery."

### Sensitivity Analysis for Missing Data

Connected to the ICH E9(R1) estimand framework:

- **Delta adjustment**: Shifts imputed values by a fixed amount to explore MNAR scenarios.
- **Reference-based methods**: Assume dropouts revert to the reference group trajectory (J2R, copy reference).
- **Pattern-mixture sensitivity**: Varies the conditional distribution of missing outcomes across dropout patterns.

## Assumptions

1. **Known bias structure**: QBA assumes the correct type of bias is identified (confounding, misclassification, selection).
2. **Bias parameters are informed**: QBA and PBA require external information (prior studies, expert opinion, validation data) to specify bias parameter values or distributions.
3. **Single bias at a time**: Most methods address one bias at a time. Multiple simultaneous biases require more complex modeling.
4. **Monotone bias direction**: The E-value assumes a consistent direction of confounding. If the confounder increases treatment probability and outcome risk, the bias is in one direction.

## Variants and Extensions

### Sensitivity Analysis for Specific Designs

- **Regression discontinuity**: Sensitivity to the bandwidth choice and functional form near the cutoff.
- **Difference-in-differences**: Sensitivity to violations of the parallel trends assumption.
- **Instrumental variables**: Sensitivity to violations of the exclusion restriction.
- **Mediation analysis**: Sensitivity to unmeasured confounders of the mediator-outcome relationship (mediational E-value).

### Multi-Bias Sensitivity Analysis

Simultaneous adjustment for multiple biases (e.g., unmeasured confounding + outcome misclassification + selection bias) using sequential correction or joint probabilistic models. The `episensr` R package supports this through chained bias analysis.

### Sensitivity for Interaction/Effect Modification

The E-value framework has been extended to assess sensitivity of interaction effects on additive and multiplicative scales.

## When to Use This Method

- **Every observational study** should include some form of sensitivity analysis for unmeasured confounding (E-value at minimum).
- **Clinical trials with missing data** should include tipping-point or delta-adjustment sensitivity analyses.
- **When specific biases are suspected** (known exposure misclassification, selection issues), QBA provides a structured correction.
- **Matched studies** should include Rosenbaum bounds.
- **Studies with residual confounding concerns** should consider negative controls and falsification tests.

## Strengths and Limitations

### Strengths
- Quantifies the degree of bias needed to overturn a result, moving beyond "there might be unmeasured confounding."
- E-values are simple to compute and widely understood.
- Probabilistic bias analysis captures uncertainty in bias parameters.
- Rosenbaum bounds are non-parametric and exact for matched designs.
- Negative controls provide empirical evidence about residual bias.

### Limitations
- QBA results are only as good as the bias parameter assumptions.
- E-values provide an upper bound on the confounder strength needed, not the actual bias.
- Single-bias analyses may miss interactions between multiple biases.
- Probabilistic bias analysis requires specification of bias parameter distributions, which can be subjective.
- Sensitivity analysis cannot prove causation — it can only show whether bias could explain the result.

## Key References

1. VanderWeele TJ, Ding P. Sensitivity analysis in observational research: introducing the E-value. *Annals of Internal Medicine*. 2017;167(4):268-274.
2. Lash TL, Fox MP, MacLehose RF, Maldonado G, McCandless LC, Greenland S. Good practices for quantitative bias analysis. *International Journal of Epidemiology*. 2014;43(6):1969-1985.
3. Rosenbaum PR. *Observational Studies*. 2nd ed. Springer, 2002.
4. Lipsitch M, Tchetgen Tchetgen E, Cohen T. Negative controls: a tool for detecting confounding and bias in observational studies. *Epidemiology*. 2010;21(3):383-388.
5. Fox MP, MacLehose RF, Lash TL. *Applying Quantitative Bias Analysis to Epidemiologic Data*. 2nd ed. Springer, 2021.
6. Cinelli C, Hazlett C. Making sense of sensitivity: extending omitted variable bias. *Journal of the Royal Statistical Society: Series B*. 2020;82(1):39-67.
7. Cro S, et al. Sensitivity analysis for clinical trials with missing continuous outcome data using controlled multiple imputation. *Statistics in Medicine*. 2020;39(21):2815-2842.
8. Mathur MB, et al. Sensitivity analysis for unmeasured confounding in meta-analyses. *Journal of the American Statistical Association*. 2022;117(538):915-933.
