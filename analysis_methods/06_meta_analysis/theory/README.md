# Meta-Analysis — Theory

## Introduction

Meta-analysis is a statistical methodology for quantitatively combining results from multiple independent studies addressing a common research question. In health research, meta-analysis is the cornerstone of evidence-based medicine, providing the highest level of evidence in the traditional evidence hierarchy. By pooling effect estimates across trials, meta-analysis increases statistical power, improves precision of treatment effect estimates, and can resolve apparent conflicts between individual study findings.

The Cochrane Collaboration has systematized meta-analytic methods for clinical effectiveness research, and regulatory agencies (FDA, EMA) routinely rely on meta-analyses to inform approval decisions.

## Mathematical Foundation

### Fixed-Effect Model

The fixed-effect model assumes all studies estimate a single common true effect. Let $\hat{\theta}_i$ be the observed effect from study $i$ with variance $\sigma_i^2$. The pooled estimate is:

$$\hat{\theta}_{FE} = \frac{\sum_{i=1}^{k} w_i \hat{\theta}_i}{\sum_{i=1}^{k} w_i}$$

where $w_i = 1/\sigma_i^2$ (inverse-variance weights). The variance of the pooled estimate is $\text{Var}(\hat{\theta}_{FE}) = 1/\sum w_i$.

### Random-Effects Model

The random-effects model assumes the true effect varies across studies. Each study estimates $\theta_i = \mu + u_i$, where $u_i \sim N(0, \tau^2)$. The between-study variance $\tau^2$ captures heterogeneity.

$$\hat{\theta}_{RE} = \frac{\sum_{i=1}^{k} w_i^* \hat{\theta}_i}{\sum_{i=1}^{k} w_i^*}$$

where $w_i^* = 1/(\sigma_i^2 + \hat{\tau}^2)$.

### DerSimonian-Laird Estimator

The most common method for estimating $\tau^2$:

$$\hat{\tau}^2_{DL} = \max\left(0, \frac{Q - (k-1)}{\sum w_i - \sum w_i^2 / \sum w_i}\right)$$

where $Q = \sum w_i(\hat{\theta}_i - \hat{\theta}_{FE})^2$ is Cochran's Q statistic.

### REML Estimation

Restricted Maximum Likelihood (REML) provides a less biased estimate of $\tau^2$ by maximizing the restricted log-likelihood:

$$\ell_R(\tau^2) = -\frac{1}{2}\sum_{i=1}^{k}\left[\log(\sigma_i^2 + \tau^2) + \frac{(\hat{\theta}_i - \hat{\theta}_{RE})^2}{\sigma_i^2 + \tau^2}\right] - \frac{1}{2}\log\left(\sum \frac{1}{\sigma_i^2 + \tau^2}\right)$$

REML is generally preferred over DerSimonian-Laird when the number of studies is small.

## Key Concepts

### Heterogeneity Assessment

- **Cochran's Q test**: Tests $H_0: \tau^2 = 0$. Has low power when $k$ is small.
- **I-squared ($I^2$)**: The proportion of total variability due to between-study heterogeneity: $I^2 = \max(0, (Q - (k-1))/Q) \times 100\%$. Benchmarks: 25% low, 50% moderate, 75% high.
- **Tau-squared ($\tau^2$)**: The absolute between-study variance on the effect-size scale.
- **Prediction interval**: A 95% prediction interval for the true effect in a new study: $\hat{\mu} \pm t_{k-2, 0.025}\sqrt{\hat{\tau}^2 + \text{Var}(\hat{\mu})}$. This is more clinically informative than confidence intervals for the mean effect.

### Forest Plots

Forest plots display each study's point estimate and confidence interval alongside the pooled estimate. The size of each marker is proportional to the study weight. They provide an immediate visual summary of effect direction, magnitude, precision, and heterogeneity.

### Funnel Plots and Publication Bias

Funnel plots display each study's effect against its precision (or standard error). In the absence of bias, points should form a symmetric inverted funnel. Asymmetry suggests publication bias, small-study effects, or other systematic differences.

- **Egger's test**: A regression test for funnel plot asymmetry, regressing the standardized effect on its precision.
- **Trim-and-fill**: A non-parametric method that estimates the number of "missing" studies and adjusts the pooled estimate.
- **Selection models (Copas, Vevea-Hedges)**: Parametric models for publication bias.

## Assumptions

1. **Study-level independence**: Effect estimates from different studies are independent.
2. **Correct within-study variances**: Standard errors reported by individual studies are accurate.
3. **Exchangeability (random-effects)**: Studies sample from a common distribution of true effects.
4. **No systematic reporting bias**: Published results are representative (often violated in practice).
5. **Comparable effect measures**: All studies estimate the same type of effect (e.g., all report hazard ratios).
6. **Normality of random effects**: The distribution of true effects is approximately normal.

## Variants and Extensions

### Network Meta-Analysis (NMA)

Network meta-analysis (also called mixed-treatment comparison) extends pairwise meta-analysis to simultaneously compare three or more interventions, even when not all have been compared in head-to-head trials. It relies on both direct and indirect evidence.

- **Consistency assumption**: Direct and indirect evidence agree ($\theta_{AB} = \theta_{AC} - \theta_{BC}$).
- **SUCRA (Surface Under the Cumulative Ranking)**: Provides a numerical summary (0-100%) of where each treatment ranks relative to an ideal treatment always ranked first.
- **Node-splitting**: A diagnostic to check for inconsistency between direct and indirect evidence at each comparison.

### Individual Patient Data (IPD) Meta-Analysis

IPD meta-analysis uses original participant-level data from each study, rather than aggregate published results. Benefits include:

- Ability to investigate patient-level moderators (treatment-covariate interactions).
- More flexible modeling (e.g., non-linear dose-response, time-to-event outcomes).
- Better handling of missing data and standardized analyses.
- Two-stage approach (analyze each study, then combine) or one-stage approach (single hierarchical model).

### Meta-Regression

Meta-regression extends the random-effects model by including study-level covariates:

$$\theta_i = \beta_0 + \beta_1 x_{i1} + \cdots + \beta_p x_{ip} + u_i + \epsilon_i$$

This is used to explore sources of heterogeneity (e.g., dose, follow-up duration, risk of bias). Caution: ecological bias may arise since associations at the study level may not reflect individual-level relationships. Meta-regression generally requires at least 10 studies per covariate.

### Subgroup Analysis

Subgroup analysis partitions studies into categories (e.g., by design, population, or dose) and examines whether the effect differs across groups. A formal test for subgroup differences compares the Q-between statistic to a chi-squared distribution.

## When to Use This Method

- **Multiple studies exist** on the same intervention or exposure-outcome relationship.
- **Quantitative synthesis** is more informative than a narrative review.
- **Studies are sufficiently similar** in design, population, and outcome measurement.
- **Regulatory submissions** require pooled efficacy or safety summaries.
- **Clinical guidelines** demand the best available combined evidence.

Prefer meta-analysis over a single large trial when studies already exist, when a single trial cannot be conducted (ethical/logistical constraints), or when exploring between-study variation is itself of interest.

## Strengths and Limitations

### Strengths
- Increases precision and statistical power beyond any single study.
- Provides a transparent, reproducible summary of evidence.
- Identifies and quantifies heterogeneity across settings and populations.
- Can reveal small effects not detectable in individual studies.
- NMA enables indirect comparisons where head-to-head trials are unavailable.

### Limitations
- "Garbage in, garbage out" — pooling biased studies yields biased conclusions.
- Heterogeneity may be so large that a single pooled estimate is misleading.
- Publication bias can distort results despite detection methods.
- Study-level meta-regression is prone to ecological fallacy.
- Requires judgment about which studies are sufficiently similar to combine.
- NMA consistency assumption may be violated in practice.

## Key References

1. Higgins JPT, Thomas J, et al. (eds). *Cochrane Handbook for Systematic Reviews of Interventions*, Version 6.3, 2022.
2. Borenstein M, Hedges LV, Higgins JPT, Rothstein HR. *Introduction to Meta-Analysis*. 2nd ed. Wiley, 2021.
3. DerSimonian R, Laird N. Meta-analysis in clinical trials. *Controlled Clinical Trials*. 1986;7(3):177-188.
4. Veroniki AA, et al. Methods to estimate the between-study variance and its uncertainty in meta-analysis. *Research Synthesis Methods*. 2016;7(1):55-79.
5. Salanti G. Indirect and mixed-treatment comparison, network, or multiple-treatments meta-analysis. *Research Synthesis Methods*. 2012;3(2):80-97.
6. Riley RD, et al. *Individual Participant Data Meta-Analysis: A Handbook for Healthcare Research*. Wiley, 2021.
7. Egger M, et al. Bias in meta-analysis detected by a simple, graphical test. *BMJ*. 1997;315:629-634.
8. IntHout J, et al. The Hartung-Knapp-Sidik-Jonkman method for random effects meta-analysis is straightforward and considerably outperforms the standard DerSimonian-Laird method. *BMC Medical Research Methodology*. 2014;14:25.
