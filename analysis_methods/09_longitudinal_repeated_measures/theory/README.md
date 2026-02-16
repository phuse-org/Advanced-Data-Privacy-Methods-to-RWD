# Longitudinal / Repeated Measures — Theory

## Introduction

Longitudinal data arise when the same subjects are measured repeatedly over time. In clinical trials, patients are assessed at multiple visits (e.g., baseline, weeks 4, 8, 12, 16); in observational studies, outcomes are tracked across follow-up periods. These repeated measurements within subjects are correlated, violating the independence assumption of standard regression methods. Specialized statistical methods — mixed-effects models, GEE, and MMRM — account for this within-subject correlation, improving both efficiency and validity of inference.

The choice of longitudinal method directly affects the interpretation of treatment effects and the handling of missing data, making it central to the statistical analysis plan of clinical trials. The mixed model for repeated measures (MMRM) has become the de facto standard for continuous endpoints in confirmatory clinical trials.

## Mathematical Foundation

### Linear Mixed-Effects Models (LMMs)

The general LMM for subject $i$ at time $j$ is:

$$Y_{ij} = X_{ij}^T \beta + Z_{ij}^T b_i + \epsilon_{ij}$$

where:
- $X_{ij}^T \beta$ is the fixed-effects component (population-average effects of covariates).
- $Z_{ij}^T b_i$ is the random-effects component, with $b_i \sim N(0, G)$.
- $\epsilon_{ij}$ is the residual error, with $\epsilon_i \sim N(0, R_i)$.

The marginal distribution of $Y_i$ is $N(X_i \beta, V_i)$ where $V_i = Z_i G Z_i^T + R_i$.

### Random Intercepts and Slopes

The simplest LMM includes a random intercept:

$$Y_{ij} = (\beta_0 + b_{0i}) + \beta_1 t_{ij} + \beta_2 x_i + \epsilon_{ij}$$

where $b_{0i} \sim N(0, \sigma_{b_0}^2)$ captures the between-subject variability in baseline levels. Adding a random slope for time:

$$Y_{ij} = (\beta_0 + b_{0i}) + (\beta_1 + b_{1i}) t_{ij} + \beta_2 x_i + \epsilon_{ij}$$

where $(b_{0i}, b_{1i})^T \sim N(0, G)$ and $G$ is a 2x2 covariance matrix capturing the correlation between intercepts and slopes.

### Covariance Structures

The choice of covariance structure for $R_i$ (and $V_i$) is critical:

- **Compound symmetry (CS)**: Equal variance at all times, equal correlation between all pairs: $\text{Corr}(Y_{ij}, Y_{ik}) = \rho$ for all $j \neq k$. Equivalent to a random intercept model.
- **AR(1) (autoregressive order 1)**: Correlation decays exponentially with temporal lag: $\text{Corr}(Y_{ij}, Y_{ik}) = \rho^{|j-k|}$. Appropriate when closer time points are more correlated.
- **Unstructured (UN)**: Each variance and covariance is estimated freely. Most flexible but requires the most parameters ($p(p+1)/2$ for $p$ time points). The preferred choice for MMRM in clinical trials.
- **Toeplitz**: Banded structure where correlation depends only on lag distance but not exponentially.
- **Heterogeneous variants**: Allow different variances at each time point combined with CS or AR(1) correlation patterns.

### MMRM (Mixed Model for Repeated Measures)

MMRM is a special case of the LMM widely used in clinical trials. Key features:

$$Y_{ij} = \mu + \alpha_i \cdot \text{trt} + \beta_j \cdot \text{visit} + \gamma_{ij} \cdot (\text{trt} \times \text{visit}) + \delta \cdot Y_{i,\text{baseline}} + \epsilon_{ij}$$

- Fixed effects for treatment, visit (as categorical), treatment-by-visit interaction, and baseline value.
- Unstructured covariance for the residuals across visits.
- No random effects (the within-subject correlation is handled entirely by the residual covariance structure).
- Estimated using REML.
- The treatment effect at the final visit is the primary estimand: the treatment-by-visit interaction at the last time point.
- Valid under MAR when the model includes covariates predictive of dropout.

### Marginal Models (GEE)

Generalized Estimating Equations (GEE; Liang and Zeger, 1986) directly model the marginal mean:

$$g(E[Y_{ij}]) = X_{ij}^T \beta$$

with a "working" correlation structure for within-subject observations. Key features:

- **Population-averaged interpretation**: $\beta$ describes the effect on the average response, not conditional on random effects.
- **Robust (sandwich) standard errors**: Consistent even if the working correlation structure is misspecified.
- **No distributional assumption**: GEE is a quasi-likelihood method.
- **Limitation**: GEE requires data to be MCAR (not just MAR) for valid inference with incomplete data. Weighted GEE (WGEE) can handle MAR by incorporating inverse probability weights.

### Conditional vs Marginal Models

- **Conditional (LMM)**: Effects are interpreted conditional on the random effects (subject-specific). For a binary outcome: "for a given patient, how does the log-odds change?"
- **Marginal (GEE)**: Effects are interpreted as population-averaged. For a binary outcome: "how does the population-level probability change?"

For linear models with identity link, marginal and conditional effect estimates coincide. For non-linear models (logistic, Poisson), they differ — the marginal effect is attenuated relative to the conditional effect.

### Generalized Linear Mixed Models (GLMMs)

GLMMs extend LMMs to non-normal outcomes (binary, count, ordinal):

$$g(E[Y_{ij} | b_i]) = X_{ij}^T \beta + Z_{ij}^T b_i$$

where $g$ is a link function (logit, log, etc.) and $Y_{ij}$ follows an exponential family distribution conditional on $b_i$. Estimation is more complex than LMMs because the marginal likelihood involves integrals over the random effects distribution, typically approximated via Laplace approximation or adaptive Gauss-Hermite quadrature.

### Intraclass Correlation Coefficient (ICC)

The ICC measures the proportion of total variance attributable to between-subject differences:

$$\text{ICC} = \frac{\sigma_{b_0}^2}{\sigma_{b_0}^2 + \sigma_\epsilon^2}$$

A high ICC (e.g., > 0.5) indicates strong within-subject correlation and reinforces the need for repeated measures methods. If ICC is near zero, observations within subjects are essentially independent.

## Key Concepts

### Repeated Measures ANOVA: Limitations

Traditional repeated measures ANOVA requires:
- Complete data (no missing time points).
- Sphericity (equal variances of all pairwise differences).
- Balanced designs.

These restrictions make it impractical for most clinical trial settings where dropout occurs and designs are unbalanced.

### Growth Curve Models

Growth curve models parameterize the trajectory over time (linear, quadratic, piecewise linear) using fixed and random effects:

$$Y_{ij} = (\beta_0 + b_{0i}) + (\beta_1 + b_{1i}) t_j + (\beta_2 + b_{2i}) t_j^2 + \epsilon_{ij}$$

These are useful when the research question focuses on the shape of change over time rather than effects at specific time points.

### Joint Models (Longitudinal + Survival)

Joint models simultaneously model a longitudinal process and a time-to-event outcome:

- **Longitudinal submodel**: $Y_i(t) = X_i(t)^T \beta + Z_i(t)^T b_i + \epsilon_i(t)$
- **Survival submodel**: $h_i(t) = h_0(t) \exp(\gamma^T W_i + \alpha m_i(t))$

where $m_i(t)$ is the true (latent) longitudinal trajectory, linking the two submodels through $\alpha$. Joint models handle informative dropout (where dropout depends on the unobserved longitudinal trajectory), a form of MNAR.

## Assumptions

1. **LMM**: Normality of random effects and residuals; correct specification of the mean structure and covariance structure; MAR for valid inference with missing data.
2. **GEE**: Correct specification of the marginal mean; MCAR for valid inference (or WGEE for MAR); no distributional assumptions beyond first two moments.
3. **MMRM**: Normality of the outcome conditional on covariates; correct covariance structure (unstructured is recommended to avoid misspecification); MAR.
4. **GLMM**: Correct distributional family, link function, and random effects distribution.

## Variants and Extensions

- **Non-linear mixed-effects models**: For pharmacokinetic/pharmacodynamic (PK/PD) data where trajectories follow known non-linear functional forms.
- **Multivariate longitudinal models**: Jointly model two or more longitudinal outcomes.
- **Bayesian mixed models**: Enable incorporation of prior information and natural handling of complex hierarchical structures.
- **Functional data analysis**: When observations are densely sampled, treating trajectories as functions rather than discrete measurements.

## When to Use This Method

| Scenario | Recommended Method |
|---|---|
| Confirmatory clinical trial, continuous endpoint | MMRM (unstructured covariance) |
| Exploratory trajectory analysis | LMM with random slopes |
| Binary/count outcome over time | GLMM or GEE |
| Population-averaged effects (epidemiology) | GEE with robust SE |
| Informative dropout | Joint model (longitudinal + survival) |
| Clustered data (patients within sites) | LMM with nested random effects |

## Strengths and Limitations

### Strengths
- Properly accounts for within-subject correlation.
- Handles unbalanced data and irregular time points naturally (LMM/GEE).
- MMRM is valid under MAR and is the regulatory standard for clinical trials.
- Random effects capture individual-level variability, enabling personalized predictions.
- Joint models handle informative dropout.

### Limitations
- Covariance structure misspecification can bias standard errors (LMM with restricted structures).
- GEE is invalid under MAR without inverse probability weighting.
- MMRM with unstructured covariance can have convergence issues with many time points.
- Joint models are computationally intensive and require distributional assumptions.
- Distinguishing between marginal and conditional interpretations requires care.

## Key References

1. Fitzmaurice GM, Laird NM, Ware JH. *Applied Longitudinal Analysis*. 2nd ed. Wiley, 2011.
2. Verbeke G, Molenberghs G. *Linear Mixed Models for Longitudinal Data*. Springer, 2000.
3. Mallinckrodt CH, et al. Recommendations for the primary analysis of continuous endpoints in longitudinal clinical trials. *Drug Information Journal*. 2008;42(4):303-319.
4. Liang K-Y, Zeger SL. Longitudinal data analysis using generalized linear models. *Biometrika*. 1986;73(1):13-22.
5. Rizopoulos D. *Joint Models for Longitudinal and Time-to-Event Data*. CRC Press, 2012.
6. Pinheiro JC, Bates DM. *Mixed-Effects Models in S and S-PLUS*. Springer, 2000.
7. National Research Council. *The Prevention and Treatment of Missing Data in Clinical Trials*. National Academies Press, 2010.
8. MMRM R package documentation. Li et al. mmrm: Mixed Models for Repeated Measures. CRAN, 2023.
