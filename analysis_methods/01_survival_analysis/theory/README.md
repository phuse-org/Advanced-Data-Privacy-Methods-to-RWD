# Survival Analysis — Theory

## Introduction

Survival analysis is a branch of statistics concerned with modeling the time until an event of interest occurs. Originally developed for studying mortality, it is now applied broadly across health research to analyze time-to-event outcomes such as disease progression, hospital readmission, treatment failure, and adverse events. The distinguishing feature of survival data is **censoring**: not all subjects experience the event during the observation period, yet their partial information remains valuable.

In clinical trials, survival analysis is the primary tool for evaluating time-dependent endpoints such as overall survival (OS), progression-free survival (PFS), and event-free survival (EFS). Regulatory agencies including the FDA and EMA rely on survival methods for drug approvals in oncology, cardiology, and many other therapeutic areas.

## Mathematical Foundation

### The Survival Function

The survival function S(t) gives the probability that an individual survives beyond time t:

```
S(t) = P(T > t) = 1 - F(t)
```

where F(t) is the cumulative distribution function of the event time T. S(t) is a non-increasing function with S(0) = 1 and S(infinity) = 0.

### The Hazard Function

The hazard function h(t) represents the instantaneous rate of the event occurring at time t, given survival up to that point:

```
h(t) = lim_{dt->0} P(t <= T < t + dt | T >= t) / dt = f(t) / S(t)
```

where f(t) is the probability density function. The hazard is not a probability; it can exceed 1.

### Cumulative Hazard Function

The cumulative hazard H(t) accumulates hazard over time:

```
H(t) = integral from 0 to t of h(u) du = -log(S(t))
```

This relationship links the survival and hazard functions: S(t) = exp(-H(t)).

### Kaplan-Meier Estimator

The Kaplan-Meier (KM) estimator is a non-parametric estimate of the survival function:

```
S_hat(t) = product over t_i <= t of (1 - d_i / n_i)
```

where t_i are the distinct event times, d_i is the number of events at time t_i, and n_i is the number at risk just before t_i. The variance is estimated by Greenwood's formula:

```
Var(S_hat(t)) = S_hat(t)^2 * sum over t_i <= t of d_i / (n_i * (n_i - d_i))
```

### Log-Rank Test

The log-rank test compares survival distributions between groups under the null hypothesis of no difference. The test statistic is:

```
chi^2 = (sum(O_j - E_j))^2 / sum(V_j)
```

where O_j and E_j are observed and expected events in group j, and V_j is the variance. Under H0, this follows a chi-squared distribution with (G-1) degrees of freedom, where G is the number of groups.

### Cox Proportional Hazards Model

The Cox PH model specifies the hazard for subject i as:

```
h_i(t) = h_0(t) * exp(beta_1 * X_i1 + beta_2 * X_i2 + ... + beta_p * X_ip)
```

where h_0(t) is the unspecified baseline hazard and the betas are regression coefficients. The model is semi-parametric: the baseline hazard is left unspecified, and inference proceeds via partial likelihood:

```
PL(beta) = product over event times t_j of [exp(X_j * beta) / sum over i in R(t_j) of exp(X_i * beta)]
```

where R(t_j) is the risk set at time t_j. Exponentiated coefficients exp(beta) are hazard ratios (HR).

### Parametric Models

Parametric survival models fully specify the distribution of survival times:

- **Exponential**: Constant hazard h(t) = lambda. S(t) = exp(-lambda * t).
- **Weibull**: h(t) = (alpha / lambda) * (t / lambda)^(alpha - 1). Reduces to exponential when alpha = 1.
- **Log-normal**: log(T) ~ Normal(mu, sigma^2). Hazard is non-monotone.
- **Log-logistic**: Similar to log-normal but with closed-form survival function.
- **Gompertz**: h(t) = alpha * exp(beta * t). Commonly used in actuarial science.

### Accelerated Failure Time (AFT) Models

AFT models model the log of survival time directly:

```
log(T) = mu + beta_1 * X_1 + ... + beta_p * X_p + sigma * epsilon
```

where epsilon follows a specified distribution (extreme value, normal, logistic). The acceleration factor exp(beta) indicates how covariates speed up or slow down the time scale.

### Restricted Mean Survival Time (RMST)

RMST is the area under the survival curve up to a time horizon tau:

```
RMST(tau) = integral from 0 to tau of S(t) dt
```

The difference in RMST between groups provides a clinically interpretable measure of treatment effect that does not require the PH assumption.

## Key Concepts

### Censoring

- **Right censoring**: The event has not occurred by the end of follow-up. Most common type.
- **Left censoring**: The event occurred before the start of observation (e.g., seroconversion before enrollment).
- **Interval censoring**: The event is known only to have occurred within an interval (e.g., between clinic visits).
- **Informative censoring**: Censoring is related to the event process, violating assumptions.

### Risk Set

At any time t, the risk set R(t) consists of all subjects who are still under observation and have not yet experienced the event. The size of the risk set directly affects precision.

### Median Survival Time

The median survival time is the time at which S(t) = 0.5. It is more robust than the mean for skewed survival distributions and is a standard reporting metric in oncology trials.

## Assumptions

1. **Non-informative censoring**: The censoring mechanism is independent of the event process, conditional on covariates.
2. **Proportional hazards (Cox model)**: The hazard ratio between groups is constant over time.
3. **Correct model specification**: Covariates are correctly specified (linearity on the log-hazard scale for Cox).
4. **Independent observations**: Subjects' event times are independent (violated in clustered data).

### Checking the PH Assumption

- **Schoenfeld residuals**: Regress scaled Schoenfeld residuals against time. A non-zero slope suggests violation. The `cox.zph()` test formalizes this.
- **Log-log plots**: Plot log(-log(S(t))) vs log(t). Parallel curves support PH.
- **Time-varying coefficients**: Fit models with interaction terms between covariates and time.

## Variants and Extensions

### Time-Varying Covariates

Covariates that change during follow-up (e.g., lab values, treatment switching) can be incorporated by splitting each subject's record at times when covariates change. The extended Cox model handles these via the counting process formulation (start, stop, event).

### Frailty Models

Frailty models add a random effect to the Cox model to account for unobserved heterogeneity or clustering:

```
h_ij(t) = h_0(t) * Z_i * exp(X_ij * beta)
```

where Z_i is a frailty term (typically gamma or log-normal distributed) for cluster i. Shared frailty models handle multi-center trials or recurrent events.

### Landmark Analysis

Landmark analysis addresses immortal time bias by defining a landmark time point and restricting analysis to subjects still at risk at that time. Covariates measured by the landmark time are used for subsequent prediction. This avoids conditioning on the future.

### Multi-State Models

Generalize survival analysis to transitions between multiple states (e.g., healthy -> diseased -> dead). Transition-specific hazards are modeled, and state occupation probabilities are estimated via the Aalen-Johansen estimator.

### Cure Models

Mixture cure models assume a fraction of the population will never experience the event:

```
S(t) = pi + (1 - pi) * S_u(t)
```

where pi is the cure fraction and S_u(t) is the survival function for the uncured.

## When to Use This Method

- **Primary use**: Any study where the outcome is the time to a specific event and censoring is present.
- **Prefer KM and log-rank**: For descriptive survival comparisons and simple group comparisons without adjustment.
- **Prefer Cox PH**: When adjusting for covariates and the PH assumption is reasonable.
- **Prefer parametric models**: When smooth hazard estimation is needed, for extrapolation beyond observed data, or for health economic modeling.
- **Prefer AFT**: When the PH assumption fails and a time-ratio interpretation is more natural.
- **Prefer RMST**: As a supplement or alternative when PH does not hold, or for easier clinical interpretation.

## Strengths and Limitations

### Strengths
- Handles censored data naturally, using all available information.
- KM is non-parametric and widely understood.
- Cox model is flexible, requiring no distributional assumptions on the baseline hazard.
- Well-developed diagnostic tools and extensions.
- Established regulatory acceptance for pivotal clinical trials.

### Limitations
- Relies on non-informative censoring, which can be difficult to verify.
- Cox PH assumption may be violated, particularly for long follow-up periods.
- Interpretation of hazard ratios can be non-collapsible and difficult for non-statisticians.
- Parametric models require correct distribution specification.
- Single-event analysis does not account for competing risks.

## Key References

1. Kaplan EL, Meier P. Nonparametric estimation from incomplete observations. *J Am Stat Assoc.* 1958;53(282):457-481.
2. Cox DR. Regression models and life-tables (with discussion). *J R Stat Soc Series B.* 1972;34(2):187-220.
3. Collett D. *Modelling Survival Data in Medical Research.* 3rd ed. Chapman & Hall/CRC; 2015.
4. Klein JP, Moeschberger ML. *Survival Analysis: Techniques for Censored and Truncated Data.* 2nd ed. Springer; 2003.
5. Royston P, Parmar MKB. Restricted mean survival time: an alternative to the hazard ratio for the design and analysis of randomized trials with a time-to-event outcome. *BMC Med Res Methodol.* 2013;13:152.
6. Therneau TM, Grambsch PM. *Modeling Survival Data: Extending the Cox Model.* Springer; 2000.
7. Kalbfleisch JD, Prentice RL. *The Statistical Analysis of Failure Time Data.* 2nd ed. Wiley; 2002.
