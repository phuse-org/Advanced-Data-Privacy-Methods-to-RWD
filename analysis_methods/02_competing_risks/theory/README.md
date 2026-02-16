# Competing Risks — Theory

## Introduction

Competing risks arise when subjects are at risk of experiencing more than one type of event, and the occurrence of one event precludes or fundamentally alters the probability of the other events. In health research, this is ubiquitous: a patient in a cancer trial may die from the cancer (event of interest), die from cardiovascular disease (competing event), or remain alive (censored). Standard survival methods that treat competing events as censored observations produce biased estimates of the event-specific cumulative incidence.

Competing risks analysis provides the appropriate statistical framework for estimating the probability of a specific event type in the presence of other events. It is critical in oncology (cancer-specific vs. other-cause mortality), transplant medicine (graft failure vs. death), cardiology (cardiovascular death vs. non-cardiovascular death), and geriatric research where comorbid conditions create multiple event pathways.

## Mathematical Foundation

### Setting and Notation

Let T denote the time to the first event, and D denote the type of event, where D takes values in {1, 2, ..., K} for K competing event types. The observed data for subject i are (T_i, D_i, delta_i), where delta_i indicates whether the event was observed (delta = 1) or right-censored (delta = 0).

### Cause-Specific Hazard

The cause-specific hazard for event type k is the instantaneous rate of occurrence of event k, given that no event of any type has occurred:

```
h_k(t) = lim_{dt->0} P(t <= T < t + dt, D = k | T >= t) / dt
```

The overall hazard is the sum of all cause-specific hazards:

```
h(t) = sum_{k=1}^{K} h_k(t)
```

The overall survival function is:

```
S(t) = exp(-integral from 0 to t of h(u) du) = exp(-sum_{k=1}^{K} H_k(t))
```

where H_k(t) is the cause-specific cumulative hazard.

### Cumulative Incidence Function (CIF)

The cumulative incidence function (also called the subdistribution function) for event k gives the probability of experiencing event k by time t:

```
F_k(t) = P(T <= t, D = k) = integral from 0 to t of h_k(u) * S(u-) du
```

where S(u-) is the overall survival function just before time u. Key properties:

- The CIF is bounded: 0 <= F_k(t) <= 1 for all t and k.
- The CIFs sum with the overall survival: sum_{k=1}^{K} F_k(t) + S(t) = 1.
- The CIF for event k depends on ALL cause-specific hazards, not just h_k(t).

This last point is critical: unlike the standard survival setting, the probability of event k is influenced by the hazards of all competing events.

### Non-Parametric CIF Estimation (Aalen-Johansen)

The non-parametric estimator of the CIF, often attributed to Aalen and Johansen (1978), is:

```
F_hat_k(t) = sum over t_j <= t of [d_kj / n_j] * S_hat(t_j-)
```

where d_kj is the number of type-k events at time t_j, n_j is the risk set size, and S_hat(t_j-) is the KM-type overall survival estimate just before t_j. In the two-event case, this is equivalent to the estimator described by Kalbfleisch and Prentice.

### Subdistribution Hazard (Fine-Gray Model)

Fine and Gray (1999) introduced a regression model based on the subdistribution hazard:

```
lambda_k(t) = lim_{dt->0} P(t <= T < t + dt, D = k | T >= t or (T < t and D != k)) / dt
```

The key distinction: the risk set for the subdistribution hazard includes subjects who have already experienced a competing event (they are "kept in" the risk set with zero probability of experiencing event k). This is a mathematical construction that directly models the CIF:

```
F_k(t | X) = 1 - exp(-Lambda_k(t | X))
```

where Lambda_k is the cumulative subdistribution hazard. The Fine-Gray model specifies:

```
lambda_k(t | X) = lambda_{k,0}(t) * exp(beta * X)
```

The coefficients beta have interpretations in terms of the CIF: exp(beta) is the subdistribution hazard ratio, and positive beta implies higher cumulative incidence.

### Cause-Specific Cox Model

An alternative regression approach applies the standard Cox model to each cause-specific hazard separately:

```
h_k(t | X) = h_{k,0}(t) * exp(gamma_k * X)
```

For event k, subjects who experience competing events are censored at their event time. The model is fit separately for each event type.

## Key Concepts

### One Minus KM Is Wrong

A fundamental mistake in competing risks analysis is using the Kaplan-Meier complement (1 - KM) to estimate the cumulative incidence of a specific event. When competing events are censored in a standard KM analysis, the complement overestimates the true cumulative incidence because it assumes that censored subjects (including those who experienced competing events) would eventually experience the event of interest.

### CIF vs. KM Complement

- **1 - KM(t)**: Estimates the probability of event k in a hypothetical world where competing events do not exist. This is the marginal or "net" probability.
- **CIF(t)**: Estimates the probability of event k in the real world where competing events do exist. This is the "crude" or "real-world" probability.

The CIF is always less than or equal to 1 - KM, and the difference grows as competing events become more frequent.

### Etiologic vs. Prognostic Research Questions

The choice between cause-specific and subdistribution approaches depends on the research question:

- **Etiologic (causal) questions**: "Does treatment affect the biological mechanism leading to event k?" Use **cause-specific hazard** models. These directly model the rate at which event k occurs among those still at risk of any event.

- **Prognostic (predictive) questions**: "What is the predicted probability of event k by time t for a patient with characteristics X?" Use **subdistribution hazard** (Fine-Gray) models. These directly model the cumulative incidence and are better for prediction and risk communication.

In practice, reporting both approaches provides the most complete picture.

## Assumptions

1. **Non-informative censoring**: Right censoring is independent of all event types, conditional on covariates.
2. **Independent competing risks** (for cause-specific models): The cause-specific hazard for event k does not depend on the competing event process, conditional on covariates. This assumption is untestable.
3. **Proportional hazards**: For both the cause-specific Cox model and the Fine-Gray model, the respective hazards must be proportional over time.
4. **Correct covariate specification**: Covariates are correctly modeled (e.g., linearity on the log-hazard scale).

Note: The cause-specific and subdistribution proportional hazards assumptions cannot both hold simultaneously unless the covariate has no effect. If one is proportional, the other generally is not.

## Variants and Extensions

### Multi-State Models

Competing risks are a special case of multi-state models with an initial state and K absorbing states. More generally, multi-state models allow transitions between non-absorbing states (e.g., healthy -> sick -> dead, with possible recovery). The Aalen-Johansen estimator generalizes to estimate state occupation probabilities.

### Subdistribution Hazard with Time-Varying Covariates

Extensions of the Fine-Gray model to handle time-varying covariates exist but are computationally more complex because the artificially retained subjects in the risk set need appropriate covariate histories.

### Random Effects (Frailty) Models

Shared frailty can be added to both cause-specific and subdistribution models to account for clustering (e.g., multi-center studies).

### Parametric Competing Risks Models

Parametric specifications of cause-specific hazards (e.g., Weibull) can be used. The CIF is then obtained by integration. These are useful for extrapolation in health economic modeling.

### Inverse Probability of Censoring Weighting (IPCW)

IPCW methods can be applied in competing risks settings to handle dependent censoring or to construct pseudo-observations for regression on the CIF scale.

## When to Use This Method

- **Use competing risks** when subjects can experience events that prevent the event of interest from occurring (e.g., death prevents disease progression).
- **Use cause-specific models** for understanding etiology and the direct effect of covariates on event rates.
- **Use Fine-Gray models** for prediction, risk communication, and when the primary goal is to estimate cumulative incidence as a function of covariates.
- **Report both** cause-specific and subdistribution analyses for a comprehensive understanding.
- **Do not use** standard KM with 1 - KM to estimate cumulative incidence when competing risks are present.

## Strengths and Limitations

### Strengths
- Properly accounts for the presence of competing events, avoiding bias in cumulative incidence estimation.
- The CIF provides a clinically meaningful probability of experiencing the event of interest.
- Well-developed non-parametric (Aalen-Johansen) and semi-parametric (Fine-Gray, cause-specific Cox) methods.
- Complementary perspectives from cause-specific and subdistribution approaches.
- Multi-state framework provides a unifying structure.

### Limitations
- The independent competing risks assumption is untestable from observed data.
- The Fine-Gray model's risk set construction (retaining subjects who experienced competing events) can be counterintuitive.
- The subdistribution hazard ratio lacks a direct causal interpretation.
- Cause-specific and subdistribution proportional hazards cannot both hold simultaneously.
- Sample size requirements increase with the number of event types.
- Software implementation is less mature than standard survival analysis in some environments.

## Key References

1. Fine JP, Gray RJ. A proportional hazards model for the subdistribution of a competing risk. *J Am Stat Assoc.* 1999;94(446):496-509.
2. Putter H, Fiocco M, Geskus RB. Tutorial in biostatistics: competing risks and multi-state models. *Stat Med.* 2007;26(11):2389-2430.
3. Austin PC, Lee DS, Fine JP. Introduction to the analysis of survival data in the presence of competing risks. *Circulation.* 2016;133(6):601-609.
4. Aalen OO, Johansen S. An empirical transition matrix for non-homogeneous Markov chains based on censored observations. *Scand J Stat.* 1978;5(3):141-150.
5. Latouche A, Allignol A, Beyersmann J, Labopin M, Fine JP. A competing risks analysis should report results on all cause-specific hazards and cumulative incidence functions. *J Clin Epidemiol.* 2013;66(6):648-653.
6. Beyersmann J, Allignol A, Schumacher M. *Competing Risks and Multistate Models with R.* Springer; 2012.
7. Kalbfleisch JD, Prentice RL. *The Statistical Analysis of Failure Time Data.* 2nd ed. Wiley; 2002.
