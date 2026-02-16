# Adaptive Trial Design — Theory

## Introduction

Adaptive clinical trial designs allow pre-planned modifications to one or more aspects
of the trial — sample size, randomisation ratio, patient population, or endpoints —
based on accumulating data, without undermining the validity and integrity of the study.
These designs improve efficiency by stopping early for efficacy or futility, re-sizing
the trial when initial variance assumptions are wrong, or adapting the allocation to
favour the better-performing arm.

Regulatory guidance (FDA 2019, EMA Reflection Paper 2007) requires that adaptations be
pre-specified in the protocol and that the operating characteristics be demonstrated
through simulation.

## Mathematical Foundation

### Group Sequential Designs

A group sequential design performs K planned interim analyses at information fractions
`t_1, t_2, ..., t_K = 1`. At each analysis k, a test statistic Z_k is compared against
upper and lower boundaries:

```
Reject H0 if Z_k >= u_k        (efficacy boundary)
Stop for futility if Z_k <= l_k (futility boundary)
Continue if l_k < Z_k < u_k
```

The overall Type I error alpha is controlled by distributing ("spending") it across
the K looks.

### Alpha Spending Functions

The alpha spending function `alpha*(t)` maps the information fraction t in [0, 1] to
the cumulative Type I error spent by that point:

- `alpha*(0) = 0`, `alpha*(1) = alpha`.
- The incremental alpha at look k: `Delta_k = alpha*(t_k) - alpha*(t_{k-1})`.

Common spending functions:

| Name | Formula | Behaviour |
|------|---------|-----------|
| **O'Brien-Fleming** | `alpha*(t) = 2 - 2*Phi(z_{alpha/2} / sqrt(t))` | Very conservative early, liberal late |
| **Pocock** | `alpha*(t) = alpha * ln(1 + (e-1)*t)` | Uniform spending |
| **Lan-DeMets (OF approx)** | `alpha*(t) = 2[1 - Phi(z_{alpha/2}/sqrt(t))]` | Approximates OF boundaries |
| **Hwang-Shih-DeCani** | `alpha*(t) = alpha*(1-exp(-gamma*t))/(1-exp(-gamma))` | Parameterised family |

### Information Fraction

The information fraction at analysis k is:

```
t_k = I_k / I_K
```

where `I_k` is the statistical information (e.g., the number of events for a survival
trial, or the number of patients for a continuous endpoint). For a Z-test on means:

```
I_k = n_k / sigma^2
```

### Boundary Computation

Given the spending function and information fractions, boundaries are computed
recursively using the joint distribution of (Z_1, ..., Z_K), which is multivariate
normal with known covariance:

```
Cov(Z_j, Z_k) = sqrt(I_j / I_k)   for j <= k
```

At each look, the incremental boundary is found by solving for u_k such that:

```
P(Z_1 < u_1, ..., Z_{k-1} < u_{k-1}, Z_k >= u_k | H0) = Delta_k
```

This is computed numerically via the Armitage-McPherson-Rowe recursion.

## Key Concepts

### Conditional Power and Futility

**Conditional power** is the probability of rejecting H0 at the final analysis given
the data observed so far:

```
CP(theta) = P(Reject H0 at look K | Z_k = z_k, true effect = theta)
```

When computed under the current trend (`theta_hat`), low conditional power (e.g., <10%)
suggests futility. When computed under the original design alternative, it assesses
whether the trial is on track.

### Predictive Probability (Bayesian Futility)

Instead of conditioning on a fixed theta, the predictive probability averages over the
posterior distribution of theta:

```
PP = integral CP(theta) * pi(theta | data) d(theta)
```

This accounts for uncertainty in the treatment effect and is often more calibrated than
conditional power.

### Adaptive Sample Size Re-estimation

If the interim variance estimate differs from the planning assumption, the sample size
can be increased (rarely decreased) to maintain the target power. The Cui-Hung-Wang
(CHW) method preserves the Type I error by using pre-specified combination test
statistics:

```
T = w_1 * Z_1 + w_2 * Z_2      (fixed weights based on original sample fractions)
```

Alternatively, the inverse-normal method or Fisher combination test can be used.

### Response-Adaptive Randomisation (RAR)

In RAR, the allocation probability is updated based on accumulating outcome data,
directing more patients to the better-performing arm. Common rules:

- **Thompson sampling:** allocate proportional to the posterior probability that each
  arm is best.
- **Doubly adaptive biased coin:** targets a specified allocation based on response rates.
- **Bayesian adaptive randomisation:** allocation proportional to
  `P(theta_k = max(theta_1, ..., theta_K) | data)`.

RAR raises ethical appeal (fewer patients on inferior arms) but can inflate Type I
error and reduce power if not combined with appropriate inference.

## Assumptions

1. **Pre-specification:** All adaptations must be fully specified in the protocol before
   unblinding. Post-hoc adaptations are not adaptive designs — they are protocol
   amendments.
2. **Information monitoring:** Information fractions are correctly calculated.
3. **Independent increments:** The test statistic increments between looks are
   independent (ensured by canonical joint distribution).
4. **No operational bias:** The adaptation does not introduce bias (e.g., through
   differential knowledge of treatment assignment).

## Variants and Extensions

### Seamless Phase II/III Designs

Combines dose-selection (Phase II) and confirmatory testing (Phase III) in a single
trial. At the interim, the best dose is selected, and the trial continues with only
that dose and control. Alpha is controlled using closed testing or combination tests.

### Platform Trials (Basket/Umbrella)

- **Basket trial:** One treatment tested across multiple tumour types (indications).
- **Umbrella trial:** One disease with multiple biomarker-defined subgroups, each tested
  with a matched therapy.
- **Platform trial:** A standing infrastructure that allows adding/dropping arms over
  time (e.g., I-SPY 2, RECOVERY).

Platform trials use a shared control arm and Bayesian decision rules for arm graduation
or dropping. Master protocols and perpetual platform designs reduce startup costs.

### Bayesian Adaptive Designs

Bayesian designs use posterior probabilities for decision-making:

```
Stop for efficacy if P(theta > 0 | data) > theta_U   (e.g., 0.99)
Stop for futility if P(theta > 0 | data) < theta_L   (e.g., 0.05)
```

The design's operating characteristics (Type I error, power) are evaluated by simulation
under a range of true effect sizes. The frequentist properties must still satisfy
regulatory requirements.

### DSMB Role

The Data Safety Monitoring Board (DSMB) is an independent committee that reviews interim
data. They may recommend:

- Continuing the trial as planned.
- Stopping for efficacy, futility, or safety.
- Modifying the sample size.

The DSMB operates under a charter that specifies stopping guidelines aligned with the
statistical design.

## When to Use This Method

- Large confirmatory trials where early stopping can save resources and reduce patient
  exposure to inferior treatments.
- Dose-finding studies where seamless designs improve efficiency.
- Rare diseases where sample size is limited and adaptive randomisation is valuable.
- Pandemic settings where rapid evidence generation is critical (e.g., RECOVERY trial).
- Health technology assessments requiring interim evidence for conditional approval.

## Strengths and Limitations

### Strengths
- Ethical: reduces patient exposure to inferior treatments.
- Efficient: uses fewer patients on average than fixed designs.
- Flexible: accommodates uncertainty in design parameters.
- Regulatory acceptance is established (FDA/EMA guidance available).

### Limitations
- Logistical complexity: requires real-time data monitoring infrastructure.
- Statistical complexity: boundary computation and combination tests are non-trivial.
- Potential for operational bias if blinding is compromised.
- Simulation burden: operating characteristics must be verified by extensive simulation.
- Publication bias: trials stopped early for efficacy tend to overestimate effects.

## Key References

1. Jennison, C., & Turnbull, B. W. (2000). *Group Sequential Methods with Applications
   to Clinical Trials*. Chapman & Hall/CRC.
2. FDA (2019). Adaptive Designs for Clinical Trials of Drugs and Biologics: Guidance for
   Industry.
3. Lan, K. K. G., & DeMets, D. L. (1983). Discrete sequential boundaries for clinical
   trials. *Biometrika*, 70(3), 659-663.
4. Wassmer, G., & Brannath, W. (2016). *Group Sequential and Confirmatory Adaptive
   Designs in Clinical Trials*. Springer.
5. Berry, S. M., et al. (2011). *Bayesian Adaptive Methods for Clinical Trials*. CRC.
6. Pallmann, P., et al. (2018). Adaptive designs in clinical trials: why use them, and
   how to run and report them. *BMC Medicine*, 16, 29.
