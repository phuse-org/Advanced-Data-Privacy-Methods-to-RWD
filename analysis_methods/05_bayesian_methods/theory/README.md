# Bayesian Methods — Theory

## Introduction

Bayesian statistics provides a coherent framework for combining prior knowledge with observed data to make probabilistic statements about unknown quantities. In contrast to frequentist methods, which interpret probability as long-run frequency, Bayesian methods interpret probability as a degree of belief, updated in light of evidence.

In health research, Bayesian methods are increasingly valued for their ability to incorporate historical data, provide intuitive probability statements about treatment effects ("there is a 95% probability that the treatment effect lies between X and Y"), handle complex hierarchical structures, and support adaptive clinical trial designs. Regulatory agencies including the FDA have issued guidance supporting Bayesian approaches for medical device trials and adaptive designs.

## Mathematical Foundation

### Bayes' Theorem

The foundation of Bayesian inference is Bayes' theorem:

```
P(theta | data) = P(data | theta) * P(theta) / P(data)
```

In continuous form:

```
pi(theta | y) = L(y | theta) * pi(theta) / integral of L(y | theta) * pi(theta) d_theta
```

where:
- **pi(theta | y)**: Posterior distribution — our updated belief about parameters after observing data.
- **L(y | theta)**: Likelihood — the probability of the data given the parameters.
- **pi(theta)**: Prior distribution — our belief about parameters before seeing data.
- **P(data)**: Marginal likelihood (normalizing constant) — ensures the posterior integrates to 1.

The posterior is proportional to the prior times the likelihood:

```
pi(theta | y) is proportional to L(y | theta) * pi(theta)
```

### Prior Distributions

Priors encode knowledge or assumptions about parameters before observing data:

**Non-informative (vague) priors**: Express minimal prior information.
- Flat prior: pi(theta) is proportional to 1 (uniform)
- Jeffreys prior: pi(theta) is proportional to sqrt(det(I(theta))), where I is the Fisher information

**Weakly informative priors**: Constrain parameters to reasonable ranges without dominating the likelihood.
- Normal(0, 10) for regression coefficients (on the log-odds scale for logistic regression)
- Half-Cauchy(0, 2.5) for standard deviations
- Student-t(3, 0, 2.5) for robust regression coefficients

**Informative priors**: Incorporate substantial existing knowledge.
- Prior from a previous clinical trial
- Expert elicitation
- Meta-analytic priors

**Conjugate priors**: Priors that yield a posterior in the same distributional family.
- Beta prior for binomial likelihood yields Beta posterior
- Normal prior for normal likelihood (known variance) yields Normal posterior
- Gamma prior for Poisson likelihood yields Gamma posterior

### Posterior Distribution

The posterior distribution is the complete inferential summary in Bayesian statistics. Key summaries:

**Posterior mean**: E[theta | y] — the mean of the posterior distribution.

**Posterior median**: The 50th percentile; robust to skewness.

**Maximum a Posteriori (MAP)**: The mode of the posterior; the single most probable value.

**Credible intervals**: A 95% credible interval [a, b] satisfies P(a < theta < b | y) = 0.95. Unlike frequentist confidence intervals, this is a direct probability statement about the parameter.

- **Equal-tailed interval**: 2.5th to 97.5th percentile.
- **Highest Posterior Density (HPD) interval**: The shortest interval containing 95% of the posterior mass.

### Credible Intervals vs. Confidence Intervals

A 95% **credible interval** means: "Given the data and prior, there is a 95% probability the parameter lies in this interval."

A 95% **confidence interval** means: "If we repeated the experiment many times, 95% of the computed intervals would contain the true parameter."

The Bayesian interpretation is more intuitive for clinical decision-making.

### Markov Chain Monte Carlo (MCMC)

For most models, the posterior cannot be computed analytically. MCMC algorithms generate samples from the posterior distribution.

**Gibbs Sampling**: Iteratively samples each parameter from its full conditional distribution, holding others fixed. Works well when full conditionals are available in closed form.

**Metropolis-Hastings**: Proposes new parameter values from a proposal distribution and accepts/rejects based on the posterior ratio. More general but can be slow.

**Hamiltonian Monte Carlo (HMC) / NUTS**: Uses gradient information to propose moves in parameter space, exploring the posterior more efficiently than random-walk MH. The No-U-Turn Sampler (NUTS) automatically tunes the trajectory length. Stan implements NUTS and is the engine behind `brms` and `rstanarm`.

### Convergence Diagnostics

MCMC samples must be checked for convergence to the target posterior:

**Rhat (Gelman-Rubin statistic)**: Compares between-chain and within-chain variance. Rhat < 1.01 indicates convergence.

**Effective Sample Size (ESS)**: The equivalent number of independent samples. Low ESS indicates high autocorrelation. Aim for ESS > 400 for reliable inference.

**Trace plots**: Visual inspection of the MCMC chains. Well-mixed chains look like "hairy caterpillars" with no trends or stuck regions.

**Divergent transitions** (HMC-specific): Indicate regions of the posterior that are difficult to explore. Address by reparameterizing the model or increasing `adapt_delta`.

### Bayesian Hierarchical Models

Hierarchical (multilevel) models are a natural application of Bayesian methods. For data from J groups:

```
y_ij | theta_j ~ f(y | theta_j)          (data level)
theta_j | mu, tau ~ g(theta | mu, tau)    (group level)
mu, tau ~ pi(mu, tau)                      (hyperprior level)
```

The group-level parameters theta_j are "partially pooled" toward the overall mean mu. Groups with less data borrow more strength from others. This is particularly useful for multi-center clinical trials, meta-analyses, and rare disease studies.

### Model Comparison

**Bayes Factors**: The ratio of marginal likelihoods for two models:
```
BF_12 = P(data | M1) / P(data | M2)
```
BF > 10 is "strong evidence" for M1 (Kass and Raftery scale).

**WAIC (Widely Applicable Information Criterion)**: A Bayesian analog of AIC that accounts for the effective number of parameters. Lower is better.

**LOO-CV (Leave-One-Out Cross-Validation)**: Approximated efficiently using Pareto-smoothed importance sampling (PSIS-LOO). The `loo` package provides `elpd_loo` (expected log pointwise predictive density).

## Key Concepts

### Prior-Data Conflict

When the prior and data strongly disagree, the posterior may be heavily influenced by the prior (if the sample size is small) or by the data (if the sample is large). Formal diagnostics such as the prior-data conflict check compare the observed data's marginal likelihood under the prior to expected values.

### Prior Sensitivity Analysis

Varying the prior and examining how the posterior changes is essential for credible Bayesian analysis. If conclusions are robust to a range of reasonable priors, confidence in the results is strengthened.

### Bayesian Adaptive Trial Designs

Bayesian methods naturally support adaptive designs that modify the trial based on accumulating data:
- **Bayesian response-adaptive randomization**: Allocate more patients to the better-performing arm.
- **Bayesian interim analyses**: Compute posterior probability of efficacy/futility at interim looks without multiplicity adjustments.
- **Bayesian platform trials**: Add or drop experimental arms based on posterior probabilities.

### Posterior Predictive Checking

Generate simulated datasets from the posterior predictive distribution and compare to the observed data. Systematic discrepancies indicate model misspecification.

## Assumptions

1. **Prior specification**: The prior must be explicitly stated and justified. Sensitivity analyses are expected.
2. **Likelihood specification**: The data-generating model must be correctly specified.
3. **MCMC convergence**: Inference is valid only if the MCMC chains have converged to the stationary distribution.
4. **Sufficient computation**: Enough posterior samples must be drawn for accurate summarization.

## Variants and Extensions

### Bayesian Nonparametrics

Dirichlet process mixtures and Gaussian process priors allow flexible modeling without fixing the number of parameters.

### Bayesian Variable Selection

Spike-and-slab priors place a mixture of a point mass at zero and a diffuse prior on each coefficient, performing variable selection within the Bayesian framework.

### Bayesian Survival Analysis

Bayesian extensions of Cox PH, parametric survival, and cure models allow prior incorporation and full posterior inference for survival endpoints.

### Approximate Bayesian Methods

When MCMC is computationally prohibitive:
- **Variational Bayes (VB)**: Approximates the posterior with a simpler distribution. Faster but potentially less accurate.
- **Integrated Nested Laplace Approximation (INLA)**: Fast and accurate for latent Gaussian models.

## When to Use This Method

- **Small sample sizes**: Priors stabilize estimation when data are limited.
- **Hierarchical/multilevel data**: Bayesian methods handle partial pooling naturally.
- **Incorporating prior information**: Historical data, expert knowledge, or meta-analytic evidence.
- **Adaptive clinical trials**: Natural framework for sequential updating.
- **Complex models**: Bayesian software (Stan) handles models that are difficult to fit with maximum likelihood.
- **Direct probability statements**: When stakeholders want to know "the probability that the treatment works."

## Strengths and Limitations

### Strengths
- Provides intuitive probability statements about parameters.
- Naturally incorporates prior information and handles small samples.
- Coherent framework for prediction, model comparison, and decision-making.
- Handles complex hierarchical structures and missing data elegantly.
- Supports adaptive trial designs without multiplicity corrections.

### Limitations
- Requires explicit prior specification, which can be subjective and controversial.
- MCMC computation can be slow for large datasets or complex models.
- Convergence diagnostics require expertise and careful checking.
- Prior sensitivity may be difficult to fully explore.
- Communication of Bayesian results to non-statistical audiences can be challenging.

## Key References

1. Gelman A, Carlin JB, Stern HS, Dunson DB, Vehtari A, Rubin DB. *Bayesian Data Analysis.* 3rd ed. Chapman & Hall/CRC; 2013.
2. McElreath R. *Statistical Rethinking: A Bayesian Course with Examples in R and Stan.* 2nd ed. Chapman & Hall/CRC; 2020.
3. Kruschke JK. *Doing Bayesian Data Analysis.* 2nd ed. Academic Press; 2015.
4. Carpenter B, Gelman A, Hoffman MD, et al. Stan: A probabilistic programming language. *J Stat Softw.* 2017;76(1):1-32.
5. Vehtari A, Gelman A, Gabry J. Practical Bayesian model evaluation using leave-one-out cross-validation and WAIC. *Stat Comput.* 2017;27(5):1413-1432.
6. Berry SM, Carlin BP, Lee JJ, Muller P. *Bayesian Adaptive Methods for Clinical Trials.* Chapman & Hall/CRC; 2010.
7. Spiegelhalter DJ, Abrams KR, Myles JP. *Bayesian Approaches to Clinical Trials and Health-Care Evaluation.* Wiley; 2004.
