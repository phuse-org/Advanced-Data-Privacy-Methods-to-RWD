# Machine Learning for Clinical Prediction — Theory

## Introduction

Machine learning (ML) methods for clinical prediction aim to develop models that estimate a patient's risk of a future outcome (e.g., mortality, disease recurrence, hospital readmission) based on their clinical characteristics. Unlike traditional explanatory models that seek to understand causal mechanisms, prediction models prioritize out-of-sample accuracy. ML extends the traditional statistical toolkit (logistic regression, Cox regression) with algorithms capable of capturing complex non-linear relationships and high-order interactions, potentially improving prediction in settings with many features and large sample sizes.

Clinical prediction models directly impact patient care through risk stratification, treatment selection, and resource allocation. The TRIPOD (Transparent Reporting of a multivariable prediction model for Individual Prognosis or Diagnosis) guidelines provide a framework for rigorous development and reporting.

## Mathematical Foundation

### Supervised Learning Framework

Given training data $\{(x_i, y_i)\}_{i=1}^{n}$ where $x_i \in \mathbb{R}^p$ are patient features and $y_i$ is the outcome, we seek a function $\hat{f}(x)$ that minimizes expected prediction error:

$$\hat{f} = \arg\min_{f \in \mathcal{F}} \frac{1}{n}\sum_{i=1}^{n} L(y_i, f(x_i)) + \lambda \Omega(f)$$

where $L$ is the loss function (e.g., log-loss for classification, squared error for regression), $\Omega(f)$ is a regularization term, and $\lambda$ controls the bias-variance tradeoff.

### Logistic Regression as Baseline

For binary outcomes: $P(Y=1|X=x) = \frac{1}{1 + e^{-(\beta_0 + \beta^T x)}}$

Penalized logistic regression (LASSO, ridge, elastic net) extends this by adding $\lambda_1 \|\beta\|_1 + \lambda_2 \|\beta\|_2^2$ to the loss, enabling variable selection and shrinkage.

### Random Forests

An ensemble of $B$ decision trees, each trained on a bootstrap sample with a random subset of $m$ features considered at each split. The prediction aggregates individual tree predictions:

- Classification: majority vote or averaged probabilities.
- Regression: averaged predictions.
- Key hyperparameters: number of trees ($B$), features per split ($m$), minimum node size, maximum depth.

### Gradient Boosting (XGBoost, LightGBM)

Sequentially builds an additive model: $\hat{f}(x) = \sum_{m=1}^{M} \eta \cdot h_m(x)$, where each $h_m$ is a shallow decision tree fit to the negative gradient of the loss function (pseudo-residuals). The learning rate $\eta$ controls the contribution of each tree.

- **XGBoost**: Uses regularized objective, second-order Taylor approximation, column/row subsampling, and supports custom objectives.
- **LightGBM**: Uses histogram-based splitting and leaf-wise tree growth for faster training on large datasets.

### Support Vector Machines

SVMs find the maximum-margin hyperplane separating classes: $\min \frac{1}{2}\|w\|^2 + C\sum \xi_i$ subject to $y_i(w^T \phi(x_i) + b) \geq 1 - \xi_i$. Kernel functions $K(x_i, x_j) = \phi(x_i)^T \phi(x_j)$ enable non-linear decision boundaries.

### Neural Networks

Composed of layers of interconnected nodes with non-linear activations: $h^{(l)} = \sigma(W^{(l)} h^{(l-1)} + b^{(l)})$. Deep learning architectures (convolutional, recurrent, transformers) can handle images, time series, and structured clinical data.

## Key Concepts

### Model Validation

- **Train/test split**: Hold out 20-30% of data for final evaluation. Simple but wastes data.
- **K-fold cross-validation**: Partition data into $k$ folds, train on $k-1$, validate on the held-out fold. Repeat $k$ times and average performance.
- **Repeated CV**: Repeat the $k$-fold procedure multiple times with different random partitions to reduce variance.
- **Bootstrap validation (0.632+)**: Draw bootstrap samples for training, evaluate on out-of-bag observations. The 0.632+ estimator corrects for optimistic bias.
- **Temporal validation**: Train on earlier data, validate on later data. Essential for models deployed in evolving clinical settings.
- **External validation**: Evaluate on data from a completely independent site, population, or time period. The gold standard for generalizability.

### Performance Metrics

- **AUC / c-statistic**: Area under the ROC curve. Measures discrimination — the probability that a randomly chosen positive case is ranked higher than a negative case. AUC of 0.5 indicates no discrimination; 1.0 is perfect.
- **Calibration**: Agreement between predicted probabilities and observed event rates. Assessed via calibration plots (predicted vs. observed), Hosmer-Lemeshow test, or calibration-in-the-large/slope.
- **Brier score**: $BS = \frac{1}{n}\sum(y_i - \hat{p}_i)^2$. Combines discrimination and calibration. Lower is better.
- **Sensitivity / Specificity / PPV / NPV**: Threshold-dependent metrics; clinically interpretable but require a chosen cutpoint.
- **Net Reclassification Improvement (NRI)**: Quantifies the improvement in risk classification compared to a reference model.

### Decision Curve Analysis

Decision curve analysis (DCA) evaluates the clinical utility of a prediction model by plotting net benefit across a range of threshold probabilities:

$$NB(p_t) = \frac{TP}{n} - \frac{FP}{n} \cdot \frac{p_t}{1 - p_t}$$

A model is clinically useful if its net benefit exceeds both the "treat all" and "treat none" strategies across a clinically relevant range of thresholds.

### Feature Importance and Interpretability

- **SHAP (SHapley Additive exPlanations)**: Based on game-theoretic Shapley values, SHAP provides consistent, locally accurate feature attributions for any model. SHAP values decompose each prediction into additive contributions from each feature.
- **Permutation importance**: Measures the increase in prediction error when a single feature is randomly shuffled, breaking its relationship with the outcome.
- **Partial dependence plots (PDPs)**: Show the marginal effect of a feature on the predicted outcome, averaging over all other features.

### Class Imbalance

Clinical outcomes are often rare (e.g., 5% mortality). Standard algorithms may perform poorly on the minority class.

- **SMOTE (Synthetic Minority Over-sampling Technique)**: Generates synthetic minority class examples by interpolating between existing ones.
- **Class weights**: Assign higher misclassification cost to the minority class.
- **Threshold adjustment**: Optimize the classification threshold for a specific metric (e.g., sensitivity).
- **Note**: AUC is invariant to class imbalance; calibration is not. Recalibration after resampling is essential.

## Assumptions

1. **Representativeness**: Training data must be representative of the population where the model will be deployed.
2. **Stability**: The relationships between features and outcomes do not change substantially over time (stationarity). Violated in evolving clinical practice (dataset shift).
3. **Feature availability**: All features used in the model must be available at the time of prediction in clinical practice.
4. **Missing data handled appropriately**: Models require complete input; imputation strategy must be consistent between development and deployment.
5. **No information leakage**: No feature should contain information about the outcome that would not be available at prediction time.

## Variants and Extensions

### Ensemble Methods

Stacking (super learner) combines predictions from multiple base learners using a meta-learner, often improving over individual models. The super learner is asymptotically optimal among the candidate set.

### AutoML

Automated machine learning frameworks (H2O, AutoML, TPOT) automate feature engineering, model selection, and hyperparameter tuning. Useful for benchmarking but must still follow rigorous validation procedures.

### Survival Prediction

Random survival forests, DeepSurv (deep learning for Cox models), and other methods handle time-to-event outcomes while accounting for censoring.

### Fairness and Bias

Clinical prediction models may perform differently across demographic subgroups. Fairness-aware methods aim to ensure equitable performance across race, sex, age, and socioeconomic status.

## When to Use This Method

- **Large datasets** with many features and sufficient events (at least 10-20 events per variable for traditional models; more for ML).
- **Complex, non-linear relationships** where traditional models may underfit.
- **Prediction is the primary goal**, not causal inference or hypothesis testing.
- **Decision support** applications where clinical utility can be demonstrated via DCA.
- **External validation data** is available or can be obtained.

Prefer simpler models (penalized regression) when sample size is limited, interpretability is paramount, or the number of features is small. ML methods are not inherently superior to regression for all clinical prediction tasks.

## Strengths and Limitations

### Strengths
- Can capture non-linear relationships and interactions without pre-specification.
- Handles high-dimensional feature spaces effectively (especially with regularization).
- Ensemble methods are robust to individual model instability.
- SHAP provides model-agnostic interpretability.
- May improve discrimination in complex clinical scenarios.

### Limitations
- Requires larger sample sizes than traditional regression to realize benefits.
- Risk of overfitting, especially with aggressive hyperparameter tuning.
- "Black box" perception may hinder clinical adoption despite interpretability tools.
- Calibration often worse than logistic regression without recalibration.
- External validity frequently poor — models may not transport across populations.
- Computationally expensive to train and validate compared to regression.

## Key References

1. Steyerberg EW. *Clinical Prediction Models*. 2nd ed. Springer, 2019.
2. Collins GS, et al. Transparent Reporting of a multivariable prediction model for Individual Prognosis or Diagnosis (TRIPOD). *Annals of Internal Medicine*. 2015;162(1):55-63.
3. Lundberg SM, Lee S-I. A unified approach to interpreting model predictions (SHAP). *NeurIPS*. 2017.
4. Vickers AJ, Elkin EB. Decision curve analysis. *Medical Decision Making*. 2006;26(6):565-574.
5. Chen T, Guestrin C. XGBoost: A scalable tree boosting system. *KDD*. 2016.
6. Breiman L. Random forests. *Machine Learning*. 2001;45(1):5-32.
7. Van Calster B, et al. Calibration: the Achilles heel of predictive analytics. *BMC Medicine*. 2019;17:230.
8. Riley RD, et al. Calculating the sample size required for developing a clinical prediction model. *BMJ*. 2020;368:m441.
9. Christodoulou E, et al. A systematic review shows no performance benefit of machine learning over logistic regression for clinical prediction models. *Journal of Clinical Epidemiology*. 2019;110:12-22.
