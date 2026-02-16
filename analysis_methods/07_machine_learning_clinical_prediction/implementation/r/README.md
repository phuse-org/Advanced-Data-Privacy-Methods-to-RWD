# Machine Learning for Clinical Prediction — R Implementation

## Required Packages

```r
install.packages(c("tidymodels", "xgboost", "randomForest", "glmnet",
                   "pROC", "DALEX", "dcurves", "shapviz", "probably",
                   "themis", "vip"))

library(tidymodels)
library(xgboost)
library(randomForest)
library(glmnet)
library(pROC)
library(DALEX)
library(dcurves)
library(shapviz)
library(vip)
```

- **tidymodels**: Unified modeling framework (recipes, parsnip, rsample, tune, yardstick).
- **xgboost**: Gradient boosting implementation.
- **randomForest**: Random forest implementation.
- **glmnet**: Penalized logistic/linear regression (LASSO, ridge, elastic net).
- **pROC**: ROC curve analysis, AUC calculation, confidence intervals.
- **DALEX**: Model-agnostic explainability (variable importance, partial dependence).
- **dcurves**: Decision curve analysis.
- **shapviz**: SHAP value visualization.

## Example Dataset

We simulate an ICU mortality prediction dataset with 2000 patients and 10 clinical features.

```r
set.seed(42)
n <- 2000

icu_data <- data.frame(
  age          = round(rnorm(n, mean = 65, sd = 15)),
  sex          = factor(sample(c("Male", "Female"), n, replace = TRUE)),
  sofa_score   = rpois(n, lambda = 5),
  apache_ii    = round(rnorm(n, mean = 18, sd = 7)),
  heart_rate   = round(rnorm(n, mean = 90, sd = 20)),
  sbp          = round(rnorm(n, mean = 120, sd = 25)),
  creatinine   = round(rlnorm(n, meanlog = 0.3, sdlog = 0.6), 2),
  wbc          = round(rlnorm(n, meanlog = 2.2, sdlog = 0.5), 1),
  lactate      = round(rlnorm(n, meanlog = 0.5, sdlog = 0.7), 2),
  mech_vent    = factor(sample(c("Yes", "No"), n, replace = TRUE, prob = c(0.4, 0.6)))
)

# Simulate mortality outcome (~20% event rate)
log_odds <- -4 + 0.03 * icu_data$age + 0.15 * icu_data$sofa_score +
            0.08 * icu_data$apache_ii + 0.5 * icu_data$lactate -
            0.01 * icu_data$sbp + 0.3 * (icu_data$mech_vent == "Yes")
icu_data$mortality <- factor(rbinom(n, 1, plogis(log_odds)),
                             levels = c(0, 1), labels = c("Survived", "Died"))

table(icu_data$mortality)
```

This simulates a realistic ICU cohort where mortality is driven by age, severity scores, lactate, blood pressure, and ventilation status.

## Complete Worked Example

### Step 1: Data Splitting

```r
set.seed(123)
icu_split <- initial_split(icu_data, prop = 0.75, strata = mortality)
train_data <- training(icu_split)
test_data  <- testing(icu_split)

cat("Training set:", nrow(train_data), "patients\n")
cat("Test set:", nrow(test_data), "patients\n")
cat("Event rate (train):", mean(train_data$mortality == "Died"), "\n")
```

**Output interpretation**: Stratified splitting ensures both sets have similar mortality rates. The test set is held out entirely until final evaluation.

### Step 2: Preprocessing Recipe

```r
icu_recipe <- recipe(mortality ~ ., data = train_data) %>%
  step_normalize(all_numeric_predictors()) %>%
  step_dummy(all_nominal_predictors()) %>%
  step_zv(all_predictors())

# Preview the processed data
icu_recipe %>% prep() %>% bake(new_data = NULL) %>% head()
```

### Step 3: Model Specifications

```r
# Logistic regression (baseline)
lr_spec <- logistic_reg(penalty = 0.01, mixture = 1) %>%
  set_engine("glmnet") %>%
  set_mode("classification")

# Random forest
rf_spec <- rand_forest(mtry = 4, trees = 500, min_n = 10) %>%
  set_engine("randomForest") %>%
  set_mode("classification")

# XGBoost
xgb_spec <- boost_tree(trees = 500, tree_depth = 4, learn_rate = 0.05,
                        min_n = 10, sample_size = 0.8) %>%
  set_engine("xgboost") %>%
  set_mode("classification")
```

### Step 4: Cross-Validation

```r
set.seed(456)
cv_folds <- vfold_cv(train_data, v = 10, strata = mortality)

# Create workflows
lr_wf  <- workflow() %>% add_recipe(icu_recipe) %>% add_model(lr_spec)
rf_wf  <- workflow() %>% add_recipe(icu_recipe) %>% add_model(rf_spec)
xgb_wf <- workflow() %>% add_recipe(icu_recipe) %>% add_model(xgb_spec)

# Fit with CV
cv_metrics <- metric_set(roc_auc, brier_class, sensitivity, specificity)

lr_cv  <- fit_resamples(lr_wf, resamples = cv_folds, metrics = cv_metrics)
rf_cv  <- fit_resamples(rf_wf, resamples = cv_folds, metrics = cv_metrics)
xgb_cv <- fit_resamples(xgb_wf, resamples = cv_folds, metrics = cv_metrics)

# Compare models
bind_rows(
  collect_metrics(lr_cv) %>% mutate(model = "Logistic Regression"),
  collect_metrics(rf_cv) %>% mutate(model = "Random Forest"),
  collect_metrics(xgb_cv) %>% mutate(model = "XGBoost")
) %>%
  select(model, .metric, mean, std_err) %>%
  arrange(.metric, desc(mean)) %>%
  print(n = 20)
```

**Output interpretation**: Compare AUC (higher is better) and Brier score (lower is better) across models. Standard errors indicate stability across folds. If models perform similarly, prefer the simpler one (logistic regression). An AUC above 0.75 is generally considered acceptable for clinical prediction.

### Step 5: Final Model Training and Test Set Evaluation

```r
# Select XGBoost as the final model (assuming best CV performance)
xgb_final <- xgb_wf %>% fit(data = train_data)

# Predict on test set
test_preds <- predict(xgb_final, test_data, type = "prob") %>%
  bind_cols(predict(xgb_final, test_data)) %>%
  bind_cols(test_data %>% select(mortality))

# AUC
roc_obj <- roc(test_preds$mortality, test_preds$.pred_Died,
               levels = c("Survived", "Died"), direction = "<")
cat("Test AUC:", auc(roc_obj), "\n")
cat("95% CI:", ci.auc(roc_obj), "\n")

# Brier score
brier <- mean((as.numeric(test_preds$mortality == "Died") -
               test_preds$.pred_Died)^2)
cat("Brier score:", round(brier, 4), "\n")
```

**Output interpretation**: The test set AUC provides an unbiased estimate of discrimination. Brier score below 0.25 for a ~20% event rate indicates reasonable overall accuracy. Compare with the null model Brier score (prevalence * (1 - prevalence)).

### Step 6: ROC Curve

```r
library(pROC)

roc_obj <- roc(test_preds$mortality, test_preds$.pred_Died,
               levels = c("Survived", "Died"), direction = "<")

plot(roc_obj, col = "steelblue", lwd = 2,
     main = "ROC Curve: ICU Mortality Prediction (XGBoost)",
     print.auc = TRUE, print.auc.x = 0.4, print.auc.y = 0.3,
     ci = TRUE, ci.type = "shape", ci.col = adjustcolor("steelblue", 0.2))
abline(a = 1, b = -1, lty = 2, col = "gray50")
```

**Output interpretation**: The ROC curve plots sensitivity (true positive rate) against 1-specificity (false positive rate) across all thresholds. The shaded region represents the 95% confidence band for the AUC. A curve well above the diagonal indicates good discrimination.

### Step 7: Calibration Plot

```r
library(probably)

cal_data <- test_preds %>%
  mutate(prob_died = .pred_Died,
         obs = as.numeric(mortality == "Died"))

# Calibration plot using grouped binning
cal_plot_data <- cal_data %>%
  mutate(bin = cut(prob_died, breaks = seq(0, 1, by = 0.1),
                   include.lowest = TRUE)) %>%
  group_by(bin) %>%
  summarise(mean_pred = mean(prob_died),
            mean_obs  = mean(obs),
            n = n())

plot(cal_plot_data$mean_pred, cal_plot_data$mean_obs,
     xlim = c(0, 1), ylim = c(0, 1),
     xlab = "Predicted Probability", ylab = "Observed Proportion",
     main = "Calibration Plot: XGBoost", pch = 19, col = "steelblue",
     cex = sqrt(cal_plot_data$n) / 5)
abline(a = 0, b = 1, lty = 2, col = "gray50")
lines(lowess(cal_plot_data$mean_pred, cal_plot_data$mean_obs),
      col = "steelblue", lwd = 2)
legend("topleft", legend = paste("n per bin:", paste(cal_plot_data$n, collapse = ", ")),
       cex = 0.7)
```

**Output interpretation**: Points on the 45-degree line indicate perfect calibration. Points above the line mean the model underestimates risk; below means it overestimates. The LOESS smooth reveals systematic calibration patterns. Many ML models require post-hoc recalibration (e.g., Platt scaling).

## Advanced Example

### SHAP Values for Model Interpretability

```r
library(shapviz)

# Extract the xgboost model object
xgb_model <- extract_fit_engine(xgb_final)

# Prepare the data in the format xgboost expects
test_processed <- icu_recipe %>% prep() %>% bake(new_data = test_data)
test_matrix <- as.matrix(test_processed %>% select(-mortality))

# Compute SHAP values
shp <- shapviz(xgb_model, X_pred = test_matrix,
               X = test_processed %>% select(-mortality))

# SHAP summary (beeswarm) plot
sv_importance(shp, kind = "beeswarm", max_display = 10)

# SHAP waterfall for a single patient
sv_waterfall(shp, row_id = 1)

# SHAP dependence plot for top feature
sv_dependence(shp, v = "sofa_score", color_var = "auto")
```

**Output interpretation**: The beeswarm plot shows how each feature contributes to predictions across all patients. Features at the top have the strongest influence. Red points (high feature values) on the right (positive SHAP) mean that higher feature values increase the predicted risk. The waterfall plot decomposes one specific patient's prediction into additive feature contributions.

### Decision Curve Analysis

```r
library(dcurves)

dca_data <- data.frame(
  mortality = as.numeric(test_preds$mortality == "Died"),
  xgboost = test_preds$.pred_Died
)

# Add logistic regression predictions for comparison
lr_final <- lr_wf %>% fit(data = train_data)
lr_preds <- predict(lr_final, test_data, type = "prob")
dca_data$logistic <- lr_preds$.pred_Died

dca_result <- dca(mortality ~ xgboost + logistic,
                  data = dca_data,
                  thresholds = seq(0.01, 0.60, by = 0.01))

plot(dca_result,
     smooth = TRUE,
     style = "color")
```

**Output interpretation**: Decision curve analysis plots net benefit against threshold probability. A model with higher net benefit at a clinically relevant threshold (e.g., 10-30% for ICU mortality) provides better clinical utility. If both models track the "treat all" line, they add no value beyond treating everyone. The range where a model exceeds both "treat all" and "treat none" defines its useful operating range.

## Visualization

### Variable Importance Comparison

```r
library(vip)

# Permutation importance for XGBoost
xgb_vip <- xgb_final %>%
  extract_fit_engine() %>%
  vip(num_features = 10, geom = "col",
      aesthetics = list(fill = "steelblue")) +
  ggtitle("XGBoost: Variable Importance (Gain)")

print(xgb_vip)

# Permutation-based importance using DALEX
library(DALEX)
explainer_xgb <- explain(
  extract_fit_engine(xgb_final),
  data = test_matrix,
  y = as.numeric(test_preds$mortality == "Died"),
  type = "classification",
  label = "XGBoost"
)

vi_perm <- model_parts(explainer_xgb, type = "difference")
plot(vi_perm, max_vars = 10, show_boxplots = TRUE) +
  ggtitle("Permutation Variable Importance")
```

### Confusion Matrix at Optimal Threshold

```r
# Find optimal threshold (Youden's J)
coords_obj <- coords(roc_obj, "best", ret = c("threshold", "sensitivity",
                                                "specificity", "ppv", "npv"),
                     best.method = "youden")
print(coords_obj)

# Confusion matrix at optimal threshold
test_preds$pred_class <- factor(
  ifelse(test_preds$.pred_Died >= coords_obj$threshold, "Died", "Survived"),
  levels = c("Survived", "Died")
)

conf_matrix <- conf_mat(test_preds, truth = mortality, estimate = pred_class)
autoplot(conf_matrix, type = "heatmap") +
  ggtitle(paste("Confusion Matrix (threshold =",
                round(coords_obj$threshold, 3), ")"))

summary(conf_matrix)
```

**Output interpretation**: The confusion matrix shows true/false positives/negatives at the chosen threshold. Youden's J index maximizes sensitivity + specificity - 1. In clinical settings, the threshold should reflect the relative costs of false positives vs false negatives, not simply maximize Youden's J.

## Tips and Best Practices

1. **Always start with logistic regression** as a baseline. Many studies show ML does not significantly outperform well-specified regression models in clinical prediction.
2. **Use nested cross-validation** if you are tuning hyperparameters: the inner loop tunes, the outer loop estimates generalization error.
3. **Never evaluate on data used for any part of model development** (including feature selection, imputation parameter estimation, or threshold optimization).
4. **Report calibration alongside discrimination**: a model with high AUC but poor calibration can lead to harmful clinical decisions.
5. **For class imbalance**, prefer class weights or threshold adjustment over SMOTE; always recalibrate after resampling.
6. **Perform decision curve analysis** to demonstrate clinical utility — AUC alone does not justify clinical adoption.
7. **Follow TRIPOD guidelines** for transparent reporting of prediction model development and validation.
8. **Validate externally** before deployment. Internal validation (even rigorous CV) often overestimates performance.
9. **Monitor for dataset shift** after deployment: retrain periodically and track performance metrics over time.
10. **Document the intended use population** clearly — a model trained on ICU patients should not be applied to outpatients without revalidation.
