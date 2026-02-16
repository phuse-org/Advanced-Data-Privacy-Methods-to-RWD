# Privacy Method — Analysis Method — Dataset Matrix

This matrix maps each **privacy method** in the repository to **recommended analytical methods** (at three complexity tiers) and the **best-suited datasets** for evaluating that privacy method's efficacy. It serves as a decision guide for researchers choosing how to benchmark and validate privacy-preserving techniques.

---

## How to Read This Matrix

- **Privacy Method**: One of the 7 privacy methods documented in `privacy_methods/`.
- **Analytical Methods**: At least 3 recommended methods from `analysis_methods/`, tiered as:
  - **Simple** — straightforward, quick to implement, good starting point
  - **Medium** — requires more statistical sophistication, richer evaluation
  - **Complex** — state-of-the-art, publication-grade analysis
- **Datasets**: The best-fit datasets from `datasets/`, tagged by access type:
  - **[FREE]** — open download, no registration (registry level: **Open**)
  - **[REG]** — free but requires registration, training, or DUA (registry level: **Controlled** or **Restricted**; see `datasets/README.md` Access Level Legend for definitions)

---

## Master Matrix

| # | Privacy Method | Data Type | Simple | Medium | Complex | Primary Datasets | Secondary Datasets |
|---|---------------|-----------|--------|--------|---------|-----------------|-------------------|
| 1 | Baseline De-identification | D1 Structured | Regression & GLMs (04) | Sensitivity Analysis (10) | Machine Learning (07) | MIMIC-IV [REG], eICU-CRD [REG] | NHANES [FREE], EHRShot [FREE] |
| 2 | Differential Privacy | D1 Structured | Regression & GLMs (04) | Machine Learning (07) | Bayesian Methods (05) | MIMIC-IV [REG], eICU-CRD [REG] | NHANES [FREE], BRFSS [FREE], HiRID [REG] |
| 3 | Synthetic Data Generation | D1 Structured | Regression & GLMs (04) | Machine Learning (07) | Causal Inference (03) | MIMIC-IV [REG], eICU-CRD [REG] | EHRShot [FREE], NHANES [FREE] |
| 4 | Federated Learning | D1 Structured | Regression & GLMs (04) | Machine Learning (07) | Survival Analysis (01) | MIMIC-IV [REG], eICU-CRD [REG] | AmsterdamUMCdb [REG], HiRID [REG] |
| 5 | Text De-identification | D2 Unstructured | Diagnostic Accuracy (15) | Machine Learning (07) | Sensitivity Analysis (10) | MIMIC-IV-Note [REG], i2b2/n2c2 [REG] | MIMIC-CXR [REG] |
| 6 | Privacy Attacks | D2 Unstructured | Diagnostic Accuracy (15) | Machine Learning (07) | Bayesian Methods (05) | MIMIC-IV-Note [REG], i2b2/n2c2 [REG] | MIMIC-CXR [REG], EHRShot [FREE] |
| 7 | LLM Privacy Controls | D2 Unstructured | Diagnostic Accuracy (15) | Sensitivity Analysis (10) | Causal Mediation (16) | MIMIC-IV-Note [REG], i2b2/n2c2 [REG] | MIMIC-CXR [REG], OpenFDA [FREE] |

---

## Detailed Breakdown by Privacy Method

---

### 1. Baseline De-identification (Structured Data)

**What it does:** Removes direct identifiers (names, MRNs, dates) and handles quasi-identifiers (age, ZIP, diagnosis combinations) through suppression, generalization, k-anonymity, and l-diversity checks.

**What we need to measure:** Whether de-identification preserves data utility while reducing re-identification risk.

#### Analytical Methods

| Tier | Method | Folder | What It Evaluates | Specific Application |
|------|--------|--------|-------------------|---------------------|
| **Simple** | Regression & GLMs | `04_regression_and_glms/` | Utility preservation | Fit logistic regression on original vs. de-identified data; compare coefficients, odds ratios, and p-values. If key associations survive generalization, utility is preserved. |
| **Medium** | Sensitivity Analysis | `10_sensitivity_analysis/` | Disclosure risk under assumptions | Use quantitative bias analysis (QBA) to model how much information an attacker could recover under varying generalization levels. E-value computation for residual risk after quasi-identifier suppression. |
| **Complex** | Machine Learning | `07_machine_learning_clinical_prediction/` | Predictive utility + re-ID risk | Train ML models (XGBoost, random forest) on original vs. de-identified data. Compare AUROC, calibration, and Brier scores. Separately, train a re-identification classifier to quantify residual linkage risk. SHAP analysis reveals which quasi-identifiers drive both prediction and re-ID risk. |

#### Recommended Datasets

| Dataset | # | Access | Why It Fits |
|---------|---|--------|-------------|
| **MIMIC-IV** | 01 | [REG] PhysioNet | Rich structured EHR with demographics, labs, diagnoses — ideal for testing quasi-identifier generalization on real ICU data. Contains the exact data types targeted by de-identification. |
| **eICU-CRD** | 03 | [REG] PhysioNet | Multi-center (200+ hospitals) adds geographic diversity. Tests whether de-identification holds across sites with different demographic distributions. |
| **NHANES** | 17 | [FREE] CDC | Public survey data with demographics, labs, health conditions. Good for testing k-anonymity methods on population-representative data without any registration barrier. |
| **EHRShot** | 26 | [FREE] GitHub | Small, open EHR benchmark (~6,700 patients). Quick prototyping of de-identification pipelines before scaling to MIMIC-IV. |

---

### 2. Differential Privacy (Structured Data)

**What it does:** Adds calibrated noise to queries (DP analytics) or to model training gradients (DP-SGD) to provide mathematically provable privacy guarantees parameterized by epsilon (privacy budget).

**What we need to measure:** The privacy-utility tradeoff — how much predictive/statistical accuracy is lost at different epsilon values.

#### Analytical Methods

| Tier | Method | Folder | What It Evaluates | Specific Application |
|------|--------|--------|-------------------|---------------------|
| **Simple** | Regression & GLMs | `04_regression_and_glms/` | Statistical accuracy under DP noise | Compare DP vs. non-DP summary statistics (means, proportions, regression coefficients). Run logistic regression with and without DP-SGD at epsilon = 1, 2, 4, 8. Plot coefficient bias vs. epsilon. |
| **Medium** | Machine Learning | `07_machine_learning_clinical_prediction/` | Predictive utility under DP | Train DP-SGD neural nets and XGBoost models at varying epsilon. Evaluate AUROC, calibration (ECE), and fairness (equalized odds) across subgroups. Construct utility-vs-epsilon curves. |
| **Complex** | Bayesian Methods | `05_bayesian_methods/` | Posterior privacy guarantees | Use Bayesian inference with DP posterior sampling (e.g., posterior DP via exponential mechanism). Compare posterior credible intervals under DP vs. non-DP to quantify uncertainty inflation. Bayesian model comparison (WAIC, LOO-CV) to assess model degradation at different privacy budgets. |

#### Additional Method: Subgroup Analysis (11)
At medium-to-complex level, test whether DP noise disproportionately degrades performance for minority subgroups. Use causal forests from `11_subgroup_analysis_hte/` to identify heterogeneous utility loss across demographic groups — critical for fairness evaluation.

#### Recommended Datasets

| Dataset | # | Access | Why It Fits |
|---------|---|--------|-------------|
| **MIMIC-IV** | 01 | [REG] PhysioNet | Primary benchmark for DP-SGD on ICU mortality/readmission tasks. Rich enough to test DP at scale with meaningful clinical endpoints. |
| **eICU-CRD** | 03 | [REG] PhysioNet | Multi-center data for testing whether DP noise impacts generalizability differently across hospital sites. |
| **NHANES** | 17 | [FREE] CDC | Excellent for DP analytics (counts, means, histograms) on nationally representative survey data. No registration needed — fastest path to prototype. |
| **BRFSS** | 19 | [FREE] CDC | Largest U.S. health survey (~400k respondents/year). Tests DP histogram and summary statistics at scale. |
| **HiRID** | 05 | [REG] PhysioNet | High-resolution time series — tests DP-SGD on temporal models where noise could disrupt sequential patterns. |

---

### 3. Synthetic Data Generation (Structured Data)

**What it does:** Generates synthetic datasets (via CTGAN, TVAE, or DP-synthetic methods) that mimic the statistical properties of real data without containing actual patient records.

**What we need to measure:** Statistical fidelity, downstream ML utility (train-on-synthetic / test-on-real), and privacy leakage (membership inference).

#### Analytical Methods

| Tier | Method | Folder | What It Evaluates | Specific Application |
|------|--------|--------|-------------------|---------------------|
| **Simple** | Regression & GLMs | `04_regression_and_glms/` | Statistical fidelity | Fit the same logistic/Poisson regression on real and synthetic data. Compare coefficients, standard errors, and p-values. Compute column-wise distributional metrics (KS test, Jensen-Shannon divergence). |
| **Medium** | Machine Learning | `07_machine_learning_clinical_prediction/` | Train-synthetic / test-real utility | Train XGBoost/random forest on synthetic data, evaluate on held-out real test set. Compare AUROC, calibration, and feature importance rankings against models trained on real data. Run membership inference attacks (MIA) to quantify privacy leakage. |
| **Complex** | Causal Inference | `03_causal_inference/` | Causal structure preservation | Estimate ATEs using propensity score matching and IPTW on both real and synthetic data. If causal effect estimates diverge, synthetic data has broken the causal structure. Use TMLE/AIPW for doubly robust estimates. This is the most stringent test — preserving marginal distributions is easy; preserving causal relationships is hard. |

#### Additional Method: Sensitivity Analysis (10)
Quantitative bias analysis on synthetic data evaluates how much confounding structure is preserved or distorted. E-value computation on synthetic vs. real causal estimates quantifies robustness.

#### Recommended Datasets

| Dataset | # | Access | Why It Fits |
|---------|---|--------|-------------|
| **MIMIC-IV** | 01 | [REG] PhysioNet | Gold standard for synthetic data benchmarking. Rich relational structure (20+ tables) tests whether generative models capture complex inter-table dependencies. |
| **eICU-CRD** | 03 | [REG] PhysioNet | Multi-center baseline for external validity — generate synthetic data from one center, test against another. |
| **EHRShot** | 26 | [FREE] GitHub | Small benchmark (~6,700 patients) — fast iteration on synthetic generation pipelines before scaling. Open access accelerates prototyping. |
| **NHANES** | 17 | [FREE] CDC | Complex survey design with sampling weights. Tests whether synthetic methods preserve survey design properties — an under-studied challenge. |

---

### 4. Federated Learning (Structured Data)

**What it does:** Trains models across multiple hospital sites without centralizing raw data. Each site trains locally and shares only model updates (gradients/weights) with a central aggregator.

**What we need to measure:** Whether the federated model matches centralized performance, and whether gradient sharing leaks private information.

#### Analytical Methods

| Tier | Method | Folder | What It Evaluates | Specific Application |
|------|--------|--------|-------------------|---------------------|
| **Simple** | Regression & GLMs | `04_regression_and_glms/` | Federated vs. centralized accuracy | Implement federated logistic regression via Flower. Compare coefficients and AUROC against a centralized logistic regression trained on pooled data. Quantify convergence speed (rounds to target accuracy). |
| **Medium** | Machine Learning | `07_machine_learning_clinical_prediction/` | Federated ML performance + site heterogeneity | Train federated XGBoost/neural nets. Evaluate per-site AUROC, global AUROC, and calibration. Test with non-IID data splits (different case mix per site) to measure federation robustness. |
| **Complex** | Survival Analysis | `01_survival_analysis/` | Federated time-to-event models | Implement federated Cox PH or federated RMST estimation. Compare against centralized Cox model. Evaluate whether hazard ratios, survival curves, and concordance indices are preserved under federation — critical for oncology and cardiovascular endpoints. |

#### Additional Method: Subgroup Analysis (11)
Analyze heterogeneous treatment effects across federated sites using causal forests. Determines whether federation preserves or distorts site-specific treatment effect estimates — important for multi-center trials.

#### Recommended Datasets

| Dataset | # | Access | Why It Fits |
|---------|---|--------|-------------|
| **MIMIC-IV** | 01 | [REG] PhysioNet | Single-center data that can be split by care unit (MICU, SICU, CCU, TSICU) to simulate multi-site federation with natural heterogeneity. |
| **eICU-CRD** | 03 | [REG] PhysioNet | Actual multi-center data (200+ hospitals). The most realistic federated learning testbed — each hospital becomes a federation client with genuine non-IID distributions. |
| **AmsterdamUMCdb** | 04 | [REG] AmsterdamUMC | European ICU data as an out-of-distribution federation partner. Tests cross-border federated learning with different coding systems and populations. |
| **HiRID** | 05 | [REG] PhysioNet | Swiss ICU data — another international node for realistic cross-silo federation experiments. |

---

### 5. Text De-identification (Unstructured Data)

**What it does:** Detects and masks Protected Health Information (PHI) in clinical notes using NER models (BERT, BiLSTM-CRF), rule-based systems, or hybrid LLM-assisted workflows.

**What we need to measure:** PHI detection accuracy (especially recall — missed PHI is dangerous), and whether downstream NLP tasks remain useful after masking.

#### Analytical Methods

| Tier | Method | Folder | What It Evaluates | Specific Application |
|------|--------|--------|-------------------|---------------------|
| **Simple** | Diagnostic Accuracy & Biomarkers | `15_diagnostic_accuracy_biomarkers/` | NER detection performance | Compute entity-level precision, recall, F1 by PHI category (names, dates, locations, IDs). Build ROC curves for token-level classification confidence. Optimize detection thresholds using Youden index — recall is prioritized over precision for safety. |
| **Medium** | Machine Learning | `07_machine_learning_clinical_prediction/` | Downstream NLP utility after de-ID | Train clinical NLP models (e.g., ICD coding, mortality prediction from notes) on original vs. de-identified text. Compare AUROC and F1. Use SHAP on text features to identify which masked tokens drive utility loss. Cross-validate across note types (discharge summaries, radiology, nursing). |
| **Complex** | Sensitivity Analysis | `10_sensitivity_analysis/` | Residual disclosure risk | Apply quantitative bias analysis to estimate residual PHI leakage after de-identification. Use tipping-point analysis: how many missed PHI entities (at what severity) would make re-identification feasible? Model attacker capabilities under different knowledge assumptions. Probabilistic bias analysis with Monte Carlo simulation of detection failure rates. |

#### Additional Method: Meta-Analysis (06)
Conduct a meta-analysis of de-ID performance across multiple PHI categories and note types to produce pooled precision/recall estimates with heterogeneity (I-squared). This quantifies how consistent de-ID performance is across clinical settings.

#### Recommended Datasets

| Dataset | # | Access | Why It Fits |
|---------|---|--------|-------------|
| **MIMIC-IV-Note** | 02 | [REG] PhysioNet | Millions of real clinical notes (discharge summaries, radiology reports, nursing notes). The primary benchmark for clinical text de-identification at scale. |
| **i2b2/n2c2** | 25 | [REG] Harvard DBMI | Gold-standard annotated clinical NLP datasets from shared tasks. Includes manually labeled PHI annotations — the definitive ground truth for de-ID evaluation. |
| **MIMIC-CXR** | 06 | [REG] PhysioNet | Radiology reports paired with images. Tests de-ID in a multimodal context where report text must be cleaned before linking with imaging data. |

---

### 6. Privacy Attacks / Evaluations (Unstructured Data)

**What it does:** Tests whether private information can be extracted from trained models through membership inference attacks (MIA), training data extraction, dataset inference, and targeted PHI probing.

**What we need to measure:** Attack success rates — how much private information leaks from models trained on clinical text.

#### Analytical Methods

| Tier | Method | Folder | What It Evaluates | Specific Application |
|------|--------|--------|-------------------|---------------------|
| **Simple** | Diagnostic Accuracy & Biomarkers | `15_diagnostic_accuracy_biomarkers/` | Attack classifier performance | Compute ROC-AUC for membership inference classifiers. Report TPR@FPR=0.01 (the clinically relevant operating point — low false positive). Build precision-recall curves for extraction attacks. Decision curve analysis to assess whether the attack provides actionable discrimination at realistic thresholds. |
| **Medium** | Machine Learning | `07_machine_learning_clinical_prediction/` | Attack model development | Train shadow models for MIA (reference-model calibration). Implement perplexity-based and likelihood-ratio extraction attacks. Use XGBoost/random forest on attack features (loss, perplexity, zlib entropy) to build composite attack classifiers. Feature importance (SHAP) reveals which signals drive successful attacks. |
| **Complex** | Bayesian Methods | `05_bayesian_methods/` | Posterior-based membership inference | Implement Bayesian membership inference using posterior predictive distributions. Compare posterior uncertainty for member vs. non-member samples. Bayesian hypothesis testing (Bayes factors) for membership decisions provides calibrated uncertainty estimates and avoids the multiple-testing problem inherent in testing thousands of candidate members. |

#### Additional Method: Multiplicity Adjustment (13)
When testing thousands of candidate sequences for memorization, apply Benjamini-Hochberg FDR control from `13_multiplicity_adjustment/` to distinguish true extractions from false positives.

#### Recommended Datasets

| Dataset | # | Access | Why It Fits |
|---------|---|--------|-------------|
| **MIMIC-IV-Note** | 02 | [REG] PhysioNet | Train language models on real clinical notes and test for memorization. The richness and redundancy of clinical text makes extraction attacks particularly relevant here. |
| **i2b2/n2c2** | 25 | [REG] Harvard DBMI | Annotated clinical NLP data — ground truth annotations enable precise evaluation of whether attacks recover actual PHI vs. benign text. |
| **MIMIC-CXR** | 06 | [REG] PhysioNet | Radiology reports — shorter, more templated text. Tests whether extraction attacks are more or less effective on repetitive clinical language. |
| **EHRShot** | 26 | [FREE] GitHub | Open structured EHR data for testing MIA on tabular models as a comparison baseline against text-based attacks. |

---

### 7. LLM Privacy Controls (Unstructured Data)

**What it does:** Applies privacy safeguards to LLM-based clinical workflows: RAG over de-identified summaries, DP synthetic text generation, DP fine-tuning of small models, and safety evaluation (HELM/MedHELM framework).

**What we need to measure:** Whether privacy controls reduce information leakage without degrading clinical accuracy, hallucination rates, or guideline compliance.

#### Analytical Methods

| Tier | Method | Folder | What It Evaluates | Specific Application |
|------|--------|--------|-------------------|---------------------|
| **Simple** | Diagnostic Accuracy & Biomarkers | `15_diagnostic_accuracy_biomarkers/` | LLM output accuracy metrics | Compute accuracy, F1, and hallucination rate for LLM outputs under different privacy controls. ROC analysis on LLM confidence scores vs. correctness. Compare factual accuracy of RAG-based vs. direct fine-tuned approaches. |
| **Medium** | Sensitivity Analysis | `10_sensitivity_analysis/` | Robustness to prompt perturbations | Use prompt perturbation testing as a form of sensitivity analysis. Vary prompt wording, instruction framing, and adversarial prefixes. Tipping-point analysis: how much adversarial pressure before the LLM leaks private information? E-value analogy: quantify how strong an attack would need to be to overcome the privacy control. |
| **Complex** | Causal Mediation Analysis | `16_causal_mediation_analysis/` | Decomposing privacy leakage pathways | Model the LLM pipeline as a causal chain: input prompt -> retrieval/context -> model inference -> output. Use 4-way decomposition to separate: (a) direct leakage from the model's parameters, (b) indirect leakage through retrieved context, (c) interaction effects between prompt engineering and model memorization. Identifies which pipeline component is the primary leakage channel. |

#### Additional Methods
- **Meta-Analysis (06):** Pool LLM evaluation results across multiple clinical tasks (ICD coding, summarization, QA) to estimate overall privacy-controlled performance with heterogeneity.
- **Quasi-Experimental (12):** Use interrupted time series to measure whether deploying a privacy control (e.g., switching from direct access to RAG) changes information leakage rates over time.

#### Recommended Datasets

| Dataset | # | Access | Why It Fits |
|---------|---|--------|-------------|
| **MIMIC-IV-Note** | 02 | [REG] PhysioNet | Primary corpus for clinical LLM fine-tuning and RAG experiments. Tests whether privacy controls work on real clinical text at scale. |
| **i2b2/n2c2** | 25 | [REG] Harvard DBMI | Annotated NLP tasks (de-ID, relation extraction, coding) — structured evaluation of LLM performance under privacy controls with ground-truth labels. |
| **MIMIC-CXR** | 06 | [REG] PhysioNet | Multimodal (report + image) — tests privacy controls in vision-language LLM pipelines where information can leak through text or image channels. |
| **OpenFDA** | 29 | [FREE] API | Fully open regulatory text (drug labels, adverse events). Excellent for testing RAG pipelines and LLM summarization without any access barriers. |

---

## Summary: Dataset Suitability by Access Type

### Free / Open Access (No Registration)

| Dataset | # | Best For Privacy Methods |
|---------|---|------------------------|
| NHANES | 17 | DP analytics, baseline de-ID, synthetic data |
| BRFSS | 19 | DP analytics (large scale) |
| EHRShot | 26 | Synthetic data prototyping, MIA on tabular models |
| OpenFDA | 29 | LLM RAG testing, regulatory text privacy |
| NIH ChestX-ray14 | 12 | Imaging de-ID (if scope extends to imaging) |
| NHIS | 18 | DP analytics on survey data |
| CDC SVI | 21 | DP aggregation on area-level data |
| WHO GHO | 20 | DP analytics on global indicators |
| County Health Rankings | 23 | DP aggregation on county-level data |
| TCIA | 10 | Imaging privacy (if scope extends) |

### Registration Required (Free with DUA)

| Dataset | # | Best For Privacy Methods |
|---------|---|------------------------|
| MIMIC-IV | 01 | All D1 methods (baseline de-ID, DP, synthetic, FL) — the primary benchmark |
| MIMIC-IV-Note | 02 | All D2 methods (text de-ID, attacks, LLM controls) — the primary text benchmark |
| eICU-CRD | 03 | FL (multi-center), DP (cross-site validation), synthetic data |
| i2b2/n2c2 | 25 | Text de-ID (gold-standard annotations), LLM evaluation |
| MIMIC-CXR | 06 | Multimodal privacy (text + imaging), LLM report generation |
| HiRID | 05 | DP on high-resolution time series, FL (international node) |
| AmsterdamUMCdb | 04 | FL (European node), cross-border privacy evaluation |
| SEER | 08 | DP on cancer registry data, synthetic data for oncology |
| All of Us | 31 | Multi-modal DP, FL on genomics + EHR + wearables |

---

## Quick Decision Guide

```
START HERE: What privacy method are you evaluating?
│
├─ Baseline De-identification
│   ├─ Quick test? → Logistic regression on NHANES [FREE]
│   ├─ Full benchmark? → ML + sensitivity analysis on MIMIC-IV [REG]
│   └─ Publication? → ML re-ID risk + QBA on MIMIC-IV + eICU [REG]
│
├─ Differential Privacy
│   ├─ Quick test? → DP means/histograms on NHANES or BRFSS [FREE]
│   ├─ Full benchmark? → DP-SGD ML models on MIMIC-IV [REG]
│   └─ Publication? → Bayesian DP + fairness + subgroup on MIMIC-IV + eICU [REG]
│
├─ Synthetic Data
│   ├─ Quick test? → Regression fidelity on EHRShot [FREE]
│   ├─ Full benchmark? → Train-synth/test-real ML on MIMIC-IV [REG]
│   └─ Publication? → Causal structure preservation (IPTW/TMLE) on MIMIC-IV [REG]
│
├─ Federated Learning
│   ├─ Quick test? → Federated logistic regression on MIMIC-IV (unit split) [REG]
│   ├─ Full benchmark? → Federated XGBoost on eICU (hospital split) [REG]
│   └─ Publication? → Federated Cox PH on eICU + AmsterdamUMCdb + HiRID [REG]
│
├─ Text De-identification
│   ├─ Quick test? → F1/recall by PHI category on i2b2/n2c2 [REG]
│   ├─ Full benchmark? → NLP utility after de-ID on MIMIC-IV-Note [REG]
│   └─ Publication? → Residual risk QBA + meta-analysis on both [REG]
│
├─ Privacy Attacks
│   ├─ Quick test? → MIA ROC-AUC on EHRShot [FREE]
│   ├─ Full benchmark? → Shadow model MIA + extraction on MIMIC-IV-Note [REG]
│   └─ Publication? → Bayesian MIA + FDR-controlled extraction on MIMIC-IV-Note [REG]
│
└─ LLM Privacy Controls
    ├─ Quick test? → RAG accuracy on OpenFDA [FREE]
    ├─ Full benchmark? → Prompt perturbation sensitivity on MIMIC-IV-Note [REG]
    └─ Publication? → Causal mediation decomposition on MIMIC-IV-Note + i2b2 [REG]
```

---

## Cross-Reference: Which Analysis Methods Apply to Multiple Privacy Methods

| Analysis Method | Folder | Privacy Methods It Supports |
|----------------|--------|---------------------------|
| **Regression & GLMs** | `04_regression_and_glms/` | Baseline De-ID, DP, Synthetic Data, FL (all as Simple tier) |
| **Machine Learning** | `07_machine_learning_clinical_prediction/` | All 7 methods (Medium tier for most) |
| **Sensitivity Analysis** | `10_sensitivity_analysis/` | Baseline De-ID, Text De-ID, LLM Controls (Medium tier) |
| **Diagnostic Accuracy** | `15_diagnostic_accuracy_biomarkers/` | Text De-ID, Privacy Attacks, LLM Controls (Simple tier) |
| **Bayesian Methods** | `05_bayesian_methods/` | DP, Privacy Attacks (Complex tier) |
| **Causal Inference** | `03_causal_inference/` | Synthetic Data (Complex tier) |
| **Survival Analysis** | `01_survival_analysis/` | FL (Complex tier) |
| **Subgroup / HTE** | `11_subgroup_analysis_hte/` | DP, FL (additional fairness analysis) |
| **Meta-Analysis** | `06_meta_analysis/` | Text De-ID, LLM Controls (additional cross-task pooling) |
| **Multiplicity** | `13_multiplicity_adjustment/` | Privacy Attacks (additional FDR control for extraction tests) |
| **Causal Mediation** | `16_causal_mediation_analysis/` | LLM Controls (Complex tier) |
| **Quasi-Experimental** | `12_quasi_experimental_methods/` | LLM Controls (additional longitudinal evaluation) |
