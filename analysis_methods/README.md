# Analysis Methods

This folder contains **20 analysis method families** commonly used in health data research, clinical trials, pharmacoepidemiology, and real-world evidence (RWE) studies. Each method was identified through a systematic review of the literature, FDA/EMA guidance documents, PHUSE working group publications, and standard biostatistics textbooks.

No raw data is stored here — only documentation, theory, and implementation guides.

---

## Folder Structure

Each method folder follows the same structure:

```
XX_method_name/
├── theory/
│   └── README.md          # Mathematical foundations, assumptions, key concepts
└── implementation/
    ├── r/
    │   └── README.md      # Full R implementation with runnable examples
    └── python/
        └── README.md      # Full Python implementation with runnable examples
```

---

## Quick Reference — All Methods

| # | Method | Primary Application | Folder |
|---|--------|-------------------|--------|
| 1 | [Survival / Time-to-Event Analysis](#1-survival-analysis) | Clinical trials, oncology, cardiovascular | `01_survival_analysis/` |
| 2 | [Competing Risks Analysis](#2-competing-risks) | Oncology, geriatrics, transplant | `02_competing_risks/` |
| 3 | [Causal Inference & Confounding Adjustment](#3-causal-inference) | Pharmacoepidemiology, RWE, comparative effectiveness | `03_causal_inference/` |
| 4 | [Regression & Generalized Linear Models](#4-regression-and-glms) | Universal across all domains | `04_regression_and_glms/` |
| 5 | [Bayesian Methods](#5-bayesian-methods) | Adaptive trials, rare diseases, devices | `05_bayesian_methods/` |
| 6 | [Meta-Analysis & Network Meta-Analysis](#6-meta-analysis) | Systematic reviews, HTA, guidelines | `06_meta_analysis/` |
| 7 | [Machine Learning for Clinical Prediction](#7-machine-learning) | EHR-based prediction, risk stratification | `07_machine_learning_clinical_prediction/` |
| 8 | [Missing Data Methods](#8-missing-data) | Clinical trials (ICH E9(R1)), registries | `08_missing_data_methods/` |
| 9 | [Longitudinal / Repeated Measures](#9-longitudinal) | Chronic disease trials, PROs | `09_longitudinal_repeated_measures/` |
| 10 | [Sensitivity Analysis & QBA](#10-sensitivity-analysis) | Pharmacoepidemiology, RWE | `10_sensitivity_analysis/` |
| 11 | [Subgroup Analysis & HTE](#11-subgroup-hte) | Precision medicine, regulatory submissions | `11_subgroup_analysis_hte/` |
| 12 | [Quasi-Experimental Methods](#12-quasi-experimental) | Policy evaluation, vaccine effectiveness | `12_quasi_experimental_methods/` |
| 13 | [Multiplicity Adjustment](#13-multiplicity) | Confirmatory clinical trials | `13_multiplicity_adjustment/` |
| 14 | [Dose-Response Modeling](#14-dose-response) | Clinical pharmacology, dose optimization | `14_dose_response_modeling/` |
| 15 | [Diagnostic Accuracy & Biomarkers](#15-diagnostic-accuracy) | Screening, diagnostics, biomarker validation | `15_diagnostic_accuracy_biomarkers/` |
| 16 | [Causal Mediation Analysis](#16-mediation) | Mechanism-of-action, pathway decomposition | `16_causal_mediation_analysis/` |
| 17 | [Healthcare Cost & Resource Analysis](#17-cost-analysis) | Health economics, HTA | `17_healthcare_cost_analysis/` |
| 18 | [Adaptive Trial Design](#18-adaptive-design) | Trial monitoring, DSMBs | `18_adaptive_trial_design/` |
| 19 | [Study Design Frameworks](#19-design-frameworks) | RWE study planning, regulatory submissions | `19_study_design_frameworks/` |
| 20 | [Safety Signal Detection](#20-safety-signals) | Post-marketing surveillance, pharmacovigilance | `20_safety_signal_detection/` |

---

## Method Summaries

### 1. Survival Analysis
Time-to-event methods with censoring — Kaplan-Meier, Cox PH, AFT models. See [`01_survival_analysis/`](01_survival_analysis/).

### 2. Competing Risks
Multiple event types in time-to-event data — Fine-Gray, cause-specific hazards. See [`02_competing_risks/`](02_competing_risks/).

### 3. Causal Inference
Propensity scores, IPTW, TMLE, target trial emulation for observational data. See [`03_causal_inference/`](03_causal_inference/).

### 4. Regression and GLMs
Linear, logistic, Poisson, penalized regression, GAMs. See [`04_regression_and_glms/`](04_regression_and_glms/).

### 5. Bayesian Methods
Priors, MCMC, hierarchical models, adaptive designs. See [`05_bayesian_methods/`](05_bayesian_methods/).

### 6. Meta-Analysis
Fixed/random effects meta-analysis, network meta-analysis. See [`06_meta_analysis/`](06_meta_analysis/).

### 7. Machine Learning
Random forests, XGBoost, validation, SHAP for EHR-based clinical prediction. See [`07_machine_learning_clinical_prediction/`](07_machine_learning_clinical_prediction/).

### 8. Missing Data
Multiple imputation (MICE), reference-based methods, ICH E9(R1) alignment. See [`08_missing_data_methods/`](08_missing_data_methods/).

### 9. Longitudinal
Mixed models, GEE, MMRM for repeated measures and chronic disease trials. See [`09_longitudinal_repeated_measures/`](09_longitudinal_repeated_measures/).

### 10. Sensitivity Analysis
E-values, quantitative bias analysis, tipping-point analysis. See [`10_sensitivity_analysis/`](10_sensitivity_analysis/).

### 11. Subgroup HTE
Heterogeneity of treatment effects, causal forests, precision medicine. See [`11_subgroup_analysis_hte/`](11_subgroup_analysis_hte/).

### 12. Quasi-Experimental
Interrupted time series, difference-in-differences, regression discontinuity. See [`12_quasi_experimental_methods/`](12_quasi_experimental_methods/).

### 13. Multiplicity
Family-wise error rate control, gatekeeping, graphical approaches. See [`13_multiplicity_adjustment/`](13_multiplicity_adjustment/).

### 14. Dose-Response
MCP-Mod, Emax/sigmoid models, model averaging for dose-finding. See [`14_dose_response_modeling/`](14_dose_response_modeling/).

### 15. Diagnostic Accuracy
ROC analysis, decision curve analysis, biomarker evaluation. See [`15_diagnostic_accuracy_biomarkers/`](15_diagnostic_accuracy_biomarkers/).

### 16. Mediation
Causal mediation analysis — decomposing direct and indirect effects. See [`16_causal_mediation_analysis/`](16_causal_mediation_analysis/).

### 17. Cost Analysis
Two-part models, cost-effectiveness analysis, ICER. See [`17_healthcare_cost_analysis/`](17_healthcare_cost_analysis/).

### 18. Adaptive Design
Group sequential, adaptive enrichment, platform trials, DSMBs. See [`18_adaptive_trial_design/`](18_adaptive_trial_design/).

### 19. Design Frameworks
Target trial emulation, estimand framework, RWE study planning. See [`19_study_design_frameworks/`](19_study_design_frameworks/).

### 20. Safety Signals
Disproportionality analysis, SCCS, pharmacovigilance signal detection. See [`20_safety_signal_detection/`](20_safety_signal_detection/).

---

## Method Categories

### Clinical Trials & Experimental Design
- **Survival Analysis** (01) — time-to-event outcomes with censoring
- **Competing Risks** (02) — multiple event types in time-to-event data
- **Adaptive Trial Design** (18) — group sequential, adaptive, platform trials
- **Multiplicity Adjustment** (13) — controlling type I error across endpoints
- **Dose-Response Modeling** (14) — characterizing dose-outcome relationships

### Causal Inference from Observational Data
- **Causal Inference** (03) — propensity scores, IPTW, TMLE, target trial emulation
- **Quasi-Experimental Methods** (12) — ITS, DiD, regression discontinuity
- **Causal Mediation** (16) — decomposing direct and indirect effects
- **Study Design Frameworks** (19) — target trial emulation, estimand framework

### Statistical Modeling & Machine Learning
- **Regression & GLMs** (04) — linear, logistic, Poisson, penalized, GAMs
- **Bayesian Methods** (05) — priors, MCMC, hierarchical models, adaptive designs
- **Machine Learning** (07) — random forests, XGBoost, validation, SHAP
- **Longitudinal / Repeated Measures** (09) — mixed models, GEE, MMRM

### Robustness & Validity
- **Missing Data Methods** (08) — multiple imputation, MICE, reference-based methods
- **Sensitivity Analysis** (10) — E-values, QBA, tipping-point analysis
- **Subgroup Analysis & HTE** (11) — heterogeneity of treatment effects, causal forests

### Specialized Domains
- **Meta-Analysis** (06) — combining studies, network meta-analysis
- **Diagnostic Accuracy** (15) — ROC, DCA, biomarker evaluation
- **Healthcare Cost Analysis** (17) — two-part models, CEA, ICER
- **Safety Signal Detection** (20) — disproportionality, SCCS, pharmacovigilance

---

## How to Use This Resource

1. **Identify your research question** — What outcome? What study design? What data?
2. **Find the matching method(s)** — Use the table above or browse by category.
3. **Read the theory** — Understand assumptions and when the method applies.
4. **Follow the implementation** — Copy and adapt the R or Python code examples.
5. **Check robustness** — Use sensitivity analysis (10) and missing data methods (08) to validate findings.

---

## Key Cross-References

| If you are doing... | You likely also need... |
|---------------------|----------------------|
| Survival analysis (01) | Competing risks (02), sensitivity analysis (10) |
| Causal inference (03) | Sensitivity analysis (10), missing data (08), study design (19) |
| Clinical prediction (07) | Diagnostic accuracy (15), subgroup analysis (11) |
| Clinical trial analysis | Multiplicity (13), adaptive design (18), missing data (08) |
| Pharmacoepidemiology | Causal inference (03), quasi-experimental (12), safety (20) |
| Health economics | Cost analysis (17), meta-analysis (06) |

---

## References

Key guidance documents informing this collection:

- **ICH E9(R1)** — Estimands and Sensitivity Analysis in Clinical Trials (2019)
- **FDA RWE Guidance** — Framework for Real-World Evidence (2018, updated 2025)
- **PHUSE Safety Analytics** — Working group deliverables on safety methods
- **Hernan & Robins** — Causal Inference: What If (2020)
- **Harrell** — Regression Modeling Strategies (2015)
- **Collett** — Modelling Survival Data in Medical Research (2014)
- **Higgins & Green** — Cochrane Handbook for Systematic Reviews (2019)
