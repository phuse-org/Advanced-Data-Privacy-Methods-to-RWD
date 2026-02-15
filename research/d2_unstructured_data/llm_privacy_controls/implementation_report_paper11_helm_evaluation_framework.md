# Implementation Report — Paper 11
# HELM (Liang et al., 2022): Holistic Evaluation of Language Models

**PHUSE Working Group: Applying Advanced Data Privacy Methods to Real-World Data**
**Reviewer:** Onkar, Alex | **Date:** 12 February 2026

---

## 1. Paper Summary

HELM is a comprehensive evaluation framework from Stanford CRFM that benchmarks LLMs across **42 scenarios** (task/domain pairs) using **7 metric dimensions**: accuracy, calibration, robustness, fairness, bias, toxicity, and efficiency. Evaluating 30 models, HELM reveals that prior work covered only 17.9% of relevant scenarios and that no single model dominates across all dimensions. The framework's core contribution is the **taxonomy + standardized evaluation protocol**, not any specific model result.

## 2. Relevance to PHUSE WS1/WS2

HELM provides the **evaluation spine** for WS1 benchmarks. The Tech Implementation Plan requires metrics across utility, privacy, fairness, robustness, and efficiency [§4.1–4.3, Metrics]. HELM's multi-metric approach directly maps to this requirement. However, HELM must be **augmented** with healthcare-specific dimensions (privacy leakage, hallucination rate, guideline adherence) not present in the original framework.

**MedHELM** (Stanford, 2025) — an adaptation of HELM to 121 real-world medical tasks — is the most directly applicable derivative for our clinical benchmarks.

## 3. Implementation Plan

### 3.1 What We Will Implement

A **PHUSE Clinical Evaluation Framework** using HELM/MedHELM as the scaffold, extended with privacy and clinical safety metrics:

| Evaluation Dimension | HELM Original? | PHUSE Extension? | Metric Examples |
|---------------------|---------------|-------------------|-----------------|
| Accuracy | Yes | — | AUC, F1, clinical accuracy |
| Calibration | Yes | — | Expected Calibration Error (ECE), reliability diagrams |
| Robustness | Yes | Extended | Prompt perturbation delta; cross-site (MIMIC vs eICU) accuracy gap |
| Fairness | Yes | — | Demographic parity, equalized odds by sex/age/race |
| Bias | Yes | — | Stereotypical association scores |
| Efficiency | Yes | — | Inference latency, memory, cost-per-query |
| **Privacy leakage** | **No** | **PHUSE** | MIA AUC-ROC, extraction yield, PHI leak rate |
| **Clinical safety** | **No** | **PHUSE** | Hallucination rate, tool hallucination rate per patient |
| **Guideline adherence** | **No** | **PHUSE** | Compliance % with clinical practice guidelines |

### 3.2 Datasets

| Dataset | Data Type | Evaluation Tasks | Plan Reference |
|---------|-----------|-----------------|----------------|
| **MIMIC-IV** (structured) | Tabular EHR | ICU mortality/readmission prediction (accuracy, fairness, robustness) | §4.1.4 |
| **eICU** | Tabular EHR | Cross-site robustness evaluation (MIMIC-trained → eICU-tested) | §4.1.4 |
| **HiRID** | Tabular EHR | Additional cross-site validation | §4.1.4 |
| **MIMIC-IV-Note** | Clinical text | De-ID, concept extraction, classification, QA | §4.2.4 |
| **i2b2/n2c2** | Clinical NLP challenge | De-identification benchmark, NER evaluation | §4.2.4 |
| **MIMIC-CXR reports** | Radiology text | Document classification, RAG evaluation | §4.2.4 |

### 3.3 Tools and Language

| Component | Tool | Language |
|-----------|------|----------|
| Core evaluation framework | [HELM toolkit](https://crfm.stanford.edu/helm/latest/) / MedHELM | Python |
| Calibration metrics | scikit-learn (ECE) + custom reliability plots | Python |
| Fairness metrics | Fairlearn / AIF360 | Python |
| Robustness testing | TextAttack (perturbation) + custom cross-site scripts | Python |
| Statistical analysis | R (companion) | R |
| Experiment tracking | MLflow | Python |
| Reporting | Quarto | Python + R |

### 3.4 Evaluation Scenarios (WS1 Benchmark Tasks)

**Structured data scenarios (Q2 2026):**

| Scenario | Task | PET Under Test | Key Metrics |
|----------|------|---------------|-------------|
| `tabular_dp/mortality` | ICU mortality prediction | DP-SGD (Opacus) | AUC vs ε; fairness across demographics |
| `tabular_synthetic/fidelity` | Train-synthetic/test-real | SDV (CTGAN, TVAE) | ML utility; statistical similarity; MIA rate |
| `fl_icupred/readmission` | ICU readmission prediction | Flower (FL) | Accuracy vs centralized; convergence cost |

**Clinical text scenarios (Q4 2026):**

| Scenario | Task | PET Under Test | Key Metrics |
|----------|------|---------------|-------------|
| `text_deid/phi_detection` | PHI de-identification | Presidio, Philter, BERT-NER | F1, recall (prioritize recall), FN rate |
| `text_llm_privacy/extraction` | Training data extraction | DP fine-tuning defense | Extraction yield; privacy-utility curve |
| `text_llm_safety/hallucination` | Clinical hallucination rate | Hager-style simulation | Events per patient; prompt robustness |
| `text_llm_safety/rag_qa` | RAG over DP summaries | RAG + OpenDP | QA accuracy; extraction resistance |

### 3.5 Cross-Site Robustness Protocol

A key insight from Hager (Paper 12) is that prompt/input sensitivity breaks clinical utility. We implement:

1. **Prompt perturbation suite:** For each text scenario, test 5 systematic perturbations (synonym substitution, reordering, format changes, casing, abbreviation expansion). Report delta-accuracy.
2. **Cross-dataset generalization:** For structured scenarios, train on MIMIC-IV → test on eICU and HiRID. Report accuracy gap.
3. **Threshold:** If delta-accuracy exceeds 5% on any perturbation, flag as a robustness failure in the benchmark report.

## 4. Timeline (Aligned to Tech Plan §6)

| When | Milestone |
|------|-----------|
| **Q1 2026** (now) | Set up HELM/MedHELM toolkit; define PHUSE-extended evaluation config; prototype on one structured scenario |
| **Q2 2026** | Full structured data evaluation (all tabular scenarios); generate benchmark v1.0 report |
| **Q3 2026** | Design clinical text evaluation scenarios; build perturbation and cross-site test suites |
| **Q4 2026** | Full clinical text evaluation; generate benchmark v1.0 report with privacy + safety metrics |

## 5. Anticipated Bottlenecks

- **Compute for multi-model evaluation** — evaluating 5+ models across all scenarios requires significant GPU; plan for cloud/HPC
- **Clinical scenario design** requires physician input for guideline adherence targets and hallucination adjudication
- **Demographic annotations** are needed for fairness metrics — not all datasets have complete demographic fields
- **Bias statistic interpretation** — 91.7% of studies (not models) found bias; frame carefully in reports to avoid overgeneralization

## 6. Validation Checklist

- [ ] All metrics reported with confidence intervals (bootstrap)
- [ ] Fairness metrics computed on subgroups with sufficient sample size (n > 100)
- [ ] Robustness tested with minimum 5 perturbation types
- [ ] Cross-site evaluation on at least 2 datasets
- [ ] ECE calibration plots included for all classification tasks
- [ ] All evaluation configs stored as versioned YAML in repository
- [ ] Results reproducible via `uv sync --frozen` + DVC pull + single command
