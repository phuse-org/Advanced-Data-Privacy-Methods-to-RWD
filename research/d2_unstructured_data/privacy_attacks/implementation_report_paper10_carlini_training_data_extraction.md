# Implementation Report — Paper 10
# Carlini et al. (USENIX Security 2021): Extracting Training Data from LLMs

**PHUSE Working Group: Applying Advanced Data Privacy Methods to Real-World Data**
**Reviewer:** Onkar, Alex | **Date:** 12 February 2026

---

## 1. Paper Summary

Carlini et al. demonstrate that an adversary with **black-box query access** to GPT-2 can extract verbatim training data — including PII, code, and UUIDs. The attack generates 600,000 samples, ranks them using perplexity-based and zlib entropy heuristics, then verifies memorized content through internet searching and author-assisted dataset matching. At least **604 memorized sequences** were confirmed (a lower bound). Larger models, duplicated training data, and unique/high-entropy strings all increase extraction risk.

## 2. Relevance to PHUSE WS1/WS2

This paper directly motivates the **Privacy Attacks / Evaluations** benchmark tasks in the Tech Implementation Plan [§4.2.2], which lists membership inference, data extraction, and dataset inference as required evaluations for clinical text. Clinical notes are the **highest-risk data type** because they contain direct PHI and are "extremely susceptible to LLM data extraction attacks" [Tech Plan §4.2.1].

## 3. Implementation Plan

### 3.1 What We Will Implement

| Benchmark Task | Description | WS1 Pipeline Name |
|---------------|-------------|-------------------|
| **Training data extraction** | Carlini-style: generate → rank → verify for PHI leakage | `text_llm_privacy/extraction` |
| **Membership inference (MIA)** | Determine if a clinical record was used in training | `text_llm_privacy/mia` |
| **Dataset inference** | Detect if a dataset was used for pre-training/fine-tuning | `text_llm_privacy/dataset_inference` |
| **PHI probing** | Red-team prompting to induce PHI emission | `text_llm_privacy/phi_probing` |

### 3.2 Datasets

| Dataset | Type | Access | Role in Benchmark |
|---------|------|--------|-------------------|
| **MIMIC-IV-Note** | Clinical notes (discharge summaries, radiology) | PhysioNet credentialed | Primary text corpus for fine-tuning target models |
| **i2b2/n2c2 2014** | De-identification challenge data | DUA required | Secondary evaluation corpus; gold-standard PHI annotations |
| **MIMIC-CXR reports** | Radiology report text | PhysioNet credentialed | Additional clinical text modality |

### 3.3 Tools and Language

| Component | Tool | Language |
|-----------|------|----------|
| Extraction pipeline | [LM_Memorization](https://github.com/ftramer/LM_Memorization) + custom extensions | Python |
| MIA implementation | Loss-based + SPV-MIA [Fu et al., NeurIPS 2024] | Python (PyTorch) |
| Dataset inference | Min-K%++ detector | Python (HuggingFace Transformers) |
| PHI detection in outputs | Presidio + manual review | Python |
| Experiment tracking | MLflow | Python |
| Data versioning | DVC | CLI |

### 3.4 Metrics

| Attack | Primary Metric | Secondary Metrics |
|--------|---------------|-------------------|
| Extraction | Extraction yield (memorized sequences per N samples) | Precision@k; PHI-category breakdown |
| MIA | AUC-ROC | TPR@FPR=0.01 (critical for health — low false positive required) |
| Dataset inference | AUROC on membership labels | Calibration of confidence scores |
| PHI probing | PHI leak rate per 1,000 queries | Category-specific rates (name, date, MRN, SSN) |

### 3.5 Defense Evaluation

For each attack, compare success rates across three conditions:
1. **Unprotected** model (fine-tuned directly on clinical text)
2. **DP fine-tuned** model (DP-SGD via Opacus, at ε = {1, 4, 8})
3. **De-identified training data** (Presidio/Philter pre-processed)

This produces **privacy-utility tradeoff curves** — the core deliverable for WS1.

## 4. Threat Model

| Element | Specification |
|---------|--------------|
| Adversary goal | Extract memorized PHI or determine training membership |
| Adversary access | Black-box query (API); gray-box for MIA (loss available) |
| Adversary knowledge | Knows target domain is clinical text; partial training distribution knowledge |
| Target model | Any LLM fine-tuned on clinical notes |

## 5. Timeline (Aligned to Tech Plan §6)

| When | Milestone |
|------|-----------|
| **Q1 2026** (now) | Literature review complete; prototype extraction attack on public GPT-2 to validate methodology |
| **Q3 2026** | Define clinical text privacy benchmark module; select datasets; build pipeline skeleton |
| **Q4 2026** | Full implementation: extraction + MIA + dataset inference + PHI probing on clinical LLMs; generate privacy-utility curves; produce governance artifacts |

## 6. Anticipated Bottlenecks

- **PhysioNet credentialing** for MIMIC datasets (start immediately)
- **Compute cost** for large-scale generation (600K+ samples per model); coordinate GPU access
- **Manual verification** of extracted candidates is labor-intensive; develop semi-automated PHI matching
- **MIMIC DUA constraints** prevent sending data to external closed-source APIs [Hager et al.] — must use locally deployed or open-source models

## 7. Validation Checklist

- [ ] Repeat attacks across 3+ random seeds and sampling configurations
- [ ] Deduplicate outputs before counting leaks
- [ ] Log all prompts used (reproducibility)
- [ ] Report confidence intervals on MIA AUC-ROC
- [ ] Record PHI category breakdown
- [ ] All runs tracked in MLflow with DVC data lineage
