# PHUSE Working Group — LLM Safety & Privacy in Health Context
# Combined Paper Review + Similar Work + Strategy Analysis Plan (WS1/WS2-Aligned)

**Reviewer / Lead:** Onkar, Alex
**Papers Assigned:** Items 10, 11, 12
**Project:** Applying Advanced Data Privacy Methods to Real-World Data (WS1 & WS2)
**WS1/WS2 Co-Leads:** Long Luu, Mahesh Kondepati
**Date:** 12 February 2026

---

## Table of Contents

1. [Project Context and Why These 3 Papers Matter](#1-project-context-and-why-these-3-papers-matter)
2. [Detailed Paper Reviews (with Assigned Questions)](#2-detailed-paper-reviews)
   - 2.1 Paper 10 — Carlini et al. (USENIX Security 2021)
   - 2.2 Paper 11 — HELM (Liang et al., 2022)
   - 2.3 Paper 12 — Hager et al. (Nature Medicine 2024)
3. [Cross-Paper Synthesis into a PHUSE Evaluation "Stack"](#3-cross-paper-synthesis)
4. [Similar and Related Papers](#4-similar-and-related-papers)
5. [Recommended Analysis Methods](#5-recommended-analysis-methods)
6. [Tooling, Language, and WS2 Engineering Alignment](#6-tooling-language-and-ws2-engineering-alignment)
7. [Strategy Analysis Plan (Aligned to Integrated Timeline)](#7-strategy-analysis-plan-aligned-to-integrated-timeline)
8. [Validation / Diagnostic Checks ("Do Not Ship Without")](#8-validation-and-diagnostic-checks)
9. [Expertise and Training Requirements](#9-expertise-and-training-requirements)
10. [Bottom-Line Recommendation: What To Do Next](#10-bottom-line-recommendation)
11. [References (Verified)](#11-references)

---

## 1. Project Context and Why These 3 Papers Matter

The Tech Implementation Plan frames WS1/WS2 around a **data-type-driven risk model**:

- **Structured → Clinical Text → Medical Imaging** (increasing complexity/risk)
  [Tech Implementation Plan §3, Guiding Principles]
- Clinical notes are explicitly the **highest-risk modality** because they contain direct PHI and are
  "extremely susceptible to LLM data extraction attacks."
  [Tech Implementation Plan §4.2.1, Risk Profile]
- WS2 must deliver **reproducible, auditable pipelines** with `uv` locking, DVC versioning, MLflow tracking,
  and CI/CD governance gates.
  [Tech Implementation Plan §5]

The three assigned papers map cleanly onto the plan's clinical-text risk and evaluation requirements:

| Paper | PHUSE Pillar | Maps to Plan Section |
|-------|-------------|---------------------|
| Carlini et al. (2021) | **Privacy failure mode:** memorization & extraction | §4.2.2 — Privacy Attacks / Evaluations (membership inference, data extraction, dataset inference) |
| HELM (Liang et al., 2022) | **Multi-metric evaluation** beyond accuracy-only | §4.1–4.3 — Metrics (utility, privacy, fairness, robustness, efficiency) |
| Hager et al. (2024) | **Clinical failure modes:** hallucination, instruction fragility, bedside risk | §4.2.3 — Benchmark Tasks; §5 — Governance & auditability |

---

## 2. Detailed Paper Reviews

### 2.1 Paper 10 — Carlini et al. (USENIX Security 2021)
**"Extracting Training Data from Large Language Models"**

#### Summary

This landmark paper demonstrates that an adversary with only **black-box query access** to GPT-2 can extract hundreds of verbatim training sequences — including PII, code, URLs, and 128-bit UUIDs — directly disproving the assumption that training on large corpora provides inherent anonymization.

#### Q1: What specific attack methodologies successfully extracted memorized training data?

Carlini et al. demonstrate a practical extraction pipeline with three stages:

**Stage 1 — Large-scale sampling (generation)**
- Generate a very large set of model outputs (reported scale: **600,000 samples** in one analysis). [Carlini et al. §5, USENIX]
- Use multiple sampling/configuration variants (top-k, temperature-based) to increase diversity and expose rare memorized strings.

**Stage 2 — Ranking candidates with "membership-style" heuristics**
Generated outputs are ranked using multiple scoring signals:
- **Perplexity-based ranking** — sequences the model assigns unusually low perplexity to are likely memorized.
- **Zlib entropy comparison** — identifies content that is "surprising for compression, but high-likelihood for the model" (outliers).
- **Casing-based comparison** — checking if the model's confidence changes when text is lowercased.
- **Sliding-window approach** — detects memorized substrings within longer outputs.
- **Likelihood ratios against smaller/medium models** — content memorized by a large model but not a small one signals memorization, not generic pattern.

**Stage 3 — Manual + confirmatory verification**
- **Manual inspection uses internet searching** to find exact matches for extracted candidates. [Carlini et al. §5.1, USENIX]
- They then **collaborate with the GPT-2 authors** for limited matching against the original training dataset to confirm memorization. [Carlini et al. §5.2, USENIX]
- *Note:* This is NOT a simple automated lookup against WebText; it involves manual internet search followed by author-assisted dataset confirmation.

**Key quantitative anchor:** Among 600,000 generated samples, their attacks find at least **604 containing memorized text** (explicitly a lower bound, not an exhaustive count). [Carlini et al. §5, USENIX]

#### Q2: What model/data characteristics made extraction easier (informing "high-risk" profiles)?

| Characteristic | Effect on Risk | Evidence |
|---------------|---------------|----------|
| **Larger model size** | Higher risk | GPT-2 XL (1.5B params) leaked significantly more than GPT-2 Small (117M) |
| **Data duplication** | Strongly increases risk | Repeated training strings are far more likely to be memorized and recovered |
| **High-entropy / unique strings** | Higher risk | UUIDs, contact info, exact sequences stand out under scoring heuristics |
| **Prompt/context overlap** | Higher risk | Conditioning on text resembling training distribution increases recovery (they sample short contexts from scraped sources for conditioning) [Carlini et al. §4, USENIX] |

#### Translating to PHUSE "High-Risk Data Type Profiles" (Health/RWD)

The Carlini paper is about text, but the Tech Implementation Plan already defines modality risk profiles. Combined, they imply:

| Data Type | Risk Level | Primary Threat | Plan Reference |
|-----------|------------|----------------|----------------|
| Clinical notes / unstructured text | **Critical** | Direct PHI + high susceptibility to extraction attacks | §4.2.1 |
| Radiology/pathology reports | **Critical** | Unstructured text with embedded PHI and rare disease descriptions | §4.2.1 |
| Medical images | **High** | PHI in overlays, facial reconstruction risk, inversion attacks | §4.3.1 |
| Structured tabular (EHR) | **High** | Risk concentrates in quasi-identifiers/linkage/uniqueness rather than verbatim memorization (different threat model) | §4.1.1 |

#### Review Guidelines Answers (Carlini)

- **Tools:** Reference extraction codebase is public: [LM_Memorization](https://github.com/ftramer/LM_Memorization); Python (HuggingFace Transformers, PyTorch)
- **Suitable data types:** Any text data used for LLM training/fine-tuning, especially clinical notes
- **Programming language:** Python (primary)
- **Bottlenecks:**
  - Compute cost (large generation volume — 600K+ samples)
  - Manual verification burden (internet search + author collaboration)
  - Extraction success depends heavily on training data duplication and domain
- **Validation checks:** See [Section 8](#8-validation-and-diagnostic-checks)
- **Expertise required:** ML/NLP + privacy attack familiarity
- **Generalization:** Methodology extends beyond GPT-2; later work (Nasr et al., 2023) shows scalable extraction against production systems

---

### 2.2 Paper 11 — HELM (Liang et al., 2022)
**"Holistic Evaluation of Language Models"**

#### Summary

HELM is a taxonomy-driven evaluation framework from Stanford CRFM that measures models across standardized **scenarios** (task/domain pairs) and **7 metric dimensions**. HELM reports evaluating **30 models on 42 scenarios** and notes that most prior work covered only **17.9%** of the scenarios they consider. [Liang et al., arXiv:2211.09110]

#### Q1: What is the comprehensive evaluation framework proposed?

**Top-Down Taxonomy:**
- **Scenarios** = (Task, Domain) pairs — 42 total, with 16 designated "core" scenarios
- **Metrics** = 7 dimensions capturing pluralistic desiderata:

| Metric | Definition | Clinical/Health Relevance |
|--------|-----------|--------------------------|
| **Accuracy** | Correctness of outputs | Diagnostic accuracy, coding accuracy |
| **Calibration** | Alignment of confidence with correctness | Critical — overconfident wrong answers are dangerous at the bedside |
| **Robustness** | Performance under distribution shift/perturbation | Vital — clinical data varies across institutions, demographics, time periods |
| **Fairness** | Equitable performance across demographic groups | Essential for health equity |
| **Bias** | Stereotypical or prejudicial associations | Can perpetuate healthcare disparities |
| **Toxicity** | Generation of harmful/offensive content | Risk of inappropriate clinical communication |
| **Efficiency** | Computational cost | Practical constraint for clinical deployment (on-prem, secure environments) |

#### Q2: Which HELM dimensions are most critical/transferable for clinical/health LLM safety?

**Priority ordering for PHUSE WS1:**

1. **Robustness** (HIGHEST) — Clinical inputs vary across institutions, time, and documentation style. Robustness also covers prompt sensitivity — an issue Hager shows is clinically dangerous.
2. **Calibration** (HIGH) — Overconfident wrong outputs are particularly hazardous in clinical decision support.
3. **Fairness / Bias** (HIGH) — Must be audited because bias is frequently reported in medical AI/LLM evaluation studies. *Note on statistics:* A systematic review of 24 peer-reviewed studies found that **91.7% of included studies** reported measurable bias — this is a study-level finding (i.e., "91.7% of studies found bias"), not a claim that 91.7% of individual models are biased. The distinction matters for benchmark specification: it means bias is *pervasive across research contexts*, not a property of a specific model percentage. [Systematic review, PMC]
4. **Accuracy** (BASELINE) — Necessary but insufficient. Hager shows accuracy on medical exams does not predict clinical readiness.
5. **Toxicity** (MODERATE) — Less directly clinical but important for patient-facing applications.
6. **Efficiency** (PRACTICAL) — Relevant for WS2 deployment considerations.

#### PHUSE Extension: Augmenting HELM for WS1/WS2

The Tech Implementation Plan already requires metrics beyond generic "accuracy," including privacy and robustness [§4.1–4.3, Metrics]. For clinical safety + privacy, HELM should be augmented with:

| Additional Dimension | Source | Not in Original HELM |
|---------------------|--------|---------------------|
| **Privacy leakage metrics** | Carlini (Paper 10) | Extraction success rate, MIA AUC, dataset inference accuracy |
| **Hallucination / instruction-following rate** | Hager (Paper 12) | Tool hallucination count, format error rate per encounter |
| **Guideline adherence** | Hager (Paper 12) | Compliance with clinical practice guidelines (task-dependent) |

*Labeling:* These are **Proposed PHUSE extensions** (not HELM paper recommendations). The HELM paper provides the scaffold; we add the clinical-privacy dimensions.

#### Review Guidelines Answers (HELM)

- **Tools:** HELM is open-source — [crfm.stanford.edu/helm](https://crfm.stanford.edu/helm/latest/); Python-based toolkit
- **Suitable data types:** Any text-based LLM evaluation task; adaptable to structured data with appropriate scenario definitions
- **Programming language:** Python (primary)
- **Bottlenecks:** Evaluation requires substantial compute for 30+ models; scenario design for clinical domains needs expert clinical input; fairness/bias metrics require demographic annotations
- **Validation:** Standardized leaderboards and comparison dashboards; confidence intervals; ablation by scenario
- **Expertise required:** ML engineering + clinical domain expertise + statistical interpretation
- **Generalization:** Highly generalizable — the core contribution is the framework taxonomy, not specific scenarios. Can be extended to clinical scenarios using MIMIC, i2b2/n2c2 datasets

---

### 2.3 Paper 12 — Hager et al. (Nature Medicine 2024)
**"Evaluation and Mitigation of the Limitations of Large Language Models in Clinical Decision-Making"**

#### Summary

Using **2,400 real MIMIC-based patient cases** across four common abdominal pathologies, this paper simulates a realistic clinical decision-making environment. It demonstrates that current LLMs fail at autonomous clinical tasks — performing significantly worse than physicians, hallucinating non-existent tools, failing to follow guidelines, and being unable to interpret lab results.

Importantly, the authors note that **MIMIC-IV data-use constraints prevent sending data to external closed-source model APIs**, which has implications for how clinical LLMs can be evaluated/deployed. [Hager et al., PMC]

#### Q1: What concrete clinical risks/potential harms are identified?

All findings below are **directly supported by the paper** with citations:

| Risk | Quantitative Finding | Citation |
|------|---------------------|----------|
| **Instruction-following failures** | "Making errors every two to four patients" when providing actions | [Hager et al., PMC] |
| **Tool hallucinations** | "Hallucinating nonexistent tools every two to five patients" | [Hager et al., PMC] |
| **Inferior diagnostic performance** | LLMs performed significantly worse than physicians across all pathologies | [Hager et al., PMC] |
| **Prompt fragility** | Changing "final diagnosis" → "main diagnosis" or reordering lab results (same content) caused large performance swings | [Hager et al., §Results] |
| **Information overload paradox** | More clinical information sometimes *decreased* accuracy, suggesting sensitivity to noise-to-signal ratio | [Hager et al., §Discussion] |

#### Q2: What governance/safety measures are recommended?

**Directly supported by the paper's findings:**

| Measure | Paper Basis | Citation |
|---------|------------|----------|
| **Human supervision is required** | Frequent instruction errors and tool hallucinations necessitate mandatory clinician oversight | [Hager et al., PMC] |
| **Robust evaluation must include workflow realism** | Exam-style QA is insufficient; models fail at execution even when they "know facts" | [Hager et al., §Discussion] |
| **Structured output validation** | Format/instruction errors every 2–4 patients make unvalidated integration unsafe | [Hager et al., §Results] |

**Proposed PHUSE extensions (engineering best practice, NOT explicit Hager recommendations):**

| Measure | Rationale | WS1/WS2 Mapping |
|---------|-----------|------------------|
| Multi-model ensemble with disagreement detection | Reduces single-model failure risk; disagreement signals uncertainty | WS2 implementation pattern |
| Tiered deployment (low-risk tasks first) | Start with summarization/coding before clinical decision support | WS3 risk-based prioritization |
| Continuous post-deployment monitoring | Track hallucination rates, bias drift, accuracy degradation over time | WS2 MLflow + governance logs |
| Prompt perturbation test suites | Systematic robustness testing before any deployment | WS1 benchmark task |
| CI/CD governance gates on critical safety metrics | Automated gating on hallucination rate, guideline adherence thresholds | WS2 CI/CD [Tech Plan §5.7] |

#### Review Guidelines Answers (Hager)

- **Tools:** Custom simulation framework built on MIMIC; Python-based evaluation pipeline
- **Suitable data types:** Structured EHR (labs, vitals, demographics) + clinical notes; multi-modal clinical scenarios
- **Programming language:** Python (primary)
- **Bottlenecks:** PhysioNet credentialing required; simulation needs clinical expert input; evaluation is disease-specific; prompt sensitivity makes reproducibility challenging; MIMIC DUA prevents use of external closed-source APIs
- **Validation:** Physician baselines; hallucination rates per encounter; controlled perturbation studies
- **Expertise required:** Clinical domain expertise (physician involvement essential); ML/NLP engineering; clinical informatics
- **Generalization:** Framework generalizes to other clinical tasks but requires disease-specific scenario design; systematic failure nature suggests findings apply to current-generation LLMs broadly

---

## 3. Cross-Paper Synthesis

### The Combined Insight

- **Carlini** shows privacy leakage can occur via memorization/extraction. [USENIX]
- **HELM** shows single-metric "accuracy" evaluation is inadequate — we need multi-dimensional measurement. [arXiv]
- **Hager** shows clinical failure modes are frequent and workflow-breaking, so evaluation must include execution realism and instruction adherence. [PMC]

### The "Knowledge-Execution Gap"

This is the critical insight that emerges across all three papers:
- HELM documents that models can score well on accuracy benchmarks.
- Hager demonstrates they fail in realistic clinical execution.
- Carlini shows that even "well-behaved" models carry hidden privacy risks invisible to standard benchmarks.

**Implication for WS1:** Benchmark design MUST include execution-realistic scenarios (not just exam-style QA) and privacy attack evaluation (not just utility metrics).

### PHUSE WS1/WS2 Translation (Practical)

For any benchmark involving LLMs on health/RWD, evaluate along these dimensions:

| # | Dimension | Source | Metric Examples |
|---|-----------|--------|-----------------|
| 1 | **Privacy risk** | Carlini (Paper 10) | Extraction yield, MIA AUC-ROC, dataset inference accuracy |
| 2 | **Utility / accuracy** | HELM (Paper 11) | AUC, F1, BLEU, clinical accuracy |
| 3 | **Robustness** | HELM + Hager (Papers 11, 12) | Performance under perturbation + cross-site generalization |
| 4 | **Calibration** | HELM (Paper 11) | Expected Calibration Error (ECE), reliability diagrams |
| 5 | **Fairness / bias** | HELM (Paper 11) | Subgroup metrics (equalized odds, demographic parity) |
| 6 | **Clinical safety** | Hager (Paper 12) | Hallucination rate, tool hallucination rate, instruction adherence |
| 7 | **Efficiency** | HELM (Paper 11) | Compute cost, latency, memory footprint |
| 8 | **Reproducibility** | WS2 Principles | Hash consistency, environment portability |

This is directly compatible with WS1's deliverable structure [§4.4] and WS2's logging expectations [§5.6].

---

## 4. Similar and Related Papers

### 4.1 Privacy Attacks / Leakage (closest to Carlini)

| Paper | Year | Venue | Key Contribution |
|-------|------|-------|------------------|
| Nasr et al., "Scalable Extraction of Training Data from Production LMs" | 2023 | arXiv | Divergence attacks on production ChatGPT — important for threat realism |
| ProPILE (Kim et al.) | 2023 | NeurIPS | Probing privacy leakage at scale — useful for building benchmark probes |
| Lukas et al., "Analyzing Leakage of PII in Language Models" | 2023 | IEEE S&P | Systematic PII leakage perspective with attack taxonomy |
| Fu et al. (SPV-MIA) | 2024 | NeurIPS | Practical membership inference against fine-tuned LLMs with "self-prompt calibration" |
| Lehman et al., "Does BERT on Clinical Notes Reveal PHI?" | 2021 | NAACL | Baseline showing leakage may differ by architecture/training |
| Min-K%++ | 2024 | arXiv | Improved methods for detecting pre-training data membership |
| SoK: Privacy-aware LLM in Healthcare | 2026 | arXiv | Phase-aware threat model (preprocessing/fine-tuning/inference); directly maps to WS1 risk profiles |

### 4.2 Evaluation Frameworks (closest to HELM)

| Paper | Year | Venue | Key Contribution |
|-------|------|-------|------------------|
| MedHELM (Stanford) | 2025 | Stanford CRFM | HELM adapted for 121 medical tasks — primary candidate for WS1 evaluation framework |
| MEDIC | 2024 | arXiv | 5-dimension clinical competence framework; identifies "knowledge-execution gap" |
| CSEDB (Clinical Safety-Effectiveness Dual-Track) | 2025 | npj Digital Medicine | 30 metrics across 26 clinical departments; 2,069 expert Q&A items |
| QUEST | 2024 | PMC | Human evaluation framework (Planning → Implementation → Adjudication) |
| AgentClinic | 2024 | arXiv | Multimodal agent benchmark simulating clinical environments |

### 4.3 Clinical Safety / Hallucinations (closest to Hager)

| Paper | Year | Venue | Key Contribution |
|-------|------|-------|------------------|
| Asgari et al. — Clinical Safety Framework | 2025 | npj Digital Medicine | 1.47% hallucination rate, 3.45% omission rate across 12,999 annotated sentences |
| Omar et al. — Adversarial Hallucination Attacks | 2025 | Communications Medicine | LLMs repeat planted fake clinical data in up to 83% of cases |
| Kim et al. — Medical Hallucination in Foundation Models | 2025 | medRxiv | Taxonomy of medical hallucination types and causes |
| Systematic review of bias in medical LLMs | 2025 | PMC | 91.7% of 24 studies reported measurable bias (study-level, not model-level) |

### 4.4 PET Methods & Implementation Guidance (ties to Tech Plan)

| Paper | Year | Venue | Key Contribution |
|-------|------|-------|------------------|
| Privacy-Preserving Strategies for EHRs in the Era of LLMs | 2025 | npj Digital Medicine | Overview: local deployment, synthetic data, DP, de-identification |
| Differential Privacy for Medical Deep Learning | 2026 | npj Digital Medicine | Methods, tradeoffs, deployment; privacy-utility-fairness interactions |
| Evaluation of Open-Source DP Tools | 2023 | PMC | Benchmarked OpenDP/SmartNoise, Opacus, TF Privacy — tool selection guide for WS2 |
| FL + DP for Breast Cancer | 2025 | Scientific Reports | FL+DP achieves 96.1% accuracy at epsilon=1.9 — validates privacy-utility tradeoff |
| SoK: Privacy-aware LLM in Healthcare | 2026 | arXiv | Comprehensive systematization; phase-aware threat model |

---

## 5. Recommended Analysis Methods

### 5.1 Privacy Evaluation Methods (LLM + Health Text)

These align directly with the plan's "Privacy Attacks / Evaluations" for clinical notes [§4.2.2].

#### A) Training-Data Extraction (Carlini-style)

- **Goal:** Can the model output verbatim PHI-like strings from its training/fine-tuning set?
- **Method:** Large-scale sampling → ranking (perplexity, zlib, likelihood ratios) → verification (internet search + held-out known strings)
- **Primary metrics:** Extraction yield; precision@k; PHI-category breakdown
- **Threat model assumption:** Black-box query access to target model

#### B) Membership Inference Attacks (MIA)

- **Goal:** Can an attacker determine whether a specific record was used in training?
- **Method options:** Loss/perplexity-based; reference-model calibration; SPV-MIA style [Fu et al., NeurIPS 2024]
- **Primary metrics:** AUC-ROC; TPR at low FPR (health context demands very low FPR)
- **Threat model assumption:** Black-box or gray-box access; attacker knows data distribution

#### C) Dataset Inference / Pre-training Data Detection

- **Goal:** "Did this dataset appear in training?" (governance + contamination detection)
- **Method options:** Min-K%++ style detectors [arXiv, 2024]
- **Primary metrics:** AUROC on membership labels
- **Threat model assumption:** Black-box query access

#### D) PII/PHI Leakage Probing

- **Goal:** Can the model be induced to emit PHI even if not verbatim memorized?
- **Method:** Targeted prompts; red-teaming evaluation; PII detectors + manual adjudication
- **Primary metrics:** PHI leak rate by category (name, date, MRN, SSN, etc.)
- **Threat model assumption:** Adversarial user with knowledge of data domain

#### How These Become Benchmarkable WS1 Tasks

| Attack Type | WS1 Benchmark Task Name | Input | Output | Success Metric |
|-------------|------------------------|-------|--------|----------------|
| Extraction | `text_llm_privacy/extraction` | Model under test + generation config | Ranked candidate sequences | Extraction yield, precision@k |
| MIA | `text_llm_privacy/mia` | Model + member/non-member samples | Membership predictions | AUC-ROC, TPR@FPR=0.01 |
| Dataset inference | `text_llm_privacy/dataset_inference` | Model + candidate datasets | In/out predictions | AUROC |
| PHI probing | `text_llm_privacy/phi_probing` | Model + targeted prompts | Detected PHI in outputs | PHI leak rate per 1000 queries |

### 5.2 Utility & Safety Evaluation Methods (HELM + Clinical Extensions)

**Use HELM dimensions as the core scaffold [arXiv:2211.09110].**

| Category | Method | Metric | Datasets |
|----------|--------|--------|----------|
| **Utility (structured)** | ICU mortality/readmission prediction | AUC, F1 | MIMIC-IV, eICU, HiRID |
| **Utility (text)** | De-ID F1, concept extraction, classification, QA | F1, recall, BLEU | MIMIC-IV-Note, i2b2/n2c2 |
| **Robustness** | Prompt perturbation (synonyms, reorder, formatting); cross-dataset shift | Delta-accuracy under perturbation | MIMIC vs eICU vs HiRID |
| **Calibration** | Expected Calibration Error (ECE), reliability diagrams | ECE | All tasks |
| **Fairness / bias** | Subgroup performance by sex/age/race | Equalized odds, demographic parity | Where demographic annotations available |
| **Clinical safety** | Hallucination rate, tool hallucination rate, instruction-following error rate | Events per N patients | MIMIC-based simulation |
| **Guideline adherence** | Compare LLM outputs against clinical practice guidelines | Compliance % | Where task has guideline target |

### 5.3 PET Method Evaluations (WS1/WS2 scope)

From the Tech Implementation Plan, the main PET families in scope:

| PET Family | Tool(s) | Data Type | Plan Reference |
|-----------|---------|-----------|----------------|
| DP analytics | SmartNoise/OpenDP | Structured tabular | §4.1.2 |
| DP-SGD | PyTorch Opacus | Tabular + text | §4.1.2, §4.2.2 |
| Synthetic data | SDV/SDGym | Structured tabular | §4.1.2 |
| Federated Learning | Flower | All modalities | §4.1.2, §4.3.2 |
| Secure aggregation | Flower + custom | Imaging + tabular | §4.3.2 |
| De-identification (text) | Presidio, Philter, BERT-NER | Clinical notes | §4.2.2 |
| De-identification (imaging) | DICOM scrubbing, face blurring | Medical images | §4.3.2 |
| RAG over DP/de-ID summaries | LangChain + OpenDP | Clinical text | §4.2.2 |

---

## 6. Tooling, Language, and WS2 Engineering Alignment

### Programming Language Choice

The Tech Implementation Plan mandates **Python-first**, with optional R for companion statistical workflows [§5.2]:

| Task Category | Primary | Secondary | Rationale |
|---------------|---------|-----------|-----------|
| Privacy attacks (MIA, extraction) | Python | — | All attack frameworks are Python-native |
| LLM evaluation (HELM, MEDIC) | Python | — | HELM toolkit is Python |
| DP analytics | Python | R | OpenDP/SmartNoise (Python); diffpriv/DPpack (R) |
| Synthetic data | Python | R | SDV (Python); synthpop (R) |
| Federated learning | Python | — | Flower is Python-native |
| Biostatistical analysis | R | Python | R excels at survival analysis, classical biostats |
| De-identification | Python | — | Presidio, Philter, HuggingFace all Python |
| Reporting / Documentation | Quarto | — | Supports both Python and R; outputs HTML/PDF |

### Non-Negotiable WS2 Infrastructure (from Tech Plan §5)

| Component | Tool | Plan Reference |
|-----------|------|----------------|
| Environment management | `uv` (Python), `renv` (R) | §5.3 |
| Data/model versioning | DVC | §5.6 |
| Experiment tracking | MLflow | §5.6 |
| Workflow orchestration | Snakemake / Nextflow + Hydra configs | §5.5 |
| CI/CD | GitHub Actions | §5.7 |
| Documentation | Quarto + nbdev | §5.8 |
| Containerization | Docker / Singularity | §5.9 |

---

## 7. Strategy Analysis Plan (Aligned to Integrated Timeline)

This is mapped directly to the Tech Implementation Plan's "Integrated Timeline" [§6] so WS1/WS2 leads see immediate alignment.

### Phase 1: Now → End of Q1 2026 — Research + Scoping for Structured Data
*[Maps to Tech Plan: "Q1 2026 — Structured Data (Research + Scoping)"]*

**Primary objective:** Lock WS1 structured benchmarks + stand up WS2 "minimum viable pipeline."

| # | Deliverable | Status |
|---|-------------|--------|
| 1 | Literature review on DP, synthetic data, FL for structured EHRs | In progress (this document) |
| 2 | Finalize dataset selection (MIMIC-IV, eICU, HiRID) [Tech Plan §4.1.4] | To confirm |
| 3 | Draft WS1 benchmark specs for structured data (tasks, metrics, PET mapping) | To draft |
| 4 | WS2: Repository skeleton + `uv` environment + DVC + MLflow + CI/CD skeleton [Tech Plan §5] | To set up |
| 5 | WS2: E2E test pipeline on an example dataset/model/benchmark [Tech Plan §6, Q1] | To implement |
| 6 | Obtain PhysioNet credentials for MIMIC-IV, MIMIC-IV-Note, MIMIC-CXR | Start immediately |

**For the E2E test pipeline,** implement one simple tabular benchmark (e.g., ICU mortality on MIMIC-IV with DP-SGD at a single epsilon) and ensure the run emits:
- Config YAML/JSON
- DP parameters (ε, δ)
- ML metrics (AUC, F1)
- MLflow artifact
- DVC-tracked data pointer

This validates the entire WS2 infrastructure stack before investing in full benchmarks.

### Phase 2: Q2 2026 — Structured Data (Implementation + Validation)
*[Maps to Tech Plan: "Q2 2026 — Structured Data (Implementation + Validation)"]*

**Implement and validate:**

| Benchmark | Tool | Dataset | Tasks | Metrics |
|-----------|------|---------|-------|---------|
| DP analytics | OpenDP/SmartNoise | MIMIC-IV | Demographic tables, risk factor summaries | Utility vs ε |
| DP-SGD tabular | Opacus | MIMIC-IV | ICU mortality prediction | AUC vs ε |
| Synthetic tabular | SDV (CTGAN, TVAE), SDGym | MIMIC-IV, eICU | Train-synthetic / test-real fidelity | Statistical similarity, ML utility, MIA success rate |
| Federated learning | Flower | MIMIC-IV + eICU | ICU readmission prediction | Accuracy vs centralized baseline |
| Privacy attacks (structured) | Custom | Models from above | MIA against DP vs non-DP models | MIA AUC-ROC; privacy-utility curves |

**Additional deliverables:**
- Draft reports, governance logs, dataset/model cards [Tech Plan §5.8]
- Validate reproducibility across multiple volunteer environments [Tech Plan §6, Q2]
- **Release WS1–WS2 Structured Benchmarks v1.0** [Tech Plan §6, Q2]

### <a id="phase-3-q3-2026--clinical-notes-research--design"></a>Phase 3: Q3 2026 — Clinical Notes (Research + Design)
*[Maps to Tech Plan: "Q3 2026 — Clinical Notes (Research + Design)"]*

**This is where the three assigned papers become central.** Plan-aligned goals:

| # | Task | Informed By |
|---|------|-------------|
| 1 | Deep-dive research on clinical text risk profiles | Carlini (Paper 10) + SoK (2026) |
| 2 | Evaluate de-identification methods (NER, LLM-assisted, hybrid) | Presidio, Philter, BERT-NER benchmarks |
| 3 | Review LLM privacy attack literature (extraction, MIAs) | Carlini + Nasr + Fu + Min-K%++ |
| 4 | Select text datasets (MIMIC-IV-Note, CXR reports, i2b2/n2c2, TCIA notes) | Tech Plan §4.2.4 |
| 5 | Define benchmark tasks: de-ID, concept extraction, classification, RAG on safe summaries, DP synthetic text | Tech Plan §4.2.3 |
| 6 | Define privacy attack benchmark module (see threat model below) | Carlini + WS1 §4.2.2 |
| 7 | WS2: Build initial pipelines + configs for text workflows; E2E test on example | Tech Plan §6, Q3 |

**Explicit Threat Model for LLM Privacy Attacks on Clinical Text:**

| Element | Specification |
|---------|--------------|
| **Adversary goal** | Extract memorized PHI or determine training set membership for clinical text |
| **Adversary access** | Black-box query access (realistic for API-deployed models); gray-box for MIA (loss values available) |
| **Adversary knowledge** | Knows target data domain (clinical text); may have partial knowledge of training distribution |
| **Target model** | Any LLM fine-tuned on clinical notes (e.g., models fine-tuned on MIMIC-IV-Note) |
| **Attack types** | (a) Extraction [Carlini], (b) MIA [Fu/SPV-MIA], (c) Dataset inference [Min-K%++], (d) PHI probing |
| **Success metrics** | Extraction yield; MIA AUC-ROC; MIA TPR@FPR=0.01; PHI leak rate per 1000 queries |
| **Defense evaluation** | Compare attack success: unprotected vs DP fine-tuned vs de-identified training data |

### Phase 4: Q4 2026 — Clinical Notes (Implementation + Privacy Benchmarks)
*[Maps to Tech Plan: "Q4 2026 — Clinical Notes (Implementation + Privacy Benchmarks)"]*

| Benchmark | Tools | Datasets | Tasks | Key Metrics |
|-----------|-------|----------|-------|-------------|
| De-identification | Presidio, Philter, BERT-NER, LLM-assisted | MIMIC-IV-Note, i2b2/n2c2 | PHI detection | F1, recall (prioritize recall), FN rate |
| LLM privacy attacks | LM_Memorization, custom MIA, Min-K%++ | Fine-tuned clinical LLMs | Extraction, MIA, dataset inference, PHI probing | See threat model above |
| LLM safety evaluation | HELM/MedHELM adapted + Hager-style simulation | MIMIC-IV-Note | Hallucination rate, guideline adherence, prompt robustness | Events/patient, compliance %, delta-accuracy |
| DP fine-tuning | Opacus (DP-SGD) | Clinical text models | DP fine-tuning for small models | Utility (BLEU, ROUGE) vs ε |
| RAG on safe summaries | LangChain + OpenDP | De-ID/DP summaries | QA over privacy-protected text | QA accuracy, extraction resistance |
| DP/synthetic text eval | Custom | Generated text | Utility vs leakage | Privacy-utility curves |

**Output:** Release **WS1–WS2 Clinical Notes Benchmarks v1.0** [Tech Plan §6, Q4]

### Phase 5 (Optional): Q1 2027 — Medical Imaging
*[Maps to Tech Plan: "Optional (Q1 2027) — Medical Imaging"]*

The plan notes "DP imaging currently poor utility" and focuses on de-ID + FL + secure aggregation [§4.3]:
- DICOM metadata scrubbing, burn-in text removal, face blurring
- FL imaging benchmarks (MIMIC-CXR, CheXpert, NIH ChestX-ray14)
- Secure aggregation evaluations
- Release WS1–WS2 Imaging Benchmarks v1.0

---

## 8. Validation and Diagnostic Checks

These are the **"do not ship without"** checklist items for WS2 pipeline quality.

### Privacy (LLM/Text) Validation

- [ ] Repeat attacks across seeds + sampling parameters (minimum 3 runs)
- [ ] Deduplicate outputs before counting leaks (avoid inflated extraction counts)
- [ ] Measure precision@k on manually inspected candidates (not just automated counts)
- [ ] Log all prompts used for extraction (for reproducibility)
- [ ] Record "PHI-like" category breakdown (name, date, MRN, SSN, location, etc.)
- [ ] Report MIA AUC-ROC with confidence intervals

### Privacy (Structured/PET) Validation

- [ ] Explicit privacy accounting logs (ε, δ, clipping norm, sampling rate) for every run
- [ ] Utility vs privacy curves (not just single-point results)
- [ ] Measure privacy-induced fairness degradation (DP can disproportionately affect underrepresented groups) [npj Digital Medicine, 2026]
- [ ] Synthetic data: statistical similarity tests + ML fidelity + MIA evaluation

### Robustness / Safety Validation

- [ ] Prompt perturbation test suite (format changes, content reordering, synonym substitution)
- [ ] Distribution shift tests across at least 2 datasets (e.g., MIMIC vs eICU)
- [ ] Instruction-following error rate + tool hallucination rate reporting (per Hager)
- [ ] Guideline adherence scoring where applicable

### Engineering Reproducibility

- [ ] Frozen environment builds (`uv.lock`) [Tech Plan §5.3]
- [ ] Container image validation (Docker/Singularity) [Tech Plan §5.9]
- [ ] Hash consistency of outputs across runs with same seed
- [ ] DVC-tracked data lineage for every experiment
- [ ] MLflow artifact store populated with all metrics + DP parameters

---

## 9. Expertise and Training Requirements

To execute the plan successfully, the following expertise profiles are needed:

| Role | Key Skills | Maps To |
|------|-----------|---------|
| **Privacy / ML security lead** | Privacy attack implementation, DP theory, MIA, memorization analysis | WS1 privacy evaluation; Papers 10 + related |
| **ML engineer** | Pipeline development, model training/eval, CI/CD, containerization | WS2 pipelines; all benchmarks |
| **Clinician / clinical informatician** | Scenario realism, guideline targets, harm adjudication, clinical validation | WS1 benchmark design; Paper 12 |
| **Data steward / governance liaison** | Dataset access constraints, audit logs, compliance alignment, DUA management | WS3 interface; PhysioNet credentialing |
| **Biostatistician** (optional, R companion) | Survival analysis, classical DP analytics, fairness metrics | WS1 structured data; companion R pipelines |

This matches the plan's emphasis on governance-aware logs and cross-environment reproducibility [§5.1, §5.4].

---

## 10. Bottom-Line Recommendation

Given today is **12 February 2026**, the highest-leverage "lead actions" that unblock everything else:

### Immediate (This Week)

1. **Lock the WS2 repo skeleton + reproducibility contract** — `uv` / DVC / MLflow / CI. This is the foundation for every pipeline that follows. [Tech Plan §5]
2. **Start PhysioNet credentialing** for MIMIC-IV, MIMIC-IV-Note — this is on the critical path and can be slow.
3. **Circulate this review** to WS1/WS2 leads (Long Luu, Mahesh Kondepati) for alignment and feedback.

### Next 2–4 Weeks

4. **Implement one E2E "hello benchmark"** (structured data, single DP-SGD run on MIMIC-IV). Validate that the run emits all required governance artifacts (config, DP params, metrics, MLflow entry, DVC pointer). "Copy this pattern for every other pipeline." [Tech Plan §6, Q1]
5. **Define the privacy attack benchmark module** for clinical text [§4.2.2] using the explicit threat model in [Section 7, Phase 3](#phase-3-q3-2026--clinical-notes-research--design), with:
   - Extraction (Carlini/Nasr style)
   - Membership inference (Fu/SPV-MIA style)
   - Dataset inference (Min-K%++ style)
   - PHI probing (red-team style)

### Short-Term (Rest of Q1)

6. **Adopt HELM/MedHELM as the evaluation "spine"** and add the missing healthcare-critical metrics: privacy leakage + hallucination/tool hallucination + instruction adherence. [arXiv]
7. **Draft WS1 structured benchmark specification** with tasks, datasets, metrics, and PET mapping.

### Anticipated Bottlenecks

| Bottleneck | Mitigation | Priority |
|-----------|-----------|----------|
| MIMIC credentialing delay | Start now; run initial prototypes on synthetic/public data | **Critical path** |
| GPU compute for LLM evaluation | Coordinate cloud/HPC access with WS2 leads early | High |
| Clinical expertise for Hager-style evaluation | Identify physician collaborators within PHUSE working group | High |
| Reproducibility across institutions | Containerize from day one (Docker/Singularity) | Medium |

---

## 11. References

### Assigned Papers (Verified)
1. Carlini, N., Tramèr, F., Wallace, E., et al. (2021). "Extracting Training Data from Large Language Models." *USENIX Security Symposium*, pp. 2633–2650.
   [USENIX](https://www.usenix.org/conference/usenixsecurity21/presentation/carlini-extracting) | [arXiv](https://arxiv.org/abs/2012.07805)
2. Liang, P., Bommasani, R., Lee, T., et al. (2022). "Holistic Evaluation of Language Models." *arXiv:2211.09110*.
   [arXiv](https://arxiv.org/abs/2211.09110) | [HELM Benchmark](https://crfm.stanford.edu/helm/latest/)
3. Hager, P., Jungmann, F., Holland, R., et al. (2024). "Evaluation and Mitigation of the Limitations of Large Language Models in Clinical Decision-Making." *Nature Medicine*, 30(9), pp. 2613–2622.
   [Nature Medicine](https://www.nature.com/articles/s41591-024-03097-1) | [PubMed/PMC](https://pubmed.ncbi.nlm.nih.gov/38965432/)

### Related Papers — Privacy Attacks
4. Carlini, N., et al. (2023). "Quantifying Memorization Across Neural Language Models." *ICLR 2023*.
5. Nasr, M., et al. (2023). "Scalable Extraction of Training Data from (Production) Language Models." *arXiv:2311.17035*.
6. Fu, Z., et al. (2024). "Membership Inference Attacks Against Fine-Tuned LLMs via Self-Prompt Calibration." *NeurIPS 2024*.
7. Lukas, N., et al. (2023). "Analyzing Leakage of PII in Language Models." *IEEE S&P 2023*.
8. Lehman, E., et al. (2021). "Does BERT Pretrained on Clinical Notes Reveal Sensitive Data?" *NAACL 2021*.
9. Shi, W., et al. (2024). "Detecting Pretraining Data from Large Language Models" (Min-K%++). *arXiv*.
10. "SoK: Privacy-aware LLM in Healthcare" (2026). [arXiv:2601.10004](https://arxiv.org/html/2601.10004v1)

### Related Papers — Evaluation Frameworks
11. MedHELM (Stanford, 2025). [Stanford CRFM](https://crfm.stanford.edu/helm/latest/)
12. MEDIC (2024). [arXiv:2409.07314](https://arxiv.org/abs/2409.07314)
13. CSEDB (2025). [npj Digital Medicine](https://www.nature.com/articles/s41746-025-02277-8)
14. QUEST (2024). [PMC](https://pmc.ncbi.nlm.nih.gov/)

### Related Papers — Clinical Safety
15. Asgari et al. (2025). [npj Digital Medicine](https://www.nature.com/articles/s41746-025-01670-7)
16. Omar et al. (2025). [Communications Medicine](https://www.nature.com/articles/s43856-025-01021-3)
17. Systematic review of bias in medical LLMs (2025). [PMC](https://pmc.ncbi.nlm.nih.gov/)

### Related Papers — Privacy-Preserving Technologies
18. "Differential Privacy for Medical Deep Learning." (2026). [npj Digital Medicine](https://www.nature.com/articles/s41746-025-02280-z)
19. "Privacy Preserving Strategies for EHRs in the Era of LLMs." (2025). [npj Digital Medicine](https://www.nature.com/articles/s41746-025-01429-0)
20. "Evaluation of Open-Source Tools for Differential Privacy." (2023). [PMC](https://pmc.ncbi.nlm.nih.gov/articles/PMC10386022/)

### Tools and Frameworks
21. OpenDP / SmartNoise — [opendp.org/tools](https://opendp.org/tools/)
22. HELM Toolkit — [crfm.stanford.edu/helm](https://crfm.stanford.edu/helm/latest/)
23. Flower (Federated Learning) — [flower.dev](https://flower.dev/)
24. SDV (Synthetic Data Vault) — [sdv.dev](https://sdv.dev/)
25. Microsoft Presidio — [microsoft.github.io/presidio](https://microsoft.github.io/presidio/)
26. PyTorch Opacus (DP-SGD) — [opacus.ai](https://opacus.ai/)
27. LM_Memorization (Carlini attack code) — [github.com/ftramer/LM_Memorization](https://github.com/ftramer/LM_Memorization)

---

*This document is structured to paste directly into PHUSE working group deliverables. All claims labeled "Directly supported by the paper" include citations. All items labeled "Proposed PHUSE extension" are clearly distinguished as engineering recommendations not explicit in the source papers.*
