# Implementation Report — Paper 12
# Hager et al. (Nature Medicine 2024): Hallucinations and Clinical Risk

**PHUSE Working Group: Applying Advanced Data Privacy Methods to Real-World Data**
**Reviewer:** Onkar, Alex | **Date:** 12 February 2026

---

## 1. Paper Summary

Using **2,400 MIMIC-based patient cases** across four abdominal pathologies, Hager et al. simulate a realistic clinical decision-making workflow. Key findings: LLMs make **instruction-following errors every 2–4 patients**, **hallucinate nonexistent tools every 2–5 patients**, perform **worse than physicians** on diagnosis, and are **highly sensitive to trivial prompt changes** (e.g., "final diagnosis" vs "main diagnosis"). The paper concludes that human supervision is mandatory, and evaluation must include workflow realism — not just exam-style QA.

Critically, the authors note that **MIMIC-IV data-use constraints prevent sending data to external closed-source APIs**, constraining clinical LLM evaluation to open-source or locally deployed models.

## 2. Relevance to PHUSE WS1/WS2

This paper shapes **two key WS1 deliverables**:
1. **Clinical safety benchmark tasks** [Tech Plan §4.2.3] — hallucination rate, instruction adherence, guideline compliance
2. **Governance requirements** [Tech Plan §5.7, §5.8] — mandatory validation checks, CI/CD gates on safety metrics, human-in-the-loop protocols

It also directly informs how WS2 pipelines must handle LLM outputs: **structured output validation** and **prompt robustness testing** should be built into every LLM-involving pipeline.

## 3. Implementation Plan

### 3.1 What We Will Implement

| Benchmark Task | Description | WS1 Pipeline Name |
|---------------|-------------|-------------------|
| **Hallucination rate measurement** | Count hallucinated facts/tools per clinical encounter | `text_llm_safety/hallucination` |
| **Instruction adherence scoring** | Measure format errors, invalid outputs per encounter | `text_llm_safety/instruction_adherence` |
| **Prompt robustness evaluation** | Systematic perturbation → measure accuracy delta | `text_llm_safety/prompt_robustness` |
| **Guideline adherence scoring** | Compare LLM recommendations vs clinical guidelines | `text_llm_safety/guideline_adherence` |
| **Physician baseline comparison** | Benchmark LLM performance against physician gold standard | `text_llm_safety/physician_baseline` |

### 3.2 Datasets

| Dataset | Type | Role | Access |
|---------|------|------|--------|
| **MIMIC-IV** (structured) | Labs, vitals, demographics | Patient case construction for clinical simulation | PhysioNet credentialed |
| **MIMIC-IV-Note** | Discharge summaries, radiology reports | Clinical text for NLP tasks; simulation context | PhysioNet credentialed |
| **i2b2/n2c2 Clinical NLP** | Annotated clinical text | Gold-standard annotations for concept extraction | DUA required |
| **TCIA reports** | Pathology/radiology reports | Additional text modality | Public |

**Important constraint:** MIMIC DUA prohibits sending data to external closed-source APIs. All models must be **locally deployed** or **open-source** (e.g., LLaMA, Mistral, open clinical models).

### 3.3 Tools and Language

| Component | Tool | Language |
|-----------|------|----------|
| Clinical simulation framework | Custom (adapted from Hager methodology) | Python |
| LLM inference | HuggingFace Transformers (local models) | Python |
| Hallucination detection | Manual annotation protocol + semi-automated NER cross-check | Python |
| Prompt perturbation | Custom perturbation suite (5 types: synonym, reorder, format, casing, abbreviation) | Python |
| Guideline comparison | Rule-based matching + clinical expert review | Python |
| Statistical testing | McNemar's test, paired t-tests (LLM vs physician) | Python / R |
| Experiment tracking | MLflow | Python |
| Reporting | Quarto (HTML/PDF benchmark reports) | Python + R |

### 3.4 Clinical Simulation Design

Adapted from Hager et al., our simulation will:

1. **Present patient cases** to the LLM as structured clinical encounters (demographics + labs + vitals + clinical notes)
2. **Request sequential actions:** diagnostic workup → interpretation → diagnosis → treatment plan
3. **Score outputs** on:
   - **Hallucination rate** — Did the LLM reference nonexistent tools, tests, or facts?
   - **Instruction adherence** — Did the output match the requested format/schema?
   - **Diagnostic accuracy** — Was the diagnosis correct?
   - **Guideline compliance** — Did recommendations align with clinical practice guidelines?
4. **Perturbation testing** — Repeat with systematic prompt modifications to measure stability

### 3.5 Governance Artifacts (WS2/WS3 Integration)

Every benchmark run must produce:

| Artifact | Content | Purpose |
|----------|---------|---------|
| **Safety scorecard** | Hallucination rate, instruction error rate, accuracy per pathology | WS1 deliverable |
| **Perturbation report** | Delta-accuracy for each perturbation type | Robustness assessment |
| **Failure log** | Every hallucinated tool/fact with context | Governance audit trail |
| **Model card** | Model version, config, known limitations | WS3 compliance |
| **Dataset card** | MIMIC subset used, preprocessing steps, DVC hash | Reproducibility |

These feed directly into WS3 DPIAs and regulatory documentation [Tech Plan §5.8].

### 3.6 Safety Measures: Paper-Supported vs PHUSE Extensions

| Measure | Source | Implementation |
|---------|--------|---------------|
| Human-in-the-loop oversight | **Paper-supported** (Hager) | Mandatory clinician review of all LLM clinical outputs |
| Workflow-realistic evaluation | **Paper-supported** (Hager) | Clinical simulation framework (not just exam-style QA) |
| Structured output validation | **Paper-supported** (Hager) | JSON schema validation for all LLM outputs |
| Multi-model ensemble | **PHUSE extension** | Run 2+ models; flag disagreements as uncertainty signals |
| Tiered deployment | **PHUSE extension** | Start benchmarks with low-risk tasks (summarization) before clinical decision support |
| CI/CD safety gates | **PHUSE extension** | Fail pipeline if hallucination rate > threshold (configurable) |
| Continuous monitoring | **PHUSE extension** | MLflow dashboards tracking safety metrics over time |

## 4. Timeline (Aligned to Tech Plan §6)

| When | Milestone |
|------|-----------|
| **Q1 2026** (now) | Review complete; identify clinical collaborators for scenario validation |
| **Q2 2026** | Prototype clinical simulation on structured MIMIC-IV data (simple diagnostic scenario) |
| **Q3 2026** | Full simulation design: scenarios, perturbation suite, scoring rubric; clinical expert review |
| **Q4 2026** | Implementation: run evaluation against 3+ open-source LLMs; produce safety scorecards; compare against physician baselines where feasible; release Clinical Notes Benchmarks v1.0 |

## 5. Anticipated Bottlenecks

| Bottleneck | Mitigation | Priority |
|-----------|-----------|----------|
| **Physician involvement** for scenario design + hallucination adjudication | Recruit clinical collaborators from PHUSE working group early | **Critical** |
| **MIMIC DUA** prevents use of closed-source APIs | Use locally deployed open-source models only | High |
| **Prompt sensitivity** makes reproducibility difficult | Fix all prompts in versioned YAML configs; report results over multiple prompt variants | High |
| **Hallucination annotation** is labor-intensive | Develop semi-automated pipeline: NER cross-check + manual review for ambiguous cases | Medium |
| **Disease-specific results** may not generalize | Start with 2–3 pathologies; expand based on resources | Medium |

## 6. Validation Checklist

- [ ] Minimum 3 clinical pathologies evaluated
- [ ] Minimum 5 prompt perturbation types tested per scenario
- [ ] All hallucinations logged with full context (input, output, expected)
- [ ] Statistical comparison to physician baseline (where feasible) with appropriate tests
- [ ] All prompt templates versioned in repository YAML
- [ ] Safety scorecards generated automatically from MLflow data
- [ ] Pipeline reproducible via `uv sync --frozen` + DVC pull + single command
- [ ] Clinical expert sign-off on scenario design and hallucination adjudication rubric
