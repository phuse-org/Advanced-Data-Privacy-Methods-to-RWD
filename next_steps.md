# PHUSE Privacy Project — Next Steps

**Last updated:** 15 February 2026

---

## Current Status Summary

### What's Complete

| Deliverable | Status | Location |
|-------------|--------|----------|
| Repo structure — D1/D2 hierarchy across privacy_methods, docs, research | **Done** | Root `README.md` |
| Dataset registry (33 datasets with full metadata, links, access info) | **Done** | `datasets/README.md` |
| Paper reviews & implementation reports | **Done** | `research/planning/` |

### What's Not Started Yet

| Deliverable | Priority | Blocked By |
|-------------|----------|------------|
| Actual benchmark code (replace scaffold `run.py` with real logic) | High | Dataset access |
| WS2 infrastructure setup (uv, DVC, MLflow, CI/CD) | High | Repo decision |
| Data Type 2 (Clinical Text) guide — sources, access, methods | Medium | Follows Data Type 1 completion |
| E2E "hello benchmark" test run | High | Dataset access + infrastructure |
| PhysioNet credentialing (team-wide) | **Critical path** | Individual action required |

---

## Recommended Next Steps (Ordered by Priority)

### IMMEDIATE — This Week (Feb 14–21, 2026)

#### 1. Start PhysioNet Credentialing
- **What:** Every team member who will touch MIMIC-IV, eICU, or HiRID must complete:
  1. PhysioNet account creation
  2. CITI training (Human Subjects Research)
  3. Credentialing application
  4. Individual DUA signing
- **Why:** This is the single biggest bottleneck. Credentialing can take 1–4 weeks.
- **Action owner:** Each team member individually
- **Note:** PhysioNet DUA prohibits sharing data with third-party APIs/online platforms

#### 2. Circulate Deliverables for Stakeholder Review
- **What:** Share the dataset registry, repo structure, and research with WS1/WS2 leads
- **Why:** Get alignment and feedback before building real pipelines
- **Deliverables to share:**
  - `datasets/README.md` (full dataset registry with 33 datasets)
  - `research/planning/` (paper reviews, implementation reports)
  - Repo structure overview (`README.md`)
- **Action owner:** Onkar / Alex

---

### SHORT-TERM — Next 2–4 Weeks (Feb 21 – Mar 14, 2026)

#### 3. Set Up WS2 Infrastructure

| Component | Tool | What to Do |
|-----------|------|-----------|
| Python env | `uv` | Add `pyproject.toml` + `uv.lock`; pin Python 3.11+ |
| Data versioning | DVC | Initialize DVC; add `.dvc/` config; create remote storage pointer |
| Experiment tracking | MLflow | Add MLflow config; set up local or shared tracking server |
| CI/CD | GitHub Actions | Add workflow: lint, test, environment lock check |
| Branch protection | GitHub settings | Protect `main`; require PR + 1 approval; enable CODEOWNERS |

#### 4. Build the "Hello Benchmark" (E2E Pipeline Test)
- **What:** Implement ONE real benchmark end-to-end to validate the entire stack
- **Recommended:** DP analytics on MIMIC-IV (simplest first win)
  - Load MIMIC-IV demographics/vitals tables
  - Compute DP vs non-DP summary statistics using OpenDP/SmartNoise
  - Output: `params.json` (epsilon, delta, clipping), `metrics.json` (utility error), MLflow artifact
- **Alternative (if MIMIC access pending):** Use a public synthetic dataset (e.g., Synthea) as a stand-in
- **Target folder:** `privacy_methods/d1_structured_data/differential_privacy/`

#### 5. Assign Benchmark Ownership
- **What:** Assign a subgroup lead to each method folder
- **Why:** Parallel development requires clear ownership; CODEOWNERS auto-review depends on this

| Method Folder | Suggested Owner Profile |
|---------------|------------------------|
| `d1_structured_data/differential_privacy/` | Privacy/stats lead |
| `d1_structured_data/synthetic_data/` | ML engineer |
| `d1_structured_data/federated_learning/` | ML/systems engineer |
| `d1_structured_data/baseline_deidentification/` | Data engineer |
| `d2_unstructured_data/deidentification/` | NLP engineer + clinical informatician |
| `d2_unstructured_data/privacy_attacks/` | Privacy/security researcher |
| `d2_unstructured_data/llm_privacy_controls/` | ML engineer + privacy lead |

---

### MID-TERM — Rest of Q1 2026 (Mar 14 – Mar 31, 2026)

#### 6. Write Data Type 2 (Clinical Text) Guide
- **What:** Mirror the Data Type 1 guide for clinical text data:
  - Ranked text data sources (MIMIC-IV-Note, i2b2/n2c2, MIMIC-CXR reports, TCIA reports)
  - Step-by-step access instructions for each
  - Ranked privacy methods (de-identification, LLM privacy controls, privacy attacks, DP text)
  - Method-to-benchmark mapping for text tasks
- **Target folder:** `docs/d2_unstructured_data/`
- **Why:** Text is the highest-risk data type; Q3 2026 implementation needs design now

#### 7. Implement First Real Tabular Benchmarks
Once MIMIC-IV access is secured, replace scaffold `run.py` scripts with real code:

| Priority | Target Folder | What to Implement |
|----------|---------------|-------------------|
| 1st | `d1_structured_data/differential_privacy/` | OpenDP/SmartNoise: DP counts, means, histograms vs non-DP |
| 2nd | `d1_structured_data/synthetic_data/` | SDV CTGAN/TVAE: synthetic data generation + ML fidelity |
| 3rd | `d1_structured_data/federated_learning/` | Flower: cross-silo FL on eICU hospital splits |

#### 8. Draft WS1 Benchmark Specification Document
- **What:** A formal specification listing every benchmark task, its inputs/outputs, metrics, datasets, and pass/fail criteria
- **Why:** This becomes the contract between WS1 (what to evaluate) and WS2 (how to build it)

---

### Q2 2026 (Apr – Jun 2026)

#### 9. Full Structured Data Implementation + Validation
- Run all tabular benchmarks on MIMIC-IV + eICU
- Generate privacy-utility curves (AUC vs epsilon)
- Cross-site robustness testing (train MIMIC-IV -> test eICU)
- Fairness subgroup analysis (where demographic data available)
- Produce governance artifacts: model cards, dataset cards, MLflow logs
- **Release: WS1-WS2 Structured Benchmarks v1.0**

#### 10. Begin Clinical Text Research + Design (Q3 prep)
- Deep-dive into clinical text risk profiles (Carlini attacks on health text)
- Evaluate de-ID tools (Presidio, Philter, BERT-NER) on i2b2/n2c2
- Design Hager-style clinical simulation scenarios
- Recruit physician collaborators for hallucination adjudication

---

## Critical Path Items (Risks to Monitor)

| Risk | Impact | Mitigation | Owner |
|------|--------|------------|-------|
| PhysioNet credentialing delays | Blocks all real benchmarks | Start NOW; use Synthea as interim stand-in | Each team member |
| Benchmark ownership unassigned | Parallel development stalls | Assign leads by end of Feb | WS1/WS2 leads |
| GPU/compute access for LLM evaluation | Blocks text benchmarks (Q3/Q4) | Identify cloud/HPC resources by Q2 | WS2 leads |
| No physician collaborator identified | Blocks Hager-style simulation design | Recruit from PHUSE working group by Q2 | WS1 leads |

---

## Decision Log (For WS1/WS2 Leads)

| # | Decision Needed | Options | Recommended | By When |
|---|----------------|---------|-------------|---------|
| 1 | Primary dataset for first benchmark | (a) MIMIC-IV (b) Synthea stand-in | (a) if access ready; (b) if not | Feb 28 |
| 2 | MLflow hosting | (a) Local only (b) Shared server (c) Databricks community | (a) for now; migrate later | Mar 7 |
| 3 | Who owns which benchmark? | See ownership table above | Assign based on expertise | Feb 28 |
| 4 | Cloud/HPC for LLM evaluation | (a) AWS/GCP credits (b) Institutional HPC (c) Volunteer GPUs | Depends on team resources | Mar 31 |

---

*This document should be reviewed weekly and updated as decisions are made and milestones are hit.*
