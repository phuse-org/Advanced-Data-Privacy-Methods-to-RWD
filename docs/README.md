# Documentation

Organized by data type, mirroring the `privacy_methods/` folder structure.

## Structure

| Data Type | Folder | Privacy Methods Covered |
|-----------|--------|------------------------|
| D1 — Structured (tabular) clinical data | `d1_structured_data/` | Baseline de-identification, differential privacy, federated learning, synthetic data |
| D2 — Unstructured clinical text | `d2_unstructured_data/` | De-identification, LLM privacy controls, privacy attacks |

## What belongs here

Each method subfolder should contain:

- **Method specification** — What the method does, which datasets it targets, and how it relates to the privacy-analysis matrix
- **Benchmark design** — Task definitions, input/output contracts, evaluation criteria
- **Configuration reference** — Parameter descriptions for the corresponding `run.py` runner
- **Results interpretation guide** — How to read the `metrics.json` output and what thresholds matter

## Current status

Documentation is under active development. See `next_steps.md` for the roadmap. Contributions that add method specifications and benchmark design docs are welcome — see `CONTRIBUTING.md`.
