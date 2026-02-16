# Data Type 1 — Structured Data: Baseline De-identification

## Purpose (plain language)
Baseline de-identification for structured tabular clinical data. Mostly structured scrubbing (low risk relative to text). Ensures direct identifiers and quasi-identifiers are handled before applying other privacy methods.

## What this privacy method will produce
- De-identified structured datasets (stored locally, never committed)
- Summaries of identifiers removed / scrubbed
- All outputs saved locally under `results/`

## Datasets (recommended starting points)
See `datasets/datasets.yaml` and dataset cards in `datasets/`.

## Methods in scope (initial)
- Structured scrubbing of direct identifiers (pandas / custom scripts)
- Quasi-identifier assessment and suppression / generalization (ARX Java library or sdcMicro R package)
- K-anonymity / l-diversity checks (optional; ARX or custom implementation)

## Benchmark tasks
- Assess re-identification risk before and after de-identification
- Validate that downstream utility is preserved after scrubbing

## Metrics (v0)
- Re-identification risk metrics (e.g., uniqueness, k-anonymity)
- Utility metrics (e.g., AUROC, F1 on downstream tasks before vs after)
- Basic fairness checks (optional)

## How to run
- `python run.py --help`
