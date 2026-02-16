# Data Type 2 — Unstructured Data: De-identification

## Purpose (plain language)
PHI de-identification: NER-based PHI detection and masking; optional LLM-assisted workflows.

## What this benchmark will produce
- A small set of metrics (saved locally under `results/`)
- A short README documenting assumptions and parameters

## Datasets (recommended starting points)
See `datasets/datasets.yaml` and dataset cards in `datasets/`.

## Methods in scope (initial)
- **Rule-based de-identification:** Regular-expression and dictionary-based PHI scrubbing (Philter)
- **NER-based de-identification:** Transformer NER models (e.g., BERT-NER, Presidio) for PHI span detection and masking
- **Baseline:** No de-identification (raw clinical text) — measures upper-bound utility and lower-bound privacy

## Benchmark tasks
- Detect and mask 18 HIPAA PHI categories in clinical notes (i2b2/n2c2 2014 de-identification track)
- Evaluate PHI recall (missed leaks) and precision (unnecessary redactions)
- Measure downstream NLP task performance on de-identified vs original text (e.g., NER, relation extraction)

## Metrics (v0)
- De-identification metrics: token-level precision, recall, F1 for PHI detection
- Privacy metrics: residual PHI leak rate (false negatives per note)
- Utility metrics: downstream NLP task F1 before vs after de-identification
- Basic fairness checks (optional): PHI detection performance across demographic groups

## How to run
- `python run.py --help`
