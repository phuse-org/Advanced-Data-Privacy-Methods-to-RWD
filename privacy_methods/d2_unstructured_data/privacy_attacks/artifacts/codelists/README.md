# Data Type 2 — Unstructured Data: Privacy Attacks

## Purpose (plain language)
Privacy attacks/evaluations: membership inference, extraction, dataset inference tests.

## What this benchmark will produce
- A small set of metrics (saved locally under `results/`)
- A short README documenting assumptions and parameters

## Datasets (recommended starting points)
See `datasets/datasets.yaml` and dataset cards in `datasets/`.

## Methods in scope (initial)
- **Training data extraction:** Carlini-style attacks that probe LLMs for memorized sequences (perplexity ranking, Zlib entropy, casing heuristics)
- **Membership inference attacks (MIA):** Determine whether a specific record was in the training set (shadow-model and loss-threshold approaches)
- **Dataset inference attacks:** Determine whether a specific *dataset* was used during training

## Benchmark tasks
- Run extraction attacks on fine-tuned clinical LLMs; count extractable memorized sequences per N generated samples
- Run MIA against tabular and text models; measure attack AUROC and true-positive rate at low false-positive rates
- Evaluate attack effectiveness before and after applying each privacy method (DP, synthetic data, de-identification)

## Metrics (v0)
- Extraction metrics: number of memorized sequences extracted per N samples, extraction precision at top-K
- MIA metrics: attack AUROC, TPR @ 1% FPR, TPR @ 0.1% FPR
- Dataset inference metrics: attack accuracy, confidence calibration
- Comparison metrics: attack success reduction after privacy method is applied (delta)

## How to run
- `python run.py --help`
