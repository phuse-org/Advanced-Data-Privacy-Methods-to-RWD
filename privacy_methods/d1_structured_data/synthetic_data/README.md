# Data Type 1 — Structured Data: Synthetic Data

## Purpose (plain language)
Non-DP and DP synthetic tabular generation and evaluation (train synth -> test real).

## What this benchmark will produce
- A small set of metrics (saved locally under `results/`)
- A short README documenting assumptions and parameters

## Datasets (recommended starting points)
See `datasets/datasets.yaml` and dataset cards in `datasets/`.

## Methods in scope (initial)
- **Non-DP synthetic generation:** SDV library (CTGAN, TVAE, Gaussian Copula) trained on real tabular data
- **DP synthetic generation:** DPGAN / DP-CTGAN with formal (epsilon, delta) guarantees
- **Baseline:** Train-on-real / test-on-real (upper bound on utility)

## Benchmark tasks
- Generate synthetic replicas of ICU demographics/vitals tables
- Train downstream ML models (mortality, readmission) on synthetic data, evaluate on held-out real data (train-synthetic / test-real)
- Compare distributional fidelity (marginal and pairwise correlations, column-shape scores)

## Metrics (v0)
- Utility metrics: AUROC, F1, calibration error (downstream ML on synthetic vs real)
- Fidelity metrics: column-shape score, column-pair-trend score (SDMetrics)
- Privacy metrics: nearest-neighbour distance ratio, membership inference attack success rate
- Basic fairness checks (optional): subgroup utility gaps

## How to run
- `python run.py --help`
