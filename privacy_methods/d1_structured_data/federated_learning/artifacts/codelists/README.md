# Data Type 1 — Structured Data: Federated Learning

## Purpose (plain language)
Cross-silo federated learning for tabular ICU prediction tasks.

## What this benchmark will produce
- A small set of metrics (saved locally under `results/`)
- A short README documenting assumptions and parameters

## Datasets (recommended starting points)
See `datasets/datasets.yaml` and dataset cards in `datasets/`.

## Methods in scope (initial)
- **Cross-silo federated learning:** Flower framework; FedAvg and FedProx aggregation strategies
- **DP-FL:** Federated learning with per-client DP-SGD clipping and noise (Opacus + Flower)
- **Baseline:** Centralized (pooled) training on combined data (upper bound on utility)

## Benchmark tasks
- Partition eICU-CRD by hospital site to simulate cross-silo federation
- Train ICU mortality / readmission prediction models under FedAvg vs centralized pooling
- Evaluate convergence speed, communication rounds, and per-site performance heterogeneity

## Metrics (v0)
- Utility metrics: AUROC, F1, calibration error (global model and per-site)
- Convergence metrics: rounds to target AUROC, communication cost
- Privacy metrics: DP guarantee (epsilon per client), membership inference attack success rate
- Fairness metrics (optional): per-site performance variance, demographic subgroup gaps

## How to run
- `python run.py --help`
