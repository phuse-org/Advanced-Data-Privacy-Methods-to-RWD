# D1 Structured Data — Documentation

Guides and reference documents for structured (tabular) clinical data privacy methods.

## Methods

| Method | Folder | Summary |
|--------|--------|---------|
| Baseline De-identification | `baseline_deidentification/` | Direct/quasi-identifier removal, k-anonymity checks |
| Differential Privacy | `differential_privacy/` | DP analytics (OpenDP/SmartNoise) and DP-SGD models (Opacus) |
| Federated Learning | `federated_learning/` | Cross-silo training via Flower (FedAvg, FedProx) |
| Synthetic Data | `synthetic_data/` | CTGAN/TVAE generation with train-synthetic/test-real evaluation |

## Related resources

- Privacy method code and runners: `privacy_methods/d1_structured_data/`
- Research and paper reviews: `research/d1_structured_data/`
- Privacy-analysis method mapping: `analysis_methods/privacy_analysis_matrix.md`
- Dataset registry: `datasets/README.md`
