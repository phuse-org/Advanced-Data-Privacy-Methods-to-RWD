# PHUSE: Applying Advanced Data Privacy Methods to Real-World Data (RWD)

This repository is a **shared workspace** for the PHUSE project team to collaborate on:
- **Benchmark specifications** (what we evaluate)
- **Reference implementations** (how we run the benchmarks)
- **Documentation** that helps stakeholders review scope, structure, and progress

**What this repo is not:**
- A place to store raw patient-level datasets. We only keep **dataset metadata** in GitHub.

## Quick links

- Dataset registry (33 datasets): [`datasets/README.md`](datasets/README.md)
- Privacy methods: [`privacy_methods/`](privacy_methods/)
- Research & paper reviews: [`research/`](research/)
- Documentation by data type: [`docs/`](docs/)
- Next steps & roadmap: [`next_steps.md`](next_steps.md)
- Contributing guide: [`CONTRIBUTING.md`](CONTRIBUTING.md)

## Repo structure

```
phuseprivacy/
├── privacy_methods/                 # Privacy method implementations
│   ├── d1_structured_data/          # Structured (tabular) clinical data
│   │   ├── baseline_deidentification/
│   │   ├── differential_privacy/
│   │   ├── federated_learning/
│   │   └── synthetic_data/
│   └── d2_unstructured_data/        # Clinical notes / unstructured text
│       ├── deidentification/
│       ├── llm_privacy_controls/
│       └── privacy_attacks/
│
├── datasets/                        # Dataset registry — metadata only (33 datasets)
│   ├── README.md                    # Full reference with links & access info
│   ├── datasets.yaml                # Machine-readable registry
│   ├── 01_mimic-iv_icu_ehr/
│   ├── 02_mimic-iv-note_clinical_notes/
│   ├── ...
│   └── 33_openmhealth_wearables/
│
├── docs/                            # Documentation (mirrors privacy_methods/)
│   ├── d1_structured_data/
│   │   ├── baseline_deidentification/
│   │   ├── differential_privacy/
│   │   ├── federated_learning/
│   │   └── synthetic_data/
│   └── d2_unstructured_data/
│       ├── deidentification/
│       ├── llm_privacy_controls/
│       └── privacy_attacks/
│
├── research/                        # Paper reviews & implementation reports
│   ├── d1_structured_data/          # Papers on structured data methods
│   ├── d2_unstructured_data/        # Papers on unstructured text methods
│   └── planning/                    # Tech Implementation Plan & strategy
│
├── experiments/                     # Local experiment outputs (ignored by git)
├── data/                            # Local data directory (ignored by git)
└── results/                         # Local outputs directory (ignored by git)
```

### Folder conventions

All three core folders — `privacy_methods/`, `docs/`, and `research/` — share the same D1/D2 hierarchy:

| Data Type | Folder | Methods |
|-----------|--------|---------|
| **D1** Structured (tabular) | `d1_structured_data/` | baseline_deidentification, differential_privacy, federated_learning, synthetic_data |
| **D2** Unstructured (text) | `d2_unstructured_data/` | deidentification, llm_privacy_controls, privacy_attacks |

## Contributing

See [`CONTRIBUTING.md`](CONTRIBUTING.md) for the full guide. In short:
1. Work in a **group branch**
2. Open a **Pull Request** into `main`
3. A maintainer reviews and merges

## License

See [`LICENSE`](LICENSE).
