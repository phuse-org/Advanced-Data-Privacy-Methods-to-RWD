# D2 Unstructured Data — Documentation

Guides and reference documents for clinical notes and unstructured text privacy methods.

## Methods

| Method | Folder | Summary |
|--------|--------|---------|
| De-identification | `deidentification/` | NER-based PHI detection and masking (Presidio, Philter, BERT-NER) |
| LLM Privacy Controls | `llm_privacy_controls/` | RAG over de-identified text, DP fine-tuning, HELM/MedHELM safety evaluation |
| Privacy Attacks | `privacy_attacks/` | Membership inference, training data extraction (Carlini-style), dataset inference |

## Related resources

- Privacy method code and runners: `privacy_methods/d2_unstructured_data/`
- Research and paper reviews: `research/d2_unstructured_data/`
- Privacy-analysis method mapping: `analysis_methods/privacy_analysis_matrix.md`
- Dataset registry: `datasets/README.md`
