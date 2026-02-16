# Contributing

Thank you for contributing. This is a volunteer project, so we keep the process lightweight.

## Where to contribute

| What you're adding | Where to put it |
|--------------------|-----------------|
| Privacy method code | `privacy_methods/d1_structured_data/<method>/` or `d2_unstructured_data/<method>/` |
| Method documentation | `docs/d1_structured_data/<method>/` or `d2_unstructured_data/<method>/` |
| Paper reviews / research | `research/d1_structured_data/<method>/` or `d2_unstructured_data/<method>/` |
| Planning docs | `research/planning/` |
| Dataset metadata | `datasets/<dataset_folder>/` |

## How to contribute

1. Create or switch to your **group branch**
2. Make your changes
3. Open a **Pull Request** into `main`
4. A maintainer reviews and merges

## Folder conventions

All three core folders — `privacy_methods/`, `docs/`, and `research/` — use the same D1/D2 hierarchy:

- `d1_structured_data/` — baseline_deidentification, differential_privacy, federated_learning, synthetic_data
- `d2_unstructured_data/` — deidentification, llm_privacy_controls, privacy_attacks

## Minimum requirements for new method folders

- A `README.md` with concrete content: purpose, specific methods in scope, defined benchmark tasks (inputs/outputs/evaluation), and named metrics — not template prompts
- A runner script (`run.py` or `run.R`) that writes output to `results/` (local only); the scaffold already provides CLI structure, so new contributions should add real benchmark logic

## Moving beyond scaffolds

The initial repository structure uses scaffold runners and template READMEs to bootstrap the folder layout. Contributions that replace scaffold content with working implementations are the highest priority. See `next_steps.md` for the current roadmap and ownership assignments.

## Data safety

- **Never** commit raw patient-level data
- **Never** commit credentials or secrets
- Store local data in `data/` (git-ignored)
- Store experiment outputs in `results/` or `experiments/` (git-ignored)
