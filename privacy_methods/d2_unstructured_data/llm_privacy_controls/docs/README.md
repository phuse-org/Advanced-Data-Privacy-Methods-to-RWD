# Data Type 2 — Unstructured Data: LLM Privacy Controls

## Purpose (plain language)
LLM privacy controls: RAG over safe summaries; DP synthetic text; optional DP fine-tuning.

## What this benchmark will produce
- A small set of metrics (saved locally under `results/`)
- A short README documenting assumptions and parameters

## Datasets (recommended starting points)
See `datasets/datasets.yaml` and dataset cards in `datasets/`.

## Methods in scope (initial)
- **RAG over de-identified summaries:** Retrieval-augmented generation using pre-scrubbed clinical text (no raw PHI in context)
- **DP fine-tuning:** DP-SGD fine-tuning of LLMs on clinical text (Opacus / dp-transformers)
- **Safety evaluation:** HELM / MedHELM multi-metric evaluation of clinical LLM outputs (hallucination, toxicity, bias)
- **Baseline:** Standard (non-private) LLM inference on raw clinical text

## Benchmark tasks
- Evaluate clinical question-answering accuracy with RAG over de-identified vs original notes
- Measure memorization and extraction risk in fine-tuned clinical LLMs (Carlini-style probing)
- Run HELM/MedHELM safety evaluation suite on clinical LLM outputs (hallucination rate, instruction fragility)

## Metrics (v0)
- Utility metrics: QA accuracy (F1, exact match), clinical summarization ROUGE scores
- Privacy metrics: memorization rate (extractable sequences per N samples), membership inference AUROC
- Safety metrics: hallucination rate, toxicity score, instruction-following fragility (IFEval-style; Zhou et al. 2023)
- Basic fairness checks (optional): performance disparities across patient demographics

## How to run
- `python run.py --help`
