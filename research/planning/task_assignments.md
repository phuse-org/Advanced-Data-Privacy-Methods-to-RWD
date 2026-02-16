# Task Assignments — LLM Safety & Privacy (Health Context)

## Paper Reviews

| # | Paper | Assignees | Key Questions |
|---|-------|-----------|---------------|
| 10 | Carlini et al. (USENIX Sec 2021), "Extracting Training Data from LLMs" — shows real training-data leakage; motivates privacy testing for any LLM over RWD. | Onkar, Alex | What were the specific attack methodologies that successfully extracted memorized training data from large language models? What model and data characteristics made such extraction easier, informing our "high-risk" data type profiles? |
| 11 | HELM (Liang et al., 2022) — multi-metric LLM evaluation (accuracy, calibration, robustness, fairness, toxicity, efficiency); useful safety/quality framing. | Onkar, Alex | What is the comprehensive evaluation framework proposed? Which of its evaluation dimensions (e.g., accuracy, robustness, fairness, toxicity) are most critical and transferable for assessing the safety of LLMs used in clinical or health data analysis contexts? |
| 12 | Hager et al. (Nature Medicine 2024), "Hallucinations and clinical risk" — why hallucinations matter at the bedside; safety considerations. | Onkar, Alex | What are the concrete clinical risks and potential harms identified from LLM hallucinations or biases in healthcare? What high-level governance and safety measures (e.g., human-in-the-loop, strict validation) are recommended for deploying AI models on sensitive health data? |

## Cross-Cutting Review Questions

In addition to the article-specific questions, reviewers should consider:

- Which tool(s) are most appropriate, and what type of data would be suitable for them?
- Which programming language is best suited to implement the method?
- What bottlenecks/assumptions do you anticipate when applying the method?
- What validation or diagnostic checks are recommended to assess the robustness of the results?
- What expertise level or training is required to implement the methods?
- How well do the methods generalize across the different data sources?
