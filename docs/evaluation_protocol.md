# Evaluation Protocol and Metrics

[cite_start]This document defines the quantitative and qualitative metrics used to compare Method A (Long-context), Method B (RAG), and Method C (Iterative Reading)[cite: 121, 129].

## 1. Primary Metrics

### A. Answer Accuracy ($Acc$)
[cite_start]Answers are scored against the ground truth in `seed_v0.json` using a rubric-based scale (0.0 to 1.0) or exact match for factual queries[cite: 129].

### B. Citation Correctness ($C_{score}$)
A citation is considered correct if it identifies the correct file and a valid line range.
$$C_{score} = \frac{1}{N} \sum_{i=1}^{N} (\alpha \cdot \text{Match}_{\text{file}} + \beta \cdot \text{Overlap}_{\text{lines}})$$
[cite_start]*Where $\alpha=0.5, \beta=0.5$, and $\text{Overlap}_{\text{lines}}$ measures the intersection of the predicted and ground truth line ranges*[cite: 129].

### C. Performance Metrics
- [cite_start]**Cost:** Total input/output tokens and number of model calls per query[cite: 31, 75, 129].
- [cite_start]**Latency:** End-to-end time (seconds) per query[cite: 31, 45, 129].

## 2. Context-Rot Protocol
To quantify performance degradation as corpus size increases:
1. **Baseline:** Run QA on the minimal relevant context.
2. [cite_start]**Stress Test:** Scale the corpus by adding irrelevant files (distractors) up to 1M tokens[cite: 120, 129].
3. [cite_start]**Analysis:** Plot $Acc$ and $C_{score}$ vs. Corpus Size to identify the "breaking point" for each method[cite: 38, 76, 129].

## 3. Failure Case Analysis
Each failure will be categorized into:
- **Retrieval Miss:** Correct evidence was not found (primarily for Method B).
- **Reasoning Error:** Evidence found, but the model failed to synthesize the answer.
- [cite_start]**Citation Hallucination:** Answer is correct, but the cited file/line is wrong[cite: 77, 129].
