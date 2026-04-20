# Metric Card — Repository QA Evaluation

**Project:** Evaluating Long-Context RAG and Recursive Reading for LLM Repository QA  
**Frozen:** 2026-03-21  
**Config Reference:** `configs/eval_config.json`

---

## 1. Answer Accuracy — 3 Named Fields

> **IMPORTANT:** There is no single `accuracy_score` field. Three explicit fields are stored, each with a clear definition. Never collapse them into one.

### Field: `accuracy_em` — Normalized Exact Match (Tier 1)
- **Applies when:** Answer is short, discrete, and factual (e.g., return type, class name, IAM action name)
- **Method:** `normalize(predicted) == normalize(ground_truth)`
- **Normalization rules:**
  1. Lowercase all text
  2. Strip punctuation: `. , : ; ' " ( ) [ ] { } !`
  3. Collapse consecutive whitespace to single space, strip leading/trailing
  4. **Do NOT normalize:** backtick-wrapped code tokens, file paths, class/method names (case-sensitive identifiers)
  5. File path normalization: convert `\` → `/`, match basename if full paths differ only in prefix

### Field: `accuracy_human` — Human Rubric Score (Tier 2)
- **Applies to:** 20–24 sample manual audit subset (balanced across method/dataset/reasoning_type/difficulty)
- **Method:** Human reviewer assigns:
  - `1.0` — Answer is factually correct and complete
  - `0.5` — Answer is partially correct, has right direction but missing key detail
  - `0.0` — Answer is wrong, hallucinated, or does not address the question
- **Purpose:** Validates `accuracy_llmjudge` consistency; cited in Methodology chapter

### Field: `accuracy_llmjudge` — LLM-as-Judge Semantic Score (Tier 3)
- **Applies to:** All samples, especially long/paraphrase answers
- **Method:** LLM judge (same `qwen2.5:7b`, `temperature=0.0`, `judge_v1` prompt) rates 0.0–1.0
- **Guardrails:** Must be validated against `accuracy_human` on the audited subset. Correlation must be documented.
- **Prompt version in config:** `judge_prompt_version: "judge_v1"`

### Field: `accuracy_primary`
- Selection rule:
  - Use `accuracy_em` if the question is `single_hop_lookup` with short expected answer (≤ 10 words in ground truth)
  - Otherwise: use `accuracy_llmjudge` on full set, `accuracy_human` on audited subset
- **Main result table always states which field is used**

---

## 2. Citation Metrics — File, Coverage, and Span Tightness

> **IMPORTANT:** Report citation dimensions independently. Do not collapse file correctness, evidence coverage, span tightness, and LLM support into one score.

### Layer 1 — File-Level (Citation Hit@File)

| Field | Definition |
|---|---|
| `citation_file_precision` | `|predicted_files ∩ gt_files| / |predicted_files|` |
| `citation_file_recall` | `|predicted_files ∩ gt_files| / |gt_files|` |
| `citation_file_f1` | `2 × precision × recall / (precision + recall)` |

**File matching rules:**
- Normalize path separators (`\` → `/`)
- Basename match accepted if: `basename(pred) == basename(gt)` AND one path is a suffix of the other
- Case-insensitive for `.md`, `.yml`, `.yaml` files; case-sensitive for `.py`, `.js`, `.go`

### Layer 2 — Evidence Coverage and Span Tightness

| Field | Definition |
|---|---|
| `citation_containment_recall` | Fraction of each gold span contained inside predicted spans on the same file: `overlap_lines / gold_lines`. Rewards retrieving a chunk that fully contains the gold evidence. |
| `citation_span_precision` | Fraction of predicted citation lines that overlap gold spans on the same file: `overlap_lines / predicted_lines`. Penalizes overly broad chunks. |
| `citation_span_f2` | F-beta over span precision/containment recall with `beta=2`, prioritizing coverage over tightness. |
| `citation_weighted_span_score` | `0.7 * citation_containment_recall + 0.3 * citation_span_precision`. |
| `evidence_line_iou` | Legacy localization: `intersection_lines / union_lines`. Kept for continuity, but no longer the primary span metric. |
| `citation_support_score` | Legacy `file_match × line_iou` for each GT citation, averaged. |

**Scoring evidence basis:**
- Raw method output is preserved as `predicted_evidence`.
- The scorer applies a deterministic narrowing post-process to create `scored_predicted_evidence`.
- Span metrics and citation-support judge use `scored_predicted_evidence`.
- Narrowing uses only prediction-side information: question, predicted answer, cited file text, and cited chunk lines. It does not use gold evidence.

**Line range intersection:**
- `intersection = max(0, min(pred_end, gt_end) - max(pred_start, gt_start) + 1)`
- `union = max(pred_end, gt_end) - min(pred_start, gt_start) + 1`
- `iou = intersection / union`

### Gold Evidence Policy
- Each QA entry has **one authoritative evidence set** (minimal sufficient evidence to answer the question)
- This is NOT exhaustive — it cites the lines needed, not all related code
- A prediction citing a **superset** (more lines/files than GT) is NOT penalized on file-level metrics
- Partial overlap (right file, broad/wrong lines) earns separate coverage and span-tightness scores. `evidence_line_iou` is retained only as a legacy diagnostic.

---

## 3. Failure Handling — Fine-grained Rules

| `error_type` | `accuracy_llmjudge` | `accuracy_em` | All `citation_*` fields |
|---|---|---|---|
| `timeout` | 0.0 | 0.0 | 0.0 |
| `malformed_json_recovered` | Score normally from recovered text | N/A | 0.0 |
| `malformed_json_unrecovered` | 0.0 | 0.0 | 0.0 |
| `empty_answer` | 0.0 | 0.0 | 0.0 |
| `no_citation` | **Score normally** (answer may still be correct) | Score normally | 0.0 for ALL citation fields |
| `tool_error` | 0.0 | 0.0 | 0.0 |

> **NOTE:** `no_citation` does NOT zero out accuracy. Grading answer correctness is independent of whether citations were provided.

---

## 4. Latency

Report all six statistics. Never only report mean.

| Statistic | Why |
|---|---|
| `mean` | Central tendency |
| `std` | Spread |
| `median` | Robust central tendency (less sensitive to outliers) |
| `p75` | Upper quartile |
| `p95` | Near-worst-case (important for RLM with occasional long chains) |
| `max` | True worst case |

---

## 5. Cost

| Field | Description |
|---|---|
| `input_tokens` | LLM prompt tokens (all calls combined for Method C) |
| `output_tokens` | LLM completion tokens (all calls combined) |
| `total_tokens` | `input_tokens + output_tokens` |
| `num_calls` | Number of LLM API calls (always 1 for A and B; 5–11 for C) |

---

## 6. Expected Reasoning Type Taxonomy

| Tag | Meaning | Typical Evidence Files |
|---|---|---|
| `single_hop_lookup` | One file, one discrete fact | 1 file |
| `multi_hop_same_source` | Multiple locations within same file | 1 file, multiple ranges |
| `multi_hop_cross_source` | Crosses 2+ files in same repo | 2+ files, same repo |
| `cross_file_code_reasoning` | Must trace call chain across files | 3+ files, same repo |
| `doc_repo_hybrid` | Answer requires code + docs | Files from both repos |
| `aggregation_or_compare` | Compares things (two patterns, two repos) | 2+ files, possibly cross-repo |

---

## 7. Error Taxonomy (for `src/analyze.py`)

| Category | Definition |
|---|---|
| `retrieval_miss` | GT evidence in corpus but not retrieved at all |
| `insufficient_evidence` | Partially retrieved — right direction but missing pieces for multi-hop |
| `reasoning_failure` | Evidence retrieved correctly; answer wrong |
| `citation_failure` | Answer correct; citations wrong or missing |
| `over_generation` | Answer contains unsupported claims |
| `trace_inefficiency` | (Method C) Too many calls for question complexity |
