# Repository QA Evaluation

Research code for comparing three repository-level question answering methods on
software source code and technical documentation:

- `Method A`: long-context prompting
- `Method B`: standard RAG
- `Method C`: structure-aware iterative reader

The project evaluates not only answer correctness, but also citation grounding,
latency, and robustness under context-rot noise.

## Repository Layout

```text
configs/         Evaluation configuration
docs/            Dataset, metric, and evaluation notes
qa_dataset/      QA benchmarks and subset ID files
src/             Methods, scoring, analysis, and plotting scripts
```

Large local artifacts are intentionally excluded from git:

- `dataset/` repository corpora
- `chroma_db/` vector store data
- `results/` and `results_context_rot/` experiment outputs
- `docs_index.json` generated chunk index
- `archive/` local thesis assets

## Methods

### Method A: Long-Context

Implemented in [src/method_a_longcontext.py](src/method_a_longcontext.py).
This baseline packs a large bounded context into a single prompt and produces
one answer in one model call.

### Method B: Standard RAG

Implemented in [src/method_b_rag.py](src/method_b_rag.py). It retrieves top-k
chunks from the vector store, reranks them, and answers from the selected
evidence in a single synthesis step.

### Method C: Structure-Aware Iterative Reader

Implemented in [src/method_c_iterative.py](src/method_c_iterative.py). This is
a bounded iterative reader and it combines:

- initial retrieval and reranking
- quick chunk role classification (`direct`, `bridge`, `noise`)
- evidence-linked verified facts
- follow-up retrieval and deterministic repository tools
- final synthesis from verified facts plus raw evidence

## Benchmark Data

The main dissertation benchmark is:

- [qa_dataset/seed_v3_test.json](qa_dataset/seed_v3_test.json)

Each QA item includes:

- question metadata (`dataset`, `difficulty`, `reasoning_type`)
- ground-truth answer
- gold evidence as file paths and line spans
- grading notes

Supporting notes:

- [docs/dataset_selection.md](docs/dataset_selection.md)
- [docs/evaluation_protocol.md](docs/evaluation_protocol.md)
- [docs/metric_card.md](docs/metric_card.md)

## Setup

### 1. Environment

```powershell
python -m venv venv
.\venv\Scripts\Activate.ps1
pip install -r requirements.txt
```

### 2. Configure model access

Set environment variables in `.env` as needed by
[src/config.py](src/config.py). The codebase supports local-model execution via
the OpenAI-compatible client configuration defined there.

### 3. Build the chunk index and vector store

```powershell
python src\ingest.py
python src\vector_store.py --build
```

This produces:

- `docs_index.json` for chunk metadata
- `chroma_db/` for embeddings and retrieval

## Standard Evaluation Pipeline

### Run inference

Run one method:

```powershell
python src\evaluate.py --qa-file qa_dataset\seed_v3_test.json --method C --output-dir results\seed_v3_final
```

Run all three methods:

```powershell
python src\evaluate.py --qa-file qa_dataset\seed_v3_test.json --method all --output-dir results\seed_v3_final
```

Outputs are written as `predictions_*.jsonl`.

### Score predictions

```powershell
python src\score.py --predictions results\seed_v3_final --ground-truth qa_dataset\seed_v3_test.json --output results\seed_v3_scored_final_rawbert --with-citation-judge
```

The scorer computes:

- `answer_score` via LLM-as-judge
- file-level citation metrics
- span-level citation metrics
- latency and token usage
- optional diagnostic BERTScore

### Analyze scored runs

```powershell
python src\analyze.py --scored results\seed_v3_scored_final_rawbert --output results\analysis_seed_v3_final
```

### Generate figures

```powershell
python src\visualize.py --scored results\seed_v3_scored_final_rawbert --analysis results\analysis_seed_v3_final --output results\plots_seed_v3_final_rawbert
```

## Context-Rot Evaluation

Context-rot experiments are driven by
[src/context_rot_eval.py](src/context_rot_eval.py). The script constructs a
controlled evidence pool per question and per noise level, then runs the
selected methods against the same pool.

Key design choices:

- `L0` starts from gold-overlapping chunks when available
- `L1`, `L2`, `L3` add increasing noise
- Method B and Method C are restricted to the same pool for fair comparison

Supported noise modes:

- `random_noise`
- `semantic_noise`
- `same_file_noise`
- `inverted_gold_noise`
- `mixed_noise`

Example run:

```powershell
python src\context_rot_eval.py --qa-file qa_dataset\seed_v3_test.json --methods B C --subset qa_dataset\dev_ids.txt --noise-mode inverted_gold_noise --levels L0 L2 L3 --output-dir results_context_rot\phase2_inverted_gold_60 --score --with-citation-judge
```

### Method C ablations

Supported `--c-ablation` modes:

- `full`
- `no_tools`
- `no_followup`

Example:

```powershell
python src\context_rot_eval.py --qa-file qa_dataset\seed_v3_test.json --methods C --noise-mode semantic_noise --levels L0 L3 --c-ablation no_followup --output-dir results_context_rot\phase3_semantic_60_no_followup --score
```

## Deep Analysis

For post-hoc robustness and failure-taxonomy analysis:

```powershell
python src\context_rot_deep_analysis.py --scored results_context_rot\phase2_inverted_gold_60\scored\context_rot_scored_*.jsonl --output-dir results_context_rot\deep_analysis_final
```

This script aggregates:

- grouped context-rot summaries
- pairwise B vs C gaps
- same-file utility diagnostics
- heuristic failure taxonomy tables

## Main Metrics

The primary reported metrics are:

- `answer_score`
- `citation_file_f1`
- `citation_containment_recall`
- `citation_span_precision`
- `citation_weighted_span_score`
- `latency_sec`
- `model_calls`

`BERTScore` is retained only as a diagnostic similarity metric and should not
be treated as the primary correctness metric for implementation-heavy QA.

## Notes On Reproducibility

- Use the same benchmark file and subset files when comparing methods.
- Keep `configs/eval_config.json` fixed for reported runs.
- Treat `results/` and `results_context_rot/` as local experiment outputs rather
  than version-controlled assets.
- Method C is implemented in [src/method_c_iterative.py](src/method_c_iterative.py).
