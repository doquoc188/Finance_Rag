# Phase 1.3C — retrieval evaluation harness and honest baseline

Implemented on 2026-10-10 from clean Phase 1.3B commit `46fc08d`. This report separates
actual implementation measurements from deterministic test evidence. No LLM judge,
generation evaluation, retrieval tuning, or fabricated metric is included.

## 1. Benchmark file locations

The benchmark is under `evals/retrieval/`:

- `corpus.jsonl`: fixed chunks with explicit IDs and scalar provenance metadata.
- `queries.jsonl`: questions with manually assigned relevant chunk IDs.
- `README.md`: schema, methodology, fixed configuration, commands, and limitations.
- `run_benchmark.py`: local-only composition for actual implementations.
- `results/baseline.json`: deterministic machine output and per-query diagnostics.
- `results/baseline.md`: rounded human comparison table.

## 2. Corpus size

The corpus contains **47** synthetic/curated chunks: FPT 10, VNM 8, HPG 8, MWG 7,
VCB 9, and SSI 5. Text includes adjacent annual/quarterly periods, nearby finance metrics,
percentages, Vietnamese `tỷ`/`triệu`, and English units. Every text begins with a synthetic
fixture label. It is not private report content or a claim about the named companies.

## 3. Query count

There are **27** queries, within the requested 20–30 range. Every query is evaluated once
per measured strategy in file order and has at least one relevant chunk.

## 4. Query/category composition

Languages: 11 Vietnamese, 13 English, and 3 mixed. Categories: bilingual terminology 2,
exact company/metric 6, hard metric 4, hard period 2, hard year 6, multi-relevant 2, and
semantic paraphrase 5. The two multi-relevant queries each require two chunks. Nearby
year/metric distractors prevent a one-document-per-token toy layout.

## 5. Gold-label methodology

Relevant IDs were written manually while curating each query and corpus fixture. Dense,
BM25, RRF, BGE, and LLM output did not generate or revise the labels. The data was fixed
before the first actual baseline run. Validation fails on empty or duplicate gold sets and
on any ID absent from the corpus.

## 6. Stable benchmark-ID policy

Corpus IDs are explicit `bench_*` strings stored in JSONL; document IDs are explicit too.
They never call the production order-dependent ID generator. Query IDs are explicit `q_*`
strings. Loaders reject duplicate chunk/query IDs, malformed JSON/records, unexpected
fields, blank required strings, invalid metadata, and dangling relevance judgments.

## 7. Metric formulas

For each query:

```text
Recall@K = unique relevant chunk IDs retrieved in top K / total relevant chunk IDs
MRR@K    = 1 / first relevant rank within top K, otherwise 0
```

Recall@5, Recall@10, and MRR@10 are arithmetic macro averages over all queries. Metric
unit tests cover ranks 1 and 5, outside cutoff, multiple relevant IDs, no hit, and fewer
than K returned results with exact expectations. Internal results are unrounded.

## 8. Evaluation top-k policy

The runner always requests ten results because ten is the largest metric cutoff. Dense and
BM25 branch limits are 20; RRF keeps 20; reranking returns 10. RRF `k=60`. Thus the
production `final_top_k=4` cannot truncate evaluation. These initial values were recorded
before measurement and were not adjusted after viewing results.

## 9. Strategy definitions

Stable labels are exactly:

- `dense`: production `QdrantDenseRetriever`, current MiniLM embeddings, in-memory cosine
  Qdrant.
- `bm25`: current `BM25Retriever` over the benchmark chunks.
- `hybrid_rrf`: current `HybridRetriever` over those actual dense and BM25 branches.
- `hybrid_rrf_reranked`: the same full fused set plus Phase 1.3B
  `BgeCrossEncoderReranker`.

No label is inferred from a class name and no strategy receives different query/gold data.

## 10. Evaluation architecture

`finrag.evaluation` contains immutable benchmark/result values, strict JSONL loading,
metric functions, the named Retriever-protocol runner, deterministic JSON serialization,
and Markdown formatting. It imports only standard-library, domain, and retrieval code.
Architecture tests block application, LangChain, Qdrant, ML runtimes, and network during
its import.

The separate `evals/retrieval/run_benchmark.py` composition boundary owns optional model
and in-memory Qdrant construction. This keeps the core clean while using the actual
production dense query adapter where local assets permit.

## 11. Actual BM25 benchmark execution status

**MEASURED.** The runner instantiated the actual `BM25Retriever` over all 47 chunks and
evaluated all 27 queries. `rank-bm25==0.2.2` is recorded in the result configuration.

## 12. Actual Dense benchmark execution status

**NOT MEASURED.** The configured `sentence-transformers/all-MiniLM-L6-v2` snapshot was not
present in the inspected local Hugging Face/Torch cache. The runner uses the current
LangChain HuggingFace embedding path with `local_files_only=True` and the actual
`QdrantDenseRetriever`, but it does not call the legacy helper because that helper can
silently substitute constant embeddings. No substitute model/vector was used.

## 13. Actual Hybrid RRF benchmark execution status

**NOT MEASURED.** Actual Hybrid RRF requires the unavailable actual dense branch. BM25
alone was not relabelled as hybrid, and fake dense results were not published.

## 14. Actual BGE reranker benchmark execution status

**NOT MEASURED.** Neither the MiniLM embedding snapshot nor
`BAAI/bge-reranker-v2-m3` was cached locally. The runner checked both in local-only mode.
No download, fake reranker, or RRF fallback metric was substituted.

## 15. Measured metrics

Only actual BM25 was executed:

| Strategy | Recall@5 | Recall@10 | MRR@10 |
| --- | ---: | ---: | ---: |
| BM25 | 0.7593 | 0.8148 | 0.7025 |

Machine output retains `0.7592592592592593`, `0.8148148148148148`, and
`0.7024691358024691`. These values describe only this 47-chunk curated synthetic fixture.
They do not establish production quality or superiority over unmeasured strategies.

## 16. Fake/deterministic test evidence

Recording retrievers prove every question is called once at limit ten, sequence order
controls ranks, relevant hit ranks are retained, macro aggregation is exact, unknown names
and malformed rank sequences fail, and JSON rendering is repeatable. These fake outcomes
exist only inside tests. The actual BM25 smoke verifies implementation compatibility but
does not assert a minimum score or claim improvement.

## 17. Machine-readable output location

`evals/retrieval/results/baseline.json` contains benchmark/corpus/query identity,
configuration and package versions, only measured strategy metrics, every retrieved ID,
relevant hit/rank, and explicit unmeasured reasons. It has no timestamp and sorts keys for
deterministic rendering.

## 18. Human-readable output location

`evals/retrieval/results/baseline.md` rounds measured values to four decimals and marks
Dense, Hybrid RRF, and Hybrid RRF plus reranker as NOT MEASURED with exact missing-model
reasons. It does not place fake values in empty cells.

## 19. Files changed

Created:

- `.agent/plans/phase-1.3c.md`
- `evals/retrieval/{README.md,corpus.jsonl,queries.jsonl,run_benchmark.py}`
- `evals/retrieval/results/{baseline.json,baseline.md}`
- `src/finrag/evaluation/{__init__.py,retrieval.py}`
- `tests/test_retrieval_evaluation.py`
- `tests/test_retrieval_benchmark_cli.py`
- `docs/phase-1.3c-baseline.md`

Modified:

- `AGENTS.md`
- `docs/development.md`
- `tests/test_finrag_architecture.py`

No production retrieval/runtime/graph/configuration module was changed or moved.

## 20. Commands executed

`uv` used `.tools/bootstrap/bin/uv.exe` with workspace-local cache/Python directories.
Equivalent root commands were:

```powershell
uv run pytest tests/test_retrieval_evaluation.py --tb=short
uv run pytest tests/test_retrieval_evaluation.py tests/test_retrieval_benchmark_cli.py
uv run pytest tests/test_retrieval_evaluation.py tests/test_retrieval_benchmark_cli.py tests/test_finrag_architecture.py
uv run python -m evals.retrieval.run_benchmark
uv sync
uv lock --check --offline
uv run pytest
uv run ruff check <6 changed Python files>
uv run ruff format --check <6 changed Python files>
uv run mypy --follow-imports=silent <evaluation, runner, and changed tests>
uv run mypy src/finrag
uv run mypy --follow-imports=silent evals/retrieval/run_benchmark.py
uv run ruff check . --output-format concise
uv run ruff format --check .
uv run mypy app
git diff --check
```

The initial focused test failed collection because `finrag.evaluation` did not exist,
recording the missing capability. The baseline runner was repeated only to add environment
version fields to output; data, judgments, retrieval parameters, and measured metrics did
not change.

## 21. Pytest result

Final full suite: **366 passed with two warnings** on Python 3.12.15, compared with the
333-test Phase 1.3B baseline. The warnings are unchanged Pydantic deprecations in
`app/config.py` and `app/api/schemas.py`. No test was skipped, xfailed, or weakened.

## 22. Ruff, format, and mypy results

Targeted Ruff and Ruff format pass for all six changed Python files. Scoped mypy passes
the evaluation package, runner, and changed tests. Exact `mypy src/finrag` passes for 28
source files; the standalone benchmark runner also passes separately. `git diff --check`
passes after documentation completion.

Repository-wide baseline remains eight Ruff findings in four untouched legacy Python
files, five unformatted files (including one historical Markdown code block), and six mypy
errors in four untouched application files. These are pre-existing Phase 1 debt.

## 23. Dependency and lockfile changes

None. `pyproject.toml`, `uv.lock`, and requirements files are unchanged. `uv sync` and
offline lock validation resolve the existing 191 packages. The result records Python
3.12.15, qdrant-client 1.19.1, rank-bm25 0.2.2, and sentence-transformers 6.1.0.

## 24. Network/model-download behavior

Pytest retains global socket/model-cache guards and imports no evaluation ML/application
runtime. The manual runner forces `HF_HUB_OFFLINE=1`, `TRANSFORMERS_OFFLINE=1`, disables
telemetry, and calls snapshot resolution with `local_files_only=True`. It uses in-memory
Qdrant and contains no OpenAI/provider call. Neither MiniLM nor BGE weights were downloaded
or loaded during this phase.

## 25. Benchmark limitations

The benchmark is small, synthetic, manually curated by one pass, and limited to six named
Vietnamese companies and short text chunks. It has no tables, OCR noise, long-document
chunking, aliases beyond its fixtures, adjudicated labels, graded relevance, production
distribution, multilingual breadth beyond Vietnamese/English, or statistical confidence
interval. Its queries and gold are not a held-out production sample. Metrics must not be
generalized to answer correctness, financial accuracy, or deployment performance.

## 26. Intentionally deferred work

No generation/answer/faithfulness/citation evaluation, RAGAS, LLM judge, retrieval tuning,
hyperparameter search, alternate embedding/reranker, GraphRAG, SPLADE, ColBERT, routing,
web search, production ID redesign, incremental indexing, provider/prompt change, or
Phase 1.4 work was implemented. Completing Dense/Hybrid/BGE measurements requires the
documented local model assets and a separately named result run, without tuning this
baseline.
