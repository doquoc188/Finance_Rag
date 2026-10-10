# Phase 1.3B — optional local cross-encoder reranking

Implemented on 2026-10-10 from clean commit `e3f90cd`. The results below are
deterministic offline regression evidence. The real BGE model was not downloaded or run,
and no retrieval-quality or latency improvement is claimed.

## 1. Previous post-RRF behavior

`HybridRetriever.retrieve(query, *, limit)` independently requested `dense_top_k` and
`sparse_top_k`, then called RRF with `limit=min(limit, fused_top_k)`. Runtime supplied
`final_top_k` as the graph retrieval limit, so RRF produced only the final-sized list.
Fused evidence used immutable `RetrievedEvidence` values with `HYBRID/RRF` semantics.
`compose_runtime` created `HybridFinancialRetriever`; no reranking-like step existed.
Baseline: **319 passed with two warnings**.

## 2. New reranking architecture

```text
query
  |-- dense top-k
  `-- BM25 top-k
           |
           v
     RRF fused_top_k
           |
           v
   optional reranker
           |
           v
  final_top_k evidence
```

RRF owns recall candidate fusion. The optional cross-encoder receives that complete fused
set and owns precision ordering. Disabled retrieval slices the same deterministic RRF
order. Dense and sparse lists are never reranked independently.

## 3. Reranker contract

`finrag.retrieval.Reranker` is a synchronous typed protocol:

```python
rerank(
    query: str,
    candidates: Sequence[RetrievedEvidence],
    *,
    limit: int,
) -> Sequence[RetrievedEvidence]
```

It has no model loading, global instance, LangGraph, FastAPI, settings, or provider logic.
`HybridRetriever` receives one explicitly and validates its result limit, membership,
unique stable IDs, sequential ranks, preserved strategy/chunk object, and score type.

## 4. BGE implementation and runtime library

`BgeCrossEncoderReranker` targets `BAAI/bge-reranker-v2-m3` through
`sentence_transformers.CrossEncoder`. The installed API was inspected at
sentence-transformers 6.1.0: `CrossEncoder(model_name)` constructs the model and
`predict([(query, passage), ...])` returns per-pair scores. This is also the interface in
the [publisher's model card](https://huggingface.co/BAAI/bge-reranker-v2-m3) and the
[Sentence Transformers CrossEncoder documentation](https://www.sbert.net/docs/cross_encoder/usage/usage.html).

The adapter creates one `(query, chunk.text)` pair per fused candidate, requires one
numeric score per candidate, sorts descending with fused input order as the stable tie
breaker, and returns at most the requested limit.

## 5. Lazy loading behavior

The BGE module imports only standard-library typing/import helpers and clean domain code at
module import. It dynamically imports `sentence_transformers` inside the explicit adapter
constructor. Importing `finrag`, `finrag.retrieval`, or even
`finrag.retrieval.bge` therefore does not load sentence-transformers, Transformers, Torch,
weights, or network code. Construction is the intentional model-load boundary.

## 6. Optional dependency changes

No dependency was added. `sentence-transformers>=3.0.0` was already a default runtime
dependency for the current local embedding path, and the lock resolves version 6.1.0.
Adding a duplicate extra would not make that existing dependency optional. Phase 1.3B
makes reranker construction and model weights optional; `pyproject.toml`, `uv.lock`, and
`requirements.txt` are unchanged.

## 7. Settings added

`RetrievalSettings` adds:

- `reranker_enabled: bool = False`
- `reranker_model: str = "BAAI/bge-reranker-v2-m3"`

The model name must be nonblank. Existing `fused_top_k`, `final_top_k`, and validation that
`final_top_k <= fused_top_k` are reused. Environment names are
`FINRAG_RETRIEVAL__RERANKER_ENABLED` and `FINRAG_RETRIEVAL__RERANKER_MODEL`.

## 8. Disabled behavior

Disabled is the default. RRF now computes at most `fused_top_k`, then the hybrid retriever
slices to the requested final limit. The visible result order/count remains compatible.
A runtime test replaces the concrete factory with a recording function and proves neither
it nor the returned fake's `rerank` method is called. Retrieval regressions prove RRF order
and `RRF` scores remain, truncated to the final limit.

## 9. Enabled behavior

When enabled, runtime constructs or accepts one instance-owned reranker and supplies it to
`HybridFinancialRetriever`. Each indexed `HybridRetriever` passes all RRF
`fused_top_k` candidates to `rerank(..., limit=final_top_k)`. The graph remains unaware of
the concrete adapter, model name, and settings.

## 10. Failure semantics

Construction, model inference, invalid score counts/values, and injected reranker errors
propagate explicitly. A regression asserts the same raised exception object is observed.
There is no fallback to RRF results, dynamic disabling, broad exception catch, or fake
production response.

## 11. Score semantics

Before reranking, fused candidates are `HYBRID/RRF`. After reranking, they remain
`HYBRID` and use only the cross-encoder value with `ScoreType.RERANKER`. RRF and reranker
scores are not normalized, added, or presented as calibrated probabilities.

## 12. Identity and metadata preservation

Reranking creates new immutable evidence wrappers around each exact original
`DocumentChunk` object. A focused test proves `chunk_id`, `document_id`, text, and the same
immutable metadata object survive. Only order, sequential rank, and score semantics change.
The hybrid boundary rejects candidates outside the fused set or duplicate IDs.

## 13. Finance-style scenario

For query `FPT EPS 2025`, fixed RRF candidates are EPS 2024, revenue 2025, and EPS 2025.
The deterministic fake assigns the exact EPS 2025 chunk the largest score and the test
proves it becomes rank one with `RERANKER` semantics. This is an architectural regression,
not a BGE evaluation or measured accuracy result.

## 14. Real graph integration evidence

A real compiled LangGraph uses local chat-model fixtures plus real `HybridRetriever` and
RRF with a fake reranker. Dense and BM25 both execute, the fake sees the complete fused
candidate order, the exact EPS 2025 evidence moves first, grading keeps the evidence, and
generation receives context in the reranked order. The graph itself is not mocked. No
Qdrant server, provider API, embedding model, or real reranker is used.

## 15. Proof normal tests use no model download or network

The concrete adapter is tested through an injected fake CrossEncoder. An isolated process
places import blockers in front of `sentence_transformers`, `transformers`, and `torch`,
replaces socket connection methods with failures, then successfully imports the generic
retrieval packages and BGE module with zero recorded attempts. Runtime tests replace the
factory. The suite's existing autouse network/model-download guards remain active.

## 16. Files changed

Created:

- `.agent/plans/phase-1.3b.md`
- `src/finrag/retrieval/reranking.py`
- `src/finrag/retrieval/bge.py`
- `tests/rerankers.py`
- `tests/test_bge_reranker.py`
- `tests/test_reranking.py`
- `tests/test_reranking_graph.py`
- `docs/phase-1.3b-baseline.md`

Modified:

- `AGENTS.md`, `.env.example`, `docs/development.md`
- `src/finrag/config/settings.py`
- `src/finrag/retrieval/__init__.py`, `src/finrag/retrieval/hybrid.py`
- `app/core/vectorstore.py`, `app/runtime.py`
- `tests/test_finrag_architecture.py`, `tests/test_finrag_retrieval.py`
- `tests/test_finrag_settings.py`, `tests/test_graph_runtime.py`

No package/module was renamed or moved. Graph topology, prompts, providers, indexing
lifecycle, API schemas, and model roles are unchanged.

## 17. Commands executed

`uv` was invoked through `.tools/bootstrap/bin/uv.exe` with workspace-local
`UV_CACHE_DIR` and `UV_PYTHON_INSTALL_DIR`. Equivalent root commands were:

```powershell
uv run pytest -q
uv run pytest <six focused test modules> --tb=short
uv sync
uv lock --check --offline
uv run pytest
uv run ruff check <15 changed Python files>
uv run ruff format --check <15 changed Python files>
uv run mypy --follow-imports=silent src/finrag <changed app and test modules>
uv run mypy src/finrag
uv run mypy --follow-imports=silent app/core/vectorstore.py app/runtime.py
uv run ruff check . --output-format concise
uv run ruff format --check .
uv run mypy app
git diff --check
```

The first focused pre-implementation run failed collection because the new BGE module and
Reranker contract did not exist, recording the intended missing capability. A later
sandboxed editable rebuild could not fetch its isolated `setuptools>=68` build
requirement; the required network-approved `uv sync` retry succeeded. No dependency was
added or upgraded.

## 18. Pytest result

Final full suite: **333 passed, two warnings** on Python 3.12.15, up from the 319-test
baseline through added regressions only. The warnings are unchanged legacy Pydantic APIs
in `app/config.py` and `app/api/schemas.py`. No test was skipped, xfailed, or weakened.

## 19. Targeted Ruff, format, and mypy

Targeted Ruff and Ruff format pass for all 15 changed Python files. Scoped mypy passes for
36 source/application/test files. Exact `mypy src/finrag` passes for 26 source files, and
the two changed application modules pass separately. Final `git diff --check` is recorded
after documentation completion.

## 20. Dependency and lock changes

`uv sync` resolves 191 packages and rebuilds the editable local package. The offline lock
check also resolves 191 packages. `pyproject.toml`, `uv.lock`, provider extras, and the
legacy install path have no diff. The real BGE weights were not cached or downloaded by
verification.

## 21. Legacy debt

Repository-wide baseline still has eight Ruff findings in four untouched legacy Python
files, five unformatted files (including one historical Markdown code block), and six
mypy errors in four untouched application files. The two pytest deprecation warnings
remain. Existing embedding fallback, pre-ingestion mock evidence, heuristic grading,
destructive indexing, unregistered web helper, upload/security issues, and unsupported
README/UI claims are unchanged and were not silently fixed.

## 22. Limitations and deferred work

The real BGE model's multilingual relevance, memory, latency, device behavior, maximum
length handling, and production capacity are unmeasured. No batching/device/fp16/cache/
trust-remote-code setting, model registry, second reranker, query routing, score blending,
benchmark dataset, Recall/MRR comparison, RAGAS, GraphRAG, SPLADE, ColBERT, multimodal
retrieval, or incremental indexing was added. A documented manual load smoke is opt-in;
it was not run. Phase 1.3C and later phases remain untouched.
