# Phase 1.3A — true dense + BM25 hybrid retrieval with RRF

Implemented on 2026-10-10 from clean commit `29a3f6e`. Results below are deterministic
offline regression evidence. They are not retrieval-quality benchmarks or measured
production improvements.

## 1. Previous behavior and exact pseudo-hybrid defect

`HybridFinancialRetriever.index_documents` attempted to create both a Qdrant store and a
BM25 object. In this environment, `langchain-community==0.4.2` no longer exports the
imported `langchain_community.vectorstores.Qdrant`, and a broad catch silently set the
store to `None`. Even where a vector store existed, query retrieval returned
`similarity_search` immediately and never queried BM25. The fallback ranked set-token
overlap rather than the constructed BM25 index. No RRF ran, scores were discarded,
documents lacked chunk IDs, and graph/API module globals owned the active retriever.

Baseline: **300 passed with four warnings**.

## 2. New retrieval architecture

```text
query
  |-- Qdrant cosine dense ranking
  `-- BM25 sparse ranking
             |
             v
      Reciprocal Rank Fusion
             |
             v
        fused evidence
```

`finrag.retrieval.Retriever` is the synchronous typed boundary:
`retrieve(query, *, limit) -> Sequence[RetrievedEvidence]`. `BM25Retriever`, pure RRF,
and `HybridRetriever` are framework-independent. The application adapter owns Qdrant,
embedding, LangChain-document conversion, and corpus preparation. Dependencies are
injected; the graph sees only the retriever contract and final typed evidence.

## 3. Dense adapter behavior

The obsolete LangChain Community wrapper was replaced at the existing application
boundary with the already declared `qdrant-client`. Ingestion retains the configured
location, collection name, embedding source, and legacy recreate-on-ingest behavior. The
collection now explicitly uses Qdrant `Distance.COSINE`. `QdrantDenseRetriever` embeds the
query, calls `query_points`, maps payload chunk IDs to the prepared corpus, and returns
`strategy=DENSE`, `score_type=COSINE`, and the native Qdrant score. It does not catch query
errors. A local in-memory Qdrant test verifies the metric and score label.

## 4. BM25 corpus and ownership

The application adapter converts each ingestion batch once into an ordered immutable
`DocumentChunk` corpus. The same chunk objects feed dense points and `BM25Retriever`, so
the sparse index cannot silently use an unrelated list. BM25 tokenizes Unicode words with
case folding, ranks independently through the existing `rank-bm25` dependency, and emits
`strategy=SPARSE`, `score_type=BM25`. Ties retain corpus order.

## 5. RRF formula and default

For every stable chunk ID, pure `reciprocal_rank_fusion` computes:

```text
RRF(d) = sum(1 / (k + rank_i(d)))
```

Ranks are 1-based list positions. `RetrievalSettings.rrf_k` defaults to `60`, rejects
booleans/nonpositive values, and loads from `FINRAG_RETRIEVAL__RRF_K`. Branch scores are
never read by fusion. `dense_top_k`, `sparse_top_k`, `fused_top_k`, and `final_top_k` are
reused; configuration rejects a fused limit larger than the combined candidate limits.

## 6. Stable identity rule

Fusion keys only on `DocumentChunk.chunk_id`, never text. Existing nonblank
`document_id`/`chunk_id` metadata is retained. Legacy documents without IDs receive
deterministic SHA-256 IDs from source/company/year identity plus page/section and ordered
corpus position. Repeating the same ordered corpus reproduces IDs; position keeps two
equal-text chunks distinct. One chunk returned by both branches is fused once. Conflicting
chunks that claim the same ID fail explicitly.

## 7. Tie-breaking

Fused results sort by descending RRF score, then by first appearance while traversing the
dense list followed by the sparse list. Dictionary/set iteration is not an ordering input.
A duplicate within one source ranking contributes only at its first position.

## 8. Score semantics

Dense candidates are `DENSE/COSINE`; sparse candidates are `SPARSE/BM25`; fused candidates
are `HYBRID/RRF`. The graph-facing LangChain metadata includes stable IDs, fused rank,
strategy, score, and score type while retaining source/page/company/year metadata.
No normalization, raw-score addition, weighting, or probability claim is present.

## 9. Settings

Only `RetrievalSettings.rrf_k=60` was added. Existing top-k settings feed
`HybridRetriever`, and runtime passes `final_top_k` to the graph. The prior `strategy`
field remains configuration metadata; query routing is outside this phase. `.env.example`
documents `FINRAG_RETRIEVAL__RRF_K=60`.

## 10. Runtime integration

`compose_runtime` creates one `HybridFinancialRetriever`, injects it into both compiled and
fallback graph engines, and stores it on `GraphRuntime`. The ingestion route indexes that
runtime-owned instance. The graph/API `global_retriever` and `retriever_instance` mutable
singletons were removed. Relevance grading, rewrite routing, generation prompts, numeric
verification, and HTTP response schemas are unchanged. Before ingestion, the existing
demo evidence remains an explicit compatibility path.

## 11. Pure RRF evidence

For dense `A,B,C` and sparse `C,D,A`, tests assert exact values: A and C receive
`1/61 + 1/63`; B and D receive `1/62`. Equal-score ties produce `A,C,B,D` by stable first
appearance. Every output has sequential rank and `HYBRID/RRF` semantics.

## 12. Raw-score independence

A regression changes dense scores from approximately `0.9` to `-999/999` and BM25 scores
from `42/3` to `0.0001/1,000,000` without changing list order. Fused identities and RRF
scores remain identical, proving raw branch magnitudes do not affect fusion.

## 13. Finance-style BM25 regression

The fixed local corpus contains FPT revenue, `FPT EPS 2025 ... 7,200 VND`, and VNM margin
evidence. Query `FPT EPS 2025` ranks the exact EPS chunk first with `SPARSE/BM25` semantics.
This is a lexical regression, not a benchmark result.

## 14. Hybrid-improvement scenario

A fake dense rank places relevant FPT EPS evidence third. Sparse retrieval places it first.
RRF moves it to fused rank one. The test establishes the intended complementary ranking
behavior without inventing a quality percentage.

## 15. Real graph integration evidence

A real compiled LangGraph test injects a deterministic `HybridRetriever`. Dense and sparse
both execute for the original and rewritten query. Initial fused evidence is graded
irrelevant, one bounded rewrite runs, and the second fused result reaches grading and the
generation prompt with `HYBRID/RRF` metadata. The terminal real numeric verifier reports
SUPPORTED. Existing policy regressions still cover success, zero budget, exhaustion,
identical rewrites, larger budgets, and terminal contradictions.

## 16. Failure semantics

An empty dense list, empty sparse list, and both empty are valid and tested. Dense and
sparse exceptions are parameterized separately; the exact exception object propagates.
Qdrant query/index and BM25 construction errors are no longer converted into empty or
token-overlap results.

## 17. Files changed

Created:

- `.agent/plans/phase-1.3a.md`
- `src/finrag/retrieval/__init__.py`
- `src/finrag/retrieval/bm25.py`
- `src/finrag/retrieval/hybrid.py`
- `src/finrag/retrieval/rrf.py`
- `tests/retrievers.py`
- `tests/test_finrag_retrieval.py`
- `tests/test_hybrid_graph.py`
- `tests/test_vectorstore.py`
- `docs/phase-1.3a-baseline.md`

Modified:

- `AGENTS.md`, `.env.example`, `docs/development.md`
- `src/finrag/config/settings.py`
- `app/core/vectorstore.py`, `app/runtime.py`, `app/api/routes.py`
- `app/graph/nodes.py`, `app/graph/workflow.py`
- `tests/conftest.py`, `tests/test_finrag_architecture.py`
- `tests/test_finrag_settings.py`, `tests/test_graph_policy.py`
- `tests/test_graph_runtime.py`, `tests/test_numeric_graph.py`, `tests/test_rag.py`

No application package or existing module was renamed or moved.

## 18. Commands executed

`uv` was invoked through `.tools/bootstrap/bin/uv.exe` with workspace-local
`UV_CACHE_DIR` and `UV_PYTHON_INSTALL_DIR`. Equivalent root commands were:

```powershell
uv run pytest -q
uv sync
uv lock --check --offline
uv run pytest -q
uv run ruff check <21 changed Python files>
uv run ruff format --check <21 changed Python files>
uv run mypy --follow-imports=silent src/finrag <changed app modules>
uv run mypy --follow-imports=silent <changed test modules>
uv run mypy src/finrag
uv run ruff check . --output-format concise
uv run ruff format --check .
uv run mypy app
git diff --check
```

The first sandboxed sync could not fetch the isolated `setuptools>=68` build requirement.
The required retry with approved network access succeeded. No application dependency was
added or upgraded.

## 19. Pytest result

Final full suite: **319 passed, two warnings** on Python 3.12.15. Baseline was 300 passed
with four warnings. The remaining warnings are unchanged legacy Pydantic APIs in
`app/config.py` and `app/api/schemas.py`. Tests use local fakes and in-memory Qdrant; no
external API, model download, credential, persistent vector store, or network is used.

## 20. Ruff, formatting, and mypy

Targeted Ruff and Ruff format pass for all 21 changed Python files. Scoped mypy passes for
29 source/application files and 11 changed test files. The exact requested
`mypy src/finrag` passes for 24 files. `git diff --check` passes; LF/CRLF notices are Git
working-copy warnings rather than whitespace errors.

## 21. Dependencies and lockfile

`pyproject.toml`, `uv.lock`, `requirements.txt`, and provider extras are unchanged.
`qdrant-client` and `rank-bm25` were already required runtime dependencies. `uv sync`
resolved 191 packages, rebuilt the editable local package, and completed; the offline lock
check also resolved 191 packages and passed.

## 22. Remaining warnings and legacy debt

Repository-wide baseline still has eight Ruff findings in four untouched legacy Python
files, five unformatted files (including one historical Markdown code block), and six mypy
errors in four untouched application files. The two pytest deprecation warnings described
above remain. Legacy embedding fallback, pre-ingestion mock evidence, heuristic grading,
unregistered web helper, API error disclosure, upload paths, permissive CORS, PDF parsing,
README/UI claims, and all previously documented numeric limitations remain out of scope.

## 23. Corpus/indexing limitations

Collection replacement remains destructive, matching the legacy `force_recreate=True`
intent. Generated IDs depend on ordered corpus position because parsing supplies no
durable chunk ID. A changed order changes IDs. Dense/BM25 rebuild has no transaction or
recovery protocol; an interruption can leave branches/state inconsistent. There is no
incremental update, deletion reconciliation, checksum/version metadata, tenant isolation,
or concurrent-ingestion control. Phase 1.4 owns these issues.

## 24. Intentionally deferred

No cross-encoder/Cohere/Qwen reranker, weighted fusion, score normalization, SPLADE,
ColBERT, GraphRAG, multimodal retrieval, semantic chunking redesign, query routing, web
search, incremental/checksum indexing, retrieval benchmark, RAGAS, provider/model change,
prompt change, numeric redesign, README rewrite, or Phase 1.3B work was implemented.
