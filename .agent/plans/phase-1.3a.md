# Phase 1.3A — true dense + BM25 hybrid retrieval with RRF

## Objective
Replace the legacy pseudo-hybrid path with independently ranked cosine-vector and BM25
candidates fused by deterministic Reciprocal Rank Fusion, while preserving the current
graph policies, downstream LangChain `Document` shape, and recreate-on-ingest lifecycle.

## Current-state observations
Baseline at clean commit `29a3f6e` on 2026-10-10 is 300 passed with four existing
warnings. `HybridFinancialRetriever.index_documents` attempts to build Qdrant and BM25,
but the installed `langchain-community==0.4.2` no longer provides its imported `Qdrant`
module, and the broad exception handler silently leaves the dense store unset. If a dense
store is present, `get_relevant_documents` returns `similarity_search` immediately, so
BM25 is never queried. Its fallback computes set-token overlap rather than using the BM25
object. Documents have no chunk IDs, scores are discarded, module globals own the active
retriever, and the graph calls the legacy method after checking its mutable document list.

The intended Qdrant call keeps `QDRANT_LOCATION`, `QDRANT_COLLECTION_NAME`, and
`force_recreate=True`; no distance was passed. This phase will restore that path using the
already installed `qdrant-client`, preserve recreation, and explicitly create the vector
collection with Qdrant `Distance.COSINE` so returned scores can be labelled accurately.

## Scope
Add a typed retrieval contract, BM25 implementation, pure RRF, hybrid composition,
LangChain/Qdrant boundary adapter, deterministic corpus identities, one `rrf_k` setting,
instance-owned runtime/graph injection, focused offline tests, and phase documentation.

## Non-goals
No reranker, GraphRAG, retrieval evaluation, semantic chunking, query routing, web search,
provider/model changes, incremental/checksum indexing, deletion/update lifecycle,
collection migration, prompt change, numeric-verifier change, or broad quality cleanup.

## Architecture decisions
- `finrag.retrieval.Retriever` is a synchronous protocol returning
  `Sequence[RetrievedEvidence]`; the framework-independent core imports only domain types,
  standard library, and the existing `rank_bm25` implementation in its sparse adapter.
- The application adapter prepares one ordered corpus for both branches. Explicit IDs are
  retained; otherwise deterministic SHA-256 IDs derive from source identity and corpus
  position. Position keeps equal-text chunks distinct and is stable when the same ordered
  corpus is ingested again. Phase 1.4 still owns durable ingestion identity.
- The Qdrant adapter embeds and ranks independently, uses an explicit cosine collection,
  and returns typed dense scores. BM25 ranks the exact same `DocumentChunk` instances.
- RRF sums `1 / (k + rank)` per stable chunk ID, with `k=60` by default. It never reads
  raw branch scores. Descending fused score and then first appearance across input lists
  provide deterministic ordering.
- `HybridRetriever` receives dense/sparse dependencies and top-k/RRF configuration. Empty
  lists are valid; exceptions propagate unchanged.
- The graph receives a retriever and retrieval limit from composition, converts typed
  evidence back to its established LangChain `Document` state, and knows nothing about
  BM25 or RRF. Runtime ownership replaces graph/API module globals. The existing mock
  document remains only for the deliberate no-index compatibility path.

## Implementation steps
1. Add `rrf_k` validation and clean retrieval modules with focused unit tests.
2. Replace the legacy adapter internals with shared-corpus preparation, direct Qdrant
   cosine retrieval, BM25, and injected `HybridRetriever` composition.
3. Bind retrieval explicitly in graph construction and runtime; make ingestion update the
   runtime-owned adapter without changing request/response schemas.
4. Migrate existing graph fixtures from the global legacy subclass to typed retrievers;
   add a real compiled-graph hybrid regression and preserve self-correction coverage.
5. Document exact behavior and run sync, offline lock, pytest, targeted quality checks,
   scoped mypy, and whitespace verification.

## Test strategy
Pure tests cover exact RRF contributions/order, raw-score independence, stable identity,
equal-text distinctness, empty branches, exception propagation, and hybrid improvement.
A finance-style BM25 corpus proves exact ticker/year/metric lexical ranking. A local fake
embedding plus in-memory Qdrant exercises the application dense adapter without network.
A deterministic hybrid dependency runs through the actual compiled graph and proves both
branches execute and fused evidence reaches grading/generation. Existing bounded rewrite
and numeric tests remain regression coverage.

## Migration / compatibility concerns
The collection is still recreated on each ingestion, matching the legacy
`force_recreate=True` intent; Phase 1.4 must replace this destructive lifecycle. Generated
chunk IDs depend on ordered corpus position when upstream metadata has no explicit ID, so
reordering changes those IDs. Qdrant and BM25 are rebuilt from the same in-memory corpus in
one adapter call, but there is no transactional recovery from a partial rebuild. The old
pre-ingestion demo document remains for current behavior. No dependency or lock change is
planned because both `qdrant-client` and `rank-bm25` are already runtime dependencies.

## Acceptance criteria
Both branches execute independently and fuse only by RRF rank; stable IDs merge the same
chunk and retain equal-text distinct chunks; score labels remain cosine/BM25/RRF; empty
branches work while failures propagate; runtime has no global retriever; real graph and
self-correction tests pass; full pytest and changed-file Ruff/format/mypy pass; all later
retrieval, indexing, GraphRAG, and evaluation work remains deferred.

## Progress checklist
- [x] Inspect current retrieval, metadata, settings, graph/runtime ownership, tests, and baseline.
- [x] Add clean retrieval core, settings, and unit regressions.
- [x] Add Qdrant/BM25 application adapter and deterministic corpus identity.
- [x] Inject runtime-owned retrieval into the graph and migrate tests.
- [x] Complete documentation and required verification.

## Final results
Implemented true independent Qdrant cosine and BM25 retrieval over one prepared corpus,
deterministic stable-ID RRF with default k=60, typed score semantics, and instance-owned
runtime/graph injection. Removed silent query fallback and retrieval globals while keeping
the pre-ingestion demo and legacy recreate-on-ingest lifecycle. Final full pytest is 319
passed with two legacy warnings; sync/offline lock, changed-file Ruff/format, scoped mypy,
standalone mypy src/finrag, and whitespace checks pass. No dependency/lock, reranking,
GraphRAG, benchmark, prompt, provider, numeric-verifier, or indexing-lifecycle change was
made. The complete evidence and limitations are in `docs/phase-1.3a-baseline.md`.
