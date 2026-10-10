# Phase 1.3B — optional local cross-encoder reranking

## Objective
Add an optional second-stage reranker after hybrid RRF so enabled retrieval reranks the
full fused candidate set before returning `final_top_k`, while the default disabled path
keeps deterministic RRF order and never imports, constructs, or downloads a model.

## Current-state observations
Inspected the clean retrieval/domain code, configuration, application adapter, composition
root, graph, required phase reports, and tests at clean commit `e3f90cd`. Baseline on
2026-10-10 is 319 passed with two existing Pydantic warnings. `HybridRetriever.retrieve`
currently requests `min(limit, fused_top_k)` directly from RRF. Runtime passes
`final_top_k` as `limit`, so only the final-sized RRF list is produced. Scores are typed
`RetrievalScore`; fused results are `HYBRID/RRF`. There is no reranker or reranking-like
step. `compose_runtime` creates an instance-owned `HybridFinancialRetriever`, which builds
the clean `HybridRetriever` when a corpus is indexed, then injects it into the graph.

The existing runtime dependency `sentence-transformers>=3.0.0` is installed at 6.1.0
with transformers 5.19.0. Its inspected `CrossEncoder` constructor accepts a model path,
and `predict` accepts query/text pairs and returns one score array. The current BAAI model
card explicitly documents `CrossEncoder("BAAI/bge-reranker-v2-m3")` followed by
`predict([(query, passage), ...])`. No new Python dependency is required. The model is
about 2.3 GB remotely, so normal imports/tests must never construct it or fetch weights.

## Scope
Add a typed reranker contract, optional BGE CrossEncoder adapter, post-RRF composition,
two retrieval settings, composition-root ownership, deterministic fakes/unit and real
compiled-graph tests, configuration examples, developer documentation, and phase report.

## Non-goals
No second reranker, registry, dynamic routing, Qwen/Cohere/LLM reranking, score blending,
benchmarking, retrieval evaluation, query routing, GraphRAG, incremental indexing,
provider/model changes, prompt changes, numeric redesign, or broad legacy cleanup.

## Architecture decisions
- Add `Reranker` beside the existing `Retriever` protocol. It accepts a query, typed fused
  candidates, and final limit; it owns final ordering and `RERANKER` scores.
- `HybridRetriever` always fuses up to `fused_top_k`. With no reranker it slices that tuple
  to the requested final limit. With an injected reranker it passes only those fused
  candidates to `rerank`; exceptions propagate.
- Implement `BgeCrossEncoderReranker` in an integration module that dynamically imports
  `sentence_transformers.CrossEncoder` only in explicit construction. Generic
  `finrag`/`finrag.retrieval` imports remain standard-library/domain-only and load no ML
  runtime or weights.
- CrossEncoder receives `(query, chunk.text)` pairs once, must return exactly one finite
  score per candidate, and sorts descending with input order as the stable tie-breaker.
  Output reuses each immutable chunk, assigns sequential ranks, retains `HYBRID` strategy,
  and replaces RRF score semantics with `RERANKER`; scores are not added.
- Add frozen settings `reranker_enabled=False` and
  `reranker_model="BAAI/bge-reranker-v2-m3"`. The runtime creates the concrete adapter
  only when enabled and no fully composed retriever was explicitly supplied.
- Keep sentence-transformers in its existing runtime dependency role. Adding a duplicate
  optional extra would not make installation optional while the default embedding path
  already requires it; the optional resource is model construction/download.

## Implementation steps
1. Add settings and clean protocol/composition changes with unit regressions for enabled,
   disabled, final limit, identity, finance ordering, and failure propagation.
2. Add the lazily imported BGE CrossEncoder adapter and fake-model tests for pair input,
   stable sorting, score count validation, and `RERANKER` semantics.
3. Pass optional reranking through `HybridFinancialRetriever`; construct it in
   `app/runtime.py` only when enabled, with tests proving disabled imports/construction do
   not occur and enabled ownership is explicit.
4. Add one real compiled-graph offline test proving dense + BM25 -> RRF -> fake reranker
   ordering reaches grading and generation without Qdrant/server/model/network access.
5. Update contract/config/developer documentation, add a manual opt-in smoke command, and
   complete required verification without running the real model.

## Test strategy
Use typed recording branch retrievers and a test-only score reranker. Assert exact order,
score type, rank, final count, calls, candidate set, exception identity, and unchanged
chunk object/metadata. Test the concrete BGE adapter with an injected fake CrossEncoder,
never its real constructor. In an isolated process, block sentence-transformers,
transformers, torch, network, and model construction while importing generic packages and
exercising disabled composition. Run the real compiled graph with local chat models and
fake retrieval/reranking. Retain the entire Phase 1.3A/1.2C suite.

## Migration / compatibility concerns
Default `reranker_enabled=False` preserves returned RRF order and final count. Fused RRF
now computes up to `fused_top_k` before disabled truncation, which is the intended boundary
and leaves the visible default result unchanged. Enabling the feature may download/load
the configured model at application composition and can require substantial local disk,
memory, and inference time; failures are startup errors, not silent fallback. Stored
indexes, collection lifecycle, APIs, graph policy, prompts, lockfile, and providers do not
change. Rollback is setting `FINRAG_RETRIEVAL__RERANKER_ENABLED=false`.

## Acceptance criteria
Typed contract and one lazy BGE adapter exist; reranking runs only after full RRF;
disabled behavior constructs nothing and preserves truncated RRF order; enabled behavior
uses `RERANKER` scores, final limit, stable identity, and explicit failures; runtime owns
construction; real graph receives reranked evidence; normal tests are offline and make no
model/network call; full pytest and changed-file quality gates pass; later phases remain
untouched.

## Progress checklist
- [x] Inspect implementation, installed API/dependencies, required reports/tests, and baseline.
- [x] Add reranker contract, pipeline behavior, settings, and focused tests.
- [x] Add lazy BGE adapter and runtime ownership tests.
- [x] Add real compiled-graph regression and documentation.
- [x] Complete required verification and final report.

## Final results
Implemented one dependency-injected post-RRF reranking contract and one lazy
sentence-transformers BGE adapter. Disabled remains the default and preserves RRF order;
enabled retrieval sends the full fused set to the reranker and returns final-sized
RERANKER-scored evidence without changing chunk identity. Runtime owns construction and
the graph remains adapter/settings agnostic.

Offline verification completed on Python 3.12.15: `uv sync` and the offline lock check
passed; the full suite passed with 333 tests and two unchanged warnings; all 15 changed
Python files passed targeted Ruff/format; scoped mypy passed 36 files and exact
`mypy src/finrag` passed 26 files. Repository-wide Ruff/format/mypy retain only the
documented legacy baseline. No real model was downloaded or executed, and dependencies
and lockfile are unchanged. See `docs/phase-1.3b-baseline.md` for the complete evidence.
