# Phase 1.3C — retrieval evaluation harness and honest baseline

## Objective
Add a reproducible retrieval-only benchmark that evaluates Dense, BM25, Hybrid RRF, and
Hybrid RRF plus the existing BGE reranker against one fixed curated finance corpus and
manual gold labels using macro Recall@5, Recall@10, and MRR@10. Normal verification stays
offline and results distinguish deterministic test evidence from actual measured runs.

## Current-state observations
Inspected the Phase 1.3A/1.3B reports, immutable domain values, retrieval contracts,
settings, application dense adapter, embedding helper, runtime composition, and tests.
The clean `Retriever` contract returns ordered `RetrievedEvidence`; BM25 and Hybrid RRF
are framework-independent. `QdrantDenseRetriever` is the production dense boundary and
supports in-memory Qdrant when supplied embeddings/chunks. The legacy embedding helper
uses `HuggingFaceEmbeddings` with `sentence-transformers/all-MiniLM-L6-v2`, but catches all
failures and substitutes constant vectors, which is unacceptable for measured evaluation.
The benchmark runner must therefore construct the same local embedding model explicitly,
require local files, and use the real Qdrant adapter without the fallback.

The installed SentenceTransformer API supports `local_files_only=True`; the LangChain
wrapper passes model kwargs. Inspected local Hugging Face/Torch cache locations contain
neither `sentence-transformers/all-MiniLM-L6-v2` nor `BAAI/bge-reranker-v2-m3`. Therefore
only actual BM25 can currently be measured without a download. Dense, Hybrid RRF, and
reranked measurements must be marked NOT MEASURED with reproducible local-only commands.
Phase 1.3B's completed working tree is the starting point. Its full offline baseline is
333 passed with two documented Pydantic warnings.

## Scope
Add `src/finrag/evaluation`, a fixed JSONL benchmark under `evals/retrieval`, one local-only
manual runner, deterministic JSON/Markdown reporting, metric/data/runner tests, an actual
BM25 result artifact, architecture contract updates, developer documentation, and the
Phase 1.3C report.

## Non-goals
No generation evaluation, LLM judge, RAGAS, answer/faithfulness/citation scoring, metric or
retrieval tuning, hyperparameter search, new embeddings/rerankers, GraphRAG, query routing,
web search, provider/prompt changes, production ID changes, or incremental indexing.

## Architecture decisions
- Store explicit `bench_*` chunk/document/query IDs in JSONL. Loaders never derive IDs
  through production indexing and reject malformed/duplicate/dangling data.
- Keep benchmark models, metric math, runner, serialization, and table formatting in a
  standard-library/domain/retrieval-only `finrag.evaluation` package. It consumes the
  existing `Retriever` protocol and never imports application/model/provider frameworks.
- Fix strategy labels to `dense`, `bm25`, `hybrid_rrf`, and `hybrid_rrf_reranked`.
  Evaluation always requests ten results. Hybrid branches use top 20, RRF keeps 20, and
  the final layer returns ten; production settings remain unchanged.
- Place optional local composition in `evals/retrieval/run_benchmark.py`. BM25 uses the
  actual core implementation. Dense uses the current `QdrantDenseRetriever`, in-memory
  Qdrant cosine collection, and the current MiniLM embedding model through the current
  LangChain wrapper, with a resolved local snapshot and no fallback. Hybrid and reranked
  strategies reuse those actual components. BGE uses the Phase 1.3B adapter.
- Force local-only model resolution. Missing snapshots produce an explicit NOT MEASURED
  reason while other available requested strategies still run. No remote APIs or hidden
  substitutions are allowed.
- Persist unrounded numeric results in deterministic JSON without timestamps; round only
  the Markdown table. Publish only actually executed implementation metrics.

## Implementation steps
1. Add explicit curated corpus/query JSONL plus README, then validate counts, categories,
   stable IDs, gold references, and schema through focused tests.
2. Add typed benchmark values/loaders, exact metric functions, named runner, deterministic
   serializers, and human table with independent unit/negative tests.
3. Add the local-only strategy composition CLI using actual BM25, Qdrant dense, Hybrid RRF,
   and optional BGE components with evaluation-specific 20/20/20/10 limits.
4. Run the actual BM25 baseline once. Inspect local assets before any dense/reranker run;
   mark unavailable strategies NOT MEASURED and do not change data/configuration to improve
   scores.
5. Update contracts/docs, complete full offline verification, record existing repository
   debt separately, and write the 26-part phase report.

## Test strategy
Test Recall@K and reciprocal rank directly for rank 1, rank 5, outside cutoff, multiple
relevant IDs, no hit, and fewer-than-K lists using exact rational expectations. Validate
real benchmark data and isolated invalid files for duplicate IDs, blank questions, empty
gold, dangling IDs, bad metadata/types, and malformed JSON. A recording fake proves each
query is called once with limit ten, returned order controls ranks, names are stable, and
serialization is byte-deterministic. An actual BM25 smoke uses the benchmark corpus but is
not reported as fake evidence. Normal pytest retains global offline/model-cache guards.

## Migration / compatibility concerns
Production configuration, graph/runtime composition, stored indexes, retrieval defaults,
and public application contracts do not change. New evaluation imports remain lightweight.
Result schema is versioned by benchmark name and deterministic configuration fields; there
is no historical result migration. Deleting the new eval package/artifacts fully rolls back
this phase.

## Acceptance criteria
A fixed 20–30-query curated finance benchmark with explicit stable IDs and manual gold
validates; exact Recall@5/10 and MRR@10 math passes; all four fixed labels share one dataset
and fair top-k policy; real BM25 and opt-in local-only Dense/Hybrid/BGE paths exist; missing
assets yield NOT MEASURED without substitution; JSON and Markdown output are deterministic;
normal tests make no network/model calls; full pytest and changed-file quality gates pass;
no tuning or later-phase work appears.

## Progress checklist
- [x] 2026-10-10: Inspect contracts, production dense path, installed APIs, caches, tests,
  and Phase 1.3B baseline.
- [x] 2026-10-10: Add and validate the curated corpus and manual query judgments.
- [x] 2026-10-10: Implement evaluation values, metrics, runner, reporting, and focused tests.
- [x] 2026-10-10: Implement local-only actual strategy composition and run available
  baseline(s).
- [x] 2026-10-10: Complete documentation, verification, and final results.

## Final results
Added a 47-chunk, 27-query curated synthetic benchmark with explicit IDs and manual gold;
standard-library/domain/retrieval-only evaluation values, validation, Recall@5/10 and
MRR@10, per-query runner diagnostics, deterministic JSON, and Markdown output; plus a
local-only actual strategy composition path. Production runtime and settings are unchanged.

The untuned actual run measured BM25 only: Recall@5 0.7593, Recall@10 0.8148, and MRR@10
0.7025. Dense and Hybrid are NOT MEASURED because the MiniLM snapshot was absent; the
reranked path additionally lacked BGE. No substitute or download was used. Machine and
human artifacts record those statuses and the fixed 20/20/20/10, RRF k=60 configuration.

Verification completed on Python 3.12.15: `uv sync` and offline lock validation passed;
full pytest passed 366 tests with two unchanged warnings; six changed Python files passed
targeted Ruff/format and scoped mypy; exact `mypy src/finrag` passed 28 files; and final
`git diff --check` passed. Repository-wide failures remain only the documented untouched
legacy baseline. See `docs/phase-1.3c-baseline.md` for the 26-part result report.
