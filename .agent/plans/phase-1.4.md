# Phase 1.4 — stable IDs and non-destructive incremental indexing

## Objective

Replace the application adapter's destructive, order-dependent indexing lifecycle with
stable document/chunk identities, an explicit persisted manifest, and merge/upsert
semantics. Repeated ingestion must be a no-op, separate API ingestions must accumulate,
and Dense and BM25 must continue to operate over the same logical corpus.

## Current-state observations

- `app/core/vectorstore.py` derives document IDs from source/company/year and chunk IDs
  from a global input position. Moving an unrelated document therefore changes IDs.
- Every non-empty `index_documents` call embeds the full supplied batch, deletes the
  named Qdrant collection, recreates it, and builds BM25 from only that batch.
- Qdrant payloads contain only `chunk_id`, so a persistent collection cannot reconstruct
  the sparse corpus after process restart.
- `app/api/routes.py` calls `index_documents` once per uploaded file. Uploading A and then
  B currently removes A from both branches.
- `FinancialPDFParser` propagates basename `source`, `company`, `year`, page, and table
  flags to chunks. It does not create stable IDs.
- The clean `finrag` packages currently have no indexing package or indexing settings.
- Baseline on 2026-10-10: `uv run pytest -q` passed 366 tests with the two documented
  Pydantic deprecation warnings.

## Scope

- Add standard-library/domain-only identity, manifest, planning, and result contracts
  under `src/finrag/indexing/`.
- Adapt `app/core/vectorstore.py` to execute those plans against Qdrant and rebuild the
  existing BM25/dense/RRF composition from the complete corpus.
- Add manifest-path configuration and application composition wiring.
- Add focused offline unit, Qdrant integration, failure-order, and API-style regression
  tests plus Phase 1.4 documentation.

## Non-goals

- No embedding-model change, retrieval tuning, GraphRAG, generation evaluation, provider
  work, parser redesign, package moves, concurrency protocol, or Phase 1.5 feature.
- No broad Ruff, format, mypy, README, upload-security, or legacy fallback cleanup.

## Architecture decisions

- Explicit IDs are preserved. Generated document IDs hash canonical logical source,
  company, and period. Generated chunk IDs hash document identity, page/section, text,
  and the occurrence among equal chunks in that document.
- A versioned deterministic JSON manifest is the committed index state. Qdrant point IDs
  are deterministic UUIDv5 values derived from stable chunk IDs.
- The manifest signature records collection, identity/payload format, embedding identity,
  vector size, and cosine distance. Existing incompatible state fails explicitly.
- The adapter writes current points first, deletes stale points second, and atomically
  replaces the manifest last. This is recoverable ordering, not a cross-system transaction.
- Qdrant payloads store enough scalar provenance and text to reconstruct the full corpus.
  On restart, exact manifest/payload agreement is required before hybrid retrieval.
- The default in-memory Qdrant runtime keeps its manifest in memory because a disk
  manifest cannot safely describe an ephemeral collection. Persistent Qdrant uses the
  configured manifest path.

## Implementation steps

1. Add and unit-test stable identity and fingerprint functions.
2. Add strict manifest serialization/loading, signature validation, atomic replacement,
   pure NEW/UNCHANGED/CHANGED planning, and typed results.
3. Replace collection recreation with compatibility checks, changed-document embedding,
   upsert-before-stale-delete ordering, and manifest-last commit.
4. Persist reconstructable payloads and restore the complete sparse/dense logical corpus.
5. Wire `FINRAG_INDEXING__MANIFEST_PATH`, update boundaries/docs, and add API A-then-B and
   failure-path regressions.
6. Run focused checks, the full suite, scoped type/lint checks, and repository gates.

## Test strategy

- Pure tests cover explicit/generated IDs, cross-document equality, duplicate occurrences,
  unrelated reordering/insertion, fingerprints, malformed manifests, and planner states.
- In-memory Qdrant tests cover first ingestion, unchanged no-op, changed replacement,
  new-document merge, deterministic point IDs, schema mismatch, restart reconstruction,
  and embedding/upsert failures before stale deletion or manifest advancement.
- An API-style test performs two separate ingestion calls and proves both documents remain
  retrievable. All models are deterministic local fakes; no network or API key is used.
- Compare final pytest results with the 366-test baseline and report repository-wide legacy
  quality-gate debt separately.

## Migration/compatibility concerns

- Collections created by the old payload/schema lifecycle have no compatible manifest and
  are rejected rather than deleted. Operators must explicitly rebuild them.
- Changing embedding identity, vector dimension, distance, identity version, payload
  schema, or collection name requires an explicit rebuild/migration.
- The public `index_documents(documents)` call remains valid and now returns a result that
  callers may ignore. Retrieval ordering/fusion/reranking contracts remain unchanged.

## Acceptance criteria

- IDs and fingerprints are deterministic under the specified invariants.
- Re-ingesting unchanged content performs zero embedding/upsert/delete work.
- A changed document replaces only its stale points; a new document preserves prior data.
- Qdrant is never implicitly deleted or recreated, and incompatible state fails clearly.
- Manifest replacement is deterministic and atomic; failures do not advance it or delete
  stale points before successful embedding/upsert.
- Restart reconstruction cannot silently create an incomplete hybrid corpus.
- Focused checks and the full pytest suite pass without network access.

## Progress checklist

- [x] 2026-10-10: inspected current parser, API, runtime, Qdrant adapter, settings, and tests.
- [x] 2026-10-10: recorded 366-pass baseline and destructive/order-dependent behavior.
- [x] 2026-10-10: implemented clean indexing contracts and focused tests.
- [x] 2026-10-10: integrated non-destructive Qdrant/BM25 lifecycle and restart handling.
- [x] 2026-10-10: added settings, documentation, migration notes, and architecture checks.
- [x] 2026-10-10: ran final verification and classified unchanged repository debt.

## Final results

- Added `finrag.indexing` identities, fingerprints, manifest I/O, planner, and typed result;
  rewrote the application adapter to merge/upsert, reconstruct payloads, and refuse unsafe
  state; wired manifest settings; added 25 focused tests and Phase 1.4 documentation.
- `uv sync` and `uv lock --check --offline` passed with the unchanged 191-package lock.
  Full pytest passed 393 tests with the two pre-existing Pydantic warnings. Targeted Ruff,
  format, and mypy passed all 12 changed Python files; `mypy src/finrag` passed 32 files;
  `git diff --check` passed.
- Repository-wide results remain the documented baseline: eight Ruff findings in four
  untouched legacy files, five format targets, and six mypy errors in four untouched app
  files. No dependency, lockfile, GraphRAG, retrieval-tuning, or generation change occurred.
- Persistent mutation remains single-writer and recoverable rather than cross-system
  transactional. The next scoped task should define explicit document removal and/or
  crash-recovery tooling before concurrent indexing is considered.
