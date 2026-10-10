# Phase 1.4 — stable IDs and non-destructive incremental indexing

Implemented on 2026-10-10 from the clean Phase 1.3C baseline. This report describes
deterministic offline behavior and local in-memory Qdrant tests. It does not claim a live
provider, remote Qdrant, concurrent-writer, crash-transaction, or retrieval-quality result.

## 1. Previous indexing lifecycle

`HybridFinancialRetriever.index_documents` prepared one supplied sequence, embedded every
chunk, deleted the named Qdrant collection when present, recreated it, and rebuilt BM25
from only that call. API upload A followed by B therefore removed A. Qdrant payloads stored
only `chunk_id`, so BM25 could not be reconstructed after restart.

## 2. Previous ID behavior

Generated document IDs hashed source/company/year, but missing sources became the shared
string `unknown`. Generated chunk IDs included the global sequence position. Reordering or
inserting an unrelated chunk could therefore change existing IDs.

## 3. Clean indexing boundary

`src/finrag/indexing/` contains identity/fingerprint functions, immutable manifest values,
pure planning, typed results, and atomic JSON persistence. Architecture tests permit only
standard-library and `finrag.domain` dependencies. LangChain and Qdrant remain in the
application adapter.

## 4. Stable document-ID scheme

Explicit nonblank document IDs are retained. Otherwise the ID is `document-` plus SHA-256
of canonical JSON containing identity version, normalized logical source, company, and
period/year. It never includes content, global position, randomness, or a temporary upload
path. Missing/placeholder sources and absolute filesystem paths fail clearly.

## 5. Stable chunk-ID scheme

Explicit nonblank chunk IDs are retained. Otherwise the ID is `chunk-` plus SHA-256 of
document ID, page, section, normalized-text SHA-256, identity version, and duplicate
occurrence among equal document-local chunks. Equal text in different documents and equal
repeated chunks in one document remain distinct.

## 6. Reordering and insertion behavior

IDs do not contain corpus position. Tests prove an unrelated document reorder/insertion
does not change an existing ID and inserting a distinct chunk into the same document keeps
the existing chunk's ID. Ordered prepared content still changes the document fingerprint,
so a structural reorder is intentionally planned as a document update.

## 7. Prepared-content fingerprint

Each document receives a SHA-256 digest of canonical ordered records containing normalized
text, page, section, and sorted meaningful scalar metadata. Identity fields are excluded.
Tests distinguish text, order, and page changes and prove explicit ID changes alone do not
change the content fingerprint.

## 8. Manifest format

The JSON manifest records `manifest_version`, an `index_signature`, and sorted document
records with document ID, source, content SHA-256, and ordered chunk IDs. Loading is strict:
malformed JSON, missing/extra fields, unsupported versions, invalid digests, and global ID
collisions fail. JSON output is UTF-8, sorted, deterministic, and contains no pickle data,
credentials, vectors, or report text.

## 9. Index signature

The signature records collection name, effective embedding identity, vector size, cosine
distance, identity version, and payload-schema version. A changed model/dimension,
distance, collection, identity format, or payload schema is incompatible and requires an
explicit migration/rebuild.

## 10. Pure incremental plan

`plan_incremental_index` classifies each supplied document as NEW, UNCHANGED, or CHANGED
from source, fingerprint, and ordered chunk IDs. It has no storage side effects. It also
rejects incoming IDs that collide with an absent document retained by the manifest.

## 11. Batch merge semantics

Applying a plan replaces only documents present in the batch and retains all absent
manifest documents. There is no implicit delete-missing behavior. Explicit whole-document
deletion remains deferred because it needs a separate policy and API.

## 12. Unchanged-document behavior

Re-ingesting byte-for-byte equivalent prepared content performs zero document-embedding,
Qdrant upsert, and Qdrant delete calls and leaves manifest bytes unchanged. The typed result
reports the unchanged count and complete corpus totals.

## 13. Changed-document behavior

Only current chunks for CHANGED documents are embedded/upserted. Stale point IDs are the
previous chunk IDs absent from the current document. Tests change A while retaining B and
prove only A is embedded/upserted/deleted and B's payload remains identical.

## 14. New-document behavior

NEW documents are embedded and upserted without altering existing documents. A batch may
contain NEW and UNCHANGED records together; only NEW content is embedded. Separate API
uploads A then B leave both in the complete corpus.

## 15. Qdrant collection lifecycle

A missing collection is created with the measured vector dimension and cosine distance.
An existing compatible empty collection is reused. Existing vector size/distance mismatch
or nonempty storage without a compatible manifest fails explicitly. Production code no
longer calls `delete_collection`.

## 16. Deterministic point IDs

Every stable chunk ID maps through UUIDv5 with a fixed project namespace. Repeated runs
therefore address the same Qdrant point without relying on input order or random UUIDs.

## 17. Reconstruction payload

Each Qdrant point stores payload schema version, text, source, document ID, chunk ID, page,
section, and scalar metadata. Payload identity fields must agree, and the point UUID must
match the chunk ID. This is enough to rebuild the same `DocumentChunk` corpus safely.

## 18. Dense/BM25 consistency

After commit, both `QdrantDenseRetriever` and `BM25Retriever` are composed over chunk IDs in
the complete merged manifest. Tests compare dense payload sources with application/BM25
document sources after NEW and CHANGED ingestions. Dense → BM25 → RRF → optional reranker
semantics and score labels are unchanged.

## 19. Restart behavior

When a manifest exists, startup requires the collection and validates its vector schema.
It scrolls all payloads, reconstructs chunks, and requires exact equality between manifest
and Qdrant chunk-ID sets plus matching provenance. A missing collection, extra/missing
payload, invalid schema, or signature mismatch fails instead of serving incomplete hybrid
results.

## 20. Lazy embedding behavior after restart

Payload reconstruction does not embed documents or load an embedding model. The first
retrieval resolves the existing embedding path for query embedding and then composes both
branches over the restored full corpus. A test proves restart retrieval makes no document
embedding call.

## 21. Mutation and commit order

The adapter prepares/plans, embeds, validates/creates storage, upserts all current points,
deletes stale points, and writes the new manifest last. Runtime corpus state is replaced
only after all these steps report success.

## 22. Failure behavior

Embedding failure occurs before storage mutation. Upsert failure occurs before stale
deletion or manifest advancement; a recording Qdrant test proves the old payload and
manifest remain. After any reported Qdrant/manifest mutation failure, retrieval is blocked
until the same ingestion is retried successfully.

## 23. Atomic manifest boundary

Manifest writes use a sibling temporary file, flush plus `fsync`, and `os.replace`, with
temporary-file cleanup. This protects against partially written JSON. It is not an atomic
transaction with Qdrant; a process crash after Qdrant mutation can leave exact-set
validation failure and require retry/rebuild.

## 24. Indexing result

`IndexingResult` reports NEW, UNCHANGED, and CHANGED document counts; embedded, upserted,
and deleted chunk counts; and final document/chunk totals. `index_documents` returns it
without changing the existing API response contract, whose `chunks_created` still means
parser output chunks.

## 25. Configuration

`IndexingSettings.manifest_path` defaults to `.finrag/index-manifest.json` and supports
`FINRAG_INDEXING__MANIFEST_PATH`. The default directory is gitignored. Application runtime
uses disk manifests with persistent legacy `QDRANT_LOCATION`; the default `:memory:` store
keeps its manifest in memory to avoid a false persistence guarantee.

## 26. Tests added

Pure tests cover logical document IDs, duplicate occurrence, UUIDs, fingerprints,
deterministic/strict manifest I/O, planner states, retained-ID collisions, and malformed
formats. Qdrant tests cover no-op, insert/reorder stability, duplicate text, explicit IDs,
changed-only replacement, restart, failure injection, schema mismatch, orphan collection,
signature mismatch, payload mismatch, and missing collection. A real FastAPI test performs
two separate multipart ingestion calls. All are local and network-free.

## 27. Migration procedure

Legacy collections have no compatible manifest/payload and are intentionally not adopted
silently. Stop all writers, archive or explicitly remove both old collection and manifest,
then ingest the complete source corpus under one fixed embedding configuration. There is
no automatic migration, destructive reset endpoint, or remove-document API in this phase.

## 28. Limitations and deferred debt

Index mutation assumes one writer; no filesystem/distributed lock exists. Qdrant and the
manifest do not share a transaction. The existing legacy embedding helper can still fall
back to a deterministic dummy on initialization error, though the manifest now records the
effective object identity rather than falsely naming it as the requested model. Upload
path validation, parser redesign, broad lint/type cleanup, GraphRAG, retrieval tuning,
generation evaluation, and document-deletion policy remain out of scope.

## 29. Verification and next task

The pre-change full baseline was **366 passed with two existing warnings**. Final
`uv run pytest` passed **393 tests with the same two warnings**; the focused identity and
incremental-indexing suite passed **25 tests**. `uv sync` and offline lock validation passed
with 191 resolved packages. Targeted Ruff/format/mypy passed all 12 changed Python files,
`mypy src/finrag` passed 32 files, and `git diff --check` passed. Repository-wide debt is
unchanged: eight Ruff findings in four untouched legacy Python files, five format targets,
and six mypy errors in four untouched application files. No dependency or lockfile change
was required. The next scoped task should design explicit document removal and/or
crash-recovery tooling; concurrent-writer coordination remains a separate decision.

Commands were run from the repository root with the workspace-local uv cache/Python:

```powershell
uv run pytest -q
uv run pytest tests/test_finrag_indexing.py tests/test_incremental_indexing.py -q
uv sync
uv lock --check --offline
uv run pytest
uv run ruff check <12 changed Python files>
uv run ruff format --check <12 changed Python files>
uv run mypy --follow-imports=silent <12 changed Python files>
uv run mypy src/finrag
uv run ruff check . --output-format concise
uv run ruff format --check .
uv run mypy app
git diff --check
```
