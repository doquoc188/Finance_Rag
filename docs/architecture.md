# Architecture decisions

FinRAG keeps framework, provider, retrieval, and financial-verification concerns behind
small explicit boundaries. This page records the decisions a reviewer needs to understand;
historical implementation details remain in the phase reports.

## Runtime composition

`app/runtime.py` is the application composition root. A FastAPI lifespan creates one fresh
provider registry, resolves the generation, grading, and query-rewrite roles, builds the
graph, and owns the retriever. Graph nodes receive already constructed `BaseChatModel`
instances and do not read credentials or select providers.

### Role-based models instead of one global LLM

Each graph responsibility has an explicit provider/model/temperature setting. This allows
capability checks and provider changes at the role boundary while avoiding import-time
clients, a global model cache, and hidden cross-request configuration.

### No silent production fake

Missing credentials, optional SDKs, or provider initialization fail with contextual errors.
Deterministic chat models exist only as explicitly injected test doubles. Production
composition never catches a provider failure and substitutes a canned answer.

## Retrieval and reranking

Dense retrieval and BM25 independently rank the same stable chunk corpus. Reciprocal Rank
Fusion combines their rank positions with `sum(1 / (k + rank))`; it does not add or compare
raw cosine and BM25 scores, whose meanings differ.

The optional cross-encoder receives the complete fused candidate set. Its output replaces
the ranking score semantics with `RERANKER`; it is not arithmetically added to the RRF score.
Every result continues to carry its stable chunk and source identity.

## Graph policy and numeric checks

Evidence grading can request a query rewrite only while an explicit attempt budget remains.
The bound prevents an unproductive model or evidence set from creating an unbounded graph
loop.

Numeric verification is deterministic and uses `Decimal`. It extracts supported explicit
quantities, keeps percentage/currency semantics, and compares compatible lexical contexts.
The outcomes `SUPPORTED`, `CONTRADICTED`, and `NOT_FOUND` remain distinct. This component
does not claim semantic entailment, derived arithmetic, currency conversion, or general
hallucination detection.

## Incremental indexing

Document and chunk IDs derive from canonical source identity and stable content boundaries.
A versioned manifest records source fingerprints, ordered chunk IDs, collection identity,
embedding identity and dimension, distance, and payload/identity schema versions.

Repeated unchanged ingestion performs no embedding or storage mutation. Changed documents
upsert their current chunks and delete their own stale chunks; sources absent from a batch
remain indexed. The manifest is atomically replaced after storage mutation. This design is
retryable, but it is not a transaction across Qdrant and the filesystem; interrupted-commit
recovery and concurrent writers remain deferred.

## GraphRAG isolation

`src/finrag/graph_rag` exposes a small async LOCAL/GLOBAL contract and lazily imports the
pinned Microsoft GraphRAG 3.3.0 package. The dependency, credential, workspace, artifacts,
and manual CLI are optional. `app/runtime.py` does not import or route to GraphRAG, so its
absence cannot affect default imports, indexing, retrieval, tests, or application startup.

GLOBAL has one recorded live capability smoke on the curated synthetic graph. LOCAL dispatch
is tested offline; a live LOCAL run still needs a prepared `entity_description` LanceDB
table. Neither result is a GraphRAG-vs-Hybrid quality comparison.

## Evaluation evidence

Test retrievers and mocked Microsoft calls verify contracts, edge cases, and deterministic
orchestration. They are not benchmark evidence. Measured quality comes only from versioned
fixtures and result artifacts. The current retrieval artifact records actual BM25 metrics;
Dense, Hybrid RRF, and reranked strategies remain explicitly `NOT MEASURED`.
