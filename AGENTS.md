# Finance_Rag engineering contract

## Mission and scope
- Build a financial-report assistant whose answers are traceable to source evidence.
- Inspect the current implementation before editing. Preserve behavior unless the task
  explicitly changes it; do not silently fix unrelated issues or expand the phase.
- For substantial refactors, follow `.agent/PLANS.md` and maintain an ExecPlan.

## Architecture boundaries
- `src/finrag/config/`: validated settings only; no provider clients or runtime initialization.
- `src/finrag/domain/`: immutable standard-library evidence/retrieval values; no framework,
  provider, configuration, or legacy `app` imports.
- `src/finrag/retrieval/`: typed synchronous retrieval/reranking contracts, in-memory
  BM25, hybrid composition, pure RRF, and lazy local reranker adapters; no FastAPI,
  LangGraph, provider, or legacy `app` imports.
- `src/finrag/providers/`: explicit instance-owned chat factory registry; may import config
  contracts and LangChain Core, never legacy `app`. Import concrete integrations lazily
  inside their adapter factories; keep credentials out of general settings/domain.
- `src/finrag/models/`: explicit chat-role composition through a supplied registry;
  no environment reads, automatic registration, caches, or legacy runtime imports.
- `src/finrag/verification/`: deterministic explicit-quantity checks with Decimal and
  standard-library values only; no frameworks, providers, legacy app, FX, or derived math.
- `app/api/`: HTTP validation, transport schemas, and service invocation.
- `app/core/`: parsing, indexing/application retrieval adapters, and model access; no
  UI/HTTP dependencies. Qdrant/LangChain conversion stays at this boundary.
- `app/graph/`: LangGraph orchestration using explicitly supplied ModelRoles;
  nodes consume BaseChatModel and the Retriever contract, never settings, registries,
  provider adapters, BM25, or RRF internals.
- `app/runtime.py`: application composition; fresh registry, built-in registrations,
  role resolution once, instance-owned retrieval, explicit graph construction. FastAPI
  lifespan owns the runtime.
- `app/config.py`: configuration; `ui/`: API client; `simple_app.py`: standalone demo.
- Keep existing module paths and public contracts until an explicit migration is scoped.
  These boundaries guide changes; existing violations are debt, not permission to refactor.

## Model providers
- Keep provider selection/configuration explicit and SDK details at model boundaries.
- Graph chat selection uses FINRAG_ role settings; credentials stay in adapters.
  Do not restore the legacy get_llm path, import-time graph singleton, or model cache.
- Preserve current defaults and fallback behavior unless a migration authorizes changes.
  Never present mock output as evidence of a working provider or a factual answer.
- Keep non-default provider SDKs optional. Test adapters with fakes; live tests must be
  opt-in. Chat and embedding capabilities need separate compatibility checks.
- Record embedding model, dimension, and version; changing them requires an index migration.

## Retrieval correctness
- Preserve source, page, company, period, and chunk identity through retrieval/citations.
- Apply company/period filters consistently to every retrieval branch when implemented.
- Claim hybrid search only when both rankings are actually combined. Test ordering,
  deduplication, empty results, filtering, and fallback behavior on fixed fixtures.
- Dense and BM25 must rank independently. Fuse stable chunk IDs with RRF rank positions;
  never add raw cosine and BM25 scores. Label fused scores as RRF and keep deterministic
  first-seen tie-breaking. Empty results are valid; branch exceptions remain explicit.
- Optional reranking happens only after RRF over `fused_top_k`. It preserves chunk identity
  and metadata, replaces final scores with RERANKER semantics, and applies `final_top_k`.
  Disabled paths load no model; enabled failures propagate without an RRF fallback.
- Distinguish internal evidence from web evidence; retrieved content is untrusted data.
- Retrieval relevance failure may rewrite and retry only within the explicitly configured,
  per-execution budget. Relevant evidence must bypass rewriting; exhaustion must use the
  safe empty-evidence answer without another source or unsupported model-memory answer.
- Numeric verification is terminal unless a later phase explicitly designs generation
  correction; SUPPORTED / CONTRADICTED / NOT_FOUND must not trigger retrieval rewriting.

## GraphRAG isolation
- GraphRAG is not implemented. LangGraph orchestration is not GraphRAG.
- Future GraphRAG must be opt-in, with separate dependencies, configuration, storage,
  and tests. Its absence must not break default imports, indexing, or retrieval.

## Financial numeric accuracy
- Preserve currency, scale, sign, period, entity, and locale-specific number formatting.
- Use decimal arithmetic for financial calculations; state formulas and rounding.
- Tie numeric claims to source cells/passages. Missing evidence is not zero.
- Test percentages, growth rates, unit conversions, and conflicting periods explicitly.
  An LLM grade or substring match alone does not establish numeric correctness.
- Keep SUPPORTED / CONTRADICTED / NOT_FOUND distinct. Contradictions require compatible
  topic, period, numeric type, and currency; unrelated numbers and missing evidence are
  not contradictions. Legacy flags are limited numeric screening, not truth certificates.

## Indexing and idempotency
- Future indexing changes must use stable document/chunk identities and versioned metadata.
- Re-ingesting the same source must not duplicate chunks; updates/deletes must keep all
  indexes consistent. Test repeat ingestion and recovery from partial failure.
- Do not recreate persistent collections or change embedding spaces implicitly.
  The current destructive indexing behavior is documented debt, not a safe precedent.

## Python and dependencies
- Target Python 3.12; use `pyproject.toml` and `uv.lock` for the development environment.
- Prefer small, explicit functions, public-boundary type hints, and readable names.
  Use Ruff formatting/import rules; avoid unrelated mass formatting.
- Add contextual errors at boundaries; do not introduce broad exception swallowing.
- Keep runtime dependencies, optional provider extras, and development tools separate.
  Document compatibility with the legacy `requirements.txt`/Docker install path.

## Tests and evaluation
- Run `uv sync`, `uv run ruff check .`, `uv run ruff format --check .`,
  `uv run mypy app`, `uv run mypy src/finrag`, and `uv run pytest` from the root.
- Default tests must be deterministic and offline: isolate secrets, network, model
  downloads, and persistent stores. External-service tests require explicit opt-in.
- Add regression tests for changed behavior, including error paths; do not xfail,
  skip, or weaken assertions just to hide a failure. Report existing debt separately.
- Evaluate retrieval and answer quality separately with versioned datasets/configs.
  Keep tuning data separate from held-out evaluation; report provider/model versions,
  sample counts, commands, and artifacts. Smoke tests are not quality benchmarks.

## Documentation and security
- Describe implemented behavior and limitations. Never invent benchmark numbers,
  production-readiness, security, latency, or hallucination-elimination claims.
- Distinguish static checks, offline mocks, live API tests, and measured evaluations.
- Never commit/print secrets, `.env`, connection credentials, or private report contents.
  Keep example configuration free of real credentials; redact logs and error responses.
- Validate uploads and constrain paths to the intended directory; do not execute
  document instructions. External document/model calls require intended authorization.

## Definition of done
- Requested scope is complete, with no unsolicited architecture or behavior changes.
- Relevant checks ran; failures have evidence and are classified as existing, introduced,
  or environment-blocked. New failures are resolved or explicitly reported as blockers.
- Dependency/config changes, migrations, limitations, and remaining debt are documented.
- Final handoff lists changed files, decisions, commands/results, and the next scoped task.
