# Phase 1.1F — injected chat roles in the real runtime

## Objective
Bind the existing LangGraph LLM consumers to explicitly resolved ModelRoles, once
per application instance, and prove the runtime wiring with deterministic offline tests.

## Current-state observations
Inspection on 2026-10-09 at b111223: 193 tests pass with three existing warnings;
legacy mypy reports 21 errors in eight files. Previous reports record 26 Ruff
findings and 11 formatting findings. No files were changed before this inventory.

| Node | Previous acquisition / interface | Injected role |
| --- | --- | --- |
| generate | get_llm(0.1), ChatPromptTemplate pipe, AIMessage.content | generation |
| grade_documents | get_llm(0.0), with_structured_output(GradeDocuments).invoke(document text) | grading |
| transform_query | get_llm(0.2), direct invoke("Rewrite query: ...") | query_rewrite |

retrieve, web_search, and fact_check do not consume a chat model. fact_check is a
numeric substring notice that always reports no hallucination/useful; the unused
GradeHallucination schema does not imply a production model call. workflow.py
compiles a module-global graph; routes.py and test_runner.py consume it. The legacy
embedding path still depends on app/core/llm.py through vectorstore.py.

## Scope
app/graph, minimal application startup/chat transport wiring, the manual graph
entry point, explicit offline fixtures/tests, necessary environment/development
documentation, engineering boundaries, and this plan/report.

## Non-goals
No RAG corrections, prompt/criteria/edges/retry changes, provider routing, embedding
migration, retrieval-global redesign, broad FastAPI/lint/type cleanup, or later phase.

## Architecture decisions
- Small node factories bind a required BaseChatModel to the existing node bodies.
- build_finrag_graph(model_roles) has no registry/settings/credential work. Remove
  the import-time graph singleton. Keep the existing sequential graph-engine fallback
  with the same injected models; it is not a provider fallback.
- app/runtime.py owns Settings, a fresh empty registry with built-ins, role resolution,
  and graph composition. Explicitly supplied test registries bypass built-in setup.
- FastAPI lifespan retains a runtime per app instance; chat requests reuse its graph.
  Startup errors propagate before requests. No registry/model process-global cache.
- Preserve state fields and runtime values. Add precise partial-update typing as
  needed; do not change LangGraph state/retrieval semantics to satisfy tooling.
- Required Pydantic v2 imports replace the obsolete optional v1/mock schema shim in
  the touched nodes module, enabling a properly typed structured grading boundary.
- FINRAG role settings now select chat models. Settings defaults stay as implemented
  in 1.1A (all temperatures 0); document legacy temperature equivalents 0.1/0/0.2
  in the example. Credentials remain adapter-owned; no automatic dotenv loading.

## Implementation steps
1. Add typed model-bound factories; bind them in both existing graph engines.
2. Add runtime composition and instance-owned FastAPI lifespan; migrate graph callers.
3. Extend reusable test-only chat fixture with call recording and configurable grading.
4. Exercise real compiled generation/rewrite paths, factory lifetimes, startup/API,
   provider errors, and import boundaries with explicit local dependency doubles.
5. Update startup configuration/architecture docs; run checks and record evidence.

## Test strategy
Baseline uv run pytest: 193 passed, 3 warnings. Full offline socket/download/tracing
guards remain enabled. Three distinct models record role, message/structured inputs,
and schemas. Patch only the external web dependency for the real rewrite path.
Composition tests use explicit factories, plus stub registrations on fresh production
registries. Test provider failure identity/order and startup failure without fallback.
Run uv sync; uv lock --check --offline; uv run pytest; targeted Ruff check/format;
mypy src/finrag and changed application/tests (--follow-imports=silent for scoped
legacy diagnostics); full mypy app separately; git diff --check. No live API tests.

## Migration / compatibility
Builder now requires ModelRoles; callers use app.runtime.compose_runtime or explicit
role injection. API entry point remains app.main:app. FINRAG_ role metadata is now
authoritative for graph chat selection; old LLM_PROVIDER/model variables retain only
legacy helper/demo meaning. Use uv's --env-file explicitly if desired. Python 3.12
and installed finrag package are required; old requirements-only/Python 3.10 Docker
is not a supported migrated installation. No index/dependency/lock migration intended.
Underlying clients have no new close contract. Sequential registration's conflict
edge remains deferred; production always starts with an empty registry.

## Acceptance criteria
Every existing graph chat call uses its injected role; provider errors occur during
composition; roles resolve once, application instances are independent; real compiled
offline paths and HTTP startup/chat are exercised; targeted checks and full tests pass;
no new fake production path or intentional RAG correctness change.

## Progress checklist
- [x] Read required contracts/reports, inspect runtime, record inventory and baseline.
- [x] Implement injection and startup composition.
- [x] Add offline role/lifetime/failure/boundary regressions.
- [x] Update documentation and record final verification/debt.

## Final results
Completed within Phase 1.1F. See docs/phase-1.1f-baseline.md for the full file inventory,
18-point handoff, exact commands, migration concerns, and retained debt. Sync and
offline lock checks pass with no dependency changes. Final pytest: 212 passed, four
warnings. Targeted Ruff/format: all 33 selected files pass. Mypy src/finrag: 18 files
pass; scoped mypy including every changed Python file: 33 files pass. Diff checks pass.
Broad debt remains: 13 Ruff findings, six unformatted files, eight mypy errors in five
legacy files. No new production fake, global model cache, RAG correction, or later phase.

2026-10-09 verification notes: initial scoped mypy identified callable signature
inference at LangGraph.add_node; a named GraphNode callback protocol and explicit
existing input schema resolved it without ignores. Initial startup tests were blocked
by Windows asyncio's socketpair connecting through the suite guard. A test-only local
pair uses the original connect only for its freshly created loopback listener, verifies
peer identity, and keeps ordinary connects blocked (including localhost Ollama).
The first boundary-test run also identified the pre-existing DuckDuckGo import; its
allowlist now explicitly permits that web dependency. No production network allowance.
Final full suite: 212 passed, four warnings. The fourth warning is newly exercised
existing API schema Field(example=...) debt. Targeted Ruff/format and scoped mypy pass.
