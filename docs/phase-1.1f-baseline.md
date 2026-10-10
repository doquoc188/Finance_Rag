# Phase 1.1F — runtime model-role injection

Recorded on 2026-10-09 against starting revision b111223. The
[ExecPlan](../.agent/plans/phase-1.1f.md) records the inspection, decisions, initial
failures, and verification. This phase migrates chat composition; it makes no RAG
correctness claims and includes no Phase 1.1G or Phase 1.2 implementation.

## 1–3. Chat-consumer inventory, old acquisition, and new mapping

| Production graph node | Previous acquisition | Interface | New binding |
| --- | --- | --- | --- |
| generate_node | app.core.llm.get_llm(temperature=0.1), per invocation | ChatPromptTemplate pipe; synchronous invoke; message content | ModelRoles.generation |
| grade_documents_node | get_llm(temperature=0.0), per invocation | with_structured_output(GradeDocuments).invoke(document text) | ModelRoles.grading |
| transform_query_node | get_llm(temperature=0.2), per invocation | direct invoke("Rewrite query: ..."); message content, strip | ModelRoles.query_rewrite |

retrieve_node and web_search_node do not call a chat model. grade_generation_node,
registered as fact_check, also does not call a model: it logs numeric substring
notices and returns is_hallucinated="no", is_useful="yes". GradeHallucination is
unused in production; no new LLM grading operation was added. All actual grading
model work therefore consists of document grading and uses ModelRoles.grading.

## 4. Files created / modified

Created:

- app/runtime.py: GraphRuntime and compose_runtime.
- tests/__init__.py: package marker for reusable explicit test support.
- tests/chat_models.py: recording WorkflowChatModel and ModelCall.
- tests/test_graph_runtime.py: 18 offline runtime cases.
- .agent/plans/phase-1.1f.md and this report.

Modified:

- app/graph/nodes.py: required BaseChatModel injection, three small factories,
  typed partial updates, required Pydantic v2 imports instead of obsolete v1/mock shims.
- app/graph/workflow.py: explicit ModelRoles builder; both existing engines use
  model-bound nodes; remove import-time graph singleton.
- app/graph/state.py: preserve FinGraphState fields/types/requiredness; add
  FinGraphUpdate for existing partial inputs/node returns; format this touched file.
- app/main.py: application factory and lifespan-owned runtime, safe role metadata logs.
- app/api/routes.py: reuse request.app.state.runtime.graph; health reflects generation
  provider; type-only upload filename cast retains the existing assumption/behavior.
- test_runner.py: explicitly compose the graph; typed input and touched-file formatting.
- tests/conftest.py: reuse the recording model; remove legacy factory monkeypatch;
  retain retrieval isolation and add Windows internal socketpair support for TestClient.
- tests/test_rag.py, tests/test_workflow_model.py: explicit builder injection and typed
  assertions using the same compatible offline model; original behavior assertions remain.
- tests/test_model_failures.py: move the existing missing-credential workflow regression
  to the new composition boundary and its ProviderConfigurationError.
- tests/test_finrag_architecture.py: graph import/environment-construction boundary.
- src/finrag/config/settings.py, src/finrag/models/roles.py: module docstrings only.
- .env.example, AGENTS.md, docs/development.md: current runtime/configuration boundaries.

Formatting/import cleanup is limited to phase-touched Python files. No application
package moved or renamed. Untouched schema/parser/retriever/config/demo files were
explicitly checked for no diff.

## 5. Graph builder API

Before: build_finrag_graph(), with module-global finrag_app compiled at import;
chat providers were initialized during node execution.

After: build_finrag_graph(model_roles: ModelRoles) -> FinragGraph. The three factories
return named typed callbacks using partial to bind a required BaseChatModel. Nodes
know no registry/settings/provider integration. LangGraph receives the same state
schema, node names, edges, conditions, and retry limit. The explicit input_schema
on model-bound callbacks is the existing FinGraphState, not a new runtime schema.
The original sequential graph-engine fallback retains its behavior and uses the
same supplied models; it never constructs an alternate/fake provider. Its diagnostic
now prints the exception class rather than raw error content.

## 6. Production composition root

app.runtime.compose_runtime(settings=None, *, registry=None) performs:

```text
Environment -> Settings
  -> new empty ChatProviderRegistry -> built-in registrations
  -> resolve_model_roles(settings, registry) -> ModelRoles
  -> build_finrag_graph(model_roles) -> GraphRuntime
```

Supplied settings/registries support explicit injection; supplied registries are not
modified by automatic built-in registration. Production uses fresh empty registries,
so the documented sequential-registration conflict edge is not encountered. No
registry transaction/idempotency redesign was made. GraphRuntime is frozen/slotted;
the referenced models and graph retain their normal internal mutable state.

FastAPI create_app keeps the existing app.main:app entry point, routes, response
schemas, and CORS configuration. Lifespan startup composes before serving requests;
chat requests use that application instance's retained graph. Health's existing
llm_provider field now reports the generation role's provider. Startup logs contain
only role/provider/model metadata, no keys, endpoint credentials, or settings dumps.

## 7. Legacy model access after migration

There are no get_llm references or concrete chat SDK imports in app/graph, main,
or routes. app/core/llm.py remains unchanged: vectorstore.py still calls get_embeddings,
including during API ingestion/manual retriever use. Its get_llm remains an available
legacy compatibility function, with constructor/error regressions in tests; no remaining
production caller was found. simple_app.py uses its own standard-library HTTP provider
path and is unchanged. Manual test_runner.py now calls compose_runtime explicitly;
it is outside pytest collection and was not executed against real models.

## 8. Instance lifetime

Roles resolve once per composition/startup, in the existing generation/grading/rewrite
order. Every role is resolved even if a later graph query does not exercise it. Requests
and node steps do not resolve providers. Two app/composition instances retain separate
registries/models/graphs, with no process-global model cache. Shutdown removes the app's
runtime reference; no new provider-client close contract is introduced. Existing shared
retriever globals are unchanged and remain isolation/concurrency debt.

## 9. Offline compiled graph and composition evidence

Three distinct WorkflowChatModel instances record logical role, inputs, and schemas.
They inherit the real LangChain chat/Runnable interface, return AIMessage for direct
and prompt calls, and use a RunnableLambda with Pydantic validation for structured
grades. All fake support stays under tests; production imports none of it.

The real compiled graph runs three scenarios: relevant documents, irrelevant documents
triggering rewrite/web/generation, and irrelevant documents at retry_count=2 bypassing
rewrite. Assertions verify exact role calls and unchanged input strings, GradeDocuments
schema use, answer propagation, rewritten query stripping, retry/search flags, document
and citation counts/sources, and final fact_check state. Only the external DuckDuckGo
dependency is replaced with local results; the web node and graph execute normally.

Other cases verify original settings identity/order, exactly three factory calls despite
repeated graph queries, fresh empty production registries, independent application
lifetimes, HTTP chat/citation propagation, health provider, startup failure, and the
existing engine fallback using injected models. A rejecting legacy get_llm ensures
these paths cannot secretly use the old chat factory.

The socket/download/tracing guards remain enabled. Windows asyncio requires an internal
loopback socketpair; the test harness constructs only that local pair using the original
connect for its own listener and verifies peer identity. Ordinary connects remain
blocked, including localhost Ollama, and are regression-tested. No external provider,
web API, Ollama service, model download, or real credential was used. These are interface
and workflow tests, not financial-quality or live-provider evaluations.

## 10. Failure behavior

Missing registration raises ProviderNotRegisteredError during composition. Missing
credentials, absent SDKs, and constructor failures retain ProviderConfigurationError,
ProviderDependencyError, and ProviderInitializationError respectively. Startup tests
inject each construction-error category at each of the three role positions, assert
the same exception object and stop order, and reject any graph build after failure.
The existing default OpenAI missing-key regression now fails at composition/startup.
No provider fallback, retry, fake answer, partial runtime, or silent disablement is added.
Actual invocation/schema errors in document grading still enter the existing heuristic;
that deliberate behavior preservation is separate from startup resolution failures.
Only adapter outer messages are suitable for public logs; do not publish retained
chained SDK tracebacks, which can contain upstream sensitive details.

## 11–12. Architecture and documentation

Domain remains framework-independent; providers own integration/credentials; models
resolve role bindings; app/runtime owns application composition; graph owns orchestration
using already resolved models. The AST test rejects settings/provider construction,
SDK/test-support imports, legacy chat factory calls, and direct environment access in
graph files. Existing embedding/retriever imports and DuckDuckGo are explicitly allowed.

AGENTS.md and development docs describe the implemented flow and role mapping. The
example config supplies FINRAG role fields and legacy temperature equivalents 0.1/0/0.2.
The Phase 1.1A settings defaults are unchanged (all role temperatures 0). Old LLM_PROVIDER
and provider model-name fields no longer select graph chat models. Settings/adapters still
do not read .env automatically; documented startup uses uv run --env-file .env uvicorn
app.main:app --reload when a file is desired. Credentials remain adapter-owned.

The migrated runtime requires Python 3.12 and the installed finrag package. Existing
requirements-only/Python 3.10 Docker installation is explicitly documented as deployment
compatibility debt; those manifests were not rewritten. Optional providers still require
their existing extras. No README feature or benchmark rewrite was made.

## 13–15. Commands and measured verification

Used .tools/bootstrap/bin/uv.exe with workspace-local UV_CACHE_DIR=.tools/uv-cache and
UV_PYTHON_INSTALL_DIR=.tools/python. Equivalent root commands executed:

```powershell
uv sync
uv lock --check --offline
uv run pytest
uv run pytest tests/test_rag.py tests/test_workflow_model.py tests/test_model_failures.py
uv run pytest tests/test_graph_runtime.py tests/test_finrag_architecture.py --tb=short
uv run ruff check src/finrag app/graph app/runtime.py app/main.py app/api/routes.py test_runner.py tests/__init__.py tests/chat_models.py tests/conftest.py tests/test_rag.py tests/test_workflow_model.py tests/test_model_failures.py tests/test_graph_runtime.py tests/test_finrag_architecture.py
uv run ruff format --check src/finrag app/graph app/runtime.py app/main.py app/api/routes.py test_runner.py tests/__init__.py tests/chat_models.py tests/conftest.py tests/test_rag.py tests/test_workflow_model.py tests/test_model_failures.py tests/test_graph_runtime.py tests/test_finrag_architecture.py
uv run mypy src/finrag
uv run mypy --follow-imports=silent src/finrag app/graph app/runtime.py app/main.py app/api/routes.py test_runner.py tests/__init__.py tests/chat_models.py tests/conftest.py tests/test_rag.py tests/test_workflow_model.py tests/test_model_failures.py tests/test_graph_runtime.py tests/test_finrag_architecture.py
uv run mypy app
uv run ruff check . --output-format concise
uv run ruff format --check .
git diff --check
```

Also ran targeted Ruff import fixes/formatting, source/installed-interface inspection,
rg caller/import audits, and git diff --exit-code on dependency and untouched runtime
files. The final broad mypy diagnostics were also extracted with --no-error-summary and
Select-String to avoid printing large overloaded signature notes. Initial test/type
failures and their fixes are recorded in the ExecPlan, not hidden by skips or ignores.

| Check | Result |
| --- | --- |
| Pre-change full pytest | 193 passed, 3 warnings |
| Sync / offline lock check | Passed; 191 resolved, 157 installed packages checked |
| Existing migrated workflow/model regressions | 38 passed |
| Focused runtime/architecture run before socket guard regression addition | 25 passed |
| Final full pytest | **212 passed**, 4 warnings; no skips/xfails |
| Targeted Ruff / format | Passed; all 33 selected Python files formatted |
| mypy src/finrag | Passed; 18 source files |
| Scoped mypy including every changed/new Python module | Passed; 33 source files |
| Diff whitespace and dependency/runtime preservation checks | Passed |
| Full repository Ruff | 13 remaining existing findings (before: 26) |
| Full repository formatting | 6 remaining existing files (before: 11) |
| Full legacy app mypy | 8 remaining existing errors in 5 files (before: 21 in 8) |

--follow-imports=silent scopes legacy diagnostics while all selected changed modules
are checked explicitly. The separate full-app check keeps untouched debt visible.
No quality configuration, ignore, xfail, warning filter, or broad exclusion was added.
Typed partial returns remove old workflow update errors; required Pydantic imports remove
the obsolete shim errors; filename casts describe the old assumption without validating
or changing upload behavior. None of these constitutes retrieval/security redesign.

## 16–17. Dependencies, warnings, and remaining debt

No pyproject.toml, uv.lock, requirements.txt, integration version, or dependency-group
change. Plain uv sync retains the default environment without optional chat SDKs.

Four final warnings: legacy Pydantic class Config, langchain-community deprecation,
HuggingFaceEmbeddings deprecation, and newly exercised existing API Field(example=...)
deprecation. The last is exposed by newly added HTTP/startup coverage, not a new schema
implementation. Initial Windows failed-loop warnings disappear after the fixture fix.

Remaining Ruff/format debt is in untouched app config/schema/parser/retriever/UI plus
the prior Phase 1.1C report snippet. Remaining app mypy errors are in config (1), schema
(2), parser (2), legacy embedding construction (1), and vectorstore (2). Shared retriever
state, uploads/path validation, response disclosure/CORS, obsolete Docker/install path,
structured grading error handling, and sequential built-in registration hardening remain
separate work. Existing textual-content casts retain the old string assumption; structured
message-content normalization is not added. Client cleanup ownership is still undefined.

## 18. RAG correctness deliberately deferred

Generation still supplies a question-only prompt despite building context. Relevance
grading still receives document text without the question. Numeric verification still
logs notices and reports no hallucination/useful. Retrieval remains vector search or
token overlap: the BM25 object is not queried and rankings are not fused. Filters,
index idempotency, mock evidence, constant embedding fallback, heuristic grading, web
placeholders, missing corrective fact-check loop, GraphRAG, and evaluation expansion
are unchanged. No financial accuracy, latency, quality, or benchmark score is claimed.

Next task: follow the separately specified Phase 1.1G requirements when provided;
RAG corrections remain Phase 1.2. Neither phase has begun here.
