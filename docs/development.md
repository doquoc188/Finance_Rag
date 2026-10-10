# Development foundation and runtime composition

This phase adds engineering contracts and tooling. Application modules, existing test
assertions, provider defaults, and the legacy runtime installation path are preserved.
It does not implement a retrieval refactor, GraphRAG, or numeric-validation improvements.

Phase 1.0.1 adds an explicit compatible chat-model fixture for workflow tests and
strengthens their assertions. Application source remains unchanged. See the
[green test baseline](phase-1.0.1-baseline.md) for the current test results.

Phase 1.0.2 removes production chat-model and canned-answer fallbacks. Missing model
configuration or provider failures now raise explicit errors; successful provider
selection and constructor arguments are preserved. See the
[Phase 1.0.2 report](phase-1.0.2-baseline.md) for failure semantics and current results.

Phase 1.1A adds `src/finrag` configuration and immutable domain contracts without
migrating the application. See the [foundation report](phase-1.1a-baseline.md) for
environment names, domain constraints, packaging changes, and verification.

Phase 1.1B adds explicit chat-provider factory registration. Factories return LangChain
Core `BaseChatModel` to preserve Runnable composition, message output, and the standard
structured-output interface. Registries belong to individual callers, with no global
state or automatic provider discovery. Concrete adapters are deferred to Phase 1.1C+;
the legacy runtime still uses its existing model construction. See the
[provider contract report](phase-1.1b-baseline.md) for semantics and verification.

Phase 1.1C adds `create_openai_chat_model` and `create_groq_chat_model` in their
respective `finrag.providers.openai` / `.groq` modules. Import and register factories
explicitly on your own registry; there is no automatic registration or role resolver.
Factories read `OPENAI_API_KEY` / `GROQ_API_KEY` only when called, do not load `.env`,
and keep credentials out of model-role settings. Integrations import lazily; Groq
still requires `uv sync --extra groq`. Configuration, dependency, and initialization
failures raise distinct errors without fallback. Only outer messages are safe for
clients; retained exception causes may contain SDK-provided secrets. Both factories
return the integration's `BaseChatModel` unchanged, preserving Runnable and standard
structured-output interfaces. The legacy runtime remains unmigrated. See the
[adapter report](phase-1.1c-baseline.md) for verified versions, arguments, and results.

Phase 1.1D completes the five explicit factories: OpenAI, Anthropic, and Google Gemini
for cloud APIs; Groq for its inference service; Ollama for a local server. Canonical
registry names are `openai`, `anthropic`, `google`, `groq`, and `ollama`. Import the
factories from their matching `finrag.providers.<name>` modules and register them on
your own registry. Only OpenAI remains in the default dependency path. Install other
integrations with `uv sync --extra groq`, `--extra anthropic`, `--extra google`, or
`--extra ollama`; flags can be combined.

Anthropic reads `ANTHROPIC_API_KEY`; Google reads `GOOGLE_API_KEY` and explicitly uses
the Gemini Developer API backend. Ollama requires no key and reads `OLLAMA_BASE_URL`,
defaulting to `http://127.0.0.1:11434` when unset. A blank explicit URL is rejected.
Construction does not check provider/server availability, list or pull models, or
invoke them; successful construction does not prove an Ollama server is available.
Remote availability is only established when the caller uses the model. No runtime
migration or model-role resolver is included. See the
[five-provider report](phase-1.1d-baseline.md) for exact interfaces and verification.

Phase 1.1E adds explicit model-role composition:
`Settings -> resolve_model_roles -> ChatProviderRegistry -> factory -> BaseChatModel`.
Call `register_builtin_chat_providers` from `finrag.providers.builtin` on a registry
you own, then pass validated settings and that registry to
`finrag.models.resolve_model_roles`. Registration constructs no models or SDK clients.
The helper fails on duplicates and never overwrites; sequential registrations before
a later conflict remain, so it is neither idempotent nor transactional.

`ModelRoles` has frozen `generation`, `grading`, and `query_rewrite` bindings matching
the existing settings names. Each role may choose its own provider/model/temperature.
Resolution uses that order and returns only after all succeed; original provider errors
propagate without fallback. The resolver reads no environment and caches nothing;
credentials stay inside adapters. Underlying LangChain models remain stateful, and
caller-owned factories control instance sharing/lifetime. The legacy application still
uses its existing construction path. See the
[role-composition report](phase-1.1e-baseline.md) for behavior and verification.

## Current graph runtime (Phase 1.1F)

The Phase 1.1A–E paragraphs above describe those historical increments. The real
application now consumes their contracts:

```text
Process environment -> finrag.config.Settings
  -> fresh ChatProviderRegistry -> register_builtin_chat_providers
  -> resolve_model_roles -> ModelRoles
  -> build_finrag_graph(model_roles) -> LangGraph
```

`app/runtime.py:compose_runtime` is the production composition root. An explicitly
supplied registry is used unchanged (useful for offline tests); production defaults
always register built-ins on a new empty registry. The sequential registration conflict
edge remains a hardening item for callers reusing registries, not a production startup
case. `app/main.py:create_app` composes once at lifespan startup and stores GraphRuntime
on that application instance. Chat requests reuse its graph/models; a separate app gets
separate composition. There is no import-time graph, model cache, or global registry.
Lifespan shutdown releases the app's runtime reference; no new SDK close contract is added.

| Graph node | Model role | Existing model interface |
| --- | --- | --- |
| generate | generation | ChatPromptTemplate pipe, message content |
| grade_documents | grading | with_structured_output(GradeDocuments), invoke(document text) |
| transform_query | query_rewrite | invoke("Rewrite query: ...") |

retrieve, web_search, and fact_check use no chat model. fact_check retains its existing
numeric notice logic; the unused GradeHallucination schema is not an implemented LLM
grader. Node factories accept only BaseChatModel. Graph construction imports no settings
or provider adapters. The existing graph-engine fallback also uses the supplied models;
it never resolves a replacement provider. Startup composition errors propagate before
graph construction: ProviderNotRegisteredError, ProviderConfigurationError,
ProviderDependencyError, or ProviderInitializationError. The existing document-grading
heuristic on invocation/schema failure remains unchanged and is separate technical debt.

Chat selection now uses `FINRAG_GENERATION__*`, `FINRAG_GRADING__*`, and
`FINRAG_QUERY_REWRITE__*`. Old `LLM_PROVIDER`, `OPENAI_MODEL`, `GROQ_MODEL`, and
`OLLAMA_MODEL` no longer select graph chat models. Role settings retain Phase 1.1A
defaults (OpenAI/gpt-4o-mini, all temperatures 0). `.env.example` explicitly configures
the old graph temperatures 0.1/0/0.2. Supply credentials in the process environment:
OPENAI_API_KEY, GROQ_API_KEY, ANTHROPIC_API_KEY, GOOGLE_API_KEY, or OLLAMA_BASE_URL
as appropriate; optional SDK extras still need explicit installation. No connection or
model availability is established merely by constructing clients.

The new settings/adapters do not automatically read `.env`. To use a local file, load
it explicitly at the command boundary, supplying real credentials only locally:

```powershell
uv run --env-file .env uvicorn app.main:app --reload
```

The existing application name/embedding/storage settings still use legacy app.config;
its existing dotenv behavior is unchanged. `app/core/llm.py` remains for get_embeddings
and the compatibility get_llm API, but graph code has no legacy chat-factory reference.
The standalone simple_app.py remains independent. The manual test_runner.py now composes
its graph explicitly and requires valid runtime configuration; it is not an offline test.

Startup logs show only role/provider/model metadata. The health response's existing
llm_provider field now identifies the generation role. Do not publish chained SDK
tracebacks; only existing adapter outer messages are credential-safe. Existing API error
and health disclosure debt is not redesigned here.

Implemented: runtime model-role injection, explicit instance ownership, and offline
compiled graph/composition/startup tests. Still pending: RAG correctness, true hybrid
retrieval, GraphRAG, and evaluation expansion. See the
[Phase 1.1F report](phase-1.1f-baseline.md) for measured checks and deferred defects.

## Python and installation

Use Python 3.12 and [uv](https://docs.astral.sh/uv/getting-started/installation/).
From the repository root:

```powershell
uv sync
uv run ruff check .
uv run ruff format --check .
uv run mypy app
uv run pytest
```

`.python-version` selects 3.12. `requires-python` deliberately limits this baseline to
3.12; newer interpreters are not yet validated. `uv.lock` records the resolved runtime,
optional, and development dependencies. Use `uv sync --locked` in automation. The
legacy application still runs directly from the repository. Since Phase 1.1A,
setuptools packages only `src/finrag`, and `uv sync` installs it editably. The existing
`app/` modules remain at their original paths and now import the installed package
for chat-role composition.

`uv run uvicorn app.main:app --reload` and `uv run streamlit run ui/streamlit_app.py`
retain the existing entry points. Running the application can use configured credentials
and services; the offline restrictions below apply to tests only.

## Dependency decisions

- Runtime requirements retain the existing lower bounds. OpenAI stays in the default
  set because `LLM_PROVIDER` defaults to `openai`, and embeddings also consult its key.
- Groq is lazy-imported and optional: use `uv sync --extra groq` when selecting it.
  Plain `uv sync` omits all optional chat integrations. The new Ollama factory uses
  the optional `langchain-ollama` extra; the legacy app still uses its existing community
  adapter only through the retained compatibility helper. The migrated graph uses
  the new factory. The standalone demo uses the standard library for its providers.
- Anthropic/Google/Ollama extras have lower bounds matching the inspected constructor
  interfaces; `uv.lock` fixes exact versions. Google GenAI requires Tenacity below 9.2,
  so this phase changes the shared lock from 9.2.1 to 9.1.4; other existing versions
  are unchanged. The full offline suite verifies compatibility, not live model behavior.
- Future provider SDKs belong in `project.optional-dependencies`. A future GraphRAG
  extra must be separate; no placeholder SDK or graph dependency is installed now.
- `requests` and `typing-extensions` become explicit dependencies because existing
  modules import them directly. Other existing runtime packages remain, including
  currently unused packages; removing them is a separate dependency audit.
- Ruff, mypy, pytest, pytest-cov, and pre-commit live in the default `dev` dependency
  group. This follows [uv's dependency separation](https://docs.astral.sh/uv/concepts/projects/dependencies/).
- `requirements.txt` remains unchanged for compatibility with current pip/Docker
  workflows, including its Groq and pytest entries. It is a legacy install path, not
  a generated export of the new lock. Docker still targets Python 3.10. Migrating it
  and retiring the duplicate dependency manifest requires a separately scoped change.
  The migrated graph requires the installed finrag package and Python 3.12; use uv sync
  or an editable project installation. The requirements-only/Python 3.10 Docker path
  cannot run this migrated runtime and remains an explicit deployment compatibility debt.
- The old requirements had no upper bounds or lockfile. The new lock is a reproducible
  development baseline, not proof that every permitted library version is compatible.

## Quality gates and test isolation

Ruff checks correctness, unused names, and import order, with a 100-column formatter.
Mypy understands the existing namespace-package layout and checks untyped bodies as
well as annotations. Compact diagnostics avoid a Windows encoding crash when pretty
output includes Vietnamese source lines. There are no blanket application exclusions
or error suppressions. Missing-import exceptions cover only optional integration
namespaces; installed-SDK checks verify their real types separately. Anthropic 1.7.5
has one documented constructor typing exception for falsely required timeout/stop
arguments; see the phase report. Errors in `app.core.llm` remain visible.
Existing lint, formatting, and type debt intentionally remains visible.

Pytest collects `tests/` and adds the repository root to its import path. The root
`test_runner.py` is a manual script with import-time execution and console mutation;
it is intentionally outside collection. No application modules were moved to achieve this.

`tests/conftest.py` sets test credentials/provider settings and selects in-memory Qdrant
before collection. It also sets a valid `DEBUG` value, disables tracing, model downloads,
and socket connections, and uses a temporary Hugging Face cache. The workflow test
explicitly requests `workflow_chat_model`, a real LangChain fake chat-model interface
with fixed test answers, call records, and schema-validated relevance grades, defined
in tests/chat_models.py. Tests supply ModelRoles explicitly and reset shared retriever
state for isolation. The graph and its nodes still execute. The real rewrite scenario
replaces only the external web-search dependency with local results. TestClient exercises
lifespan startup and HTTP chat using explicit local factories. On Windows, the fixture
allows only the internally created loopback socketpair needed by asyncio's wakeup pipe;
normal socket connections, including localhost Ollama connections, remain blocked and
are regression-tested. No provider API or external web request is made.
The retriever smoke test continues to exercise existing fallback paths. These tests
do not validate real embeddings, provider APIs, retrieval quality, or persistent
storage. Tests are neither skipped nor converted to expected failures.
The `integration` marker is reserved, not a network opt-in switch: future live tests
need an explicit separate harness and authorization.

Optional coverage and Git hooks:

```powershell
uv run pytest --cov=app --cov-report=term-missing
uv run pre-commit validate-config
uv run pre-commit install
uv run pre-commit run --all-files
```

Hook installation is an explicit local choice. Hooks use `uv run --locked` and check
only staged Python files; full commands above check the repository. They report
violations without editing files. Existing debt also makes all-file hooks fail.
There is no invented coverage target or claimed evaluation score.

## Existing technical debt observed by source inspection

These are observations, not fixes or acceptance claims:

- `app/core/vectorstore.py` returns vector results when available, otherwise token
  overlap. A BM25 object is constructed but never queried; no rank fusion is applied.
- `app/api/routes.py` accepts company/year but does not pass them into chat retrieval.
  Indexing replaces the current documents and sets `force_recreate=True`; stable IDs,
  incremental updates, concurrency isolation, and idempotent ingestion are absent.
- `app/graph/nodes.py` builds context but its generation prompt contains only the
  question. Document grading does not include the question. The numeric guard logs
  unmatched numbers but always returns no hallucination and a useful answer.
- Phase 1.0.2 deletes `SimpleFakeLLM` and the other canned chat-answer fallbacks.
  Constant embedding fallback, mock retrieval evidence, heuristic grading, and web
  placeholders remain separate retrieval/workflow debt; they are not model-quality evidence.
- LangGraph does not loop back after fact checking. GraphRAG is not implemented.
- PDF failure falls back to decoding bytes as text, not pypdf. Zero-valued table cells
  can become empty strings; character splitting does not preserve table boundaries.
- Upload paths use a client-supplied filename; error/health responses can expose internal
  details, CORS is permissive, and shared retriever state has no tenant isolation.
  These findings require a separate security/ingestion task before external deployment.
- README/UI contain unsupported production, numeric-accuracy, hybrid-search, streaming,
  reranker, and benchmark claims. No supporting evaluation artifacts were found.
  The historical claims must not be cited as validated results.

See the [Phase 1.0 verification baseline](phase-1.0-baseline.md) for the original results,
the [Phase 1.0.1 baseline](phase-1.0.1-baseline.md) for its green offline test suite,
and the [Phase 1.0.2 baseline](phase-1.0.2-baseline.md) for the current failure semantics.
The [Phase 1.1A baseline](phase-1.1a-baseline.md) covers the new configuration/domain code.
Future agents should follow [AGENTS.md](../AGENTS.md) and the
[ExecPlan contract](../.agent/PLANS.md).
