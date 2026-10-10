# Phase 1.1E - explicit model-role composition

Recorded on 2026-10-09 against starting revision `72927e6`. The
[ExecPlan](../.agent/plans/phase-1.1e.md) records inspection, decisions and baseline.

## Files and location

Created:
- `src/finrag/models/__init__.py`
- `src/finrag/models/roles.py`
- `src/finrag/providers/builtin.py`
- `tests/test_finrag_models.py`
- `tests/test_finrag_builtin.py`
- `.agent/plans/phase-1.1e.md`
- `docs/phase-1.1e-baseline.md`

Updated `AGENTS.md`, `docs/development.md`, and `tests/test_finrag_architecture.py`.
No provider/config/domain contract, adapter, dependency or legacy runtime changes.
Existing packaging discovers the new models subpackage without metadata changes.

## Composition API and role container

`finrag.models` is the composition boundary. Public exports:

- `ModelRoles`: frozen, slotted dataclass with generation, grading, query_rewrite
  fields, each precisely typed as LangChain Core BaseChatModel. Names match Settings.
- `resolve_model_roles(settings: Settings, registry: ChatProviderRegistry) -> ModelRoles`.

Architecture flow:

```text
Settings
  -> resolve_model_roles
  -> caller-owned ChatProviderRegistry.get(role.provider)
  -> selected ChatProviderFactory(role settings)
  -> BaseChatModel
  -> complete ModelRoles
```

Resolution always constructs generation, then grading, then query_rewrite. Each lookup
uses that role's provider and passes that exact original ChatModelSettings object to
the selected factory. Different provider/model/temperature values are independent;
embeddings are not included. The resolver reads no environment, dotenv, credentials,
or files, performs no registration, and owns no global state or cache.

The result is created only after all three factory calls succeed. Missing-registration
and provider configuration/dependency/initialization errors propagate unchanged with
their causes; resolution stops immediately without returning partial roles or choosing
a substitute. Models created before a later failure may exist locally; no resource
cleanup contract is added. Caller/factory owns model lifetime and instance sharing.
Every invocation calls the factories afresh, even if role settings are identical.

Frozen ModelRoles prevents changing role bindings; it does not freeze LangChain's
internal model state. The same model may intentionally be returned by caller-supplied
factories; the resolver neither caches nor clones objects. BaseChatModel's Runnable,
message and structured-output interface is retained; actual schema capabilities remain
the provider/model's responsibility.

## Explicit built-in registration

`finrag.providers.builtin.register_builtin_chat_providers(registry) -> None` registers
exactly openai, groq, anthropic, google, ollama in that order on the supplied registry.
It uses existing concrete factory identities and constructs no model or SDK client.
The helper is explicitly imported; generic provider exports remain unchanged.

The duplicate decision is option A: repeated calls raise existing DuplicateProviderError,
starting with openai, and never overwrite. Registration follows ordinary sequential
registry semantics, not a transaction: if a later name conflicts, earlier successful
entries remain and subsequent names are not registered. Existing conflicting factories
stay intact. No batch API, rollback, idempotency shortcut, singleton or service locator
is introduced. A separate registry remains empty until the caller configures it.

Caller-owned usage (not executed against live providers during verification):

```python
from finrag.config import Settings
from finrag.models import resolve_model_roles
from finrag.providers import ChatProviderRegistry
from finrag.providers.builtin import register_builtin_chat_providers

settings = Settings()  # Or supply independently configured chat roles.
registry = ChatProviderRegistry()
register_builtin_chat_providers(registry)  # No credential/SDK construction.
roles = resolve_model_roles(settings, registry)  # Explicit model construction.
```

Instantiating Settings still loads its defined configuration environment; the resolver
only consumes that validated object. Credentials/dependencies are checked by concrete
adapters when their factories are invoked. Registration itself requires no credentials,
optional integration package or provider connectivity.

## Offline tests and evidence

Added 15 resolver cases, three registration cases, and one model-package boundary
case (19 additional cases total). The focused run includes seven architecture cases.

- Three distinct fake factories receive exact original configs and produce independent
  FakeListChatModel instances. Assertions cover provider selection, model identifiers,
  role-config identity/order and immutable bindings.
- Each resolved role separately composes with ChatPromptTemplate, invokes locally,
  and returns an AIMessage with its own exact response. No legacy graph is exercised
  by these new tests, and no real adapter is invoked.
- Missing providers and each existing construction-error category are tested at every
  role position. Error object/cause identity and stop order are asserted; no partially
  usable result is returned.
- Repeated resolution calls with identical settings create six distinct models through
  the test factory, proving no resolver reuse/cache. Environment reads are blocked
  with an unreadable os.environ mapping around resolution with prebuilt fake models.
- The registration test records exactly the five names and real factory identities,
  verifies caller isolation, and tests repeated calls and later-name conflict behavior.
- The fresh-process test blocks raw/integration SDK imports and socket connections,
  imports models/helper without registration, then explicitly registers all built-ins
  and asserts zero blocked attempts. Credentials are removed. Thus registering the
  real factory references performs no model construction or external API/network work.

The new package's AST boundary permits only models internals, config, provider registry,
LangChain Core and standard-library imports. Domain/config/provider SDK isolation rules
remain intact. No testing dependency was added. Existing socket/download/tracing guards
stay active; all new model responses come from explicit local test factories.

## Commands and results

Used `.tools/bootstrap/bin/uv.exe` with workspace-local
`UV_CACHE_DIR=.tools/uv-cache` and `UV_PYTHON_INSTALL_DIR=.tools/python`.
Equivalent root commands executed:

```powershell
uv sync
uv lock --check --offline
uv run pytest
uv run pytest tests/test_finrag_models.py tests/test_finrag_builtin.py tests/test_finrag_architecture.py
uv run ruff check src/finrag tests/test_finrag_models.py tests/test_finrag_builtin.py tests/test_finrag_architecture.py
uv run ruff format --check src/finrag tests/test_finrag_models.py tests/test_finrag_builtin.py tests/test_finrag_architecture.py
uv run mypy src/finrag
uv run mypy src/finrag tests/test_finrag_models.py tests/test_finrag_builtin.py tests/test_finrag_architecture.py
uv run ruff check . --output-format concise
uv run ruff format --check .
uv run mypy app
git diff --check
```

Also formatted only the two new test files after their initial format check failed.
Initial lint/mypy passed; final targeted formatting passed. Inspected git diff and
used diff --exit-code on dependency/config/provider/runtime paths to verify preservation.

| Check | Result |
| --- | --- |
| Sync / offline lock check | Passed; unchanged 191-package resolution, 157 installed packages checked |
| Pre-change pytest | 174 passed, three existing warnings |
| Focused tests | 25 passed |
| Final full pytest | **193 passed**, three existing warnings; no skips/xfails |
| Targeted Ruff / formatting | Passed; 21 Python files formatted |
| Mypy src/finrag | Passed; 18 files |
| Mypy src plus new/changed tests | Passed; 21 files |
| Diff whitespace / runtime preservation | Passed |
| Repository Ruff | 26 pre-existing findings, unchanged |
| Repository formatting | 11 pre-existing findings, unchanged |
| Legacy app mypy | 21 pre-existing errors in 8 files, unchanged |

The formatting debt includes the prior Phase 1.1C report snippet; no unrelated file
was repaired. Three warnings remain in untouched legacy Pydantic class configuration,
langchain-community, and HuggingFaceEmbeddings. No new warning, type suppression or
quality configuration change was introduced. The documented prior Anthropic constructor
typing exception is unchanged.

## Compatibility and scope

No pyproject.toml, uv.lock, requirements, SDK version or dependency-group changes.
The default environment remains usable without Groq/Anthropic/Google/Ollama integrations.
All existing adapters, registry/base/errors, config/domain, app/, LangGraph nodes,
demo/UI/entry points, Docker files and shared fixtures are unchanged. The legacy runtime
does not import this composition layer, so its construction behavior is preserved.
Live provider quality/availability and legacy retrieval/numeric/security debt are not
resolved by these offline interface tests. No runtime migration, caching, lifecycle,
routing, embeddings, or Phase 1.1F work is included.
