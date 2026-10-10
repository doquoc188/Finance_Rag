# Phase 1.1C - OpenAI and Groq chat adapters

Recorded on 2026-10-09 against starting revision `147c3ab`. The
[ExecPlan](../.agent/plans/phase-1.1c.md) contains the pre-change inspection and decisions.

## Files created and changed

Created:
- `src/finrag/providers/openai.py`
- `src/finrag/providers/groq.py`
- `tests/test_finrag_adapters.py`
- `.agent/plans/phase-1.1c.md`
- `docs/phase-1.1c-baseline.md`

Updated provider `errors.py` and `__init__.py`, `tests/test_finrag_architecture.py`,
`AGENTS.md`, and `docs/development.md`. Registry/base contracts, settings/domain,
legacy app, README, existing workflow tests, and dependency files are unchanged.

## Factory contract and credentials

`create_openai_chat_model(config: ChatModelSettings) -> BaseChatModel` constructs
`langchain_openai.ChatOpenAI`. `create_groq_chat_model` has the same signature and
constructs `langchain_groq.ChatGroq`. Both satisfy the existing ChatProviderFactory
callable contract. The model object is returned unchanged: no wrapper erases Runnable,
message, or standard structured-output interfaces. No separate provider class or
structured-output abstraction is introduced. Live schema behavior remains unverified.

The caller's process supplies `OPENAI_API_KEY` or `GROQ_API_KEY`. Each factory reads
only its own variable when called, validates presence/nonblank content before loading
the integration, and passes a SecretStr credential to the constructor. Nonblank key
content is preserved. Keys are not placed in settings/domain models, logged, or read
from .env. The factory does not reuse another provider's key.

Both integrations import inside their concrete factory functions. Generic finrag,
providers, and registry imports load neither integration. Even importing the adapter
modules does not import their SDKs or construct models. OpenAI stays in the existing
default dependency set; Groq remains the optional `groq` extra. No import registers a
factory; each caller owns its registry and model instances.

## Constructor signatures and exact arguments

Inspected the actual installed signatures of the locked integrations:

| Integration | Version | Supported constructor fields used |
| --- | --- | --- |
| langchain-openai | 1.7.0 | model: str; temperature: float or None; api_key accepts SecretStr |
| langchain-groq | 1.1.3 | model: str; temperature: float; api_key accepts SecretStr or None |

The exact keyword arguments in both factories are:

```python
model=config.model,
temperature=config.temperature,
api_key=SecretStr(key),
```

No endpoint, timeout, retry, routing, or other speculative options were added. No
third-party version was changed to accommodate constructor syntax. Groq was temporarily
installed via `uv sync --locked --extra groq` for inspection and type checking, then
removed by plain `uv sync` before the final full test run. The inspection extra used
the already-locked `groq==0.37.1` and `langchain-groq==1.1.3`.

## Errors

| Condition | New error | Cause retained |
| --- | --- | --- |
| Missing/blank provider key | ProviderConfigurationError(ValueError) | No upstream failure |
| ImportError loading integration or transitive dependency | ProviderDependencyError(ImportError) | Yes |
| Other integration-load failure | ProviderInitializationError(RuntimeError) | Yes |
| Constructor exception | ProviderInitializationError(RuntimeError) | Yes |

Dependency errors identify the integration and installation command. Groq guidance is
`uv sync --extra groq`; OpenAI guidance is `uv sync`. Import/constructor exceptions are
immediately re-raised with `raise ... from exc`. Public outer messages contain only
fixed guidance and configuration field names, never credential values, model values,
or raw SDK messages. Original causes can contain sensitive SDK text; do not publish
full chained tracebacks to clients. No fake, alternate provider, or silent disablement
is used. Legacy app errors are not imported.

## Explicit registry interoperability

```python
from finrag.providers import ChatProviderRegistry
from finrag.providers.groq import create_groq_chat_model
from finrag.providers.openai import create_openai_chat_model

registry = ChatProviderRegistry()
registry.register("openai", create_openai_chat_model)
registry.register("groq", create_groq_chat_model)
# Caller supplies settings and calls the chosen factory explicitly:
# model = registry.get(settings.provider)(settings)
```

No registry redesign, singleton, discovery, automatic registration, or model-role
resolver was needed. Existing registry isolation tests remain intact.

## Tests and evidence of offline execution

Added 16 parametrized adapter cases, covering both integrations' exact constructor
forwarding and object identity, explicit registry registration, AIMessage propagation
through `prompt | model`, missing/empty/whitespace credentials, ignored .env files,
missing integrations, import-time errors, initialization failures, preserved causes,
and secret-free public error messages. Constructors are replaced at the integration
module boundary with functions returning LangChain Core FakeListChatModel. Tests work
whether Groq is installed or absent and explicitly clear both key variables per case.

Architecture tests now additionally check configuration imports and a cold subprocess
import path. Matching SDK imports are allowed only inside their concrete factory;
Pydantic SecretStr is permitted in those adapter modules. Domain retains its original
standard-library-only rule. Generic providers cannot import either integration. The
cold process blocks both integration/raw SDK imports and socket connections, verifies
zero attempts, and checks no global/automatic registry appeared.

The full test suite retains the original socket guard, disabled tracing/downloads,
and isolated test credentials. An additional manual construction check used the real
locked ChatOpenAI and ChatGroq constructors, dummy process-local keys, socket guards,
and assertions on BaseChatModel/model/temperature. Both succeeded with **zero network
attempts**. Those real models were never invoked. This is constructor/interface
validation, not a live API, credential, response-quality, or structured-output evaluation.

The optional-package download used PyPI, not a model API. Its first attempt failed
because sandbox networking was restricted; the permitted retry succeeded. This did
not change dependency requirements or the lockfile.

## Commands and measured results

Used `.tools/bootstrap/bin/uv.exe` with workspace-local `UV_CACHE_DIR=.tools/uv-cache`
and `UV_PYTHON_INSTALL_DIR=.tools/python`. Equivalent root commands:

```powershell
uv sync
uv run pytest
uv sync --locked --extra groq
uv run --extra groq pytest tests/test_finrag_adapters.py tests/test_finrag_architecture.py
# While the extra was installed, direct venv Ruff/mypy and Python inspection/checks ran.
uv sync
uv lock --check --offline
uv run pytest
uv run ruff check src/finrag tests/test_finrag_adapters.py tests/test_finrag_architecture.py
uv run ruff format --check src/finrag tests/test_finrag_adapters.py tests/test_finrag_architecture.py
uv run mypy src/finrag
uv run mypy src/finrag tests/test_finrag_adapters.py tests/test_finrag_architecture.py
uv run ruff check . --output-format concise
uv run ruff format --check .
uv run mypy app
git diff --check
```

Also ran targeted `ruff format` on three touched files after the initial formatting
check identified layout differences. No unrelated files or quality configuration changed.
Signature inspection used `inspect.signature(ChatOpenAI/ChatGroq)` and
`importlib.metadata.version` through `.venv/Scripts/python.exe`; constructor checks
used that same interpreter while the extra was installed.

| Check | Result |
| --- | --- |
| Pre-change pytest | 132 passed, 3 existing warnings |
| Targeted tests with locked Groq extra installed | 22 passed |
| Final default sync and offline lock check | Passed; Groq extra removed |
| Final default pytest | **150 passed**, 3 existing warnings; no skips/xfails |
| Targeted Ruff / formatting | Passed; 14 files formatted |
| Mypy src/finrag | Passed; 12 files |
| Mypy src plus changed tests | Passed; 14 files, both with and without Groq installed |
| Diff whitespace check | Passed |
| Repository Ruff | 26 existing findings, unchanged |
| Repository formatting | 10 legacy files need formatting, unchanged |
| Legacy mypy app | 21 existing errors in 8 files, unchanged |

The existing optional-Groq missing-import mypy rule is unchanged; checking while
Groq was installed additionally validated the real constructor types. Three unchanged
warnings concern legacy Pydantic class configuration, the community LangChain package,
and HuggingFaceEmbeddings. No warning/error suppression was introduced.

## Compatibility and remaining scope

No pyproject.toml, uv.lock, or requirements.txt changes. Diff checks confirmed no
changes to app/, the demo/UI, entry points, Docker files, registry/base, config/domain,
or shared test fixtures. The legacy application still uses its existing model factory;
its runtime behavior is unchanged. Existing retrieval/numeric/security debt stays in
the earlier reports. No Anthropic, Gemini, Ollama, role resolution, or Phase 1.1D work
was implemented. Further provider/runtime work requires its next scoped task.
