# Phase 1.1B - chat provider contracts and registry

Recorded on 2026-10-09 against starting revision `07a84f9`. The
[ExecPlan](../.agent/plans/phase-1.1b.md) records inspection and pre-change results.
The legacy runtime is unchanged; Phase 1.1C has not begun.

## Files

Created:
- `src/finrag/providers/__init__.py`: public exports only.
- `src/finrag/providers/base.py`: `ChatProviderFactory` type alias.
- `src/finrag/providers/registry.py`: `ChatProviderRegistry`.
- `src/finrag/providers/errors.py`: three registration/lookup errors.
- `tests/test_finrag_providers.py`: nine offline test cases.
- `.agent/plans/phase-1.1b.md` and this report.

Modified `tests/test_finrag_architecture.py`, `AGENTS.md`, and `docs/development.md`.
No dependency, lockfile, packaging, config/domain implementation, or legacy
application changes were required. Existing package discovery includes providers.

## Contract and ownership

```python
type ChatProviderFactory = Callable[[ChatModelSettings], BaseChatModel]
```

The exact result type is `langchain_core.language_models.chat_models.BaseChatModel`.
Factories consume the existing immutable chat settings. A callable alias accepts
functions or callable instances without a second provider class hierarchy.

Inspection of `app/graph/nodes.py` confirmed prompt composition and message content
for generation, direct synchronous invocation for rewriting, and
`with_structured_output(GradeDocuments).invoke(...)` for relevance grading. Returning
BaseChatModel preserves these LangChain interfaces, including asynchronous invocation,
without reducing models to string-in/string-out functions. Domain remains independent
of LangChain; this coupling belongs to the provider boundary.

The installed BaseChatModel structured-output implementation delegates to bind_tools
and parsers; it raises NotImplementedError when a concrete model lacks tool support.
Thus the return type preserves the interface but does not certify every provider's
schema capabilities. Later adapters must test actual structured-output support; no
parallel schema abstraction or fabricated production implementation is introduced.

Each registry owns a private `dict[str, ChatProviderFactory]` initialized in its
constructor. Registrations are not shared. There is no global registry, singleton,
discovery, automatic registration, environment lookup, client construction, or model
cache. Register/get do not invoke factories. The caller resolves a factory and then
explicitly calls it; settings pass through unchanged, as do construction exceptions.
The caller owns model lifetime and any later capability checks.

Usage, with a caller-supplied factory:

```python
registry = ChatProviderRegistry()
registry.register("local", factory)
model = registry.get(settings.provider)(settings)
```

Both operations canonicalize names using `strip().lower()`. No provider allowlist is
implied, and configuration values themselves are not rewritten. Empty or whitespace-only
names raise `InvalidProviderNameError(ValueError)`. A repeated canonical name raises
`DuplicateProviderError(ValueError)` and preserves the original factory. Missing lookup
raises `ProviderNotRegisteredError(LookupError)` naming the provider and advising explicit
registration on that instance. There is no overwrite option or alternate-provider fallback.
These errors do not import or duplicate legacy SDK/configuration error handling.

Concrete adapters remain deferred to Phase 1.1C+ so later tasks can separately specify
provider dependencies, constructor parameters, credentials, and capability tests.
No model-role resolver or embedding/reranker registry is included.

## Tests and results

Nine new registry cases cover lazy registration/lookup, exact settings handoff,
canonical names, blank names, actionable missing lookup, duplicate preservation,
instance isolation, LangChain composition, and construction-error propagation.

The test-only factory creates LangChain Core `FakeListChatModel`, annotated as
BaseChatModel. The test invokes `ChatPromptTemplate | resolved_model`, asserts an
AIMessage with the exact configured response, and exercises direct model invocation.
No real-provider SDK, key, model download, or network request is needed. The existing
socket guard remains active. This proves local interface compatibility, not provider
quality or live structured-output behavior. The fake's structured-output mechanism is
not extended; that capability remains on the standard BaseChatModel interface.

The existing AST import-boundary check now also checks providers against standard
library, `finrag.providers`, `finrag.config`, and `langchain_core`. The domain allowlist
is unchanged, as is its isolated import/construction check without site-packages.
No architecture testing dependency was added. Ten additional cases overall give
132 passing tests compared with 122 before this phase.

## Commands executed

Used `.tools/bootstrap/bin/uv.exe` with `UV_CACHE_DIR=.tools/uv-cache` and
`UV_PYTHON_INSTALL_DIR=.tools/python`; these are the equivalent root commands:

```powershell
uv sync
uv run pytest
uv run pytest tests/test_finrag_providers.py tests/test_finrag_architecture.py
uv run ruff check src/finrag tests/test_finrag_providers.py tests/test_finrag_architecture.py
uv run ruff format --check src/finrag tests/test_finrag_providers.py tests/test_finrag_architecture.py
uv run mypy src/finrag
uv run mypy src/finrag tests/test_finrag_providers.py tests/test_finrag_architecture.py
uv run ruff check . --output-format concise
uv run ruff format --check .
uv run mypy app
git diff --check
```

| Check | Result |
| --- | --- |
| Sync | Passed; existing 180-package resolution, 157 packages checked |
| Full pytest before changes | 122 passed, 3 warnings |
| Targeted provider/architecture pytest | 13 passed |
| Full pytest after changes | **132 passed**, 3 existing warnings; no skips/xfails |
| Targeted Ruff | Passed |
| Targeted formatting | Passed; 12 files |
| New architecture mypy | Passed; 10 source files |
| New architecture plus touched tests mypy | Passed; 12 files |
| Diff whitespace check | Passed |
| Repository Ruff | 26 existing findings, unchanged |
| Repository formatting | 10 existing files need formatting, unchanged |
| Legacy application mypy | 21 existing errors in 8 files, unchanged |

Initial targeted Ruff found one import-order error in the modified architecture test;
the import block was corrected and the check passed. No quality-gate suppression or
configuration relaxation was added. Git may emit its existing LF-to-CRLF checkout
notices; these are not whitespace-check failures.

The three warnings still originate from untouched legacy Pydantic class configuration,
the LangChain community package, and the community HuggingFaceEmbeddings adapter.
No new warning or warning suppression was introduced.

Diff inspection confirmed no changes to app/, the standalone demo, UI, entry points,
Docker files, requirements, pyproject.toml, uv.lock, config/domain, or test fixtures.
The application does not consume this registry, so its runtime behavior is unchanged.
Legacy retrieval, numeric, security, lint, and type debt remains documented in the
prior baselines. The next scoped task is Phase 1.1C provider adapters, after its
requirements are supplied; none is implemented here.
