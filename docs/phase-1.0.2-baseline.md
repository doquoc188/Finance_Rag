# Phase 1.0.2 - Remove production fake-model fallback

Recorded on 2026-10-09 with the existing Python 3.12.15 / uv lock environment.
The [ExecPlan and pre-change audit](../.agent/plans/phase-1.0.2.md) were written before
changing behavior. The previous phase's pending edits were preserved.

## 1. Production SimpleFakeLLM references found

All three were in `app/core/llm.py`: definition at original line 5, construction in
the final provider-selection `else` at line 61, and construction in the broad
exception handler at line 63. No other production module instantiated or imported
the class. Graph nodes consumed it indirectly through `get_llm`.

## 2. Previous activation/fallback conditions

- OpenAI configured without a key or with the existing `mock` prefix: return fake.
- Groq configured without a key: return fake.
- Unsupported/empty provider: return fake.
- Missing SDK/transitive import, dependency initialization failure, or any exception
  from a provider constructor: return fake.
- Other paths: missing prompt dependency installed a fake `ChatPromptTemplate`;
  exhausted/missing providers in `simple_app.py` returned a fixed financial answer.

## 3. New behavior for each condition

| Condition | Outcome |
| --- | --- |
| Missing/blank required key, rejected OpenAI mock key, unsupported provider | `ModelConfigurationError` naming the relevant setting, without its value |
| Blank model name or missing Ollama base URL | `ModelConfigurationError` before SDK import |
| Missing SDK or transitive dependency | `ProviderUnavailableError` naming the package/provider and installation guidance; original cause chained |
| Non-import exception during lazy SDK loading | `ProviderUnavailableError` with safe installation guidance; original cause chained |
| SDK constructor failure | `ProviderUnavailableError` naming provider/configuration fields; original cause chained |
| Missing `langchain_core.prompts` | Explicit dependency error at import; no substitute prompt class |
| Standalone demo with neither key | `ModelConfigurationError` |
| Standalone demo request/response failure or empty answer | Existing Groq-then-Gemini attempt order preserved; exhausted attempts raise `ProviderUnavailableError`, never a fixed answer |
| Successful configured provider | Original constructor parameters and return object preserved; successful demo requests/responses preserved |

Public exception messages contain no credential values or raw SDK error strings.
Original causes are retained for debugging; do not expose full exception chains to
clients because upstream exceptions may themselves contain sensitive details.
The existing API catches the safe outer message; HTTP response structure is unchanged.

## 4. Fate of SimpleFakeLLM

Deleted. No production reference or construction remains. Test doubles remain explicit
fixtures in `tests/`; the real compiled workflow and production nodes still execute.
The fake prompt class and standalone canned answer were also removed.

## 5. Exceptions introduced

`app/core/errors.py` defines only `ModelConfigurationError(ValueError)` and
`ProviderUnavailableError(RuntimeError)`. It has no external dependencies, so the
standalone demo remains usable without the application SDK stack.

## 6. Files changed in this phase

- `app/core/llm.py`: delete fake, validate settings, wrap provider import/constructor failures.
- `app/core/errors.py`: new minimal error types.
- `app/graph/nodes.py`: replace the fake prompt import branch with an explicit error.
- `simple_app.py`: remove terminal canned answer and preserve safe chained request errors.
- `tests/test_model_failures.py`, `tests/test_demo_model_failures.py`: new offline regressions.
- `.agent/plans/phase-1.0.2.md`, this report, and `docs/development.md`: audit, decisions, results.
- `README.md`: correct only the setup sentence that promised automatic fake-model fallback.

Ruff cleanup/formatting is limited to touched Python files. The unused grading local
was renamed `_question` to preserve its existing state lookup. No prompt, graph edge,
retrieval algorithm, grading algorithm, or generation algorithm was redesigned.
Pre-existing changes to `tests/test_rag.py` and `docs/phase-1.0.1-baseline.md` are from
Phase 1.0.1 and were not edited by this phase. Dependencies and lockfile are unchanged.

## 7. Tests added/modified

Added 38 parametrized cases: missing/rejected configuration, all existing provider
import/initialization paths, safe messages and exception causes, constructor argument
and return-value preservation, workflow error propagation without test-model injection,
missing prompt imports, demo missing keys/errors/empty output, successful demo response
parsing, and preservation of the demo's real-provider attempt order.
The seven existing tests and their explicit model fixtures are unchanged.

Tests replace SDK constructors/imports or HTTP openers explicitly and use dummy key
strings, never real credentials. The existing socket guard stays enabled. No provider
API, model download, or external service was invoked.

## 8. Commands executed

Workspace-local uv and cache paths are documented in the Phase 1.0 report.

```powershell
uv sync
uv run pytest
uv run ruff check app/core/llm.py app/core/errors.py app/graph/nodes.py simple_app.py tests/test_model_failures.py tests/test_demo_model_failures.py
uv run ruff format --check app/core/llm.py app/core/errors.py app/graph/nodes.py simple_app.py tests/test_model_failures.py tests/test_demo_model_failures.py
uv run mypy --follow-imports=silent app/core/errors.py tests/test_model_failures.py tests/test_demo_model_failures.py
uv run mypy app
uv run ruff check . --output-format concise
uv run ruff format --check .
git diff --check
```

Also used targeted `ruff check --fix` and `ruff format` on the same six files and
`rg` to audit production fake references. `--follow-imports=silent` scopes diagnostics
to the three new modules without hiding the separately reported full application run;
no mypy configuration or suppression was added to the repository.

## 9. Pytest result

**45 passed**, no skips or xfails. Successful provider behavior is checked at constructor
and HTTP adapter boundaries with local doubles; live provider correctness is unverified.

## 10. Targeted quality results

All six changed/new Python files pass Ruff lint and formatting. New exception and test
modules pass targeted mypy (3 source files). `git diff --check` passes. The new chat-model
failure handling adds no application mypy diagnostics; existing errors remain below.

## 11. Remaining deprecation warnings

Three existing warnings: Pydantic class-based settings configuration,
`langchain-community` deprecation, and the community `HuggingFaceEmbeddings` adapter.

## 12. Remaining legacy lint/type debt

Repository Ruff: 26 findings (previously 39). Formatting: 10 untouched files need
formatting (previously 13). Full application mypy: 21 errors in 8 files, checking 11
files (previously 24 errors). Removed mock code and touched-file cleanup account for
the decrease; no blanket ignore or repository-wide cleanup was performed.

Existing embedding constructor typing, Pydantic import shims, parser/retriever optional
parameters, missing third-party stubs, TypedDict updates, and upload/schema types remain.
In particular, the unchanged embedding function has one existing mypy diagnostic, and
the unchanged Pydantic fallback block in the touched nodes file has five; these are
not claimed green by the targeted new-module run.

Constant `DummyEmbeddings`, mock retrieval evidence, heuristic grading, and web
placeholders remain explicitly out-of-scope retrieval/workflow debt. This task removes
chat-model substitution, not every fallback in the repository. No Phase 1.1 work began.
