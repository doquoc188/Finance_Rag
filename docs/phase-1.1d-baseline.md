# Phase 1.1D - five explicit chat providers

Recorded on 2026-10-09 against starting revision `8092f44`. The
[ExecPlan](../.agent/plans/phase-1.1d.md) records inspection, decisions, and verification.

## Files

Created `src/finrag/providers/anthropic.py`, `google.py`, `ollama.py`,
`tests/test_finrag_ollama.py`, `.agent/plans/phase-1.1d.md`, and this report.
Changed `tests/test_finrag_adapters.py`, `tests/test_finrag_architecture.py`,
`pyproject.toml`, `uv.lock`, and `docs/development.md`.

No existing application files, config/domain contracts, registry/base/errors, generic
provider exports, OpenAI/Groq adapters, shared fixtures, or main README changed.

## Provider contracts and ownership

All five factories satisfy `Callable[[ChatModelSettings], BaseChatModel]`; the concrete
LangChain model is returned unchanged. Domain remains standard-library-only; config
and generic provider contracts never import concrete integrations. Factory exports
remain in their own modules, matching OpenAI/Groq. There is no automatic registration,
default registry, provider routing, or model-role construction.

| Canonical name | Module/factory | Category | Credential/endpoint ownership |
| --- | --- | --- | --- |
| openai | openai.create_openai_chat_model | Cloud API | OPENAI_API_KEY |
| anthropic | anthropic.create_anthropic_chat_model | Cloud API | ANTHROPIC_API_KEY |
| google | google.create_google_chat_model | Cloud Gemini Developer API | GOOGLE_API_KEY |
| groq | groq.create_groq_chat_model | Inference service | GROQ_API_KEY |
| ollama | ollama.create_ollama_chat_model | Local server | No key; OLLAMA_BASE_URL |

Module paths are under `finrag.providers`. Register explicitly on a caller-owned
ChatProviderRegistry. Names are trimmed/lowercased by the existing registry; no
Google/Gemini aliases were introduced. The existing registry intentionally has no
provider allowlist; these are the five supplied adapter names, not automatic entries.

Cloud factories reject missing/blank credentials before lazy SDK import, use SecretStr,
and neither load .env nor place credentials into general settings. Google requires
GOOGLE_API_KEY even if GEMINI_API_KEY is present: one documented credential source
avoids implicit selection. Its fixed `vertexai=False` chooses the intended Developer
API instead of the integration's ambient cloud/backend detection. No new generic
settings fields or speculative generation options were added.

Ollama reads OLLAMA_BASE_URL at factory invocation, trims it, and rejects an explicitly
blank value. When unset it passes `http://127.0.0.1:11434`, the inspected Ollama SDK's
standard local default. This explicit default is independent of ambient OLLAMA_HOST.
It does not require any key. Credentials embedded in an endpoint are not copied into
public error text. No URL connectivity check, tags request, model validation, pull,
download, or invocation is performed. Server/model availability remains unverified
until the caller uses the model; successful construction is only client creation.

## Selected integrations and constructor arguments

Official LangChain references identify the integrations:
[Anthropic](https://reference.langchain.com/python/langchain-anthropic/chat_models/ChatAnthropic),
[Google](https://reference.langchain.com/python/langchain-google-genai/chat_models/ChatGoogleGenerativeAI),
and [Ollama](https://reference.langchain.com/python/langchain-ollama/chat_models/ChatOllama).
The implementation decisions below were verified against actual installed signatures,
model fields, and source for the versions selected by this repository's lock.

| New integration | Locked version | Exact forwarded keywords |
| --- | --- | --- |
| langchain-anthropic / ChatAnthropic | 1.7.5 | model_name=config.model, temperature=config.temperature, api_key=SecretStr(key) |
| langchain-google-genai / ChatGoogleGenerativeAI | 4.4.0 | model=config.model, temperature=config.temperature, api_key=SecretStr(key), vertexai=False |
| langchain-ollama / ChatOllama | 1.1.0 | model=config.model, temperature=config.temperature, base_url=adapter endpoint |

Anthropic's current field alias/signature uses model_name. Google supports api_key
and explicit vertexai selection; its source gives the explicit selection priority.
Ollama's validate_model_on_init defaults to False; the adapter preserves that default
without adding a validation option. No timeout, retries, token limit, safety, seed,
reasoning, or other speculative parameter is forwarded.

## Dependencies and typing

Retained the existing `groq` extra. Added `anthropic`, `google`, and `ollama` extras,
with lower bounds equal to their verified integration versions. OpenAI remains in
the default path for legacy compatibility. Install extras with repeated flags, e.g.
`uv sync --extra anthropic --extra google --extra ollama`. Default sync installs none
of the four optional chat integrations.

Lock resolution added 11 packages: anthropic, docstring-parser, filetype, google-auth,
google-genai, langchain-anthropic, langchain-google-genai, langchain-ollama, ollama,
pyasn1, and pyasn1-modules. Only one existing package version changed: Tenacity
9.2.1 to 9.1.4. Installed google-genai 2.29.0 declares `tenacity<9.2.0,>=8.2.3`, so
the shared lock must satisfy that constraint. Other existing versions are unchanged;
no broad upgrade or legacy requirements/Docker edit was made. Offline lock checking
passed after resolution.

Expanded the existing missing-import mypy rule only to the new optional integration
namespaces. Installed-SDK checks separately validate constructor/result types.
Anthropic 1.7.5 uses `Field(None, alias=...)` for timeout and stop; mypy incorrectly
requires those arguments although both the runtime signature and Pydantic defaults
are None. One constructor-line `ignore[call-arg, unused-ignore]` documents this proven
upstream limitation. The unused-ignore code is necessary when the optional package
is absent. No return-type cast, Any factory result, extra runtime argument, or global
type-check relaxation was introduced. Both installed and absent SDK checks pass.

## Errors

Reused ProviderConfigurationError, ProviderDependencyError, and
ProviderInitializationError without changing their definitions. Cloud missing keys
and explicit blank Ollama endpoints raise configuration errors. Integration ImportError
raises dependency error with the package/extra installation command. Other import or
constructor exceptions raise initialization error with the original cause chained.
Fixed outer messages never include credentials, endpoint values, or raw SDK text.
Chained upstream exceptions can contain secrets and remain for debugging, not public
tracebacks. No fake/alternate-provider fallback or legacy error import was added.

## Offline tests and evidence

Extended existing cloud parametrization from OpenAI/Groq to Anthropic/Google rather
than creating duplicate suites. Sixteen additional cloud cases verify exact keyword
forwarding, instance identity, explicit registry use, BaseChatModel/Runnable behavior,
AIMessage response propagation, missing/blank keys, ignored .env/credential aliases,
dependencies, constructor/import errors, causes, and safe public error messages.
Google forwarding is tested with ambient Vertex selection enabled to verify the
explicit backend argument.

Added eight Ollama cases covering default/explicit endpoint, no credentials or server,
ignored .env/OLLAMA_HOST, local prompt composition, blank endpoints, missing dependency,
import errors, and safe chained constructor failure. Tests replace constructors with
local FakeListChatModel instances and explicitly isolate credentials/endpoint variables.
The existing socket guard stays active; no live model is invoked or downloaded.

Expanded import-boundary and fresh-process tests to all five concrete modules and
their raw/integration SDK namespaces. They allow each matching integration only inside
its factory, block SDK/network import attempts, and check no global or automatically
registered provider appears. Domain/config boundaries remain intact.

Manual checks while extras were installed constructed actual ChatAnthropic,
ChatGoogleGenerativeAI, and ChatOllama with dummy process-local credentials and socket
plus DNS guards. All three returned BaseChatModel instances with the expected model
and temperature; Google selected vertexai=False despite ambient Vertex settings,
and Ollama retained the local endpoint and disabled validation. Recorded **zero network
attempts**; no real model invocation, API call, Ollama HTTP request, health check,
tags query, or model pull occurred. Package downloads from PyPI were development
dependency installation, not provider requests. This proves construction/interface
compatibility, not live provider availability, response quality, or schema behavior.

## Commands and results

Used `.tools/bootstrap/bin/uv.exe`, workspace `UV_CACHE_DIR=.tools/uv-cache`, and
`UV_PYTHON_INSTALL_DIR=.tools/python`. Equivalent commands from the root:

```powershell
uv sync
uv run pytest
uv lock
uv sync --locked --extra anthropic --extra google --extra ollama
uv lock --offline
uv run --extra anthropic --extra google --extra ollama pytest tests/test_finrag_adapters.py tests/test_finrag_ollama.py tests/test_finrag_architecture.py
uv sync
uv lock --check --offline
uv run pytest
uv run ruff check src/finrag tests/test_finrag_adapters.py tests/test_finrag_ollama.py tests/test_finrag_architecture.py
uv run ruff format --check src/finrag tests/test_finrag_adapters.py tests/test_finrag_ollama.py tests/test_finrag_architecture.py
uv run mypy src/finrag
uv run mypy src/finrag tests/test_finrag_adapters.py tests/test_finrag_ollama.py tests/test_finrag_architecture.py
uv sync --locked --offline --extra anthropic --extra google --extra ollama
# Direct venv mypy verifies installed SDK types; default sync then removes extras.
uv sync
uv lock --check --offline
uv run ruff check . --output-format concise
uv run ruff format --check .
uv run mypy app
git diff --check
```

Also used direct venv Python for inspect.signature, installed metadata/source, lock
version comparison, real constructor guards, and optional-import absence checks.
Targeted `ruff check --fix` corrected one changed-test import block; targeted format
corrected four touched files. No unrelated formatting changes. The initial sandbox
lock attempt could not reach PyPI; a permitted network retry resolved/downloaded
the extras. Subsequent cache/offline checks passed.

| Check | Result |
| --- | --- |
| Pre-change sync/pytest | Passed; 150 tests, 3 warnings |
| Targeted tests with new extras installed | 46 passed |
| Default sync and offline lock checks | Passed; optional integrations absent |
| Full default pytest | **174 passed**, 3 existing warnings, no skips/xfails |
| Targeted Ruff/format | Passed; 18 files formatted |
| Mypy src/finrag | Passed; 15 files |
| Mypy src plus changed tests | Passed; 18 files, with and without optional integrations |
| Diff whitespace/runtime-preservation checks | Passed |
| Repository Ruff | 26 pre-existing findings |
| Repository formatting | 11 pre-existing findings: 10 legacy Python files plus Phase 1.1C report snippet |
| Legacy app mypy | 21 pre-existing errors in 8 files |

The extra formatting finding is in unchanged committed docs/phase-1.1c-baseline.md:
the prior phase report was written after its broad format run. It is newly observed,
not introduced here, and was left untouched. Three existing warnings still concern
legacy Pydantic class configuration, langchain-community, and HuggingFaceEmbeddings.
No new warning or warning suppression was introduced.

## Compatibility and phase limits

The legacy runtime, LangGraph nodes, model defaults, entry points, config/domain,
registry/base, existing adapters, and shared fixtures are unchanged. The new factories
are not consumed by app/. The required shared Tenacity lock change is explicitly
recorded above and the full existing suite passes with it. Main README claims remain
outside scope; legacy retrieval/numeric/security debt remains in prior reports.
No role resolution, runtime migration, automatic registration, other provider family,
or Phase 1.1E implementation was added.
