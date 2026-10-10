# Phase 1.1C - OpenAI and Groq chat factories

## Objective
Add explicit OpenAI/Groq factories satisfying ChatProviderFactory, with safe errors,
lazy integration imports, and offline tests. Preserve the existing application.

## Current-state observations
- Starting revision `147c3ab`, clean working tree. Read AGENTS.md, .agent/PLANS.md,
  Phase 1.0.2/1.1A/1.1B reports, config/providers, and existing tests before editing.
- Factories take ChatModelSettings and return BaseChatModel. Registry stores factories
  per instance and never constructs models. Config has no credentials or SDK imports.
- OpenAI 1.7.0 is installed/default; Groq integration 1.1.3 is locked as optional.
  Both installed signatures inspected: model, temperature, api_key are supported;
  api_key accepts SecretStr. No dependency version changes needed.
- Baseline sync passed; pytest: 132 passed, 3 existing warnings. Prior debt is
  26 Ruff findings, 10 unformatted files, 21 app mypy errors in 8 files.
- Temporary `uv sync --locked --extra groq` initially failed on sandbox networking;
  permitted retry installed only locked groq 0.37.1 and langchain-groq 1.1.3 for
  signature inspection. Restore default sync after optional-package validation.

## Scope
Two adapter modules, minimal errors/exports, focused parametrized adapter tests,
architecture boundary tests, and engineering/development documentation.

## Non-goals
No registry redesign, global registration, role resolver, new provider families,
embedding/reranker adapter, runtime migration, retrieval changes, or broad cleanup.

## Architecture decisions
- 2026-10-09: Separate typed functions with lazy integration imports. Keep explicit
  imports from concrete adapter modules; generic package exports errors/contracts only.
- Read only the provider's API-key environment variable at factory call time; reject
  absent/whitespace-only keys before import. Do not load .env or store keys in settings.
  Pass the supplied nonblank key as SecretStr, plus model and temperature, to the SDK.
- Add ProviderConfigurationError, ProviderDependencyError, ProviderInitializationError.
  ImportError during integration load maps to dependency error with install guidance;
  other import-time failures and constructor exceptions map to initialization error.
  Chain original causes; public outer messages include no supplied values or SDK text.
- BaseChatModel preserves Runnable/message and standard structured-output interfaces.
  Do not wrap the returned model or introduce a second structured-output API.
- Allow only each adapter's matching integration import, inside its factory. Config
  and domain boundaries stay independent of provider SDKs. No new dependencies.

## Implementation steps
1. Add factories/errors/exports without changing existing config/registry/app.
2. Add offline constructor-boundary tests and cold-import isolation check; extend
   AST boundary checks narrowly for concrete adapters and config.
3. Verify with the locked extra installed and then in the default environment.
4. Review diff; record results, unchanged debt, limitations, and next scoped phase.

## Test strategy
Parametrize OpenAI/Groq success, missing/blank keys, import failures, constructor
failures, cause preservation, and secret-safe outer messages. Constructor replacements
return FakeListChatModel; assert exact forwarded arguments, explicit registry resolution,
prompt composition, and AIMessage content. Clear both keys per test and retain socket
guard; subprocess import checks install their own network/import guards.
No live model invocation or API validation. Run from root using local uv/cache:
```powershell
uv sync
uv run pytest
uv run ruff check src/finrag tests/test_finrag_adapters.py tests/test_finrag_architecture.py
uv run ruff format --check src/finrag tests/test_finrag_adapters.py tests/test_finrag_architecture.py
uv run mypy src/finrag
uv run mypy src/finrag tests/test_finrag_adapters.py tests/test_finrag_architecture.py
uv lock --check --offline
uv run ruff check . --output-format concise
uv run ruff format --check .
uv run mypy app
git diff --check
```

## Migration/compatibility concerns
No runtime migration, entry-point changes, data migration, dependency upgrades, or
provider default change. Groq remains optional; default sync removes the inspection
extra. Only outer error messages are safe for clients; chained upstream exceptions
may contain secrets and must not be exposed as a public traceback.

## Acceptance criteria
Both factories satisfy the existing contract, preserve LangChain behavior, fail
explicitly without fallback, retain causes and protect public error text. Generic
imports work without either SDK. Tests are offline and all targeted quality gates
and full pytest pass; dependency lock and existing runtime remain unchanged.

## Progress checklist
- [x] Inspect repository/locked constructors and establish baseline.
- [x] Implement factories and focused tests. With locked Groq extra: 22 targeted tests
  passed; mypy passed for all src plus changed tests (14 files); Ruff lint passed.
  Initial format check flagged three touched files; formatted only those files.
  Real constructors also passed under socket guards with dummy keys and zero network
  attempts; no real model was invoked.
- [x] Verify optional/default environments and quality gates. Final default pytest:
  150 passed, 3 unchanged warnings; Groq absent. Targeted lint/format/mypy passed.
- [x] Document results and confirm scope/runtime preservation.

## Final results
All acceptance criteria met. Added two factories, three error types, 16 adapter
cases, and two architecture cases. Targeted Ruff/format passed for 14 files; mypy
passed for src (12 files) and src plus changed tests (14 files), including a check
against the installed optional SDK. Sync, offline lock check, and diff check passed.
Repository-wide debt unchanged: 26 Ruff findings, 10 unformatted legacy files,
21 app mypy errors in 8 files. Dependency manifests/lock, registry/base, config/domain,
and legacy runtime untouched. No new warning or quality-gate relaxation. Full details
and commands: docs/phase-1.1c-baseline.md. Phase 1.1D not implemented.
