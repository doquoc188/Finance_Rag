# Phase 1.1B - chat provider contracts and registry

## Objective
Add explicit, typed provider registration/resolution while preserving LangChain
chat-model capabilities and leaving the legacy runtime unchanged.

## Current-state observations
- Starting revision: `07a84f9`; working tree clean.
- Read AGENTS.md, .agent/PLANS.md, both requested baseline reports, config/domain
  contracts, existing tests, and app/graph/nodes.py before editing code.
- Generation composes `ChatPromptTemplate | llm` and reads response.content;
  rewriting invokes directly; document grading invokes with_structured_output.
- Installed BaseChatModel provides invoke and with_structured_output. The latter
  uses bind_tools and raises NotImplementedError if the concrete model lacks support.
- ChatModelSettings already supplies provider/model/temperature. Packaging discovers
  finrag subpackages; LangChain Core is already a runtime dependency.
- Pre-change `uv sync` passed; `uv run pytest`: 122 passed, 3 legacy warnings.
  Prior baseline records 26 Ruff findings, 10 unformatted files, 21 app mypy errors.

## Scope
New providers package, focused registry tests, expanded architecture test, and
minimal development/engineering documentation plus this plan and phase report.

## Non-goals
No concrete adapters, SDKs, role resolver, discovery, globals, runtime migration,
retrieval changes, or broad quality-debt cleanup. No Phase 1.1C implementation.

## Architecture decisions
- 2026-10-09: Callable[[ChatModelSettings], BaseChatModel] is the factory contract;
  use a type alias rather than an additional provider class hierarchy.
- Registry owns its own dict of canonical names to factories. Registration/lookup
  never call factories; callers explicitly construct models using returned factories.
- Names use strip().lower() for both operations. Reject blank names, duplicates,
  and missing providers with three narrowly scoped errors. No overwrite API/fallback.
- Preserve BaseChatModel.with_structured_output; do not invent another interface.
  Concrete adapters must later prove schema/tool support; BaseChatModel alone does
  not guarantee it. The local composition test uses LangChain Core FakeListChatModel.
- Domain remains standard-library-only; provider imports limited to standard library,
  provider internals, config contracts, and LangChain Core.

## Implementation steps
1. Create providers/base.py, registry.py, errors.py and public exports.
2. Add focused offline tests and extend the static import boundary check.
3. Run checks below, inspect diff, document measured results and remaining debt.

## Test strategy
Test lookup identity/lazy construction, normalization, blank/missing/duplicate errors,
instance isolation, and prompt composition with a locally constructed AIMessage result.
Keep the existing socket guard; no keys/services needed. Test factory exceptions are
not swallowed. Preserve existing domain isolation tests. Commands from repo root:
```powershell
uv sync
uv run pytest
uv run ruff check src/finrag tests/test_finrag_providers.py tests/test_finrag_architecture.py
uv run ruff format --check src/finrag tests/test_finrag_providers.py tests/test_finrag_architecture.py
uv run mypy src/finrag
uv run mypy src/finrag tests/test_finrag_providers.py tests/test_finrag_architecture.py
uv run ruff check . --output-format concise
uv run ruff format --check .
uv run mypy app
git diff --check
```
Use the workspace-local .tools/bootstrap/bin/uv.exe and workspace caches as in
previous phases when uv is not on PATH. Live-provider correctness is not tested.

## Migration/compatibility concerns
No app imports/entry points, dependencies, settings fields, indexes, or stored data
change. Registry canonicalizes lookup names only; supplied settings stay unchanged.
New instances start empty; caller owns registration and client lifetime. No factory
construction, caching, capability validation, or exception wrapping in the registry.

## Acceptance criteria
Typed contract and explicit isolated registry; meaningful errors; real offline
Runnable composition; preserved domain boundary; green full pytest and targeted
quality gates; no new dependencies/adapters or legacy runtime changes.

## Progress checklist
- [x] 2026-10-09: Inspect instructions, contracts, workflow, installed LangChain; baseline.
- [x] 2026-10-09: Implement registry/contracts and focused tests (13 targeted cases pass).
  Initial Ruff found one import-order issue in the modified architecture test; fixed
  only that import block. Initial formatting and mypy checks already passed.
- [x] 2026-10-09: Verify quality/runtime preservation and update documentation.

## Final results
Completed all acceptance criteria. Added four provider modules, nine registry cases,
and one provider architecture-boundary case. Updated engineering/development docs.
Commands above ran: sync passed; full pytest 132 passed (3 unchanged warnings);
targeted Ruff/format passed (12 files); mypy passed for src (10 files) and src plus
touched tests (12 files). Diff whitespace checks passed. Repository baseline remains
26 Ruff findings, 10 unformatted legacy files, and 21 app mypy errors in 8 files.
Git diff confirmed no changes to legacy runtime, dependencies, config/domain, or
existing fixtures. No suppressed errors or unmet criteria. See
docs/phase-1.1b-baseline.md for exact behavior, files, and verification. Next task is
separately scoped Phase 1.1C adapters; none implemented here.
