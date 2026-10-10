# Phase 1.1E - explicit chat model-role composition

## Objective
Resolve independently configured chat roles into typed LangChain models, through a
caller-owned registry; explicitly register built-in factories without constructing them.

## Current-state observations
- Starting revision `72927e6`, clean working tree. Read AGENTS.md, .agent/PLANS.md,
  Phase 1.1A-D reports, config/providers and existing provider/architecture tests.
- Settings supplies generation, grading, query_rewrite ChatModelSettings; embeddings
  are separate. Factories take those settings and return BaseChatModel.
- Registry stores factories per instance, normalizes names, and rejects duplicates.
  All five adapters import their SDK lazily only when called; existing errors suffice.
- Pre-change sync and offline lock check passed; pytest 174 passed with 3 warnings.
  Current legacy debt: 26 Ruff findings, 11 formatting findings, 21 app mypy errors.

## Scope
New finrag.models package, providers/builtin.py helper, focused offline tests,
extended boundary checks, engineering/development docs and phase report.

## Non-goals
No app/LangGraph/FastAPI migration, config redesign, registry redesign, routing,
fallback, caching, lifecycle cleanup, embeddings, retrieval or later Phase 1.1F.

## Architecture decisions
- 2026-10-09: finrag.models is the small composition boundary; use a resolver function
  instead of a stateful resolver class. API resolve_model_roles(Settings, registry).
- ModelRoles is a frozen slotted dataclass with generation, grading, query_rewrite
  BaseChatModel fields, matching existing config names. Frozen bindings do not claim
  to freeze the underlying LangChain model internals.
- Resolve generation, grading, query_rewrite in that order. Pass the original validated
  role config to its selected factory. Build the result only after all three succeed;
  do not catch/wrap failures, read environment, cache, or reuse models across roles.
- Register built-ins explicitly via providers.builtin.register_builtin_chat_providers.
  Keep generic provider exports unchanged. The helper registers openai, groq,
  anthropic, google, ollama sequentially using existing registry semantics.
- Duplicate calls fail, never overwrite. Registration is not transactional: a conflict
  encountered later leaves earlier successful registrations in that supplied registry.
  Document/test this rather than add registry batch/rollback APIs. Model resolution,
  independently, never returns a partial result.
- New imports limited to config, provider registry, LangChain Core, dataclasses and
  models internals; SDKs remain inside concrete factories. Helper creates no client.

## Implementation steps
1. Create ModelRoles/resolver/public models exports and explicit built-in helper.
2. Test independent configs, deterministic failure order/error identity, all-or-fail
   output, immutable bindings, no cache/environment reads and local prompt composition.
3. Test real built-in factory identities, exact registration names, no construction,
   duplicate/conflict behavior and caller isolation; extend cold-import/boundary tests.
4. Run gates, review runtime/dependency diff and write results.

## Test strategy
Test-only factories return distinct FakeListChatModel objects; never invoke real
adapters. Assert per-role original config identity and exact selected response.
Exercise missing registration and existing configuration/dependency/initialization
errors across role positions. Guard environment access around resolution with
prebuilt fakes. Registration cold-process checks block integrations/network and ensure
no auto-registration/global registry. Run using local uv with workspace caches:
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

## Migration/compatibility concerns
No module moves, legacy behavior change, SDK/dependency or stored-data changes.
Provider errors propagate unchanged. Earlier constructed local objects may exist if
a later role fails; no public container is produced and no resource cleanup guarantee
is added. Caller owns registry/model lifetime. Built-in registration requires no keys
or optional SDK; SDK/credential validation happens only when resolving real adapters.

## Acceptance criteria
Typed immutable role bindings; independent factories/configs; deterministic explicit
all-or-fail resolution; explicit five-provider registration without model construction;
no env reads/cache/globals; offline LangChain compatibility; full pytest and targeted
gates pass; dependencies and legacy runtime remain unchanged.

## Progress checklist
- [x] Inspect current state and establish baseline.
- [x] Implement composition and registration helper. Focused tests: 25 passed;
  initial Ruff lint and mypy passed. Initial format check flagged only two new test
  files; formatted those files without unrelated changes.
- [x] Verify focused/full tests and quality gates: 193 full tests passed, three existing
  warnings; targeted Ruff/format/mypy and offline lock checks passed.
- [x] Record final results and scope preservation.

## Final results
All acceptance criteria met. Created models roles/resolver/exports, explicit built-in
helper, and two focused test modules; extended boundaries and docs. Added 19 cases;
focused 25 passed, full 193 passed with three unchanged warnings. Targeted Ruff/format
passed for 21 files, mypy src for 18 and src plus touched tests for 21. Sync/offline
lock/diff checks passed. Existing broad debt remains 26 Ruff, 11 format, and 21 app
mypy errors in eight files. Dependencies/provider contracts/adapters/config/domain/
legacy runtime unchanged; no new quality exceptions. See docs/phase-1.1e-baseline.md
for exact APIs, sequential registration caveat, error ownership, commands and evidence.
Next work requires a separately scoped Phase 1.1F request; none implemented here.
