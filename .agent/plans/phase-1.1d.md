# Phase 1.1D - Anthropic, Google, and Ollama adapters

## Objective
Complete the five explicitly registered chat-provider factories with three optional
integrations, retaining BaseChatModel and leaving the legacy runtime untouched.

## Current-state observations
- Starting revision `8092f44`; clean working tree. Read AGENTS.md, .agent/PLANS.md,
  Phase 1.0.2/1.1A/1.1B/1.1C reports, config/providers, and adapter/boundary tests.
- ChatProviderFactory takes immutable ChatModelSettings and returns BaseChatModel;
  registration is instance-owned. OpenAI/Groq integrations import inside factories.
- Keys are adapter-owned environment values; errors already distinguish configuration,
  dependency, and initialization failures and preserve causes without public secrets.
- Pre-change sync passed; pytest 150 passed with 3 existing warnings. Prior debt:
  26 Ruff findings, 10 unformatted legacy files, 21 app mypy errors in 8 files.
- Official LangChain references identify langchain-anthropic, langchain-google-genai,
  and langchain-ollama. Exact selected versions/signatures require lock and inspection.

## Scope
Three adapter modules, optional dependency metadata/lock, parametrized cloud tests,
focused Ollama tests, updated import-boundary checks and development/report docs.

## Non-goals
No registry redesign, provider aliases, automatic registration, default registry,
role resolver, model/server requests, health checks, app/LangGraph migration, extra
providers, retrieval changes, or broad legacy cleanup. No Phase 1.1E.

## Architecture decisions
- 2026-10-09: Use function factories in anthropic.py, google.py, and ollama.py,
  imported explicitly like OpenAI/Groq. Canonical names anthropic, google, ollama.
- Preserve groq extra; add anthropic/google/ollama extras only. Resolve with existing
  lock preferences, inspect installed signatures/source, set justified lower bounds,
  and document any existing transitive package change rather than upgrading broadly.
- Cloud keys: ANTHROPIC_API_KEY and supported Google convention after inspection.
  Validate before lazy import; no dotenv loading or generic settings changes.
- Ollama uses OLLAMA_BASE_URL when supplied, otherwise the verified standard local
  endpoint. It requires no key and does not verify server/models or pull/invoke anything.
- Reuse existing errors and LangChain BaseChatModel interface. Use no casts to conceal
  provider incompatibility. Factor common behavior only if it makes implementation simpler.
- Extend existing cloud cases instead of duplicating the suite. Allow matching SDK
  imports only inside concrete factories; generic provider/config/domain rules persist.

## Implementation steps
1. Add optional extras and lock without broad upgrades; temporarily install extras
   for exact constructor/source inspection and type verification.
2. Implement factories using confirmed keywords and existing error semantics.
3. Extend cloud/import tests, add focused local-provider tests, verify offline behavior.
4. Restore default environment, run required gates, inspect diff, document results.

## Test strategy
Existing cloud tests cover forwarding, missing/blank credentials, imports, causes,
safe errors, explicit registration, and prompt composition with a local fake model.
Ollama cases cover default/explicit endpoint, no keys, dependencies, initialization,
LangChain composition, and no connectivity/model checks. Clear credentials/endpoint
variables, block sockets, and use replaced constructors. Cold subprocess forbids
integration imports and network access. Real constructor checks, if practical, use
dummy credentials and guards; never invoke models. Run from root using local uv/cache:
```powershell
uv lock
uv sync --locked --extra anthropic --extra google --extra ollama
uv run --extra anthropic --extra google --extra ollama pytest tests/test_finrag_adapters.py tests/test_finrag_ollama.py tests/test_finrag_architecture.py
uv sync
uv lock --check --offline
uv run pytest
uv run ruff check src/finrag tests/test_finrag_adapters.py tests/test_finrag_ollama.py tests/test_finrag_architecture.py
uv run ruff format --check src/finrag tests/test_finrag_adapters.py tests/test_finrag_ollama.py tests/test_finrag_architecture.py
uv run mypy src/finrag
uv run mypy src/finrag tests/test_finrag_adapters.py tests/test_finrag_ollama.py tests/test_finrag_architecture.py
uv run ruff check . --output-format concise
uv run ruff format --check .
uv run mypy app
git diff --check
```

## Migration/compatibility concerns
Optional integrations absent from default install. Missing-import mypy rules may
cover only new optional integration namespaces, like the existing Groq rule; check
real SDK types while installed too. No persisted data/import-path/config migration.
Model/server availability is not proven by construction. Exception chains remain
debug-only since upstream causes can contain credentials/endpoint values.

## Acceptance criteria
All five explicit factories preserve BaseChatModel; three new integrations optional
and lazy; safe chained errors; cloud credentials required/Ollama keyless; no network
or automatic registration; full pytest and targeted gates green; legacy app unchanged.

## Progress checklist
- [x] Inspect current state and establish baseline.
- [x] Resolve dependencies and inspect selected constructors: Anthropic 1.7.5
  (model_name, temperature, api_key), Google 4.4.0 (model, temperature, api_key,
  vertexai), Ollama 1.1.0 (model, temperature, base_url). All subclass BaseChatModel.
  Google source prioritizes explicit vertexai=False over ambient cloud settings;
  use that fixed backend to preserve the API-key contract. Ollama defaults model
  validation off; pass only base_url with verified default http://127.0.0.1:11434.
  Resolver added 11 packages and changed only existing tenacity 9.2.1 -> 9.1.4:
  google-genai 2.29.0 requires tenacity<9.2.0,>=8.2.3. Sandbox lock attempt failed;
  permitted network retry resolved/downloaded extras, followed by offline re-lock.
- [x] Implement factories and focused offline tests: 46 targeted cases passed.
  Initial Ruff found one changed-test import-order issue; fixed it. Initial format
  check flagged four touched files; formatted only those. Installed-SDK mypy found
  two false missing-argument errors for Anthropic timeout/stop: upstream fields use
  Field(None, alias=...), runtime signature and field defaults both prove None.
  One local ignore[call-arg, unused-ignore] on that constructor avoids speculative
  arguments; return type stays BaseChatModel with no cast/Any. Default absent-SDK
  checking reported the ignore unused, so only that line's unused-ignore is allowed.
- [x] Verify optional/default environments and quality gates. Full default pytest:
  174 passed, 3 unchanged warnings, no optional integrations installed. Real constructors
  for all new providers passed with dummy keys, no invocation, zero socket/DNS attempts.
  Targeted Ruff/format passed (18 files); src mypy passed (15 files), plus changed
  tests (18 files) in default environment. Offline lock check passed. Global Ruff/type
  debt remains 26/21. Format reports 11: the prior 10 legacy Python files plus an
  unchanged committed Phase 1.1C report code fence; do not repair that unrelated file.
- [x] Document results and scope preservation.

## Final results
All acceptance criteria met. Added three precise factories and optional extras,
extended cloud cases (16 additional), and added 8 local-provider cases. Default
pytest 174 passed; targeted optional-environment tests 46 passed. Targeted Ruff/format
passed for 18 files; mypy src passed for 15 and src plus tests for 18, in installed
and absent optional environments. Real construction: zero socket/DNS attempts, no
model invocation. Sync/offline lock/diff checks passed. Only existing dependency
change is required Tenacity 9.2.1 -> 9.1.4, with 11 new optional packages. Documented
one Anthropic constructor typing exception; no interface weakening. Legacy lint/type
debt unchanged (26/21); newly observed existing report snippet brings formatting
findings to 11. See docs/phase-1.1d-baseline.md for full evidence, files, commands,
and limitations. App/registry/config/domain untouched; Phase 1.1E not implemented.
