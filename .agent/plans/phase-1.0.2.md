# Phase 1.0.2: remove production fake-model fallback

## Objective
Fail explicitly when a configured chat model cannot be used; keep the successful
provider paths and explicit test-only model injection working. No Phase 1.1 work.

## Current-state observations and pre-change audit (2026-10-09)
Read `AGENTS.md`, `.agent/PLANS.md`, and `docs/phase-1.0.1-baseline.md` before changes.
The working tree already contains the previous phase's edits to `tests/test_rag.py`
and `docs/phase-1.0.1-baseline.md`; preserve them. Baseline: 7 tests pass, 39 Ruff
findings, 13 unformatted files, and 24 mypy errors.

All production `SimpleFakeLLM` references before editing:

| Location (original lines) | Activation and classification |
| --- | --- |
| `app/core/llm.py:5` | Class definition: fixed financial answer and always-positive structured grades; no legitimate production use |
| `app/core/llm.py:61` | `get_llm` final `else`: OpenAI key absent or starts with `mock`; Groq key absent; unsupported/empty provider. Missing configuration or invalid provider selects a fake |
| `app/core/llm.py:63` | `except Exception`: provider imports (including missing SDK/transitive dependencies) or model construction fail. Invalid model/key settings and constructor/runtime errors can all reach this return |

`get_llm` is consumed by document grading, generation, and query rewriting in
`app/graph/nodes.py`. Initialization occurs before the grading node's heuristic
exception handler, so an explicit initialization error will propagate.

Other model-like canned response paths found:

- `app/graph/nodes.py:12-22`: missing `langchain_core.prompts` installs a fake
  `ChatPromptTemplate` whose composition discards the model and whose invocation
  returns a fixed financial answer. Remove this import fallback, without changing prompts.
- `simple_app.py:72-84`: the standalone demo tries its existing Groq then Gemini HTTP
  paths; absent keys, empty answers, or failures eventually return a fixed financial
  answer. Remove only that fake terminal result. Preserve successful HTTP requests,
  response parsing, and the existing real-provider retry order. Existing helpers print
  raw exception text and drop its cause; replace this with safe chained errors.

Related paths deliberately outside the chat-model change:

- `get_embeddings` catches errors and returns constant `DummyEmbeddings`: retrieval debt.
- Document/Pydantic/settings import shims and `FallbackStateGraph`: not chat-model
  construction. Mock retrieval evidence, heuristic grading, web placeholders, and numeric
  checks are existing retrieval/workflow debt. Do not redesign them in this phase.
- `test_runner.py` reports exceptions; it does not construct a fake. UI/API code exposes
  exception messages but does not select models. Existing tests explicitly inject
  `WorkflowChatModel` in `tests/conftest.py`.

## Scope
`app/core/llm.py`, a small exception module, the fake prompt import branch in
`app/graph/nodes.py`, the standalone demo's model failure paths, narrow tests, and
phase documentation. Format/lint only touched Python files as required.

## Non-goals
No provider registry, new providers, async support, graph redesign, prompt changes,
retrieval/indexing changes, numeric validation redesign, global lint/type cleanup,
module moves, or README architecture rewrite.

## Architecture decisions
Keep the existing provider conditional. Add only `ModelConfigurationError` and
`ProviderUnavailableError` in a standard-library-only exception module shared with
the standalone demo. Validate required settings before imports; distinguish import
and initialization failures. Preserve exception causes with `raise ... from ...`;
public messages name settings/packages, never secret values or raw provider errors.
Delete `SimpleFakeLLM`, rather than making it Runnable-compatible.

## Implementation steps
1. Record this audit before behavior edits.
2. Remove the three chat canned-answer paths; preserve successful provider arguments.
3. Add offline configuration/import/initialization/constructor-forwarding tests plus
   the existing real compiled workflow tests. Test that errors propagate to callers.
4. Run pytest, targeted Ruff/format and mypy checks, repository baseline checks,
   `git diff --check`, and a final production reference search.
5. Document exact results and remaining debt without claiming live provider validation.

## Test strategy
Monkeypatch settings and external SDK constructors/imports at their boundaries; never
call a real API or use a real key. The existing socket/network guard stays enabled.
Assert exception types, safe actionable text, preserved causes, constructor arguments
and return identity on success, and continued execution of the compiled graph.
Exercise missing prompt imports and the standalone demo's missing/failing providers.

## Migration/compatibility
Deployments that previously returned a fabricated answer now raise explicit errors.
Existing OpenAI/Groq/Ollama selection, model settings, temperature and credentials
remain the same on successful initialization; no dependency/lock/index migration.
The demo retains its existing real-provider retry order. Exception chains are for
debugging and must not be serialized as public error text because SDK causes may
contain sensitive details. Historical baseline reports remain snapshots.

## Acceptance criteria
Full pytest passes; no chat runtime path selects a fake; missing settings and provider
failures are explicit; test fakes remain test-only; successful constructor contracts
are verified without network; targeted Ruff/format passes; new code adds no type debt.

## Progress checklist
- [x] 2026-10-09: inspect instructions, baseline, and all production fake-model paths.
- [x] 2026-10-09: remove silent chat fallback paths and introduce narrow errors.
- [x] 2026-10-09: add 38 offline regression cases; all 45 tests pass.
- [x] 2026-10-09: record targeted quality results and remaining debt.

## Final results
See `docs/phase-1.0.2-baseline.md` for the complete 12-part handoff, file list, and
exact commands. Six changed Python files pass Ruff lint/format; the three new
exception/test modules pass scoped mypy. Full application mypy still reports 21
existing errors; repository Ruff reports 26 findings and 10 unformatted files.
The three pre-existing deprecation warnings remain. No live providers were called.
An AST comparison verified unchanged logic in 11 out-of-scope functions/classes.
No dependencies, registry, prompts, graph edges, or indexes changed.
