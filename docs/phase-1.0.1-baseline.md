# Phase 1.0.1 - Green test baseline

## Scope and decision

Read `AGENTS.md` and `.agent/PLANS.md` before editing. Reproduced the original workflow
failure in the locked Python 3.12.15 environment: LangChain rejects `SimpleFakeLLM`
when composing `prompt_template | llm` because it is not a Runnable or callable.

To preserve production runtime behavior, the compatibility fix is confined to tests.
`tests/conftest.py` provides an explicit `workflow_chat_model` fixture based on the
installed LangChain `FakeListChatModel`. It returns actual `AIMessage` objects and
supports prompt composition. A small structured-output adapter returns a Runnable
that validates fixed test grades against the requested Pydantic schema.

The fixture patches `app.graph.nodes.get_llm`, where the imported factory is consumed,
and isolates `global_retriever`. It is not autouse; only tests requesting it receive
the replacement. Pytest restores the patched references afterward. No graph invocation,
retrieval node, generation node, or final workflow result is replaced with a stub.

The existing workflow test retains its original assertions and additionally checks
the exact test response, original question, relevance, retry/search flags, and citation
source/page propagation. Four new test cases check direct/prompt invocation, both
existing grade schemas, and propagation of invalid-schema validation errors.

No application file, provider default, dependency, lockfile, or tool configuration
changed in this phase. There is no architecture refactor or new RAG feature.

## Files changed

- `tests/conftest.py`: opt-in compatible model fixture alongside existing offline isolation.
- `tests/test_rag.py`: request the fixture and strengthen workflow assertions.
- `tests/test_workflow_model.py`: model-interface and schema-validation regression checks.
- `docs/development.md` and this file: current results and test/runtime distinction.

## Verification

Recorded on 2026-10-09, using the workspace-local uv installation and cache settings
documented in [Phase 1.0](phase-1.0-baseline.md#environment-and-reproduction).

| Command | Result |
| --- | --- |
| `uv sync` | Passed; existing locked environment unchanged |
| `uv run pytest tests/test_rag.py::test_finrag_langgraph_invocation` before edits | Failed with the original Runnable TypeError |
| `uv run pytest` after edits | **7 passed**, 3 existing dependency deprecation warnings; no skips/xfails |
| `uv run ruff check tests/conftest.py tests/test_workflow_model.py` | Passed |
| `uv run ruff check . --output-format concise` | 41 existing findings, unchanged |
| `uv run ruff format --check .` | 14 existing files need formatting, unchanged |
| `uv run mypy app` | 24 existing errors in 8 files, unchanged |

Existing application sources have no diff. Existing failures were not suppressed, and
the workflow test was not skipped or weakened. The historical Phase 1.0 report remains
unchanged so its original failing baseline can still be inspected.

## Remaining limits

The production `SimpleFakeLLM` compatibility defect still exists. This phase establishes
a deterministic, compatible model boundary for workflow tests; it does not claim the
application fallback now works with LangChain. Fixed answers and positive test grades
are fixtures, not evidence of financial accuracy or model quality. No live provider,
network service, or model download was used.

Recommended next task: explicitly scope a repair of the application fallback contract
with regression tests before changing runtime behavior. General lint/type cleanup and
the architecture refactor remain separate work.
