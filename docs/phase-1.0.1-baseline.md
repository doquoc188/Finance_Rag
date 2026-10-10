# Phase 1.0.1 - Green test baseline

## Scope and decision

Read `AGENTS.md` and `.agent/PLANS.md` before editing. Reproduced the original workflow
failure in the locked Python 3.12.15 environment: LangChain rejects `SimpleFakeLLM`
when composing `prompt_template | llm` because it is not a Runnable or callable.

The production interfaces were verified directly in `app/graph/nodes.py`,
`app/graph/workflow.py`, and `app/api/routes.py`:

| Consumer | Required interface |
| --- | --- |
| `generate_node` | A component that LangChain can compose with `ChatPromptTemplate` using `|`; the resulting chain must support synchronous `.invoke()` |
| Generated/rewrite response | Reads `.content` when present, otherwise converts the response to a string; `AIMessage` satisfies this interface but is not explicitly required by application type checks |
| `grade_documents_node` | `.with_structured_output(GradeDocuments).invoke(document_text)` returning an object with a string `binary_score` |
| `transform_query_node` | Direct synchronous `.invoke(string)` |
| API/workflow | Calls `finrag_app.invoke(...)`; no production `.ainvoke()` call exists |

The current fact-check node does not call a model. `GradeHallucination` is covered by
the fixture's schema regression tests but is not consumed by the production grader.
Subclassing a chat model is a convenient way to supply the observed Runnable/message
contracts; the workflow does not explicitly require `BaseChatModel` inheritance.

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
source/page propagation. It also asserts that the compiled LangGraph runs and that
the final fact-check node populates its state fields; those assertions do not certify
numeric correctness. Four new test cases check direct/prompt invocation, both
existing grade schemas, and propagation of invalid-schema validation errors.

No application file, provider default, dependency, lockfile, or tool configuration
changed in this phase. There is no architecture refactor or new RAG feature.

## Files changed

- `tests/conftest.py`: opt-in compatible model fixture alongside existing offline isolation.
- `tests/test_rag.py`: request the fixture, strengthen workflow assertions, and clean
  imports/formatting only in this touched test file.
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
| `uv run ruff check tests/conftest.py tests/test_rag.py tests/test_workflow_model.py` | Passed for all three Phase 1.0.1 Python files |
| `uv run ruff format --check tests/conftest.py tests/test_rag.py tests/test_workflow_model.py` | Passed for all three files |
| `uv run ruff check . --output-format concise` | 39 remaining existing findings |
| `uv run ruff format --check .` | 13 untouched files need formatting |
| `uv run mypy app` | 24 existing errors in 8 files, unchanged |
| `git diff --check` | Passed |

The earlier targeted check omitted `tests/test_rag.py`. Rechecking all phase-touched
Python files found two existing import findings and formatting debt in that file.
Removing its unused import, sorting imports, and running `uv run ruff format
tests/test_rag.py` resolved this acceptance gap. The repository-wide counts decreased
from 41 to 39 lint findings and from 14 to 13 files needing formatting; no unrelated
application file was reformatted.

Existing application sources have no diff. Existing failures were not suppressed, and
the workflow test was not skipped or weakened. The historical Phase 1.0 report remains
unchanged so its original failing baseline can still be inspected.

## Remaining limits

The production `SimpleFakeLLM` compatibility defect still exists. This phase establishes
a deterministic, compatible model boundary for workflow tests; it does not claim the
application fallback now works with LangChain. The pre-existing broad exception handler
and silent fake fallback in `app/core/llm.py` remain documented runtime debt; this task
introduces no production fallback and leaves runtime behavior unchanged. Fixed answers and positive test grades
are fixtures, not evidence of financial accuracy or model quality. No live provider,
network service, or model download was used.

Recommended next task: explicitly scope a repair of the application fallback contract
with regression tests before changing runtime behavior. General lint/type cleanup and
the architecture refactor remain separate work.
