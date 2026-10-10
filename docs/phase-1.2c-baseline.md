# Phase 1.2C — bounded retrieval self-correction and graph policy

Implemented on 2026-10-10 from clean commit `b38ef80`. Results below are deterministic
offline regression evidence, not retrieval-quality, model-quality, or production metrics.

## 1. Graph policy before the change

The implementation, rather than README claims, showed this compiled route:

```text
START -> retrieve -> grade_documents
                          | relevant or retry_count >= 2 -> generate
                          | otherwise
                          v
                    transform_query -> web_search -> generate
                                                   -> fact_check -> END
```

Retrieval was never repeated after a rewrite. The integer limit `2` was hard-coded in a
conditional edge. `transform_query` incremented `retry_count`, but the common route could
rewrite only once before immediately using web content. Rejected candidates set
`web_search_needed`; if every candidate was rejected, grading restored the rejected
originals rather than returning an empty list. A partial selection could request fallback
even though relevant evidence existed. The fallback engine mirrored the one-shot web path.
`fact_check` was terminal, so numeric statuses already had no outgoing correction edge.

Baseline: **286 passed, four existing warnings**.

## 2. Graph topology after the change

```text
START -> retrieve -> grade_documents
                    | relevant evidence
                    +---------------------> generate -> fact_check -> END
                    |
                    | no relevant evidence
                    v
              attempts remaining?
                 | yes                 | no
                 v                     v
          transform_query          generate(empty)
                 |                     |
                 +-> retrieve           +-> fact_check -> END
```

Only relevant documents survive grading. Relevant evidence generates immediately,
regardless of other rejected candidates. A failed grade rewrites and returns to the same
retrieval node while budget remains. Neither compiled nor fallback topology registers
web search. The existing helper remains unregistered for import compatibility. The
compiled graph raises its finite LangGraph recursion ceiling from the validated budget,
so a valid larger limit is not cut off by the framework default.

## 3. Maximum rewrite configuration

`RetrievalSettings.max_rewrite_attempts` is a frozen validated integer with default `2`.
It is the maximum number of query-rewrite model invocations. Zero disables rewriting.
Negative values, booleans, and fractional values are rejected. The existing nested
environment convention loads `FINRAG_RETRIEVAL__MAX_REWRITE_ATTEMPTS`; `.env.example`
documents it. Direct graph-builder calls also reject invalid limits defensively.

## 4. Rewrite-attempt state semantics

`rewrite_attempts` is per graph execution. API/manual inputs initialize it to zero;
`retrieve_node` preserves it, and only `transform_query_node` increments it, once after
an actual model invocation. No conditional-edge evaluation increments the counter.
No policy/counter/settings object is module-global. An unchanged rewrite still consumes
an attempt. Reusing one graph for two invocations starts both at their supplied zero.

The externally consumed `retry_count` key remains and mirrors `rewrite_attempts` for
compatibility. Old partial callers that omit the new field fall back to retry_count or
zero. The API continues exposing retry_count, transformed_question, and the legacy
web_search_needed flag; it does not add rewrite_attempts to the HTTP response schema.
The current graph keeps web_search_needed false because no web search is performed.

## 5. Exhausted-budget behavior

When grading leaves no documents and the budget is consumed, routing goes to the existing
Phase 1.2A generation node with `documents=[]`. It returns:

> Không đủ bằng chứng từ tài liệu được cung cấp để trả lời câu hỏi này.

Citations are empty, the generation model is not invoked, and fact_check produces an
empty numeric report with the existing compatibility flags. There is no error, extra
rewrite, web request, alternate provider, or unsupported model-memory answer.

## 6. Numeric contradiction policy

`generate -> fact_check -> END` is unchanged. SUPPORTED, CONTRADICTED, and NOT_FOUND
remain visible through Phase 1.2B's structured report. A CONTRADICTED claim sets the
existing legacy flag but does not rewrite retrieval or regenerate. Retrieval rewriting
addresses evidence relevance failure only; this is not post-generation self-correction.

## 7. Files changed

Created:

- `.agent/plans/phase-1.2c.md`
- `tests/test_graph_policy.py`
- `docs/phase-1.2c-baseline.md`

Modified:

- `src/finrag/config/settings.py`: validated rewrite budget.
- `app/graph/state.py`: explicit attempt field and compatibility comments.
- `app/graph/nodes.py`: preserve/increment attempts and retain only relevant evidence.
- `app/graph/workflow.py`: bounded compiled/fallback routing and no registered web node.
- `app/runtime.py`: pass validated settings to the builder.
- `app/api/routes.py`: initialize per-request attempts without response-schema expansion.
- `.env.example`: document the nested setting and inactive legacy web settings.
- `test_runner.py`: initialize the new field in the manual invocation.
- `tests/chat_models.py`: deterministic per-call grading sequence for policy tests.
- `tests/test_finrag_settings.py`: default/environment/validation/zero regressions.
- `tests/test_graph_runtime.py`: current role route, composition ownership, API boundary,
  and fallback parity.
- `tests/test_grounded_inputs.py`: all-rejected candidates now yield empty evidence.
- `tests/test_numeric_graph.py`, `tests/test_rag.py`: explicit zero initialization.
- `tests/test_finrag_architecture.py`: honest legacy-helper boundary comment.
- `AGENTS.md`, `docs/development.md`: policy contract, topology, limits, and claims.

No application package or existing module was moved or renamed.

## 8. Builder and composition root

The builder is now:

```python
build_finrag_graph(model_roles, max_rewrite_attempts=...)
```

Its default remains two for direct-call compatibility. Production `compose_runtime`
resolves model roles exactly as before, then passes
`configuration.retrieval.max_rewrite_attempts` explicitly. Tests record this handoff,
including zero. Graph modules still import no settings/provider construction, read no
environment variables, and preserve `ModelRoles.query_rewrite` injection.

## 9. Happy-path test

A real compiled graph retrieves one relevant and one rejected local document. Both are
graded; only the relevant document reaches generation. Retrieval and generation occur
once, rewrite is never called, attempts remain zero, the web node is absent from compiled
topology, and the terminal numeric result is SUPPORTED.

## 10. One rewrite then success

The first exact query returns irrelevant local evidence and receives `no`. The rewrite
model runs once, the second exact query retrieves relevant revenue evidence, and grading
returns `yes`. Assertions prove two retrievals/two grades, one rewrite, one generation,
`rewrite_attempts == retry_count == 1`, and that generation receives the successful
evidence without the rejected first document.

## 11. Retry-exhaustion test

With maximum two and a retriever/grader that always fails, the compiled graph performs
three retrievals and grades (initial plus two rewritten queries), exactly two rewrites,
and no third rewrite. Generation-model calls remain zero. The graph terminates with empty
documents/citations, the fixed insufficient-evidence answer, and attempts equal to two.

## 12. Zero-budget test

With maximum zero and irrelevant evidence, the compiled graph performs one retrieval and
grade, no rewrite, and no generation-model call. It terminates through the same safe
empty-evidence and numeric-verification path. The fallback engine has a separate zero-budget
regression proving the same outcome.

## 13. Identical-rewrite termination

The rewrite model returns the original question unchanged. Two invocations of the same
compiled graph each stop at the configured limit of two, proving counters are execution
state rather than graph/model globals. A separate maximum-eight regression performs
exactly eight rewrites and proves the derived LangGraph recursion ceiling preserves a
larger finite configured budget.

## 14. Numeric contradiction regression

Relevant evidence states revenue of 50 billion VND; deterministic generation returns
55 billion VND. The actual Phase 1.2B verifier returns CONTRADICTED and the structured
result remains in graph output. Rewrite calls remain zero and the graph terminates.

## 15. Commands executed

`uv` is not on PATH in this environment. Commands used `.tools/bootstrap/bin/uv.exe`
with workspace-local `UV_CACHE_DIR=.tools/uv-cache` and
`UV_PYTHON_INSTALL_DIR=.tools/python`. Equivalent root commands were:

```powershell
uv run pytest
uv run pytest tests/test_finrag_settings.py tests/test_grounded_inputs.py tests/test_graph_policy.py --tb=short
uv run pytest tests/test_finrag_settings.py tests/test_grounded_inputs.py tests/test_graph_policy.py tests/test_graph_runtime.py tests/test_numeric_graph.py tests/test_rag.py --tb=short
uv sync
uv lock --check --offline
uv run pytest
uv run ruff check <all 15 changed Python files>
uv run ruff format --check <all 15 changed Python files>
uv run mypy --follow-imports=silent src/finrag <all changed application/test Python files>
uv run mypy src/finrag
uv run mypy app
uv run ruff check . --output-format concise
uv run ruff format --check .
git diff --check
git diff --exit-code -- <protected prompt/retrieval/provider/verification/dependency paths>
```

Focused regressions were first run before implementation: 10 expected failures and
42 passes demonstrated the missing setting/builder policy and rejected-document behavior.

## 16. Pytest result

Final full suite: **300 passed, four existing warnings** on Python 3.12.15. Baseline was
286 passed with the same warnings. Ten cases in `test_graph_policy.py` exercise the real
compiled graph/builder; four new settings cases cover configuration. Existing runtime
coverage was updated from the removed web route to the implemented policy without skips,
xfails, weakened assertions, network calls, real credentials, or provider APIs.

## 17. Ruff, formatting, and mypy

Targeted Ruff and formatting pass for all 15 changed Python files. Scoped mypy passes
for 34 source files, including `src/finrag` and every changed application/test module.
The separately requested `mypy src/finrag` passes for 20 files. No type ignores,
configuration exclusions, warning filters, or broad cleanup were added.

Repository-wide baseline debt is unchanged: 13 Ruff findings; six unformatted files;
eight mypy errors in five untouched legacy application files. `git diff --check` passes;
reported LF/CRLF notices are Git working-copy warnings, not whitespace failures.

## 18. Dependencies and lockfile

`uv sync` and `uv lock --check --offline` pass: 191 packages resolved and 157 installed
packages checked. `pyproject.toml`, `uv.lock`, `requirements.txt`, SDKs, provider/model
settings, embeddings, and optional dependency groups are unchanged.

## 19. Remaining warnings and legacy debt

The four warnings remain: legacy Pydantic class Config, deprecated Pydantic Field example,
`langchain-community`, and `HuggingFaceEmbeddings`. The existing global retriever and mock
retrieval fallback remain. The document-grading exception heuristic remains. The inactive
legacy web helper remains importable but is absent from both graph engines. Existing API
error disclosure, upload paths, CORS, indexing/idempotency, Docker/Python compatibility,
README/UI feature claims, and all prior numeric-verifier limitations remain documented.

## 20. Intentionally deferred

No generation retry/regeneration, LLM hallucination grader, numeric-triggered rewrite,
hybrid retrieval, BM25 use, RRF, reranking, GraphRAG, query routing, web search, provider
fallback/retry, citation verification, financial derivation, incremental indexing, API
redesign, or broad lint/type cleanup was implemented. Phase 1.3 has not begun.
