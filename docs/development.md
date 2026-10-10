# Phase 1.0 development foundation

This phase adds engineering contracts and tooling. Application modules, existing test
assertions, provider defaults, and the legacy runtime installation path are preserved.
It does not implement a retrieval refactor, GraphRAG, or numeric-validation improvements.

Phase 1.0.1 adds an explicit compatible chat-model fixture for workflow tests and
strengthens their assertions. Application source remains unchanged. See the
[green test baseline](phase-1.0.1-baseline.md) for the current test results.

Phase 1.0.2 removes production chat-model and canned-answer fallbacks. Missing model
configuration or provider failures now raise explicit errors; successful provider
selection and constructor arguments are preserved. See the
[Phase 1.0.2 report](phase-1.0.2-baseline.md) for failure semantics and current results.

Phase 1.1A adds `src/finrag` configuration and immutable domain contracts without
migrating the application. See the [foundation report](phase-1.1a-baseline.md) for
environment names, domain constraints, packaging changes, and verification.

## Python and installation

Use Python 3.12 and [uv](https://docs.astral.sh/uv/getting-started/installation/).
From the repository root:

```powershell
uv sync
uv run ruff check .
uv run ruff format --check .
uv run mypy app
uv run pytest
```

`.python-version` selects 3.12. `requires-python` deliberately limits this baseline to
3.12; newer interpreters are not yet validated. `uv.lock` records the resolved runtime,
optional, and development dependencies. Use `uv sync --locked` in automation. The
legacy application still runs directly from the repository. Since Phase 1.1A,
setuptools packages only `src/finrag`, and `uv sync` installs it editably. The existing
`app/` modules remain at their original paths and do not import the new package.

`uv run uvicorn app.main:app --reload` and `uv run streamlit run ui/streamlit_app.py`
retain the existing entry points. Running the application can use configured credentials
and services; the offline restrictions below apply to tests only.

## Dependency decisions

- Runtime requirements retain the existing lower bounds. OpenAI stays in the default
  set because `LLM_PROVIDER` defaults to `openai`, and embeddings also consult its key.
- Groq is lazy-imported and optional: use `uv sync --extra groq` when selecting it.
  Plain `uv sync` omits Groq. Ollama currently uses the existing community adapter and
  needs no new SDK. The standalone demo uses the standard library for its providers.
- Future provider SDKs belong in `project.optional-dependencies`. A future GraphRAG
  extra must be separate; no placeholder SDK or graph dependency is installed now.
- `requests` and `typing-extensions` become explicit dependencies because existing
  modules import them directly. Other existing runtime packages remain, including
  currently unused packages; removing them is a separate dependency audit.
- Ruff, mypy, pytest, pytest-cov, and pre-commit live in the default `dev` dependency
  group. This follows [uv's dependency separation](https://docs.astral.sh/uv/concepts/projects/dependencies/).
- `requirements.txt` remains unchanged for compatibility with current pip/Docker
  workflows, including its Groq and pytest entries. It is a legacy install path, not
  a generated export of the new lock. Docker still targets Python 3.10. Migrating it
  and retiring the duplicate dependency manifest requires a separately scoped change.
- The old requirements had no upper bounds or lockfile. The new lock is a reproducible
  development baseline, not proof that every permitted library version is compatible.

## Quality gates and test isolation

Ruff checks correctness, unused names, and import order, with a 100-column formatter.
Mypy understands the existing namespace-package layout and checks untyped bodies as
well as annotations. Compact diagnostics avoid a Windows encoding crash when pretty
output includes Vietnamese source lines. There are no blanket application exclusions
or error suppressions. A targeted missing-import exception covers only the optional
`langchain_groq` SDK; it does not suppress errors in `app.core.llm`.
Existing lint, formatting, and type debt intentionally remains visible.

Pytest collects `tests/` and adds the repository root to its import path. The root
`test_runner.py` is a manual script with import-time execution and console mutation;
it is intentionally outside collection. No application modules were moved to achieve this.

`tests/conftest.py` sets test credentials/provider settings and selects in-memory Qdrant
before collection. It also sets a valid `DEBUG` value, disables tracing, model downloads,
and socket connections, and uses a temporary Hugging Face cache. The workflow test
explicitly requests `workflow_chat_model`, a real LangChain fake chat-model interface
with fixed test answers and schema-validated relevance grades. It patches only the
model factory reference used by graph nodes and resets shared retriever state for
isolation; pytest restores both afterward. The graph and its nodes still execute.
The retriever smoke test continues to exercise existing fallback paths. These tests
do not validate real embeddings, provider APIs, retrieval quality, or persistent
storage. Tests are neither skipped nor converted to expected failures.
The `integration` marker is reserved, not a network opt-in switch: future live tests
need an explicit separate harness and authorization.

Optional coverage and Git hooks:

```powershell
uv run pytest --cov=app --cov-report=term-missing
uv run pre-commit validate-config
uv run pre-commit install
uv run pre-commit run --all-files
```

Hook installation is an explicit local choice. Hooks use `uv run --locked` and check
only staged Python files; full commands above check the repository. They report
violations without editing files. Existing debt also makes all-file hooks fail.
There is no invented coverage target or claimed evaluation score.

## Existing technical debt observed by source inspection

These are observations, not fixes or acceptance claims:

- `app/core/vectorstore.py` returns vector results when available, otherwise token
  overlap. A BM25 object is constructed but never queried; no rank fusion is applied.
- `app/api/routes.py` accepts company/year but does not pass them into chat retrieval.
  Indexing replaces the current documents and sets `force_recreate=True`; stable IDs,
  incremental updates, concurrency isolation, and idempotent ingestion are absent.
- `app/graph/nodes.py` builds context but its generation prompt contains only the
  question. Document grading does not include the question. The numeric guard logs
  unmatched numbers but always returns no hallucination and a useful answer.
- Phase 1.0.2 deletes `SimpleFakeLLM` and the other canned chat-answer fallbacks.
  Constant embedding fallback, mock retrieval evidence, heuristic grading, and web
  placeholders remain separate retrieval/workflow debt; they are not model-quality evidence.
- LangGraph does not loop back after fact checking. GraphRAG is not implemented.
- PDF failure falls back to decoding bytes as text, not pypdf. Zero-valued table cells
  can become empty strings; character splitting does not preserve table boundaries.
- Upload paths use a client-supplied filename; error/health responses can expose internal
  details, CORS is permissive, and shared retriever state has no tenant isolation.
  These findings require a separate security/ingestion task before external deployment.
- README/UI contain unsupported production, numeric-accuracy, hybrid-search, streaming,
  reranker, and benchmark claims. No supporting evaluation artifacts were found.
  The historical claims must not be cited as validated results.

See the [Phase 1.0 verification baseline](phase-1.0-baseline.md) for the original results,
the [Phase 1.0.1 baseline](phase-1.0.1-baseline.md) for its green offline test suite,
and the [Phase 1.0.2 baseline](phase-1.0.2-baseline.md) for the current failure semantics.
The [Phase 1.1A baseline](phase-1.1a-baseline.md) covers the new configuration/domain code.
Future agents should follow [AGENTS.md](../AGENTS.md) and the
[ExecPlan contract](../.agent/PLANS.md).
