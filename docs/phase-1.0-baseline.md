# Phase 1.0 verification baseline

Recorded on 2026-10-09 against starting revision `8b726db` on Windows/PowerShell.
This is an engineering baseline, not an evaluation or production-readiness report.

## Changes and scope

Created:

- `AGENTS.md`: engineering contract, boundaries, correctness rules, definition of done.
- `.agent/PLANS.md`: requirements for future substantial-refactor ExecPlans.
- `pyproject.toml`, `.python-version`, `uv.lock`: Python 3.12 application environment,
  runtime/provider/dev dependency separation, pytest/Ruff/mypy/coverage settings.
- `.pre-commit-config.yaml`: non-mutating Ruff hooks using the locked tool version.
- `tests/conftest.py`: test-only environment, network, and model-cache isolation.
- `docs/development.md` and this file: usage, decisions, limitations, and results.

Modified `.gitignore` for environments/caches/coverage/secrets and added a Phase 1.0
notice to `README.md`. Existing application files, tests in `tests/test_rag.py`, the
manual test runner, runtime requirements, Docker files, and module paths are unchanged.
No RAG feature, architecture migration, or benchmark was added.

## Environment and reproduction

Neither `uv` nor Python 3.12 was initially available on PATH. The first bootstrap
attempt could not reach PyPI inside the sandbox. A permitted network-enabled retry
installed uv 0.12.24 into ignored `.tools/bootstrap`; uv downloaded CPython 3.12.15
into `.tools/python` and created `.venv`. No existing global/Conda environment changed.

The verification commands used that local executable. To reuse it in this checkout:

```powershell
$env:PATH = (Join-Path $PWD '.tools\bootstrap\bin') + ';' + $env:PATH
$env:UV_CACHE_DIR = Join-Path $PWD '.tools\uv-cache'
$env:UV_PYTHON_INSTALL_DIR = Join-Path $PWD '.tools\python'
$env:PRE_COMMIT_HOME = Join-Path $PWD '.tools\pre-commit'
```

With a normal uv installation these workspace-specific paths are unnecessary.
Reproduction after checkout requires `uv sync --locked` and the commands below.
The raw lint/format/type/test outputs from this run are local ignored `.tools/*.txt`
artifacts; this committed summary records the outcomes without requiring those files.

Locked development versions: Ruff 0.16.10, mypy 2.4.0, pytest 9.1.1, pytest-cov 7.1.0,
pre-commit 4.6.2. The resolved runtime includes LangChain 1.4.4, langchain-core 1.6.9,
LangGraph 1.2.14, and Pydantic 2.14.0. These satisfy the old open-ended requirements;
they are not evidence of compatibility across all earlier/later releases.

## Commands and final results

| Command | Result | Classification |
| --- | --- | --- |
| `uv sync` | Passed; 180 packages resolved, 156 installed for the default Windows environment | New environment works |
| `uv lock --check --offline --cache-dir .tools/uv-cache` | Passed | Lock matches project metadata |
| `uv run ruff check . --output-format concise` | Failed: 41 findings | Existing source/import/style debt |
| `uv run ruff format --check .` | Failed: 14 files need formatting | Existing formatting debt; files not rewritten |
| `uv run mypy app` | Failed: 24 errors in 8 files; 10 application files checked | Existing source/library compatibility debt exposed by the new gate |
| `uv run pytest` | 2 passed, 1 failed; no skips/xfails | Existing mock/workflow incompatibility |
| `uv run ruff check tests/conftest.py` | Passed | New Python code clean |
| `uv run ruff format --check tests/conftest.py` | Passed | New Python code formatted |
| `uv run pre-commit validate-config` | Passed with workspace cache | Hook configuration valid |
| `uv run pre-commit run --files tests/conftest.py` | Both hooks passed | Locked local hooks execute correctly |
| `git diff --check` | Passed | No whitespace errors in tracked edits |

Additional inspection verified that all 24 non-Groq entries in the legacy
`requirements.txt` are satisfied by installed versions. `langchain_groq` is absent
from the default environment, as intended, and present in the optional lock graph.
The optional Groq SDK/provider was not installed or exercised live.

## Failure evidence and configuration decisions

- Ruff findings are import order, unused imports/locals, one unnecessary f-string,
  and imports following the existing `sys.path` setup in `app/main.py`. All reported
  paths existed before this task. No application file is excluded to obtain a green gate.
- Type errors include fallback class redefinitions, implicit optional parameters,
  Pydantic/OpenAI constructor signatures, missing legacy `langchain_core.pydantic_v1`,
  untyped `rank_bm25`, dictionary updates into a TypedDict, and optional upload filenames.
  The first completed check had 25 errors: one was the deliberately uninstalled Groq
  SDK. A targeted missing-import override removes only that newly introduced diagnostic;
  the final count is 24. No application type error is blanket-suppressed.
- The failing test is `tests/test_rag.py::test_finrag_langgraph_invocation`.
  `app/graph/nodes.py:111` composes `prompt_template | llm`; LangChain raises
  `TypeError: Expected a Runnable, callable or dict` for `SimpleFakeLLM`.
  Both that implementation and the test assertion predate this task and remain unchanged.
  The test is left failing visibly rather than replacing the model with a passing fake.
- Initial collection failed because the host supplied `DEBUG=release`, which is invalid
  for the existing boolean setting. The test harness now overrides this value, provider
  credentials, and storage/model settings before imports. No production setting changed.
- Initial mypy pretty output crashed with `UnicodeEncodeError` while printing Vietnamese
  source through a Windows `cp1252` stream. Disabling pretty output fixed the tool crash;
  it did not change type checking or require downgrading mypy.
- Pre-commit initially could not write its default cache outside the sandbox. Setting
  `PRE_COMMIT_HOME` inside `.tools` resolved it. No Git hooks were installed implicitly.
- Pytest collection is restricted to `tests/`: the root `test_runner.py` executes on
  import and mutates stdout, so it is a manual runner rather than a pytest test module.

There was no previous lockfile or quality-gate configuration to establish an earlier
green baseline. These counts describe unchanged application source under the new
locked environment; they do not claim every finding occurred under an unknown prior
installation. The observed Phase 1.0 tooling issues above were resolved; outstanding
application issues remain explicitly documented.

## Limits and next task

Only offline smoke tests ran. No provider API, model download, GPU, external Qdrant,
Docker runtime, retrieval evaluation, financial-accuracy evaluation, or coverage target
was validated. Passing parser/retriever smoke tests is not a retrieval-quality result.
See [development debt](development.md#existing-technical-debt-observed-by-source-inspection)
for source-level findings, including destructive indexing and unsupported README claims.

Recommended next task: a separately scoped compatibility/baseline cleanup, starting
with the mock/Runnable failure and regression tests, then existing lint/type debt.
Plan the larger architecture refactor only after its behavior and compatibility
acceptance criteria are explicit. Phase 1.1 and later work have not been implemented.
