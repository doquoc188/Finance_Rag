# Phase 1.5 — optional Microsoft GraphRAG vertical slice

Implemented on 2026-10-10 from Phase 1.4 commit `fc4eff7`. All automated query
tests use injected local boundaries. No hosted model request, real credential read, or
quality benchmark is included.

## 1. Selected Microsoft GraphRAG version

The optional extra pins `graphrag==3.3.0`, the newest inspected 3.x release. The package
metadata explicitly supports Python 3.11–3.13, including the project's Python 3.12 target.

## 2. Compatibility and dependency reasoning

GraphRAG 3.3.0 resolved with the existing project constraints and its own Pydantic 2.13,
NumPy 2.4, pandas 3.0, PyArrow 25, and split `graphrag-*` packages. Its `openai<3`
constraint changed the universal lock selection from OpenAI 3.26.1 to 2.54.0; this remains
inside `langchain-openai==1.7.0`'s accepted range. The default suite passed after the lock
change. No live provider compatibility claim is made because no provider call was run.

## 3. Optional dependency group

`pyproject.toml` now defines `graphrag = ["graphrag==3.3.0"]` under
`project.optional-dependencies`. A normal `uv sync` does not install it.

## 4. Lockfile changes

`uv.lock` records GraphRAG 3.3.0 and its optional transitive packages. The universal lock
also records OpenAI 2.54.0 because GraphRAG requires OpenAI below 3. Default synchronization
removed 60 GraphRAG-only packages, and `importlib.util.find_spec("graphrag")` returned none.

## 5. `graph_rag` package structure

`src/finrag/graph_rag/` contains `contracts.py`, `errors.py`, `artifacts.py`,
`microsoft.py`, and a small public `__init__.py`. Generic imports load no GraphRAG or pandas
module.

## 6. Project-owned contracts

`GraphQueryMode` has only `LOCAL` and `GLOBAL`. `GraphQueryBackend` is an async protocol.
`GraphQueryResult` contains a text answer, selected mode, and immutable
`GraphProvenance` records with stable context and document IDs. Microsoft DataFrames,
configuration objects, and callbacks do not cross this boundary.

## 7. Microsoft adapter isolation

`MicrosoftGraphRAGBackend` owns artifact conversion and dispatch. It imports pandas,
`graphrag.api`, and the GraphRAG config loader lazily after artifact, workspace, and
credential checks. It has no import or route from `app/runtime.py`.

## 8. LOCAL API used

LOCAL dispatch calls the v3.3.0 async `graphrag.api.local_search` API with entities,
communities, reports, text units, relationships, no covariates, community level zero, and
an explicit response type. There is no Hybrid RAG fallback.

## 9. GLOBAL API used

GLOBAL dispatch calls the v3.3.0 async `graphrag.api.global_search` API with entities,
communities, reports, community level zero, and dynamic community selection disabled. It is
never selected automatically.

## 10. BYOG artifact schema

The project fixture stores JSON tables for documents, entities, relationships, text units,
communities, and community reports. The adapter translates these strict project records to
the exact v3.3.0 DataFrame columns documented in `evals/graphrag/README.md`. JSON keeps
offline validation independent of pandas/PyArrow; DataFrames exist only at the adapter.

## 11. Fixture counts

The synthetic curated fixture contains four documents, five entities, four relationships,
four text units, one community, and one manually written community report.

## 12. Cross-document relationship scenario

The chain Northstar Manufacturing → Expansion Program → Delta Components → Harbor Bank is
supported by three distinct synthetic document IDs. A test verifies both endpoint continuity
and those three source IDs. Names and financial values are explicitly labelled test data.

## 13. Artifact validation rules

The loader rejects missing/invalid JSON, an unsupported manifest, unexpected/missing
fields, blank strings, invalid numbers, empty required entities/relationships, LOCAL without
text units, GLOBAL without reports, duplicate IDs/entity titles/community numbers, unknown
relationship endpoints, and dangling document/entity/relationship/text-unit/community
references.

## 14. GraphRAG settings added

`GraphRAGSettings.root_dir` is a `Path` defaulting to `.finrag/graphrag`. Default settings
still instantiate without artifacts or credentials, and `Settings.env_file` remains `None`.

## 15. `GRAPHRAG_API_KEY` usage

Only the dedicated `GRAPHRAG_API_KEY` is accepted for live adapter use. The adapter checks
the explicit process environment or selected workspace `.env`; it never copies
`OPENAI_API_KEY` or stores a credential in `finrag.config.Settings`.

## 16. `.env` safety behavior

`.gitignore` ignores both the repository `.env` and a workspace `.env`. Tests inject an
explicit mapping or a temporary `.env` containing only `test-not-a-real-key`; they do not
load the developer's file. Diagnostics report configuration failure without key material.

## 17. `.env.example` change

The existing example now includes a names-only `GRAPHRAG_API_KEY=` entry and a comment that
real values remain local. No duplicate example file was created.

## 18. Missing-key behavior

Missing or blank `GRAPHRAG_API_KEY` raises actionable `GraphRAGConfigurationError` before
the optional dependency loader or Microsoft query function can run. There is no fake,
provider, or Hybrid RAG fallback.

## 19. Offline test strategy

Tests use project JSON, a small fake DataFrame, fake config loader, and injected async LOCAL
and GLOBAL functions. A subprocess blocks GraphRAG/pandas imports. Credential tests isolate
the environment, and upstream errors deliberately containing the fake key prove project
error/result/log output does not expose it.

## 20. LOCAL test evidence

The focused test confirms exactly one LOCAL call, zero GLOBAL calls, expected v3.3.0 keyword
arguments, project-owned result type, answer propagation, and entity/relationship source
provenance.

## 21. GLOBAL test evidence

The focused test confirms exactly one GLOBAL call, zero LOCAL calls, disabled dynamic
selection, answer propagation, and report provenance covering all four documents.

## 22. Multi-document test evidence

The fixture test independently verifies the three-edge chain and source sequence
`doc_northstar_2024`, `doc_delta_2024`, and `doc_harbor_2024`.

## 23. Demo command

Offline validation:

```powershell
uv run python evals/graphrag/run_query.py --mode local --question "Which supplier supports the synthetic expansion?"
```

Explicit live GLOBAL after installing the extra and configuring the dedicated variable:

```powershell
uv run --extra graphrag python evals/graphrag/run_query.py --live --mode global --question "Summarize the synthetic cross-document dependency chain."
```

## 24. Real live smoke status

**REAL GRAPHRAG QUERY: NOT EXECUTED.** GLOBAL is technically configured for deliberate live
use. LOCAL dispatch is implemented, but an immediately runnable LOCAL smoke additionally
needs a compatible `entity_description` LanceDB table generated with the configured
embedding model. Phase 1.5 deliberately did not create that billable embedding artifact.

## 25. External API cost

External model API cost incurred: **$0**. No real secret was read and no hosted
completion/embedding request was sent. During package source inspection only, eager
`graphrag.api` import attempted unauthenticated LiteLLM price-map and tiktoken-data downloads;
the restricted environment blocked them. The adapter design now delays that import until an
explicit credentialed live operation.

## 26. Pytest result

The original baseline was 393 passed with two existing Pydantic warnings. Final default and
optional-environment runs both completed with **408 passed, 2 existing warnings**. The 15
new tests are deterministic and offline.

## 27. Ruff, format, and mypy results

Targeted Ruff check and format check pass for all changed Python files. Mypy passes for
`src/finrag` plus `evals/graphrag/run_query.py` (38 files). Repository-wide pre-existing
debt remains unchanged: 8 Ruff findings in four legacy files, 5 format targets, and 6 mypy
errors in four `app` files.

## 28. Files changed

Created the ExecPlan, `src/finrag/graph_rag/`, curated fixture/workspace/CLI/README,
`tests/test_graph_rag.py`, and this report. Updated `.env.example`, `AGENTS.md`,
`pyproject.toml`, `uv.lock`, `src/finrag/config/settings.py`, and its settings tests.
`app/runtime.py` and all normal retrieval/workflow modules are unchanged.

## 29. Limitations

No live answer, provider compatibility result, GraphRAG quality measurement, or performance
claim exists. LOCAL requires a separately prepared compatible entity vector table. The
context translator intentionally retains stable IDs/source provenance rather than upstream
DataFrames or full context text. GraphRAG labels its query API as subject to change, so a
future dependency upgrade needs schema/signature revalidation.

## 30. Deferred work

An explicitly scoped later task may design a small, cost-controlled LOCAL embedding
preparation command and live opt-in smoke procedure. Automatic routing, normal runtime
integration, GraphRAG indexing/extraction, production data migration, and retrieval quality
evaluation remain deferred. Phase 1.6 was not started.
