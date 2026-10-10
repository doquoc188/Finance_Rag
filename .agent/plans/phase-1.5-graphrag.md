# Phase 1.5 — optional Microsoft GraphRAG vertical slice

## Objective

Add an explicit, isolated Microsoft GraphRAG 3.x query capability with LOCAL and GLOBAL
modes, a small curated synthetic finance graph, offline validation/tests, and a deliberate
live CLI. Keep the existing Hybrid RAG application runtime and default installation
unchanged. Normal development checks must not load `.env`, use a real credential, contact
an external service, or incur API cost.

## Current-state observations

- Baseline commit is `fc4eff7`; `UV_CACHE_DIR=.uv-cache` is needed in this restricted
  workspace because the default user cache is not writable.
- Baseline `uv run pytest -q` completed with 393 passed and two pre-existing Pydantic
  deprecation warnings. Plain `uv` is not on this shell's `PATH`; the repository bootstrap
  executable is `.tools/bootstrap/bin/uv.exe`.
- `GraphRAGSettings` is disabled by default and stores only `enabled` and `index_name`.
  `Settings` deliberately has `env_file=None` and contains no credentials.
- `app/runtime.py` always composes `HybridFinancialRetriever`; it has no GraphRAG imports or
  routing. Phase 1.4 established dense → BM25 → RRF → optional reranker behavior.
- Microsoft GraphRAG v3.3.0 supports Python 3.12. Its public async query surface is
  `graphrag.api.local_search(...)` and `graphrag.api.global_search(...)`; each accepts
  GraphRAG config plus artifact DataFrames and returns `(answer, context)`.
- Current Microsoft BYOG data uses entity, relationship, text-unit, community, and
  community-report tables. LOCAL requires entity/relationship/text-unit context; GLOBAL
  requires communities and community reports.
- The current GraphRAG YAML schema uses `completion_models` and `embedding_models` and
  supports `${GRAPHRAG_API_KEY}` interpolation. No GraphRAG package is installed in the
  current default environment.
- `.env` is gitignored and contains user secrets. It must never be read by tests or printed.
  The existing `.env.example` needs a names-only `GRAPHRAG_API_KEY` entry.

## Scope

- Add the optional `graphrag` dependency extra and lock it without making it a default
  runtime dependency.
- Add project-owned contracts, fixture values/validation, errors, and a lazy Microsoft
  adapter under `src/finrag/graph_rag/`.
- Add only a `Path` workspace field to `GraphRAGSettings` if needed; keep credentials out.
- Add a curated fixture, matching safe workspace template, offline tests, CLI, and Phase
  1.5 documentation.
- Update the engineering contract only to describe the implemented isolation boundary.

## Non-goals

- No automatic routing, application-chat integration, runtime replacement, GraphRAG
  indexing pipeline, LLM extraction, provider registry changes, or production-scale data.
- No live query, paid embedding, workspace initialization, or use of the developer's real
  environment values.
- No retrieval tuning, evaluation claims, broad lint/type cleanup, or later-phase work.

## Architecture decisions

- **2026-10-10:** Pin Microsoft GraphRAG `3.3.0` in the `graphrag` optional extra. Confirm
  resolution under Python 3.12 before retaining the pin.
- **2026-10-10:** Public `finrag.graph_rag` imports only standard-library/project code.
  Microsoft GraphRAG and pandas imports occur lazily inside the concrete adapter.
- **2026-10-10:** Store the tiny fixture as deterministic JSON with project-owned records,
  then translate it to the exact GraphRAG v3.3 DataFrame columns at the adapter boundary.
  This permits default/offline validation without pandas or GraphRAG.
- **2026-10-10:** Treat concrete backend queries as live and check only
  `GRAPHRAG_API_KEY` before loading GraphRAG configuration or invoking query functions. The
  CLI defaults to artifact validation and constructs the backend only when `--live` is set.
- **2026-10-10:** Translate upstream context to immutable project-owned provenance IDs;
  never return upstream DataFrames/config objects, credentials, or exception text.
- **2026-10-10:** Inject narrow local/global async call boundaries in tests. Default adapter
  resolution remains lazy so dependency absence is clear only when concrete use starts.

## Implementation steps

1. Add and resolve the optional GraphRAG v3.3.0 dependency; inspect the installed public API,
   config loader, and BYOG consumers to confirm signatures/columns.
2. Implement immutable query contracts/errors and strict fixture loading/validation.
3. Implement the lazy Microsoft adapter, credential preflight, DataFrame translation,
   LOCAL/GLOBAL dispatch, safe errors, and result translation.
4. Add the four-document synthetic fixture and a GraphRAG v3.3.0 settings template using
   `${GRAPHRAG_API_KEY}`. Validate cross-source relationship provenance.
5. Add the validation-first CLI and usage/limitations documentation.
6. Add offline tests for imports, dependency absence, dispatch, translation, malformed
   artifacts, credential isolation/redaction, fixture provenance, and unchanged runtime.
7. Run the acceptance checks and update this plan's checklist/final results with evidence.

## Test strategy

- Re-run the full offline pytest suite with no `--env-file` and no live flag.
- Block/replace the exact Microsoft query functions in adapter tests; use only
  `test-not-a-real-key` in isolated mappings and explicitly empty mappings for missing-key
  tests.
- Exercise all required invalid artifact cases with small temporary JSON fixtures.
- Test generic imports in a subprocess whose import guard rejects `graphrag` and pandas.
- Assert runtime composition is still `HybridFinancialRetriever` and has no GraphRAG field.
- Run targeted Ruff/format, mypy for `src/finrag` and the CLI, lock/offline checks, default
  import without the extra, and `git diff --check`.

## Migration/compatibility concerns

The default dependency set, public application imports, existing indexes, stored manifests,
and runtime behavior remain unchanged. Installing the `graphrag` extra introduces its own
pinned transitive stack and may update packages in that environment; it is verified
separately and removed again with a default `uv sync`. The curated fixture is demonstration
data, not a migration path for existing Qdrant data. GraphRAG minor releases may change YAML
or DataFrame schemas, so the version and template stay pinned together.

## Acceptance criteria

- `import finrag.graph_rag` succeeds without the optional package.
- Validated LOCAL and GLOBAL calls dispatch only to their corresponding Microsoft v3.3.0
  API boundary and return `GraphQueryResult`.
- Required malformed artifacts fail before Microsoft code or network access.
- A missing/blank `GRAPHRAG_API_KEY` fails before config loading/query dispatch, and fake
  credentials never appear in results, errors, logs, or committed files.
- The fixture has 3–5 synthetic documents and a tested relationship chain spanning sources.
- The CLI separates offline validation from explicit `--live` execution.
- Default runtime remains Hybrid RAG; default imports/install do not require GraphRAG.
- Full pytest and changed-file quality checks pass offline at $0 external API cost.

## Progress checklist

- [x] 2026-10-10: Read required contracts, baselines, configuration, domain, and runtime.
- [x] 2026-10-10: Inspect current Microsoft GraphRAG v3.3.0 docs/source API and BYOG schema.
- [x] 2026-10-10: Record clean baseline: 393 passed, two legacy warnings.
- [x] 2026-10-10: Resolve/pin optional dependency and confirm installed v3.3.0 interfaces.
- [x] 2026-10-10: Implement project contracts, fixture validation, and Microsoft adapter.
- [x] 2026-10-10: Add fixture, workspace template, CLI, documentation, and offline tests.
- [x] 2026-10-10: Run all required verification and default/optional compatibility checks.

## Final results

- Added optional `graphrag==3.3.0`, locked its transitive stack, and verified its installed
  Python metadata, async LOCAL/GLOBAL signatures, indexer adapters, and settings schema.
  GraphRAG's `openai<3` constraint selected OpenAI 2.54.0 in the universal lock; all default
  tests passed, but no live provider compatibility claim was made.
- Added project contracts/errors, strict standard-library artifacts, lazy Microsoft
  conversion/dispatch, four-document curated data, v3.3.0 workspace YAML, validation-first
  CLI, documentation, and 15 offline tests. `app/runtime.py` was not changed.
- Baseline: 393 passed, two legacy Pydantic warnings. Final optional and default runs:
  408 passed with the same warnings. Targeted Ruff/format passed; mypy passed for 38
  `src/finrag` and demo files; offline lock and default imports passed; `git diff --check`
  passed.
- Repository-wide legacy results remain 8 Ruff findings in four files, 5 format targets,
  and 6 mypy errors in four `app` files. These match the Phase 1.4 debt and were not edited.
- No real `.env` was loaded, no secret was printed or committed, and no hosted model API
  call ran. External API cost was $0. An eager upstream import attempted ancillary public
  metadata/tokenizer downloads during inspection; the sandbox blocked both.
- GLOBAL is technically ready for a deliberate manual live query. LOCAL dispatch is ready,
  but actual LOCAL execution still needs a compatible prebuilt `entity_description`
  LanceDB table; creating it was intentionally deferred because it invokes embeddings.
