# Optional Microsoft GraphRAG vertical slice

This directory contains a small, manually curated **synthetic** finance graph for the
explicit Phase 1.5 Microsoft GraphRAG integration. Its names, amounts, percentages, and
relationships are test data. They are not verified facts about real organizations.

## What is included

`fixtures/curated_finance/` stores project-owned JSON tables for four documents, five
entities, four relationships, four text units, one community, and one community report.
The adapter converts them at the Microsoft boundary to the GraphRAG 3.3.0 columns used by
`graphrag.api.local_search` and `graphrag.api.global_search`:

- entities: `id`, `human_readable_id`, `title`, `type`, `description`,
  `description_embedding`, `degree`, `text_unit_ids`;
- relationships: `id`, `human_readable_id`, `source`, `target`, `description`, `weight`,
  `combined_degree`, `text_unit_ids`;
- text units: `id`, `text`, `document_id`, `entity_ids`, `relationship_ids`,
  `covariate_ids`, `n_tokens`;
- communities: `id`, `community`, `title`, `level`, `parent`, `children`, `entity_ids`,
  `relationship_ids`, `text_unit_ids`;
- community reports: `id`, `community`, `title`, `summary`, `full_content`, `rank`, `level`.

The chain `Northstar Manufacturing → Expansion Program → Delta Components → Harbor Bank`
is supported by three different synthetic documents. The community report is also manually
written, so fixture creation performs no LLM extraction or indexing.

`workspace/settings.yaml` follows the Microsoft GraphRAG 3.3.0 schema. Both hosted model
entries reference `${GRAPHRAG_API_KEY}`; no resolved credential is committed.

## Offline validation

The CLI validates only unless `--live` is present. This command makes no API call and does
not need the optional GraphRAG dependency or a credential:

```powershell
uv run python evals/graphrag/run_query.py --mode local --question "Which supplier supports the synthetic expansion?"
```

Expected status includes `REAL GRAPHRAG QUERY: NOT EXECUTED`. GLOBAL artifacts can be
validated by changing `--mode local` to `--mode global`.

## Deliberate live execution

Install the optional environment explicitly:

```powershell
uv sync --extra graphrag
```

Choose one credential path. For a workspace-local file, create the ignored
`evals/graphrag/workspace/.env` with `GRAPHRAG_API_KEY=<user-secret>`. The upstream config
loader reads that file. To use the repository-root `.env` for this manual operation, pass
it explicitly to `uv`:

```powershell
uv run --extra graphrag --env-file .env python evals/graphrag/run_query.py --live --mode global --question "Summarize the synthetic cross-document dependency chain."
```

For the current PowerShell session, enter the value without displaying it and then run the
same command without `--env-file`:

```powershell
$env:GRAPHRAG_API_KEY = Read-Host -MaskInput "GRAPHRAG_API_KEY"
uv run --extra graphrag python evals/graphrag/run_query.py --live --mode global --question "Summarize the synthetic cross-document dependency chain."
```

Missing or blank `GRAPHRAG_API_KEY` fails before Microsoft imports or query dispatch. The
adapter never copies `OPENAI_API_KEY`, selects a fallback provider, returns a fake answer,
or falls back to Hybrid RAG.

## LOCAL live prerequisite

Microsoft GraphRAG 3.3.0 LOCAL search maps a question through an entity-description vector
store. The curated repository fixture deliberately does not contain hosted-model embeddings
and Phase 1.5 does not run the upstream embedding workflow. A live LOCAL query therefore
requires a compatible `entity_description` LanceDB table under the workspace's
`output/lancedb` created with the configured embedding model. After that explicit,
potentially billable preparation, the dispatch command is:

```powershell
uv run --extra graphrag python evals/graphrag/run_query.py --live --mode local --question "Which supplier supports the synthetic expansion?"
```

GLOBAL is the immediately runnable live slice because its manually curated community report
does not require that entity vector table. No live command is part of pytest or the normal
quality gates.

## Recorded GLOBAL live evidence

On 2026-10-10, one deliberate GLOBAL smoke used Microsoft GraphRAG 3.3.0. Offline validation
reported four documents, five entities, four relationships, four text units, and one
community report. The hosted query returned a response and provenance containing all four
curated document IDs. No credential appeared in the observed terminal output, and cost was
limited to this one small deliberate hosted query.

This establishes live integration capability only. It does not measure answer quality or
show that GraphRAG outperforms the normal hybrid path. A real LOCAL query has not been run.
Do not commit terminal screenshots containing credentials, private content, personal data,
or avoidable local absolute paths.

## Isolation and limits

GraphRAG is not connected to `app/runtime.py`. The normal Dense + BM25 → RRF → optional
reranker path remains the application behavior. The fixture proves schema validation,
dispatch, result translation, and multi-document provenance offline with mocked Microsoft
calls; the recorded GLOBAL smoke additionally proves a real hosted response was returned.
Neither is a retrieval-quality benchmark. The public
GraphRAG query API warns that compatibility may change, so the package and workspace schema
are pinned together at 3.3.0.
