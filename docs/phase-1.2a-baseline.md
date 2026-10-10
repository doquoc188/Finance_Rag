# Phase 1.2A — grounded generation and question-aware grading

Recorded on 2026-10-09 at starting revision 5a3a548, with a clean working tree.
Read the engineering contracts, Phase 1.1F report, current graph/state/model roles,
and node/workflow tests before editing. The [ExecPlan](../.agent/plans/phase-1.2a.md)
records scope, decisions, the failing input regressions, and verification.

## 1–2. Generation before / after

Documents already reached generate_node through FinGraphState.documents: retrieve
populates them, grade_documents returns its selection, and generation reads that list.
There is no separate context state field. Before this phase, generation constructed
a source/page/text context string and supplied context to chain.invoke, but the only
template message was {question}. Consequently the actual model input was the question
alone; different retrieved contexts did not change it.

Now generation sends a system instruction to answer the financial question using only
supplied evidence and explicitly acknowledge insufficiency rather than invent facts.
The user message contains separate EVIDENCE and QUESTION sections. Evidence is marked
as data, not instructions. Output extraction and citations for nonempty evidence remain
unchanged. This is evidence grounding at prompt-input level, not proof of model obedience,
financial accuracy, citation correctness, or hallucination elimination.

## 3. Context serialization

app/graph/context.py provides serialize_evidence(documents: Sequence[Document]) -> str.
It uses the existing LangChain Document type, with no new runtime document abstraction.
Documents retain their input order and receive numbered source blocks:

```text
[SOURCE 1]
source: report.pdf
page: 14
company: FPT
year: 2023
content:
<verbatim document text>
```

Available scalar source/page/company/year/has_table metadata appears in that fixed
order. Missing, None, or complex values are omitted; no filename/page defaults or
arbitrary object representations are inserted into prompts. Other metadata fields are
not dumped. Content is unchanged, including newlines, 1,250, 12.5%, VND, USD, billion,
and million. This is serialization, not numeric normalization or verification.

## 4. Empty context

When documents is empty or all document text is whitespace/empty, generation returns:

> Không đủ bằng chứng từ tài liệu được cung cấp để trả lời câu hỏi này.

Citations are empty and the generation model is not called. Metadata alone cannot
enable answering. No extra retrieval, web request, fallback model, or provider call
is introduced. For nonempty but inadequate evidence, acknowledgement is a prompt
instruction; offline test models do not establish real model compliance. Existing
graph policy can still populate mock/web placeholder documents before generation;
this phase does not validate their provenance or redesign that routing.

## 5–6. Relevance grading before / after

Before: the node read question but did not use it; GradeDocuments structured output
was invoked on candidate.page_content alone. The question and source metadata were
absent from the model input.

Now: the grading prompt asks whether the candidate contains information relevant to
answering the user's question. Its user message separates QUESTION and CANDIDATE
DOCUMENT, and the prompt composes with model.with_structured_output(GradeDocuments).
The schema is unchanged (binary_score), with no free-text grade parser or new schema.
Each candidate receives its own request with the same question. Source metadata is
not needed for this relevance request and remains outside it.

Generation/grading/rewrite still use their existing injected BaseChatModel roles.
No provider/settings/environment access was added to nodes. The old error heuristic,
document filtering/retention, search flags, and conditional routes remain unchanged.

## 7. Files changed

Created:

- app/graph/context.py: deterministic evidence serializer.
- tests/test_grounded_inputs.py: nine observable input/empty-evidence regressions.
- .agent/plans/phase-1.2a.md: inspected state, decisions, progress, results.
- docs/phase-1.2a-baseline.md: this completion report.

Modified:

- app/graph/nodes.py: corrected generation/grading prompts and empty-evidence result.
- tests/chat_models.py: structured input recording now captures actual message text
  from PromptValue rather than its Python representation; schema validation remains.
- tests/test_graph_runtime.py: replace the old defective-input assertions with role,
  question, evidence, metadata, and structured schema assertions.
- docs/development.md: current corrected flow, prompt-level guarantees, remaining debt.

No graph topology/state/role/provider/retriever/API/startup module changed. Dependency
and untouched runtime paths were explicitly checked for no diff. Historical phase
reports remain snapshots; their old defect observations were not rewritten.

## 8–9. Tests and real compiled-graph evidence

Nine new focused cases inspect actual model requests: question plus financial evidence,
available metadata, different contexts changing input for the same question, omitted
missing/complex metadata, stable formatting/document order, empty/blank evidence
(three cases), changed grading question with identical candidate, and per-candidate
question-aware negative grading. The existing recording WorkflowChatModel is local,
deterministic, returns AIMessage, and supports Pydantic-validated structured grades.
The existing schema/interface tests remain and verify actual GradeDocuments instances.

All nine new cases failed before the production input fix and passed afterward. No
skip/xfail or quality suppression was used. Their financial values are fixtures, not
benchmarks or asserted company facts.

The existing real compiled graph test now checks that every result document's content,
source, and page reaches generation's recorded request, and grading's request contains
the original question and candidate content with GradeDocuments. The relevant-document
scenario proves the corrected flow without rewriting/web; existing rewrite and retry-limit
scenarios retain their role/routing assertions. The graph and production nodes are not
mocked. Existing web tests replace only the external dependency with local results.

The full offline socket/tracing/download guards remain enabled; no provider API, real
credential, external web request, Ollama service, or model download was used. No new
fake model was introduced into production. Input-delivery tests do not measure RAG quality.

## 10. Commands executed

Used .tools/bootstrap/bin/uv.exe with workspace-local UV_CACHE_DIR=.tools/uv-cache and
UV_PYTHON_INSTALL_DIR=.tools/python. Equivalent root commands:

```powershell
uv run pytest
uv run pytest tests/test_grounded_inputs.py --tb=line
uv run pytest tests/test_grounded_inputs.py tests/test_graph_runtime.py tests/test_workflow_model.py --tb=short
uv sync
uv lock --check --offline
uv run pytest
uv run ruff check app/graph/nodes.py app/graph/context.py tests/chat_models.py tests/test_grounded_inputs.py tests/test_graph_runtime.py
uv run ruff format app/graph/nodes.py app/graph/context.py tests/chat_models.py tests/test_grounded_inputs.py tests/test_graph_runtime.py
uv run ruff format --check app/graph/nodes.py app/graph/context.py tests/chat_models.py tests/test_grounded_inputs.py tests/test_graph_runtime.py
uv run mypy --follow-imports=silent src/finrag app/graph/nodes.py app/graph/context.py tests/chat_models.py tests/test_grounded_inputs.py tests/test_graph_runtime.py
uv run mypy src/finrag
uv run mypy app --no-error-summary
uv run ruff check . --output-format concise
uv run ruff format --check .
git diff --check
```

The full-app mypy and broad formatter diagnostics were filtered with PowerShell
Select-String for compact baseline output; their failure exit codes remain visible.
Source/diff audits also checked workflow/state/core/models/providers/runtime/API and
dependency files for preservation. Formatting touches only the five changed Python
files. Scoped mypy checks every changed application/test module explicitly; silent
import following does not exclude those files or add repository configuration ignores.

## 11–12. Results

| Check | Measured result |
| --- | --- |
| Pre-change full pytest | 212 passed, four existing warnings |
| New regressions before production fix | Nine failed, demonstrating both defects and empty-evidence gap |
| Focused corrected node/runtime/schema tests | 31 passed |
| Sync / offline lock check | Passed; 191 resolved packages, 157 installed packages checked |
| Final full pytest | **221 passed**, four existing warnings; no skips/xfails |
| Targeted Ruff / formatting | Passed; all five changed Python files |
| Scoped mypy, including every changed Python module | Passed; 23 source files |
| mypy src/finrag | Passed; 18 source files |
| Diff whitespace / unchanged runtime-dependency paths | Passed |
| Repository Ruff | 13 pre-existing findings, unchanged |
| Repository formatting | Six pre-existing files, unchanged |
| Full legacy app mypy | Eight pre-existing errors in five files, unchanged |

## 13–15. Dependencies, warnings, deferred correctness

No pyproject.toml, uv.lock, requirements, provider SDK, or dependency-group changes.
Four unchanged warnings: legacy Pydantic class Config, API Field(example=...),
langchain-community, and HuggingFaceEmbeddings. No warning filter or broad cleanup.
The eight full-app mypy errors remain in untouched config/schema/parser/embedding/
retriever modules; the 13 broad Ruff findings and six formatting findings also remain
in the previously documented untouched files. No new diagnostic was introduced.

Numeric verification still performs substring notices and reports no hallucination/
useful. Financial arithmetic, citation verification and missing citation metadata
defaults remain unresolved. BM25 is still not queried; hybrid fusion/RRF/reranking,
GraphRAG, incremental indexing, and evaluation expansion remain deferred.

Phase 1.2C policy debt includes retention of original candidates when all are rejected,
question-unaware heuristic grading on errors, an existing web path that adds placeholder
content on failure, mock retrieval evidence, and absence of a corrective fact-check loop.
Shared retrieval state, upload/security/installation debt remains as documented in
Phase 1.1F. Model block-content normalization is unchanged. None is silently fixed here.

This task stops at Phase 1.2A. Numeric verification requires the separately scoped
Phase 1.2B request; it has not begun.
