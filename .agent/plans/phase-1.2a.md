# Phase 1.2A — grounded inputs and question-aware relevance

## Objective
Supply retrieved evidence to generation and question + candidate document to the
existing structured relevance grader, preserving injected ModelRoles and graph policy.

## Current-state observations
2026-10-09, clean checkout at 5a3a548. Read AGENTS.md, PLANS.md, the 1.1F report,
graph/state/nodes, models, and workflow/test-model tests before edits. Baseline pytest:
212 passed, four existing warnings. FinGraphState.documents carries LangChain Documents
from retrieve through grade_documents to generate. Generation builds source/page/text
context, but its only prompt message is {question}; unused context in chain.invoke is
never sent. Grading reads but ignores question and invokes GradeDocuments structured
output on d.page_content alone. No context field exists in state. Old generation has
no explicit empty-evidence behavior. Existing tests assert the defective old inputs.

## Scope
app/graph/nodes.py, one small context serializer, current recording test support,
focused node/compiled-graph tests, development docs, this plan and a completion report.

## Non-goals
No graph edges/routing/retry/rewrite/retrieval/provider/state schema changes; no numeric
redesign, citation parser, hybrid retrieval, web additions, GraphRAG, or broad cleanup.

## Architecture decisions
- serialize_evidence uses existing LangChain Documents, stable source numbering and
  fixed-order available source/page/company/year/has_table metadata. Serialize only
  scalar metadata, omit unavailable/complex values, and preserve document text verbatim.
- Generation uses a system instruction and a user message with separate EVIDENCE and
  QUESTION sections. Require supplied evidence and acknowledgement of insufficiency;
  treat evidence as data. Keep response extraction and nonempty-evidence citations.
- Empty/all-whitespace document text returns a deterministic Vietnamese insufficient-
  evidence acknowledgement and empty citations, with no model/retrieval/web invocation.
  This is an explicit no-evidence result, not a fake factual/model answer.
- Grading uses system instructions and QUESTION/CANDIDATE DOCUMENT sections composed
  with the existing GradeDocuments structured model. Keep schema and grading fallback.
- Normalize recorded structured PromptValue inputs to actual message text in the
  existing test-only model; do not record Python repr instead of observable input.

## Implementation steps
1. Add focused input/empty-context regressions and normalize the recorder; run to
   demonstrate pre-change failures.
2. Add serializer and minimally fix the two node inputs; preserve roles and output.
3. Update the existing real compiled graph assertions for corrected inputs and verify
   evidence propagation through real nodes without graph mocks.
4. Update docs and run full offline pytest, dependency/lock, targeted quality/diff checks.

## Test strategy
Inspect actual recorded model inputs for question, full financial text/units, metadata,
ordering, context/question sensitivity, per-document grading isolation, and schema.
Empty/blank documents must not call a model. Use existing local WorkflowChatModel,
socket/tracing/download guards, and real compiled graph. No live provider or API tests.
Run uv sync; uv lock --check --offline; uv run pytest; targeted Ruff check/format;
uv run mypy src/finrag; scoped mypy --follow-imports=silent on changed application/tests;
git diff --check. Keep baseline debt visible, without suppressions.

## Migration / compatibility
Intentional model-input correction and explicit empty-evidence response only. Existing
builder/factories, ModelRoles, state, role lifetime, dependencies/indexes, and HTTP output
schema remain. The grader's heuristic fallback, retention of rejected documents when
all rejected, existing web route, and always-successful numeric flags remain policy debt
for Phase 1.2C/1.2B. Prompt grounding is not proof of model obedience or citation accuracy.

## Acceptance criteria
Generation receives evidence + question; stable available metadata/text survive;
empty evidence cannot use unsupported memory; grading receives question + document
through GradeDocuments; real graph demonstrates these inputs; full offline suite and
targeted type/lint/format checks pass; no later-phase change.

## Progress checklist
- [x] Inspect contracts/runtime/tests and record baseline.
- [x] Record failing regressions and implement corrected inputs.
- [x] Verify real graph, offline tests, and targeted gates.
- [x] Update docs and final measured handoff.

## Final results
Completed: see docs/phase-1.2a-baseline.md for the 15-point report and exact commands.
Final pytest: 221 passed, four unchanged warnings. Focused node/runtime/schema run:
31 passed. Sync and offline lock check pass, with no dependency changes. All five
changed Python files pass Ruff/format. Scoped mypy passes for 23 files including
every changed application/test module; standalone src/finrag passes for 18 files.
Diff checks confirm workflow/state/providers/retrieval/API/dependencies unchanged.
Baseline debt remains unchanged: 13 Ruff findings, six formatting files, eight app
mypy errors in five untouched files. No suppressions or later-phase work was added.

Pre-change focused regressions: all nine cases failed as expected, demonstrating
question-only generation, unchanged generation input across contexts, unsupported
empty-evidence answers, and document-only grading. No skips or xfails were introduced.
