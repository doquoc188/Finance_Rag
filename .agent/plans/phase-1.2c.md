# Phase 1.2C — bounded retrieval self-correction and graph policy

## Objective
Make relevance-failure recovery explicit and bounded: retrieve, grade, generate when
relevant, otherwise rewrite and retrieve again only within a validated per-run budget.
Exhaustion must use the existing empty-evidence response and terminal numeric verifier.

## Current-state observations
Inspected the required phase reports, graph nodes/state/builder, composition root,
settings, API consumers, and workflow tests at clean revision `b38ef80`. Baseline on
2026-10-10: 286 passed, four existing warnings. Current compiled topology is
START -> retrieve -> grade_documents; rejection with hard-coded retry_count < 2 routes
to transform_query -> web_search -> generate -> fact_check -> END. It never retrieves
or grades again. All-rejected grading returns the rejected originals, and any rejected
candidate sets web_search_needed even when another relevant candidate remains. The
fallback engine mirrors that one-shot web path. retry_count is incremented by rewrite,
but its limit is neither validated nor composition-owned. fact_check is terminal.
API execution_info exposes retry_count/web_search_needed/transformed_question but not
the structured numeric report. Full graph input callers currently initialize retry_count.

## Scope
Retrieval configuration, graph state/nodes/builder/fallback engine, explicit runtime
composition, API initial compatibility state, focused offline graph/settings/runtime/
architecture tests, `.env.example`, AGENTS/development documentation, and phase report.

## Non-goals
No generation retry, numeric-triggered rewrite, web search, new retrieval source,
hybrid/BM25/RRF/reranking, GraphRAG, query routing, provider/model changes, citation
verification, arithmetic, indexing changes, API redesign, or broad quality cleanup.

## Architecture decisions
- Add `RetrievalSettings.max_rewrite_attempts: int = 2`, rejecting booleans and values
  below zero. Environment name is `FINRAG_RETRIEVAL__MAX_REWRITE_ATTEMPTS`.
- `build_finrag_graph` receives the integer explicitly; `compose_runtime` passes it from
  validated settings. Nodes import no settings and read no environment.
- Add per-execution `rewrite_attempts`; transform increments it exactly once. Preserve
  externally consumed retry_count as a mirrored compatibility field. Initial inputs set
  both to zero; old callers lacking the new field start from retry_count or zero.
- Grading retains Phase 1.2A question-aware per-document yes/no semantics, but returns
  only relevant documents. An all-rejected result is therefore empty and safely reaches
  Phase 1.2A generation after exhaustion. web_search_needed remains for compatibility
  but is false because this graph no longer performs web search.
- Compiled and fallback engines use the same loop: rejection with budget rewrites then
  returns to retrieve; success or exhaustion generates. Numeric verification stays
  terminal regardless of its tri-state result.
- Preserve rewrite prompt and original-question input. Identical rewrites still consume
  an attempt, so no semantic duplicate detector or hidden mutable state is needed.

## Implementation steps
1. Add configuration/state contracts and tests for defaults, environment, zero, invalid
   negatives/booleans, and compatibility mirror semantics.
2. Filter rejected documents, increment explicit attempts in rewrite, and implement the
   same bounded retrieve/grade loop in compiled and fallback engines.
3. Pass the setting through composition; initialize request state without exposing the
   internal field through the existing API response.
4. Add real compiled-graph offline regressions for happy path, one rewrite then success,
   exhaustion, zero budget, identical rewrite, numeric contradiction, and per-run state.
5. Update architecture/development/config documentation and run all required checks.

## Test strategy
Use local recording BaseChatModel implementations and an injected deterministic retriever;
do not mock graph nodes or fact checking. Assert exact retrieval/grade/rewrite/generation
counts, successful evidence in the generation prompt, final documents/state, safe answer,
numeric report, and termination. Retain existing provider/network/download guards. Run
baseline/full pytest, targeted Ruff/format, mypy src/finrag and every changed Python
module, full quality checks for debt reporting, offline lock validation, and diff checks.

## Migration / compatibility concerns
Default two rewrites preserves the old numeric limit while changing the authorized policy
from one-shot web fallback to repeated internal retrieval. retry_count stays in state/API
as a mirror; rewrite_attempts remains internal to Python graph state. web_search_needed
stays present but remains false. No dependency, lock, stored-index, prompt, provider,
model-role, numeric-verifier, or HTTP schema migration. Rejected documents no longer reach
generation, which is required for the safe exhaustion behavior.

## Acceptance criteria
Validated explicit budget; per-run exact attempt accounting; relevant evidence bypasses
rewrite; failure retries retrieval only within budget; zero/repeated rewrites terminate;
exhaustion returns the existing no-evidence message without generation model invocation;
numeric contradiction remains terminal/reportable; real compiled tests and scoped quality
checks pass; later retrieval/graph work remains untouched.

## Progress checklist
- [x] Inspect current topology, state/config ownership, consumers, tests, and baseline.
- [x] Add failing policy/configuration regressions (10 failed, 42 passed before source fix).
- [x] Implement bounded graph policy and composition wiring.
- [x] Update documentation and complete required verification (2026-10-10).

## Final results
Completed from clean revision `b38ef80`. Added validated default-two retrieval policy,
per-execution rewrite_attempts with retry_count compatibility mirroring, relevant-only
grading output, and compiled/fallback rewrite -> retrieve loops. Exhaustion uses empty
evidence without a generation-model call; web search is unregistered; numeric verification
remains terminal. Composition passes settings explicitly and the HTTP schema remains stable.

Final full pytest: 300 passed with the same four warnings as the 286-test baseline.
Sync/offline lock check passed. Targeted Ruff/format passed for all 15 changed Python
files. Scoped mypy passed for 34 files and standalone mypy src/finrag passed for 20.
Repository-wide baseline remains 13 Ruff findings, six unformatted files, and eight mypy
errors in five untouched files. Diff whitespace and protected-path checks passed; no
dependency/lock/provider/prompt/numeric-verifier changes. The complete 20-point result,
command record, limitations, and deferred defects are in `docs/phase-1.2c-baseline.md`.
All acceptance criteria are met; Phase 1.3 has not begun.
