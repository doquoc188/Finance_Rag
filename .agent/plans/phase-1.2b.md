# Phase 1.2B — deterministic financial numeric verification

## Objective
Replace the fact_check substring notice with LLM-free Decimal verification of explicit
answer quantities against supplied evidence, retaining a tri-state per-claim report.

## Current-state observations
2026-10-09 at 2ff539e, clean checkout. Read required contracts/reports and current
graph/state/API/tests. Baseline: 221 passed, four existing warnings. generate writes
generation/documents, then the unchanged generate -> fact_check -> END edge runs
grade_generation_node. Its regex is \d+(?:\.\d+)?, collapses occurrences into a set,
loses separators/signs/units, checks string substrings, and logs unmatched long values.
It always returns is_hallucinated="no", is_useful="yes". Those fields are present in
state/tests but do not control edges and are not exported by HTTP execution_info.
The UI reads only existing execution_info routing fields. State additions can be optional;
no HTTP schema/provider/retrieval/prompt changes are needed for graph/runtime results.

## Scope
src/finrag/verification, fact-check node/state typing, numeric/real-graph tests,
architecture boundary tests, engineering/development docs, plan and completion report.

## Non-goals
No derived arithmetic, FX, citation checking, retrieval/hybrid/GraphRAG/provider work,
prompt changes, graph retries/topology changes, API/UI expansion, or broad quality cleanup.

## Architecture decisions
- Standard-library verification package; frozen/slotted dataclasses and StrEnum statuses.
  Accept only strings/sequence of evidence strings, with zero framework/provider access.
- Extract signed explicit quantities, percentages, listed English/Vietnamese scales,
  VND/VNĐ/đồng and USD/$ prefixes/suffixes. Preserve raw text. Dot is decimal; validated
  comma groups are thousands, otherwise one/two fractional comma digits are decimal.
  Reject unsupported formats as whole mentions rather than partially matching them.
- Use exact Decimal equality and exact scale multiplication with sufficient local
  precision; percentages are percentage points, distinct from plain numbers/money.
- Split sentences without splitting decimal dots, then conjunction/semicolon clauses.
  Mask numeric mentions and remove small explicit stopword sets. Context compatibility
  requires identical nontrivial topic-token sets and equal explicit year anchors.
  This intentionally conservative lexical approach returns NOT_FOUND for paraphrases
  it cannot safely align. Ambiguous multiple same-type quantities per clause cannot
  establish support/contradiction. Bare equal quantities can support one another;
  bare unequal quantities cannot establish contradiction without a topic.
- Same currency/type is mandatory; no inferred currency, FX, tolerance, or calculations.
  Compatible conflicting evidence takes CONTRADICTED precedence over supporting evidence;
  retain both passages in original evidence order so conflicts are reviewable.
- Add optional numeric_verification tuple to graph state/update. Match results retain
  evidence input index and sentence; that index maps to current documents, not a verified
  citation. Existing HTTP transport is unchanged.
- Legacy is_hallucinated="yes" only for CONTRADICTED; NOT_FOUND is not proven hallucination.
  is_useful="yes" when all extracted claims are supported (including empty report for
  compatibility), otherwise "no". Neither flag certifies overall truth/usefulness; document
  this semantic debt rather than redesigning graph policy.

## Implementation steps
1. Implement immutable verifier types and bounded-format parsing/context comparison.
2. Add parametrized extraction/normalization/tri-state/context/robustness regressions.
3. Replace only grade_generation_node and add optional report typing; preserve prompts.
4. Run real compiled graph scenarios with explicit offline upstream models/documents,
   including supported, contradicted, NOT_FOUND and independent multiple claims.
5. Extend stdlib-only/import architecture checks; document actual semantics/limitations;
   run full suite and targeted checks, update this plan with measured results.

## Test strategy
Representative integer/decimal/grouped/percentage/scale/currency/sign extraction;
exact equivalents, type/currency mismatch, unrelated equal/different numbers, entity/
period/metric separation, multiple claims, conflicting evidence, no derived percentage,
unsupported formats, exact long Decimal values, immutable results, and malicious strings.
Use existing offline guards/fakes and real compiled graph/nodes; no live model/API calls.
Run uv sync, uv lock --check --offline, uv run pytest, targeted Ruff/format, mypy src/finrag,
scoped mypy for every changed application/test module, baseline checks, git diff --check.

## Migration / compatibility
Additive graph field, unchanged state keys and edges. Legacy flag behavior intentionally
changes on numeric failures; no-number responses retain old flag values with empty report.
Report values remain Decimal objects for Python graph consumers; no HTTP JSON contract
is added. Evidence provenance, tables with external unit/year headers, lexical paraphrase,
locale ambiguity, and general hallucination detection remain limitations. No index,
dependency, SDK, lockfile, or provider/model migration expected.

## Acceptance criteria
Explicit mentions yield ordered SUPPORTED/CONTRADICTED/NOT_FOUND entries; exact Decimal
normalization supports required syntax; unrelated quantities never establish contradiction;
structured results and conservative flags propagate through the real graph; offline full
tests and targeted quality checks pass; no later-phase change.

## Progress checklist
- [x] Inspect current code/output consumers and record baseline.
- [x] Implement verifier and focused tests.
- [x] Integrate state/node and exercise real graph.
- [x] Verify checks, document limitations, finalize handoff (2026-10-09).

## Final results
Initial focused result: 54 numeric unit cases + five real-node/compiled-graph cases +
10 architecture cases passed (69 total). Parser review additionally covers dash ranges
and negative-value contradictions. No application prompts/topology/API changed.

Final full pytest: 286 passed, four existing warnings; the baseline was 221 passed.
uv sync and offline lock check pass. Targeted Ruff/format pass for all seven changed
Python files. mypy src/finrag passes (20 files); explicit scoped mypy with
--follow-imports=silent passes (25 files, including every changed application/test file).
Repository-wide baseline checks retain 13 lint findings, six unformatted files, and
eight mypy errors in five untouched legacy files. No new suppressions or cleanup.
git diff --check passes. Protected provider/model/config/retrieval/API/workflow/context/
dependency paths have an empty diff. Six files created, five modified; exact commands,
inventory, behavior, limits, and deferred defects are in docs/phase-1.2b-baseline.md.
No acceptance criteria remain unmet. Phase 1.2C has not begun.
