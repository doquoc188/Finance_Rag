# Phase 1.2B — deterministic financial numeric verification

Implemented on 2026-10-09 from commit `2ff539e`. Verification is offline regression
evidence, not a financial accuracy benchmark or live-provider evaluation.

## 1. Previous behavior

Inspected the actual fact-check node, state, workflow, API/UI consumers, and existing
tests before editing. `grade_generation_node` extracted `\d+(?:\.\d+)?`, collapsed
mentions into a set, and checked string substrings against joined document bodies.
It lost signs, grouping, units, currency, and claim context. Unmatched values of at
least three characters produced a notice, but both flags always returned success:
`is_hallucinated="no"`, `is_useful="yes"`. Baseline pytest: 221 passed, four warnings.

## 2. Verification architecture

`src/finrag/verification` accepts only answer text and a sequence of evidence strings.
It imports standard-library modules only, performs no I/O, and has no LangChain,
LangGraph, FastAPI, Qdrant, provider, settings, or legacy app dependency. Parsing and
comparison are deterministic; no model is consulted.

```text
Generated answer + retrieved evidence
                 |
       Deterministic Numeric Verifier
                 |
      SUPPORTED / CONTRADICTED / NOT_FOUND
```

## 3. Result and status types

`VerificationStatus` and `NumericType` are explicit string enums. The latter separates
NUMBER, PERCENTAGE, and MONEY. Frozen/slotted `NumericMention` retains raw text, Decimal
value, type, and currency. `NumericVerification` retains the answer sentence, mention,
status, and immutable evidence matches. `EvidenceMatch` retains evidence input index,
sentence, and normalized mention. Results preserve answer order and repeated mentions.

## 4. Supported syntax

Representative formats include `1250`, `1,250`, `1,250.50`, `12.5`, `12.5%`, `12,5%`,
ASCII signed values, and case-insensitive scale/currency words. English scales are
thousand, million, billion, trillion; Vietnamese scales are nghìn, ngàn, triệu, tỷ.
Currencies can precede or follow a quantity: `VND 50 billion`, `50 tỷ VNĐ`, `$50 million`.
Raw matched spans are retained. Explicit years also produce numeric entries.

## 5. Normalization

Validated comma thousands groups are removed; otherwise one/two comma fractional digits
are decimal. Dot is always decimal. Known scales multiply by exact powers of ten using
Decimal with sufficient local precision, including long mantissas. Thus `1.2 billion VND`,
`1,200 million VND`, and `1,200,000,000 VND` are equivalent. Equality is exact: no
rounding tolerance. Percentages retain percentage points, distinct from plain numbers.

## 6. Currency rules

VND, VNĐ, and đồng normalize to VND; USD and `$` normalize to USD. Type and currency
must both match. Currency-less amounts do not inherit a currency from nearby text or
document metadata. Conflicting explicit currency notation is skipped. No FX conversion
is performed; VND and USD never match just because their numbers match.

## 7. Contradiction context

Split sentences without splitting decimal dots, then split clear conjunction/semicolon
clauses. Mask numeric notation, case/diacritic-fold text, tokenize, and remove a small
explicit stopword set. Compare identical meaningful token sets and identical recognized
year anchors (sentence start or after in/for/year/FY/năm). Entity, metric, quarter, and
negation tokens are retained. Period mentions are compared separately from plain amounts.
Multiple same-type/currency nonperiod quantities in one clause are ambiguous and cannot
establish a match. Bare unequal quantities cannot establish contradiction without a topic.

Compatible exact values support; compatible differing values contradict. When supplied
evidence both supports and contradicts a claim, CONTRADICTED takes precedence and both
passages are retained in evidence order for review. Unrelated numbers do not contradict.

## 8. NOT_FOUND

No supporting or conflicting compatible evidence yields NOT_FOUND. This includes empty
evidence, incompatible currency/type, unrelated metric/entity/period, and contexts the
conservative lexical rule cannot align. It means unsupported by supplied evidence,
not proven false or deliberate hallucination.

## 9. Graph integration

The actual `grade_generation_node` consumes `state["generation"]` and existing document
`page_content` values, without environment reads, provider resolution, or model calls.
It writes an optional `numeric_verification` tuple to `FinGraphState`/`FinGraphUpdate`.
Evidence indexes map to the existing documents; they are not verified citations.
The Python graph/runtime result contains Decimal-valued reports. Existing HTTP schemas,
`execution_info`, and UI remain unchanged and do not expose this report yet.

## 10. Legacy flags

Retained existing fields: `is_hallucinated="yes"` only if a result is CONTRADICTED;
NOT_FOUND alone leaves it `"no"`. `is_useful="yes"` when all extracted results are
SUPPORTED, otherwise `"no"`. For an empty report, old `"no"`/`"yes"` behavior remains
for compatibility. Empty extraction does not certify complete coverage or truth.
These misleading legacy names describe limited numeric screening, not general
hallucination/usefulness. The inspected graph edges and API do not consume these flags
for retry policy or transport output; their terminology/policy redesign is deferred.

## 11. Files changed

Created:

- `.agent/plans/phase-1.2b.md`
- `src/finrag/verification/__init__.py`
- `src/finrag/verification/numeric.py`
- `tests/test_numeric_verification.py`
- `tests/test_numeric_graph.py`
- `docs/phase-1.2b-baseline.md`

Modified:

- `app/graph/nodes.py`: replace only numeric fact-check behavior plus required import.
- `app/graph/state.py`: optional report typing and honest legacy-flag comments.
- `tests/test_finrag_architecture.py`: verifier import boundary and stdlib-only execution.
- `AGENTS.md`: verifier boundary and tri-state accuracy contract.
- `docs/development.md`: current flow, semantics, limitations, and report link.

## 12. Regression coverage

Added 58 numeric unit cases and five graph/node cases; architecture tests grow from
eight to ten. Coverage includes required syntax, exact scale equivalence, sign/type/
currency distinction, unrelated matching/differing values, periods/entities/metrics,
independent multiple claims, ambiguous clauses, source conflicts, missing evidence,
unsupported notation, no derived percentage, exact long Decimal normalization under
low ambient precision, immutable results, and inert code-like input strings.
Architecture checks reject forbidden imports/direct eval/exec and execute the verifier
in an isolated Python process with site packages disabled.

## 13. Real graph evidence

Three parametrized cases invoke the actual compiled LangGraph and real fact-check node.
Explicit local chat models generate Revenue 50 billion VND (SUPPORTED), Revenue 55
billion VND (CONTRADICTED), or Profit 55 billion VND (NOT_FOUND) against supplied Revenue
50 billion VND evidence. Assertions check report contents, document index, Decimal values,
flags, answer propagation, zero retries/web search, single grading/generation calls, and
unused rewrite model. The retriever fixture supplies documents without embeddings or
storage; fact checking is never mocked. Two direct-node regressions check independent
claim statuses/no model call and numeric-free compatibility. Existing test isolation
blocks network/model downloads. No real external API was used.

## 14. Commands executed

`uv` is not on PATH in this environment. Commands used the existing local executable
and workspace caches:

```powershell
$env:UV_CACHE_DIR = Join-Path $PWD '.tools\uv-cache'
$env:UV_PYTHON_INSTALL_DIR = Join-Path $PWD '.tools\python'
$uvExecutable = '.\.tools\bootstrap\bin\uv.exe'
& $uvExecutable sync
& $uvExecutable lock --check --offline
& $uvExecutable run pytest
& $uvExecutable run pytest tests/test_numeric_verification.py
& $uvExecutable run pytest tests/test_numeric_verification.py tests/test_numeric_graph.py tests/test_finrag_architecture.py --tb=short
& $uvExecutable run ruff check --fix src/finrag/verification app/graph/nodes.py app/graph/state.py tests/test_numeric_verification.py tests/test_numeric_graph.py tests/test_finrag_architecture.py
& $uvExecutable run ruff format src/finrag/verification app/graph/nodes.py app/graph/state.py tests/test_numeric_verification.py tests/test_numeric_graph.py tests/test_finrag_architecture.py
& $uvExecutable run ruff check src/finrag/verification app/graph/nodes.py app/graph/state.py tests/test_numeric_verification.py tests/test_numeric_graph.py tests/test_finrag_architecture.py
& $uvExecutable run ruff format --check src/finrag/verification app/graph/nodes.py app/graph/state.py tests/test_numeric_verification.py tests/test_numeric_graph.py tests/test_finrag_architecture.py
& $uvExecutable run ruff format --check AGENTS.md .agent/plans/phase-1.2b.md docs/development.md docs/phase-1.2b-baseline.md
& $uvExecutable run mypy src/finrag
& $uvExecutable run mypy --follow-imports=silent src/finrag app/graph/nodes.py app/graph/state.py tests/test_numeric_verification.py tests/test_numeric_graph.py tests/test_finrag_architecture.py
& $uvExecutable run ruff check .
& $uvExecutable run ruff format --check .
& $uvExecutable run mypy app
git diff --check
git diff --exit-code -- app/graph/workflow.py app/graph/context.py app/core src/finrag/config src/finrag/domain src/finrag/models src/finrag/providers app/runtime.py app/main.py app/api pyproject.toml uv.lock requirements.txt
```

## 15. Pytest results

Full final suite: **286 passed, four existing warnings** (Python 3.12.15).
Initial focused suite: 69 passed before four extra dash/sign cases were added; those
additional cases pass in the final full suite. Initial numeric-only run: 54 passed.
No skips, xfails, weakened assertions, fake benchmarks, or live-provider claims.

## 16. Quality results

Targeted Ruff check passes; all seven changed Python files pass formatting.
`mypy src/finrag` passes for 20 files. Scoped mypy uses `--follow-imports=silent` to
check the explicit clean package and changed application/test files without reporting
untouched legacy dependencies; it passes for 25 files. No new ignore/exclusion added.
Repository-wide checks retain existing debt: 13 lint findings, six files needing
formatting, and eight mypy errors in five untouched legacy files. Full checks are
baseline reporting, not claimed green gates. `git diff --check` passes; Git's LF/CRLF
notices are working-copy warnings rather than whitespace failures.

## 17. Dependencies and compatibility

`uv sync` and `uv lock --check --offline` pass. No dependency, lockfile, SDK, provider,
model settings, prompt, graph topology/retry, index, retrieval, or HTTP schema changes.
Only the two listed application graph files change. No application module is moved.

## 18. Known limits

Conservative token-set matching misses paraphrases and does not provide semantic
entailment, relation direction, or general hallucination detection. Year anchors are
limited explicit text patterns; document metadata/table headings do not supply missing
periods/currencies/units. Bare equal values support without broader semantic proof.
Clause ambiguity deliberately reduces recall. Duplicate/conflicting source credibility
and citation validity are not assessed.

The fixed locale convention can misinterpret an ambiguous single dotted Vietnamese
group as decimal; no automatic locale inference exists. Mixed European separators,
multi-dot grouping, malformed comma groups, scientific notation, compact ranges,
slash dates, Unicode minus, accounting parentheses, and currency/unit inheritance
are unsupported. Unsupported forms can be omitted rather than yielding a report entry;
an empty report is not proof that an answer contains no numeric claim. No FX, fuzzy
tolerance, growth/rate/ratio derivation, or arithmetic expression execution is present.

## 19. Intentionally deferred

Phase 1.2C retry-policy changes remain untouched. Derived financial calculations,
citation/provenance verification, hybrid retrieval/BM25/RRF/reranking, GraphRAG, query
routing, indexing/idempotency, provider changes, and broad lint/mypy cleanup are outside
this phase. Existing mock retrieval/web content, grading heuristics, ingestion/security
issues, and Python 3.10/requirements-only Docker incompatibility remain documented debt.
Next work should be separately scoped; this implementation does not begin Phase 1.2C.
