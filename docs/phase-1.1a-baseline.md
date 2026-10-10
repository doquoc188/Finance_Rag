# Phase 1.1A - configuration and domain contracts

Recorded on 2026-10-09 against starting revision `d8fc799`. Scope follows the
[ExecPlan](../.agent/plans/phase-1.1a.md). No Phase 1.1B/provider registry work is included.

## Files and package structure

Created:

```text
src/finrag/
    __init__.py
    py.typed
    config/
        __init__.py
        settings.py
    domain/
        __init__.py
        documents.py
        retrieval.py
tests/test_finrag_settings.py
tests/test_finrag_domain.py
tests/test_finrag_architecture.py
.agent/plans/phase-1.1a.md
docs/phase-1.1a-baseline.md
```

Updated `pyproject.toml`, `uv.lock`, `.gitignore`, `AGENTS.md`, and
`docs/development.md`. No existing Python source or test file changed.

Setuptools discovers packages under `src`, restricted to `finrag*`, and includes
the `py.typed` marker. `uv sync` now installs `finance-rag` editably; the lockfile
changes only its source from `virtual = "."` to `editable = "."`. Runtime dependency
requirements, extras, and all third-party locked versions remain unchanged.
`mypy_path = ["src"]` gives the package its canonical `finrag` module names.
Legacy `app/` entry points still run from the root and are not packaged or migrated.

## Settings and environment convention

`finrag.config.Settings` uses Pydantic v2 `SettingsConfigDict`; nested models use
`ConfigDict`. There is no settings singleton, SDK/client initialization, network
validation, credential requirement, or automatic `.env` loading. Only the new
`FINRAG_` prefix with `__` nesting is read; old unprefixed settings are separate.

| Field | Default |
| --- | --- |
| `generation` | Independent `ChatModelSettings`: `openai`, `gpt-4o-mini`, temperature `0.0` |
| `grading` | Separate instance with the same defaults |
| `query_rewrite` | Separate instance with the same defaults |
| `embedding` | `EmbeddingSettings`: `huggingface`, `sentence-transformers/all-MiniLM-L6-v2`; no temperature |
| `retrieval` | Strategy `dense`; dense/sparse/fused top-k `20`; final top-k `4` |
| `graphrag` | Disabled, `index_name=None` |

Examples (configuration metadata only; no provider adapter is introduced):

```text
FINRAG_GENERATION__PROVIDER=openai
FINRAG_GENERATION__MODEL=gpt-4o-mini
FINRAG_GENERATION__TEMPERATURE=0.2
FINRAG_GRADING__PROVIDER=groq
FINRAG_GRADING__MODEL=grading-model
FINRAG_QUERY_REWRITE__PROVIDER=ollama
FINRAG_QUERY_REWRITE__MODEL=rewrite-model
FINRAG_EMBEDDING__PROVIDER=huggingface
FINRAG_EMBEDDING__MODEL=sentence-transformers/all-MiniLM-L6-v2
FINRAG_RETRIEVAL__FUSED_TOP_K=12
FINRAG_RETRIEVAL__FINAL_TOP_K=4
FINRAG_GRAPHRAG__ENABLED=false
```

Each role can select a different provider/model; a partial override keeps that
role's other defaults and leaves other roles untouched. Provider names are nonblank
identifiers, not a supported-provider registry. Defaults/identifiers do not claim a
provider or model is available. Provider-specific temperature/capability/key checks
belong to later work. Configuration classes log nothing and define no secret fields.

Validation rejects blank provider/model names after whitespace stripping, nonfinite
or negative temperatures, unknown constructor/nested fields, nonpositive/nonintegral
retrieval counts, boolean counts, and `final_top_k > fused_top_k`. Settings and nested
models are frozen. Graph enablement requires a nonblank index name; choosing graph
retrieval also requires enablement. The flag/name neither loads nor implements GraphRAG.

The environment mechanism follows the official
[Pydantic settings documentation](https://docs.pydantic.dev/latest/concepts/pydantic_settings/).

## Document and retrieval contracts

All domain imports are standard-library or other `finrag.domain` modules. Domain
objects use frozen dataclasses with slots; they do not depend on Pydantic or SDKs.

| Type | Contract |
| --- | --- |
| `SourceReference` | Nonblank document ID and source, optional one-based integer page and nonblank section |
| `DocumentChunk` | Nonblank chunk/document IDs and text, matching `SourceReference.document_id`, scalar metadata |
| `RetrievalStrategy` | String enum: `dense`, `sparse`, `hybrid`, `graph` |
| `ScoreType` | String enum: `cosine`, `bm25`, `rrf`, `reranker` |
| `RetrievalScore` | Finite numeric value paired with a required `ScoreType`; immutable and unordered |
| `RetrievedEvidence` | Chunk, positive integer rank, explicit strategy, optional typed score |

Page/rank reject booleans, strings, fractions, zero, and negative values. Source and
chunk identity are supplied by callers; this phase does not generate IDs or indexes.
Evidence text is retained verbatim rather than trimmed or normalized.

Metadata is defensively copied into a read-only mapping. Values are limited to
strings, integers, finite floats, booleans, or `None`; nested containers and arbitrary
objects are rejected. This keeps metadata immutable without a recursive freezing
framework. Metadata participates in value equality; hashes omit the mapping. These
ordinary Python hashes are not persistent document IDs. Explicit serializers/adapters
are future work; callers can obtain a plain scalar mapping with `dict(chunk.metadata)`.

A raw score cannot be attached to evidence without its type. Equal numbers with
different score types are unequal `RetrievalScore` values; ordering scores raises
`TypeError`. Values are not normalized or treated as probabilities, and reranker
scores can coexist with any retrieval strategy. Even scores sharing a category need
matching model/index/query context before comparison. No fusion, ranking, reranking,
or retrieval implementation is added by these contracts.

## Tests and verification

Added 77 offline test cases: settings (33), domain (41), and architecture/import (3).
The original 45 tests are unchanged. Tests isolate new environment variables and
use the existing network guard; no API credentials or model downloads are needed.

The architectural checks inspect domain imports against a standard-library/domain
allowlist, construct domain objects in a Python process with site-packages disabled,
and import the installed settings package outside the repository with deprecations
treated as errors. No architecture-testing dependency was added.

| Command | Result |
| --- | --- |
| `uv sync` | Passed; installed only the new editable project |
| `uv run pytest` | **122 passed**, no skips/xfails; 3 existing warnings |
| `uv run ruff check src/finrag tests/test_finrag_settings.py tests/test_finrag_domain.py tests/test_finrag_architecture.py` | Passed |
| `uv run ruff format --check src/finrag tests/test_finrag_settings.py tests/test_finrag_domain.py tests/test_finrag_architecture.py` | Passed; 9 Python files |
| `uv run mypy src/finrag` | Passed; 6 source files |
| `uv run mypy src/finrag tests/test_finrag_settings.py tests/test_finrag_domain.py tests/test_finrag_architecture.py` | Passed; 9 files including all new tests |
| `git diff --check` | Passed |
| `uv run ruff check . --output-format concise` | 26 existing findings, unchanged |
| `uv run ruff format --check .` | 10 legacy files need formatting, unchanged |
| `uv run mypy app` | 21 existing errors in 8 files, unchanged |

The initial sandboxed sync could not reach PyPI, and the offline cache lacked some
resolution metadata. A permitted network-enabled sync installed the editable package
without upgrading dependencies. Subsequent sync/checks use the existing workspace
uv installation and caches documented in Phase 1.0. A negative ordering test initially
triggered mypy as intended; an explicit test-only `Any` boundary now exercises the
runtime rejection too. No domain type or tool rule was weakened.

## Deprecations and compatibility

No new Pydantic deprecation warning is introduced. The existing settings warning
originates at untouched `app/config.py:7` (`class Config`). The existing community
package and `HuggingFaceEmbeddings` warnings also remain. No warning was suppressed.

The legacy application, prompts, provider construction, graph, retrieval, UI, Docker
files, and existing tests are unchanged. They do not import the new package. The new
packaging and contracts establish no change to their runtime behavior. No provider
registry, adapters, routing, or Phase 1.1B implementation is included.
