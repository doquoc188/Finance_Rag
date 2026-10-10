# Phase 1.1A - configuration and domain contracts

## Objective
Add an installable, typed `src/finrag` foundation for validated role configuration
and immutable evidence/retrieval values. Keep the existing `app/` runtime unchanged.

## Current-state observations
Read `AGENTS.md`, `.agent/PLANS.md`, and `docs/phase-1.0.2-baseline.md` before edits.
Starting revision: `d8fc799`; clean working tree. There is no `src/` package or build
backend. `tool.uv.package = false` currently installs dependencies only. The baseline
records 45 passing tests, 3 legacy deprecation warnings, 26 Ruff findings, 10 files
needing formatting, and 21 application mypy errors. Legacy settings use `class Config`.
Existing tests isolate credentials/network and inject their model doubles explicitly.

## Scope
Create the requested `finrag` package, settings, documents, retrieval types, focused
tests, a lightweight architecture boundary check, and documentation. Configure only
the build/import/type-check paths needed for the package, retaining runtime dependencies.

## Non-goals
No runtime imports of the new package, provider registry/adapters, routing, SDK clients,
async support, graph/prompt migration, retrieval algorithms, GraphRAG adapter, indexing,
numeric redesign, or broad legacy cleanup. Do not start Phase 1.1B.

## Architecture decisions
- Use setuptools with explicit `src` package discovery so `uv sync` installs `finrag`
  editably. Do not package/move `app`; retain existing root-based entry points.
- Use `SettingsConfigDict` / `ConfigDict`, with `FINRAG_` and `__` for environment
  loading. No global settings instance, automatic `.env` loading, secrets, or clients.
- Independent frozen chat-role settings for generation/grading/rewrite; a separate
  embedding configuration has no temperature. Providers are nonblank identifiers,
  not a registry whitelist. Defaults are metadata only, not provider availability claims.
- Retrieval counts are positive and `final_top_k <= fused_top_k`. Graph settings are
  disabled by default, with an optional index name required when enabled. Graph retrieval
  selection requires explicit enablement; this does not implement GraphRAG execution.
- Domain code uses standard-library frozen dataclasses and string enums only. Chunk
  metadata is a defensive read-only copy of scalar values; reject nested mutable values
  rather than build a recursive freezing framework. Preserve evidence text verbatim.
- Attach optional scores through a frozen `RetrievalScore` pairing value and `ScoreType`.
  No natural ordering, normalization, or cross-system comparisons are implemented.

## Implementation steps
1. Add package files and build/type-check configuration, then sync the environment.
2. Add role/retrieval/graph settings without importing legacy runtime or SDKs.
3. Add validated immutable evidence and retrieval value objects.
4. Add offline configuration, value-semantics, score, and import-boundary tests.
5. Run required checks, verify unchanged runtime/dependency versions, and document results.

## Test strategy
Clear only `FINRAG_` variables in the new settings tests; use no real credentials.
Test defaults, role independence, nested overrides, invalid values, immutability,
metadata aliasing, provenance consistency, score identity, and enum serialization.
Check domain import ASTs against a standard-library/domain allowlist and import the
domain in an isolated Python process without site-packages. Verify installed-package
imports outside the repository. Run all existing tests unchanged under the socket guard.

## Migration/compatibility
No application settings or behavior migration. The new package is available after
`uv sync`; legacy pip/Docker workflows still use their existing manifests/entry points.
Only package metadata changes in the lock; preserve installed runtime dependency versions.
New settings defaults and validation apply only to explicit callers of `finrag.config`.

## Acceptance criteria
Requested package exists; modern settings support independent roles and environment
loading; document and retrieval values are immutable and framework-independent; basic
invalid state fails; score semantics are explicit; all tests and targeted lint/format/
types pass; legacy runtime sources are unchanged; no provider registry is introduced.

## Progress checklist
- [x] 2026-10-09: inspect instructions, baseline, packaging, and installed settings API.
- [x] 2026-10-09: add editable package, role settings, immutable evidence and score contracts.
- [x] 2026-10-09: add 77 focused cases, including three import/architecture checks.
- [x] 2026-10-09: verify 122 passing tests and clean targeted lint/format/mypy; document debt.

## Final results
See `docs/phase-1.1a-baseline.md` for files, environment examples, validation choices,
commands, and results. New code: 6 source modules plus 3 test modules, all passing
Ruff/format/mypy. Full pytest: 122 passed with the same 3 legacy warnings. Full legacy
checks remain at 26 Ruff findings, 10 unformatted files, and 21 mypy errors in 8 files.
The lock changed only from virtual to editable project metadata; dependency versions
and legacy runtime/test sources are unchanged. No provider registry or Phase 1.1B work.
