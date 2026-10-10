# ExecPlans

Write an ExecPlan before a substantial refactor: changes spanning module boundaries,
provider contracts, persistence/index formats, or externally visible behavior. Store it
as `.agent/plans/<short-name>.md`. Small tooling/documentation changes need not create one.

Plans must be self-contained, grounded in inspected files, and executable by a new
contributor. Use repository-relative paths and exact commands. Separate observations
from proposals; name unresolved assumptions. Do not imply a planned feature already
exists. Stay within the authorized phase; a plan does not authorize additional scope.

Use the following sections:

1. **Objective** — observable user/developer outcome and why it matters.
2. **Current-state observations** — entry points, relevant files, dependencies,
   behavior, baseline checks, and known defects, with evidence.
3. **Scope** — modules/contracts/artifacts allowed to change.
4. **Non-goals** — later phases and behaviors deliberately preserved.
5. **Architecture decisions** — boundaries, alternatives, rationale, and tradeoffs;
   maintain a dated decision log when choices change.
6. **Implementation steps** — small ordered steps with exact paths, dependencies,
   commands, expected outcomes, and checkpoints. Make retries safe.
7. **Test strategy** — baseline, regression, negative-path, and compatibility checks;
   separate offline tests from opt-in live tests and actual quality evaluations.
8. **Migration/compatibility concerns** — import paths, APIs, configuration, dependencies,
   stored indexes, rollout/rollback, and recovery. State explicitly when none apply.
9. **Acceptance criteria** — observable pass/fail conditions tied to the objective.
10. **Progress checklist** — dated completed/remaining steps and blockers. Update during
    work, including deviations and discoveries; do not rewrite a failure as a success.
11. **Final results** — changed files, exact commands and outcomes, measured evidence,
    unmet criteria, technical debt, and the next scoped task.

Keep plans concise but concrete. Record test failures before changing code, and compare
under the same environment. Explain every compatibility break or quality-gate exception.
Never invent evaluation results or mark a step complete based only on intended behavior.
