# eps/k targeted refinement implementation

Approved source: `/Users/tommasomoriondo/.opencode/plan/arrowspace-tuner-lowest-k-spec.md`.
Repository contract: `docs/spec-lowest-k.md`.
The user explicitly directed implementation of the complete approved feature.

## T-001 — eps/k-only tuning with quality-first low-k refinement

Dependency order within this task:
1. eps/k-only objective and shared configuration validation.
2. Pure deterministic scheduling and valid-trial selection helpers.
3. Main/refinement orchestration and compatible-study checks.
4. Consistent Python, CLI, MCP, report, and reload contracts.
5. Regression tests, examples, migration notes, and verification.

No unrelated dependency, CI, or release-version changes.
Use TDD for behavior, then full pytest, Ruff, mypy, and wheel build.
Do not merge with failing checks or an unresolved checkpoint.
