# Implementation ledger

## T-001 — eps/k-only tuning with targeted low-k refinement

- State: IMPLEMENTED — final CI pending
- Dependencies: approved implementation spec; no other tasks.
- Acceptance:
  - Optimize eps/k only; tau is fixed internally and absent from public results.
  - Explicit extra refinement budget explores low-k/high-eps and full-range eps.
  - Select minimum evaluated k within absolute tolerance of the global best score.
  - Python/CLI/MCP/reports/reload agree; incompatible resumes are rejected.
  - Regression tests, lint, types, and build pass.
- Verification:
  - `python -m pytest`
  - `python -m ruff check src tests`
  - `python -m ruff format --check src tests`
  - `python -m mypy src/arrowspace_tuner`
  - `uv build --wheel --out-dir dist`
- TDD evidence:
  - Obsolete tau-bound test failed before removal, then passed.
  - Selection tests failed because the helper module was missing, then passed.
  - Refinement regression failed before orchestration, then passed.
  - Legacy study regression failed before compatibility checks, then passed.
  - Report/CLI JSON regression failed before summary moved to stderr, then passed.
- Notes:
  - incremental-implementation skill is unavailable; TDD skill was loaded.
  - `python -m pip wheel` is unavailable (no pip in the project venv); use uv build.
  - Repository-wide format check flags untouched tests/test_arrowspace_030_metadata.py.
    Do not change unrelated formatting.
- Verification results:
  - Full suite: 310 passed, 90% total coverage (selection helper 100%).
  - Ruff lint and strict mypy: pass.
  - Wheel build: pass using uv.
  - Full-tree format check: only the confirmed pre-existing metadata-test issue.
  - One run timed out; diagnostic verbose rerun completed in 5.42 seconds.
  - Optuna experimental and SQLite resource warnings do not fail checks.
  - Initial CI exposed report tests assuming optional Plotly. Tests now assert the
    missing-extra error without skipping, and retain complete report assertions
    when extras are installed. Simulated no-Plotly verification: 2 passed.
  - Added checkpoint hardening for full-range eps exploration, inclusive tolerance
    boundaries, and reporting a selected score below the raw maximum.
- Commit: e0c9315 (implementation).
- PR: https://github.com/Genefold/arrowspace_tuner/pull/49
- Checkpoint: no blocking correctness/security/test findings in bounded source
  reviews. Reviewers could not independently execute commands or read the external
  spec; the contract is now available at docs/spec-lowest-k.md. Parent verification
  supplies the diff/check results. Optional cache cleanup and additional resume
  variants are deferred, not mixed into this feature.
- Accepted risks: stochastic native builds; Optuna/SQLite warnings; pre-existing
  unrelated format issue. No global minimum-k guarantee.
- Rollback: revert this task's commits together; use a new compatible study.
