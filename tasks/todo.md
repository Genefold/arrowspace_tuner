# Implementation ledger

## T-001 — eps/k-only tuning with targeted low-k refinement

- State: DONE — implementation/local verification complete; merge gated by PR CI
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
- Commits: e0c9315 (implementation), af2c19c (checkpoint/report-contract hardening).
- PR: https://github.com/Genefold/arrowspace_tuner/pull/49
- Checkpoint: APPROVE from the correctness reviewer after full diff/spec review;
  security GO with no findings; final test-quality signoff GO at af2c19c.
  Reviewers did not independently execute commands; parent verification supplies
  the recorded check results. Optional cache cleanup and additional resume variants
  are deferred, not mixed into this feature.
- CI at af2c19c: Python 3.12/3.13 tests, Ruff/mypy on both versions, and installed
  wheel smoke all pass (run 43604903995). Final ledger-only update retains the
  same verified source/tests.
- Accepted risks: stochastic native builds; Optuna/SQLite warnings; pre-existing
  unrelated format issue. No global minimum-k guarantee.
- Rollback: revert this task's commits together; use a new compatible study.

## T-002 — prepare release 0.30.1

- State: DONE — prepared for review, not tagged or published
- Depends on: merged T-001 / PR #49.
- Acceptance: tuner version 0.30.1; ArrowSpace compatibility remains 0.30.x;
  manifest/lock/artifacts agree; release notes prominently document breaking
  tau removal; no tag or publication.
- TDD: release metadata and README/changelog tests failed on 0.30.0/stale
  introduction before preparation changes.
- Verification: release-focused tests, full pytest, Ruff, mypy, uv lock check,
  sdist/wheel build and twine validation.
- Results: 312 tests pass; Ruff, strict mypy, lock check, sdist/wheel build,
  and twine checks pass. Dependency range is unchanged; no CI/runtime changes.
- Artifacts: dist/release-0.30.1/ (wheel and source distribution).
- Release notes: docs/releases/0.30.1.md.
- Commit: release-preparation commit containing this ledger.
- PR: https://github.com/Genefold/arrowspace_tuner/pull/50
- User subsequently explicitly authorized publication of tuner 0.30.1.
- Pre-publication checkpoint caught stale quickstart tau discovery and SemVer
  claims; corrected both with a failing-then-passing documentation regression.
- Publishing is gated on green PR CI and final checkpoint approval. Release
  workflow and PyPI verification are tracked in the GitHub release.
