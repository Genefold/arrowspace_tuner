# Changelog

All notable changes to `arrowspace_tuner` are documented here.
Format follows [Keep a Changelog](https://keepachangelog.com/en/1.0.0/).
Versioning follows [Semantic Versioning](https://semver.org/).

---

## [0.2.1] — 2026-05-02

### Changed
- `EpsTuner` configuration defaults — set `.max_clusters` and `.min_clusters` (or similar clustering bounds) to `None`. This delegates cluster sizing entirely to `arrowspace`'s internal heuristics, allowing it to determine the optimal number of clusters dynamically during the search rather than being constrained by the tuner.

---

## [0.2.0] — 2026-04-29

### Added
- `StudyConfig.n_jobs` — new field (default `1`) for parallel trial execution
- `EpsTuner.__init__` — new `n_jobs` keyword argument passed through to
  `study.optimize()`; set to `-1` to use all available CPU cores
- `GPSampler` with `n_startup_trials=4` as the primary sampler when
  `optuna[botorch]` is installed; silently falls back to TPE otherwise
- Warm-start `enqueue_trial` — injects one known-good anchor trial
  `{eps=1.0, k=15, tau=0.5}` before `study.optimize()` to seed the surrogate
- `[tool.hatch.build.targets.wheel] exclude` in `pyproject.toml` — keeps
  `tests/`, `notebooks/`, `docs/`, `.github/`, caches, and build artefacts
  out of the published wheel (56 KB, 14 files)
- `test` job in `.github/workflows/ci.yml` — runs pytest with coverage after
  the `lint` job succeeds; uses `uv sync --extra dev` to install the full
  project including the `arrowspace` wheel

### Changed
- `EpsTuner.fit()` — sampler replaced: default `TPESampler` (with
  `n_startup_trials=10`) swapped for `GPSampler` → `TPESampler(multivariate=True,
  n_startup_trials=4)` fallback, raising informed trials from 5 to 11 out of 15
- `TPESampler` fallback now uses `multivariate=True` and `group=True` for a
  joint posterior over `(eps, k, tau)` instead of three independent 1-D models
- CI `lint` job — previously skipped pytest entirely; now a prerequisite for
  the new `test` job rather than the sole CI job

### Fixed
- `n_startup_trials=10` (Optuna default) left only 5 informed trials out of 15
  for a 3-D continuous search space; reduced to `4` across both samplers

---

## [0.1.0] — 2026-04-29

Initial release.

### Added
- `EpsTuner` — main public class for hyperparameter discovery over `eps`, `k`, `tau`
- `arrowspace_tuner.optuna()` — one-liner convenience API:
  `aspace, gl = arrowspace.optuna(embeddings)`
- `StudyConfig` / `BuildParams` — typed dataclasses for power-user configuration
- Query-free spectral objective: weighted composite of MRR-Top0 proxy, Fiedler
  value, and lambda variance
- Optuna TPE sampler with pruning on degenerate graphs (NNZ ≤ N, disconnected,
  flat spectrum)
- `sample_n` subsampling: 33x speedup on 50k corpus with identical best params
  (validated)
- `storage` parameter for SQLite-backed persistence and parallel/resumed runs
- `tuner.save_report()` — saves `trials.csv`, `best_params.json`, and Plotly
  HTML plots
- `[report]` optional extra (pandas + plotly) — kept out of hard dependencies
- `py.typed` marker — PEP 561 compliant, full mypy strict mode
- Comprehensive test suite: `test_objective.py`, `test_tuner.py`, `conftest.py`
- CI workflow: pytest + ruff + mypy on every push and pull request