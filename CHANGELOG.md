# Changelog

All notable changes to `arrowspace_tuner` are documented here.
Format follows [Keep a Changelog](https://keepachangelog.com/en/1.0.0/).
Versioning follows [Semantic Versioning](https://semver.org/).

---

## Unreleased

### Breaking Changes
- `EpsTuner.fit()` now returns `dict[str, Any]` (graph_params) instead of
  `tuple[ArrowSpace, GraphLaplacian]`. Callers must own the build step:
      graph_params = tuner.fit(embeddings)
      aspace, gl = ArrowSpaceBuilder().build(graph_params, embeddings)
- `api.optuna()` return type updated accordingly.
- `EpsTuner._final_build()` removed (internal method, was not public API).

### Fixed
- `load_best_params()` now returns `"top_k"` instead of `"topk"` to match
  `best_params` output.

---

## [0.3.0] — 2026-05-07

### Fixed
- **Critical — C-1:** `excep` syntax stump in `core/objective.py`
  `build_and_score()` — the bare token made the entire package unimportable.
  Restored as a proper `except BaseException as exc:` block with correct
  indentation.
- **Critical — C-2:** `save_report()` raised `ModuleNotFoundError` because
  `reporting.py` was missing and `reporter.py` was empty. Implemented
  `reporting.py` with `save_results()` producing `trials.csv`,
  `best_params.json`, and Plotly HTML plots. `reporter.py` is kept as a
  backward-compat stub.
- **Critical — C-3:** `StudyConfig.n_probe` defaulted to `100` while
  `api.optuna()` passed `n_probe=50`, causing silent behavioural divergence
  between the two entry points. Introduced `_DEFAULT_N_PROBE = 50` constant
  in `config.py`; both `StudyConfig` and `api.optuna()` now reference it.
- **High — H-1:** `BuildParams.topk` defaulted to hardcoded `5` regardless
  of `k`, contradicting the docstring's "k // 2" contract. Replaced with a
  sentinel (`-1`) resolved by `__post_init__` to `max(1, k // 2)`.
- **High — H-2:** `gl_to_scipy` comment claimed `to_csr()` returned a
  4-tuple including `shape`; the Protocol and Rust binding return a 3-tuple.
  Comment corrected; docstring updated.
- **High — H-4:** MRR vectorisation crashed when `search_batch` returned no
  results for all probe anchors (issues #16, #22 — small corpus or
  `n_probe ≥ corpus_size`). Added explicit `row_widths.sum() == 0` guard
  that raises `TrialPruned` before the undefined computation.
- **Medium — M-1:** `logger.warning` call in `build_and_score()` exception
  handler was mis-indented. Fixed.
- **Medium — M-2:** Trailing space in `ArrowSpaceProtocol.search_batch`
  signature (`gl: PyGraphLaplacian ,`). Removed.
- **Medium — M-5:** `StudyConfig` accepted invalid inputs silently
  (e.g. `eps_low >= eps_high`, `n_trials=0`). Added `__post_init__`
  with bounds validation and clear error messages.
- **Low — L-1:** `__version__` was hardcoded as `"0.1.0"` in both
  `__init__.py` and `core/__init__.py` while `pyproject.toml` declared
  `0.2.3`. All three copies are now replaced by a single
  `importlib.metadata.version()` call in `__init__.py`; `core/__init__.py`
  no longer declares `__version__`.
- **Low — L-2:** CHANGELOG v0.2.1 entry referenced a non-existent
  `min_clusters` field. Corrected.
- **Low — L-3:** `gl_to_scipy` was exported in `core/__all__` but never
  used externally. Removed from `__all__` (function kept for power users
  who import it directly).
- **Style — S-3:** `StudyConfig` used as return-type annotation in
  `conftest.py` without being imported, requiring `# type: ignore` on every
  fixture. Added `from arrowspace_tuner import StudyConfig` import.

### Changed
- `pyproject.toml` version bumped from `0.2.3` → `0.3.0`.
- `core/__init__.py` — removed duplicate `__version__` attribute;
  removed `gl_to_scipy` from `__all__`.
- Fiedler dense path in `graph.py` — removed redundant `sorted()` call
  (eigvalsh guarantees ascending order); documented why ARPACK path
  still needs `sorted()`.

---

## [0.2.1] — 2026-05-02

### Changed
- `EpsTuner` configuration defaults — set `max_clusters` and `cluster_radius`
  to `None`. This delegates cluster sizing entirely to `arrowspace`'s internal
  heuristics, allowing it to determine the optimal number of clusters
  dynamically during the search rather than being constrained by the tuner.

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
