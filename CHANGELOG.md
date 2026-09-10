# Changelog

All notable changes to `arrowspace_tuner` are documented here.
Format follows [Keep a Changelog](https://keepachangelog.com/en/1.0.0/).
Versioning follows [Semantic Versioning](https://semver.org/).

---

## [0.5.0] — 2026-09-09

### Added

- `arrowspace-tuner` Click CLI for local `.npy` and `.npz` embedding
  matrices, with `tune`, `validate`, `inspect`, `version`, and `mcp`
  commands. The console script is also reachable as
  `python -m arrowspace_tuner`.
- Human-readable text output and stable JSON machine output: exactly one
  JSON document on stdout, all logs and error envelopes on stderr.
- Input validation and embedding inspection commands, including SHA-256
  fingerprinting and L2-norm diagnostics.
- Optional stdio MCP server for local LLM clients
  (`pip install "arrowspace-tuner[mcp]"`) exposing exactly four tools:
  `inspect_embeddings`, `tune_graph`, `build_instruction`, and
  `get_tuner_info`.
- Stable `TuneResult` schema (`schema_version` "1.0") with graph
  parameters, the separate search-time `best_tau`, diagnostics, input
  metadata, version information, and warnings — shared by the Python
  service, CLI, and MCP server (new `arrowspace_tuner.service` layer).
- Local MCP path restrictions using `ARROWSPACE_TUNER_ALLOWED_ROOTS`
  (required) plus `ARROWSPACE_TUNER_MAX_INPUT_BYTES`,
  `ARROWSPACE_TUNER_MAX_TRIALS`, and `ARROWSPACE_TUNER_MAX_N_JOBS` limits.
- Atomic `--output` persistence (temp file, fsync, atomic replace).
- Documented exit codes: 0 ok, 2 usage, 3 invalid input, 4 tuning failure,
  5 requested output/report could not be written (`output_error`), 6
  interrupted, 7 internal error. A requested report that fails to persist
  returns `status: "output_error"` with no partial graph configuration.
- Corpus-aware neighbour validation: `k_low` above `n_items - 1` is
  rejected before tuning; an out-of-range `k_high` is clipped with a
  warning; MCP environment limits must be positive integers;
  `build_instruction` validates every graph parameter value.

### Security

- MCP supports local `.npy` and `.npz` files only.
- Remote URLs, pickle/object arrays, and unconstrained filesystem paths
  are rejected before any data is loaded.
- The server does not upload embeddings or make network requests.

### Notes

- `graph_params` contains only ArrowSpace build-time keys: `eps`, `k`,
  `topk`, `p`, and `sigma` — never `top_k`, never `tau`.
- `best_tau` is a separate search-time result; use it at query time:
  `aspace.search(q, gl, tau=result["best_tau"])`.
- The MCP server calls the shared service directly — the CLI is never
  invoked through subprocess from MCP.
- This release does not change the existing Python API (`tune`, `EpsTuner`).

---

## [0.4.2] — 2026-09-09

### Fixed

- Corrected the README and PyPI quickstart/power-user examples to import
  `ArrowSpaceBuilder` from `arrowspace`, not `arrowspace_tuner`. The
  previous examples raised `ImportError` on the released wheel. Closes #40.
- Fixed clean-install tuning failures when PyTorch is not installed.
  `GPSampler` imports PyTorch lazily; the tuner now detects unavailable
  PyTorch before sampler selection and falls back to `TPESampler`.
  Users with PyTorch installed keep `GPSampler` as the default. Found
  by the wheel smoke test.
- `scripts/test_eval.py` updated to the v0.4.x API (`fit()` returns
  graph_params; the caller owns the build step; `tau` read from
  `best_tau`).

### Compatibility

- Added tested compatibility with ArrowSpace 0.28.x. The support range is
  now `arrowspace >= 0.26.0, < 0.29`; tested with 0.26.0, 0.27.3, and
  0.28.1. Closes #41.
- Added regression coverage proving that the graph-parameter dictionaries
  returned by `tune()`, `EpsTuner.fit()`, `EpsTuner.graph_params`, and
  `load_graph_params()` round-trip directly into
  `ArrowSpaceBuilder.build()` on every supported ArrowSpace version.
- Added executable example tests: `examples/quickstart.py` and
  `examples/power_user.py` mirror the README snippets and run in
  `tests/test_examples.py` on every CI build.
- Added an ArrowSpace compatibility matrix to CI (Python 3.12/3.13 ×
  ArrowSpace 0.26.0/0.28.1) using isolated resolution so the lockfile
  cannot conceal an unsupported resolution.

### Notes

- This release does not change the public API.
- Graph-build parameter dictionaries continue to use ArrowSpace's native
  `topk` key — never `top_k`.
- `best_tau` remains a separate search-time result and is intentionally
  excluded from graph-build parameters. Use it at query time:
  `aspace.search(q, gl, tau=tuner.best_tau)`.

---

## [0.4.1] — 2026-09-04

### Fixed

- All graph-params dicts (`BuildParams.to_dict()`, `EpsTuner.best_params`,
  `EpsTuner.graph_params`, `EpsTuner.load_graph_params()`) emit the
  bindings-native `"topk"` key again. v0.4.0 renamed it to `"top_k"`,
  which pyarrowspace's `build()` validation rejects (`unknown key(s)
  'top_k'`), so every Optuna trial was pruned. Closes #38.
- When every trial is pruned by an exception (not statistical pruning),
  `EpsTuner.fit()` now records the exception on each pruned trial and
  surfaces the distinct build errors in the raised `RuntimeError`,
  instead of blaming the corpus size / eps bounds.
- `None` rows returned by `search_batch()` (pyarrowspace >= 0.26.7) no
  longer crash the objective with `TypeError` before the all-zero
  row_widths guard; `None` rows are treated as empty and the trial is
  pruned. Closes #37, #24.
- Probe anchors with zero |λ| (isolated items) are filtered out before
  `search_batch()` instead of triggering the bindings' `Lambda is zero
  for query N` error, which pruned every trial on corpora containing
  isolated nodes. Trials with at least one usable anchor now complete.
  Closes #24.

---

## [0.4.0] — 2026-08-14

### Breaking Changes

- `EpsTuner.fit()` now returns `dict[str, Any]` (graph_params) instead of
  `tuple[ArrowSpace, GraphLaplacian]`. Callers must own the build step:
      graph_params = tuner.fit(embeddings)
      aspace, gl = ArrowSpaceBuilder().build(graph_params, embeddings)
- `api.optuna()` return type updated accordingly.
- `EpsTuner._final_build()` removed (internal method, was not public API).
- `tune()` no longer defaults to `sample_n=5_000`. It inherits
  `EpsTuner`'s default (`sample_n=None`, i.e. full corpus every trial).
  Users relying on the implicit subsampling of `optuna()` must now pass
  `sample_n=5_000` explicitly to `tune()`.

### Added

- `EpsTuner.best_tau` — query-time-only attribute for the optimal `tau`
  value, separated from `best_params`. Closes #26.
- `EpsTuner.graph_params` — property returning `best_params` after
  fitting without requiring file I/O. Raises `RuntimeError` before
  `.fit()`. Closes #29.
- `arrowspace_tuner.tune(embeddings, *, tuner=None, **kwargs)` — new
  primary one-liner entry point. Forwards all kwargs to `EpsTuner`
  with zero parameter duplication. Supports `n_jobs`, `max_clusters`,
  and `cluster_radius`, which were silently missing from `optuna()`.
  Closes #27.

### Changed

- `best_params` now contains only build-time parameters:
  `{eps, k, top_k, p, sigma}`. `tau` is excluded (see `best_tau`).
- `BuildParams.to_dict()` returns `"top_k"` instead of `"topk"`.
  Closes #28.
- `load_best_params()` renamed to `load_graph_params()`. The old name
  is retained as a deprecated alias emitting `DeprecationWarning`.
- `EpsTuner.__repr__` includes `best_tau=...` when fitted.

### Deprecated

- `arrowspace_tuner.optuna()` — emits `DeprecationWarning` and delegates
  to `tune()`. Will be removed in the next minor release.
- `EpsTuner.load_best_params()` — deprecated in favour of
  `load_graph_params()`.

### Fixed

- `load_graph_params()` (formerly `load_best_params()`) now returns
  `"top_k"` instead of `"topk"`, consistent with `EpsTuner.best_params`.

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
