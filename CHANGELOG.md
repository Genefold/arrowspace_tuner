# Changelog

All notable changes to `arrowspace_tuner` are documented here.
Format follows [Keep a Changelog](https://keepachangelog.com/en/1.0.0/).
Versioning follows [Semantic Versioning](https://semver.org/).

---

## [0.1.0] — 2026-04-29

Initial release.

### Added

- `EpsTuner` — main public class for hyperparameter discovery over `eps`, `k`, `tau`
- `arrowspace_tuner.optuna()` — one-liner convenience API: `aspace, gl = arrowspace.optuna(embeddings)`
- `StudyConfig` / `BuildParams` — typed dataclasses for power-user configuration
- Query-free spectral objective: weighted composite of MRR-Top0 proxy, Fiedler value, and lambda variance
- Optuna TPE sampler with pruning on degenerate graphs (NNZ ≤ N, disconnected, flat spectrum)
- `sample_n` subsampling: 33x speedup on 50k corpus with identical best params (validated)
- `storage` parameter for SQLite-backed persistence and parallel/resumed runs
- `tuner.save_report()` — saves `trials.csv`, `best_params.json`, and Plotly HTML plots
- `[report]` optional extra (pandas + plotly) — kept out of hard dependencies
- `py.typed` marker — PEP 561 compliant, full mypy strict mode
- Comprehensive test suite: `test_objective.py`, `test_tuner.py`, `conftest.py`
- CI workflow: pytest + ruff + mypy on every push and pull request
