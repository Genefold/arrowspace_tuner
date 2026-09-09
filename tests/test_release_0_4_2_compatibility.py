"""
test_release_0_4_2_compatibility.py — v0.4.2 release regression suite.

Proves, on the installed ArrowSpace (run under
``uv run --with "arrowspace==<version>" pytest`` for the full matrix):

- issue #40: the README imports ``ArrowSpaceBuilder`` from ``arrowspace``
  and the wrong import genuinely raises ``ImportError``;
- round trips: dictionaries returned by ``tune()``, ``EpsTuner.fit()``,
  ``EpsTuner.graph_params``, and ``load_graph_params()`` pass verbatim to
  ``ArrowSpaceBuilder().build()``;
- the graph-parameter contract: native ``topk`` key, never ``top_k``,
  and ``tau``/``best_tau`` excluded from build-time dictionaries;
- the v0.4.1 fixes stay fixed: ``None`` ``search_batch()`` rows are pruned
  instead of crashing, zero-λ probe anchors are filtered before
  ``search_batch()``, and the all-pruned ``RuntimeError`` lists distinct
  build failures.
"""

from __future__ import annotations

import json
import subprocess
import sys
from collections.abc import Callable
from pathlib import Path
from typing import Any

import numpy as np
import optuna
import pytest
from arrowspace import ArrowSpaceBuilder

import arrowspace_tuner
from arrowspace_tuner import BuildParams, EpsTuner, StudyConfig

EXAMPLES_DIR = Path(__file__).resolve().parents[1] / "examples"
README_PATH = Path(__file__).resolve().parents[1] / "README.md"


# ── helpers ───────────────────────────────────────────────────────────────────


def _fast_tuner() -> EpsTuner:
    """EpsTuner configured for fast, deterministic tests."""
    return EpsTuner(
        n_trials=3,
        seed=42,
        eps_low=0.5,
        eps_high=2.0,
        k_low=3,
        k_high=10,
        tau_low=0.3,
        tau_high=1.2,
        n_probe=20,
    )


def _run_example(name: str) -> None:
    result = subprocess.run(
        [sys.executable, str(EXAMPLES_DIR / name)],
        capture_output=True,
        text=True,
        timeout=300,
    )
    assert result.returncode == 0, (
        f"{name} failed (exit {result.returncode}):\n"
        f"stdout:\n{result.stdout}\nstderr:\n{result.stderr}"
    )


@pytest.fixture(scope="module")
def fitted_tuner(embeddings_small: np.ndarray) -> EpsTuner:
    """One shared fit() result reused by the round-trip tests."""
    tuner = _fast_tuner()
    tuner.fit(embeddings_small)
    return tuner


# ── issue #40: README / import contract ───────────────────────────────────────


def test_readme_quickstart_imports() -> None:
    """No README snippet may import ArrowSpaceBuilder from arrowspace_tuner."""
    readme = README_PATH.read_text(encoding="utf-8")
    assert "from arrowspace_tuner import ArrowSpaceBuilder" not in readme
    # quickstart and power-user snippets both import it from `arrowspace`
    assert readme.count("from arrowspace import ArrowSpaceBuilder") >= 2
    # exact failure from issue #40, reproduced on the installed package:
    with pytest.raises(ImportError):
        from arrowspace_tuner import ArrowSpaceBuilder  # noqa: F401


def test_readme_quickstart_builds_graph() -> None:
    """The README quickstart is mirrored by examples/quickstart.py, which runs."""
    _run_example("quickstart.py")


def test_power_user_example_builds_graph() -> None:
    """The README power-user flow is mirrored by examples/power_user.py."""
    _run_example("power_user.py")


# ── round trips: tuner output → ArrowSpaceBuilder().build() ──────────────────


def test_tune_output_round_trips_to_arrowspace_builder(
    embeddings_small: np.ndarray,
) -> None:
    graph_params = arrowspace_tuner.tune(embeddings_small, n_trials=3, seed=42, n_probe=20)
    aspace, gl = ArrowSpaceBuilder().build(graph_params, embeddings_small)
    assert aspace is not None and gl is not None
    results = aspace.search_batch(embeddings_small[:5], gl, 0.5)
    assert results is not None
    assert len(results) == 5


def test_fit_output_round_trips_to_arrowspace_builder(
    fitted_tuner: EpsTuner,
    embeddings_small: np.ndarray,
) -> None:
    graph_params = fitted_tuner.best_params
    assert graph_params is not None
    aspace, gl = ArrowSpaceBuilder().build(graph_params, embeddings_small)
    results = aspace.search_batch(embeddings_small[:5], gl, fitted_tuner.best_tau)
    assert results is not None
    assert len(results) == 5


def test_graph_params_round_trip_to_arrowspace_builder(
    fitted_tuner: EpsTuner,
    embeddings_small: np.ndarray,
) -> None:
    aspace, gl = ArrowSpaceBuilder().build(fitted_tuner.graph_params, embeddings_small)
    results = aspace.search_batch(embeddings_small[:5], gl, fitted_tuner.best_tau)
    assert results is not None


def _write_report(out_dir: Path) -> None:
    """Write a best_params.json in the layout save_report() produces."""
    run_dir = out_dir / "arrowspace_tuner" / "20260909_000000"
    run_dir.mkdir(parents=True)
    (run_dir / "best_params.json").write_text(
        json.dumps({"params": {"eps": 1.25, "k": 14}, "score": 0.85}),
        encoding="utf-8",
    )


def test_loaded_graph_params_round_trip_to_arrowspace_builder(
    tmp_path: Path,
    embeddings_small: np.ndarray,
) -> None:
    """Params loaded from a saved report must build verbatim."""
    _write_report(tmp_path)
    tuner = EpsTuner()
    graph_params = tuner.load_graph_params(out_dir=str(tmp_path))

    aspace, gl = ArrowSpaceBuilder().build(graph_params, embeddings_small)
    results = aspace.search_batch(embeddings_small[:5], gl, 0.5)
    assert results is not None


# ── graph-parameter contract ──────────────────────────────────────────────────


def test_public_graph_params_emit_topk_not_top_k(
    fitted_tuner: EpsTuner,
    embeddings_small: np.ndarray,
    tmp_path: Path,
) -> None:
    """Every public graph dict uses native `topk` and never `top_k`."""
    _write_report(tmp_path)
    loaded = EpsTuner().load_graph_params(out_dir=str(tmp_path))

    public_dicts = [
        arrowspace_tuner.tune(embeddings_small, n_trials=3, seed=42, n_probe=20),
        fitted_tuner.best_params,
        fitted_tuner.graph_params,
        loaded,
        BuildParams(k=14).to_dict(),
    ]
    expected = {"eps", "k", "topk", "p", "sigma"}
    for gp in public_dicts:
        assert set(gp.keys()) == expected, gp
        assert "top_k" not in gp
        assert "tau" not in gp


def test_best_tau_is_not_in_graph_params(fitted_tuner: EpsTuner) -> None:
    """best_tau is a search-time result, excluded from graph-build dicts."""
    assert isinstance(fitted_tuner.best_tau, float)
    assert "tau" not in fitted_tuner.graph_params
    assert "tau" not in fitted_tuner.best_params


# ── issue-24-style corpus: isolated items + non-zero-λ anchors ────────────────


def test_corpus_with_isolated_items_round_trips() -> None:
    """
    Unnormalised two-Gaussian corpus contains isolated items at low eps
    (zero-λ anchors). fit() must complete, and the returned graph_params
    must build and search on the installed ArrowSpace.
    """
    rng = np.random.default_rng(3407)
    X = np.vstack(
        [
            rng.normal(0, 1, (40, 12)) + 2.0,
            rng.normal(4, 1, (40, 12)),
        ]
    )
    tuner = EpsTuner(n_trials=5, seed=3407, eps_low=0.5, eps_high=3.0, n_probe=20)
    graph_params = tuner.fit(X)
    aspace, gl = ArrowSpaceBuilder().build(graph_params, X)
    results = aspace.search_batch(X[:10], gl, tuner.best_tau)
    assert results is not None
    assert len(results) == 10


# ── v0.4.1 fixes: objective-level guards, exercised through make_objective ────


class _FakeArrowSpace:
    """
    Stands in for the Rust ArrowSpace object inside make_objective.

    search_batch() returns one row per query (as the bindings do) via
    ``row_fn``; it optionally rejects probes whose embedding maps to a
    zero-λ item, mirroring the bindings' "Lambda is zero for query N"
    failure that motivated the v0.4.1 anchor filtering.
    """

    def __init__(
        self,
        lambdas: np.ndarray,
        row_fn: Callable[[int], Any],
        corpus: np.ndarray | None = None,
        reject_zero_lambda: bool = False,
    ) -> None:
        self._lambdas = lambdas
        self._row_fn = row_fn
        self._reject = reject_zero_lambda
        if reject_zero_lambda and corpus is not None:
            self._lambda_by_row = {
                np.ascontiguousarray(row, dtype=np.float64).tobytes(): lam
                for row, lam in zip(corpus, lambdas)
            }

    def lambdas(self) -> np.ndarray:
        return self._lambdas

    def search_batch(
        self,
        queries: np.ndarray,
        gl: object,
        tau: float,
    ) -> list[Any]:
        if self._reject:
            for i, row in enumerate(np.asarray(queries)):
                lam = self._lambda_by_row.get(np.ascontiguousarray(row).tobytes())
                if lam is not None and abs(lam) <= 1e-12:
                    raise ValueError(f"Lambda is zero for query {i}")
        return [self._row_fn(i) for i in range(len(queries))]


def _make_study(
    embeddings: np.ndarray,
    cfg: StudyConfig,
) -> optuna.Study:
    from arrowspace_tuner.core.objective import make_objective

    objective_fn, _cache = make_objective(embeddings, cfg)
    study = optuna.create_study(
        direction="maximize",
        sampler=optuna.samplers.TPESampler(seed=cfg.seed),
        pruner=optuna.pruners.NopPruner(),
    )
    study.optimize(objective_fn, n_trials=cfg.n_trials)
    return study


def _compat_cfg(name: str) -> StudyConfig:
    return StudyConfig(
        n_trials=2,
        seed=42,
        study_name=name,
        eps_low=0.5,
        eps_high=2.0,
        k_low=3,
        k_high=10,
        tau_low=0.3,
        tau_high=1.2,
        n_probe=20,
    )


def _patch_build_and_score(
    monkeypatch: pytest.MonkeyPatch,
    aspace: _FakeArrowSpace,
) -> None:
    import arrowspace_tuner.core.objective as obj_mod

    def fake_build_and_score(
        embeddings: np.ndarray,
        params: BuildParams,
        trial: optuna.Trial | None = None,
    ) -> tuple[float, float, Any, Any]:
        return 0.5, 0.2, aspace, object()

    monkeypatch.setattr(obj_mod, "build_and_score", fake_build_and_score)


def test_none_search_rows_are_pruned_not_crashed(
    embeddings_small: np.ndarray,
    monkeypatch: pytest.MonkeyPatch,
) -> None:
    """
    pyarrowspace >= 0.26.7 returns None rows for probes with no hits (#37).
    The objective must skip them and complete the trial, not raise TypeError.
    """
    n = len(embeddings_small)
    rng = np.random.default_rng(7)
    lambdas = rng.normal(0.6, 0.1, n)  # no zero-λ anchors

    def row_fn(i: int) -> list[tuple[int, float]] | None:
        # odd probe rows return hits, even rows are None
        if i % 2 == 0:
            return None
        return [(j, 0.9 - 0.1 * c) for c, j in enumerate(range(i, i + 4))]

    _patch_build_and_score(monkeypatch, _FakeArrowSpace(lambdas, row_fn))

    study = _make_study(embeddings_small, _compat_cfg("compat_none_rows"))
    assert len(study.trials) == 2
    assert all(t.state == optuna.trial.TrialState.COMPLETE for t in study.trials)
    assert study.best_value > 0.0


def test_none_search_rows_all_none_prunes_trial(
    embeddings_small: np.ndarray,
    monkeypatch: pytest.MonkeyPatch,
) -> None:
    """When every probe row is None the trial is pruned, not crashed."""
    n = len(embeddings_small)
    rng = np.random.default_rng(7)
    lambdas = rng.normal(0.6, 0.1, n)
    _patch_build_and_score(monkeypatch, _FakeArrowSpace(lambdas, lambda i: None))

    study = _make_study(embeddings_small, _compat_cfg("compat_all_none"))
    assert len(study.trials) == 2
    assert all(t.state == optuna.trial.TrialState.PRUNED for t in study.trials)


def test_zero_lambda_probe_is_filtered(
    embeddings_small: np.ndarray,
    monkeypatch: pytest.MonkeyPatch,
) -> None:
    """
    Probes sitting on isolated items (zero λ) must be filtered BEFORE
    search_batch() — passing them would raise the bindings'
    "Lambda is zero for query N" error (#24).
    """
    n = len(embeddings_small)
    lambdas = np.where(np.arange(n) % 2 == 0, 0.0, 0.7)  # half the items isolated

    def row_fn(i: int) -> list[tuple[int, float]]:
        # hits for every probe row — only reachable if zero-λ probes are
        # filtered out, otherwise the fake raises "Lambda is zero for query N"
        return [(j, 0.9 - 0.1 * c) for c, j in enumerate(range(i, i + 4))]

    _patch_build_and_score(
        monkeypatch,
        _FakeArrowSpace(lambdas, row_fn, corpus=embeddings_small, reject_zero_lambda=True),
    )

    study = _make_study(embeddings_small, _compat_cfg("compat_zero_lambda"))
    assert len(study.trials) == 2
    assert all(t.state == optuna.trial.TrialState.COMPLETE for t in study.trials)
    # some probes were filtered out before search_batch
    n_probe = study.best_trial.user_attrs["n_probe"]
    assert 0 < n_probe < 20


def test_all_pruned_error_lists_distinct_build_failures(
    embeddings_small: np.ndarray,
    monkeypatch: pytest.MonkeyPatch,
) -> None:
    """
    When every trial fails with a build error, fit() must surface the
    distinct errors instead of blaming the corpus (#38).
    """
    import arrowspace_tuner.core.objective as obj_mod

    distinct_errors = [
        "ValueError: unknown key(s) 'top_k'",
        "TypeError: sigma must be float or None",
    ]

    def fake_build_and_score(
        embeddings: np.ndarray,
        params: BuildParams,
        trial: optuna.Trial | None = None,
    ) -> tuple[float, float, Any, Any]:
        assert trial is not None
        trial.set_user_attr("build_error", distinct_errors[trial.number % 2])
        raise optuna.TrialPruned()

    monkeypatch.setattr(obj_mod, "build_and_score", fake_build_and_score)

    tuner = _fast_tuner()
    with pytest.raises(RuntimeError) as excinfo:
        tuner.fit(embeddings_small)
    msg = str(excinfo.value)
    assert "distinct exception" in msg
    for error in distinct_errors:
        assert error in msg
