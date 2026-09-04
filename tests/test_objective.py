"""
test_objective.py — unit tests for core/graph.py and core/objective.py.

These tests exercise the internal building blocks in isolation.
They require the arrowspace Rust wheel to be installed.
"""
from __future__ import annotations

from typing import Any

import numpy as np
import optuna

from arrowspace_tuner.core import (
    BuildParams,
    build_and_score,
    fiedler_normalized,
    make_objective,
)
from arrowspace_tuner.core.config import StudyConfig

# ── BuildParams.to_dict ─────────────────────────────────────────────────────────

class TestBuildParams:

    def test_to_dict_keys(self) -> None:
        p = BuildParams(eps=1.0, k=10, topk=5)
        d = p.to_dict()
        assert set(d.keys()) == {"eps", "k", "topk", "p", "sigma"}

    def test_to_dict_values(self) -> None:
        p = BuildParams(eps=1.5, k=8, topk=4, p=2.0, sigma=None)
        d = p.to_dict()
        assert d["eps"]   == 1.5
        assert d["k"]     == 8
        assert d["topk"] == 4
        assert d["sigma"] is None

    def test_to_dict_topk_value(self) -> None:
        """topk value in dict must equal the topk attribute."""
        p = BuildParams(eps=1.0, k=10, topk=5)
        d = p.to_dict()
        assert d["topk"] == p.topk == 5

    def test_to_dict_sigma_value_passthrough(self) -> None:
        """A set sigma (not None) must pass through unchanged."""
        p = BuildParams(eps=1.0, k=10, topk=5, sigma=1.5)
        assert p.to_dict()["sigma"] == 1.5

    def test_to_dict_matches_bindings_schema(self) -> None:
        """
        Guard against key-schema drift with pyarrowspace (#38): to_dict()
        must emit exactly the bindings-native keys. pyarrowspace >= 0.27
        rejects unknown keys, which pruned every Optuna trial in 0.4.0.
        """
        d = BuildParams().to_dict()
        assert set(d.keys()) == {"eps", "k", "topk", "p", "sigma"}

    def test_to_dict_round_trip_bindings(self, embeddings_small: np.ndarray) -> None:
        """The dict must be accepted verbatim by ArrowSpaceBuilder.build()."""
        from arrowspace import ArrowSpaceBuilder

        params = BuildParams(eps=1.5, k=8, topk=4)
        aspace, gl = ArrowSpaceBuilder().build(params.to_dict(), embeddings_small)
        assert gl is not None
        assert len(aspace.lambdas()) == len(embeddings_small)

    def test_topk_default_is_half_k(self) -> None:
        p = BuildParams(k=12)
        assert p.topk == 6   # k=12, so __post_init__ sets topk = max(1, 12 // 2) = 6


# ── build_and_score ───────────────────────────────────────────────────────────

class TestBuildAndScore:

    def test_returns_four_values(self, embeddings_small: np.ndarray) -> None:
        params = BuildParams(eps=1.5, k=8, topk=4)
        result = build_and_score(embeddings_small, params)
        assert len(result) == 4

    def test_valid_graph_nonzero_fiedler(self, embeddings_small: np.ndarray) -> None:
        # eps=1.5 reliably connects the 4-cluster L2-normalised fixture.
        # eps=1.0 produced a disconnected graph (all unit-sphere vectors
        # have pairwise distances tightly clustered around 1.0).
        params = BuildParams(eps=1.5, k=8, topk=4)
        fiedler, var_lambda, aspace, gl = build_and_score(embeddings_small, params)
        assert fiedler > 0.0
        assert var_lambda >= 0.0
        assert aspace is not None
        assert gl is not None

    def test_degenerate_eps_too_large_returns_zeros(self, embeddings_small: np.ndarray) -> None:
        # eps so large every item connects to every other → NNZ >> N but
        # spectrum collapses; OR eps so small graph is empty → NNZ <= N
        params = BuildParams(eps=1e-6, k=3, topk=1)
        fiedler, var_lambda, aspace, gl = build_and_score(embeddings_small, params)
        # degenerate path: aspace and gl are None
        assert aspace is None
        assert gl is None

    def test_degenerate_raises_pruned_with_trial(self, embeddings_small: np.ndarray) -> None:
        study = optuna.create_study(direction="maximize")

        def obj(trial: optuna.Trial) -> float:
            params = BuildParams(eps=1e-6, k=3, topk=1)
            build_and_score(embeddings_small, params, trial=trial)
            return 0.0

        # TrialPruned is caught by Optuna internally — study should complete
        study.optimize(obj, n_trials=1)
        assert study.trials[0].state == optuna.trial.TrialState.PRUNED

    def test_fiedler_in_unit_interval(self, embeddings_small: np.ndarray) -> None:
        params = BuildParams(eps=1.5, k=8, topk=4)
        fiedler, _, _, gl = build_and_score(embeddings_small, params)
        if gl is not None:
            assert 0.0 <= fiedler <= 1.0 + 1e-9   # small float tolerance

    def test_var_lambda_nonnegative(self, embeddings_small: np.ndarray) -> None:
        params = BuildParams(eps=1.5, k=8, topk=4)
        _, var_lambda, _, _ = build_and_score(embeddings_small, params)
        assert var_lambda >= 0.0


# ── fiedler_normalized ───────────────────────────────────────────────────────────

class TestFiedlerNormalized:

    def test_returns_float(self, embeddings_small: np.ndarray) -> None:
        params = BuildParams(eps=1.5, k=8, topk=4)
        _, _, _, gl = build_and_score(embeddings_small, params)
        if gl is not None:
            result = fiedler_normalized(gl)
            assert isinstance(result, float)

    def test_value_in_unit_interval(self, embeddings_small: np.ndarray) -> None:
        params = BuildParams(eps=1.5, k=8, topk=4)
        _, _, _, gl = build_and_score(embeddings_small, params)
        if gl is not None:
            f = fiedler_normalized(gl)
            assert 0.0 <= f <= 1.0 + 1e-9


# ── make_objective ──────────────────────────────────────────────────────────────

class TestMakeObjective:

    def test_returns_callable(self, embeddings_small: np.ndarray, fast_study_config: StudyConfig) -> None:  # type: ignore[name-defined]  # noqa: F821
        obj, cache = make_objective(embeddings_small, fast_study_config)
        assert callable(obj)
        assert isinstance(cache, dict)

    def test_objective_returns_float(self, embeddings_small: np.ndarray, fast_study_config: StudyConfig) -> None:  # type: ignore[name-defined]  # noqa: F821
        study      = optuna.create_study(direction="maximize")
        obj, _     = make_objective(embeddings_small, fast_study_config)
        study.optimize(obj, n_trials=1)

        completed = [
            t for t in study.trials
            if t.state == optuna.trial.TrialState.COMPLETE
        ]
        if completed:
            assert isinstance(completed[0].value, float)

    def test_objective_score_nonnegative(self, embeddings_small: np.ndarray, fast_study_config: StudyConfig) -> None:  # type: ignore[name-defined]  # noqa: F821
        study      = optuna.create_study(direction="maximize")
        obj, _     = make_objective(embeddings_small, fast_study_config)
        study.optimize(obj, n_trials=fast_study_config.n_trials)

        completed = [
            t for t in study.trials
            if t.state == optuna.trial.TrialState.COMPLETE
        ]
        for t in completed:
            assert t.value >= 0.0

    def test_user_attrs_populated(self, embeddings_small: np.ndarray, fast_study_config: StudyConfig) -> None:  # type: ignore[name-defined]  # noqa: F821
        study      = optuna.create_study(direction="maximize")
        obj, _     = make_objective(embeddings_small, fast_study_config)
        study.optimize(obj, n_trials=fast_study_config.n_trials)

        completed = [
            t for t in study.trials
            if t.state == optuna.trial.TrialState.COMPLETE
        ]
        if completed:
            attrs = completed[0].user_attrs
            assert "fiedler"    in attrs
            assert "var_lambda" in attrs
            assert "mrr_proxy"  in attrs
            assert "tau"        in attrs
            assert "n_sample"   in attrs
            assert "n_probe"    in attrs

    def test_three_params_suggested(self, embeddings_small: np.ndarray, fast_study_config: StudyConfig) -> None:  # type: ignore[name-defined]  # noqa: F821
        study      = optuna.create_study(direction="maximize")
        obj, _     = make_objective(embeddings_small, fast_study_config)
        study.optimize(obj, n_trials=fast_study_config.n_trials)

        completed = [
            t for t in study.trials
            if t.state == optuna.trial.TrialState.COMPLETE
        ]
        if completed:
            assert set(completed[0].params.keys()) == {"eps", "k", "tau"}

    def test_sample_n_respected(self, embeddings_medium: np.ndarray, fast_study_config: StudyConfig) -> None:  # type: ignore[name-defined]  # noqa: F821
        cfg          = fast_study_config
        cfg.sample_n = 50   # force subsampling on the 600-item fixture
        study        = optuna.create_study(direction="maximize")
        obj, _       = make_objective(embeddings_medium, cfg)
        study.optimize(obj, n_trials=1)

        completed = [
            t for t in study.trials
            if t.state == optuna.trial.TrialState.COMPLETE
        ]
        if completed:
            assert completed[0].user_attrs["n_sample"] == 50

    def test_best_cache_populated_when_full_corpus(
        self, embeddings_small: np.ndarray, fast_study_config: StudyConfig  # type: ignore[name-defined]  # noqa: F821
    ) -> None:
        """best_cache is filled when sample_n=None (full corpus path)."""
        study      = optuna.create_study(direction="maximize")
        obj, cache = make_objective(embeddings_small, fast_study_config)
        study.optimize(obj, n_trials=fast_study_config.n_trials)

        completed = [
            t for t in study.trials
            if t.state == optuna.trial.TrialState.COMPLETE
        ]
        if completed:
            assert "aspace" in cache
            assert "gl"     in cache
            assert "score"  in cache
            assert cache["score"] > 0.0

    def test_best_cache_empty_when_subsampling(
        self, embeddings_medium: np.ndarray, fast_study_config: StudyConfig  # type: ignore[name-defined]  # noqa: F821
    ) -> None:
        """best_cache stays empty when sample_n is set (subsample path)."""
        fast_study_config.sample_n = 50
        study      = optuna.create_study(direction="maximize")
        obj, cache = make_objective(embeddings_medium, fast_study_config)
        study.optimize(obj, n_trials=1)
        assert cache == {}

    def test_flat_embeddings_all_pruned_or_zero(
        self, embeddings_flat: np.ndarray, flat_study_config: StudyConfig  # type: ignore[name-defined]  # noqa: F821
    ) -> None:
        """
        On near-identical embeddings with eps bounds below the data scale,
        every trial must be either pruned or return a zero score.

        Uses flat_study_config (eps_high=0.05) instead of fast_study_config
        (eps_high=2.0) so that arrowspace cannot form a connected graph on
        the 0.01-scaled vectors, keeping the test deterministic regardless
        of test collection order.
        """
        study      = optuna.create_study(direction="maximize")
        obj, _     = make_objective(embeddings_flat, flat_study_config)
        study.optimize(obj, n_trials=flat_study_config.n_trials)

        for t in study.trials:
            is_pruned = t.state == optuna.trial.TrialState.PRUNED
            is_zero   = t.value == 0.0 if t.value is not None else True
            assert is_pruned or is_zero, (
                f"Expected pruned or zero score on flat embeddings, "
                f"got state={t.state} value={t.value}"
            )

    # ── None search rows (pyarrowspace >= 0.26.7, issue #37) ──────────────────

    @staticmethod
    def _stub_build_and_score(monkeypatch: object, fake_aspace: object) -> None:
        """Patch core.objective.build_and_score to return a fake aspace."""
        import arrowspace_tuner.core.objective as obj_mod

        def fake_build_and_score(
            embeddings: np.ndarray,
            params: object,
            trial: optuna.Trial | None = None,
        ) -> tuple[float, float, object, object]:
            return 0.5, 0.1, fake_aspace, object()

        monkeypatch.setattr(obj_mod, "build_and_score", fake_build_and_score)

    def test_all_none_search_rows_pruned_not_crashed(
        self,
        embeddings_small: np.ndarray,  # type: ignore[name-defined]  # noqa: F821
        fast_study_config: StudyConfig,  # type: ignore[name-defined]  # noqa: F821
        monkeypatch: object,
    ) -> None:
        """
        pyarrowspace >= 0.26.7 returns None rows from search_batch for
        probes with no hits (#37). The objective must prune via the
        row_widths guard, not crash with TypeError before it.
        """
        class FakeAspace:
            def lambdas(self) -> list[float]:
                return list(np.linspace(0.1, 0.9, len(embeddings_small)))

            def search_batch(self, queries: object, gl: object, tau: float) -> list[Any]:
                return [None] * len(queries)

        self._stub_build_and_score(monkeypatch, FakeAspace())
        study = optuna.create_study(direction="maximize")
        obj, _ = make_objective(embeddings_small, fast_study_config)
        study.optimize(obj, n_trials=1)
        assert study.trials[0].state == optuna.trial.TrialState.PRUNED

    def test_mixed_none_search_rows_completes(
        self,
        embeddings_small: np.ndarray,  # type: ignore[name-defined]  # noqa: F821
        fast_study_config: StudyConfig,  # type: ignore[name-defined]  # noqa: F821
        monkeypatch: object,
    ) -> None:
        """Mixed valid/None/[] rows: valid rows still count toward the MRR."""
        n_items = len(embeddings_small)

        class FakeAspace:
            def lambdas(self) -> list[float]:
                return list(np.linspace(0.1, 0.9, n_items))

            def search_batch(
                self, queries: object, gl: object, tau: float
            ) -> list[Any]:
                out = []
                for r in range(len(queries)):
                    if r % 2 == 0:
                        out.append([
                            (i * 7 % n_items, 1.0 / (j + 1))
                            for j, i in enumerate(range(3))
                        ])
                    elif r % 4 == 1:
                        out.append(None)
                    else:
                        out.append([])
                return out

        self._stub_build_and_score(monkeypatch, FakeAspace())
        study = optuna.create_study(direction="maximize")
        obj, _ = make_objective(embeddings_small, fast_study_config)
        study.optimize(obj, n_trials=1)
        assert study.trials[0].state == optuna.trial.TrialState.COMPLETE
        assert study.trials[0].value > 0.0

    def test_none_batch_results_pruned(
        self,
        embeddings_small: np.ndarray,  # type: ignore[name-defined]  # noqa: F821
        fast_study_config: StudyConfig,  # type: ignore[name-defined]  # noqa: F821
        monkeypatch: object,
    ) -> None:
        """A None whole-batch return must prune via the `or []` guard, not crash."""
        class FakeAspace:
            def lambdas(self) -> list[float]:
                return list(np.linspace(0.1, 0.9, len(embeddings_small)))

            def search_batch(
                self, queries: object, gl: object, tau: float
            ) -> object:
                return None

        self._stub_build_and_score(monkeypatch, FakeAspace())
        study = optuna.create_study(direction="maximize")
        obj, _ = make_objective(embeddings_small, fast_study_config)
        study.optimize(obj, n_trials=1)
        assert study.trials[0].state == optuna.trial.TrialState.PRUNED

    def test_build_failure_sets_build_error_attr(
        self,
        embeddings_small: np.ndarray,  # type: ignore[name-defined]  # noqa: F821
        fast_study_config: StudyConfig,  # type: ignore[name-defined]  # noqa: F821
    ) -> None:
        """A .build() exception must be persisted on the pruned trial (#38)."""
        import unittest.mock as mock

        import arrowspace

        with mock.patch.object(
            arrowspace.ArrowSpaceBuilder, "build",
            side_effect=ValueError("boom"),
        ):
            study = optuna.create_study(direction="maximize")
            obj, _ = make_objective(embeddings_small, fast_study_config)
            study.optimize(obj, n_trials=1)

        t = study.trials[0]
        assert t.state == optuna.trial.TrialState.PRUNED
        assert t.user_attrs["build_error"] == "ValueError: boom"

    def test_search_batch_failure_sets_build_error_attr(
        self,
        embeddings_small: np.ndarray,  # type: ignore[name-defined]  # noqa: F821
        fast_study_config: StudyConfig,  # type: ignore[name-defined]  # noqa: F821
        monkeypatch: object,
    ) -> None:
        """A search_batch exception must be persisted on the pruned trial (#38)."""
        class FakeAspace:
            def lambdas(self) -> list[float]:
                return list(np.linspace(0.1, 0.9, len(embeddings_small)))

            def search_batch(
                self, queries: object, gl: object, tau: float
            ) -> object:
                raise ValueError("search blew up")

        self._stub_build_and_score(monkeypatch, FakeAspace())
        study = optuna.create_study(direction="maximize")
        obj, _ = make_objective(embeddings_small, fast_study_config)
        study.optimize(obj, n_trials=1)

        t = study.trials[0]
        assert t.state == optuna.trial.TrialState.PRUNED
        assert t.user_attrs["build_error"] == "ValueError: search blew up"

    # ── zero-lambda probe anchors (issue #24) ─────────────────────────────────

    @staticmethod
    def _make_zero_lambda_aspace(
        embeddings: np.ndarray, zero_predicate: object
    ) -> object:
        """
        Fake aspace mimicking pyarrowspace search_batch: raises ValueError
        when a query anchor sits on a zero-lambda item (issue #24), maps
        each query back to its corpus index by value comparison.
        """
        zero_set = {
            i for i in range(len(embeddings)) if zero_predicate(i)
        }

        class FakeAspace:
            def lambdas(self) -> list[float]:
                return [
                    0.0 if i in zero_set else 0.1 + i * 0.001
                    for i in range(len(embeddings))
                ]

            def search_batch(self, queries: object, gl: object, tau: float) -> list[Any]:
                out = []
                for q in queries:
                    for j in range(len(embeddings)):
                        if np.allclose(q, embeddings[j]):
                            if j in zero_set:
                                raise ValueError(f"Lambda is zero for query {j}")
                            out.append(
                                [(k, 1.0 / (r + 1)) for r, k in enumerate(range(3))]
                            )
                            break
                    else:
                        out.append(None)
                return out

        return FakeAspace()

    def test_zero_lambda_anchors_filtered_not_failed(
        self,
        embeddings_small: np.ndarray,  # type: ignore[name-defined]  # noqa: F821
        fast_study_config: StudyConfig,  # type: ignore[name-defined]  # noqa: F821
        monkeypatch: object,
    ) -> None:
        """
        Half the corpus has zero lambda (isolated nodes). search_batch raises
        for those anchors (#24); the objective must filter them out and
        complete the trial, not prune it.
        """
        fake = self._make_zero_lambda_aspace(
            embeddings_small, lambda i: i % 2 == 0
        )
        self._stub_build_and_score(monkeypatch, fake)
        study = optuna.create_study(direction="maximize")
        obj, _ = make_objective(embeddings_small, fast_study_config)
        study.optimize(obj, n_trials=1)
        assert study.trials[0].state == optuna.trial.TrialState.COMPLETE
        assert study.trials[0].value > 0.0
        # n_probe attr must reflect the FILTERED anchor count (#24)
        n_probe_used = study.trials[0].user_attrs["n_probe"]
        assert 0 < n_probe_used < fast_study_config.n_probe

    def test_all_zero_lambdas_pruned(
        self,
        embeddings_small: np.ndarray,  # type: ignore[name-defined]  # noqa: F821
        fast_study_config: StudyConfig,  # type: ignore[name-defined]  # noqa: F821
        monkeypatch: object,
    ) -> None:
        """Every item has zero lambda — no usable anchors — must prune."""
        fake = self._make_zero_lambda_aspace(embeddings_small, lambda i: True)
        self._stub_build_and_score(monkeypatch, fake)
        study = optuna.create_study(direction="maximize")
        obj, _ = make_objective(embeddings_small, fast_study_config)
        study.optimize(obj, n_trials=1)
        assert study.trials[0].state == optuna.trial.TrialState.PRUNED
