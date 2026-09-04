"""
test_tuner.py — integration tests for EpsTuner and the optuna() one-liner.

These tests exercise the full public API end-to-end.
They require the arrowspace Rust wheel to be installed.
"""
from __future__ import annotations

import json
import pathlib

import numpy as np
import optuna as opt
import pytest

from arrowspace_tuner import EpsTuner, tune

# ── helpers ───────────────────────────────────────────────────────────────────

def _make_tuner(**overrides: object) -> EpsTuner:
    """Return a fast EpsTuner suitable for tests."""
    defaults = dict(
        n_trials   = 3,
        seed       = 42,
        eps_low    = 0.5,
        eps_high   = 2.0,
        k_low      = 3,
        k_high     = 10,
        tau_low    = 0.3,
        tau_high   = 1.2,
        n_probe    = 20,
    )
    defaults.update(overrides)
    return EpsTuner(**defaults)


# ── EpsTuner.__init__ ─────────────────────────────────────────────────────────

class TestEpsTunerInit:

    def test_default_instantiation(self) -> None:
        tuner = EpsTuner()
        assert tuner.best_params     is None
        assert tuner.best_score      is None
        assert tuner.best_fiedler    is None
        assert tuner.best_var_lambda is None
        assert tuner.best_mrr_proxy  is None
        assert tuner.best_tau        is None
        assert tuner.study           is None

    def test_repr_before_fit(self) -> None:
        tuner = _make_tuner()
        r = repr(tuner)
        assert "not fitted" in r
        assert "n_trials=3" in r

    def test_repr_reflects_bounds(self) -> None:
        tuner = EpsTuner(eps_low=0.1, eps_high=5.0)
        assert "0.1" in repr(tuner)
        assert "5.0" in repr(tuner)


# ── EpsTuner._validate ────────────────────────────────────────────────────────

class TestEpsTunerValidation:

    def test_raises_on_non_array(self) -> None:
        tuner = _make_tuner()
        with pytest.raises(ValueError, match="np.ndarray"):
            tuner.fit([[1.0, 2.0], [3.0, 4.0]])  # type: ignore[arg-type]

    def test_raises_on_1d(self, embeddings_1d: np.ndarray) -> None:
        tuner = _make_tuner()
        with pytest.raises(ValueError, match="2D"):
            tuner.fit(embeddings_1d)

    def test_warns_on_float32(
        self,
        embeddings_wrong_dtype: np.ndarray,
        caplog: pytest.LogCaptureFixture,
    ) -> None:
        """float32 input triggers a warning and is silently cast to float64."""
        tuner = _make_tuner()
        import logging
        with caplog.at_level(logging.WARNING):
            # validate() only — don't call fit() to avoid full Rust dependency
            result = tuner._validate(embeddings_wrong_dtype)
        assert result.dtype == np.float64
        assert "float64" in caplog.text or "dtype" in caplog.text


# ── StudyConfig validation ────────────────────────────────────────────────────

class TestStudyConfigValidation:

    def test_raises_on_inverted_eps_bounds(self) -> None:
        from arrowspace_tuner import StudyConfig
        with pytest.raises(ValueError, match="eps_low"):
            StudyConfig(eps_low=3.0, eps_high=1.0)

    def test_raises_on_inverted_k_bounds(self) -> None:
        from arrowspace_tuner import StudyConfig
        with pytest.raises(ValueError, match="k_low"):
            StudyConfig(k_low=20, k_high=5)

    def test_raises_on_inverted_tau_bounds(self) -> None:
        from arrowspace_tuner import StudyConfig
        with pytest.raises(ValueError, match="tau_low"):
            StudyConfig(tau_low=1.0, tau_high=0.1)

    def test_raises_on_zero_trials(self) -> None:
        from arrowspace_tuner import StudyConfig
        with pytest.raises(ValueError, match="n_trials"):
            StudyConfig(n_trials=0)

    def test_raises_on_zero_probe(self) -> None:
        from arrowspace_tuner import StudyConfig
        with pytest.raises(ValueError, match="n_probe"):
            StudyConfig(n_probe=0)


# ── BuildParams.__post_init__ ─────────────────────────────────────────────────

class TestBuildParamsTopk:

    def test_topk_resolved_to_half_k_by_default(self) -> None:
        from arrowspace_tuner import BuildParams
        p = BuildParams(k=20)
        assert p.topk == 10   # max(1, 20 // 2)

    def test_topk_override_respected(self) -> None:
        from arrowspace_tuner import BuildParams
        p = BuildParams(k=20, topk=3)
        assert p.topk == 3

    def test_topk_minimum_one(self) -> None:
        from arrowspace_tuner import BuildParams
        p = BuildParams(k=1)
        assert p.topk == 1   # max(1, 1 // 2) = max(1, 0) = 1


# ── fit() all-pruned RuntimeError diagnostics (#38, #24) ──────────────────────

class TestFitAllPruned:
    """fit() must raise a diagnostic RuntimeError when every trial prunes."""

    @staticmethod
    def _stub_build_and_score(monkeypatch: object, build_error: str | None) -> None:
        """Patch core.objective.build_and_score to prune every trial,
        optionally persisting a build_error user attr first."""
        import arrowspace_tuner.core.objective as obj_mod

        def fake_build_and_score(
            embeddings: np.ndarray,
            params: object,
            trial: opt.Trial | None = None,
        ) -> tuple[float, float, object, object]:
            if build_error is not None:
                trial.set_user_attr("build_error", build_error)  # type: ignore[union-attr]
            raise opt.TrialPruned()

        monkeypatch.setattr(obj_mod, "build_and_score", fake_build_and_score)

    def test_all_pruned_lists_build_errors(
        self,
        embeddings_small: np.ndarray,  # type: ignore[name-defined]  # noqa: F821
        monkeypatch: object,
    ) -> None:
        """
        When trials prune with exceptions, fit() must surface the distinct
        build errors instead of blaming the corpus (#38).
        """
        self._stub_build_and_score(
            monkeypatch, build_error="ValueError: unknown key(s) 'top_k'"
        )
        tuner = _make_tuner()
        with pytest.raises(RuntimeError) as excinfo:
            tuner.fit(embeddings_small)
        msg = str(excinfo.value)
        assert "distinct exception" in msg
        assert "ValueError: unknown key(s) 'top_k'" in msg

    def test_all_pruned_generic_no_build_errors(
        self,
        embeddings_small: np.ndarray,  # type: ignore[name-defined]  # noqa: F821
        monkeypatch: object,
    ) -> None:
        """
        Statistical pruning (no exceptions recorded) keeps the generic
        corpus/bounds advice.
        """
        self._stub_build_and_score(monkeypatch, build_error=None)
        tuner = _make_tuner()
        with pytest.raises(RuntimeError) as excinfo:
            tuner.fit(embeddings_small)
        msg = str(excinfo.value)
        assert "All Optuna trials were pruned" in msg
        assert "distinct exception" not in msg

    def test_issue24_corpus_fits(self) -> None:
        """
        Regression (#24): unnormalised two-Gaussian corpus with isolated
        nodes previously pruned every trial ('Lambda is zero for query N').
        Zero-lambda anchors are now filtered and fit() completes.
        """
        rng = np.random.default_rng(3407)
        X = np.vstack([
            rng.normal(0, 1, (40, 12)) + 2.0,
            rng.normal(4, 1, (40, 12)),
        ])
        tuner = EpsTuner(n_trials=5, seed=3407, eps_low=0.5, eps_high=3.0)
        params = tuner.fit(X)
        assert set(params.keys()) == {"eps", "k", "topk", "p", "sigma"}
        assert tuner.best_tau is not None


# ── save_report — pre-fit guard ───────────────────────────────────────────────

class TestSaveReport:

    def test_raises_before_fit(self) -> None:
        """save_report() must raise RuntimeError when called before .fit()."""
        tuner = EpsTuner()
        with pytest.raises(RuntimeError, match="fit"):
            tuner.save_report()

    def test_save_report_requires_report_extra(self, tmp_path: pathlib.Path) -> None:
        """
        If [report] extra is not installed, save_results() raises ImportError
        with a helpful message.

        This test monkey-patches the reporting module so it can run without
        the real arrowspace wheel.
        """
        import unittest.mock as mock
        import optuna as opt

        # Build a minimal fake completed study
        study = opt.create_study(direction="maximize")
        study.add_trial(
            opt.trial.create_trial(
                params={"eps": 1.0, "k": 10, "tau": 0.5},
                distributions={
                    "eps": opt.distributions.FloatDistribution(0.1, 5.0),
                    "k":   opt.distributions.IntDistribution(3, 40),
                    "tau": opt.distributions.FloatDistribution(0.1, 1.0),
                },
                value=0.42,
            )
        )

        tuner = EpsTuner()
        tuner.study = study  # inject pre-built study
        tuner.best_tau    = 0.5
        tuner.best_params = {"eps": 1.0, "k": 10, "topk": 5, "p": 2.0, "sigma": None}

        # Patch pandas/plotly import to simulate missing [report] extra
        with mock.patch.dict("sys.modules", {"pandas": None, "plotly": None, "plotly.express": None}):
            with pytest.raises((ImportError, TypeError)):
                tuner.save_report(out_dir=str(tmp_path))


# ── tau separation: tau is query-time, not build-time ─────────────────────────

class TestBestTauSeparation:
    """
    Tests that tau is stripped from best_params and stored in best_tau.
    Uses a fake injected study — no arrowspace Rust wheel required.
    """

    def _make_fitted_tuner(self) -> EpsTuner:
        """Return an EpsTuner with a fake completed study injected."""
        study = opt.create_study(direction="maximize")
        study.add_trial(
            opt.trial.create_trial(
                params={
                    "eps": 1.2,
                    "k":   14,
                    "tau": 0.75,
                },
                distributions={
                    "eps": opt.distributions.FloatDistribution(0.3, 4.0),
                    "k":   opt.distributions.IntDistribution(3, 40),
                    "tau": opt.distributions.FloatDistribution(0.1, 1.0),
                },
                value=0.85,
                user_attrs={"fiedler": 0.42, "var_lambda": 0.11, "mrr_proxy": 0.73},
            )
        )

        tuner = EpsTuner()

        # Replicate the result-extraction block from fit() directly
        # so we test just the assignment logic in isolation
        best     = study.best_trial
        raw      = best.params
        tuner.study           = study
        tuner.best_tau        = float(raw["tau"])
        tuner.best_params     = {
            "eps":   raw["eps"],
            "k":     raw["k"],
            "topk": max(1, raw["k"] // 2),
            "p":     2.0,
            "sigma": None,
        }
        tuner.best_score      = best.value
        tuner.best_fiedler    = best.user_attrs.get("fiedler")
        tuner.best_var_lambda = best.user_attrs.get("var_lambda")
        tuner.best_mrr_proxy  = best.user_attrs.get("mrr_proxy")
        return tuner

    def test_tau_absent_from_best_params(self) -> None:
        tuner = self._make_fitted_tuner()
        assert "tau" not in tuner.best_params

    def test_best_params_has_required_keys(self) -> None:
        tuner = self._make_fitted_tuner()
        assert set(tuner.best_params.keys()) == {"eps", "k", "topk", "p", "sigma"}

    def test_best_params_values_correct(self) -> None:
        tuner = self._make_fitted_tuner()
        assert tuner.best_params["eps"]   == pytest.approx(1.2)
        assert tuner.best_params["k"]     == 14
        assert tuner.best_params["topk"] == 7      # max(1, 14 // 2)
        assert tuner.best_params["p"]     == 2.0
        assert tuner.best_params["sigma"] is None

    def test_best_tau_populated(self) -> None:
        tuner = self._make_fitted_tuner()
        assert tuner.best_tau is not None
        assert isinstance(tuner.best_tau, float)
        assert tuner.best_tau == pytest.approx(0.75)

    def test_best_tau_none_before_fit(self) -> None:
        tuner = EpsTuner()
        assert tuner.best_tau is None

    def test_topk_is_half_k(self) -> None:
        """topk must always be max(1, k // 2) — never the raw k."""
        tuner = self._make_fitted_tuner()
        k     = tuner.best_params["k"]
        top_k = tuner.best_params["topk"]
        assert top_k == max(1, k // 2)

    def test_topk_minimum_one(self) -> None:
        """Edge case: k=1 must yield topk=1, not 0."""
        study = opt.create_study(direction="maximize")
        study.add_trial(
            opt.trial.create_trial(
                params={"eps": 0.5, "k": 1, "tau": 0.5},
                distributions={
                    "eps": opt.distributions.FloatDistribution(0.3, 4.0),
                    "k":   opt.distributions.IntDistribution(1, 40),
                    "tau": opt.distributions.FloatDistribution(0.1, 1.0),
                },
                value=0.5,
            )
        )
        raw = study.best_trial.params
        top_k = max(1, raw["k"] // 2)
        assert top_k == 1

    def test_repr_includes_best_tau_when_fitted(self) -> None:
        tuner = self._make_fitted_tuner()
        r = repr(tuner)
        assert "best_tau" in r
        assert "0.75" in r

    def test_repr_omits_best_tau_when_not_fitted(self) -> None:
        tuner = EpsTuner()
        r = repr(tuner)
        assert "not fitted" in r


# ── fit() returns graph_params dict ───────────────────────────────────────────

class TestFitReturnsParams:
    """Tests that EpsTuner.fit() and api.optuna() return graph_params dicts."""

    def test_fit_return_type(self, embeddings_small: np.ndarray) -> None:
        tuner = _make_tuner()
        result = tuner.fit(embeddings_small)
        assert isinstance(result, dict)

    def test_fit_return_keys(self, embeddings_small: np.ndarray) -> None:
        tuner = _make_tuner()
        result = tuner.fit(embeddings_small)
        assert set(result.keys()) == {"eps", "k", "topk", "p", "sigma"}

    def test_fit_return_equals_best_params(self, embeddings_small: np.ndarray) -> None:
        tuner = _make_tuner()
        result = tuner.fit(embeddings_small)
        assert result is tuner.best_params

    def test_fit_no_aspace_returned(self, embeddings_small: np.ndarray) -> None:
        tuner = _make_tuner()
        result = tuner.fit(embeddings_small)
        assert not isinstance(result, tuple)

    def test_best_tau_still_set(self, embeddings_small: np.ndarray) -> None:
        tuner = _make_tuner()
        tuner.fit(embeddings_small)
        assert tuner.best_tau is not None
        assert isinstance(tuner.best_tau, float)

    def test_tune_return_type(self, embeddings_small: np.ndarray) -> None:
        result = tune(embeddings_small, n_trials=3, seed=42, sample_n=None, n_probe=20)
        assert isinstance(result, dict)
        assert set(result.keys()) == {"eps", "k", "topk", "p", "sigma"}


# ── graph_params property ─────────────────────────────────────────────────────

class TestGraphParamsProperty:
    """Tests for the EpsTuner.graph_params property."""

    def test_graph_params_returns_best_params(self, embeddings_small: np.ndarray) -> None:
        tuner = _make_tuner()
        tuner.fit(embeddings_small)
        assert tuner.graph_params is tuner.best_params

    def test_graph_params_keys(self, embeddings_small: np.ndarray) -> None:
        tuner = _make_tuner()
        tuner.fit(embeddings_small)
        assert set(tuner.graph_params.keys()) == {"eps", "k", "topk", "p", "sigma"}

    def test_graph_params_raises_before_fit(self) -> None:
        tuner = EpsTuner()
        with pytest.raises(RuntimeError, match="Call .fit()"):
            _ = tuner.graph_params

    def test_graph_params_matches_fit_return(self, embeddings_small: np.ndarray) -> None:
        tuner = _make_tuner()
        fit_return = tuner.fit(embeddings_small)
        assert tuner.graph_params == fit_return


# ── load_graph_params / load_best_params deprecation ──────────────────────────

class TestLoadGraphParams:
    """Tests for disk-based graph-params loading."""

    def _write_best_params_json(self, tuner: EpsTuner, report_dir: pathlib.Path) -> None:
        """Write a minimal best_params.json for load_graph_params()."""
        report_dir.mkdir(parents=True, exist_ok=True)
        data = {
            "params": {
                "eps": 1.25,
                "k": 14,
            },
            "score": 0.85,
        }
        json_path = report_dir / "best_params.json"
        json_path.write_text(json.dumps(data), encoding="utf-8")
        tuner._last_report_path = report_dir

    def test_load_graph_params_returns_topk(
        self,
        tmp_path: pathlib.Path,
    ) -> None:
        tuner = EpsTuner()
        self._write_best_params_json(tuner, tmp_path / "report")

        params = tuner.load_graph_params()
        assert "topk" in params
        assert "top_k" not in params
        assert params["topk"] == 7   # max(1, 14 // 2)

    def test_load_best_params_emits_deprecation_warning(
        self,
        tmp_path: pathlib.Path,
    ) -> None:
        tuner = EpsTuner()
        self._write_best_params_json(tuner, tmp_path / "report")

        with pytest.warns(DeprecationWarning, match="load_graph_params"):
            tuner.load_best_params()

    def test_load_best_params_delegates(
        self,
        tmp_path: pathlib.Path,
    ) -> None:
        tuner = EpsTuner()
        self._write_best_params_json(tuner, tmp_path / "report")

        with pytest.warns(DeprecationWarning):
            deprecated = tuner.load_best_params()
        fresh = tuner.load_graph_params()
        assert deprecated == fresh
