"""
test_tuner.py — integration tests for EpsTuner and the optuna() one-liner.

These tests exercise the full public API end-to-end.
They require the arrowspace Rust wheel to be installed.
"""
from __future__ import annotations

import pathlib

import numpy as np
import pytest

from arrowspace_tuner import EpsTuner, optuna

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
        tuner.best_params = {"eps": 1.0, "k": 10, "tau": 0.5}

        # Patch pandas/plotly import to simulate missing [report] extra
        with mock.patch.dict("sys.modules", {"pandas": None, "plotly": None, "plotly.express": None}):
            with pytest.raises((ImportError, TypeError)):
                tuner.save_report(out_dir=str(tmp_path))
