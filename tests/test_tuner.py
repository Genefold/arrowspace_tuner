"""
test_tuner.py — integration tests for EpsTuner and the optuna() one-liner.

These tests exercise the full public API end-to-end.
They require the arrowspace Rust wheel to be installed.
"""
from __future__ import annotations

import pytest

from arrowspace_tuner import EpsTuner, optuna

# ── helpers ───────────────────────────────────────────────────────────────────

def _make_tuner(**overrides) -> EpsTuner:
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

    def test_default_instantiation(self):
        tuner = EpsTuner()
        assert tuner.best_params     is None
        assert tuner.best_score      is None
        assert tuner.best_fiedler    is None
        assert tuner.best_var_lambda is None
        assert tuner.best_mrr_proxy  is None
        assert tuner.study           is None

    def test_repr_before_fit(self):
        tuner = _make_tuner()
        r = repr(tuner)
        assert "not fitted" in r
        assert "n_trials=3" in r

    def test_repr_reflects_bounds(self):
        tuner = EpsTuner(eps_low=0.1, eps_high=5.0)
        assert "0.1" in repr(tuner)
        assert "5.0" in repr(tuner)


# ── EpsTuner._validate ────────────────────────────────────────────────────────

class TestEpsTunerValidation:

    def test_raises_on_non_array(self):
        tuner = _make_tuner()
        with pytest.raises(ValueError, match="np.ndarray"):
            tuner.fit([[1.0, 2.0], [3.0, 4.0]])  # list, not ndarray

    def test_raises_on_1d(self, embeddings_1d):
        tuner = _make_tuner()
        with pytest.raises(ValueError, match="2D"):
            tuner.fit(embeddings_1d)

    def test_warns_on_float32(self, embeddings_wrong_dtype, caplog):
        import logging
        tuner = _make_tuner()
        with caplog.at_level(logging.WARNING, logger="arrowspace_tuner.tuner"):
            try:
                tuner.fit(embeddings_wrong_dtype)
            except Exception:
                pass   # may fail for other reasons on tiny array; warning is what we test
        assert any("float64" in m for m in caplog.messages)


# ── EpsTuner.fit — happy path ─────────────────────────────────────────────────

class TestEpsTunerFit:

    def test_returns_tuple_of_two(self, embeddings_small):
        tuner  = _make_tuner()
        result = tuner.fit(embeddings_small)
        assert isinstance(result, tuple)
        assert len(result) == 2

    def test_aspace_and_gl_not_none(self, embeddings_small):
        tuner       = _make_tuner()
        aspace, gl  = tuner.fit(embeddings_small)
        assert aspace is not None
        assert gl     is not None

    def test_best_params_populated(self, embeddings_small):
        tuner = _make_tuner()
        tuner.fit(embeddings_small)
        assert tuner.best_params is not None
        assert "eps" in tuner.best_params
        assert "k"   in tuner.best_params
        assert "tau" in tuner.best_params

    def test_best_params_within_bounds(self, embeddings_small):
        tuner = _make_tuner(eps_low=0.5, eps_high=2.0, k_low=3, k_high=10)
        tuner.fit(embeddings_small)
        assert 0.5 <= tuner.best_params["eps"] <= 2.0
        assert 3   <= tuner.best_params["k"]   <= 10

    def test_best_score_positive(self, embeddings_small):
        tuner = _make_tuner()
        tuner.fit(embeddings_small)
        assert tuner.best_score is not None
        assert tuner.best_score > 0.0

    def test_spectral_attrs_populated(self, embeddings_small):
        tuner = _make_tuner()
        tuner.fit(embeddings_small)
        assert tuner.best_fiedler    is not None
        assert tuner.best_var_lambda is not None
        assert tuner.best_mrr_proxy  is not None
        assert 0.0 <= tuner.best_fiedler    <= 1.0 + 1e-9
        assert tuner.best_var_lambda        >= 0.0
        assert tuner.best_mrr_proxy         >= 0.0

    def test_study_stored(self, embeddings_small):
        import optuna as optuna_lib
        tuner = _make_tuner()
        tuner.fit(embeddings_small)
        assert isinstance(tuner.study, optuna_lib.Study)

    def test_repr_after_fit(self, embeddings_small):
        tuner = _make_tuner()
        tuner.fit(embeddings_small)
        r = repr(tuner)
        assert "not fitted" not in r
        assert "best_score" in r

    def test_fit_is_reproducible(self, embeddings_small):
        tuner_a = _make_tuner(seed=0)
        tuner_b = _make_tuner(seed=0)
        tuner_a.fit(embeddings_small)
        tuner_b.fit(embeddings_small)
        assert tuner_a.best_params == tuner_b.best_params

    def test_different_seeds_may_differ(self, embeddings_small):
        tuner_a = _make_tuner(seed=0,  n_trials=5)
        tuner_b = _make_tuner(seed=99, n_trials=5)
        tuner_a.fit(embeddings_small)
        tuner_b.fit(embeddings_small)
        # Not guaranteed to differ, but almost always will across 5 trials
        # We just check both completed without error
        assert tuner_a.best_score is not None
        assert tuner_b.best_score is not None

    def test_sample_n_subsampling(self, embeddings_medium):
        tuner = _make_tuner(sample_n=80)
        tuner.fit(embeddings_medium)
        # study user_attrs should report n_sample <= 80
        completed = [
            t for t in tuner.study.trials
            if t.state.name == "COMPLETE"
        ]
        if completed:
            assert completed[0].user_attrs["n_sample"] <= 80

    def test_final_build_uses_full_corpus(self, embeddings_medium):
        # Even with sample_n set, the returned gl should reflect full N
        tuner      = _make_tuner(sample_n=80)
        aspace, gl = tuner.fit(embeddings_medium)
        n_full     = len(embeddings_medium)
        assert len(aspace.lambdas()) == n_full


# ── EpsTuner.fit — error paths ────────────────────────────────────────────────

class TestEpsTunerFitErrors:

    def test_raises_runtime_error_on_all_pruned(self, embeddings_flat):
        # Flat embeddings → all trials pruned → RuntimeError
        tuner = EpsTuner(
            n_trials  = 3,
            eps_low   = 1e-7,
            eps_high  = 1e-6,   # so small every graph is empty
            k_low     = 3,
            k_high    = 5,
            tau_low   = 0.1,
            tau_high  = 0.5,
            n_probe   = 10,
        )
        with pytest.raises(RuntimeError, match="pruned"):
            tuner.fit(embeddings_flat)


# ── EpsTuner.save_report ──────────────────────────────────────────────────────

class TestEpsTunerSaveReport:

    def test_raises_before_fit(self, tmp_path):
        tuner = _make_tuner()
        with pytest.raises(RuntimeError, match="fit"):
            tuner.save_report(out_dir=str(tmp_path))

    def test_save_report_creates_files(self, embeddings_small, tmp_path):
        pytest.importorskip("pandas")   # skip if [report] not installed
        pytest.importorskip("plotly")

        tuner = _make_tuner()
        tuner.fit(embeddings_small)
        run_dir = tuner.save_report(out_dir=str(tmp_path))

        assert (run_dir / "best_params.json").exists()
        assert (run_dir / "trials.csv").exists()

    def test_best_params_json_content(self, embeddings_small, tmp_path):
        pytest.importorskip("pandas")
        pytest.importorskip("plotly")

        import json
        tuner = _make_tuner()
        tuner.fit(embeddings_small)
        run_dir = tuner.save_report(out_dir=str(tmp_path))

        data = json.loads((run_dir / "best_params.json").read_text())
        assert "params" in data
        assert "eps" in data["params"]
        assert "k"   in data["params"]
        assert "tau" in data["params"]
        assert "score" in data


# ── optuna() one-liner ────────────────────────────────────────────────────────

class TestOptunaOneLiner:

    def test_returns_two_objects(self, embeddings_small):
        result = optuna(
            embeddings_small,
            n_trials = 3,
            eps_low  = 0.5,
            eps_high = 2.0,
            k_low    = 3,
            k_high   = 10,
            tau_low  = 0.3,
            tau_high = 1.2,
            n_probe  = 20,
        )
        assert isinstance(result, tuple)
        assert len(result) == 2

    def test_aspace_and_gl_not_none(self, embeddings_small):
        aspace, gl = optuna(
            embeddings_small,
            n_trials = 3,
            eps_low  = 0.5,
            eps_high = 2.0,
            k_low    = 3,
            k_high   = 10,
            n_probe  = 20,
        )
        assert aspace is not None
        assert gl     is not None

    def test_accepts_keyword_overrides(self, embeddings_small):
        # Should not raise — all params are keyword-only
        aspace, gl = optuna(
            embeddings_small,
            n_trials   = 3,
            seed       = 7,
            eps_low    = 0.8,
            eps_high   = 1.5,
            k_low      = 4,
            k_high     = 8,
            tau_low    = 0.5,
            tau_high   = 1.0,
            n_probe    = 15,
            study_name = "test_one_liner",
        )
        assert aspace is not None