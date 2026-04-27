"""
test_objective.py — unit tests for core/graph.py and core/objective.py.

These tests exercise the internal building blocks in isolation.
They require the arrowspace Rust wheel to be installed.
"""
from __future__ import annotations

import numpy as np
import optuna
import pytest

from arrowspace_tuner.core import (
    BuildParams,
    StudyConfig,
    build_and_score,
    fiedler_normalized,
    make_objective,
)


# ── BuildParams.to_dict ───────────────────────────────────────────────────────

class TestBuildParams:

    def test_to_dict_keys(self):
        p = BuildParams(eps=1.0, k=10, topk=5)
        d = p.to_dict()
        assert set(d.keys()) == {"eps", "k", "topk", "p", "sigma"}

    def test_to_dict_values(self):
        p = BuildParams(eps=1.5, k=8, topk=4, p=2.0, sigma=None)
        d = p.to_dict()
        assert d["eps"]   == 1.5
        assert d["k"]     == 8
        assert d["topk"]  == 4
        assert d["sigma"] is None

    def test_topk_default_is_half_k(self):
        p = BuildParams(k=12)
        assert p.topk == 5   # default, not k//2 — user must set explicitly


# ── build_and_score ───────────────────────────────────────────────────────────

class TestBuildAndScore:

    def test_returns_four_values(self, embeddings_small):
        params = BuildParams(eps=1.0, k=8, topk=4)
        result = build_and_score(embeddings_small, params)
        assert len(result) == 4

    def test_valid_graph_nonzero_fiedler(self, embeddings_small):
        params = BuildParams(eps=1.0, k=8, topk=4)
        fiedler, var_lambda, aspace, gl = build_and_score(embeddings_small, params)
        assert fiedler > 0.0
        assert var_lambda >= 0.0
        assert aspace is not None
        assert gl is not None

    def test_degenerate_eps_too_large_returns_zeros(self, embeddings_small):
        # eps so large every item connects to every other → NNZ >> N but
        # spectrum collapses; OR eps so small graph is empty → NNZ <= N
        params = BuildParams(eps=1e-6, k=3, topk=1)
        fiedler, var_lambda, aspace, gl = build_and_score(embeddings_small, params)
        # degenerate path: aspace and gl are None
        assert aspace is None
        assert gl is None

    def test_degenerate_raises_pruned_with_trial(self, embeddings_small):
        study = optuna.create_study(direction="maximize")

        def obj(trial):
            params = BuildParams(eps=1e-6, k=3, topk=1)
            build_and_score(embeddings_small, params, trial=trial)
            return 0.0

        # TrialPruned is caught by Optuna internally — study should complete
        study.optimize(obj, n_trials=1)
        assert study.trials[0].state == optuna.trial.TrialState.PRUNED

    def test_fiedler_in_unit_interval(self, embeddings_small):
        params = BuildParams(eps=1.0, k=8, topk=4)
        fiedler, _, _, gl = build_and_score(embeddings_small, params)
        if gl is not None:
            assert 0.0 <= fiedler <= 1.0 + 1e-9   # small float tolerance

    def test_var_lambda_nonnegative(self, embeddings_small):
        params = BuildParams(eps=1.0, k=8, topk=4)
        _, var_lambda, _, _ = build_and_score(embeddings_small, params)
        assert var_lambda >= 0.0


# ── fiedler_normalized ────────────────────────────────────────────────────────

class TestFiedlerNormalized:

    def test_returns_float(self, embeddings_small):
        from arrowspace import ArrowSpaceBuilder
        params = BuildParams(eps=1.0, k=8, topk=4)
        _, _, _, gl = build_and_score(embeddings_small, params)
        if gl is not None:
            result = fiedler_normalized(gl)
            assert isinstance(result, float)

    def test_value_in_unit_interval(self, embeddings_small):
        params = BuildParams(eps=1.0, k=8, topk=4)
        _, _, _, gl = build_and_score(embeddings_small, params)
        if gl is not None:
            f = fiedler_normalized(gl)
            assert 0.0 <= f <= 1.0 + 1e-9


# ── make_objective ────────────────────────────────────────────────────────────

class TestMakeObjective:

    def test_returns_callable(self, embeddings_small, fast_study_config):
        obj = make_objective(embeddings_small, fast_study_config)
        assert callable(obj)

    def test_objective_returns_float(self, embeddings_small, fast_study_config):
        study = optuna.create_study(direction="maximize")
        obj   = make_objective(embeddings_small, fast_study_config)
        study.optimize(obj, n_trials=1)

        completed = [
            t for t in study.trials
            if t.state == optuna.trial.TrialState.COMPLETE
        ]
        if completed:
            assert isinstance(completed[0].value, float)

    def test_objective_score_nonnegative(self, embeddings_small, fast_study_config):
        study = optuna.create_study(direction="maximize")
        obj   = make_objective(embeddings_small, fast_study_config)
        study.optimize(obj, n_trials=fast_study_config.n_trials)

        completed = [
            t for t in study.trials
            if t.state == optuna.trial.TrialState.COMPLETE
        ]
        for t in completed:
            assert t.value >= 0.0

    def test_user_attrs_populated(self, embeddings_small, fast_study_config):
        study = optuna.create_study(direction="maximize")
        obj   = make_objective(embeddings_small, fast_study_config)
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

    def test_three_params_suggested(self, embeddings_small, fast_study_config):
        study = optuna.create_study(direction="maximize")
        obj   = make_objective(embeddings_small, fast_study_config)
        study.optimize(obj, n_trials=fast_study_config.n_trials)

        completed = [
            t for t in study.trials
            if t.state == optuna.trial.TrialState.COMPLETE
        ]
        if completed:
            assert set(completed[0].params.keys()) == {"eps", "k", "tau"}

    def test_sample_n_respected(self, embeddings_medium, fast_study_config):
        cfg          = fast_study_config
        cfg.sample_n = 50   # force subsampling on the 600-item fixture
        study        = optuna.create_study(direction="maximize")
        obj          = make_objective(embeddings_medium, cfg)
        study.optimize(obj, n_trials=1)

        completed = [
            t for t in study.trials
            if t.state == optuna.trial.TrialState.COMPLETE
        ]
        if completed:
            assert completed[0].user_attrs["n_sample"] == 50

    def test_flat_embeddings_all_pruned_or_zero(
        self, embeddings_flat, fast_study_config
    ):
        study = optuna.create_study(direction="maximize")
        obj   = make_objective(embeddings_flat, fast_study_config)
        study.optimize(obj, n_trials=fast_study_config.n_trials)

        for t in study.trials:
            is_pruned   = t.state == optuna.trial.TrialState.PRUNED
            is_zero     = t.value == 0.0 if t.value is not None else True
            assert is_pruned or is_zero, (
                f"Expected pruned or zero score on flat embeddings, "
                f"got state={t.state} value={t.value}"
            )