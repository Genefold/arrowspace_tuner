"""
test_api.py — tests for the one-liner convenience API.

These tests exercise the public tune() / optuna() entry points.
They require the arrowspace Rust wheel to be installed.
"""
from __future__ import annotations

import numpy as np
import pytest

import arrowspace_tuner as at
from arrowspace_tuner import EpsTuner


class TestTune:
    """Tests for arrowspace_tuner.tune()."""

    def test_minimal_usage(self, embeddings_small: np.ndarray) -> None:
        graph_params = at.tune(embeddings_small, n_trials=3, n_probe=20)
        assert isinstance(graph_params, dict)
        assert set(graph_params.keys()) == {"eps", "k", "topk", "p", "sigma"}

    def test_kwargs_forwarded(self, embeddings_small: np.ndarray) -> None:
        # n_trials and seed must reach EpsTuner without raising
        graph_params = at.tune(
            embeddings_small,
            n_trials=3,
            seed=42,
            n_probe=20,
        )
        assert isinstance(graph_params, dict)

    def test_n_jobs_forwarded(self, embeddings_small: np.ndarray) -> None:
        # n_jobs was not exposed by the old optuna() shim
        graph_params = at.tune(
            embeddings_small,
            n_trials=3,
            n_jobs=1,
            n_probe=20,
        )
        assert isinstance(graph_params, dict)

    def test_max_clusters_forwarded(self, embeddings_small: np.ndarray) -> None:
        # max_clusters was not exposed by the old optuna() shim
        graph_params = at.tune(
            embeddings_small,
            n_trials=3,
            max_clusters=10,
            n_probe=20,
        )
        assert isinstance(graph_params, dict)

    def test_tuner_arg_used(self, embeddings_small: np.ndarray) -> None:
        tuner = EpsTuner(n_trials=3, n_probe=20)
        graph_params = at.tune(embeddings_small, tuner=tuner)
        assert tuner.study is not None
        assert tuner.best_params is graph_params

    def test_tuner_arg_ignores_kwargs(self, embeddings_small: np.ndarray) -> None:
        """When tuner= is provided, extra kwargs are ignored."""
        tuner = EpsTuner(n_trials=3, n_probe=20, seed=42)
        # n_trials=99 is ignored because a pre-built tuner is passed in
        graph_params = at.tune(embeddings_small, tuner=tuner, n_trials=99)
        assert tuner.study is not None
        # The tuner used its own seed=42, n_trials=3 configuration
        assert len(tuner.study.trials) == 3
        assert isinstance(graph_params, dict)

    def test_best_tau_accessible_via_tuner_arg(self, embeddings_small: np.ndarray) -> None:
        tuner = EpsTuner(n_trials=3, n_probe=20)
        at.tune(embeddings_small, tuner=tuner)
        assert tuner.best_tau is not None
        assert isinstance(tuner.best_tau, float)


class TestOptuna:
    """Tests for the deprecated arrowspace_tuner.optuna() alias."""

    def test_optuna_emits_deprecation(self, embeddings_small: np.ndarray) -> None:
        with pytest.warns(DeprecationWarning, match="tune\\(\\)"):
            at.optuna(embeddings_small, n_trials=3, n_probe=20)

    def test_optuna_delegates_to_tune(self, embeddings_small: np.ndarray) -> None:
        with pytest.warns(DeprecationWarning):
            deprecated = at.optuna(
                embeddings_small,
                n_trials=3,
                seed=42,
                n_probe=20,
            )
        fresh = at.tune(
            embeddings_small,
            n_trials=3,
            seed=42,
            n_probe=20,
        )
        assert deprecated == fresh
