from __future__ import annotations

import numpy as np
import optuna
import pytest

from arrowspace_tuner import BuildParams, StudyConfig
from arrowspace_tuner.core.objective import build_and_score, make_objective


def test_relative_pruning_does_not_discard_refinement(
    embeddings_small: np.ndarray,
) -> None:
    study = optuna.create_study(pruner=optuna.pruners.ThresholdPruner(lower=1e100))
    trial = study.ask()
    trial.set_user_attr("phase", "refinement")
    # Same graph validity checks, but a statistical pruner cannot kill the trial.
    fiedler, variance, aspace, gl = build_and_score(
        embeddings_small,
        BuildParams(eps=1.0, k=10),
        trial,
    )
    assert fiedler > 1e-6 and variance > 0
    assert aspace is not None and gl is not None
    main = study.ask()
    main.set_user_attr("phase", "main")
    with pytest.raises(optuna.TrialPruned):
        build_and_score(embeddings_small, BuildParams(eps=1.0, k=10), main)


def test_objective_failures_are_pruned_not_zero_winners(
    monkeypatch: pytest.MonkeyPatch,
    embeddings_small: np.ndarray,
) -> None:
    import arrowspace_tuner.core.objective as module

    def broken(*args: object, **kwargs: object) -> object:
        raise ValueError("evaluation failed")

    monkeypatch.setattr(module, "build_and_score", broken)
    objective, _ = make_objective(embeddings_small, StudyConfig())
    study = optuna.create_study()
    study.optimize(objective, n_trials=1)
    trial = study.trials[0]
    assert trial.state == optuna.trial.TrialState.PRUNED
    assert not trial.user_attrs.get("evaluation_success")
    assert "evaluation failed" in trial.user_attrs["build_error"]


def test_subsample_and_probes_shared_across_phases(
    monkeypatch: pytest.MonkeyPatch,
    embeddings_small: np.ndarray,
) -> None:
    import arrowspace_tuner.core.objective as module

    corpora: list[np.ndarray] = []
    queries: list[np.ndarray] = []

    class FakeSpace:
        def lambdas(self) -> np.ndarray:
            return np.linspace(0.1, 1.0, 60)

        def search_batch(self, probe: np.ndarray, gl: object, tau: float) -> list[object]:
            assert tau == 0.5
            queries.append(probe.copy())
            return [[(0, 1.0)]] * len(probe)

    def build(data: np.ndarray, params: object, trial: object) -> tuple[object, ...]:
        corpora.append(data.copy())
        return 0.5, 0.2, FakeSpace(), object()

    monkeypatch.setattr(module, "build_and_score", build)
    objective, cache = make_objective(embeddings_small, StudyConfig(sample_n=60))
    study = optuna.create_study(direction="maximize")
    study.enqueue_trial({"eps": 1.0, "k": 10}, user_attrs={"phase": "main"})
    study.enqueue_trial({"eps": 2.0, "k": 3}, user_attrs={"phase": "refinement"})
    study.optimize(objective, n_trials=2)
    assert all(t.state == optuna.trial.TrialState.COMPLETE for t in study.trials)
    np.testing.assert_array_equal(corpora[0], corpora[1])
    np.testing.assert_array_equal(queries[0], queries[1])
    assert cache == {}
