from __future__ import annotations

import importlib.util
import json
from pathlib import Path

import numpy as np
import optuna
import pytest

from arrowspace_tuner import EpsTuner


def test_refinement_finds_missed_low_k_region(monkeypatch: pytest.MonkeyPatch) -> None:
    import arrowspace_tuner.tuner as module

    def factory(embeddings: np.ndarray, cfg: object) -> tuple[object, dict[str, object]]:
        def objective(trial: optuna.Trial) -> float:
            k = trial.suggest_int("k", 3, 40)
            eps = trial.suggest_float("eps", 0.3, 4.0, log=True)
            trial.set_user_attr("evaluation_success", True)
            # Both corners are equally good, other configurations are worse.
            return 1.0 if (k == 15 and eps == 1.0) or (k == 3 and eps == 4.0) else 0.1

        return objective, {}

    monkeypatch.setattr(module, "make_objective", factory)
    tuner = EpsTuner(n_trials=1, selection_policy="lowest_k", refinement_trials=4)
    result = tuner.fit(np.ones((50, 4)))
    assert result == {"eps": 4.0, "k": 3, "topk": 1, "p": 2.0, "sigma": None}
    assert len(tuner.study.trials) == 5
    assert tuner.selection["refinement_trials_actual"] == 4
    assert tuner.study.best_trial.params["k"] == 15
    assert all("tau" not in trial.params for trial in tuner.study.trials)
    assert not hasattr(tuner, "best_tau")


def test_report_uses_selected_trial_and_reload(
    monkeypatch: pytest.MonkeyPatch, tmp_path: Path
) -> None:
    import arrowspace_tuner.tuner as module

    def factory(embeddings: np.ndarray, cfg: object) -> tuple[object, dict[str, object]]:
        def objective(trial: optuna.Trial) -> float:
            trial.suggest_int("k", 3, 40)
            trial.suggest_float("eps", 0.3, 4.0, log=True)
            trial.set_user_attr("evaluation_success", True)
            return 1.0 if trial.user_attrs["phase"] == "main" else 0.875

        return objective, {}

    monkeypatch.setattr(module, "make_objective", factory)
    tuner = EpsTuner(
        n_trials=1,
        selection_policy="lowest_k",
        refinement_trials=2,
        score_tolerance=0.125,
    )
    params = dict(tuner.fit(np.ones((50, 4))))
    assert params["k"] == 3
    assert tuner.best_score == 0.875
    selected_number = tuner.selection["selected_trial_number"]
    if importlib.util.find_spec("plotly") is None or importlib.util.find_spec("pandas") is None:
        with pytest.raises(ImportError):
            tuner.save_report(str(tmp_path))
        assert tuner.graph_params == params
        assert tuner.selection["selected_trial_number"] == selected_number
        return
    report = tuner.save_report(str(tmp_path))
    saved = json.loads((report / "best_params.json").read_text())
    assert saved["params"] == params
    assert saved["trial"] == selected_number
    assert saved["selection"]["raw_best_trial_number"] == 0
    assert saved["score"] == 0.875
    assert saved["selection"]["raw_best_score"] == 1.0
    assert saved["selection"]["selected_score_gap"] == 0.125
    assert tuner.load_graph_params() == params
    assert "tau" not in (report / "trials.csv").read_text()
    # A later invocation must not overwrite earlier report snapshots.
    tuner.fit(np.ones((50, 4)))
    assert json.loads((report / "best_params.json").read_text()) == saved


def test_compatible_resume_and_data_mismatch(
    monkeypatch: pytest.MonkeyPatch, tmp_path: Path
) -> None:
    import arrowspace_tuner.tuner as module

    def factory(embeddings: np.ndarray, cfg: object) -> tuple[object, dict[str, object]]:
        def objective(trial: optuna.Trial) -> float:
            trial.suggest_int("k", 3, 40)
            trial.suggest_float("eps", 0.3, 4.0, log=True)
            trial.set_user_attr("evaluation_success", True)
            return 1.0

        return objective, {}

    monkeypatch.setattr(module, "make_objective", factory)
    storage = f"sqlite:///{tmp_path / 'resume.db'}"
    first = EpsTuner(n_trials=1, storage=storage)
    first.fit(np.ones((50, 4)))
    second = EpsTuner(
        n_trials=1,
        storage=storage,
        selection_policy="lowest_k",
        refinement_trials=2,
        score_tolerance=0.01,
    )
    second.fit(np.ones((50, 4)))
    assert len(second.study.trials) == 4
    assert second.selection["refinement_trials_actual"] == 2
    with pytest.raises(RuntimeError, match="new study"):
        second.fit(np.zeros((50, 4)))
    assert len(second.study.trials) == 4


def test_legacy_study_rejected_before_new_trials(tmp_path: object) -> None:
    from pathlib import Path

    storage = f"sqlite:///{Path(str(tmp_path)) / 'legacy.db'}"
    study = optuna.create_study(storage=storage, study_name="legacy")
    study.enqueue_trial({"eps": 1.0, "k": 15, "tau": 0.8})
    tuner = EpsTuner(storage=storage, study_name="legacy")
    with pytest.raises(RuntimeError, match="new study"):
        tuner.fit(np.ones((50, 4)))
    assert len(optuna.load_study(storage=storage, study_name="legacy").trials) == 1


@pytest.mark.parametrize("fail_refinement", [False, True])
def test_refinement_improvement_or_all_failed_fallback(
    monkeypatch: pytest.MonkeyPatch,
    fail_refinement: bool,
) -> None:
    import arrowspace_tuner.tuner as module

    def factory(embeddings: np.ndarray, cfg: object) -> tuple[object, dict[str, object]]:
        def objective(trial: optuna.Trial) -> float:
            trial.suggest_int("k", 3, 40)
            trial.suggest_float("eps", 0.3, 4.0, log=True)
            if trial.user_attrs["phase"] == "refinement" and fail_refinement:
                raise optuna.TrialPruned()
            trial.set_user_attr("evaluation_success", True)
            return 1.0 if trial.user_attrs["phase"] == "main" else 2.0

        return objective, {}

    monkeypatch.setattr(module, "make_objective", factory)
    tuner = EpsTuner(n_trials=1, selection_policy="lowest_k", refinement_trials=2)
    params = tuner.fit(np.ones((50, 4)))
    assert params["k"] == (15 if fail_refinement else 3)
    assert tuner.best_score == (1.0 if fail_refinement else 2.0)
    assert len(tuner.study.trials) == 3
    assert tuner.selection["refinement_trials_actual"] == 2
