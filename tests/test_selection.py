from __future__ import annotations

import optuna
import pytest

from arrowspace_tuner.core.config import StudyConfig
from arrowspace_tuner.core.selection import refinement_schedule, select_trial


def test_refinement_deliberately_evaluates_low_k_high_eps() -> None:
    cfg = StudyConfig(selection_policy="lowest_k", refinement_trials=8)
    schedule = refinement_schedule(cfg, k_anchor=20, eps_anchor=1.0)
    assert schedule[0] == {"k": cfg.k_low, "eps": cfg.eps_high}
    assert len(schedule) == 8
    assert len({point["k"] for point in schedule}) == 8
    assert all(cfg.k_low <= point["k"] < 20 for point in schedule)
    assert all(cfg.eps_low <= point["eps"] <= cfg.eps_high for point in schedule)
    assert schedule == refinement_schedule(cfg, k_anchor=20, eps_anchor=1.0)


def test_refinement_skips_when_anchor_is_already_minimum() -> None:
    cfg = StudyConfig(selection_policy="lowest_k", refinement_trials=4)
    assert refinement_schedule(cfg, k_anchor=cfg.k_low, eps_anchor=1.0) == []


def test_full_range_slots_explore_below_anchor_eps() -> None:
    cfg = StudyConfig(selection_policy="lowest_k", refinement_trials=8)
    points = refinement_schedule(cfg, k_anchor=20, eps_anchor=2.99)
    assert any(p["eps"] < 2.99 for p in points)


def test_selection_bounds_quality_loss_against_global_maximum() -> None:
    study = optuna.create_study(direction="maximize")
    for k, score in [(20, 1.0), (8, 0.995), (3, 0.989)]:
        study.add_trial(
            optuna.trial.create_trial(
                value=score,
                params={"k": k, "eps": 1.0},
                distributions={
                    "k": optuna.distributions.IntDistribution(3, 40),
                    "eps": optuna.distributions.FloatDistribution(0.5, 3.0, log=True),
                },
                user_attrs={"evaluation_success": True},
            )
        )
    cfg = StudyConfig(selection_policy="lowest_k", refinement_trials=4, score_tolerance=0.01)
    assert select_trial(study.trials, cfg).params["k"] == 8
    cfg.score_tolerance = 0.0
    assert select_trial(study.trials, cfg).params["k"] == 20


def test_tolerance_boundary_is_inclusive() -> None:
    study = optuna.create_study(direction="maximize")
    for k, score in [(20, 1.0), (3, 0.875)]:
        study.add_trial(
            optuna.trial.create_trial(
                value=score,
                params={"k": k, "eps": 1.0},
                distributions={
                    "k": optuna.distributions.IntDistribution(3, 40),
                    "eps": optuna.distributions.FloatDistribution(0.5, 3.0, log=True),
                },
                user_attrs={"evaluation_success": True},
            )
        )
    cfg = StudyConfig(
        selection_policy="lowest_k",
        refinement_trials=1,
        score_tolerance=0.125,
    )
    assert select_trial(study.trials, cfg).params["k"] == 3
    cfg.score_tolerance = 0.124
    assert select_trial(study.trials, cfg).params["k"] == 20


@pytest.mark.parametrize(
    "kwargs",
    [
        {"selection_policy": "unknown"},
        {"selection_policy": "lowest_k"},
        {"refinement_trials": True},
        {"refinement_trials": 1.5},
        {"score_tolerance": float("nan")},
        {"score_tolerance": float("inf")},
        {"score_tolerance": -1.0},
        {"score_tolerance": 0.1},
    ],
)
def test_invalid_selection_configuration(kwargs: dict[str, object]) -> None:
    with pytest.raises(ValueError):
        StudyConfig(**kwargs)


@pytest.mark.parametrize("budget", [1, 2, 3, 4, 8, 20])
@pytest.mark.parametrize("anchor_k", [4, 5, 20])
@pytest.mark.parametrize("anchor_eps", [1.0, 3.0])
def test_schedule_small_budgets_and_collapsed_regions(
    budget: int,
    anchor_k: int,
    anchor_eps: float,
) -> None:
    cfg = StudyConfig(selection_policy="lowest_k", refinement_trials=budget)
    points = refinement_schedule(cfg, k_anchor=anchor_k, eps_anchor=anchor_eps)
    assert len(points) == budget
    assert points[0] == {"k": 3, "eps": 3.0}
    assert all(3 <= p["k"] < anchor_k and 0.5 <= p["eps"] <= 3.0 for p in points)
    if budget >= anchor_k - 3:
        assert set(p["k"] for p in points[: anchor_k - 3]) == set(range(3, anchor_k))
    if budget >= 2:
        assert any(p["eps"] != 3.0 for p in points)


def test_selection_excludes_sentinels_and_breaks_ties_deterministically() -> None:
    from optuna.distributions import FloatDistribution, IntDistribution
    from optuna.trial import TrialState, create_trial

    study = optuna.create_study(direction="maximize")
    dist = {"k": IntDistribution(3, 40), "eps": FloatDistribution(0.5, 3.0, log=True)}
    for k, score, attrs in [
        (3, 10.0, {}),  # failure-sentinel COMPLETE without success
        (3, float("inf"), {"evaluation_success": True}),
        (8, 1.00000001, {"evaluation_success": True}),
        (8, 1.00000002, {"evaluation_success": True}),
        (8, 1.00000002, {"evaluation_success": True}),
        (20, 1.00000003, {"evaluation_success": True}),
    ]:
        study.add_trial(
            create_trial(
                value=score,
                params={"k": k, "eps": 1.0},
                distributions=dist,
                user_attrs=attrs,
            )
        )
    study.add_trial(create_trial(state=TrialState.PRUNED, value=100.0))
    cfg = StudyConfig(selection_policy="lowest_k", refinement_trials=1, score_tolerance=0.01)
    assert select_trial(study.trials, cfg).number == 3
    assert select_trial(list(reversed(study.trials)), cfg).number == 3
    cfg.score_tolerance = 0.0
    assert select_trial(study.trials, cfg).number == 5
    with pytest.raises(RuntimeError, match="No valid"):
        select_trial(study.trials[:2], cfg)
