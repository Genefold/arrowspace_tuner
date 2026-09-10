"""
test_service.py — shared service layer contract (spec §12.2).

EpsTuner is monkeypatched where the test target is the service's own logic
(dry run, error conversion); real tuning runs use the fast 3-trial fixture.
"""
from __future__ import annotations

import dataclasses

import arrowspace_tuner.service as service_module
from arrowspace_tuner.io import InputValidationError, TuningExecutionError
from arrowspace_tuner.models import (
    GRAPH_BUILD_KEYS,
    SCHEMA_VERSION,
    TuneRequest,
    TuneResult,
    tune_result_to_dict,
)
from arrowspace_tuner.service import inspect_embeddings, run_tuning, validate_request

import pytest


# ── request validation ────────────────────────────────────────────────────────

def test_invalid_tau_bounds_return_validation_error(fast_tune_request) -> None:
    result = run_tuning(dataclasses.replace(fast_tune_request, tau_high=1.2))
    assert result.status == "validation_error"
    assert result.error_code == "invalid_tau_bounds"
    assert result.graph_params is None and result.best_tau is None


def test_invalid_eps_bounds_return_validation_error(fast_tune_request) -> None:
    low = run_tuning(dataclasses.replace(fast_tune_request, eps_low=0.0))
    assert low.error_code == "invalid_eps_bounds"
    high = run_tuning(dataclasses.replace(fast_tune_request, eps_high=0.1))
    assert high.error_code == "invalid_eps_bounds"


def test_invalid_k_bounds_return_validation_error(fast_tune_request) -> None:
    low = run_tuning(dataclasses.replace(fast_tune_request, k_low=0))
    assert low.error_code == "invalid_k_bounds"
    high = run_tuning(dataclasses.replace(fast_tune_request, k_high=1))
    assert high.error_code == "invalid_k_bounds"


def test_invalid_trial_and_job_counts_rejected(fast_tune_request) -> None:
    for field, value in (("n_trials", 0), ("n_probe", 0), ("n_jobs", 0),
                         ("sample_n", 1)):
        result = run_tuning(dataclasses.replace(fast_tune_request, **{field: value}))
        assert result.status == "validation_error"


def test_tau_outside_unit_interval_rejected_not_clamped(fast_tune_request) -> None:
    for kwargs in ({"tau_low": -0.1}, {"tau_high": 1.5}):
        result = run_tuning(dataclasses.replace(fast_tune_request, **kwargs))
        assert result.status == "validation_error"
        assert result.error_code == "invalid_tau_bounds"


def test_validate_request_raises(fast_tune_request) -> None:
    request = dataclasses.replace(fast_tune_request, n_probe=0)
    try:
        validate_request(request)
    except InputValidationError as exc:
        assert exc.code == "invalid_n_probe"
    else:
        raise AssertionError("expected InputValidationError")


# ── dry run ───────────────────────────────────────────────────────────────────

def test_dry_run_does_not_instantiate_eps_tuner(
    monkeypatch, fast_tune_request
) -> None:
    class _ExplodingTuner:
        def __init__(self, **kwargs: object) -> None:
            raise AssertionError("EpsTuner must not be constructed on dry_run")

    monkeypatch.setattr(service_module, "EpsTuner", _ExplodingTuner)
    result = run_tuning(dataclasses.replace(fast_tune_request, dry_run=True))
    assert result.status == "ok"
    assert result.graph_params is None and result.best_tau is None
    assert result.input_info is not None


# ── full tuning runs ──────────────────────────────────────────────────────────

def test_valid_request_returns_ok_result(fast_tune_request) -> None:
    result = run_tuning(fast_tune_request)
    assert result.status == "ok"
    assert result.graph_params is not None
    assert result.best_tau is not None
    assert result.input_info is not None


def test_result_graph_params_contain_exactly_expected_keys(fast_tune_request) -> None:
    result = run_tuning(fast_tune_request)
    assert result.graph_params is not None
    assert set(result.graph_params) == set(GRAPH_BUILD_KEYS)
    assert isinstance(result.graph_params["k"], int)
    assert isinstance(result.graph_params["topk"], int)


def test_result_excludes_tau_from_graph_params(fast_tune_request) -> None:
    result = run_tuning(fast_tune_request)
    assert result.graph_params is not None
    assert "tau" not in result.graph_params
    assert "top_k" not in result.graph_params
    assert result.best_tau is not None
    assert 0.0 <= result.best_tau <= 1.0


def test_k_high_above_n_items_adds_clip_warning(fast_tune_request) -> None:
    result = run_tuning(dataclasses.replace(fast_tune_request, k_high=200))
    assert result.status == "ok"
    assert result.graph_params is not None
    assert any("k_high exceeds" in w for w in result.warnings)
    assert result.graph_params["k"] <= 119


def test_all_pruned_tuner_failure_returns_tuning_error(
    monkeypatch, fast_tune_request
) -> None:
    class _PruningTuner:
        def __init__(self, **kwargs: object) -> None:
            pass

        def fit(self, embeddings: object) -> dict[str, object]:
            raise RuntimeError("All Optuna trials were pruned")

    monkeypatch.setattr(service_module, "EpsTuner", _PruningTuner)
    result = run_tuning(fast_tune_request)
    assert result.status == "tuning_error"
    assert result.error_code == "no_valid_trials"
    assert result.graph_params is None and result.best_tau is None
    assert result.error_message is not None


def test_unexpected_failure_raises_tuning_execution_error(
    monkeypatch, fast_tune_request
) -> None:
    class _BrokenTuner:
        def __init__(self, **kwargs: object) -> None:
            pass

        def fit(self, embeddings: object) -> dict[str, object]:
            raise ValueError("boom")

    monkeypatch.setattr(service_module, "EpsTuner", _BrokenTuner)
    with pytest.raises(TuningExecutionError):
        run_tuning(fast_tune_request)


def test_save_report_false_does_not_create_directory(
    tmp_path, fast_tune_request
) -> None:
    report_dir = tmp_path / "reports"
    request = dataclasses.replace(
        fast_tune_request, report_dir=report_dir, save_report=False
    )
    result = run_tuning(request)
    assert result.status == "ok"
    assert not report_dir.exists()
    assert result.report_path is None


def test_save_report_true_returns_report_path(tmp_path, fast_tune_request) -> None:
    report_dir = tmp_path / "reports"
    request = dataclasses.replace(
        fast_tune_request, report_dir=report_dir, save_report=True
    )
    result = run_tuning(request)
    assert result.status == "ok"
    assert result.report_path is not None
    assert report_dir.exists()


def test_same_seed_and_fixture_give_same_params_and_tau(fast_tune_request) -> None:
    first = run_tuning(fast_tune_request)
    second = run_tuning(fast_tune_request)
    assert first.graph_params == second.graph_params
    assert first.best_tau == second.best_tau


def test_result_json_serialisation_converts_non_finite_to_null() -> None:
    payload = tune_result_to_dict(
        TuneResult(best_score=float("nan"), best_tau=float("inf"))
    )
    assert payload["best_score"] is None
    assert payload["best_tau"] is None
    assert payload["schema_version"] == SCHEMA_VERSION


# ── inspect_embeddings ────────────────────────────────────────────────────────

def test_inspect_embeddings_returns_metadata_without_tuning(npy_file) -> None:
    info = inspect_embeddings(npy_file)
    assert info.n_items == 120
    assert info.n_dimensions == 64
    assert info.sha256 is not None