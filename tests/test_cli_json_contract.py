"""
test_cli_json_contract.py — the JSON machine-output contract (spec §12.4).

One real tune run is shared across the tests via a module fixture; the
stdout contract (exactly one JSON document, no ANSI escapes, topk not
top_k, best_tau outside graph_params) is checked against that output.
"""

from __future__ import annotations

import json
from pathlib import Path

import numpy as np
import pytest
from click.testing import CliRunner

import arrowspace_tuner.service as service_module
from arrowspace_tuner import cli

CLI_BOUNDS = [
    "--trials",
    "3",
    "--seed",
    "42",
    "--k-low",
    "3",
    "--k-high",
    "10",
    "--eps-low",
    "0.5",
    "--eps-high",
    "2.0",
    "--n-probe",
    "20",
]


@pytest.fixture
def runner() -> CliRunner:
    return CliRunner()


@pytest.fixture
def json_run(embeddings_small: np.ndarray, tmp_path: Path) -> str:
    """Run `tune --format json` once and share (stdout, result object)."""
    runner = CliRunner()
    path = tmp_path / "embeddings.npy"
    np.save(path, embeddings_small)
    result = runner.invoke(
        cli.main,
        ["tune", str(path), *CLI_BOUNDS, "--format", "json"],
    )
    assert result.exit_code == 0, result.output
    return result.stdout


def test_json_output_parses(json_run: str) -> None:
    payload = json.loads(json_run)
    assert payload["status"] == "ok"


def test_stdout_contains_exactly_one_json_document(json_run: str) -> None:
    decoder = json.JSONDecoder()
    text = json_run.strip()
    payload, idx = decoder.raw_decode(text)
    assert payload["schema_version"] == "1.0"
    assert text[idx:].strip() == ""


def test_stdout_has_no_ansi_escapes(json_run: str) -> None:
    assert "\x1b" not in json_run


def test_logs_never_appear_on_stdout(json_run: str) -> None:
    # every non-empty stdout line must belong to the single JSON document
    lines = [line for line in json_run.strip().splitlines() if line.strip()]
    assert len(lines) == 1
    json.loads(lines[0])


def test_graph_params_contain_topk_not_top_k(json_run: str) -> None:
    payload = json.loads(json_run)
    assert "topk" in payload["graph_params"]
    assert "top_k" not in payload["graph_params"]


def test_graph_params_contain_no_tau(json_run: str) -> None:
    payload = json.loads(json_run)
    assert "tau" not in payload["graph_params"]


def test_best_tau_is_top_level(json_run: str) -> None:
    payload = json.loads(json_run)
    assert 0.0 <= payload["best_tau"] <= 1.0


def test_schema_version_is_present(json_run: str) -> None:
    assert json.loads(json_run)["schema_version"] == "1.0"


def test_failure_output_is_valid_json(runner: CliRunner, npy_file: Path) -> None:
    result = runner.invoke(
        cli.main, ["tune", str(npy_file), "--tau-high", "2.0", "--format", "json"]
    )
    assert result.exit_code != 0
    payload = json.loads(result.stdout)
    assert payload["status"] == "validation_error"
    assert payload["error_code"] == "invalid_tau_bounds"
    assert payload["graph_params"] is None
    assert payload["best_tau"] is None


def test_failure_output_on_invalid_matrix_is_json(runner: CliRunner, tmp_path: Path) -> None:
    path = tmp_path / "nan.npy"
    arr = np.zeros((10, 4))
    arr[0, 0] = np.nan
    np.save(path, arr)
    result = runner.invoke(cli.main, ["tune", str(path), "--format", "json"])
    assert result.exit_code == 3
    payload = json.loads(result.stdout)
    assert payload["status"] == "validation_error"


def test_non_finite_values_become_null() -> None:
    from arrowspace_tuner.models import TuneResult, tune_result_to_dict

    payload = tune_result_to_dict(TuneResult(status="ok", best_fiedler=float("nan")))
    dumped = json.dumps(payload)
    assert json.loads(dumped)["best_fiedler"] is None


# ── hardening: one-JSON-object contract on failure paths ──────────────────────


def _single_json_document(text: str) -> dict:
    decoder = json.JSONDecoder()
    payload, end = decoder.raw_decode(text.strip())
    assert text.strip()[end:].strip() == ""
    return payload


def test_output_error_is_exactly_one_json_document(
    runner: CliRunner, npy_file: Path, tmp_path: Path
) -> None:
    target = tmp_path / "is_a_directory"
    target.mkdir()
    result = runner.invoke(
        cli.main,
        ["tune", str(npy_file), *CLI_BOUNDS, "--format", "json", "--output", str(target)],
    )
    assert result.exit_code == 5
    payload = _single_json_document(result.stdout)
    assert payload["status"] == "output_error"
    assert payload["schema_version"] == "1.0"


def test_report_error_is_exactly_one_json_document(
    runner: CliRunner,
    monkeypatch: pytest.MonkeyPatch,
    npy_file: Path,
    tmp_path: Path,
) -> None:
    import arrowspace_tuner.service as service_module

    class _ReportFailingTuner:
        def __init__(self, **kwargs: object) -> None:
            pass

        def fit(self, embeddings: object) -> dict[str, object]:
            return {"eps": 1.0, "k": 3, "topk": 1, "p": 2.0, "sigma": None}

        def save_report(self, out_dir: object) -> object:
            raise OSError("permission denied")

    monkeypatch.setattr(service_module, "EpsTuner", _ReportFailingTuner)
    result = runner.invoke(
        cli.main,
        [
            "tune",
            str(npy_file),
            *CLI_BOUNDS,
            "--save-report",
            "--report-dir",
            str(tmp_path / "reports"),
            "--format",
            "json",
        ],
    )
    assert result.exit_code == 5
    payload = _single_json_document(result.stdout)
    assert payload["status"] == "output_error"
    assert payload["error_code"] == "report_write_failed"


def test_json_failure_does_not_emit_second_json_object_to_stderr(
    runner: CliRunner, npy_file: Path
) -> None:
    result = runner.invoke(
        cli.main,
        ["tune", str(npy_file), "--tau-high", "2.0", "--format", "json"],
    )
    assert result.exit_code == 3
    # stdout: exactly one JSON document
    _single_json_document(result.stdout)
    # stderr: no schema-bearing JSON envelope (logs/diagnostics only)
    assert '"schema_version"' not in result.stderr


def test_json_failure_modes_keep_stderr_free_of_json(
    runner: CliRunner,
    monkeypatch: pytest.MonkeyPatch,
    npy_file: Path,
    tmp_path: Path,
) -> None:
    class _ReportFailingTuner:
        def __init__(self, **kwargs: object) -> None:
            pass

        def fit(self, embeddings: object) -> dict[str, object]:
            return {"eps": 1.0, "k": 3, "topk": 1, "p": 2.0, "sigma": None}

        def save_report(self, out_dir: object) -> object:
            raise OSError("denied")

    monkeypatch.setattr(service_module, "EpsTuner", _ReportFailingTuner)
    result = runner.invoke(
        cli.main,
        [
            "tune",
            str(npy_file),
            *CLI_BOUNDS,
            "--save-report",
            "--report-dir",
            str(tmp_path / "reports"),
            "--format",
            "json",
        ],
    )
    assert result.exit_code == 5
    _single_json_document(result.stdout)
    assert '"schema_version"' not in result.stderr


def test_json_output_write_failure_has_status_output_error(
    runner: CliRunner, npy_file: Path, tmp_path: Path
) -> None:
    target = tmp_path / "out_dir"
    target.mkdir()
    result = runner.invoke(
        cli.main,
        ["tune", str(npy_file), *CLI_BOUNDS, "--format", "json", "--output", str(target)],
    )
    payload = _single_json_document(result.stdout)
    assert payload["status"] == "output_error"
    assert payload["error_code"] == "output_write_failed"
    assert payload["graph_params"] is None
    assert payload["best_tau"] is None


def test_json_output_write_failure_exits_5(
    runner: CliRunner, npy_file: Path, tmp_path: Path
) -> None:
    target = tmp_path / "out_dir"
    target.mkdir()
    result = runner.invoke(
        cli.main,
        ["tune", str(npy_file), *CLI_BOUNDS, "--format", "json", "--output", str(target)],
    )
    assert result.exit_code == 5
