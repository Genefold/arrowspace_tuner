"""
test_cli.py — Click CLI behaviour and exit codes (spec §12.3).

The only allowed subprocess use for CLI testing is the installed executable
itself (covered by the wheel smoke test); these tests use Click's CliRunner
in-process.
"""

from __future__ import annotations

import json
from pathlib import Path

import numpy as np
import pytest
from click.testing import CliRunner

from arrowspace_tuner import cli
from arrowspace_tuner.models import SCHEMA_VERSION


@pytest.fixture
def runner() -> CliRunner:
    return CliRunner()


@pytest.fixture
def tuned_result(runner: CliRunner, npy_file: Path, tmp_path: Path) -> Path:
    """One real tune run shared by output-persistence tests."""
    out = tmp_path / "result.json"
    res = runner.invoke(
        cli.main,
        [
            "tune",
            str(npy_file),
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
            "--format",
            "json",
            "--output",
            str(out),
        ],
    )
    assert res.exit_code == 0, res.output
    return out


# ── help surfaces ─────────────────────────────────────────────────────────────


@pytest.mark.parametrize(
    "args",
    [
        ["--help"],
        ["tune", "--help"],
        ["validate", "--help"],
        ["inspect", "--help"],
        ["version", "--help"],
        ["mcp", "--help"],
    ],
)
def test_help_works(runner: CliRunner, args: list[str]) -> None:
    result = runner.invoke(cli.main, args)
    assert result.exit_code == 0
    assert "Usage" in result.output


def test_version_option(runner: CliRunner) -> None:
    result = runner.invoke(cli.main, ["--version"])
    assert result.exit_code == 0


# ── tune ──────────────────────────────────────────────────────────────────────

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


def test_tune_valid_npy_exits_zero(runner: CliRunner, npy_file: Path) -> None:
    result = runner.invoke(cli.main, ["tune", str(npy_file), *CLI_BOUNDS])
    assert result.exit_code == 0
    assert "Graph parameters" in result.output


def test_tune_invalid_file_exits_3(runner: CliRunner, tmp_path: Path) -> None:
    result = runner.invoke(cli.main, ["tune", str(tmp_path / "missing.npy")])
    assert result.exit_code == 3


def test_tune_invalid_tau_exits_3(runner: CliRunner, npy_file: Path) -> None:
    # tau bounds are validated by the service layer → exit 3 (EXIT_INPUT)
    result = runner.invoke(cli.main, ["tune", str(npy_file), "--tau-high", "1.5", *CLI_BOUNDS])
    assert result.exit_code == 3


def test_dry_run_exits_zero_and_skips_tuning(
    runner: CliRunner, monkeypatch: pytest.MonkeyPatch, npy_file: Path
) -> None:
    import arrowspace_tuner.service as service_module

    class _ExplodingTuner:
        def __init__(self, **kwargs: object) -> None:
            raise AssertionError("EpsTuner must not be constructed on --dry-run")

    monkeypatch.setattr(service_module, "EpsTuner", _ExplodingTuner)
    result = runner.invoke(cli.main, ["tune", str(npy_file), "--dry-run"])
    assert result.exit_code == 0
    assert "no tuning performed" in result.output


def test_output_written_atomically(runner: CliRunner, npy_file: Path, tmp_path: Path) -> None:
    out = tmp_path / "result.json"
    result = runner.invoke(
        cli.main,
        ["tune", str(npy_file), *CLI_BOUNDS, "--format", "json", "--output", str(out)],
    )
    assert result.exit_code == 0
    assert out.exists()
    leftovers = [p.name for p in tmp_path.iterdir() if ".tmp-" in p.name]
    assert leftovers == []
    payload = json.loads(out.read_text(encoding="utf-8"))
    assert payload["schema_version"] == SCHEMA_VERSION


def test_output_contains_valid_tune_result_json(tuned_result: Path) -> None:
    payload = json.loads(tuned_result.read_text(encoding="utf-8"))
    assert payload["status"] == "ok"
    assert set(payload["graph_params"]) == {"eps", "k", "topk", "p", "sigma"}
    assert "top_k" not in payload["graph_params"]
    assert "tau" not in payload["graph_params"]
    assert payload["best_tau"] is not None
    assert payload["seed"] == 42
    assert payload["input_sha256"]


def test_keyboard_interrupt_maps_to_exit_6(
    runner: CliRunner, monkeypatch: pytest.MonkeyPatch, npy_file: Path
) -> None:
    def _interrupt(*args: object, **kwargs: object) -> None:
        raise KeyboardInterrupt

    monkeypatch.setattr(cli, "run_tuning", _interrupt)
    result = runner.invoke(cli.main, ["tune", str(npy_file), "--format", "json"])
    assert result.exit_code == 6


def test_unexpected_failure_maps_to_exit_7(
    runner: CliRunner, monkeypatch: pytest.MonkeyPatch, npy_file: Path
) -> None:
    from arrowspace_tuner.io import TuningExecutionError

    def _boom(*args: object, **kwargs: object) -> None:
        raise TuningExecutionError("boom")

    monkeypatch.setattr(cli, "run_tuning", _boom)
    result = runner.invoke(cli.main, ["tune", str(npy_file)])
    assert result.exit_code == 7


# ── hardening: report and output failures ─────────────────────────────────────


def test_save_report_without_report_dir_exits_3(runner: CliRunner, npy_file: Path) -> None:
    result = runner.invoke(
        cli.main,
        ["tune", str(npy_file), *CLI_BOUNDS, "--save-report", "--format", "json"],
    )
    assert result.exit_code == 3
    payload = json.loads(result.stdout)
    assert payload["schema_version"] == "1.0"
    assert payload["status"] == "validation_error"
    assert payload["error_code"] == "missing_report_dir"
    assert payload["graph_params"] is None
    assert payload["best_tau"] is None


def test_report_write_failure_exits_5(
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
    payload = json.loads(result.stdout)
    assert payload["schema_version"] == "1.0"
    assert payload["status"] == "output_error"
    assert payload["error_code"] == "report_write_failed"
    assert payload["graph_params"] is None
    assert payload["best_tau"] is None


def test_output_write_failure_exits_5(runner: CliRunner, npy_file: Path, tmp_path: Path) -> None:
    # --output pointing at an existing directory makes the atomic replace fail
    target = tmp_path / "is_a_directory"
    target.mkdir()
    result = runner.invoke(
        cli.main,
        ["tune", str(npy_file), *CLI_BOUNDS, "--format", "json", "--output", str(target)],
    )
    assert result.exit_code == 5
    payload = json.loads(result.stdout)
    assert payload["schema_version"] == "1.0"
    assert payload["status"] == "output_error"
    assert payload["error_code"] == "output_write_failed"
    assert payload["graph_params"] is None
    assert payload["best_tau"] is None


def test_k_low_exceeds_corpus_exits_3(runner: CliRunner, npy_file: Path) -> None:
    result = runner.invoke(
        cli.main,
        ["tune", str(npy_file), "--k-low", "120", "--k-high", "200", "--format", "json"],
    )
    assert result.exit_code == 3
    payload = json.loads(result.stdout)
    assert payload["schema_version"] == "1.0"
    assert payload["error_code"] == "k_low_exceeds_corpus"
    assert payload["graph_params"] is None
    assert payload["best_tau"] is None


def test_k_high_clip_succeeds_and_warns(runner: CliRunner, npy_file: Path) -> None:
    result = runner.invoke(
        cli.main,
        [
            "tune",
            str(npy_file),
            "--k-low",
            "10",
            "--k-high",
            "200",
            "--eps-low",
            "0.5",
            "--eps-high",
            "2.0",
            "--trials",
            "3",
            "--n-probe",
            "20",
            "--format",
            "json",
        ],
    )
    assert result.exit_code == 0
    payload = json.loads(result.stdout)
    assert payload["status"] == "ok"
    assert payload["graph_params"]["k"] <= 119
    assert any("k_high exceeds n_items - 1" in warning for warning in payload["warnings"])


# ── validate ──────────────────────────────────────────────────────────────────


def test_validate_ok(runner: CliRunner, npy_file: Path) -> None:
    result = runner.invoke(cli.main, ["validate", str(npy_file), "--format", "json"])
    assert result.exit_code == 0
    payload = json.loads(result.output)
    assert payload["n_items"] == 120


def test_validate_invalid_input_exits_3(runner: CliRunner, tmp_path: Path) -> None:
    path = tmp_path / "bad.npy"
    np.save(path, np.array([1.0, 2.0, 3.0]))  # 1D
    result = runner.invoke(cli.main, ["validate", str(path)])
    assert result.exit_code == 3


def test_validate_multi_array_npz_without_key_exits_3(
    runner: CliRunner, npz_multi_file: Path
) -> None:
    result = runner.invoke(cli.main, ["validate", str(npz_multi_file)])
    assert result.exit_code == 3


def test_validate_multi_array_npz_with_key_exits_0(runner: CliRunner, npz_multi_file: Path) -> None:
    result = runner.invoke(cli.main, ["validate", str(npz_multi_file), "--array-key", "small"])
    assert result.exit_code == 0


def test_multi_array_npz_requires_array_key(runner: CliRunner, npz_multi_file: Path) -> None:
    result = runner.invoke(cli.main, ["tune", str(npz_multi_file), *CLI_BOUNDS])
    assert result.exit_code == 3


# ── inspect ───────────────────────────────────────────────────────────────────


def test_inspect_reads_previous_result(runner: CliRunner, tuned_result: Path) -> None:
    result = runner.invoke(cli.main, ["inspect", str(tuned_result)])
    assert result.exit_code == 0
    assert "graph_params" in result.output or "eps" in result.output


def test_inspect_json_mode_echoes_document(runner: CliRunner, tuned_result: Path) -> None:
    result = runner.invoke(cli.main, ["inspect", str(tuned_result), "--format", "json"])
    assert result.exit_code == 0
    assert json.loads(result.output)["schema_version"] == SCHEMA_VERSION


def test_inspect_rejects_non_json(runner: CliRunner, tmp_path: Path) -> None:
    path = tmp_path / "not_json.json"
    path.write_text("definitely not json")
    result = runner.invoke(cli.main, ["inspect", str(path)])
    assert result.exit_code == 3


def test_inspect_rejects_unknown_schema_version(runner: CliRunner, tmp_path: Path) -> None:
    path = tmp_path / "future.json"
    path.write_text(json.dumps({"schema_version": "9.9"}))
    result = runner.invoke(cli.main, ["inspect", str(path)])
    assert result.exit_code == 3
    assert "schema" in result.output


# ── version ───────────────────────────────────────────────────────────────────


def test_version_json(runner: CliRunner) -> None:
    result = runner.invoke(cli.main, ["version", "--format", "json"])
    assert result.exit_code == 0
    payload = json.loads(result.output)
    assert payload["graph_build_keys"] == ["eps", "k", "topk", "p", "sigma"]
    assert payload["search_time_result"] == "best_tau"


# ── text output sample ────────────────────────────────────────────────────────


def test_tune_text_output_mentions_best_tau(runner: CliRunner, npy_file: Path) -> None:
    result = runner.invoke(cli.main, ["tune", str(npy_file), *CLI_BOUNDS])
    assert result.exit_code == 0
    assert "Search-time parameter" in result.output
    assert "best_tau" in result.output
