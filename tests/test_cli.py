"""
test_cli.py — Click CLI behaviour and exit codes (spec §12.3).

The only allowed subprocess use for CLI testing is the installed executable
itself (covered by the wheel smoke test); these tests use Click's CliRunner
in-process.
"""
from __future__ import annotations

import json

import numpy as np
from click.testing import CliRunner

from arrowspace_tuner import cli
from arrowspace_tuner.models import SCHEMA_VERSION

import pytest


@pytest.fixture
def runner() -> CliRunner:
    return CliRunner()


@pytest.fixture
def tuned_result(runner, npy_file, tmp_path):
    """One real tune run shared by output-persistence tests."""
    out = tmp_path / "result.json"
    res = runner.invoke(
        cli.main,
        ["tune", str(npy_file), "--trials", "3", "--seed", "42",
         "--k-low", "3", "--k-high", "10", "--eps-low", "0.5",
         "--eps-high", "2.0", "--n-probe", "20",
         "--format", "json", "--output", str(out)],
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
def test_help_works(runner, args) -> None:
    result = runner.invoke(cli.main, args)
    assert result.exit_code == 0
    assert "Usage" in result.output


def test_version_option(runner) -> None:
    result = runner.invoke(cli.main, ["--version"])
    assert result.exit_code == 0


# ── tune ──────────────────────────────────────────────────────────────────────

CLI_BOUNDS = [
    "--trials", "3", "--seed", "42",
    "--k-low", "3", "--k-high", "10",
    "--eps-low", "0.5", "--eps-high", "2.0", "--n-probe", "20",
]


def test_tune_valid_npy_exits_zero(runner, npy_file) -> None:
    result = runner.invoke(cli.main, ["tune", str(npy_file), *CLI_BOUNDS])
    assert result.exit_code == 0
    assert "Graph parameters" in result.output


def test_tune_invalid_file_exits_3(runner, tmp_path) -> None:
    result = runner.invoke(cli.main, ["tune", str(tmp_path / "missing.npy")])
    assert result.exit_code == 3


def test_tune_invalid_tau_exits_3(runner, npy_file) -> None:
    # tau bounds are validated by the service layer → exit 3 (EXIT_INPUT)
    result = runner.invoke(
        cli.main, ["tune", str(npy_file), "--tau-high", "1.5", *CLI_BOUNDS]
    )
    assert result.exit_code == 3


def test_dry_run_exits_zero_and_skips_tuning(runner, monkeypatch, npy_file) -> None:
    import arrowspace_tuner.service as service_module

    class _ExplodingTuner:
        def __init__(self, **kwargs: object) -> None:
            raise AssertionError("EpsTuner must not be constructed on --dry-run")

    monkeypatch.setattr(service_module, "EpsTuner", _ExplodingTuner)
    result = runner.invoke(cli.main, ["tune", str(npy_file), "--dry-run"])
    assert result.exit_code == 0
    assert "no tuning performed" in result.output


def test_output_written_atomically(runner, npy_file, tmp_path) -> None:
    out = tmp_path / "result.json"
    result = runner.invoke(
        cli.main,
        ["tune", str(npy_file), *CLI_BOUNDS, "--format", "json",
         "--output", str(out)],
    )
    assert result.exit_code == 0
    assert out.exists()
    leftovers = [p.name for p in tmp_path.iterdir() if ".tmp-" in p.name]
    assert leftovers == []
    payload = json.loads(out.read_text(encoding="utf-8"))
    assert payload["schema_version"] == SCHEMA_VERSION


def test_output_contains_valid_tune_result_json(tuned_result) -> None:
    payload = json.loads(tuned_result.read_text(encoding="utf-8"))
    assert payload["status"] == "ok"
    assert set(payload["graph_params"]) == {"eps", "k", "topk", "p", "sigma"}
    assert "top_k" not in payload["graph_params"]
    assert "tau" not in payload["graph_params"]
    assert payload["best_tau"] is not None
    assert payload["seed"] == 42
    assert payload["input_sha256"]


def test_keyboard_interrupt_maps_to_exit_6(runner, monkeypatch, npy_file) -> None:
    def _interrupt(*args: object, **kwargs: object) -> None:
        raise KeyboardInterrupt

    monkeypatch.setattr(cli, "run_tuning", _interrupt)
    result = runner.invoke(cli.main, ["tune", str(npy_file), "--format", "json"])
    assert result.exit_code == 6


def test_unexpected_failure_maps_to_exit_7(runner, monkeypatch, npy_file) -> None:
    from arrowspace_tuner.io import TuningExecutionError

    def _boom(*args: object, **kwargs: object) -> None:
        raise TuningExecutionError("boom")

    monkeypatch.setattr(cli, "run_tuning", _boom)
    result = runner.invoke(cli.main, ["tune", str(npy_file)])
    assert result.exit_code == 7


# ── validate ──────────────────────────────────────────────────────────────────

def test_validate_ok(runner, npy_file) -> None:
    result = runner.invoke(cli.main, ["validate", str(npy_file), "--format", "json"])
    assert result.exit_code == 0
    payload = json.loads(result.output)
    assert payload["n_items"] == 120


def test_validate_invalid_input_exits_3(runner, tmp_path) -> None:
    path = tmp_path / "bad.npy"
    np.save(path, np.array([1.0, 2.0, 3.0]))  # 1D
    result = runner.invoke(cli.main, ["validate", str(path)])
    assert result.exit_code == 3


def test_validate_multi_array_npz_without_key_exits_3(runner, npz_multi_file) -> None:
    result = runner.invoke(cli.main, ["validate", str(npz_multi_file)])
    assert result.exit_code == 3


def test_validate_multi_array_npz_with_key_exits_0(runner, npz_multi_file) -> None:
    result = runner.invoke(
        cli.main, ["validate", str(npz_multi_file), "--array-key", "small"]
    )
    assert result.exit_code == 0


def test_multi_array_npz_requires_array_key(runner, npz_multi_file) -> None:
    result = runner.invoke(cli.main, ["tune", str(npz_multi_file), *CLI_BOUNDS])
    assert result.exit_code == 3


# ── inspect ───────────────────────────────────────────────────────────────────

def test_inspect_reads_previous_result(runner, tuned_result) -> None:
    result = runner.invoke(cli.main, ["inspect", str(tuned_result)])
    assert result.exit_code == 0
    assert "graph_params" in result.output or "eps" in result.output


def test_inspect_json_mode_echoes_document(runner, tuned_result) -> None:
    result = runner.invoke(cli.main, ["inspect", str(tuned_result), "--format", "json"])
    assert result.exit_code == 0
    assert json.loads(result.output)["schema_version"] == SCHEMA_VERSION


def test_inspect_rejects_non_json(runner, tmp_path) -> None:
    path = tmp_path / "not_json.json"
    path.write_text("definitely not json")
    result = runner.invoke(cli.main, ["inspect", str(path)])
    assert result.exit_code == 3


def test_inspect_rejects_unknown_schema_version(runner, tmp_path) -> None:
    path = tmp_path / "future.json"
    path.write_text(json.dumps({"schema_version": "9.9"}))
    result = runner.invoke(cli.main, ["inspect", str(path)])
    assert result.exit_code == 3
    assert "schema" in result.output


# ── version ───────────────────────────────────────────────────────────────────

def test_version_json(runner) -> None:
    result = runner.invoke(cli.main, ["version", "--format", "json"])
    assert result.exit_code == 0
    payload = json.loads(result.output)
    assert payload["graph_build_keys"] == ["eps", "k", "topk", "p", "sigma"]
    assert payload["search_time_result"] == "best_tau"


# ── text output sample ────────────────────────────────────────────────────────

def test_tune_text_output_mentions_best_tau(runner, npy_file) -> None:
    result = runner.invoke(cli.main, ["tune", str(npy_file), *CLI_BOUNDS])
    assert result.exit_code == 0
    assert "Search-time parameter" in result.output
    assert "best_tau" in result.output