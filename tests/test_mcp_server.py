"""
test_mcp_server.py — MCP tool contract (spec §12.5).

Tool handlers are exercised directly (they are plain functions over the
shared service); the server-structure tests require the optional mcp
package and are skipped when it is not installed.
"""

from __future__ import annotations

import asyncio
from pathlib import Path

import pytest

from arrowspace_tuner import mcp_server
from arrowspace_tuner.mcp_server import (
    McpConfigurationError,
    McpContext,
    McpLimits,
    _tool_build_instruction,
    _tool_get_tuner_info,
    _tool_inspect_embeddings,
    _tool_tune_graph,
    build_server,
    load_allowed_roots,
)
from arrowspace_tuner.models import TuneRequest
from arrowspace_tuner.service import run_tuning

pytest.importorskip("mcp", reason="MCP tests need the [mcp] extra")

EXPECTED_TOOLS = {
    "inspect_embeddings",
    "tune_graph",
    "build_instruction",
    "get_tuner_info",
}


@pytest.fixture
def ctx(npy_file: Path) -> McpContext:
    return McpContext(allowed_roots=(str(npy_file.parent),), limits=McpLimits())


# ── server lifecycle ──────────────────────────────────────────────────────────


def test_server_initialises_when_allowed_roots_configured(ctx: McpContext) -> None:
    server = build_server(ctx)
    tools = asyncio.run(server.list_tools())
    assert {t.name for t in tools} == EXPECTED_TOOLS


def test_tools_list_contains_exactly_four_tools(ctx: McpContext) -> None:
    tools = asyncio.run(build_server(ctx).list_tools())
    assert len(tools) == 4


def test_server_refuses_startup_without_allowed_roots() -> None:
    with pytest.raises(McpConfigurationError, match="ARROWSPACE_TUNER_ALLOWED_ROOTS"):
        load_allowed_roots({})


def test_get_tuner_info_schema_is_stable() -> None:
    info = _tool_get_tuner_info()
    assert set(info) == {
        "schema_version",
        "arrowspace_tuner_version",
        "arrowspace_requirement",
        "supported_input_formats",
        "graph_build_keys",
        "default_n_trials",
        "default_n_probe",
        "security",
    }
    assert info["supported_input_formats"] == ["npy", "npz"]
    assert info["graph_build_keys"] == ["eps", "k", "topk", "p", "sigma"]
    assert "search_time_key" not in info
    assert info["security"]["transport"] == "stdio"
    assert info["security"]["remote_urls_supported"] is False


# ── tool handlers ─────────────────────────────────────────────────────────────


def test_inspect_embeddings_returns_embedding_info(ctx: McpContext, npy_file: Path) -> None:
    payload = _tool_inspect_embeddings(ctx, str(npy_file))
    assert payload["format"] == "npy"
    assert payload["n_items"] == 120
    assert payload["sha256"] is not None
    assert "error_code" not in payload


def test_tune_graph_returns_tune_result(ctx: McpContext, npy_file: Path) -> None:
    payload = _tool_tune_graph(
        ctx,
        str(npy_file),
        n_trials=3,
        k_low=3,
        k_high=10,
        eps_low=0.5,
        eps_high=2.0,
        n_probe=20,
    )
    assert payload["status"] == "ok"
    assert set(payload["graph_params"]) == {"eps", "k", "topk", "p", "sigma"}
    assert "tau" not in payload["graph_params"]
    assert "best_tau" not in payload


def test_refinement_cannot_bypass_mcp_trial_cap(ctx: McpContext) -> None:
    payload = _tool_tune_graph(
        ctx,
        "/must-not-be-read.npy",
        n_trials=ctx.limits.max_trials,
        selection_policy="lowest_k",
        refinement_trials=1,
    )
    assert payload["error_code"] == "invalid_trials"


def test_build_instruction_separates_graph_params_from_search_tau() -> None:
    payload = _tool_build_instruction({"eps": 1.6, "k": 38, "topk": 19, "p": 2.0, "sigma": None})
    assert "search_tau" not in payload
    assert set(payload["graph_params"]) == {"eps", "k", "topk", "p", "sigma"}
    assert "ArrowSpaceBuilder().build(graph_params, embeddings)" in payload["python_example"]
    assert "tau" not in payload["python_example"]
    assert any("build-time" in note for note in payload["notes"])


def test_build_instruction_rejects_tau_outside_unit_interval() -> None:
    with pytest.raises(TypeError):
        _tool_build_instruction({"eps": 1.0, "k": 10, "topk": 5, "p": 2.0, "sigma": None}, 1.5)


def test_build_instruction_rejects_missing_keys() -> None:
    payload = _tool_build_instruction({"eps": 1.0, "k": 10})
    assert payload["error_code"] == "invalid_graph_params"


def test_same_seed_and_file_match_service_result(
    ctx: McpContext,
    npy_file: Path,
    fast_tune_request: TuneRequest,
    monkeypatch: pytest.MonkeyPatch,
) -> None:
    import optuna

    import arrowspace_tuner.tuner as tuner_module

    def factory(embeddings: object, cfg: object) -> tuple[object, dict[str, object]]:
        def objective(trial: optuna.Trial) -> float:
            trial.suggest_int("k", 3, 10)
            trial.suggest_float("eps", 0.5, 2.0, log=True)
            trial.set_user_attr("evaluation_success", True)
            return 1.0

        return objective, {}

    # Native eigensolver results can vary slightly. This test checks adapter
    # parity for identical evaluated results, not native floating-point repeatability.
    monkeypatch.setattr(tuner_module, "make_objective", factory)
    expected = run_tuning(fast_tune_request)
    got = _tool_tune_graph(
        ctx,
        str(npy_file),
        n_trials=3,
        seed=42,
        k_low=3,
        k_high=10,
        eps_low=0.5,
        eps_high=2.0,
        n_probe=20,
    )
    assert got["graph_params"] == expected.graph_params
    assert got["best_score"] == expected.best_score


def test_mcp_validation_failure_is_structured(ctx: McpContext, npz_multi_file: Path) -> None:
    payload = _tool_tune_graph(ctx, str(npz_multi_file))
    assert payload["status"] == "validation_error"
    assert payload["error_code"] == "ambiguous_npz_array"
    assert payload["error_message"]
    assert "Traceback" not in payload["error_message"]


def test_mcp_internal_failure_does_not_leak_traceback(
    monkeypatch: pytest.MonkeyPatch, ctx: McpContext, npy_file: Path
) -> None:
    def _boom(request: object) -> object:
        raise RuntimeError("secret traceback /local/secret/path.npy leaked")

    monkeypatch.setattr(mcp_server, "run_tuning", _boom)
    payload = _tool_tune_graph(ctx, str(npy_file))
    assert payload["error_code"] == "internal_error"
    assert "secret" not in str(payload["error_message"])
    assert "Traceback" not in str(payload)


# ── hardening: report persistence and build_instruction values ────────────────


def test_tune_graph_missing_report_dir_returns_validation_error(
    ctx: McpContext, npy_file: Path
) -> None:
    payload = _tool_tune_graph(ctx, str(npy_file), save_report=True, report_dir=None)
    assert payload["status"] == "validation_error"
    assert payload["error_code"] == "missing_report_dir"


def test_tune_graph_report_write_failure_returns_output_error(
    monkeypatch: pytest.MonkeyPatch, ctx: McpContext, npy_file: Path
) -> None:
    from arrowspace_tuner.models import TuneResult

    def _output_error(request: object) -> object:
        return TuneResult(
            status="output_error",
            error_code="report_write_failed",
            error_message="Could not save requested report: denied",
        )

    monkeypatch.setattr(mcp_server, "run_tuning", _output_error)
    payload = _tool_tune_graph(ctx, str(npy_file))
    assert payload["status"] == "output_error"
    assert payload["error_code"] == "report_write_failed"
    assert payload["graph_params"] is None
    assert "best_tau" not in payload


VALID_PARAMS = {"eps": 1.6, "k": 38, "topk": 19, "p": 2.0, "sigma": None}


def test_build_instruction_rejects_negative_eps() -> None:
    payload = _tool_build_instruction({**VALID_PARAMS, "eps": -1})
    assert payload["error_code"] == "invalid_graph_params"
    assert "eps" in payload["error_message"]


def test_build_instruction_rejects_bool_k() -> None:
    payload = _tool_build_instruction({**VALID_PARAMS, "k": True})
    assert payload["error_code"] == "invalid_graph_params"
    assert "k" in payload["error_message"]


def test_build_instruction_rejects_topk_larger_than_k() -> None:
    payload = _tool_build_instruction({**VALID_PARAMS, "topk": 50})
    assert payload["error_code"] == "invalid_graph_params"
    assert "topk" in payload["error_message"]


def test_build_instruction_rejects_zero_p() -> None:
    payload = _tool_build_instruction({**VALID_PARAMS, "p": 0})
    assert payload["error_code"] == "invalid_graph_params"
    assert "p" in payload["error_message"]


def test_build_instruction_rejects_non_finite_sigma() -> None:
    payload = _tool_build_instruction({**VALID_PARAMS, "sigma": float("nan")})
    assert payload["error_code"] == "invalid_graph_params"
    assert "sigma" in payload["error_message"]


def test_build_instruction_accepts_sigma_none() -> None:
    payload = _tool_build_instruction(dict(VALID_PARAMS))
    assert "error_code" not in payload
    assert payload["graph_params"]["sigma"] is None
    assert "search_tau" not in payload


def test_build_instruction_rejects_string_eps() -> None:
    payload = _tool_build_instruction({**VALID_PARAMS, "eps": "1.5"})
    assert payload["error_code"] == "invalid_graph_params"


@pytest.mark.parametrize("value", [True, float("nan"), "0.5"])
def test_build_instruction_rejects_obsolete_tau(value: object) -> None:
    with pytest.raises(TypeError):
        _tool_build_instruction(dict(VALID_PARAMS), value)
