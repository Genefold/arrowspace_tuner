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
        "search_time_key",
        "tau_range",
        "default_n_trials",
        "default_n_probe",
        "security",
    }
    assert info["supported_input_formats"] == ["npy", "npz"]
    assert info["graph_build_keys"] == ["eps", "k", "topk", "p", "sigma"]
    assert info["search_time_key"] == "best_tau"
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
    assert 0.0 <= payload["best_tau"] <= 1.0


def test_build_instruction_separates_graph_params_from_search_tau() -> None:
    payload = _tool_build_instruction(
        {"eps": 1.6, "k": 38, "topk": 19, "p": 2.0, "sigma": None}, 0.8
    )
    assert payload["search_tau"] == 0.8
    assert set(payload["graph_params"]) == {"eps", "k", "topk", "p", "sigma"}
    assert "ArrowSpaceBuilder().build(graph_params, embeddings)" in payload["python_example"]
    assert "tau=search_tau" in payload["python_example"]
    assert any("must not be passed" in note for note in payload["notes"])


def test_build_instruction_rejects_tau_outside_unit_interval() -> None:
    payload = _tool_build_instruction(
        {"eps": 1.0, "k": 10, "topk": 5, "p": 2.0, "sigma": None}, 1.5
    )
    assert payload["error_code"] == "invalid_tau"


def test_build_instruction_rejects_missing_keys() -> None:
    payload = _tool_build_instruction({"eps": 1.0, "k": 10}, 0.5)
    assert payload["error_code"] == "invalid_graph_params"


def test_same_seed_and_file_match_service_result(
    ctx: McpContext, npy_file: Path, fast_tune_request: TuneRequest
) -> None:
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
    assert got["best_tau"] == expected.best_tau


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
    assert payload["best_tau"] is None


VALID_PARAMS = {"eps": 1.6, "k": 38, "topk": 19, "p": 2.0, "sigma": None}


def test_build_instruction_rejects_negative_eps() -> None:
    payload = _tool_build_instruction({**VALID_PARAMS, "eps": -1}, 0.8)
    assert payload["error_code"] == "invalid_graph_params"
    assert "eps" in payload["error_message"]


def test_build_instruction_rejects_bool_k() -> None:
    payload = _tool_build_instruction({**VALID_PARAMS, "k": True}, 0.8)
    assert payload["error_code"] == "invalid_graph_params"
    assert "k" in payload["error_message"]


def test_build_instruction_rejects_topk_larger_than_k() -> None:
    payload = _tool_build_instruction({**VALID_PARAMS, "topk": 50}, 0.8)
    assert payload["error_code"] == "invalid_graph_params"
    assert "topk" in payload["error_message"]


def test_build_instruction_rejects_zero_p() -> None:
    payload = _tool_build_instruction({**VALID_PARAMS, "p": 0}, 0.8)
    assert payload["error_code"] == "invalid_graph_params"
    assert "p" in payload["error_message"]


def test_build_instruction_rejects_non_finite_sigma() -> None:
    payload = _tool_build_instruction({**VALID_PARAMS, "sigma": float("nan")}, 0.8)
    assert payload["error_code"] == "invalid_graph_params"
    assert "sigma" in payload["error_message"]


def test_build_instruction_accepts_sigma_none() -> None:
    payload = _tool_build_instruction(dict(VALID_PARAMS), 0.8)
    assert "error_code" not in payload
    assert payload["graph_params"]["sigma"] is None
    assert payload["search_tau"] == 0.8


def test_build_instruction_rejects_string_eps() -> None:
    payload = _tool_build_instruction({**VALID_PARAMS, "eps": "1.5"}, 0.8)
    assert payload["error_code"] == "invalid_graph_params"


def test_build_instruction_rejects_bool_and_nan_tau() -> None:
    assert _tool_build_instruction(dict(VALID_PARAMS), True)["error_code"] == "invalid_tau"
    assert _tool_build_instruction(dict(VALID_PARAMS), float("nan"))["error_code"] == (
        "invalid_tau"
    )
    assert _tool_build_instruction(dict(VALID_PARAMS), "0.5")["error_code"] == ("invalid_tau")
