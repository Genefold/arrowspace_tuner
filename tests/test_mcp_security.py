"""
test_mcp_security.py — MCP security model (spec §12.6).

Allowed roots, path policy, resource limits, and the no-network guarantee.
"""

from __future__ import annotations

import ast
from pathlib import Path

import numpy as np
import pytest

from arrowspace_tuner.mcp_server import (
    McpContext,
    McpLimits,
    _tool_inspect_embeddings,
    _tool_tune_graph,
    ensure_allowed_path,
    ensure_allowed_report_dir,
    load_limits,
)

EXPECTED_FORBIDDEN_MODULES = {
    "requests",
    "httpx",
    "aiohttp",
    "urllib",
    "urllib3",
    "http",
    "socket",
    "ftplib",
    "smtplib",
}


@pytest.fixture
def ctx(npy_file: Path) -> McpContext:
    return McpContext(allowed_roots=(str(npy_file.parent),), limits=McpLimits())


def _reject(payload: dict, code: str) -> None:
    assert payload["status"] == "validation_error"
    assert payload["error_code"] == code
    assert "Traceback" not in payload["error_message"]


# ── path policy ───────────────────────────────────────────────────────────────


def test_relative_input_path_rejected(ctx: McpContext) -> None:
    with pytest.raises(Exception, match="absolute"):
        ensure_allowed_path("embeddings.npy", allowed_roots=ctx.allowed_roots, must_exist=True)


def test_path_outside_allowed_root_rejected(ctx: McpContext, tmp_path: Path) -> None:
    outside = tmp_path.parent / "outside.npy"
    np.save(outside, np.ones((10, 4)))
    try:
        with pytest.raises(Exception, match="outside"):
            ensure_allowed_path(str(outside), allowed_roots=ctx.allowed_roots, must_exist=True)
    finally:
        outside.unlink(missing_ok=True)


def test_symlink_escaping_allowed_root_rejected(
    ctx: McpContext, npy_file: Path, tmp_path: Path
) -> None:
    link = npy_file.parent / "escape.npy"
    link.symlink_to(tmp_path / "elsewhere.npy")
    try:
        with pytest.raises(Exception, match="exist|outside"):
            ensure_allowed_path(str(link), allowed_roots=ctx.allowed_roots, must_exist=True)
    finally:
        link.unlink(missing_ok=True)


def test_nonexistent_input_rejected(ctx: McpContext, npy_file: Path) -> None:
    with pytest.raises(Exception, match="not accessible"):
        ensure_allowed_path(
            str(npy_file.parent / "missing.npy"),
            allowed_roots=ctx.allowed_roots,
            must_exist=True,
        )


def test_url_like_input_rejected(ctx: McpContext) -> None:
    for url in (
        "https://example.com/embeddings.npy",
        "http://example.com/embeddings.npy",
        "file:///workspace/embeddings.npy",
    ):
        with pytest.raises(Exception, match="URL"):
            ensure_allowed_path(url, allowed_roots=ctx.allowed_roots, must_exist=True)


def test_pickle_file_rejected(ctx: McpContext, tmp_path: Path) -> None:
    pkl = Path(ctx.allowed_roots[0]) / "model.pkl"
    pkl.write_bytes(b"\x80\x04\x95")
    try:
        with pytest.raises(Exception, match="Unsupported file type"):
            ensure_allowed_path(str(pkl), allowed_roots=ctx.allowed_roots, must_exist=True)
    finally:
        pkl.unlink(missing_ok=True)


def test_object_array_npy_rejected(ctx: McpContext) -> None:
    root = Path(ctx.allowed_roots[0])
    path = root / "object.npy"
    np.save(path, np.array([[object()], [object()]], dtype=object), allow_pickle=True)
    try:
        payload = _tool_inspect_embeddings(ctx, str(path))
        _reject(payload, "object_array_rejected")
    finally:
        path.unlink(missing_ok=True)


def test_report_path_outside_allowed_root_rejected(ctx: McpContext, tmp_path: Path) -> None:
    outside = tmp_path.parent / "outside_reports_mcp"
    with pytest.raises(Exception, match="outside"):
        ensure_allowed_report_dir(str(outside), allowed_roots=ctx.allowed_roots)


def test_report_path_relative_rejected(ctx: McpContext) -> None:
    with pytest.raises(Exception, match="absolute"):
        ensure_allowed_report_dir("reports", allowed_roots=ctx.allowed_roots)


# ── resource limits ───────────────────────────────────────────────────────────


def test_max_file_size_enforced(ctx: McpContext, npy_file: Path) -> None:
    tight = McpContext(
        allowed_roots=ctx.allowed_roots,
        limits=McpLimits(max_input_bytes=10, max_trials=100, max_n_jobs=4),
    )
    payload = _tool_tune_graph(tight, str(npy_file))
    _reject(payload, "input_too_large")


def test_max_trials_enforced(ctx: McpContext, npy_file: Path) -> None:
    tight = McpContext(
        allowed_roots=ctx.allowed_roots,
        limits=McpLimits(max_input_bytes=2**31, max_trials=5, max_n_jobs=4),
    )
    payload = _tool_tune_graph(tight, str(npy_file), n_trials=101)
    _reject(payload, "invalid_trials")


def test_max_n_jobs_enforced(ctx: McpContext, npy_file: Path) -> None:
    tight = McpContext(
        allowed_roots=ctx.allowed_roots,
        limits=McpLimits(max_input_bytes=2**31, max_trials=100, max_n_jobs=4),
    )
    payload = _tool_tune_graph(tight, str(npy_file), n_jobs=5)
    _reject(payload, "invalid_n_jobs")


def test_limits_parse_from_environment() -> None:
    limits = load_limits(
        {
            "ARROWSPACE_TUNER_MAX_INPUT_BYTES": "1024",
            "ARROWSPACE_TUNER_MAX_TRIALS": "7",
            "ARROWSPACE_TUNER_MAX_N_JOBS": "2",
        }
    )
    assert limits == McpLimits(max_input_bytes=1024, max_trials=7, max_n_jobs=2)


def test_limits_defaults() -> None:
    assert load_limits({}) == McpLimits(max_input_bytes=2 * 1024**3, max_trials=100, max_n_jobs=4)


def test_limits_reject_non_integer() -> None:
    from arrowspace_tuner.mcp_server import McpConfigurationError

    with pytest.raises(McpConfigurationError, match="positive integer"):
        load_limits({"ARROWSPACE_TUNER_MAX_TRIALS": "not-a-number"})


@pytest.mark.parametrize(
    ("key", "value"),
    [
        ("ARROWSPACE_TUNER_MAX_INPUT_BYTES", "0"),
        ("ARROWSPACE_TUNER_MAX_INPUT_BYTES", "-1"),
        ("ARROWSPACE_TUNER_MAX_TRIALS", "0"),
        ("ARROWSPACE_TUNER_MAX_TRIALS", "-10"),
        ("ARROWSPACE_TUNER_MAX_N_JOBS", "0"),
        ("ARROWSPACE_TUNER_MAX_N_JOBS", "-1"),
    ],
)
def test_limits_reject_non_positive_integer(key: str, value: str) -> None:
    from arrowspace_tuner.mcp_server import McpConfigurationError

    with pytest.raises(McpConfigurationError) as excinfo:
        load_limits({key: value})
    assert key in str(excinfo.value)
    assert "positive integer" in str(excinfo.value)


def test_mcp_report_dir_missing_with_save_report_rejected(ctx: McpContext, npy_file: Path) -> None:
    payload = _tool_tune_graph(ctx, str(npy_file), save_report=True, report_dir=None)
    assert payload["status"] == "validation_error"
    assert payload["error_code"] == "missing_report_dir"
    assert "Traceback" not in payload["error_message"]


def test_mcp_k_low_exceeds_corpus_returns_structured_error(ctx: McpContext, npy_file: Path) -> None:
    payload = _tool_tune_graph(ctx, str(npy_file), n_trials=3, k_low=120, k_high=200)
    assert payload["status"] == "validation_error"
    assert payload["error_code"] == "k_low_exceeds_corpus"
    assert "Traceback" not in payload["error_message"]
    assert payload["graph_params"] is None


# ── no-network guarantee ──────────────────────────────────────────────────────


def test_no_network_client_imported_by_mcp_server() -> None:
    source = Path(
        __import__("arrowspace_tuner.mcp_server", fromlist=["__file__"]).__file__
    ).read_text(encoding="utf-8")
    tree = ast.parse(source)
    imported: set[str] = set()
    for node in ast.walk(tree):
        if isinstance(node, ast.Import):
            imported.update(alias.name.split(".")[0] for alias in node.names)
        elif isinstance(node, ast.ImportFrom) and node.module:
            imported.add(node.module.split(".")[0])
    assert imported & EXPECTED_FORBIDDEN_MODULES == set()


def test_report_dir_inside_allowed_root_is_accepted(ctx: McpContext, tmp_path: Path) -> None:
    root = Path(ctx.allowed_roots[0])
    report_dir = root / "reports"
    resolved = ensure_allowed_report_dir(str(report_dir), allowed_roots=ctx.allowed_roots)
    assert resolved == str(report_dir.resolve())
    assert not report_dir.exists()  # creation is deferred to the service


def test_existing_report_dir_is_not_overwritten(ctx: McpContext, tmp_path: Path) -> None:
    root = Path(ctx.allowed_roots[0])
    existing = root / "existing_reports"
    existing.mkdir()
    try:
        with pytest.raises(Exception, match="already exists"):
            ensure_allowed_report_dir(str(existing), allowed_roots=ctx.allowed_roots)
    finally:
        existing.rmdir()
