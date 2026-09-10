"""
models.py — typed, frozen data models shared by the Python service, CLI, and MCP server.

These are the stable contracts for issue #17:

- ``TuneRequest``    — validated tuning request (service input).
- ``EmbeddingInfo``  — validated local embedding-array metadata.
- ``TuneResult``     — stable result schema (schema_version "1.0").

Two naming rules are load-bearing (see CHANGELOG 0.4.1):

- graph parameters use ArrowSpace's native ``topk`` key — never ``top_k``;
- ``best_tau`` is a search-time result and is NEVER part of ``graph_params``.
"""

from __future__ import annotations

import math
from dataclasses import dataclass, field
from pathlib import Path
from typing import Literal

#: Version of the TuneResult JSON schema emitted by ``tune_result_to_dict``.
SCHEMA_VERSION: str = "1.0"

#: The exact set of ArrowSpace build-time graph parameter keys.
GRAPH_BUILD_KEYS: tuple[str, ...] = ("eps", "k", "topk", "p", "sigma")

#: Supported input file formats.
SUPPORTED_FORMATS: tuple[str, ...] = ("npy", "npz")

#: Valid search-time interval for tau.
TAU_LOW: float = 0.0
TAU_HIGH: float = 1.0


@dataclass(frozen=True)
class TuneRequest:
    """Validated graph-parameter tuning request."""

    input_path: Path

    array_key: str | None = None

    n_trials: int = 15
    sample_n: int | None = None
    seed: int = 42

    eps_low: float = 0.5
    eps_high: float = 12.0

    k_low: int = 10
    k_high: int = 45

    tau_low: float = 0.0
    tau_high: float = 1.0

    n_probe: int = 50
    n_jobs: int = 1

    report_dir: Path | None = None
    save_report: bool = False

    output_path: Path | None = None
    output_format: Literal["text", "json"] = "text"

    dry_run: bool = False
    include_input_hash: bool = True


@dataclass(frozen=True)
class EmbeddingInfo:
    """Validated local embedding-array metadata."""

    path: str
    format: Literal["npy", "npz"]
    array_key: str | None

    n_items: int
    n_dimensions: int
    shape: tuple[int, int]
    dtype: str

    finite: bool
    l2_norm_min: float | None
    l2_norm_mean: float | None
    l2_norm_max: float | None

    estimated_memory_bytes: int
    sha256: str | None

    warnings: tuple[str, ...]


@dataclass(frozen=True)
class TuneResult:
    """Stable result schema shared by Python service, CLI, and MCP."""

    schema_version: str = SCHEMA_VERSION
    status: Literal["ok", "validation_error", "tuning_error", "output_error", "interrupted"] = "ok"

    graph_params: dict[str, float | int | None] | None = None
    best_tau: float | None = None

    best_score: float | None = None
    best_fiedler: float | None = None
    best_var_lambda: float | None = None
    best_mrr_proxy: float | None = None

    input_info: EmbeddingInfo | None = None

    n_trials_requested: int = 0
    n_trials_complete: int = 0
    n_trials_pruned: int = 0

    elapsed_seconds: float | None = None

    arrowspace_tuner_version: str = "0.0.0+unknown"
    arrowspace_version: str | None = None

    report_path: str | None = None
    output_path: str | None = None

    warnings: tuple[str, ...] = field(default=())
    error_code: str | None = None
    error_message: str | None = None


def _json_float(value: float | None) -> float | None:
    """Convert non-finite floats to ``None`` so JSON output stays valid."""
    if value is None:
        return None
    as_float = float(value)
    return as_float if math.isfinite(as_float) else None


def embedding_info_to_dict(info: EmbeddingInfo) -> dict[str, object]:
    """Serialise ``EmbeddingInfo`` to a JSON-compatible dict."""
    return {
        "path": info.path,
        "format": info.format,
        "array_key": info.array_key,
        "n_items": info.n_items,
        "n_dimensions": info.n_dimensions,
        "shape": [info.shape[0], info.shape[1]],
        "dtype": info.dtype,
        "finite": info.finite,
        "l2_norm_min": _json_float(info.l2_norm_min),
        "l2_norm_mean": _json_float(info.l2_norm_mean),
        "l2_norm_max": _json_float(info.l2_norm_max),
        "estimated_memory_bytes": info.estimated_memory_bytes,
        "sha256": info.sha256,
        "warnings": list(info.warnings),
    }


def tune_result_to_dict(result: TuneResult) -> dict[str, object]:
    """
    Serialise a TuneResult to the stable JSON schema (version "1.0").

    Rules:
    - preserve ``graph_params["topk"]`` (never emit ``top_k``);
    - ``best_tau`` stays a top-level key, outside ``graph_params``;
    - tuples become JSON arrays;
    - non-finite floats become ``null``;
    - ``schema_version`` is always present.
    """
    return {
        "schema_version": result.schema_version,
        "status": result.status,
        "graph_params": (dict(result.graph_params) if result.graph_params is not None else None),
        "best_tau": _json_float(result.best_tau),
        "best_score": _json_float(result.best_score),
        "best_fiedler": _json_float(result.best_fiedler),
        "best_var_lambda": _json_float(result.best_var_lambda),
        "best_mrr_proxy": _json_float(result.best_mrr_proxy),
        "input_info": (
            embedding_info_to_dict(result.input_info) if result.input_info is not None else None
        ),
        "n_trials_requested": result.n_trials_requested,
        "n_trials_complete": result.n_trials_complete,
        "n_trials_pruned": result.n_trials_pruned,
        "elapsed_seconds": _json_float(result.elapsed_seconds),
        "arrowspace_tuner_version": result.arrowspace_tuner_version,
        "arrowspace_version": result.arrowspace_version,
        "report_path": result.report_path,
        "output_path": result.output_path,
        "warnings": list(result.warnings),
        "error_code": result.error_code,
        "error_message": result.error_message,
    }
