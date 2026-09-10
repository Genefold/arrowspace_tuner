"""
mcp_server.py — local stdio MCP server exposing the shared tuning service.

The MCP server calls ``service.run_tuning`` / ``service.inspect_embeddings``
directly — never the CLI, never subprocess. Transport is stdio only.

Security model (see README "Security model for MCP"):

- every file path must be absolute, exist, resolve (symlinks included)
  inside one configured allowed root, and end in .npy/.npz;
- ``ARROWSPACE_TUNER_ALLOWED_ROOTS`` is mandatory — the server refuses to
  start without it and never defaults to ``/`` or the CWD;
- input size, trial count, and worker count are capped via environment
  variables;
- no network access: embeddings are read from local files only and are
  never uploaded.
"""

from __future__ import annotations

import importlib
import math
import os
from collections.abc import Callable
from dataclasses import dataclass
from pathlib import Path
from typing import Any, Protocol, cast

from arrowspace_tuner.io import (
    SUPPORTED_SUFFIXES,
    InputValidationError,
    UnsupportedInputFormatError,
)
from arrowspace_tuner.models import (
    GRAPH_BUILD_KEYS,
    SCHEMA_VERSION,
    TAU_HIGH,
    TAU_LOW,
    TuneRequest,
    embedding_info_to_dict,
    tune_result_to_dict,
)
from arrowspace_tuner.service import inspect_embeddings, run_tuning

_ToolDecorator = Callable[[Callable[..., object]], Callable[..., object]]


class _ServerLike(Protocol):
    """Structural type covering FastMCP (mcp 1.x) and MCPServer (mcp 2.x)."""

    tool: Callable[[], _ToolDecorator]

    def run(self) -> None: ...


_ALLOWED_ROOTS_ENV = "ARROWSPACE_TUNER_ALLOWED_ROOTS"
_MAX_INPUT_BYTES_ENV = "ARROWSPACE_TUNER_MAX_INPUT_BYTES"
_MAX_TRIALS_ENV = "ARROWSPACE_TUNER_MAX_TRIALS"
_MAX_N_JOBS_ENV = "ARROWSPACE_TUNER_MAX_N_JOBS"

_DEFAULT_MAX_INPUT_BYTES = 2 * 1024 * 1024 * 1024  # 2 GiB
_DEFAULT_MAX_TRIALS = 100
_DEFAULT_MAX_N_JOBS = 4

_SERVER_DESCRIPTION = (
    "Local, query-free tuning of ArrowSpace graph-construction parameters from "
    "trusted local embedding arrays. Returns ArrowSpace-compatible graph "
    "parameters (eps, k, topk, p, sigma) and a separate best_tau search-time "
    "value."
)


class McpConfigurationError(RuntimeError):
    """The MCP server cannot start with the current configuration."""


@dataclass(frozen=True)
class McpLimits:
    """Environment-configurable resource limits."""

    max_input_bytes: int = _DEFAULT_MAX_INPUT_BYTES
    max_trials: int = _DEFAULT_MAX_TRIALS
    max_n_jobs: int = _DEFAULT_MAX_N_JOBS


@dataclass(frozen=True)
class McpContext:
    """Everything the tool handlers need: allowed roots and limits."""

    allowed_roots: tuple[str, ...]
    limits: McpLimits


_LIMIT_ENV_KEYS: dict[str, str] = {
    "max_input_bytes": _MAX_INPUT_BYTES_ENV,
    "max_trials": _MAX_TRIALS_ENV,
    "max_n_jobs": _MAX_N_JOBS_ENV,
}


def load_limits(environ: dict[str, str] | None = None) -> McpLimits:
    """Parse resource-limit environment variables with documented defaults.

    All limits must be positive integers; zero, negative, and non-integer
    values are invalid server configuration.
    """
    env = os.environ if environ is None else environ
    defaults = {
        "max_input_bytes": _DEFAULT_MAX_INPUT_BYTES,
        "max_trials": _DEFAULT_MAX_TRIALS,
        "max_n_jobs": _DEFAULT_MAX_N_JOBS,
    }
    values: dict[str, int] = {}
    for field, key in _LIMIT_ENV_KEYS.items():
        raw = env.get(key)
        if raw is None or raw == "":
            values[field] = defaults[field]
            continue
        try:
            parsed = int(raw)
        except ValueError as exc:
            raise McpConfigurationError(f"{key} must be a positive integer; got {raw!r}.") from exc
        if parsed <= 0:
            raise McpConfigurationError(f"{key} must be a positive integer; got {parsed}.")
        values[field] = parsed
    return McpLimits(**values)


def load_allowed_roots(
    environ: dict[str, str] | None = None,
) -> tuple[str, ...]:
    """
    Parse ``ARROWSPACE_TUNER_ALLOWED_ROOTS`` (os.pathsep-separated).

    Every root must exist; there is no default — the server refuses to start
    without an explicit configuration.
    """
    env = os.environ if environ is None else environ
    raw = env.get(_ALLOWED_ROOTS_ENV)
    if not raw or not raw.strip():
        raise McpConfigurationError(
            f"{_ALLOWED_ROOTS_ENV} is required. Point it at one or more "
            f"directories (separated by {os.pathsep!r}) that the server may "
            "read, e.g. ARROWSPACE_TUNER_ALLOWED_ROOTS=/workspace:/data"
        )
    roots: list[str] = []
    for value in raw.split(os.pathsep):
        candidate = value.strip()
        if not candidate:
            continue
        try:
            resolved = Path(candidate).expanduser().resolve(strict=True)
        except OSError as exc:
            raise McpConfigurationError(
                f"Allowed root {candidate!r} does not exist or is not accessible: {exc}"
            ) from exc
        roots.append(str(resolved))
    if not roots:
        raise McpConfigurationError(
            f"{_ALLOWED_ROOTS_ENV} is empty; configure at least one directory."
        )
    return tuple(roots)


def _error(error_code: str, error_message: str) -> dict[str, object]:
    return {
        "schema_version": SCHEMA_VERSION,
        "status": "validation_error",
        "error_code": error_code,
        "error_message": error_message,
        "graph_params": None,
        "best_tau": None,
    }


def ensure_allowed_path(
    raw_path: str,
    *,
    allowed_roots: tuple[str, ...],
    must_exist: bool,
) -> str:
    """
    Enforce the MCP path policy: absolute, resolvable, regular file (when
    ``must_exist``), inside an allowed root after symlink resolution, and
    .npy/.npz only. Remote URLs and relative paths are rejected.
    """
    if not isinstance(raw_path, str) or not raw_path.strip():
        raise InputValidationError("A non-empty path string is required.", code="invalid_path")
    if "://" in raw_path:
        raise InputValidationError(
            "Remote URLs are not supported; provide a local file path.",
            code="url_rejected",
        )
    candidate = Path(raw_path)
    if not candidate.is_absolute():
        raise InputValidationError(f"Path must be absolute: {raw_path}", code="relative_path")
    try:
        resolved = candidate.expanduser().resolve(strict=must_exist)
    except OSError as exc:
        raise InputValidationError(
            f"Path does not exist or is not accessible: {exc.strerror or exc}",
            code="input_not_found",
        ) from exc
    if must_exist and not resolved.is_file():
        raise InputValidationError(
            f"Input must be a regular file: {resolved}", code="not_a_regular_file"
        )
    suffix = resolved.suffix.lower()
    if suffix not in SUPPORTED_SUFFIXES:
        raise UnsupportedInputFormatError(
            f"Unsupported file type '{suffix}'; only .npy and .npz are accepted."
        )
    if not any(resolved.is_relative_to(root) for root in allowed_roots):
        raise InputValidationError(
            f"Path is outside the configured allowed roots: {resolved}",
            code="outside_allowed_roots",
        )
    return str(resolved)


def ensure_allowed_report_dir(
    raw_path: str,
    *,
    allowed_roots: tuple[str, ...],
) -> str:
    """Report directories must be absolute, new, and inside an allowed root."""
    if not isinstance(raw_path, str) or not raw_path.strip():
        raise InputValidationError(
            "A non-empty report_dir string is required.", code="invalid_path"
        )
    candidate = Path(raw_path)
    if not candidate.is_absolute():
        raise InputValidationError(
            f"report_dir must be an absolute path: {raw_path}",
            code="relative_path",
        )
    resolved = candidate.expanduser().resolve(strict=False)
    if resolved.exists():
        raise InputValidationError(
            f"report_dir already exists and will not be overwritten: {resolved}",
            code="report_dir_exists",
        )
    if not any(resolved.is_relative_to(root) for root in allowed_roots):
        raise InputValidationError(
            f"report_dir is outside the configured allowed roots: {resolved}",
            code="outside_allowed_roots",
        )
    return str(resolved)


def _enforce_size(path: str, limits: McpLimits) -> None:
    size = Path(path).stat().st_size
    if size > limits.max_input_bytes:
        raise InputValidationError(
            f"Input file exceeds the configured maximum size "
            f"({size} > {limits.max_input_bytes} bytes); raise "
            f"{_MAX_INPUT_BYTES_ENV} to allow larger files.",
            code="input_too_large",
        )


def _tool_inspect_embeddings(
    ctx: McpContext,
    path: str,
    array_key: str | None = None,
    include_hash: bool = True,
) -> dict[str, object]:
    try:
        resolved = ensure_allowed_path(path, allowed_roots=ctx.allowed_roots, must_exist=True)
        _enforce_size(resolved, ctx.limits)
        info = inspect_embeddings(
            Path(resolved),
            array_key=array_key,
            include_hash=include_hash,
        )
    except InputValidationError as exc:
        return _error(exc.code, str(exc))
    return embedding_info_to_dict(info)


def _tool_tune_graph(
    ctx: McpContext,
    path: str,
    array_key: str | None = None,
    n_trials: int = 15,
    sample_n: int | None = None,
    seed: int = 42,
    eps_low: float = 0.5,
    eps_high: float = 12.0,
    k_low: int = 10,
    k_high: int = 45,
    tau_low: float = 0.0,
    tau_high: float = 1.0,
    n_probe: int = 50,
    n_jobs: int = 1,
    save_report: bool = False,
    report_dir: str | None = None,
) -> dict[str, object]:
    # Reject over-limit requests before touching the filesystem or Optuna.
    if n_trials > ctx.limits.max_trials:
        return _error(
            "invalid_trials",
            f"n_trials {n_trials} exceeds the configured maximum "
            f"{ctx.limits.max_trials}; raise {_MAX_TRIALS_ENV} to allow more.",
        )
    if n_jobs > ctx.limits.max_n_jobs:
        return _error(
            "invalid_n_jobs",
            f"n_jobs {n_jobs} exceeds the configured maximum "
            f"{ctx.limits.max_n_jobs}; raise {_MAX_N_JOBS_ENV} to allow more.",
        )
    try:
        resolved = ensure_allowed_path(path, allowed_roots=ctx.allowed_roots, must_exist=True)
        _enforce_size(resolved, ctx.limits)
        report_resolved: str | None = None
        if save_report:
            if report_dir is None:
                return _error(
                    "missing_report_dir",
                    "save_report=true requires report_dir.",
                )
            report_resolved = ensure_allowed_report_dir(report_dir, allowed_roots=ctx.allowed_roots)
    except InputValidationError as exc:
        return _error(exc.code, str(exc))

    request = TuneRequest(
        input_path=Path(resolved),
        array_key=array_key,
        n_trials=n_trials,
        sample_n=sample_n,
        seed=seed,
        eps_low=eps_low,
        eps_high=eps_high,
        k_low=k_low,
        k_high=k_high,
        tau_low=tau_low,
        tau_high=tau_high,
        n_probe=n_probe,
        n_jobs=n_jobs,
        report_dir=Path(report_resolved) if report_resolved else None,
        save_report=save_report,
    )
    try:
        result = run_tuning(request)
    except Exception:
        # Never leak tracebacks or raw embedding data to the client; the
        # server-side log holds the details.
        return _error(
            "internal_error",
            "Tuning failed unexpectedly; see the MCP server logs for details.",
        )
    return tune_result_to_dict(result)


def _validate_build_instruction_params(
    graph_params: dict[str, object],
) -> dict[str, float | int | None]:
    """
    Validate build-parameter values (keys are checked by the caller):
    eps finite > 0, k integer >= 1, topk integer in [1, k], p finite > 0,
    sigma null or finite > 0. Booleans are rejected explicitly.
    """
    eps = graph_params.get("eps")
    if isinstance(eps, bool) or not isinstance(eps, (int, float)):
        raise InputValidationError(
            "eps must be a finite number greater than zero.",
            code="invalid_graph_params",
        )
    if not math.isfinite(float(eps)) or float(eps) <= 0:
        raise InputValidationError(
            "eps must be a finite number greater than zero.",
            code="invalid_graph_params",
        )

    k = graph_params.get("k")
    if isinstance(k, bool) or not isinstance(k, int):
        raise InputValidationError(
            "k must be an integer greater than or equal to 1.",
            code="invalid_graph_params",
        )
    if k < 1:
        raise InputValidationError(
            "k must be an integer greater than or equal to 1.",
            code="invalid_graph_params",
        )

    topk = graph_params.get("topk")
    if isinstance(topk, bool) or not isinstance(topk, int):
        raise InputValidationError(
            f"topk must be an integer between 1 and k ({k}).",
            code="invalid_graph_params",
        )
    if not 1 <= topk <= k:
        raise InputValidationError(
            f"topk must be an integer between 1 and k ({k}).",
            code="invalid_graph_params",
        )

    p = graph_params.get("p")
    if isinstance(p, bool) or not isinstance(p, (int, float)):
        raise InputValidationError(
            "p must be a finite number greater than zero.",
            code="invalid_graph_params",
        )
    if not math.isfinite(float(p)) or float(p) <= 0:
        raise InputValidationError(
            "p must be a finite number greater than zero.",
            code="invalid_graph_params",
        )

    sigma = graph_params.get("sigma")
    if sigma is not None:
        if isinstance(sigma, bool) or not isinstance(sigma, (int, float)):
            raise InputValidationError(
                "sigma must be null or a finite number greater than zero.",
                code="invalid_graph_params",
            )
        if not math.isfinite(float(sigma)) or float(sigma) <= 0:
            raise InputValidationError(
                "sigma must be null or a finite number greater than zero.",
                code="invalid_graph_params",
            )

    return {
        "eps": float(eps),
        "k": k,
        "topk": topk,
        "p": float(p),
        "sigma": None if sigma is None else float(sigma),
    }


def _validate_best_tau(best_tau: object) -> float:
    """Validate best_tau: real number in [0, 1]; strings, booleans, NaN,
    and infinity are rejected."""
    if isinstance(best_tau, bool) or not isinstance(best_tau, (int, float)):
        raise InputValidationError("best_tau must be a number.", code="invalid_tau")
    tau = float(best_tau)
    if not math.isfinite(tau) or not TAU_LOW <= tau <= TAU_HIGH:
        raise InputValidationError(
            f"best_tau must be within [{TAU_LOW}, {TAU_HIGH}]; got {tau}.",
            code="invalid_tau",
        )
    return tau


def _tool_build_instruction(
    graph_params: dict[str, object],
    best_tau: float,
) -> dict[str, object]:
    if not isinstance(graph_params, dict) or set(graph_params) != set(GRAPH_BUILD_KEYS):
        return _error(
            "invalid_graph_params",
            f"graph_params must contain exactly the keys: {', '.join(GRAPH_BUILD_KEYS)}.",
        )
    try:
        validated = _validate_build_instruction_params(graph_params)
        tau = _validate_best_tau(best_tau)
    except InputValidationError as exc:
        return _error(exc.code, str(exc))
    return {
        "graph_params": validated,
        "search_tau": tau,
        "python_example": (
            "from arrowspace import ArrowSpaceBuilder\n"
            "\n"
            "aspace, gl = ArrowSpaceBuilder().build(graph_params, embeddings)\n"
            "hits = aspace.search(query_embedding, gl, tau=search_tau)\n"
        ),
        "notes": [
            "graph_params contains build-time parameters only",
            "search_tau is separate and must not be passed to ArrowSpaceBuilder.build",
        ],
    }


def _tool_get_tuner_info() -> dict[str, object]:
    from arrowspace_tuner import __version__

    return {
        "schema_version": SCHEMA_VERSION,
        "arrowspace_tuner_version": __version__,
        "arrowspace_requirement": ">=0.26.0,<0.29",
        "supported_input_formats": ["npy", "npz"],
        "graph_build_keys": list(GRAPH_BUILD_KEYS),
        "search_time_key": "best_tau",
        "tau_range": [TAU_LOW, TAU_HIGH],
        "default_n_trials": 15,
        "default_n_probe": 50,
        "security": {
            "transport": "stdio",
            "local_files_only": True,
            "allowed_roots_required": True,
            "remote_urls_supported": False,
        },
    }


def _instantiate_server() -> _ServerLike:
    """Instantiate mcp 2.x MCPServer, or the 1.x FastMCP fallback."""
    from arrowspace_tuner import __version__

    try:
        module = importlib.import_module("mcp.server.mcpserver")  # mcp >= 2
        server_cls: Any = module.MCPServer
        return cast(
            _ServerLike,
            server_cls(
                name="arrowspace-tuner",
                instructions=_SERVER_DESCRIPTION,
                version=__version__,
            ),
        )
    except ImportError:
        module = importlib.import_module("mcp.server.fastmcp")  # mcp 1.x
        server_cls = module.FastMCP
    return cast(
        _ServerLike,
        server_cls(name="arrowspace-tuner", instructions=_SERVER_DESCRIPTION),
    )


def build_server(ctx: McpContext) -> _ServerLike:
    """
    Create the MCP server with the four documented tools registered.
    """
    server = _instantiate_server()

    @server.tool()
    def inspect_embeddings(
        path: str,
        array_key: str | None = None,
        include_hash: bool = True,
    ) -> dict[str, object]:
        """Validate a local .npy/.npz embedding matrix and return metadata without tuning."""
        return _tool_inspect_embeddings(ctx, path, array_key, include_hash)

    @server.tool()
    def tune_graph(
        path: str,
        array_key: str | None = None,
        n_trials: int = 15,
        sample_n: int | None = None,
        seed: int = 42,
        eps_low: float = 0.5,
        eps_high: float = 12.0,
        k_low: int = 10,
        k_high: int = 45,
        tau_low: float = 0.0,
        tau_high: float = 1.0,
        n_probe: int = 50,
        n_jobs: int = 1,
        save_report: bool = False,
        report_dir: str | None = None,
    ) -> dict[str, object]:
        """Tune ArrowSpace graph parameters (eps, k, topk, p, sigma) from a
        local embedding file and return a TuneResult with a separate
        best_tau search-time value."""
        return _tool_tune_graph(
            ctx,
            path,
            array_key,
            n_trials,
            sample_n,
            seed,
            eps_low,
            eps_high,
            k_low,
            k_high,
            tau_low,
            tau_high,
            n_probe,
            n_jobs,
            save_report,
            report_dir,
        )

    @server.tool()
    def build_instruction(
        graph_params: dict[str, object],
        best_tau: float,
    ) -> dict[str, object]:
        """Turn tuned graph_params plus best_tau into an ArrowSpace build/search instruction."""
        return _tool_build_instruction(graph_params, best_tau)

    @server.tool()
    def get_tuner_info() -> dict[str, object]:
        """Return server version, supported formats, and security posture."""
        return _tool_get_tuner_info()

    return server


def run() -> None:
    """Load configuration, build the server, and serve over stdio."""
    ctx = McpContext(
        allowed_roots=load_allowed_roots(),
        limits=load_limits(),
    )
    try:
        server = build_server(ctx)
    except ImportError as exc:
        raise McpConfigurationError(
            "The MCP extra is not installed. Run: pip install 'arrowspace_tuner[mcp]'"
        ) from exc
    server.run()
