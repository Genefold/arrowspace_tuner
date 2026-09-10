"""
cli.py — Click command-line interface for arrowspace_tuner (issue #17).

Commands: tune, validate, inspect, version, mcp.

Output contract:
- ``--format text`` is for humans;
- ``--format json`` emits exactly one JSON document on stdout and sends all
  logs and errors to stderr as JSON.
"""

from __future__ import annotations

import json
import logging
import os
import sys
import uuid
from pathlib import Path

import click

from arrowspace_tuner.io import InputValidationError, TuningExecutionError
from arrowspace_tuner.models import (
    SCHEMA_VERSION,
    SUPPORTED_FORMATS,
    TAU_HIGH,
    TAU_LOW,
    TuneRequest,
    TuneResult,
    embedding_info_to_dict,
    tune_result_to_dict,
)
from arrowspace_tuner.service import inspect_embeddings, run_tuning

logger = logging.getLogger(__name__)

EXIT_OK = 0
EXIT_USAGE = 2
EXIT_INPUT = 3
EXIT_TUNING = 4
EXIT_OUTPUT = 5
EXIT_INTERRUPTED = 6
EXIT_INTERNAL = 7

STATUS_EXIT_CODES: dict[str, int] = {
    "ok": EXIT_OK,
    "validation_error": EXIT_INPUT,
    "tuning_error": EXIT_TUNING,
    "output_error": EXIT_OUTPUT,
    "interrupted": EXIT_INTERRUPTED,
}


def _output_error_result(error_message: str) -> TuneResult:
    """TuneResult for a requested artifact that could not be persisted."""
    return TuneResult(
        schema_version=SCHEMA_VERSION,
        status="output_error",
        error_code="output_write_failed",
        error_message=error_message,
    )


def _atomic_write_json(payload: dict[str, object], output_path: Path) -> None:
    """Write JSON via a sibling temp file, then atomically replace."""
    tmp_path = output_path.parent / f"{output_path.name}.tmp-{uuid.uuid4().hex}"
    try:
        with open(tmp_path, "w", encoding="utf-8") as handle:
            json.dump(payload, handle, sort_keys=True, indent=2)
            handle.write("\n")
            handle.flush()
            os.fsync(handle.fileno())
        tmp_path.replace(output_path)
    finally:
        tmp_path.unlink(missing_ok=True)


def _json_error(status: str, error_code: str, error_message: str) -> str:
    return json.dumps(
        {
            "schema_version": SCHEMA_VERSION,
            "status": status,
            "error_code": error_code,
            "error_message": error_message,
            "graph_params": None,
            "best_tau": None,
        },
        sort_keys=True,
    )


def _emit_result(
    result: TuneResult,
    output_format: str,
    extra: dict[str, object] | None = None,
) -> int:
    """Echo the result in the requested format and return the exit code."""
    if output_format == "json":
        payload = tune_result_to_dict(result)
        if extra:
            payload = {**payload, **extra}
        click.echo(json.dumps(payload, sort_keys=True))
        return STATUS_EXIT_CODES.get(result.status, EXIT_INTERNAL)

    click.echo(_render_text(result))
    if result.status != "ok":
        click.echo(f"error [{result.error_code}]: {result.error_message}", err=True)
    return STATUS_EXIT_CODES.get(result.status, EXIT_INTERNAL)


def _render_text(result: TuneResult) -> str:
    lines: list[str] = [f"arrowspace_tuner {result.arrowspace_tuner_version}"]
    if result.input_info is not None:
        info = result.input_info
        sha_short = f"{info.sha256[:16]}…" if info.sha256 else "n/a"
        lines += [
            "",
            "Input:",
            f"  Path:       {info.path}",
            f"  Shape:      {info.shape[0]} × {info.shape[1]}",
            f"  Dtype:      {info.dtype}",
            f"  SHA-256:    {sha_short}",
        ]
    if result.status == "ok" and result.graph_params is not None:
        lines += [
            "",
            "Tuning complete.",
            "",
            "Graph parameters:",
            f"  eps:        {result.graph_params['eps']:.6f}",
            f"  k:          {result.graph_params['k']}",
            f"  topk:       {result.graph_params['topk']}",
            f"  p:          {result.graph_params['p']:.6f}",
            "  sigma:      auto"
            if result.graph_params["sigma"] is None
            else f"  sigma:      {result.graph_params['sigma']:.6f}",
            "",
            "Search-time parameter:",
            f"  best_tau:   {result.best_tau:.6f}",
            "",
            "Diagnostics:",
            f"  score:      {result.best_score:.6f}",
            f"  fiedler:    {result.best_fiedler:.6f}",
            f"  var_lambda: {result.best_var_lambda:.6f}",
            f"  mrr_proxy:  {result.best_mrr_proxy:.6f}",
            "",
            "Execution:",
            f"  trials:     {result.n_trials_complete} complete / "
            f"{result.n_trials_pruned} pruned / {result.n_trials_requested} requested",
            f"  elapsed:    {result.elapsed_seconds:.2f} s"
            if result.elapsed_seconds is not None
            else "  elapsed:    n/a",
        ]
    elif result.status == "ok" and result.graph_params is None:
        lines += ["", "Dry run: input validated, no tuning performed."]
    if result.warnings:
        lines += ["", "Warnings:"] + [f"  - {w}" for w in result.warnings]
    if result.status != "ok":
        lines += [
            "",
            f"Error [{result.error_code}]: {result.error_message}",
        ]
    return "\n".join(lines)


@click.group()
@click.version_option(package_name="arrowspace_tuner", prog_name="arrowspace-tuner")
def main() -> None:
    """Tune ArrowSpace graph-construction parameters from local embeddings."""


_FORMAT_OPTION = click.option(
    "--format",
    "output_format",
    type=click.Choice(["text", "json"]),
    default="text",
    show_default=True,
    help="Output format: text for humans, json for automation.",
)


@main.command()
@click.argument("input_path", type=click.Path(path_type=Path))
@click.option("--array-key", default=None, help="Array key for multi-array .npz files.")
@click.option("--trials", type=int, default=15, show_default=True, help="Optuna trials.")
@click.option("--sample-n", type=int, default=None, help="Subsample size per trial.")
@click.option("--seed", type=int, default=42, show_default=True, help="Random seed.")
@click.option("--eps-low", type=float, default=0.5, show_default=True)
@click.option("--eps-high", type=float, default=12.0, show_default=True)
@click.option("--k-low", type=int, default=10, show_default=True)
@click.option("--k-high", type=int, default=45, show_default=True)
@click.option("--tau-low", type=float, default=TAU_LOW, show_default=True)
@click.option("--tau-high", type=float, default=TAU_HIGH, show_default=True)
@click.option("--n-probe", type=int, default=50, show_default=True)
@click.option("--n-jobs", type=int, default=1, show_default=True)
@click.option("--report-dir", type=click.Path(path_type=Path), default=None)
@click.option(
    "--save-report/--no-save-report",
    default=False,
    show_default=True,
    help="Save the trial report under --report-dir.",
)
@click.option(
    "--output",
    type=click.Path(path_type=Path),
    default=None,
    help="Write the TuneResult JSON to this file atomically.",
)
@_FORMAT_OPTION
@click.option(
    "--dry-run", is_flag=True, default=False, help="Validate input only; do not run tuning."
)
@click.option(
    "--no-progress",
    is_flag=True,
    default=False,
    help="Accepted for compatibility; progress goes to stderr only.",
)
@click.option(
    "--include-hash/--no-include-hash",
    default=True,
    show_default=True,
    help="Compute the SHA-256 of the input file.",
)
def tune(
    input_path: Path,
    array_key: str | None,
    trials: int,
    sample_n: int | None,
    seed: int,
    eps_low: float,
    eps_high: float,
    k_low: int,
    k_high: int,
    tau_low: float,
    tau_high: float,
    n_probe: int,
    n_jobs: int,
    report_dir: Path | None,
    save_report: bool,
    output: Path | None,
    output_format: str,
    dry_run: bool,
    no_progress: bool,  # noqa: ARG001
    include_hash: bool,
) -> None:
    """Tune graph-construction parameters (eps, k, topk, p, sigma) from INPUT."""
    request = TuneRequest(
        input_path=input_path,
        array_key=array_key,
        n_trials=trials,
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
        report_dir=report_dir,
        save_report=save_report,
        output_path=output,
        output_format="json" if output_format == "json" else "text",
        dry_run=dry_run,
        include_input_hash=include_hash,
    )
    try:
        result = run_tuning(request)
    except TuningExecutionError as exc:
        _fail(output_format, "tuning_error", "tuning_failed", str(exc), EXIT_INTERNAL)
        return
    except KeyboardInterrupt:
        _interrupted(output_format)
        return

    extra: dict[str, object] | None = None
    if output is not None:
        extra = {
            "seed": seed,
            "input_sha256": (result.input_info.sha256 if result.input_info else None),
        }
        try:
            payload = tune_result_to_dict(result)
            payload.update(extra)
            _atomic_write_json(payload, output)
        except OSError as exc:
            failure = _output_error_result(f"Could not write output file: {exc}")
            if output_format == "json":
                click.echo(json.dumps(tune_result_to_dict(failure), sort_keys=True))
            else:
                click.echo(
                    f"error [output_write_failed]: {failure.error_message}",
                    err=True,
                )
            sys.exit(EXIT_OUTPUT)

    sys.exit(_emit_result(result, output_format, extra))


def _fail(
    output_format: str,
    status: str,
    error_code: str,
    error_message: str,
    exit_code: int,
) -> None:
    if output_format == "json":
        click.echo(_json_error(status, error_code, error_message))
    else:
        click.echo(f"error [{error_code}]: {error_message}", err=True)
    sys.exit(exit_code)


def _interrupted(output_format: str) -> None:
    if output_format == "json":
        click.echo(_json_error("interrupted", "interrupted", "Interrupted."))
    else:
        click.echo("Interrupted.", err=True)
    sys.exit(EXIT_INTERRUPTED)


@main.command()
@click.argument("input_path", type=click.Path(path_type=Path))
@click.option("--array-key", default=None, help="Array key for multi-array .npz files.")
@_FORMAT_OPTION
@click.option(
    "--output",
    type=click.Path(path_type=Path),
    default=None,
    help="Write the EmbeddingInfo JSON to this file atomically.",
)
@click.option("--include-hash/--no-include-hash", default=True, show_default=True)
def validate(
    input_path: Path,
    array_key: str | None,
    output_format: str,
    output: Path | None,
    include_hash: bool,
) -> None:
    """Validate INPUT and print embedding metadata without tuning."""
    try:
        info = inspect_embeddings(input_path, array_key=array_key, include_hash=include_hash)
    except InputValidationError as exc:
        _fail(output_format, "validation_error", exc.code, str(exc), EXIT_INPUT)
        return
    except KeyboardInterrupt:
        _interrupted(output_format)
        return

    payload = embedding_info_to_dict(info)
    if output is not None:
        try:
            _atomic_write_json(payload, output)
        except OSError as exc:
            _fail(
                output_format,
                "output_error",
                "output_write_failed",
                f"Could not write output file: {exc}",
                EXIT_OUTPUT,
            )
            return
    if output_format == "json":
        click.echo(json.dumps(payload, sort_keys=True))
    else:
        click.echo(
            f"{info.path} — {info.shape[0]} × {info.shape[1]} {info.dtype} "
            f"({info.format})\n"
            f"  finite: {info.finite}   "
            f"l2 norms: min={info.l2_norm_min:.4f} "
            f"mean={info.l2_norm_mean:.4f} max={info.l2_norm_max:.4f}\n"
            f"  sha256: {info.sha256 or 'not computed'}"
        )
        for warning in info.warnings:
            click.echo(f"  - {warning}")
    sys.exit(EXIT_OK)


@main.command()
@click.argument("result_path", type=click.Path(path_type=Path))
@_FORMAT_OPTION
def inspect(result_path: Path, output_format: str) -> None:
    """Inspect a previously written TuneResult JSON file."""
    try:
        raw: object = json.loads(result_path.read_text(encoding="utf-8"))
    except (OSError, json.JSONDecodeError) as exc:
        _fail(output_format, "validation_error", "invalid_result_file", str(exc), EXIT_INPUT)
        return
    if not isinstance(raw, dict):
        _fail(
            output_format,
            "validation_error",
            "invalid_result_file",
            "Result file must contain a JSON object.",
            EXIT_INPUT,
        )
        return
    schema = raw.get("schema_version")
    if schema != SCHEMA_VERSION:
        _fail(
            output_format,
            "validation_error",
            "unsupported_schema_version",
            f"Unsupported result schema_version {schema!r}; expected {SCHEMA_VERSION!r}.",
            EXIT_INPUT,
        )
        return
    if output_format == "json":
        click.echo(json.dumps(raw, sort_keys=True))
    else:
        click.echo(_render_inspect_text(raw))
    sys.exit(EXIT_OK)


def _render_inspect_text(raw: dict[str, object]) -> str:
    gp = raw.get("graph_params")
    lines = [
        f"schema_version: {raw.get('schema_version')}",
        f"status:         {raw.get('status')}",
        "Graph parameters:",
    ]
    if isinstance(gp, dict):
        for key in ("eps", "k", "topk", "p", "sigma"):
            lines.append(f"  {key}:{' ' * (10 - len(key))}{gp.get(key, 'n/a')}")
    else:
        lines.append("  n/a")
    lines += [
        f"best_tau:   {raw.get('best_tau')}",
        f"score:      {raw.get('best_score')}",
        f"fiedler:    {raw.get('best_fiedler')}",
        f"var_lambda: {raw.get('best_var_lambda')}",
        f"mrr_proxy:  {raw.get('best_mrr_proxy')}",
    ]
    input_info = raw.get("input_info")
    if isinstance(input_info, dict):
        lines.append(f"input sha256: {input_info.get('sha256', 'n/a')}")
    lines += [
        f"arrowspace_tuner: {raw.get('arrowspace_tuner_version')}   "
        f"arrowspace: {raw.get('arrowspace_version')}",
    ]
    warnings = raw.get("warnings")
    if isinstance(warnings, list) and warnings:
        lines.append("Warnings:")
        lines += [f"  - {w}" for w in warnings]
    if raw.get("report_path"):
        lines.append(f"report: {raw['report_path']}")
    return "\n".join(lines)


@main.command()
@_FORMAT_OPTION
def version(output_format: str) -> None:
    """Print version and capability information."""
    from arrowspace_tuner import __version__

    payload = {
        "arrowspace_tuner_version": __version__,
        "arrowspace_requirement": ">=0.26.0,<0.29",
        "supported_input_formats": list(SUPPORTED_FORMATS),
        "graph_build_keys": ["eps", "k", "topk", "p", "sigma"],
        "search_time_result": "best_tau",
        "tau_range": [TAU_LOW, TAU_HIGH],
        "result_schema_version": SCHEMA_VERSION,
    }
    if output_format == "json":
        click.echo(json.dumps(payload, sort_keys=True))
    else:
        for key, value in payload.items():
            click.echo(f"{key}: {value}")
    sys.exit(EXIT_OK)


@main.command()
def mcp() -> None:
    """Run the local stdio MCP server (requires arrowspace_tuner[mcp])."""
    from arrowspace_tuner import mcp_server

    try:
        mcp_server.run()
    except mcp_server.McpConfigurationError as exc:
        click.echo(f"configuration error: {exc}", err=True)
        sys.exit(EXIT_USAGE)
