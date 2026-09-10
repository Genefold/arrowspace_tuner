"""
service.py — the shared application service for the CLI and MCP server.

Layering (issue #17):

    arrowspace_tuner.core → arrowspace_tuner.service → cli.py / mcp_server.py

Both adapters call exactly these two functions and nothing else from the
tuner internals; neither may touch private EpsTuner methods, and the MCP
server must never shell out to the CLI.
"""

from __future__ import annotations

import logging
import time
from importlib.metadata import PackageNotFoundError
from importlib.metadata import version as _pkg_version
from pathlib import Path

import optuna

from . import io as _io
from .io import InputValidationError, TuningExecutionError
from .models import SCHEMA_VERSION, EmbeddingInfo, TuneRequest, TuneResult
from .tuner import EpsTuner

logger = logging.getLogger(__name__)


def _arrowspace_version() -> str | None:
    try:
        return _pkg_version("arrowspace")
    except PackageNotFoundError:
        return None


def validate_request(request: TuneRequest) -> None:
    """
    Validate scalar request bounds. Raises InputValidationError; tau is
    search-time and must stay within [0, 1] — invalid values are rejected,
    never clamped.
    """
    if request.n_trials < 1:
        raise InputValidationError(
            f"n_trials must be >= 1; got {request.n_trials}.", code="invalid_trials"
        )
    if request.sample_n is not None and request.sample_n < 2:
        raise InputValidationError(
            f"sample_n must be >= 2 (or null), got {request.sample_n}.",
            code="invalid_sample_n",
        )
    if request.eps_low <= 0:
        raise InputValidationError(
            f"eps_low must be > 0; got {request.eps_low}.", code="invalid_eps_bounds"
        )
    if request.eps_high <= request.eps_low:
        raise InputValidationError(
            f"eps_high must be greater than eps_low; got [{request.eps_low}, {request.eps_high}].",
            code="invalid_eps_bounds",
        )
    if request.k_low < 1:
        raise InputValidationError(
            f"k_low must be >= 1; got {request.k_low}.", code="invalid_k_bounds"
        )
    if request.k_high < request.k_low:
        raise InputValidationError(
            f"k_high must be >= k_low; got [{request.k_low}, {request.k_high}].",
            code="invalid_k_bounds",
        )
    if request.tau_low < 0.0:
        raise InputValidationError(
            f"tau_low must be within [0.0, 1.0]; got {request.tau_low}.",
            code="invalid_tau_bounds",
        )
    if request.tau_high > 1.0:
        raise InputValidationError(
            f"tau_high must be within [0.0, 1.0]; got {request.tau_high}.",
            code="invalid_tau_bounds",
        )
    if request.tau_high < request.tau_low:
        raise InputValidationError(
            f"tau_high must be >= tau_low; got [{request.tau_low}, {request.tau_high}].",
            code="invalid_tau_bounds",
        )
    if request.n_probe < 1:
        raise InputValidationError(
            f"n_probe must be >= 1; got {request.n_probe}.", code="invalid_n_probe"
        )
    if request.n_jobs < 1:
        raise InputValidationError(
            f"n_jobs must be >= 1; got {request.n_jobs}.", code="invalid_n_jobs"
        )
    if request.save_report and request.report_dir is None:
        raise InputValidationError(
            "save_report=true requires report_dir.", code="missing_report_dir"
        )


def inspect_embeddings(
    input_path: Path,
    *,
    array_key: str | None = None,
    include_hash: bool = True,
) -> EmbeddingInfo:
    """Load and validate a local embedding matrix without tuning."""
    info, _ = _io.load_embeddings(input_path, array_key=array_key, include_hash=include_hash)
    return info


def _error_result(
    status: str,
    error_code: str,
    error_message: str,
    request: TuneRequest | None = None,
    input_info: EmbeddingInfo | None = None,
    warnings: tuple[str, ...] = (),
) -> TuneResult:
    return TuneResult(
        schema_version=SCHEMA_VERSION,
        status=status,  # type: ignore[arg-type]
        graph_params=None,
        best_tau=None,
        input_info=input_info,
        n_trials_requested=request.n_trials if request is not None else 0,
        arrowspace_tuner_version=_tuner_version(),
        arrowspace_version=_arrowspace_version(),
        warnings=warnings,
        error_code=error_code,
        error_message=error_message,
    )


def _tuner_version() -> str:
    from arrowspace_tuner import __version__

    return __version__


def run_tuning(request: TuneRequest) -> TuneResult:
    """
    Run graph-parameter tuning and return a stable structured result.

    Expected input failures become ``validation_error`` results; unexpected
    failures are logged with traceback and raised as TuningExecutionError.
    """
    started = time.perf_counter()

    # ── 1. request bounds ────────────────────────────────────────────────────
    try:
        validate_request(request)
    except InputValidationError as exc:
        return _error_result("validation_error", exc.code, str(exc), request)

    # ── 2. input inspection and loading ─────────────────────────────────────
    try:
        input_info, embeddings = _io.load_embeddings(
            request.input_path,
            array_key=request.array_key,
            include_hash=request.include_input_hash,
        )
    except InputValidationError as exc:
        return _error_result("validation_error", exc.code, str(exc), request)

    warnings_list: list[str] = list(input_info.warnings)

    # ── corpus-aware neighbour bounds: the largest legal k is n_items - 1 ────
    max_valid_k = input_info.n_items - 1
    if request.k_low > max_valid_k:
        return _error_result(
            "validation_error",
            "k_low_exceeds_corpus",
            (
                f"k_low={request.k_low} exceeds the maximum valid neighbour "
                f"count {max_valid_k} for a corpus with "
                f"{input_info.n_items} rows."
            ),
            request,
            input_info=input_info,
            warnings=tuple(warnings_list),
        )
    effective_k_high = min(request.k_high, max_valid_k)
    if effective_k_high < request.k_high:
        warnings_list.append(_io.k_high_warning(input_info.n_items))
    if effective_k_high == request.k_low:
        # EpsTuner searches k_low < k_high; a corpus this small leaves a
        # single legal k and no search room — reject before tuning instead
        # of failing inside Optuna.
        return _error_result(
            "validation_error",
            "k_range_too_narrow",
            (
                f"Corpus with {input_info.n_items} rows leaves a single valid "
                f"neighbour count (k={request.k_low}); tuning requires "
                "k_low < k_high. Use a larger corpus "
                f"(n_items >= {request.k_low + 2}) or lower k_low."
            ),
            request,
            input_info=input_info,
            warnings=tuple(warnings_list),
        )

    # ── dry run: no EpsTuner, no Optuna study ────────────────────────────────
    if request.dry_run:
        return TuneResult(
            schema_version=SCHEMA_VERSION,
            status="ok",
            graph_params=None,
            best_tau=None,
            input_info=input_info,
            elapsed_seconds=time.perf_counter() - started,
            arrowspace_tuner_version=_tuner_version(),
            arrowspace_version=_arrowspace_version(),
            warnings=tuple(warnings_list),
        )

    # ── 5-7. run the shared EpsTuner ─────────────────────────────────────────
    try:
        tuner = EpsTuner(
            n_trials=request.n_trials,
            sample_n=request.sample_n,
            seed=request.seed,
            eps_low=request.eps_low,
            eps_high=request.eps_high,
            k_low=request.k_low,
            k_high=effective_k_high,
            tau_low=request.tau_low,
            tau_high=request.tau_high,
            n_probe=request.n_probe,
            n_jobs=request.n_jobs,
        )
        graph_params = tuner.fit(embeddings)
    except RuntimeError as exc:
        # All trials pruned / no valid result — expected tuning failure.
        return _error_result(
            "tuning_error",
            "no_valid_trials",
            str(exc),
            request,
            input_info=input_info,
            warnings=tuple(warnings_list),
        )
    except Exception as exc:
        logger.exception("Tuning failed unexpectedly")
        raise TuningExecutionError(f"{type(exc).__name__}: {exc}") from exc

    # ── 8. optional report ───────────────────────────────────────────────────
    report_path: str | None = None
    if request.save_report and request.report_dir is not None:
        try:
            report_path = str(tuner.save_report(out_dir=str(request.report_dir)))
        except (ImportError, OSError) as exc:
            logger.warning("Could not save requested report", exc_info=True)
            return _error_result(
                "output_error",
                "report_write_failed",
                f"Could not save requested report: {exc}",
                request,
                input_info=input_info,
                warnings=tuple(warnings_list),
            )

    # ── 9. trial bookkeeping ─────────────────────────────────────────────────
    n_complete = 0
    n_pruned = 0
    if tuner.study is not None:
        states = [t.state for t in tuner.study.trials]
        n_complete = sum(s == optuna.trial.TrialState.COMPLETE for s in states)
        n_pruned = sum(s == optuna.trial.TrialState.PRUNED for s in states)

    return TuneResult(
        schema_version=SCHEMA_VERSION,
        status="ok",
        graph_params=dict(graph_params),
        best_tau=tuner.best_tau,
        best_score=tuner.best_score,
        best_fiedler=tuner.best_fiedler,
        best_var_lambda=tuner.best_var_lambda,
        best_mrr_proxy=tuner.best_mrr_proxy,
        input_info=input_info,
        n_trials_requested=request.n_trials,
        n_trials_complete=n_complete,
        n_trials_pruned=n_pruned,
        elapsed_seconds=time.perf_counter() - started,
        arrowspace_tuner_version=_tuner_version(),
        arrowspace_version=_arrowspace_version(),
        report_path=report_path,
        output_path=str(request.output_path) if request.output_path else None,
        warnings=tuple(warnings_list),
    )
