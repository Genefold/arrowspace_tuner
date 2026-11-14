"""
reporter.py — optional result persistence for arrowspace_tuner.

Saves trial data and Plotly visualisations to disk after a study run.
Requires the [report] extra: pip install arrowspace-tuner[report]

Never called automatically by EpsTuner — must be invoked explicitly:
    tuner.save_report(out_dir="results")
    # or directly:
    from arrowspace_tuner.reporting import save_results
    save_results(study, out_dir="results")
"""

from __future__ import annotations

import json
import logging
import sys
from datetime import datetime
from pathlib import Path

import numpy as np
import optuna
from optuna.trial import FrozenTrial

logger = logging.getLogger(__name__)

try:
    import pandas as pd
    from optuna.visualization import (
        plot_contour,
        plot_optimization_history,
        plot_param_importances,
    )

    _REPORT_DEPS_OK = True
except ImportError:
    _REPORT_DEPS_OK = False
    logger.warning(
        "arrowspace-tuner[report] extras not installed — "
        "HTML plots and CSV output unavailable. "
        "Run: pip install arrowspace-tuner[report]"
    )


def save_results(
    study: optuna.Study,
    out_dir: str | Path = "results",
    *,
    selected_trial: FrozenTrial | None = None,
    selection: dict[str, object] | None = None,
) -> Path:
    """
    Persist study results to out_dir/<study_name>/<timestamp>/.

    Always written (no extra deps required):
        - best_params.json

    Written only with [report] extras installed:
        - trials.csv
        - optimization_history.html
        - param_importances.html   (only if variance > 0 and trials >= 4)
        - contour_eps_k.html       (only if variance > 0 and trials >= 4)

    Parameters
    ----------
    study : optuna.Study
        Completed Optuna study returned by EpsTuner after .fit().
    out_dir : str | Path
        Root directory for output. Subdirectories are created automatically.

    Returns
    -------
    Path
        The timestamped run directory where files were saved.
    """
    if not _REPORT_DEPS_OK:
        raise ImportError(
            "save_results requires the [report] extras. Run: pip install arrowspace-tuner[report]"
        )

    ts = datetime.now().strftime("%Y%m%d_%H%M%S_%f")
    run_dir = Path(out_dir) / study.study_name / ts
    run_dir.mkdir(parents=True, exist_ok=True)

    # ── 1. trials.csv ─────────────────────────────────────────────────────────
    rows = []
    for t in study.trials:
        row = {
            "trial": t.number,
            "score": t.value if t.value is not None else float("nan"),
            "state": t.state.name,
            "duration_s": (
                (t.datetime_complete - t.datetime_start).total_seconds()
                if t.datetime_complete and t.datetime_start
                else None
            ),
        }
        row.update({k: v for k, v in t.params.items() if k not in ("tau", "best_tau")})
        row.update(
            {f"attr_{k}": v for k, v in t.user_attrs.items() if k not in ("tau", "best_tau")}
        )
        rows.append(row)

    df = pd.DataFrame(rows).sort_values("score", ascending=False)
    csv_path = run_dir / "trials.csv"
    df.to_csv(csv_path, index=False, float_format="%.8f")
    logger.info("Saved trials → %s", csv_path)

    # ── 2. best_params.json ───────────────────────────────────────────────────
    best = selected_trial if selected_trial is not None else study.best_trial
    best_dict = {
        "trial": best.number,
        "score": best.value,
        "params": {
            "eps": best.params["eps"],
            "k": best.params["k"],
            "topk": max(1, best.params["k"] // 2),
            "p": 2.0,
            "sigma": None,
        },
        "user_attrs": {k: v for k, v in best.user_attrs.items() if k not in ("tau", "best_tau")},
        "timestamp": ts,
        "study_name": study.study_name,
        "n_trials": len(study.trials),
    }
    if selection is not None:
        best_dict["selection"] = dict(selection)
    json_path = run_dir / "best_params.json"
    json_path.write_text(json.dumps(best_dict, indent=2))
    logger.info("Saved best params → %s", json_path)

    # ── 3. HTML plots ─────────────────────────────────────────────────────────
    if len(study.trials) > 1:
        history = plot_optimization_history(study)
        history.update_layout(title="Raw objective history (not lowest-k selection)")
        _save_html(
            history,
            run_dir / "optimization_history.html",
        )

        completed = [t for t in study.trials if t.state == optuna.trial.TrialState.COMPLETE]
        objective_values = [t.value for t in completed if t.value is not None]
        has_variance = len(objective_values) >= 4 and np.var(objective_values) > 1e-12

        if has_variance:
            _save_html(
                plot_param_importances(study, params=["eps", "k"]),
                run_dir / "param_importances.html",
            )
            _save_html(
                plot_contour(study, params=["eps", "k"]),
                run_dir / "contour_eps_k.html",
            )
        else:
            logger.warning(
                "Skipping param importances: no variance across trials "
                "(sample too small or graph collapsed — try sample_n >= 1000)"
            )

    # ── 4. console summary ────────────────────────────────────────────────────
    _print_summary(best, run_dir, df)
    if selection:
        print(f"  Raw-score winner : #{selection['raw_best_trial_number']}", file=sys.stderr)
        print(f"  Selected trial   : #{selection['selected_trial_number']}", file=sys.stderr)

    return run_dir


def _save_html(fig: object, path: Path) -> None:
    try:
        from plotly.graph_objs import Figure

        if isinstance(fig, Figure):
            fig.write_html(str(path))
            logger.info("Saved plot → %s", path)
        else:
            logger.warning("Cannot save %s: unexpected figure type %s", path.name, type(fig))
    except Exception as exc:
        logger.warning("Could not save %s: %s", path.name, exc)


def _print_summary(
    best: FrozenTrial,
    run_dir: Path,
    df: pd.DataFrame,
) -> None:
    completed = df[df["state"] == "COMPLETE"]
    print("\n" + "=" * 60, file=sys.stderr)
    print(f"  Results saved → {run_dir}", file=sys.stderr)
    print("=" * 60, file=sys.stderr)
    print(f"  Trials total     : {len(df)}", file=sys.stderr)
    print(f"  Trials completed : {len(completed)}", file=sys.stderr)
    if not completed.empty:
        nonzero = completed[completed["score"] > 0]
        print(f"  Trials score > 0 : {len(nonzero)}", file=sys.stderr)
        print(f"  Score mean       : {completed['score'].mean():.8f}", file=sys.stderr)
        print(f"  Score std        : {completed['score'].std():.8f}", file=sys.stderr)
    print("-" * 60, file=sys.stderr)
    print(f"  Selected trial   : #{best.number}", file=sys.stderr)
    print(f"  Selected score   : {best.value:.8f}", file=sys.stderr)
    for k, v in best.params.items():
        if k not in ("tau", "best_tau"):
            print(f"  {k:<16} : {v}", file=sys.stderr)
    for k, v in best.user_attrs.items():
        if k not in ("tau", "best_tau"):
            print(f"  {k:<16} : {v}", file=sys.stderr)
    print("=" * 60, file=sys.stderr)
