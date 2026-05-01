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
from datetime import datetime
from pathlib import Path

import numpy as np
import optuna

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
    study:   optuna.Study,
    out_dir: str | Path = "results",
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
            "save_results requires the [report] extras. "
            "Run: pip install arrowspace-tuner[report]"
        )

    ts      = datetime.now().strftime("%Y%m%d_%H%M%S")
    run_dir = Path(out_dir) / study.study_name / ts
    run_dir.mkdir(parents=True, exist_ok=True)

    # ── 1. trials.csv ─────────────────────────────────────────────────────────
    rows = []
    for t in study.trials:
        row = {
            "trial":      t.number,
            "score":      t.value if t.value is not None else float("nan"),
            "state":      t.state.name,
            "duration_s": (
                (t.datetime_complete - t.datetime_start).total_seconds()
                if t.datetime_complete and t.datetime_start else None
            ),
        }
        row.update(t.params)
        row.update({f"attr_{k}": v for k, v in t.user_attrs.items()})
        rows.append(row)

    df = pd.DataFrame(rows).sort_values("score", ascending=False)
    csv_path = run_dir / "trials.csv"
    df.to_csv(csv_path, index=False, float_format="%.8f")
    logger.info("Saved trials → %s", csv_path)

    # ── 2. best_params.json ───────────────────────────────────────────────────
    best      = study.best_trial
    best_dict = {
        "trial":      best.number,
        "score":      best.value,
        "params":     best.params,
        "user_attrs": best.user_attrs,
        "timestamp":  ts,
        "study_name": study.study_name,
        "n_trials":   len(study.trials),
    }
    json_path = run_dir / "best_params.json"
    json_path.write_text(json.dumps(best_dict, indent=2))
    logger.info("Saved best params → %s", json_path)

    # ── 3. HTML plots ─────────────────────────────────────────────────────────
    if len(study.trials) > 1:
        _save_html(
            plot_optimization_history(study),
            run_dir / "optimization_history.html",
        )

        completed        = [t for t in study.trials
                            if t.state == optuna.trial.TrialState.COMPLETE]
        objective_values = [t.value for t in completed if t.value is not None]
        has_variance     = (
            len(objective_values) >= 4
            and np.var(objective_values) > 1e-12
        )

        if has_variance:
            _save_html(
                plot_param_importances(study),
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

    return run_dir


def _save_html(fig: object, path: Path) -> None:
    try:
        fig.write_html(str(path))  # type: ignore[union-attr]
        logger.info("Saved plot → %s", path)
    except Exception as exc:
        logger.warning("Could not save %s: %s", path.name, exc)


def _print_summary(
    best:    optuna.Trial,
    run_dir: Path,
    df:      pd.DataFrame,
) -> None:
    completed = df[df["state"] == "COMPLETE"]
    print("\n" + "=" * 60)
    print(f"  Results saved → {run_dir}")
    print("=" * 60)
    print(f"  Trials total     : {len(df)}")
    print(f"  Trials completed : {len(completed)}")
    if not completed.empty:
        nonzero = completed[completed["score"] > 0]
        print(f"  Trials score > 0 : {len(nonzero)}")
        print(f"  Score mean       : {completed['score'].mean():.8f}")
        print(f"  Score std        : {completed['score'].std():.8f}")
    print("-" * 60)
    print(f"  Best trial       : #{best.number}")
    print(f"  Best score       : {best.value:.8f}")
    for k, v in best.params.items():
        print(f"  {k:<16} : {v}")
    for k, v in best.user_attrs.items():
        print(f"  {k:<16} : {v}")
    print("=" * 60)