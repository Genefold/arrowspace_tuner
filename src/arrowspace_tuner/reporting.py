"""
reporting.py — save Optuna study results to disk.

Required by ``EpsTuner.save_report()``.
Install the [report] extra to use this module::

    pip install arrowspace-tuner[report]
"""
from __future__ import annotations

import json
import logging
from datetime import datetime, timezone
from pathlib import Path

logger = logging.getLogger(__name__)


def save_results(
    study: object,
    out_dir: str = "results",
) -> Path:
    """
    Save trial CSV, best_params.json, and Plotly HTML plots to disk.

    Parameters
    ----------
    study : optuna.Study
        A completed Optuna study (passed by EpsTuner after .fit()).
    out_dir : str
        Root output directory. Files are written to
        ``out_dir/<study_name>/<YYYYMMDD_HHMMSS>/``.

    Returns
    -------
    Path
        The timestamped run directory where all files were saved.

    Raises
    ------
    ImportError
        If pandas or plotly are not installed. Install with
        ``pip install arrowspace-tuner[report]``.
    """
    try:
        import pandas as pd
        import plotly.express as px
    except ImportError as exc:  # pragma: no cover
        raise ImportError(
            "save_report() requires the [report] extra. "
            "Install with: pip install arrowspace-tuner[report]"
        ) from exc

    import optuna as opt

    assert isinstance(study, opt.Study), (
        f"Expected optuna.Study, got {type(study).__name__}"
    )

    # ── build output directory ─────────────────────────────────────────────────
    ts      = datetime.now(tz=timezone.utc).strftime("%Y%m%d_%H%M%S")
    run_dir = Path(out_dir) / study.study_name / ts
    run_dir.mkdir(parents=True, exist_ok=True)

    # ── trials.csv ────────────────────────────────────────────────────────────
    rows = []
    for t in study.trials:
        row: dict[str, object] = {
            "number": t.number,
            "state":  t.state.name,
            "value":  t.value,
        }
        row.update(t.params)
        row.update(t.user_attrs)
        rows.append(row)

    df = pd.DataFrame(rows)
    csv_path = run_dir / "trials.csv"
    df.to_csv(csv_path, index=False)
    logger.info("Saved trials CSV → %s", csv_path)

    # ── best_params.json ──────────────────────────────────────────────────────
    best = study.best_trial
    best_data = {
        "study_name": study.study_name,
        "best_trial": best.number,
        "best_value": best.value,
        "params":     best.params,
        "user_attrs": best.user_attrs,
        "timestamp":  ts,
    }
    json_path = run_dir / "best_params.json"
    json_path.write_text(json.dumps(best_data, indent=2))
    logger.info("Saved best_params JSON → %s", json_path)

    # ── HTML plots (requires plotly) ──────────────────────────────────────────
    completed = df[df["state"] == "COMPLETE"].copy()

    if not completed.empty and "value" in completed.columns:
        # Objective score over trials
        fig_score = px.line(
            completed,
            x="number",
            y="value",
            title=f"{study.study_name} — objective score per trial",
            labels={"number": "Trial", "value": "Score"},
        )
        score_path = run_dir / "score_history.html"
        fig_score.write_html(str(score_path))
        logger.info("Saved score history plot → %s", score_path)

        # eps vs score scatter (if eps column present)
        if "eps" in completed.columns:
            fig_eps = px.scatter(
                completed,
                x="eps",
                y="value",
                color="k" if "k" in completed.columns else None,
                title=f"{study.study_name} — eps vs score",
                labels={"value": "Score"},
            )
            eps_path = run_dir / "eps_vs_score.html"
            fig_eps.write_html(str(eps_path))
            logger.info("Saved eps vs score plot → %s", eps_path)

    logger.info("Report saved to %s", run_dir)
    return run_dir
