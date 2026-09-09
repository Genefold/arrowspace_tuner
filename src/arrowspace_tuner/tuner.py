"""
tuner.py — EpsTuner: the main public class for hyperparameter discovery.

Typical usage:
    from arrowspace import ArrowSpaceBuilder
    from arrowspace_tuner import EpsTuner

    tuner = EpsTuner(n_trials=15)
    graph_params = tuner.fit(embeddings)

    # graph_params is also accessible as a property after fitting
    assert tuner.graph_params is graph_params

    # caller owns the build step
    aspace, gl = ArrowSpaceBuilder().build(graph_params, embeddings)

    # inspect results
    print(tuner.best_params)   # {"eps": 1.2, "k": 14, "topk": 7, "p": 2.0, "sigma": None}
    print(tuner.best_tau)      # 0.8  — use at search time: aspace.search(q, gl, tau=tuner.best_tau)
    print(tuner.best_score)

    # optional: save full report (requires [report] extra)
    tuner.save_report(out_dir="results")
"""
from __future__ import annotations

import json
import logging
from collections.abc import Callable
from pathlib import Path
from typing import Any

import numpy as np
import optuna

from .core import StudyConfig, make_objective
from .core.config import _DEFAULT_N_TRIALS

logger = logging.getLogger(__name__)

# Silence Optuna's verbose output by default.
# Users can re-enable with: optuna.logging.set_verbosity(optuna.logging.INFO)
optuna.logging.set_verbosity(optuna.logging.WARNING)


class EpsTuner:
    """
    Hyperparameter discovery for ArrowSpace via Optuna.

    Searches over eps, k, and tau simultaneously using the spectral
    MRR-Top0 proxy as the objective (query-free, label-agnostic).

    Parameters
    ----------
    n_trials : int
        Number of Optuna trials. Default 15. More trials = better coverage
        of the search space at the cost of runtime.
    sample_n : int | None
        Subsample this many embeddings per trial for speed. None = full
        corpus every trial. Recommended 5_000 for corpora > 50k items
        (validated: 33x speedup, identical best params).
    seed : int
        Random seed for reproducibility.
    study_name : str
        Optuna study name. Used as folder name by save_report().
    storage : str | None
        Optuna storage URI (e.g. "sqlite:///tune.db") for persistence
        across runs. None = in-memory (results lost after .fit() returns).
    eps_low, eps_high : float
        Log-scale search bounds for eps.
    k_low, k_high : int
        Search bounds for k (number of nearest neighbours).
    tau_low, tau_high : float
        Search bounds for tau (ArrowSpace search temperature).
    n_probe : int
        Number of anchor queries per trial for the MRR proxy.
        Scales search_batch cost linearly; 50 is the default (~ 14% s.e.).
    n_jobs : int
        Number of parallel workers. Default 1 (serial, fully reproducible).

        Set to -1 to use all available CPU cores for a ~N× speedup where
        N = min(n_trials, cpu_count). Typical production usage::

            EpsTuner(n_trials=30, n_jobs=-1)   # parallel, fast
            EpsTuner(n_trials=15, n_jobs=1)    # serial, reproducible

        Threading safety: each trial runs in a separate thread sharing
        the same process. The objective closure is stateless, but
        ArrowSpace's .build() must be thread-safe. If you observe crashes
        or corrupted results with n_jobs > 1, fall back to n_jobs=1.

        Reproducibility: with n_jobs > 1 the trial execution order is
        non-deterministic even with a fixed seed, so best_params may vary
        between runs. Use n_jobs=1 for exact reproducibility.
    max_clusters : int | None
        Optional upper bound on the number of clusters passed to the
        ArrowSpace builder. None = use builder default.
    cluster_radius : float | None
        Optional squared L2 threshold for cluster creation passed to the
        ArrowSpace builder. None = use builder default.

    Attributes (available after .fit())
    ------------------------------------
    best_params : dict[str, Any]
        Optimised **build-time** graph parameters:
        {"eps": float, "k": int, "topk": int, "p": float, "sigma": None}.
        Ready to be passed directly to ArrowSpaceBuilder.build().
        Note: tau is NOT included here — see best_tau.
    graph_params : dict[str, Any]
        Convenience property returning the same dict as ``best_params``.
        Raises ``RuntimeError`` if called before ``.fit()``.
    best_tau : float
        Optimal search temperature found during tuning. **Query-time** param.
        Use as: aspace.search(query, gl, tau=tuner.best_tau)
    best_score : float
        Best composite objective score achieved.
    best_fiedler : float
        Fiedler value at the best trial (connectivity health).
    best_var_lambda : float
        Lambda variance at the best trial (spectral richness).
    best_mrr_proxy : float
        MRR proxy at the best trial (retrieval coherence).
    study : optuna.Study
        The raw Optuna study object for custom analysis.
    """

    def __init__(
        self,
        *,
        n_trials:       int          = _DEFAULT_N_TRIALS,
        sample_n:       int | None   = None,
        seed:           int          = 54,
        study_name:     str          = "arrowspace_tuner",
        storage:        str | None   = None,
        eps_low:        float        = 0.3,
        eps_high:       float        = 4.0,
        k_low:          int          = 3,
        k_high:         int          = 40,
        tau_low:        float        = 0.1,
        tau_high:       float        = 1.0,
        n_probe:        int          = 50,
        n_jobs:         int          = 1,
        max_clusters:   int | None   = None,
        cluster_radius: float | None = None,
    ) -> None:
        self._cfg = StudyConfig(
            n_trials       = n_trials,
            sample_n       = sample_n,
            seed           = seed,
            study_name     = study_name,
            storage        = storage,
            eps_low        = eps_low,
            eps_high       = eps_high,
            k_low          = k_low,
            k_high         = k_high,
            tau_low        = tau_low,
            tau_high       = tau_high,
            n_probe        = n_probe,
            n_jobs         = n_jobs,
            max_clusters   = max_clusters,
            cluster_radius = cluster_radius,
        )

        # Results — populated by .fit()
        self.best_params:     dict[str, Any] | None = None
        self.best_score:      float | None          = None
        self.best_fiedler:    float | None          = None
        self.best_var_lambda: float | None          = None
        self.best_mrr_proxy:  float | None          = None
        self.best_tau:        float | None          = None   # query-time param, NOT in best_params
        self.study:           optuna.Study | None   = None
        self._last_report_path: Path | None = None

    # ── public interface ─────────────────────────────────────────────────────────

    @property
    def graph_params(self) -> dict[str, Any]:
        """
        Optimised graph parameters ready for ArrowSpaceBuilder.build().

        This is a read-only view of ``best_params``: the same dict object is
        returned, not a copy. It is not mutated after ``.fit()`` returns.

        Returns
        -------
        dict[str, Any]
            ``{"eps": float, "k": int, "topk": int, "p": float, "sigma": None}``

        Raises
        ------
        RuntimeError
            If called before ``.fit()``.
        """
        if self.best_params is None:
            raise RuntimeError(
                "Call .fit() before accessing .graph_params."
            )
        return self.best_params

    def fit(
        self,
        embeddings: np.ndarray,
    ) -> dict[str, Any]:
        """
        Run hyperparameter search and return the best graph parameters.

        Parameters
        ----------
        embeddings : np.ndarray
            Shape (N, D) float64 corpus embeddings. N should be at least
            a few hundred for meaningful spectral analysis.

        Returns
        -------
        dict[str, Any]
            Optimised **build-time** graph parameters:
            {"eps": float, "k": int, "topk": int, "p": float, "sigma": None}.
            Ready to be passed directly to ArrowSpaceBuilder.build().
            Note: tau is NOT included here — see ``best_tau``.

        Attributes (available after .fit())
        ------------------------------------
        best_params : dict[str, Any]
            Same dict as the return value (identity).
        best_tau : float
            Optimal search temperature found during tuning. **Query-time** param.
            Use as: aspace.search(query, gl, tau=tuner.best_tau)
        best_score : float
            Best composite objective score achieved.
        best_fiedler : float
            Fiedler value at the best trial (connectivity health).
        best_var_lambda : float
            Lambda variance at the best trial (spectral richness).
        best_mrr_proxy : float
            MRR proxy at the best trial (retrieval coherence).
        study : optuna.Study
            The raw Optuna study object for custom analysis.

        Raises
        ------
        ValueError
            If embeddings are not a 2D array.
        RuntimeError
            If all trials were pruned (corpus too small or search bounds
            too narrow — try widening eps_low/eps_high).
        """
        embeddings = self._validate(embeddings)

        cfg = self._cfg

        # ── pre-flight: warn when n_trials is below the pruner startup threshold ──
        if cfg.n_trials < 5:
            logger.warning(
                "n_trials=%d is very low — the MedianPruner needs at least "
                "n_startup_trials=4 completed trials to function. "
                "With n_trials=1 every trial will be pruned and fit() will raise. "
                "Recommend n_trials >= 10.",
                cfg.n_trials,
            )

        logger.info(
            "Starting EpsTuner: n_trials=%d  sample_n=%s  seed=%d  n_jobs=%d",
            cfg.n_trials, cfg.sample_n, cfg.seed, cfg.n_jobs,
        )

        # ── sampler: GPSampler → TPE multivariate fallback (#4) ─────────────────
        # GPSampler (Gaussian Process) is the best choice for small trial
        # budgets (≤30) on a low-dimensional continuous space.
        # A bare `from optuna.samplers import GPSampler` succeeds without
        # torch, but GPSampler only fails deep inside study.optimize()
        # (torch is imported lazily by optuna). Probe torch here so clean
        # installs — the built wheel with bare optuna — fall back to TPE
        # instead of failing every trial. Discovered by the v0.4.2 wheel
        # smoke test.
        try:
            # probe only — GPSampler needs torch at optimize() time
            import torch  # noqa: F401
            from optuna.samplers import GPSampler
            sampler: optuna.samplers.BaseSampler = GPSampler(
                seed             = cfg.seed,
                n_startup_trials = 4,
            )
            logger.info("Sampler: GPSampler (BoTorch backend)")
        except ImportError:
            sampler = optuna.samplers.TPESampler(
                seed             = cfg.seed,
                n_startup_trials = 4,       # default 10 → 4: more informed trials
                multivariate     = True,    # joint posterior over (eps, k, tau)
                group            = True,
            )
            logger.info(
                "Sampler: TPESampler(multivariate=True, n_startup_trials=4) "
                "[install optuna[botorch] for GPSampler]"
            )

        # ── pruner ──────────────────────────────────────────────────────────────
        pruner = optuna.pruners.MedianPruner(
            n_startup_trials = 4,
            n_warmup_steps   = 0,
        )

        # ── create study ─────────────────────────────────────────────────────────
        study = optuna.create_study(
            direction      = "maximize",
            study_name     = cfg.study_name,
            storage        = cfg.storage,
            sampler        = sampler,
            pruner         = pruner,
            load_if_exists = cfg.storage is not None,
        )

        # ── warm-start: enqueue one known-good anchor trial (#4) ───────────────
        # Gives the surrogate a reasonable starting point so trial 0 is
        # never wasted on an arbitrary corner of the search space.
        # Only enqueue when starting fresh (not resuming from storage).
        completed_so_far = [
            t for t in study.trials
            if t.state == optuna.trial.TrialState.COMPLETE
        ]
        if not completed_so_far:
            study.enqueue_trial(
                {
                    "eps": max(cfg.eps_low,  min(cfg.eps_high,  1.0)),
                    "k":   max(cfg.k_low,    min(cfg.k_high,    15)),
                    "tau": max(cfg.tau_low,  min(cfg.tau_high,  0.5)),
                },
                skip_if_exists=True,
            )

        # ── run optimisation ────────────────────────────────────────────────────
        objective_fn, best_cache = make_objective(embeddings, cfg)
        objective: Callable[[optuna.Trial], float] = objective_fn  # type: ignore[assignment]
        study.optimize(objective, n_trials=cfg.n_trials, n_jobs=cfg.n_jobs)

        # ── guard: all trials pruned ───────────────────────────────────────────
        completed = [
            t for t in study.trials
            if t.state == optuna.trial.TrialState.COMPLETE
        ]
        if not completed:
            # If trials failed with exceptions (not statistical pruning),
            # surface the actual errors instead of the generic corpus advice.
            build_errors = sorted({
                t.user_attrs["build_error"]
                for t in study.trials
                if "build_error" in t.user_attrs
            })
            detail = ""
            if build_errors:
                shown = build_errors[:3]
                detail = (
                    f" The trials failed with {len(build_errors)} distinct "
                    f"exception(s) — these are build errors, not corpus/"
                    f"bounds problems:\n"
                    + "\n".join(f"  - {e}" for e in shown)
                )
            raise RuntimeError(
                "All Optuna trials were pruned — no valid graph was built. "
                f"n_trials={cfg.n_trials}: the MedianPruner requires at least "
                "n_startup_trials=4 completed trials; use n_trials >= 5. "
                "The corpus may also be too small, too uniform, or the eps "
                "bounds too narrow. "
                f"Current bounds: eps=[{cfg.eps_low}, {cfg.eps_high}]. "
                "Try widening eps_low/eps_high, reducing k_high, or increasing "
                "sample_n."
                + detail
            )

        # ── store results ─────────────────────────────────────────────────────
        best                 = study.best_trial
        raw_params           = best.params          # {"eps": float, "k": int, "tau": float}

        self.study           = study
        self.best_tau        = float(raw_params["tau"])  # saved separately — query-time only
        self.best_params     = {
            "eps":   raw_params["eps"],
            "k":     raw_params["k"],
            "topk":  max(1, raw_params["k"] // 2),
            "p":     2.0,
            "sigma": None,
        }
        self.best_score      = best.value
        self.best_fiedler    = best.user_attrs.get("fiedler")
        self.best_var_lambda = best.user_attrs.get("var_lambda")
        self.best_mrr_proxy  = best.user_attrs.get("mrr_proxy")

        logger.info(
            "EpsTuner finished | best score=%.6f | eps=%.5f k=%d tau=%.3f",
            self.best_score,
            self.best_params["eps"],
            self.best_params["k"],
            self.best_tau,
        )

        return self.best_params

    def save_report(self, out_dir: str = "results") -> Path:
        """
        Save trial CSV, best_params.json, and HTML plots to disk.

        Requires the [report] extra:
            pip install arrowspace-tuner[report]

        Parameters
        ----------
        out_dir : str
            Root directory. Files land in out_dir/<study_name>/<timestamp>/.

        Returns
        -------
        Path
            The timestamped run directory where files were saved.

        Raises
        ------
        RuntimeError
            If called before .fit().
        """
        if self.study is None:
            raise RuntimeError("Call .fit() before .save_report().")

        from .reporting import save_results
        path = save_results(self.study, out_dir=out_dir)
        self._last_report_path = path
        return path


    def load_graph_params(self, out_dir: str = "results") -> dict[str, Any]:
        """
        Load optimised graph parameters from the most recently saved JSON report.

        Use this for cross-session reuse: loading parameters from a previous
        run without re-fitting. For in-session use after ``.fit()``, prefer the
        ``.graph_params`` property instead.

        Parameters
        ----------
        out_dir : str
            Root results directory. Ignored when a report was saved in the
            current session (uses that path directly).

        Returns
        -------
        dict[str, Any]
            ``{"eps": float, "k": int, "topk": int, "p": float, "sigma": None}``

        Raises
        ------
        FileNotFoundError
            If no study directory or timestamped run is found.
        ValueError
            If best_params.json is missing required keys.
        """
        if self._last_report_path is not None:
            report_dir = self._last_report_path
        else:
            study_dir = Path(out_dir) / self._cfg.study_name
            if not study_dir.exists() or not study_dir.is_dir():
                raise FileNotFoundError(f"No study directory found at {study_dir}")
            subdirs = sorted(d for d in study_dir.iterdir() if d.is_dir())
            if not subdirs:
                raise FileNotFoundError(f"No timestamped runs found in {study_dir}")
            report_dir = subdirs[-1]  # YYYYMMDD_HHMMSS sorts chronologically

        # Read the JSON
        json_path = report_dir / "best_params.json"
        if not json_path.exists():
            raise FileNotFoundError(
                f"Could not find 'best_params.json' in {report_dir}"
            )
        with open(json_path, encoding="utf-8") as f:
            data = json.load(f)

        # params are nested under "params" key — not at the top level
        params = data.get("params", {})
        eps = params.get("eps")
        k   = params.get("k")

        if eps is None or k is None:
            raise ValueError(
                f"best_params.json is missing required keys under 'params'. "
                f"Found params: {params}"
            )

        return {
            "eps":   float(eps),
            "k":     int(k),
            "topk":  max(1, int(k) // 2),
            "p":     2.0,
            "sigma": None,
        }

    def load_best_params(self, out_dir: str = "results") -> dict[str, Any]:
        """
        Deprecated. Use load_graph_params() instead.

        .. deprecated:: 0.4.0
            load_best_params() is deprecated and will be removed in a future
            release. Use load_graph_params(out_dir) for disk-based loading, or
            the .graph_params property for in-session access after .fit().
        """
        import warnings
        warnings.warn(
            "load_best_params() is deprecated, use load_graph_params() instead.",
            DeprecationWarning,
            stacklevel=2,
        )
        return self.load_graph_params(out_dir=out_dir)

    def _validate(self, embeddings: np.ndarray) -> np.ndarray:
        """
        Validate and cast embeddings to float64.

        Returns the (possibly cast) array so callers always work with float64.
        """
        if not isinstance(embeddings, np.ndarray):
            raise ValueError(
                f"embeddings must be np.ndarray, got {type(embeddings).__name__}"
            )
        if embeddings.ndim != 2:
            raise ValueError(
                f"embeddings must be 2D (N, D), got shape {embeddings.shape}"
            )
        if embeddings.dtype != np.float64:
            logger.warning(
                "embeddings dtype is %s — casting to float64", embeddings.dtype
            )
            embeddings = embeddings.astype(np.float64)
        return embeddings

    def __repr__(self) -> str:
        fitted = self.best_params is not None
        status = (
            f"best_score={self.best_score:.6f} "
            f"params={self.best_params} "
            f"best_tau={self.best_tau:.3f}"
            if fitted else "not fitted"
        )
        return (
            f"EpsTuner("
            f"n_trials={self._cfg.n_trials}, "
            f"eps=[{self._cfg.eps_low}, {self._cfg.eps_high}], "
            f"k=[{self._cfg.k_low}, {self._cfg.k_high}], "
            f"tau=[{self._cfg.tau_low}, {self._cfg.tau_high}], "
            f"{status})"
        )
