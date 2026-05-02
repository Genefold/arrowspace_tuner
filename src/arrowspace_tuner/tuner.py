"""
tuner.py — EpsTuner: the main public class for hyperparameter discovery.

Typical usage:
    from arrowspace_tuner import EpsTuner

    tuner = EpsTuner(n_trials=15)
    aspace, gl = tuner.fit(embeddings)

    # inspect results
    print(tuner.best_params)   # {"eps": 1.2, "k": 14}
    print(tuner.best_score)

    # optional: save full report (requires [report] extra)
    tuner.save_report(out_dir="results")

    # use the recommended search_tau when calling search
    results = aspace.search(query_embedding, gl, tau=tuner.search_tau)
"""
from __future__ import annotations

import logging
from collections.abc import Callable
from pathlib import Path
from typing import Any

import numpy as np
import optuna

from .core import BuildParams, StudyConfig, make_objective
from .core.config import _DEFAULT_N_TRIALS, DEFAULT_SEARCH_TAU

logger = logging.getLogger(__name__)

optuna.logging.set_verbosity(optuna.logging.WARNING)


class EpsTuner:
    """
    Hyperparameter discovery for ArrowSpace via Optuna.

    Searches over eps and k using the spectral MRR-Top0 proxy as the
    objective (query-free, label-agnostic).

    tau is NOT optimised. It is a search-time parameter controlling the
    spectral/cosine blend in search_batch(). Optimising it alongside
    eps/k creates a circular reward (see objective.py for details).
    Use the ``search_tau`` parameter to set it; the same value is stored
    on ``tuner.search_tau`` for convenience after fit().

    Parameters
    ----------
    n_trials : int
        Number of Optuna trials. Default 15.
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
    search_tau : float
        Fixed tau passed to search_batch() inside every trial.
        Not optimised — see class docstring. Default: 0.5.
    n_probe : int
        Number of anchor queries per trial for the MRR proxy.
        Scales search_batch cost linearly; 50 is the default (~14% s.e.).
    n_jobs : int
        Number of parallel workers. Default 1 (serial, fully reproducible).

    Attributes (available after .fit())
    ------------------------------------
    best_params : dict[str, Any]
        Best hyperparameters found: {"eps": float, "k": int}.
    best_score : float
        Best composite objective score achieved.
    best_fiedler : float
        Fiedler value at the best trial (connectivity health).
    best_var_lambda : float
        Lambda variance at the best trial (spectral richness).
    best_mrr_proxy : float
        MRR proxy at the best trial (retrieval coherence).
    search_tau : float
        The tau value used during the study — pass this to aspace.search().
    study : optuna.Study
        The raw Optuna study object for custom analysis.
    """

    def __init__(
        self,
        *,
        n_trials:   int          = _DEFAULT_N_TRIALS,
        sample_n:   int | None   = None,
        seed:       int          = 54,
        study_name: str          = "arrowspace_tuner",
        storage:    str | None   = None,
        eps_low:    float        = 0.3,
        eps_high:   float        = 4.0,
        k_low:      int          = 3,
        k_high:     int          = 40,
        search_tau: float        = DEFAULT_SEARCH_TAU,
        n_probe:    int          = 50,
        n_jobs:     int          = 1,
    ) -> None:
        self._cfg = StudyConfig(
            n_trials   = n_trials,
            sample_n   = sample_n,
            seed       = seed,
            study_name = study_name,
            storage    = storage,
            eps_low    = eps_low,
            eps_high   = eps_high,
            k_low      = k_low,
            k_high     = k_high,
            search_tau = search_tau,
            n_probe    = n_probe,
            n_jobs     = n_jobs,
        )
        self.search_tau: float = search_tau

        # Results — populated by .fit()
        self.best_params:     dict[str, Any] | None = None
        self.best_score:      float | None          = None
        self.best_fiedler:    float | None          = None
        self.best_var_lambda: float | None          = None
        self.best_mrr_proxy:  float | None          = None
        self.study:           optuna.Study | None   = None

    # ── public interface ─────────────────────────────────────────────────────────

    def fit(
        self,
        embeddings: np.ndarray,
    ) -> tuple[object, object]:
        """
        Run hyperparameter search and return the best (aspace, gl) pair.

        Parameters
        ----------
        embeddings : np.ndarray
            Shape (N, D) float64 corpus embeddings.

        Returns
        -------
        aspace : ArrowSpace
            ArrowSpace index built with the best hyperparameters found.
        gl : GraphLaplacian
            Corresponding graph Laplacian.

        Raises
        ------
        ValueError
            If embeddings are not a 2D array.
        RuntimeError
            If all trials were pruned.
        """
        embeddings = self._validate(embeddings)

        cfg = self._cfg
        logger.info(
            "Starting EpsTuner: n_trials=%d  sample_n=%s  seed=%d  n_jobs=%d  search_tau=%.3f",
            cfg.n_trials, cfg.sample_n, cfg.seed, cfg.n_jobs, cfg.search_tau,
        )

        try:
            from optuna.samplers import GPSampler
            sampler: optuna.samplers.BaseSampler = GPSampler(
                seed             = cfg.seed,
                n_startup_trials = 4,
            )
            logger.info("Sampler: GPSampler (BoTorch backend)")
        except ImportError:
            sampler = optuna.samplers.TPESampler(
                seed             = cfg.seed,
                n_startup_trials = 4,
                multivariate     = True,
                group            = True,
            )
            logger.info(
                "Sampler: TPESampler(multivariate=True, n_startup_trials=4) "
                "[install optuna[botorch] for GPSampler]"
            )

        pruner = optuna.pruners.MedianPruner(
            n_startup_trials = 4,
            n_warmup_steps   = 0,
        )

        study = optuna.create_study(
            direction      = "maximize",
            study_name     = cfg.study_name,
            storage        = cfg.storage,
            sampler        = sampler,
            pruner         = pruner,
            load_if_exists = cfg.storage is not None,
        )

        completed_so_far = [
            t for t in study.trials
            if t.state == optuna.trial.TrialState.COMPLETE
        ]
        if not completed_so_far:
            study.enqueue_trial(
                {
                    "eps": max(cfg.eps_low, min(cfg.eps_high, 1.0)),
                    "k":   max(cfg.k_low,   min(cfg.k_high,   15)),
                },
                skip_if_exists=True,
            )

        objective_fn, best_cache = make_objective(embeddings, cfg)
        objective: Callable[[optuna.Trial], float] = objective_fn  # type: ignore[assignment]
        study.optimize(objective, n_trials=cfg.n_trials, n_jobs=cfg.n_jobs)

        completed = [
            t for t in study.trials
            if t.state == optuna.trial.TrialState.COMPLETE
        ]
        if not completed:
            raise RuntimeError(
                "All Optuna trials were pruned — no valid graph was built. "
                "Try widening eps_low/eps_high or increasing sample_n."
            )

        best             = study.best_trial
        self.study       = study
        self.best_params     = best.params
        self.best_score      = best.value
        self.best_fiedler    = best.user_attrs.get("fiedler")
        self.best_var_lambda = best.user_attrs.get("var_lambda")
        self.best_mrr_proxy  = best.user_attrs.get("mrr_proxy")

        logger.info(
            "EpsTuner finished | best score=%.6f | eps=%.5f k=%d",
            self.best_score,
            self.best_params["eps"],
            self.best_params["k"],
        )

        if best_cache:
            logger.info("Returning cached best-trial objects (skipping redundant build)")
            return best_cache["aspace"], best_cache["gl"]

        return self._final_build(embeddings)

    def save_report(self, out_dir: str = "results") -> Path:
        """
        Save trial CSV, best_params.json, and HTML plots to disk.

        Requires the [report] extra:
            pip install arrowspace-tuner[report]
        """
        if self.study is None:
            raise RuntimeError("Call .fit() before .save_report().")

        from .reporting import save_results
        return save_results(self.study, out_dir=out_dir)

    # ── private helpers ──────────────────────────────────────────────────────────

    def _validate(self, embeddings: np.ndarray) -> np.ndarray:
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

    def _final_build(self, embeddings: np.ndarray) -> tuple[object, object]:
        """
        Rebuild ArrowSpace once with the best params found by the study.
        Only called when sample_n is set (subsample path).
        """
        from arrowspace import ArrowSpaceBuilder

        assert self.best_params is not None, "best_params must be set before _final_build"

        params = BuildParams(
            eps  = self.best_params["eps"],
            k    = self.best_params["k"],
            topk = max(1, self.best_params["k"] // 2),
        )

        embeddings = np.ascontiguousarray(embeddings, dtype=np.float64)

        aspace, gl = (
            ArrowSpaceBuilder()
            .with_dims_reduction(enabled=False, eps=None)
            .with_cluster_max_clusters(self._cfg.max_clusters)
            .with_cluster_radius(self._cfg.cluster_radius)
            .build(params.to_dict(), embeddings)
        )

        logger.info(
            "Final build complete | eps=%.5f k=%d topk=%d",
            params.eps, params.k, params.topk,
        )
        return aspace, gl

    def __repr__(self) -> str:
        fitted = self.best_params is not None
        status = (
            f"best_score={self.best_score:.6f} params={self.best_params}"
            if fitted else "not fitted"
        )
        return (
            f"EpsTuner("
            f"n_trials={self._cfg.n_trials}, "
            f"eps=[{self._cfg.eps_low}, {self._cfg.eps_high}], "
            f"k=[{self._cfg.k_low}, {self._cfg.k_high}], "
            f"search_tau={self.search_tau}, "
            f"{status})"
        )
