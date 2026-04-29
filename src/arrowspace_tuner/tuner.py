"""
tuner.py — EpsTuner: the main public class for hyperparameter discovery.

Typical usage:
    from arrowspace_tuner import EpsTuner

    tuner = EpsTuner(n_trials=15)
    aspace, gl = tuner.fit(embeddings)

    # inspect results
    print(tuner.best_params)   # {"eps": 1.2, "k": 14, "tau": 0.8}
    print(tuner.best_score)

    # optional: save full report (requires [report] extra)
    tuner.save_report(out_dir="results")
"""
from __future__ import annotations

import logging
from typing import Any

import numpy as np
import optuna

from .core import BuildParams, StudyConfig, make_objective
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
        ArrowSpace’s .build() must be thread-safe. If you observe crashes
        or corrupted results with n_jobs > 1, fall back to n_jobs=1.

        Reproducibility: with n_jobs > 1 the trial execution order is
        non-deterministic even with a fixed seed, so best_params may vary
        between runs. Use n_jobs=1 for exact reproducibility.

    Attributes (available after .fit())
    ------------------------------------
    best_params : dict[str, Any]
        Best hyperparameters found: {"eps": float, "k": int, "tau": float}.
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
        n_trials:   int          = _DEFAULT_N_TRIALS,
        sample_n:   int | None   = None,
        seed:       int          = 42,
        study_name: str          = "arrowspace_tuner",
        storage:    str | None   = None,
        eps_low:    float        = 0.3,
        eps_high:   float        = 4.0,
        k_low:      int          = 3,
        k_high:     int          = 40,
        tau_low:    float        = 0.1,
        tau_high:   float        = 2.0,
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
            tau_low    = tau_low,
            tau_high   = tau_high,
            n_probe    = n_probe,
            n_jobs     = n_jobs,
        )

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
            Shape (N, D) float64 corpus embeddings. N should be at least
            a few hundred for meaningful spectral analysis.

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
            If all trials were pruned (corpus too small or search bounds
            too narrow — try widening eps_low/eps_high).
        """
        embeddings = self._validate(embeddings)

        cfg = self._cfg
        logger.info(
            "Starting EpsTuner: n_trials=%d  sample_n=%s  seed=%d  n_jobs=%d",
            cfg.n_trials, cfg.sample_n, cfg.seed, cfg.n_jobs,
        )

        # ── create study ─────────────────────────────────────────────────────────
        sampler = optuna.samplers.TPESampler(seed=cfg.seed)
        pruner  = optuna.pruners.MedianPruner(
            n_startup_trials = 4,
            n_warmup_steps   = 0,
        )
        study   = optuna.create_study(
            direction  = "maximize",
            study_name = cfg.study_name,
            storage    = cfg.storage,
            sampler    = sampler,
            pruner     = pruner,
            load_if_exists = cfg.storage is not None,
        )

        # ── run optimisation ────────────────────────────────────────────────────
        objective, best_cache = make_objective(embeddings, cfg)
        study.optimize(objective, n_trials=cfg.n_trials, n_jobs=cfg.n_jobs)

        # ── guard: all trials pruned ───────────────────────────────────────────
        completed = [
            t for t in study.trials
            if t.state == optuna.trial.TrialState.COMPLETE
        ]
        if not completed:
            raise RuntimeError(
                "All Optuna trials were pruned — no valid graph was built. "
                "Try widening eps_low/eps_high or increasing sample_n."
            )

        # ── store results ─────────────────────────────────────────────────────
        best         = study.best_trial
        self.study   = study
        self.best_params     = best.params
        self.best_score      = best.value
        self.best_fiedler    = best.user_attrs.get("fiedler")
        self.best_var_lambda = best.user_attrs.get("var_lambda")
        self.best_mrr_proxy  = best.user_attrs.get("mrr_proxy")

        logger.info(
            "EpsTuner finished | best score=%.6f | eps=%.5f k=%d tau=%.3f",
            self.best_score,
            self.best_params["eps"],
            self.best_params["k"],
            self.best_params["tau"],
        )

        # ── final build: use cached objects if available (#9) ──────────────────
        # When sample_n=None every trial built on the full corpus, so the
        # best trial's aspace+gl are already correct. Skip _final_build.
        if best_cache:
            logger.info(
                "Returning cached best-trial objects (skipping redundant build)"
            )
            return best_cache["aspace"], best_cache["gl"]

        return self._final_build(embeddings)

    def save_report(self, out_dir: str = "results") -> "Path":  # noqa: F821
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
        return save_results(self.study, out_dir=out_dir)

    # ── private helpers ──────────────────────────────────────────────────────────

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

    def _final_build(self, embeddings: np.ndarray) -> tuple[object, object]:
        """
        Rebuild ArrowSpace once with the best params found by the study.
        This is the (aspace, gl) pair returned to the user.
        Only called when sample_n is set (subsample path), because in that
        case the trial objects were built on a subset, not the full corpus.
        """
        from arrowspace import ArrowSpaceBuilder

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
            f"tau=[{self._cfg.tau_low}, {self._cfg.tau_high}], "
            f"{status})"
        )
