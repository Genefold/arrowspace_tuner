from __future__ import annotations

from dataclasses import dataclass, field
from typing import Any

# Single source of truth for the default number of trials.
# Referenced by StudyConfig, EpsTuner.__init__, and api.optuna().
_DEFAULT_N_TRIALS: int = 15

# Single source of truth for the MRR probe count.
# 50 probes gives ~14% standard error — adequate for trial ranking.
# Referenced by StudyConfig and api.optuna() to prevent silent divergence.
_DEFAULT_N_PROBE: int = 50


@dataclass
class BuildParams:
    """
    Parameters passed to ArrowSpaceBuilder for a single trial build.

    Attributes
    ----------
    eps : float
        Neighbourhood radius for graph construction.
        Primary hyperparameter being optimised.
    k : int
        Number of nearest neighbours used when building the graph.
    topk : int
        Number of results returned by search.
        Defaults to ``max(1, k // 2)`` via ``__post_init__`` when left at
        the sentinel value ``-1``. Can be overridden for the final build.
    p : float
        Minkowski distance exponent (2.0 = Euclidean).
    sigma : float | None
        Optional Gaussian kernel bandwidth. None = auto.
    max_clusters : int
        Upper bound on the number of clusters fed to the builder.
    cluster_radius : float
        Squared L2 threshold for cluster creation.
    sampling_rate : float
        Fraction of embeddings used per trial build (1.0 = all).
    """

    eps:            float        = 0.8
    k:              int          = 10
    topk:           int          = -1   # sentinel: resolved in __post_init__
    p:              float        = 2.0
    sigma:          float | None = None
    max_clusters:   int | None   = None
    cluster_radius: float | None = None
    sampling_rate:  float        = 1.0

    def __post_init__(self) -> None:
        # Resolve topk sentinel: -1 means "use k // 2"
        if self.topk == -1:
            self.topk = max(1, self.k // 2)

    def to_dict(self) -> dict[str, Any]:
        """Return graph_params dict expected by ArrowSpaceBuilder.build()."""
        return {
            "eps":   self.eps,
            "k":     self.k,
            "topk":  self.topk,
            "p":     self.p,
            "sigma": self.sigma,
        }


@dataclass
class StudyConfig:
    """
    Configuration for the Optuna study loop.

    Attributes
    ----------
    n_trials : int
        Number of Optuna trials to run. Default: 15.
    sample_n : int | None
        Subsample this many embeddings per trial for speed.
        None = use all embeddings every trial.
        Recommended: 5_000 for corpora > 50k items (33x speedup,
        identical best params found vs full-corpus run).
    seed : int
        Random seed for reproducibility.
    study_name : str
        Optuna study identifier. Used as folder name in reporter output.
    storage : str | None
        Optuna storage URL (e.g. "sqlite:///optuna.db"). None = in-memory.
    n_jobs : int
        Number of parallel workers for study.optimize(). Default: 1 (serial).
        Set to -1 to use all available CPU cores, or any positive integer.

        Threading safety note: Optuna n_jobs > 1 runs each trial in a
        separate thread sharing the same Python process. The objective
        closure itself is stateless (captures read-only numpy arrays), so
        it is thread-safe. However, parallelism is only safe if the
        underlying ArrowSpace Rust extension is thread-safe under concurrent
        .build() calls. Verify this before setting n_jobs > 1 in production.

        Reproducibility note: with n_jobs > 1 and TPESampler the trial
        execution order is non-deterministic, so best_params may differ
        across runs even with the same seed. Use n_jobs=1 for reproducible
        comparisons.

    Search space — graph structure
    ------------------------------
    eps_low, eps_high : float
        Log-scale bounds for eps search.
    k_low, k_high : int
        Bounds for k (nearest neighbours) search.

    Search space — retrieval
    ------------------------
    tau_low, tau_high : float
        Bounds for tau search. tau controls the ArrowSpace search
        temperature passed to search_batch(). Optimising tau alongside
        eps and k ensures the graph is evaluated at its best retrieval
        operating point, not an arbitrary fixed tau.

    MRR proxy
    ---------
    n_probe : int
        Number of corpus items used as query anchors per trial when
        computing the spectral MRR-Top0 proxy. Default: 50 (shared with
        api.optuna() via _DEFAULT_N_PROBE). 50 probes gives ~14% MRR
        standard error, which is adequate for ranking trials. Use 200
        only for a final high-accuracy evaluation where trial speed is
        not a concern.
    """

    n_trials:   int          = _DEFAULT_N_TRIALS
    sample_n:   int | None   = None
    seed:       int          = 54
    study_name: str          = "arrowspace_tuner"
    storage:    str | None   = None
    n_jobs:     int          = 1

    # Search space — graph
    eps_low:  float = 0.5
    eps_high: float = 3.0
    k_low:    int   = 3
    k_high:   int   = 40

    # Search space — retrieval
    tau_low:  float = 0.1
    tau_high: float = 1.0

    # MRR proxy — shared constant ensures api.optuna() and EpsTuner agree
    n_probe:  int   = _DEFAULT_N_PROBE

    max_clusters:   int | None   = None
    cluster_radius: float | None = None

    def __post_init__(self) -> None:
        """Validate search-space bounds at construction time."""
        if self.n_trials < 1:
            raise ValueError(f"n_trials must be >= 1, got {self.n_trials}")
        if self.n_probe < 1:
            raise ValueError(f"n_probe must be >= 1, got {self.n_probe}")
        if self.eps_low >= self.eps_high:
            raise ValueError(
                f"eps_low must be < eps_high, got [{self.eps_low}, {self.eps_high}]"
            )
        if self.k_low >= self.k_high:
            raise ValueError(
                f"k_low must be < k_high, got [{self.k_low}, {self.k_high}]"
            )
        if self.tau_low >= self.tau_high:
            raise ValueError(
                f"tau_low must be < tau_high, got [{self.tau_low}, {self.tau_high}]"
            )
        if self.sample_n is not None and self.sample_n < 1:
            raise ValueError(f"sample_n must be >= 1 or None, got {self.sample_n}")
