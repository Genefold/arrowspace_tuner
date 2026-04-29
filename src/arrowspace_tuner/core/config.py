from __future__ import annotations

from dataclasses import dataclass

# Single source of truth for the default number of trials.
# Referenced by StudyConfig, EpsTuner.__init__, and api.optuna().
_DEFAULT_N_TRIALS: int = 15


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
        Number of results returned by search. Automatically set to k // 2
        during optimisation; can be overridden for the final build.
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
    topk:           int          = 5
    p:              float        = 2.0
    sigma:          float | None = None
    max_clusters:   int          = 5000
    cluster_radius: float        = 0.42
    sampling_rate:  float        = 1.0

    def to_dict(self) -> dict:
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
        computing the spectral MRR-Top0 proxy. Scales search_batch cost
        linearly — 50 probes gives ~14% MRR standard error, which is
        more than adequate for ranking trials. Use 200 only for a final
        high-accuracy evaluation where trial speed is not a concern.
    """

    n_trials:   int          = _DEFAULT_N_TRIALS
    sample_n:   int | None   = None
    seed:       int          = 54
    study_name: str          = "arrowspace_tuner"
    storage:    str | None   = None

    # Search space — graph
    eps_low:  float = 0.3
    eps_high: float = 4.0
    k_low:    int   = 3
    k_high:   int   = 40

    # Search space — retrieval
    tau_low:  float = 0.1
    tau_high: float = 1.0

    # MRR proxy — 50 gives ~14% s.e., adequate for trial ranking (was 200)
    n_probe:  int   = 50
    max_clusters:   int   = 50
    cluster_radius: float = 0.5
