"""
api.py — one-liner convenience function for hyperparameter discovery.

    aspace, gl = arrowspace_tuner.optuna(embeddings)

Thin shim over EpsTuner with sensible defaults.
"""
from __future__ import annotations

import numpy as np

from .core.config import _DEFAULT_N_TRIALS, DEFAULT_SEARCH_TAU
from .tuner import EpsTuner


def optuna(
    embeddings: np.ndarray,
    *,
    n_trials:   int        = _DEFAULT_N_TRIALS,
    sample_n:   int | None = 5_000,
    seed:       int        = 54,
    study_name: str        = "arrowspace_fstar",
    storage:    str | None = None,
    eps_low:    float      = 0.3,
    eps_high:   float      = 4.0,
    k_low:      int        = 3,
    k_high:     int        = 40,
    search_tau: float      = DEFAULT_SEARCH_TAU,
    n_probe:    int        = 50,
) -> tuple[object, object]:
    """
    Auto-discover eps and k and return a ready-to-use (aspace, gl) pair.

    tau is NOT optimised — it is a search-time parameter that controls the
    spectral/cosine blend in search_batch(). Pass ``search_tau`` here to
    set the value used during the study; then pass the same value to
    aspace.search() / aspace.search_batch() at query time.

    Parameters
    ----------
    embeddings : np.ndarray
        Shape (N, D) float64 corpus embeddings.
    n_trials : int
        Number of Optuna trials. Default 15.
    sample_n : int | None
        Subsample size per trial. Default 5_000. None = full corpus.
    seed : int
        Random seed for reproducibility.
    study_name : str
        Optuna study identifier.
    storage : str | None
        Optuna storage URI for persistence. None = in-memory.
    eps_low, eps_high : float
        Log-scale search bounds for eps.
    k_low, k_high : int
        Search bounds for k.
    search_tau : float
        Fixed tau used inside the study's search_batch calls.
        Default 0.5 (balanced spectral + cosine blend).
        Not optimised — see EpsTuner docstring for rationale.
    n_probe : int
        Number of anchor queries per trial for the MRR proxy. Default 50.

    Returns
    -------
    aspace : ArrowSpace
        ArrowSpace index built with the best hyperparameters found.
    gl : GraphLaplacian
        Corresponding graph Laplacian.

    Examples
    --------
    Minimal usage:

        import numpy as np
        import arrowspace_tuner as arrowspace

        embeddings = np.load("corpus.npy")
        aspace, gl = arrowspace.optuna(embeddings)

        # tau for search is independent — pass what makes sense for your workload
        results = aspace.search(query_embedding, gl, tau=0.5)

    With a custom search_tau:

        aspace, gl = arrowspace.optuna(embeddings, search_tau=0.3)
        results = aspace.search(query_embedding, gl, tau=0.3)
    """
    tuner = EpsTuner(
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
    )
    return tuner.fit(embeddings)
