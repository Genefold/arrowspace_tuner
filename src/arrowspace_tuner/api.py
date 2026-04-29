"""
api.py — one-liner convenience function for hyperparameter discovery.

This module exists solely to satisfy the acceptance criteria:

    aspace, gl = arrowspace.optuna(embeddings)

It is a thin shim over EpsTuner with sensible defaults.
For any non-trivial use case, instantiate EpsTuner directly.
"""
from __future__ import annotations

import numpy as np

from .tuner import EpsTuner


def optuna(
    embeddings: np.ndarray,
    *,
    n_trials:   int        = 15,
    sample_n:   int | None = None,
    seed:       int        = 54,
    study_name: str        = "arrowspace_fstar",
    storage:    str | None = None,
    eps_low:    float      = 0.3,
    eps_high:   float      = 4.0,
    k_low:      int        = 3,
    k_high:     int        = 40,
    tau_low:    float      = 0.1,
    tau_high:   float      = 1.0,
    n_probe:    int        = 2000,
) -> tuple[object, object]:
    """
    Auto-discover eps, k, and tau and return a ready-to-use (aspace, gl) pair.

    This is the simplest entry point to arrowspace_tuner. It runs an Optuna
    study with default settings and returns the ArrowSpace index built with
    the best hyperparameters found.

    Parameters
    ----------
    embeddings : np.ndarray
        Shape (N, D) float64 corpus embeddings.
    n_trials : int
        Number of Optuna trials. Default 50.
    sample_n : int | None
        Subsample size per trial. None = full corpus. Recommended for
        large corpora (> 50k items).
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
    tau_low, tau_high : float
        Search bounds for tau.
    n_probe : int
        Number of anchor queries per trial for the MRR proxy.

    Returns
    -------
    aspace : ArrowSpace
        ArrowSpace index built with the best hyperparameters found.
    gl : GraphLaplacian
        Corresponding graph Laplacian.

    Examples
    --------
    Minimal usage — matches the acceptance criteria exactly:

        import numpy as np
        import arrowspace_tuner as arrowspace

        embeddings = np.load("corpus.npy")
        aspace, gl = arrowspace.optuna(embeddings)

        results = aspace.search(query_embedding, gl, tau=0.8)

    With a custom search range:

        aspace, gl = arrowspace.optuna(
            embeddings,
            n_trials=100,
            sample_n=10_000,
            eps_low=0.5,
            eps_high=3.0,
        )

    Inspecting the study after the fact:

        from arrowspace_tuner import EpsTuner

        tuner = EpsTuner(n_trials=100, sample_n=10_000)
        aspace, gl = tuner.fit(embeddings)
        print(tuner.best_params)
        tuner.save_report()
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
        tau_low    = tau_low,
        tau_high   = tau_high,
        n_probe    = n_probe,
    )
    return tuner.fit(embeddings)