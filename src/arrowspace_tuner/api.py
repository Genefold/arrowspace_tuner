"""
api.py — one-liner convenience function for hyperparameter discovery.

For any non-trivial use case, instantiate EpsTuner directly.
"""
from __future__ import annotations

import warnings
from typing import Any

import numpy as np

from .tuner import EpsTuner


def tune(
    embeddings: np.ndarray,
    *,
    tuner: EpsTuner | None = None,
    **kwargs: Any,  # noqa: ANN401
) -> dict[str, Any]:
    """
    Auto-tune ArrowSpace hyperparameters for a given embedding corpus.

    The simplest entry point to arrowspace_tuner. Runs an Optuna study
    and returns optimised graph parameters ready for ArrowSpaceBuilder.build().

    Parameters
    ----------
    embeddings : np.ndarray
        Shape (N, D) float64 corpus embeddings.
    tuner : EpsTuner | None
        A pre-configured EpsTuner instance. If None, a default EpsTuner
        is created from any extra kwargs provided.
    **kwargs
        Forwarded to EpsTuner(**kwargs) when tuner is None.
        All EpsTuner constructor params are supported: n_trials, sample_n,
        seed, study_name, storage, eps_low, eps_high, k_low, k_high,
        tau_low, tau_high, n_probe, n_jobs, max_clusters, cluster_radius.

    Returns
    -------
    dict[str, Any]
        Optimised graph parameters:
        {"eps": float, "k": int, "topk": int, "p": float, "sigma": None}.
        Ready to be passed to ArrowSpaceBuilder.build().

    Notes
    -----
    tau is not included in the returned dict — it is a query-time parameter.
    To access best_tau, pass a pre-configured EpsTuner and inspect it after:

        tuner = EpsTuner(n_trials=30)
        graph_params = tune(embeddings, tuner=tuner)
        print(tuner.best_tau)

    Examples
    --------
    Minimal usage::

        import arrowspace_tuner as at
        from arrowspace import ArrowSpaceBuilder

        graph_params = at.tune(embeddings)
        aspace, gl = ArrowSpaceBuilder().build(graph_params, embeddings)

    With custom bounds::

        graph_params = at.tune(embeddings, n_trials=30, eps_low=0.5, eps_high=3.0)

    Parallel search::

        graph_params = at.tune(embeddings, n_trials=50, n_jobs=-1)

    With a pre-configured tuner (for introspection)::

        tuner = EpsTuner(n_trials=30, n_jobs=-1)
        graph_params = at.tune(embeddings, tuner=tuner)
        print(tuner.best_tau)   # optimal search temperature
        tuner.save_report()     # persist trial data

    Resuming an interrupted run::

        graph_params = at.tune(embeddings, storage="sqlite:///tune.db")
    """
    t = tuner if tuner is not None else EpsTuner(**kwargs)
    return t.fit(embeddings)


def optuna(
    embeddings: np.ndarray,
    **kwargs: Any,  # noqa: ANN401
) -> dict[str, Any]:
    """
    Deprecated. Use tune() instead.

    .. deprecated:: 0.4.0
        optuna() is deprecated and will be removed in a future release.
        Use arrowspace_tuner.tune() instead.
    """
    warnings.warn(
        "arrowspace_tuner.optuna() is deprecated, use tune() instead.",
        DeprecationWarning,
        stacklevel=2,
    )
    return tune(embeddings, **kwargs)
