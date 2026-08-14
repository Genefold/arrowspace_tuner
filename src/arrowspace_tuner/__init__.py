"""
arrowspace_tuner — hyperparameter discovery for ArrowSpace.

Quickstart
----------
    import numpy as np
    import arrowspace_tuner as at
    from arrowspace import ArrowSpaceBuilder

    embeddings = np.load("corpus.npy")

    # one-liner: auto-discover eps, k, tau
    graph_params = at.tune(embeddings)
    aspace, gl = ArrowSpaceBuilder().build(graph_params, embeddings)

    # power-user: full control + post-run inspection
    from arrowspace_tuner import EpsTuner

    tuner = EpsTuner(n_trials=100, sample_n=10_000, eps_low=0.5, eps_high=3.0)
    graph_params = tuner.fit(embeddings)
    print(tuner.best_params)    # {"eps": 1.2, "k": 14, "top_k": 7, "p": 2.0, "sigma": None}
    print(tuner.best_tau)       # 0.8  (query-time — use at search time)
    print(tuner.best_score)
    tuner.save_report()         # requires pip install arrowspace-tuner[report]
"""
from importlib.metadata import PackageNotFoundError, version

from .api import optuna, tune
from .core import BuildParams, StudyConfig
from .tuner import EpsTuner

try:
    __version__: str = version("arrowspace_tuner")
except PackageNotFoundError:
    __version__ = "0.0.0+unknown"

__all__ = [
    # primary public API
    "tune",
    "EpsTuner",
    # deprecated — remove in next minor bump
    "optuna",
    # config — for power users
    "BuildParams",
    "StudyConfig",
    # version
    "__version__",
]
