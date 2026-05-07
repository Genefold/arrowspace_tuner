"""
arrowspace_tuner — hyperparameter discovery for ArrowSpace.

Quickstart
----------
    import numpy as np
    import arrowspace_tuner as arrowspace

    embeddings = np.load("corpus.npy")

    # one-liner: auto-discover eps, k, tau
    aspace, gl = arrowspace.optuna(embeddings)

    # power-user: full control + post-run inspection
    from arrowspace_tuner import EpsTuner

    tuner = EpsTuner(n_trials=100, sample_n=10_000, eps_low=0.5, eps_high=3.0)
    aspace, gl = tuner.fit(embeddings)
    print(tuner.best_params)    # {"eps": 1.2, "k": 14, "tau": 0.8}
    print(tuner.best_score)
    tuner.save_report()         # requires pip install arrowspace-tuner[report]
"""
from importlib.metadata import PackageNotFoundError, version

from .api import optuna

# Power-user exports: config dataclasses for advanced customisation
from .core import BuildParams, StudyConfig
from .tuner import EpsTuner

try:
    __version__: str = version("arrowspace_tuner")
except PackageNotFoundError:
    __version__ = "0.0.0+unknown"

__all__ = [
    # primary public API
    "optuna",
    "EpsTuner",
    # config — for power users
    "BuildParams",
    "StudyConfig",
    # version
    "__version__",
]
