"""
arrowspace_tuner.core — internal building blocks.

This subpackage is not part of the public API.
Import from arrowspace_tuner directly:

    from arrowspace_tuner import EpsTuner, optuna
    from arrowspace_tuner import StudyConfig, BuildParams  # for power users
"""
from .config import BuildParams, StudyConfig
from .graph import fiedler_normalized, gl_to_scipy
from .objective import build_and_score, make_objective

__version__ = "0.1.0"

__all__ = [
    # config
    "BuildParams",
    "StudyConfig",
    # graph
    "fiedler_normalized",
    "gl_to_scipy",
    # objective
    "build_and_score",
    "make_objective",
]