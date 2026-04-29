"""
conftest.py — shared pytest fixtures for arrowspace_tuner tests.

All fixtures are purely synthetic — no disk reads, no real embeddings,
no arrowspace Rust wheel required for the fixtures themselves.
"""
from __future__ import annotations

import numpy as np
import pytest


# ── constants ─────────────────────────────────────────────────────────────────

N_SMALL  = 120    # fast: enough for a non-degenerate graph
N_MEDIUM = 600    # realistic: closer to a real corpus sample
D        = 64     # embedding dimension
SEED     = 42


# ── embedding fixtures ────────────────────────────────────────────────────────

@pytest.fixture(scope="session")
def rng() -> np.random.Generator:
    """Seeded RNG reused across all fixtures."""
    return np.random.default_rng(SEED)


@pytest.fixture(scope="session")
def embeddings_small(rng) -> np.ndarray:
    """
    120 × 64 float64 embeddings with mild cluster structure.

    Four loose clusters of 30 items each. Small enough for fast tests,
    structured enough to produce a non-degenerate graph at reasonable eps.
    """
    centres = rng.standard_normal((4, D))
    chunks  = [
        centres[i] + 0.4 * rng.standard_normal((30, D))
        for i in range(4)
    ]
    arr = np.vstack(chunks).astype(np.float64)
    # L2-normalise so eps bounds are dataset-agnostic
    norms = np.linalg.norm(arr, axis=1, keepdims=True)
    return arr / np.clip(norms, 1e-9, None)


@pytest.fixture(scope="session")
def embeddings_medium(rng) -> np.ndarray:
    """
    600 × 64 float64 embeddings with six clusters of 100 items.

    Used for integration tests where sample_n subsampling matters.
    """
    centres = rng.standard_normal((6, D))
    chunks  = [
        centres[i] + 0.35 * rng.standard_normal((100, D))
        for i in range(6)
    ]
    arr = np.vstack(chunks).astype(np.float64)
    norms = np.linalg.norm(arr, axis=1, keepdims=True)
    return arr / np.clip(norms, 1e-9, None)


@pytest.fixture(scope="session")
def embeddings_flat(rng) -> np.ndarray:
    """
    80 × 64 float64 embeddings drawn from a single tight Gaussian near origin.

    Intentionally degenerate: very low spectral variance.
    Used to test pruning paths in objective.py.

    NOTE: do NOT L2-normalise — normalisation projects every vector onto
    the unit sphere, destroying the "flat" property entirely. The * 0.01
    scale is what makes all vectors nearly identical and thus produces a
    degenerate graph when eps is small.
    """
    return rng.standard_normal((80, D)).astype(np.float64) * 0.01


@pytest.fixture(scope="session")
def embeddings_wrong_dtype(rng) -> np.ndarray:
    """float32 array — used to test dtype validation in EpsTuner."""
    return rng.standard_normal((50, D)).astype(np.float32)


@pytest.fixture(scope="session")
def embeddings_1d(rng) -> np.ndarray:
    """1D array — used to test shape validation in EpsTuner."""
    return rng.standard_normal(D).astype(np.float64)


# ── StudyConfig fixture ───────────────────────────────────────────────────────

@pytest.fixture
def fast_study_config():
    """
    Minimal StudyConfig for fast unit tests.

    3 trials, small probe set, tight search bounds centred on a region
    known to produce valid graphs for the clustered fixtures above.
    """
    from arrowspace_tuner import StudyConfig
    return StudyConfig(
        n_trials   = 3,
        sample_n   = None,
        seed       = SEED,
        study_name = "test_study",
        eps_low    = 0.5,
        eps_high   = 2.0,
        k_low      = 3,
        k_high     = 10,
        tau_low    = 0.3,
        tau_high   = 1.2,
        n_probe    = 20,
    )
