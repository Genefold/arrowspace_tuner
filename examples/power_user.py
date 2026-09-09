"""
arrowspace_tuner power-user flow — executable mirror of the README
Power-user API section.

Run with:
    python examples/power_user.py
"""

import numpy as np
from arrowspace import ArrowSpaceBuilder

from arrowspace_tuner import EpsTuner

# Synthetic stand-in for np.load("corpus.npy"): (N, D) float64 embeddings.
rng = np.random.default_rng(42)
centres = rng.standard_normal((4, 64))
corpus = np.vstack([centres[i] + 0.4 * rng.standard_normal((30, 64)) for i in range(4)])
embeddings = corpus / np.clip(np.linalg.norm(corpus, axis=1, keepdims=True), 1e-9, None)

tuner = EpsTuner(
    n_trials=15,
    seed=42,
)

graph_params = tuner.fit(embeddings)
best_tau = tuner.best_tau  # query-time only — not in graph_params

# The caller owns the build step.
aspace, gl = ArrowSpaceBuilder().build(graph_params, embeddings)

print("graph_params:", graph_params)
print("best_tau:    ", best_tau)
print("best_score:  ", tuner.best_score)
print("best_fiedler:", tuner.best_fiedler)

# graph_params is also accessible without file I/O after .fit()
assert tuner.graph_params == graph_params

# Search with the tuned tau.
hits = aspace.search(embeddings[0], gl, tau=best_tau)
print("top hit:", hits[0] if hits else "no hits")
