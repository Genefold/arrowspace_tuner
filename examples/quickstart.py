"""
arrowspace_tuner quickstart — executable mirror of the README Quickstart.

Run with:
    python examples/quickstart.py
"""

import numpy as np
from arrowspace import ArrowSpaceBuilder  # builder comes from `arrowspace`

import arrowspace_tuner

# Synthetic stand-in for np.load("corpus.npy"): (N, D) float64 embeddings
# with mild cluster structure.
rng = np.random.default_rng(42)
centres = rng.standard_normal((4, 64))
corpus = np.vstack([centres[i] + 0.4 * rng.standard_normal((30, 64)) for i in range(4)])
embeddings = corpus / np.clip(np.linalg.norm(corpus, axis=1, keepdims=True), 1e-9, None)

# One-liner: auto-discover eps, k, tau.
# graph_params is build-time only: {"eps", "k", "topk", "p", "sigma"}.
graph_params = arrowspace_tuner.tune(embeddings)

# The caller owns the build step.
aspace, gl = ArrowSpaceBuilder().build(graph_params, embeddings)

# Search as normal — tau is a query-time parameter.
query = embeddings[0]
results = aspace.search(query, gl, tau=0.8)

print("graph_params:", graph_params)
print("top hit:", results[0] if results else "no hits")
