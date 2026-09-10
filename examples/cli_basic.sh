#!/usr/bin/env bash
# cli_basic.sh — human-facing tuning workflow (text output).
set -euo pipefail
cd "$(dirname "$0")"

python - <<'PY'
import numpy as np
rng = np.random.default_rng(42)
arr = rng.normal(size=(120, 32))
arr /= np.linalg.norm(arr, axis=1, keepdims=True)
np.save("demo_embeddings.npy", arr)
PY

arrowspace-tuner validate demo_embeddings.npy

arrowspace-tuner tune demo_embeddings.npy \
  --trials 3 \
  --seed 42 \
  --k-low 3 \
  --k-high 10 \
  --n-probe 20

echo
echo "graph_params feed ArrowSpaceBuilder().build(); best_tau is used at search time:"
echo "  aspace.search(query, gl, tau=best_tau)"