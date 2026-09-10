#!/usr/bin/env bash
# cli_validate.sh — validate inputs before tuning (exit 3 on invalid matrices).
set -euo pipefail
cd "$(dirname "$0")"

python - <<'PY'
import numpy as np
rng = np.random.default_rng(42)
arr = rng.normal(size=(120, 32))
arr /= np.linalg.norm(arr, axis=1, keepdims=True)
np.save("demo_embeddings.npy", arr)

# a deliberately invalid matrix
bad = np.array([[1.0, np.nan], [2.0, 3.0]])
np.save("bad_embeddings.npy", bad)
PY

echo "--- valid input (exit 0) ---"
arrowspace-tuner validate demo_embeddings.npy --format json

echo "--- invalid input (exit 3) ---"
set +e
arrowspace-tuner validate bad_embeddings.npy --format json
echo "exit code: $?"
set -e