#!/usr/bin/env bash
# cli_json.sh — machine-readable tuning result (one JSON document on stdout).
set -euo pipefail
cd "$(dirname "$0")"

python - <<'PY'
import numpy as np
rng = np.random.default_rng(42)
arr = rng.normal(size=(120, 32))
arr /= np.linalg.norm(arr, axis=1, keepdims=True)
np.save("demo_embeddings.npy", arr)
PY

arrowspace-tuner tune demo_embeddings.npy \
  --trials 3 \
  --seed 42 \
  --k-low 3 \
  --k-high 10 \
  --n-probe 20 \
  --format json \
  --output demo_result.json

echo "result written to demo_result.json (stderr carries logs only):" >&2
python3 -c "
import json
result = json.load(open('demo_result.json'))
print(json.dumps({
    'status': result['status'],
    'graph_params': result['graph_params'],
    'best_tau': result['best_tau'],
}, indent=2))
"