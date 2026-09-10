# arrowspace_tuner

[![CI](https://github.com/Genefold/arrowspace_tuner/actions/workflows/ci.yml/badge.svg)](https://github.com/Genefold/arrowspace_tuner/actions/workflows/ci.yml)
[![PyPI](https://img.shields.io/pypi/v/arrowspace-tuner)](https://pypi.org/project/arrowspace-tuner/)
[![Python](https://img.shields.io/pypi/pyversions/arrowspace-tuner)](https://pypi.org/project/arrowspace-tuner/)
[![License](https://img.shields.io/badge/license-Apache--2.0-blue)](LICENSE)

Hyperparameter discovery for [ArrowSpace](https://github.com/tuned-org-uk/arrowspace-rs) — automatically finds the best `eps`, `k`, and `tau` for your corpus using a query-free spectral objective.

## Why

ArrowSpace's retrieval quality depends on three parameters:

| Parameter | What it controls |
|---|---|
| `eps` | Neighbourhood radius for graph edges |
| `k` | Number of nearest neighbours per node |
| `tau` | Search temperature (query-time, tuned automatically) |

Setting these by hand is tedious and corpus-dependent. `arrowspace_tuner` uses [Optuna](https://optuna.org/) and a label-free spectral MRR proxy to find them automatically in minutes.

## Install

```bash
# Core (no pandas/plotly)
pip install arrowspace-tuner

# With HTML/CSV reporting
pip install arrowspace-tuner[report]

# With the local stdio MCP server for LLM clients
pip install "arrowspace-tuner[mcp]"
```

## Quickstart

Executable versions of these snippets live in
[`examples/quickstart.py`](examples/quickstart.py) and
[`examples/power_user.py`](examples/power_user.py), and are run on every
CI build.

```python
import numpy as np
import arrowspace_tuner
from arrowspace import ArrowSpaceBuilder   # builder comes from `arrowspace`

embeddings = np.load("corpus.npy")   # shape (N, D) float64

# One-liner: auto-discover eps, k, tau — runs in ~15 min on 50k corpus
graph_params = arrowspace_tuner.tune(embeddings)

# The caller owns the build step
aspace, gl = ArrowSpaceBuilder().build(graph_params, embeddings)

# Search as normal — tau is a query-time parameter
results = aspace.search(query_embedding, gl, tau=0.8)
```

> [!WARNING]
> **Upgrading from v0.3.x?** `optuna()` is deprecated — use `tune()`.
> `load_best_params()` is deprecated — use `load_graph_params()`.
> `EpsTuner.fit()` now returns `dict` (graph_params), not `(aspace, gl)`.

### Build-time vs. search-time parameters

```text
graph_params:
  Build-time parameters only.
  Expected native ArrowSpace keys:
  eps, k, topk, p, sigma.

best_tau:
  Search-time parameter.
  It is intentionally excluded from graph_params.
```

Every public result dictionary — from `tune()`, `EpsTuner.fit()`,
`EpsTuner.graph_params`, and `load_graph_params()` — uses the bindings-native
`topk` key and can be passed verbatim to `ArrowSpaceBuilder().build()`.
`best_tau` is a separate search-time result: use it at query time as
`aspace.search(q, gl, tau=tuner.best_tau)`.

## Power-user API

Executable version: [`examples/power_user.py`](examples/power_user.py).

```python
from arrowspace import ArrowSpaceBuilder
from arrowspace_tuner import EpsTuner

tuner = EpsTuner(
    n_trials  = 15,
    seed      = 42,
    sample_n  = 50_000,
    eps_low   = 0.8,
    eps_high  = 10,
    k_low     = 15,
    k_high    = 40,
    n_probe   = 50,
    storage   = "sqlite:///tune.db",   # resume interrupted runs
)

graph_params = tuner.fit(embeddings)
best_tau = tuner.best_tau           # query-time only — not in graph_params

# The caller owns the build step
aspace, gl = ArrowSpaceBuilder().build(graph_params, embeddings)

print(graph_params)          # {"eps": 1.615, "k": 38, "topk": 19, "p": ..., "sigma": ...}
print(tuner.best_tau)        # 0.114  — query-time only, not in graph_params
print(tuner.best_score)      # 2.138
print(tuner.best_fiedler)    # 0.718  — graph connectivity health
print(tuner.best_mrr_proxy)  # 2.896  — retrieval coherence proxy

# Access graph params without file I/O
print(tuner.graph_params)    # same dict as best_params, raises RuntimeError before .fit()

# Save CSV + HTML plots (requires [report] extra)
tuner.save_report(out_dir="results")
```

The final build after the study always uses the full corpus.

## CLI

The `arrowspace-tuner` command (also available as `python -m arrowspace_tuner`)
runs the same tuning engine against local embedding files:

```bash
pip install "arrowspace-tuner[mcp]"

arrowspace-tuner tune corpus_embeddings.npy \
  --trials 15 \
  --format json \
  --output tuning_result.json
```

Validate the input first — invalid matrices fail before tuning with exit
code 3:

```bash
arrowspace-tuner validate corpus_embeddings.npy --format json
```

Other commands: `inspect RESULT.json` (re-read a written TuneResult),
`version`, and `mcp`. Run `arrowspace-tuner --help` for every option.

Exit codes: `0` ok · `2` CLI usage error · `3` invalid input · `4` tuning
failed (all trials pruned) · `5` output/report write failure · `6`
interrupted · `7` unexpected internal error.

Build with the result:

```python
import json
import numpy as np
from arrowspace import ArrowSpaceBuilder

embeddings = np.load("corpus_embeddings.npy")
result = json.load(open("tuning_result.json"))

aspace, gl = ArrowSpaceBuilder().build(
    result["graph_params"],
    embeddings,
)

hits = aspace.search(
    query_embedding,
    gl,
    tau=result["best_tau"],
)
```

## JSON output for automation

`--format json` prints exactly one JSON document on stdout; logs, warnings,
and progress go to stderr only, so the stream is safe to pipe. The schema
(`schema_version: "1.0"`) is stable:

```json
{
  "schema_version": "1.0",
  "status": "ok",
  "graph_params": {"eps": 1.615376, "k": 38, "topk": 19, "p": 2.0, "sigma": null},
  "best_tau": 0.8,
  "best_score": 2.138421,
  "n_trials_complete": 12,
  "n_trials_pruned": 3,
  "elapsed_seconds": 84.27,
  "input_info": {"path": "corpus_embeddings.npy", "sha256": "9d4b…"},
  "error_code": null,
  "error_message": null
}
```

Guarantees: `graph_params` uses the native `topk` key (never `top_k`) and
never contains `tau`; `best_tau` is a top-level search-time key; failures
also print valid JSON (`status: "validation_error" | "tuning_error"`, with
`error_code` and `error_message`) and exit non-zero. `--output` writes the
same document atomically (temp file + fsync + atomic replace).

## MCP server

`arrowspace-tuner mcp` starts a local stdio MCP server for LLM clients
(requires the `mcp` extra). It calls the same shared service as the CLI —
never a subprocess — and exposes exactly four tools:
`inspect_embeddings`, `tune_graph`, `build_instruction`, and
`get_tuner_info`.

```json
{
  "mcpServers": {
    "arrowspace-tuner": {
      "command": "arrowspace-tuner",
      "args": ["mcp"],
      "env": {
        "ARROWSPACE_TUNER_ALLOWED_ROOTS": "/workspace:/data"
      }
    }
  }
}
```

See [`examples/mcp_usage.md`](examples/mcp_usage.md) and
[`examples/mcp_config.json`](examples/mcp_config.json).

## Security model for MCP

- The server runs locally over stdio; it makes no network requests and
  accepts no remote URLs.
- Embeddings are read only from `.npy`/`.npz` files under explicitly
  configured roots (`ARROWSPACE_TUNER_ALLOWED_ROOTS`, mandatory — the
  server refuses to start without it). Symlinks escaping a root, relative
  paths, pickle/object arrays, and files above
  `ARROWSPACE_TUNER_MAX_INPUT_BYTES` (default 2 GiB) are rejected.
- `ARROWSPACE_TUNER_MAX_TRIALS` (default 100) and
  `ARROWSPACE_TUNER_MAX_N_JOBS` (default 4) cap tuning requests before
  Optuna starts.
- No embedding data is uploaded; reports are written only when explicitly
  requested, beneath an allowed root, and never overwrite an existing
  report directory.

## Objective

The objective is a weighted composite of three spectral signals — no ground-truth labels required:

```
score = 0.70 * mrr_top0_spectral   # retrieval coherence
      + 0.20 * log1p(fiedler)      # graph connectivity health
      + 0.10 * log1p(var_lambda)   # spectral richness
```

## Parallel runs

Optuna + SQLite lets you run multiple workers simultaneously:

```bash
# Terminal 1
python -m arrowspace_tuner --storage sqlite:///tune.db --trials 15

# Terminal 2 (simultaneously)
python -m arrowspace_tuner --storage sqlite:///tune.db --trials 15
```

## Requirements

- Python ≥ 3.12
- `arrowspace >= 0.26.0, < 0.29` — tested with 0.26.0, 0.27.3, and 0.28.1
- `optuna >= 4.8.0`
- `scipy >= 1.17.1`
- `numpy >= 2.4.4`

## License

Apache-2.0 — see [LICENSE](LICENSE).
