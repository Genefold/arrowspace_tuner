# MCP usage: arrowspace-tuner over stdio

The `arrowspace-tuner mcp` command starts a local stdio MCP server that
exposes the shared tuning service to LLM clients. It never shells out to
the CLI, reads only local `.npy`/`.npz` files under explicitly allowed
roots, and makes no network requests.

## Install

```bash
pip install "arrowspace-tuner[mcp]"
```

## Configure your client

`ARROWSPACE_TUNER_ALLOWED_ROOTS` is mandatory. Directories are separated
with the platform path separator (`:` on Unix, `;` on Windows).

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

## Tools

| Tool | Purpose |
|---|---|
| `inspect_embeddings` | Validate a local matrix; returns shape, dtype, norms, hash. |
| `tune_graph` | Optimize eps/k; returns `graph_params`, metrics, and selection metadata. |
| `build_instruction` | Turn graph_params into an ArrowSpace build snippet. |
| `get_tuner_info` | Server version, supported formats, security posture. |

## Example session

1. `inspect_embeddings(path="/workspace/corpus_embeddings.npy")` — confirms
   the file is a valid 2D matrix and reports any warnings.
2. `tune_graph(path="/workspace/corpus_embeddings.npy", n_trials=30)` —
   returns the `TuneResult` JSON.
3. `build_instruction(graph_params=<result["graph_params"]>)` — returns
   the exact Python build snippet. Tau is neither optimized nor returned.

For targeted low-k exploration, pass `selection_policy="lowest_k"`,
`refinement_trials=8`, and optionally `score_tolerance=0.01` to `tune_graph`.
Tolerance is an absolute difference in composite score and defaults to zero.

## Environment variables

| Variable | Default | Meaning |
|---|---:|---|
| `ARROWSPACE_TUNER_ALLOWED_ROOTS` | — (required) | `os.pathsep`-separated roots the server may read. |
| `ARROWSPACE_TUNER_MAX_INPUT_BYTES` | `2147483648` (2 GiB) | Reject larger files before loading. |
| `ARROWSPACE_TUNER_MAX_TRIALS` | `100` | Cap `n_trials + refinement_trials`. |
| `ARROWSPACE_TUNER_MAX_N_JOBS` | `4` | Reject `n_jobs` above this cap. |

## Security model

- Transport is stdio only; no HTTP, SSE, or WebSocket.
- Paths must be absolute, exist, resolve (symlinks included) inside an
  allowed root, and end in `.npy` or `.npz`.
- Remote URLs, pickle/object arrays, and over-limit files are rejected
  before any data is loaded.
- Embeddings are never uploaded or transmitted; reports are written only
  when explicitly requested, inside allowed roots, and never overwrite an
  existing report directory.
