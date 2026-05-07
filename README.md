# arrowspace_tuner

[![CI](https://github.com/Genefold/arrowspace_tuner/actions/workflows/ci.yml/badge.svg)](https://github.com/Genefold/arrowspace_tuner/actions/workflows/ci.yml)
[![PyPI](https://img.shields.io/pypi/v/arrowspace-tuner)](https://pypi.org/project/arrowspace-tuner/)
[![Python](https://img.shields.io/pypi/pyversions/arrowspace-tuner)](https://pypi.org/project/arrowspace-tuner/)
[![License](https://img.shields.io/badge/license-Apache--2.0-blue)](LICENSE)

Hyperparameter discovery for [ArrowSpace](https://github.com/tuned-org-uk/arrowspace-rs) — automatically finds the best `eps`, `k`, and `tau` for your corpus using a query-free spectral objective.

## Why

ArrowSpace's retrieval quality depends on three graph-construction parameters:

| Parameter | What it controls |
|---|---|
| `eps` | Neighbourhood radius for graph edges |
| `k` | Number of nearest neighbours per node |


Setting these by hand is tedious and corpus-dependent. `arrowspace_tuner` uses [Optuna](https://optuna.org/) and a label-free spectral MRR proxy to find them automatically in minutes.

## Install

```bash
# Core (no pandas/plotly)
pip install arrowspace-tuner

# With HTML/CSV reporting
pip install arrowspace-tuner[report]
```

## Quickstart

```python
import numpy as np
import arrowspace_tuner as arrowspace

embeddings = np.load("corpus.npy")   # shape (N, D) float64

# One-liner: auto-discover eps, k, tau — runs in ~15 min on 50k corpus
aspace, gl = arrowspace.optuna(embeddings)

# Search as normal
results = aspace.search(query_embedding, gl, tau=0.8)
```

## Power-user API

```python
from arrowspace_tuner import EpsTuner

tuner = EpsTuner(
    n_trials  = 15,
    sample_n  = 50_000,   
    eps_low   = 0.8,      
    eps_high  = 10,
    k_low     = 15,
    k_high    = 40,
    n_probe   = 50,
    storage   = "sqlite:///tune.db",   # resume interrupted runs
)

aspace, gl = tuner.fit(embeddings)

print(tuner.best_params)    # {"eps": 1.615, "k": 38, "tau": 0.114}
print(tuner.best_score)     # 2.138
print(tuner.best_fiedler)   # 0.718  — graph connectivity health
print(tuner.best_mrr_proxy) # 2.896  — retrieval coherence proxy

# Save CSV + HTML plots (requires [report] extra)
tuner.save_report(out_dir="results")
```


The final build after the study always uses the full corpus.

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
- `arrowspace >= 0.26.0`
- `optuna >= 4.8.0`
- `scipy >= 1.17.1`
- `numpy >= 2.4.4`

## License

Apache-2.0 — see [LICENSE](LICENSE).
