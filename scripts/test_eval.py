#!/usr/bin/env python3
"""
scripts/test_eval.py
====================
Run the arrowspace_tuner optimisation pipeline on the CVE .npy corpus.

    uv run python scripts/test_eval.py \
        --data  data/cve_embs/cve1999-2025.npy \
        --n     50000 \
        --trials 20 \
        --seed  42
"""
from __future__ import annotations

import argparse
import logging

import numpy as np

from arrowspace_tuner.tuner import EpsTuner

logging.basicConfig(
    level=logging.INFO,
    format="[%(asctime)s] %(levelname)s %(name)s | %(message)s",
)
log = logging.getLogger(__name__)


def load_npy(path: str, n: int, seed: int) -> np.ndarray:
    log.info("Loading %s …", path)
    X = np.load(path)
    log.info("  full shape : %s  dtype=%s", X.shape, X.dtype)
    n = min(n, len(X))
    rng = np.random.default_rng(seed)
    idx = rng.choice(len(X), size=n, replace=False)
    idx.sort()
    X = X[idx].astype(np.float64)
    norms = np.linalg.norm(X, axis=1, keepdims=True)
    X = X / np.clip(norms, 1e-12, None)
    log.info("  subsample  : %s  (L2-normalised)", X.shape)
    return X


def main() -> None:
    parser = argparse.ArgumentParser()
    parser.add_argument("--data",   default="data/cve_embs/cve1999-2025.npy")
    parser.add_argument("--n",      type=int, default=5000)
    parser.add_argument("--trials", type=int, default=20)
    parser.add_argument("--seed",   type=int, default=54)

    args = parser.parse_args()

    embeddings = load_npy(args.data, args.n, args.seed)

    tuner = EpsTuner(
        n_trials   = args.trials,
        sample_n   = None,          # already subsampled above
        seed       = args.seed,
        study_name = "cve_arrowspace_fstar",
        storage    = None,
    )

    log.info("Starting | n=%d  trials=%d  seed=%d", len(embeddings), args.trials, args.seed)

    aspace, gl = tuner.fit(embeddings)

    print("\n=== Best result ===")
    print(f"  F**        : {tuner.best_score:.8f}")
    print(f"  eps        : {tuner.best_params['eps']:.5f}")
    print(f"  k          : {tuner.best_params['k']}")
    print(f"  tau        : {tuner.best_params['tau']:.4f}")
    print(f"  fiedler    : {tuner.best_fiedler}")
    print(f"  var_lambda : {tuner.best_var_lambda}")
    print(f"  mrr_proxy  : {tuner.best_mrr_proxy}")
    print(f"\n  ArrowSpace : {aspace}")
    print(f"  Graph      : {gl}")

    tuner.save_report(out_dir="results")
    log.info("Report saved to results/")


if __name__ == "__main__":
    main()