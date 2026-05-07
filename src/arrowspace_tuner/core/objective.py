"""
objective_v2.py — query-free retrieval-quality objective for ArrowSpace.

Proxy: fully-spectral MRR-Top0 (paper §2.8).
Every corpus item acts as its own query anchor.
Rel(q) = k-NN neighbourhood in the graph (label-agnostic definition).
Topology factor T_qi = exp(-|λ_q - λ_i| / σ_λ)  — no PPR, no adjacency needed.

This is strictly query-free: only the graph Laplacian, λ values, and
the k-NN structure built by ArrowSpace are required.

Objective (maximise):
    score = W_MRR  * mrr_top0_spectral   # retrieval coherence
          + W_FIED * log1p(fiedler)      # connectivity health
          + W_VAR  * log1p(var_lambda)   # spectral richness

Hyperparameters optimised by Optuna: eps, k, tau

Pruning checkpoints (MedianPruner in tuner.py):
    step=0  after Fiedler   — cheap proxy for connectivity
    step=1  after var_lambda — proxy for spectral richness
    Both are BEFORE the expensive search_batch + MRR path.

Best-trial cache (#9):
    When sample_n is None (full corpus path), make_objective returns a
    best_cache dict that is updated in-place whenever a trial beats the
    current best. EpsTuner reads this after study.optimize() and skips
    the otherwise-redundant _final_build call.
"""
from __future__ import annotations

import logging
import threading
from collections.abc import Sequence
from typing import Any, Protocol

import numpy as np
import optuna
import scipy.sparse as sp

from .config import BuildParams, StudyConfig
from .graph import PyGraphLaplacian, fiedler_normalized_from_csr

logger = logging.getLogger(__name__)

# ── objective weights ─────────────────────────────────────────────────────────
W_MRR  = 0.70
W_FIED = 0.20
W_VAR  = 0.10

K_EVAL = 10   # top-k cutoff for MRR-Top0


class ArrowSpaceProtocol(Protocol):
    """
    Structural protocol for the ArrowSpace object returned by
    ArrowSpaceBuilder.build() (Rust FFI type).

    Only the methods used inside this module are declared so that mypy
    can type-check call-sites without importing the extension wheel.
    """

    def search_batch(
        self,
        queries: np.ndarray,
        gl: PyGraphLaplacian,
        tau: float,
    ) -> Sequence[Sequence[tuple[int, float]]]:
        """Run a batch of queries against the index."""
        ...

    def lambdas(self) -> Sequence[float]:
        """Return taumode λ values for all indexed items."""
        ...


# ── graph build + spectral diagnostics ───────────────────────────────────────

def build_and_score(
    embeddings: np.ndarray,
    params:     BuildParams,
    trial:      optuna.Trial | None = None,
) -> tuple[float, float, ArrowSpaceProtocol | None, PyGraphLaplacian | None]:
    """
    Build an ArrowSpace graph and compute spectral diagnostics.

    Parameters
    ----------
    embeddings : np.ndarray
        Shape (N, D) float64 corpus embeddings for this trial.
    params : BuildParams
        Graph construction parameters for this trial.
    trial : optuna.Trial | None
        If provided, degenerate graphs raise TrialPruned instead of
        returning zeros. Also used for pruner checkpoints.

    Returns
    -------
    fiedler : float
        Normalised Fiedler value λ₂ — algebraic connectivity.
    var_lambda : float
        Variance of ArrowSpace taumode λ values — spectral richness.
    aspace : ArrowSpaceProtocol | None
        Live ArrowSpace object, needed for search_batch downstream.
        None on degenerate graphs.
    gl : PyGraphLaplacian | None
        Live GraphLaplacian object. None on degenerate graphs.
    """
    from arrowspace import ArrowSpaceBuilder

    # ── build graph ───────────────────────────────────────────────────────────
    try:
        aspace: ArrowSpaceProtocol
        gl: PyGraphLaplacian

        builder = (
            ArrowSpaceBuilder()
            .with_dims_reduction(enabled=False, eps=None)
            .with_sampling("simple", params.sampling_rate)
        )

        if params.max_clusters is not None:
            builder = builder.with_cluster_max_clusters(params.max_clusters)

        if params.cluster_radius is not None:
            builder = builder.with_cluster_radius(params.cluster_radius)

        aspace, gl = builder.build(params.to_dict(), embeddings)

    except (KeyboardInterrupt, SystemExit):
        raise
    except BaseException as exc:
        logger.warning(
            "ArrowSpace .build() failed (eps=%.4f k=%d): %s — pruning",
            params.eps, params.k, exc,
        )
        if trial:
            raise optuna.TrialPruned()
        return 0.0, 0.0, None, None

    # ── pre-flight: check graph shape ─────────────────────────────────────────
    shape = gl.shape()
    if shape[1] < 2:
        logger.warning(
            "Degenerate graph shape=%s (single cluster) | eps=%.4f", shape, params.eps
        )
        if trial:
            raise optuna.TrialPruned()
        return 0.0, 0.0, None, None

    # ── single FFI call: materialise CSR once (#10) ───────────────────────────
    raw     = gl.to_csr()
    data    = np.asarray(raw[0], dtype=np.float64)
    indices = np.asarray(raw[1], dtype=np.int32)
    indptr  = np.asarray(raw[2], dtype=np.int32)
    n       = shape[1]
    nnz     = len(data)
    L       = sp.csr_matrix((data, indices, indptr), shape=(n, n))

    # ── degenerate guard 1: nearly empty graph ────────────────────────────────
    if nnz <= n:
        logger.warning(
            "Degenerate graph NNZ=%d <= N=%d | eps=%.4f", nnz, n, params.eps
        )
        if trial:
            raise optuna.TrialPruned()
        return 0.0, 0.0, None, None

    fiedler    = fiedler_normalized_from_csr(L, nnz)
    lambdas    = np.array(aspace.lambdas(), dtype=np.float64)
    spread     = float(lambdas.max() - lambdas.min()) if len(lambdas) > 1 else 0.0
    var_lambda = float(np.var(lambdas))               if len(lambdas) > 1 else 0.0

    # ── degenerate guard 2: disconnected graph ────────────────────────────────
    if fiedler <= 1e-6:
        logger.warning(
            "Disconnected graph fiedler=%.2e — pruning", fiedler
        )
        if trial:
            raise optuna.TrialPruned()
        return 0.0, 0.0, None, None

    # ── pruner checkpoint 0: after Fiedler ────────────────────────────────────
    if trial is not None:
        trial.report(fiedler, step=0)
        if trial.should_prune():
            raise optuna.TrialPruned()

    # ── degenerate guard 3: flat spectrum ─────────────────────────────────────
    if spread < 1e-10:
        logger.warning(
            "Flat spectrum spread=%.2e — pruning", spread
        )
        if trial:
            raise optuna.TrialPruned()
        return fiedler, 0.0, None, None

    # ── pruner checkpoint 1: after var_lambda ─────────────────────────────────
    if trial is not None:
        trial.report(fiedler + 0.5 * var_lambda, step=1)
        if trial.should_prune():
            raise optuna.TrialPruned()

    return fiedler, var_lambda, aspace, gl


# ── main factory ──────────────────────────────────────────────────────────────

def make_objective(
    embeddings: np.ndarray,
    cfg:        StudyConfig,
) -> tuple[object, dict[str, Any]]:
    """
    Return ``(objective_fn, best_cache)`` closed over embeddings and cfg.

    ``objective_fn`` is passed directly to ``study.optimize()``.
    It searches over eps, k, and tau simultaneously.

    ``best_cache`` is a dict updated in-place whenever a trial beats the
    current best score::

        {"aspace": <ArrowSpace>, "gl": <GraphLaplacian>, "score": float}

    When ``cfg.sample_n`` is ``None`` (full corpus path) the cached objects
    were built on the full corpus and can be returned directly by
    ``EpsTuner._final_build``, saving one redundant ``.build()`` call.
    When ``cfg.sample_n`` is set the cache is never populated (the trial
    objects were built on a subsample) so ``_final_build`` runs as normal.

    The corpus subsample and probe anchor indices are drawn ONCE here,
    before the inner closure is created, so every trial evaluates on the
    same fixed slice of the corpus. This gives TPE a deterministic,
    stationary response surface to model — a prerequisite for the
    surrogate to be meaningful.

    Parameters
    ----------
    embeddings : np.ndarray
        Full corpus embeddings shape (N, D).
    cfg : StudyConfig
        Study configuration including search bounds and MRR probe settings.

    Returns
    -------
    objective_fn : Callable[[optuna.Trial], float]
        The Optuna objective function.
    best_cache : dict
        Mutable cache populated with the best trial's objects (see above).
    """
    # ── draw fixed subsample ONCE ─────────────────────────────────────────────
    using_subsample = bool(cfg.sample_n and cfg.sample_n < len(embeddings))
    if using_subsample:
        rng       = np.random.default_rng(cfg.seed)
        idx       = rng.choice(len(embeddings), size=cfg.sample_n, replace=False)
        emb_fixed = embeddings[idx]
    else:
        emb_fixed = embeddings

    # ── draw fixed probe indices ONCE ─────────────────────────────────────────
    n_fixed         = len(emb_fixed)
    rng_probe       = np.random.default_rng(cfg.seed + 42)
    probe_idx_fixed = rng_probe.choice(
        n_fixed, size=min(cfg.n_probe, n_fixed), replace=False
    )

    # ── best-trial cache (#9) ─────────────────────────────────────────────────
    # Only populated when using the full corpus (not a subsample), because
    # subsample-built objects cannot be returned as the final (aspace, gl).
    best_cache: dict[str, Any] = {}  # keys: "aspace", "gl", "score" when populated
    _cache_lock = threading.Lock()   # protect concurrent updates (n_jobs > 1)

    def objective(trial: optuna.Trial) -> float:

        # ── 1. fixed corpus slice ─────────────────────────────────────────────
        emb_trial = emb_fixed

        # ── 2. suggest hyperparameters ────────────────────────────────────────
        k   = trial.suggest_int(  "k",   cfg.k_low,   cfg.k_high)
        eps = trial.suggest_float("eps", cfg.eps_low,  cfg.eps_high, log=True)
        tau = trial.suggest_float("tau", cfg.tau_low,  cfg.tau_high)

        params = BuildParams(
            eps=eps,
            k=k,
            topk=max(1, k // 2),
        )

        # ── 3. build graph + spectral diagnostics (includes pruner steps 0,1) ──
        try:
            fiedler, var_lambda, aspace, gl = build_and_score(
                emb_trial, params, trial
            )
        except optuna.TrialPruned:
            raise
        except (KeyboardInterrupt, SystemExit):
            raise
        except Exception as exc:
            logger.warning(
                "Trial %d unexpected error: %s", trial.number, exc, exc_info=True
            )
            return 0.0

        # ── 4. fixed probe anchors ────────────────────────────────────────────
        probe_idx  = probe_idx_fixed
        probe_embs = np.ascontiguousarray(
            emb_trial[probe_idx], dtype=np.float64
        )

        # ── 5. k-NN retrieval via search_batch ────────────────────────────────
        if aspace is None or gl is None:
            raise optuna.TrialPruned()
        try:
            batch_results = aspace.search_batch(probe_embs, gl, tau)
        except Exception as exc:
            logger.warning(
                "Trial %d search_batch failed (eps=%.4f k=%d tau=%.3f): %s",
                trial.number, params.eps, params.k, tau, exc,
            )
            raise optuna.TrialPruned()

        # ── 6. build k-NN index table ─────────────────────────────────────────
        P = len(probe_idx)
        knn_indices = np.zeros((P, K_EVAL), dtype=np.int64)
        row_widths  = np.zeros(P, dtype=np.int64)
        for row, results in enumerate(batch_results):
            hits = results[:K_EVAL]
            w    = len(hits)
            row_widths[row] = w
            for col, (idx_item, _) in enumerate(hits):
                knn_indices[row, col] = idx_item

        # ── guard: prune if search returned no results for all probes ──────────
        # Fixes issues #16 and #22: when n_probe approaches corpus_size or
        # the graph is nearly disconnected, search_batch may return empty
        # result lists for every probe, making the MRR computation undefined.
        if row_widths.sum() == 0:
            logger.warning(
                "Trial %d: search_batch returned no results for any probe — pruning",
                trial.number,
            )
            raise optuna.TrialPruned()

        # ── 7. spectral MRR-Top0 proxy (vectorised) ───────────────────────────
        lambdas      = np.array(aspace.lambdas(), dtype=np.float64)
        sigma        = float(np.std(lambdas)) + 1e-9
        lambda_probe = lambdas[probe_idx]

        l_q    = lambda_probe[:, None]
        l_nbrs = lambdas[knn_indices]
        T      = np.exp(-np.abs(l_q - l_nbrs) / sigma)

        col_idx   = np.arange(K_EVAL)
        mask      = col_idx[None, :] < row_widths[:, None]
        inv_rk    = 1.0 / np.arange(1, K_EVAL + 1)
        mrr_proxy = float(((T * inv_rk) * mask).sum(axis=1).mean())

        # ── 8. composite objective ────────────────────────────────────────────
        score = (
            W_MRR  * mrr_proxy
          + W_FIED * float(np.log1p(fiedler))
          + W_VAR  * float(np.log1p(var_lambda))
        )

        # ── 9. update best-trial cache (full-corpus path only) ────────────────
        if not using_subsample and aspace is not None:
            with _cache_lock:
                if score > best_cache.get("score", -1.0):
                    best_cache["aspace"] = aspace
                    best_cache["gl"]     = gl
                    best_cache["score"]  = score

        # ── 10. log trial attributes ──────────────────────────────────────────
        trial.set_user_attr("fiedler",    round(fiedler,    8))
        trial.set_user_attr("var_lambda", round(var_lambda, 8))
        trial.set_user_attr("mrr_proxy",  round(mrr_proxy,  6))
        trial.set_user_attr("tau",        round(tau,        6))
        trial.set_user_attr("n_sample",   len(emb_trial))
        trial.set_user_attr("n_probe",    len(probe_idx))

        logger.info(
            "Trial %03d | eps=%.5f k=%2d topk=%2d tau=%.3f | "
            "fiedler=%.4f var=%.4f mrr=%.4f → score=%.6f",
            trial.number, params.eps, params.k, params.topk, tau,
            fiedler, var_lambda, mrr_proxy, score,
        )
        return score

    return objective, best_cache
