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
"""
from __future__ import annotations

import logging

import numpy as np
import optuna

from .config import BuildParams, StudyConfig
from .graph import fiedler_normalized

logger = logging.getLogger(__name__)

# ── objective weights ─────────────────────────────────────────────────────────
W_MRR  = 0.70
W_FIED = 0.20
W_VAR  = 0.10

K_EVAL = 10   # top-k cutoff for MRR-Top0


# ── graph build + spectral diagnostics ───────────────────────────────────────

def build_and_score(
    embeddings: np.ndarray,
    params:     BuildParams,
    trial:      optuna.Trial | None = None,
) -> tuple[float, float, object, object]:
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
        returning zeros.

    Returns
    -------
    fiedler : float
        Normalised Fiedler value λ₂ — algebraic connectivity.
    var_lambda : float
        Variance of ArrowSpace taumode λ values — spectral richness.
    aspace : ArrowSpace | None
        Live ArrowSpace object, needed for search_batch downstream.
        None on degenerate graphs.
    gl : GraphLaplacian | None
        Live GraphLaplacian object. None on degenerate graphs.
    """
    # Deferred import: arrowspace is a hard dep but we don't want it
    # imported at module level so the library is importable in environments
    # where only the pure-Python layer is installed (CI, docs, type checkers).
    from arrowspace import ArrowSpaceBuilder

    # ── build graph ───────────────────────────────────────────────────────────
    # ArrowSpaceBuilder.build() can raise pyo3_runtime.PanicException when the
    # corpus collapses into a single cluster (e.g. flat/near-identical vectors,
    # or cluster_radius too large). PanicException is a BaseException, NOT an
    # Exception, so a plain `except Exception` will NOT catch it.
    # We catch BaseException here and re-raise only KeyboardInterrupt /
    # SystemExit so the study loop can continue normally on degenerate inputs.
    try:
        aspace, gl = (
            ArrowSpaceBuilder()
            .with_dims_reduction(enabled=False, eps=None)
            .with_sampling("simple", params.sampling_rate)
            .with_cluster_max_clusters(params.max_clusters)
            .with_cluster_radius(params.cluster_radius)
            .build(params.to_dict(), embeddings)
        )
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

    # ── pre-flight: check graph shape before touching CSR data ───────────────
    # If all embeddings collapsed into 1 cluster the Laplacian has shape (N,1)
    # and any eigendecomposition will panic. Detect and prune early.
    shape = gl.shape()
    if shape[1] < 2:
        logger.warning(
            "Degenerate graph shape=%s (single cluster) | eps=%.4f", shape, params.eps
        )
        if trial:
            raise optuna.TrialPruned()
        return 0.0, 0.0, None, None

    raw = gl.to_csr()
    nnz = len(raw[0])
    n   = shape[1]

    # ── degenerate guard 1: nearly empty graph ────────────────────────────────
    if nnz <= n:
        logger.warning(
            "Degenerate graph NNZ=%d <= N=%d | eps=%.4f", nnz, n, params.eps
        )
        if trial:
            raise optuna.TrialPruned()
        return 0.0, 0.0, None, None

    fiedler    = fiedler_normalized(gl)
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

    # ── degenerate guard 3: flat spectrum ─────────────────────────────────────
    if spread < 1e-10:
        logger.warning(
            "Flat spectrum spread=%.2e — pruning", spread
        )
        if trial:
            raise optuna.TrialPruned()
        return fiedler, 0.0, None, None

    return fiedler, var_lambda, aspace, gl


# ── main factory ──────────────────────────────────────────────────────────────

def make_objective(
    embeddings: np.ndarray,
    cfg:        StudyConfig,
):
    """
    Return an Optuna objective function closed over embeddings and cfg.

    The returned callable is passed directly to study.optimize().
    It searches over eps, k, and tau simultaneously.

    Parameters
    ----------
    embeddings : np.ndarray
        Full corpus embeddings shape (N, D).
    cfg : StudyConfig
        Study configuration including search bounds and MRR probe settings.

    Returns
    -------
    Callable[[optuna.Trial], float]
        The Optuna objective function.
    """
    def objective(trial: optuna.Trial) -> float:

        # ── 1. sample corpus for this trial ──────────────────────────────────
        if cfg.sample_n and cfg.sample_n < len(embeddings):
            rng       = np.random.default_rng(trial.number + cfg.seed)
            idx       = rng.choice(len(embeddings), size=cfg.sample_n, replace=False)
            emb_trial = embeddings[idx]
        else:
            emb_trial = embeddings

        # ── 2. suggest hyperparameters ────────────────────────────────────────
        k   = trial.suggest_int(  "k",   cfg.k_low,   cfg.k_high)
        eps = trial.suggest_float("eps", cfg.eps_low,  cfg.eps_high, log=True)
        tau = trial.suggest_float("tau", cfg.tau_low,  cfg.tau_high)

        params = BuildParams(
            eps=eps,
            k=k,
            topk=max(1, k // 2),
        )

        # ── 3. build graph + spectral diagnostics ─────────────────────────────
        # Note: build_and_score already catches BaseException (Rust panics)
        # and converts them to TrialPruned. We only need to re-raise
        # TrialPruned here; all other exceptions are already handled.
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

        # ── 4. sample probe anchors ───────────────────────────────────────────
        n         = len(emb_trial)
        rng_probe = np.random.default_rng(trial.number + cfg.seed + 42)
        probe_idx = rng_probe.choice(n, size=min(cfg.n_probe, n), replace=False)

        # ── 5. k-NN retrieval via search_batch ────────────────────────────────
        probe_embs = np.ascontiguousarray(
            emb_trial[probe_idx], dtype=np.float64
        )

        try:
            batch_results = aspace.search_batch(probe_embs, gl, tau)
        except Exception as exc:
            logger.warning(
                "Trial %d search_batch failed (eps=%.4f k=%d tau=%.3f): %s",
                trial.number, params.eps, params.k, tau, exc,
            )
            raise optuna.TrialPruned()

        # ── 6. build k-NN index table ─────────────────────────────────────────
        knn_indices = np.zeros((len(probe_idx), K_EVAL), dtype=np.int64)
        for row, results in enumerate(batch_results):
            for col, (idx_item, _score) in enumerate(results[:K_EVAL]):
                knn_indices[row, col] = idx_item

        # ── 7. spectral MRR-Top0 proxy ────────────────────────────────────────
        # Topology factor: T_qi = exp(-|λ_q - λ_i| / σ_λ)
        # Items spectrally close to the query anchor are treated as relevant.
        # No ground-truth labels required.
        lambdas      = np.array(aspace.lambdas(), dtype=np.float64)   # (N,)
        sigma        = float(np.std(lambdas)) + 1e-9
        lambda_probe = lambdas[probe_idx]                              # (P,)

        mrr_scores = []
        for row in range(len(probe_idx)):
            lq     = lambda_probe[row]
            nbrs   = knn_indices[row]                                  # (K,)
            l_nbrs = lambdas[nbrs]
            t_qi   = np.exp(-np.abs(lq - l_nbrs) / sigma)
            mrr_scores.append(
                float(np.sum(t_qi / np.arange(1, K_EVAL + 1)))
            )

        mrr_proxy = float(np.mean(mrr_scores))

        # ── 8. composite objective ────────────────────────────────────────────
        score = (
            W_MRR  * mrr_proxy
          + W_FIED * float(np.log1p(fiedler))
          + W_VAR  * float(np.log1p(var_lambda))
        )

        # ── 9. log trial attributes ───────────────────────────────────────────
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

    return objective
