from __future__ import annotations

import logging

import numpy as np
import scipy.sparse as sp
import scipy.sparse.linalg as spla

logger = logging.getLogger(__name__)


def gl_to_scipy(gl: object) -> sp.csr_matrix:
    """
    Convert a PyGraphLaplacian (from the ArrowSpace Rust extension) to a
    SciPy CSR sparse matrix.

    Parameters
    ----------
    gl : PyGraphLaplacian
        The graph Laplacian returned by ArrowSpaceBuilder.build().

    Returns
    -------
    sp.csr_matrix
        The Laplacian as a SciPy sparse matrix, ready for eigendecomposition.
    """
    raw = gl.to_csr()          # returns (data, indices, indptr, shape)
    shape = gl.shape()
    data    = np.asarray(raw[0], dtype=np.float64)
    indices = np.asarray(raw[1], dtype=np.int32)
    indptr  = np.asarray(raw[2], dtype=np.int32)
    return sp.csr_matrix((data, indices, indptr), shape=shape)


def _get_nnz(gl: object) -> int:
    """Return the number of non-zero entries in the graph Laplacian."""
    return len(gl.to_csr()[0])


def fiedler_normalized(gl: object) -> float:
    """
    Compute the normalised Fiedler value (λ₂) of the graph Laplacian.

    The Fiedler value is the second-smallest eigenvalue of the normalised
    Laplacian L_norm = D^{-1/2} L D^{-1/2}. It measures algebraic
    connectivity: λ₂ = 0 means the graph is disconnected, higher values
    indicate stronger connectivity.

    Used as one factor in the composite objective.

    Eigenvalue strategy
    -------------------
    N ≤ 5_000 : dense path via np.linalg.eigvalsh.
        Always converges, zero ARPACK overhead, fastest at this scale.
        Covers the sample_n=5_000 default path entirely.
    N > 5_000 : shift-invert ARPACK (sigma=0.0, which="LM").
        Finds the largest eigenvalues of L^{-1}, equivalent to the
        smallest eigenvalues of L. 5–20× faster than which="SM" for
        sparse PSD matrices and far more numerically stable.
        tol=1e-4 is sufficient because the Fiedler value feeds into
        log1p() — 4 significant digits is more than adequate.

    Parameters
    ----------
    gl : PyGraphLaplacian
        The graph Laplacian returned by ArrowSpaceBuilder.build().

    Returns
    -------
    float
        λ₂ ∈ [0, 1]. Returns 0.0 on degenerate or disconnected graphs,
        and on any numerical failure.
    """
    try:
        nnz   = _get_nnz(gl)
        shape = gl.shape()
        n     = shape[0]

        # Degenerate guard: fewer edges than nodes → nearly empty graph
        if nnz <= n:
            logger.warning(
                "Degenerate graph NNZ=%d <= N=%d — returning 0.0", nnz, n
            )
            return 0.0

        L = gl_to_scipy(gl)

        # Normalise: L_norm = D^{-1/2} L D^{-1/2}
        diag       = np.array(L.diagonal(), dtype=np.float64)
        safe_diag  = np.where(diag > 1e-12, diag, 1e-12)
        d_inv_sqrt = sp.diags(1.0 / np.sqrt(safe_diag))
        L_norm     = d_inv_sqrt @ L @ d_inv_sqrt

        # ── eigenvalue computation ───────────────────────────────────────────────
        if n <= 5_000:
            # Dense path: materialise and use full symmetric eigensolver.
            # Always converges; competitive with ARPACK at this scale.
            all_vals = np.linalg.eigvalsh(L_norm.toarray())
            vals = all_vals[:2]  # two smallest: λ₀ ≈ 0, λ₁ = Fiedler
        else:
            # Shift-invert: finds largest eigenvalues of (L_norm - 0·I)^{-1}
            # = L_norm^{-1}, which correspond to the *smallest* eigenvalues
            # of L_norm. Correct mode for bottom-k of a PSD matrix.
            # tol=1e-4: sufficient for log1p-weighted objective term.
            vals = spla.eigsh(
                L_norm,
                k=2,
                sigma=0.0,
                which="LM",
                return_eigenvectors=False,
                tol=1e-4,
                maxiter=500,
            )

        fiedler = max(0.0, float(sorted(np.real(vals))[1]))

        logger.debug(
            "fiedler_normalized: λ₂=%.6f  NNZ=%d  N=%d  path=%s",
            fiedler, nnz, n, "dense" if n <= 5_000 else "shift-invert",
        )
        return fiedler

    except Exception as exc:
        logger.warning("fiedler_normalized failed: %s", exc, exc_info=True)
        return 0.0
