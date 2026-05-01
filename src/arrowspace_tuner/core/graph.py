from __future__ import annotations

import logging
from typing import Protocol, Sequence, Tuple

import numpy as np
import scipy.sparse as sp
import scipy.sparse.linalg as spla

logger = logging.getLogger(__name__)


class PyGraphLaplacian(Protocol):
    """
    Structural protocol matching the PyGraphLaplacian type exposed by the
    ArrowSpace Rust extension (arrowspace._arrowspace.PyGraphLaplacian).

    Declaring it here as a Protocol lets mypy check call-sites without
    importing the extension at type-check time, which is correct because
    the Rust wheel may not be present in the type-checking environment.
    """

    def to_csr(self) -> Tuple[Sequence[float], Sequence[int], Sequence[int]]:
        """Return (data, indices, indptr) arrays for CSR construction."""
        ...

    def shape(self) -> Tuple[int, int]:
        """Return (nrows, ncols) of the Laplacian matrix."""
        ...


def gl_to_scipy(gl: PyGraphLaplacian) -> sp.csr_matrix:
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


def fiedler_normalized_from_csr(L: sp.csr_matrix, nnz: int) -> float:
    """
    Compute the normalised Fiedler value (λ₂) from a pre-built SciPy CSR
    Laplacian matrix.

    This is the hot path called from build_and_score. The caller is
    responsible for building L and computing nnz from a single gl.to_csr()
    call, avoiding redundant FFI roundtrips (#10).

    Eigenvalue strategy
    -------------------
    N ≤ 5_000 : dense path via np.linalg.eigvalsh.
        Always converges, zero ARPACK overhead, fastest at this scale.
        Covers the sample_n=5_000 default path entirely.
    N > 5_000 : shift-invert ARPACK (sigma=0.0, which="LM").
        Finds the largest eigenvalues of L^{-1}, equivalent to the
        smallest eigenvalues of L. 5–20× faster than which="SM" and
        far more numerically stable.
        tol=1e-4 is sufficient because the Fiedler value feeds into
        log1p() — 4 significant digits is more than adequate.

    Parameters
    ----------
    L : sp.csr_matrix
        Pre-built normalised Laplacian (caller's responsibility).
    nnz : int
        Number of non-zero entries (already computed by caller).

    Returns
    -------
    float
        λ₂ ∈ [0, 1]. Returns 0.0 on degenerate/disconnected graphs
        and on any numerical failure.
    """
    try:
        n = L.shape[0]

        # Degenerate guard: fewer edges than nodes → nearly empty graph
        if nnz <= n:
            logger.warning(
                "Degenerate graph NNZ=%d <= N=%d — returning 0.0", nnz, n
            )
            return 0.0

        # Normalise: L_norm = D^{-1/2} L D^{-1/2}
        diag       = np.array(L.diagonal(), dtype=np.float64)
        safe_diag  = np.where(diag > 1e-12, diag, 1e-12)
        d_inv_sqrt = sp.diags(1.0 / np.sqrt(safe_diag))
        L_norm     = d_inv_sqrt @ L @ d_inv_sqrt

        # ── eigenvalue computation ──────────────────────────────────────────
        if n <= 5_000:
            all_vals = np.linalg.eigvalsh(L_norm.toarray())
            vals = all_vals[:2]
        else:
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


def fiedler_normalized(gl: PyGraphLaplacian) -> float:
    """
    Public wrapper: compute the normalised Fiedler value from a raw
    PyGraphLaplacian. Calls gl.to_csr() once internally.

    Prefer fiedler_normalized_from_csr() in hot paths where the CSR
    matrix has already been materialised to avoid a redundant FFI call.

    Parameters
    ----------
    gl : PyGraphLaplacian
        The graph Laplacian returned by ArrowSpaceBuilder.build().

    Returns
    -------
    float
        λ₂ ∈ [0, 1].
    """
    raw     = gl.to_csr()
    shape   = gl.shape()
    data    = np.asarray(raw[0], dtype=np.float64)
    indices = np.asarray(raw[1], dtype=np.int32)
    indptr  = np.asarray(raw[2], dtype=np.int32)
    L       = sp.csr_matrix((data, indices, indptr), shape=shape)
    nnz     = len(data)
    return fiedler_normalized_from_csr(L, nnz)
