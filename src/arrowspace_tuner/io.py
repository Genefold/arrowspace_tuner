"""
io.py — loading and validation of local embedding matrices (.npy / .npz).

This is the single input contract for the CLI and MCP server:

- ``allow_pickle=False`` always — object arrays are rejected;
- 2D numeric arrays only (float/int/uint), converted to float64;
- no NaN, no infinity, at least 2 rows and 1 column;
- NPZ: single-array files auto-selected, multi-array files require a key.

Embeddings are never normalised or modified — diagnostics only.
"""
from __future__ import annotations

import hashlib
from pathlib import Path
from typing import Literal

import numpy as np

from .models import SUPPORTED_FORMATS, EmbeddingInfo

#: File extensions accepted by the CLI/MCP input contract.
SUPPORTED_SUFFIXES: frozenset[str] = frozenset({".npy", ".npz"})

_WARN_FEW_ROWS = "Input has fewer than 100 rows; tuning may be unstable."
_WARN_NORMALISED = "Embeddings appear approximately L2-normalised."
_WARN_VARYING_NORMS = (
    "Embedding norms vary substantially; confirm that raw Euclidean geometry "
    "is intended."
)
_WARN_K_HIGH_CLIP = (
    "k_high exceeds n_items - 1; effective upper bound will be clipped."
)


class InputValidationError(ValueError):
    """Embedding input violates the CLI/MCP contract."""

    def __init__(self, message: str, *, code: str = "invalid_input") -> None:
        super().__init__(message)
        self.code = code


class UnsupportedInputFormatError(InputValidationError):
    """File extension is unsupported."""

    def __init__(self, message: str, *, code: str = "unsupported_format") -> None:
        super().__init__(message, code=code)


class AmbiguousNpzArrayError(InputValidationError):
    """An NPZ contains multiple arrays but no key was supplied."""

    def __init__(self, message: str, *, code: str = "ambiguous_npz_array") -> None:
        super().__init__(message, code=code)


class TuningExecutionError(RuntimeError):
    """No valid tuning result could be produced."""


def sha256_file(path: Path, chunk_size: int = 1024 * 1024) -> str:
    """Hash file bytes in chunks — never holds the whole file in memory."""
    digest = hashlib.sha256()
    with open(path, "rb") as handle:
        while chunk := handle.read(chunk_size):
            digest.update(chunk)
    return digest.hexdigest()


def _reject(condition: bool, message: str, code: str) -> None:
    if condition:
        raise InputValidationError(message, code=code)


def _validate_matrix(arr: np.ndarray, source: str) -> np.ndarray:
    """Shared 2D/numeric/finiteness validation; returns a float64 copy."""
    _reject(
        arr.ndim != 2,
        f"{source} must be a 2D (N, D) matrix, got {arr.ndim}D with shape "
        f"{tuple(arr.shape)}.",
        "not_2d",
    )
    _reject(
        arr.dtype.kind not in {"f", "i", "u"},
        f"{source} dtype {arr.dtype} is not supported; use floating-point or "
        "integer arrays (object, complex, and boolean arrays are rejected).",
        "unsupported_dtype",
    )
    embeddings = np.ascontiguousarray(arr, dtype=np.float64)
    _reject(
        bool(np.isnan(embeddings).any()),
        f"{source} contains NaN values.",
        "nan_detected",
    )
    _reject(
        not bool(np.isfinite(embeddings).all()),
        f"{source} contains positive or negative infinity.",
        "inf_detected",
    )
    _reject(
        embeddings.shape[0] < 2,
        f"{source} must contain at least 2 rows, got {embeddings.shape[0]}.",
        "too_few_rows",
    )
    _reject(
        embeddings.shape[1] < 1,
        f"{source} must contain at least 1 column, got {embeddings.shape[1]}.",
        "too_few_columns",
    )
    return embeddings


def _norm_warnings(embeddings: np.ndarray, n_items: int) -> tuple[str, ...]:
    """Diagnostics-only norm warnings. Never normalises the input."""
    norms = np.linalg.norm(embeddings, axis=1)
    n_min = float(norms.min())
    n_max = float(norms.max())
    warnings: list[str] = []
    if n_items < 100:
        warnings.append(_WARN_FEW_ROWS)
    if 0.95 <= n_min and n_max <= 1.05:
        warnings.append(_WARN_NORMALISED)
    elif n_min > 0.0 and n_max > 3.0 * n_min:
        warnings.append(_WARN_VARYING_NORMS)
    return tuple(warnings)


def _load_npy(path: Path, array_key: str | None) -> np.ndarray:
    if array_key is not None:
        raise InputValidationError(
            "--array-key is only valid for .npz files; the .npy file holds a "
            "single array.",
            code="array_key_not_applicable",
        )
    try:
        loaded: np.ndarray = np.load(path, allow_pickle=False)
        return loaded
    except ValueError as exc:
        raise InputValidationError(
            f"Could not load .npy file: {exc}", code="object_array_rejected"
        ) from exc


def _load_npz(path: Path, array_key: str | None) -> tuple[np.ndarray, str]:
    try:
        with np.load(path, allow_pickle=False) as npz:
            keys = list(npz.files)
            if not keys:
                raise InputValidationError(
                    "The .npz file contains no arrays.", code="empty_npz"
                )
            if array_key is None:
                if len(keys) > 1:
                    raise AmbiguousNpzArrayError(
                        "The .npz file contains multiple arrays "
                        f"({', '.join(sorted(keys))}); supply an array key to "
                        "select one explicitly.",
                    )
                selected = keys[0]
            else:
                _reject(
                    array_key not in keys,
                    f"Array key '{array_key}' not found in .npz; available keys: "
                    f"{', '.join(sorted(keys))}.",
                    "invalid_array_key",
                )
                selected = array_key
            return np.asarray(npz[selected]), selected
    except InputValidationError:
        raise
    except (OSError, ValueError) as exc:
        raise InputValidationError(
            f"Could not load .npz file: {exc}", code="invalid_file"
        ) from exc


def load_embeddings(
    input_path: Path | str,
    *,
    array_key: str | None = None,
    include_hash: bool = True,
    max_bytes: int | None = None,
) -> tuple[EmbeddingInfo, np.ndarray]:
    """
    Load and validate a local embedding matrix.

    Returns ``(EmbeddingInfo, float64 ndarray)``. Raises InputValidationError
    (or a subclass) for every violation of the CLI/MCP input contract.
    """
    try:
        path = Path(input_path).expanduser().resolve(strict=True)
    except OSError as exc:
        raise InputValidationError(
            f"Input path does not exist or is not accessible: {input_path}",
            code="input_not_found",
        ) from exc

    _reject(
        not path.is_file(),
        f"Input path is not a regular file: {path}",
        "not_a_regular_file",
    )

    suffix = path.suffix.lower()
    if suffix not in SUPPORTED_SUFFIXES:
        raise UnsupportedInputFormatError(
            f"Unsupported input format '{suffix}'; supported formats: "
            f"{', '.join(SUPPORTED_FORMATS)}."
        )

    if max_bytes is not None:
        size = path.stat().st_size
        _reject(
            size > max_bytes,
            f"Input file exceeds the configured maximum size "
            f"({size} > {max_bytes} bytes).",
            "input_too_large",
        )

    if suffix == ".npy":
        raw = _load_npy(path, array_key)
        selected_key: str | None = None
        fmt: Literal["npy", "npz"] = "npy"
    else:
        raw, selected_key = _load_npz(path, array_key)
        fmt = "npz"

    source = (
        f"Array '{selected_key}' in {path.name}" if selected_key else f"Array in {path.name}"
    )
    embeddings = _validate_matrix(raw, source)

    n_items, n_dimensions = int(embeddings.shape[0]), int(embeddings.shape[1])
    norms = np.linalg.norm(embeddings, axis=1)

    sha: str | None = sha256_file(path) if include_hash else None

    info = EmbeddingInfo(
        path=str(path),
        format=fmt,
        array_key=selected_key,
        n_items=n_items,
        n_dimensions=n_dimensions,
        shape=(n_items, n_dimensions),
        dtype=str(raw.dtype),
        finite=True,
        l2_norm_min=float(norms.min()),
        l2_norm_mean=float(norms.mean()),
        l2_norm_max=float(norms.max()),
        estimated_memory_bytes=int(embeddings.nbytes),
        sha256=sha,
        warnings=_norm_warnings(embeddings, n_items),
    )
    return info, embeddings


def k_high_warning(n_items: int) -> str:
    """Warning text emitted when k_high exceeds n_items - 1."""
    return _WARN_K_HIGH_CLIP