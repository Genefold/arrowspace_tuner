"""
test_io.py — input loading and validation contract (spec §12.1).
"""
from __future__ import annotations

import hashlib

import numpy as np
import pytest

from arrowspace_tuner.io import (
    AmbiguousNpzArrayError,
    InputValidationError,
    UnsupportedInputFormatError,
    load_embeddings,
    sha256_file,
)


# ── valid inputs ──────────────────────────────────────────────────────────────

def test_valid_float64_npy_loads(npy_file) -> None:
    info, embeddings = load_embeddings(npy_file)
    assert embeddings.dtype == np.float64
    assert embeddings.shape == (120, 64)
    assert info.format == "npy"
    assert info.array_key is None
    assert info.finite is True
    assert info.sha256 is not None


def test_valid_float32_npy_loads_and_converts(tmp_path, rng) -> None:
    path = tmp_path / "float32.npy"
    np.save(path, rng.standard_normal((50, 8)).astype(np.float32))
    info, embeddings = load_embeddings(path)
    assert info.dtype == "float32"
    assert embeddings.dtype == np.float64


def test_valid_integer_npy_loads_and_converts(tmp_path, rng) -> None:
    path = tmp_path / "int.npy"
    np.save(path, rng.integers(0, 10, size=(50, 8)).astype(np.int32))
    info, embeddings = load_embeddings(path)
    assert info.dtype.startswith("int")
    assert embeddings.dtype == np.float64


def test_valid_single_array_npz_loads(npz_single_file) -> None:
    info, embeddings = load_embeddings(npz_single_file)
    assert info.format == "npz"
    assert info.array_key == "embeddings"
    assert embeddings.shape == (120, 64)


def test_multi_array_npz_with_valid_key_loads(npz_multi_file) -> None:
    info, embeddings = load_embeddings(npz_multi_file, array_key="medium")
    assert info.array_key == "medium"
    assert embeddings.shape == (600, 64)


def test_hash_is_stable(npy_file) -> None:
    expected = hashlib.sha256(npy_file.read_bytes()).hexdigest()
    assert sha256_file(npy_file) == expected
    info, _ = load_embeddings(npy_file, include_hash=True)
    assert info.sha256 == expected
    info_no_hash, _ = load_embeddings(npy_file, include_hash=False)
    assert info_no_hash.sha256 is None


def test_norm_diagnostics_are_calculated(tmp_path) -> None:
    rng = np.random.default_rng(7)
    arr = rng.standard_normal((150, 8)) * 3.0  # deliberately not normalised
    path = tmp_path / "scaled.npy"
    np.save(path, arr)
    info, _ = load_embeddings(path)
    norms = np.linalg.norm(arr, axis=1)
    assert info.l2_norm_min == pytest.approx(float(norms.min()))
    assert info.l2_norm_mean == pytest.approx(float(norms.mean()))
    assert info.l2_norm_max == pytest.approx(float(norms.max()))
    assert any("vary substantially" in w for w in info.warnings)


# ── invalid inputs ────────────────────────────────────────────────────────────

def test_1d_array_rejected(tmp_path, rng) -> None:
    path = tmp_path / "one_d.npy"
    np.save(path, rng.standard_normal(64))
    with pytest.raises(InputValidationError, match="2D"):
        load_embeddings(path)


def test_3d_array_rejected(tmp_path, rng) -> None:
    path = tmp_path / "three_d.npy"
    np.save(path, rng.standard_normal((10, 4, 4)))
    with pytest.raises(InputValidationError, match="2D"):
        load_embeddings(path)


def test_empty_matrix_rejected(tmp_path) -> None:
    path = tmp_path / "empty.npy"
    np.save(path, np.empty((0, 8), dtype=np.float64))
    with pytest.raises(InputValidationError, match="at least 2 rows"):
        load_embeddings(path)


def test_object_dtype_rejected(tmp_path) -> None:
    path = tmp_path / "object.npy"
    np.save(path, np.array([[object()], [object()]], dtype=object), allow_pickle=True)
    with pytest.raises(InputValidationError, match="Could not load"):
        load_embeddings(path)


def test_complex_dtype_rejected(tmp_path, rng) -> None:
    path = tmp_path / "complex.npy"
    np.save(path, (rng.standard_normal((10, 4)) + 1j).astype(np.complex128))
    with pytest.raises(InputValidationError, match="not supported"):
        load_embeddings(path)


def test_nan_rejected(tmp_path, rng) -> None:
    arr = rng.standard_normal((50, 8))
    arr[3, 2] = np.nan
    path = tmp_path / "nan.npy"
    np.save(path, arr)
    with pytest.raises(InputValidationError, match="NaN"):
        load_embeddings(path)


def test_infinity_rejected(tmp_path, rng) -> None:
    arr = rng.standard_normal((50, 8))
    arr[0, 0] = np.inf
    path = tmp_path / "inf.npy"
    np.save(path, arr)
    with pytest.raises(InputValidationError, match="infinity"):
        load_embeddings(path)


def test_multi_array_npz_without_key_rejected(npz_multi_file) -> None:
    with pytest.raises(AmbiguousNpzArrayError):
        load_embeddings(npz_multi_file)


def test_invalid_npz_key_rejected(npz_multi_file) -> None:
    with pytest.raises(InputValidationError, match="not found"):
        load_embeddings(npz_multi_file, array_key="nope")


def test_unsupported_extension_rejected(tmp_path) -> None:
    path = tmp_path / "data.pkl"
    path.write_bytes(b"\x80\x04\x95")
    with pytest.raises(UnsupportedInputFormatError):
        load_embeddings(path)


def test_nonexistent_path_rejected(tmp_path) -> None:
    with pytest.raises(InputValidationError, match="does not exist"):
        load_embeddings(tmp_path / "missing.npy")


def test_directory_rejected(tmp_path) -> None:
    with pytest.raises(InputValidationError, match="not a regular file"):
        load_embeddings(tmp_path)


def test_max_bytes_policy_enforced(tmp_path, rng) -> None:
    path = tmp_path / "big.npy"
    np.save(path, rng.standard_normal((100, 8)))
    with pytest.raises(InputValidationError, match="maximum size"):
        load_embeddings(path, max_bytes=10)