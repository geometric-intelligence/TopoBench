"""Bounded sparse feature conversion for native hypergraph assets."""

from __future__ import annotations
import hashlib
import json
from pathlib import Path

import numpy as np
import pytest
import torch
from scipy.sparse import csc_matrix, csr_matrix

import topobench.data.utils.hypergraph_io as hypergraph_io
from topobench.data.utils.hypergraph_io import (
    SAFE_HYPERGRAPH_CONVERTER_VERSION,
    SAFE_HYPERGRAPH_FORMAT,
    SAFE_HYPERGRAPH_FORMAT_VERSION,
    _as_feature_tensor,
    load_hypergraph_npz_dataset,
)


def test_high_shape_csr_converts_without_dense_allocation(
    monkeypatch: pytest.MonkeyPatch,
) -> None:
    """Tiny-NNZ CSR input never allocates storage proportional to its shape."""
    features = csr_matrix(
        (
            np.array([1.25, -3.5], dtype=np.float64),
            (
                np.array([0, 2], dtype=np.int64),
                np.array([999_999_999, 7], dtype=np.int64),
            ),
        ),
        shape=(3, 1_000_000_000),
    )

    def unexpected_dense(*_: object, **__: object) -> object:
        raise AssertionError("CSR conversion attempted a dense allocation")

    monkeypatch.setattr(csr_matrix, "toarray", unexpected_dense)
    monkeypatch.setattr(csr_matrix, "todense", unexpected_dense)

    tensor = _as_feature_tensor(features)

    assert tensor.layout == torch.sparse_coo
    assert tensor.is_coalesced()
    assert tensor.dtype == torch.float32
    assert tensor.shape == torch.Size((3, 1_000_000_000))
    assert tensor._indices().dtype == torch.int64
    assert tensor._nnz() == 2
    torch.testing.assert_close(
        tensor._indices(),
        torch.tensor([[0, 2], [999_999_999, 7]], dtype=torch.long),
    )
    torch.testing.assert_close(
        tensor._values(), torch.tensor([1.25, -3.5], dtype=torch.float32)
    )


def test_duplicate_csr_coordinates_are_coalesced() -> None:
    features = csr_matrix(
        (
            np.array([1.0, 2.0], dtype=np.float32),
            np.array([1, 1], dtype=np.int64),
            np.array([0, 2], dtype=np.int64),
        ),
        shape=(1, 3),
    )

    tensor = _as_feature_tensor(features)

    assert tensor.is_coalesced()
    assert tensor._nnz() == 1
    torch.testing.assert_close(tensor._values(), torch.tensor([3.0]))


@pytest.mark.parametrize(
    ("features", "error_type", "message"),
    [
        (
            csc_matrix(np.eye(2, dtype=np.float32)),
            TypeError,
            "scipy CSR",
        ),
        (
            csr_matrix(np.eye(2, dtype=np.int64)),
            TypeError,
            "floating dtype",
        ),
        (
            csr_matrix(np.array([[np.nan]], dtype=np.float32)),
            ValueError,
            "finite values",
        ),
        (
            csr_matrix(np.array([[np.inf]], dtype=np.float64)),
            ValueError,
            "finite values",
        ),
    ],
    ids=["csc-layout", "integer-dtype", "nan", "infinity"],
)
def test_sparse_conversion_rejects_unsupported_or_nonfinite_input(
    features: object,
    error_type: type[Exception],
    message: str,
) -> None:
    with pytest.raises(error_type, match=message):
        _as_feature_tensor(features)


def test_sparse_conversion_rejects_invalid_csr_bounds() -> None:
    features = csr_matrix(
        (
            np.array([1.0], dtype=np.float32),
            np.array([0], dtype=np.int64),
            np.array([0, 1, 1], dtype=np.int64),
        ),
        shape=(2, 3),
    )
    features.indices[0] = 3

    with pytest.raises(ValueError, match="column indices.*bounds"):
        _as_feature_tensor(features)


def test_torch_sparse_coo_is_validated_and_coalesced() -> None:
    features = torch.sparse_coo_tensor(
        torch.tensor([[0, 0], [1, 1]], dtype=torch.long),
        torch.tensor([1.0, 2.0], dtype=torch.float64),
        size=(2, 3),
    )

    tensor = _as_feature_tensor(features)

    assert tensor.layout == torch.sparse_coo
    assert tensor.is_coalesced()
    assert tensor.dtype == torch.float32
    torch.testing.assert_close(tensor._values(), torch.tensor([3.0]))


def test_torch_sparse_coo_rejects_indices_outside_declared_shape() -> None:
    features = torch.sparse_coo_tensor(
        torch.tensor([[2], [0]], dtype=torch.long),
        torch.tensor([1.0]),
        size=(2, 3),
        check_invariants=False,
    )

    with pytest.raises(ValueError, match="indices.*shape bounds"):
        _as_feature_tensor(features)


def test_oversized_dense_conversion_rejects_before_torch_allocation(
    monkeypatch: pytest.MonkeyPatch,
) -> None:
    backing = np.zeros(1, dtype=np.float64)
    features = np.lib.stride_tricks.as_strided(
        backing,
        shape=(2, 1_000_000_000),
        strides=(0, 0),
        writeable=False,
    )

    def unexpected_allocation(*_: object, **__: object) -> object:
        raise AssertionError("dense tensor allocation ran before preflight")

    monkeypatch.setattr(torch, "as_tensor", unexpected_allocation)

    with pytest.raises(ValueError) as error:
        _as_feature_tensor(features, max_dense_bytes=1024)

    message = str(error.value)
    assert "shape=(2, 1000000000)" in message
    assert "dtype=float64" in message
    assert "estimate=8000000000" in message
    assert "limit=1024" in message


@pytest.mark.parametrize("limit", [True, 0, -1, 1.5])
def test_dense_allocation_limit_must_be_a_positive_integer(limit: object) -> None:
    with pytest.raises((TypeError, ValueError), match="max_dense_bytes"):
        _as_feature_tensor(
            np.ones((1, 1), dtype=np.float32),
            max_dense_bytes=limit,
        )


def test_safe_dense_descriptor_rejects_before_archive_member_access(
    tmp_path: Path,
    monkeypatch: pytest.MonkeyPatch,
) -> None:
    name = "oversized"
    npz_path = tmp_path / f"{name}.npz"
    npz_path.write_bytes(b"authenticated-but-never-opened")
    metadata = {
        "arrays": {
            "features": {
                "dtype": np.dtype(np.float64).str,
                "shape": [2, 1_000_000_000],
            },
            "incidence": {
                "dtype": np.dtype(np.int64).str,
                "shape": [2, 1],
            },
            "labels": {
                "dtype": np.dtype(np.int64).str,
                "shape": [2],
            },
        },
        "converter_version": SAFE_HYPERGRAPH_CONVERTER_VERSION,
        "feature_storage": "dense",
        "format": SAFE_HYPERGRAPH_FORMAT,
        "format_version": SAFE_HYPERGRAPH_FORMAT_VERSION,
        "incidence_roles": ["node", "hyperedge"],
        "npz_sha256": hashlib.sha256(npz_path.read_bytes()).hexdigest(),
        "num_hyperedges": 1,
        "num_nodes": 2,
        "padding_count": 0,
        "padding_sentinel": None,
        "raw_sha256": "a" * 64,
    }
    encoded = (
        json.dumps(
            metadata,
            allow_nan=False,
            ensure_ascii=True,
            separators=(",", ":"),
            sort_keys=True,
        )
        + "\n"
    ).encode("utf-8")
    (tmp_path / f"{name}.json").write_bytes(encoded)
    accessed: list[str] = []

    class NoMaterializationArchive:
        files = ["features", "incidence", "labels"]

        def __enter__(self):
            return self

        def __exit__(self, *_: object) -> None:
            return None

        def __getitem__(self, member: str) -> np.ndarray:
            accessed.append(member)
            raise AssertionError("archive member accessed before preflight")

    monkeypatch.setattr(
        hypergraph_io.np,
        "load",
        lambda *_args, **_kwargs: NoMaterializationArchive(),
    )

    with pytest.raises(ValueError) as error:
        load_hypergraph_npz_dataset(
            tmp_path,
            name,
            max_dense_bytes=1024,
        )

    assert accessed == []
    message = str(error.value)
    assert "shape=(2, 1000000000)" in message
    assert "dtype=float64" in message
    assert "estimate=8000000000" in message
    assert "limit=1024" in message
