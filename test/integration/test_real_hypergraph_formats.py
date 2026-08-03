"""Adapter-level evidence for the published non-executable hypergraph format."""

from __future__ import annotations

import hashlib
import json
from pathlib import Path
from typing import Any

import numpy as np
import pytest
import torch
from omegaconf import OmegaConf

from topobench.data import HYPERGRAPH_CACHE_FILENAME, HypergraphData
from topobench.data.datasets import (
    CitationHypergraphDataset,
    HypergraphDataset,
)
from topobench.data.utils.hypergraph_io import (
    SAFE_HYPERGRAPH_CONVERTER_VERSION,
    SAFE_HYPERGRAPH_FORMAT,
    SAFE_HYPERGRAPH_FORMAT_VERSION,
)


def _canonical_json(payload: dict[str, Any]) -> bytes:
    return (
        json.dumps(
            payload,
            allow_nan=False,
            ensure_ascii=True,
            separators=(",", ":"),
            sort_keys=True,
        )
        + "\n"
    ).encode("utf-8")


def _write_safe_asset(raw_dir: Path, name: str, *, valid_digest: bool) -> None:
    raw_dir.mkdir(parents=True, exist_ok=True)
    arrays = {
        "features": np.asarray(
            [[0.0, 0.0], [1.0, 2.0], [3.0, 4.0]], dtype=np.float32
        ),
        "incidence": np.asarray([[1, 1, 2], [1, 1, 0]], dtype=np.int64),
        "labels": np.asarray([0, 2, 1], dtype=np.int64),
    }
    npz_path = raw_dir / f"{name}.npz"
    np.savez(npz_path, **arrays)
    digest = hashlib.sha256(npz_path.read_bytes()).hexdigest()
    metadata = {
        "arrays": {
            key: {"dtype": value.dtype.str, "shape": list(value.shape)}
            for key, value in arrays.items()
        },
        "converter_version": SAFE_HYPERGRAPH_CONVERTER_VERSION,
        "feature_storage": "dense",
        "format": SAFE_HYPERGRAPH_FORMAT,
        "format_version": SAFE_HYPERGRAPH_FORMAT_VERSION,
        "incidence_roles": ["node", "hyperedge"],
        "npz_sha256": digest if valid_digest else "0" * 64,
        "num_hyperedges": 2,
        "num_nodes": 3,
        "padding_count": 0,
        "padding_sentinel": None,
        "raw_sha256": "a" * 64,
    }
    (raw_dir / f"{name}.json").write_bytes(_canonical_json(metadata))


@pytest.mark.parametrize(
    "dataset_class",
    [CitationHypergraphDataset, HypergraphDataset],
)
def test_safe_asset_processes_and_caches_without_losing_declared_rows(
    tmp_path: Path,
    dataset_class: type[CitationHypergraphDataset] | type[HypergraphDataset],
) -> None:
    name = dataset_class.__name__.lower()
    _write_safe_asset(tmp_path / name / "raw", name, valid_digest=True)

    dataset = dataset_class(
        root=str(tmp_path),
        name=name,
        parameters=OmegaConf.create({"data_name": name}),
    )

    data = dataset[0]
    assert isinstance(data, HypergraphData)
    assert torch.equal(data.x[0], torch.tensor([0.0, 0.0]))
    assert int(data.y[0]) == 0
    assert 0 not in data.hyperedge_index[0]
    assert torch.equal(
        data.hyperedge_index,
        torch.tensor([[1, 1, 2], [1, 1, 0]], dtype=torch.long),
    )
    assert (
        tmp_path / name / "processed" / HYPERGRAPH_CACHE_FILENAME
    ).is_file()


@pytest.mark.parametrize(
    "dataset_class",
    [CitationHypergraphDataset, HypergraphDataset],
)
def test_malformed_safe_asset_fails_before_cache_creation(
    tmp_path: Path,
    dataset_class: type[CitationHypergraphDataset] | type[HypergraphDataset],
) -> None:
    name = f"invalid-{dataset_class.__name__.lower()}"
    _write_safe_asset(tmp_path / name / "raw", name, valid_digest=False)

    with pytest.raises(ValueError, match="NPZ SHA-256 mismatch"):
        dataset_class(
            root=str(tmp_path),
            name=name,
            parameters=OmegaConf.create({"data_name": name}),
        )

    assert not (
        tmp_path / name / "processed" / HYPERGRAPH_CACHE_FILENAME
    ).exists()


def test_unpublished_remote_selectors_have_no_runtime_asset_fallback() -> None:
    assert dict(CitationHypergraphDataset.ASSETS) == {}
    assert dict(HypergraphDataset.ASSETS) == {}
