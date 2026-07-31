"""Offline tests for native hypergraph parsers and dataset caches."""

from __future__ import annotations

import pickle
from pathlib import Path

import numpy as np
import pytest
import torch
from omegaconf import OmegaConf
from scipy.sparse import csr_matrix

from topobench.data import (
    HYPERGRAPH_CACHE_FILENAME,
    HYPERGRAPH_REPRESENTATION_VERSION,
    HypergraphData,
    validate_hypergraph_structure,
)
from topobench.data.datasets import CitationHypergraphDataset, HypergraphDataset
from topobench.data.loaders.hypergraph.citation_hypergraph_dataset_loader import (
    CitationHypergraphDatasetLoader,
)
from topobench.data.loaders.hypergraph.hypergraph_dataset_loader import (
    HypergraphDatasetLoader,
)
from topobench.data.utils.hypergraph_io import (
    load_hypergraph_content_dataset,
    load_hypergraph_pickle_dataset,
)


def _write_pickle_fixture(raw_dir: Path, *, empty: bool = False) -> None:
    raw_dir.mkdir(parents=True, exist_ok=True)
    features = csr_matrix(
        np.array(
            [
                [10.0, 11.0],
                [20.0, 21.0],
                [30.0, 31.0],
                [40.0, 41.0],
            ],
            dtype=np.float32,
        )
    )
    labels = np.array([2, 0, 1, 2], dtype=np.int64)
    hypergraph = {
        "sparse-edge": [] if empty else [1],
        700: [2, 0, 2],
    }
    for filename, value in (
        ("features.pickle", features),
        ("labels.pickle", labels),
        ("hypergraph.pickle", hypergraph),
    ):
        with (raw_dir / filename).open("wb") as stream:
            pickle.dump(value, stream)


def _write_content_fixture(raw_dir: Path, name: str) -> None:
    raw_dir.mkdir(parents=True, exist_ok=True)
    (raw_dir / f"{name}.content").write_text(
        "\n".join(
            [
                "10 1.0 1.5 3",
                "20 2.0 2.5 5",
                "30 3.0 3.5 4",
                "700 0.0 0.0 0",
                "900 0.0 0.0 0",
            ]
        )
        + "\n",
        encoding="utf-8",
    )
    (raw_dir / f"{name}.edges").write_text(
        "30 900\n10 700\n30 700\n10 700\n",
        encoding="utf-8",
    )


def _assert_native(data: HypergraphData) -> None:
    assert validate_hypergraph_structure(data) is data
    assert data.representation_version == HYPERGRAPH_REPRESENTATION_VERSION
    assert int(data["representation_version"]) == HYPERGRAPH_REPRESENTATION_VERSION
    assert data.hyperedge_index.dtype == torch.long
    assert data.hyperedge_index.shape[0] == 2
    pairs = list(map(tuple, data.hyperedge_index.t().tolist()))
    assert pairs == sorted(set(pairs))
    assert torch.equal(
        torch.unique(data.hyperedge_index[1]),
        torch.arange(data.num_hyperedges),
    )
    for legacy_field in ("x_0", "edge_index", "incidence_hyperedges", "n_x"):
        assert legacy_field not in data


def test_pickle_parser_canonicalizes_sparse_mixed_ids_and_adds_singletons(
    tmp_path: Path,
) -> None:
    raw_dir = tmp_path / "pickle-data"
    _write_pickle_fixture(raw_dir)

    data, resolved_dir = load_hypergraph_pickle_dataset(tmp_path, "pickle-data")

    _assert_native(data)
    assert resolved_dir == str(raw_dir)
    assert data.num_nodes == 4
    assert data.num_hyperedges == 3
    assert torch.equal(
        data.x,
        torch.tensor(
            [[10.0, 11.0], [20.0, 21.0], [30.0, 31.0], [40.0, 41.0]]
        ),
    )
    assert torch.equal(data.y, torch.tensor([2, 0, 1, 2]))
    assert torch.equal(
        data.hyperedge_index,
        torch.tensor([[0, 1, 2, 3], [0, 1, 0, 2]], dtype=torch.long),
    )


def test_pickle_parser_rejects_declared_empty_hyperedges(tmp_path: Path) -> None:
    raw_dir = tmp_path / "pickle-empty"
    _write_pickle_fixture(raw_dir, empty=True)

    with pytest.raises(ValueError, match="empty hyperedge"):
        load_hypergraph_pickle_dataset(tmp_path, "pickle-empty")


def test_content_parser_preserves_isolated_node_rows_and_alignment(
    tmp_path: Path,
) -> None:
    name = "content-data"
    _write_content_fixture(tmp_path, name)

    data, resolved_dir = load_hypergraph_content_dataset(tmp_path, name)

    _assert_native(data)
    assert resolved_dir == str(tmp_path)
    assert data.num_nodes == 3
    assert data.num_hyperedges == 2
    assert torch.equal(
        data.x,
        torch.tensor([[1.0, 1.5], [2.0, 2.5], [3.0, 3.5]]),
    )
    assert torch.equal(data.y, torch.tensor([0, 2, 1]))
    assert torch.equal(
        data.hyperedge_index,
        torch.tensor([[0, 2, 2], [0, 0, 1]], dtype=torch.long),
    )
    assert 1 not in data.hyperedge_index[0]


def test_citation_loader_bypasses_old_cache_and_loads_versioned_native_data(
    tmp_path: Path,
    monkeypatch: pytest.MonkeyPatch,
) -> None:
    name = "pickle-cache"
    raw_dir = tmp_path / name / "raw"
    _write_pickle_fixture(raw_dir)
    processed_dir = tmp_path / name / "processed"
    processed_dir.mkdir(parents=True)
    (processed_dir / "data.pt").write_bytes(b"legacy rank-based cache")
    monkeypatch.setattr(CitationHypergraphDataset, "download", lambda self: None)
    parameters = OmegaConf.create({"data_dir": str(tmp_path), "data_name": name})

    with pytest.warns(UserWarning, match="Ignoring legacy processed cache"):
        dataset = CitationHypergraphDatasetLoader(parameters).load_dataset()

    data = dataset[0]
    _assert_native(data)
    assert dataset.processed_file_names == HYPERGRAPH_CACHE_FILENAME
    assert (processed_dir / HYPERGRAPH_CACHE_FILENAME).is_file()
    assert (processed_dir / "data.pt").read_bytes() == b"legacy rank-based cache"


def test_content_loader_round_trips_only_the_versioned_native_cache(
    tmp_path: Path,
    monkeypatch: pytest.MonkeyPatch,
) -> None:
    name = "content-cache"
    raw_dir = tmp_path / name / "raw"
    _write_content_fixture(raw_dir, name)
    monkeypatch.setattr(HypergraphDataset, "download", lambda self: None)
    parameters = OmegaConf.create({"data_dir": str(tmp_path), "data_name": name})

    first = HypergraphDatasetLoader(parameters).load_dataset()
    first_data = first[0]
    _assert_native(first_data)

    def fail_if_reparsed(*args: object, **kwargs: object) -> None:
        raise AssertionError("versioned cache should be loaded without reparsing raw data")

    monkeypatch.setattr(
        "topobench.data.datasets.hypergraph_datasets.load_hypergraph_content_dataset",
        fail_if_reparsed,
    )
    second = HypergraphDatasetLoader(parameters).load_dataset()
    second_data = second[0]
    _assert_native(second_data)
    assert torch.equal(second_data.x, first_data.x)
    assert torch.equal(second_data.y, first_data.y)
    assert torch.equal(second_data.hyperedge_index, first_data.hyperedge_index)
    assert second.processed_file_names == HYPERGRAPH_CACHE_FILENAME
