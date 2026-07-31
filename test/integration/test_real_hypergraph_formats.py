"""Opt-in smoke tests for the two supported real hypergraph raw formats."""

import os
from pathlib import Path

import pytest
import torch
from omegaconf import OmegaConf

from topobench.data import (
    HYPERGRAPH_CACHE_FILENAME,
    HYPERGRAPH_REPRESENTATION_VERSION,
    HypergraphData,
    validate_hypergraph_structure,
)
from topobench.data.datasets import (
    CitationHypergraphDataset,
    HypergraphDataset,
)

pytestmark = pytest.mark.skipif(
    os.getenv("TOPOBENCH_RUN_DOWNLOAD_TESTS") != "1",
    reason="set TOPOBENCH_RUN_DOWNLOAD_TESTS=1 to run network smokes",
)


def _assert_real_native_data(data: HypergraphData) -> None:
    assert validate_hypergraph_structure(data) is data
    assert data.representation_version == HYPERGRAPH_REPRESENTATION_VERSION
    assert int(data["representation_version"]) == HYPERGRAPH_REPRESENTATION_VERSION
    assert data.hyperedge_index.dtype == torch.long
    assert torch.equal(
        torch.unique(data.hyperedge_index[1]),
        torch.arange(data.num_hyperedges),
    )


@pytest.mark.integration
@pytest.mark.download
def test_real_cocitation_cora_pickle_format(tmp_path: Path) -> None:
    """Download, extract, parse, and cache the small Cora pickle archive."""
    name = "cocitation_cora"
    dataset = CitationHypergraphDataset(
        root=str(tmp_path),
        name=name,
        parameters=OmegaConf.create({"data_name": name}),
    )
    data = dataset[0]

    _assert_real_native_data(data)
    assert data.x.shape[1] == 1433
    assert int(data.y.min()) == 0
    assert int(data.y.max()) == 6
    assert (tmp_path / name / "raw" / "features.pickle").is_file()
    assert (tmp_path / name / "raw" / "labels.pickle").is_file()
    assert (tmp_path / name / "raw" / "hypergraph.pickle").is_file()
    assert (tmp_path / name / "processed" / HYPERGRAPH_CACHE_FILENAME).is_file()


@pytest.mark.integration
@pytest.mark.download
def test_real_zoo_content_edges_format(tmp_path: Path) -> None:
    """Download, extract, parse, and cache the small Zoo content archive."""
    name = "zoo"
    dataset = HypergraphDataset(
        root=str(tmp_path),
        name=name,
        parameters=OmegaConf.create({"data_name": name}),
    )
    data = dataset[0]

    _assert_real_native_data(data)
    assert data.x.shape[1] == 16
    assert int(data.y.min()) == 0
    assert int(data.y.max()) == 6
    assert (tmp_path / name / "raw" / f"{name}.content").is_file()
    assert (tmp_path / name / "raw" / f"{name}.edges").is_file()
    assert (tmp_path / name / "processed" / HYPERGRAPH_CACHE_FILENAME).is_file()
