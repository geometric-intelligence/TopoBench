"""Native PyG loading contracts for hypergraph node data."""

from __future__ import annotations

import pytest
import torch
from torch_geometric.data import Batch
from torch_geometric.loader import DataLoader

from topobench.data import HypergraphData
from topobench.data.datasets.synthetic_hypergraph_dataset import (
    make_synthetic_hypergraph_data,
)
from topobench.dataloader import GraphDataModule


def test_native_loader_offsets_each_incidence_row_by_its_entity_count() -> None:
    """Node and hyperedge identifiers use independent cumulative offsets."""
    first = make_synthetic_hypergraph_data(
        seed=1,
        num_nodes=6,
        num_hyperedges=2,
    )
    second = make_synthetic_hypergraph_data(
        seed=2,
        num_nodes=8,
        num_hyperedges=3,
    )

    batch = next(iter(DataLoader([first, second], batch_size=2, shuffle=False)))
    second_incidence = batch.hyperedge_index[
        :, first.hyperedge_index.size(1) :
    ]

    assert isinstance(batch, Batch)
    assert batch.num_graphs == 2
    assert torch.equal(
        second_incidence[0],
        second.hyperedge_index[0] + first.num_nodes,
    )
    assert torch.equal(
        second_incidence[1],
        second.hyperedge_index[1] + first.num_hyperedges,
    )
    assert torch.equal(batch.num_hyperedges, torch.tensor([2, 3]))


@pytest.mark.parametrize(
    "loader_name",
    ["train_dataloader", "val_dataloader", "test_dataloader"],
)
def test_transductive_phase_loaders_preserve_native_hypergraph_fields(
    loader_name: str,
) -> None:
    """Every phase emits the same one-graph native batch with node masks."""
    data = make_synthetic_hypergraph_data(
        seed=7,
        num_nodes=9,
        num_hyperedges=4,
    )
    module = GraphDataModule(
        dataset_train=[data],
        learning_setting="transductive",
        batch_size=1,
        num_workers=0,
    )

    batch = next(iter(getattr(module, loader_name)()))

    assert isinstance(batch, Batch)
    assert isinstance(batch, HypergraphData)
    assert batch.num_graphs == 1
    assert torch.equal(batch.x, data.x)
    assert torch.equal(batch.y, data.y)
    assert torch.equal(batch.hyperedge_index, data.hyperedge_index)
    assert torch.equal(batch.num_hyperedges, torch.tensor([data.num_hyperedges]))
    for mask_name in ("train_mask", "val_mask", "test_mask"):
        mask = batch[mask_name]
        assert mask.dtype == torch.bool
        assert mask.shape == (data.num_nodes,)
        assert torch.equal(mask, data[mask_name])


def test_transductive_hypergraph_loading_requires_one_graph_per_batch() -> None:
    """The native graph module enforces the v1 transductive batch size."""
    data = make_synthetic_hypergraph_data()

    with pytest.raises(
        ValueError,
        match="^transductive graph loading requires batch_size=1$",
    ):
        GraphDataModule(
            dataset_train=[data],
            learning_setting="transductive",
            batch_size=2,
        )
