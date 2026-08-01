"""Deterministic native hypergraph data for network-free lifecycles."""

from __future__ import annotations

import torch
from torch_geometric.data import InMemoryDataset

from topobench.data.hypergraph import (
    HYPERGRAPH_REPRESENTATION_VERSION,
    HypergraphData,
    validate_hypergraph_node_data,
)


def make_synthetic_hypergraph_data(
    *,
    seed: int = 0,
    num_nodes: int = 12,
    num_hyperedges: int = 5,
) -> HypergraphData:
    """Build a small deterministic hypergraph node-classification example.

    The fixture uses only native dense PyTorch tensors. Every node belongs to
    two cyclic hyperedges, every hyperedge is nonempty, and node IDs modulo
    three define a complete fixed train/validation/test partition.

    Parameters
    ----------
    seed : int, default=0
        Seed for a factory-local random generator.
    num_nodes : int, default=12
        Number of nodes. Must be at least three and no smaller than the number
        of hyperedges.
    num_hyperedges : int, default=5
        Positive number of nonempty hyperedges.

    Returns
    -------
    HypergraphData
        A fully validated native node-classification example.
    """
    seed = int(seed)
    num_nodes = int(num_nodes)
    num_hyperedges = int(num_hyperedges)
    if num_nodes < 3:
        raise ValueError("num_nodes must be at least 3")
    if num_hyperedges < 1 or num_hyperedges > num_nodes:
        raise ValueError(
            "num_hyperedges must satisfy 1 <= num_hyperedges <= num_nodes"
        )

    generator = torch.Generator().manual_seed(seed)
    node_ids = torch.arange(num_nodes, dtype=torch.long)
    labels = node_ids % 2
    x = 0.05 * torch.randn(num_nodes, 4, generator=generator)
    x[node_ids, labels] += 1.0

    hyperedge_index = torch.stack(
        [
            torch.cat([node_ids, node_ids]),
            torch.cat(
                [
                    node_ids % num_hyperedges,
                    (node_ids + 1) % num_hyperedges,
                ]
            ),
        ]
    )

    split_ids = node_ids % 3
    data = HypergraphData(
        x=x,
        hyperedge_index=hyperedge_index,
        num_hyperedges=num_hyperedges,
        representation_version=HYPERGRAPH_REPRESENTATION_VERSION,
        y=labels,
        train_mask=split_ids == 0,
        val_mask=split_ids == 1,
        test_mask=split_ids == 2,
    )
    return validate_hypergraph_node_data(data)


class SyntheticHypergraphDataset(InMemoryDataset):
    """Package one deterministic native hypergraph as a PyG dataset."""

    def __init__(
        self,
        *,
        seed: int = 0,
        num_nodes: int = 12,
        num_hyperedges: int = 5,
    ) -> None:
        self.seed = int(seed)
        self.num_fixture_nodes = int(num_nodes)
        self.num_fixture_hyperedges = int(num_hyperedges)
        super().__init__(root=None)
        data = make_synthetic_hypergraph_data(
            seed=self.seed,
            num_nodes=self.num_fixture_nodes,
            num_hyperedges=self.num_fixture_hyperedges,
        )
        self.data, self.slices = self.collate([data])


__all__ = [
    "SyntheticHypergraphDataset",
    "make_synthetic_hypergraph_data",
]
