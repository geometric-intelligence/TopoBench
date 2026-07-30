"""Heterogeneous Graph Transformer for batched cell complexes."""

from __future__ import annotations

import torch
from torch_geometric.data import Data

from topobench.data.utils import get_routes_from_neighborhoods


class CellHGT(torch.nn.Module):
    """Map cell ranks and incidence relations to a heterogeneous graph."""

    def __init__(
        self,
        hidden_channels: int,
        num_layers: int,
        heads: int,
        neighborhoods,
        max_rank: int = 2,
        dropout: float = 0.0,
        activation: str = "relu",
    ) -> None:
        super().__init__()
        if hidden_channels % heads != 0:
            raise ValueError(
                "hidden_channels must be divisible by the number of heads"
            )
        if num_layers < 1:
            raise ValueError("num_layers must be at least 1")

        self.hidden_channels = hidden_channels
        self.out_channels = hidden_channels
        self.num_layers = num_layers
        self.heads = heads
        self.max_rank = max_rank
        self.dropout_probability = dropout
        self.activation_name = activation
        self.neighborhoods = list(neighborhoods)

        if not self.neighborhoods:
            raise ValueError("At least one incidence neighborhood is required")
        if any("incidence" not in name for name in self.neighborhoods):
            raise ValueError(
                "CellHGT version 1 supports incidence neighborhoods only"
            )
        if len(set(self.neighborhoods)) != len(self.neighborhoods):
            raise ValueError("Neighborhood names must be unique")

        self.routes = [
            tuple(route)
            for route in get_routes_from_neighborhoods(self.neighborhoods)
        ]
        if any(max(route) > max_rank for route in self.routes):
            raise ValueError("A neighborhood route exceeds max_rank")

        self.node_types = [
            self.node_type(rank) for rank in range(self.max_rank + 1)
        ]
        self.edge_types = [
            (
                self.node_type(src_rank),
                neighborhood,
                self.node_type(dst_rank),
            )
            for neighborhood, (src_rank, dst_rank) in zip(
                self.neighborhoods, self.routes, strict=True
            )
        ]
        self.metadata = (self.node_types, self.edge_types)

    @staticmethod
    def node_type(rank: int) -> str:
        """Return the PyG node-type name for a cell rank."""
        return f"rank_{rank}"

    def to_heterogeneous_inputs(self, batch: Data):
        """Convert a batched TopoBench complex to PyG HGT dictionaries."""
        x_dict = {}
        for rank in range(self.max_rank + 1):
            field = f"x_{rank}"
            if batch.get(field) is None:
                raise KeyError(f"Missing cell feature field: {field}")
            x_dict[self.node_type(rank)] = batch[field]

        edge_index_dict = {}
        for neighborhood, edge_type in zip(
            self.neighborhoods, self.edge_types, strict=True
        ):
            matrix = batch.get(neighborhood)
            if matrix is None:
                raise KeyError(
                    f"Missing configured neighborhood: {neighborhood}"
                )
            if not matrix.is_sparse:
                raise TypeError(
                    f"Neighborhood {neighborhood} must be a sparse COO tensor"
                )
            edge_index_dict[edge_type] = (
                matrix.coalesce()
                .indices()
                .flip(0)
                .contiguous()
                .long()
            )
        return x_dict, edge_index_dict

    def forward(self, batch: Data):
        """Apply HGT layers; implemented in the next TDD task."""
        raise NotImplementedError
