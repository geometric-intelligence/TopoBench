"""Heterogeneous Graph Transformer for batched cell complexes."""

from __future__ import annotations

import torch
from torch_geometric.data import Data

from topobench.data.utils.utils import get_routes_from_neighborhoods
from topobench.nn.backbones.heterogeneous.hgt import HGTBackbone


class CellHGT(HGTBackbone):
    """Apply heterogeneous attention across cell ranks and incidence relations.

    Parameters
    ----------
    hidden_channels : int
        Number of feature channels for every cell rank.
    num_layers : int
        Number of stacked HGT layers.
    heads : int
        Number of attention heads. Must be positive and divide
        ``hidden_channels`` evenly.
    neighborhoods : sequence of str
        Ordered incidence-neighborhood names to expose as edge types.
    max_rank : int, optional
        Highest cell rank represented by the backbone.
    dropout : float, optional
        Dropout probability applied after each updated rank.
    activation : str, optional
        Name of the activation applied after layer normalization.
    """

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
        self.max_rank = max_rank
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
        if any(
            rank < 0 or rank > max_rank
            for route in self.routes
            for rank in route
        ):
            raise ValueError(
                "Neighborhood route ranks must be between 0 and max_rank"
            )

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
        super().__init__(
            metadata=(self.node_types, self.edge_types),
            hidden_channels=hidden_channels,
            num_layers=num_layers,
            heads=heads,
            dropout=dropout,
            activation=activation,
            _canonicalize_relations=False,
        )

    @staticmethod
    def node_type(rank: int) -> str:
        """Return the PyG node-type name for a cell rank.

        Parameters
        ----------
        rank : int
            Cell rank.

        Returns
        -------
        str
            Node-type name in ``rank_<rank>`` form.
        """
        return f"rank_{rank}"

    def to_heterogeneous_inputs(self, batch: Data):
        """Convert a batched TopoBench complex to PyG HGT dictionaries.

        Parameters
        ----------
        batch : torch_geometric.data.Data
            Batched complex containing per-rank features and configured sparse
            incidence neighborhoods.

        Returns
        -------
        tuple of dict
            Per-node-type feature tensors and per-edge-type PyG edge indices.
        """
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
                matrix.coalesce().indices().flip(0).contiguous().long()
            )
        return x_dict, edge_index_dict

    def forward(  # type: ignore[override]
        self,
        batch: Data,
    ) -> dict[int, torch.Tensor]:
        """Apply all HGT layers to one disjoint-union mini-batch.

        Parameters
        ----------
        batch : torch_geometric.data.Data
            Batched cell complex.

        Returns
        -------
        dict of int to torch.Tensor
            Updated feature tensor for every represented cell rank.
        """
        x_dict, edge_index_dict = self.to_heterogeneous_inputs(batch)
        output = super().forward(x_dict, edge_index_dict)
        return {
            rank: output[self.node_type(rank)]
            for rank in range(self.max_rank + 1)
        }
