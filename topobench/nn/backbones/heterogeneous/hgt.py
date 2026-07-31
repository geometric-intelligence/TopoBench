"""Metadata-driven Heterogeneous Graph Transformer backbone."""

from __future__ import annotations

from collections.abc import Mapping

import torch
from torch import Tensor
from torch_geometric.nn import HGTConv
from torch_geometric.typing import EdgeType, Metadata

from topobench.nn.activation import make_activation
from topobench.nn.backbones.heterogeneous.common import (
    _HeterogeneousMetadataAdapter,
    validate_backbone_arguments,
    validate_forward_dictionaries,
)


class HGTBackbone(torch.nn.Module):
    """Apply stacked HGT message passing to typed feature dictionaries."""

    def __init__(
        self,
        metadata: Metadata,
        hidden_channels: int,
        num_layers: int,
        heads: int,
        dropout: float,
        activation: str,
    ) -> None:
        super().__init__()
        node_types, edge_types = metadata
        normalized_metadata = validate_backbone_arguments(
            node_types=node_types,
            edge_types=edge_types,
            hidden_channels=hidden_channels,
            num_layers=num_layers,
            heads=heads,
            dropout=dropout,
        )
        self.metadata_adapter = _HeterogeneousMetadataAdapter(
            normalized_metadata
        )
        self.node_types = list(self.metadata_adapter.node_types)
        self.edge_types = list(self.metadata_adapter.edge_types)
        self.metadata: Metadata = (self.node_types, self.edge_types)
        self.internal_metadata = self.metadata_adapter.internal_metadata
        self.hidden_channels = int(hidden_channels)
        self.out_channels = self.hidden_channels
        self.num_layers = int(num_layers)
        self.heads = int(heads)
        self.dropout_probability = float(dropout)
        self.activation_name = activation
        self.convs = torch.nn.ModuleList(
            [
                HGTConv(
                    in_channels=self.hidden_channels,
                    out_channels=self.hidden_channels,
                    metadata=self.internal_metadata,
                    heads=self.heads,
                )
                for _ in range(self.num_layers)
            ]
        )
        internal_node_types = self.internal_metadata[0]
        self.norms = torch.nn.ModuleList(
            [
                torch.nn.ModuleDict(
                    {
                        node_type: torch.nn.LayerNorm(self.hidden_channels)
                        for node_type in internal_node_types
                    }
                )
                for _ in range(self.num_layers)
            ]
        )
        self.activation = make_activation(activation)
        self.dropout = torch.nn.Dropout(self.dropout_probability)

    def forward(
        self,
        x_dict: Mapping[str, Tensor],
        edge_index_dict: Mapping[EdgeType, Tensor],
    ) -> dict[str, Tensor]:
        """Apply all HGT layers and return externally named features."""
        validate_forward_dictionaries(
            x_dict=x_dict,
            edge_index_dict=edge_index_dict,
            node_types=self.node_types,
            edge_types=self.edge_types,
            hidden_channels=self.hidden_channels,
        )
        current = self.metadata_adapter.to_internal_x_dict(x_dict)
        internal_edges = self.metadata_adapter.to_internal_edge_index_dict(
            edge_index_dict
        )
        for conv, norms in zip(self.convs, self.norms, strict=True):
            updates = conv(current, internal_edges) if internal_edges else {}
            current = {
                node_type: (
                    features
                    if updates.get(node_type) is None
                    else self.dropout(
                        self.activation(norms[node_type](updates[node_type]))
                    )
                )
                for node_type, features in current.items()
            }
        return self.metadata_adapter.to_external_x_dict(current)


__all__ = ["HGTBackbone"]
