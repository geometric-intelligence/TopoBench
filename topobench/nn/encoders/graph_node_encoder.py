"""Native homogeneous graph node feature encoder."""

from __future__ import annotations

from numbers import Integral

import torch
from torch import Tensor
from torch_geometric.data import Data, HeteroData
from torch_geometric.nn import GraphNorm

from topobench.nn.encoders.base import AbstractFeatureEncoder


class GraphNodeFeatureEncoder(AbstractFeatureEncoder):
    """Normalize and project native homogeneous node features.

    Parameters
    ----------
    in_channels : int
        Width of the input ``data.x`` tensor.
    out_channels : int
        Width assigned to the encoded ``data.x`` tensor.
    dropout : float, default=0.0
        Dropout probability applied after projection and activation.
    """

    def __init__(
        self,
        in_channels: int,
        out_channels: int,
        dropout: float = 0.0,
    ) -> None:
        super().__init__()
        for name, value in (
            ("in_channels", in_channels),
            ("out_channels", out_channels),
        ):
            if isinstance(value, bool) or not isinstance(value, Integral):
                raise TypeError(f"{name} must be an integer")
        in_channels = int(in_channels)
        out_channels = int(out_channels)
        self.norm = GraphNorm(in_channels)
        self.projection = torch.nn.Linear(in_channels, out_channels)
        self.activation = torch.nn.ReLU()
        self.dropout = torch.nn.Dropout(dropout)

    def forward(self, data: Data) -> Data:
        """Replace ``data.x`` with graph-aware encoded node features."""
        if not isinstance(data, Data) or isinstance(data, HeteroData):
            raise TypeError(
                "GraphNodeFeatureEncoder requires homogeneous Data"
            )

        x = data.get("x")
        if not isinstance(x, Tensor) or x.ndim != 2:
            raise ValueError("data.x must be a rank-2 tensor")
        if not x.is_floating_point():
            raise TypeError("data.x must have a floating dtype")

        batch = data.get("batch")
        if batch is None:
            batch = torch.zeros(x.shape[0], dtype=torch.long, device=x.device)
        data.x = self.dropout(
            self.activation(self.projection(self.norm(x, batch=batch)))
        )
        return data


__all__ = ["GraphNodeFeatureEncoder"]
