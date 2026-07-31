"""Deterministic constant node features for homogeneous graphs."""

from __future__ import annotations

from numbers import Integral

import torch
from torch import Tensor
from torch_geometric.data import Data, HeteroData
from torch_geometric.transforms import BaseTransform


class ConstantNodeFeatures(BaseTransform):
    """Replace homogeneous node features with deterministic ones.

    Parameters
    ----------
    num_features : int
        Positive width of the constant node feature matrix.
    transform_name : str | None, optional
        Name supplied by the TopoBench transform registry.
    transform_type : str | None, optional
        Category supplied by the TopoBench transform registry.
    """

    def __init__(
        self,
        num_features: int,
        transform_name: str | None = None,
        transform_type: str | None = None,
    ) -> None:
        super().__init__()
        if (
            isinstance(num_features, bool)
            or not isinstance(num_features, Integral)
        ):
            raise TypeError("num_features must be an integer")
        if num_features <= 0:
            raise ValueError("num_features must be positive")
        self.num_features = int(num_features)
        self.transform_name = transform_name
        self.transform_type = transform_type

    def forward(self, data: Data) -> Data:
        """Replace ``data.x`` with a float matrix filled with ones."""
        if not isinstance(data, Data) or isinstance(data, HeteroData):
            raise TypeError("ConstantNodeFeatures requires homogeneous Data")
        num_nodes = data.num_nodes
        if num_nodes is None:
            raise ValueError("data.num_nodes is required for constant features")

        x = data.get("x")
        edge_index = data.get("edge_index")
        device = (
            x.device
            if isinstance(x, Tensor)
            else edge_index.device
            if isinstance(edge_index, Tensor)
            else None
        )
        data.x = torch.ones(
            (num_nodes, self.num_features),
            dtype=torch.float,
            device=device,
        )
        return data


__all__ = ["ConstantNodeFeatures"]
