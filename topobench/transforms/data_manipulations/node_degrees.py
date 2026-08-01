"""Native graph node-degree transform."""

from collections.abc import Sequence

import torch
from torch_geometric.data import Data
from torch_geometric.transforms import BaseTransform
from torch_geometric.utils import degree


class NodeDegrees(BaseTransform):
    """Add outgoing graph degrees as ``node_degrees``."""

    def __init__(
        self,
        selected_fields: Sequence[str] = ("edge_index",),
        **kwargs: object,
    ) -> None:
        super().__init__()
        unsupported_fields = set(selected_fields) - {"edge_index"}
        if unsupported_fields:
            raise ValueError(
                "NodeDegrees supports only native graph edge_index; received "
                f"{sorted(unsupported_fields)}"
            )
        self.type = "node_degrees"
        self.selected_fields = tuple(selected_fields)
        self.parameters = {
            "selected_fields": list(self.selected_fields),
            **kwargs,
        }

    def __repr__(self) -> str:
        return (
            f"{self.__class__.__name__}(type={self.type!r}, "
            f"parameters={self.parameters!r})"
        )

    def forward(self, data: Data) -> Data:
        """Add the outgoing degree for every node when requested."""
        edge_index = data.get("edge_index")
        if "edge_index" not in self.selected_fields or edge_index is None:
            return data

        data.node_degrees = degree(
            edge_index[0],
            num_nodes=data.num_nodes,
            dtype=torch.float,
        ).unsqueeze(1)
        return data
