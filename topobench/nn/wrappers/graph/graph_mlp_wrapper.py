"""Native homogeneous graph adapter for GraphMLP backbones."""

from __future__ import annotations

from torch import Tensor, nn
from torch_geometric.data import Data

from .gnn_wrapper import (
    EdgeMode,
    _edge_kwargs,
    _validate_edge_mode,
    _validated_graph_fields,
)


class GraphMLPWrapper(nn.Module):
    """Translate native PyG features into a GraphMLP backbone call."""

    def __init__(
        self,
        backbone: nn.Module,
        edge_attr_mode: EdgeMode,
        edge_weight_mode: EdgeMode,
    ) -> None:
        super().__init__()
        if not isinstance(backbone, nn.Module):
            raise TypeError("backbone must be a torch.nn.Module")
        self.backbone = backbone
        self.edge_modes = {
            "edge_attr": _validate_edge_mode("edge_attr", edge_attr_mode),
            "edge_weight": _validate_edge_mode(
                "edge_weight", edge_weight_mode
            ),
        }

    def forward(self, batch: Data) -> dict[str, Tensor | None]:
        """Return only embeddings, labels, and native batch membership."""
        x, _edge_index, labels, batch_index = _validated_graph_fields(batch)
        output = self.backbone(x, **_edge_kwargs(batch, self.edge_modes))
        embeddings = output[0] if isinstance(output, tuple) else output
        if not isinstance(embeddings, Tensor):
            raise TypeError(
                "GraphMLP backbone must return a tensor or tensor-first tuple"
            )
        return {"x": embeddings, "labels": labels, "batch": batch_index}


__all__ = ["GraphMLPWrapper"]
