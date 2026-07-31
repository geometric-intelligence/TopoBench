"""Base class for native homogeneous node and graph readouts."""

from abc import abstractmethod
from typing import Any

import torch
from torch import Tensor, nn
from torch_geometric.data import Data
from torch_geometric.utils import scatter


class AbstractZeroCellReadOut(nn.Module):
    """Read node embeddings and optionally pool them to graph embeddings.

    The historical class name remains public for configuration stability. Its
    runtime contract is native PyG: embeddings are ``model_out["x"]`` and graph
    membership is ``model_out["batch"]``.
    """

    def __init__(
        self,
        hidden_dim: int,
        out_channels: int,
        task_level: str,
        pooling_type: str = "sum",
        logits_linear_layer: bool = True,
        **kwargs: Any,
    ) -> None:
        super().__init__()
        del kwargs
        if task_level not in {"graph", "node", "node_inductive"}:
            raise ValueError(
                "task_level must be graph, node, or node_inductive"
            )
        if pooling_type not in {"max", "sum", "mean"}:
            raise ValueError("pooling_type must be max, sum, or mean")
        self.linear = (
            nn.Linear(hidden_dim, out_channels)
            if hidden_dim != out_channels or logits_linear_layer
            else nn.Identity()
        )
        self.task_level = task_level
        self.logits_linear_layer = logits_linear_layer
        self.pooling_type = pooling_type

    def __repr__(self) -> str:
        return (
            f"{self.__class__.__name__}(task_level={self.task_level}, "
            f"pooling_type={self.pooling_type})"
        )

    def __call__(
        self, model_out: dict[str, Any], batch: Data
    ) -> dict[str, Any]:
        """Run the specialized readout then provide logits when absent."""
        model_out = super().__call__(model_out, batch)
        if model_out.get("logits") is None:
            model_out["logits"] = self.compute_logits(
                model_out.get("x"), model_out.get("batch")
            )
        return model_out

    def compute_logits(
        self, x: object, batch_index: object
    ) -> Tensor:
        """Apply graph pooling when configured, then the output head."""
        if not isinstance(x, Tensor) or x.ndim != 2:
            raise TypeError("model_out['x'] must be a rank-2 tensor")
        if self.task_level == "graph":
            if not isinstance(batch_index, Tensor):
                raise TypeError(
                    "model_out['batch'] must be a tensor for graph readout"
                )
            if (
                batch_index.ndim != 1
                or batch_index.dtype is not torch.long
                or batch_index.size(0) != x.size(0)
            ):
                raise ValueError(
                    "model_out['batch'] must be a rank-1 long tensor "
                    "matching model_out['x']"
                )
            x = scatter(
                x,
                batch_index,
                dim=0,
                reduce=self.pooling_type,
            )
        return self.linear(x)

    @abstractmethod
    def forward(
        self, model_out: dict[str, Any], batch: Data
    ) -> dict[str, Any]:
        """Apply readout-specific feature transformations."""


__all__ = ["AbstractZeroCellReadOut"]
