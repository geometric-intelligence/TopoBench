"""Self-contained MLP readout over native graph node embeddings."""

from __future__ import annotations

from collections.abc import Iterable
from typing import Any

import torch
from torch import Tensor, nn
from torch_geometric.data import Data
from torch_geometric.nn.resolver import (
    activation_resolver,
    normalization_resolver,
)
from torch_geometric.utils import scatter


class MLPReadout(nn.Module):
    """Apply an MLP to nodes and optionally pool to graph logits."""

    def __init__(
        self,
        in_channels: int,
        hidden_layers: int | Iterable[int],
        out_channels: int,
        pooling_type: str = "sum",
        dropout: float = 0.25,
        norm: str | None = None,
        norm_kwargs: dict[str, Any] | None = None,
        act: str | None = "relu",
        act_kwargs: dict[str, Any] | None = None,
        final_act: str | None = None,
        final_act_kwargs: dict[str, Any] | None = None,
        task_level: str | None = None,
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
        if isinstance(hidden_layers, int):
            dimensions = [hidden_layers]
        else:
            dimensions = list(hidden_layers)
        if any(
            not isinstance(dimension, int) or dimension < 1
            for dimension in dimensions
        ):
            raise ValueError("hidden_layers must contain positive integers")

        self.in_channels = in_channels
        self.hidden_layers = dimensions
        self.out_channels = out_channels
        self.dropout = dropout
        self.task_level = task_level
        self.pooling_type = pooling_type

        layers: list[nn.Module] = []
        input_dim = in_channels
        for hidden_dim in dimensions:
            normalization = (
                normalization_resolver(
                    norm,
                    hidden_dim,
                    **(norm_kwargs or {}),
                )
                if norm is not None
                else nn.Identity()
            )
            activation = (
                activation_resolver(act, **(act_kwargs or {}))
                if act is not None
                else nn.Identity()
            )
            layers.append(
                nn.Sequential(
                    nn.Linear(input_dim, hidden_dim),
                    normalization,
                    activation,
                    nn.AlphaDropout(p=dropout, inplace=True),
                )
            )
            input_dim = hidden_dim
        self.mlp_layers = nn.Sequential(*layers)
        self.output_layer = nn.Linear(input_dim, out_channels)
        self.final_act = (
            activation_resolver(final_act, **(final_act_kwargs or {}))
            if final_act is not None
            else nn.Identity()
        )

    def forward(
        self, model_out: dict[str, Any], batch: Data
    ) -> dict[str, Any]:
        """Compute native node or pooled graph logits in place."""
        del batch
        x = model_out.get("x")
        if not isinstance(x, Tensor) or x.ndim != 2:
            raise TypeError("model_out['x'] must be a rank-2 tensor")
        x = self.mlp_layers(x)
        if self.task_level == "graph":
            batch_index = model_out.get("batch")
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
        x = self.final_act(self.output_layer(x))
        model_out["x"] = x
        model_out["logits"] = x
        return model_out


__all__ = ["MLPReadout"]
