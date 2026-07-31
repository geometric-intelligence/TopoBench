"""Target-node classifier for native heterogeneous model outputs."""

from __future__ import annotations

from collections.abc import Mapping
from numbers import Integral

import torch
from torch import Tensor
from torch_geometric.data import HeteroData


def _positive_dimension(name: str, value: object, *, minimum: int) -> int:
    """Validate one eager classifier dimension."""
    if isinstance(value, bool) or not isinstance(value, Integral):
        raise TypeError(f"{name} must be a non-boolean integer")
    normalized = int(value)
    if normalized < minimum:
        raise ValueError(f"{name} must be at least {minimum}")
    return normalized


class HeterogeneousNodeReadout(torch.nn.Module):
    """Classify every embedding of one configured target node type.

    Mask and neighbor-seed selection intentionally happen later in the
    supervision adapter.

    Parameters
    ----------
    target_node_type : str
        Node type to classify.
    hidden_channels : int
        Width of every selected target embedding.
    out_channels : int
        Number of classification logits. Must be at least two.
    """

    task_level = "node"

    def __init__(
        self,
        target_node_type: str,
        hidden_channels: int,
        out_channels: int,
    ) -> None:
        super().__init__()
        if not isinstance(target_node_type, str):
            raise TypeError("target_node_type must be a non-empty string")
        if not target_node_type.strip():
            raise ValueError("target_node_type must be a non-empty string")
        self.target_node_type = target_node_type
        self.hidden_channels = _positive_dimension(
            "hidden_channels", hidden_channels, minimum=1
        )
        self.out_channels = _positive_dimension(
            "out_channels", out_channels, minimum=2
        )
        self.linear = torch.nn.Linear(
            self.hidden_channels,
            self.out_channels,
        )

    def forward(
        self,
        model_out: dict[str, object],
        batch: HeteroData,
    ) -> dict[str, object]:
        """Add unfiltered target-node logits to ``model_out``.

        Parameters
        ----------
        model_out : dict[str, object]
            Wrapper output containing a complete ``x_dict``.
        batch : torch_geometric.data.HeteroData
            Full graph or sampled batch represented by ``x_dict``.

        Returns
        -------
        dict[str, object]
            The same mapping with only ``logits`` added or replaced.
        """
        embeddings = self._validate_inputs(model_out, batch)
        logits = self.linear(embeddings)
        model_out["logits"] = logits
        return model_out

    def _validate_inputs(
        self,
        model_out: dict[str, object],
        batch: HeteroData,
    ) -> Tensor:
        """Validate without mutating ``model_out``."""
        if not isinstance(model_out, dict):
            raise TypeError("model_out must be a dictionary")
        if not isinstance(batch, HeteroData):
            raise TypeError(
                "HeterogeneousNodeReadout requires native HeteroData batch"
            )
        if self.target_node_type not in batch.node_types:
            raise ValueError(
                f"batch is missing target node store {self.target_node_type!r}"
            )
        x_dict = model_out.get("x_dict")
        if not isinstance(x_dict, Mapping):
            raise TypeError("model_out must contain an x_dict mapping")
        if self.target_node_type not in x_dict:
            raise ValueError(
                f"no embeddings for target node type {self.target_node_type!r}"
            )
        embeddings = x_dict[self.target_node_type]
        if not isinstance(embeddings, Tensor):
            raise TypeError(
                f"target embeddings {self.target_node_type!r} must be a tensor"
            )
        if embeddings.ndim != 2:
            raise ValueError(
                f"target embeddings {self.target_node_type!r} must be rank-2"
            )
        if embeddings.size(1) != self.hidden_channels:
            raise ValueError(
                f"target embeddings {self.target_node_type!r} width "
                f"must be {self.hidden_channels}; received "
                f"{embeddings.size(1)}"
            )
        target_count = batch[self.target_node_type].num_nodes
        if embeddings.size(0) != target_count:
            raise ValueError(
                f"target embeddings {self.target_node_type!r} node count "
                f"must be {target_count}; received {embeddings.size(0)}"
            )
        return embeddings


__all__ = ["HeterogeneousNodeReadout"]
