"""Native PyG hypergraph backbone adapter."""

from __future__ import annotations

from numbers import Integral

import torch
from torch import Tensor, nn
from torch_geometric.data import Data

from topobench.dataloader.graph import require_hypergraph_validation


def _require_tensor(batch: Data, field: str) -> Tensor:
    value = batch.get(field)
    if not isinstance(value, Tensor):
        raise TypeError(f"batch.{field} must be a tensor")
    return value


def _hyperedge_counts(
    batch: Data,
    *,
    device: torch.device,
) -> int:
    """Return the graph count without scanning incidence storage."""
    value = batch.get("num_hyperedges")
    if isinstance(value, Tensor):
        if value.ndim != 1:
            raise ValueError("batch.num_hyperedges must be rank-1")
        if value.dtype is not torch.long:
            raise TypeError("batch.num_hyperedges must use torch.long")
        if value.device != device:
            raise ValueError(
                "batch.num_hyperedges and batch.hyperedge_index "
                "must use the same device"
            )
        if not value.numel():
            raise ValueError(
                "batch.num_hyperedges must contain one count per graph"
            )
        return value.size(0)
    if isinstance(value, Integral) and not isinstance(value, bool):
        count = int(value)
        if count < 1:
            raise ValueError("batch.num_hyperedges counts must be positive")
        return 1
    raise TypeError(
        "batch.num_hyperedges must be an integer or rank-1 long tensor"
    )


def _validated_fields(
    batch: Data,
) -> tuple[Tensor, Tensor, Tensor, Tensor]:
    if not isinstance(batch, Data):
        raise TypeError("batch must be a torch_geometric.data.Data")

    x = _require_tensor(batch, "x")
    if x.ndim != 2:
        raise ValueError("batch.x must be rank-2")
    if not x.is_floating_point():
        raise TypeError("batch.x must use a floating dtype")

    if x.size(1) == 0:
        raise ValueError("batch.x must have positive feature width")
    labels = _require_tensor(batch, "y")
    if labels.ndim != 1:
        raise ValueError("batch.y must be rank-1")
    if labels.dtype is not torch.long:
        raise TypeError("batch.y must use torch.long")
    if labels.size(0) != x.size(0):
        raise ValueError("batch.y must contain one label per node")
    if labels.device != x.device:
        raise ValueError("batch.x and batch.y must use the same device")

    batch_index = _require_tensor(batch, "batch")
    if batch_index.ndim != 1:
        raise ValueError("batch.batch must be rank-1")
    if batch_index.dtype is not torch.long:
        raise TypeError("batch.batch must use torch.long")
    if batch_index.size(0) != x.size(0):
        raise ValueError("batch.batch length must match batch.x rows")
    if batch_index.device != x.device:
        raise ValueError("batch.x and batch.batch must use the same device")
    if not batch_index.numel():
        raise ValueError("batch.batch must contain graph IDs")

    hyperedge_index = _require_tensor(batch, "hyperedge_index")
    if hyperedge_index.layout != torch.strided:
        raise TypeError("batch.hyperedge_index must be dense")
    if hyperedge_index.ndim != 2 or hyperedge_index.size(0) != 2:
        raise ValueError("batch.hyperedge_index must have shape [2, M]")
    if hyperedge_index.dtype is not torch.long:
        raise TypeError("batch.hyperedge_index must use torch.long")
    if hyperedge_index.device != x.device:
        raise ValueError(
            "batch.x and batch.hyperedge_index must use the same device"
        )

    graph_count = _hyperedge_counts(
        batch,
        device=hyperedge_index.device,
    )
    if graph_count > x.size(0):
        raise ValueError(
            "batch.num_hyperedges contains more graphs than batch.x rows"
        )
    if not hyperedge_index.numel():
        raise ValueError("batch.hyperedge_index must contain every hyperedge")

    require_hypergraph_validation(batch)

    return x, hyperedge_index, labels, batch_index


class HypergraphWrapper(nn.Module):
    """Translate native PyG hypergraph fields into a backbone call."""

    def __init__(self, backbone: nn.Module) -> None:
        super().__init__()
        if not isinstance(backbone, nn.Module):
            raise TypeError("backbone must be a torch.nn.Module")
        self.backbone = backbone

    def forward(self, batch: Data) -> dict[str, Tensor]:
        """Validate transactionally and return the exact readout contract."""
        x, hyperedge_index, labels, batch_index = _validated_fields(batch)
        output = self.backbone(x, hyperedge_index)
        if not isinstance(output, Tensor):
            raise TypeError("hypergraph backbone must return a tensor")
        if output.ndim != 2:
            raise ValueError("hypergraph backbone output must be rank-2")
        if output.size(0) != x.size(0):
            raise ValueError(
                "hypergraph backbone output must contain one row per node"
            )
        if not output.is_floating_point():
            raise TypeError("hypergraph backbone output must be floating")
        return {"x": output, "labels": labels, "batch": batch_index}


__all__ = ["HypergraphWrapper"]
