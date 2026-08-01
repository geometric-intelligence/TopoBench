"""Native homogeneous graph adapter for GNN backbones."""

from __future__ import annotations

from typing import Literal

import torch
from torch import Tensor, nn
from torch_geometric.data import Data

EdgeMode = Literal["consume", "ignore", "reject"]
_EDGE_MODES = frozenset({"consume", "ignore", "reject"})


def _validate_edge_mode(field: str, mode: object) -> EdgeMode:
    """Return one validated optional-edge-field mode."""
    if not isinstance(mode, str):
        raise TypeError(f"{field}_mode must be consume, ignore, or reject")
    if mode not in _EDGE_MODES:
        raise ValueError(f"{field}_mode must be consume, ignore, or reject")
    return mode  # type: ignore[return-value]


def _require_tensor(data: Data, field: str) -> Tensor:
    """Read one required native tensor field."""
    value = data.get(field)
    if not isinstance(value, Tensor):
        raise TypeError(f"batch.{field} must be a tensor")
    return value


def _validate_node_targets(labels: Tensor, node_count: int) -> None:
    """Validate the only supported node-level target contract."""
    if labels.size(0) != node_count:
        raise ValueError(
            "batch.y count must match batch.x rows for node targets"
        )
    if labels.dtype is not torch.long:
        raise TypeError(
            "batch.y must have dtype torch.long for node classification"
        )
    if labels.ndim != 1:
        raise ValueError("batch.y must be rank-1 for node classification")


def _validate_targets(
    labels: Tensor,
    *,
    node_count: int,
    graph_count: int | None,
) -> None:
    """Validate targets inferred from native graph membership and counts."""
    if labels.ndim == 0:
        raise ValueError("batch.y must have a leading example dimension")

    if graph_count is None:
        _validate_node_targets(labels, node_count)
        return

    if labels.size(0) == graph_count:
        if labels.dtype is torch.long:
            if labels.ndim != 1:
                raise ValueError(
                    "batch.y must be rank-1 for graph classification"
                )
            return
        if labels.is_floating_point():
            if labels.ndim != 1 and (labels.ndim != 2 or labels.size(1) != 1):
                raise ValueError(
                    "batch.y must have shape [B] or [B, 1] for graph scalar "
                    "regression"
                )
            return
        raise TypeError(
            "batch.y must be floating for graph scalar regression or have "
            "dtype torch.long for graph classification"
        )

    if labels.size(0) == node_count:
        _validate_node_targets(labels, node_count)
        return

    raise ValueError(
        "batch.y count must match graphs or nodes described by batch.batch"
    )


def _validated_graph_fields(
    data: Data,
) -> tuple[Tensor, Tensor, Tensor, Tensor | None]:
    """Validate native homogeneous fields before any backbone call."""
    if not isinstance(data, Data):
        raise TypeError("batch must be a torch_geometric.data.Data")

    x = _require_tensor(data, "x")
    if x.ndim != 2 or not x.is_floating_point():
        raise ValueError("batch.x must be a rank-2 floating tensor")

    edge_index = _require_tensor(data, "edge_index")
    if edge_index.ndim != 2 or edge_index.size(0) != 2:
        raise ValueError("batch.edge_index must have shape [2, E]")
    if edge_index.dtype is not torch.long:
        raise TypeError("batch.edge_index must have dtype torch.long")
    if edge_index.numel() and (
        int(edge_index.min()) < 0 or int(edge_index.max()) >= x.size(0)
    ):
        raise ValueError("batch.edge_index contains an invalid node index")

    labels = _require_tensor(data, "y")

    batch_index = data.get("batch")
    if batch_index is None:
        ptr = data.get("ptr")
        has_multiple_boundaries = (
            isinstance(ptr, Tensor) and ptr.ndim == 1 and ptr.numel() > 2
        )
        if not has_multiple_boundaries:
            num_graphs = getattr(data, "num_graphs", 1)
            has_multiple_boundaries = (
                isinstance(num_graphs, int) and num_graphs > 1
            )
        if has_multiple_boundaries:
            raise ValueError("batch.batch is required for graph-level targets")
        _validate_targets(
            labels,
            node_count=x.size(0),
            graph_count=None,
        )
        return x, edge_index, labels, None
    if not isinstance(batch_index, Tensor):
        raise TypeError("batch.batch must be a tensor")
    if batch_index.ndim != 1:
        raise ValueError("batch.batch must be a rank-1 tensor")
    if batch_index.dtype is not torch.long:
        raise TypeError("batch.batch must have dtype torch.long")
    if batch_index.size(0) != x.size(0):
        raise ValueError("batch.batch length must match batch.x rows")
    if batch_index.numel() == 0:
        raise ValueError("batch.batch must contain at least one node")
    if int(batch_index.min()) < 0:
        raise ValueError("batch.batch indices must be non-negative")
    graph_count = int(batch_index.max()) + 1
    if not torch.equal(
        torch.unique(batch_index),
        torch.arange(graph_count, device=batch_index.device),
    ):
        raise ValueError("batch.batch indices must be contiguous from zero")
    _validate_targets(
        labels,
        node_count=x.size(0),
        graph_count=graph_count,
    )
    return x, edge_index, labels, batch_index


def _edge_kwargs(data: Data, modes: dict[str, EdgeMode]) -> dict[str, Tensor]:
    """Translate explicitly consumed optional edge tensors."""
    kwargs: dict[str, Tensor] = {}
    edge_count = _require_tensor(data, "edge_index").size(1)
    for field, mode in modes.items():
        value = data.get(field)
        if value is None:
            continue
        if mode == "reject":
            raise ValueError(f"{field} is unsupported by this model")
        if mode == "consume":
            if not isinstance(value, Tensor):
                raise TypeError(f"batch.{field} must be a tensor")
            if field == "edge_weight" and value.ndim != 1:
                raise ValueError("batch.edge_weight must be rank-1")
            if field == "edge_attr" and value.ndim < 1:
                raise ValueError("batch.edge_attr must have rank at least 1")
            if value.size(0) != edge_count:
                raise ValueError(
                    f"batch.{field} length must match batch.edge_index edges"
                )
            kwargs[field] = value
    return kwargs


class GNNWrapper(nn.Module):
    """Translate native PyG ``Data`` fields into a GNN backbone call."""

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
        """Return exactly the native readout contract."""
        x, edge_index, labels, batch_index = _validated_graph_fields(batch)
        kwargs: dict[str, Tensor | None] = {"batch": batch_index}
        kwargs.update(_edge_kwargs(batch, self.edge_modes))
        output = self.backbone(x, edge_index, **kwargs)
        if not isinstance(output, Tensor):
            raise TypeError("GNN backbone must return a tensor")
        return {"x": output, "labels": labels, "batch": batch_index}


__all__ = ["GNNWrapper"]
