"""Native homogeneous graph adapter for GNN backbones."""

from __future__ import annotations

from dataclasses import dataclass
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


@dataclass(frozen=True, slots=True)
class _TensorBinding:
    identity: int
    version: int
    shape: tuple[int, ...]
    dtype: torch.dtype
    device: torch.device


@dataclass(frozen=True, slots=True)
class _OptionalEdgeEvidence:
    field: str
    present: bool
    is_tensor: bool
    binding: _TensorBinding | None
    finite: bool | None


@dataclass(frozen=True, slots=True)
class _GraphBatchEvidence:
    node_count: int
    edge_count: int
    graph_count: int
    edge_index: _TensorBinding
    batch_index: _TensorBinding
    optional_edges: tuple[_OptionalEdgeEvidence, ...]


_GRAPH_EVIDENCE_ATTRIBUTE = "_topobench_graph_batch_evidence"


def _tensor_binding(tensor: Tensor) -> _TensorBinding:
    """Capture immutable metadata identifying one exact tensor version."""
    return _TensorBinding(
        identity=id(tensor),
        version=tensor._version,
        shape=tuple(tensor.shape),
        dtype=tensor.dtype,
        device=tensor.device,
    )


def _binding_is_current(tensor: Tensor, binding: _TensorBinding) -> bool:
    """Return whether evidence still names this exact unmodified tensor."""
    return (
        id(tensor) == binding.identity
        and tensor._version == binding.version
        and tuple(tensor.shape) == binding.shape
        and tensor.dtype == binding.dtype
        and tensor.device == binding.device
    )


def _validate_synchronously(predicate: Tensor, message: str) -> None:
    """Validate untrusted tensor contents before entering a backbone."""
    try:
        valid = bool(predicate)
    except (RuntimeError, TypeError) as error:
        raise ValueError(
            f"untrusted graph batch could not validate: {message}"
        ) from error
    if not valid:
        raise ValueError(message)


def _optional_edge_evidence(data: Data) -> tuple[_OptionalEdgeEvidence, ...]:
    """Record optional fields without enforcing any model-specific mode."""
    records: list[_OptionalEdgeEvidence] = []
    for field in ("edge_attr", "edge_weight"):
        value = data.get(field)
        if value is None:
            records.append(
                _OptionalEdgeEvidence(field, False, False, None, None)
            )
            continue
        if not isinstance(value, Tensor):
            records.append(
                _OptionalEdgeEvidence(field, True, False, None, None)
            )
            continue
        finite: bool | None = None
        if value.device.type == "cpu":
            try:
                finite = bool(torch.isfinite(value).all())
            except (RuntimeError, TypeError):
                finite = False
        records.append(
            _OptionalEdgeEvidence(
                field,
                True,
                True,
                _tensor_binding(value),
                finite,
            )
        )
    return tuple(records)


def _prepare_graph_batch_evidence(data: Data) -> _GraphBatchEvidence:
    """Validate a CPU graph batch and preserve content-derived evidence."""
    if not isinstance(data, Data):
        raise TypeError("batch must be a torch_geometric.data.Data")
    x = _require_tensor(data, "x")
    edge_index = _require_tensor(data, "edge_index")
    batch_index = _require_tensor(data, "batch")
    if x.ndim != 2 or not x.is_floating_point():
        raise ValueError("batch.x must be a rank-2 floating tensor")
    if edge_index.ndim != 2 or edge_index.size(0) != 2:
        raise ValueError("batch.edge_index must have shape [2, E]")
    if edge_index.dtype is not torch.long:
        raise TypeError("batch.edge_index must have dtype torch.long")
    if batch_index.ndim != 1:
        raise ValueError("batch.batch must be a rank-1 tensor")
    if batch_index.dtype is not torch.long:
        raise TypeError("batch.batch must have dtype torch.long")
    if any(
        tensor.device.type != "cpu"
        for tensor in (x, edge_index, batch_index)
    ):
        raise ValueError("graph batch evidence must be prepared on CPU")
    if edge_index.device != x.device or batch_index.device != x.device:
        raise ValueError("graph structural tensors must share one device")
    if batch_index.size(0) != x.size(0):
        raise ValueError("batch.batch length must match batch.x rows")
    if batch_index.numel() == 0:
        raise ValueError("batch.batch must contain at least one node")
    if edge_index.numel():
        _validate_synchronously(
            (edge_index.min() >= 0) & (edge_index.max() < x.size(0)),
            "batch.edge_index contains an invalid node index",
        )
    _validate_synchronously(
        batch_index.min() >= 0,
        "batch.batch indices must be non-negative",
    )
    ordered_batch = torch.sort(batch_index).values
    _validate_synchronously(
        (ordered_batch[0] == 0)
        & torch.all(ordered_batch[1:] - ordered_batch[:-1] <= 1),
        "batch.batch indices must be contiguous from zero",
    )
    if edge_index.numel():
        _validate_synchronously(
            torch.all(
                batch_index.index_select(0, edge_index[0])
                == batch_index.index_select(0, edge_index[1])
            ),
            "batch.edge_index crosses graph boundaries described by batch.batch",
        )
    return _GraphBatchEvidence(
        node_count=x.size(0),
        edge_count=edge_index.size(1),
        graph_count=int(batch_index.max()) + 1,
        edge_index=_tensor_binding(edge_index),
        batch_index=_tensor_binding(batch_index),
        optional_edges=_optional_edge_evidence(data),
    )


def _bind_graph_batch_evidence(
    data: Data,
    prepared: _GraphBatchEvidence,
) -> None:
    """Bind CPU-derived evidence to framework-transferred tensor objects."""
    x = _require_tensor(data, "x")
    edge_index = _require_tensor(data, "edge_index")
    batch_index = _require_tensor(data, "batch")
    if (
        x.size(0) != prepared.node_count
        or tuple(edge_index.shape) != prepared.edge_index.shape
        or edge_index.dtype != prepared.edge_index.dtype
        or tuple(batch_index.shape) != prepared.batch_index.shape
        or batch_index.dtype != prepared.batch_index.dtype
    ):
        raise ValueError("device transfer changed graph batch metadata")
    rebound_optional: list[_OptionalEdgeEvidence] = []
    for record in prepared.optional_edges:
        value = data.get(record.field)
        present = value is not None
        is_tensor = isinstance(value, Tensor)
        if present != record.present or is_tensor != record.is_tensor:
            raise ValueError(
                f"device transfer changed batch.{record.field} metadata"
            )
        rebound_optional.append(
            _OptionalEdgeEvidence(
                record.field,
                present,
                is_tensor,
                _tensor_binding(value) if isinstance(value, Tensor) else None,
                record.finite,
            )
        )
    bound = _GraphBatchEvidence(
        node_count=prepared.node_count,
        edge_count=prepared.edge_count,
        graph_count=prepared.graph_count,
        edge_index=_tensor_binding(edge_index),
        batch_index=_tensor_binding(batch_index),
        optional_edges=tuple(rebound_optional),
    )
    data.__dict__[_GRAPH_EVIDENCE_ATTRIBUTE] = bound


def _current_graph_batch_evidence(
    data: Data,
    x: Tensor,
    edge_index: Tensor,
    batch_index: Tensor,
) -> _GraphBatchEvidence | None:
    """Return current structural evidence or reject a stale marker."""
    evidence = data.__dict__.get(_GRAPH_EVIDENCE_ATTRIBUTE)
    if not isinstance(evidence, _GraphBatchEvidence):
        return None
    if (
        x.size(0) == evidence.node_count
        and x.device == edge_index.device
        and edge_index.size(1) == evidence.edge_count
        and _binding_is_current(edge_index, evidence.edge_index)
        and _binding_is_current(batch_index, evidence.batch_index)
    ):
        return evidence
    return None


def _current_optional_evidence(
    data: Data,
    field: str,
    value: Tensor,
) -> _OptionalEdgeEvidence | None:
    """Return current evidence for one consumed optional tensor."""
    evidence = data.__dict__.get(_GRAPH_EVIDENCE_ATTRIBUTE)
    if not isinstance(evidence, _GraphBatchEvidence):
        return None
    for record in evidence.optional_edges:
        if (
            record.field == field
            and record.binding is not None
            and _binding_is_current(value, record.binding)
        ):
            return record
    return None




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
    if edge_index.device != x.device:
        raise ValueError(
            "batch.edge_index must be on the same device as batch.x"
        )

    labels = _require_tensor(data, "y")
    batch_index = data.get("batch")
    if batch_index is None:
        if edge_index.numel():
            _validate_synchronously(
                (edge_index.min() >= 0) & (edge_index.max() < x.size(0)),
                "batch.edge_index contains an invalid node index",
            )
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
    if batch_index.device != x.device:
        raise ValueError("batch.batch must be on the same device as batch.x")
    if batch_index.size(0) != x.size(0):
        raise ValueError("batch.batch length must match batch.x rows")
    if batch_index.numel() == 0:
        raise ValueError("batch.batch must contain at least one node")

    evidence = _current_graph_batch_evidence(
        data,
        x,
        edge_index,
        batch_index,
    )
    if evidence is None:
        if edge_index.numel():
            _validate_synchronously(
                (edge_index.min() >= 0) & (edge_index.max() < x.size(0)),
                "batch.edge_index contains an invalid node index",
            )
        _validate_synchronously(
            batch_index.min() >= 0,
            "batch.batch indices must be non-negative",
        )
        ordered_batch = torch.sort(batch_index).values
        _validate_synchronously(
            (ordered_batch[0] == 0)
            & torch.all(ordered_batch[1:] - ordered_batch[:-1] <= 1),
            "batch.batch indices must be contiguous from zero",
        )
        if edge_index.numel():
            _validate_synchronously(
                torch.all(
                    batch_index.index_select(0, edge_index[0])
                    == batch_index.index_select(0, edge_index[1])
                ),
                "batch.edge_index crosses graph boundaries described by "
                "batch.batch",
            )
        graph_count = int(batch_index.max()) + 1
    else:
        graph_count = evidence.graph_count
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
    x = _require_tensor(data, "x")
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
            if not value.is_floating_point():
                raise TypeError(f"batch.{field} must have a floating dtype")
            if value.dtype != x.dtype:
                raise TypeError(
                    f"batch.{field} dtype must match batch.x dtype"
                )
            if value.device != x.device:
                raise ValueError(
                    f"batch.{field} must be on the same device as batch.x"
                )
            if value.size(0) != edge_count:
                raise ValueError(
                    f"batch.{field} length must match batch.edge_index edges"
                )
            evidence = _current_optional_evidence(data, field, value)
            if evidence is not None and evidence.finite is not None:
                if not evidence.finite:
                    raise ValueError(
                        f"batch.{field} must contain only finite values"
                    )
            else:
                _validate_synchronously(
                    torch.isfinite(value).all(),
                    f"batch.{field} must contain only finite values",
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
