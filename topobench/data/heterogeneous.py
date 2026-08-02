"""Validated metadata contracts for native heterogeneous node data."""

from __future__ import annotations

from collections.abc import Sequence
from dataclasses import dataclass
from numbers import Integral
from typing import cast

import torch
from torch_geometric.data import HeteroData
from torch_geometric.typing import EdgeType, Metadata
from topobench.data.capabilities import validate_classification_vocabulary

_MASK_NAMES = ("train_mask", "val_mask", "test_mask")


def _require_ordered_sequence(
    value: object,
    *,
    field_name: str,
) -> Sequence[object]:
    """Return an ordered non-string sequence or raise a contract error."""
    if isinstance(value, (str, bytes)) or not isinstance(value, Sequence):
        raise TypeError(f"{field_name} must be an ordered sequence")
    return value


def _normalize_integral(value: object, *, field_name: str) -> int:
    """Normalize a non-boolean integral scalar to built-in ``int``."""
    if isinstance(value, bool) or not isinstance(value, Integral):
        raise TypeError(f"{field_name} must be an integer")
    return int(value)


def _normalize_node_types(value: object) -> tuple[str, ...]:
    """Normalize and structurally validate ordered node types."""
    sequence = _require_ordered_sequence(value, field_name="node_types")
    if not sequence:
        raise ValueError("node_types must not be empty")
    if not all(isinstance(node_type, str) for node_type in sequence):
        raise TypeError("node_types entries must be strings")
    normalized = tuple(cast(str, node_type) for node_type in sequence)
    if any(not node_type for node_type in normalized):
        raise ValueError("node_types entries must be non-empty strings")
    if len(set(normalized)) != len(normalized):
        raise ValueError("node_types must not contain duplicates")
    return normalized


def _normalize_edge_types(
    value: object,
    *,
    node_types: tuple[str, ...],
) -> tuple[EdgeType, ...]:
    """Normalize typed relations and validate their referenced node types."""
    sequence = _require_ordered_sequence(value, field_name="edge_types")
    normalized: list[EdgeType] = []
    available = set(node_types)
    for relation in sequence:
        if isinstance(relation, (str, bytes)) or not isinstance(
            relation, Sequence
        ):
            raise TypeError("edge_types entries must be relation triples")
        if len(relation) != 3:
            raise ValueError(
                "edge_types entries must contain exactly three values"
            )
        if not all(isinstance(component, str) for component in relation):
            raise TypeError("edge_types relation components must be strings")
        edge_type = (relation[0], relation[1], relation[2])
        if any(not component for component in edge_type):
            raise ValueError(
                "edge_types relation components must be non-empty strings"
            )
        if edge_type[0] not in available or edge_type[2] not in available:
            raise ValueError(
                f"edge_types relation {edge_type!r} references an unknown "
                f"node type; node_types={node_types!r}"
            )
        normalized.append(edge_type)
    if len(set(normalized)) != len(normalized):
        raise ValueError("edge_types must not contain duplicates")
    return tuple(normalized)


def _normalize_target_node_type(
    value: object,
    *,
    node_types: tuple[str, ...] | None = None,
) -> str:
    """Normalize a target node selector, optionally checking membership."""
    if not isinstance(value, str):
        raise TypeError("target_node_type must be str")
    if not value:
        raise ValueError("target_node_type must be a non-empty string")
    if node_types is not None and value not in node_types:
        raise ValueError(
            f"target_node_type {value!r} must be present in "
            f"node_types={node_types!r}"
        )
    return value


def _normalize_input_channels(
    value: object,
    *,
    node_types: tuple[str, ...],
) -> tuple[tuple[str, int], ...]:
    """Normalize ordered feature widths and align them with node types."""
    sequence = _require_ordered_sequence(
        value,
        field_name="input_channels",
    )
    normalized: list[tuple[str, int]] = []
    for entry in sequence:
        if isinstance(entry, (str, bytes)) or not isinstance(entry, Sequence):
            raise TypeError(
                "input_channels entries must be node type/width pairs"
            )
        if len(entry) != 2:
            raise ValueError(
                "input_channels entries must contain exactly two values"
            )
        node_type, width_value = entry
        if not isinstance(node_type, str):
            raise TypeError("input_channels node type must be str")
        width = _normalize_integral(
            width_value,
            field_name="input_channels width",
        )
        if width < 1:
            raise ValueError("input_channels widths must be positive")
        normalized.append((node_type, width))

    channel_node_types = tuple(node_type for node_type, _ in normalized)
    if channel_node_types != node_types:
        raise ValueError(
            "input_channels node types must match node_types in order; "
            f"received={channel_node_types!r}, expected={node_types!r}"
        )
    return tuple(normalized)


@dataclass(frozen=True)
class HeterogeneousDataSpec:
    """Immutable metadata required by heterogeneous node models.

    Parameters
    ----------
    node_types : tuple[str, ...]
        Node types in their native PyG metadata order.
    edge_types : tuple[EdgeType, ...]
        Typed relations in their native PyG metadata order.
    target_node_type : str
        Node type carrying labels and split masks.
    num_classes : int
        Number of target classes.
    input_channels : tuple[tuple[str, int], ...]
        Feature width for every node type, in node metadata order.
    """

    node_types: tuple[str, ...]
    edge_types: tuple[EdgeType, ...]
    target_node_type: str
    num_classes: int
    input_channels: tuple[tuple[str, int], ...]

    def __post_init__(self) -> None:
        """Normalize direct construction into deeply immutable state."""
        node_types = _normalize_node_types(self.node_types)
        edge_types = _normalize_edge_types(
            self.edge_types,
            node_types=node_types,
        )
        target_node_type = _normalize_target_node_type(
            self.target_node_type,
            node_types=node_types,
        )
        num_classes = _normalize_integral(
            self.num_classes,
            field_name="num_classes",
        )
        if num_classes < 2:
            raise ValueError("num_classes must be at least 2")
        input_channels = _normalize_input_channels(
            self.input_channels,
            node_types=node_types,
        )

        object.__setattr__(self, "node_types", node_types)
        object.__setattr__(self, "edge_types", edge_types)
        object.__setattr__(self, "target_node_type", target_node_type)
        object.__setattr__(self, "num_classes", num_classes)
        object.__setattr__(self, "input_channels", input_channels)

    @property
    def input_channels_dict(self) -> dict[str, int]:
        """Return a fresh mapping from node type to feature width."""
        return dict(self.input_channels)

    def pyg_metadata(self) -> Metadata:
        """Return fresh mutable containers accepted by PyG models."""
        return list(self.node_types), list(self.edge_types)


def _validate_node_features(
    data: HeteroData,
    *,
    selector: str,
) -> tuple[tuple[str, int], ...]:
    """Validate feature matrices and collect their ordered widths."""
    channels: list[tuple[str, int]] = []
    for node_type in data.node_types:
        context = f"{selector}: node type {node_type!r} field x"
        store = data[node_type]
        if store.num_nodes is None:
            raise ValueError(f"{context} has no num_nodes")
        if "x" not in store:
            raise ValueError(f"{context} has no x after preprocessing")
        if not isinstance(store.x, torch.Tensor):
            raise TypeError(
                f"{context} observed shape=None, "
                f"dtype={type(store.x).__name__}; expected a torch.Tensor"
            )
        if store.num_nodes == 0 or (
            store.x.ndim >= 1 and store.x.size(0) == 0
        ):
            raise ValueError(
                f"{context} observed shape={tuple(store.x.shape)}, "
                f"dtype={store.x.dtype}; expected at least one node"
            )
        if store.x.ndim != 2 or store.x.size(0) != store.num_nodes:
            raise ValueError(
                f"{context} has invalid x shape {tuple(store.x.shape)} "
                f"for {store.num_nodes} nodes; expected rank-2 alignment"
            )
        if store.x.size(-1) < 1:
            raise ValueError(f"{context} has zero feature width")
        if not store.x.is_floating_point():
            raise TypeError(
                f"{context} observed shape={tuple(store.x.shape)}, "
                f"dtype={store.x.dtype}; expected floating point"
            )
        if not torch.isfinite(store.x).all():
            raise ValueError(
                f"{context} observed shape={tuple(store.x.shape)}, "
                f"dtype={store.x.dtype}; expected finite values"
            )
        channels.append((node_type, int(store.x.size(-1))))
    return tuple(channels)


def _validate_edge_indices(data: HeteroData) -> None:
    """Validate typed-edge tensor types before PyG inspects their shapes."""
    for edge_type in data.edge_types:
        store = data[edge_type]
        if "edge_index" not in store:
            continue
        edge_index = store.edge_index
        if not isinstance(edge_index, torch.Tensor):
            raise TypeError(
                f"Edge type {edge_type!r} edge_index must be a torch.Tensor"
            )
        if edge_index.dtype != torch.long:
            raise TypeError(
                f"Edge type {edge_type!r} edge_index must use torch.long; "
                f"received {edge_index.dtype}"
            )


def _validate_target_labels(
    data: HeteroData,
    *,
    target_node_type: str,
    selector: str,
) -> torch.Tensor:
    """Validate and return the target node label vector."""
    context = f"{selector}: node type {target_node_type!r} field y"
    available = tuple(data.node_types)
    if target_node_type not in data.node_types:
        raise ValueError(
            f"Unknown target node type {target_node_type!r}; "
            f"available={available!r}"
        )

    store = data[target_node_type]
    if "y" not in store:
        raise ValueError(f"{context} has no y labels")
    labels = store.y
    if not isinstance(labels, torch.Tensor):
        raise TypeError(
            f"{context} observed shape=None, "
            f"dtype={type(labels).__name__}; expected a torch.Tensor"
        )
    if labels.dtype != torch.long:
        raise TypeError(
            f"{context} observed shape={tuple(labels.shape)}, "
            f"dtype={labels.dtype}; expected torch.long"
        )
    if labels.ndim != 1:
        raise ValueError(
            f"{context} observed shape={tuple(labels.shape)}, "
            f"dtype={labels.dtype}; expected one-dimensional labels"
        )
    if labels.size(0) != store.num_nodes:
        raise ValueError(
            f"{context} observed shape={tuple(labels.shape)}, "
            f"dtype={labels.dtype}; expected one label for each of "
            f"{store.num_nodes} nodes"
        )
    return labels


def _validate_target_masks(
    data: HeteroData,
    *,
    target_node_type: str,
    labels: torch.Tensor,
) -> tuple[torch.Tensor, ...]:
    """Validate required, nonempty, pairwise-disjoint split masks."""
    store = data[target_node_type]
    masks: list[torch.Tensor] = []
    for mask_name in _MASK_NAMES:
        if mask_name not in store:
            raise ValueError(
                f"Target node type {target_node_type!r} has no {mask_name}"
            )
        mask = store[mask_name]
        if not isinstance(mask, torch.Tensor):
            raise TypeError(
                f"{mask_name} on target node type "
                f"{target_node_type!r} must be a torch.Tensor"
            )
        if mask.dtype != torch.bool:
            raise TypeError(
                f"{mask_name} on target node type "
                f"{target_node_type!r} must be boolean"
            )
        if mask.shape != labels.shape:
            raise ValueError(
                f"{mask_name} on target node type "
                f"{target_node_type!r} has shape {tuple(mask.shape)}; "
                f"expected {tuple(labels.shape)}"
            )
        if not bool(mask.any()):
            raise ValueError(
                f"{mask_name} on target node type "
                f"{target_node_type!r} must be non-empty"
            )
        masks.append(mask)

    for first_index, first_name in enumerate(_MASK_NAMES):
        for second_index in range(first_index + 1, len(_MASK_NAMES)):
            second_name = _MASK_NAMES[second_index]
            if bool((masks[first_index] & masks[second_index]).any()):
                raise ValueError(
                    f"{first_name} and {second_name} overlap on target "
                    f"node type {target_node_type!r}"
                )
    return tuple(masks)


def _validate_target_store(
    data: HeteroData,
    *,
    target_node_type: str,
    num_classes: int,
    selector: str,
) -> None:
    """Validate the full target vocabulary and phase masks."""
    labels = _validate_target_labels(
        data,
        target_node_type=target_node_type,
        selector=selector,
    )
    validate_classification_vocabulary(
        [
            (
                f"node type {target_node_type!r}",
                labels,
                int(data[target_node_type].num_nodes),
            )
        ],
        selector=selector,
        field="y",
        configured_num_classes=num_classes,
    )
    _validate_target_masks(
        data,
        target_node_type=target_node_type,
        labels=labels,
    )


def validate_heterogeneous_node_data(
    data: HeteroData,
    *,
    target_node_type: str,
    num_classes: int,
    selector: str = "<heterogeneous>",
) -> HeterogeneousDataSpec:
    """Validate native heterogeneous node-classification data.

    The validator owns the post-preprocessing schema contract. It validates
    node features, typed relations, target labels, and explicit split masks
    without mutating the input graph.

    Parameters
    ----------
    data : HeteroData
        Native PyG heterogeneous graph after preprocessing.
    target_node_type : str
        Node type that owns labels and split masks.
    num_classes : int
        Number of target classes. Must be at least two.

    Returns
    -------
    HeterogeneousDataSpec
        Frozen, ordered metadata derived from ``data``.

    Raises
    ------
    TypeError
        If the top-level data family, features, labels, or masks have invalid
        types.
    ValueError
        If graph structure, shapes, split masks, or supervised labels violate
        the heterogeneous node-classification contract.
    """
    if not isinstance(data, HeteroData):
        raise TypeError("Expected native torch_geometric.data.HeteroData")
    target_node_type = _normalize_target_node_type(target_node_type)
    num_classes = _normalize_integral(
        num_classes,
        field_name="num_classes",
    )
    if num_classes < 2:
        raise ValueError("num_classes must be at least 2")
    _validate_edge_indices(data)
    try:
        data.validate(raise_on_error=True)
    except ValueError as error:
        raise ValueError(f"Invalid heterogeneous graph: {error}") from error

    input_channels = _validate_node_features(data, selector=selector)
    _validate_target_store(
        data,
        target_node_type=target_node_type,
        num_classes=num_classes,
        selector=selector,
    )
    node_types, edge_types = data.metadata()
    return HeterogeneousDataSpec(
        node_types=tuple(node_types),
        edge_types=tuple(edge_types),
        target_node_type=target_node_type,
        num_classes=num_classes,
        input_channels=input_channels,
    )


__all__ = [
    "HeterogeneousDataSpec",
    "validate_heterogeneous_node_data",
]
