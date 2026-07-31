"""Shared contracts and metadata adaptation for heterogeneous backbones."""

from __future__ import annotations

import math
from collections.abc import Mapping, Sequence
from numbers import Integral, Real

import torch
from torch import Tensor
from torch_geometric.typing import EdgeType, Metadata


def _positive_integer(value: object, *, name: str) -> int:
    """Normalize a non-boolean positive integer."""
    if isinstance(value, bool) or not isinstance(value, Integral):
        raise TypeError(f"{name} must be an integer")
    normalized = int(value)
    if normalized < 1:
        qualifier = "at least 1" if name == "num_layers" else "positive"
        raise ValueError(f"{name} must be {qualifier}")
    return normalized


def _dropout_probability(value: object) -> float:
    """Normalize a finite dropout probability in ``[0, 1)``."""
    if isinstance(value, bool) or not isinstance(value, Real):
        raise TypeError("dropout must be a real number")
    normalized = float(value)
    if not math.isfinite(normalized):
        raise ValueError("dropout must be finite")
    if not 0.0 <= normalized < 1.0:
        raise ValueError("dropout must be in [0, 1)")
    return normalized


def _normalize_metadata(
    node_types: Sequence[str],
    edge_types: Sequence[EdgeType],
) -> Metadata:
    """Copy and validate canonical heterogeneous metadata."""
    if isinstance(node_types, (str, bytes)) or not isinstance(
        node_types, Sequence
    ):
        raise TypeError("node types must be a sequence")
    if isinstance(edge_types, (str, bytes)) or not isinstance(
        edge_types, Sequence
    ):
        raise TypeError("edge types must be a sequence")

    normalized_nodes = list(node_types)
    if not normalized_nodes:
        raise ValueError("node types must not be empty")
    for node_type in normalized_nodes:
        if not isinstance(node_type, str):
            raise TypeError("node types must be strings")
        if not node_type:
            raise ValueError("node types must be non-empty strings")
    if len(set(normalized_nodes)) != len(normalized_nodes):
        raise ValueError("node types must be unique")

    normalized_edges: list[EdgeType] = []
    for edge_type in edge_types:
        if not isinstance(edge_type, tuple) or len(edge_type) != 3:
            raise TypeError(
                "edge types must be (source, relation, destination) tuples"
            )
        if any(not isinstance(part, str) for part in edge_type):
            raise TypeError("edge type components must be strings")
        if any(not part for part in edge_type):
            raise ValueError("edge type components must be non-empty strings")
        normalized_edges.append(edge_type)

    if not normalized_edges:
        raise ValueError("edge types must not be empty")
    if len(set(normalized_edges)) != len(normalized_edges):
        raise ValueError("edge types must be unique")

    known_nodes = set(normalized_nodes)
    for source_type, _, destination_type in normalized_edges:
        for endpoint in (source_type, destination_type):
            if endpoint not in known_nodes:
                raise ValueError(
                    f"Edge endpoint {endpoint!r} is not a known node type"
                )
    return normalized_nodes, normalized_edges


def validate_backbone_arguments(
    *,
    node_types: Sequence[str],
    edge_types: Sequence[EdgeType],
    hidden_channels: object,
    num_layers: object,
    heads: object,
    dropout: object,
) -> Metadata:
    """Validate shared metadata and scalar backbone arguments.

    Returns
    -------
    Metadata
        Defensive, canonical copies of the node and edge type sequences.
    """
    metadata = _normalize_metadata(node_types, edge_types)
    normalized_hidden_channels = _positive_integer(
        hidden_channels,
        name="hidden_channels",
    )
    normalized_heads = _positive_integer(heads, name="heads")
    _positive_integer(num_layers, name="num_layers")
    _dropout_probability(dropout)
    if normalized_hidden_channels % normalized_heads != 0:
        raise ValueError(
            "hidden_channels must be divisible by the number of heads"
        )
    return metadata


def validate_forward_dictionaries(
    *,
    x_dict: Mapping[str, Tensor],
    edge_index_dict: Mapping[EdgeType, Tensor],
    node_types: Sequence[str],
    edge_types: Sequence[EdgeType],
    hidden_channels: int,
) -> None:
    """Validate one heterogeneous mini-batch without mutating it."""
    if not isinstance(x_dict, Mapping):
        raise TypeError("x_dict must be a mapping")
    if not isinstance(edge_index_dict, Mapping):
        raise TypeError("edge_index_dict must be a mapping")

    expected_nodes = set(node_types)
    actual_nodes = set(x_dict)
    missing_nodes = expected_nodes - actual_nodes
    unexpected_nodes = actual_nodes - expected_nodes
    if missing_nodes or unexpected_nodes:
        raise ValueError(
            "Heterogeneous node types differ from backbone metadata: "
            f"missing={sorted(missing_nodes)!r}, "
            f"unexpected={sorted(unexpected_nodes, key=repr)!r}"
        )

    for node_type in node_types:
        features = x_dict[node_type]
        if not isinstance(features, Tensor):
            raise TypeError(
                f"Node type {node_type!r} features must be a torch.Tensor"
            )
        if features.ndim != 2:
            raise ValueError(
                f"Node type {node_type!r} features must have rank 2; "
                f"received shape {tuple(features.shape)!r}"
            )
        if features.size(1) != hidden_channels:
            raise ValueError(
                f"Node type {node_type!r} expected feature width "
                f"{hidden_channels}, received {features.size(1)}"
            )
        if not features.is_floating_point():
            raise TypeError(
                f"Node type {node_type!r} features must be floating point"
            )

    known_edges = set(edge_types)
    for edge_type, edge_index in edge_index_dict.items():
        if edge_type not in known_edges:
            raise ValueError(f"Unknown edge type: {edge_type!r}")
        if not isinstance(edge_index, Tensor):
            raise TypeError(
                f"Edge index for {edge_type!r} must be a torch.Tensor"
            )
        if edge_index.ndim != 2 or edge_index.size(0) != 2:
            raise ValueError(
                f"Edge index for {edge_type!r} must have shape [2, E]"
            )
        if edge_index.dtype not in {
            torch.uint8,
            torch.int8,
            torch.int16,
            torch.int32,
            torch.int64,
        }:
            raise TypeError(f"Edge index for {edge_type!r} must be integer")
        if edge_index.dtype != torch.long:
            raise TypeError(
                f"Edge index for {edge_type!r} must use torch.long dtype"
            )
        if edge_index.numel() == 0:
            continue

        source_type, _, destination_type = edge_type
        source = edge_index[0]
        destination = edge_index[1]
        if source.min().item() < 0 or source.max().item() >= x_dict[
            source_type
        ].size(0):
            raise ValueError(
                f"Edge index for {edge_type!r} has source indices out of range"
            )
        if destination.min().item() < 0 or destination.max().item() >= x_dict[
            destination_type
        ].size(0):
            raise ValueError(
                f"Edge index for {edge_type!r} has destination indices "
                "out of range"
            )


def _encoded_node_type(node_type: str, *, prefix: str) -> str:
    """Return an injective, module-safe node alias."""
    return f"{prefix}{node_type.encode('utf-8').hex()}"


def _encoded_edge_type(edge_type: EdgeType, *, prefix: str) -> str:
    """Return an injective, tuple-safe relation alias."""
    payload = b"".join(
        len(encoded).to_bytes(8, "big") + encoded
        for part in edge_type
        for encoded in (part.encode("utf-8"),)
    )
    return f"{prefix}{payload.hex()}"


def _is_safe_node_module_key(node_type: str) -> bool:
    """Check all PyTorch registries used by :class:`HGTConv`."""
    if "." in node_type or "__" in node_type:
        return False
    module_dict = torch.nn.ModuleDict()
    parameter_dict = torch.nn.ParameterDict()
    return not hasattr(module_dict, node_type) and not hasattr(
        parameter_dict, node_type
    )


def _is_safe_joined_edge_key(edge_type: EdgeType) -> bool:
    """Check the ``HGTConv`` parameter key derived from one edge tuple."""
    joined = "__".join(edge_type)
    return "." not in joined and not hasattr(torch.nn.ParameterDict(), joined)


class _HeterogeneousMetadataAdapter:
    """Losslessly map external PyG metadata to safe ``HGTConv`` names.

    Identity aliases are retained whenever PyTorch can safely register them.
    Unsafe names and ambiguous relation joins use deterministic UTF-8-derived
    aliases. This preserves historical checkpoint keys for ordinary CellHGT
    rank metadata while supporting arbitrary valid PyG names.
    """

    def __init__(self, metadata: Metadata) -> None:
        node_types, edge_types = _normalize_metadata(*metadata)
        self.node_types = list(node_types)
        self.edge_types = list(edge_types)
        self.metadata: Metadata = (self.node_types, self.edge_types)

        identity_nodes = {
            node_type
            for node_type in self.node_types
            if _is_safe_node_module_key(node_type)
        }
        prefix = "tbN"
        while any(
            node_type.startswith(prefix) for node_type in identity_nodes
        ):
            prefix += "N"
        self._node_to_internal = {
            node_type: (
                node_type
                if node_type in identity_nodes
                else _encoded_node_type(node_type, prefix=prefix)
            )
            for node_type in self.node_types
        }
        self._internal_to_node = {
            internal: external
            for external, internal in self._node_to_internal.items()
        }

        provisional_edges = {
            edge_type: (
                self._node_to_internal[edge_type[0]],
                edge_type[1],
                self._node_to_internal[edge_type[2]],
            )
            for edge_type in self.edge_types
        }
        joined_groups: dict[str, list[EdgeType]] = {}
        for external_edge, internal_edge in provisional_edges.items():
            joined_groups.setdefault("__".join(internal_edge), []).append(
                external_edge
            )

        edge_prefix = "tbE"
        relation_names = {edge_type[1] for edge_type in self.edge_types}
        while any(
            relation.startswith(edge_prefix) for relation in relation_names
        ):
            edge_prefix += "E"
        self._edge_to_internal: dict[EdgeType, EdgeType] = {}
        for external_edge, provisional_edge in provisional_edges.items():
            joined = "__".join(provisional_edge)
            ambiguous = len(joined_groups[joined]) > 1
            relation_unsafe = "__" in provisional_edge[
                1
            ] or not _is_safe_joined_edge_key(provisional_edge)
            relation = (
                _encoded_edge_type(external_edge, prefix=edge_prefix)
                if ambiguous or relation_unsafe
                else provisional_edge[1]
            )
            self._edge_to_internal[external_edge] = (
                provisional_edge[0],
                relation,
                provisional_edge[2],
            )

        internal_edges = list(self._edge_to_internal.values())
        joined_edges = ["__".join(edge_type) for edge_type in internal_edges]
        if len(set(joined_edges)) != len(joined_edges):
            raise RuntimeError(
                "Internal heterogeneous relation aliases collide"
            )
        self._internal_to_edge = {
            internal: external
            for external, internal in self._edge_to_internal.items()
        }
        self.internal_metadata: Metadata = (
            [
                self._node_to_internal[node_type]
                for node_type in self.node_types
            ],
            [
                self._edge_to_internal[edge_type]
                for edge_type in self.edge_types
            ],
        )

    def node_to_internal(self, node_type: str) -> str:
        """Return the internal alias for one external node type."""
        try:
            return self._node_to_internal[node_type]
        except KeyError as error:
            raise KeyError(f"Unknown node type: {node_type!r}") from error

    def to_internal_x_dict(
        self,
        x_dict: Mapping[str, Tensor],
    ) -> dict[str, Tensor]:
        """Remap feature keys while preserving canonical node order."""
        return {
            self._node_to_internal[node_type]: x_dict[node_type]
            for node_type in self.node_types
        }

    def to_internal_edge_index_dict(
        self,
        edge_index_dict: Mapping[EdgeType, Tensor],
    ) -> dict[EdgeType, Tensor]:
        """Remap the known relation subset in sample iteration order."""
        return {
            self._edge_to_internal[edge_type]: edge_index
            for edge_type, edge_index in edge_index_dict.items()
        }

    def to_external_x_dict(
        self,
        x_dict: Mapping[str, Tensor],
    ) -> dict[str, Tensor]:
        """Restore semantic feature keys in canonical node order."""
        return {
            node_type: x_dict[self._node_to_internal[node_type]]
            for node_type in self.node_types
        }


__all__ = [
    "validate_backbone_arguments",
    "validate_forward_dictionaries",
]
