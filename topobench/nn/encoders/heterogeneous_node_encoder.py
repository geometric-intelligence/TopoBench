"""Feature encoding for native heterogeneous node data."""

from __future__ import annotations

import math
from collections.abc import Mapping
from numbers import Integral, Real

import torch
from torch_geometric.data import HeteroData

from topobench.nn.activation import make_activation
from topobench.nn.encoders.base import AbstractFeatureEncoder


def _normalize_positive_integer(value: object, *, name: str) -> int:
    """Normalize a non-boolean positive integral value."""
    if isinstance(value, bool) or not isinstance(value, Integral):
        raise TypeError(f"{name} must be an integer")
    normalized = int(value)
    if normalized < 1:
        raise ValueError(f"{name} must be positive")
    return normalized


def _normalize_input_channels(
    input_channels: object,
) -> tuple[tuple[str, int], ...]:
    """Copy and validate ordered per-node-type input widths."""
    if not isinstance(input_channels, Mapping):
        raise TypeError("input_channels must be a mapping")
    if not input_channels:
        raise ValueError("input_channels must not be empty")

    normalized: dict[str, int] = {}
    for node_type, width in input_channels.items():
        if not isinstance(node_type, str):
            raise TypeError("input_channels node type must be a string")
        if not node_type:
            raise ValueError(
                "input_channels node type must be a non-empty string"
            )
        normalized[node_type] = _normalize_positive_integer(
            width,
            name=f"input_channels width for {node_type!r}",
        )
    return tuple(normalized.items())


def _projection_module_key(node_type: str) -> str:
    """Encode an external node type as a safe, stable PyTorch module key.

    PyG accepts arbitrary non-empty strings as node types, while PyTorch child
    modules reject dots and names colliding with attributes such as
    ``training`` or ``forward``. UTF-8 hex encoding is injective and depends
    only on the semantic node type, so checkpoint keys remain stable when
    equivalent metadata is supplied in another insertion order.
    """
    return f"node_type_{node_type.encode('utf-8').hex()}"


def _normalize_dropout(dropout: object) -> float:
    """Normalize a finite dropout probability in ``[0, 1)``."""
    if isinstance(dropout, bool) or not isinstance(dropout, Real):
        raise TypeError("dropout must be a real number")
    normalized = float(dropout)
    if not math.isfinite(normalized):
        raise ValueError("dropout must be finite")
    if not 0.0 <= normalized < 1.0:
        raise ValueError("dropout must be in [0, 1)")
    return normalized


class HeterogeneousNodeFeatureEncoder(AbstractFeatureEncoder):
    """Project each heterogeneous node type to one common feature width.

    All projections are constructed eagerly from validated metadata, so an
    optimizer created before the first batch sees the complete parameter set.
    Forward validation and computation finish before any feature store is
    updated, making mutations transactional on failure.

    Encoding replaces feature tensors in place. A preprocessed batch is
    therefore a single-pass input; use a fresh batch object for another model
    call.

    Parameters
    ----------
    input_channels : Mapping[str, int]
        Ordered mapping from every node type to its input feature width.
    hidden_channels : int
        Common output feature width.
    activation : str, default="relu"
        Exact activation name accepted by
        :func:`topobench.nn.activation.make_activation`.
    dropout : float, default=0.0
        Dropout probability in ``[0, 1)``.
    """

    def __init__(
        self,
        input_channels: Mapping[str, int],
        hidden_channels: int,
        activation: str = "relu",
        dropout: float = 0.0,
    ) -> None:
        super().__init__()
        normalized_channels = _normalize_input_channels(input_channels)
        normalized_hidden_channels = _normalize_positive_integer(
            hidden_channels,
            name="hidden_channels",
        )
        normalized_dropout = _normalize_dropout(dropout)

        self._input_channels = normalized_channels
        self.hidden_channels = normalized_hidden_channels
        self.activation_name = activation
        self.dropout_probability = normalized_dropout
        self._projections = torch.nn.ModuleDict(
            {
                _projection_module_key(node_type): torch.nn.Linear(
                    width,
                    normalized_hidden_channels,
                )
                for node_type, width in normalized_channels
            }
        )
        self.activation = make_activation(activation)
        self.dropout = torch.nn.Dropout(normalized_dropout)

    @property
    def input_channels(self) -> dict[str, int]:
        """Return a defensive copy of ordered input-channel metadata."""
        return dict(self._input_channels)

    @property
    def projections(self) -> dict[str, torch.nn.Linear]:
        """Return a fresh semantic view of the registered projections."""
        return {
            node_type: self.projection_for(node_type)
            for node_type, _ in self._input_channels
        }

    def projection_for(self, node_type: str) -> torch.nn.Linear:
        """Return the registered projection for an external PyG node type.

        Parameters
        ----------
        node_type : str
            Semantic node type from the input metadata.

        Returns
        -------
        torch.nn.Linear
            Eagerly registered projection for ``node_type``.

        Raises
        ------
        KeyError
            If the node type was not present in the validated constructor
            metadata.
        """
        if node_type not in dict(self._input_channels):
            raise KeyError(f"Unknown node type: {node_type!r}")
        projection = self._projections[_projection_module_key(node_type)]
        if not isinstance(projection, torch.nn.Linear):
            raise TypeError(
                f"Projection for node type {node_type!r} is not Linear"
            )
        return projection

    def _validate_store_keys(self, data: HeteroData) -> None:
        """Require node stores and feature stores to match metadata exactly."""
        expected = {node_type for node_type, _ in self._input_channels}
        actual_nodes = set(data.node_types)
        actual_features = set(data.x_dict)
        missing = expected - actual_nodes
        unexpected = actual_nodes - expected
        if missing or unexpected:
            raise ValueError(
                "Heterogeneous node types differ from validated metadata: "
                f"missing={sorted(missing)!r}, "
                f"unexpected={sorted(unexpected)!r}"
            )

        missing_features = expected - actual_features
        unexpected_features = actual_features - expected
        if missing_features or unexpected_features:
            raise ValueError(
                "Heterogeneous feature keys differ from validated metadata: "
                f"missing={sorted(missing_features)!r}, "
                f"unexpected={sorted(unexpected_features)!r}"
            )

    def _validate_features(
        self,
        data: HeteroData,
    ) -> dict[str, torch.Tensor]:
        """Preflight every feature tensor without mutating the graph."""
        features_by_type: dict[str, torch.Tensor] = {}
        for node_type, expected_width in self._input_channels:
            features = data[node_type].x
            if not isinstance(features, torch.Tensor):
                raise TypeError(
                    f"Node type {node_type!r} features must be a torch.Tensor"
                )
            if features.ndim != 2:
                raise ValueError(
                    f"Node type {node_type!r} features must have rank 2; "
                    f"received shape {tuple(features.shape)!r}"
                )

            num_nodes = data[node_type].num_nodes
            if num_nodes is not None and features.size(0) != num_nodes:
                raise ValueError(
                    f"Node type {node_type!r} declares {num_nodes} nodes but "
                    f"received {features.size(0)} feature rows"
                )
            if features.size(1) != expected_width:
                raise ValueError(
                    f"Node type {node_type!r} expected feature width "
                    f"{expected_width}, received {features.size(1)}"
                )
            if not features.is_floating_point():
                raise TypeError(
                    f"Node type {node_type!r} features must be floating point; "
                    f"received {features.dtype}"
                )
            features_by_type[node_type] = features
        return features_by_type

    def forward(self, data: HeteroData) -> HeteroData:
        """Encode one fresh batch in place and return that same graph object."""
        if not isinstance(data, HeteroData):
            raise TypeError(
                "Heterogeneous node feature encoder requires HeteroData"
            )
        self._validate_store_keys(data)
        features_by_type = self._validate_features(data)

        encoded_by_type = {
            node_type: self.dropout(
                self.activation(
                    self.projection_for(node_type)(features_by_type[node_type])
                )
            )
            for node_type, _ in self._input_channels
        }
        for node_type, encoded_features in encoded_by_type.items():
            data[node_type].x = encoded_features
        return data


__all__ = ["HeterogeneousNodeFeatureEncoder"]
