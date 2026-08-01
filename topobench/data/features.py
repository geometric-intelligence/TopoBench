"""Post-transform feature contracts for native homogeneous graphs."""

from __future__ import annotations

from collections.abc import Sequence, Sized
from numbers import Integral
from typing import Literal

import torch
from torch import Tensor
from torch_geometric.data import Data, HeteroData

GraphFeaturePolicy = Literal[
    "continuous",
    "categorical_one_hot",
    "degree",
    "constant",
]

_FEATURE_POLICIES = frozenset(
    {"continuous", "categorical_one_hot", "degree", "constant"}
)

OGB_ATOM_FEATURE_CARDINALITIES = (119, 5, 12, 12, 10, 6, 6, 2, 2)


def encode_categorical_columns(
    categories: Tensor,
    cardinalities: Sequence[int],
) -> Tensor:
    """Encode columns as deterministic concatenated one-hot blocks."""
    if not isinstance(categories, Tensor):
        raise TypeError("categories must be a torch.Tensor")
    if categories.ndim != 2:
        raise ValueError("categories must be rank-2")

    cardinality_values = tuple(cardinalities)
    if len(cardinality_values) != categories.shape[1]:
        raise ValueError(
            "cardinality count must match the number of category columns"
        )
    if not cardinality_values or any(
        isinstance(cardinality, bool)
        or not isinstance(cardinality, Integral)
        or cardinality <= 0
        for cardinality in cardinality_values
    ):
        raise ValueError("cardinalities must be positive integers")

    if categories.dtype == torch.bool or categories.is_complex():
        raise TypeError("categories must contain integral values")
    if categories.is_floating_point() and (
        not torch.all(torch.isfinite(categories))
        or not torch.all(categories == torch.trunc(categories))
    ):
        raise ValueError("categories must contain integral values")

    limits = torch.tensor(
        cardinality_values,
        dtype=torch.long,
        device=categories.device,
    )
    if torch.any(categories < 0) or torch.any(categories >= limits):
        raise ValueError("category value out of range for its column")

    values = categories.to(dtype=torch.long)
    offsets = torch.zeros_like(limits)
    offsets[1:] = torch.cumsum(limits[:-1], dim=0)
    column_indices = values + offsets
    encoded = torch.zeros(
        (categories.shape[0], sum(cardinality_values)),
        dtype=torch.float,
        device=categories.device,
    )
    return encoded.scatter_(1, column_indices, 1.0)


def _node_feature_channels(num_features: object) -> int:
    """Return the native node channel width from scalar or edge-aware config."""
    if isinstance(num_features, bool):
        raise TypeError("num_features must declare an integer node width")
    if isinstance(num_features, Integral):
        channels = int(num_features)
    else:
        try:
            channels = int(num_features[0])  # type: ignore[index]
        except (IndexError, KeyError, TypeError, ValueError) as error:
            raise TypeError(
                "num_features must declare an integer node width"
            ) from error
    if channels <= 0:
        raise ValueError("num_features must be positive")
    return channels


def _validate_one_hot(x: Tensor, policy: str) -> None:
    """Validate an exact one-hot post-transform representation."""
    if not torch.all((x == 0) | (x == 1)) or not torch.all(x.sum(dim=-1) == 1):
        raise ValueError(f"{policy} policy requires one-hot data.x")


def _validate_consistent_multi_hot(x: Tensor, policy: str) -> None:
    """Validate binary rows with the same positive number of active columns."""
    is_binary = torch.all((x == 0) | (x == 1))
    if x.shape[0] == 0:
        if not is_binary:
            raise ValueError(
                f"{policy} policy requires consistent multi-hot data.x"
            )
        return
    active_counts = x.sum(dim=-1)
    if (
        not is_binary
        or not torch.all(active_counts > 0)
        or not torch.all(active_counts == active_counts[0])
    ):
        raise ValueError(
            f"{policy} policy requires consistent multi-hot data.x"
        )


def validate_graph_features(
    data: Data,
    feature_policy: GraphFeaturePolicy | str,
    num_features: object,
) -> Data:
    """Validate one post-transform homogeneous graph.

    The boundary intentionally validates representation, rather than inferring
    or repairing it. Dataset defaults are responsible for deterministic feature
    creation before this function is called.
    """
    if not isinstance(data, Data) or isinstance(data, HeteroData):
        raise TypeError("graph feature validation requires homogeneous Data")
    if feature_policy not in _FEATURE_POLICIES:
        raise ValueError(
            f"unsupported graph feature policy: {feature_policy!r}"
        )

    x = data.get("x")
    if not isinstance(x, Tensor):
        raise ValueError("data.x is required after graph transforms")
    if x.ndim != 2:
        raise ValueError("data.x must be rank-2 after graph transforms")
    if not x.is_floating_point():
        raise TypeError(
            "data.x must have a floating dtype after graph transforms"
        )

    expected_channels = _node_feature_channels(num_features)
    if x.shape[1] != expected_channels:
        raise ValueError(
            f"expected {expected_channels} feature channels, got {x.shape[1]}"
        )
    if data.num_nodes is not None and x.shape[0] != data.num_nodes:
        raise ValueError("data.x must have one row per node")

    if feature_policy == "categorical_one_hot":
        _validate_consistent_multi_hot(x, feature_policy)
    elif feature_policy == "degree":
        _validate_one_hot(x, feature_policy)
    elif feature_policy == "constant" and not torch.all(x == 1):
        raise ValueError("constant policy requires data.x filled with ones")
    return data


def prepare_graph_features(
    dataset_train: Sized,
    dataset_val: Sized | None,
    dataset_test: Sized | None,
    *,
    feature_policy: GraphFeaturePolicy | str,
    num_features: object,
) -> tuple[Sized, Sized | None, Sized | None]:
    """Validate every graph in the post-transform phase datasets."""
    datasets = (dataset_train, dataset_val, dataset_test)
    for dataset in datasets:
        if dataset is None:
            continue
        for index in range(len(dataset)):
            validate_graph_features(
                dataset[index],  # type: ignore[index]
                feature_policy,
                num_features,
            )
    return datasets


__all__ = [
    "OGB_ATOM_FEATURE_CARDINALITIES",
    "encode_categorical_columns",
    "GraphFeaturePolicy",
    "prepare_graph_features",
    "validate_graph_features",
]
