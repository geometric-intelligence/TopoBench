"""Post-transform feature contracts for native homogeneous graphs."""

from __future__ import annotations

from collections.abc import Sized
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
    if not torch.all((x == 0) | (x == 1)) or not torch.all(
        x.sum(dim=-1) == 1
    ):
        raise ValueError(f"{policy} policy requires one-hot data.x")


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
        raise ValueError(f"unsupported graph feature policy: {feature_policy!r}")

    x = data.get("x")
    if not isinstance(x, Tensor):
        raise ValueError("data.x is required after graph transforms")
    if x.ndim != 2:
        raise ValueError("data.x must be rank-2 after graph transforms")
    if not x.is_floating_point():
        raise TypeError("data.x must have a floating dtype after graph transforms")

    expected_channels = _node_feature_channels(num_features)
    if x.shape[1] != expected_channels:
        raise ValueError(
            f"expected {expected_channels} feature channels, got {x.shape[1]}"
        )
    if data.num_nodes is not None and x.shape[0] != data.num_nodes:
        raise ValueError("data.x must have one row per node")

    if feature_policy in {"categorical_one_hot", "degree"}:
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
    "GraphFeaturePolicy",
    "prepare_graph_features",
    "validate_graph_features",
]
