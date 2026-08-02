"""Post-transform feature contracts for native homogeneous graphs."""

from __future__ import annotations

from collections.abc import Sequence, Sized
from numbers import Integral
from typing import Literal

import torch
from torch import Tensor
from torch_geometric.data import Data, HeteroData
from topobench.data.capabilities import (
    GraphDatasetCapability,
    OGB_ATOM_FEATURE_CARDINALITIES,
    validate_classification_vocabulary,
)

GraphFeaturePolicy = Literal[
    "continuous",
    "categorical_one_hot",
    "degree",
    "constant",
]

_FEATURE_POLICIES = frozenset(
    {"continuous", "categorical_one_hot", "degree", "constant"}
)



def validate_categorical_columns(
    categories: Tensor,
    cardinalities: Sequence[int],
    *,
    context: str = "categories",
    allow_integral_float: bool = False,
) -> tuple[int, ...]:
    """Validate compact categorical columns without expanding their width."""
    if not isinstance(categories, Tensor):
        raise TypeError(f"{context} must be a torch.Tensor")
    if categories.ndim != 2:
        raise ValueError(f"{context} must be rank-2")

    cardinality_values = tuple(cardinalities)
    if not cardinality_values or any(
        isinstance(cardinality, bool)
        or not isinstance(cardinality, Integral)
        or cardinality <= 0
        for cardinality in cardinality_values
    ):
        raise ValueError("cardinalities must be positive integers")
    if categories.shape[1] != len(cardinality_values):
        raise ValueError(
            f"{context} must have exactly {len(cardinality_values)} "
            "categorical columns"
        )
    if categories.dtype == torch.bool or categories.is_complex():
        raise TypeError(f"{context} must have an integral dtype")
    if categories.is_floating_point():
        if not allow_integral_float:
            raise TypeError(f"{context} must have an integral dtype")
        if (
            not torch.isfinite(categories).all()
            or not torch.all(categories == torch.trunc(categories))
        ):
            raise ValueError(
                f"{context} must contain exact integral categories"
            )

    for column, cardinality in enumerate(cardinality_values):
        values = categories[:, column]
        if torch.any(values < 0) or torch.any(values >= cardinality):
            raise ValueError(
                f"{context} column {column} contains a category outside "
                f"the valid range [0, {cardinality})"
            )
    return tuple(int(value) for value in cardinality_values)


def encode_categorical_columns(
    categories: Tensor,
    cardinalities: Sequence[int],
    *,
    context: str = "categories",
    allow_integral_float: bool = False,
) -> Tensor:
    """Encode validated columns as concatenated one-hot blocks."""
    cardinality_values = validate_categorical_columns(
        categories,
        cardinalities,
        context=context,
        allow_integral_float=allow_integral_float,
    )
    values = categories.to(dtype=torch.long)
    limits = torch.tensor(
        cardinality_values,
        dtype=torch.long,
        device=categories.device,
    )
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
    *,
    base_num_features: object,
    total_num_features: object,
    categorical_cardinalities: Sequence[int] = (),
    selector: str = "<graph>",
    item: str = "graph",
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

    context = f"{selector}: {item} field x"
    x = data.get("x")
    if not isinstance(x, Tensor):
        raise ValueError(
            f"{context} observed shape=None, dtype=None; data.x is required "
            "as a rank-2 floating tensor after graph transforms"
        )
    if x.ndim != 2:
        raise ValueError(
            f"{context} observed shape={tuple(x.shape)}, dtype={x.dtype}; "
            "data.x must be rank-2 after graph transforms"
        )
    if x.shape[0] == 0:
        raise ValueError(
            f"{context} observed shape={tuple(x.shape)}, dtype={x.dtype}; "
            "expected at least one node"
        )

    base_channels = _node_feature_channels(base_num_features)
    expected_channels = _node_feature_channels(total_num_features)
    cardinalities = tuple(categorical_cardinalities)
    if cardinalities:
        if base_channels != len(cardinalities):
            raise ValueError(
                f"{context} categorical cardinality count must equal "
                f"base_num_features={base_channels}"
            )
        if expected_channels < len(cardinalities):
            raise ValueError(
                f"{context} categorical cardinality count exceeds "
                f"total_num_features={expected_channels}"
            )
        if x.shape[1] != expected_channels:
            raise ValueError(
                f"{context} observed shape={tuple(x.shape)}, dtype={x.dtype}; "
                f"expected shape=(N, {expected_channels})"
            )
        categorical_prefix = x[:, : len(cardinalities)]
        validate_categorical_columns(
            categorical_prefix,
            cardinalities,
            context=f"{context} categorical prefix",
            allow_integral_float=True,
        )
        continuous_suffix = x[:, len(cardinalities) :]
        if continuous_suffix.shape[1]:
            if not x.is_floating_point():
                raise TypeError(
                    f"{context} continuous suffix must have a floating dtype"
                )
            if not torch.isfinite(continuous_suffix).all():
                raise ValueError(
                    f"{context} continuous suffix must contain finite values"
                )
        if data.num_nodes is not None and x.shape[0] != data.num_nodes:
            raise ValueError(
                f"{context} observed shape={tuple(x.shape)}, "
                f"dtype={x.dtype}; expected one row per "
                f"num_nodes={data.num_nodes}"
            )
        return data

    if not x.is_floating_point():
        raise TypeError(
            f"{context} observed shape={tuple(x.shape)}, dtype={x.dtype}; "
            "data.x must have a floating dtype after graph transforms"
        )
    if not torch.isfinite(x).all():
        raise ValueError(
            f"{context} observed shape={tuple(x.shape)}, dtype={x.dtype}; "
            "expected finite feature values"
        )
    if expected_channels < base_channels:
        raise ValueError(
            f"{context} observed shape={tuple(x.shape)}, dtype={x.dtype}; "
            "expected total_num_features to be at least "
            f"base_num_features={base_channels}"
        )
    if x.shape[1] != expected_channels:
        raise ValueError(
            f"{context} observed shape={tuple(x.shape)}, dtype={x.dtype}; "
            f"expected shape=(N, {expected_channels}); expected "
            f"{expected_channels} feature channels"
        )
    if data.num_nodes is not None and x.shape[0] != data.num_nodes:
        raise ValueError(
            f"{context} observed shape={tuple(x.shape)}, dtype={x.dtype}; "
            f"expected one row per num_nodes={data.num_nodes}"
        )

    base_features = x[:, :base_channels]
    appended_features = x[:, base_channels:]

    if feature_policy == "categorical_one_hot":
        _validate_consistent_multi_hot(base_features, feature_policy)
    elif feature_policy == "degree":
        _validate_one_hot(base_features, feature_policy)
    elif feature_policy == "constant" and not torch.all(base_features == 1):
        raise ValueError("constant policy requires data.x filled with ones")
    return data


def validate_qualified_graph_source(
    dataset: Sized,
    *,
    capability: GraphDatasetCapability,
    configured_num_classes: object,
    total_num_features: object,
) -> Sized:
    """Validate one complete qualified graph source before phase splitting."""
    selector = capability.selector
    if len(dataset) == 0:
        raise ValueError(
            f"{selector}: full source observed shape=(0,); expected a "
            "non-empty graph source"
        )

    def classification_labels():
        for index in range(len(dataset)):
            item = f"graph[{index}]"
            data = dataset[index]  # type: ignore[index]
            validate_graph_features(
                data,
                capability.feature_policy,
                base_num_features=capability.feature_width,
                total_num_features=total_num_features,
                categorical_cardinalities=capability.feature_cardinalities,
                selector=selector,
                item=item,
            )
            expected_size = (
                1
                if capability.task_level == "graph"
                else int(data.num_nodes)
            )
            yield item, data.get("y"), expected_size

    if capability.task == "classification":
        validate_classification_vocabulary(
            classification_labels(),
            selector=selector,
            field="y",
            configured_num_classes=configured_num_classes,
            manifest_num_classes=capability.num_classes,
            allow_incomplete=capability.allow_incomplete_class_vocabulary,
        )
    else:
        for index in range(len(dataset)):
            item = f"graph[{index}]"
            data = dataset[index]  # type: ignore[index]
            validate_graph_features(
                data,
                capability.feature_policy,
                base_num_features=capability.feature_width,
                total_num_features=total_num_features,
                categorical_cardinalities=capability.feature_cardinalities,
                selector=selector,
                item=item,
            )
            target = data.get("y")
            shape = (
                tuple(target.shape)
                if isinstance(target, Tensor)
                else None
            )
            dtype = (
                target.dtype
                if isinstance(target, Tensor)
                else type(target).__name__
            )
            context = (
                f"{selector}: {item} field y observed "
                f"shape={shape}, dtype={dtype}"
            )
            if not isinstance(target, Tensor):
                raise ValueError(
                    f"{context}; target is required with shape=(1,)"
                )
            if not target.is_floating_point():
                raise TypeError(
                    f"{context}; expected floating dtype"
                )
            if target.shape != (1,):
                raise ValueError(
                    f"{context}; expected shape=(1,)"
                )
            if not torch.isfinite(target).all():
                raise ValueError(
                    f"{context}; expected one finite scalar target"
                )
    return dataset


def prepare_graph_features(
    dataset_train: Sized,
    dataset_val: Sized | None,
    dataset_test: Sized | None,
    *,
    feature_policy: GraphFeaturePolicy | str,
    base_num_features: object,
    total_num_features: object,
    categorical_cardinalities: Sequence[int] = (),
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
                base_num_features=base_num_features,
                total_num_features=total_num_features,
                categorical_cardinalities=categorical_cardinalities,
            )
    return datasets


__all__ = [
    "OGB_ATOM_FEATURE_CARDINALITIES",
    "encode_categorical_columns",
    "validate_categorical_columns",
    "GraphFeaturePolicy",
    "prepare_graph_features",
    "validate_graph_features",
    "validate_qualified_graph_source",
]
