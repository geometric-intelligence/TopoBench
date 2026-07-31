"""Canonical homogeneous graph split representations and validation."""

from __future__ import annotations

from collections.abc import Mapping, Sequence
from numbers import Integral
from typing import Any

import torch
from torch.utils.data import Dataset, Subset
from torch_geometric.data import Data

_SPLIT_MASK_NAMES = ("train_mask", "val_mask", "test_mask")


def _normalize_num_nodes(num_nodes: int) -> int:
    """Return a validated node count."""
    if isinstance(num_nodes, bool) or not isinstance(num_nodes, Integral):
        raise TypeError("num_nodes must be an integer")
    normalized = int(num_nodes)
    if normalized < 0:
        raise ValueError("num_nodes must be non-negative")
    return normalized


def _index_tensor(indices: Any, num_nodes: int) -> torch.Tensor:
    """Normalize and validate one rank-one integer index collection."""
    try:
        tensor = torch.as_tensor(indices)
    except (TypeError, ValueError) as error:
        raise TypeError("indices must contain integers") from error
    if tensor.ndim != 1:
        raise ValueError("indices must be rank-1")
    if tensor.numel() == 0:
        return torch.empty(0, dtype=torch.long)
    if tensor.dtype == torch.bool or tensor.is_floating_point() or tensor.is_complex():
        raise TypeError("indices must contain integers")
    tensor = tensor.to(dtype=torch.long, device="cpu")
    if torch.unique(tensor).numel() != tensor.numel():
        raise ValueError("indices must be unique")
    if torch.any(tensor < 0) or torch.any(tensor >= num_nodes):
        raise ValueError(f"indices must be in [0, {num_nodes})")
    return tensor


def indices_to_mask(
    indices: Sequence[int] | torch.Tensor,
    num_nodes: int,
) -> torch.Tensor:
    """Convert node indices to a full-length rank-one boolean mask.

    Parameters
    ----------
    indices : Sequence[int] | torch.Tensor
        Unique, rank-one integer indices.
    num_nodes : int
        Length of the returned mask.

    Returns
    -------
    torch.Tensor
        Boolean mask of shape ``[num_nodes]``.
    """
    num_nodes = _normalize_num_nodes(num_nodes)
    index = _index_tensor(indices, num_nodes)
    mask = torch.zeros(num_nodes, dtype=torch.bool)
    mask[index] = True
    return mask


def _labeled_mask(data: Data, num_nodes: int) -> torch.Tensor:
    """Return the full labeled-node policy for homogeneous node tasks."""
    if data.y is None or data.y.ndim == 0 or data.y.shape[0] != num_nodes:
        raise ValueError("node labels must have one entry per node")
    return torch.ones(num_nodes, dtype=torch.bool)


def _validate_masks(
    masks: tuple[torch.Tensor, torch.Tensor, torch.Tensor],
    labeled_mask: torch.Tensor,
) -> None:
    """Validate canonical mask shape, partition, and coverage invariants."""
    num_nodes = labeled_mask.numel()
    for name, mask in zip(_SPLIT_MASK_NAMES, masks, strict=True):
        if (
            not isinstance(mask, torch.Tensor)
            or mask.dtype != torch.bool
            or mask.ndim != 1
        ):
            raise ValueError(f"{name} must be a rank-1 boolean mask")
        if mask.numel() != num_nodes:
            raise ValueError(f"{name} must have length {num_nodes}")
        if not torch.any(mask):
            raise ValueError("split masks must be non-empty")

    train_mask, val_mask, test_mask = masks
    if torch.any(train_mask & val_mask) or torch.any(
        train_mask & test_mask
    ) or torch.any(val_mask & test_mask):
        raise ValueError("split masks must be disjoint")
    if not torch.equal(
        train_mask | val_mask | test_mask,
        labeled_mask,
    ):
        raise ValueError("split masks must cover all labeled nodes")


def validate_transductive_masks(data: Data) -> None:
    """Validate canonical train, validation, and test masks on one graph."""
    num_nodes = data.num_nodes
    if num_nodes is None:
        raise ValueError("transductive data must define num_nodes")
    num_nodes = _normalize_num_nodes(num_nodes)
    masks = tuple(getattr(data, name, None) for name in _SPLIT_MASK_NAMES)
    _validate_masks(
        masks,  # type: ignore[arg-type]
        _labeled_mask(data, num_nodes),
    )


def apply_transductive_split(
    data: Data,
    *,
    train: Sequence[int] | torch.Tensor,
    val: Sequence[int] | torch.Tensor,
    test: Sequence[int] | torch.Tensor,
) -> Data:
    """Apply index splits to one graph as validated canonical boolean masks."""
    num_nodes = data.num_nodes
    if num_nodes is None:
        raise ValueError("transductive data must define num_nodes")
    num_nodes = _normalize_num_nodes(num_nodes)
    masks = (
        indices_to_mask(train, num_nodes),
        indices_to_mask(val, num_nodes),
        indices_to_mask(test, num_nodes),
    )
    _validate_masks(masks, _labeled_mask(data, num_nodes))
    data.train_mask, data.val_mask, data.test_mask = masks
    return data


def inductive_split_views(
    dataset: Dataset[Data],
    split_idx: Mapping[str, Sequence[int] | torch.Tensor],
) -> tuple[Subset[Data], Subset[Data], Subset[Data]]:
    """Return non-empty index-backed phase views over one source dataset."""
    num_graphs = len(dataset)
    phase_keys = ("train", "valid", "test")
    normalized: list[list[int]] = []
    for key in phase_keys:
        if key not in split_idx:
            raise ValueError(f"split_idx must contain {key!r}")
        indices = _index_tensor(split_idx[key], num_graphs)
        if indices.numel() == 0:
            raise ValueError(f"{key} split must not be empty")
        normalized.append(indices.tolist())

    return (
        Subset(dataset, normalized[0]),
        Subset(dataset, normalized[1]),
        Subset(dataset, normalized[2]),
    )


__all__ = [
    "apply_transductive_split",
    "indices_to_mask",
    "inductive_split_views",
    "validate_transductive_masks",
]
