"""Canonical homogeneous graph split representations and validation."""

from __future__ import annotations

from collections.abc import Mapping, Sequence
from numbers import Integral
from typing import Any

import torch
from torch.utils.data import Dataset, Subset
from torch_geometric.data import Data

_SPLIT_MASK_NAMES = ("train_mask", "val_mask", "test_mask")


def _ensure_global_nid(data: Data, num_nodes: int) -> None:
    """Attach or validate stable source node ordinals without copying features."""
    global_nid = getattr(data, "global_nid", None)
    if global_nid is None:
        data.global_nid = torch.arange(num_nodes, dtype=torch.long)
        return
    if (
        not isinstance(global_nid, torch.Tensor)
        or global_nid.dtype == torch.bool
        or global_nid.is_floating_point()
        or global_nid.is_complex()
        or global_nid.ndim != 1
        or global_nid.numel() != num_nodes
    ):
        raise ValueError(
            "global_nid must be a rank-one integer tensor aligned to nodes"
        )
    if bool(torch.any(global_nid < 0)):
        raise ValueError(
            "global_nid must contain non-negative source ordinals"
        )
    if torch.unique(global_nid).numel() != num_nodes:
        raise ValueError("global_nid must contain unique source ordinals")


def _source_ordinal(dataset: Dataset[Data], index: int) -> int:
    """Resolve an index through nested shallow Subset views."""
    current: Dataset[Data] = dataset
    ordinal = int(index)
    while isinstance(current, Subset):
        ordinal = int(current.indices[ordinal])
        current = current.dataset
    return ordinal


class _SourceOrdinalSubset(Subset[Data]):
    """Lazy phase view that attaches a stable graph source ordinal on access."""

    def __getitem__(self, index: int) -> Data:
        source_index = int(self.indices[index])
        data = self.dataset[source_index]
        if not isinstance(data, Data):
            raise TypeError("inductive graph views must contain native Data")
        if getattr(data, "sample_id", None) is None:
            data.sample_id = torch.tensor(
                [_source_ordinal(self.dataset, source_index)],
                dtype=torch.long,
            )
        return data

    def __getitems__(self, indices: list[int]) -> list[Data]:
        return [self[index] for index in indices]


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
    if (
        tensor.dtype == torch.bool
        or tensor.is_floating_point()
        or tensor.is_complex()
    ):
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
    if (
        torch.any(train_mask & val_mask)
        or torch.any(train_mask & test_mask)
        or torch.any(val_mask & test_mask)
    ):
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
    _ensure_global_nid(data, num_nodes)


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
    _ensure_global_nid(data, num_nodes)
    return data


def _normalize_inductive_partition(
    split_idx: Mapping[str, Sequence[int] | torch.Tensor],
    num_graphs: int,
) -> tuple[list[int], list[int], list[int]]:
    """Normalize and validate one complete graph-level phase partition."""
    phase_keys = ("train", "valid", "test")
    normalized: list[torch.Tensor] = []
    for key in phase_keys:
        if key not in split_idx:
            raise ValueError(f"split_idx must contain {key!r}")
        try:
            indices = _index_tensor(split_idx[key], num_graphs)
        except (TypeError, ValueError) as error:
            raise ValueError(f"{key} split {error}") from error
        if indices.numel() == 0:
            raise ValueError(f"{key} split must not be empty")
        normalized.append(indices)

    combined = torch.cat(normalized)
    if torch.unique(combined).numel() != combined.numel():
        raise ValueError(
            "train, valid, and test splits must be pairwise disjoint"
        )
    if not torch.equal(
        torch.sort(combined).values,
        torch.arange(num_graphs),
    ):
        raise ValueError(
            "train, valid, and test splits must cover every source index "
            "exactly once"
        )

    return tuple(indices.tolist() for indices in normalized)  # type: ignore[return-value]


def inductive_split_views(
    dataset: Dataset[Data],
    split_idx: Mapping[str, Sequence[int] | torch.Tensor],
) -> tuple[Subset[Data], Subset[Data], Subset[Data]]:
    """Return validated index-backed phase views over one source dataset."""
    train_indices, valid_indices, test_indices = (
        _normalize_inductive_partition(split_idx, len(dataset))
    )
    return (
        _SourceOrdinalSubset(dataset, train_indices),
        _SourceOrdinalSubset(dataset, valid_indices),
        _SourceOrdinalSubset(dataset, test_indices),
    )


__all__ = [
    "apply_transductive_split",
    "indices_to_mask",
    "inductive_split_views",
    "validate_transductive_masks",
]
