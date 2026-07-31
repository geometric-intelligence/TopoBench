"""Native PyG hypergraph representation and validation contracts."""

from __future__ import annotations

from numbers import Integral
from typing import Any, ClassVar, Final

import torch
from torch import Tensor
from torch_geometric.data import Data

HYPERGRAPH_REPRESENTATION_VERSION: Final = 2
HYPERGRAPH_CACHE_FILENAME: Final = "hypergraph_data_v2.pt"
_MASK_NAMES: Final = ("train_mask", "val_mask", "test_mask")


class HypergraphData(Data):
    """Native hypergraph data whose incidence rows batch independently.

    ``hyperedge_index[0]`` contains node IDs and ``hyperedge_index[1]``
    contains hyperedge IDs. PyG therefore has to increment the two rows by
    different per-example counts when constructing a batch.
    """

    representation_version: ClassVar[int] = HYPERGRAPH_REPRESENTATION_VERSION

    def __inc__(
        self,
        key: str,
        value: Tensor,
        *args: Any,
        **kwargs: Any,
    ) -> int | Tensor:
        """Return independent node and hyperedge incidence offsets."""
        if key == "hyperedge_index":
            if self.num_nodes is None or self.num_hyperedges is None:
                raise ValueError(
                    "hypergraph batching requires node and hyperedge counts"
                )
            return torch.tensor(
                [[self.num_nodes], [self.num_hyperedges]],
                dtype=torch.long,
                device=value.device,
            )
        return super().__inc__(key, value, *args, **kwargs)


def _require_tensor(data: HypergraphData, field_name: str) -> Tensor:
    """Return one required tensor attribute with a field-local error."""
    if field_name not in data:
        raise ValueError(f"hypergraph data requires {field_name}")
    value = data[field_name]
    if not isinstance(value, Tensor):
        raise TypeError(f"{field_name} must be a torch.Tensor")
    return value


def _validate_features(data: HypergraphData) -> tuple[Tensor, int]:
    """Validate the native node feature matrix."""
    x = _require_tensor(data, "x")
    if x.ndim != 2:
        raise ValueError(f"x must be rank-2; received shape {tuple(x.shape)}")
    if not x.is_floating_point():
        raise TypeError(f"x must use a floating dtype; received {x.dtype}")
    if not bool(torch.isfinite(x).all()):
        raise ValueError("x must contain only finite values")

    num_nodes = int(x.size(0))
    if "num_nodes" in data:
        stored_num_nodes = data["num_nodes"]
        if (
            isinstance(stored_num_nodes, bool)
            or not isinstance(stored_num_nodes, Integral)
        ):
            raise TypeError("num_nodes must be an integer when explicitly set")
        if int(stored_num_nodes) != num_nodes:
            raise ValueError(
                "x row count must equal num_nodes; "
                f"received {num_nodes} rows for num_nodes={stored_num_nodes}"
            )
    return x, num_nodes


def _validate_num_hyperedges(data: HypergraphData) -> int:
    """Validate and return the explicit scalar hyperedge count."""
    if "num_hyperedges" not in data:
        raise ValueError("hypergraph data requires num_hyperedges")
    value = data["num_hyperedges"]
    if isinstance(value, bool) or not isinstance(value, Integral):
        raise TypeError("num_hyperedges must be an integer")
    num_hyperedges = int(value)
    if num_hyperedges < 1:
        raise ValueError("num_hyperedges must be positive")
    return num_hyperedges


def _validate_incidence(
    data: HypergraphData,
    *,
    num_nodes: int,
    num_hyperedges: int,
) -> Tensor:
    """Validate the native node-to-hyperedge incidence index."""
    hyperedge_index = _require_tensor(data, "hyperedge_index")
    if hyperedge_index.ndim != 2 or hyperedge_index.size(0) != 2:
        raise ValueError(
            "hyperedge_index must have shape [2, M]; "
            f"received {tuple(hyperedge_index.shape)}"
        )
    if hyperedge_index.dtype != torch.long:
        raise TypeError(
            "hyperedge_index must use torch.long; "
            f"received {hyperedge_index.dtype}"
        )

    node_ids = hyperedge_index[0]
    hyperedge_ids = hyperedge_index[1]
    if bool((node_ids < 0).any()):
        raise ValueError("hyperedge_index node indices must be nonnegative")
    if bool((hyperedge_ids < 0).any()):
        raise ValueError("hyperedge_index hyperedge IDs must be nonnegative")
    if bool((node_ids >= num_nodes).any()):
        raise ValueError(
            "hyperedge_index node indices must be smaller than num_nodes="
            f"{num_nodes}"
        )
    if bool((hyperedge_ids >= num_hyperedges).any()):
        raise ValueError(
            "hyperedge_index hyperedge IDs must be smaller than "
            f"num_hyperedges={num_hyperedges}"
        )

    counts = torch.bincount(hyperedge_ids, minlength=num_hyperedges)
    empty_ids = (counts == 0).nonzero(as_tuple=False).view(-1)
    if empty_ids.numel():
        raise ValueError(
            "hyperedge IDs must be contiguous from 0 to "
            f"num_hyperedges - 1; empty hyperedges={empty_ids.tolist()}"
        )
    return hyperedge_index


def validate_hypergraph_structure(data: HypergraphData) -> HypergraphData:
    """Validate native hypergraph structure without labels or split masks.

    This representation-only boundary is intended for raw loaders and pipeline
    preprocessing before phase masks are generated. It never normalizes,
    renumbers, or otherwise mutates ``data``.
    """
    if not isinstance(data, HypergraphData):
        raise TypeError("Expected native topobench.data.HypergraphData")
    _, num_nodes = _validate_features(data)
    num_hyperedges = _validate_num_hyperedges(data)
    _validate_incidence(
        data,
        num_nodes=num_nodes,
        num_hyperedges=num_hyperedges,
    )
    return data


def _validate_labels(data: HypergraphData, *, num_nodes: int) -> Tensor:
    """Validate and return one label per native node."""
    labels = _require_tensor(data, "y")
    if labels.ndim != 1:
        raise ValueError(f"y must be rank-1; received shape {tuple(labels.shape)}")
    if labels.size(0) != num_nodes:
        raise ValueError(
            f"y must contain one label per num_nodes={num_nodes}; "
            f"received {labels.size(0)}"
        )
    return labels


def _validate_masks(
    data: HypergraphData,
    *,
    num_nodes: int,
) -> tuple[Tensor, Tensor, Tensor]:
    """Validate a complete, nonempty, disjoint partition of labeled nodes."""
    masks: list[Tensor] = []
    for mask_name in _MASK_NAMES:
        mask = _require_tensor(data, mask_name)
        if mask.dtype != torch.bool:
            raise TypeError(
                f"{mask_name} must be boolean; received {mask.dtype}"
            )
        if mask.ndim != 1:
            raise ValueError(
                f"{mask_name} must be rank-1; received shape {tuple(mask.shape)}"
            )
        if mask.size(0) != num_nodes:
            raise ValueError(
                f"{mask_name} must have num_nodes={num_nodes} entries; "
                f"received {mask.size(0)}"
            )
        if not bool(mask.any()):
            raise ValueError(
                f"{mask_name} must select at least one labeled node"
            )
        masks.append(mask)

    for first_index, first_name in enumerate(_MASK_NAMES):
        for second_index in range(first_index + 1, len(_MASK_NAMES)):
            second_name = _MASK_NAMES[second_index]
            if bool((masks[first_index] & masks[second_index]).any()):
                raise ValueError(f"{first_name} and {second_name} overlap")

    covered = masks[0] | masks[1] | masks[2]
    if not bool(covered.all()):
        raise ValueError(
            "train_mask, val_mask, and test_mask must cover exactly all "
            "labeled nodes"
        )
    return masks[0], masks[1], masks[2]


def validate_hypergraph_node_data(data: HypergraphData) -> HypergraphData:
    """Validate a complete native hypergraph node-classification example.

    All checks are observational. The identical input object is returned only
    after its representation, labels, and masks satisfy the complete contract.
    No malformed input is normalized or partially updated.
    """
    validate_hypergraph_structure(data)
    num_nodes = int(data.x.size(0))
    _validate_labels(data, num_nodes=num_nodes)
    _validate_masks(data, num_nodes=num_nodes)
    return data


__all__ = [
    "HYPERGRAPH_CACHE_FILENAME",
    "HYPERGRAPH_REPRESENTATION_VERSION",
    "HypergraphData",
    "validate_hypergraph_node_data",
    "validate_hypergraph_structure",
]
