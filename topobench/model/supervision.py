"""Typed supervision selection for homogeneous and heterogeneous batches.

The adapters in this module isolate which predictions own the loss for one
training phase.  They intentionally do not mutate model outputs or data
batches; :class:`topobench.model.TBModel` remains the compatibility boundary
that writes the selected tensors back into its output mapping.
"""

from __future__ import annotations

from dataclasses import dataclass
from numbers import Integral
from typing import Literal, Protocol

import torch
from torch import Tensor
from torch_geometric.data import Data, HeteroData

SamplingMode = Literal["full_batch", "neighbor"]

_PHASE_MASK = {
    "Training": "train_mask",
    "Validation": "val_mask",
    "Test": "test_mask",
}


@dataclass(frozen=True)
class SupervisedBatch:
    """Frozen field container for one model step's supervised tensors.

    Freezing prevents field reassignment; it does not make tensor contents
    deeply immutable. Logits and targets are intentionally returned without
    cloning or detaching. They therefore either alias the input tensors or
    remain autograd-connected masked/sliced selections.
    """

    logits: Tensor
    targets: Tensor
    num_examples: int
    row_indices: Tensor | None = None

    def __post_init__(self) -> None:
        """Validate the public result even when constructed directly."""
        if not isinstance(self.logits, Tensor) or not isinstance(
            self.targets, Tensor
        ):
            raise TypeError("logits and targets must be tensors")
        if type(self.num_examples) is not int:
            raise TypeError("num_examples must be a positive built-in int")
        if self.num_examples < 1:
            raise ValueError("num_examples must be a positive built-in int")
        if (
            self.logits.ndim == 0
            or self.targets.ndim == 0
            or self.logits.shape[0] != self.num_examples
            or self.targets.shape[0] != self.num_examples
        ):
            raise ValueError("logits and targets must align with num_examples")
        row_indices = self.row_indices
        if row_indices is None:
            row_indices = torch.arange(
                self.num_examples,
                dtype=torch.long,
                device=self.logits.device,
            )
            object.__setattr__(self, "row_indices", row_indices)
        if not isinstance(row_indices, Tensor):
            raise TypeError("row_indices must be a tensor")
        if row_indices.dtype != torch.long or row_indices.ndim != 1:
            raise TypeError("row_indices must be a rank-1 torch.long tensor")
        if row_indices.numel() != self.num_examples:
            raise ValueError("row_indices must align with num_examples")
        if bool(torch.any(row_indices < 0)):
            raise ValueError("row_indices must be non-negative")
        if row_indices.numel() > 1 and bool(
            torch.any(row_indices[1:] <= row_indices[:-1])
        ):
            raise ValueError(
                "row_indices must be unique and strictly increasing"
            )


class SupervisionAdapter(Protocol):
    """Select the predictions and labels owned by one training phase."""

    def select(
        self,
        model_out: dict[str, object],
        batch: Data | HeteroData,
        phase: str,
    ) -> SupervisedBatch:
        """Return the loss-owning predictions, labels, and example count."""


def _model_tensors(
    model_out: dict[str, object],
) -> tuple[Tensor, Tensor]:
    """Extract and validate prediction and label tensors."""
    logits = model_out.get("logits")
    labels = model_out.get("labels")
    if not isinstance(logits, Tensor) or not isinstance(labels, Tensor):
        raise TypeError("model_out must contain tensor logits and labels")
    if logits.ndim == 0 or labels.ndim == 0:
        raise ValueError("Logits and labels must have a leading dimension")
    if logits.size(0) != labels.size(0):
        raise ValueError("Logit and label counts must match before selection")
    return logits, labels


def _homogeneous_targets(
    logits: Tensor,
    labels: Tensor,
    task_level: str,
) -> Tensor:
    """Validate and normalize one native homogeneous target contract."""
    if logits.ndim != 2 or logits.size(1) < 1:
        raise ValueError(
            "Homogeneous logits must have shape [examples, outputs]"
        )

    if logits.size(1) == 1:
        if task_level != "graph":
            raise ValueError(
                "Homogeneous node tasks support classification only"
            )
        if not labels.is_floating_point():
            raise TypeError("Scalar regression targets must be floating")
        if not torch.isfinite(labels).all():
            raise ValueError("Scalar regression targets must be finite")
        if labels.ndim == 1:
            targets = labels.unsqueeze(1)
        elif labels.ndim == 2 and labels.size(1) == 1:
            targets = labels
        else:
            raise ValueError(
                "Scalar regression targets must have shape [B] or [B, 1]"
            )
        if logits.shape != targets.shape:
            raise ValueError(
                "Scalar regression logits and targets must have shape [B, 1]"
            )
        return targets

    if labels.ndim != 1:
        raise ValueError(
            "classification targets must be a rank-1 tensor [B] or [N]"
        )
    if labels.dtype is not torch.long:
        raise TypeError("classification targets must have dtype torch.long")
    return labels


def _mask_from_store(
    store: object,
    mask_name: str,
    expected_count: int,
) -> Tensor:
    """Read a valid one-dimensional boolean phase mask from a data store."""
    try:
        mask = store[mask_name]  # type: ignore[index]
    except (KeyError, TypeError, AttributeError) as error:
        raise ValueError(f"Batch is missing boolean {mask_name}") from error
    if not isinstance(mask, Tensor) or mask.dtype is not torch.bool:
        raise TypeError(f"{mask_name} must be a rank-1 bool tensor")
    if mask.ndim != 1:
        raise ValueError(f"{mask_name} must be a rank-1 bool tensor")
    if mask.size(0) != expected_count:
        raise ValueError(
            f"{mask_name} count must match target outputs "
            f"({mask.size(0)} != {expected_count})"
        )
    return mask


class DefaultSupervisionAdapter:
    """Select strictly shaped native homogeneous supervision."""

    def __init__(self, task_level: str) -> None:
        """Create an adapter for one supported homogeneous task level."""
        if not isinstance(task_level, str):
            raise TypeError(
                "task_level must be graph, node, or node_inductive"
            )
        if task_level not in {"graph", "node", "node_inductive"}:
            raise ValueError(
                "task_level must be graph, node, or node_inductive"
            )
        self.task_level = task_level

    def select(
        self,
        model_out: dict[str, object],
        batch: Data | HeteroData,
        phase: str,
    ) -> SupervisedBatch:
        """Validate targets and select transductive node masks."""
        logits, source_labels = _model_tensors(model_out)
        targets = _homogeneous_targets(
            logits,
            source_labels,
            self.task_level,
        )
        if self.task_level != "node":
            count = targets.size(0)
            return SupervisedBatch(
                logits,
                targets,
                num_examples=count,
                row_indices=torch.arange(
                    count,
                    dtype=torch.long,
                    device=logits.device,
                ),
            )

        try:
            mask_name = _PHASE_MASK[phase]
        except KeyError as error:
            raise ValueError(f"Invalid state_str: {phase}") from error
        mask = _mask_from_store(batch, mask_name, targets.size(0))
        row_indices = mask.nonzero(as_tuple=False).view(-1)
        selected = int(row_indices.numel())
        if selected == 0:
            raise ValueError(f"No supervised examples for {phase}")
        return SupervisedBatch(
            logits=logits.index_select(0, row_indices),
            targets=targets.index_select(0, row_indices),
            num_examples=selected,
            row_indices=row_indices,
        )


class HeterogeneousNodeSupervisionAdapter:
    """Select target-node supervision from full or sampled hetero batches."""

    def __init__(
        self,
        target_node_type: str,
        mode: SamplingMode,
    ) -> None:
        """Create an adapter for one target node type and sampling mode."""
        if not isinstance(target_node_type, str):
            raise TypeError("target_node_type must be a non-empty string")
        if not target_node_type.strip():
            raise ValueError("target_node_type must be a non-empty string")
        if not isinstance(mode, str):
            raise TypeError(
                f"Unsupported sampling mode: {mode!r}; expected a string"
            )
        if mode not in {
            "full_batch",
            "neighbor",
        }:
            raise ValueError(f"Unsupported sampling mode: {mode}")
        self.target_node_type = target_node_type
        self.mode: SamplingMode = mode

    def select(
        self,
        model_out: dict[str, object],
        batch: Data | HeteroData,
        phase: str,
    ) -> SupervisedBatch:
        """Select target masks in full mode or leading seeds in neighbor mode."""
        if not isinstance(batch, HeteroData):
            raise TypeError("Heterogeneous supervision requires HeteroData")
        if phase not in _PHASE_MASK:
            raise ValueError(f"Invalid state_str: {phase}")
        if self.target_node_type not in batch.node_types:
            raise ValueError(
                f"Batch has no target store {self.target_node_type!r}"
            )

        logits, labels = _model_tensors(model_out)
        if logits.ndim != 2 or labels.ndim != 1:
            raise ValueError(
                "Heterogeneous classification expects [N, C] logits "
                "and [N] labels"
            )
        target_store = batch[self.target_node_type]

        if self.mode == "neighbor":
            raw_seed_count = target_store.get("batch_size")
            if raw_seed_count is None:
                raise ValueError(
                    "Neighbor batch is missing target-store batch_size"
                )
            if isinstance(raw_seed_count, bool) or not isinstance(
                raw_seed_count, Integral
            ):
                raise TypeError(
                    "Target-store batch_size must be a non-boolean integer"
                )
            seed_count = int(raw_seed_count)
            if seed_count < 1 or seed_count > labels.size(0):
                raise ValueError(f"Invalid target seed count: {seed_count}")
            return SupervisedBatch(
                logits=logits[:seed_count],
                targets=labels[:seed_count],
                num_examples=seed_count,
                row_indices=torch.arange(
                    seed_count,
                    dtype=torch.long,
                    device=logits.device,
                ),
            )

        mask_name = _PHASE_MASK[phase]
        mask = _mask_from_store(
            target_store,
            mask_name,
            expected_count=labels.size(0),
        )
        row_indices = mask.nonzero(as_tuple=False).view(-1)
        selected = int(row_indices.numel())
        if selected == 0:
            raise ValueError(f"No supervised target nodes for {phase}")
        return SupervisedBatch(
            logits=logits.index_select(0, row_indices),
            targets=labels.index_select(0, row_indices),
            num_examples=selected,
            row_indices=row_indices,
        )


__all__ = [
    "DefaultSupervisionAdapter",
    "HeterogeneousNodeSupervisionAdapter",
    "SamplingMode",
    "SupervisedBatch",
    "SupervisionAdapter",
]
