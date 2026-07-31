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
    """Predictions, targets, and reduction weight owned by one model step."""

    logits: Tensor
    targets: Tensor
    num_examples: int

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
    """Preserve TopoBench's legacy homogeneous supervision semantics."""

    def __init__(self, task_level: str) -> None:
        """Create an adapter for a readout task level."""
        if not isinstance(task_level, str):
            raise TypeError("task_level must be a non-empty string")
        if not task_level.strip():
            raise ValueError("task_level must be a non-empty string")
        self.task_level = task_level

    def select(
        self,
        model_out: dict[str, object],
        batch: Data | HeteroData,
        phase: str,
    ) -> SupervisedBatch:
        """Select top-level masks for transductive node classification."""
        logits, labels = _model_tensors(model_out)
        if self.task_level != "node":
            # Preserve historical Lightning weighting for graph and
            # node-inductive tasks.
            return SupervisedBatch(logits, labels, num_examples=1)

        try:
            mask_name = _PHASE_MASK[phase]
        except KeyError as error:
            raise ValueError(f"Invalid state_str: {phase}") from error
        mask = _mask_from_store(batch, mask_name, labels.size(0))
        selected = int(mask.sum().item())
        if selected == 0:
            raise ValueError(f"No supervised examples for {phase}")
        return SupervisedBatch(
            logits=logits[mask],
            targets=labels[mask],
            num_examples=selected,
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
            )

        mask_name = _PHASE_MASK[phase]
        mask = _mask_from_store(
            target_store,
            mask_name,
            expected_count=labels.size(0),
        )
        selected = int(mask.sum().item())
        if selected == 0:
            raise ValueError(f"No supervised target nodes for {phase}")
        return SupervisedBatch(
            logits=logits[mask],
            targets=labels[mask],
            num_examples=selected,
        )


__all__ = [
    "DefaultSupervisionAdapter",
    "HeterogeneousNodeSupervisionAdapter",
    "SamplingMode",
    "SupervisedBatch",
    "SupervisionAdapter",
]
