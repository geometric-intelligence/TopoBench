"""TopoBench-owned evaluator lifecycle and backend contracts."""

from __future__ import annotations

from abc import ABC, abstractmethod
from collections.abc import Mapping
from typing import Protocol, runtime_checkable

from .types import (
    EvaluationBatch,
    EvaluationContext,
    EvaluationResult,
    MetricScalar,
)


@runtime_checkable
class EvaluatorBackend(Protocol):
    """Minimal injected state backend, independent of any metric library."""

    def begin(self, context: EvaluationContext) -> None:
        """Prepare backend state for one context."""

    def update(self, batch: EvaluationBatch) -> None:
        """Accumulate one already-validated batch."""

    def compute(self) -> MetricScalar | Mapping[str, MetricScalar]:
        """Compute scalar state without resetting it."""

    def reset(self) -> None:
        """Clear all mutable state for the active context."""


class AbstractEvaluator(ABC):
    """Explicit lifecycle implemented by every TopoBench evaluator."""

    @abstractmethod
    def begin(self, context: EvaluationContext) -> None:
        """Open one otherwise idle evaluation context."""

    @abstractmethod
    def update(self, batch: EvaluationBatch) -> None:
        """Jointly commit one supervised batch to every backend."""

    @abstractmethod
    def snapshot(self) -> EvaluationResult:
        """Return current immutable results without resetting state."""

    @abstractmethod
    def finalize(self) -> EvaluationResult:
        """Return final immutable results and clear mutable state."""

    @abstractmethod
    def abort(self) -> None:
        """Clear the active or recorded-failure context without results."""
