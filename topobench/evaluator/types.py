"""Immutable TopoBench-owned evaluator value contracts."""

from __future__ import annotations

from collections.abc import Hashable, Mapping
from dataclasses import dataclass, field
from numbers import Real
from types import MappingProxyType
from typing import Any, Literal, TypeAlias

import torch

from .prediction import PredictionPayload

EvaluationTask: TypeAlias = Literal["classification", "regression"]
EvaluationSplit: TypeAlias = Literal["train", "val", "test"]
EvaluationPassKind: TypeAlias = Literal["fit_epoch", "selected_checkpoint"]
EvaluationPolicy: TypeAlias = Literal["online", "exact", "audit"]
MetricStatus: TypeAlias = Literal["exact", "approximate", "undefined"]
MetricScalar: TypeAlias = Real | torch.Tensor

_TASKS = frozenset({"classification", "regression"})
_SPLITS = frozenset({"train", "val", "test"})
_PASS_KINDS = frozenset({"fit_epoch", "selected_checkpoint"})
_POLICIES = frozenset({"online", "exact", "audit"})
_METRIC_STATUSES = frozenset({"exact", "approximate", "undefined"})


def _require_int(name: str, value: object, *, positive: bool = True) -> int:
    if isinstance(value, bool) or not isinstance(value, int):
        raise TypeError(f"{name} must be an integer, not a boolean")
    if positive and value <= 0:
        raise ValueError(f"{name} must be positive")
    if not positive and value < 0:
        raise ValueError(f"{name} must be non-negative")
    return value


def _require_literal(
    name: str, value: object, allowed: frozenset[str]
) -> None:
    if value not in allowed:
        choices = ", ".join(sorted(allowed))
        raise ValueError(f"{name} must be one of: {choices}")


def _require_optional_identity(name: str, value: object) -> None:
    if value is not None and (not isinstance(value, str) or not value):
        raise ValueError(f"{name} must be a non-empty string or None")


def _deep_freeze(value: Any) -> Any:
    if isinstance(value, Mapping):
        return MappingProxyType(
            {key: _deep_freeze(item) for key, item in value.items()}
        )
    if isinstance(value, (list, tuple)):
        return tuple(_deep_freeze(item) for item in value)
    if isinstance(value, (set, frozenset)):
        return frozenset(_deep_freeze(item) for item in value)
    return value


@dataclass(frozen=True, slots=True)
class EvaluationContext:
    """One validated metric-accumulation window."""

    split: EvaluationSplit
    pass_kind: EvaluationPassKind
    policy: EvaluationPolicy
    task: EvaluationTask
    num_classes: int
    expected_num_examples: int | None = None
    vocabulary_id: str | None = None
    model_id: str | None = None
    checkpoint_id: str | None = None
    qualified: bool = True

    def __post_init__(self) -> None:
        _require_literal("split", self.split, _SPLITS)
        _require_literal("pass_kind", self.pass_kind, _PASS_KINDS)
        _require_literal("policy", self.policy, _POLICIES)
        _require_literal("task", self.task, _TASKS)
        _require_int("num_classes", self.num_classes)
        if self.task == "classification" and self.num_classes < 2:
            raise ValueError("classification num_classes must be at least 2")
        if self.task == "regression" and self.num_classes != 1:
            raise ValueError("regression num_classes must be 1")
        if self.expected_num_examples is not None:
            _require_int("expected_num_examples", self.expected_num_examples)
        _require_optional_identity("vocabulary_id", self.vocabulary_id)
        _require_optional_identity("model_id", self.model_id)
        _require_optional_identity("checkpoint_id", self.checkpoint_id)
        if not isinstance(self.qualified, bool):
            raise TypeError("qualified must be a boolean")


@dataclass(frozen=True, slots=True)
class EvaluationBatch:
    """Tensor references and exact participation count for one update."""

    outputs: torch.Tensor
    targets: torch.Tensor
    num_examples: int
    context: EvaluationContext | None = None
    sequence_id: Hashable | None = None
    prediction_payload: PredictionPayload | None = None

    def __post_init__(self) -> None:
        if not isinstance(self.outputs, torch.Tensor) or not isinstance(
            self.targets, torch.Tensor
        ):
            raise TypeError("outputs and targets must be tensors")
        count = _require_int("num_examples", self.num_examples)
        if self.outputs.ndim == 0 or self.targets.ndim == 0:
            raise ValueError(
                "outputs and targets must have leading dimensions"
            )
        if self.outputs.shape[0] != count or self.targets.shape[0] != count:
            raise ValueError(
                "num_examples must equal both leading tensor dimensions"
            )
        if self.context is not None and not isinstance(
            self.context, EvaluationContext
        ):
            raise TypeError("context must be an EvaluationContext or None")
        if self.sequence_id is not None:
            try:
                hash(self.sequence_id)
            except TypeError as error:
                raise TypeError(
                    "sequence_id must be hashable or None"
                ) from error
        if self.prediction_payload is not None:
            payload = self.prediction_payload
            if not isinstance(payload, PredictionPayload):
                raise TypeError(
                    "prediction_payload must be a PredictionPayload or None"
                )
            if payload.num_rows != count:
                raise ValueError(
                    "prediction_payload rows must align with num_examples"
                )
            if payload.columns["target"] is not self.targets:
                raise ValueError(
                    "prediction_payload target must be the exact EvaluationBatch "
                    "targets alias"
                )
            if payload.columns["raw_output"] is not self.outputs:
                raise ValueError(
                    "prediction_payload raw_output must be the exact "
                    "EvaluationBatch outputs alias"
                )


@dataclass(frozen=True, slots=True)
class EvaluationResult:
    """Ordered scalar results and immutable evaluation metadata."""

    metrics: Mapping[str, MetricScalar]
    num_examples: int
    context: EvaluationContext
    status: Mapping[str, MetricStatus] = field(default_factory=dict)
    support: Mapping[str, Any] = field(default_factory=dict)
    reason: Mapping[str, str | None] = field(default_factory=dict)
    provenance: Mapping[str, Any] = field(default_factory=dict)

    def __post_init__(self) -> None:
        _require_int("num_examples", self.num_examples, positive=False)
        if not isinstance(self.context, EvaluationContext):
            raise TypeError("context must be an EvaluationContext")
        if not isinstance(self.metrics, Mapping):
            raise TypeError("metrics must be a mapping")

        ordered_metrics: dict[str, MetricScalar] = {}
        for name, value in self.metrics.items():
            if not isinstance(name, str) or not name:
                raise ValueError("metric names must be non-empty strings")
            if name == "num_examples":
                raise ValueError("num_examples is reserved outside metrics")
            if isinstance(value, bool) or not isinstance(
                value, (Real, torch.Tensor)
            ):
                raise TypeError(f"metric {name!r} must be a scalar")
            if isinstance(value, torch.Tensor) and value.ndim != 0:
                raise ValueError(f"metric {name!r} must be a scalar tensor")
            ordered_metrics[name] = (
                value.detach().clone()
                if isinstance(value, torch.Tensor)
                else value
            )

        for name, value in self.status.items():
            if name not in ordered_metrics:
                raise ValueError(f"status references unknown metric {name!r}")
            _require_literal("status", value, _METRIC_STATUSES)
        for metadata_name, metadata in (
            ("support", self.support),
            ("reason", self.reason),
        ):
            unknown = set(metadata).difference(ordered_metrics)
            if unknown:
                raise ValueError(
                    f"{metadata_name} references unknown metrics: "
                    f"{sorted(unknown)}"
                )
        for name, value in self.reason.items():
            if value is not None and not isinstance(value, str):
                raise TypeError(
                    f"reason for {name!r} must be a string or None"
                )

        object.__setattr__(self, "metrics", MappingProxyType(ordered_metrics))
        object.__setattr__(self, "status", _deep_freeze(self.status))
        object.__setattr__(self, "support", _deep_freeze(self.support))
        object.__setattr__(self, "reason", _deep_freeze(self.reason))
        object.__setattr__(self, "provenance", _deep_freeze(self.provenance))

    @property
    def split(self) -> EvaluationSplit:
        return self.context.split

    @property
    def pass_kind(self) -> EvaluationPassKind:
        return self.context.pass_kind

    @property
    def policy(self) -> EvaluationPolicy:
        return self.context.policy

    @property
    def task(self) -> EvaluationTask:
        return self.context.task

    @property
    def expected_num_examples(self) -> int | None:
        return self.context.expected_num_examples
