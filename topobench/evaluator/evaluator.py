"""Typed, failure-safe evaluator lifecycle."""

from __future__ import annotations

from collections.abc import Mapping, Sequence
from types import MappingProxyType
from typing import Any, Literal

import torch

from .backends import (
    DEFAULT_MAX_EXACT_RANKING_BYTES,
    ONLINE_RANKING_THRESHOLDS,
    BackendSnapshot,
    MetricPolicyBackend,
)
from .base import AbstractEvaluator, EvaluatorBackend
from .registry import MetricSpec, resolve_metric_specs
from .types import (
    EvaluationBatch,
    EvaluationContext,
    EvaluationResult,
    MetricScalar,
    MetricStatus,
)

_SUPPORTED_TASKS = frozenset({"classification", "regression"})


class TBEvaluator(AbstractEvaluator):
    """Evaluate one reduced task through an explicit lifecycle."""

    def __init__(
        self,
        task: str,
        *,
        num_classes: int,
        metrics: Sequence[str],
        backends: Mapping[str, EvaluatorBackend] | None = None,
        custom_specs: Sequence[MetricSpec] = (),
        ranking_thresholds: int = ONLINE_RANKING_THRESHOLDS,
        max_exact_ranking_bytes: int = DEFAULT_MAX_EXACT_RANKING_BYTES,
        undefined_metric_policy: str = "error",
        device: torch.device | str | None = None,
    ) -> None:
        if task not in _SUPPORTED_TASKS:
            raise ValueError("Supported tasks are exactly: classification, regression")
        if isinstance(num_classes, bool) or not isinstance(num_classes, int):
            raise TypeError("num_classes must be an integer, not a boolean")
        if task == "classification" and num_classes < 2:
            raise ValueError("classification num_classes must be at least 2")
        if task == "regression" and num_classes != 1:
            raise ValueError("regression num_classes must be 1")
        if isinstance(metrics, (str, bytes)):
            raise TypeError("metrics must be an ordered sequence of names")

        metric_names = tuple(metrics)
        if not metric_names:
            raise ValueError("At least one metric must be configured")
        if any(not isinstance(name, str) or not name for name in metric_names):
            raise ValueError("Metric names must be non-empty strings")
        if len(set(metric_names)) != len(metric_names):
            raise ValueError("Duplicate metric names are not allowed")

        metric_backend: MetricPolicyBackend | None = None
        if backends is None:
            resolved = resolve_metric_specs(
                metric_names,
                task=task,
                num_classes=num_classes,
                policy="exact",
                custom_specs=custom_specs,
            )
            audit_names: list[str] = []
            for spec in resolved:
                audit_names.append(spec.name)
                if spec.name in {"auroc", "auprc", "somers_d"}:
                    audit_names.extend(
                        (
                            f"{spec.name}_online",
                            f"{spec.name}_online_abs_error",
                        )
                    )
            if len(set(audit_names)) != len(audit_names):
                collisions = sorted(
                    name
                    for name in set(audit_names)
                    if audit_names.count(name) > 1
                )
                raise ValueError(f"Generated audit-key collision: {collisions}")
            metric_backend = MetricPolicyBackend(
                task=task,
                num_classes=num_classes,
                metrics=metric_names,
                custom_specs=custom_specs,
                ranking_thresholds=ranking_thresholds,
                max_exact_ranking_bytes=max_exact_ranking_bytes,
                undefined_metric_policy=undefined_metric_policy,
                device=device,
            )
            mutable_backends: dict[str, EvaluatorBackend] = {
                "metrics": metric_backend
            }
        else:
            if custom_specs:
                raise ValueError(
                    "custom_specs cannot be combined with injected lifecycle backends"
                )
            if not isinstance(backends, Mapping) or not backends:
                raise ValueError("backends must be a non-empty mapping")
            mutable_backends = dict(backends)
            for name, backend in mutable_backends.items():
                if not isinstance(name, str) or not name:
                    raise ValueError("Backend names must be non-empty strings")
                if not isinstance(backend, EvaluatorBackend):
                    raise TypeError(
                        f"Backend {name!r} does not implement EvaluatorBackend"
                    )

        self.task = task
        self.num_classes = num_classes
        self.metric_names = metric_names
        self._metric_backend = metric_backend
        self._backends = mutable_backends
        self._state: Literal["idle", "active", "failed"] = "idle"
        self._context: EvaluationContext | None = None
        self._num_examples = 0
        self._allow_idle_abort = False

    @property
    def state(self) -> str:
        return self._state

    @property
    def num_examples(self) -> int:
        return self._num_examples

    @property
    def context(self) -> EvaluationContext | None:
        return self._context

    @property
    def backends(self) -> Mapping[str, EvaluatorBackend]:
        return MappingProxyType(self._backends)

    @property
    def metric_backend(self) -> MetricPolicyBackend:
        if self._metric_backend is None:
            raise AttributeError(
                "metric_backend is unavailable with injected lifecycle backends"
            )
        return self._metric_backend

    def __repr__(self) -> str:
        return (
            f"{type(self).__name__}(task={self.task!r}, "
            f"num_classes={self.num_classes}, metrics={self.metric_names!r})"
        )

    def begin(self, context: EvaluationContext) -> None:
        if self._state != "idle":
            raise RuntimeError("begin requires an idle evaluator")
        if not isinstance(context, EvaluationContext):
            raise TypeError("context must be an EvaluationContext")
        if context.task != self.task or context.num_classes != self.num_classes:
            raise ValueError(
                "EvaluationContext task and vocabulary must match construction"
            )

        self._context = context
        self._num_examples = 0
        self._allow_idle_abort = False
        self._state = "active"
        try:
            for backend in self._backends.values():
                backend.begin(context)
        except BaseException:
            self._state = "failed"
            raise

    def update(self, batch: EvaluationBatch) -> None:
        context = self._require_active("update")
        if not isinstance(batch, EvaluationBatch):
            raise TypeError("batch must be an EvaluationBatch")
        if batch.context is not None and batch.context != context:
            raise ValueError(
                "EvaluationBatch does not match the active EvaluationContext"
            )
        self._validate_batch(batch, context)

        try:
            for backend in self._backends.values():
                backend.update(batch)
        except BaseException:
            self._state = "failed"
            raise
        self._num_examples += batch.num_examples

    def snapshot(self) -> EvaluationResult:
        context = self._require_active("snapshot")
        try:
            return self._build_result(context)
        except BaseException:
            self._state = "failed"
            raise

    def finalize(self) -> EvaluationResult:
        context = self._require_active("finalize")
        if self._num_examples == 0 and self._metric_backend is None:
            raise RuntimeError("finalize requires at least one supervised example")
        if (
            context.expected_num_examples is not None
            and self._num_examples != context.expected_num_examples
        ):
            observed = self._num_examples
            expected = context.expected_num_examples
            self._clear_after_failure()
            raise ValueError(
                f"Evaluation expected {expected} examples but observed {observed}"
            )

        try:
            result = self._build_result(context)
        except BaseException:
            self._clear_after_failure()
            raise

        try:
            self._reset_backends()
        except BaseException:
            self._state = "failed"
            raise
        self._clear_fields(allow_idle_abort=False)
        return result

    def abort(self) -> None:
        if self._state == "idle":
            if self._allow_idle_abort:
                return
            raise RuntimeError("abort is invalid while idle")

        failed = self._state == "failed"
        try:
            self._reset_backends()
        except BaseException:
            self._state = "failed"
            raise
        self._clear_fields(allow_idle_abort=failed)

    def _require_active(self, operation: str) -> EvaluationContext:
        if self._state != "active" or self._context is None:
            raise RuntimeError(f"{operation} requires an active evaluator")
        return self._context

    def _validate_batch(
        self, batch: EvaluationBatch, context: EvaluationContext
    ) -> None:
        outputs = batch.outputs
        targets = batch.targets
        if context.task == "classification":
            if outputs.ndim != 2:
                raise ValueError("classification outputs must be rank-2")
            if targets.ndim != 1:
                raise ValueError("classification targets must be rank-1")
            if outputs.shape[1] != context.num_classes:
                raise ValueError(
                    "classification output class dimension must equal num_classes"
                )
            if not outputs.is_floating_point():
                raise TypeError("classification outputs must be floating tensors")
            if targets.dtype != torch.long:
                raise TypeError("classification targets must have dtype torch.long")
            if not torch.isfinite(outputs).all():
                raise ValueError("classification outputs must be finite")
            if bool(torch.any(targets < 0)) or bool(
                torch.any(targets >= context.num_classes)
            ):
                raise ValueError(
                    "classification targets contain out-of-vocabulary class IDs"
                )
            return

        if (
            outputs.ndim != 2
            or targets.ndim != 2
            or outputs.shape[1] != 1
            or targets.shape[1] != 1
            or outputs.shape != targets.shape
        ):
            raise ValueError(
                "regression outputs and targets must have equal shape [N, 1]"
            )
        if not outputs.is_floating_point() or not targets.is_floating_point():
            raise TypeError("regression outputs and targets must be floating tensors")
        if not torch.isfinite(outputs).all() or not torch.isfinite(targets).all():
            raise ValueError("regression outputs and targets must be finite")

    def _build_result(self, context: EvaluationContext) -> EvaluationResult:
        computed: dict[str, MetricScalar] = {}
        status: dict[str, MetricStatus] = {}
        support: dict[str, Any] = {}
        reason: dict[str, str | None] = {}
        provenance: dict[str, Any] = {}
        for backend_name, backend in self._backends.items():
            output = backend.compute()
            if isinstance(output, Mapping):
                for metric_name, value in output.items():
                    if metric_name in computed:
                        raise ValueError(f"Duplicate computed metric {metric_name!r}")
                    computed[metric_name] = value
                if isinstance(output, BackendSnapshot):
                    status.update(output.status)
                    support.update(output.support)
                    reason.update(output.reason)
                    provenance.update(output.provenance)
            else:
                if backend_name in computed:
                    raise ValueError(f"Duplicate computed metric {backend_name!r}")
                computed[backend_name] = output

        output_order: list[str] = []
        for name in self.metric_names:
            output_order.append(name)
            if (
                self._metric_backend is not None
                and context.policy == "audit"
                and name in {"auroc", "auprc", "somers_d"}
            ):
                output_order.extend(
                    (f"{name}_online", f"{name}_online_abs_error")
                )
        missing = set(output_order).difference(computed)
        extra = set(computed).difference(output_order)
        if missing or extra:
            raise ValueError(
                "Backend outputs must match configured metrics; "
                f"missing={sorted(missing)}, extra={sorted(extra)}"
            )
        ordered = {name: computed[name] for name in output_order}
        default_status: MetricStatus = (
            "approximate" if context.policy == "online" else "exact"
        )
        for name in output_order:
            status.setdefault(name, default_status)
        return EvaluationResult(
            metrics=ordered,
            num_examples=self._num_examples,
            context=context,
            status=status,
            support=support,
            reason=reason,
            provenance=provenance,
        )

    def _reset_backends(self) -> None:
        first_error: BaseException | None = None
        for backend in self._backends.values():
            try:
                backend.reset()
            except BaseException as error:
                if first_error is None:
                    first_error = error
        if first_error is not None:
            raise first_error

    def _clear_after_failure(self) -> None:
        self._state = "failed"
        self._reset_backends()
        self._clear_fields(allow_idle_abort=True)

    def _clear_fields(self, *, allow_idle_abort: bool) -> None:
        self._context = None
        self._num_examples = 0
        self._state = "idle"
        self._allow_idle_abort = allow_idle_abort
