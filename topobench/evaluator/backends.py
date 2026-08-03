"""TopoBench-owned metric backends and scalable ranking state."""

from __future__ import annotations

from collections import OrderedDict
from collections.abc import Callable, Iterator, Mapping, Sequence
from dataclasses import dataclass, field
from math import ceil
from types import MappingProxyType
from typing import Any, Protocol, runtime_checkable

import torch
from torch import Tensor
from torchmetrics import Metric
from torchmetrics.classification import (
    BinaryAUROC,
    BinaryAveragePrecision,
    MulticlassAccuracy,
    MulticlassAUROC,
    MulticlassF1Score,
    MulticlassPrecision,
    MulticlassRecall,
)
from torchmetrics.functional.classification import (
    binary_auroc,
    binary_average_precision,
    multiclass_auroc,
)
from torchmetrics.regression import (
    MeanAbsoluteError,
    MeanSquaredError,
    R2Score,
)

from .types import EvaluationBatch, EvaluationContext

ONLINE_RANKING_THRESHOLDS = 512
DEFAULT_MAX_EXACT_RANKING_BYTES = 512 * 1024 * 1024
DEFAULT_EXACT_MEMORY_SAFETY_FACTOR = 1.25


@dataclass(frozen=True, slots=True)
class BackendFactoryContext:
    """Immutable construction inputs supplied to a metric backend factory."""

    task: str
    num_classes: int
    policy: str
    device: torch.device
    ranking_thresholds: int
    max_exact_ranking_bytes: int
    undefined_metric_policy: str
    threshold_grid: Tensor | None = field(
        default=None, compare=False, repr=False
    )


@runtime_checkable
class MetricBackend(Protocol):
    """Small structural protocol for constructor-injected metric state."""

    def update(self, predictions: Tensor, targets: Tensor) -> None:
        """Accumulate one prediction view and its targets."""

    def compute(self) -> Tensor:
        """Compute one scalar without clearing state."""

    def reset(self) -> None:
        """Release or zero mutable state."""

    def to(self, device: torch.device | str) -> MetricBackend:
        """Move fixed-size state to the evaluation device."""

    @property
    def retained_bytes(self) -> int:
        """Return recursively owned tensor-storage bytes."""


@dataclass(frozen=True, slots=True)
class ExactRankingMemoryEstimate:
    """Conservative retained-plus-sequential-workspace estimate."""

    layout: str
    num_examples: int
    num_classes: int
    score_dtype: torch.dtype
    target_dtype: torch.dtype
    retained_bytes: int
    workspace_bytes: int
    estimated_peak_bytes: int
    safety_factor: float
    binary_state_shared: bool


def estimate_exact_ranking_memory(
    *,
    num_examples: int,
    num_classes: int,
    score_dtype: torch.dtype = torch.float32,
    target_dtype: torch.dtype = torch.int64,
    safety_factor: float = DEFAULT_EXACT_MEMORY_SAFETY_FACTOR,
) -> ExactRankingMemoryEstimate:
    """Estimate one shared exact buffer plus the largest sequential workspace."""
    if (
        isinstance(num_examples, bool)
        or not isinstance(num_examples, int)
        or num_examples < 0
    ):
        raise ValueError("num_examples must be a non-negative integer")
    if (
        isinstance(num_classes, bool)
        or not isinstance(num_classes, int)
        or num_classes < 2
    ):
        raise ValueError("num_classes must be an integer of at least 2")
    if (
        not isinstance(safety_factor, (int, float))
        or isinstance(safety_factor, bool)
        or safety_factor < 1
    ):
        raise ValueError("safety_factor must be at least 1")

    score_width = 1 if num_classes == 2 else num_classes
    score_size = torch.empty((), dtype=score_dtype).element_size()
    target_size = torch.empty((), dtype=target_dtype).element_size()
    score_bytes = num_examples * score_width * score_size
    target_bytes = num_examples * target_size
    retained_bytes = score_bytes + target_bytes
    concatenation_bytes = retained_bytes
    sorting_bytes = score_bytes + num_examples * score_width * 8
    workspace_bytes = concatenation_bytes + sorting_bytes
    estimated_peak_bytes = ceil(
        (retained_bytes + workspace_bytes) * float(safety_factor)
    )
    return ExactRankingMemoryEstimate(
        layout="binary_positive_scores"
        if num_classes == 2
        else "multiclass_probabilities",
        num_examples=num_examples,
        num_classes=num_classes,
        score_dtype=score_dtype,
        target_dtype=target_dtype,
        retained_bytes=retained_bytes,
        workspace_bytes=workspace_bytes,
        estimated_peak_bytes=estimated_peak_bytes,
        safety_factor=float(safety_factor),
        binary_state_shared=num_classes == 2,
    )


class ExactRankingMemoryError(RuntimeError):
    """Raised before exact ranking state would exceed its declared ceiling."""

    def __init__(
        self,
        *,
        split: str,
        observed_examples: int,
        projected_examples: int,
        projected_bytes: int,
        configured_limit: int,
    ) -> None:
        self.split = split
        self.observed_examples = observed_examples
        self.projected_examples = projected_examples
        self.projected_bytes = projected_bytes
        self.configured_limit = configured_limit
        super().__init__(
            "Exact ranking memory ceiling exceeded: "
            f"split={split}, observed_examples={observed_examples}, "
            f"projected_examples={projected_examples}, projected_bytes={projected_bytes}, "
            f"configured_limit={configured_limit}. Select policy='online' explicitly "
            "or increase the qualified exact-ranking ceiling."
        )


class UndefinedMetricError(RuntimeError):
    """Structured failure for a mathematically undefined scalar metric."""

    def __init__(
        self,
        *,
        metric: str,
        split: str,
        reason: str,
        support: Mapping[Any, Any],
        num_examples: int,
    ) -> None:
        self.metric = metric
        self.split = split
        self.reason = reason
        self.support = dict(support)
        self.num_examples = num_examples
        super().__init__(
            f"Metric {metric!r} is undefined on split={split}: reason={reason}, "
            f"support={dict(support)}, num_examples={num_examples}. Use "
            "undefined_metric_policy='nan' only for an explicitly exploratory run."
        )


def reachable_objects(root: object) -> tuple[object, ...]:
    """Return objects recursively owned by ``root`` without revisiting cycles."""
    found: list[object] = []
    visited: set[int] = set()

    def visit(value: object) -> None:
        identity = id(value)
        if identity in visited:
            return
        visited.add(identity)
        found.append(value)
        if isinstance(value, Mapping):
            for key, item in value.items():
                visit(key)
                visit(item)
            return
        if isinstance(value, (list, tuple, set, frozenset)):
            for item in value:
                visit(item)
            return
        if isinstance(
            value,
            (
                str,
                bytes,
                int,
                float,
                bool,
                type(None),
                torch.dtype,
                torch.device,
            ),
        ):
            return
        attributes = getattr(value, "__dict__", None)
        if isinstance(attributes, dict):
            for item in attributes.values():
                visit(item)
        slots = getattr(type(value), "__slots__", ())
        if isinstance(slots, str):
            slots = (slots,)
        for slot in slots:
            if slot.startswith("__") or not hasattr(value, slot):
                continue
            visit(getattr(value, slot))

    visit(root)
    return tuple(found)


def owned_tensor_bytes(root: object) -> int:
    """Count unique recursively reachable tensor storages."""
    storages: dict[tuple[str, int, int], int] = {}
    for value in reachable_objects(root):
        if not isinstance(value, Tensor):
            continue
        storage = value.untyped_storage()
        key = (str(value.device), storage.data_ptr(), storage.nbytes())
        storages[key] = storage.nbytes()
    return sum(storages.values())


def _state_tensors(root: object) -> tuple[Tensor, ...]:
    return tuple(
        value for value in reachable_objects(root) if isinstance(value, Tensor)
    )


def _clone_checkpoint_state(
    value: Any,
    *,
    tensor_device: torch.device | str | None = None,
) -> Any:
    """Clone checkpoint-safe values without mutable tensor aliases."""
    if isinstance(value, Tensor):
        detached = value.detach()
        if tensor_device is not None:
            detached = detached.to(tensor_device)
        return detached.clone()
    if isinstance(value, Mapping):
        return {
            key: _clone_checkpoint_state(item, tensor_device=tensor_device)
            for key, item in value.items()
        }
    if isinstance(value, tuple):
        return tuple(
            _clone_checkpoint_state(item, tensor_device=tensor_device)
            for item in value
        )
    if isinstance(value, list):
        return [
            _clone_checkpoint_state(item, tensor_device=tensor_device)
            for item in value
        ]
    if isinstance(value, (str, int, float, bool, type(None))):
        return value
    raise TypeError(
        "evaluator checkpoint state may contain only mappings, sequences, "
        "primitive scalars, and tensors"
    )


def _metric_type_name(metric: Metric) -> str:
    metric_type = type(metric)
    return f"{metric_type.__module__}.{metric_type.__qualname__}"


def _torch_metric_state_dict(metric: Metric) -> dict[str, Any]:
    """Serialize all TorchMetrics states, including non-persistent defaults."""
    return {
        "format_version": "torch-metric-state-v1",
        "metric_type": _metric_type_name(metric),
        "update_count": metric.update_count,
        "metric_state": _clone_checkpoint_state(
            metric.metric_state,
            tensor_device="cpu",
        ),
    }


def _load_torch_metric_state_dict(
    metric: Metric,
    state_dict: Mapping[str, Any],
) -> None:
    """Restore all TorchMetrics states onto the metric's configured device."""
    expected = {
        "format_version",
        "metric_type",
        "update_count",
        "metric_state",
    }
    if set(state_dict) != expected:
        raise ValueError("TorchMetrics state_dict keys do not match schema")
    if state_dict["format_version"] != "torch-metric-state-v1":
        raise ValueError("unsupported TorchMetrics state_dict format_version")
    if state_dict["metric_type"] != _metric_type_name(metric):
        raise ValueError("TorchMetrics state type does not match construction")
    update_count = state_dict["update_count"]
    if (
        isinstance(update_count, bool)
        or not isinstance(update_count, int)
        or update_count < 0
    ):
        raise ValueError(
            "TorchMetrics update_count must be a non-negative integer"
        )
    serialized = state_dict["metric_state"]
    current = metric.metric_state
    if not isinstance(serialized, Mapping):
        raise TypeError("TorchMetrics metric_state must be a mapping")
    if set(serialized) != set(current):
        raise ValueError("TorchMetrics metric_state keys do not match metric")

    nonnegative_states = {
        "tp",
        "fp",
        "tn",
        "fn",
        "confmat",
        "total",
        "sum_abs_error",
        "sum_squared_error",
        "residual",
    }
    for name, current_value in current.items():
        serialized_value = serialized[name]
        if isinstance(current_value, Tensor):
            if not isinstance(serialized_value, Tensor):
                raise TypeError(
                    f"TorchMetrics state {name!r} must be a tensor"
                )
            if serialized_value.dtype != current_value.dtype:
                raise ValueError(
                    f"TorchMetrics state {name!r} tensor dtype does not match"
                )
            if serialized_value.shape != current_value.shape and not (
                serialized_value.numel() == 1 and current_value.numel() == 1
            ):
                raise ValueError(
                    f"TorchMetrics state {name!r} tensor shape does not match"
                )
            if not bool(torch.isfinite(serialized_value).all()):
                raise ValueError(f"TorchMetrics state {name!r} must be finite")
            if name in nonnegative_states and bool(
                (serialized_value < 0).any()
            ):
                raise ValueError(
                    f"TorchMetrics state {name!r} must be non-negative"
                )
        elif not isinstance(serialized_value, list):
            raise TypeError(f"TorchMetrics state {name!r} must be a list")
        elif any(
            not isinstance(item, Tensor)
            or not bool(torch.isfinite(item).all())
            for item in serialized_value
        ):
            raise ValueError(
                f"TorchMetrics list state {name!r} must contain finite tensors"
            )

    restored = _clone_checkpoint_state(
        serialized,
        tensor_device=metric.device,
    )
    metric.reset()
    for name, restored_value in restored.items():
        setattr(metric, name, restored_value)
    metric._update_count = update_count
    metric._computed = None
    metric._forward_cache = None
    metric._cache = None
    metric._is_synced = False


class TorchMetricBackend:
    """Adapter hiding a public stateful TorchMetrics implementation."""

    def __init__(self, metric: Metric) -> None:
        self._metric = metric

    @property
    def metric(self) -> Metric:
        return self._metric

    def update(self, predictions: Tensor, targets: Tensor) -> None:
        self._metric.update(predictions, targets)

    def compute(self) -> Tensor:
        return self._metric.compute()

    def reset(self) -> None:
        self._metric.reset()

    def to(self, device: torch.device | str) -> TorchMetricBackend:
        self._metric.to(device)
        return self

    @property
    def retained_bytes(self) -> int:
        return owned_tensor_bytes(self._metric)

    @property
    def state_tensors(self) -> tuple[Tensor, ...]:
        return _state_tensors(self._metric)

    def state_dict(self) -> dict[str, Any]:
        """Return every detached TorchMetrics state on CPU."""
        return _torch_metric_state_dict(self._metric)

    def load_state_dict(
        self,
        state_dict: Mapping[str, Any],
        *,
        strict: bool,
    ) -> None:
        """Restore every TorchMetrics state onto the configured device."""
        if strict is not True:
            raise TypeError("strict must be True")
        if not isinstance(state_dict, Mapping):
            raise TypeError("metric state_dict must be a mapping")
        _load_torch_metric_state_dict(self._metric, state_dict)


def _backend_type_name(backend: MetricBackend) -> str:
    backend_type = type(backend)
    return f"{backend_type.__module__}.{backend_type.__qualname__}"


def _backend_state_dict(backend: MetricBackend) -> dict[str, Any]:
    state_method = getattr(backend, "state_dict", None)
    if not callable(state_method):
        raise TypeError(
            f"Metric backend {_backend_type_name(backend)!r} does not support "
            "strict checkpoint state"
        )
    state = state_method()
    if not isinstance(state, Mapping):
        raise TypeError("metric backend state_dict must return a mapping")
    return {
        "backend_type": _backend_type_name(backend),
        "state": _clone_checkpoint_state(state, tensor_device="cpu"),
    }


def _load_backend_state_dict(
    backend: MetricBackend,
    state_dict: Mapping[str, Any],
) -> None:
    if set(state_dict) != {"backend_type", "state"}:
        raise ValueError("metric backend state keys do not match schema")
    if state_dict["backend_type"] != _backend_type_name(backend):
        raise ValueError(
            "metric backend state type does not match construction"
        )
    state = state_dict["state"]
    if not isinstance(state, Mapping):
        raise TypeError("metric backend state must be a mapping")
    load_method = getattr(backend, "load_state_dict", None)
    if not callable(load_method):
        raise TypeError(
            f"Metric backend {_backend_type_name(backend)!r} does not support "
            "strict checkpoint state"
        )
    load_method(_clone_checkpoint_state(state), strict=True)


class _FunctionalExactMetric:
    def __init__(self, name: str) -> None:
        self.name = name

    def update(self, predictions: Tensor, targets: Tensor) -> None:
        del predictions, targets

    def compute(self) -> Tensor:
        raise RuntimeError(
            "functional exact metrics compute through ExactRankingBackend"
        )

    def reset(self) -> None:
        return None

    def to(self, device: torch.device | str) -> _FunctionalExactMetric:
        del device
        return self

    @property
    def retained_bytes(self) -> int:
        return 0


class PredictionViews:
    """Lazy per-batch classification views, each derived at most once."""

    def __init__(self, outputs: Tensor) -> None:
        self.outputs = outputs
        self._probabilities: Tensor | None = None
        self._positive_probabilities: Tensor | None = None
        self._hard_classes: Tensor | None = None

    def _get_probabilities(self) -> Tensor:
        if self._probabilities is None:
            self._probabilities = torch.softmax(self.outputs, dim=1)
        return self._probabilities

    @property
    def probabilities(self) -> Tensor:
        return self._get_probabilities()

    @property
    def positive_probabilities(self) -> Tensor:
        if self._positive_probabilities is None:
            self._positive_probabilities = self._get_probabilities()[:, 1]
        return self._positive_probabilities

    @property
    def hard_classes(self) -> Tensor:
        if self._hard_classes is None:
            self._hard_classes = torch.argmax(self.outputs, dim=1)
        return self._hard_classes

    @property
    def raw(self) -> Tensor:
        return self.outputs


class ExactRankingBackend:
    """One TopoBench-owned detached CPU buffer for exact ranking metrics."""

    def __init__(
        self,
        *,
        num_classes: int,
        metrics: Sequence[str],
        split: str,
        max_bytes: int,
        expected_num_examples: int | None,
    ) -> None:
        self.num_classes = num_classes
        self.metrics = tuple(dict.fromkeys(metrics))
        self.split = split
        self.max_bytes = max_bytes
        self.score_chunks: list[Tensor] = []
        self.target_chunks: list[Tensor] = []
        self.num_examples = 0
        self._score_dtype = torch.float32
        self._target_dtype = torch.int64
        if expected_num_examples is not None:
            self._guard(
                expected_num_examples, self._score_dtype, self._target_dtype
            )

    @property
    def binary_state_shared(self) -> bool:
        return self.num_classes == 2

    def _guard(
        self,
        projected_examples: int,
        score_dtype: torch.dtype,
        target_dtype: torch.dtype,
    ) -> None:
        estimate = estimate_exact_ranking_memory(
            num_examples=projected_examples,
            num_classes=self.num_classes,
            score_dtype=score_dtype,
            target_dtype=target_dtype,
        )
        if estimate.estimated_peak_bytes > self.max_bytes:
            raise ExactRankingMemoryError(
                split=self.split,
                observed_examples=self.num_examples,
                projected_examples=projected_examples,
                projected_bytes=estimate.estimated_peak_bytes,
                configured_limit=self.max_bytes,
            )

    def guard_append(
        self,
        num_examples: int,
        score_dtype: torch.dtype,
        target_dtype: torch.dtype,
    ) -> None:
        self._guard(
            self.num_examples + num_examples, score_dtype, target_dtype
        )

    def update(self, predictions: Tensor, targets: Tensor) -> None:
        projected = self.num_examples + targets.shape[0]
        self._guard(projected, predictions.dtype, targets.dtype)
        detached_predictions = predictions.detach()
        detached_targets = targets.detach()
        score_chunk = (
            detached_predictions.clone()
            if detached_predictions.device.type == "cpu"
            else detached_predictions.to(device="cpu")
        )
        target_chunk = (
            detached_targets.clone()
            if detached_targets.device.type == "cpu"
            else detached_targets.to(device="cpu")
        )
        self.score_chunks.append(score_chunk.contiguous())
        self.target_chunks.append(target_chunk.contiguous())
        self.num_examples = projected
        self._score_dtype = score_chunk.dtype
        self._target_dtype = target_chunk.dtype

    def compute(self) -> OrderedDict[str, Tensor]:
        scores = torch.cat(self.score_chunks, dim=0)
        targets = torch.cat(self.target_chunks, dim=0)
        values: OrderedDict[str, Tensor] = OrderedDict()
        for name in self.metrics:
            if name == "auroc":
                values[name] = (
                    binary_auroc(scores, targets, thresholds=None)
                    if self.num_classes == 2
                    else multiclass_auroc(
                        scores,
                        targets,
                        num_classes=self.num_classes,
                        average="macro",
                        thresholds=None,
                    )
                )
            elif name == "auprc":
                values[name] = binary_average_precision(
                    scores, targets, thresholds=None
                )
            else:
                raise RuntimeError(
                    f"Unsupported exact ranking metric {name!r}"
                )
        return values

    def reset(self) -> None:
        self.score_chunks.clear()
        self.target_chunks.clear()
        self.num_examples = 0

    @property
    def retained_bytes(self) -> int:
        return owned_tensor_bytes((self.score_chunks, self.target_chunks))

    def estimate(self) -> ExactRankingMemoryEstimate:
        return estimate_exact_ranking_memory(
            num_examples=self.num_examples,
            num_classes=self.num_classes,
            score_dtype=self._score_dtype,
            target_dtype=self._target_dtype,
        )

    def reachable_objects(self) -> tuple[object, ...]:
        return reachable_objects(self)


class OnlineRankingBackend:
    """Bounded stateful ranking metrics on one shared threshold grid."""

    def __init__(
        self,
        *,
        num_classes: int,
        metrics: Sequence[str],
        threshold_count: int,
        device: torch.device,
    ) -> None:
        if (
            isinstance(threshold_count, bool)
            or not isinstance(threshold_count, int)
            or threshold_count < 2
        ):
            raise ValueError(
                "ranking_thresholds must be an integer of at least 2"
            )
        self.num_classes = num_classes
        self.threshold_grid = torch.linspace(
            0.0, 1.0, threshold_count, device=device
        )
        modules: OrderedDict[str, Metric] = OrderedDict()
        for name in dict.fromkeys(metrics):
            if name == "auroc":
                module: Metric = (
                    BinaryAUROC(thresholds=self.threshold_grid)
                    if num_classes == 2
                    else MulticlassAUROC(
                        num_classes=num_classes,
                        average="macro",
                        thresholds=self.threshold_grid,
                    )
                )
            elif name == "auprc" and num_classes == 2:
                module = BinaryAveragePrecision(thresholds=self.threshold_grid)
            else:
                raise ValueError(f"Unsupported online ranking metric {name!r}")
            modules[name] = module.to(device)
        self.metrics = modules

    def update(self, predictions: Tensor, targets: Tensor) -> None:
        for metric in self.metrics.values():
            metric.update(predictions, targets)

    def compute(self) -> OrderedDict[str, Tensor]:
        return OrderedDict(
            (name, metric.compute()) for name, metric in self.metrics.items()
        )

    def reset(self) -> None:
        for metric in self.metrics.values():
            metric.reset()

    def to(self, device: torch.device | str) -> OnlineRankingBackend:
        resolved = torch.device(device)
        self.threshold_grid = self.threshold_grid.to(resolved)
        for metric in self.metrics.values():
            metric.to(resolved)
        return self

    @property
    def state_tensors(self) -> tuple[Tensor, ...]:
        return _state_tensors(self.metrics)

    @property
    def retained_bytes(self) -> int:
        return owned_tensor_bytes(self.metrics)

    def state_dict(self) -> dict[str, Any]:
        """Return every bounded ranking state detached on CPU."""
        return {
            name: _torch_metric_state_dict(metric)
            for name, metric in self.metrics.items()
        }

    def load_state_dict(
        self,
        state_dict: Mapping[str, Any],
        *,
        strict: bool,
    ) -> None:
        """Restore bounded ranking metrics with exact key validation."""
        if strict is not True:
            raise TypeError("strict must be True")
        if not isinstance(state_dict, Mapping):
            raise TypeError("online ranking state_dict must be a mapping")
        if set(state_dict) != set(self.metrics):
            raise ValueError("online ranking state keys do not match metrics")
        for name, metric in self.metrics.items():
            item = state_dict[name]
            if not isinstance(item, Mapping):
                raise TypeError(
                    "online ranking metric state must be a mapping"
                )
            _load_torch_metric_state_dict(metric, item)


@dataclass(frozen=True)
class BackendSnapshot(Mapping[str, Tensor]):
    metrics: Mapping[str, Tensor]
    status: Mapping[str, str]
    support: Mapping[str, Any]
    reason: Mapping[str, str | None]
    provenance: Mapping[str, Any]

    def __post_init__(self) -> None:
        object.__setattr__(
            self, "metrics", MappingProxyType(OrderedDict(self.metrics))
        )
        object.__setattr__(self, "status", MappingProxyType(dict(self.status)))
        object.__setattr__(
            self, "support", MappingProxyType(dict(self.support))
        )
        object.__setattr__(self, "reason", MappingProxyType(dict(self.reason)))
        object.__setattr__(
            self, "provenance", MappingProxyType(dict(self.provenance))
        )

    def __getitem__(self, key: str) -> Tensor:
        return self.metrics[key]

    def __iter__(self) -> Iterator[str]:
        return iter(self.metrics)

    def __len__(self) -> int:
        return len(self.metrics)


class _SupportTracker:
    def __init__(
        self, task: str, num_classes: int, device: torch.device
    ) -> None:
        self.task = task
        self.num_classes = num_classes
        self.num_examples = 0
        self.class_counts = (
            torch.zeros(num_classes, dtype=torch.long, device=device)
            if task == "classification"
            else None
        )
        self.target_sum = torch.tensor(0.0, dtype=torch.float64, device=device)
        self.target_square_sum = torch.tensor(
            0.0, dtype=torch.float64, device=device
        )
        self._cached_support: dict[Any, Any] | None = None

    def to(self, device: torch.device) -> None:
        if self.class_counts is not None:
            self.class_counts = self.class_counts.to(device)
        self.target_sum = self.target_sum.to(device)
        self.target_square_sum = self.target_square_sum.to(device)
        self._cached_support = None

    def update(self, targets: Tensor) -> None:
        detached_targets = targets.detach()
        self.num_examples += targets.shape[0]
        self._cached_support = None
        if self.task == "classification":
            self.class_counts += torch.bincount(
                detached_targets, minlength=self.num_classes
            )
        else:
            flattened = detached_targets.to(torch.float64).reshape(-1)
            self.target_sum += flattened.sum()
            self.target_square_sum += (flattened * flattened).sum()

    def class_support(self) -> dict[Any, Any]:
        if self.class_counts is None:
            return {}
        if self._cached_support is None:
            counts = self.class_counts.detach().cpu().tolist()
            self._cached_support = {
                index: value for index, value in enumerate(counts)
            }
        return self._cached_support

    def regression_support(self) -> dict[str, Any]:
        if self._cached_support is None:
            self._cached_support = {
                "num_examples": self.num_examples,
                "target_sum": float(self.target_sum),
                "target_square_sum": float(self.target_square_sum),
            }
        return self._cached_support


class MetricPolicyBackend:
    """One lifecycle backend coordinating specs, views, policies, and metadata."""

    def __init__(
        self,
        *,
        task: str,
        num_classes: int,
        metrics: Sequence[str],
        custom_specs: Sequence[Any] = (),
        ranking_thresholds: int = ONLINE_RANKING_THRESHOLDS,
        max_exact_ranking_bytes: int = DEFAULT_MAX_EXACT_RANKING_BYTES,
        undefined_metric_policy: str = "error",
        device: torch.device | str | None = None,
        prediction_views_factory: Callable[
            [Tensor], PredictionViews
        ] = PredictionViews,
    ) -> None:
        if undefined_metric_policy not in {"error", "nan"}:
            raise ValueError(
                "undefined_metric_policy must be 'error' or 'nan'"
            )
        if (
            isinstance(max_exact_ranking_bytes, bool)
            or not isinstance(max_exact_ranking_bytes, int)
            or max_exact_ranking_bytes <= 0
        ):
            raise ValueError(
                "max_exact_ranking_bytes must be a positive integer"
            )
        if (
            isinstance(ranking_thresholds, bool)
            or not isinstance(ranking_thresholds, int)
            or ranking_thresholds < 2
        ):
            raise ValueError(
                "ranking_thresholds must be an integer of at least 2"
            )
        self.task = task
        self.num_classes = num_classes
        self.metric_names = tuple(metrics)
        self.custom_specs = tuple(custom_specs)
        self.ranking_thresholds = ranking_thresholds
        self.max_exact_ranking_bytes = max_exact_ranking_bytes
        self.undefined_metric_policy = undefined_metric_policy
        self._device_explicit = device is not None
        self._auto_device_pending = not self._device_explicit
        resolved_device = torch.device("cpu" if device is None else device)
        if resolved_device.type == "cuda" and resolved_device.index is None:
            resolved_device = torch.device("cuda", torch.cuda.current_device())
        self.device = resolved_device
        self.prediction_views_factory = prediction_views_factory
        self.policy: str | None = None
        self.context: EvaluationContext | None = None
        self._specs: tuple[Any, ...] = ()
        self._fixed: OrderedDict[str, MetricBackend] = OrderedDict()
        self._fixed_views: dict[str, str] = {}
        self.exact_ranking_backend: ExactRankingBackend | None = None
        self.online_ranking_backend: OnlineRankingBackend | None = None
        self._support = _SupportTracker(task, num_classes, self.device)

    def begin(self, context: EvaluationContext) -> None:
        if self.context is not None:
            raise RuntimeError(
                "An active metric policy context cannot change policy"
            )
        if (
            context.task != self.task
            or context.num_classes != self.num_classes
        ):
            raise ValueError(
                "EvaluationContext task/class vocabulary does not match "
                "backend construction"
            )
        from .registry import resolve_metric_specs

        specs = resolve_metric_specs(
            self.metric_names,
            task=self.task,
            num_classes=self.num_classes,
            policy=context.policy,
            custom_specs=self.custom_specs,
        )
        staged_support = _SupportTracker(
            self.task, self.num_classes, self.device
        )
        factory_context = BackendFactoryContext(
            task=self.task,
            num_classes=self.num_classes,
            policy=context.policy,
            device=self.device,
            ranking_thresholds=self.ranking_thresholds,
            max_exact_ranking_bytes=self.max_exact_ranking_bytes,
            undefined_metric_policy=self.undefined_metric_policy,
        )
        staged_fixed: OrderedDict[str, MetricBackend] = OrderedDict()
        staged_views: dict[str, str] = {}
        ranking_names: list[str] = []
        for spec in specs:
            if spec.derived_from is not None:
                if spec.derived_from not in ranking_names:
                    ranking_names.append(spec.derived_from)
                continue
            if spec.name in {"auroc", "auprc"}:
                ranking_names.append(spec.name)
                continue
            factory = (
                spec.online_factory
                if context.policy == "online"
                else spec.exact_factory
            )
            if factory is None:
                raise ValueError(
                    f"Metric {spec.name!r} does not support policy "
                    f"{context.policy!r}"
                )
            created_backend = factory(factory_context)
            if not isinstance(created_backend, MetricBackend):
                raise TypeError(
                    f"Factory for metric {spec.name!r} did not return "
                    "a TopoBench MetricBackend"
                )
            staged_fixed[spec.name] = created_backend
            staged_views[spec.name] = spec.prediction_view

        ranking_names = list(dict.fromkeys(ranking_names))
        staged_exact = None
        if ranking_names and context.policy in {"exact", "audit"}:
            staged_exact = ExactRankingBackend(
                num_classes=self.num_classes,
                metrics=ranking_names,
                split=context.split,
                max_bytes=self.max_exact_ranking_bytes,
                expected_num_examples=context.expected_num_examples,
            )
        staged_online = None
        if ranking_names and context.policy in {"online", "audit"}:
            staged_online = OnlineRankingBackend(
                num_classes=self.num_classes,
                metrics=ranking_names,
                threshold_count=self.ranking_thresholds,
                device=self.device,
            )

        self.context = context
        self.policy = context.policy
        self._specs = specs
        self._support = staged_support
        self._fixed = staged_fixed
        self._fixed_views = staged_views
        self.exact_ranking_backend = staged_exact
        self.online_ranking_backend = staged_online
        self._auto_device_pending = not self._device_explicit

    def _move_streaming_state(self, device: torch.device) -> None:
        self.device = device
        for backend in self._fixed.values():
            backend.to(device)
        if self.online_ranking_backend is not None:
            self.online_ranking_backend.to(device)
        self._support.to(device)
        self._auto_device_pending = False

    def update(self, batch: EvaluationBatch) -> None:
        if self.context is None:
            raise RuntimeError("MetricPolicyBackend.update requires begin")
        if self._auto_device_pending:
            self._move_streaming_state(batch.outputs.device)
        if (
            batch.outputs.device != self.device
            or batch.targets.device != self.device
        ):
            raise ValueError(
                f"batch tensors must be on evaluation device {self.device}"
            )
        if self.exact_ranking_backend is not None:
            self.exact_ranking_backend.guard_append(
                batch.num_examples, batch.outputs.dtype, batch.targets.dtype
            )
        required_views = set(self._fixed_views.values())
        if (
            self.exact_ranking_backend is not None
            or self.online_ranking_backend is not None
        ):
            required_views.add(
                "positive_probabilities"
                if self.num_classes == 2
                else "probabilities"
            )
        if "positive_probabilities" in required_views:
            required_views.add("probabilities")
        views = self.prediction_views_factory(batch.outputs.detach())
        materialized: dict[str, Tensor] = {}
        for name in (
            "probabilities",
            "positive_probabilities",
            "hard_classes",
            "raw",
        ):
            if name in required_views:
                materialized[name] = getattr(views, name)
        detached_targets = batch.targets.detach()
        for name, backend in self._fixed.items():
            backend.update(
                materialized[self._fixed_views[name]].detach(),
                detached_targets,
            )
        ranking_view_name = (
            "positive_probabilities"
            if self.num_classes == 2
            else "probabilities"
        )
        if self.exact_ranking_backend is not None:
            self.exact_ranking_backend.update(
                materialized[ranking_view_name], detached_targets
            )
        if self.online_ranking_backend is not None:
            self.online_ranking_backend.update(
                materialized[ranking_view_name].detach(), detached_targets
            )
        self._support.update(batch.targets)

    def _undefined_reason(self, metric: str) -> str | None:
        if self._support.num_examples == 0:
            return "empty_evaluation"
        if self.task == "classification":
            support = self._support.class_support()
            if (
                metric in {"auroc", "auprc", "somers_d"}
                and self.num_classes == 2
                and any(count == 0 for count in support.values())
            ):
                return "binary_target_single_class"
            if (
                metric == "auroc"
                and self.num_classes > 2
                and any(count == 0 for count in support.values())
            ):
                return "multiclass_target_missing_class"
            if metric in {"precision", "recall", "f1"} and any(
                count == 0 for count in support.values()
            ):
                return "macro_target_missing_class"
            return None
        if metric == "r2":
            if self._support.num_examples < 2:
                return "r2_too_few_examples"
            count = self._support.num_examples
            centered = (
                float(self._support.target_square_sum)
                - float(self._support.target_sum) ** 2 / count
            )
            if abs(centered) <= torch.finfo(torch.float64).eps * max(
                1.0, abs(float(self._support.target_square_sum))
            ):
                return "r2_constant_target"
        return None

    def _support_for(self, metric: str) -> Mapping[Any, Any]:
        del metric
        return (
            self._support.class_support()
            if self.task == "classification"
            else self._support.regression_support()
        )

    def _handle_undefined(self, metric: str, reason: str) -> Tensor:
        support = self._support_for(metric)
        if self.undefined_metric_policy == "error":
            assert self.context is not None
            raise UndefinedMetricError(
                metric=metric,
                split=self.context.split,
                reason=reason,
                support=support,
                num_examples=self._support.num_examples,
            )
        return torch.tensor(float("nan"), device=self.device)

    def _compute_sources(
        self,
    ) -> tuple[OrderedDict[str, Tensor], OrderedDict[str, Tensor]]:
        exact: OrderedDict[str, Tensor] = OrderedDict()
        online: OrderedDict[str, Tensor] = OrderedDict()
        for name, backend in self._fixed.items():
            reason = self._undefined_reason(name)
            exact[name] = (
                self._handle_undefined(name, reason)
                if reason
                else backend.compute()
            )
        ranking_names = list(
            dict.fromkeys(
                spec.derived_from or spec.name
                for spec in self._specs
                if spec.name in {"auroc", "auprc", "somers_d"}
            )
        )
        undefined_ranking = {
            name: self._undefined_reason(name) for name in ranking_names
        }
        if self.exact_ranking_backend is not None:
            if any(undefined_ranking.values()):
                for name in ranking_names:
                    reason = undefined_ranking[name]
                    exact[name] = (
                        self._handle_undefined(name, reason)
                        if reason
                        else torch.tensor(float("nan"), device=self.device)
                    )
            else:
                exact.update(self.exact_ranking_backend.compute())
        if self.online_ranking_backend is not None:
            if any(undefined_ranking.values()):
                for name in ranking_names:
                    reason = undefined_ranking[name]
                    online[name] = (
                        self._handle_undefined(name, reason)
                        if reason
                        else torch.tensor(float("nan"), device=self.device)
                    )
            else:
                online.update(self.online_ranking_backend.compute())
        return exact, online

    def compute(self) -> BackendSnapshot:
        if self.context is None or self.policy is None:
            raise RuntimeError("MetricPolicyBackend.compute requires begin")
        if self.undefined_metric_policy == "error":
            for spec in self._specs:
                undefined_reason = self._undefined_reason(spec.name)
                if undefined_reason is not None:
                    self._handle_undefined(spec.name, undefined_reason)
        exact, online = self._compute_sources()
        values: OrderedDict[str, Tensor] = OrderedDict()
        statuses: dict[str, str] = {}
        support: dict[str, Any] = {}
        reasons: dict[str, str | None] = {}
        thresholds: dict[str, int | None] = {}
        for spec in self._specs:
            name = spec.name
            reason = self._undefined_reason(name)
            if spec.derived_from is not None:
                base = spec.derived_from
                exact_value = 2 * exact[base] - 1 if base in exact else None
                online_value = 2 * online[base] - 1 if base in online else None
            else:
                exact_value = exact.get(name)
                online_value = online.get(name)
            if self.policy == "online" and name in {
                "auroc",
                "auprc",
                "somers_d",
            }:
                values[name] = online_value
                statuses[name] = "undefined" if reason else "approximate"
                thresholds[name] = self.ranking_thresholds
            elif self.policy == "audit" and name in {
                "auroc",
                "auprc",
                "somers_d",
            }:
                values[name] = exact_value
                values[f"{name}_online"] = online_value
                values[f"{name}_online_abs_error"] = torch.abs(
                    exact_value - online_value
                )
                for output_name in (
                    name,
                    f"{name}_online",
                    f"{name}_online_abs_error",
                ):
                    statuses[output_name] = (
                        "undefined"
                        if reason
                        else (
                            "exact" if output_name == name else "approximate"
                        )
                    )
                    support[output_name] = self._support_for(name)
                    reasons[output_name] = reason
                thresholds[name] = None
                thresholds[f"{name}_online"] = self.ranking_thresholds
                thresholds[f"{name}_online_abs_error"] = (
                    self.ranking_thresholds
                )
                continue
            else:
                values[name] = exact_value
                statuses[name] = "undefined" if reason else "exact"
                thresholds[name] = None
            support[name] = self._support_for(name)
            reasons[name] = reason
        exact_memory: Mapping[str, Any] | None = None
        if self.exact_ranking_backend is not None:
            estimate = self.exact_ranking_backend.estimate()
            exact_memory = {
                "expected_num_examples": self.context.expected_num_examples,
                "observed_num_examples": self._support.num_examples,
                "retained_bytes": self.exact_ranking_backend.retained_bytes,
                "workspace_bytes": estimate.workspace_bytes,
                "estimated_peak_bytes": estimate.estimated_peak_bytes,
                "safety_factor": estimate.safety_factor,
                "configured_limit": self.max_exact_ranking_bytes,
                "buffer_device": "cpu",
                "score_dtype": str(estimate.score_dtype),
                "target_dtype": str(estimate.target_dtype),
                "class_count": self.num_classes,
                "layout": estimate.layout,
                "binary_state_shared": estimate.binary_state_shared,
            }
        metric_semantics: dict[str, Mapping[str, Any]] = {}
        for spec in self._specs:
            output_names = [spec.name]
            if self.policy == "audit" and spec.name in {
                "auroc",
                "auprc",
                "somers_d",
            }:
                output_names.extend(
                    (
                        f"{spec.name}_online",
                        f"{spec.name}_online_abs_error",
                    )
                )
            aggregation = spec.aggregation
            if spec.name == "auroc":
                aggregation = (
                    "binary" if self.num_classes == 2 else "macro_ovr"
                )
            semantics = {
                "aggregation": aggregation,
                "higher_is_better": spec.higher_is_better,
                "positive_class": spec.positive_class,
                "orientation": spec.orientation,
                "derived_from": spec.derived_from,
                "undefined_metric_policy": self.undefined_metric_policy,
            }
            for output_name in output_names:
                metric_semantics[output_name] = semantics
        provenance = {
            "thresholds": thresholds,
            "metric_semantics": metric_semantics,
            "exact_ranking_memory": exact_memory,
            "device_policy": {
                "evaluation_device": str(self.device),
                "exact_buffer_device": (
                    "cpu" if self.exact_ranking_backend is not None else None
                ),
                "cuda_qualified": self.device.type == "cuda",
            },
        }
        return BackendSnapshot(values, statuses, support, reasons, provenance)

    def state_dict(self) -> dict[str, Any]:
        """Serialize an active online backend using checkpoint-safe values."""
        context = self.context
        if context is None or self.policy != "online":
            raise RuntimeError(
                "evaluator checkpointing requires an active online backend"
            )
        fixed = {
            name: _backend_state_dict(backend)
            for name, backend in self._fixed.items()
        }
        return {
            "format_version": "metric-policy-backend-state-v1",
            "context": {
                "split": context.split,
                "pass_kind": context.pass_kind,
                "policy": context.policy,
                "task": context.task,
                "num_classes": context.num_classes,
                "expected_num_examples": context.expected_num_examples,
                "vocabulary_id": context.vocabulary_id,
                "model_id": context.model_id,
                "checkpoint_id": context.checkpoint_id,
                "qualified": context.qualified,
            },
            "fixed": fixed,
            "online_ranking": (
                None
                if self.online_ranking_backend is None
                else self.online_ranking_backend.state_dict()
            ),
            "support": {
                "num_examples": self._support.num_examples,
                "class_counts": (
                    None
                    if self._support.class_counts is None
                    else self._support.class_counts.detach().clone()
                ),
                "target_sum": self._support.target_sum.detach().clone(),
                "target_square_sum": (
                    self._support.target_square_sum.detach().clone()
                ),
            },
        }

    def load_state_dict(
        self,
        state_dict: Mapping[str, Any],
        *,
        strict: bool,
    ) -> None:
        """Restore one active online backend after exact schema validation."""
        if strict is not True:
            raise TypeError("strict must be True")
        if not isinstance(state_dict, Mapping):
            raise TypeError("backend state_dict must be a mapping")
        expected = {
            "format_version",
            "context",
            "fixed",
            "online_ranking",
            "support",
        }
        if set(state_dict) != expected:
            raise ValueError("backend state_dict keys do not match schema")
        if state_dict["format_version"] != "metric-policy-backend-state-v1":
            raise ValueError("unsupported backend state_dict format_version")
        context_record = state_dict["context"]
        if not isinstance(context_record, Mapping):
            raise TypeError("backend context state must be a mapping")
        context_keys = {
            "split",
            "pass_kind",
            "policy",
            "task",
            "num_classes",
            "expected_num_examples",
            "vocabulary_id",
            "model_id",
            "checkpoint_id",
            "qualified",
        }
        if set(context_record) != context_keys:
            raise ValueError("backend context state keys do not match schema")
        context = EvaluationContext(**dict(context_record))
        if context.policy != "online":
            raise ValueError(
                "backend checkpoint context must use online policy"
            )
        if (
            context.task != self.task
            or context.num_classes != self.num_classes
        ):
            raise ValueError(
                "backend checkpoint context does not match construction"
            )
        fixed = state_dict["fixed"]
        support = state_dict["support"]
        if not isinstance(fixed, Mapping):
            raise TypeError("backend fixed state must be a mapping")
        if not isinstance(support, Mapping):
            raise TypeError("backend support state must be a mapping")
        if set(support) != {
            "num_examples",
            "class_counts",
            "target_sum",
            "target_square_sum",
        }:
            raise ValueError("backend support state keys do not match schema")
        num_examples = support["num_examples"]
        if (
            isinstance(num_examples, bool)
            or not isinstance(num_examples, int)
            or num_examples < 0
        ):
            raise ValueError(
                "backend support num_examples must be a non-negative integer"
            )
        class_counts = support["class_counts"]
        target_sum = support["target_sum"]
        target_square_sum = support["target_square_sum"]
        if not isinstance(target_sum, Tensor) or not isinstance(
            target_square_sum, Tensor
        ):
            raise TypeError("backend support sums must be tensors")
        if target_sum.ndim != 0 or target_square_sum.ndim != 0:
            raise ValueError("backend support sums must be scalar tensors")
        if (
            target_sum.dtype != torch.float64
            or target_square_sum.dtype != torch.float64
        ):
            raise ValueError("backend support sums must use float64 dtype")
        if not bool(torch.isfinite(target_sum)) or not bool(
            torch.isfinite(target_square_sum)
        ):
            raise ValueError("backend support sums must be finite")
        if bool(target_square_sum < 0):
            raise ValueError("backend target_square_sum must be non-negative")
        if self.task == "classification":
            if not isinstance(class_counts, Tensor):
                raise TypeError("classification class_counts must be a tensor")
            if class_counts.dtype != torch.long:
                raise ValueError(
                    "classification class_counts must use long dtype"
                )
            if tuple(class_counts.shape) != (self.num_classes,):
                raise ValueError(
                    "classification class_counts shape does not match classes"
                )
            if bool((class_counts < 0).any()):
                raise ValueError(
                    "classification class_counts must be non-negative"
                )
            if int(class_counts.sum()) != num_examples:
                raise ValueError(
                    "backend class_counts do not match classification support"
                )
        elif class_counts is not None:
            raise ValueError("regression backend class_counts must be None")
        staged = MetricPolicyBackend(
            task=self.task,
            num_classes=self.num_classes,
            metrics=self.metric_names,
            custom_specs=self.custom_specs,
            ranking_thresholds=self.ranking_thresholds,
            max_exact_ranking_bytes=self.max_exact_ranking_bytes,
            undefined_metric_policy=self.undefined_metric_policy,
            device=self.device if self._device_explicit else None,
            prediction_views_factory=self.prediction_views_factory,
        )
        staged.begin(context)
        if set(fixed) != set(staged._fixed):
            raise ValueError(
                "backend fixed state keys do not match configured metrics"
            )
        for name, backend in staged._fixed.items():
            item = fixed[name]
            if not isinstance(item, Mapping):
                raise TypeError("fixed metric state must be a mapping")
            _load_backend_state_dict(backend, item)
        ranking_state = state_dict["online_ranking"]
        if staged.online_ranking_backend is None:
            if ranking_state is not None:
                raise ValueError("unexpected online ranking checkpoint state")
        else:
            if not isinstance(ranking_state, Mapping):
                raise TypeError(
                    "online ranking checkpoint state must be a mapping"
                )
            staged.online_ranking_backend.load_state_dict(
                ranking_state,
                strict=True,
            )
        staged._support.num_examples = num_examples
        staged._support.class_counts = (
            None
            if class_counts is None
            else class_counts.detach().to(staged.device).clone()
        )
        staged._support.target_sum = (
            target_sum.detach().to(staged.device).clone()
        )
        staged._support.target_square_sum = (
            target_square_sum.detach().to(staged.device).clone()
        )
        staged._support._cached_support = None

        self.device = staged.device
        self.policy = staged.policy
        self.context = staged.context
        self._specs = staged._specs
        self._fixed = staged._fixed
        self._fixed_views = staged._fixed_views
        self.exact_ranking_backend = staged.exact_ranking_backend
        self.online_ranking_backend = staged.online_ranking_backend
        self._support = staged._support
        self._auto_device_pending = staged._auto_device_pending

    def reset(self) -> None:
        for backend in self._fixed.values():
            backend.reset()
        if self.exact_ranking_backend is not None:
            self.exact_ranking_backend.reset()
        if self.online_ranking_backend is not None:
            self.online_ranking_backend.reset()
        self._fixed.clear()
        self._fixed_views.clear()
        self.exact_ranking_backend = None
        self.online_ranking_backend = None
        self._specs = ()
        self.context = None
        self.policy = None
        self._support = _SupportTracker(
            self.task, self.num_classes, self.device
        )
        self._auto_device_pending = not self._device_explicit

    @property
    def retained_bytes(self) -> int:
        return owned_tensor_bytes(
            (
                self._fixed,
                self.exact_ranking_backend,
                self.online_ranking_backend,
            )
        )

    @property
    def fixed_state_tensors(self) -> tuple[Tensor, ...]:
        return _state_tensors(self._fixed)

    def reachable_objects(self) -> tuple[object, ...]:
        return reachable_objects(self)


def make_accuracy_backend(context: BackendFactoryContext) -> MetricBackend:
    return TorchMetricBackend(
        MulticlassAccuracy(
            num_classes=context.num_classes, average="micro"
        ).to(context.device)
    )


def make_precision_backend(context: BackendFactoryContext) -> MetricBackend:
    return TorchMetricBackend(
        MulticlassPrecision(
            num_classes=context.num_classes, average="macro", zero_division=0
        ).to(context.device)
    )


def make_recall_backend(context: BackendFactoryContext) -> MetricBackend:
    return TorchMetricBackend(
        MulticlassRecall(
            num_classes=context.num_classes, average="macro", zero_division=0
        ).to(context.device)
    )


def make_f1_backend(context: BackendFactoryContext) -> MetricBackend:
    return TorchMetricBackend(
        MulticlassF1Score(
            num_classes=context.num_classes, average="macro", zero_division=0
        ).to(context.device)
    )


def make_mae_backend(context: BackendFactoryContext) -> MetricBackend:
    return TorchMetricBackend(MeanAbsoluteError().to(context.device))


def make_mse_backend(context: BackendFactoryContext) -> MetricBackend:
    return TorchMetricBackend(
        MeanSquaredError(squared=True).to(context.device)
    )


def make_rmse_backend(context: BackendFactoryContext) -> MetricBackend:
    return TorchMetricBackend(
        MeanSquaredError(squared=False).to(context.device)
    )


def make_r2_backend(context: BackendFactoryContext) -> MetricBackend:
    return TorchMetricBackend(R2Score().to(context.device))


def make_exact_auroc_backend(context: BackendFactoryContext) -> MetricBackend:
    del context
    return _FunctionalExactMetric("auroc")


def make_exact_auprc_backend(context: BackendFactoryContext) -> MetricBackend:
    del context
    return _FunctionalExactMetric("auprc")


def make_online_auroc_backend(context: BackendFactoryContext) -> MetricBackend:
    thresholds = context.threshold_grid
    if thresholds is None:
        thresholds = torch.linspace(
            0.0, 1.0, context.ranking_thresholds, device=context.device
        )
    metric: Metric = (
        BinaryAUROC(thresholds=thresholds)
        if context.num_classes == 2
        else MulticlassAUROC(
            num_classes=context.num_classes,
            average="macro",
            thresholds=thresholds,
        )
    )
    return TorchMetricBackend(metric.to(context.device))


def make_online_auprc_backend(context: BackendFactoryContext) -> MetricBackend:
    thresholds = context.threshold_grid
    if thresholds is None:
        thresholds = torch.linspace(
            0.0, 1.0, context.ranking_thresholds, device=context.device
        )
    return TorchMetricBackend(
        BinaryAveragePrecision(thresholds=thresholds).to(context.device)
    )
