"""Immutable explicit evaluator metric specifications."""

from __future__ import annotations

from collections.abc import Callable, Sequence
from dataclasses import dataclass
from types import MappingProxyType
from typing import Literal

from .backends import (
    BackendFactoryContext,
    MetricBackend,
    make_accuracy_backend,
    make_exact_auprc_backend,
    make_exact_auroc_backend,
    make_f1_backend,
    make_mae_backend,
    make_mse_backend,
    make_online_auprc_backend,
    make_online_auroc_backend,
    make_precision_backend,
    make_r2_backend,
    make_recall_backend,
    make_rmse_backend,
)

PredictionView = Literal[
    "hard_classes",
    "probabilities",
    "positive_probabilities",
    "raw",
    "derived",
]
BackendFactory = Callable[[BackendFactoryContext], MetricBackend]

_RESERVED_RESULT_NAMES = frozenset(
    {
        "num_examples",
        "metrics",
        "context",
        "status",
        "support",
        "reason",
        "provenance",
        "split",
        "pass_kind",
        "policy",
        "task",
        "expected_num_examples",
    }
)
_ALLOWED_TASKS = frozenset({"classification", "regression"})
_ALLOWED_POLICIES = frozenset({"online", "exact", "audit"})
_ALLOWED_VIEWS = frozenset(
    {"hard_classes", "probabilities", "positive_probabilities", "raw", "derived"}
)


@dataclass(frozen=True, slots=True)
class MetricSpec:
    """Complete immutable semantics and factories for one scalar metric."""

    name: str
    tasks: frozenset[str]
    prediction_view: PredictionView
    backend_group: str
    exact_factory: BackendFactory | None
    online_factory: BackendFactory | None
    scalar: bool
    higher_is_better: bool
    undefined_reasons: frozenset[str]
    aggregation: str = "global"
    positive_class: int | None = None
    binary_only: bool = False
    derived_from: str | None = None
    derived_transform: str | None = None
    orientation: str | None = None

    def __post_init__(self) -> None:
        if not isinstance(self.name, str) or not self.name:
            raise ValueError("MetricSpec name must be a non-empty string")
        if not isinstance(self.tasks, frozenset):
            raise TypeError("MetricSpec tasks must be a frozenset")
        if not self.tasks or not self.tasks <= _ALLOWED_TASKS:
            raise ValueError("MetricSpec tasks must contain supported evaluator tasks")
        if self.prediction_view not in _ALLOWED_VIEWS:
            raise ValueError(f"Unsupported prediction view {self.prediction_view!r}")
        if not isinstance(self.backend_group, str) or not self.backend_group:
            raise ValueError("MetricSpec backend_group must be a non-empty string")
        if self.exact_factory is not None and not callable(self.exact_factory):
            raise TypeError("MetricSpec exact_factory must be callable or None")
        if self.online_factory is not None and not callable(self.online_factory):
            raise TypeError("MetricSpec online_factory must be callable or None")
        if not isinstance(self.scalar, bool) or not self.scalar:
            raise ValueError("MetricSpec must declare a scalar output")
        if not isinstance(self.higher_is_better, bool):
            raise TypeError("higher_is_better must be a boolean")
        if not isinstance(self.undefined_reasons, frozenset):
            raise TypeError("MetricSpec undefined_reasons must be a frozenset")
        if self.derived_from is None and self.exact_factory is None:
            raise ValueError("A non-derived MetricSpec requires an exact factory")
        if self.derived_from is not None:
            if self.prediction_view != "derived":
                raise ValueError("Derived metrics must use the derived prediction view")
            if self.derived_transform is None:
                raise ValueError("Derived metrics require a named transform")


def _spec(
    name: str,
    *,
    tasks: frozenset[str],
    view: PredictionView,
    group: str,
    exact: BackendFactory | None,
    online: BackendFactory | None,
    higher: bool,
    undefined: frozenset[str] = frozenset({"empty_evaluation"}),
    aggregation: str = "global",
    positive_class: int | None = None,
    binary_only: bool = False,
    derived_from: str | None = None,
    derived_transform: str | None = None,
    orientation: str | None = None,
) -> MetricSpec:
    return MetricSpec(
        name=name,
        tasks=tasks,
        prediction_view=view,
        backend_group=group,
        exact_factory=exact,
        online_factory=online,
        scalar=True,
        higher_is_better=higher,
        undefined_reasons=undefined,
        aggregation=aggregation,
        positive_class=positive_class,
        binary_only=binary_only,
        derived_from=derived_from,
        derived_transform=derived_transform,
        orientation=orientation,
    )


_CLASSIFICATION = frozenset({"classification"})
_REGRESSION = frozenset({"regression"})

_BUILTINS = {
    "accuracy": _spec(
        "accuracy",
        tasks=_CLASSIFICATION,
        view="hard_classes",
        group="classification_confusion",
        exact=make_accuracy_backend,
        online=make_accuracy_backend,
        higher=True,
        aggregation="micro",
    ),
    "precision": _spec(
        "precision",
        tasks=_CLASSIFICATION,
        view="hard_classes",
        group="classification_confusion",
        exact=make_precision_backend,
        online=make_precision_backend,
        higher=True,
        aggregation="macro",
        undefined=frozenset({"empty_evaluation", "macro_target_missing_class"}),
    ),
    "recall": _spec(
        "recall",
        tasks=_CLASSIFICATION,
        view="hard_classes",
        group="classification_confusion",
        exact=make_recall_backend,
        online=make_recall_backend,
        higher=True,
        aggregation="macro",
        undefined=frozenset({"empty_evaluation", "macro_target_missing_class"}),
    ),
    "f1": _spec(
        "f1",
        tasks=_CLASSIFICATION,
        view="hard_classes",
        group="classification_confusion",
        exact=make_f1_backend,
        online=make_f1_backend,
        higher=True,
        aggregation="macro",
        undefined=frozenset({"empty_evaluation", "macro_target_missing_class"}),
    ),
    "auroc": _spec(
        "auroc",
        tasks=_CLASSIFICATION,
        view="probabilities",
        group="exact_binary_ranking",
        exact=make_exact_auroc_backend,
        online=make_online_auroc_backend,
        higher=True,
        aggregation="macro_ovr",
        undefined=frozenset(
            {
                "empty_evaluation",
                "binary_target_single_class",
                "multiclass_target_missing_class",
            }
        ),
        positive_class=1,
    ),
    "auprc": _spec(
        "auprc",
        tasks=_CLASSIFICATION,
        view="positive_probabilities",
        group="exact_binary_ranking",
        exact=make_exact_auprc_backend,
        online=make_online_auprc_backend,
        higher=True,
        aggregation="average_precision",
        undefined=frozenset({"empty_evaluation", "binary_target_single_class"}),
        positive_class=1,
        binary_only=True,
    ),
    "somers_d": _spec(
        "somers_d",
        tasks=_CLASSIFICATION,
        view="derived",
        group="derived_binary_ranking",
        exact=None,
        online=None,
        higher=True,
        aggregation="asymmetric_rank_association",
        undefined=frozenset({"empty_evaluation", "binary_target_single_class"}),
        positive_class=1,
        binary_only=True,
        derived_from="auroc",
        derived_transform="two_auc_minus_one",
        orientation="D_S_given_Y",
    ),
    "mae": _spec(
        "mae",
        tasks=_REGRESSION,
        view="raw",
        group="regression_streaming",
        exact=make_mae_backend,
        online=make_mae_backend,
        higher=False,
    ),
    "mse": _spec(
        "mse",
        tasks=_REGRESSION,
        view="raw",
        group="regression_streaming",
        exact=make_mse_backend,
        online=make_mse_backend,
        higher=False,
    ),
    "rmse": _spec(
        "rmse",
        tasks=_REGRESSION,
        view="raw",
        group="regression_streaming",
        exact=make_rmse_backend,
        online=make_rmse_backend,
        higher=False,
    ),
    "r2": _spec(
        "r2",
        tasks=_REGRESSION,
        view="raw",
        group="regression_streaming",
        exact=make_r2_backend,
        online=make_r2_backend,
        higher=True,
        undefined=frozenset(
            {"empty_evaluation", "r2_too_few_examples", "r2_constant_target"}
        ),
    ),
}

BUILTIN_METRIC_SPECS = MappingProxyType(_BUILTINS)


def resolve_evaluation_policy(split: str, requested: str | None = None) -> str:
    """Resolve split defaults without silently changing an explicit policy."""
    if split not in {"train", "val", "test"}:
        raise ValueError("split must be train, val, or test")
    if requested is not None:
        if requested not in _ALLOWED_POLICIES:
            raise ValueError("policy must be online, exact, or audit")
        return requested
    return "online" if split == "train" else "exact"


def _audit_output_keys(specs: Sequence[MetricSpec]) -> tuple[str, ...]:
    keys: list[str] = []
    for spec in specs:
        keys.append(spec.name)
        if spec.name in {"auroc", "auprc", "somers_d"}:
            keys.extend(
                (f"{spec.name}_online", f"{spec.name}_online_abs_error")
            )
    return tuple(keys)


def resolve_metric_specs(
    names: Sequence[str],
    *,
    task: str,
    num_classes: int,
    policy: str,
    custom_specs: Sequence[MetricSpec] = (),
) -> tuple[MetricSpec, ...]:
    """Resolve constructor-local specs and reject all collisions eagerly."""
    if task not in _ALLOWED_TASKS:
        raise ValueError("Supported tasks are exactly: classification, regression")
    if policy not in _ALLOWED_POLICIES:
        raise ValueError("policy must be online, exact, or audit")
    if isinstance(names, (str, bytes)):
        raise TypeError("metric names must be an ordered sequence")
    requested = tuple(names)
    if not requested:
        raise ValueError("At least one metric must be configured")
    if len(set(requested)) != len(requested):
        raise ValueError("Duplicate metric names are not allowed")

    combined = dict(BUILTIN_METRIC_SPECS)
    seen_custom: set[str] = set()
    for spec in custom_specs:
        if not isinstance(spec, MetricSpec):
            raise TypeError("custom_specs must contain MetricSpec values")
        if spec.name in combined or spec.name in seen_custom:
            raise ValueError(f"Duplicate metric specification {spec.name!r}")
        seen_custom.add(spec.name)
        combined[spec.name] = spec

    resolved: list[MetricSpec] = []
    for name in requested:
        if name in _RESERVED_RESULT_NAMES:
            raise ValueError(f"reserved metric name {name!r}")
        if name not in combined:
            raise ValueError(
                f"Unknown metric {name!r}; Unsupported metric for {task}"
            )
        spec = combined[name]
        if task not in spec.tasks:
            raise ValueError(f"Metric {name!r} does not support task {task!r}")
        if spec.binary_only and num_classes != 2:
            raise ValueError(
                f"{name} requires classification num_classes == 2 "
                "(num_classes=2)"
            )
        if policy == "exact" and spec.exact_factory is None and spec.derived_from is None:
            raise ValueError(f"Metric {name!r} does not support exact policy")
        if policy == "online" and spec.online_factory is None and spec.derived_from is None:
            raise ValueError(f"Metric {name!r} does not support online policy")
        if policy == "audit" and (
            (spec.exact_factory is None or spec.online_factory is None)
            and spec.derived_from is None
        ):
            raise ValueError(f"Metric {name!r} does not support audit policy")
        resolved.append(spec)

    if policy == "audit":
        if task != "classification" or not any(
            spec.name in {"auroc", "auprc", "somers_d"} for spec in resolved
        ):
            metric_list = ", ".join(requested)
            raise ValueError(f"Metrics {metric_list} do not support audit policy")
        output_keys = _audit_output_keys(resolved)
        if len(set(output_keys)) != len(output_keys):
            collisions = sorted(
                key for key in set(output_keys) if output_keys.count(key) > 1
            )
            raise ValueError(f"Generated audit-key collision: {collisions}")
        reserved = _RESERVED_RESULT_NAMES.intersection(output_keys)
        if reserved:
            raise ValueError(f"Generated audit keys collide with reserved names: {sorted(reserved)}")

    return tuple(resolved)
