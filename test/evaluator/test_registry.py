"""Contracts for the explicit immutable evaluator metric registry."""

from dataclasses import FrozenInstanceError, fields
from types import MappingProxyType

import pytest

import topobench.evaluator as evaluator_public
from topobench.evaluator.backends import MetricBackend
from topobench.evaluator.registry import (
    BUILTIN_METRIC_SPECS,
    MetricSpec,
    resolve_metric_specs,
)

BUILTIN_NAMES = (
    "accuracy",
    "precision",
    "recall",
    "f1",
    "auroc",
    "auprc",
    "somers_d",
    "mae",
    "mse",
    "rmse",
    "r2",
)


def test_builtin_registry_is_immutable_and_exactly_reduced_surface():
    assert isinstance(BUILTIN_METRIC_SPECS, MappingProxyType)
    assert tuple(BUILTIN_METRIC_SPECS) == BUILTIN_NAMES
    with pytest.raises(TypeError):
        BUILTIN_METRIC_SPECS["other"] = BUILTIN_METRIC_SPECS["accuracy"]


def test_every_builtin_spec_declares_complete_semantics():
    required = {
        "name",
        "tasks",
        "prediction_view",
        "backend_group",
        "exact_factory",
        "online_factory",
        "scalar",
        "higher_is_better",
        "undefined_reasons",
    }
    assert required <= {field.name for field in fields(MetricSpec)}
    for name, spec in BUILTIN_METRIC_SPECS.items():
        assert spec.name == name
        assert spec.tasks
        assert spec.prediction_view in {
            "hard_classes",
            "probabilities",
            "positive_probabilities",
            "raw",
            "derived",
        }
        assert spec.backend_group
        assert spec.exact_factory is not None or spec.derived_from is not None
        assert spec.scalar is True
        assert isinstance(spec.higher_is_better, bool)
        assert isinstance(spec.undefined_reasons, frozenset)
        with pytest.raises(FrozenInstanceError):
            spec.name = "changed"


def test_binary_ranking_specs_share_exact_group_and_record_semantics():
    auroc = BUILTIN_METRIC_SPECS["auroc"]
    auprc = BUILTIN_METRIC_SPECS["auprc"]
    somers_d = BUILTIN_METRIC_SPECS["somers_d"]

    assert auroc.backend_group == auprc.backend_group == "exact_binary_ranking"
    assert auprc.aggregation == "average_precision"
    assert auprc.positive_class == 1
    assert somers_d.derived_from == "auroc"
    assert somers_d.derived_transform == "two_auc_minus_one"
    assert somers_d.orientation == "D_S_given_Y"
    assert somers_d.positive_class == 1
    assert somers_d.exact_factory is None
    assert somers_d.online_factory is None


@pytest.mark.parametrize("metric", ["auprc", "somers_d"])
@pytest.mark.parametrize("num_classes", [1, 3, 8])
def test_binary_only_metrics_reject_other_class_counts(metric, num_classes):
    with pytest.raises(ValueError, match=rf"{metric}.*num_classes=2"):
        resolve_metric_specs(
            [metric],
            task="classification",
            num_classes=num_classes,
            policy="exact",
        )


def test_duplicate_injected_name_is_rejected():
    duplicate = MetricSpec(
        name="accuracy",
        tasks=frozenset({"classification"}),
        prediction_view="hard_classes",
        backend_group="custom",
        exact_factory=lambda context: object(),
        online_factory=lambda context: object(),
        scalar=True,
        higher_is_better=True,
        undefined_reasons=frozenset(),
    )
    with pytest.raises(ValueError, match="Duplicate metric.*accuracy"):
        resolve_metric_specs(
            ["accuracy"],
            task="classification",
            num_classes=2,
            policy="exact",
            custom_specs=[duplicate],
        )


def test_reserved_result_name_is_rejected_before_factory_runs():
    called = False

    def factory(context):
        nonlocal called
        called = True
        return object()

    reserved = MetricSpec(
        name="num_examples",
        tasks=frozenset({"regression"}),
        prediction_view="raw",
        backend_group="custom",
        exact_factory=factory,
        online_factory=factory,
        scalar=True,
        higher_is_better=False,
        undefined_reasons=frozenset(),
    )
    with pytest.raises(ValueError, match="reserved.*num_examples"):
        resolve_metric_specs(
            ["num_examples"],
            task="regression",
            num_classes=1,
            policy="exact",
            custom_specs=[reserved],
        )
    assert called is False


@pytest.mark.parametrize(
    ("custom_name", "expected_collision"),
    [
        ("auroc_online", "auroc_online"),
        ("auroc_online_abs_error", "auroc_online_abs_error"),
    ],
)
def test_generated_audit_key_collision_is_rejected(
    custom_name, expected_collision
):
    custom = MetricSpec(
        name=custom_name,
        tasks=frozenset({"classification"}),
        prediction_view="hard_classes",
        backend_group="custom",
        exact_factory=lambda context: object(),
        online_factory=lambda context: object(),
        scalar=True,
        higher_is_better=True,
        undefined_reasons=frozenset(),
    )
    with pytest.raises(ValueError, match=expected_collision):
        resolve_metric_specs(
            ["auroc", custom_name],
            task="classification",
            num_classes=3,
            policy="audit",
            custom_specs=[custom],
        )


@pytest.mark.parametrize(
    ("task", "policy", "metric"),
    [
        ("regression", "exact", "accuracy"),
        ("classification", "exact", "mae"),
        ("regression", "audit", "mse"),
    ],
)
def test_unsupported_task_or_policy_fails_during_resolution(
    task, policy, metric
):
    with pytest.raises(ValueError, match=metric):
        resolve_metric_specs(
            [metric],
            task=task,
            num_classes=2 if task == "classification" else 1,
            policy=policy,
        )


def test_no_global_registration_or_torchmetrics_public_exports():
    assert not hasattr(evaluator_public, "METRICS")
    assert not hasattr(evaluator_public, "register_metric")
    assert not hasattr(evaluator_public, "Metric")
    assert all(
        "torchmetrics" not in value.__class__.__module__
        for value in evaluator_public.__dict__.values()
    )


def test_metric_backend_is_a_structural_topobench_protocol():
    assert MetricBackend.__module__ == "topobench.evaluator.backends"
