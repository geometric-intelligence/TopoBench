"""Contract tests for the intentionally small evaluator vocabulary."""

import inspect

import hydra
import pytest
import torchmetrics

import topobench.evaluator as evaluator_api
from topobench.evaluator import TBEvaluator
from topobench.utils.config_resolvers import (
    get_default_metrics,
    register_all_resolvers,
)


@pytest.mark.parametrize(
    ("task", "num_classes", "expected"),
    [
        (
            "classification",
            3,
            ["accuracy", "precision", "recall", "f1", "auroc"],
        ),
        (
            "classification",
            2,
            [
                "accuracy",
                "precision",
                "recall",
                "f1",
                "auroc",
                "auprc",
                "somers_d",
            ],
        ),
        ("regression", 1, ["mae", "mse", "rmse", "r2"]),
    ],
)
def test_default_metrics_are_exact_and_ordered(
    task: str, num_classes: int, expected: list[str]
) -> None:
    assert get_default_metrics(task, num_classes) == expected


@pytest.mark.parametrize(
    "task",
    ["multilabel classification", "multioutput classification", "other"],
)
def test_default_metrics_reject_removed_tasks(task: str) -> None:
    with pytest.raises(ValueError, match="Supported tasks"):
        get_default_metrics(task, 2)


@pytest.mark.parametrize(
    ("task", "num_classes", "metrics"),
    [
        ("classification", 3, ["auprc"]),
        ("classification", 4, ["somers_d"]),
        ("regression", 1, ["accuracy"]),
        ("classification", 3, ["mae"]),
        ("classification", 3, ["example"]),
        ("classification", 3, ["accuracy-0"]),
    ],
)
def test_explicit_metrics_must_belong_to_active_vocabulary(
    task: str, num_classes: int, metrics: list[str]
) -> None:
    with pytest.raises(ValueError, match="metric"):
        get_default_metrics(task, num_classes, metrics)


@pytest.mark.parametrize("num_classes", [True, 0, -1])
def test_default_metrics_reject_invalid_class_counts(num_classes: int) -> None:
    with pytest.raises((TypeError, ValueError), match="num_classes"):
        get_default_metrics("classification", num_classes)


def test_public_evaluator_module_exports_no_torchmetrics_types() -> None:
    public_names = evaluator_api.__all__

    assert "METRICS" not in public_names
    assert "ExampleRegressionMetric" not in public_names
    assert all(
        not (inspect.isclass(value) and issubclass(value, torchmetrics.Metric))
        for name in public_names
        if inspect.isclass(value := getattr(evaluator_api, name))
    )


@pytest.mark.parametrize(
    "metric",
    [
        "example",
        "confusion_matrix",
        "f1_macro",
        "f1_weighted",
        "accuracy-0",
        "accuracy_0",
    ],
)
def test_removed_names_are_not_constructible(metric: str) -> None:
    with pytest.raises(ValueError, match="Unsupported metric"):
        TBEvaluator(task="classification", num_classes=3, metrics=[metric])


def _compose_selectable_evaluator_metrics(
    evaluator: str, dataset: str
) -> list[str]:
    hydra.core.global_hydra.GlobalHydra.instance().clear()
    register_all_resolvers()
    with hydra.initialize(
        version_base="1.3",
        config_path="../../configs",
    ):
        cfg = hydra.compose(
            config_name="run.yaml",
            overrides=[
                f"dataset=graph/{dataset}",
                f"evaluator={evaluator}",
            ],
        )
    return list(cfg.evaluator.metrics)


@pytest.mark.parametrize(
    ("evaluator", "dataset", "expected"),
    [
        (
            "classification",
            "SyntheticGraph",
            [
                "accuracy",
                "precision",
                "recall",
                "f1",
                "auroc",
                "auprc",
                "somers_d",
            ],
        ),
        (
            "classification",
            "IMDB-MULTI",
            ["accuracy", "precision", "recall", "f1", "auroc"],
        ),
        (
            "regression",
            "SyntheticGraphRegression",
            ["mae", "mse", "rmse", "r2"],
        ),
    ],
)
def test_selectable_evaluator_configs_use_authoritative_ordered_defaults(
    evaluator, dataset, expected
):
    assert (
        _compose_selectable_evaluator_metrics(evaluator, dataset) == expected
    )
