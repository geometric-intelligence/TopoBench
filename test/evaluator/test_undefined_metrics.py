"""Structured undefined metric behavior for qualified and exploratory runs."""

import math

import pytest
import torch

from topobench.evaluator import EvaluationBatch, EvaluationContext, TBEvaluator
from topobench.evaluator.backends import UndefinedMetricError


def _context(task: str, classes: int, count: int | None, policy: str = "exact") -> EvaluationContext:
    return EvaluationContext(
        split="val",
        pass_kind="fit_epoch",
        policy=policy,
        task=task,
        num_classes=classes,
        expected_num_examples=count,
        qualified=True,
    )


def _finalize(
    *,
    task: str,
    classes: int,
    metrics: list[str],
    outputs: torch.Tensor | None,
    targets: torch.Tensor | None,
    undefined_policy: str,
):
    count = None if targets is None else len(targets)
    evaluator = TBEvaluator(
        task,
        num_classes=classes,
        metrics=metrics,
        undefined_metric_policy=undefined_policy,
    )
    evaluator.begin(_context(task, classes, count))
    if outputs is not None and targets is not None:
        evaluator.update(EvaluationBatch(outputs=outputs, targets=targets, num_examples=len(targets)))
    return evaluator.finalize()


@pytest.mark.parametrize("metric", ["auroc", "auprc", "somers_d"])
def test_single_class_binary_ranking_raises_structured_error_by_default(metric):
    evaluator = TBEvaluator("classification", num_classes=2, metrics=[metric])
    evaluator.begin(_context("classification", 2, 4))
    evaluator.update(
        EvaluationBatch(
            outputs=torch.tensor([[2.0, 0.0], [1.0, 0.1], [0.3, 0.2], [4.0, -1.0]]),
            targets=torch.zeros(4, dtype=torch.long),
            num_examples=4,
        )
    )
    with pytest.raises(UndefinedMetricError) as caught:
        evaluator.finalize()
    error = caught.value
    assert error.metric == metric
    assert error.split == "val"
    assert error.reason == "binary_target_single_class"
    assert error.support == {0: 4, 1: 0}
    assert error.num_examples == 4
    assert "undefined_metric_policy='nan'" in str(error)


@pytest.mark.parametrize("metric", ["auroc", "auprc", "somers_d"])
def test_single_class_binary_nan_mode_returns_reason_and_support(metric):
    result = _finalize(
        task="classification",
        classes=2,
        metrics=[metric],
        outputs=torch.tensor([[2.0, 0.0], [1.0, 0.1], [0.3, 0.2], [4.0, -1.0]]),
        targets=torch.zeros(4, dtype=torch.long),
        undefined_policy="nan",
    )
    assert math.isnan(float(result.metrics[metric]))
    assert result.status[metric] == "undefined"
    assert result.reason[metric] == "binary_target_single_class"
    assert result.support[metric] == {0: 4, 1: 0}
    assert float(result.metrics[metric]) != 0.0


def test_multiclass_auroc_requires_every_vocabulary_class_support():
    result = _finalize(
        task="classification",
        classes=3,
        metrics=["auroc"],
        outputs=torch.tensor([[2.0, 0.0, -1.0], [0.1, 1.4, 0.0], [0.3, 0.5, -0.2]]),
        targets=torch.tensor([0, 1, 1]),
        undefined_policy="nan",
    )
    assert math.isnan(float(result.metrics["auroc"]))
    assert result.reason["auroc"] == "multiclass_target_missing_class"
    assert result.support["auroc"] == {0: 1, 1: 2, 2: 0}


@pytest.mark.parametrize("metric", ["precision", "recall", "f1"])
def test_macro_metrics_report_absent_class_support(metric):
    with pytest.raises(UndefinedMetricError) as caught:
        _finalize(
            task="classification",
            classes=3,
            metrics=[metric],
            outputs=torch.tensor([[2.0, 0.0, -1.0], [0.1, 1.4, 0.0], [0.3, 0.5, -0.2]]),
            targets=torch.tensor([0, 1, 1]),
            undefined_policy="error",
        )
    assert caught.value.metric == metric
    assert caught.value.reason == "macro_target_missing_class"
    assert caught.value.support == {0: 1, 1: 2, 2: 0}


@pytest.mark.parametrize(
    ("outputs", "targets", "reason"),
    [
        (torch.tensor([[1.0]]), torch.tensor([[1.5]]), "r2_too_few_examples"),
        (
            torch.tensor([[0.0], [1.0], [2.0]]),
            torch.tensor([[4.0], [4.0], [4.0]]),
            "r2_constant_target",
        ),
    ],
)
def test_r2_undefined_cases_are_not_coerced_to_zero(outputs, targets, reason):
    result = _finalize(
        task="regression",
        classes=1,
        metrics=["r2"],
        outputs=outputs,
        targets=targets,
        undefined_policy="nan",
    )
    assert math.isnan(float(result.metrics["r2"]))
    assert result.reason["r2"] == reason
    assert result.status["r2"] == "undefined"
    assert result.support["r2"]["num_examples"] == len(targets)


def test_empty_evaluation_raises_structured_error_in_qualified_default():
    evaluator = TBEvaluator("regression", num_classes=1, metrics=["mae", "r2"])
    evaluator.begin(_context("regression", 1, None))
    with pytest.raises(UndefinedMetricError) as caught:
        evaluator.finalize()
    assert caught.value.metric == "mae"
    assert caught.value.reason == "empty_evaluation"
    assert caught.value.num_examples == 0


def test_empty_evaluation_nan_mode_returns_all_metrics_with_metadata():
    evaluator = TBEvaluator(
        "regression",
        num_classes=1,
        metrics=["mae", "mse", "rmse", "r2"],
        undefined_metric_policy="nan",
    )
    evaluator.begin(_context("regression", 1, None))
    result = evaluator.finalize()
    assert result.num_examples == 0
    assert tuple(result.metrics) == ("mae", "mse", "rmse", "r2")
    assert all(math.isnan(float(value)) for value in result.metrics.values())
    assert result.status == {name: "undefined" for name in result.metrics}
    assert result.reason == {name: "empty_evaluation" for name in result.metrics}
    assert all(metadata["num_examples"] == 0 for metadata in result.support.values())


def test_invalid_undefined_policy_fails_at_construction():
    with pytest.raises(ValueError, match="undefined_metric_policy"):
        TBEvaluator(
            "classification",
            num_classes=2,
            metrics=["accuracy"],
            undefined_metric_policy="zero",
        )
