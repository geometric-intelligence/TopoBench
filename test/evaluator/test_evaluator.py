"""Tests for the reduced evaluator construction surface."""

from collections import OrderedDict

import pytest
import torch

from topobench.evaluator import EvaluationBatch, EvaluationContext, TBEvaluator


@pytest.mark.parametrize(
    "task",
    ["multilabel classification", "multioutput classification", "wrong_task"],
)
def test_unsupported_tasks_fail_during_construction(task: str) -> None:
    with pytest.raises(ValueError, match="Supported tasks"):
        TBEvaluator(task=task, num_classes=3, metrics=["accuracy"])


@pytest.mark.parametrize("metric", ["auprc", "somers_d"])
def test_binary_only_metrics_reject_nonbinary_vocabularies(
    metric: str,
) -> None:
    with pytest.raises(ValueError, match="num_classes == 2"):
        TBEvaluator(task="classification", num_classes=3, metrics=[metric])


@pytest.mark.parametrize(
    "metric",
    [
        "example",
        "confusion_matrix",
        "f1_macro",
        "f1_weighted",
        "accuracy-0",
        "f1_1",
    ],
)
def test_removed_metric_names_fail_during_construction(metric: str) -> None:
    with pytest.raises(ValueError, match="Unsupported metric"):
        TBEvaluator(task="classification", num_classes=3, metrics=[metric])


def test_duplicate_metrics_fail_during_construction() -> None:
    with pytest.raises(ValueError, match="Duplicate metric"):
        TBEvaluator(
            task="classification",
            num_classes=3,
            metrics=["accuracy", "accuracy"],
        )


def test_classification_round_trip_uses_typed_lifecycle() -> None:
    evaluator = TBEvaluator(
        task="classification", num_classes=3, metrics=["accuracy"]
    )
    context = EvaluationContext(
        split="val",
        pass_kind="fit_epoch",
        policy="exact",
        task="classification",
        num_classes=3,
        expected_num_examples=3,
    )
    evaluator.begin(context)
    evaluator.update(
        EvaluationBatch(
            outputs=torch.tensor(
                [[9.0, 0.0, 0.0], [0.0, 8.0, 0.0], [0.0, 0.0, 7.0]]
            ),
            targets=torch.tensor([0, 1, 2], dtype=torch.long),
            num_examples=3,
        )
    )

    result = evaluator.finalize()

    assert tuple(result.metrics) == ("accuracy",)
    assert result.metrics["accuracy"] == pytest.approx(1.0)
    assert result.num_examples == 3
    assert evaluator.state == "idle"


def test_regression_round_trip_preserves_metric_order() -> None:
    evaluator = TBEvaluator(
        task="regression",
        num_classes=1,
        metrics=OrderedDict.fromkeys(["mae", "mse", "rmse", "r2"]),
    )
    context = EvaluationContext(
        split="test",
        pass_kind="selected_checkpoint",
        policy="exact",
        task="regression",
        num_classes=1,
        expected_num_examples=3,
    )
    evaluator.begin(context)
    evaluator.update(
        EvaluationBatch(
            outputs=torch.tensor([[0.0], [2.0], [5.0]]),
            targets=torch.tensor([[0.0], [1.0], [4.0]]),
            num_examples=3,
        )
    )

    result = evaluator.finalize()

    assert tuple(result.metrics) == ("mae", "mse", "rmse", "r2")


def test_repr_names_task_and_configured_metric_order() -> None:
    evaluator = TBEvaluator(
        task="classification",
        num_classes=3,
        metrics=["accuracy", "f1"],
    )

    assert repr(evaluator) == (
        "TBEvaluator(task='classification', num_classes=3, "
        "metrics=('accuracy', 'f1'))"
    )


def test_audit_result_integrates_expanded_metric_keys_and_metadata() -> None:
    evaluator = TBEvaluator(
        task="classification",
        num_classes=2,
        metrics=["accuracy", "auroc", "auprc", "somers_d"],
    )
    context = EvaluationContext(
        split="val",
        pass_kind="fit_epoch",
        policy="audit",
        task="classification",
        num_classes=2,
        expected_num_examples=4,
    )
    evaluator.begin(context)
    evaluator.update(
        EvaluationBatch(
            outputs=torch.tensor(
                [[3.0, 0.0], [0.0, 3.0], [1.2, 0.4], [0.1, 1.7]]
            ),
            targets=torch.tensor([0, 1, 0, 1]),
            num_examples=4,
        )
    )

    result = evaluator.finalize()

    assert tuple(result.metrics) == (
        "accuracy",
        "auroc",
        "auroc_online",
        "auroc_online_abs_error",
        "auprc",
        "auprc_online",
        "auprc_online_abs_error",
        "somers_d",
        "somers_d_online",
        "somers_d_online_abs_error",
    )
    assert result.status["accuracy"] == "exact"
    assert result.status["auroc"] == "exact"
    assert result.status["auroc_online"] == "approximate"
    assert (
        result.provenance["exact_ranking_memory"]["binary_state_shared"]
        is True
    )


def test_evaluator_exposes_only_topobench_composite_backend_state() -> None:
    evaluator = TBEvaluator(
        task="classification",
        num_classes=2,
        metrics=["auroc", "auprc", "somers_d"],
    )

    assert tuple(evaluator.backends) == ("metrics",)
    assert evaluator.metric_backend is evaluator.backends["metrics"]
    assert evaluator.metric_backend.__class__.__module__ == (
        "topobench.evaluator.backends"
    )
