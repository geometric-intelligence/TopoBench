"""Validation tests for evaluator value contracts."""

from collections import OrderedDict
from dataclasses import FrozenInstanceError

import pytest
import torch

from topobench.evaluator import EvaluationBatch, EvaluationContext, EvaluationResult


def _context(**overrides: object) -> EvaluationContext:
    values = {
        "split": "val",
        "pass_kind": "fit_epoch",
        "policy": "exact",
        "task": "classification",
        "num_classes": 3,
        "expected_num_examples": 3,
    }
    values.update(overrides)
    return EvaluationContext(**values)


@pytest.mark.parametrize("value", [True, False, 0, -1])
def test_context_rejects_nonpositive_or_boolean_expected_counts(
    value: object,
) -> None:
    with pytest.raises((TypeError, ValueError), match="expected_num_examples"):
        _context(expected_num_examples=value)


@pytest.mark.parametrize("value", [True, False, 0, -1, 1.5])
def test_context_rejects_invalid_num_classes(value: object) -> None:
    with pytest.raises((TypeError, ValueError), match="num_classes"):
        _context(num_classes=value)


@pytest.mark.parametrize(
    ("field", "value"),
    [
        ("split", "validation"),
        ("pass_kind", "predict"),
        ("policy", "automatic"),
        ("task", "multilabel classification"),
    ],
)
def test_context_rejects_invalid_literals(field: str, value: str) -> None:
    with pytest.raises(ValueError, match=field):
        _context(**{field: value})


def test_context_rejects_non_scalar_regression_vocabulary() -> None:
    with pytest.raises(ValueError, match="num_classes must be 1"):
        _context(task="regression", num_classes=2)


def test_context_is_immutable() -> None:
    context = _context()

    with pytest.raises(FrozenInstanceError):
        context.policy = "online"


@pytest.mark.parametrize("value", [True, False, 0, -1, 2.5])
def test_batch_rejects_invalid_num_examples(value: object) -> None:
    with pytest.raises((TypeError, ValueError), match="num_examples"):
        EvaluationBatch(
            outputs=torch.randn(3, 2),
            targets=torch.tensor([0, 1, 0]),
            num_examples=value,
        )


@pytest.mark.parametrize(
    ("outputs", "targets"),
    [
        ([[1.0], [2.0]], torch.tensor([[1.0], [2.0]])),
        (torch.tensor([[1.0], [2.0]]), [[1.0], [2.0]]),
    ],
)
def test_batch_requires_tensor_outputs_and_targets(
    outputs: object, targets: object
) -> None:
    with pytest.raises(TypeError, match="tensors"):
        EvaluationBatch(outputs=outputs, targets=targets, num_examples=2)


@pytest.mark.parametrize(
    ("outputs", "targets", "count"),
    [
        (torch.randn(3, 2), torch.tensor([0, 1, 0]), 2),
        (torch.randn(2, 2), torch.tensor([0, 1, 0]), 2),
    ],
)
def test_batch_count_matches_both_leading_tensor_dimensions(
    outputs: torch.Tensor, targets: torch.Tensor, count: int
) -> None:
    with pytest.raises(ValueError, match="leading tensor dimensions"):
        EvaluationBatch(outputs=outputs, targets=targets, num_examples=count)


def test_batch_preserves_tensor_storage_identity() -> None:
    outputs = torch.randn(3, 2, requires_grad=True)
    targets = torch.tensor([0, 1, 0])

    batch = EvaluationBatch(
        outputs=outputs, targets=targets, num_examples=3
    )

    assert batch.outputs.data_ptr() == outputs.data_ptr()
    assert batch.targets.data_ptr() == targets.data_ptr()
    assert batch.outputs.requires_grad


def test_batch_is_immutable() -> None:
    batch = EvaluationBatch(
        outputs=torch.randn(2, 2),
        targets=torch.tensor([0, 1]),
        num_examples=2,
    )

    with pytest.raises(FrozenInstanceError):
        batch.num_examples = 3


@pytest.mark.parametrize("value", [True, False, -1, 1.5])
def test_result_rejects_invalid_observed_counts(value: object) -> None:
    with pytest.raises((TypeError, ValueError), match="num_examples"):
        EvaluationResult(
            metrics={"accuracy": 1.0},
            num_examples=value,
            context=_context(),
        )


def test_result_allows_zero_observed_examples_for_explicit_empty_results() -> None:
    result = EvaluationResult(
        metrics={"mae": float("nan")},
        num_examples=0,
        context=_context(
            task="regression",
            num_classes=1,
            expected_num_examples=None,
        ),
        status={"mae": "undefined"},
        support={"mae": {"num_examples": 0}},
        reason={"mae": "empty_evaluation"},
    )

    assert result.num_examples == 0


@pytest.mark.parametrize("status", ["online", "ok", "invalid"])
def test_result_rejects_invalid_metric_status(status: str) -> None:
    with pytest.raises(ValueError, match="status"):
        EvaluationResult(
            metrics={"accuracy": 1.0},
            num_examples=3,
            context=_context(),
            status={"accuracy": status},
        )


def test_result_mapping_is_ordered_deeply_immutable_and_separates_count() -> None:
    result = EvaluationResult(
        metrics=OrderedDict(
            [("recall", torch.tensor(0.75)), ("accuracy", 0.8)]
        ),
        num_examples=3,
        context=_context(),
        status=OrderedDict(
            [("recall", "exact"), ("accuracy", "exact")]
        ),
        support={"recall": {"classes": [0, 1]}},
        reason={"recall": None},
        provenance={"backend": {"name": "fake", "thresholds": [0.1, 0.9]}},
    )

    assert tuple(result.metrics) == ("recall", "accuracy")
    assert "num_examples" not in result.metrics
    assert result.num_examples == 3
    assert result.split == "val"
    assert result.expected_num_examples == 3
    with pytest.raises(TypeError):
        result.metrics["f1"] = 0.5
    with pytest.raises(TypeError):
        result.support["recall"]["classes"] = (0,)
    with pytest.raises(TypeError):
        result.provenance["backend"]["name"] = "changed"
    assert result.support["recall"]["classes"] == (0, 1)
    assert result.provenance["backend"]["thresholds"] == (0.1, 0.9)


def test_result_is_immutable() -> None:
    result = EvaluationResult(
        metrics={"accuracy": 1.0},
        num_examples=3,
        context=_context(),
    )

    with pytest.raises(FrozenInstanceError):
        result.num_examples = 4
