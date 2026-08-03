"""Evaluator tests for the surviving target-shape contracts."""

import pytest
import torch

from topobench.evaluator import EvaluationBatch, EvaluationContext, TBEvaluator


def _context(task: str, num_classes: int) -> EvaluationContext:
    return EvaluationContext(
        split="val",
        pass_kind="fit_epoch",
        policy="exact",
        task=task,
        num_classes=num_classes,
    )


@pytest.mark.parametrize("batch_size", [3, 1])
def test_scalar_regression_accepts_only_equal_floating_n_by_one(
    batch_size: int,
) -> None:
    evaluator = TBEvaluator(task="regression", num_classes=1, metrics=["mae"])
    evaluator.begin(_context("regression", 1))
    evaluator.update(
        EvaluationBatch(
            outputs=torch.randn(batch_size, 1),
            targets=torch.randn(batch_size, 1),
            num_examples=batch_size,
        )
    )

    assert evaluator.finalize().num_examples == batch_size


@pytest.mark.parametrize(
    ("outputs", "targets", "message"),
    [
        (torch.randn(3), torch.randn(3, 1), r"shape \[N, 1\]"),
        (torch.randn(3, 1), torch.randn(3), r"shape \[N, 1\]"),
        (torch.randn(3, 2), torch.randn(3, 2), r"shape \[N, 1\]"),
        (torch.randn(3, 1), torch.randn(3, 2), r"shape \[N, 1\]"),
        (
            torch.randn(3, 1),
            torch.tensor([[1], [2], [3]], dtype=torch.long),
            "floating tensors",
        ),
    ],
)
def test_scalar_regression_rejects_broadcast_and_dtype_variants(
    outputs: torch.Tensor, targets: torch.Tensor, message: str
) -> None:
    evaluator = TBEvaluator(task="regression", num_classes=1, metrics=["mae"])
    evaluator.begin(_context("regression", 1))

    with pytest.raises((TypeError, ValueError), match=message):
        evaluator.update(
            EvaluationBatch(
                outputs=outputs,
                targets=targets,
                num_examples=outputs.shape[0],
            )
        )


@pytest.mark.parametrize("batch_size", [4, 1])
def test_classification_accepts_rank_two_outputs_and_long_targets(
    batch_size: int,
) -> None:
    evaluator = TBEvaluator(
        task="classification", num_classes=3, metrics=["accuracy"]
    )
    evaluator.begin(_context("classification", 3))
    evaluator.update(
        EvaluationBatch(
            outputs=torch.randn(batch_size, 3),
            targets=torch.arange(batch_size, dtype=torch.long) % 3,
            num_examples=batch_size,
        )
    )

    assert evaluator.finalize().num_examples == batch_size


def test_binary_classification_is_n_by_two_with_zero_one_targets() -> None:
    evaluator = TBEvaluator(
        task="classification", num_classes=2, metrics=["accuracy"]
    )
    evaluator.begin(_context("classification", 2))
    evaluator.update(
        EvaluationBatch(
            outputs=torch.tensor([[0.1, 0.9], [0.8, 0.2]]),
            targets=torch.tensor([1, 0], dtype=torch.long),
            num_examples=2,
        )
    )

    assert evaluator.finalize().num_examples == 2


@pytest.mark.parametrize(
    ("outputs", "targets", "message"),
    [
        (torch.randn(3), torch.tensor([0, 1, 2]), "rank-2"),
        (torch.randn(3, 3), torch.tensor([[0], [1], [2]]), "rank-1"),
        (
            torch.randn(3, 3),
            torch.tensor([0.0, 1.0, 2.0]),
            "torch.long",
        ),
        (torch.randn(3, 2), torch.tensor([0, 1, 2]), "class dimension"),
        (torch.randn(3, 3), torch.tensor([0, 1, 3]), "class IDs"),
    ],
)
def test_classification_rejects_wrong_rank_dtype_vocabulary_or_ids(
    outputs: torch.Tensor, targets: torch.Tensor, message: str
) -> None:
    evaluator = TBEvaluator(
        task="classification", num_classes=3, metrics=["accuracy"]
    )
    evaluator.begin(_context("classification", 3))

    with pytest.raises((TypeError, ValueError), match=message):
        evaluator.update(
            EvaluationBatch(
                outputs=outputs,
                targets=targets,
                num_examples=outputs.shape[0],
            )
        )
