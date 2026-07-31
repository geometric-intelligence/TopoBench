"""Evaluator tests for native graph target shapes."""

import pytest
import torch

from topobench.evaluator import TBEvaluator


def _regression_evaluator() -> TBEvaluator:
    return TBEvaluator(task="regression", metrics=["mae"])


@pytest.mark.parametrize("batch_size", [3, 1])
def test_scalar_regression_updates_exact_b_by_one(batch_size: int) -> None:
    """Metric state accepts full and final singleton batches without squeeze."""
    evaluator = _regression_evaluator()
    logits = torch.randn(batch_size, 1)
    targets = torch.randn(batch_size, 1)

    evaluator.update({"logits": logits, "labels": targets})

    assert torch.isfinite(evaluator.compute()["mae"])


@pytest.mark.parametrize(
    ("logits", "targets"),
    [
        (torch.randn(3, 1), torch.randn(3)),
        (torch.randn(3), torch.randn(3, 1)),
        (torch.randn(1, 1), torch.randn(1)),
        (torch.randn(3, 1), torch.randn(3, 1, 1)),
        (torch.randn(3, 2), torch.randn(3, 2)),
        (torch.randn(2, 1), torch.randn(3, 1)),
    ],
)
def test_scalar_regression_rejects_every_broadcast_opportunity(
    logits: torch.Tensor, targets: torch.Tensor
) -> None:
    """Metric updates do not normalize or broadcast regression targets."""
    evaluator = _regression_evaluator()

    with pytest.raises(ValueError, match="regression.*shape"):
        evaluator.update({"logits": logits, "labels": targets})


@pytest.mark.parametrize(
    "targets",
    [
        torch.ones(3, 1, dtype=torch.long),
        torch.tensor([[0.0], [float("nan")], [1.0]]),
        torch.tensor([[0.0], [float("inf")], [1.0]]),
    ],
)
def test_scalar_regression_rejects_invalid_target_values(
    targets: torch.Tensor,
) -> None:
    """Bad target dtype or values fail before torchmetrics state changes."""
    evaluator = _regression_evaluator()
    message = "floating" if not targets.is_floating_point() else "finite"

    with pytest.raises((TypeError, ValueError), match=message):
        evaluator.update(
            {"logits": torch.randn(3, 1), "labels": targets}
        )


@pytest.mark.parametrize("batch_size", [4, 1])
def test_classification_updates_rank_one_targets(batch_size: int) -> None:
    """Classification metrics accept graph or node leading dimensions."""
    evaluator = TBEvaluator(
        task="classification", num_classes=3, metrics=["accuracy"]
    )

    evaluator.update(
        {
            "logits": torch.randn(batch_size, 3),
            "labels": torch.arange(batch_size, dtype=torch.long) % 3,
        }
    )

    assert "accuracy" in evaluator.compute()


@pytest.mark.parametrize(
    "targets",
    [
        torch.zeros(3, 1, dtype=torch.long),
        torch.zeros(3, 1, 1, dtype=torch.long),
    ],
)
def test_classification_rejects_wrong_target_rank(
    targets: torch.Tensor,
) -> None:
    """Evaluator never squeezes classification labels."""
    evaluator = TBEvaluator(
        task="classification", num_classes=3, metrics=["accuracy"]
    )

    with pytest.raises(ValueError, match="classification.*rank-1"):
        evaluator.update(
            {"logits": torch.randn(3, 3), "labels": targets}
        )
