"""Dataset loss tests for native graph target shapes."""

import pytest
import torch
from torch_geometric.data import Data

from topobench.loss.dataset import DatasetLoss


def _regression_loss() -> DatasetLoss:
    return DatasetLoss({"task": "regression", "loss_type": "mse"})


@pytest.mark.parametrize("batch_size", [3, 1])
def test_scalar_regression_accepts_exact_b_by_one(batch_size: int) -> None:
    """Full and final singleton batches keep exact regression shape."""
    logits = torch.randn(batch_size, 1)
    targets = torch.randn(batch_size, 1)

    loss = _regression_loss()({"logits": logits, "labels": targets}, Data())

    assert loss.ndim == 0
    assert torch.isfinite(loss)


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
    """DatasetLoss never inserts dimensions or relies on broadcasting."""
    with pytest.raises(ValueError, match="regression.*shape"):
        _regression_loss()({"logits": logits, "labels": targets}, Data())


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
    """Regression targets are floating and finite before criterion dispatch."""
    message = "floating" if not targets.is_floating_point() else "finite"
    with pytest.raises((TypeError, ValueError), match=message):
        _regression_loss()(
            {"logits": torch.randn(3, 1), "labels": targets}, Data()
        )


@pytest.mark.parametrize("batch_size", [4, 1])
def test_classification_accepts_rank_one_targets(batch_size: int) -> None:
    """Graph and node classification share a rank-one target contract."""
    loss = DatasetLoss(
        {"task": "classification", "loss_type": "cross_entropy"}
    )

    value = loss(
        {
            "logits": torch.randn(batch_size, 3),
            "labels": torch.arange(batch_size, dtype=torch.long) % 3,
        },
        Data(),
    )

    assert value.ndim == 0


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
    """Classification never squeezes [B, 1] or deeper labels."""
    loss = DatasetLoss(
        {"task": "classification", "loss_type": "cross_entropy"}
    )

    with pytest.raises(ValueError, match="classification.*rank-1"):
        loss({"logits": torch.randn(3, 3), "labels": targets}, Data())
