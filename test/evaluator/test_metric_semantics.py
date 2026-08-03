"""Known-value and partition-invariance evaluator contracts."""

import pytest
import torch
from scipy.stats import somersd
from sklearn.metrics import (
    accuracy_score,
    average_precision_score,
    f1_score,
    mean_absolute_error,
    mean_squared_error,
    precision_score,
    r2_score,
    recall_score,
    roc_auc_score,
)

from topobench.evaluator import EvaluationBatch, EvaluationContext, TBEvaluator


def _context(
    task: str, num_classes: int, count: int, policy: str = "exact"
) -> EvaluationContext:
    return EvaluationContext(
        split="val",
        pass_kind="fit_epoch",
        policy=policy,
        task=task,
        num_classes=num_classes,
        expected_num_examples=count,
    )


def _evaluate(
    *,
    task: str,
    num_classes: int,
    metrics: list[str],
    outputs: torch.Tensor,
    targets: torch.Tensor,
    partitions: list[int],
    policy: str = "exact",
):
    evaluator = TBEvaluator(task, num_classes=num_classes, metrics=metrics)
    evaluator.begin(_context(task, num_classes, len(targets), policy))
    offset = 0
    for width in partitions:
        evaluator.update(
            EvaluationBatch(
                outputs=outputs[offset : offset + width],
                targets=targets[offset : offset + width],
                num_examples=width,
            )
        )
        offset += width
    assert offset == len(targets)
    return evaluator.finalize()


def _values(result) -> dict[str, float]:
    return {name: float(value) for name, value in result.metrics.items()}


def test_multiclass_known_values_match_sklearn():
    logits = torch.tensor(
        [
            [3.0, 0.2, -1.0],
            [0.0, 1.5, 0.4],
            [0.1, 0.8, 1.2],
            [1.2, 1.0, -0.2],
            [-0.4, 2.1, 0.1],
            [0.3, -0.2, 2.4],
        ]
    )
    targets = torch.tensor([0, 1, 2, 1, 0, 2])
    result = _values(
        _evaluate(
            task="classification",
            num_classes=3,
            metrics=["accuracy", "precision", "recall", "f1", "auroc"],
            outputs=logits,
            targets=targets,
            partitions=[6],
        )
    )
    hard = logits.argmax(dim=1).numpy()
    probabilities = logits.softmax(dim=1).numpy()
    expected = {
        "accuracy": accuracy_score(targets.numpy(), hard),
        "precision": precision_score(
            targets.numpy(), hard, average="macro", zero_division=0
        ),
        "recall": recall_score(
            targets.numpy(), hard, average="macro", zero_division=0
        ),
        "f1": f1_score(
            targets.numpy(), hard, average="macro", zero_division=0
        ),
        "auroc": roc_auc_score(
            targets.numpy(), probabilities, average="macro", multi_class="ovr"
        ),
    }
    assert result == pytest.approx(expected, abs=1e-6)


def test_binary_average_precision_and_somers_d_match_reference_libraries():
    positive_scores = torch.tensor([0.05, 0.25, 0.25, 0.70, 0.80, 0.95])
    logits = torch.stack(
        (torch.log1p(-positive_scores), torch.log(positive_scores)), dim=1
    )
    targets = torch.tensor([0, 1, 0, 1, 0, 1])
    result = _values(
        _evaluate(
            task="classification",
            num_classes=2,
            metrics=["auroc", "auprc", "somers_d"],
            outputs=logits,
            targets=targets,
            partitions=[6],
        )
    )
    expected_auc = roc_auc_score(targets.numpy(), positive_scores.numpy())
    assert result["auroc"] == pytest.approx(expected_auc, abs=1e-6)
    assert result["auprc"] == pytest.approx(
        average_precision_score(targets.numpy(), positive_scores.numpy()),
        abs=1e-6,
    )
    assert result["somers_d"] == pytest.approx(
        2.0 * expected_auc - 1.0, abs=1e-6
    )
    assert result["somers_d"] == pytest.approx(
        somersd(targets.numpy(), positive_scores.numpy()).statistic, abs=1e-6
    )


def test_binary_positive_class_and_label_reversal_semantics():
    scores = torch.tensor([0.05, 0.25, 0.25, 0.70, 0.80, 0.95])
    logits = torch.stack((torch.log1p(-scores), torch.log(scores)), dim=1)
    targets = torch.tensor([0, 1, 0, 1, 0, 1])
    original = _values(
        _evaluate(
            task="classification",
            num_classes=2,
            metrics=["auroc", "somers_d"],
            outputs=logits,
            targets=targets,
            partitions=[6],
        )
    )
    flipped_targets = 1 - targets
    flipped_logits = logits.flip(dims=(1,))
    flipped = _values(
        _evaluate(
            task="classification",
            num_classes=2,
            metrics=["auroc", "somers_d"],
            outputs=flipped_logits,
            targets=flipped_targets,
            partitions=[6],
        )
    )
    same_scores_reversed_labels = _values(
        _evaluate(
            task="classification",
            num_classes=2,
            metrics=["auroc"],
            outputs=logits,
            targets=flipped_targets,
            partitions=[6],
        )
    )
    assert flipped == pytest.approx(original, abs=1e-6)
    assert same_scores_reversed_labels["auroc"] == pytest.approx(
        1.0 - original["auroc"], abs=1e-6
    )


def test_regression_known_values_match_sklearn():
    outputs = torch.tensor([[0.0], [1.5], [2.0], [4.0], [7.5]])
    targets = torch.tensor([[1.0], [1.0], [3.0], [5.0], [8.0]])
    result = _values(
        _evaluate(
            task="regression",
            num_classes=1,
            metrics=["mae", "mse", "rmse", "r2"],
            outputs=outputs,
            targets=targets,
            partitions=[5],
        )
    )
    expected_mse = mean_squared_error(targets.numpy(), outputs.numpy())
    assert result == pytest.approx(
        {
            "mae": mean_absolute_error(targets.numpy(), outputs.numpy()),
            "mse": expected_mse,
            "rmse": expected_mse**0.5,
            "r2": r2_score(targets.numpy(), outputs.numpy()),
        },
        abs=1e-6,
    )


@pytest.mark.parametrize(
    ("task", "num_classes", "metrics", "outputs", "targets"),
    [
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
            torch.tensor(
                [
                    [2.0, -1.0],
                    [-1.0, 2.0],
                    [0.7, 0.6],
                    [0.2, 1.3],
                    [1.5, 0.0],
                    [0.4, 0.9],
                    [1.0, 1.0],
                ]
            ),
            torch.tensor([0, 1, 1, 1, 0, 0, 1]),
        ),
        (
            "classification",
            3,
            ["accuracy", "precision", "recall", "f1", "auroc"],
            torch.tensor(
                [
                    [2.0, 0.0, -1.0],
                    [0.0, 2.0, 0.1],
                    [0.1, 0.2, 2.0],
                    [1.0, 0.9, 0.0],
                    [0.0, 1.2, 1.1],
                    [0.4, 0.0, 1.3],
                    [1.4, 0.3, 0.2],
                ]
            ),
            torch.tensor([0, 1, 2, 1, 2, 0, 0]),
        ),
        (
            "regression",
            1,
            ["mae", "mse", "rmse", "r2"],
            torch.tensor([[0.0], [1.0], [2.5], [3.0], [5.0], [8.0], [13.0]]),
            torch.tensor([[0.5], [1.5], [2.0], [4.0], [5.5], [7.5], [12.0]]),
        ),
    ],
)
def test_every_metric_is_partition_invariant(
    task, num_classes, metrics, outputs, targets
):
    all_at_once = _values(
        _evaluate(
            task=task,
            num_classes=num_classes,
            metrics=metrics,
            outputs=outputs,
            targets=targets,
            partitions=[7],
        )
    )
    uneven = _values(
        _evaluate(
            task=task,
            num_classes=num_classes,
            metrics=metrics,
            outputs=outputs,
            targets=targets,
            partitions=[1, 3, 2, 1],
        )
    )
    assert uneven == pytest.approx(all_at_once, abs=1e-6)
