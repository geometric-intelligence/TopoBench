"""Evaluators for model evaluation."""

from torchmetrics.classification import AUROC, Accuracy, Precision, Recall
from torchmetrics.regression import MeanAbsoluteError, MeanSquaredError

from .metrics import ExampleRegressionMetric

# Define metrics
METRICS = {
    "accuracy": Accuracy,
    "precision": Precision,
    "recall": Recall,
    "auroc": AUROC,
    "mae": MeanAbsoluteError,
    "mse": MeanSquaredError,
    "example": ExampleRegressionMetric,
}

from .base import AbstractEvaluator  # noqa: E402
from .evaluator import TBEvaluator  # noqa: E402

__all__ = [
    "METRICS",
    "AbstractEvaluator",
    "TBEvaluator",
]
