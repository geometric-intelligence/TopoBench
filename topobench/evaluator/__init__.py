"""Evaluators for model evaluation."""

from torchmetrics.classification import (
    AUROC,
    Accuracy,
    Precision,
    Recall,
    AveragePrecision,
    F1Score,
    CohenKappa,
    JaccardIndex,
    MatthewsCorrCoef,
)
from torchmetrics.regression import (
    MeanAbsoluteError,
    MeanSquaredError,
    R2Score,
)

from .metrics import ExampleRegressionMetric

# Define metrics
METRICS = {
    "accuracy": Accuracy,
    "precision": Precision,
    "average_precision": AveragePrecision,
    "recall": Recall,
    "f1_score": F1Score,
    "cohen_kappa": CohenKappa,
    "jaccard": JaccardIndex,
    "mcc": MatthewsCorrCoef,
    "auroc": AUROC,
    "f1": F1Score,
    "f1_macro": F1Score,
    "f1_weighted": F1Score,
    "confusion_matrix": ConfusionMatrix,
    "mae": MeanAbsoluteError,
    "mse": MeanSquaredError,
    "r2": R2Score,
    "example": ExampleRegressionMetric,
}

from .base import AbstractEvaluator  # noqa: E402
from .evaluator import TBEvaluator  # noqa: E402

__all__ = [
    "METRICS",
    "AbstractEvaluator",
    "TBEvaluator",
]
