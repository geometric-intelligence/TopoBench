"""TopoBench-owned evaluator contracts and lifecycle."""

from .backends import (
    ExactRankingMemoryError,
    ExactRankingMemoryEstimate,
    MetricBackend,
    UndefinedMetricError,
    estimate_exact_ranking_memory,
)
from .base import AbstractEvaluator, EvaluatorBackend
from .evaluator import TBEvaluator
from .prediction import PredictionIdentity, PredictionPayload
from .registry import MetricSpec
from .types import (
    EvaluationBatch,
    EvaluationContext,
    EvaluationPassKind,
    EvaluationPolicy,
    EvaluationResult,
    EvaluationSplit,
    EvaluationTask,
    MetricScalar,
    MetricStatus,
)

__all__ = [
    "AbstractEvaluator",
    "EvaluationBatch",
    "EvaluationContext",
    "EvaluationPassKind",
    "EvaluationPolicy",
    "EvaluationResult",
    "EvaluationSplit",
    "EvaluationTask",
    "PredictionIdentity",
    "PredictionPayload",
    "EvaluatorBackend",
    "MetricScalar",
    "MetricStatus",
    "ExactRankingMemoryError",
    "ExactRankingMemoryEstimate",
    "MetricBackend",
    "MetricSpec",
    "TBEvaluator",
    "UndefinedMetricError",
    "estimate_exact_ranking_memory",
]
