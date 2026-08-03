"""TopoBench-owned evaluator contracts and lifecycle."""

from .base import AbstractEvaluator, EvaluatorBackend
from .backends import (
    ExactRankingMemoryError,
    ExactRankingMemoryEstimate,
    MetricBackend,
    UndefinedMetricError,
    estimate_exact_ranking_memory,
)
from .evaluator import TBEvaluator
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
