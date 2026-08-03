"""Exact, online, audit, and prediction-view policy contracts."""

import pytest
import torch
from torchmetrics.classification import (
    BinaryAUROC,
    BinaryAveragePrecision,
    MulticlassAUROC,
)

from topobench.evaluator import EvaluationBatch, EvaluationContext
from topobench.evaluator.backends import (
    ExactRankingBackend,
    MetricPolicyBackend,
    PredictionViews,
)
from topobench.evaluator.registry import resolve_evaluation_policy


def _context(
    policy: str, *, split: str = "val", classes: int = 2
) -> EvaluationContext:
    return EvaluationContext(
        split=split,
        pass_kind="fit_epoch",
        policy=policy,
        task="classification",
        num_classes=classes,
    )


def _batch(classes: int = 2) -> EvaluationBatch:
    outputs = (
        torch.tensor([[2.0, -0.5], [-1.0, 2.0], [0.2, 1.0], [1.5, 0.0]])
        if classes == 2
        else torch.tensor(
            [
                [2.0, 0.0, -1.0],
                [0.0, 2.0, 0.1],
                [0.1, 0.2, 2.0],
                [1.0, 0.9, 0.0],
            ]
        )
    )
    targets = (
        torch.tensor([0, 1, 1, 0])
        if classes == 2
        else torch.tensor([0, 1, 2, 1])
    )
    return EvaluationBatch(outputs=outputs, targets=targets, num_examples=4)


@pytest.mark.parametrize(
    ("split", "requested", "expected"),
    [
        ("train", None, "online"),
        ("val", None, "exact"),
        ("test", None, "exact"),
        ("train", "audit", "audit"),
        ("val", "online", "online"),
    ],
)
def test_split_default_policy_resolution(split, requested, expected):
    assert resolve_evaluation_policy(split, requested) == expected


def test_binary_audit_constructs_one_exact_group_and_thresholded_online_modules():
    backend = MetricPolicyBackend(
        task="classification",
        num_classes=2,
        metrics=["auroc", "auprc", "somers_d"],
    )
    backend.begin(_context("audit"))
    assert isinstance(backend.exact_ranking_backend, ExactRankingBackend)
    assert set(backend.exact_ranking_backend.metrics) == {"auroc", "auprc"}
    assert isinstance(
        backend.online_ranking_backend.metrics["auroc"], BinaryAUROC
    )
    assert isinstance(
        backend.online_ranking_backend.metrics["auprc"], BinaryAveragePrecision
    )
    assert "somers_d" not in backend.online_ranking_backend.metrics


def test_online_ranking_uses_one_configured_512_threshold_grid():
    backend = MetricPolicyBackend(
        task="classification",
        num_classes=2,
        metrics=["auroc", "auprc", "somers_d"],
        ranking_thresholds=512,
    )
    backend.begin(_context("online", split="train"))
    grid = backend.online_ranking_backend.threshold_grid
    assert grid.shape == (512,)
    assert torch.equal(grid, torch.linspace(0, 1, 512, device=grid.device))
    for metric in backend.online_ranking_backend.metrics.values():
        assert torch.equal(metric.thresholds, grid)


def test_exact_binary_metrics_retain_one_shared_positive_score_target_buffer():
    backend = MetricPolicyBackend(
        task="classification",
        num_classes=2,
        metrics=["auroc", "auprc", "somers_d"],
    )
    backend.begin(_context("exact"))
    backend.update(_batch())
    exact = backend.exact_ranking_backend
    assert len(exact.score_chunks) == len(exact.target_chunks) == 1
    assert exact.score_chunks[0].shape == (4,)
    assert exact.target_chunks[0].shape == (4,)
    assert set(exact.metrics) == {"auroc", "auprc"}


def test_exact_multiclass_auroc_retains_one_probability_target_buffer():
    backend = MetricPolicyBackend(
        task="classification", num_classes=3, metrics=["auroc"]
    )
    backend.begin(_context("exact", classes=3))
    backend.update(_batch(classes=3))
    exact = backend.exact_ranking_backend
    assert exact.score_chunks[0].shape == (4, 3)
    assert exact.target_chunks[0].shape == (4,)


def test_no_stateful_exact_torchmetrics_ranking_module_is_reachable():
    backend = MetricPolicyBackend(
        task="classification",
        num_classes=2,
        metrics=["auroc", "auprc", "somers_d"],
    )
    backend.begin(_context("exact"))
    backend.update(_batch())
    reachable = backend.reachable_objects()
    assert not any(
        isinstance(
            value, (BinaryAUROC, BinaryAveragePrecision, MulticlassAUROC)
        )
        for value in reachable
    )


def test_exact_snapshot_records_exact_status_without_threshold_grid():
    backend = MetricPolicyBackend(
        task="classification", num_classes=2, metrics=["auroc", "auprc"]
    )
    backend.begin(_context("exact"))
    backend.update(_batch())
    snapshot = backend.compute()
    assert snapshot.status == {"auroc": "exact", "auprc": "exact"}
    assert snapshot.provenance["thresholds"] == {"auroc": None, "auprc": None}


def test_binary_audit_keys_and_derived_somers_d_are_exactly_expanded():
    backend = MetricPolicyBackend(
        task="classification",
        num_classes=2,
        metrics=["auroc", "auprc", "somers_d"],
    )
    backend.begin(_context("audit"))
    backend.update(_batch())
    snapshot = backend.compute()
    assert tuple(snapshot) == (
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
    assert float(snapshot["somers_d"]) == pytest.approx(
        2 * float(snapshot["auroc"]) - 1
    )
    assert float(snapshot["somers_d_online"]) == pytest.approx(
        2 * float(snapshot["auroc_online"]) - 1
    )
    for metric in ("auroc", "auprc", "somers_d"):
        assert float(snapshot[f"{metric}_online_abs_error"]) == pytest.approx(
            abs(float(snapshot[metric]) - float(snapshot[f"{metric}_online"]))
        )


def test_multiclass_audit_expands_only_auroc():
    backend = MetricPolicyBackend(
        task="classification",
        num_classes=3,
        metrics=["accuracy", "f1", "auroc"],
    )
    backend.begin(_context("audit", classes=3))
    backend.update(_batch(classes=3))
    assert tuple(backend.compute()) == (
        "accuracy",
        "f1",
        "auroc",
        "auroc_online",
        "auroc_online_abs_error",
    )


class CountingPredictionViews(PredictionViews):
    probability_calls = 0
    positive_probability_calls = 0
    hard_class_calls = 0

    @property
    def probabilities(self):
        type(self).probability_calls += 1
        return super().probabilities

    @property
    def positive_probabilities(self):
        type(self).positive_probability_calls += 1
        return super().positive_probabilities

    @property
    def hard_classes(self):
        type(self).hard_class_calls += 1
        return super().hard_classes


def _reset_view_counts():
    CountingPredictionViews.probability_calls = 0
    CountingPredictionViews.positive_probability_calls = 0
    CountingPredictionViews.hard_class_calls = 0


def test_required_prediction_views_are_derived_at_most_once_per_update():
    _reset_view_counts()
    backend = MetricPolicyBackend(
        task="classification",
        num_classes=2,
        metrics=[
            "accuracy",
            "precision",
            "recall",
            "f1",
            "auroc",
            "auprc",
            "somers_d",
        ],
        prediction_views_factory=CountingPredictionViews,
    )
    backend.begin(_context("audit"))
    backend.update(_batch())
    assert CountingPredictionViews.probability_calls == 1
    assert CountingPredictionViews.positive_probability_calls == 1
    assert CountingPredictionViews.hard_class_calls == 1


def test_softmax_is_not_computed_without_probability_metric():
    _reset_view_counts()
    backend = MetricPolicyBackend(
        task="classification",
        num_classes=2,
        metrics=["accuracy", "precision"],
        prediction_views_factory=CountingPredictionViews,
    )
    backend.begin(_context("online", split="train"))
    backend.update(_batch())
    assert CountingPredictionViews.probability_calls == 0
    assert CountingPredictionViews.positive_probability_calls == 0
    assert CountingPredictionViews.hard_class_calls == 1


def test_policy_cannot_change_during_an_active_context():
    backend = MetricPolicyBackend(
        task="classification", num_classes=2, metrics=["auroc"]
    )
    backend.begin(_context("online", split="train"))
    with pytest.raises(RuntimeError, match="active.*policy"):
        backend.begin(_context("exact"))
    assert backend.policy == "online"
