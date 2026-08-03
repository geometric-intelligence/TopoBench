"""Owned-state, shared-buffer, and exact-memory guard contracts."""

import pytest
import torch
from torchmetrics.classification import (
    BinaryAUROC,
    BinaryAveragePrecision,
    MulticlassAUROC,
)

from topobench.evaluator import EvaluationBatch, EvaluationContext, TBEvaluator
from topobench.evaluator.backends import (
    ExactRankingMemoryError,
    MetricPolicyBackend,
    estimate_exact_ranking_memory,
    owned_tensor_bytes,
)


def _context(
    policy: str,
    *,
    classes: int = 2,
    expected: int | None = None,
    split: str = "val",
) -> EvaluationContext:
    return EvaluationContext(
        split=split,
        pass_kind="fit_epoch",
        policy=policy,
        task="classification",
        num_classes=classes,
        expected_num_examples=expected,
    )


def _classification_batch(count: int, classes: int = 2) -> EvaluationBatch:
    generator = torch.Generator().manual_seed(count * 17 + classes)
    return EvaluationBatch(
        outputs=torch.randn(count, classes, generator=generator),
        targets=torch.arange(count, dtype=torch.long) % classes,
        num_examples=count,
    )


def test_decomposable_state_bytes_do_not_grow_with_examples():
    backend = MetricPolicyBackend(
        task="classification",
        num_classes=3,
        metrics=["accuracy", "precision", "recall", "f1"],
    )
    backend.begin(_context("online", classes=3, split="train"))
    backend.update(_classification_batch(9, classes=3))
    first = owned_tensor_bytes(backend)
    backend.update(_classification_batch(900, classes=3))
    assert owned_tensor_bytes(backend) == first


def test_thresholded_online_ranking_state_is_bounded_in_population():
    backend = MetricPolicyBackend(
        task="classification",
        num_classes=2,
        metrics=["auroc", "auprc", "somers_d"],
        ranking_thresholds=512,
    )
    backend.begin(_context("online", split="train"))
    backend.update(_classification_batch(10))
    first = owned_tensor_bytes(backend.online_ranking_backend)
    backend.update(_classification_batch(10_000))
    second = owned_tensor_bytes(backend.online_ranking_backend)
    assert first == second
    assert second > 0


@pytest.mark.parametrize("classes", [2, 5])
def test_exact_retained_bytes_grow_by_recorded_layout(classes):
    backend = MetricPolicyBackend(
        task="classification", num_classes=classes, metrics=["auroc"]
    )
    backend.begin(_context("exact", classes=classes))
    backend.update(_classification_batch(3, classes))
    first = backend.exact_ranking_backend.retained_bytes
    backend.update(_classification_batch(7, classes))
    second = backend.exact_ranking_backend.retained_bytes
    estimate3 = estimate_exact_ranking_memory(
        num_examples=3, num_classes=classes
    )
    estimate10 = estimate_exact_ranking_memory(
        num_examples=10, num_classes=classes
    )
    assert first == estimate3.retained_bytes
    assert second == estimate10.retained_bytes
    assert second > first


def test_binary_auroc_auprc_somers_share_exact_observations():
    backend = MetricPolicyBackend(
        task="classification",
        num_classes=2,
        metrics=["auroc", "auprc", "somers_d"],
    )
    backend.begin(_context("exact"))
    backend.update(_classification_batch(11))
    exact = backend.exact_ranking_backend
    assert len(exact.score_chunks) == len(exact.target_chunks) == 1
    assert exact.retained_bytes == 11 * (
        torch.tensor(0.0).element_size() + torch.tensor(0).element_size()
    )
    assert exact.binary_state_shared is True


def test_every_tensor_reachable_from_exact_backend_is_cpu_and_accounted():
    backend = MetricPolicyBackend(
        task="classification", num_classes=3, metrics=["auroc"]
    )
    backend.begin(_context("exact", classes=3))
    backend.update(_classification_batch(13, classes=3))
    exact = backend.exact_ranking_backend
    tensors = [
        value
        for value in exact.reachable_objects()
        if isinstance(value, torch.Tensor)
    ]
    assert tensors
    assert all(tensor.device.type == "cpu" for tensor in tensors)
    unique_bytes = {}
    for tensor in tensors:
        storage = tensor.untyped_storage()
        unique_bytes[(storage.data_ptr(), storage.nbytes())] = storage.nbytes()
    assert exact.retained_bytes == sum(unique_bytes.values())


def test_no_stateful_exact_ranking_metric_holds_hidden_observations():
    backend = MetricPolicyBackend(
        task="classification",
        num_classes=2,
        metrics=["auroc", "auprc", "somers_d"],
    )
    backend.begin(_context("exact"))
    backend.update(_classification_batch(10))
    assert not any(
        isinstance(
            value, (BinaryAUROC, BinaryAveragePrecision, MulticlassAUROC)
        )
        for value in backend.reachable_objects()
    )


def test_exact_snapshot_does_not_duplicate_retained_storage():
    backend = MetricPolicyBackend(
        task="classification", num_classes=2, metrics=["auroc", "auprc"]
    )
    backend.begin(_context("exact"))
    backend.update(_classification_batch(20))
    exact = backend.exact_ranking_backend
    pointers_before = tuple(
        tensor.data_ptr()
        for tensor in (*exact.score_chunks, *exact.target_chunks)
    )
    bytes_before = exact.retained_bytes
    snapshot = backend.compute()
    assert set(snapshot) == {"auroc", "auprc"}
    assert (
        tuple(
            tensor.data_ptr()
            for tensor in (*exact.score_chunks, *exact.target_chunks)
        )
        == pointers_before
    )
    assert exact.retained_bytes == bytes_before


def test_reset_releases_all_exact_chunk_references():
    backend = MetricPolicyBackend(
        task="classification", num_classes=2, metrics=["auroc"]
    )
    backend.begin(_context("exact"))
    backend.update(_classification_batch(8))
    assert backend.exact_ranking_backend.retained_bytes > 0
    backend.reset()
    assert backend.exact_ranking_backend is None
    assert backend.retained_bytes == 0


def test_finalize_and_abort_release_evaluator_exact_state():
    evaluator = TBEvaluator(
        "classification", num_classes=2, metrics=["auroc", "auprc"]
    )
    evaluator.begin(_context("exact", expected=8))
    evaluator.update(_classification_batch(8))
    backend = evaluator.metric_backend
    evaluator.finalize()
    assert backend.retained_bytes == 0

    evaluator.begin(_context("exact"))
    evaluator.update(_classification_batch(8))
    backend = evaluator.metric_backend
    evaluator.abort()
    assert backend.retained_bytes == 0


def test_public_estimator_covers_binary_and_multiclass_layouts():
    binary = estimate_exact_ranking_memory(num_examples=100, num_classes=2)
    multiclass = estimate_exact_ranking_memory(num_examples=100, num_classes=7)
    assert binary.layout == "binary_positive_scores"
    assert binary.binary_state_shared is True
    assert multiclass.layout == "multiclass_probabilities"
    assert multiclass.binary_state_shared is False
    assert binary.retained_bytes == 100 * (4 + 8)
    assert multiclass.retained_bytes == 100 * (7 * 4 + 8)
    assert binary.estimated_peak_bytes >= binary.retained_bytes
    assert multiclass.estimated_peak_bytes > binary.estimated_peak_bytes


def test_known_expected_count_fails_memory_guard_before_state_allocation():
    estimate = estimate_exact_ranking_memory(num_examples=1_000, num_classes=2)
    backend = MetricPolicyBackend(
        task="classification",
        num_classes=2,
        metrics=["auroc", "auprc"],
        max_exact_ranking_bytes=estimate.estimated_peak_bytes - 1,
    )
    with pytest.raises(ExactRankingMemoryError) as caught:
        backend.begin(_context("exact", expected=1_000))
    message = str(caught.value)
    assert "split=val" in message
    assert "observed_examples=0" in message
    assert "projected_examples=1000" in message
    assert f"projected_bytes={estimate.estimated_peak_bytes}" in message
    assert f"configured_limit={estimate.estimated_peak_bytes - 1}" in message
    assert "policy='online'" in message
    assert backend.retained_bytes == 0


def test_unknown_count_runtime_guard_rejects_before_offending_append():
    limit = estimate_exact_ranking_memory(
        num_examples=4, num_classes=3
    ).estimated_peak_bytes
    backend = MetricPolicyBackend(
        task="classification",
        num_classes=3,
        metrics=["auroc"],
        max_exact_ranking_bytes=limit,
    )
    backend.begin(_context("exact", classes=3, expected=None))
    backend.update(_classification_batch(3, classes=3))
    exact = backend.exact_ranking_backend
    pointers = tuple(
        tensor.data_ptr()
        for tensor in (*exact.score_chunks, *exact.target_chunks)
    )
    with pytest.raises(ExactRankingMemoryError) as caught:
        backend.update(_classification_batch(2, classes=3))
    message = str(caught.value)
    assert "observed_examples=3" in message
    assert "projected_examples=5" in message
    assert f"configured_limit={limit}" in message
    assert exact.num_examples == 3
    assert (
        tuple(
            tensor.data_ptr()
            for tensor in (*exact.score_chunks, *exact.target_chunks)
        )
        == pointers
    )


@pytest.mark.parametrize(
    ("retry_policy", "retry_expected"),
    [("exact", 4), ("online", 100)],
)
def test_failed_begin_rolls_back_all_state_and_allows_retry(
    retry_policy, retry_expected
):
    limit = estimate_exact_ranking_memory(
        num_examples=4, num_classes=2
    ).estimated_peak_bytes
    backend = MetricPolicyBackend(
        task="classification",
        num_classes=2,
        metrics=["accuracy", "auroc"],
        max_exact_ranking_bytes=limit,
    )

    with pytest.raises(ExactRankingMemoryError):
        backend.begin(_context("exact", expected=100))

    assert backend.context is None
    assert backend.policy is None
    assert backend.exact_ranking_backend is None
    assert backend.online_ranking_backend is None
    assert backend.retained_bytes == 0

    backend.begin(
        _context(
            retry_policy,
            expected=retry_expected,
            split="train" if retry_policy == "online" else "val",
        )
    )
    backend.update(_classification_batch(retry_expected))
    snapshot = backend.compute()
    assert set(snapshot) == {"accuracy", "auroc"}
