"""Evaluator device separation contracts."""

import pytest
import torch
from torchmetrics.classification import BinaryAUROC, BinaryAveragePrecision, MulticlassAUROC

from topobench.evaluator import EvaluationBatch, EvaluationContext
from topobench.evaluator.backends import MetricPolicyBackend, estimate_exact_ranking_memory


def _context(policy: str, classes: int, split: str = "val") -> EvaluationContext:
    return EvaluationContext(
        split=split,
        pass_kind="fit_epoch",
        policy=policy,
        task="classification",
        num_classes=classes,
    )


def _batch(device: torch.device, classes: int = 2, count: int = 12) -> EvaluationBatch:
    outputs = torch.randn(count, classes, device=device, requires_grad=True)
    targets = torch.arange(count, device=device, dtype=torch.long) % classes
    return EvaluationBatch(outputs=outputs, targets=targets, num_examples=count)


def test_fixed_and_online_state_follow_cpu_evaluation_device():
    backend = MetricPolicyBackend(
        task="classification",
        num_classes=2,
        metrics=["accuracy", "precision", "auroc", "auprc"],
        device=torch.device("cpu"),
    )
    backend.begin(_context("online", 2, split="train"))
    backend.update(_batch(torch.device("cpu")))
    state = backend.fixed_state_tensors + backend.online_ranking_backend.state_tensors
    assert state
    assert all(tensor.device.type == "cpu" for tensor in state)
    snapshot = backend.compute()
    assert snapshot.provenance["device_policy"] == {
        "evaluation_device": "cpu",
        "exact_buffer_device": None,
        "cuda_qualified": False,
    }


@pytest.mark.parametrize("classes", [2, 4])
def test_exact_chunks_are_detached_cpu_storage(classes):
    backend = MetricPolicyBackend(
        task="classification",
        num_classes=classes,
        metrics=["auroc"] if classes > 2 else ["auroc", "auprc"],
        device=torch.device("cpu"),
    )
    backend.begin(_context("exact", classes))
    backend.update(_batch(torch.device("cpu"), classes))
    exact = backend.exact_ranking_backend
    chunks = (*exact.score_chunks, *exact.target_chunks)
    assert chunks
    assert all(tensor.device.type == "cpu" for tensor in chunks)
    assert all(tensor.grad_fn is None and not tensor.requires_grad for tensor in chunks)
    snapshot = backend.compute()
    assert snapshot.provenance["device_policy"]["exact_buffer_device"] == "cpu"


def test_exact_binary_device_metadata_records_shared_observation_state():
    backend = MetricPolicyBackend(
        task="classification",
        num_classes=2,
        metrics=["auroc", "auprc", "somers_d"],
    )
    backend.begin(_context("exact", 2))
    backend.update(_batch(torch.device("cpu"), 2))
    snapshot = backend.compute()
    memory = snapshot.provenance["exact_ranking_memory"]
    assert memory["buffer_device"] == "cpu"
    assert memory["binary_state_shared"] is True
    assert memory["score_dtype"] == "torch.float32"
    assert memory["target_dtype"] == "torch.int64"
    assert memory["class_count"] == 2


@pytest.mark.skipif(not torch.cuda.is_available(), reason="mandatory CUDA runner only")
def test_cuda_audit_keeps_exact_observations_off_accelerator_and_shares_binary_state():
    device = torch.device("cuda")
    backend = MetricPolicyBackend(
        task="classification",
        num_classes=2,
        metrics=["accuracy", "auroc", "auprc", "somers_d"],
        device=device,
    )
    backend.begin(_context("audit", 2))
    backend.update(_batch(device, 2, 128))
    exact = backend.exact_ranking_backend
    assert all(chunk.device.type == "cpu" for chunk in (*exact.score_chunks, *exact.target_chunks))
    assert exact.binary_state_shared is True
    assert not any(
        isinstance(value, (BinaryAUROC, BinaryAveragePrecision, MulticlassAUROC))
        for value in exact.reachable_objects()
    )
    assert all(tensor.device.type == "cuda" for tensor in backend.fixed_state_tensors)
    assert all(tensor.device.type == "cuda" for tensor in backend.online_ranking_backend.state_tensors)
    snapshot = backend.compute()
    assert snapshot.provenance["device_policy"]["evaluation_device"] == "cuda:0"
    assert snapshot.provenance["device_policy"]["exact_buffer_device"] == "cpu"
    assert snapshot.provenance["device_policy"]["cuda_qualified"] is True


@pytest.mark.skipif(not torch.cuda.is_available(), reason="mandatory CUDA runner only")
def test_guarded_cuda_multiclass_exact_finalization_respects_declared_cpu_estimate():
    device = torch.device("cuda")
    count = 2_000
    classes = 5
    estimate = estimate_exact_ranking_memory(num_examples=count, num_classes=classes)
    backend = MetricPolicyBackend(
        task="classification",
        num_classes=classes,
        metrics=["auroc"],
        max_exact_ranking_bytes=estimate.estimated_peak_bytes,
        device=device,
    )
    backend.begin(
        EvaluationContext(
            split="test",
            pass_kind="selected_checkpoint",
            policy="exact",
            task="classification",
            num_classes=classes,
            expected_num_examples=count,
        )
    )
    torch.cuda.reset_peak_memory_stats(device)
    backend.update(_batch(device, classes, count))
    assert backend.exact_ranking_backend.retained_bytes == estimate.retained_bytes
    assert all(
        chunk.device.type == "cpu"
        for chunk in (
            *backend.exact_ranking_backend.score_chunks,
            *backend.exact_ranking_backend.target_chunks,
        )
    )
    result = backend.compute()
    assert torch.isfinite(result["auroc"])
    assert result.provenance["exact_ranking_memory"]["estimated_peak_bytes"] == estimate.estimated_peak_bytes
