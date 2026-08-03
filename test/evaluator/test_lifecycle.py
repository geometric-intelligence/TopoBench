"""Failure-safe evaluator lifecycle tests independent of TorchMetrics."""

from collections import OrderedDict
from collections.abc import Mapping

import pytest
import torch

from topobench.evaluator import EvaluationBatch, EvaluationContext, TBEvaluator


class FakeBackend:
    """Small stateful scalar backend used only to exercise lifecycle ordering."""

    def __init__(
        self,
        value: float = 1.0,
        *,
        fail_update_at: int | None = None,
        reset_failures: int = 0,
        output: (
            float | torch.Tensor | Mapping[str, float | torch.Tensor] | None
        ) = None,
    ) -> None:
        self.value = value
        self.fail_update_at = fail_update_at
        self.reset_failures = reset_failures
        self.output = output
        self.context: EvaluationContext | None = None
        self.update_calls = 0
        self.reset_calls = 0
        self.active_examples = 0
        self.last_batch: EvaluationBatch | None = None

    def begin(self, context: EvaluationContext) -> None:
        self.context = context

    def update(self, batch: EvaluationBatch) -> None:
        self.update_calls += 1
        self.active_examples += batch.num_examples
        self.last_batch = batch
        if self.update_calls == self.fail_update_at:
            raise RuntimeError("backend update failed")

    def compute(
        self,
    ) -> float | torch.Tensor | Mapping[str, float | torch.Tensor]:
        if self.output is not None:
            return self.output
        return self.value

    def reset(self) -> None:
        self.reset_calls += 1
        if self.reset_failures:
            self.reset_failures -= 1
            raise RuntimeError("backend reset failed")
        self.context = None
        self.active_examples = 0


def _context(**overrides: object) -> EvaluationContext:
    values = {
        "split": "val",
        "pass_kind": "fit_epoch",
        "policy": "exact",
        "task": "classification",
        "num_classes": 2,
        "expected_num_examples": None,
    }
    values.update(overrides)
    return EvaluationContext(**values)


def _batch(
    size: int, *, context: EvaluationContext | None = None
) -> EvaluationBatch:
    return EvaluationBatch(
        outputs=torch.randn(size, 2),
        targets=torch.arange(size, dtype=torch.long) % 2,
        num_examples=size,
        context=context,
    )


def _evaluator(
    backends: Mapping[str, FakeBackend] | None = None,
) -> TBEvaluator:
    configured = backends or OrderedDict([("score", FakeBackend())])
    return TBEvaluator(
        task="classification",
        num_classes=2,
        metrics=list(configured),
        backends=configured,
    )


def test_initial_state_is_idle() -> None:
    evaluator = _evaluator()

    assert evaluator.state == "idle"
    assert evaluator.num_examples == 0


def test_begin_update_snapshot_finalize_returns_to_idle() -> None:
    backend = FakeBackend(value=0.75)
    evaluator = _evaluator(OrderedDict([("score", backend)]))
    context = _context(expected_num_examples=3)

    evaluator.begin(context)
    evaluator.update(_batch(3))
    snapshot = evaluator.snapshot()
    result = evaluator.finalize()

    assert snapshot.metrics == {"score": 0.75}
    assert snapshot.num_examples == 3
    assert result.metrics == {"score": 0.75}
    assert result.num_examples == 3
    assert evaluator.state == "idle"
    assert evaluator.num_examples == 0
    assert backend.reset_calls == 1


def test_begin_update_abort_returns_to_idle() -> None:
    backend = FakeBackend()
    evaluator = _evaluator(OrderedDict([("score", backend)]))
    evaluator.begin(_context())
    evaluator.update(_batch(2))

    evaluator.abort()

    assert evaluator.state == "idle"
    assert evaluator.num_examples == 0
    assert backend.reset_calls == 1


@pytest.mark.parametrize("batch_sizes", [[1], [2, 5, 3], [4, 4, 1]])
def test_uneven_batches_accumulate_exact_num_examples(
    batch_sizes: list[int],
) -> None:
    evaluator = _evaluator()
    evaluator.begin(_context(expected_num_examples=sum(batch_sizes)))

    for batch_size in batch_sizes:
        evaluator.update(_batch(batch_size))

    assert evaluator.finalize().num_examples == sum(batch_sizes)


def test_snapshot_neither_resets_nor_increments_count() -> None:
    backend = FakeBackend()
    evaluator = _evaluator(OrderedDict([("score", backend)]))
    evaluator.begin(_context())
    evaluator.update(_batch(2))

    first = evaluator.snapshot()
    second = evaluator.snapshot()

    assert first.num_examples == second.num_examples == 2
    assert evaluator.num_examples == 2
    assert backend.reset_calls == 0


def test_batch_is_counted_only_after_every_backend_update_succeeds() -> None:
    accepted = FakeBackend()
    failed = FakeBackend(fail_update_at=1)
    evaluator = _evaluator(
        OrderedDict([("accepted", accepted), ("failed", failed)])
    )
    evaluator.begin(_context())

    with pytest.raises(RuntimeError, match="backend update failed"):
        evaluator.update(_batch(4))

    assert evaluator.state == "failed"
    assert evaluator.num_examples == 0
    assert accepted.active_examples == 4
    evaluator.abort()
    assert accepted.active_examples == 0
    assert failed.active_examples == 0
    assert evaluator.state == "idle"


def test_abort_is_idempotent_only_after_recorded_active_failure() -> None:
    failed = FakeBackend(fail_update_at=1)
    evaluator = _evaluator(OrderedDict([("score", failed)]))

    with pytest.raises(RuntimeError, match="idle"):
        evaluator.abort()

    evaluator.begin(_context())
    with pytest.raises(RuntimeError, match="backend update failed"):
        evaluator.update(_batch(1))
    evaluator.abort()
    evaluator.abort()

    evaluator.begin(_context())
    evaluator.update(_batch(1))
    evaluator.finalize()
    with pytest.raises(RuntimeError, match="idle"):
        evaluator.abort()


@pytest.mark.parametrize("operation", ["update", "snapshot", "finalize"])
def test_active_operations_fail_while_idle(operation: str) -> None:
    evaluator = _evaluator()

    with pytest.raises(RuntimeError, match="active"):
        if operation == "update":
            evaluator.update(_batch(1))
        else:
            getattr(evaluator, operation)()


def test_begin_fails_while_active() -> None:
    evaluator = _evaluator()
    evaluator.begin(_context())

    with pytest.raises(RuntimeError, match="idle"):
        evaluator.begin(_context(split="test"))


def test_empty_finalize_fails() -> None:
    evaluator = _evaluator()
    evaluator.begin(_context())

    with pytest.raises(RuntimeError, match="at least one"):
        evaluator.finalize()

    evaluator.abort()


def test_mixed_context_batch_fails_before_backend_update() -> None:
    backend = FakeBackend()
    evaluator = _evaluator(OrderedDict([("score", backend)]))
    evaluator.begin(_context(split="val"))

    with pytest.raises(ValueError, match="active EvaluationContext"):
        evaluator.update(_batch(2, context=_context(split="test")))

    assert backend.update_calls == 0
    assert evaluator.num_examples == 0


def test_context_must_match_evaluator_construction() -> None:
    evaluator = _evaluator()

    with pytest.raises(ValueError, match="construction"):
        evaluator.begin(_context(task="regression", num_classes=1))


def test_expected_count_mismatch_fails_before_result_is_published() -> None:
    backend = FakeBackend()
    evaluator = _evaluator(OrderedDict([("score", backend)]))
    evaluator.begin(_context(expected_num_examples=3))
    evaluator.update(_batch(2))

    with pytest.raises(ValueError, match="expected 3.*observed 2"):
        evaluator.finalize()

    assert evaluator.state == "idle"
    assert backend.reset_calls == 1
    evaluator.abort()


def test_finalize_clears_backend_before_downstream_consumer_can_fail() -> None:
    backend = FakeBackend()
    evaluator = _evaluator(OrderedDict([("score", backend)]))
    evaluator.begin(_context(expected_num_examples=2))
    evaluator.update(_batch(2))

    result = evaluator.finalize()
    assert backend.active_examples == 0
    assert backend.reset_calls == 1

    with pytest.raises(RuntimeError, match="consumer failed"):
        raise RuntimeError(f"consumer failed after {result.num_examples}")

    assert evaluator.state == "idle"
    assert backend.active_examples == 0


def test_result_order_follows_configured_metric_order() -> None:
    backends = OrderedDict(
        [
            ("recall", FakeBackend(0.2)),
            ("accuracy", FakeBackend(0.9)),
            ("f1", FakeBackend(0.3)),
        ]
    )
    evaluator = _evaluator(backends)
    evaluator.begin(_context())
    evaluator.update(_batch(2))

    result = evaluator.finalize()

    assert tuple(result.metrics) == ("recall", "accuracy", "f1")


@pytest.mark.parametrize("operation", ["finalize", "abort"])
def test_reset_failure_keeps_context_failed_until_retry_abort_succeeds(
    operation: str,
) -> None:
    backend = FakeBackend(reset_failures=1)
    evaluator = _evaluator(OrderedDict([("score", backend)]))
    first_context = _context(split="val")
    evaluator.begin(first_context)
    evaluator.update(_batch(2))

    with pytest.raises(RuntimeError, match="backend reset failed"):
        getattr(evaluator, operation)()

    assert evaluator.state == "failed"
    assert evaluator.context == first_context
    assert evaluator.num_examples == 2
    assert backend.active_examples == 2
    with pytest.raises(RuntimeError, match="idle"):
        evaluator.begin(_context(split="test"))

    evaluator.abort()

    assert evaluator.state == "idle"
    assert evaluator.context is None
    assert evaluator.num_examples == 0
    assert backend.active_examples == 0

    evaluator.begin(_context(split="test"))
    evaluator.update(_batch(1))
    result = evaluator.finalize()
    assert result.num_examples == 1


def test_snapshot_tensor_mutation_cannot_affect_backend_or_later_result() -> (
    None
):
    backend_value = torch.tensor(0.75, requires_grad=True)
    backend = FakeBackend(output=backend_value)
    evaluator = _evaluator(OrderedDict([("score", backend)]))
    evaluator.begin(_context())
    evaluator.update(_batch(2))

    first = evaluator.snapshot()
    first.metrics["score"].fill_(0.0)
    second = evaluator.snapshot()

    assert backend_value.item() == pytest.approx(0.75)
    assert second.metrics["score"].item() == pytest.approx(0.75)
    assert first.metrics["score"].data_ptr() != backend_value.data_ptr()
    assert (
        second.metrics["score"].data_ptr() != first.metrics["score"].data_ptr()
    )
    evaluator.abort()


def test_backend_receives_exact_typed_batch_references_and_canonical_id() -> (
    None
):
    """Lifecycle routing does not clone tensors or reinterpret sequence IDs."""
    backend = FakeBackend()
    evaluator = _evaluator(OrderedDict([("score", backend)]))
    context = _context(split="train", policy="online")
    outputs = torch.randn(2, 2, requires_grad=True)
    targets = torch.tensor([0, 1])
    batch = EvaluationBatch(
        outputs=outputs,
        targets=targets,
        num_examples=2,
        context=context,
        sequence_id=("partition", 17),
    )
    evaluator.begin(context)

    evaluator.update(batch)

    assert backend.last_batch is batch
    assert backend.last_batch.outputs is outputs
    assert backend.last_batch.targets is targets
    assert backend.last_batch.sequence_id == ("partition", 17)
    assert evaluator.snapshot().num_examples == 2
    evaluator.abort()
