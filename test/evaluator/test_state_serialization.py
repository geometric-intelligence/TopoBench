"""Checkpoint-safe evaluator metric-state round trips."""
import copy

from collections.abc import Mapping

import pytest
import torch

from topobench.evaluator import EvaluationBatch, EvaluationContext, TBEvaluator
from topobench.evaluator.registry import MetricSpec


def _context(*, task: str, num_classes: int, expected: int) -> EvaluationContext:
    return EvaluationContext(
        split="train",
        pass_kind="fit_epoch",
        policy="online",
        task=task,
        num_classes=num_classes,
        expected_num_examples=expected,
    )


def _batch(outputs: list[list[float]], targets: list[object]) -> EvaluationBatch:
    return EvaluationBatch(
        outputs=torch.tensor(outputs),
        targets=torch.tensor(targets),
        num_examples=len(targets),
    )


def _checkpoint_tensors(value: object) -> tuple[torch.Tensor, ...]:
    tensors: list[torch.Tensor] = []

    def visit(item: object) -> None:
        if isinstance(item, torch.Tensor):
            tensors.append(item)
        elif isinstance(item, Mapping):
            for nested in item.values():
                visit(nested)
        elif isinstance(item, (list, tuple)):
            for nested in item:
                visit(nested)

    visit(value)
    return tuple(tensors)


@pytest.mark.parametrize(
    ("task", "num_classes", "metrics", "batches"),
    [
        (
            "classification",
            3,
            ("accuracy", "precision", "recall", "f1"),
            (
                _batch(
                    [[8.0, 0.0, 0.0], [0.0, 7.0, 0.0], [0.0, 0.0, 6.0]],
                    [0, 1, 2],
                ),
                _batch(
                    [[0.0, 5.0, 0.0], [4.0, 0.0, 0.0], [0.0, 3.0, 0.0]],
                    [1, 2, 0],
                ),
            ),
        ),
        (
            "regression",
            1,
            ("mae", "mse", "rmse", "r2"),
            (
                _batch([[0.0], [2.0], [5.0]], [[0.0], [1.0], [4.0]]),
                _batch([[3.0], [8.0]], [[2.0], [4.0]]),
            ),
        ),
    ],
)
def test_mid_context_round_trip_preserves_all_builtin_scalar_state_and_count(
    task: str,
    num_classes: int,
    metrics: tuple[str, ...],
    batches: tuple[EvaluationBatch, EvaluationBatch],
) -> None:
    expected = sum(batch.num_examples for batch in batches)
    context = _context(task=task, num_classes=num_classes, expected=expected)

    uninterrupted = TBEvaluator(
        task=task,
        num_classes=num_classes,
        metrics=metrics,
    )
    uninterrupted.begin(context)
    for batch in batches:
        uninterrupted.update(batch)
    reference = uninterrupted.finalize()

    interrupted = TBEvaluator(
        task=task,
        num_classes=num_classes,
        metrics=metrics,
    )
    interrupted.begin(context)
    interrupted.update(batches[0])
    live_storage = {
        tensor.untyped_storage().data_ptr()
        for tensor in interrupted.metric_backend.fixed_state_tensors
        if tensor.untyped_storage().nbytes()
    }
    checkpoint = interrupted.state_dict()
    checkpoint_tensors = _checkpoint_tensors(checkpoint)

    assert checkpoint_tensors
    assert all(tensor.device.type == "cpu" for tensor in checkpoint_tensors)
    assert all(
        tensor.untyped_storage().data_ptr() not in live_storage
        for tensor in checkpoint_tensors
        if tensor.untyped_storage().nbytes()
    )

    resumed = TBEvaluator(
        task=task,
        num_classes=num_classes,
        metrics=metrics,
    )
    resumed.load_state_dict(checkpoint, strict=True)
    for tensor in checkpoint_tensors:
        tensor.zero_()
    resumed.update(batches[1])
    actual = resumed.finalize()

    assert actual.num_examples == reference.num_examples == expected
    assert tuple(actual.metrics) == metrics
    for name in metrics:
        torch.testing.assert_close(
            torch.as_tensor(actual.metrics[name]),
            torch.as_tensor(reference.metrics[name]),
            rtol=0,
            atol=0,
        )


def test_mid_context_round_trip_preserves_bounded_online_ranking_state() -> None:
    metrics = ("auroc", "auprc", "somers_d")
    context = _context(task="classification", num_classes=2, expected=4)
    first = _batch([[5.0, 0.0], [0.0, 5.0]], [0, 1])
    second = _batch([[0.0, 5.0], [0.0, 5.0]], [0, 1])

    uninterrupted = TBEvaluator(
        task="classification",
        num_classes=2,
        metrics=metrics,
    )
    uninterrupted.begin(context)
    uninterrupted.update(first)
    uninterrupted.update(second)
    reference = uninterrupted.finalize()

    interrupted = TBEvaluator(
        task="classification",
        num_classes=2,
        metrics=metrics,
    )
    interrupted.begin(context)
    interrupted.update(first)
    checkpoint = interrupted.state_dict()

    resumed = TBEvaluator(
        task="classification",
        num_classes=2,
        metrics=metrics,
    )
    resumed.load_state_dict(checkpoint, strict=True)
    resumed.update(second)
    actual = resumed.finalize()

    assert actual.num_examples == reference.num_examples == 4
    for name in metrics:
        torch.testing.assert_close(
            torch.as_tensor(actual.metrics[name]),
            torch.as_tensor(reference.metrics[name]),
            rtol=0,
            atol=0,
        )


class CheckpointableMeanPredictionBackend:
    """Minimal injected backend with an explicit strict checkpoint contract."""

    def __init__(self) -> None:
        self.total = torch.tensor(0.0)
        self.count = torch.tensor(0, dtype=torch.long)
        self.to_calls: list[torch.device] = []

    def update(self, predictions: torch.Tensor, targets: torch.Tensor) -> None:
        del targets
        self.total += predictions.sum()
        self.count += predictions.numel()

    def compute(self) -> torch.Tensor:
        return self.total / self.count

    def reset(self) -> None:
        self.total.zero_()
        self.count.zero_()

    def to(
        self, device: torch.device | str
    ) -> "CheckpointableMeanPredictionBackend":
        resolved = torch.device(device)
        self.to_calls.append(resolved)
        self.total = self.total.to(resolved)
        self.count = self.count.to(resolved)
        return self

    @property
    def retained_bytes(self) -> int:
        return sum(
            value.untyped_storage().nbytes() for value in (self.total, self.count)
        )

    def state_dict(self) -> dict[str, torch.Tensor]:
        return {
            "total": self.total.detach().cpu().clone(),
            "count": self.count.detach().cpu().clone(),
        }

    def load_state_dict(
        self, state_dict: Mapping[str, object], *, strict: bool
    ) -> None:
        if strict is not True or set(state_dict) != {"total", "count"}:
            raise ValueError("invalid custom metric state")
        total = state_dict["total"]
        count = state_dict["count"]
        if not isinstance(total, torch.Tensor) or not isinstance(count, torch.Tensor):
            raise TypeError("custom metric state values must be tensors")
        device = self.total.device
        self.total = total.detach().to(device).clone()
        self.count = count.detach().to(device).clone()


def _checkpointable_custom_spec(
    created: list[CheckpointableMeanPredictionBackend] | None = None,
) -> MetricSpec:
    def factory(context: object) -> CheckpointableMeanPredictionBackend:
        del context
        backend = CheckpointableMeanPredictionBackend()
        if created is not None:
            created.append(backend)
        return backend

    return MetricSpec(
        name="mean_prediction",
        tasks=frozenset({"regression"}),
        prediction_view="raw",
        backend_group="mean_prediction",
        exact_factory=factory,
        online_factory=factory,
        scalar=True,
        higher_is_better=False,
        undefined_reasons=frozenset({"empty_evaluation"}),
    )


def test_mid_context_round_trip_supports_explicit_custom_backend_state() -> None:
    context = _context(task="regression", num_classes=1, expected=4)
    spec = _checkpointable_custom_spec()
    first = _batch([[1.0], [3.0]], [[0.0], [0.0]])
    second = _batch([[5.0], [7.0]], [[0.0], [0.0]])

    interrupted = TBEvaluator(
        task="regression",
        num_classes=1,
        metrics=["mean_prediction"],
        custom_specs=[spec],
    )
    interrupted.begin(context)
    interrupted.update(first)
    checkpoint = interrupted.state_dict()

    resumed = TBEvaluator(
        task="regression",
        num_classes=1,
        metrics=["mean_prediction"],
        custom_specs=[spec],
    )
    resumed.load_state_dict(checkpoint, strict=True)
    for tensor in _checkpoint_tensors(checkpoint):
        tensor.zero_()
    resumed.update(second)
    result = resumed.finalize()

    assert result.num_examples == 4
    assert result.metrics["mean_prediction"] == pytest.approx(4.0)


def test_restored_auto_device_state_rebinds_on_first_live_batch() -> None:
    created: list[CheckpointableMeanPredictionBackend] = []
    spec = _checkpointable_custom_spec(created)
    context = _context(task="regression", num_classes=1, expected=4)
    first = _batch([[1.0], [3.0]], [[0.0], [0.0]])
    second = _batch([[5.0], [7.0]], [[0.0], [0.0]])

    source = TBEvaluator(
        task="regression",
        num_classes=1,
        metrics=["mean_prediction"],
        custom_specs=[spec],
    )
    source.begin(context)
    source.update(first)
    checkpoint = source.state_dict()

    resumed = TBEvaluator(
        task="regression",
        num_classes=1,
        metrics=["mean_prediction"],
        custom_specs=[spec],
    )
    resumed.load_state_dict(checkpoint, strict=True)
    restored_backend = created[-1]
    assert restored_backend.to_calls == []

    resumed.update(second)

    assert restored_backend.to_calls == [torch.device("cpu")]
    assert resumed.num_examples == 4


@pytest.mark.skipif(
    not torch.cuda.is_available(),
    reason="requires the mandatory CUDA evaluator job",
)
def test_restored_auto_device_state_migrates_cpu_checkpoint_to_cuda() -> None:
    metrics = ("accuracy", "auroc")
    context = _context(task="classification", num_classes=2, expected=4)
    first = _batch([[5.0, 0.0], [0.0, 5.0]], [0, 1])
    source = TBEvaluator(
        task="classification",
        num_classes=2,
        metrics=metrics,
    )
    source.begin(context)
    source.update(first)
    checkpoint = source.state_dict()

    resumed = TBEvaluator(
        task="classification",
        num_classes=2,
        metrics=metrics,
    )
    resumed.load_state_dict(checkpoint, strict=True)
    resumed.update(
        EvaluationBatch(
            outputs=torch.tensor(
                [[0.0, 5.0], [5.0, 0.0]],
                device="cuda",
            ),
            targets=torch.tensor([1, 0], device="cuda"),
            num_examples=2,
        )
    )

    assert resumed.metric_backend.device.type == "cuda"
    state_tensors = resumed.metric_backend.fixed_state_tensors
    ranking = resumed.metric_backend.online_ranking_backend
    assert ranking is not None
    state_tensors += ranking.state_tensors
    assert state_tensors
    assert all(tensor.device.type == "cuda" for tensor in state_tensors)
    assert resumed.finalize().num_examples == 4


def _classification_checkpoint() -> tuple[
    EvaluationContext,
    EvaluationBatch,
    dict[str, object],
]:
    context = _context(task="classification", num_classes=2, expected=2)
    batch = _batch([[5.0, 0.0], [0.0, 5.0]], [0, 1])
    evaluator = TBEvaluator(
        task="classification",
        num_classes=2,
        metrics=("accuracy",),
    )
    evaluator.begin(context)
    evaluator.update(batch)
    return context, batch, evaluator.state_dict()


def _tamper_checkpoint(
    checkpoint: dict[str, object],
    case: str,
) -> None:
    backend = checkpoint["backend"]
    assert isinstance(backend, dict)
    support = backend["support"]
    fixed = backend["fixed"]
    assert isinstance(support, dict)
    assert isinstance(fixed, dict)
    accuracy = fixed["accuracy"]
    assert isinstance(accuracy, dict)
    state = accuracy["state"]
    assert isinstance(state, dict)
    metric_state = state["metric_state"]
    assert isinstance(metric_state, dict)

    if case == "metric_shape":
        metric_state["tp"] = torch.tensor([1, 1], dtype=torch.long)
    elif case == "metric_negative":
        metric_state["tp"] = torch.tensor([-1], dtype=torch.long)
    elif case == "metric_nonfinite":
        metric_state["tp"] = torch.tensor([float("nan")])
    elif case == "metric_wrong_dtype":
        metric_state["tp"] = torch.tensor([1.0])
    elif case == "support_shape":
        support["target_sum"] = torch.tensor([0.0], dtype=torch.float64)
    elif case == "support_negative":
        support["class_counts"] = torch.tensor([-1, 3], dtype=torch.long)
    elif case == "support_nonfinite":
        support["target_sum"] = torch.tensor(
            float("nan"),
            dtype=torch.float64,
        )
    elif case == "support_wrong_dtype":
        support["target_sum"] = torch.tensor(0.0, dtype=torch.float32)
    else:
        raise AssertionError(f"unknown tamper case {case}")


def _assert_checkpoint_equal(left: object, right: object) -> None:
    if isinstance(left, torch.Tensor):
        assert isinstance(right, torch.Tensor)
        assert left.dtype == right.dtype
        assert left.device == right.device
        assert torch.equal(left, right)
        return
    if isinstance(left, Mapping):
        assert isinstance(right, Mapping)
        assert tuple(left) == tuple(right)
        for key in left:
            _assert_checkpoint_equal(left[key], right[key])
        return
    if isinstance(left, (list, tuple)):
        assert isinstance(right, type(left))
        assert len(left) == len(right)
        for left_item, right_item in zip(left, right, strict=True):
            _assert_checkpoint_equal(left_item, right_item)
        return
    assert left == right


@pytest.mark.parametrize(
    "case",
    (
        "metric_shape",
        "metric_negative",
        "metric_wrong_dtype",
        "support_shape",
        "support_negative",
        "support_nonfinite",
        "support_wrong_dtype",
    ),
)
def test_malformed_restore_is_rejected_atomically(case: str) -> None:
    context, batch, checkpoint = _classification_checkpoint()
    malformed = copy.deepcopy(checkpoint)
    _tamper_checkpoint(malformed, case)

    active = TBEvaluator(
        task="classification",
        num_classes=2,
        metrics=("accuracy",),
    )
    active.begin(context)
    active.update(batch)
    before = active.state_dict()

    with pytest.raises((TypeError, ValueError)):
        active.load_state_dict(malformed, strict=True)

    assert active.state == "active"
    _assert_checkpoint_equal(active.state_dict(), before)


def test_nonfinite_regression_metric_state_is_rejected_atomically() -> None:
    context = _context(task="regression", num_classes=1, expected=2)
    batch = _batch([[1.0], [3.0]], [[0.0], [0.0]])
    source = TBEvaluator(
        task="regression",
        num_classes=1,
        metrics=("mae",),
    )
    source.begin(context)
    source.update(batch)
    malformed = source.state_dict()
    metric = malformed["backend"]["fixed"]["mae"]["state"]["metric_state"]
    metric["sum_abs_error"] = torch.tensor([float("nan")])

    active = TBEvaluator(
        task="regression",
        num_classes=1,
        metrics=("mae",),
    )
    active.begin(context)
    active.update(batch)
    before = active.state_dict()

    with pytest.raises(ValueError, match="finite"):
        active.load_state_dict(malformed, strict=True)

    assert active.state == "active"
    _assert_checkpoint_equal(active.state_dict(), before)
