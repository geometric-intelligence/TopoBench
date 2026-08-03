"""Constructor-injected custom metric extension contracts."""

from collections.abc import Mapping

import pytest
import torch

from topobench.evaluator.backends import BackendFactoryContext, MetricBackend
from topobench.evaluator.registry import MetricSpec, resolve_metric_specs


class MeanPredictionBackend:
    """A custom backend that has no TorchMetrics dependency."""

    def __init__(self):
        self.total = torch.tensor(0.0)
        self.count = torch.tensor(0, dtype=torch.long)

    def update(self, predictions: torch.Tensor, targets: torch.Tensor) -> None:
        del targets
        self.total += predictions.sum()
        self.count += predictions.numel()

    def compute(self) -> torch.Tensor:
        return self.total / self.count

    def reset(self) -> None:
        self.total.zero_()
        self.count.zero_()

    def to(self, device: torch.device | str) -> "MeanPredictionBackend":
        self.total = self.total.to(device)
        self.count = self.count.to(device)
        return self

    @property
    def retained_bytes(self) -> int:
        return (
            self.total.numel() * self.total.element_size()
            + self.count.numel() * self.count.element_size()
        )


def _custom_spec(
    name: str = "mean_prediction",
    *,
    online: bool = True,
) -> MetricSpec:
    def factory(context):
        return MeanPredictionBackend()

    return MetricSpec(
        name=name,
        tasks=frozenset({"regression"}),
        prediction_view="raw",
        backend_group=name,
        exact_factory=factory,
        online_factory=factory if online else None,
        scalar=True,
        higher_is_better=False,
        undefined_reasons=frozenset({"empty"}),
    )


def test_custom_backend_satisfies_runtime_protocol_without_torchmetrics():
    backend = MeanPredictionBackend()
    assert isinstance(backend, MetricBackend)
    assert "torchmetrics" not in MeanPredictionBackend.__module__

    backend.update(torch.tensor([[1.0], [3.0]]), torch.zeros(2, 1))
    assert backend.compute().item() == pytest.approx(2.0)
    assert backend.retained_bytes == 12
    backend.reset()
    assert backend.count.item() == 0


def test_custom_spec_is_constructor_local_and_does_not_mutate_builtins():
    from topobench.evaluator.registry import BUILTIN_METRIC_SPECS

    before = tuple(BUILTIN_METRIC_SPECS)
    resolved = resolve_metric_specs(
        ["mean_prediction"],
        task="regression",
        num_classes=1,
        policy="exact",
        custom_specs=[_custom_spec()],
    )
    assert tuple(spec.name for spec in resolved) == ("mean_prediction",)
    assert tuple(BUILTIN_METRIC_SPECS) == before
    with pytest.raises(ValueError, match="Unknown metric.*mean_prediction"):
        resolve_metric_specs(
            ["mean_prediction"],
            task="regression",
            num_classes=1,
            policy="exact",
        )


def test_custom_online_policy_requires_explicit_online_factory():
    with pytest.raises(ValueError, match="mean_prediction.*online"):
        resolve_metric_specs(
            ["mean_prediction"],
            task="regression",
            num_classes=1,
            policy="online",
            custom_specs=[_custom_spec(online=False)],
        )


def test_custom_audit_requires_distinct_exact_and_online_capability():
    with pytest.raises(ValueError, match="mean_prediction.*audit"):
        resolve_metric_specs(
            ["mean_prediction"],
            task="regression",
            num_classes=1,
            policy="audit",
            custom_specs=[_custom_spec(online=False)],
        )


def test_backend_factory_receives_immutable_topobench_context():
    context = BackendFactoryContext(
        task="regression",
        num_classes=1,
        policy="exact",
        device=torch.device("cpu"),
        ranking_thresholds=512,
        max_exact_ranking_bytes=1024,
        undefined_metric_policy="error",
    )
    assert context.task == "regression"
    with pytest.raises((AttributeError, TypeError)):
        context.policy = "online"


def test_metric_spec_rejects_mutable_task_and_reason_collections():
    with pytest.raises(TypeError, match="frozenset"):
        MetricSpec(
            name="bad",
            tasks={"regression"},
            prediction_view="raw",
            backend_group="bad",
            exact_factory=lambda context: MeanPredictionBackend(),
            online_factory=None,
            scalar=True,
            higher_is_better=False,
            undefined_reasons=frozenset(),
        )


def test_factory_protocol_does_not_require_mapping_or_registry_access():
    spec = _custom_spec()
    context = BackendFactoryContext(
        task="regression",
        num_classes=1,
        policy="exact",
        device=torch.device("cpu"),
        ranking_thresholds=512,
        max_exact_ranking_bytes=1024,
        undefined_metric_policy="error",
    )
    backend = spec.exact_factory(context)
    assert isinstance(backend, MetricBackend)
    assert not isinstance(backend, Mapping)
