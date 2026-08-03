"""Representative-data contracts for the isolated pre-training probe."""

from __future__ import annotations

from collections.abc import Mapping
from pathlib import Path
from types import SimpleNamespace
from typing import Any

import numpy as np
import pytest
import torch
from lightning import LightningDataModule, LightningModule
from torch import Tensor, nn
from torch.optim import SGD
from torch.optim.lr_scheduler import StepLR
from torch_geometric.data import Data, HeteroData
from torch_geometric.loader import DataLoader as GeometricDataLoader

from test.data.dataload.test_disk_graph_datamodule import (
    exhaustive_fanout,
    materialized_heterogeneous_reference,
    task8_stores,
)
from test.data.stores.test_typed_graph_store import QualifiedStoreFixture
from topobench.data.stores.typed_graph_store import TypedGraphStore
from topobench.dataloader.disk_graph import (
    DiskGraphDataModule,
    HeterogeneousClusterStrategy,
    HeterogeneousNeighborStrategy,
    HomogeneousClusterStrategy,
)
from topobench.evaluator.base import AbstractEvaluator
from topobench.evaluator.types import (
    EvaluationBatch,
    EvaluationContext,
    EvaluationResult,
)
from topobench.preflight import PreflightResult, PreflightRunner
from topobench.transforms.fittable import FitStateError, FitStatus
from topobench.transforms.incremental_pca import IncrementalPCATransform

from .test_static_validation import (
    _compose_tiny_config,
    _qualified_pipeline_output,
)


class _ProbeEvaluator(AbstractEvaluator):
    """Small real evaluator lifecycle used at the public model boundary."""

    def __init__(self, observations: dict[str, Any]) -> None:
        self.observations = observations
        self.context: EvaluationContext | None = None
        self.num_examples = 0

    def begin(self, context: EvaluationContext) -> None:
        if self.context is not None:
            raise RuntimeError("evaluation is already active")
        self.context = context
        self.num_examples = 0
        self.observations["events"].append(f"evaluator.begin:{context.split}")

    def update(self, batch: EvaluationBatch) -> None:
        if self.context is None or batch.context != self.context:
            raise RuntimeError("evaluation update has no matching context")
        self.num_examples += batch.num_examples
        self.observations["events"].append(
            f"evaluator.update:{self.context.split}"
        )

    def snapshot(self) -> EvaluationResult:
        if self.context is None:
            raise RuntimeError("evaluation snapshot has no active context")
        self.observations["events"].append(
            f"evaluator.snapshot:{self.context.split}"
        )
        return EvaluationResult(
            metrics={"accuracy": torch.tensor(1.0)},
            num_examples=self.num_examples,
            context=self.context,
            status={"accuracy": "exact"},
        )

    def finalize(self) -> EvaluationResult:
        result = self.snapshot()
        self.abort()
        return result

    def abort(self) -> None:
        phase = "idle" if self.context is None else self.context.split
        self.observations["events"].append(f"evaluator.abort:{phase}")
        self.context = None
        self.num_examples = 0


class _BiasLogits(nn.Module):
    """Deterministic trainable logits without constructor-time RNG use."""

    def __init__(
        self,
        *,
        num_classes: int,
        nan_gradient: bool = False,
    ) -> None:
        super().__init__()
        if num_classes < 2:
            raise ValueError("num_classes must be at least two")
        self.bias = nn.Parameter(torch.linspace(0.3, -0.3, steps=num_classes))
        self.nan_gradient = nan_gradient

    def forward(self, count: int) -> Tensor:
        logits = self.bias.expand(count, -1)
        if not self.nan_gradient:
            return logits

        class _FiniteForwardNanBackward(torch.autograd.Function):
            @staticmethod
            def forward(ctx: object, value: Tensor) -> Tensor:
                del ctx
                return value.clone()

            @staticmethod
            def backward(ctx: object, gradient: Tensor) -> Tensor:
                del ctx
                return torch.full_like(gradient, float("nan"))

        return _FiniteForwardNanBackward.apply(logits)


class _RecordingSGD(SGD):
    def __init__(self, parameters: Any, observations: dict[str, Any]) -> None:
        self._observations = observations
        super().__init__(parameters, lr=0.1)

    def step(self, closure: Any = None) -> Any:
        gradients = [
            parameter.grad
            for group in self.param_groups
            for parameter in group["params"]
            if parameter.grad is not None
        ]
        self._observations["gradient_nonempty"] = bool(gradients)
        self._observations["gradient_finite"] = bool(gradients) and all(
            bool(torch.isfinite(gradient).all()) for gradient in gradients
        )
        before = [
            parameter.detach().clone()
            for group in self.param_groups
            for parameter in group["params"]
        ]
        result = super().step(closure)
        after = [
            parameter.detach()
            for group in self.param_groups
            for parameter in group["params"]
        ]
        self._observations["parameter_changed"] = any(
            not torch.equal(left, right) for left, right in zip(before, after)
        )
        self._observations["events"].append("optimizer.step")
        return result


class _RecordingStepLR(StepLR):
    def __init__(self, optimizer: SGD, observations: dict[str, Any]) -> None:
        self._observations = observations
        self._record_steps = False
        super().__init__(optimizer, step_size=1, gamma=0.5)
        self._record_steps = True

    def step(self, epoch: int | None = None) -> None:
        super().step(epoch)
        if self._record_steps:
            self._observations["events"].append("scheduler.step")


class ProbeModel(LightningModule):
    """Deterministic model implementing the same public hooks as ``TBModel``."""

    def __init__(
        self,
        observations: dict[str, Any],
        *,
        compile_enabled: bool = False,
        fail_phase: str | None = None,
        nan_gradient: bool = False,
        mutate_runtime: bool = False,
        num_classes: int = 4,
    ) -> None:
        super().__init__()
        self.observations = observations
        self.projection = _BiasLogits(
            num_classes=num_classes,
            nan_gradient=nan_gradient,
        )
        self.evaluator = _ProbeEvaluator(observations)
        self.compile_enabled = compile_enabled
        self.fail_phase = fail_phase
        self.mutate_runtime = mutate_runtime
        self.num_classes = num_classes
        self.phase: str | None = None

    def setup(self, stage: str) -> None:
        self.observations["events"].append(f"model.setup:{stage}")
        if self.compile_enabled and stage == "fit":
            self.projection = torch.compile(self.projection)

    def transfer_batch_to_device(
        self,
        batch: Data | HeteroData,
        device: torch.device,
        dataloader_idx: int,
    ) -> Data | HeteroData:
        self.observations["events"].append("transfer")
        return super().transfer_batch_to_device(batch, device, dataloader_idx)

    def _begin(self, phase: str) -> None:
        self.phase = phase
        policy = "online" if phase == "train" else "exact"
        self.evaluator.begin(
            EvaluationContext(
                split=phase,  # type: ignore[arg-type]
                pass_kind="fit_epoch",
                policy=policy,  # type: ignore[arg-type]
                task="classification",
                num_classes=self.num_classes,
            )
        )

    def on_train_epoch_start(self) -> None:
        self._begin("train")

    def on_validation_epoch_start(self) -> None:
        self._begin("val")

    def on_test_epoch_start(self) -> None:
        self._begin("test")

    @staticmethod
    def _supervision(
        batch: Data | HeteroData, phase: str
    ) -> tuple[Tensor, Tensor]:
        if isinstance(batch, HeteroData):
            candidates = [
                batch[node_type]
                for node_type in batch.node_types
                if "y" in batch[node_type]
            ]
            if len(candidates) != 1:
                raise ValueError("probe batch must expose one supervised node type")
            store = candidates[0]
        else:
            store = batch
        targets = store.y.reshape(-1).to(torch.long)
        mask = store.get(f"{phase}_mask")
        if isinstance(mask, Tensor) and mask.numel() == targets.numel():
            targets = targets[mask]
        return targets, targets

    def _mutate_runtime_state(self) -> None:
        import random

        random.random()
        np.random.random()
        torch.rand(1)
        if torch.cuda.is_available():
            torch.rand(1, device="cuda")
        torch.set_default_dtype(torch.float64)
        torch.backends.cudnn.benchmark = not torch.backends.cudnn.benchmark

    def model_step(self, batch: Data | HeteroData) -> Mapping[str, Any]:
        if self.phase is None:
            raise RuntimeError("probe model has no active phase")
        if self.mutate_runtime:
            self._mutate_runtime_state()
        self.observations["events"].append(f"forward:{self.phase}")
        _, targets = self._supervision(batch, self.phase)
        self.observations["events"].append(f"supervision:{self.phase}")
        if self.fail_phase == self.phase:
            raise RuntimeError(f"intentional {self.phase} probe failure")
        logits = self.projection(int(targets.numel()))
        loss = torch.nn.functional.cross_entropy(logits, targets)
        self.observations["events"].append(f"loss:{self.phase}")
        self.evaluator.update(
            EvaluationBatch(
                outputs=logits,
                targets=targets,
                num_examples=int(targets.numel()),
                context=self.evaluator.context,
            )
        )
        return {
            "loss": loss,
            "logits": logits,
            "labels": targets,
            "num_supervised_examples": int(targets.numel()),
            "batch": batch,
        }

    def abort_evaluation(self) -> None:
        self.evaluator.abort()
        self.phase = None

    def configure_optimizers(self) -> dict[str, Any]:
        self.observations["events"].append("optimizer.construct")
        optimizer = _RecordingSGD(self.parameters(), self.observations)
        scheduler = _RecordingStepLR(optimizer, self.observations)
        self.observations["optimizer"] = optimizer
        self.observations["scheduler"] = scheduler
        return {
            "optimizer": optimizer,
            "lr_scheduler": {"scheduler": scheduler, "interval": "step"},
        }


class StatefulPhaseDataModule(LightningDataModule):
    """Minimal phase source with an observable committed cursor."""

    def __init__(self) -> None:
        super().__init__()
        self.cursor = {"train": 0, "val": 0, "test": 0}
        self.yield_events: list[str] = []
        self.batch = Data(
            x=torch.tensor([[1.0, 0.0], [0.0, 1.0]]),
            y=torch.tensor([0, 1]),
            train_mask=torch.tensor([True, True]),
            val_mask=torch.tensor([True, True]),
            test_mask=torch.tensor([True, True]),
        )

    def state_dict(self) -> dict[str, object]:
        return {"cursor": dict(self.cursor)}

    def load_state_dict(self, state_dict: Mapping[str, object]) -> None:
        cursor = state_dict["cursor"]
        assert isinstance(cursor, Mapping)
        self.cursor = {phase: int(cursor[phase]) for phase in self.cursor}

    def _loader(self, phase: str) -> GeometricDataLoader:
        self.cursor[phase] += 1
        self.yield_events.append(phase)
        return GeometricDataLoader(
            [self.batch],
            batch_size=1,
            shuffle=False,
        )

    def train_dataloader(self) -> GeometricDataLoader:
        return self._loader("train")

    def val_dataloader(self) -> GeometricDataLoader:
        return self._loader("val")

    def test_dataloader(self) -> GeometricDataLoader:
        return self._loader("test")


def make_observations() -> dict[str, Any]:
    return {"events": []}


def qualified_runner(
    datamodule: LightningDataModule,
    *,
    train: bool = True,
    test: bool = True,
    configure: Any = None,
) -> tuple[PreflightRunner, PreflightResult, Any]:
    cfg = _compose_tiny_config()
    cfg.train = train
    cfg.test = test
    if configure is not None:
        configure(cfg)
    output = _qualified_pipeline_output()
    output.datamodule = datamodule
    runner = PreflightRunner(cfg, output)
    return runner, runner.validate_static(), cfg


def run_probe(
    datamodule: LightningDataModule,
    observations: dict[str, Any],
    **model_options: Any,
) -> PreflightResult:
    runner, static_result, _ = qualified_runner(datamodule)
    return runner.run_probe(
        model_factory=lambda: ProbeModel(observations, **model_options),
        static_result=static_result,
    )


@pytest.mark.parametrize(
    ("train", "test", "expected"),
    [
        (True, False, ["train", "val"]),
        (False, True, ["test"]),
        (True, True, ["train", "val", "test"]),
    ],
)
def test_probe_reads_exactly_one_representative_batch_per_enabled_phase(
    train: bool,
    test: bool,
    expected: list[str],
) -> None:
    datamodule = StatefulPhaseDataModule()
    observations = make_observations()
    runner, static_result, _ = qualified_runner(
        datamodule,
        train=train,
        test=test,
    )

    result = runner.run_probe(
        model_factory=lambda: ProbeModel(observations),
        static_result=static_result,
    )

    assert result.passed
    assert datamodule.yield_events == expected
    assert datamodule.cursor == {"train": 0, "val": 0, "test": 0}
    assert [
        event.removeprefix("forward:")
        for event in observations["events"]
        if event.startswith("forward:")
    ] == expected


def _strategy_modules(
    task8_stores: dict[str, QualifiedStoreFixture],
    strategy_name: str,
) -> tuple[DiskGraphDataModule, DiskGraphDataModule]:
    if strategy_name == "homogeneous-disk-cluster":
        source: Any = task8_stores["homogeneous"].store_build.path
        factory = lambda: HomogeneousClusterStrategy(
            clusters_per_batch=1,
            seed=31,
        )
    elif strategy_name == "heterogeneous-disk-cluster":
        source = task8_stores["heterogeneous"].store_build.path
        factory = lambda: HeterogeneousClusterStrategy(
            clusters_per_batch=1,
            seed=31,
        )
    else:
        fixture = task8_stores["heterogeneous"]
        with TypedGraphStore.open(fixture.store_build.path) as store:
            source = materialized_heterogeneous_reference(store)
            fanout = exhaustive_fanout(store.relation_types)
        factory = lambda: HeterogeneousNeighborStrategy(
            batch_size=1,
            num_neighbors=fanout,
            seed=31,
        )
    return (
        DiskGraphDataModule(source, factory(), train_shuffle=False),
        DiskGraphDataModule(source, factory(), train_shuffle=False),
    )


@pytest.mark.parametrize(
    "strategy_name",
    [
        "homogeneous-disk-cluster",
        "heterogeneous-disk-cluster",
        "heterogeneous-materialized-neighbor",
    ],
)
def test_probe_preserves_first_canonical_descriptor_and_committed_sampler_state(
    task8_stores: dict[str, QualifiedStoreFixture],
    strategy_name: str,
) -> None:
    control, probed = _strategy_modules(task8_stores, strategy_name)
    observations = make_observations()
    try:
        control_descriptors = {
            phase: next(
                iter(getattr(control, f"{phase}_dataloader")())
            ).sampling_descriptor
            for phase in ("train", "val", "test")
        }
        committed_before = probed.state_dict()

        result = run_probe(probed, observations)

        assert result.passed
        assert [
            event.removeprefix("forward:")
            for event in observations["events"]
            if event.startswith("forward:")
        ] == ["train", "val", "test"]
        assert probed.state_dict() == committed_before
        for phase in ("train", "val", "test"):
            batch = next(iter(getattr(probed, f"{phase}_dataloader")()))
            assert batch.sampling_descriptor == control_descriptors[phase]
    finally:
        control.teardown(None)
        probed.teardown(None)


class _ObservedHomogeneousPCA(IncrementalPCATransform):
    calls = {"begin": 0, "update": 0, "finalize": 0, "transform": 0}
    fit_features: list[np.ndarray] = []

    def __init__(self) -> None:
        super().__init__(
            n_components=1,
            max_batch_rows=1,
            max_batch_bytes=8,
            target_node_type="node",
            input_dtype="float32",
            output_dtype="float32",
            accumulation_dtype="float64",
        )

    def begin_fit(self, context: Any) -> None:
        type(self).calls["begin"] += 1
        super().begin_fit(context)

    def update_fit(
        self,
        features: np.ndarray,
        labels: np.ndarray | None = None,
    ) -> None:
        type(self).calls["update"] += 1
        type(self).fit_features.append(np.array(features, copy=True))
        super().update_fit(features, labels)

    def finalize_fit(self, state_root: str | Path) -> Any:
        type(self).calls["finalize"] += 1
        return super().finalize_fit(state_root)

    def transform(self, batch: Data | HeteroData) -> Data | HeteroData:
        type(self).calls["transform"] += 1
        return super().transform(batch)


def test_probe_exercises_training_only_fit_without_publishing_or_binding_state(
    task8_stores: dict[str, QualifiedStoreFixture],
    tmp_path: Path,
) -> None:
    production_state_root = tmp_path / "production-fitted-state"
    transform = _ObservedHomogeneousPCA()
    type(transform).calls = {name: 0 for name in type(transform).calls}
    type(transform).fit_features = []
    with TypedGraphStore.open(
        task8_stores["homogeneous"].store_build.path
    ) as store:
        node_type = store.node_types[0]
        train_ids = store.split_ids(store.active_split_tag, "train")
        expected_train_features = np.array(
            store.node_features(node_type)[np.asarray(train_ids)],
            copy=True,
        )
    module = DiskGraphDataModule(
        task8_stores["homogeneous"].store_build.path,
        HomogeneousClusterStrategy(clusters_per_batch=1, seed=19),
        fitted_transform=transform,
        fitted_state_root=production_state_root,
        train_shuffle=False,
    )
    observations = make_observations()
    try:
        result = run_probe(module, observations)

        assert result.passed
        assert all(count > 0 for count in type(transform).calls.values())
        np.testing.assert_array_equal(
            np.concatenate(type(transform).fit_features),
            expected_train_features,
        )
        assert transform.status is FitStatus.UNFITTED
        assert transform.state_key is None
        assert not production_state_root.exists()
    finally:
        module.teardown(None)


@pytest.mark.parametrize("phase", ["val", "test"])
def test_nontrain_disk_probe_requires_existing_fitted_state_without_fitting(
    task8_stores: dict[str, QualifiedStoreFixture],
    tmp_path: Path,
    phase: str,
) -> None:
    production_state_root = tmp_path / "missing-production-fitted-state"
    transform = _ObservedHomogeneousPCA()
    type(transform).calls = {name: 0 for name in type(transform).calls}
    type(transform).fit_features = []
    module = DiskGraphDataModule(
        task8_stores["homogeneous"].store_build.path,
        HomogeneousClusterStrategy(clusters_per_batch=1, seed=43),
        fitted_transform=transform,
        fitted_state_root=production_state_root,
        train_shuffle=False,
    )
    try:
        with pytest.raises(
            FitStateError,
            match="requires an existing exact fitted state",
        ):
            with module.noncommitting_probe_batches((phase,)):
                pass

        assert type(transform).calls == {
            "begin": 0,
            "update": 0,
            "finalize": 0,
            "transform": 0,
        }
        assert type(transform).fit_features == []
        assert transform.status is FitStatus.UNFITTED
        assert transform.state_key is None
        assert not production_state_root.exists()
    finally:
        module.teardown(None)
