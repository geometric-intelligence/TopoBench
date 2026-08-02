"""Lightning 2.4 optimizer-commit callback and TBModel participant tests."""

from __future__ import annotations

import copy
import json
from dataclasses import dataclass
from pathlib import Path
from types import SimpleNamespace
from typing import Any

import pytest
import torch
from hydra.utils import instantiate
from lightning import LightningDataModule, LightningModule, Trainer
from omegaconf import OmegaConf
from torch import nn
from torch.utils.data import DataLoader
from torch_geometric.data import Data

from topobench.callbacks.dataloader_commit import DataloaderCommitCallback
from topobench.dataloader.disk_graph import SamplingDescriptor
from topobench.dataloader.sequence_state import SequenceIdentity, SequenceState
from topobench.model.model import TBModel


def _descriptors() -> tuple[SamplingDescriptor, ...]:
    options = json.dumps(
        {"clusters_per_batch": 1, "partition_groups": None},
        sort_keys=True,
        separators=(",", ":"),
    )
    return tuple(
        SamplingDescriptor(
            content_sha256="a" * 64,
            active_split_tag="primary",
            phase="train",
            strategy="homogeneous-cluster",
            strategy_options_json=options,
            batch_ordinal=index,
            partition_ids=(index,),
            participant_counts=(("node", 1),),
            generator_seed=index + 10,
            generator_state_sha256=f"{index + 1:064x}",
        )
        for index in range(2)
    )


def _state() -> SequenceState:
    descriptors = _descriptors()
    identity = SequenceIdentity.from_descriptors(
        descriptors,
        partition_book_identity="b" * 64,
        fitted_transform_state_key=None,
        sampler_state={
            "format_version": "graph-sampling-state-v1",
            "seed": 5,
            "strategy": "homogeneous-cluster",
        },
    )
    return SequenceState(identity, descriptors)


def _deliver(state: SequenceState, descriptor: SamplingDescriptor) -> int:
    sequence_id = state.issue(descriptor)
    state.prepare(sequence_id, descriptor)
    state.deliver(sequence_id)
    return sequence_id


class _CommitDataModule:
    def __init__(self, state: SequenceState | None = None) -> None:
        self.sequence_state = _state() if state is None else state

    def consume_sequence(self, sequence_id: int) -> None:
        self.sequence_state.consume(sequence_id)

    def commit_optimizer_step(
        self,
        *,
        optimizer_succeeded: bool,
        model_global_step: int,
        evaluator_snapshot: dict[str, Any],
        epoch: int,
    ) -> bool:
        return self.sequence_state.commit(
            optimizer_succeeded=optimizer_succeeded,
            model_global_step=model_global_step,
            evaluator_sequence=evaluator_snapshot["sequence_id"],
            evaluator_count=evaluator_snapshot["count"],
            evaluator_state=evaluator_snapshot["state"],
            epoch=epoch,
        )

    def state_dict(self) -> dict[str, Any]:
        return self.sequence_state.state_dict()

    def load_state_dict(self, state_dict: dict[str, Any]) -> None:
        self.sequence_state.load_state_dict(state_dict)


class _CommitModel:
    def __init__(self) -> None:
        self.dataloader_optimizer_success_token = 0
        self.sequence_id = 0
        self.count = 0
        self.state: dict[str, Any] = {}
        self.restored: dict[str, Any] | None = None

    def dataloader_evaluator_snapshot(self) -> dict[str, Any]:
        return {
            "sequence_id": self.sequence_id,
            "count": self.count,
            "state": copy.deepcopy(self.state),
        }

    def dataloader_restore_evaluator(self, snapshot: dict[str, Any]) -> None:
        self.restored = copy.deepcopy(snapshot)
        self.sequence_id = snapshot["sequence_id"]
        self.count = snapshot["count"]
        self.state = copy.deepcopy(snapshot["state"])


@dataclass
class _FakeTrainer:
    datamodule: _CommitDataModule
    callbacks: list[Any]
    global_step: int = 0
    current_epoch: int = 0


def test_callback_accumulates_consumed_ids_and_commits_only_token_plus_step() -> None:
    descriptors = _descriptors()
    datamodule = _CommitDataModule()
    callback = DataloaderCommitCallback()
    model = _CommitModel()
    trainer = _FakeTrainer(datamodule, [callback])
    callback.setup(trainer, model, "fit")

    first_id = _deliver(datamodule.sequence_state, descriptors[0])
    first_batch = SimpleNamespace(sequence_id=first_id)
    callback.on_train_batch_start(trainer, model, first_batch, 0)
    model.sequence_id, model.count, model.state = 1, 1, {"sum": 1}
    callback.on_before_optimizer_step(trainer, model, object())
    callback.on_train_batch_end(trainer, model, None, first_batch, 0)
    assert datamodule.sequence_state.pending_group == (1,)
    assert datamodule.sequence_state.committed_cursor == 0

    second_id = _deliver(datamodule.sequence_state, descriptors[1])
    second_batch = SimpleNamespace(sequence_id=second_id)
    callback.on_train_batch_start(trainer, model, second_batch, 1)
    model.sequence_id, model.count, model.state = 2, 2, {"sum": 3}
    model.dataloader_optimizer_success_token += 1
    trainer.global_step += 1
    callback.on_train_batch_end(trainer, model, None, second_batch, 1)
    assert datamodule.sequence_state.committed_cursor == 2
    assert datamodule.sequence_state.pending_group == ()

    third_id = _deliver(datamodule.sequence_state, descriptors[0])
    third_batch = SimpleNamespace(sequence_id=third_id)
    callback.on_train_batch_start(trainer, model, third_batch, 2)
    model.sequence_id, model.count, model.state = 3, 3, {"sum": 5}
    trainer.global_step += 1
    callback.on_train_batch_end(trainer, model, None, third_batch, 2)
    assert datamodule.sequence_state.committed_cursor == 2
    assert datamodule.sequence_state.pending_group == (3,)

    callback.on_exception(trainer, model, RuntimeError("backward failed"))
    assert datamodule.sequence_state.committed_cursor == 2


def test_callback_rejects_success_token_without_global_step_advance() -> None:
    descriptor = _descriptors()[0]
    datamodule = _CommitDataModule()
    callback = DataloaderCommitCallback()
    model = _CommitModel()
    trainer = _FakeTrainer(datamodule, [callback])
    callback.setup(trainer, model, "fit")
    sequence_id = _deliver(datamodule.sequence_state, descriptor)
    batch = SimpleNamespace(sequence_id=sequence_id)
    callback.on_train_batch_start(trainer, model, batch, 0)
    model.sequence_id, model.count, model.state = 1, 1, {"sum": 1}
    model.dataloader_optimizer_success_token = 1

    with pytest.raises(RuntimeError, match="without global_step advance"):
        callback.on_train_batch_end(trainer, model, None, batch, 0)
    assert datamodule.sequence_state.committed_cursor == 0
    assert datamodule.sequence_state.pending_group == (1,)


def test_skipped_global_step_then_success_commits_pending_group_without_gap() -> None:
    descriptors = _descriptors()
    datamodule = _CommitDataModule()
    callback = DataloaderCommitCallback()
    model = _CommitModel()
    trainer = _FakeTrainer(datamodule, [callback])
    callback.setup(trainer, model, "fit")

    first_id = _deliver(datamodule.sequence_state, descriptors[0])
    first_batch = SimpleNamespace(sequence_id=first_id)
    callback.on_train_batch_start(trainer, model, first_batch, 0)
    model.sequence_id, model.count, model.state = 1, 1, {"sum": 1}
    trainer.global_step = 1
    callback.on_train_batch_end(trainer, model, None, first_batch, 0)
    assert datamodule.sequence_state.committed_cursor == 0
    assert datamodule.sequence_state.pending_group == (1,)

    second_id = _deliver(datamodule.sequence_state, descriptors[1])
    second_batch = SimpleNamespace(sequence_id=second_id)
    callback.on_train_batch_start(trainer, model, second_batch, 1)
    model.sequence_id, model.count, model.state = 2, 2, {"sum": 3}
    model.dataloader_optimizer_success_token = 1
    trainer.global_step = 2
    callback.on_train_batch_end(trainer, model, None, second_batch, 1)
    assert datamodule.sequence_state.committed_cursor == 2
    assert datamodule.sequence_state.committed_global_step == 2
    assert datamodule.sequence_state.pending_group == ()


def test_skipped_checkpoint_regenerates_then_later_success_uses_runtime_step() -> None:
    descriptor = _descriptors()[0]
    source_module = _CommitDataModule()
    source_callback = DataloaderCommitCallback()
    source_model = _CommitModel()
    source_trainer = _FakeTrainer(source_module, [source_callback])
    source_callback.setup(source_trainer, source_model, "fit")
    sequence_id = _deliver(source_module.sequence_state, descriptor)
    batch = SimpleNamespace(sequence_id=sequence_id)
    source_callback.on_train_batch_start(source_trainer, source_model, batch, 0)
    source_model.sequence_id = source_model.count = 1
    source_model.state = {"sum": 1}
    source_trainer.global_step = 1
    source_callback.on_train_batch_end(
        source_trainer, source_model, None, batch, 0
    )
    checkpoint = source_callback.state_dict()
    assert checkpoint["data_module"]["committed_cursor"] == 0
    assert checkpoint["data_module"]["committed_global_step"] == 0
    assert set(checkpoint) == {"format_version", "data_module"}

    restored_module = _CommitDataModule()
    restored_callback = DataloaderCommitCallback()
    restored_model = _CommitModel()
    restored_trainer = _FakeTrainer(
        restored_module,
        [restored_callback],
        global_step=1,
    )
    restored_callback.setup(restored_trainer, restored_model, "fit")
    restored_callback.load_state_dict(checkpoint)
    regenerated_id = _deliver(restored_module.sequence_state, descriptor)
    assert regenerated_id == 1
    regenerated = SimpleNamespace(sequence_id=regenerated_id)
    restored_callback.on_train_batch_start(
        restored_trainer, restored_model, regenerated, 0
    )
    restored_model.sequence_id = restored_model.count = 1
    restored_model.state = {"sum": 1}
    restored_model.dataloader_optimizer_success_token = 1
    restored_trainer.global_step = 2
    restored_callback.on_train_batch_end(
        restored_trainer, restored_model, None, regenerated, 0
    )
    assert restored_module.sequence_state.committed_cursor == 1
    assert restored_module.sequence_state.committed_global_step == 2


@pytest.mark.parametrize(
    ("current_step", "message"),
    [(0, "regressed"), (3, "jumped by more than one")],
)
def test_callback_rejects_observed_global_step_regression_or_jump(
    current_step: int,
    message: str,
) -> None:
    datamodule = _CommitDataModule()
    callback = DataloaderCommitCallback()
    model = _CommitModel()
    trainer = _FakeTrainer(
        datamodule,
        [callback],
        global_step=1,
    )
    callback.setup(trainer, model, "fit")
    trainer.global_step = current_step
    with pytest.raises(ValueError, match=message):
        callback.on_train_batch_end(
            trainer,
            model,
            None,
            SimpleNamespace(sequence_id=1),
            0,
        )


def test_callback_checkpoint_contains_only_durable_module_state_and_rejects_identity() -> None:
    descriptors = _descriptors()
    source_module = _CommitDataModule()
    _deliver(source_module.sequence_state, descriptors[0])
    source_module.consume_sequence(1)
    source_module.commit_optimizer_step(
        optimizer_succeeded=True,
        model_global_step=1,
        evaluator_snapshot={"sequence_id": 1, "count": 1, "state": {"sum": 1}},
        epoch=0,
    )
    callback = DataloaderCommitCallback()
    model = _CommitModel()
    callback.setup(_FakeTrainer(source_module, [callback]), model, "fit")
    callback_state = callback.state_dict()

    assert set(callback_state) == {"format_version", "data_module"}
    assert callback_state["data_module"] == source_module.state_dict()
    serialized = repr(callback_state).lower()
    assert "pending" not in serialized
    assert "issued" not in serialized
    assert "prepared" not in serialized
    assert "sampling_descriptor" not in serialized

    restored_module = _CommitDataModule()
    restored_callback = DataloaderCommitCallback()
    restored_model = _CommitModel()
    restored_callback.setup(
        _FakeTrainer(restored_module, [restored_callback]), restored_model, "fit"
    )
    restored_callback.load_state_dict(callback_state)
    assert restored_module.sequence_state.committed_cursor == 1

    mismatched_descriptors = tuple(
        replace_descriptor_content(descriptor, "d" * 64)
        for descriptor in descriptors
    )
    mismatched_identity = SequenceIdentity.from_descriptors(
        mismatched_descriptors,
        partition_book_identity="b" * 64,
        fitted_transform_state_key=None,
        sampler_state={
            "format_version": "graph-sampling-state-v1",
            "seed": 5,
            "strategy": "homogeneous-cluster",
        },
    )
    mismatched_module = _CommitDataModule(
        SequenceState(mismatched_identity, mismatched_descriptors)
    )
    mismatched_callback = DataloaderCommitCallback()
    mismatched_callback.setup(
        _FakeTrainer(mismatched_module, [mismatched_callback]),
        _CommitModel(),
        "fit",
    )
    with pytest.raises(ValueError, match="identity mismatch"):
        mismatched_callback.load_state_dict(callback_state)


def replace_descriptor_content(
    descriptor: SamplingDescriptor, content_sha256: str
) -> SamplingDescriptor:
    return SamplingDescriptor(
        content_sha256=content_sha256,
        active_split_tag=descriptor.active_split_tag,
        phase=descriptor.phase,
        strategy=descriptor.strategy,
        strategy_options_json=descriptor.strategy_options_json,
        batch_ordinal=descriptor.batch_ordinal,
        partition_ids=descriptor.partition_ids,
        participant_counts=descriptor.participant_counts,
        generator_seed=descriptor.generator_seed,
        generator_state_sha256=descriptor.generator_state_sha256,
    )


class _EvaluatorOwner:
    def __init__(self) -> None:
        self.value = torch.tensor(3)

    def state_dict(self) -> dict[str, torch.Tensor]:
        return {"value": self.value.clone()}

    def load_state_dict(
        self, state_dict: dict[str, torch.Tensor], strict: bool = True
    ) -> None:
        assert strict
        assert set(state_dict) == {"value"}
        self.value = state_dict["value"].clone()


class _Evaluator:
    def __init__(self) -> None:
        self.metrics = _EvaluatorOwner()


class _WrapperOptimizer:
    def __init__(
        self,
        optimizer: torch.optim.Optimizer,
        *,
        call_raw_step: bool,
        fail_after_step: bool = False,
    ) -> None:
        self.optimizer = optimizer
        self.call_raw_step = call_raw_step
        self.fail_after_step = fail_after_step

    def step(self, closure: Any = None) -> None:
        if closure is not None:
            closure()
        if self.call_raw_step:
            self.optimizer.step()
        if self.fail_after_step:
            raise RuntimeError("wrapper failed after raw optimizer")


def _tbmodel() -> TBModel:
    backbone = nn.Linear(1, 1)
    readout = nn.Identity()
    readout.task_level = "graph"
    return TBModel(
        backbone=backbone,
        readout=readout,
        loss=nn.Identity(),
        evaluator=_Evaluator(),
        optimizer=SimpleNamespace(configure_optimizer=lambda parameters: {}),
    )


def test_tbmodel_success_token_proves_raw_optimizer_completion_and_evaluator_state() -> None:
    model = _tbmodel()
    raw = torch.optim.SGD(model.backbone.parameters(), lr=0.1)
    closure = lambda: None

    model.optimizer_step(0, 0, _WrapperOptimizer(raw, call_raw_step=False), closure)
    assert model.dataloader_optimizer_success_token == 0
    with pytest.raises(RuntimeError, match="failed after raw"):
        model.optimizer_step(
            0,
            1,
            _WrapperOptimizer(raw, call_raw_step=True, fail_after_step=True),
            closure,
        )
    assert model.dataloader_optimizer_success_token == 0
    model.optimizer_step(0, 2, _WrapperOptimizer(raw, call_raw_step=True), closure)
    assert model.dataloader_optimizer_success_token == 1

    snapshot = model.dataloader_evaluator_snapshot()
    assert snapshot["sequence_id"] == 0
    assert snapshot["count"] == 0
    assert snapshot["state"]["value"].item() == 3
    snapshot["state"]["value"].fill_(9)
    model.dataloader_restore_evaluator(snapshot)
    assert model.dataloader_evaluator_snapshot()["state"]["value"].item() == 9


def test_hydra_config_has_one_exact_callback_target() -> None:
    config_path = (
        Path(__file__).parents[2] / "configs" / "callbacks" / "dataloader_commit.yaml"
    )
    config = OmegaConf.load(config_path)
    assert tuple(config) == ("dataloader_commit",)
    node = config.dataloader_commit
    assert node._target_ == (
        "topobench.callbacks.dataloader_commit.DataloaderCommitCallback"
    )
    assert isinstance(instantiate(node), DataloaderCommitCallback)

    callback = DataloaderCommitCallback()
    duplicate = DataloaderCommitCallback()
    trainer = _FakeTrainer(_CommitDataModule(), [callback, duplicate])
    with pytest.raises(RuntimeError, match="exactly one"):
        callback.setup(trainer, _CommitModel(), "fit")


class _TrainerDataModule(LightningDataModule):
    def __init__(self) -> None:
        super().__init__()
        self.sequence_state = _state()
        self._batches = tuple(
            Data(x=torch.tensor([float(index + 1)]), sequence_id=index + 1)
            for index in range(2)
        )

    def train_dataloader(self) -> DataLoader:
        if self.sequence_state.committed_cursor == 0 and not self.sequence_state.issued:
            for descriptor in _descriptors():
                _deliver(self.sequence_state, descriptor)
        return DataLoader(self._batches, batch_size=None)

    def consume_sequence(self, sequence_id: int) -> None:
        self.sequence_state.consume(sequence_id)

    def commit_optimizer_step(
        self,
        *,
        optimizer_succeeded: bool,
        model_global_step: int,
        evaluator_snapshot: dict[str, Any],
        epoch: int,
    ) -> bool:
        return self.sequence_state.commit(
            optimizer_succeeded=optimizer_succeeded,
            model_global_step=model_global_step,
            evaluator_sequence=evaluator_snapshot["sequence_id"],
            evaluator_count=evaluator_snapshot["count"],
            evaluator_state=evaluator_snapshot["state"],
            epoch=epoch,
        )

    def state_dict(self) -> dict[str, Any]:
        return self.sequence_state.state_dict()

    def load_state_dict(self, state_dict: dict[str, Any]) -> None:
        self.sequence_state.load_state_dict(state_dict)


class _TinyTrainerModel(LightningModule):
    def __init__(self) -> None:
        super().__init__()
        self.weight = nn.Parameter(torch.tensor(1.0))
        self.dataloader_optimizer_success_token = 0
        self._sequence = 0
        self._count = 0
        self._sum = 0.0

    def training_step(self, batch: Data, batch_idx: int) -> torch.Tensor:
        self._sequence = int(batch.sequence_id)
        self._count += 1
        self._sum += float(batch.x.item())
        return (self.weight * batch.x).sum()

    def optimizer_step(self, epoch, batch_idx, optimizer, optimizer_closure=None):
        super().optimizer_step(epoch, batch_idx, optimizer, optimizer_closure)
        self.dataloader_optimizer_success_token += 1

    def dataloader_evaluator_snapshot(self) -> dict[str, Any]:
        return {
            "sequence_id": self._sequence,
            "count": self._count,
            "state": {"sum": self._sum},
        }

    def dataloader_restore_evaluator(self, snapshot: dict[str, Any]) -> None:
        self._sequence = snapshot["sequence_id"]
        self._count = snapshot["count"]
        self._sum = snapshot["state"]["sum"]

    def configure_optimizers(self):
        return torch.optim.SGD(self.parameters(), lr=0.01)


def test_real_lightning_trainer_commits_one_accumulated_group_after_batch_end() -> None:
    datamodule = _TrainerDataModule()
    callback = DataloaderCommitCallback()
    trainer = Trainer(
        accelerator="cpu",
        devices=1,
        max_epochs=1,
        accumulate_grad_batches=2,
        callbacks=[callback],
        enable_checkpointing=False,
        enable_model_summary=False,
        logger=False,
        limit_val_batches=0,
        num_sanity_val_steps=0,
    )
    trainer.fit(_TinyTrainerModel(), datamodule=datamodule)

    assert trainer.global_step == 1
    assert datamodule.sequence_state.committed_cursor == 2
    assert datamodule.sequence_state.committed_global_step == 1
    assert datamodule.sequence_state.committed_evaluator_count == 2
    assert datamodule.sequence_state.pending_group == ()
