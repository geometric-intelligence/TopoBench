"""Production-Lightning exact resume for homogeneous disk cluster training."""

from __future__ import annotations

from collections.abc import Callable, Mapping
from dataclasses import dataclass
from pathlib import Path
from typing import Any

import pytest
import torch
from lightning import Callback, Trainer, seed_everything
from lightning.pytorch.callbacks import ModelCheckpoint
from omegaconf import DictConfig, open_dict

from test.callbacks.test_dataloader_commit import (
    _CommitDataModule,
    _CommitModel,
    _deliver,
    _descriptors,
    _FakeTrainer,
)
from test.pipeline.test_disk_graph_pipeline import (
    _binary_homogeneous_source,
    _build,
    _parquet_cfg,
)
from topobench.callbacks.best_epoch_metrics import BestEpochMetricsCallback
from topobench.callbacks.dataloader_commit import DataloaderCommitCallback
from topobench.data.pipelines import DataPipelineOutput
from topobench.evaluator import TBEvaluator
from topobench.evaluator.types import EvaluationBatch, EvaluationContext
from topobench.nn.capabilities import validate_capability_composition
from topobench.utils.model_instantiation import instantiate_model

INTERRUPTION_BOUNDARIES = (
    "issue",
    "prepare",
    "consume",
    "evaluator",
    "optimizer",
    "commit",
)
BITWISE_PROFILE_FIELDS = (
    "model_state",
    "optimizer_state",
    "scheduler_state",
    "sequence_state",
    "selected_checkpoint",
    "prediction_logits",
    "prediction_identities",
    "prediction_external_ids",
)
NUMERIC_PROFILE_FIELDS = ("final_metrics", "best_epoch_metrics")
NUMERIC_ABS_TOLERANCE = 1.0e-7
NUMERIC_REL_TOLERANCE = 1.0e-6
_BEST_EPOCH_STATE_KEYS = {
    "format_version",
    "monitor_metric",
    "mode",
    "best_monitored_value",
    "best_epoch_metrics",
    "best_epoch_number",
    "current_epoch_train_metrics",
}
_SEED = 310_519
_MAX_EPOCHS = 3


class _BoundaryCrash(RuntimeError):
    """Deterministic test-only process interruption."""


@dataclass(frozen=True, slots=True)
class _RunProfile:
    model_state: Mapping[str, object]
    optimizer_state: Mapping[str, object]
    scheduler_state: tuple[Mapping[str, object], ...]
    sequence_state: Mapping[str, object]
    selected_checkpoint: Mapping[str, object]
    prediction_logits: tuple[torch.Tensor, ...]
    prediction_identities: tuple[tuple[object, ...], ...]
    prediction_external_ids: tuple[tuple[object, ...], ...]
    final_metrics: Mapping[str, float]
    best_epoch_number: int | None
    best_monitored_value: float | None
    best_epoch_metrics: Mapping[str, float]
    train_descriptors: tuple[object, ...]


@dataclass(frozen=True, slots=True)
class _ResumeObservation:
    state: Mapping[str, object]
    remaining_descriptors: tuple[object, ...]
    evaluator_snapshot: Mapping[str, object]


@dataclass(frozen=True, slots=True)
class _BoundaryObservation:
    committed_state: Mapping[str, object]
    issued: tuple[int, ...]
    prepared: tuple[int, ...]
    delivered: tuple[int, ...]
    consumed: tuple[int, ...]
    pending_group: tuple[int, ...]
    evaluator_snapshot: Mapping[str, object]
    batch_start_evaluator_snapshot: Mapping[str, object] | None
    optimizer_token: int
    optimizer_state: Mapping[str, object]
    global_step: int


@dataclass(frozen=True, slots=True)
class _LifecycleCase:
    source: object
    store_path: Path
    root: Path
    reference: _RunProfile
    config_factory: Callable[[Path, Path], DictConfig]


class _DurableCheckpoint(Callback):
    """Persist the latest real Lightning checkpoint after durable boundaries."""

    def __init__(self, path: Path) -> None:
        super().__init__()
        self.path = path
        self._installed = False

    def _save(self, trainer: Trainer) -> None:
        self.path.parent.mkdir(parents=True, exist_ok=True)
        trainer.save_checkpoint(self.path)

    def setup(
        self,
        trainer: Trainer,
        pl_module: object,
        stage: str,
    ) -> None:
        del pl_module
        if stage != "fit" or self._installed:
            return
        self._installed = True
        checkpoint = _one(trainer.callbacks, ModelCheckpoint)
        original = checkpoint.on_train_epoch_end

        def save_after_selection(
            trainer: Trainer,
            pl_module: object,
        ) -> None:
            original(trainer, pl_module)
            self._save(trainer)

        checkpoint.on_train_epoch_end = save_after_selection


    def on_train_batch_end(
        self,
        trainer: Trainer,
        pl_module: object,
        outputs: object,
        batch: object,
        batch_idx: int,
    ) -> None:
        del pl_module, outputs, batch, batch_idx
        self._save(trainer)


class _ResumeProbe(Callback):
    def __init__(self) -> None:
        super().__init__()
        self.observation: _ResumeObservation | None = None

    def _capture(self, trainer: Trainer, pl_module: object) -> None:
        if self.observation is not None:
            return
        module = trainer.datamodule
        state = _clone(module.state_dict())
        descriptors = tuple(module.descriptors("train"))
        cursor = int(state["committed_cursor"])
        offset = cursor % len(descriptors)
        self.observation = _ResumeObservation(
            state,
            descriptors[offset:],
            _clone(pl_module.dataloader_evaluator_snapshot()),
        )

    def on_train_batch_start(
        self,
        trainer: Trainer,
        pl_module: object,
        batch: object,
        batch_idx: int,
    ) -> None:
        del batch, batch_idx
        self._capture(trainer, pl_module)

    def on_validation_epoch_start(
        self,
        trainer: Trainer,
        pl_module: object,
    ) -> None:
        self._capture(trainer, pl_module)


class _InterruptAtBoundary(Callback):
    """Raise after one exact production participant reaches its boundary."""

    def __init__(
        self,
        boundary: str,
        target_sequence: int,
        checkpoint_path: Path,
    ) -> None:
        super().__init__()
        if boundary not in INTERRUPTION_BOUNDARIES:
            raise ValueError(f"unknown boundary {boundary!r}")
        self.boundary = boundary
        self.target_sequence = target_sequence
        self.checkpoint_path = checkpoint_path
        self.unsafe_checkpoint_path = checkpoint_path.with_name(
            f"{boundary}-in-flight.ckpt"
        )
        self.observation: _BoundaryObservation | None = None
        self.checkpoint_rejected = False
        self._installed = False

    def _capture(self, trainer: Trainer, pl_module: object) -> None:
        module = trainer.datamodule
        state = module.sequence_state
        commit = next(
            callback
            for callback in trainer.callbacks
            if isinstance(callback, DataloaderCommitCallback)
        )
        baseline = commit._active_batch_evaluator_snapshot
        self.observation = _BoundaryObservation(
            committed_state=_clone(module.state_dict()),
            issued=tuple(state.issued),
            prepared=tuple(state.prepared),
            delivered=tuple(state.delivered),
            consumed=tuple(state.consumed),
            pending_group=tuple(state.pending_group),
            evaluator_snapshot=_clone(
                pl_module.dataloader_evaluator_snapshot()
            ),
            batch_start_evaluator_snapshot=(
                None if baseline is None else _clone(baseline)
            ),
            optimizer_token=int(pl_module.dataloader_optimizer_success_token),
            global_step=int(trainer.global_step),
            optimizer_state=_clone(trainer.optimizers[0].state_dict()),
        )

    def _crash(self) -> None:
        raise _BoundaryCrash(
            f"interrupted after {self.boundary} sequence {self.target_sequence}"
        )

    def _save_and_crash(
        self,
        trainer: Trainer,
        pl_module: object,
    ) -> None:
        trainer.save_checkpoint(self.checkpoint_path)
        self._capture(trainer, pl_module)
        self._crash()

    def _reject_and_crash(
        self,
        trainer: Trainer,
        pl_module: object,
    ) -> None:
        self._capture(trainer, pl_module)
        try:
            trainer.save_checkpoint(self.unsafe_checkpoint_path)
        except RuntimeError as error:
            if "uncommitted training work" not in str(error):
                raise
            self.checkpoint_rejected = True
        else:
            raise AssertionError(
                f"{self.boundary} in-flight checkpoint was published"
            )
        assert not self.unsafe_checkpoint_path.exists()
        self._crash()

    def setup(
        self,
        trainer: Trainer,
        pl_module: object,
        stage: str,
    ) -> None:
        if stage != "fit" or self._installed:
            return
        self._installed = True
        module = trainer.datamodule
        target = self.target_sequence
        if self.boundary == "issue":
            original = module.issue_descriptor

            def issue(descriptor: object) -> int:
                sequence_id = original(descriptor)
                if sequence_id == target:
                    self._save_and_crash(trainer, pl_module)
                return sequence_id

            module.issue_descriptor = issue
        elif self.boundary == "prepare":
            original = module.prepare_sequence

            def prepare(sequence_id: int, descriptor: object) -> None:
                original(sequence_id, descriptor)
                if sequence_id == target:
                    self._save_and_crash(trainer, pl_module)

            module.prepare_sequence = prepare
        elif self.boundary == "evaluator":
            original = pl_module.model_step

            def model_step(batch: object) -> object:
                result = original(batch)
                if (
                    getattr(pl_module, "state_str", None) == "Training"
                    and getattr(batch, "sequence_id", None) == target
                ):
                    self._reject_and_crash(trainer, pl_module)
                return result

            pl_module.model_step = model_step

    def on_train_batch_start(
        self,
        trainer: Trainer,
        pl_module: object,
        batch: object,
        batch_idx: int,
    ) -> None:
        del batch_idx
        if (
            self.boundary == "consume"
            and getattr(batch, "sequence_id", None) == self.target_sequence
        ):
            self._save_and_crash(trainer, pl_module)

    def on_train_batch_end(
        self,
        trainer: Trainer,
        pl_module: object,
        outputs: object,
        batch: object,
        batch_idx: int,
    ) -> None:
        del outputs, batch_idx
        if getattr(batch, "sequence_id", None) != self.target_sequence:
            return
        if self.boundary == "optimizer":
            self._reject_and_crash(trainer, pl_module)
        elif self.boundary == "commit":
            self._save_and_crash(trainer, pl_module)


def _clone(value: Any) -> Any:
    if isinstance(value, torch.Tensor):
        return value.detach().cpu().clone()
    if isinstance(value, Mapping):
        return {key: _clone(item) for key, item in value.items()}
    if isinstance(value, tuple):
        return tuple(_clone(item) for item in value)
    if isinstance(value, list):
        return [_clone(item) for item in value]
    return value


def _one(callbacks: list[Callback], callback_type: type[Any]) -> Any:
    matches = [item for item in callbacks if isinstance(item, callback_type)]
    assert len(matches) == 1
    return matches[0]


def _callbacks(
    root: Path,
    *,
    boundary: str | None = None,
    target_sequence: int | None = None,
    probe: _ResumeProbe | None = None,
) -> tuple[list[Callback], Path]:
    durable_path = root / "last-committed.ckpt"
    callbacks: list[Callback] = [
        DataloaderCommitCallback(),
        BestEpochMetricsCallback(monitor="val/loss", mode="min"),
        _DurableCheckpoint(durable_path),
    ]
    if probe is not None:
        callbacks.append(probe)
    if boundary is not None:
        assert target_sequence is not None
        controller = _InterruptAtBoundary(
            boundary,
            target_sequence,
            durable_path,
        )
        callbacks.insert(0 if boundary == "optimizer" else 2, controller)
    callbacks.append(
        ModelCheckpoint(
            dirpath=root / "selected",
            filename="epoch={epoch}-step={step}",
            monitor="val/loss",
            mode="min",
            save_top_k=1,
            save_last=True,
            auto_insert_metric_name=False,
        )
    )
    return callbacks, durable_path


def _trainer(root: Path, callbacks: list[Callback]) -> Trainer:
    return Trainer(
        accelerator="cpu",
        devices=1,
        max_epochs=_MAX_EPOCHS,
        deterministic=True,
        logger=False,
        callbacks=callbacks,
        default_root_dir=root,
        enable_checkpointing=True,
        enable_model_summary=False,
        enable_progress_bar=False,
        num_sanity_val_steps=0,
        log_every_n_steps=1,
    )


def _predictions(
    output: DataPipelineOutput,
    model: object,
) -> tuple[
    tuple[torch.Tensor, ...],
    tuple[tuple[object, ...], ...],
    tuple[tuple[object, ...], ...],
]:
    output.datamodule.setup("validate")
    adapter = output.prediction_row_adapter
    assert adapter is not None
    logits: list[torch.Tensor] = []
    identities: list[tuple[object, ...]] = []
    external_ids: list[tuple[object, ...]] = []
    model.eval()
    model.state_str = "Validation"
    with torch.no_grad():
        for batch in output.datamodule.val_dataloader():
            model_out = model.forward(batch)
            supervised = model.supervision_adapter.select(
                model_out,
                batch,
                "Validation",
            )
            canonical = adapter.resolve(batch, phase="val")
            assert len(canonical) == supervised.num_examples
            logits.append(supervised.logits.detach().cpu().clone())
            identities.append(tuple(canonical))
            external_ids.append(tuple(adapter.restore_external_ids(canonical)))
    return tuple(logits), tuple(identities), tuple(external_ids)


def _profile(
    output: DataPipelineOutput,
    model: object,
    trainer: Trainer,
    callbacks: list[Callback],
) -> _RunProfile:
    checkpoint = _one(callbacks, ModelCheckpoint)
    best = _one(callbacks, BestEpochMetricsCallback)
    selected_path = Path(checkpoint.best_model_path)
    assert selected_path.is_file()
    selected = torch.load(
        selected_path, map_location="cpu", weights_only=False
    )
    logits, identities, external_ids = _predictions(output, model)
    schedulers = tuple(
        _clone(item.scheduler.state_dict())
        for item in trainer.lr_scheduler_configs
    )
    metrics = {
        str(key): float(value)
        for key, value in trainer.callback_metrics.items()
        if isinstance(value, (int, float, torch.Tensor))
    }
    best_metrics = {
        str(key): float(value)
        for key, value in best.best_epoch_metrics.items()
    }
    return _RunProfile(
        model_state=_clone(model.state_dict()),
        optimizer_state=_clone(trainer.optimizers[0].state_dict()),
        scheduler_state=schedulers,
        sequence_state=_clone(output.datamodule.state_dict()),
        selected_checkpoint={
            "epoch": selected["epoch"],
            "global_step": selected["global_step"],
            "state_dict": _clone(selected["state_dict"]),
        },
        prediction_logits=logits,
        prediction_identities=identities,
        prediction_external_ids=external_ids,
        final_metrics=metrics,
        best_epoch_number=best.best_epoch_number,
        best_monitored_value=(
            None
            if best.best_monitored_value is None
            else float(best.best_monitored_value)
        ),
        best_epoch_metrics=best_metrics,
        train_descriptors=tuple(output.datamodule.descriptors("train")),
    )


def _run_complete(
    cfg: DictConfig,
    root: Path,
    *,
    ckpt_path: Path | None = None,
    probe: _ResumeProbe | None = None,
) -> _RunProfile:
    seed_everything(_SEED, workers=True)
    output = _build(cfg)
    model = instantiate_model(
        cfg,
        data_spec=output.data_spec,
        capability_validation=validate_capability_composition(
            cfg,
            observed=output.capability_spec,
        ),
    )
    callbacks, _ = _callbacks(root, probe=probe)
    trainer = _trainer(root, callbacks)
    trainer.fit(
        model=model,
        datamodule=output.datamodule,
        ckpt_path=None if ckpt_path is None else str(ckpt_path),
    )
    profile = _profile(output, model, trainer, callbacks)
    output.datamodule.close()
    return profile


def _run_interrupted(
    cfg: DictConfig,
    resume_cfg: DictConfig,
    root: Path,
    boundary: str,
) -> tuple[_RunProfile, _ResumeObservation]:
    seed_everything(_SEED, workers=True)
    output = _build(cfg)
    descriptors = tuple(output.datamodule.descriptors("train"))
    target_sequence = len(descriptors) + 1
    model = instantiate_model(
        cfg,
        data_spec=output.data_spec,
        capability_validation=validate_capability_composition(
            cfg,
            observed=output.capability_spec,
        ),
    )
    callbacks, durable_path = _callbacks(
        root / "crashed",
        boundary=boundary,
        target_sequence=target_sequence,
    )
    trainer = _trainer(root / "crashed", callbacks)
    with pytest.raises(
        _BoundaryCrash,
        match=rf"after {boundary} sequence {target_sequence}",
    ):
        trainer.fit(model=model, datamodule=output.datamodule)
    assert durable_path.is_file()

    best = _one(callbacks, BestEpochMetricsCallback)
    commit = _one(callbacks, DataloaderCommitCallback)
    controller = _one(callbacks, _InterruptAtBoundary)
    durable = torch.load(durable_path, map_location="cpu", weights_only=False)
    callback_state = durable["callbacks"][best.state_key]
    assert set(callback_state) == _BEST_EPOCH_STATE_KEYS
    assert callback_state["best_epoch_number"] == 0
    assert callback_state["best_monitored_value"] is not None
    committed_state = durable["callbacks"][commit.state_key]["data_module"]
    _assert_boundary_transition(
        controller,
        durable,
        committed_state,
        boundary,
        target_sequence,
    )
    output.datamodule.close()

    probe = _ResumeProbe()
    resumed = _run_complete(
        resume_cfg,
        root / "crashed",
        ckpt_path=durable_path,
        probe=probe,
    )
    assert probe.observation is not None
    restored_evaluator = probe.observation.evaluator_snapshot
    assert (
        restored_evaluator["sequence_id"]
        == committed_state["committed_evaluator_sequence"]
    )
    assert (
        restored_evaluator["count"]
        == committed_state["committed_evaluator_count"]
    )
    restored_state = restored_evaluator["state"]
    committed_evaluator_state = committed_state["committed_evaluator_state"]
    assert restored_state.keys() == committed_evaluator_state.keys()
    committed_cursor = int(committed_state["committed_cursor"])
    epoch_complete = committed_cursor % len(descriptors) == 0
    if epoch_complete and boundary != "commit":
        assert restored_state["num_examples"] == 0
    else:
        _assert_bitwise(
            restored_state,
            committed_evaluator_state,
            "restored_post_commit_evaluator",
        )
    _assert_bitwise(
        probe.observation.state,
        committed_state,
        "restored_committed_state",
    )
    return resumed, probe.observation


def _assert_bitwise(left: Any, right: Any, path: str = "profile") -> None:
    if isinstance(left, torch.Tensor) or isinstance(right, torch.Tensor):
        assert isinstance(left, torch.Tensor) and isinstance(
            right, torch.Tensor
        )
        assert left.dtype == right.dtype, path
        assert left.shape == right.shape, path
        assert torch.equal(left, right), path
        return
    if isinstance(left, Mapping) or isinstance(right, Mapping):
        assert isinstance(left, Mapping) and isinstance(right, Mapping), path
        assert left.keys() == right.keys(), path
        for key in left:
            _assert_bitwise(left[key], right[key], f"{path}.{key}")
        return
    if isinstance(left, (tuple, list)) or isinstance(right, (tuple, list)):
        assert isinstance(left, (tuple, list)) and isinstance(
            right, (tuple, list)
        ), path
        assert len(left) == len(right), path
        for index, (first, second) in enumerate(zip(left, right, strict=True)):
            _assert_bitwise(first, second, f"{path}[{index}]")
        return
    assert type(left) is type(right), path
    assert left == right, path


def _bitwise_equal(left: Any, right: Any) -> bool:
    try:
        _assert_bitwise(left, right)
    except AssertionError:
        return False
    return True


def _assert_boundary_transition(
    controller: _InterruptAtBoundary,
    durable: Mapping[str, object],
    committed_state: Mapping[str, object],
    boundary: str,
    target_sequence: int,
) -> None:
    observation = controller.observation
    assert observation is not None
    previous_sequence = target_sequence - 1
    interrupted_cursor = (
        target_sequence if boundary == "commit" else previous_sequence
    )
    assert observation.committed_state["committed_cursor"] == (
        interrupted_cursor
    )
    assert observation.committed_state["committed_global_step"] == (
        interrupted_cursor
    )
    assert (
        observation.committed_state["committed_sampler_state"]["cursor"]
        == interrupted_cursor
    )
    assert committed_state["committed_cursor"] == interrupted_cursor
    assert committed_state["committed_evaluator_sequence"] == (
        interrupted_cursor
    )
    assert committed_state["committed_evaluator_count"] == interrupted_cursor
    assert committed_state["committed_global_step"] == interrupted_cursor
    assert committed_state["committed_sampler_state"]["cursor"] == (
        interrupted_cursor
    )
    assert durable["global_step"] == interrupted_cursor
    assert len(durable["optimizer_states"]) == 1
    saved_optimizer = durable["optimizer_states"][0]
    if boundary != "optimizer":
        _assert_bitwise(
            observation.optimizer_state,
            saved_optimizer,
            "durable_optimizer",
        )

    target_was_issued = target_sequence in observation.issued
    target_was_prepared = target_sequence in observation.prepared
    target_was_delivered = target_sequence in observation.delivered
    target_was_consumed = target_sequence in observation.consumed
    target_is_pending = target_sequence in observation.pending_group
    evaluator = observation.evaluator_snapshot
    committed_evaluator = observation.committed_state[
        "committed_evaluator_state"
    ]

    if boundary in {"issue", "prepare", "consume", "evaluator"}:
        assert observation.optimizer_token == previous_sequence
        assert observation.global_step == previous_sequence

    if boundary == "issue":
        assert target_was_issued
        assert not target_was_prepared
        assert not target_was_delivered
        assert not target_was_consumed
        assert not target_is_pending
    elif boundary == "prepare":
        assert target_was_issued and target_was_prepared
        assert not target_was_delivered
        assert not target_was_consumed
        assert not target_is_pending
    elif boundary == "consume":
        assert (
            target_was_issued
            and target_was_prepared
            and target_was_delivered
            and target_was_consumed
            and target_is_pending
        )
        assert evaluator["sequence_id"] == previous_sequence
        assert evaluator["count"] == previous_sequence
        assert observation.batch_start_evaluator_snapshot is not None
        _assert_bitwise(
            evaluator,
            observation.batch_start_evaluator_snapshot,
            "consume_evaluator",
        )
    elif boundary == "evaluator":
        assert target_was_consumed and target_is_pending
        assert evaluator["sequence_id"] == previous_sequence
        assert evaluator["count"] == previous_sequence
        assert observation.batch_start_evaluator_snapshot is not None
        assert not _bitwise_equal(
            evaluator,
            observation.batch_start_evaluator_snapshot,
        )
        assert controller.checkpoint_rejected
    elif boundary == "optimizer":
        assert target_was_consumed and target_is_pending
        assert evaluator["sequence_id"] == target_sequence
        assert evaluator["count"] == target_sequence
        assert observation.batch_start_evaluator_snapshot is not None
        assert not _bitwise_equal(
            evaluator,
            observation.batch_start_evaluator_snapshot,
        )
        assert observation.optimizer_token == target_sequence
        assert observation.global_step == target_sequence
        assert controller.checkpoint_rejected
    else:
        assert boundary == "commit"
        assert not target_was_issued
        assert not target_was_prepared
        assert not target_was_delivered
        assert not target_was_consumed
        assert not target_is_pending
        assert evaluator["sequence_id"] == target_sequence
        assert evaluator["count"] == target_sequence
        assert observation.optimizer_token == target_sequence
        assert observation.global_step == target_sequence
        _assert_bitwise(
            evaluator["state"],
            committed_evaluator,
            "post_commit_evaluator",
        )


def _assert_numeric_mapping(
    observed: Mapping[str, float],
    expected: Mapping[str, float],
) -> None:
    assert observed.keys() == expected.keys()
    for key in observed:
        assert observed[key] == pytest.approx(
            expected[key],
            abs=NUMERIC_ABS_TOLERANCE,
            rel=NUMERIC_REL_TOLERANCE,
        )


def _descriptor_participants(
    descriptors: tuple[object, ...],
) -> tuple[tuple[tuple[int, ...], tuple[int, ...]], ...]:
    return tuple(
        (
            tuple(descriptor.partition_ids),
            tuple(descriptor.target_seed_ids),
        )
        for descriptor in descriptors
    )


def _assert_resume_equivalent(
    observed: _RunProfile,
    reference: _RunProfile,
    resume: _ResumeObservation,
    boundary: str,
) -> None:
    for field in BITWISE_PROFILE_FIELDS:
        _assert_bitwise(
            getattr(observed, field),
            getattr(reference, field),
            field,
        )
    for field in NUMERIC_PROFILE_FIELDS:
        _assert_numeric_mapping(
            getattr(observed, field),
            getattr(reference, field),
        )
    assert observed.best_epoch_number == reference.best_epoch_number
    if reference.best_monitored_value is None:
        assert observed.best_monitored_value is None
    else:
        assert observed.best_monitored_value == pytest.approx(
            reference.best_monitored_value,
            abs=NUMERIC_ABS_TOLERANCE,
            rel=NUMERIC_REL_TOLERANCE,
        )

    descriptor_count = len(reference.train_descriptors)
    expected_cursor = descriptor_count + int(boundary == "commit")
    assert resume.state["committed_cursor"] == expected_cursor
    assert resume.state["committed_evaluator_sequence"] == expected_cursor
    assert resume.state["committed_evaluator_count"] == expected_cursor
    assert resume.state["committed_global_step"] == expected_cursor
    assert resume.state["committed_sampler_state"]["cursor"] == expected_cursor
    expected_remaining = reference.train_descriptors[
        expected_cursor % descriptor_count :
    ]
    assert resume.remaining_descriptors == expected_remaining
    assert _descriptor_participants(resume.remaining_descriptors) == (
        _descriptor_participants(expected_remaining)
    )

    identity = resume.state["identity"]
    final_identity = reference.sequence_state["identity"]
    for key in (
        "store_content_sha256",
        "partition_book_identity",
        "active_split_tag",
        "fitted_transform_state_key",
        "sampling_strategy_type",
        "sampling_strategy_config",
        "sampling_strategy_fingerprint",
        "phase_descriptor_digest",
        "phase_descriptor_order",
        "sampler_rng_identity",
    ):
        assert identity[key] == final_identity[key]


def test_pending_accumulation_checkpoint_is_rejected() -> None:
    descriptors = _descriptors()
    module = _CommitDataModule()
    model = _CommitModel()
    callback = DataloaderCommitCallback()
    trainer = _FakeTrainer(module, [callback])
    callback.setup(trainer, model, "fit")

    first_id = _deliver(module.sequence_state, descriptors[0])
    first_batch = type("Batch", (), {"sequence_id": first_id})()
    callback.on_train_batch_start(trainer, model, first_batch, 0)
    second_id = _deliver(module.sequence_state, descriptors[1])
    second_batch = type("Batch", (), {"sequence_id": second_id})()
    callback.on_train_batch_start(trainer, model, second_batch, 1)

    with pytest.raises(RuntimeError, match="uncommitted training work"):
        callback._checkpoint_rng_state()

    assert module.sequence_state.pending_group == (first_id, second_id)


def test_evaluator_mutation_cannot_masquerade_as_pristine_consume() -> None:
    descriptor = _descriptors()[0]
    module = _CommitDataModule()
    model = _CommitModel()
    callback = DataloaderCommitCallback()
    trainer = _FakeTrainer(module, [callback])
    callback.setup(trainer, model, "fit")
    sequence_id = _deliver(module.sequence_state, descriptor)
    batch = type("Batch", (), {"sequence_id": sequence_id})()
    callback.on_train_batch_start(trainer, model, batch, 0)
    model.state = {"sum": torch.tensor(1.0)}

    with pytest.raises(RuntimeError, match="uncommitted training work"):
        callback.state_dict()

    assert module.sequence_state.committed_cursor == 0
    assert module.sequence_state.pending_group == (sequence_id,)


def test_optimizer_advanced_checkpoint_is_rejected_until_commit_callback() -> (
    None
):
    descriptors = _descriptors()
    module = _CommitDataModule()
    model = _CommitModel()
    callback = DataloaderCommitCallback()
    trainer = _FakeTrainer(module, [callback])
    callback.setup(trainer, model, "fit")
    sequence_id = _deliver(module.sequence_state, descriptors[0])
    batch = type("Batch", (), {"sequence_id": sequence_id})()
    callback.on_train_batch_start(trainer, model, batch, 0)
    model.sequence_id = model.count = sequence_id
    model.state = {"sum": torch.tensor(1.0)}
    model.dataloader_optimizer_success_token = 1
    trainer.global_step = 1

    with pytest.raises(RuntimeError, match="uncommitted training work"):
        callback.state_dict()

    assert module.sequence_state.committed_cursor == 0
    assert module.sequence_state.committed_global_step == 0
    assert module.sequence_state.pending_group == (sequence_id,)
    callback.on_train_batch_end(trainer, model, None, batch, 0)
    assert module.sequence_state.committed_cursor == sequence_id
    assert module.sequence_state.committed_global_step == sequence_id
    assert module.sequence_state.committed_evaluator_sequence == sequence_id
    assert (
        module.sequence_state.committed_sampler_state["cursor"] == sequence_id
    )
    assert module.sequence_state.pending_group == ()


def _state_tensors(value: object) -> tuple[torch.Tensor, ...]:
    if isinstance(value, torch.Tensor):
        return (value,)
    if isinstance(value, Mapping):
        return tuple(
            tensor
            for item in value.values()
            for tensor in _state_tensors(item)
        )
    if isinstance(value, (list, tuple)):
        return tuple(
            tensor for item in value for tensor in _state_tensors(item)
        )
    return ()


def test_active_online_evaluator_state_round_trips_without_mutable_alias() -> (
    None
):
    context = EvaluationContext(
        split="train",
        pass_kind="fit_epoch",
        policy="online",
        task="classification",
        num_classes=2,
    )
    first = EvaluationBatch(
        outputs=torch.tensor([[3.0, -1.0], [-2.0, 2.0]]),
        targets=torch.tensor([0, 1]),
        num_examples=2,
        context=context,
    )
    second = EvaluationBatch(
        outputs=torch.tensor([[-1.0, 2.0], [4.0, -2.0]]),
        targets=torch.tensor([1, 0]),
        num_examples=2,
        context=context,
    )
    uninterrupted = TBEvaluator(
        "classification",
        num_classes=2,
        metrics=("accuracy",),
    )
    uninterrupted.begin(context)
    uninterrupted.update(first)
    checkpoint = uninterrupted.state_dict()
    checkpoint_tensors = _state_tensors(checkpoint)
    live_tensors = uninterrupted.metric_backend.fixed_state_tensors
    assert checkpoint_tensors
    assert {
        tensor.untyped_storage().data_ptr() for tensor in checkpoint_tensors
    }.isdisjoint(
        tensor.untyped_storage().data_ptr() for tensor in live_tensors
    )

    resumed = TBEvaluator(
        "classification",
        num_classes=2,
        metrics=("accuracy",),
    )
    resumed.load_state_dict(checkpoint, strict=True)
    assert resumed.state == "active"
    assert resumed.context == context
    assert resumed.num_examples == 2
    uninterrupted.update(second)
    resumed.update(second)
    expected = uninterrupted.finalize()
    observed = resumed.finalize()
    assert observed.context == expected.context
    assert observed.num_examples == expected.num_examples == 4
    assert torch.equal(
        observed.metrics["accuracy"], expected.metrics["accuracy"]
    )


def test_evaluator_state_restore_rejects_schema_and_configuration_mismatch() -> (
    None
):
    context = EvaluationContext(
        split="train",
        pass_kind="fit_epoch",
        policy="online",
        task="classification",
        num_classes=2,
    )
    evaluator = TBEvaluator(
        "classification",
        num_classes=2,
        metrics=("accuracy",),
    )
    evaluator.begin(context)
    evaluator.update(
        EvaluationBatch(
            outputs=torch.tensor([[2.0, -1.0]]),
            targets=torch.tensor([0]),
            num_examples=1,
            context=context,
        )
    )
    checkpoint = evaluator.state_dict()
    wrong_format = dict(checkpoint)
    wrong_format["format_version"] = "unknown"
    with pytest.raises(ValueError, match="format_version"):
        evaluator.load_state_dict(wrong_format, strict=True)
    mismatched = TBEvaluator(
        "classification",
        num_classes=2,
        metrics=("f1",),
    )
    with pytest.raises(ValueError, match="metric_names"):
        mismatched.load_state_dict(checkpoint, strict=True)
    with pytest.raises(TypeError, match="strict"):
        mismatched.load_state_dict(checkpoint, strict=False)


def _homogeneous_lifecycle_config(
    source: object,
    run_root: Path,
    *,
    store_path: Path | None = None,
) -> DictConfig:
    cfg = _parquet_cfg(source, run_root, store_path=store_path)
    with open_dict(cfg.dataset.parameters):
        cfg.dataset.parameters.metrics = ["accuracy"]
    return cfg


@pytest.fixture(scope="module")
def homogeneous_lifecycle(
    tmp_path_factory: pytest.TempPathFactory,
) -> _LifecycleCase:
    root = tmp_path_factory.mktemp("typed-graph-resume")
    source = _binary_homogeneous_source(root / "source")
    built_cfg = _homogeneous_lifecycle_config(source, root / "build")
    built = _build(built_cfg)
    store_path = built.prediction_row_adapter.store_path
    built.datamodule.close()

    def config_factory(run_root: Path, path: Path) -> DictConfig:
        return _homogeneous_lifecycle_config(
            source,
            run_root,
            store_path=path,
        )

    reference = _run_complete(
        config_factory(root / "reference", store_path),
        root / "reference" / "trainer",
    )
    return _LifecycleCase(source, store_path, root, reference, config_factory)


@pytest.mark.parametrize("boundary", INTERRUPTION_BOUNDARIES)
def test_homogeneous_cluster_resume_is_exact_at_every_production_boundary(
    homogeneous_lifecycle: _LifecycleCase,
    boundary: str,
) -> None:
    case = homogeneous_lifecycle
    run_root = case.root / boundary
    observed, resume = _run_interrupted(
        case.config_factory(run_root / "crash-cfg", case.store_path),
        case.config_factory(run_root / "resume-cfg", case.store_path),
        run_root,
        boundary,
    )
    _assert_resume_equivalent(observed, case.reference, resume, boundary)
