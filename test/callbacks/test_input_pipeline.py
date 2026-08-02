"""Lifecycle and logger-boundary contracts for input-pipeline evidence."""

from __future__ import annotations

from pathlib import Path
from types import SimpleNamespace
from unittest.mock import Mock

import pytest
from lightning.pytorch.callbacks import ModelCheckpoint
from omegaconf import OmegaConf

from topobench.callbacks.input_pipeline import InputPipelineCallback
from topobench.data.stores.qualification_checks import (
    QualificationFailure,
    validate_store,
)
from topobench.profiling.execution_events import (
    ExecutionOperation,
    ExecutionStatus,
    summarize_events,
)
from topobench.profiling.local_event_log import LocalEventLog


class _Logger:
    def __init__(self, *, fail: bool = False) -> None:
        self.fail = fail
        self.calls: list[tuple[dict[str, int | float], int | None]] = []

    def log_metrics(
        self,
        metrics: dict[str, int | float],
        step: int | None = None,
    ) -> None:
        if self.fail:
            raise RuntimeError("remote logger unavailable")
        self.calls.append((metrics, step))


def _trainer(
    tmp_path: Path,
    *,
    loggers: list[object] | None = None,
    callbacks: list[object] | None = None,
) -> SimpleNamespace:
    data_module = SimpleNamespace(execution_monitor=None)
    return SimpleNamespace(
        datamodule=data_module,
        callbacks=list(callbacks or ()),
        loggers=list(loggers or ()),
        logger=(loggers or [None])[0],
        global_step=4,
        current_epoch=2,
        default_root_dir=str(tmp_path),
        checkpoint_callback=None,
    )


def _callback(tmp_path: Path, **changes: object) -> InputPipelineCallback:
    values: dict[str, object] = {
        "event_log_path": tmp_path / "execution" / "events.jsonl",
        "max_log_bytes": 128_000,
        "max_log_records": 128,
        "max_rotations": 1,
        "fsync_policy": "always",
        "event_capacity": 32,
        "pending_cuda_capacity": 4,
        "overflow_policy": "error",
        "sample_every_n": 1,
        "sample_offset": 0,
        "max_logger_metrics": 32,
    }
    values.update(changes)
    return InputPipelineCallback(**values)


def test_callback_owns_attach_flush_aggregate_logger_and_idempotent_teardown(
    tmp_path: Path,
) -> None:
    logger = _Logger()
    trainer = _trainer(tmp_path, loggers=[logger])
    model = SimpleNamespace(execution_monitor=None)
    callback = _callback(tmp_path)
    eager_monitor = callback.monitor
    trainer.callbacks.append(callback)

    callback.setup(trainer, model, "fit")
    assert callback.monitor is eager_monitor
    assert trainer.datamodule.execution_monitor is callback.monitor
    assert model.execution_monitor is callback.monitor
    callback.monitor.record(
        ExecutionOperation.SELECTED_READ,
        phase="fit",
        split="train",
        duration_ns=17,
        descriptor_sequence=1,
        descriptor_identity="private-descriptor",
        evidence={"token": "must-not-leak", "safe_count": 3},
    )
    callback.on_train_batch_end(trainer, model, None, object(), 0)
    callback.on_save_checkpoint(trainer, model, {})
    callback.on_train_end(trainer, model)

    events = LocalEventLog(
        tmp_path / "execution" / "events.jsonl",
        max_bytes=128_000,
        max_records=128,
        max_rotations=1,
    ).load()
    assert {event.operation for event in events} >= {
        ExecutionOperation.SELECTED_READ,
        ExecutionOperation.CHECKPOINT,
        ExecutionOperation.ARTIFACT,
    }
    assert "private-descriptor" not in repr(events)
    assert "must-not-leak" not in repr(events)
    assert logger.calls
    for metrics, step in logger.calls:
        assert step == 4
        assert len(metrics) <= 32
        assert metrics
        assert all(key.startswith("system/") for key in metrics)
        assert all(type(value) in {int, float} for value in metrics.values())
    assert any(
        key.startswith("system/summary/")
        for metrics, _ in logger.calls
        for key in metrics
    )

    provenance = callback.provenance_summary
    assert provenance is not None
    aggregates = provenance["aggregates"]
    assert isinstance(aggregates, tuple)
    with pytest.raises(TypeError):
        aggregates[0]["count"] = 999  # type: ignore[index]

    callback.teardown(trainer, model, "fit")
    callback.on_exception(trainer, model, RuntimeError("late error"))
    callback.teardown(trainer, model, "fit")
    assert callback.closed
    assert trainer.datamodule.execution_monitor is None
    assert model.execution_monitor is None


def test_local_log_remains_authoritative_when_remote_logger_fails(
    tmp_path: Path,
) -> None:
    trainer = _trainer(tmp_path, loggers=[_Logger(fail=True)])
    model = SimpleNamespace(execution_monitor=None)
    callback = _callback(tmp_path)
    callback.setup(trainer, model, "fit")
    callback.monitor.record(
        ExecutionOperation.MODEL_COMPUTE,
        phase="fit",
        split="train",
        duration_ns=31,
    )

    callback.flush(trainer)
    events = callback.event_log.load()
    assert len(events) == 1
    assert events[0].operation is ExecutionOperation.MODEL_COMPUTE
    callback.teardown(trainer, model, "fit")


def test_qualification_failure_propagates_after_emitting_stable_check_evidence(
    tmp_path: Path,
) -> None:
    trainer = _trainer(tmp_path)
    model = SimpleNamespace(execution_monitor=None)
    callback = _callback(tmp_path)
    callback.setup(trainer, model, "fit")
    report_path = tmp_path / "reports" / "qualification.json"

    with pytest.raises(QualificationFailure) as caught:
        validate_store(
            tmp_path / "missing-store",
            report_path=report_path,
            execution_monitor=callback.monitor,
        )
    callback.flush(trainer, publish=False)

    assert caught.value.check_id == "MANIFEST-001"
    event = callback.event_log.load()[-1]
    assert event.operation is ExecutionOperation.VALIDATION
    assert event.status is ExecutionStatus.ERROR
    assert event.check_id == "MANIFEST-001"
    assert event.remediation
    assert event.report_reference == "qualification.json"
    assert event.evidence
    callback.teardown(trainer, model, "fit")


def test_batch_flushes_increment_summary_without_replaying_local_log(
    tmp_path: Path,
    monkeypatch: pytest.MonkeyPatch,
) -> None:
    trainer = _trainer(tmp_path)
    model = SimpleNamespace(execution_monitor=None)
    callback = _callback(tmp_path)
    callback.setup(trainer, model, "fit")
    load = Mock(wraps=callback.event_log.load)
    monkeypatch.setattr(callback.event_log, "load", load)

    for sequence in range(1, 5):
        callback.monitor.record(
            ExecutionOperation.SELECTED_READ,
            phase="fit",
            split="train",
            descriptor_sequence=sequence,
            descriptor_identity=f"private-{sequence}",
            duration_ns=sequence,
        )
        summary = callback.flush(trainer, publish=False)

    assert load.call_count == 0
    aggregate = summary.for_operation(
        ExecutionOperation.SELECTED_READ,
        ExecutionStatus.SUCCESS,
    )
    assert aggregate.count == 4
    callback.teardown(trainer, model, "fit")



def test_summary_matches_authoritative_retained_window_after_byte_eviction(
    tmp_path: Path,
    monkeypatch: pytest.MonkeyPatch,
) -> None:
    trainer = _trainer(tmp_path)
    model = SimpleNamespace(execution_monitor=None)
    callback = _callback(
        tmp_path,
        max_log_bytes=4_000,
        max_log_records=3,
        max_rotations=1,
    )
    callback.setup(trainer, model, "fit")
    load = Mock(wraps=callback.event_log.load)
    monkeypatch.setattr(callback.event_log, "load", load)

    for sequence, padding in enumerate((10, 400, 20, 700, 30, 900), 1):
        callback.monitor.record(
            ExecutionOperation.SELECTED_READ,
            phase="fit",
            split="train",
            descriptor_sequence=sequence,
            descriptor_identity=f"private-{sequence}",
            duration_ns=sequence,
            evidence={"padding": "x" * padding},
        )
        summary = callback.flush(trainer, publish=False)

    assert load.call_count == 0
    assert callback.event_log.retained_events
    loaded = callback.event_log.load()
    assert load.call_count == 1
    assert callback.event_log.retained_events == loaded
    expected = summarize_events(
        loaded,
        sample_every_n=callback.sample_every_n,
        sample_offset=callback.sample_offset,
        dropped_event_count=summary.dropped_event_count,
        rotated_file_count=callback.event_log.rotated_file_count,
    )
    assert summary.evidence_digest == expected.evidence_digest
    assert summary.aggregates == expected.aggregates
    callback.teardown(trainer, model, "fit")



def test_system_metrics_are_prohibited_as_checkpoint_scientific_selector(
    tmp_path: Path,
) -> None:
    checkpoint = ModelCheckpoint(monitor="system/selected_read/p95_ns")
    trainer = _trainer(tmp_path, callbacks=[checkpoint])
    callback = _callback(tmp_path)
    model = SimpleNamespace(execution_monitor=None)
    with pytest.raises(ValueError, match="system/.*checkpoint"):
        callback.setup(trainer, model, "fit")


def test_callback_config_is_selectable_without_default_composition() -> None:
    root = Path(__file__).parents[2]
    config = OmegaConf.load(root / "configs" / "callbacks" / "input_pipeline.yaml")
    assert config.input_pipeline._target_ == (
        "topobench.callbacks.input_pipeline.InputPipelineCallback"
    )
    assert config.input_pipeline.warmup_steps == 20
    assert config.input_pipeline.rolling_window_steps == 100
    assert config.input_pipeline.max_input_stall_fraction == 0.05
    assert config.input_pipeline.max_consecutive_starved_steps == 3
    assert config.input_pipeline.patience_windows == 2
    assert config.input_pipeline.stall_action == "warn"
    defaults = OmegaConf.load(root / "configs" / "callbacks" / "default.yaml")
    assert "input_pipeline" not in OmegaConf.to_container(defaults, resolve=False)[
        "defaults"
    ]
