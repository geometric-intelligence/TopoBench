"""Lightning lifecycle owner for authoritative input-pipeline evidence."""

from __future__ import annotations

import contextlib
import resource
import sys
import warnings
from collections.abc import Mapping, Sequence
from pathlib import Path
from types import MappingProxyType
from typing import Any

from lightning import Callback
from lightning.pytorch.callbacks import EarlyStopping, ModelCheckpoint

from topobench.dataloader.input_monitor import (
    InputMonitor,
    OverflowPolicy,
    ResourceSnapshot,
)
from topobench.profiling.execution_events import (
    ExecutionOperation,
    ExecutionStatus,
    ExecutionSummary,
    summarize_events,
)
from topobench.profiling.local_event_log import FsyncPolicy, LocalEventLog


def create_input_monitor(
    event_log_path: str | Path,
    *,
    event_capacity: int = 4096,
    pending_cuda_capacity: int = 256,
    overflow_policy: OverflowPolicy | str = OverflowPolicy.WARN,
    sample_every_n: int = 10,
    sample_offset: int = 0,
    warmup_steps: int = 20,
    rolling_window_steps: int = 100,
    max_input_stall_fraction: float = 0.05,
    max_consecutive_starved_steps: int = 3,
    patience_windows: int = 2,
    stall_action: OverflowPolicy | str = OverflowPolicy.WARN,
) -> InputMonitor:
    """Create the monitor shared by pipeline construction and its callback."""

    path = Path(event_log_path)

    def resource_snapshot() -> ResourceSnapshot:
        maximum_rss = int(resource.getrusage(resource.RUSAGE_SELF).ru_maxrss)
        if sys.platform != "darwin":
            maximum_rss *= 1024
        gpu_bytes: int | None = None
        import torch

        if torch.cuda.is_available():
            gpu_bytes = int(torch.cuda.memory_allocated())
        final_disk_bytes = path.stat().st_size if path.exists() else 0
        return ResourceSnapshot(
            rss_bytes=maximum_rss,
            gpu_bytes=gpu_bytes,
            final_disk_bytes=final_disk_bytes,
        )

    return InputMonitor(
        resource_reader=resource_snapshot,
        event_capacity=event_capacity,
        pending_cuda_capacity=pending_cuda_capacity,
        sample_every_n=sample_every_n,
        sample_offset=sample_offset,
        overflow_policy=overflow_policy,
        warmup_steps=warmup_steps,
        rolling_window_steps=rolling_window_steps,
        max_input_stall_fraction=max_input_stall_fraction,
        max_consecutive_starved_steps=max_consecutive_starved_steps,
        patience_windows=patience_windows,
        stall_action=stall_action,
        report_reference=path.name,
    )


class InputPipelineCallback(Callback):
    """Own monitor startup, local persistence, aggregation, and cleanup."""

    def __init__(
        self,
        event_log_path: str | Path,
        *,
        max_log_bytes: int = 16 * 1024 * 1024,
        max_log_records: int = 10_000,
        max_rotations: int = 2,
        fsync_policy: FsyncPolicy | str = FsyncPolicy.ROTATION,
        event_capacity: int = 4096,
        pending_cuda_capacity: int = 256,
        overflow_policy: OverflowPolicy | str = OverflowPolicy.WARN,
        sample_every_n: int = 10,
        sample_offset: int = 0,
        warmup_steps: int = 20,
        rolling_window_steps: int = 100,
        max_input_stall_fraction: float = 0.05,
        max_consecutive_starved_steps: int = 3,
        patience_windows: int = 2,
        stall_action: OverflowPolicy | str = OverflowPolicy.WARN,
        max_logger_metrics: int = 64,
        publish_every_n_flushes: int = 1,
        monitor: InputMonitor | None = None,
    ) -> None:
        super().__init__()
        self.event_log_path = Path(event_log_path)
        self.max_log_bytes = self._positive(max_log_bytes, "max_log_bytes")
        self.max_log_records = self._positive(max_log_records, "max_log_records")
        self.max_rotations = self._nonnegative(max_rotations, "max_rotations")
        self.fsync_policy = FsyncPolicy(fsync_policy)
        self.event_capacity = self._positive(event_capacity, "event_capacity")
        self.pending_cuda_capacity = self._positive(
            pending_cuda_capacity,
            "pending_cuda_capacity",
        )
        self.overflow_policy = OverflowPolicy(overflow_policy)
        self.sample_every_n = self._positive(
            sample_every_n,
            "sample_every_n",
        )
        self.sample_offset = self._nonnegative(sample_offset, "sample_offset")
        if self.sample_offset >= self.sample_every_n:
            raise ValueError("sample_offset must be smaller than sample_every_n")
        self.warmup_steps = self._nonnegative(warmup_steps, "warmup_steps")
        self.rolling_window_steps = self._positive(
            rolling_window_steps,
            "rolling_window_steps",
        )
        if (
            isinstance(max_input_stall_fraction, bool)
            or not isinstance(max_input_stall_fraction, (int, float))
            or not 0 <= float(max_input_stall_fraction) <= 1
        ):
            raise ValueError(
                "max_input_stall_fraction must be between zero and one"
            )
        self.max_input_stall_fraction = float(max_input_stall_fraction)
        self.max_consecutive_starved_steps = self._positive(
            max_consecutive_starved_steps,
            "max_consecutive_starved_steps",
        )
        self.patience_windows = self._positive(
            patience_windows,
            "patience_windows",
        )
        self.stall_action = OverflowPolicy(stall_action)
        self.max_logger_metrics = self._positive(
            max_logger_metrics,
            "max_logger_metrics",
        )
        self.publish_every_n_flushes = self._positive(
            publish_every_n_flushes,
            "publish_every_n_flushes",
        )
        if monitor is not None and not isinstance(monitor, InputMonitor):
            raise TypeError("monitor must be an InputMonitor or None")
        expected_monitor_values = {
            "event_capacity": self.event_capacity,
            "pending_cuda_capacity": self.pending_cuda_capacity,
            "sample_every_n": self.sample_every_n,
            "sample_offset": self.sample_offset,
            "overflow_policy": self.overflow_policy,
            "warmup_steps": self.warmup_steps,
            "rolling_window_steps": self.rolling_window_steps,
            "max_input_stall_fraction": self.max_input_stall_fraction,
            "max_consecutive_starved_steps": self.max_consecutive_starved_steps,
            "patience_windows": self.patience_windows,
            "stall_action": self.stall_action,
            "report_reference": self.event_log_path.name,
        }
        if monitor is not None:
            for name, expected in expected_monitor_values.items():
                if getattr(monitor, name) != expected:
                    raise ValueError(
                        f"adopted monitor {name} does not match callback config"
                    )
        self._monitor = monitor or create_input_monitor(
            self.event_log_path,
            event_capacity=self.event_capacity,
            pending_cuda_capacity=self.pending_cuda_capacity,
            sample_every_n=self.sample_every_n,
            sample_offset=self.sample_offset,
            overflow_policy=self.overflow_policy,
            warmup_steps=self.warmup_steps,
            rolling_window_steps=self.rolling_window_steps,
            max_input_stall_fraction=self.max_input_stall_fraction,
            max_consecutive_starved_steps=self.max_consecutive_starved_steps,
            patience_windows=self.patience_windows,
            stall_action=self.stall_action,
        )
        self._event_log: LocalEventLog | None = None
        self._summary: ExecutionSummary | None = None
        self._flush_count = 0
        self._closed = False
        self._attached_model: object | None = None
        self._attached_datamodule: object | None = None

    @staticmethod
    def _positive(value: object, name: str) -> int:
        if isinstance(value, bool) or not isinstance(value, int):
            raise TypeError(f"{name} must be an integer")
        if value < 1:
            raise ValueError(f"{name} must be positive")
        return value

    @staticmethod
    def _nonnegative(value: object, name: str) -> int:
        if isinstance(value, bool) or not isinstance(value, int):
            raise TypeError(f"{name} must be an integer")
        if value < 0:
            raise ValueError(f"{name} must be non-negative")
        return value

    @property
    def monitor(self) -> InputMonitor:
        return self._monitor

    @property
    def event_log(self) -> LocalEventLog:
        if self._event_log is None:
            raise RuntimeError("InputPipelineCallback has not started")
        return self._event_log

    @property
    def summary(self) -> ExecutionSummary | None:
        return self._summary

    @property
    def closed(self) -> bool:
        return self._closed

    @property
    def provenance_summary(self) -> Mapping[str, object] | None:
        """Expose the existing summary record for Task13 provenance wiring."""

        if self._summary is None:
            return None
        record = self._summary.as_record()
        aggregates = record["aggregates"]
        assert isinstance(aggregates, list)
        record["aggregates"] = tuple(
            MappingProxyType(dict(aggregate))
            for aggregate in aggregates
        )
        return MappingProxyType(record)

    def _checkpoint_metric_guard(self, trainer: object) -> None:
        callbacks = getattr(trainer, "callbacks", ())
        for callback in callbacks:
            if callback is self or not isinstance(
                callback,
                (ModelCheckpoint, EarlyStopping),
            ):
                continue
            monitor = callback.monitor
            if isinstance(monitor, str) and monitor.startswith("system/"):
                raise ValueError(
                    "system/* resource metrics are prohibited as checkpoint "
                    "or scientific selectors"
                )

    @staticmethod
    def _cuda_event_factory(model: object) -> Any:
        device = getattr(model, "device", None)
        if getattr(device, "type", None) != "cuda":
            return None
        import torch

        if not torch.cuda.is_available():
            return None
        return lambda: torch.cuda.Event(enable_timing=True)


    def _start(self, trainer: object, model: object) -> None:
        if self._closed:
            raise RuntimeError("InputPipelineCallback cannot restart after teardown")
        self._checkpoint_metric_guard(trainer)
        cuda_event_factory = self._cuda_event_factory(model)
        if (
            self._monitor.cuda_event_factory is None
            and cuda_event_factory is not None
        ):
            self._monitor.cuda_event_factory = cuda_event_factory
        if self._event_log is None:
            self._event_log = LocalEventLog(
                self.event_log_path,
                max_bytes=self.max_log_bytes,
                max_records=self.max_log_records,
                max_rotations=self.max_rotations,
                fsync_policy=self.fsync_policy,
                allowed_root=self.event_log_path.parent,
            )
        datamodule = getattr(trainer, "datamodule", None)
        if datamodule is not None:
            attach = getattr(datamodule, "set_execution_monitor", None)
            if callable(attach):
                attach(self._monitor)
            else:
                setattr(datamodule, "execution_monitor", self._monitor)
            self._attached_datamodule = datamodule
        setattr(model, "execution_monitor", self._monitor)
        self._attached_model = model

    def setup(self, trainer: object, pl_module: object, stage: str) -> None:
        """Attach optional monitor references before data/model execution."""

        self._start(trainer, pl_module)

    def on_fit_start(self, trainer: object, pl_module: object) -> None:
        self._start(trainer, pl_module)

    @staticmethod
    def _loggers(trainer: object) -> tuple[object, ...]:
        loggers = getattr(trainer, "loggers", None)
        if isinstance(loggers, Sequence) and not isinstance(loggers, (str, bytes)):
            return tuple(logger for logger in loggers if logger is not None)
        logger = getattr(trainer, "logger", None)
        return () if logger is None else (logger,)

    def _logger_metrics(self, summary: ExecutionSummary) -> dict[str, int | float]:
        metrics: dict[str, int | float] = {
            "system/events/dropped": summary.dropped_event_count,
            "system/events/rotated_files": summary.rotated_file_count,
            "system/events/aggregate_pairs": len(summary.aggregates),
        }
        summary_values = {
            "system/summary/conversion_records_per_second": (
                summary.conversion_records_per_second
            ),
            "system/summary/conversion_bytes_per_second": (
                summary.conversion_bytes_per_second
            ),
            "system/summary/selected_read_records_per_second": (
                summary.selected_read_records_per_second
            ),
            "system/summary/selected_read_bytes_per_second": (
                summary.selected_read_bytes_per_second
            ),
            "system/summary/native_assembly_records_per_second": (
                summary.native_assembly_records_per_second
            ),
            "system/summary/native_assembly_bytes_per_second": (
                summary.native_assembly_bytes_per_second
            ),
            "system/summary/achieved_input_stall_fraction": (
                summary.achieved_input_stall_fraction
            ),
            "system/summary/host_queue_peak_depth": summary.host_queue_peak_depth,
            "system/summary/host_queue_peak_bytes": summary.host_queue_peak_bytes,
            "system/summary/device_queue_peak_depth": (
                summary.device_queue_peak_depth
            ),
            "system/summary/device_queue_peak_bytes": (
                summary.device_queue_peak_bytes
            ),
            "system/summary/rss_peak_bytes": summary.rss_peak_bytes,
            "system/summary/rss_peak_delta_bytes": summary.rss_peak_delta_bytes,
            "system/summary/pinned_peak_bytes": summary.pinned_peak_bytes,
            "system/summary/pinned_peak_delta_bytes": (
                summary.pinned_peak_delta_bytes
            ),
            "system/summary/gpu_peak_bytes": summary.gpu_peak_bytes,
            "system/summary/gpu_peak_delta_bytes": summary.gpu_peak_delta_bytes,
            "system/summary/temp_disk_peak_bytes": summary.temp_disk_peak_bytes,
            "system/summary/temp_disk_peak_delta_bytes": (
                summary.temp_disk_peak_delta_bytes
            ),
            "system/summary/final_disk_peak_bytes": summary.final_disk_peak_bytes,
            "system/summary/final_disk_peak_delta_bytes": (
                summary.final_disk_peak_delta_bytes
            ),
        }
        for key, value in summary_values.items():
            if value is None:
                continue
            if len(metrics) >= self.max_logger_metrics:
                return metrics
            metrics[key] = value
        for aggregate in summary.aggregates:
            prefix = f"system/{aggregate.operation.value}/{aggregate.status.value}"
            values: tuple[tuple[str, int | float], ...] = (
                (f"{prefix}/count", aggregate.count),
                (f"{prefix}/minimum_ns", aggregate.minimum_ns),
                (f"{prefix}/maximum_ns", aggregate.maximum_ns),
                (f"{prefix}/mean_ns", aggregate.mean_ns),
                (f"{prefix}/p50_ns", aggregate.p50_ns),
                (f"{prefix}/p95_ns", aggregate.p95_ns),
                (f"{prefix}/p99_ns", aggregate.p99_ns),
            )
            for key, value in values:
                if len(metrics) >= self.max_logger_metrics:
                    return metrics
                metrics[key] = value
        return metrics

    def _publish(self, trainer: object, summary: ExecutionSummary) -> None:
        metrics = self._logger_metrics(summary)
        step = getattr(trainer, "global_step", None)
        if isinstance(step, bool) or not isinstance(step, int):
            step = None
        for logger in self._loggers(trainer):
            log_metrics = getattr(logger, "log_metrics", None)
            if not callable(log_metrics):
                continue
            try:
                log_metrics(metrics, step=step)
            except Exception as error:
                warnings.warn(
                    f"aggregate system logger failed: {type(error).__name__}",
                    RuntimeWarning,
                    stacklevel=2,
                )




    def _update_summary(self, trainer: object) -> ExecutionSummary:
        summary = summarize_events(
            self.event_log.retained_events,
            sample_every_n=self.sample_every_n,
            sample_offset=self.sample_offset,
            dropped_event_count=(
                self.monitor.dropped_event_count
                + self.event_log.evicted_event_count
            ),
            rotated_file_count=self.event_log.rotated_file_count,
        )
        self._summary = summary
        if self._attached_datamodule is not None:
            setattr(
                self._attached_datamodule,
                "execution_summary",
                summary,
            )
        return summary

    def flush(self, trainer: object, *, publish: bool = True) -> ExecutionSummary:
        """Persist local events first, then optionally publish bounded aggregates."""

        if self._monitor is None or self._event_log is None:
            raise RuntimeError("InputPipelineCallback has not started")
        for event in self.monitor.drain():
            self.event_log.append(event)
        summary = self._update_summary(trainer)
        self._flush_count += 1
        if publish and self._flush_count % self.publish_every_n_flushes == 0:
            self._publish(trainer, summary)
        return summary

    def on_train_batch_end(
        self,
        trainer: object,
        pl_module: object,
        outputs: object,
        batch: object,
        batch_idx: int,
    ) -> None:
        self.monitor.poll()
        self.flush(trainer)

    def on_validation_batch_end(
        self,
        trainer: object,
        pl_module: object,
        outputs: object,
        batch: object,
        batch_idx: int,
        dataloader_idx: int = 0,
    ) -> None:
        self.monitor.poll()
        self.flush(trainer, publish=False)

    def on_test_batch_end(
        self,
        trainer: object,
        pl_module: object,
        outputs: object,
        batch: object,
        batch_idx: int,
        dataloader_idx: int = 0,
    ) -> None:
        self.monitor.poll()
        self.flush(trainer, publish=False)

    def on_train_epoch_end(self, trainer: object, pl_module: object) -> None:
        self.flush(trainer)

    def on_validation_epoch_end(self, trainer: object, pl_module: object) -> None:
        self.flush(trainer)

    def on_test_epoch_end(self, trainer: object, pl_module: object) -> None:
        self.flush(trainer)

    def on_save_checkpoint(
        self,
        trainer: object,
        pl_module: object,
        checkpoint: dict[str, Any],
    ) -> None:
        if self._monitor is None:
            return
        self.monitor.record(
            ExecutionOperation.CHECKPOINT,
            phase="checkpoint_save",
            epoch=self._optional_trainer_integer(trainer, "current_epoch"),
            global_step=self._optional_trainer_integer(trainer, "global_step"),
            evidence={"state": "saved"},
        )
        self.flush(trainer, publish=False)

    def on_load_checkpoint(
        self,
        trainer: object,
        pl_module: object,
        checkpoint: Mapping[str, Any],
    ) -> None:
        if self._monitor is None:
            return
        self.monitor.record(
            ExecutionOperation.CHECKPOINT,
            phase="checkpoint_load",
            epoch=self._optional_trainer_integer(trainer, "current_epoch"),
            global_step=self._optional_trainer_integer(trainer, "global_step"),
            evidence={"state": "loaded"},
        )
        self.flush(trainer, publish=False)

    @staticmethod
    def _optional_trainer_integer(trainer: object, name: str) -> int | None:
        value = getattr(trainer, name, None)
        return None if isinstance(value, bool) or not isinstance(value, int) else value

    def on_train_end(self, trainer: object, pl_module: object) -> None:
        if self._monitor is None:
            return
        self.monitor.record(
            ExecutionOperation.ARTIFACT,
            phase="training_end",
            epoch=self._optional_trainer_integer(trainer, "current_epoch"),
            global_step=self._optional_trainer_integer(trainer, "global_step"),
            evidence={"artifact": "execution-summary"},
        )
        self.flush(trainer)
        if self._summary is not None and self._attached_model is not None:
            setattr(
                self._attached_model,
                "execution_summary",
                self._summary,
            )

    def on_exception(
        self,
        trainer: object,
        pl_module: object,
        exception: BaseException,
    ) -> None:
        if self._closed or self._monitor is None:
            return
        with contextlib.suppress(Exception):
            self.monitor.record(
                ExecutionOperation.ARTIFACT,
                phase="exception",
                status=ExecutionStatus.ERROR,
                evidence={"error_type": type(exception).__name__},
            )
            self.flush(trainer, publish=False)
        self._close(trainer)

    def _detach(self) -> None:
        datamodule = self._attached_datamodule
        if (
            datamodule is not None
            and getattr(datamodule, "execution_monitor", None)
            is self._monitor
        ):
            detach = getattr(datamodule, "set_execution_monitor", None)
            if callable(detach):
                detach(None)
            else:
                setattr(datamodule, "execution_monitor", None)
        model = self._attached_model
        if (
            model is not None
            and getattr(model, "execution_monitor", None) is self._monitor
        ):
            setattr(model, "execution_monitor", None)
        self._attached_datamodule = None
        self._attached_model = None

    def _close(self, trainer: object) -> None:
        if self._closed:
            return
        try:
            if self._monitor is not None:
                self._monitor.close()
                if self._event_log is not None:
                    for event in self._monitor.drain():
                        self._event_log.append(event)
                    self._update_summary(trainer)
        finally:
            self._detach()
            if self._event_log is not None:
                self._event_log.close()
            self._closed = True

    def teardown(self, trainer: object, pl_module: object, stage: str) -> None:
        self._close(trainer)


__all__ = ["InputPipelineCallback"]
