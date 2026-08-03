"""Low-allocation asynchronous execution monitoring for input pipelines."""

from __future__ import annotations

import contextlib
import resource
import sys
import threading
import time
import warnings
from collections import deque
from collections.abc import Callable, Mapping, Sequence
from dataclasses import dataclass, replace
from datetime import UTC, datetime
from enum import StrEnum
from pathlib import Path
from types import MappingProxyType

from topobench.profiling.execution_events import (
    ExecutionEvent,
    ExecutionOperation,
    ExecutionStatus,
    Primitive,
    PrimitiveMap,
    descriptor_digest,
    redact_mapping,
)

WallClock = Callable[[], datetime]
MonotonicClock = Callable[[], int]
ResourceReader = Callable[[], "ResourceSnapshot"]
CudaEventFactory = Callable[[], object]


class OverflowPolicy(StrEnum):
    """Explicit handling for bounded event and pending-timing queues."""

    WARN = "warn"
    ERROR = "error"


class MonitorOverflowError(RuntimeError):
    """Raised when a configured error policy encounters bounded overflow."""


class InputStallError(RuntimeError):
    """Raised after the configured number of sustained starvation windows."""


def _integer(
    value: object,
    name: str,
    *,
    minimum: int | None = 0,
) -> int:
    if isinstance(value, bool) or not isinstance(value, int):
        raise TypeError(f"{name} must be an integer")
    if minimum is not None and value < minimum:
        raise ValueError(f"{name} must be at least {minimum}")
    return value


def _optional_integer(
    value: object,
    name: str,
    *,
    minimum: int | None = 0,
) -> int | None:
    return None if value is None else _integer(value, name, minimum=minimum)


@dataclass(frozen=True, slots=True)
class ResourceSnapshot:
    """One sampled process/device/disk resource observation in bytes."""

    rss_bytes: int | None = None
    pinned_bytes: int | None = None
    gpu_bytes: int | None = None
    temp_disk_bytes: int | None = None
    final_disk_bytes: int | None = None

    def __post_init__(self) -> None:
        for name in (
            "rss_bytes",
            "pinned_bytes",
            "gpu_bytes",
            "temp_disk_bytes",
            "final_disk_bytes",
        ):
            object.__setattr__(
                self,
                name,
                _optional_integer(getattr(self, name), name),
            )


@dataclass(frozen=True, slots=True)
class QueueSnapshot:
    """Configured and current host/device queue occupancy."""

    host_configured_depth: int | None = None
    host_depth: int | None = None
    host_configured_bytes: int | None = None
    host_bytes: int | None = None
    device_configured_depth: int | None = None
    device_depth: int | None = None
    device_configured_bytes: int | None = None
    device_bytes: int | None = None

    def __post_init__(self) -> None:
        for name in (
            "host_configured_depth",
            "host_depth",
            "host_configured_bytes",
            "host_bytes",
            "device_configured_depth",
            "device_depth",
            "device_configured_bytes",
            "device_bytes",
        ):
            object.__setattr__(
                self,
                name,
                _optional_integer(getattr(self, name), name),
            )
        for current, configured in (
            (self.host_depth, self.host_configured_depth),
            (self.host_bytes, self.host_configured_bytes),
            (self.device_depth, self.device_configured_depth),
            (self.device_bytes, self.device_configured_bytes),
        ):
            if (
                current is not None
                and configured is not None
                and current > configured
            ):
                raise ValueError(
                    "current queue occupancy exceeds configured bound"
                )


@dataclass(frozen=True, slots=True)
class OperationToken:
    """Opaque immutable timing token that never references a caller batch."""

    serial: int
    operation: ExecutionOperation
    phase: str
    split: str | None
    wall_time_utc: str
    start_ns: int
    epoch: int | None
    global_step: int | None
    descriptor_sequence: int | None
    descriptor_digest: str | None
    sampled: bool
    resource_before: ResourceSnapshot | None
    queue: QueueSnapshot | None
    cuda_start: object | None
    cuda_end: object | None
    evidence: PrimitiveMap


@dataclass(frozen=True, slots=True)
class _Completion:
    end_ns: int
    status: ExecutionStatus
    node_count: int | None
    edge_count: int | None
    row_count: int | None
    example_count: int | None
    unique_storage_bytes: int | None
    queue: QueueSnapshot | None
    resource_after: ResourceSnapshot | None
    check_id: str | None
    check_passed: bool | None
    check_expected: Primitive | PrimitiveMap
    check_observed: Primitive | PrimitiveMap
    evidence: PrimitiveMap
    remediation: str | None
    report_reference: str | None


@dataclass(frozen=True, slots=True)
class _PendingCuda:
    token: OperationToken
    completion: _Completion


@dataclass(slots=True)
class _StepTiming:
    host_wait_ns: int = 0
    model_compute_ns: int = 0
    optimizer_ns: int = 0
    model_seen: bool = False
    optimizer_seen: bool = False


def _default_wall() -> datetime:
    return datetime.now(UTC)


def _default_resources() -> ResourceSnapshot:
    maximum_rss = resource.getrusage(resource.RUSAGE_SELF).ru_maxrss
    rss_bytes = int(maximum_rss)
    # Linux reports KiB while Darwin reports bytes.
    if sys.platform != "darwin":
        rss_bytes *= 1024
    return ResourceSnapshot(rss_bytes=rss_bytes)


def _utc_text(value: datetime) -> str:
    if not isinstance(value, datetime):
        raise TypeError("wall_clock must return datetime")
    if value.tzinfo is None or value.utcoffset() is None:
        raise ValueError("wall_clock must return timezone-aware UTC time")
    utc = value.astimezone(UTC)
    return utc.isoformat(timespec="microseconds").replace("+00:00", "Z")


def _delta(after: int | None, before: int | None) -> int | None:
    return None if after is None or before is None else after - before


def _queue_values(queue: QueueSnapshot | None) -> dict[str, int | None]:
    if queue is None:
        return {
            "host_queue_configured_depth": None,
            "host_queue_depth": None,
            "host_queue_configured_bytes": None,
            "host_queue_bytes": None,
            "device_queue_configured_depth": None,
            "device_queue_depth": None,
            "device_queue_configured_bytes": None,
            "device_queue_bytes": None,
        }
    return {
        "host_queue_configured_depth": queue.host_configured_depth,
        "host_queue_depth": queue.host_depth,
        "host_queue_configured_bytes": queue.host_configured_bytes,
        "host_queue_bytes": queue.host_bytes,
        "device_queue_configured_depth": queue.device_configured_depth,
        "device_queue_depth": queue.device_depth,
        "device_queue_configured_bytes": queue.device_configured_bytes,
        "device_queue_bytes": queue.device_bytes,
    }


def _resource_values(
    before: ResourceSnapshot | None,
    after: ResourceSnapshot | None,
) -> dict[str, int | None]:
    if after is None:
        return {
            "rss_bytes": None,
            "rss_delta_bytes": None,
            "pinned_bytes": None,
            "pinned_delta_bytes": None,
            "gpu_bytes": None,
            "gpu_delta_bytes": None,
            "temp_disk_bytes": None,
            "temp_disk_delta_bytes": None,
            "final_disk_bytes": None,
            "final_disk_delta_bytes": None,
        }
    return {
        "rss_bytes": after.rss_bytes,
        "rss_delta_bytes": _delta(
            after.rss_bytes, None if before is None else before.rss_bytes
        ),
        "pinned_bytes": after.pinned_bytes,
        "pinned_delta_bytes": _delta(
            after.pinned_bytes, None if before is None else before.pinned_bytes
        ),
        "gpu_bytes": after.gpu_bytes,
        "gpu_delta_bytes": _delta(
            after.gpu_bytes, None if before is None else before.gpu_bytes
        ),
        "temp_disk_bytes": after.temp_disk_bytes,
        "temp_disk_delta_bytes": _delta(
            after.temp_disk_bytes,
            None if before is None else before.temp_disk_bytes,
        ),
        "final_disk_bytes": after.final_disk_bytes,
        "final_disk_delta_bytes": _delta(
            after.final_disk_bytes,
            None if before is None else before.final_disk_bytes,
        ),
    }


def _safe_check_value(value: object, name: str) -> Primitive | PrimitiveMap:
    if value is None or type(value) in {bool, int, float}:
        return value  # type: ignore[return-value]
    if isinstance(value, str):
        return redact_mapping({name: value})[name]
    if isinstance(value, Path):
        return value.name
    if isinstance(value, Mapping):
        return redact_mapping(value)
    if isinstance(value, Sequence) and not isinstance(
        value, (str, bytes, bytearray)
    ):
        return MappingProxyType({"count": len(value)})
    return f"[{type(value).__name__}]"


class InputMonitor:
    """Bounded nonblocking event capture with asynchronous CUDA completion."""

    def __init__(
        self,
        *,
        wall_clock: WallClock = _default_wall,
        monotonic_ns: MonotonicClock = time.monotonic_ns,
        resource_reader: ResourceReader = _default_resources,
        cuda_event_factory: CudaEventFactory | None = None,
        event_capacity: int = 4096,
        pending_cuda_capacity: int = 256,
        sample_every_n: int = 1,
        sample_offset: int = 0,
        overflow_policy: OverflowPolicy | str = OverflowPolicy.WARN,
        warmup_steps: int = 20,
        rolling_window_steps: int = 100,
        max_input_stall_fraction: float = 0.05,
        max_consecutive_starved_steps: int = 3,
        patience_windows: int = 2,
        stall_action: OverflowPolicy | str = OverflowPolicy.WARN,
        report_reference: str = "execution-events.jsonl",
    ) -> None:
        if not callable(wall_clock) or not callable(monotonic_ns):
            raise TypeError("wall and monotonic clocks must be callable")
        if not callable(resource_reader):
            raise TypeError("resource_reader must be callable")
        if cuda_event_factory is not None and not callable(cuda_event_factory):
            raise TypeError("cuda_event_factory must be callable")
        try:
            policy = OverflowPolicy(overflow_policy)
        except (TypeError, ValueError) as error:
            raise ValueError(
                "overflow_policy must be warn or error"
            ) from error
        self._wall_clock = wall_clock
        self._monotonic_ns = monotonic_ns
        self._resource_reader = resource_reader
        self.cuda_event_factory = cuda_event_factory
        self.event_capacity = _integer(
            event_capacity, "event_capacity", minimum=1
        )
        self.pending_cuda_capacity = _integer(
            pending_cuda_capacity,
            "pending_cuda_capacity",
            minimum=1,
        )
        self.sample_every_n = _integer(
            sample_every_n, "sample_every_n", minimum=1
        )
        self.sample_offset = _integer(sample_offset, "sample_offset")
        if self.sample_offset >= self.sample_every_n:
            raise ValueError(
                "sample_offset must be smaller than sample_every_n"
            )
        self.overflow_policy = policy
        self.warmup_steps = _integer(warmup_steps, "warmup_steps")
        self.rolling_window_steps = _integer(
            rolling_window_steps,
            "rolling_window_steps",
            minimum=1,
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
        self.max_consecutive_starved_steps = _integer(
            max_consecutive_starved_steps,
            "max_consecutive_starved_steps",
            minimum=1,
        )
        self.patience_windows = _integer(
            patience_windows,
            "patience_windows",
            minimum=1,
        )
        try:
            self.stall_action = OverflowPolicy(stall_action)
        except (TypeError, ValueError) as error:
            raise ValueError("stall_action must be warn or error") from error
        if not isinstance(report_reference, str) or not report_reference:
            raise ValueError("report_reference must be a non-empty string")
        self.report_reference = Path(report_reference).name
        self._events: deque[ExecutionEvent] = deque()
        self._pending: deque[_PendingCuda] = deque()
        self._open_tokens: set[int] = set()
        self._ready_times: dict[str, int] = {}
        self._lock = threading.RLock()
        self._serial = 0
        self._sample_serial = 0
        self._last_compute_end_ns: int | None = None
        self.dropped_event_count = 0
        self._step_timings: dict[int, _StepTiming] = {}
        self._stall_window: deque[tuple[int, int, int]] = deque(
            maxlen=self.rolling_window_steps
        )
        self._completed_steps = 0
        self._steps_since_stall_window = 0
        self._bad_stall_windows = 0
        self.dropped_cuda_timing_count = 0
        self._closed = False

    @property
    def pending_cuda_count(self) -> int:
        with self._lock:
            return len(self._pending)

    @property
    def closed(self) -> bool:
        return self._closed

    def _ensure_open(self) -> None:
        if self._closed:
            raise RuntimeError("InputMonitor is closed")

    def _sampled(self, descriptor_sequence: int | None) -> bool:
        if descriptor_sequence is not None:
            return (
                descriptor_sequence % self.sample_every_n == self.sample_offset
            )
        self._sample_serial += 1
        return self._sample_serial % self.sample_every_n == self.sample_offset

    def _overflow(self, queue_name: str) -> None:
        message = f"input monitor {queue_name} capacity exceeded"
        if self.overflow_policy is OverflowPolicy.ERROR:
            raise MonitorOverflowError(message)
        warnings.warn(message, ResourceWarning, stacklevel=3)

    def _stall_check_event(
        self,
        *,
        status: ExecutionStatus,
        fraction: float,
        consecutive: int,
    ) -> ExecutionEvent:
        return ExecutionEvent(
            operation=ExecutionOperation.HOST_WAIT,
            status=status,
            phase="input_stall_policy",
            wall_time_utc=_utc_text(self._wall_clock()),
            monotonic_ns=_integer(
                self._monotonic_ns(),
                "monotonic clock",
            ),
            duration_ns=0,
            check_id="INPUT-STALL-001",
            check_passed=False,
            check_expected=(
                f"stall_fraction<{self.max_input_stall_fraction:.6f};"
                f"consecutive<{self.max_consecutive_starved_steps}"
            ),
            check_observed=(
                f"stall_fraction={fraction:.6f};consecutive={consecutive}"
            ),
            evidence={
                "window_steps": self.rolling_window_steps,
                "window_stall_fraction": fraction,
                "consecutive_starved_steps": consecutive,
                "patience_windows": self.patience_windows,
                "bad_windows": self._bad_stall_windows,
            },
            remediation=(
                "inspect bounded host/device queue occupancy and remove the "
                "input-pipeline bottleneck; automatic tuning is prohibited"
            ),
            report_reference=self.report_reference,
        )

    def _observe_stall_locked(self, event: ExecutionEvent) -> None:
        if (
            event.global_step is None
            or event.status is not ExecutionStatus.SUCCESS
            or event.operation
            not in {
                ExecutionOperation.HOST_WAIT,
                ExecutionOperation.MODEL_COMPUTE,
                ExecutionOperation.OPTIMIZER,
            }
        ):
            return
        timing = self._step_timings.setdefault(
            event.global_step, _StepTiming()
        )
        if event.operation is ExecutionOperation.HOST_WAIT:
            timing.host_wait_ns += event.duration_ns
        elif event.operation is ExecutionOperation.MODEL_COMPUTE:
            timing.model_compute_ns += event.duration_ns
            timing.model_seen = True
        else:
            timing.optimizer_ns += event.duration_ns
            timing.optimizer_seen = True
        if not timing.model_seen or not timing.optimizer_seen:
            if len(self._step_timings) > self.event_capacity:
                oldest = next(iter(self._step_timings))
                self._step_timings.pop(oldest)
                self._overflow("stall step queue")
            return
        self._step_timings.pop(event.global_step)
        self._completed_steps += 1
        if self._completed_steps <= self.warmup_steps:
            return
        self._stall_window.append(
            (
                timing.host_wait_ns,
                timing.model_compute_ns,
                timing.optimizer_ns,
            )
        )
        self._steps_since_stall_window += 1
        if (
            len(self._stall_window) < self.rolling_window_steps
            or self._steps_since_stall_window < self.rolling_window_steps
        ):
            return
        self._steps_since_stall_window = 0
        wait_ns = sum(values[0] for values in self._stall_window)
        total_ns = sum(values[0] + values[1] for values in self._stall_window)
        fraction = 0.0 if total_ns == 0 else wait_ns / total_ns
        consecutive = 0
        maximum_consecutive = 0
        for host_wait_ns, compute_ns, _optimizer_ns in self._stall_window:
            denominator = host_wait_ns + compute_ns
            step_fraction = (
                0.0 if denominator == 0 else host_wait_ns / denominator
            )
            if step_fraction >= self.max_input_stall_fraction:
                consecutive += 1
                maximum_consecutive = max(maximum_consecutive, consecutive)
            else:
                consecutive = 0
        if (
            fraction < self.max_input_stall_fraction
            and maximum_consecutive < self.max_consecutive_starved_steps
        ):
            self._bad_stall_windows = 0
            return
        self._bad_stall_windows += 1
        strict_error = (
            self.stall_action is OverflowPolicy.ERROR
            and self._bad_stall_windows >= self.patience_windows
        )
        status = (
            ExecutionStatus.ERROR if strict_error else ExecutionStatus.WARNING
        )
        check = self._stall_check_event(
            status=status,
            fraction=fraction,
            consecutive=maximum_consecutive,
        )
        if len(self._events) >= self.event_capacity:
            self.dropped_event_count += 1
            self._overflow("event queue")
        else:
            self._events.append(check)
        message = (
            "INPUT-STALL-001: input starvation exceeded configured "
            f"policy ({fraction:.6f})"
        )
        if strict_error:
            raise InputStallError(message)
        warnings.warn(message, ResourceWarning, stacklevel=4)

    def _enqueue(self, event: ExecutionEvent) -> bool:
        if len(self._events) >= self.event_capacity:
            self.dropped_event_count += 1
            self._overflow("event queue")
            return False
        self._events.append(event)
        self._observe_stall_locked(event)
        return True

    @staticmethod
    def _resolve_descriptor_digest(
        identity: object | None,
        prehashed: str | None,
    ) -> str | None:
        if prehashed is not None:
            if (
                not isinstance(prehashed, str)
                or len(prehashed) != 64
                or any(
                    character not in "0123456789abcdef"
                    for character in prehashed
                )
            ):
                raise ValueError(
                    "descriptor_digest_value must be a lowercase SHA-256"
                )
            return prehashed
        return None if identity is None else descriptor_digest(identity)

    def begin(
        self,
        operation: ExecutionOperation,
        *,
        phase: str,
        split: str | None = None,
        epoch: int | None = None,
        global_step: int | None = None,
        descriptor_sequence: int | None = None,
        descriptor_identity: object | None = None,
        descriptor_digest_value: str | None = None,
        queue: QueueSnapshot | None = None,
        cuda_timing: bool = False,
        cuda_stream: object | None = None,
        evidence: Mapping[str, object] | None = None,
    ) -> OperationToken:
        """Begin one operation without retaining descriptors or caller batches."""

        self._ensure_open()
        if not isinstance(operation, ExecutionOperation):
            operation = ExecutionOperation(operation)
        sequence = _optional_integer(
            descriptor_sequence,
            "descriptor_sequence",
        )
        digest = self._resolve_descriptor_digest(
            descriptor_identity,
            descriptor_digest_value,
        )
        with self._lock:
            self._serial += 1
            serial = self._serial
            sampled = self._sampled(sequence)
        before = self._resource_reader() if sampled else None
        if before is not None and not isinstance(before, ResourceSnapshot):
            raise TypeError("resource_reader must return ResourceSnapshot")
        cuda_start: object | None = None
        cuda_end: object | None = None
        if cuda_timing and self.cuda_event_factory is not None:
            cuda_start = self.cuda_event_factory()
            cuda_end = self.cuda_event_factory()
            record = getattr(cuda_start, "record", None)
            if not callable(record):
                raise TypeError("CUDA timing event must expose record")
            record(cuda_stream)
        token = OperationToken(
            serial=serial,
            operation=operation,
            phase=phase,
            split=split,
            wall_time_utc=_utc_text(self._wall_clock()),
            start_ns=_integer(self._monotonic_ns(), "monotonic clock"),
            epoch=_optional_integer(epoch, "epoch"),
            global_step=_optional_integer(global_step, "global_step"),
            descriptor_sequence=sequence,
            descriptor_digest=digest,
            sampled=sampled,
            resource_before=before,
            queue=queue,
            cuda_start=cuda_start,
            cuda_end=cuda_end,
            evidence=redact_mapping(evidence or {}),
        )
        with self._lock:
            self._open_tokens.add(serial)
        return token

    def begin_for_batch(
        self,
        operation: ExecutionOperation,
        *,
        batch: object,
        phase: str,
        split: str | None = None,
        epoch: int | None = None,
        global_step: int | None = None,
        queue: QueueSnapshot | None = None,
        cuda_timing: bool = False,
        cuda_stream: object | None = None,
        evidence: Mapping[str, object] | None = None,
    ) -> OperationToken:
        """Extract only sequence and hashed descriptor, never the batch itself."""

        descriptor = getattr(batch, "sampling_descriptor", None)
        prehashed = getattr(batch, "execution_descriptor_digest", None)
        sequence = getattr(batch, "sequence_id", None)
        if sequence is not None:
            sequence = _integer(sequence, "batch.sequence_id")
        return self.begin(
            operation,
            phase=phase,
            split=split,
            epoch=epoch,
            global_step=global_step,
            descriptor_sequence=sequence,
            descriptor_identity=(descriptor if prehashed is None else None),
            descriptor_digest_value=prehashed,
            queue=queue,
            cuda_timing=cuda_timing,
            cuda_stream=cuda_stream,
            evidence=evidence,
        )

    def _completion(
        self,
        *,
        status: ExecutionStatus,
        node_count: int | None,
        edge_count: int | None,
        row_count: int | None,
        example_count: int | None,
        unique_storage_bytes: int | None,
        queue: QueueSnapshot | None,
        sampled: bool,
        check_id: str | None,
        check_passed: bool | None,
        check_expected: object,
        check_observed: object,
        evidence: Mapping[str, object] | None,
        remediation: str | None,
        report_reference: str | None,
    ) -> _Completion:
        after = self._resource_reader() if sampled else None
        if after is not None and not isinstance(after, ResourceSnapshot):
            raise TypeError("resource_reader must return ResourceSnapshot")
        return _Completion(
            end_ns=_integer(self._monotonic_ns(), "monotonic clock"),
            status=ExecutionStatus(status),
            node_count=_optional_integer(node_count, "node_count"),
            edge_count=_optional_integer(edge_count, "edge_count"),
            row_count=_optional_integer(row_count, "row_count"),
            example_count=_optional_integer(example_count, "example_count"),
            unique_storage_bytes=_optional_integer(
                unique_storage_bytes,
                "unique_storage_bytes",
            ),
            queue=queue,
            resource_after=after,
            check_id=check_id,
            check_passed=check_passed,
            check_expected=_safe_check_value(check_expected, "expected"),
            check_observed=_safe_check_value(check_observed, "observed"),
            evidence=redact_mapping(evidence or {}),
            remediation=remediation,
            report_reference=report_reference,
        )

    def _event(
        self,
        token: OperationToken,
        completion: _Completion,
        duration_ns: int,
    ) -> ExecutionEvent:
        evidence = dict(token.evidence)
        evidence.update(completion.evidence)
        return ExecutionEvent(
            operation=token.operation,
            status=completion.status,
            phase=token.phase,
            split=token.split,
            wall_time_utc=token.wall_time_utc,
            monotonic_ns=token.start_ns,
            duration_ns=max(0, duration_ns),
            epoch=token.epoch,
            global_step=token.global_step,
            descriptor_sequence=token.descriptor_sequence,
            descriptor_digest=token.descriptor_digest,
            node_count=completion.node_count,
            edge_count=completion.edge_count,
            row_count=completion.row_count,
            example_count=completion.example_count,
            unique_storage_bytes=completion.unique_storage_bytes,
            sampled=token.sampled,
            check_id=completion.check_id,
            check_passed=completion.check_passed,
            check_expected=completion.check_expected,
            check_observed=completion.check_observed,
            evidence=redact_mapping(evidence),
            remediation=completion.remediation,
            report_reference=completion.report_reference,
            **_queue_values(completion.queue or token.queue),
            **_resource_values(
                token.resource_before, completion.resource_after
            ),
        )

    def finish(
        self,
        token: OperationToken,
        *,
        status: ExecutionStatus = ExecutionStatus.SUCCESS,
        node_count: int | None = None,
        edge_count: int | None = None,
        row_count: int | None = None,
        example_count: int | None = None,
        unique_storage_bytes: int | None = None,
        queue: QueueSnapshot | None = None,
        cuda_stream: object | None = None,
        check_id: str | None = None,
        check_passed: bool | None = None,
        check_expected: object = None,
        check_observed: object = None,
        evidence: Mapping[str, object] | None = None,
        remediation: str | None = None,
        report_reference: str | None = None,
    ) -> ExecutionEvent | None:
        """Finish now or enqueue a nonblocking CUDA completion pair."""

        self._ensure_open()
        if not isinstance(token, OperationToken):
            raise TypeError("finish requires an OperationToken")
        with self._lock:
            if token.serial not in self._open_tokens:
                raise RuntimeError("operation token is not open")
            self._open_tokens.remove(token.serial)
        try:
            completion = self._completion(
                status=status,
                node_count=node_count,
                edge_count=edge_count,
                row_count=row_count,
                example_count=example_count,
                unique_storage_bytes=unique_storage_bytes,
                queue=queue,
                sampled=token.sampled,
                check_id=check_id,
                check_passed=check_passed,
                check_expected=check_expected,
                check_observed=check_observed,
                evidence=evidence,
                remediation=remediation,
                report_reference=report_reference,
            )
        except BaseException:
            with self._lock:
                self._open_tokens.add(token.serial)
            raise
        if token.cuda_end is None:
            event = self._event(
                token, completion, completion.end_ns - token.start_ns
            )
            with self._lock:
                self._enqueue(event)
            return event
        record = getattr(token.cuda_end, "record", None)
        if not callable(record):
            raise TypeError("CUDA completion event must expose record")
        record(cuda_stream)
        with self._lock:
            if len(self._pending) >= self.pending_cuda_capacity:
                self.dropped_cuda_timing_count += 1
                self._overflow("pending CUDA timing queue")
                fallback = replace(
                    completion,
                    status=(
                        ExecutionStatus.WARNING
                        if completion.status is ExecutionStatus.SUCCESS
                        else completion.status
                    ),
                    evidence=redact_mapping(
                        {
                            **dict(completion.evidence),
                            "cuda_timing": "pending-overflow",
                        }
                    ),
                )
                event = self._event(
                    token,
                    fallback,
                    completion.end_ns - token.start_ns,
                )
                self._enqueue(event)
                return event
            self._pending.append(_PendingCuda(token, completion))
        return None

    def poll(self) -> tuple[ExecutionEvent, ...]:
        """Resolve only descriptor-ordered CUDA pairs whose end event is ready."""

        self._ensure_open()
        completed: list[ExecutionEvent] = []
        with self._lock:
            while self._pending:
                pending = self._pending[0]
                query = getattr(pending.token.cuda_end, "query", None)
                if not callable(query):
                    raise TypeError("CUDA completion event must expose query")
                if query() is not True:
                    break
                elapsed = getattr(
                    pending.token.cuda_start, "elapsed_time", None
                )
                if not callable(elapsed):
                    raise TypeError(
                        "CUDA start event must expose elapsed_time"
                    )
                milliseconds = float(elapsed(pending.token.cuda_end))
                if milliseconds < 0:
                    raise ValueError("CUDA elapsed time must be non-negative")
                self._pending.popleft()
                event = self._event(
                    pending.token,
                    pending.completion,
                    round(milliseconds * 1_000_000),
                )
                self._enqueue(event)
                completed.append(event)
        return tuple(completed)

    def drain(self) -> tuple[ExecutionEvent, ...]:
        """Poll ready CUDA work and transfer queued immutable events to a writer."""

        if not self._closed:
            self.poll()
        with self._lock:
            events = tuple(self._events)
            self._events.clear()
            return events

    def record(
        self,
        operation: ExecutionOperation,
        *,
        phase: str,
        split: str | None = None,
        status: ExecutionStatus = ExecutionStatus.SUCCESS,
        duration_ns: int = 0,
        epoch: int | None = None,
        global_step: int | None = None,
        descriptor_sequence: int | None = None,
        descriptor_identity: object | None = None,
        descriptor_digest_value: str | None = None,
        node_count: int | None = None,
        edge_count: int | None = None,
        row_count: int | None = None,
        example_count: int | None = None,
        unique_storage_bytes: int | None = None,
        queue: QueueSnapshot | None = None,
        evidence: Mapping[str, object] | None = None,
        check_id: str | None = None,
        check_passed: bool | None = None,
        check_expected: object = None,
        check_observed: object = None,
        remediation: str | None = None,
        report_reference: str | None = None,
    ) -> ExecutionEvent:
        """Queue a known-duration primitive event without JSON serialization."""

        token = self.begin(
            operation,
            phase=phase,
            split=split,
            epoch=epoch,
            global_step=global_step,
            descriptor_sequence=descriptor_sequence,
            descriptor_identity=descriptor_identity,
            descriptor_digest_value=descriptor_digest_value,
            queue=queue,
            evidence=evidence,
        )
        try:
            completion = self._completion(
                status=status,
                node_count=node_count,
                edge_count=edge_count,
                row_count=row_count,
                example_count=example_count,
                unique_storage_bytes=unique_storage_bytes,
                queue=queue,
                sampled=token.sampled,
                check_id=check_id,
                check_passed=check_passed,
                check_expected=check_expected,
                check_observed=check_observed,
                evidence=None,
                remediation=remediation,
                report_reference=report_reference,
            )
            event = self._event(
                token,
                completion,
                _integer(duration_ns, "duration_ns"),
            )
        finally:
            with self._lock:
                self._open_tokens.discard(token.serial)
        with self._lock:
            self._enqueue(event)
        return event

    def mark_batch_ready(
        self,
        *,
        phase: str,
        split: str | None,
        descriptor_sequence: int | None,
        descriptor_identity: object | None = None,
        descriptor_digest_value: str | None = None,
        queue: QueueSnapshot | None = None,
    ) -> None:
        """Remember only a descriptor digest and readiness timestamp."""

        self._ensure_open()
        digest = self._resolve_descriptor_digest(
            descriptor_identity,
            descriptor_digest_value,
        )
        if digest is None:
            return
        with self._lock:
            if (
                digest not in self._ready_times
                and len(self._ready_times) >= self.event_capacity
            ):
                self._overflow("batch readiness queue")
                return
            self._ready_times[digest] = _integer(
                self._monotonic_ns(),
                "monotonic clock",
            )

    def begin_model_compute(
        self,
        *,
        phase: str,
        split: str | None,
        descriptor_sequence: int | None,
        descriptor_identity: object | None = None,
        descriptor_digest_value: str | None = None,
        epoch: int | None = None,
        global_step: int | None = None,
        cuda_timing: bool = False,
        cuda_stream: object | None = None,
    ) -> OperationToken:
        """Record starvation relative to prior compute without serialization."""

        digest = self._resolve_descriptor_digest(
            descriptor_identity,
            descriptor_digest_value,
        )
        now = _integer(self._monotonic_ns(), "monotonic clock")
        with self._lock:
            ready = (
                None if digest is None else self._ready_times.pop(digest, None)
            )
            prior_end = self._last_compute_end_ns
        if prior_end is not None and ready is not None and ready > prior_end:
            self.record(
                ExecutionOperation.HOST_WAIT,
                phase=phase,
                split=split,
                duration_ns=ready - prior_end,
                epoch=epoch,
                global_step=global_step,
                descriptor_sequence=descriptor_sequence,
                descriptor_digest_value=digest,
                evidence={"cadence_end_ns": prior_end, "ready_ns": ready},
            )
        return self.begin(
            ExecutionOperation.MODEL_COMPUTE,
            phase=phase,
            split=split,
            epoch=epoch,
            global_step=global_step,
            descriptor_sequence=descriptor_sequence,
            descriptor_digest_value=digest,
            cuda_timing=cuda_timing,
            cuda_stream=cuda_stream,
            evidence={"compute_begin_ns": now},
        )

    def finish_model_compute(
        self,
        token: OperationToken,
        **values: object,
    ) -> ExecutionEvent | None:
        """Finish model compute and advance the starvation cadence."""

        try:
            return self.finish(token, **values)  # type: ignore[arg-type]
        finally:
            with self._lock:
                self._last_compute_end_ns = _integer(
                    self._monotonic_ns(),
                    "monotonic clock",
                )

    def record_qualification(
        self,
        result: object,
        report_path: str | Path,
    ) -> ExecutionEvent:
        """Emit one existing qualification result without changing its failure."""

        passed = getattr(result, "passed", None)
        if type(passed) is not bool:
            raise TypeError("qualification result must expose boolean passed")
        evidence = getattr(result, "evidence", {})
        if not isinstance(evidence, Mapping):
            evidence = {"evidence_type": type(evidence).__name__}
        limit = getattr(result, "limit", None)
        if limit is not None:
            evidence = {**dict(evidence), "limit": limit}
        remediation_value = _safe_check_value(
            getattr(result, "remediation", None),
            "remediation",
        )
        remediation = (
            remediation_value if isinstance(remediation_value, str) else None
        )
        return self.record(
            ExecutionOperation.VALIDATION,
            phase="qualification",
            status=(
                ExecutionStatus.SUCCESS if passed else ExecutionStatus.ERROR
            ),
            check_id=getattr(result, "check_id", None),
            check_passed=passed,
            check_expected=getattr(result, "expected", None),
            check_observed=getattr(result, "observed", None),
            evidence=evidence,
            remediation=remediation,
            report_reference=Path(report_path).name,
        )

    def close(self) -> None:
        """Discard unresolved completion handles without synchronizing CUDA."""

        if self._closed:
            return
        with self._lock:
            try:
                while self._pending:
                    pending = self._pending.popleft()
                    completion = replace(
                        pending.completion,
                        status=ExecutionStatus.CANCELLED,
                        evidence=redact_mapping(
                            {
                                **dict(pending.completion.evidence),
                                "cuda_timing": "unresolved-at-close",
                            }
                        ),
                    )
                    with contextlib.suppress(MonitorOverflowError):
                        self._enqueue(
                            self._event(
                                pending.token,
                                completion,
                                completion.end_ns - pending.token.start_ns,
                            )
                        )
            finally:
                self._open_tokens.clear()
                self._ready_times.clear()
                self._closed = True


__all__ = [
    "InputMonitor",
    "MonitorOverflowError",
    "InputStallError",
    "OperationToken",
    "OverflowPolicy",
    "QueueSnapshot",
    "ResourceSnapshot",
]
