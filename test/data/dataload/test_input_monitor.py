"""Controllable-clock contracts for asynchronous input-pipeline monitoring."""

from __future__ import annotations

import gc
import weakref
from dataclasses import dataclass
from datetime import UTC, datetime, timedelta
from types import SimpleNamespace

import pytest
import torch
from torch_geometric.data import Data

from topobench.dataloader.disk_graph import (
    SamplingDescriptor,
    _DescriptorDataset,
    _ExecutionEventLoader,
)
from topobench.dataloader.input_monitor import (
    InputMonitor,
    InputStallError,
    MonitorOverflowError,
    OverflowPolicy,
    QueueSnapshot,
    ResourceSnapshot,
)
from topobench.profiling.execution_events import (
    ExecutionOperation,
    ExecutionStatus,
)


@dataclass
class _Clock:
    monotonic: int = 0

    def monotonic_ns(self) -> int:
        return self.monotonic

    def wall(self) -> datetime:
        return datetime(2026, 8, 2, tzinfo=UTC) + timedelta(
            microseconds=self.monotonic // 1000
        )

    def advance(self, nanoseconds: int) -> None:
        self.monotonic += nanoseconds


class _CudaEvent:
    def __init__(self) -> None:
        self.complete = False
        self.recorded_streams: list[object | None] = []
        self.elapsed_ms = 0.0

    def record(self, stream: object | None = None) -> None:
        self.recorded_streams.append(stream)

    def query(self) -> bool:
        return self.complete

    def elapsed_time(self, end: "_CudaEvent") -> float:
        assert end.complete
        return end.elapsed_ms


class _CudaFactory:
    def __init__(self) -> None:
        self.events: list[_CudaEvent] = []

    def __call__(self) -> _CudaEvent:
        event = _CudaEvent()
        self.events.append(event)
        return event


def _queue(host_depth: int = 1, device_depth: int = 0) -> QueueSnapshot:
    return QueueSnapshot(
        host_configured_depth=4,
        host_depth=host_depth,
        host_configured_bytes=400,
        host_bytes=host_depth * 40,
        device_configured_depth=2,
        device_depth=device_depth,
        device_configured_bytes=200,
        device_bytes=device_depth * 50,
    )

class _WorkerStrategy:
    def materialize(self, source: object, descriptor: SamplingDescriptor) -> Data:
        del source, descriptor
        return Data(
            x=torch.ones(2, 1),
            edge_index=torch.tensor([[0], [1]], dtype=torch.long),
            num_nodes=2,
        )




def test_controllable_clocks_resources_queue_deltas_and_cadence_are_exact() -> None:
    clock = _Clock(100)
    samples = iter(
        (
            ResourceSnapshot(1000, 100, 200, 300, 400),
            ResourceSnapshot(1120, 140, 250, 330, 470),
            ResourceSnapshot(1300, 180, 300, 390, 520),
        )
    )
    monitor = InputMonitor(
        wall_clock=clock.wall,
        monotonic_ns=clock.monotonic_ns,
        resource_reader=lambda: next(samples),
        event_capacity=16,
        pending_cuda_capacity=2,
        sample_every_n=2,
        sample_offset=1,
        overflow_policy=OverflowPolicy.ERROR,
    )
    first = monitor.begin(
        ExecutionOperation.SELECTED_READ,
        phase="fit",
        split="train",
        descriptor_sequence=1,
        descriptor_identity={"external_id": "raw-1"},
        queue=_queue(2, 1),
    )
    clock.advance(25)
    monitor.finish(
        first,
        node_count=3,
        edge_count=5,
        row_count=7,
        example_count=2,
        unique_storage_bytes=88,
        queue=_queue(1, 1),
    )
    second = monitor.begin(
        ExecutionOperation.SELECTED_READ,
        phase="fit",
        split="train",
        descriptor_sequence=2,
        descriptor_identity={"external_id": "raw-2"},
    )
    clock.advance(10)
    monitor.finish(second)

    first_event, second_event = monitor.drain()
    assert first_event.duration_ns == 25
    assert first_event.sampled
    assert first_event.rss_bytes == 1120
    assert first_event.rss_delta_bytes == 120
    assert first_event.pinned_delta_bytes == 40
    assert first_event.gpu_delta_bytes == 50
    assert first_event.temp_disk_delta_bytes == 30
    assert first_event.final_disk_delta_bytes == 70
    assert first_event.host_queue_configured_depth == 4
    assert first_event.host_queue_depth == 1
    assert first_event.device_queue_bytes == 50
    assert first_event.descriptor_digest is not None
    assert "raw-1" not in repr(first_event)
    assert not second_event.sampled
    assert second_event.rss_bytes is None
    assert monitor.sample_every_n == 2
    assert monitor.sample_offset == 1


def test_cuda_completion_is_polled_nonblocking_in_descriptor_order() -> None:
    clock = _Clock(10)
    factory = _CudaFactory()
    stream = object()
    monitor = InputMonitor(
        wall_clock=clock.wall,
        monotonic_ns=clock.monotonic_ns,
        cuda_event_factory=factory,
        event_capacity=8,
        pending_cuda_capacity=2,
        overflow_policy="error",
    )
    first = monitor.begin(
        ExecutionOperation.H2D_COPY,
        phase="fit",
        split="train",
        descriptor_sequence=1,
        descriptor_identity="descriptor-one",
        cuda_timing=True,
        cuda_stream=stream,
    )
    monitor.finish(first, cuda_stream=stream)
    second = monitor.begin(
        ExecutionOperation.H2D_COPY,
        phase="fit",
        split="train",
        descriptor_sequence=2,
        descriptor_identity="descriptor-two",
        cuda_timing=True,
        cuda_stream=stream,
    )
    monitor.finish(second, cuda_stream=stream)

    first_end, second_end = factory.events[1], factory.events[3]
    second_end.complete = True
    second_end.elapsed_ms = 2.5
    assert monitor.poll() == ()
    assert monitor.pending_cuda_count == 2

    first_end.complete = True
    first_end.elapsed_ms = 1.25
    completed = monitor.poll()
    assert [event.descriptor_sequence for event in completed] == [1, 2]
    assert [event.duration_ns for event in completed] == [1_250_000, 2_500_000]
    assert all(event.status is ExecutionStatus.SUCCESS for event in completed)
    assert all(event.recorded_streams == [stream] for event in factory.events)
    assert not hasattr(monitor, "synchronize")
    assert "synchronize" not in vars(monitor)


def test_pending_and_event_overflow_have_explicit_warn_or_error_policy() -> None:
    clock = _Clock()
    warning_monitor = InputMonitor(
        wall_clock=clock.wall,
        monotonic_ns=clock.monotonic_ns,
        event_capacity=1,
        pending_cuda_capacity=1,
        overflow_policy=OverflowPolicy.WARN,
    )
    warning_monitor.record(
        ExecutionOperation.HOST_PIN,
        phase="fit",
        split="train",
    )
    with pytest.warns(ResourceWarning, match="event queue"):
        warning_monitor.record(
            ExecutionOperation.HOST_PIN,
            phase="fit",
            split="train",
        )
    assert warning_monitor.dropped_event_count == 1

    error_monitor = InputMonitor(
        wall_clock=clock.wall,
        monotonic_ns=clock.monotonic_ns,
        event_capacity=1,
        pending_cuda_capacity=1,
        overflow_policy=OverflowPolicy.ERROR,
    )
    error_monitor.record(ExecutionOperation.HOST_PIN, phase="fit")
    with pytest.raises(MonitorOverflowError, match="event queue"):
        error_monitor.record(ExecutionOperation.HOST_PIN, phase="fit")


def test_input_starvation_is_wait_for_ready_batch_after_prior_compute() -> None:
    clock = _Clock(100)
    monitor = InputMonitor(
        wall_clock=clock.wall,
        monotonic_ns=clock.monotonic_ns,
        event_capacity=8,
        pending_cuda_capacity=1,
    )
    first = monitor.begin_model_compute(
        phase="fit",
        split="train",
        descriptor_sequence=1,
        descriptor_identity="one",
    )
    clock.advance(30)
    monitor.finish_model_compute(first)
    clock.advance(50)
    monitor.mark_batch_ready(
        phase="fit",
        split="train",
        descriptor_sequence=2,
        descriptor_identity="two",
        queue=_queue(),
    )
    clock.advance(10)
    second = monitor.begin_model_compute(
        phase="fit",
        split="train",
        descriptor_sequence=2,
        descriptor_identity="two",
    )
    clock.advance(20)
    monitor.finish_model_compute(second)

    events = monitor.drain()
    waits = [event for event in events if event.operation is ExecutionOperation.HOST_WAIT]
    assert len(waits) == 1
    assert waits[0].duration_ns == 50
    assert waits[0].descriptor_sequence == 2


def test_stall_policy_packages_stable_checks_and_errors_after_patience() -> None:
    clock = _Clock()
    monitor = InputMonitor(
        wall_clock=clock.wall,
        monotonic_ns=clock.monotonic_ns,
        event_capacity=64,
        pending_cuda_capacity=1,
        warmup_steps=1,
        rolling_window_steps=2,
        max_input_stall_fraction=0.1,
        max_consecutive_starved_steps=1,
        patience_windows=2,
        stall_action="error",
        report_reference="events.jsonl",
    )

    for step in range(5):
        monitor.record(
            ExecutionOperation.HOST_WAIT,
            phase="fit",
            split="train",
            global_step=step,
            duration_ns=9,
        )
        monitor.record(
            ExecutionOperation.MODEL_COMPUTE,
            phase="fit",
            split="train",
            global_step=step,
            duration_ns=1,
        )
        if step == 4:
            with pytest.raises(InputStallError, match="INPUT-STALL-001"):
                monitor.record(
                    ExecutionOperation.OPTIMIZER,
                    phase="fit",
                    split="train",
                    global_step=step,
                    duration_ns=90,
                )
        else:
            monitor.record(
                ExecutionOperation.OPTIMIZER,
                phase="fit",
                split="train",
                global_step=step,
                duration_ns=90,
            )

    checks = [
        event
        for event in monitor.drain()
        if event.check_id == "INPUT-STALL-001"
    ]
    assert [event.status for event in checks] == [
        ExecutionStatus.WARNING,
        ExecutionStatus.ERROR,
    ]
    assert all(event.remediation for event in checks)
    assert all(event.report_reference == "events.jsonl" for event in checks)
    assert checks[-1].evidence["window_stall_fraction"] == pytest.approx(0.9)


def test_stall_boundary_excludes_optimizer_and_triggers_on_third_step() -> None:
    monitor = InputMonitor(
        event_capacity=32,
        pending_cuda_capacity=1,
        warmup_steps=0,
        rolling_window_steps=3,
        max_input_stall_fraction=0.5,
        max_consecutive_starved_steps=3,
        patience_windows=1,
        stall_action="error",
    )
    for step in range(3):
        monitor.record(
            ExecutionOperation.HOST_WAIT,
            phase="fit",
            global_step=step,
            duration_ns=1,
        )
        monitor.record(
            ExecutionOperation.MODEL_COMPUTE,
            phase="fit",
            global_step=step,
            duration_ns=1,
        )
        if step == 2:
            with pytest.raises(InputStallError, match="INPUT-STALL-001"):
                monitor.record(
                    ExecutionOperation.OPTIMIZER,
                    phase="fit",
                    global_step=step,
                    duration_ns=98,
                )
        else:
            monitor.record(
                ExecutionOperation.OPTIMIZER,
                phase="fit",
                global_step=step,
                duration_ns=98,
            )

    check = next(
        event
        for event in monitor.drain()
        if event.check_id == "INPUT-STALL-001"
    )
    assert check.evidence["window_stall_fraction"] == pytest.approx(0.5)
    assert check.evidence["consecutive_starved_steps"] == 3



def test_worker_loader_returns_ordered_stage_envelopes_without_metadata() -> None:
    monitor = InputMonitor(event_capacity=16, pending_cuda_capacity=1)
    descriptors = tuple(
        SamplingDescriptor(
            content_sha256="a" * 64,
            active_split_tag="split",
            phase="val",
            strategy="worker-test",
            strategy_options_json="{}",
            batch_ordinal=ordinal,
            partition_ids=(ordinal,),
            generator_state_sha256="b" * 64,
            participant_counts=(("node", ordinal + 1),),
        )
        for ordinal in range(2)
    )
    dataset = _DescriptorDataset(
        SimpleNamespace(),
        _WorkerStrategy(),
        descriptors,
        None,
        "val",
        None,
        capture_worker_events=True,
    )
    owner = SimpleNamespace(execution_monitor=monitor)
    loader = _ExecutionEventLoader(
        owner,
        dataset,
        num_workers=1,
        persistent_workers=False,
    )

    batches = list(loader)
    events = monitor.drain()
    assert [event.operation for event in events] == [
        ExecutionOperation.SELECTED_READ,
        ExecutionOperation.NATIVE_ASSEMBLY,
        ExecutionOperation.SELECTED_READ,
        ExecutionOperation.NATIVE_ASSEMBLY,
    ]
    assert all("_execution_event_envelopes" not in batch for batch in batches)
    assert [event.descriptor_sequence for event in events] == [None] * 4
    assert all(event.descriptor_digest for event in events)
    assert [event.row_count for event in events] == [1, 1, 2, 2]
    assert all(event.unique_storage_bytes is None for event in events)
    assert "SamplingDescriptor" not in repr(events)




def test_monitor_hashes_descriptor_immediately_and_never_retains_batch_tensors() -> None:
    clock = _Clock()
    monitor = InputMonitor(
        wall_clock=clock.wall,
        monotonic_ns=clock.monotonic_ns,
        event_capacity=4,
        pending_cuda_capacity=1,
    )
    batch = Data(x=torch.ones(3, 2))
    batch.sampling_descriptor = {"external_ids": ["secret-node"]}
    reference = weakref.ref(batch)
    token = monitor.begin_for_batch(
        ExecutionOperation.NATIVE_ASSEMBLY,
        batch=batch,
        phase="fit",
        split="train",
    )
    monitor.finish(token)
    del batch
    gc.collect()

    assert reference() is None
    assert "secret-node" not in repr(monitor.drain())
