"""Standalone mandatory CUDA qualification for native graph prefetch.

This is intentionally not a pytest test: a host without a real CUDA runtime is a
qualification failure, never a skip or synthetic pass.
"""

from __future__ import annotations

import sys
import threading
import time
from collections.abc import Iterator

import torch
from lightning.fabric.utilities.apply_func import move_data_to_device
from torch.profiler import ProfilerActivity, profile, record_function
from torch_geometric.data import Data, HeteroData

from topobench.dataloader.device_prefetch import (
    DevicePrefetchLoader,
    PrefetchError,
    PrefetchLimits,
)

_MIB = 1024 * 1024


def _limits(depth: int, *, batch_bytes: int = 96 * _MIB) -> PrefetchLimits:
    return PrefetchLimits(
        host_queue_depth=max(2, depth),
        device_queue_depth=depth,
        max_batch_nodes=20_000_000,
        max_batch_edges=1_000_000,
        max_nodes_per_type={},
        max_edges_per_relation={},
        max_host_batch_bytes=batch_bytes,
        max_device_batch_bytes=batch_bytes,
        max_host_queue_bytes=(max(2, depth) + depth) * batch_bytes,
        max_device_queue_bytes=depth * batch_bytes,
        worst_case_host_bytes=batch_bytes,
        worst_case_device_bytes=batch_bytes,
    )


def _data(sequence_id: int, rows: int = 4096) -> Data:
    values = torch.arange(rows * 16, dtype=torch.float32).view(rows, 16)
    batch = Data(
        x=values,
        edge_index=torch.tensor([[0, 1, 2], [1, 2, 3]], dtype=torch.long),
        num_nodes=rows,
    )
    batch.x_alias = values[:, :4]
    batch.sequence_id = sequence_id
    return batch


def _hetero(sequence_id: int, rows: int = 4096) -> HeteroData:
    batch = HeteroData()
    batch["paper"].x = torch.arange(rows * 16, dtype=torch.float32).view(
        rows, 16
    )
    batch["paper"].x_alias = batch["paper"].x[:, :4]
    batch["author"].x = torch.ones((32, 8), dtype=torch.float32)
    relation = ("author", "writes", "paper")
    batch[relation].edge_index = torch.tensor([[0, 1, 2], [1, 2, 3]])
    batch.sequence_id = sequence_id
    return batch


def _tensors(batch: Data | HeteroData) -> tuple[torch.Tensor, ...]:
    return tuple(
        value
        for store in batch.stores
        for value in store.values()
        if isinstance(value, torch.Tensor)
    )


def _pointers(batch: Data | HeteroData) -> tuple[int, ...]:
    return tuple(tensor.data_ptr() for tensor in _tensors(batch))


def _storage_bytes(batch: Data | HeteroData) -> int:
    storages = {
        tensor.untyped_storage()._cdata: tensor.untyped_storage().nbytes()
        for tensor in _tensors(batch)
    }
    return sum(storages.values())


def _qualify_depth_and_native_kind(
    device: torch.device,
    depth: int,
    kind: str,
) -> None:
    batches = [
        _data(sequence_id) if kind == "data" else _hetero(sequence_id)
        for sequence_id in range(1, 7)
    ]
    loader = DevicePrefetchLoader(
        batches,
        _limits(depth),
        device=device,
    )
    torch.cuda.synchronize(device)
    baseline_bytes = torch.cuda.memory_allocated(device)
    torch.cuda.reset_peak_memory_stats(device)
    iterator = iter(loader)
    output = next(iterator)
    assert iterator.transfer_stream is not None
    assert (
        iterator.transfer_stream.cuda_stream
        != torch.cuda.current_stream(device).cuda_stream
    )
    assert len(iterator.completion_events) == depth
    assert (
        iterator.max_device_queued_bytes
        <= loader.limits.max_device_queue_bytes
    )
    assert iterator.max_host_queued_bytes <= loader.limits.max_host_queue_bytes
    allocation_bytes = torch.cuda.max_memory_allocated(device) - baseline_bytes
    allocation_bound = (depth + 1) * _storage_bytes(batches[0]) + _MIB
    assert allocation_bytes <= allocation_bound
    assert all(tensor.device == device for tensor in _tensors(output))

    # The current model stream must have waited on the slot completion event.
    expected = (
        batches[0].x.sum()
        if isinstance(batches[0], Data)
        else batches[0]["paper"].x.sum()
    )
    actual = (
        output.x.sum() if isinstance(output, Data) else output["paper"].x.sum()
    )
    assert torch.equal(actual.cpu(), expected.cpu())

    before = _pointers(output)
    lightning_output = move_data_to_device(output, device)
    after = _pointers(lightning_output)
    assert lightning_output is output
    assert after == before, "Lightning performed a second device allocation"

    remaining = list(iterator)
    assert [
        output.sequence_id,
        *(batch.sequence_id for batch in remaining),
    ] == list(range(1, 7))
    assert not iterator.producer_alive
    assert iterator.closed
    loader.close()
    loader.close()
    assert not loader.active_iterators


def _qualify_already_resident(device: torch.device) -> None:
    source = _data(1).to(device)
    pointers = _pointers(source)
    loader = DevicePrefetchLoader([source], _limits(1), device=device)
    output = next(iter(loader))
    assert output is source
    assert _pointers(output) == pointers
    assert all(tensor.device == device for tensor in _tensors(output))
    loader.close()


def _qualify_error_cleanup(device: torch.device) -> None:
    def values() -> Iterator[Data]:
        yield _data(1)
        invalid = _data(2)
        invalid.num_nodes = 20_000_001
        yield invalid

    loader = DevicePrefetchLoader(values(), _limits(1), device=device)
    iterator = iter(loader)
    assert next(iterator).sequence_id == 1
    try:
        next(iterator)
    except PrefetchError as error:
        assert "host_admission" in str(error)
        assert error.__cause__ is not None
    else:
        raise AssertionError("oversized CUDA batch was not rejected")
    loader.close()
    assert iterator.closed
    assert not iterator.producer_alive
    assert not loader.active_iterators


def _qualify_ready_yield_does_not_wait_for_future_source(
    device: torch.device,
) -> None:
    first_produced = threading.Event()
    release_second = threading.Event()

    def slow_source() -> Iterator[Data]:
        first_produced.set()
        yield _data(1)
        release_second.wait(timeout=2.0)
        yield _data(2)

    loader = DevicePrefetchLoader(
        slow_source(),
        _limits(3),
        device=device,
    )
    iterator = iter(loader)
    assert first_produced.wait(timeout=1.0)
    deadline = time.monotonic() + 1.0
    while iterator.max_host_queue_size == 0 and time.monotonic() < deadline:
        time.sleep(0.001)
    started = time.monotonic()
    output = next(iterator)
    elapsed = time.monotonic() - started
    release_second.set()
    loader.close()
    assert output.sequence_id == 1
    assert elapsed < 0.5, (
        "event-ready batch waited for future host materialization: "
        f"{elapsed:.3f}s"
    )


def _descendant_kernel_names(event: object) -> set[str]:
    names: set[str] = set()
    pending = list(event.cpu_children)
    while pending:
        child = pending.pop()
        names.update(kernel.name for kernel in child.kernels)
        pending.extend(child.cpu_children)
    return names


def _overlaps(left: object, right: object) -> bool:
    left_range = left.time_range
    right_range = right.time_range
    return max(left_range.start, right_range.start) < min(
        left_range.end, right_range.end
    )


def _run_overlap_workload(
    device: torch.device,
    matrix: torch.Tensor,
) -> torch.Tensor:
    rows = 1024 * 1024  # 64 MiB H2D per batch.
    batches = [_data(sequence_id, rows=rows) for sequence_id in range(1, 6)]
    loader = DevicePrefetchLoader(
        batches,
        _limits(3, batch_bytes=80 * _MIB),
        device=device,
    )
    result = matrix
    iterator = iter(loader)
    for index, batch in enumerate(iterator):
        with record_function("synthetic_model_compute"):
            result = torch.mm(matrix, matrix)
            result.add_(batch.x[index, 0])
        if index == 2:
            break
    loader.close()
    return result


def _qualify_profiler_overlap(device: torch.device) -> None:
    matrix = torch.randn((2048, 2048), device=device)
    _run_overlap_workload(device, matrix)
    torch.cuda.synchronize(device)

    with profile(
        activities=[ProfilerActivity.CPU, ProfilerActivity.CUDA]
    ) as trace:
        result = _run_overlap_workload(device, matrix)
    torch.cuda.synchronize(device)
    assert torch.isfinite(result).all()

    events = trace.events()
    cuda_events = [
        event
        for event in events
        if event.device_type == torch.autograd.DeviceType.CUDA
    ]
    h2d = [
        event
        for event in cuda_events
        if "memcpy" in event.name.lower()
        and ("htod" in event.name.lower() or "h2d" in event.name.lower())
    ]
    markers = [
        event
        for event in events
        if event.device_type == torch.autograd.DeviceType.CPU
        and event.name == "synthetic_model_compute"
    ]
    compute_kernel_names = set().union(
        *(_descendant_kernel_names(marker) for marker in markers)
    )
    compute = [
        event for event in cuda_events if event.name in compute_kernel_names
    ]
    assert markers, "torch.profiler recorded no synthetic model marker"
    assert compute_kernel_names, (
        "synthetic model marker launched no CUDA kernels"
    )
    assert h2d, "torch.profiler recorded no CUDA H2D activity"
    assert compute, "torch.profiler recorded no synthetic model CUDA kernels"
    assert any(
        _overlaps(copy_event, compute_event)
        for copy_event in h2d
        for compute_event in compute
    ), "torch.profiler found no H2D/model-compute overlap after warmup"


def main() -> int:
    if not torch.cuda.is_available() or torch.cuda.device_count() < 1:
        print(
            "CUDA required: Task11 device-prefetch qualification needs a real "
            "CUDA runner; no CUDA device is available",
            file=sys.stderr,
        )
        return 2

    device = torch.device("cuda", 0)
    for depth in (1, 3):
        for kind in ("data", "heterodata"):
            _qualify_depth_and_native_kind(device, depth, kind)
    _qualify_already_resident(device)
    _qualify_error_cleanup(device)
    _qualify_ready_yield_does_not_wait_for_future_source(device)
    _qualify_profiler_overlap(device)
    print("CUDA device-prefetch qualification passed")
    return 0


if __name__ == "__main__":
    raise SystemExit(main())
