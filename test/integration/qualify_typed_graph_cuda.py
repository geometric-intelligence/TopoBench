"""Mandatory CUDA release qualification for typed disk graph strategies.

The runner fails when CUDA or a required Parquet/partition/sampling backend is
missing.  It compares synchronous, host-only, and CUDA depths one and three,
executes exact materialized functional oracles, and publishes aggregate-only
qualification evidence with thresholds fixed before any measurement.
"""

# ruff: noqa: E402
from __future__ import annotations

import hashlib
import json
import math
import os
import statistics
import sys
import tempfile
import time
from collections.abc import Iterable, Iterator, Sequence
from dataclasses import asdict, dataclass, replace
from importlib import metadata
from pathlib import Path
from typing import Any, Literal

_REPOSITORY_ROOT = Path(__file__).resolve().parents[2]
if not (
    (_REPOSITORY_ROOT / "pyproject.toml").is_file()
    and (_REPOSITORY_ROOT / "topobench").is_dir()
    and (_REPOSITORY_ROOT / "test").is_dir()
):
    raise RuntimeError("cannot establish the TopoBench repository import root")
repository_root_text = str(_REPOSITORY_ROOT)
if repository_root_text not in sys.path:
    sys.path.insert(0, repository_root_text)


import torch
from torch import Tensor
from torch.profiler import ProfilerActivity, profile, record_function
from torch_geometric.data import Data, HeteroData
from torch_geometric.nn.models import GCN

from topobench.data.stores.typed_graph_store import TypedGraphStore
from topobench.dataloader.device_prefetch import (
    DevicePrefetchLoader,
    PrefetchLimits,
    estimate_batch_bytes,
)
from topobench.dataloader.disk_graph import (
    HeterogeneousClusterStrategy,
    HeterogeneousNeighborStrategy,
    HomogeneousClusterStrategy,
    SamplingDescriptor,
)
from topobench.dataloader.input_monitor import InputMonitor
from topobench.nn.backbones.heterogeneous.hgt import HGTBackbone
from topobench.profiling.execution_events import (
    ExecutionOperation,
    ExecutionStatus,
)

_MIB = 1024**2
QUALIFICATION_SEEDS = (17, 29, 43, 71, 101)
DEVICE_DEPTHS = (1, 3)
NUM_CLASSES = 4
NUM_STEPS = int(os.environ.get("TOPOBENCH_CUDA_QUALIFICATION_STEPS", "32"))
WARMUP_STEPS = 5
COMPUTE_REPEATS = int(os.environ.get("TOPOBENCH_CUDA_COMPUTE_REPEATS", "8"))
MAX_INPUT_STALL_FRACTION = 0.05
MAX_CONSECUTIVE_STARVED_STEPS = 3
MAX_DEGRADATION = 0.02
CONFIDENCE_LEVEL = 0.95
CONFIDENCE_Z = 1.96
ORACLE_ATOL = 5e-6
ORACLE_RTOL = 1e-5
ORACLE_METRIC_ATOL = 0.0
MAX_HOST_BATCH_BYTES = 64 * _MIB
MAX_DEVICE_BATCH_BYTES = 64 * _MIB
MAX_HOST_QUEUE_BYTES = 256 * _MIB
MAX_DEVICE_QUEUE_BYTES = 192 * _MIB
MAX_GPU_ALLOCATED_BYTES = 512 * _MIB
MAX_BATCH_NODES = 1_000_000
MAX_BATCH_EDGES = 4_000_000
PROFILE_PAYLOAD_BYTES = 32 * _MIB
PROFILE_MATRIX_WIDTH = 2048
PROFILE_STEPS = 6
PAIRED_NEIGHBOR_BATCH_SIZE = 2
PAIRED_NEIGHBOR_FANOUT = (4, 2)

NativeBatch = Data | HeteroData
ModelKind = Literal["gcn", "hgt"]


class _GCNQualificationModel(torch.nn.Module):
    def __init__(self, in_channels: int) -> None:
        super().__init__()
        self.network = GCN(
            in_channels=in_channels,
            hidden_channels=32,
            num_layers=2,
            out_channels=NUM_CLASSES,
            dropout=0.0,
            act="relu",
        )

    def forward(self, batch: NativeBatch) -> Tensor:
        if not isinstance(batch, Data):
            raise TypeError("GCN qualification requires homogeneous Data")
        return self.network(batch.x, batch.edge_index)


class _HGTQualificationModel(torch.nn.Module):
    def __init__(self, sample: HeteroData, target_node_type: str) -> None:
        super().__init__()
        self.target_node_type = target_node_type
        self.project = torch.nn.ModuleDict(
            {
                node_type: torch.nn.Linear(int(sample[node_type].x.shape[1]), 32)
                for node_type in sample.node_types
            }
        )
        self.backbone = HGTBackbone(
            sample.metadata(),
            hidden_channels=32,
            num_layers=2,
            heads=2,
            dropout=0.0,
            activation="relu",
        )
        self.head = torch.nn.Linear(32, NUM_CLASSES)

    def forward(self, batch: NativeBatch) -> Tensor:
        if not isinstance(batch, HeteroData):
            raise TypeError("HGT qualification requires HeteroData")
        features = {
            node_type: self.project[node_type](batch[node_type].x.float())
            for node_type in batch.node_types
        }
        hidden = self.backbone(features, batch.edge_index_dict)
        return self.head(hidden[self.target_node_type])


@dataclass(frozen=True, slots=True)
class _Workload:
    name: str
    model_kind: ModelKind
    store: TypedGraphStore
    strategy: Any
    descriptors: tuple[SamplingDescriptor, ...]
    target_node_type: str | None


class _MeasuredBatchSource(Iterable[NativeBatch]):
    """Repeat exact strategy descriptors while recording real selected reads."""

    def __init__(
        self,
        workload: _Workload,
        monitor: InputMonitor,
        *,
        steps: int,
        profile_payload_bytes: int = 0,
    ) -> None:
        self.workload = workload
        self.monitor = monitor
        self.steps = steps
        self.profile_payload_bytes = profile_payload_bytes

    def __iter__(self) -> Iterator[NativeBatch]:
        for index in range(self.steps):
            descriptor = self.workload.descriptors[index % len(self.workload.descriptors)]
            read = self.monitor.begin(
                ExecutionOperation.SELECTED_READ,
                phase="qualification",
                split="train",
                descriptor_sequence=index + 1,
                descriptor_identity=descriptor,
            )
            batch = self.workload.strategy.materialize(
                self.workload.store,
                descriptor,
            )
            estimate = estimate_batch_bytes(batch)
            self.monitor.finish(
                read,
                node_count=estimate.node_count,
                edge_count=estimate.edge_count,
                unique_storage_bytes=estimate.total_bytes,
            )
            assembly = self.monitor.begin(
                ExecutionOperation.NATIVE_ASSEMBLY,
                phase="qualification",
                split="train",
                descriptor_sequence=index + 1,
                descriptor_identity=descriptor,
            )
            batch.sequence_id = index + 1
            if self.profile_payload_bytes:
                count = self.profile_payload_bytes // torch.empty((), dtype=torch.float32).element_size()
                batch.qualification_payload = torch.arange(count, dtype=torch.float32)
            self.monitor.finish(
                assembly,
                node_count=estimate.node_count,
                edge_count=estimate.edge_count,
                unique_storage_bytes=estimate.total_bytes + self.profile_payload_bytes,
            )
            yield batch


def _version(package: str) -> str:
    try:
        return metadata.version(package)
    except metadata.PackageNotFoundError as error:
        raise RuntimeError(f"mandatory package {package!r} is unavailable") from error


def _cuda_prerequisites() -> dict[str, Any]:
    versions = {
        package: _version(package)
        for package in (
            "duckdb",
            "pyarrow",
            "torch",
            "torch-geometric",
            "torch-sparse",
        )
    }
    if not torch.cuda.is_available() or torch.cuda.device_count() < 1:
        raise RuntimeError(
            "mandatory CUDA qualification requires a visible CUDA device; "
            "CPU-only execution is a release-gate failure"
        )
    from torch_geometric.distributed import Partitioner  # noqa: F401
    from torch_geometric.loader import NeighborLoader  # noqa: F401

    device = torch.device("cuda", 0)
    properties = torch.cuda.get_device_properties(device)
    return {
        "versions": versions,
        "cuda_runtime": torch.version.cuda,
        "cuda_device_count": torch.cuda.device_count(),
        "cuda_device_name": properties.name,
        "cuda_total_memory_bytes": properties.total_memory,
        "cuda_driver_available": True,
        "partition_backend": "torch_geometric.distributed.Partitioner",
        "sampling_backend": "torch-sparse NeighborLoader",
    }


def _partition_backend_smoke(root: Path) -> dict[str, Any]:
    """Execute the real partition operator required by the CUDA release gate."""
    from test.data.stores.test_topology_only_pyg_partitioner import (
        asymmetric_typed_source,
    )
    from topobench.data.stores.pyg_partitioner import (
        TopologyOnlyPyGPartitioner,
    )
    from topobench.data.stores.typed_graph_ingestion import (
        ParquetTypedGraphIngestor,
    )
    from topobench.data.stores.typed_partition_book import (
        PartitionQualificationLimits,
    )

    source = asymmetric_typed_source(root / "source", num_partitions=2)
    ingestor = ParquetTypedGraphIngestor(source, root / "stores")
    relations = ingestor.build_relations()
    book = TopologyOnlyPyGPartitioner(ingestor, relations).generate(
        PartitionQualificationLimits()
    )
    try:
        if book.backend != "pyg" or not all(
            check.passed for check in book.qualification_checks
        ):
            raise RuntimeError(
                "mandatory partition backend smoke did not produce a "
                "qualified PyG partition book"
            )
        return {
            "backend": book.backend,
            "backend_version": book.backend_version,
            "num_partitions": book.num_partitions,
            "qualification_check_ids": [
                check.check_id for check in book.qualification_checks
            ],
            "partition_book_identity": book.content_identity,
        }
    finally:
        book.close()


def _limits(depth: int) -> PrefetchLimits:
    return PrefetchLimits(
        host_queue_depth=3,
        device_queue_depth=depth,
        max_batch_nodes=MAX_BATCH_NODES,
        max_batch_edges=MAX_BATCH_EDGES,
        max_host_batch_bytes=MAX_HOST_BATCH_BYTES,
        max_device_batch_bytes=MAX_DEVICE_BATCH_BYTES,
        max_host_queue_bytes=MAX_HOST_QUEUE_BYTES,
        max_device_queue_bytes=(0 if depth == 0 else MAX_DEVICE_QUEUE_BYTES),
    )


def _digest_descriptors(descriptors: Sequence[SamplingDescriptor]) -> str:
    digest = hashlib.sha256()
    for descriptor in descriptors:
        payload = json.dumps(asdict(descriptor), sort_keys=True, separators=(",", ":"))
        encoded = payload.encode("utf-8")
        digest.update(len(encoded).to_bytes(8, "big"))
        digest.update(encoded)
    return digest.hexdigest()


def _target_store(batch: NativeBatch, target_node_type: str | None) -> Any:
    return batch if isinstance(batch, Data) else batch[target_node_type]


def _loss_and_accuracy(
    logits: Tensor,
    batch: NativeBatch,
    target_node_type: str | None,
) -> tuple[Tensor, Tensor]:
    storage = _target_store(batch, target_node_type)
    mask = storage.supervised_mask
    if (
        not isinstance(mask, Tensor)
        or mask.dtype != torch.bool
        or mask.ndim != 1
        or mask.numel() == 0
    ):
        raise RuntimeError("qualification batch has invalid supervision")
    labels = storage.y[mask].long()
    selected = logits[mask]
    loss = torch.nn.functional.cross_entropy(selected, labels)
    accuracy = (selected.argmax(dim=-1) == labels).float().mean()
    return loss, accuracy


def _model(
    workload: _Workload,
    sample: NativeBatch,
    *,
    seed: int,
    device: torch.device,
) -> torch.nn.Module:
    torch.manual_seed(seed)
    if workload.model_kind == "gcn":
        assert isinstance(sample, Data)
        result: torch.nn.Module = _GCNQualificationModel(int(sample.x.shape[1]))
    else:
        assert isinstance(sample, HeteroData)
        assert workload.target_node_type is not None
        result = _HGTQualificationModel(sample, workload.target_node_type)
    return result.to(device)


def _event_quantiles(events: Sequence[Any]) -> dict[str, dict[str, int]]:
    grouped: dict[str, list[int]] = {}
    for event in events:
        if event.status is not ExecutionStatus.SUCCESS:
            continue
        grouped.setdefault(event.operation.value, []).append(int(event.duration_ns))
    result: dict[str, dict[str, int]] = {}
    for operation, values in sorted(grouped.items()):
        ordered = sorted(values)
        result[operation] = {
            "count": len(ordered),
            "p50_ns": ordered[math.ceil(0.50 * len(ordered)) - 1],
            "p95_ns": ordered[math.ceil(0.95 * len(ordered)) - 1],
            "p99_ns": ordered[math.ceil(0.99 * len(ordered)) - 1],
        }
    return result


def _has_cpu_disk_overlap(events: Sequence[Any]) -> bool:
    selected = [
        event
        for event in events
        if event.operation
        in {ExecutionOperation.SELECTED_READ, ExecutionOperation.NATIVE_ASSEMBLY}
    ]
    compute = [
        event
        for event in events
        if event.operation is ExecutionOperation.MODEL_COMPUTE
    ]
    return any(
        max(left.monotonic_ns, right.monotonic_ns)
        < min(
            left.monotonic_ns + left.duration_ns,
            right.monotonic_ns + right.duration_ns,
        )
        for left in selected
        for right in compute
    )


def _run_mode(
    workload: _Workload,
    mode: Literal["synchronous", "host-only", "device-depth-1", "device-depth-3"],
    device: torch.device,
) -> dict[str, Any]:
    depth = 0 if mode in {"synchronous", "host-only"} else int(mode.rsplit("-", 1)[1])
    monitor = InputMonitor(
        cuda_event_factory=lambda: torch.cuda.Event(enable_timing=True),
        warmup_steps=WARMUP_STEPS,
        rolling_window_steps=NUM_STEPS - WARMUP_STEPS,
        max_input_stall_fraction=MAX_INPUT_STALL_FRACTION,
        max_consecutive_starved_steps=MAX_CONSECUTIVE_STARVED_STEPS,
        patience_windows=1,
        stall_action="error",
    )
    source = _MeasuredBatchSource(workload, monitor, steps=NUM_STEPS)
    loader: Iterable[NativeBatch]
    prefetch: DevicePrefetchLoader | None = None
    if mode == "synchronous":
        loader = source
    else:
        prefetch = DevicePrefetchLoader(
            source,
            _limits(depth),
            device=(device if depth else "cpu"),
            execution_monitor=monitor,
            monitor_phase="qualification",
            monitor_split="train",
        )
        loader = prefetch
    iterator = iter(loader)
    first = next(iterator)
    canonical_order = [first.sampling_descriptor]
    if any(
        tensor.device.type == "cuda"
        for storage in first.stores
        for tensor in storage.values()
        if isinstance(tensor, Tensor)
    ):
        sample = first
    else:
        sample = first.to(device, non_blocking=False)
    model = _model(workload, sample, seed=QUALIFICATION_SEEDS[0], device=device)
    torch.cuda.empty_cache()
    torch.cuda.reset_peak_memory_stats(device)
    loss_tensors: list[Tensor] = []
    accuracy_tensors: list[Tensor] = []
    wait_ns: list[int] = []
    compute_pairs: list[tuple[torch.cuda.Event, torch.cuda.Event]] = []

    def execute(batch: NativeBatch, sequence: int) -> None:
        if not all(
            tensor.device.type == "cuda"
            for storage in batch.stores
            for tensor in storage.values()
            if isinstance(tensor, Tensor)
        ):
            batch = batch.to(device, non_blocking=mode == "host-only")
        descriptor = batch.sampling_descriptor
        token = monitor.begin_model_compute(
            phase="qualification",
            split="train",
            descriptor_sequence=sequence,
            descriptor_identity=descriptor,
            global_step=sequence,
            cuda_timing=True,
        )
        start = torch.cuda.Event(enable_timing=True)
        end = torch.cuda.Event(enable_timing=True)
        start.record()
        loss: Tensor | None = None
        accuracy: Tensor | None = None
        for _ in range(COMPUTE_REPEATS):
            logits = model(batch)
            loss, accuracy = _loss_and_accuracy(
                logits,
                batch,
                workload.target_node_type,
            )
        assert loss is not None and accuracy is not None
        end.record()
        monitor.finish_model_compute(token)
        monitor.record(
            ExecutionOperation.OPTIMIZER,
            phase="qualification",
            split="train",
            global_step=sequence,
            descriptor_sequence=sequence,
            descriptor_identity=descriptor,
            duration_ns=0,
        )
        loss_tensors.append(loss.detach())
        accuracy_tensors.append(accuracy.detach())
        compute_pairs.append((start, end))

    execute(sample, 1)
    for sequence in range(2, NUM_STEPS + 1):
        started = time.perf_counter_ns()
        batch = next(iterator)
        wait_ns.append(time.perf_counter_ns() - started)
        canonical_order.append(batch.sampling_descriptor)
        execute(batch, sequence)
    try:
        next(iterator)
    except StopIteration:
        pass
    else:
        raise RuntimeError("qualification source exceeded its declared finite length")
    torch.cuda.synchronize(device)
    losses = [float(value) for value in loss_tensors]
    accuracies = [float(value) for value in accuracy_tensors]
    compute_ns = [round(start.elapsed_time(end) * 1_000_000) for start, end in compute_pairs]
    steady_wait = wait_ns[WARMUP_STEPS:]
    steady_compute = compute_ns[WARMUP_STEPS + 1 :]
    wait_total = sum(steady_wait)
    compute_total = sum(steady_compute)
    stall_fraction = wait_total / (wait_total + compute_total)
    if stall_fraction > MAX_INPUT_STALL_FRACTION:
        raise RuntimeError(
            f"{workload.name}/{mode} input stall {stall_fraction:.6f} exceeds "
            f"{MAX_INPUT_STALL_FRACTION:.6f}"
        )
    monitor.poll()
    events = monitor.drain()
    if any(event.status is ExecutionStatus.ERROR for event in events):
        raise RuntimeError(f"{workload.name}/{mode} emitted error profiling evidence")
    operations = {event.operation for event in events}
    required = {
        ExecutionOperation.SELECTED_READ,
        ExecutionOperation.NATIVE_ASSEMBLY,
        ExecutionOperation.MODEL_COMPUTE,
        ExecutionOperation.OPTIMIZER,
    }
    if mode.startswith("device"):
        required |= {
            ExecutionOperation.HOST_PIN,
            ExecutionOperation.H2D_QUEUE,
            ExecutionOperation.H2D_COPY,
        }
    missing = required - operations
    if missing:
        raise RuntimeError(f"{workload.name}/{mode} missing evidence {sorted(item.value for item in missing)}")
    max_host_bytes = 0
    max_device_bytes = 0
    if prefetch is not None:
        assert hasattr(iterator, "max_host_queued_bytes")
        max_host_bytes = int(iterator.max_host_queued_bytes)
        max_device_bytes = int(iterator.max_device_queued_bytes)
        if max_host_bytes > MAX_HOST_QUEUE_BYTES:
            raise RuntimeError("observed pinned-host queue exceeded its budget")
        if max_device_bytes > (0 if depth == 0 else MAX_DEVICE_QUEUE_BYTES):
            raise RuntimeError("observed device queue exceeded its budget")
        prefetch.close()
    gpu_peak = int(torch.cuda.max_memory_allocated(device))
    if gpu_peak > MAX_GPU_ALLOCATED_BYTES:
        raise RuntimeError("observed CUDA allocation exceeded its predeclared budget")
    monitor.close()
    if not all(math.isfinite(value) for value in losses):
        raise RuntimeError(f"{workload.name}/{mode} produced non-finite loss")
    return {
        "mode": mode,
        "finite_loss_mean": statistics.fmean(losses),
        "accuracy_mean": statistics.fmean(accuracies),
        "descriptor_order_sha256": _digest_descriptors(canonical_order),
        "input_stall_fraction": stall_fraction,
        "max_host_queue_bytes": max_host_bytes,
        "max_device_queue_bytes": max_device_bytes,
        "gpu_peak_allocated_bytes": gpu_peak,
        "timing_quantiles": _event_quantiles(events),
        "disk_cpu_compute_overlap": _has_cpu_disk_overlap(events),
    }


def _descendant_kernel_names(event: Any) -> set[str]:
    names: set[str] = set()
    pending = list(event.cpu_children)
    while pending:
        child = pending.pop()
        names.update(kernel.name for kernel in child.kernels)
        pending.extend(child.cpu_children)
    return names


def _overlaps(left: Any, right: Any) -> bool:
    return max(left.time_range.start, right.time_range.start) < min(
        left.time_range.end,
        right.time_range.end,
    )


def _profile_overlap(
    workload: _Workload,
    device: torch.device,
    *,
    device_depth: int,
) -> dict[str, bool | int]:
    if device_depth not in DEVICE_DEPTHS:
        raise ValueError(
            f"unsupported qualification device depth {device_depth}"
        )
    monitor = InputMonitor(
        cuda_event_factory=lambda: torch.cuda.Event(enable_timing=True)
    )
    source = _MeasuredBatchSource(
        workload,
        monitor,
        steps=PROFILE_STEPS,
        profile_payload_bytes=PROFILE_PAYLOAD_BYTES,
    )
    loader = DevicePrefetchLoader(
        source,
        PrefetchLimits(
            host_queue_depth=3,
            device_queue_depth=device_depth,
            max_batch_nodes=MAX_BATCH_NODES,
            max_batch_edges=MAX_BATCH_EDGES,
            max_host_batch_bytes=MAX_HOST_BATCH_BYTES,
            max_device_batch_bytes=MAX_DEVICE_BATCH_BYTES,
            max_host_queue_bytes=MAX_HOST_QUEUE_BYTES,
            max_device_queue_bytes=MAX_DEVICE_QUEUE_BYTES,
        ),
        device=device,
        execution_monitor=monitor,
        monitor_phase="qualification",
        monitor_split="train",
    )
    matrix = torch.randn(
        (PROFILE_MATRIX_WIDTH, PROFILE_MATRIX_WIDTH),
        device=device,
    )

    def consume() -> Tensor:
        result = matrix
        for batch in loader:
            with record_function("typed_graph_model_compute"):
                result = torch.mm(matrix, matrix)
                result.add_(batch.qualification_payload[0])
        return result

    consume()
    torch.cuda.synchronize(device)
    with profile(
        activities=[ProfilerActivity.CPU, ProfilerActivity.CUDA]
    ) as trace:
        result = consume()
    torch.cuda.synchronize(device)
    if not bool(torch.isfinite(result).all()):
        raise RuntimeError("overlap workload produced non-finite output")
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
        and event.name == "typed_graph_model_compute"
    ]
    kernel_names = set().union(
        *(_descendant_kernel_names(marker) for marker in markers)
    )
    compute = [event for event in cuda_events if event.name in kernel_names]
    if not markers or not h2d or not compute:
        raise RuntimeError(
            "profiler evidence is missing model or H2D activity"
        )
    h2d_compute = any(
        _overlaps(copy_event, compute_event)
        for copy_event in h2d
        for compute_event in compute
    )
    monitor.poll()
    monitored = monitor.drain()
    disk_cpu_compute = _has_cpu_disk_overlap(monitored)
    loader.close()
    monitor.close()
    if not h2d_compute or not disk_cpu_compute:
        raise RuntimeError(
            "profiler did not prove disk/CPU/H2D overlap with model compute"
        )
    return {
        "device_queue_depth": device_depth,
        "profiler_h2d_compute_overlap": h2d_compute,
        "structured_disk_cpu_compute_overlap": disk_cpu_compute,
    }


def _allclose_or_fail(actual: Tensor, expected: Tensor, contract: str) -> float:
    difference = float((actual - expected).abs().max()) if actual.numel() else 0.0
    if not torch.allclose(actual, expected, atol=ORACLE_ATOL, rtol=ORACLE_RTOL):
        raise RuntimeError(f"{contract} functional oracle failed; max_abs={difference}")
    return difference


def _metric_error_or_fail(
    actual: float | Tensor,
    expected: float | Tensor,
    contract: str,
) -> float:
    actual_value = float(actual.detach().item()) if isinstance(actual, Tensor) else float(actual)
    expected_value = (
        float(expected.detach().item())
        if isinstance(expected, Tensor)
        else float(expected)
    )
    difference = abs(actual_value - expected_value)
    if difference > ORACLE_METRIC_ATOL:
        raise RuntimeError(
            f"{contract} metric oracle failed; abs_error={difference}"
        )
    return difference


def _functional_oracles(
    homogeneous: Any,
    heterogeneous: Any,
    device: torch.device,
) -> dict[str, Any]:
    from test.data.dataload.test_disk_graph_datamodule import (
        materialized_heterogeneous_reference,
        materialized_homogeneous_reference,
    )
    result: dict[str, Any] = {}
    with TypedGraphStore.open(homogeneous.store_build.path) as store:
        full = materialized_homogeneous_reference(store)
        strategy = HomogeneousClusterStrategy(
            partition_groups=(tuple(range(store.num_partitions)),),
            seed=QUALIFICATION_SEEDS[0],
        )
        descriptor = strategy.setup(
            store,
            phase="train",
            active_split_tag=store.active_split_tag,
            shuffle=False,
        )[0]
        disk = strategy.materialize(store, descriptor)
        model = _model(
            _Workload("gcn-cluster", "gcn", store, strategy, (descriptor,), None),
            disk,
            seed=QUALIFICATION_SEEDS[0],
            device=device,
        ).eval()
        with torch.no_grad():
            disk_device = disk.to(device)
            full_device = full.to(device)
            disk_logits = model(disk_device)
            full_logits = model(full_device).index_select(0, disk_device.global_nid)
        result["gcn_all_partitions"] = {
            "max_abs_logit_error": _allclose_or_fail(
                disk_logits,
                full_logits,
                "GCN all-partition union",
            ),
            "metric_abs_error": _metric_error_or_fail(
                _loss_and_accuracy(disk_logits, disk_device, None)[1],
                _loss_and_accuracy(full_logits, disk_device, None)[1],
                "GCN all-partition union",
            ),
        }

    with TypedGraphStore.open(heterogeneous.store_build.path) as store:
        full = materialized_heterogeneous_reference(store)
        target = store._manifest["target_node_type"]
        cluster = HeterogeneousClusterStrategy(
            partition_groups=(tuple(range(store.num_partitions)),),
            seed=QUALIFICATION_SEEDS[0],
        )
        cluster_descriptor = cluster.setup(
            store,
            phase="train",
            active_split_tag=store.active_split_tag,
            shuffle=False,
        )[0]
        disk_cluster = cluster.materialize(store, cluster_descriptor)
        cluster_model = _model(
            _Workload(
                "hgt-cluster",
                "hgt",
                store,
                cluster,
                (cluster_descriptor,),
                target,
            ),
            disk_cluster,
            seed=QUALIFICATION_SEEDS[0],
            device=device,
        ).eval()
        with torch.no_grad():
            disk_device = disk_cluster.to(device)
            full_device = full.to(device)
            disk_logits = cluster_model(disk_device)
            full_logits = cluster_model(full_device).index_select(
                0,
                disk_device[target].n_id,
            )
        result["hgt_all_partitions"] = {
            "max_abs_logit_error": _allclose_or_fail(
                disk_logits,
                full_logits,
                "HGT all-partition union",
            ),
            "metric_abs_error": _metric_error_or_fail(
                _loss_and_accuracy(disk_logits, disk_device, target)[1],
                _loss_and_accuracy(full_logits, disk_device, target)[1],
                "HGT all-partition union",
            ),
        }

        fanout = {relation: [-1, -1] for relation in store.relation_types}
        batch_size = len(store.split_ids(store.active_split_tag, "train"))
        neighbor = HeterogeneousNeighborStrategy(
            batch_size=batch_size,
            num_neighbors=fanout,
            seed=QUALIFICATION_SEEDS[0],
        )
        neighbor_descriptor = neighbor.setup(
            store,
            phase="train",
            active_split_tag=store.active_split_tag,
            shuffle=False,
        )[0]
        disk_neighbor = neighbor.materialize(store, neighbor_descriptor)
        neighbor_model = _model(
            _Workload(
                "hgt-neighbor",
                "hgt",
                store,
                neighbor,
                (neighbor_descriptor,),
                target,
            ),
            disk_neighbor,
            seed=QUALIFICATION_SEEDS[0],
            device=device,
        ).eval()
        with torch.no_grad():
            neighbor_device = disk_neighbor.to(device)
            full_device = full.to(device)
            neighbor_all_logits = neighbor_model(neighbor_device)
            seed_count = int(neighbor_device[target].batch_size)
            neighbor_logits = neighbor_all_logits[:seed_count]
            full_logits = neighbor_model(full_device).index_select(
                0,
                neighbor_device[target].n_id[:seed_count],
            )
            labels = neighbor_device[target].y[:seed_count].long()
            neighbor_accuracy = float(
                (neighbor_logits.argmax(dim=-1) == labels).float().mean()
            )
            full_accuracy = float(
                (full_logits.argmax(dim=-1) == labels).float().mean()
            )
        result["hgt_exhaustive_neighbor"] = {
            "max_abs_logit_error": _allclose_or_fail(
                neighbor_logits,
                full_logits,
                "HGT exhaustive relation fanout",
            ),
            "metric_abs_error": _metric_error_or_fail(
                neighbor_accuracy,
                full_accuracy,
                "HGT exhaustive relation fanout",
            ),
            "hop_count": 2,
            "fanout": "exhaustive-per-relation",
        }
    return result


def _one_training_result(
    workload: _Workload,
    disk_batch: NativeBatch,
    reference_batch: NativeBatch,
    *,
    seed: int,
    device: torch.device,
) -> tuple[float, float]:
    disk_batch = disk_batch.to(device)
    reference_batch = reference_batch.to(device)
    disk_model = _model(workload, disk_batch, seed=seed, device=device)
    reference_model = _model(workload, reference_batch, seed=seed, device=device)
    reference_model.load_state_dict(disk_model.state_dict())

    def train(model: torch.nn.Module, batch: NativeBatch) -> float:
        optimizer = torch.optim.SGD(model.parameters(), lr=0.01)
        optimizer.zero_grad(set_to_none=True)
        logits = model(batch)
        loss, _ = _loss_and_accuracy(logits, batch, workload.target_node_type)
        loss.backward()
        optimizer.step()
        with torch.no_grad():
            logits = model(batch)
            _, accuracy = _loss_and_accuracy(
                logits,
                batch,
                workload.target_node_type,
            )
        return float(accuracy)

    return train(disk_model, disk_batch), train(reference_model, reference_batch)


def _statistics(values: Sequence[float]) -> dict[str, float | list[float]]:
    mean = statistics.fmean(values)
    standard_deviation = statistics.stdev(values)
    half_width = CONFIDENCE_Z * standard_deviation / math.sqrt(len(values))
    return {
        "mean": mean,
        "standard_deviation": standard_deviation,
        "confidence_level": CONFIDENCE_LEVEL,
        "confidence_interval": [mean - half_width, mean + half_width],
    }


def _paired_seed_qualification(
    homogeneous: Any,
    heterogeneous: Any,
    device: torch.device,
) -> dict[str, Any]:
    from test.data.dataload.test_disk_graph_datamodule import (
        materialized_heterogeneous_reference,
        materialized_homogeneous_reference,
    )
    all_results: dict[str, Any] = {}
    with TypedGraphStore.open(homogeneous.store_build.path) as store:
        full = materialized_homogeneous_reference(store)
        partition = int(store.partition_assignment(store.node_types[0])[0])
        for seed in QUALIFICATION_SEEDS:
            strategy = HomogeneousClusterStrategy(
                partition_groups=((partition,),),
                seed=seed,
            )
            descriptor = strategy.setup(
                store,
                phase="train",
                active_split_tag=store.active_split_tag,
                shuffle=False,
            )[0]
            disk = strategy.materialize(store, descriptor)
            reference = full.subgraph(disk.global_nid)
            reference.supervised_mask = reference.train_mask.clone()
            workload = _Workload(
                "gcn-cluster",
                "gcn",
                store,
                strategy,
                (descriptor,),
                None,
            )
            disk_metric, reference_metric = _one_training_result(
                workload,
                disk,
                reference,
                seed=seed,
                device=device,
            )
            all_results.setdefault("gcn-cluster", []).append(
                (seed, disk_metric, reference_metric)
            )

    with TypedGraphStore.open(heterogeneous.store_build.path) as store:
        full = materialized_heterogeneous_reference(store)
        target = store._manifest["target_node_type"]
        train_partition = int(
            store.partition_assignment(target)[
                int(store.split_ids(store.active_split_tag, "train")[0])
            ]
        )
        for seed in QUALIFICATION_SEEDS:
            cluster = HeterogeneousClusterStrategy(
                partition_groups=((train_partition,),),
                seed=seed,
            )
            descriptor = cluster.setup(
                store,
                phase="train",
                active_split_tag=store.active_split_tag,
                shuffle=False,
            )[0]
            disk = cluster.materialize(store, descriptor)
            materialized = HeterogeneousClusterStrategy(
                partition_book=heterogeneous.partition_build.book,
                partition_groups=((train_partition,),),
                seed=seed,
            )
            reference_descriptor = materialized.setup(
                full,
                phase="train",
                active_split_tag=store.active_split_tag,
                shuffle=False,
            )[0]
            reference = materialized.materialize(full, reference_descriptor)
            workload = _Workload(
                "hgt-cluster",
                "hgt",
                store,
                cluster,
                (descriptor,),
                target,
            )
            disk_metric, reference_metric = _one_training_result(
                workload,
                disk,
                reference,
                seed=seed,
                device=device,
            )
            all_results.setdefault("hgt-cluster", []).append(
                (seed, disk_metric, reference_metric)
            )

            fanout = {
                relation: list(PAIRED_NEIGHBOR_FANOUT)
                for relation in store.relation_types
            }
            neighbor = HeterogeneousNeighborStrategy(
                batch_size=PAIRED_NEIGHBOR_BATCH_SIZE,
                num_neighbors=fanout,
                seed=seed,
            )
            descriptor = neighbor.setup(
                store,
                phase="train",
                active_split_tag=store.active_split_tag,
                shuffle=True,
            )[0]
            disk = neighbor.materialize(store, descriptor)
            materialized_neighbor = HeterogeneousNeighborStrategy(
                batch_size=PAIRED_NEIGHBOR_BATCH_SIZE,
                num_neighbors=fanout,
                seed=seed,
            )
            reference_descriptor = materialized_neighbor.setup(
                full,
                phase="train",
                active_split_tag=store.active_split_tag,
                shuffle=True,
            )[0]
            reference_descriptor = replace(
                reference_descriptor,
                target_seed_ids=descriptor.target_seed_ids,
                generator_seed=descriptor.generator_seed,
                generator_state_sha256=(
                    descriptor.generator_state_sha256
                ),
            )
            reference = materialized_neighbor.materialize(full, reference_descriptor)
            workload = _Workload(
                "hgt-neighbor",
                "hgt",
                store,
                neighbor,
                (descriptor,),
                target,
            )
            disk_metric, reference_metric = _one_training_result(
                workload,
                disk,
                reference,
                seed=seed,
                device=device,
            )
            all_results.setdefault("hgt-neighbor", []).append(
                (seed, disk_metric, reference_metric)
            )

    aggregate: dict[str, Any] = {}
    for strategy, rows in sorted(all_results.items()):
        disk_values = [row[1] for row in rows]
        reference_values = [row[2] for row in rows]
        paired = [disk - reference for disk, reference in zip(disk_values, reference_values, strict=True)]
        degradation = statistics.fmean(reference_values) - statistics.fmean(disk_values)
        if degradation > MAX_DEGRADATION:
            raise RuntimeError(
                f"{strategy} paired degradation {degradation:.6f} exceeds "
                f"{MAX_DEGRADATION:.6f}"
            )
        aggregate[strategy] = {
            "results": [
                {
                    "seed": seed,
                    "disk_metric": disk,
                    "materialized_metric": reference,
                    "paired_difference": difference,
                }
                for (seed, disk, reference), difference in zip(rows, paired, strict=True)
            ],
            "disk": _statistics(disk_values),
            "materialized": _statistics(reference_values),
            "paired_difference": _statistics(paired),
            "observed_degradation": degradation,
            "maximum_degradation": MAX_DEGRADATION,
        }
        if strategy == "hgt-neighbor":
            aggregate[strategy]["sampling_contract"] = {
                "batch_size": PAIRED_NEIGHBOR_BATCH_SIZE,
                "fanout_per_hop": list(PAIRED_NEIGHBOR_FANOUT),
                "exhaustive": False,
                "paired_target_and_sampler_seed": True,
            }
    return aggregate


def _workloads(homogeneous: Any, heterogeneous: Any) -> tuple[_Workload, ...]:
    homogeneous_store = TypedGraphStore.open(homogeneous.store_build.path)
    target_ids = homogeneous_store.split_ids(
        homogeneous_store.active_split_tag,
        "train",
    )
    assignments = homogeneous_store.partition_assignment(homogeneous_store.node_types[0])
    groups = tuple((int(assignments[int(target_ids[index % len(target_ids)])]),) for index in range(NUM_STEPS))
    gcn = HomogeneousClusterStrategy(partition_groups=groups, seed=37)
    gcn_descriptors = gcn.setup(
        homogeneous_store,
        phase="train",
        active_split_tag=homogeneous_store.active_split_tag,
        shuffle=False,
    )

    heterogeneous_store = TypedGraphStore.open(heterogeneous.store_build.path)
    target = heterogeneous_store._manifest["target_node_type"]
    target_ids = heterogeneous_store.split_ids(
        heterogeneous_store.active_split_tag,
        "train",
    )
    assignments = heterogeneous_store.partition_assignment(target)
    groups = tuple((int(assignments[int(target_ids[index % len(target_ids)])]),) for index in range(NUM_STEPS))
    cluster = HeterogeneousClusterStrategy(partition_groups=groups, seed=41)
    cluster_descriptors = cluster.setup(
        heterogeneous_store,
        phase="train",
        active_split_tag=heterogeneous_store.active_split_tag,
        shuffle=False,
    )
    neighbor = HeterogeneousNeighborStrategy(
        batch_size=1,
        num_neighbors={
            relation: [-1, -1] for relation in heterogeneous_store.relation_types
        },
        seed=43,
    )
    neighbor_descriptors = neighbor.setup(
        heterogeneous_store,
        phase="train",
        active_split_tag=heterogeneous_store.active_split_tag,
        shuffle=False,
    )
    return (
        _Workload(
            "gcn-cluster",
            "gcn",
            homogeneous_store,
            gcn,
            gcn_descriptors,
            None,
        ),
        _Workload(
            "hgt-cluster",
            "hgt",
            heterogeneous_store,
            cluster,
            cluster_descriptors,
            target,
        ),
        _Workload(
            "hgt-neighbor",
            "hgt",
            heterogeneous_store,
            neighbor,
            neighbor_descriptors,
            target,
        ),
    )


def _store_provenance(workload: _Workload) -> dict[str, Any]:
    store = workload.store
    return {
        "store_fingerprint": store.content_sha256,
        "source_fingerprint": store._manifest["source_binding"]["source_fingerprint"],
        "partition_book_identity": store.partition_book_identity,
        "output_kind": store.output_kind,
        "schema_roles": {
            "node_type_count": len(store.node_types),
            "relation_count": len(store.relation_types),
            "target_node_type": store._manifest["target_node_type"],
            "active_split_tag": store.active_split_tag,
        },
        "representation": "typed-csc-mmap",
        "strategy_state": {
            **workload.strategy.sampler_state(),
            "options": json.loads(
                workload.descriptors[0].strategy_options_json
            ),
        },
        "queue_budgets": {
            "host_batch_bytes": MAX_HOST_BATCH_BYTES,
            "device_batch_bytes": MAX_DEVICE_BATCH_BYTES,
            "host_queue_bytes": MAX_HOST_QUEUE_BYTES,
            "device_queue_bytes": MAX_DEVICE_QUEUE_BYTES,
            "device_depths": list(DEVICE_DEPTHS),
        },
    }


def main() -> int:
    try:
        prerequisites = _cuda_prerequisites()
        from test.data.stores.test_topology_only_pyg_partitioner import (
            asymmetric_typed_source,
            homogeneous_source,
        )
        from test.data.stores.test_typed_graph_store import (
            _build_qualified_store,
        )
        from topobench.data.loaders.parquet import ParquetTypedGraphSource

        if NUM_STEPS <= WARMUP_STEPS + 1 or COMPUTE_REPEATS < 1:
            raise RuntimeError("CUDA qualification dimensions are too small")
        device = torch.device("cuda", 0)
        with tempfile.TemporaryDirectory(prefix="topobench-typed-cuda-") as directory:
            root = Path(directory)
            prerequisites["partition_backend_execution"] = (
                _partition_backend_smoke(root / "partition-backend-smoke")
            )
            homogeneous_source_value = homogeneous_source(root / "homogeneous-source")
            homogeneous_source_value = ParquetTypedGraphSource(
                replace(
                    homogeneous_source_value.spec,
                    partition=replace(
                        homogeneous_source_value.spec.partition,
                        num_partitions=4,
                        memory_limit_bytes=1,
                        external_partition_map="external/manifest.json",
                    ),
                )
            )
            homogeneous = _build_qualified_store(
                homogeneous_source_value,
                root / "homogeneous-stores",
            )
            heterogeneous = _build_qualified_store(
                asymmetric_typed_source(
                    root / "heterogeneous-source",
                    num_partitions=4,
                    memory_limit_bytes=1,
                    external_partition_map="external/manifest.json",
                ),
                root / "heterogeneous-stores",
            )
            workloads = _workloads(homogeneous, heterogeneous)
            mode_evidence: dict[str, Any] = {}
            overlap_evidence: dict[str, Any] = {}
            for workload in workloads:
                runs = [
                    _run_mode(workload, mode, device)
                    for mode in (
                        "synchronous",
                        "host-only",
                        "device-depth-1",
                        "device-depth-3",
                    )
                ]
                order = {run["descriptor_order_sha256"] for run in runs}
                if len(order) != 1:
                    raise RuntimeError(
                        f"{workload.name} descriptor order changed across prefetch modes"
                    )
                for device_run in (
                    run for run in runs if run["mode"].startswith("device-")
                ):
                    if not device_run["disk_cpu_compute_overlap"]:
                        raise RuntimeError(
                            f"{workload.name}/{device_run['mode']} did not overlap "
                            "disk/CPU work"
                        )
                profiler_by_mode = {
                    f"device-depth-{depth}": _profile_overlap(
                        workload,
                        device,
                        device_depth=depth,
                    )
                    for depth in DEVICE_DEPTHS
                }
                overlap_evidence[workload.name] = profiler_by_mode
                mode_evidence[workload.name] = {
                    "provenance": _store_provenance(workload),
                    "runs": runs,
                    "profiler_overlap_by_mode": profiler_by_mode,
                }
            oracles = _functional_oracles(homogeneous, heterogeneous, device)
            paired = _paired_seed_qualification(homogeneous, heterogeneous, device)
            for workload in workloads:
                workload.store.close()

        aggregate = {
            "schema_version": "typed-graph-cuda-qualification-v1",
            "status": "passed",
            "thresholds": {
                "seeds": list(QUALIFICATION_SEEDS),
                "max_input_stall_fraction": MAX_INPUT_STALL_FRACTION,
                "max_consecutive_starved_steps": MAX_CONSECUTIVE_STARVED_STEPS,
                "max_degradation": MAX_DEGRADATION,
                "confidence_level": CONFIDENCE_LEVEL,
                "oracle_atol": ORACLE_ATOL,
                "oracle_rtol": ORACLE_RTOL,
                "oracle_metric_atol": ORACLE_METRIC_ATOL,
                "paired_neighbor_batch_size": PAIRED_NEIGHBOR_BATCH_SIZE,
                "paired_neighbor_fanout": list(PAIRED_NEIGHBOR_FANOUT),
                "max_host_batch_bytes": MAX_HOST_BATCH_BYTES,
                "max_device_batch_bytes": MAX_DEVICE_BATCH_BYTES,
                "max_host_queue_bytes": MAX_HOST_QUEUE_BYTES,
                "max_device_queue_bytes": MAX_DEVICE_QUEUE_BYTES,
                "max_gpu_allocated_bytes": MAX_GPU_ALLOCATED_BYTES,
                "device_depths": list(DEVICE_DEPTHS),
            },
            "prerequisites": prerequisites,
            "workloads": mode_evidence,
            "overlap": overlap_evidence,
            "functional_oracles": oracles,
            "paired_seed_statistics": paired,
        }
        evidence_root = Path(
            os.environ.get(
                "TOPOBENCH_QUALIFICATION_EVIDENCE_DIR",
                "qualification-evidence",
            )
        )
        evidence_root.mkdir(parents=True, exist_ok=True)
        evidence_path = evidence_root / "typed-graph-cuda-qualification.json"
        evidence_path.write_text(
            json.dumps(aggregate, indent=2, sort_keys=True) + "\n",
            encoding="utf-8",
        )
        print(json.dumps(aggregate, sort_keys=True))
        return 0
    except Exception as error:
        print(
            json.dumps(
                {
                    "schema_version": "typed-graph-cuda-qualification-v1",
                    "status": "failed",
                    "error_type": type(error).__name__,
                    "error": str(error),
                },
                sort_keys=True,
            ),
            file=sys.stderr,
        )
        return 2


if __name__ == "__main__":
    raise SystemExit(main())
