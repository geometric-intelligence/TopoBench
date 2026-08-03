"""Bounded host production and CUDA-ready transfer for native PyG batches."""

from __future__ import annotations

import contextlib
import copy
import queue
import threading
from collections import deque
from collections.abc import Callable, Iterable, Iterator, Mapping, Sequence
from dataclasses import dataclass, field, replace
from numbers import Integral
from types import MappingProxyType
from typing import Literal, TypeAlias

import torch
from torch import Tensor
from torch.utils.data import DataLoader
from torch_geometric.data import Data, HeteroData

from topobench.dataloader.input_monitor import (
    InputMonitor,
    MonitorOverflowError,
    OperationToken,
    QueueSnapshot,
)
from topobench.profiling.execution_events import (
    ExecutionOperation,
    ExecutionStatus,
)

CanonicalRelation: TypeAlias = tuple[str, str, str]
NativeBatch: TypeAlias = Data | HeteroData
Owner: TypeAlias = str | CanonicalRelation
Scope: TypeAlias = Literal["node", "edge", "global"]
PrefetchMode: TypeAlias = Literal["host-only", "cuda"]
CudaSourceMode: TypeAlias = Literal["host", "resident"]
StorageKey: TypeAlias = tuple[str, int | None, int, int]


class PrefetchLimitError(ValueError):
    """Report one exact configured or observed admission-limit violation."""


class PrefetchError(RuntimeError):
    """Attach lifecycle phase and sampling identity to a prefetch failure."""

    def __init__(
        self,
        phase: str,
        sequence: int,
        descriptor: object,
        cause: BaseException,
    ) -> None:
        self.phase = phase
        self.sequence = sequence
        self.descriptor = descriptor
        self.root_cause = cause
        super().__init__(
            f"prefetch phase={phase} sequence={sequence} descriptor={descriptor!r}: {cause}"
        )


def _integer(value: object, name: str, *, minimum: int = 0) -> int:
    if isinstance(value, bool) or not isinstance(value, Integral):
        raise TypeError(f"{name} must be an integer")
    result = int(value)
    if result < minimum:
        raise ValueError(f"{name} must be at least {minimum}")
    return result


def _optional_integer(value: object, name: str) -> int | None:
    return None if value is None else _integer(value, name)


def _strict_bool(value: object, name: str) -> bool:
    if type(value) is not bool:
        raise TypeError(f"{name} must be bool")
    return value


def _node_caps(value: object) -> Mapping[str, int]:
    if isinstance(value, Mapping):
        items = value.items()
    elif isinstance(value, Sequence) and not isinstance(value, (str, bytes)):
        items = value
    else:
        raise TypeError(
            "max_nodes_per_type must be a mapping or pair sequence"
        )
    result: dict[str, int] = {}
    for item in items:
        try:
            key, cap = item
        except (TypeError, ValueError) as error:
            raise TypeError(
                "max_nodes_per_type must contain (node_type, cap) pairs"
            ) from error
        if not isinstance(key, str) or not key:
            raise TypeError(
                "max_nodes_per_type keys must be non-empty strings"
            )
        if key in result:
            raise ValueError(f"duplicate node type cap {key!r}")
        result[key] = _integer(cap, f"max_nodes_per_type[{key!r}]")
    return MappingProxyType(dict(sorted(result.items())))


def _canonical_relation(value: object) -> CanonicalRelation:
    if (
        isinstance(value, (str, bytes))
        or not isinstance(value, Sequence)
        or len(value) != 3
        or any(not isinstance(part, str) or not part for part in value)
    ):
        raise TypeError("relation cap keys must be canonical string triples")
    return value[0], value[1], value[2]


def _relation_caps(value: object) -> Mapping[CanonicalRelation, int]:
    if isinstance(value, Mapping):
        items = value.items()
    elif isinstance(value, Sequence) and not isinstance(value, (str, bytes)):
        items = value
    else:
        raise TypeError(
            "max_edges_per_relation must be a mapping or pair sequence"
        )
    result: dict[CanonicalRelation, int] = {}
    for item in items:
        try:
            raw_relation, cap = item
        except (TypeError, ValueError) as error:
            raise TypeError(
                "max_edges_per_relation must contain (relation, cap) pairs"
            ) from error
        relation = _canonical_relation(raw_relation)
        if relation in result:
            raise ValueError(f"duplicate relation cap {relation!r}")
        result[relation] = _integer(
            cap, f"max_edges_per_relation[{relation!r}]"
        )
    return MappingProxyType(dict(sorted(result.items())))


@dataclass(frozen=True, slots=True)
class PrefetchLimits:
    """Strict immutable count and memory admission limits for both queues."""

    host_queue_depth: int
    device_queue_depth: int
    max_batch_nodes: int
    max_batch_edges: int
    max_nodes_per_type: Mapping[str, int] = field(default_factory=dict)
    max_edges_per_relation: Mapping[CanonicalRelation, int] = field(
        default_factory=dict
    )
    max_host_batch_bytes: int = 0
    max_device_batch_bytes: int = 0
    max_host_queue_bytes: int = 0
    max_device_queue_bytes: int = 0
    worst_case_host_bytes: int | None = None
    worst_case_device_bytes: int | None = None

    def __post_init__(self) -> None:
        for name, minimum in (
            ("host_queue_depth", 1),
            ("device_queue_depth", 0),
            ("max_batch_nodes", 0),
            ("max_batch_edges", 0),
            ("max_host_batch_bytes", 0),
            ("max_device_batch_bytes", 0),
            ("max_host_queue_bytes", 0),
            ("max_device_queue_bytes", 0),
        ):
            object.__setattr__(
                self,
                name,
                _integer(getattr(self, name), name, minimum=minimum),
            )
        object.__setattr__(
            self, "max_nodes_per_type", _node_caps(self.max_nodes_per_type)
        )
        object.__setattr__(
            self,
            "max_edges_per_relation",
            _relation_caps(self.max_edges_per_relation),
        )
        host_worst = _optional_integer(
            self.worst_case_host_bytes, "worst_case_host_bytes"
        )
        device_worst = _optional_integer(
            self.worst_case_device_bytes, "worst_case_device_bytes"
        )
        object.__setattr__(self, "worst_case_host_bytes", host_worst)
        object.__setattr__(self, "worst_case_device_bytes", device_worst)
        if host_worst is not None:
            if host_worst > self.max_host_batch_bytes:
                raise ValueError(
                    "worst_case_host_bytes exceeds max_host_batch_bytes"
                )
            live_host_slots = self.host_queue_depth + self.device_queue_depth
            if host_worst * live_host_slots > self.max_host_queue_bytes:
                raise ValueError(
                    "worst_case_host_bytes across the host queue and live CUDA "
                    "staging exceeds the host queue budget"
                )
        if device_worst is not None:
            if device_worst > self.max_device_batch_bytes:
                raise ValueError(
                    "worst_case_device_bytes exceeds max_device_batch_bytes"
                )
            if (
                device_worst * self.device_queue_depth
                > self.max_device_queue_bytes
            ):
                raise ValueError(
                    "worst_case_device_bytes times device queue depth exceeds the device queue budget"
                )


def _device(value: object) -> torch.device:
    if isinstance(value, bool):
        raise TypeError("device must not be bool")
    if isinstance(value, Integral):
        return torch.device("cuda", _integer(value, "device"))
    if not isinstance(value, (str, torch.device)):
        raise TypeError("device must be a string, torch.device, or CUDA index")
    try:
        result = torch.device(value)
    except (RuntimeError, TypeError, ValueError) as error:
        raise ValueError(f"unknown device {value!r}") from error
    if result.type not in {"cpu", "mps", "cuda"}:
        raise ValueError(f"unknown device type {result.type!r}")
    if result.type in {"cpu", "mps"} and result.index is not None:
        raise ValueError(f"{result.type} device must not have an index")
    return result


@dataclass(frozen=True, slots=True)
class PrefetchCapability:
    """Qualified device and pinning capability without global-state mutation."""

    device: torch.device
    cuda_available: bool
    pin_memory_supported: bool

    def __post_init__(self) -> None:
        device = _device(self.device)
        cuda_available = _strict_bool(self.cuda_available, "cuda_available")
        pin_supported = _strict_bool(
            self.pin_memory_supported, "pin_memory_supported"
        )
        if device.type == "cuda":
            if not cuda_available or not torch.cuda.is_available():
                raise RuntimeError(
                    "CUDA required for device prefetch, but CUDA is unavailable"
                )
            count = torch.cuda.device_count()
            index = 0 if device.index is None else device.index
            if index < 0 or index >= count:
                raise ValueError(
                    f"CUDA device index {index} is outside [0, {count})"
                )
            if not pin_supported:
                raise RuntimeError("CUDA prefetch requires pinned host memory")
            device = torch.device("cuda", index)
        object.__setattr__(self, "device", device)

    @classmethod
    def detect(cls, device: object = "cpu") -> PrefetchCapability:
        """Resolve a real runtime capability for one requested output device."""
        resolved = _device(device)
        available = bool(torch.cuda.is_available())
        if resolved.type == "cuda":
            if not available:
                raise RuntimeError(
                    "CUDA required for device prefetch, but CUDA is unavailable"
                )
            index = 0 if resolved.index is None else resolved.index
            count = torch.cuda.device_count()
            if index >= count:
                raise ValueError(
                    f"CUDA device index {index} is outside [0, {count})"
                )
            resolved = torch.device("cuda", index)
        return cls(resolved, available, resolved.type == "cuda")


@dataclass(frozen=True, slots=True)
class PrefetchStatus:
    """Explicitly describe whether CUDA transfer is active."""

    mode: PrefetchMode
    device: torch.device
    detail: str


@dataclass(frozen=True, slots=True)
class TensorFieldEstimate:
    """Deterministic byte attribution for one dense or sparse component."""

    scope: Scope
    owner: Owner
    field_path: str
    component: str | None
    layout: str
    dtype: str
    device: str
    logical_bytes: int
    admitted_bytes: int


@dataclass(frozen=True, slots=True)
class BatchByteEstimate:
    """Exact unique-storage counts and typed graph count estimates."""

    total_bytes: int
    host_bytes: int
    device_bytes: int
    node_count: int
    edge_count: int
    node_counts: tuple[tuple[str, int], ...]
    edge_counts: tuple[tuple[CanonicalRelation | str, int], ...]
    node_bytes: tuple[tuple[str, int], ...]
    edge_bytes: tuple[tuple[CanonicalRelation | str, int], ...]
    global_bytes: int
    fields: tuple[TensorFieldEstimate, ...]


@dataclass(frozen=True, slots=True)
class _RawComponent:
    scope: Scope
    owner: Owner
    field_path: str
    component: str | None
    layout: str
    tensor: Tensor


def _storage_key(tensor: Tensor) -> StorageKey:
    storage = tensor.untyped_storage()
    return (
        tensor.device.type,
        tensor.device.index,
        int(storage._cdata),
        int(storage.nbytes()),
    )


def _sparse_components(tensor: Tensor) -> tuple[tuple[str, Tensor], ...]:
    if tensor.layout == torch.sparse_coo:
        return (("indices", tensor._indices()), ("values", tensor._values()))
    if tensor.layout in {torch.sparse_csr, torch.sparse_bsr}:
        return (
            ("crow_indices", tensor.crow_indices()),
            ("col_indices", tensor.col_indices()),
            ("values", tensor.values()),
        )
    if tensor.layout in {torch.sparse_csc, torch.sparse_bsc}:
        return (
            ("ccol_indices", tensor.ccol_indices()),
            ("row_indices", tensor.row_indices()),
            ("values", tensor.values()),
        )
    if tensor.layout == torch.strided:
        return (("", tensor),)
    raise TypeError(f"unsupported tensor layout {tensor.layout}")


def _nested_tensors(value: object, path: str) -> Iterator[tuple[str, Tensor]]:
    if isinstance(value, Tensor):
        yield path, value
    elif isinstance(value, Mapping):
        for key in sorted(value, key=lambda item: repr(item)):
            yield from _nested_tensors(value[key], f"{path}.{key}")
    elif isinstance(value, Sequence) and not isinstance(value, (str, bytes)):
        for index, item in enumerate(value):
            yield from _nested_tensors(item, f"{path}[{index}]")


def _data_scope(batch: Data, key: str) -> Scope:
    try:
        if batch.is_node_attr(key):
            return "node"
        if batch.is_edge_attr(key):
            return "edge"
    except (AttributeError, IndexError, TypeError, ValueError):
        pass
    return "global"


def _raw_components(batch: NativeBatch) -> tuple[_RawComponent, ...]:
    values: list[_RawComponent] = []

    def append(scope: Scope, owner: Owner, key: str, value: object) -> None:
        base_path = f"{scope}.{owner!r}.{key}"
        for nested_path, tensor in _nested_tensors(value, base_path):
            layout = str(tensor.layout)
            for component, dense in _sparse_components(tensor):
                field_path = (
                    nested_path
                    if not component
                    else f"{nested_path}.{component}"
                )
                values.append(
                    _RawComponent(
                        scope,
                        owner,
                        field_path,
                        component or None,
                        layout,
                        dense,
                    )
                )

    if isinstance(batch, HeteroData):
        for key, value in sorted(batch._global_store.items()):
            append("global", "__global__", key, value)
        for store in sorted(batch.node_stores, key=lambda item: item._key):
            owner = store._key
            assert isinstance(owner, str)
            for key, value in sorted(store.items()):
                append("node", owner, key, value)
        for store in sorted(batch.edge_stores, key=lambda item: item._key):
            owner = store._key
            assert isinstance(owner, tuple) and len(owner) == 3
            relation = owner[0], owner[1], owner[2]
            for key, value in sorted(store.items()):
                append("edge", relation, key, value)
    elif isinstance(batch, Data):
        for key, value in sorted(batch._store.items()):
            scope = _data_scope(batch, key)
            owner = "__homogeneous__" if scope != "global" else "__global__"
            append(scope, owner, key, value)
    else:
        raise TypeError(
            f"prefetch source must yield native Data or HeteroData, got {type(batch).__name__}"
        )
    return tuple(sorted(values, key=lambda item: item.field_path))


def _batch_counts(
    batch: NativeBatch,
) -> tuple[
    tuple[tuple[str, int], ...],
    tuple[tuple[CanonicalRelation | str, int], ...],
]:
    if isinstance(batch, HeteroData):
        nodes: list[tuple[str, int]] = []
        for store in sorted(batch.node_stores, key=lambda item: item._key):
            count = store.num_nodes
            if count is None:
                raise ValueError(f"node type {store._key!r} has no node count")
            nodes.append(
                (store._key, _integer(count, f"nodes[{store._key!r}]"))
            )
        edges: list[tuple[CanonicalRelation, int]] = []
        for store in sorted(batch.edge_stores, key=lambda item: item._key):
            relation = _canonical_relation(store._key)
            count = store.num_edges
            if count is None:
                raise ValueError(f"relation {relation!r} has no edge count")
            edges.append((relation, _integer(count, f"edges[{relation!r}]")))
        return tuple(nodes), tuple(edges)
    if not isinstance(batch, Data):
        raise TypeError(
            f"prefetch source must yield native Data or HeteroData, got {type(batch).__name__}"
        )
    node_count = batch.num_nodes
    edge_count = batch.num_edges
    if node_count is None:
        raise ValueError("homogeneous batch has no node count")
    return (
        (("__homogeneous__", _integer(node_count, "num_nodes")),),
        (("__homogeneous__", _integer(edge_count, "num_edges")),),
    )


def estimate_batch_bytes(batch: NativeBatch) -> BatchByteEstimate:
    """Count all native graph tensor storages once without densifying sparse data."""
    node_counts, edge_counts = _batch_counts(batch)
    node_bytes: dict[str, int] = {name: 0 for name, _ in node_counts}
    edge_bytes: dict[CanonicalRelation | str, int] = {
        name: 0 for name, _ in edge_counts
    }
    global_bytes = 0
    seen: set[StorageKey] = set()
    fields: list[TensorFieldEstimate] = []
    host_bytes = 0
    device_bytes = 0
    for raw in _raw_components(batch):
        storage_bytes = int(raw.tensor.untyped_storage().nbytes())
        key = _storage_key(raw.tensor)
        admitted = 0 if key in seen else storage_bytes
        seen.add(key)
        if raw.tensor.device.type == "cpu":
            host_bytes += admitted
        else:
            device_bytes += admitted
        if raw.scope == "node":
            assert isinstance(raw.owner, str)
            node_bytes[raw.owner] = node_bytes.get(raw.owner, 0) + admitted
        elif raw.scope == "edge":
            edge_bytes[raw.owner] = edge_bytes.get(raw.owner, 0) + admitted
        else:
            global_bytes += admitted
        fields.append(
            TensorFieldEstimate(
                raw.scope,
                raw.owner,
                raw.field_path,
                raw.component,
                raw.layout,
                str(raw.tensor.dtype),
                str(raw.tensor.device),
                raw.tensor.numel() * raw.tensor.element_size(),
                admitted,
            )
        )
    return BatchByteEstimate(
        host_bytes + device_bytes,
        host_bytes,
        device_bytes,
        sum(value for _, value in node_counts),
        sum(value for _, value in edge_counts),
        node_counts,
        edge_counts,
        tuple(sorted(node_bytes.items())),
        tuple(sorted(edge_bytes.items())),
        global_bytes,
        tuple(fields),
    )


def _descriptor(batch: object) -> object:
    return getattr(batch, "sampling_descriptor", "unavailable")


def _sequence(batch: object, fallback: int) -> int:
    value = getattr(batch, "sequence_id", fallback)
    if (
        isinstance(value, bool)
        or not isinstance(value, Integral)
        or int(value) < 1
    ):
        return fallback
    return int(value)


def _all_component_tensors(batch: NativeBatch) -> tuple[Tensor, ...]:
    return tuple(component.tensor for component in _raw_components(batch))


def _cuda_source_mode(
    batch: NativeBatch,
    target: torch.device,
) -> CudaSourceMode:
    tensors = _all_component_tensors(batch)
    if not tensors or all(tensor.device.type == "cpu" for tensor in tensors):
        return "host"
    if all(tensor.device == target for tensor in tensors):
        return "resident"
    devices = sorted({str(tensor.device) for tensor in tensors})
    raise PrefetchLimitError(
        "CUDA prefetch requires a single-device batch: all tensors must be "
        f"CPU or already resident on target {target}; received tensor devices "
        f"{devices!r}"
    )


def _admit(
    batch: NativeBatch,
    estimate: BatchByteEstimate,
    limits: PrefetchLimits,
    capability: PrefetchCapability,
) -> CudaSourceMode:
    if estimate.node_count > limits.max_batch_nodes:
        raise PrefetchLimitError(
            f"batch nodes {estimate.node_count} exceed max_batch_nodes {limits.max_batch_nodes}"
        )
    if estimate.edge_count > limits.max_batch_edges:
        raise PrefetchLimitError(
            f"batch edges {estimate.edge_count} exceed max_batch_edges {limits.max_batch_edges}"
        )
    for node_type, count in estimate.node_counts:
        cap = limits.max_nodes_per_type.get(node_type)
        if cap is not None and count > cap:
            raise PrefetchLimitError(
                f"node type {node_type!r} count {count} exceeds declared cap {cap}"
            )
    for relation, count in estimate.edge_counts:
        if isinstance(relation, str):
            continue
        cap = limits.max_edges_per_relation.get(relation)
        if cap is not None and count > cap:
            raise PrefetchLimitError(
                f"relation {relation!r} edge count {count} exceeds declared cap {cap}"
            )
    if estimate.host_bytes > limits.max_host_batch_bytes:
        raise PrefetchLimitError(
            f"batch host bytes {estimate.host_bytes} exceed max_host_batch_bytes {limits.max_host_batch_bytes}"
        )
    if estimate.host_bytes > limits.max_host_queue_bytes:
        raise PrefetchLimitError(
            f"batch host bytes {estimate.host_bytes} exceed host queue budget {limits.max_host_queue_bytes}"
        )
    tensors = _all_component_tensors(batch)
    if capability.device.type != "cuda":
        non_cpu = sorted(
            {
                str(tensor.device)
                for tensor in tensors
                if tensor.device.type != "cpu"
            }
        )
        if non_cpu:
            raise PrefetchLimitError(
                "host-only prefetch requires CPU batches for Lightning-owned transfer, "
                f"received {non_cpu!r}"
            )
        return "host"
    source_mode = _cuda_source_mode(batch, capability.device)
    if estimate.total_bytes > limits.max_device_batch_bytes:
        raise PrefetchLimitError(
            f"batch device bytes {estimate.total_bytes} exceed max_device_batch_bytes {limits.max_device_batch_bytes}"
        )
    if estimate.total_bytes > limits.max_device_queue_bytes:
        raise PrefetchLimitError(
            f"batch device bytes {estimate.total_bytes} exceed device queue budget {limits.max_device_queue_bytes}"
        )
    return source_mode


def _map_nested(
    value: object,
    map_tensor: Callable[[Tensor], Tensor],
    object_cache: dict[int, Tensor],
) -> object:
    if isinstance(value, Tensor):
        identity = id(value)
        if identity not in object_cache:
            object_cache[identity] = map_tensor(value)
        return object_cache[identity]
    if isinstance(value, tuple) and hasattr(value, "_fields"):
        return type(value)(
            *(_map_nested(item, map_tensor, object_cache) for item in value)
        )
    if isinstance(value, tuple):
        return tuple(
            _map_nested(item, map_tensor, object_cache) for item in value
        )
    if isinstance(value, list):
        return [_map_nested(item, map_tensor, object_cache) for item in value]
    if isinstance(value, Mapping):
        return {
            key: _map_nested(item, map_tensor, object_cache)
            for key, item in value.items()
        }
    return value


class _TensorTransformer:
    def __init__(self, dense: Callable[[Tensor], Tensor]) -> None:
        self._dense = dense
        self._storage_cache: dict[StorageKey, Tensor] = {}

    def _flat(self, tensor: Tensor) -> Tensor:
        storage = tensor.untyped_storage()
        storage_bytes = int(storage.nbytes())
        element_size = tensor.element_size()
        if storage_bytes % element_size:
            raise TypeError(
                "shared storage size is incompatible with tensor dtype"
            )
        flat = torch.empty(0, dtype=tensor.dtype, device=tensor.device)
        return flat.set_(storage, 0, (storage_bytes // element_size,), (1,))

    def _strided(self, tensor: Tensor) -> Tensor:
        key = _storage_key(tensor)
        transformed = self._storage_cache.get(key)
        if transformed is None:
            transformed = self._dense(self._flat(tensor))
            if transformed.dtype != tensor.dtype:
                raise TypeError("tensor transformation changed dtype")
            if transformed.numel() * transformed.element_size() != key[-1]:
                raise ValueError("tensor transformation changed storage bytes")
            self._storage_cache[key] = transformed
        if transformed.dtype == tensor.dtype:
            base = transformed
        else:
            storage_bytes = int(transformed.untyped_storage().nbytes())
            if storage_bytes % tensor.element_size():
                raise TypeError(
                    "shared storage size is incompatible with tensor dtype"
                )
            base = torch.empty(
                0,
                dtype=tensor.dtype,
                device=transformed.device,
            ).set_(
                transformed.untyped_storage(),
                0,
                (storage_bytes // tensor.element_size(),),
                (1,),
            )
        return base.as_strided(
            tuple(tensor.size()),
            tuple(tensor.stride()),
            tensor.storage_offset(),
        )

    def __call__(self, tensor: Tensor) -> Tensor:
        if tensor.layout == torch.strided:
            return self._strided(tensor)
        components = {
            name: self._strided(component)
            for name, component in _sparse_components(tensor)
        }
        common = {
            "size": tuple(tensor.size()),
            "dtype": tensor.dtype,
            "device": components["values"].device,
            "requires_grad": tensor.requires_grad,
        }
        if tensor.layout == torch.sparse_coo:
            return torch.sparse_coo_tensor(
                components["indices"],
                components["values"],
                is_coalesced=tensor.is_coalesced(),
                **common,
            )
        if tensor.layout == torch.sparse_csr:
            return torch.sparse_csr_tensor(
                components["crow_indices"],
                components["col_indices"],
                components["values"],
                **common,
            )
        if tensor.layout == torch.sparse_csc:
            return torch.sparse_csc_tensor(
                components["ccol_indices"],
                components["row_indices"],
                components["values"],
                **common,
            )
        if tensor.layout == torch.sparse_bsr:
            return torch.sparse_bsr_tensor(
                components["crow_indices"],
                components["col_indices"],
                components["values"],
                **common,
            )
        if tensor.layout == torch.sparse_bsc:
            return torch.sparse_bsc_tensor(
                components["ccol_indices"],
                components["row_indices"],
                components["values"],
                **common,
            )
        raise TypeError(f"unsupported tensor layout {tensor.layout}")


def _map_batch(
    batch: NativeBatch, transform: _TensorTransformer, *, shallow_copy: bool
) -> NativeBatch:
    output = copy.copy(batch) if shallow_copy else batch
    cache: dict[int, Tensor] = {}
    for store in output.stores:
        for key, value in tuple(store.items()):
            store[key] = _map_nested(value, transform, cache)
    return output


def _prepare_host(
    batch: NativeBatch,
    capability: PrefetchCapability,
    pin_memory: Callable[[Tensor], Tensor],
) -> NativeBatch:
    def prepare(flat: Tensor) -> Tensor:
        if flat.device.type != "cpu":
            return flat
        storage = flat.untyped_storage()
        if capability.pin_memory_supported and not flat.is_pinned():
            output = pin_memory(flat)
            if output.device.type != "cpu":
                raise ValueError("pin_memory must return a CPU tensor")
            return output
        if not storage.resizable():
            return flat.clone()
        return flat

    return _map_batch(batch, _TensorTransformer(prepare), shallow_copy=True)


def _move_cuda(batch: NativeBatch, target: torch.device) -> NativeBatch:
    if all(
        tensor.device == target for tensor in _all_component_tensors(batch)
    ):
        return batch

    def move(flat: Tensor) -> Tensor:
        if flat.device == target:
            return flat
        if flat.device.type != "cpu":
            raise ValueError(
                f"cannot transfer {flat.device} tensor to {target}"
            )
        if not flat.is_pinned():
            raise RuntimeError("CUDA H2D source tensor is not pinned")
        return flat.to(device=target, non_blocking=True)

    return _map_batch(batch, _TensorTransformer(move), shallow_copy=True)


def _record_stream(batch: NativeBatch, stream: torch.cuda.Stream) -> None:
    seen: set[StorageKey] = set()
    for tensor in _all_component_tensors(batch):
        key = _storage_key(tensor)
        if key in seen:
            continue
        seen.add(key)
        tensor.record_stream(stream)


@dataclass(slots=True)
class _HostItem:
    batch: NativeBatch
    estimate: BatchByteEstimate
    sequence: int
    descriptor: object


@dataclass(slots=True)
class _DeviceSlot:
    event: torch.cuda.Event
    host_batch: NativeBatch | None = None
    device_batch: NativeBatch | None = None
    sequence: int = 0
    descriptor: object = "unavailable"
    admitted_bytes: int = 0
    host_bytes: int = 0


class DevicePrefetchIterator(Iterator[NativeBatch]):
    """One owned producer and one ordered consumer-side CUDA ring."""

    def __init__(self, owner: DevicePrefetchLoader) -> None:
        self._owner = owner
        self._source = owner.source
        self._queue: queue.Queue[_HostItem] = queue.Queue(
            maxsize=owner.limits.host_queue_depth
        )
        self._queue_slots = threading.BoundedSemaphore(
            owner.limits.host_queue_depth
        )
        self._stop = threading.Event()
        self._done = threading.Event()
        self._condition = threading.Condition()
        self._host_queued_bytes = 0
        self.max_host_queued_bytes = 0
        self.max_host_queue_size = 0
        self.max_device_queued_bytes = 0
        self._device_queued_bytes = 0
        self._producer_error: tuple[BaseException, BaseException] | None = None
        self._deferred_error: tuple[PrefetchError, BaseException] | None = None
        self._source_iterator: Iterator[NativeBatch] | None = None
        self._closed = False
        self._exhausted = False
        self._abort_notified = False
        self._pending_host: _HostItem | None = None
        self._ring: deque[int] = deque()
        self._free_slots: deque[int] = deque()
        self._retired_slots: deque[int] = deque()
        self._slots: list[_DeviceSlot] = []
        self.transfer_stream: torch.cuda.Stream | None = None
        self.completion_events: tuple[torch.cuda.Event, ...] = ()
        self._producer_thread = threading.Thread(
            target=self._produce,
            name=f"device-prefetch-{id(self):x}",
            daemon=False,
        )
        self._producer_thread.start()

    @property
    def producer_alive(self) -> bool:
        return self._producer_thread.is_alive()

    @property
    def closed(self) -> bool:
        return self._closed

    def _monitor_queue(self) -> QueueSnapshot:
        with self._condition:
            host_bytes = self._host_queued_bytes
        return QueueSnapshot(
            host_configured_depth=self._owner.limits.host_queue_depth,
            host_depth=self._queue.qsize(),
            host_configured_bytes=self._owner.limits.max_host_queue_bytes,
            host_bytes=host_bytes,
            device_configured_depth=self._owner.limits.device_queue_depth,
            device_depth=len(self._ring),
            device_configured_bytes=self._owner.limits.max_device_queue_bytes,
            device_bytes=self._device_queued_bytes,
        )

    def _monitor_begin(
        self,
        operation: ExecutionOperation,
        item: _HostItem,
        *,
        cuda_timing: bool = False,
    ) -> OperationToken | None:
        monitor = self._owner.execution_monitor
        if monitor is None:
            return None
        prehashed = getattr(item.batch, "execution_descriptor_digest", None)
        return monitor.begin(
            operation,
            phase=self._owner.monitor_phase,
            split=self._owner.monitor_split,
            descriptor_sequence=item.sequence,
            descriptor_identity=(
                item.descriptor if prehashed is None else None
            ),
            descriptor_digest_value=prehashed,
            queue=self._monitor_queue(),
            cuda_timing=cuda_timing,
            cuda_stream=self.transfer_stream,
        )

    def _monitor_finish(
        self,
        token: OperationToken | None,
        item: _HostItem,
        *,
        status: ExecutionStatus = ExecutionStatus.SUCCESS,
        evidence: Mapping[str, object] | None = None,
    ) -> None:
        monitor = self._owner.execution_monitor
        if token is None or monitor is None:
            return
        monitor.finish(
            token,
            status=status,
            node_count=item.estimate.node_count,
            edge_count=item.estimate.edge_count,
            unique_storage_bytes=item.estimate.total_bytes,
            queue=self._monitor_queue(),
            cuda_stream=self.transfer_stream,
            evidence=evidence,
        )

    def _monitor_queue_admission(self, item: _HostItem) -> None:
        monitor = self._owner.execution_monitor
        if monitor is None:
            return
        prehashed = getattr(item.batch, "execution_descriptor_digest", None)
        queue_state = self._monitor_queue()
        queue_state = replace(
            queue_state,
            host_depth=(queue_state.host_depth or 0) + 1,
        )
        monitor.record(
            ExecutionOperation.H2D_QUEUE,
            phase=self._owner.monitor_phase,
            split=self._owner.monitor_split,
            descriptor_sequence=item.sequence,
            descriptor_identity=(
                item.descriptor if prehashed is None else None
            ),
            descriptor_digest_value=prehashed,
            node_count=item.estimate.node_count,
            edge_count=item.estimate.edge_count,
            unique_storage_bytes=item.estimate.total_bytes,
            queue=queue_state,
            evidence={"prefetch_mode": self._owner.status.mode},
        )

    def __iter__(self) -> DevicePrefetchIterator:
        return self

    def _set_error(
        self,
        phase: str,
        sequence: int,
        descriptor: object,
        cause: BaseException,
    ) -> None:
        with self._condition:
            if self._producer_error is None:
                error: BaseException = (
                    cause
                    if isinstance(cause, MonitorOverflowError)
                    else PrefetchError(phase, sequence, descriptor, cause)
                )
                self._producer_error = (error, cause)
            self._condition.notify_all()

    def _reserve_host(self, byte_count: int) -> bool:
        with self._condition:
            while (
                not self._stop.is_set()
                and self._host_queued_bytes + byte_count
                > self._owner.limits.max_host_queue_bytes
            ):
                self._condition.wait(0.05)
            if self._stop.is_set():
                return False
            self._host_queued_bytes += byte_count
            self.max_host_queued_bytes = max(
                self.max_host_queued_bytes, self._host_queued_bytes
            )
            return True

    def _release_host(self, byte_count: int) -> None:
        with self._condition:
            self._host_queued_bytes -= byte_count
            self._condition.notify_all()

    def _acquire_host(self, byte_count: int) -> bool:
        while not self._stop.is_set():
            if self._queue_slots.acquire(timeout=0.05):
                if self._reserve_host(byte_count):
                    return True
                self._queue_slots.release()
                return False
        return False

    def _release_admission(self, byte_count: int) -> None:
        self._release_host(byte_count)
        self._queue_slots.release()

    def _put_host(self, item: _HostItem) -> None:
        self._queue.put_nowait(item)
        self.max_host_queue_size = max(
            self.max_host_queue_size,
            self._queue.qsize(),
        )

    def _produce(self) -> None:
        fallback_sequence = 1
        last_descriptor: object = "unavailable"
        try:
            try:
                source_iterator = iter(self._source)
                self._source_iterator = source_iterator
            except BaseException as cause:
                self._set_error(
                    "host_producer", fallback_sequence, last_descriptor, cause
                )
                return
            while not self._stop.is_set():
                try:
                    batch = next(source_iterator)
                except StopIteration:
                    break
                except BaseException as cause:
                    self._set_error(
                        "host_producer",
                        fallback_sequence,
                        last_descriptor,
                        cause,
                    )
                    break
                sequence = _sequence(batch, fallback_sequence)
                descriptor = _descriptor(batch)
                fallback_sequence = sequence + 1
                last_descriptor = descriptor
                try:
                    if not isinstance(batch, (Data, HeteroData)):
                        raise TypeError(
                            "prefetch source must yield native Data or HeteroData, "
                            f"got {type(batch).__name__}"
                        )
                    estimate = estimate_batch_bytes(batch)
                    source_mode = _admit(
                        batch,
                        estimate,
                        self._owner.limits,
                        self._owner.capability,
                    )
                except BaseException as cause:
                    self._set_error(
                        "host_admission", sequence, descriptor, cause
                    )
                    break
                if not self._acquire_host(estimate.host_bytes):
                    break
                monitor_item: _HostItem | None = None
                pin_token: OperationToken | None = None
                try:
                    if self._owner.execution_monitor is not None:
                        monitor_item = _HostItem(
                            batch,
                            estimate,
                            sequence,
                            descriptor,
                        )
                        pin_token = self._monitor_begin(
                            ExecutionOperation.HOST_PIN,
                            monitor_item,
                        )
                    prepared = (
                        batch
                        if source_mode == "resident"
                        else _prepare_host(
                            batch,
                            self._owner.capability,
                            self._owner.pin_memory,
                        )
                    )
                    if (
                        source_mode != "resident"
                        and pin_token is not None
                        and pin_token.descriptor_digest is not None
                    ):
                        prepared.execution_descriptor_digest = (
                            pin_token.descriptor_digest
                        )
                    prepared_estimate = estimate_batch_bytes(prepared)
                    if prepared_estimate.host_bytes != estimate.host_bytes:
                        raise RuntimeError(
                            "host preparation changed admitted storage bytes"
                        )
                    if self._stop.is_set():
                        self._release_admission(estimate.host_bytes)
                        break
                    queued = _HostItem(
                        prepared,
                        prepared_estimate,
                        sequence,
                        descriptor,
                    )
                    self._monitor_finish(
                        pin_token,
                        queued,
                        status=(
                            ExecutionStatus.SKIPPED
                            if source_mode == "resident"
                            else ExecutionStatus.SUCCESS
                        ),
                        evidence={"source_mode": source_mode},
                    )
                    pin_token = None
                    self._monitor_queue_admission(queued)
                    self._put_host(queued)
                except BaseException as cause:
                    if pin_token is not None and monitor_item is not None:
                        try:
                            self._monitor_finish(
                                pin_token,
                                monitor_item,
                                status=ExecutionStatus.ERROR,
                                evidence={"failure_stage": "host_pin"},
                            )
                        except Exception as monitor_error:
                            if not isinstance(cause, MonitorOverflowError):
                                cause.add_note(
                                    "secondary input monitor failure: "
                                    f"{type(monitor_error).__name__}"
                                )
                    self._release_admission(estimate.host_bytes)
                    self._set_error("host_pin", sequence, descriptor, cause)
                    break
        finally:
            self._done.set()
            with self._condition:
                self._condition.notify_all()

    def _raise_producer_error(self) -> None:
        assert self._producer_error is not None
        error, cause = self._producer_error
        if error is cause:
            raise error
        raise error from cause

    def _take_host(
        self,
        *,
        retain_host_bytes: bool = False,
        wait: bool = True,
    ) -> _HostItem | None:
        while True:
            try:
                item = (
                    self._queue.get(timeout=0.05)
                    if wait
                    else self._queue.get_nowait()
                )
            except queue.Empty:
                if self._done.is_set():
                    if self._producer_error is not None:
                        self._raise_producer_error()
                    return None
                if self._stop.is_set() or not wait:
                    return None
                continue
            self._queue_slots.release()
            if not retain_host_bytes:
                self._release_host(item.estimate.host_bytes)
            return item

    def _ensure_cuda_resources(self) -> None:
        if self.transfer_stream is not None:
            return
        target = self._owner.capability.device
        self.transfer_stream = torch.cuda.Stream(device=target)
        self.completion_events = tuple(
            torch.cuda.Event()
            for _ in range(self._owner.limits.device_queue_depth)
        )
        self._slots = [
            _DeviceSlot(event=event) for event in self.completion_events
        ]
        self._free_slots.extend(range(len(self._slots)))

    def _schedule(self, item: _HostItem, slot_index: int) -> None:
        assert self.transfer_stream is not None
        transfer_token = self._monitor_begin(
            ExecutionOperation.H2D_COPY,
            item,
            cuda_timing=True,
        )
        slot = self._slots[slot_index]
        try:
            with torch.cuda.stream(self.transfer_stream):
                device_batch = _move_cuda(
                    item.batch,
                    self._owner.capability.device,
                )
                _record_stream(item.batch, self.transfer_stream)
                slot.event.record(self.transfer_stream)
        except BaseException as cause:
            if transfer_token is not None:
                try:
                    self._monitor_finish(
                        transfer_token,
                        item,
                        status=ExecutionStatus.ERROR,
                        evidence={"failure_stage": "device_transfer"},
                    )
                except Exception as monitor_error:
                    cause.add_note(
                        "secondary input monitor failure: "
                        f"{type(monitor_error).__name__}"
                    )
            raise
        slot.host_batch = item.batch
        slot.device_batch = device_batch
        slot.sequence = item.sequence
        slot.descriptor = item.descriptor
        slot.admitted_bytes = item.estimate.total_bytes
        slot.host_bytes = item.estimate.host_bytes
        self._device_queued_bytes += slot.admitted_bytes
        self.max_device_queued_bytes = max(
            self.max_device_queued_bytes,
            self._device_queued_bytes,
        )
        self._ring.append(slot_index)
        try:
            self._monitor_finish(
                transfer_token,
                item,
                evidence={"prefetch_mode": self._owner.status.mode},
            )
        except MonitorOverflowError:
            raise
        except Exception:
            pass

    def _reap_completed_slots(self) -> None:
        retained: deque[int] = deque()
        while self._retired_slots:
            slot_index = self._retired_slots.popleft()
            slot = self._slots[slot_index]
            if not slot.event.query():
                retained.append(slot_index)
                continue
            self._release_host(slot.host_bytes)
            slot.host_batch = None
            slot.host_bytes = 0
            self._free_slots.append(slot_index)
        self._retired_slots = retained

    def _fill_ring(self, *, block_for_first: bool) -> None:
        self._ensure_cuda_resources()
        self._reap_completed_slots()
        if (
            block_for_first
            and not self._ring
            and not self._free_slots
            and self._retired_slots
        ):
            self._slots[self._retired_slots[0]].event.synchronize()
            self._reap_completed_slots()
        if self._deferred_error is not None:
            return
        depth = self._owner.limits.device_queue_depth
        while len(self._ring) < depth and self._free_slots:
            item = self._pending_host
            if item is None:
                try:
                    item = self._take_host(
                        retain_host_bytes=True,
                        wait=block_for_first and not self._ring,
                    )
                except PrefetchError as error:
                    if not self._ring:
                        raise
                    self._deferred_error = (error, error.root_cause)
                    return
                if item is None:
                    return
            if (
                self._device_queued_bytes + item.estimate.total_bytes
                > self._owner.limits.max_device_queue_bytes
            ):
                if not self._ring:
                    cause = PrefetchLimitError(
                        f"batch device bytes {item.estimate.total_bytes} exceed remaining device queue budget"
                    )
                    error = PrefetchError(
                        "device_admission",
                        item.sequence,
                        item.descriptor,
                        cause,
                    )
                    self._pending_host = item
                    self.close()
                    raise error from cause
                self._pending_host = item
                return
            self._pending_host = None
            slot_index = self._free_slots.popleft()
            try:
                self._schedule(item, slot_index)
            except BaseException as cause:
                self._release_host(item.estimate.host_bytes)
                error = PrefetchError(
                    "device_transfer", item.sequence, item.descriptor, cause
                )
                self._free_slots.appendleft(slot_index)
                self.close()
                raise error from cause

    def _mark_exhausted_if_drained(self) -> None:
        if (
            self._done.is_set()
            and self._producer_error is None
            and self._queue.empty()
            and self._pending_host is None
            and not self._ring
        ):
            self._exhausted = True

    def _deliver_callback(self, batch: NativeBatch) -> None:
        try:
            monitor = self._owner.execution_monitor
            if monitor is not None:
                prehashed = getattr(
                    batch,
                    "execution_descriptor_digest",
                    None,
                )
                monitor.mark_batch_ready(
                    phase=self._owner.monitor_phase,
                    split=self._owner.monitor_split,
                    descriptor_sequence=_sequence(batch, 1),
                    descriptor_identity=(
                        _descriptor(batch) if prehashed is None else None
                    ),
                    descriptor_digest_value=prehashed,
                    queue=self._monitor_queue(),
                )
            if self._owner.on_yield is not None:
                self._owner.on_yield(batch)
        except BaseException as cause:
            self.close()
            if isinstance(cause, MonitorOverflowError):
                raise
            error = PrefetchError(
                "delivery",
                _sequence(batch, 1),
                _descriptor(batch),
                cause,
            )
            raise error from cause

    def _next_host_only(self) -> NativeBatch:
        try:
            item = self._take_host()
        except (PrefetchError, MonitorOverflowError):
            self.close()
            raise
        if item is None:
            self._exhausted = True
            self.close()
            raise StopIteration
        self._deliver_callback(item.batch)
        self._mark_exhausted_if_drained()
        return item.batch

    def _next_cuda(self) -> NativeBatch:
        try:
            self._fill_ring(block_for_first=True)
        except (PrefetchError, MonitorOverflowError):
            self.close()
            raise
        if not self._ring:
            self._exhausted = True
            self.close()
            raise StopIteration
        slot_index = self._ring.popleft()
        slot = self._slots[slot_index]
        assert slot.device_batch is not None
        current = torch.cuda.current_stream(self._owner.capability.device)
        current.wait_event(slot.event)
        _record_stream(slot.device_batch, current)
        batch = slot.device_batch
        self._device_queued_bytes -= slot.admitted_bytes
        slot.device_batch = None
        slot.admitted_bytes = 0
        self._retired_slots.append(slot_index)
        try:
            self._fill_ring(block_for_first=False)
        except PrefetchError as error:
            self._deferred_error = (error, error.root_cause)
        except MonitorOverflowError:
            self.close()
            raise
        self._mark_exhausted_if_drained()
        self._deliver_callback(batch)
        return batch

    def __next__(self) -> NativeBatch:
        if self._deferred_error is not None and not self._ring:
            error, cause = self._deferred_error
            self._deferred_error = None
            self.close()
            raise error from cause
        if self._closed:
            raise StopIteration
        return (
            self._next_host_only()
            if self._owner.status.mode == "host-only"
            else self._next_cuda()
        )

    def _shutdown_source(self) -> None:
        if not self._owner.owns_source:
            return
        for iterator in (
            self._source_iterator,
            getattr(self._source, "_iterator", None),
        ):
            shutdown = getattr(iterator, "_shutdown_workers", None)
            if callable(shutdown):
                with contextlib.suppress(RuntimeError, TypeError, ValueError):
                    shutdown()
        if getattr(self._source, "_iterator", None) is not None:
            with contextlib.suppress(AttributeError, TypeError):
                self._source._iterator = None  # type: ignore[attr-defined]

    def _drain_host(self) -> None:
        while True:
            try:
                item = self._queue.get_nowait()
            except queue.Empty:
                return
            self._queue_slots.release()
            self._release_host(item.estimate.host_bytes)

    def close(self) -> None:
        if self._closed:
            return
        self._closed = True
        self._stop.set()
        self._shutdown_source()
        self._drain_host()
        with self._condition:
            self._condition.notify_all()
        if (
            self._producer_thread.is_alive()
            and threading.current_thread() is not self._producer_thread
        ):
            self._producer_thread.join()
        self._drain_host()
        if self._owner.owns_source:
            close = getattr(self._source_iterator, "close", None)
            if callable(close):
                with contextlib.suppress(RuntimeError, ValueError):
                    close()
        if self.transfer_stream is not None:
            self.transfer_stream.synchronize()
        if self._pending_host is not None:
            self._release_host(self._pending_host.estimate.host_bytes)
            self._pending_host = None
        for slot in self._slots:
            if slot.host_batch is not None:
                self._release_host(slot.host_bytes)
                slot.host_batch = None
                slot.host_bytes = 0
        self._ring.clear()
        self._free_slots.clear()
        self._retired_slots.clear()
        self._slots.clear()
        self._device_queued_bytes = 0
        try:
            if (
                not self._exhausted
                and not self._abort_notified
                and self._owner.on_abort is not None
            ):
                self._abort_notified = True
                self._owner.on_abort()
        finally:
            self._owner._iterator_finished(self)
        return

    def __enter__(self) -> DevicePrefetchIterator:
        return self

    def __exit__(self, *exc_info: object) -> None:
        self.close()

    def __del__(self) -> None:
        with contextlib.suppress(Exception):
            self.close()


class DevicePrefetchLoader(Iterable[NativeBatch]):
    """Wrap any finite native graph iterable with bounded ordered prefetch."""

    def __init__(
        self,
        source: Iterable[NativeBatch],
        limits: PrefetchLimits,
        *,
        device: object | None = None,
        capability: PrefetchCapability | None = None,
        pin_memory: Callable[[Tensor], Tensor] | None = None,
        owns_source: bool = False,
        on_yield: Callable[[NativeBatch], None] | None = None,
        on_abort: Callable[[], None] | None = None,
        execution_monitor: InputMonitor | None = None,
        monitor_phase: str = "fit",
        monitor_split: str | None = None,
    ) -> None:
        if not isinstance(source, Iterable):
            raise TypeError("source must be a finite iterable")
        if not isinstance(limits, PrefetchLimits):
            raise TypeError("limits must be PrefetchLimits")
        # Multiprocess DataLoader primes ``prefetch_factor * num_workers``
        # tasks; each worker fetches a complete batch and puts it on the
        # shared result queue before this loader can estimate or reserve bytes.
        if isinstance(source, DataLoader) and source.num_workers != 0:
            raise ValueError(
                "DevicePrefetchLoader requires num_workers=0 for PyTorch "
                "DataLoader sources because the worker result queue receives "
                "materialized batches before host-byte admission"
            )
        if capability is not None and not isinstance(
            capability, PrefetchCapability
        ):
            raise TypeError("capability must be PrefetchCapability")
        if (
            capability is not None
            and device is not None
            and _device(device) != capability.device
        ):
            raise ValueError("device and injected capability disagree")
        resolved = (
            capability
            if capability is not None
            else PrefetchCapability.detect("cpu" if device is None else device)
        )
        if resolved.device.type == "cuda":
            if limits.device_queue_depth < 1:
                raise ValueError(
                    "CUDA prefetch requires device_queue_depth of at least one"
                )
            if limits.max_device_queue_bytes < 1:
                raise ValueError(
                    "CUDA prefetch requires a device queue budget"
                )
        elif limits.device_queue_depth != 0:
            raise ValueError(
                "device_queue_depth must be zero for CPU/MPS host-only prefetch"
            )
        if pin_memory is not None and not callable(pin_memory):
            raise TypeError("pin_memory must be callable")
        if type(owns_source) is not bool:
            raise TypeError("owns_source must be bool")
        if on_yield is not None and not callable(on_yield):
            raise TypeError("on_yield must be callable")
        if on_abort is not None and not callable(on_abort):
            raise TypeError("on_abort must be callable")
        if execution_monitor is not None and not isinstance(
            execution_monitor,
            InputMonitor,
        ):
            raise TypeError("execution_monitor must be InputMonitor or None")
        if not isinstance(monitor_phase, str) or not monitor_phase:
            raise TypeError("monitor_phase must be a non-empty string")
        if monitor_split is not None and (
            not isinstance(monitor_split, str) or not monitor_split
        ):
            raise TypeError("monitor_split must be a non-empty string or None")
        self.source = source
        self.limits = limits
        self.capability = resolved
        self.pin_memory = pin_memory or (lambda tensor: tensor.pin_memory())
        self.owns_source = owns_source
        self.on_yield = on_yield
        self.on_abort = on_abort
        self.execution_monitor = execution_monitor
        self.monitor_phase = monitor_phase
        self.monitor_split = monitor_split
        self.status = (
            PrefetchStatus(
                "cuda",
                resolved.device,
                f"CUDA prefetch enabled on dedicated stream for {resolved.device}",
            )
            if resolved.device.type == "cuda"
            else PrefetchStatus(
                "host-only",
                resolved.device,
                f"CUDA disabled for {resolved.device.type}; batches remain on CPU for Lightning-owned device transfer",
            )
        )
        self._closed = False
        self._active: set[DevicePrefetchIterator] = set()
        self._last_iterator: DevicePrefetchIterator | None = None
        self._iterator_condition = threading.Condition()
        self._iterator_reserved = False

    @property
    def active_iterator(self) -> DevicePrefetchIterator | None:
        with self._iterator_condition:
            return self._last_iterator

    @property
    def active_iterators(self) -> tuple[DevicePrefetchIterator, ...]:
        with self._iterator_condition:
            return tuple(
                iterator for iterator in self._active if not iterator.closed
            )

    def __len__(self) -> int:
        length = getattr(self.source, "__len__", None)
        if not callable(length):
            raise TypeError("wrapped iterable has no length")
        return _integer(length(), "source length")

    def __iter__(self) -> DevicePrefetchIterator:
        with self._iterator_condition:
            if self._closed:
                raise RuntimeError("DevicePrefetchLoader is closed")
            if self._iterator_reserved or any(
                not iterator.closed for iterator in self._active
            ):
                raise RuntimeError(
                    "DevicePrefetchLoader allows one active prefetch iterator"
                )
            self._iterator_reserved = True
        try:
            iterator = DevicePrefetchIterator(self)
        except BaseException:
            with self._iterator_condition:
                self._iterator_reserved = False
                self._iterator_condition.notify_all()
            raise
        with self._iterator_condition:
            self._active.add(iterator)
            self._last_iterator = iterator
            self._iterator_reserved = False
            closed = self._closed
            self._iterator_condition.notify_all()
        if closed:
            iterator.close()
            raise RuntimeError("DevicePrefetchLoader is closed")
        return iterator

    def _iterator_finished(self, iterator: DevicePrefetchIterator) -> None:
        with self._iterator_condition:
            self._active.discard(iterator)
            self._iterator_condition.notify_all()

    def close(self) -> None:
        with self._iterator_condition:
            self._closed = True
            while self._iterator_reserved:
                self._iterator_condition.wait()
            active = tuple(self._active)
        for iterator in active:
            iterator.close()
        with self._iterator_condition:
            self._active.clear()
            self._iterator_condition.notify_all()

    def __enter__(self) -> DevicePrefetchLoader:
        return self

    def __exit__(self, *exc_info: object) -> None:
        self.close()

    def __del__(self) -> None:
        with contextlib.suppress(Exception):
            self.close()


__all__ = [
    "BatchByteEstimate",
    "DevicePrefetchIterator",
    "DevicePrefetchLoader",
    "PrefetchCapability",
    "PrefetchError",
    "PrefetchLimitError",
    "PrefetchLimits",
    "PrefetchStatus",
    "TensorFieldEstimate",
    "estimate_batch_bytes",
]
