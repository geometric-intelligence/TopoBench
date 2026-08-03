"""Strategy-driven CPU sampling over materialized and immutable graph stores."""

from __future__ import annotations

import copy
import hashlib
import inspect
import json
import os
import tempfile
import threading
import time
from collections.abc import Iterable, Iterator, Mapping, Sequence
from contextlib import contextmanager, suppress
from dataclasses import dataclass, is_dataclass, replace
from dataclasses import fields as dataclass_fields
from importlib import metadata
from numbers import Integral
from pathlib import Path
from typing import Any, Literal, Protocol, TypeAlias, runtime_checkable

import numpy as np
import torch
from lightning import LightningDataModule
from torch import Tensor
from torch.utils.data import DataLoader, Dataset, Sampler
from torch_geometric.data import Data, HeteroData
from torch_geometric.loader import NeighborLoader
from torch_geometric.sampler import NeighborSampler

from topobench.data.stores.materialized_partition import (
    MaterializedHomogeneousPartition,
)
from topobench.data.stores.pyg_store import (
    PyGTypedFeatureStore,
    PyGTypedGraphStore,
)
from topobench.data.stores.typed_graph_store import (
    TypedGraphStore,
    TypedGraphStoreState,
)
from topobench.data.stores.typed_partition_book import TypedPartitionBook
from topobench.dataloader.device_prefetch import (
    DevicePrefetchLoader,
    PrefetchCapability,
    PrefetchLimits,
)
from topobench.dataloader.graph import loader_worker_options
from topobench.dataloader.input_monitor import InputMonitor
from topobench.dataloader.sequence_state import SequenceIdentity, SequenceState
from topobench.profiling.execution_events import (
    ExecutionOperation,
    ExecutionStatus,
    descriptor_digest,
)
from topobench.transforms.fittable import (
    FitContext,
    FitStateError,
    FitStateNotFoundError,
    FitStatus,
    FittableTransform,
    build_fit_state_key,
)

Phase: TypeAlias = Literal["train", "val", "test"]
CanonicalRelation: TypeAlias = tuple[str, str, str]
NativeBatch: TypeAlias = Data | HeteroData
SamplingSource: TypeAlias = TypedGraphStore | MaterializedHomogeneousPartition | HeteroData
MaterializedSource: TypeAlias = MaterializedHomogeneousPartition | HeteroData
_MAX_SEED = 2**63 - 1
_PHASES: tuple[Phase, ...] = ("train", "val", "test")
_PYG_SAMPLER_RNG_LOCK = threading.Lock()


class SamplingCapabilityError(RuntimeError):
    """Report a contextual strategy/backend capability mismatch."""


@dataclass(frozen=True, slots=True)
class SamplingDescriptor:
    """One immutable, content-bound native batch materialization request."""

    content_sha256: str
    active_split_tag: str
    phase: Phase
    strategy: str
    strategy_options_json: str
    batch_ordinal: int
    partition_ids: tuple[int, ...] = ()
    target_node_type: str | None = None
    target_seed_ids: tuple[int, ...] = ()
    participant_counts: tuple[tuple[str, int], ...] = ()
    generator_seed: int = 0
    generator_state_sha256: str = ""

    def __post_init__(self) -> None:
        _require_sha256(self.content_sha256, "content_sha256")
        _nonempty(self.active_split_tag, "active_split_tag")
        _phase(self.phase)
        _nonempty(self.strategy, "strategy")
        try:
            options = json.loads(self.strategy_options_json)
        except (TypeError, json.JSONDecodeError) as error:
            raise ValueError("strategy_options_json must be canonical JSON") from error
        if _json(options) != self.strategy_options_json:
            raise ValueError("strategy_options_json must be canonical JSON")
        object.__setattr__(
            self,
            "batch_ordinal",
            _integer(self.batch_ordinal, "batch_ordinal", minimum=0),
        )
        object.__setattr__(
            self,
            "partition_ids",
            _identities(
                self.partition_ids,
                "partition IDs",
                "duplicate partition identity",
            ),
        )
        object.__setattr__(
            self,
            "target_seed_ids",
            _identities(
                self.target_seed_ids,
                "target seed IDs",
                "duplicate target seed identity",
            ),
        )
        if self.partition_ids and self.target_seed_ids:
            raise ValueError("sampling descriptor cannot bind both partition and target identities")
        if self.target_node_type is None and self.target_seed_ids:
            raise ValueError("target seed IDs require target_node_type")
        if self.target_node_type is not None:
            _nonempty(self.target_node_type, "target_node_type")
            if not self.target_seed_ids:
                raise ValueError("target_node_type requires target seed IDs")
        if not self.partition_ids and not self.target_seed_ids:
            raise ValueError("sampling descriptor requires participant identities")
        object.__setattr__(
            self,
            "participant_counts",
            _canonical_participant_counts(self.participant_counts),
        )
        seed = _integer(self.generator_seed, "generator_seed", minimum=0)
        object.__setattr__(self, "generator_seed", seed)
        if seed > _MAX_SEED:
            raise ValueError(f"generator_seed must be no greater than {_MAX_SEED}")
        _require_sha256(self.generator_state_sha256, "generator_state_sha256")


@runtime_checkable
class GraphSamplingStrategy(Protocol):
    """Small lifecycle shared by graph sampling implementations."""

    name: str
    seed: int

    def validate_capabilities(self, source: SamplingSource) -> None: ...

    def setup(
        self,
        source: SamplingSource,
        *,
        phase: Phase,
        active_split_tag: str,
        shuffle: bool,
    ) -> tuple[SamplingDescriptor, ...]: ...

    def materialize(self, source: SamplingSource, descriptor: SamplingDescriptor) -> NativeBatch: ...

    def sampler_state(self) -> dict[str, object]: ...


def _json(value: object) -> str:
    return json.dumps(value, sort_keys=True, separators=(",", ":"))


def _nonempty(value: object, name: str) -> str:
    if not isinstance(value, str):
        raise TypeError(f"{name} must be a string")
    if not value:
        raise ValueError(f"{name} must not be empty")
    return value


def _require_sha256(value: object, name: str) -> str:
    if not isinstance(value, str):
        raise TypeError(f"{name} must be a string")
    if len(value) != 64 or any(char not in "0123456789abcdef" for char in value):
        raise ValueError(f"{name} must be a lowercase SHA-256 digest")
    return value


def _integer(value: object, name: str, *, minimum: int) -> int:
    if isinstance(value, bool) or not isinstance(value, Integral):
        raise TypeError(f"{name} must be an integer")
    result = int(value)
    if result < minimum:
        raise ValueError(f"{name} must be at least {minimum}")
    return result


def _boolean(value: object, name: str) -> bool:
    if type(value) is not bool:
        raise TypeError(f"{name} must be bool")
    return value


def _phase(value: object) -> Phase:
    if not isinstance(value, str):
        raise TypeError("phase must be a string")
    if value not in _PHASES:
        raise ValueError(f"phase must be one of {_PHASES!r}")
    return value


def _identities(values: object, name: str, duplicate: str) -> tuple[int, ...]:
    if isinstance(values, (str, bytes)) or not isinstance(values, Sequence):
        raise TypeError(f"{name} must be an ordered sequence")
    result: list[int] = []
    seen: set[int] = set()
    for value in values:
        if isinstance(value, bool) or not isinstance(value, Integral):
            raise TypeError(f"{name} must contain non-boolean integers")
        identity = int(value)
        if identity < 0:
            raise ValueError(f"{name} must be non-negative")
        if identity in seen:
            raise ValueError(duplicate)
        result.append(identity)
        seen.add(identity)
    return tuple(result)

def _canonical_participant_counts(
    values: object,
) -> tuple[tuple[str, int], ...]:
    """Return immutable counts in deterministic participant-name order."""
    if isinstance(values, Mapping):
        entries: object = tuple(values.items())
    else:
        entries = values
    if isinstance(entries, (str, bytes)) or not isinstance(entries, Sequence):
        raise TypeError("participant_counts must be a mapping or ordered sequence")
    result: list[tuple[str, int]] = []
    seen: set[str] = set()
    for entry in entries:
        if (
            isinstance(entry, (str, bytes))
            or not isinstance(entry, Sequence)
            or len(entry) != 2
        ):
            raise TypeError(
                "participant_counts entries must be (str, integer) pairs"
            )
        participant = _nonempty(entry[0], "participant_counts participant")
        count = _integer(
            entry[1],
            "participant_counts count",
            minimum=0,
        )
        if participant in seen:
            raise ValueError("duplicate participant count identity")
        seen.add(participant)
        result.append((participant, count))
    return tuple(sorted(result))


def _partition_groups(groups: Sequence[Sequence[int]] | None) -> tuple[tuple[int, ...], ...] | None:
    if groups is None:
        return None
    if isinstance(groups, (str, bytes)) or not isinstance(groups, Sequence):
        raise TypeError("partition_groups must be an ordered sequence")
    result: list[tuple[int, ...]] = []
    for group in groups:
        normalized = _identities(group, "partition IDs", "duplicate partition identity")
        if not normalized:
            raise ValueError("partition group must not be empty")
        result.append(tuple(sorted(normalized)))
    if not result:
        raise ValueError("partition_groups must not be empty")
    return tuple(result)


def _state_sha(seed: int) -> str:
    state = torch.Generator(device="cpu").manual_seed(seed).get_state()
    return hashlib.sha256(state.numpy().tobytes()).hexdigest()


def _descriptor_seed(
    root_seed: int,
    content_sha256: str,
    tag: str,
    phase: Phase,
    strategy: str,
    options: str,
    ordinal: int,
    identities: Sequence[int],
) -> int:
    payload = _json({
        "active_split_tag": tag,
        "batch_ordinal": ordinal,
        "content_sha256": content_sha256,
        "identities": list(identities),
        "phase": phase,
        "root_seed": root_seed,
        "strategy": strategy,
        "strategy_options": json.loads(options),
    }).encode()
    return int.from_bytes(hashlib.sha256(payload).digest()[:8], "big") & _MAX_SEED


def _frame(digest: Any, name: str, payload: bytes) -> None:
    encoded = name.encode()
    digest.update(len(encoded).to_bytes(8, "big"))
    digest.update(encoded)
    digest.update(len(payload).to_bytes(8, "big"))
    digest.update(payload)


def _tensor_frame(digest: Any, name: str, tensor: Tensor) -> None:
    value = tensor.detach().cpu().contiguous()
    _frame(digest, name + ":dtype", str(value.dtype).encode())
    _frame(digest, name + ":shape", _json(list(value.shape)).encode())
    _frame(digest, name, value.numpy().tobytes())


def _materialized_sha(source: Data | HeteroData) -> str:
    """Hash the actual native tensor payload without trusting graph metadata."""
    digest = hashlib.sha256()
    if isinstance(source, HeteroData):
        _frame(digest, "kind", b"heterogeneous")
        for node_type in source.node_types:
            _frame(digest, "node_type", node_type.encode())
            _frame(
                digest,
                f"node:{node_type}:num_nodes",
                str(int(source[node_type].num_nodes)).encode(),
            )
            for name, value in sorted(source[node_type].items()):
                if isinstance(value, Tensor):
                    _tensor_frame(digest, f"node:{node_type}:{name}", value)
        for relation in source.edge_types:
            _frame(digest, "relation", _json(relation).encode())
            for name, value in sorted(source[relation].items()):
                if isinstance(value, Tensor):
                    _tensor_frame(digest, f"edge:{relation}:{name}", value)
    else:
        _frame(digest, "kind", b"homogeneous")
        _frame(digest, "num_nodes", str(int(source.num_nodes)).encode())
        for name, value in sorted(source.items()):
            if isinstance(value, Tensor):
                _tensor_frame(digest, f"field:{name}", value)
    return digest.hexdigest()


def _partition_items(
    source: MaterializedHomogeneousPartition,
) -> tuple[tuple[str, object], ...]:
    return tuple(
        (field.name, getattr(source.partition, field.name))
        for field in dataclass_fields(source.partition)
    )


def _clone_materialized(source: MaterializedSource) -> MaterializedSource:
    """Clone every admitted native tensor while retaining qualified metadata."""
    if isinstance(source, HeteroData):
        return source.clone()
    snapshot = copy.copy(source)
    snapshot._data = source._data.clone()
    snapshot._attribute_roles = dict(source._attribute_roles)
    partition_values = {
        name: value.clone() if isinstance(value, Tensor) else value
        for name, value in _partition_items(source)
    }
    snapshot.partition = type(source.partition)(**partition_values)
    snapshot.perm_to_global = snapshot.partition.node_perm
    return snapshot


def _version_signature(
    source: MaterializedSource,
) -> tuple[tuple[str, int, int], ...]:
    """Describe tensor bindings and version counters without scanning payloads."""
    data = source._data if isinstance(source, MaterializedHomogeneousPartition) else source
    result: list[tuple[str, int, int]] = []
    for storage_index, storage in enumerate(data.stores):
        for name, value in sorted(storage.items()):
            if isinstance(value, Tensor):
                result.append(
                    (
                        f"storage:{storage_index}:{name}",
                        id(value),
                        value._version,
                    )
                )
    if isinstance(source, MaterializedHomogeneousPartition):
        for name, value in _partition_items(source):
            if isinstance(value, Tensor):
                result.append((f"partition:{name}", id(value), value._version))
    return tuple(result)


def _snapshot_sha(source: MaterializedSource) -> str:
    if isinstance(source, HeteroData):
        return _materialized_sha(source)
    digest = hashlib.sha256()
    _frame(digest, "graph", _materialized_sha(source._data).encode())
    for name, value in _partition_items(source):
        if isinstance(value, Tensor):
            _tensor_frame(digest, f"partition:{name}", value)
        else:
            _frame(digest, f"partition:{name}", str(value).encode())
    return digest.hexdigest()


@dataclass(slots=True)
class _MaterializedAdmission:
    """One private immutable-by-contract materialized strategy source."""

    original: MaterializedSource
    source: MaterializedSource
    content_sha256: str
    versions: tuple[tuple[str, int, int], ...]

    def check(self, requested: MaterializedSource) -> None:
        if self.original is not requested and self.source is not requested:
            raise ValueError(
                "sampling strategy is already bound to another materialized source"
            )
        if _version_signature(self.source) != self.versions:
            raise RuntimeError("admitted materialized tensor payload was mutated")

    def __getstate__(
        self,
    ) -> tuple[MaterializedSource, MaterializedSource, str]:
        return self.original, self.source, self.content_sha256

    def __setstate__(
        self,
        state: tuple[MaterializedSource, MaterializedSource, str],
    ) -> None:
        self.original, self.source, self.content_sha256 = state
        self.versions = _version_signature(self.source)


def _admit_materialized(
    admission: _MaterializedAdmission | None,
    source: MaterializedSource,
) -> _MaterializedAdmission:
    if admission is None:
        snapshot = _clone_materialized(source)
        admission = _MaterializedAdmission(
            original=source,
            source=snapshot,
            content_sha256=_snapshot_sha(snapshot),
            versions=_version_signature(snapshot),
        )
    admission.check(source)
    return admission


def _declared_tag(source: SamplingSource) -> str | None:
    if isinstance(source, TypedGraphStore):
        return source.active_split_tag
    data = source._data if isinstance(source, MaterializedHomogeneousPartition) else source
    value = getattr(data, "active_split_tag", None)
    return value if isinstance(value, str) else None


def _active_tag(source: SamplingSource, tag: str) -> str:
    tag = _nonempty(tag, "active_split_tag")
    declared = _declared_tag(source)
    if declared is not None and declared != tag:
        raise ValueError(f"active split tag must remain {declared!r}; received {tag!r}")
    if isinstance(source, TypedGraphStore) and tag not in source._manifest["splits"]:
        raise ValueError(f"active split tag {tag!r} is not stored")
    return tag


def _target(source: TypedGraphStore | HeteroData) -> str:
    if isinstance(source, TypedGraphStore):
        return source._manifest["target_node_type"]
    declared = getattr(source, "target_node_type", None)
    if isinstance(declared, str) and declared in source.node_types:
        return declared
    candidates = [node_type for node_type in source.node_types if all(f"{p}_mask" in source[node_type] for p in _PHASES)]
    if len(candidates) != 1:
        raise ValueError("heterogeneous source must identify exactly one target node type")
    return candidates[0]


def _phase_ids(source: TypedGraphStore | HeteroData, target: str, tag: str, phase: Phase) -> tuple[int, ...]:
    if isinstance(source, TypedGraphStore):
        values = tuple(int(value) for value in source.split_ids(tag, phase))
    else:
        field = f"{phase}_mask" if _declared_tag(source) in {None, tag} else f"{tag}_{phase}_mask"
        if field not in source[target]:
            raise ValueError(f"target node type {target!r} is missing {field!r}")
        mask = source[target][field]
        if not isinstance(mask, Tensor) or mask.dtype != torch.bool or mask.dim() != 1:
            raise TypeError(f"{field} must be a one-dimensional bool tensor")
        if mask.device.type != "cpu":
            raise ValueError(f"{field} must remain on CPU")
        values = tuple(int(value) for value in mask.nonzero().reshape(-1))
    _identities(values, "phase target seed IDs", "duplicate phase target seed identity")
    if not values:
        raise ValueError(f"phase {phase!r} has no target seed IDs")
    return values


def _shuffled(values: tuple[Any, ...], shuffle: bool, seed: int) -> tuple[Any, ...]:
    if not shuffle or len(values) < 2:
        return values
    order = torch.randperm(len(values), generator=torch.Generator().manual_seed(seed)).tolist()
    return tuple(values[index] for index in order)


def _torch(value: np.ndarray) -> Tensor:
    return torch.from_numpy(np.array(value, copy=True))


def _store_nodes(store: TypedGraphStore, node_type: str, partition_ids: Sequence[int]) -> np.ndarray:
    permutation = store.partition_permutation(node_type)
    partptr = store.partition_partptr(node_type)
    return np.concatenate([permutation[int(partptr[p]):int(partptr[p + 1])] for p in partition_ids]).astype(np.int64, copy=False)


def _book_nodes(book: TypedPartitionBook, node_type: str, partition_ids: Sequence[int]) -> np.ndarray:
    permutation = book.node_permutations[node_type]
    partptr = book.node_partptr[node_type]
    return np.concatenate([permutation[int(partptr[p]):int(partptr[p + 1])] for p in partition_ids]).astype(np.int64, copy=False)


def _validate_partitions(groups: Sequence[Sequence[int]], count: int) -> None:
    for group in groups:
        for part_id in group:
            if part_id >= count:
                raise ValueError(f"partition ID {part_id} is outside [0, {count})")


def _default_groups(partition_ids: Iterable[int], size: int) -> tuple[tuple[int, ...], ...]:
    values = tuple(sorted(set(int(value) for value in partition_ids)))
    if not values:
        raise ValueError("active phase does not participate in any partition")
    return tuple(values[start:start + size] for start in range(0, len(values), size))


def _capability(store: TypedGraphStore, name: str) -> None:
    supported = tuple(store._manifest.get("supported_capabilities", ()))
    if name not in supported:
        raise SamplingCapabilityError(f"{name} is unavailable for store {store.content_sha256}; supported={supported!r}")


def _attach(batch: NativeBatch, descriptor: SamplingDescriptor, counts: Mapping[str, int]) -> NativeBatch:
    batch.sampling_descriptor = descriptor
    batch.participant_counts = dict(counts)
    batch.active_split_tag = descriptor.active_split_tag
    batch.sampling_phase = descriptor.phase
    for storage in batch.stores:
        for value in storage.values():
            if isinstance(value, Tensor) and value.device.type != "cpu":
                raise RuntimeError("graph sampling strategies emit CPU tensors only")
    return batch


class _ClusterBase:
    name: str

    def __init__(self, clusters_per_batch: int, partition_groups: Sequence[Sequence[int]] | None, seed: int) -> None:
        self.clusters_per_batch = _integer(clusters_per_batch, "clusters_per_batch", minimum=1)
        self.partition_groups = _partition_groups(partition_groups)
        self.seed = _integer(seed, "seed", minimum=0)
        if self.seed > _MAX_SEED:
            raise ValueError(f"seed must be no greater than {_MAX_SEED}")
        self._admission: _MaterializedAdmission | None = None

    def _resolve_source(self, source: SamplingSource) -> SamplingSource:
        if isinstance(source, TypedGraphStore):
            return source
        assert isinstance(source, (MaterializedHomogeneousPartition, HeteroData))
        self._admission = _admit_materialized(self._admission, source)
        return self._admission.source

    def _content_identity(self, source: SamplingSource) -> str:
        if isinstance(source, TypedGraphStore):
            return source.content_sha256
        self._resolve_source(source)
        assert self._admission is not None
        return self._admission.content_sha256

    def sampler_state(self) -> dict[str, object]:
        return {"format_version": "graph-sampling-state-v1", "seed": self.seed, "strategy": self.name}

    def _options(self) -> str:
        return _json({"clusters_per_batch": self.clusters_per_batch, "partition_groups": None if self.partition_groups is None else [list(g) for g in self.partition_groups]})

    def _ordered(self, groups: tuple[tuple[int, ...], ...], shuffle: bool, content: str, tag: str, phase: Phase) -> tuple[tuple[int, ...], ...]:
        seed = _descriptor_seed(self.seed, content, tag, phase, self.name, self._options(), 0, tuple(v for g in groups for v in g))
        return _shuffled(groups, shuffle, seed)

    def _descriptor(self, content: str, tag: str, phase: Phase, ordinal: int, parts: tuple[int, ...], counts: tuple[tuple[str, int], ...]) -> SamplingDescriptor:
        options = self._options()
        seed = _descriptor_seed(self.seed, content, tag, phase, self.name, options, ordinal, parts)
        return SamplingDescriptor(content, tag, phase, self.name, options, ordinal, partition_ids=parts, participant_counts=counts, generator_seed=seed, generator_state_sha256=_state_sha(seed))


class HomogeneousClusterStrategy(_ClusterBase):
    """Exact directed homogeneous unions over Task5 or Task7 partitions."""

    name = "homogeneous-cluster"

    def __init__(self, *, clusters_per_batch: int = 1, partition_groups: Sequence[Sequence[int]] | None = None, seed: int = 0) -> None:
        super().__init__(clusters_per_batch, partition_groups, seed)

    def validate_capabilities(self, source: SamplingSource) -> None:
        if isinstance(source, TypedGraphStore):
            if source.output_kind != "homogeneous":
                raise ValueError(f"homogeneous-cluster requires a homogeneous store; received {source.output_kind!r}")
            _capability(source, self.name)
        elif not isinstance(source, MaterializedHomogeneousPartition):
            raise TypeError("homogeneous-cluster source must be MaterializedHomogeneousPartition or TypedGraphStore")
        else:
            self._resolve_source(source)

    def setup(self, source: SamplingSource, *, phase: Phase, active_split_tag: str, shuffle: bool) -> tuple[SamplingDescriptor, ...]:
        self.validate_capabilities(source)
        source = self._resolve_source(source)
        phase, tag, shuffle = _phase(phase), _active_tag(source, active_split_tag), _boolean(shuffle, "shuffle")
        content = self._content_identity(source)
        if isinstance(source, TypedGraphStore):
            count = source.num_partitions
            if self.partition_groups is None:
                target = source._manifest["target_node_type"]
                assignment = source.partition_assignment(target)
                groups = _default_groups((int(assignment[int(i)]) for i in source.split_ids(tag, phase)), self.clusters_per_batch)
            else:
                groups = self.partition_groups
            def participant(ids: Sequence[int]) -> int:
                return len(_store_nodes(source, source.node_types[0], ids))
        else:
            count = source.num_parts
            groups = _default_groups(source.partition_ids_for_phase(phase), self.clusters_per_batch) if self.partition_groups is None else self.partition_groups
            def participant(ids: Sequence[int]) -> int:
                return sum(
                    int(source.partition.partptr[partition + 1])
                    - int(source.partition.partptr[partition])
                    for partition in ids
                )
        _validate_partitions(groups, count)
        groups = self._ordered(groups, shuffle, content, tag, phase)
        return tuple(self._descriptor(content, tag, phase, ordinal, group, (("node", participant(group)),)) for ordinal, group in enumerate(groups))

    def materialize(self, source: SamplingSource, descriptor: SamplingDescriptor) -> Data:
        self.validate_capabilities(source)
        source = self._resolve_source(source)
        self._validate(source, descriptor)
        batch = source.materialize(descriptor.partition_ids, phase=descriptor.phase) if isinstance(source, MaterializedHomogeneousPartition) else _homogeneous_store(source, descriptor)
        return _attach(batch, descriptor, {"node": int(batch.num_nodes)})

    def _validate(self, source: SamplingSource, descriptor: SamplingDescriptor) -> None:
        if descriptor.strategy != self.name:
            raise ValueError("descriptor strategy identity mismatch")
        if descriptor.content_sha256 != self._content_identity(source):
            raise ValueError("descriptor content identity mismatch")
        _active_tag(source, descriptor.active_split_tag)
        if descriptor.strategy_options_json != self._options():
            raise ValueError("descriptor strategy options mismatch")


def _mask(selected: np.ndarray, identities: np.ndarray) -> Tensor:
    return torch.from_numpy(np.isin(selected, identities, assume_unique=True))


def _induced(store: TypedGraphStore, relation: CanonicalRelation, source_ids: np.ndarray, destination_ids: np.ndarray) -> tuple[Tensor, np.ndarray]:
    row, colptr = store.relation_csc(relation)
    source_local = {int(value): index for index, value in enumerate(source_ids)}
    destination_local = {int(value): index for index, value in enumerate(destination_ids)}
    sources: list[int] = []
    destinations: list[int] = []
    positions: list[int] = []
    for destination in sorted(destination_local):
        for position in range(int(colptr[destination]), int(colptr[destination + 1])):
            source = source_local.get(int(row[position]))
            if source is not None:
                sources.append(source)
                destinations.append(destination_local[destination])
                positions.append(position)
    return torch.tensor([sources, destinations], dtype=torch.long).reshape(2, -1), np.asarray(positions, dtype=np.int64)


def _homogeneous_store(store: TypedGraphStore, descriptor: SamplingDescriptor) -> Data:
    node_type, relation = store.node_types[0], store.relation_types[0]
    selected = _store_nodes(store, node_type, descriptor.partition_ids)
    edge_index, positions = _induced(store, relation, selected, selected)
    output = Data(edge_index=edge_index, num_nodes=len(selected))
    node = store._node(node_type)
    output.x = _torch(store.node_array(node_type, "x", selected))
    if node["y"] is not None:
        output.y = _torch(store.node_array(node_type, "y", selected))
    for name in node["fields"]:
        output[name] = _torch(store.node_array(node_type, name, selected))
    if store._relation(relation)["edge_id"] is not None:
        output.edge_id = _torch(
            store.relation_field(relation, "edge_id", positions)
        )
    for name in store._relation(relation)["fields"]:
        output[name] = _torch(store.relation_field(relation, name, positions))
    for phase in _PHASES:
        output[f"{phase}_mask"] = _mask(selected, store.split_ids(descriptor.active_split_tag, phase))
    output.global_nid = _torch(selected)
    output.selected_partition_ids = torch.tensor(descriptor.partition_ids)
    output.num_selected_partitions = len(descriptor.partition_ids)
    output.supervised_mask = output[f"{descriptor.phase}_mask"].clone()
    return output


class HeterogeneousClusterStrategy(_ClusterBase):
    """Exact typed partition unions owned by ``HeteroData.subgraph`` semantics."""

    name = "heterogeneous-cluster"

    def __init__(self, *, partition_book: TypedPartitionBook | None = None, clusters_per_batch: int = 1, partition_groups: Sequence[Sequence[int]] | None = None, seed: int = 0) -> None:
        if partition_book is not None and not isinstance(partition_book, TypedPartitionBook):
            raise TypeError("partition_book must be a TypedPartitionBook")
        self.partition_book = partition_book
        super().__init__(clusters_per_batch, partition_groups, seed)

    def _content_identity(self, source: SamplingSource) -> str:
        content = super()._content_identity(source)
        if isinstance(source, TypedGraphStore):
            return content
        assert self.partition_book is not None
        digest = hashlib.sha256()
        _frame(digest, "graph", content.encode())
        _frame(
            digest,
            "partition_book",
            self.partition_book.content_identity.encode(),
        )
        return digest.hexdigest()

    def validate_capabilities(self, source: SamplingSource) -> None:
        if isinstance(source, TypedGraphStore):
            if source.output_kind != "heterogeneous":
                raise ValueError(f"heterogeneous-cluster requires a heterogeneous store; received {source.output_kind!r}")
            _capability(source, self.name)
            return
        if not isinstance(source, HeteroData):
            raise TypeError("source must be HeteroData or TypedGraphStore")
        source = self._resolve_source(source)
        assert isinstance(source, HeteroData)
        if self.partition_book is None:
            raise ValueError("materialized heterogeneous-cluster requires partition_book")
        if tuple(source.node_types) != tuple(self.partition_book.node_assignments):
            raise ValueError("partition book node types do not match HeteroData")
        declared_identity = getattr(source, "partition_book_identity", None)
        if declared_identity is None:
            raise ValueError(
                "materialized partition book identity is required"
            )
        if declared_identity != self.partition_book.content_identity:
            raise ValueError("partition book identity mismatch")

    def setup(self, source: SamplingSource, *, phase: Phase, active_split_tag: str, shuffle: bool) -> tuple[SamplingDescriptor, ...]:
        self.validate_capabilities(source)
        source = self._resolve_source(source)
        assert isinstance(source, (TypedGraphStore, HeteroData))
        phase, tag, shuffle = _phase(phase), _active_tag(source, active_split_tag), _boolean(shuffle, "shuffle")
        content, target = self._content_identity(source), _target(source)
        seeds = _phase_ids(source, target, tag, phase)
        if isinstance(source, TypedGraphStore):
            count, assignment, node_types = source.num_partitions, source.partition_assignment(target), source.node_types
            def select(node_type: str, ids: Sequence[int]) -> np.ndarray:
                return _store_nodes(source, node_type, ids)
        else:
            assert self.partition_book is not None
            count, assignment, node_types = self.partition_book.num_partitions, self.partition_book.node_assignments[target], tuple(source.node_types)
            def select(node_type: str, ids: Sequence[int]) -> np.ndarray:
                return _book_nodes(self.partition_book, node_type, ids)
        groups = _default_groups((int(assignment[seed]) for seed in seeds), self.clusters_per_batch) if self.partition_groups is None else self.partition_groups
        _validate_partitions(groups, count)
        groups = self._ordered(groups, shuffle, content, tag, phase)
        return tuple(self._descriptor(content, tag, phase, ordinal, group, tuple((node_type, len(select(node_type, group))) for node_type in node_types)) for ordinal, group in enumerate(groups))

    def materialize(self, source: SamplingSource, descriptor: SamplingDescriptor) -> HeteroData:
        self.validate_capabilities(source)
        source = self._resolve_source(source)
        if descriptor.strategy != self.name or descriptor.content_sha256 != self._content_identity(source):
            raise ValueError("descriptor strategy or content identity mismatch")
        _active_tag(source, descriptor.active_split_tag)
        if descriptor.strategy_options_json != self._options():
            raise ValueError("descriptor strategy options mismatch")
        if isinstance(source, TypedGraphStore):
            batch = _heterogeneous_store(source, descriptor)
        else:
            assert self.partition_book is not None
            subsets = {node_type: _torch(_book_nodes(self.partition_book, node_type, descriptor.partition_ids)) for node_type in source.node_types}
            batch = source.subgraph(subsets)
            for node_type, subset in subsets.items():
                if "n_id" not in batch[node_type]:
                    batch[node_type].n_id = subset
            target = _target(source)
            batch[target].supervised_mask = batch[target][f"{descriptor.phase}_mask"].clone()
            batch.selected_partition_ids = torch.tensor(descriptor.partition_ids)
            batch.num_selected_partitions = len(descriptor.partition_ids)
        return _attach(batch, descriptor, {node_type: int(batch[node_type].num_nodes) for node_type in batch.node_types})


def _attach_store_split_masks(
    store: TypedGraphStore,
    storage: Any,
    node_ids: np.ndarray,
    active_split_tag: str,
) -> None:
    """Attach every named split mask and the active-tag convenience masks."""
    for tag in store._manifest["splits"]:
        for phase in _PHASES:
            value = _mask(node_ids, store.split_ids(tag, phase))
            storage[f"{tag}_{phase}_mask"] = value
            if tag == active_split_tag:
                storage[f"{phase}_mask"] = value.clone()


def _heterogeneous_store(store: TypedGraphStore, descriptor: SamplingDescriptor) -> HeteroData:
    selected = {node_type: _store_nodes(store, node_type, descriptor.partition_ids) for node_type in store.node_types}
    output, target = HeteroData(), store._manifest["target_node_type"]
    for node_type, ids in selected.items():
        node = store._node(node_type)
        output[node_type].x = _torch(store.node_array(node_type, "x", ids))
        if node["y"] is not None:
            output[node_type].y = _torch(store.node_array(node_type, "y", ids))
        for name in node["fields"]:
            output[node_type][name] = _torch(store.node_array(node_type, name, ids))
        output[node_type].n_id = _torch(ids)
        output[node_type].num_nodes = len(ids)
        if node_type == target:
            _attach_store_split_masks(
                store,
                output[node_type],
                ids,
                descriptor.active_split_tag,
            )
    for relation in store.relation_types:
        edge_index, positions = _induced(store, relation, selected[relation[0]], selected[relation[2]])
        output[relation].edge_index = edge_index
        if store._relation(relation)["edge_id"] is not None:
            output[relation].edge_id = _torch(
                store.relation_field(relation, "edge_id", positions)
            )
        for name in store._relation(relation)["fields"]:
            output[relation][name] = _torch(store.relation_field(relation, name, positions))
    output[target].supervised_mask = output[target][f"{descriptor.phase}_mask"].clone()
    output.selected_partition_ids = torch.tensor(descriptor.partition_ids)
    output.num_selected_partitions = len(descriptor.partition_ids)
    return output


def _fanout_values(values: object) -> tuple[int, ...]:
    if isinstance(values, (str, bytes)) or not isinstance(values, Sequence):
        raise TypeError("fanout must be an ordered sequence")
    if not values:
        raise ValueError("fanout must not be empty")
    result: list[int] = []
    for value in values:
        if isinstance(value, bool) or not isinstance(value, Integral):
            raise TypeError("fanout values must contain non-boolean integers")
        value = int(value)
        if value < -1:
            raise ValueError("fanout values must be at least -1")
        result.append(value)
    return tuple(result)


def _relation(value: object) -> CanonicalRelation:
    if isinstance(value, (str, bytes)) or not isinstance(value, Sequence) or len(value) != 3 or any(not isinstance(item, str) or not item for item in value):
        raise TypeError("fanout relation keys must be canonical string triples")
    return value[0], value[1], value[2]


def _version(name: str) -> str | None:
    try:
        return metadata.version(name)
    except metadata.PackageNotFoundError:
        return None


class HeterogeneousNeighborStrategy:
    """PyG-owned typed target-seed sampling with deterministic qualification."""

    name = "heterogeneous-neighbor"

    def __init__(self, *, batch_size: int, num_neighbors: Sequence[int] | Mapping[CanonicalRelation, Sequence[int]], seed: int = 0, replace: bool = False, subgraph_type: str = "directional", sample_direction: str = "forward", filter_per_worker: bool = False) -> None:
        self.batch_size = _integer(batch_size, "batch_size", minimum=1)
        self.seed = _integer(seed, "seed", minimum=0)
        if self.seed > _MAX_SEED:
            raise ValueError(f"seed must be no greater than {_MAX_SEED}")
        self.replace, self.filter_per_worker = _boolean(replace, "replace"), _boolean(filter_per_worker, "filter_per_worker")
        if subgraph_type not in {"directional", "induced"}:
            raise ValueError("subgraph_type must be 'directional' or 'induced'")
        self.subgraph_type = subgraph_type
        if sample_direction not in {"forward", "backward"}:
            raise ValueError("sample_direction must be 'forward' or 'backward'")
        self.sample_direction = sample_direction
        if isinstance(num_neighbors, Mapping):
            self._relation_fanout: tuple[tuple[CanonicalRelation, tuple[int, ...]], ...] | None = tuple((_relation(key), _fanout_values(value)) for key, value in num_neighbors.items())
            if len({key for key, _ in self._relation_fanout}) != len(self._relation_fanout):
                raise ValueError("duplicate fanout relation identity")
            self._generic_fanout: tuple[int, ...] | None = None
        else:
            self._relation_fanout, self._generic_fanout = None, _fanout_values(num_neighbors)
        self._admission: _MaterializedAdmission | None = None

    def _resolve_source(
        self,
        source: SamplingSource,
    ) -> TypedGraphStore | HeteroData:
        if isinstance(source, TypedGraphStore):
            return source
        assert isinstance(source, HeteroData)
        self._admission = _admit_materialized(self._admission, source)
        assert isinstance(self._admission.source, HeteroData)
        return self._admission.source

    def _content_identity(self, source: SamplingSource) -> str:
        if isinstance(source, TypedGraphStore):
            return source.content_sha256
        self._resolve_source(source)
        assert self._admission is not None
        return self._admission.content_sha256

    def sampler_state(self) -> dict[str, object]:
        return {"format_version": "graph-sampling-state-v1", "seed": self.seed, "strategy": self.name}

    def _relations(self, source: TypedGraphStore | HeteroData) -> tuple[CanonicalRelation, ...]:
        return source.relation_types if isinstance(source, TypedGraphStore) else tuple(source.edge_types)

    def _resolved(self, source: TypedGraphStore | HeteroData) -> dict[CanonicalRelation, list[int]]:
        relations = self._relations(source)
        if self._relation_fanout is None:
            assert self._generic_fanout is not None
            resolved = {relation: list(self._generic_fanout) for relation in relations}
        else:
            configured = dict(self._relation_fanout)
            missing = tuple(relation for relation in relations if relation not in configured)
            extra = tuple(relation for relation in configured if relation not in relations)
            if missing or extra:
                raise ValueError(f"relation fanout keys must exactly match source relations; missing={missing!r}, extra={extra!r}")
            resolved = {relation: list(configured[relation]) for relation in relations}
        if len({len(values) for values in resolved.values()}) != 1:
            raise ValueError("relation fanout values must have equal hop counts")
        return resolved

    def validate_capabilities(self, source: SamplingSource) -> None:
        if isinstance(source, TypedGraphStore):
            if source.output_kind != "heterogeneous":
                raise ValueError(f"heterogeneous-neighbor requires a heterogeneous store; received {source.output_kind!r}")
            _capability(source, self.name)
            if self.sample_direction == "backward":
                raise SamplingCapabilityError(
                    "heterogeneous-neighbor backward sampling is unavailable "
                    "for TypedGraphStore: native CSR is not qualified"
                )
        elif not isinstance(source, HeteroData):
            raise TypeError("heterogeneous-neighbor source must be HeteroData or TypedGraphStore")
        if (
            self.sample_direction == "backward"
            and isinstance(source, HeteroData)
        ):
            raise SamplingCapabilityError(
                "heterogeneous-neighbor backward sampling is unsupported for "
                "materialized HeteroData by the installed NeighborSampler"
            )
        source = self._resolve_source(source)
        self._resolved(source)
        if self.subgraph_type == "induced":
            raise SamplingCapabilityError(
                "heterogeneous-neighbor induced subgraphs cannot expose exact "
                "per-hop counts without native sampler hop counts"
            )
        if self.replace:
            raise SamplingCapabilityError("heterogeneous-neighbor replace=True cannot satisfy deterministic local-generator parity on the installed backend")

    def _options(self, source: TypedGraphStore | HeteroData) -> str:
        fanout = self._resolved(source)
        return _json({"batch_size": self.batch_size, "fanout": [{"relation": list(relation), "values": fanout[relation]} for relation in self._relations(source)], "filter_per_worker": self.filter_per_worker, "replace": self.replace, "sample_direction": self.sample_direction, "subgraph_type": self.subgraph_type})

    def setup(self, source: SamplingSource, *, phase: Phase, active_split_tag: str, shuffle: bool) -> tuple[SamplingDescriptor, ...]:
        self.validate_capabilities(source)
        source = self._resolve_source(source)
        assert isinstance(source, (TypedGraphStore, HeteroData))
        phase, tag, shuffle = _phase(phase), _active_tag(source, active_split_tag), _boolean(shuffle, "shuffle")
        content, target, options = self._content_identity(source), _target(source), self._options(source)
        values = _phase_ids(source, target, tag, phase)
        order_seed = _descriptor_seed(self.seed, content, tag, phase, self.name, options, 0, values)
        values = _shuffled(values, shuffle, order_seed)
        result: list[SamplingDescriptor] = []
        for ordinal, start in enumerate(range(0, len(values), self.batch_size)):
            seeds = values[start:start + self.batch_size]
            seed = _descriptor_seed(self.seed, content, tag, phase, self.name, options, ordinal, seeds)
            result.append(SamplingDescriptor(content, tag, phase, self.name, options, ordinal, target_node_type=target, target_seed_ids=seeds, participant_counts=(), generator_seed=seed, generator_state_sha256=_state_sha(seed)))
        return tuple(result)

    def materialize(self, source: SamplingSource, descriptor: SamplingDescriptor) -> HeteroData:
        self.validate_capabilities(source)
        source = self._resolve_source(source)
        assert isinstance(source, (TypedGraphStore, HeteroData))
        if descriptor.strategy != self.name or descriptor.content_sha256 != self._content_identity(source):
            raise ValueError("descriptor strategy or content identity mismatch")
        _active_tag(source, descriptor.active_split_tag)
        if descriptor.strategy_options_json != self._options(source):
            raise ValueError("descriptor strategy options mismatch")
        if descriptor.target_node_type != _target(source):
            raise ValueError("descriptor target node identity mismatch")
        data: HeteroData | tuple[PyGTypedFeatureStore, PyGTypedGraphStore] = (PyGTypedFeatureStore(source), PyGTypedGraphStore(source)) if isinstance(source, TypedGraphStore) else source
        generator = torch.Generator().manual_seed(descriptor.generator_seed)
        resolved_fanout = self._resolved(source)
        sampler_fanout: Sequence[int] | Mapping[CanonicalRelation, Sequence[int]]
        sampler_fanout = (
            list(self._generic_fanout)
            if self.sample_direction == "backward"
            and self._generic_fanout is not None
            else resolved_fanout
        )
        neighbor_sampler = NeighborSampler(
            data,
            num_neighbors=sampler_fanout,
            replace=self.replace,
            subgraph_type=self.subgraph_type,
            sample_direction=self.sample_direction,
        )
        loader = NeighborLoader(data, input_nodes=(descriptor.target_node_type, torch.tensor(descriptor.target_seed_ids)), num_neighbors=sampler_fanout, batch_size=len(descriptor.target_seed_ids), shuffle=False, replace=self.replace, subgraph_type=self.subgraph_type, filter_per_worker=self.filter_per_worker, neighbor_sampler=neighbor_sampler, generator=generator)
        with _PYG_SAMPLER_RNG_LOCK, torch.random.fork_rng(devices=[]):
            torch.random.set_rng_state(generator.get_state())
            iterator = iter(loader)
            batch = next(iterator)
            try:
                next(iterator)
            except StopIteration:
                pass
            else:
                raise RuntimeError(
                    "one neighbor descriptor produced multiple batches"
                )
        if isinstance(source, TypedGraphStore):
            _disk_neighbor_fields(source, batch, descriptor)
        target = descriptor.target_node_type
        assert target is not None
        _attach_neighbor_sample_counts(
            batch,
            target_node_type=target,
            seed_count=len(descriptor.target_seed_ids),
            fanout=resolved_fanout,
        )
        batch[target].supervised_mask = batch[target][f"{descriptor.phase}_mask"].clone()
        counts = {
            node_type: len(batch[node_type].n_id)
            for node_type in batch.node_types
        }
        realized = replace(
            descriptor,
            participant_counts=tuple(counts.items()),
        )
        return _attach(
            batch,
            realized,
            dict(realized.participant_counts),
        )


def _disk_neighbor_fields(store: TypedGraphStore, batch: HeteroData, descriptor: SamplingDescriptor) -> None:
    target = store._manifest["target_node_type"]
    for node_type in batch.node_types:
        ids = batch[node_type].n_id.detach().cpu().numpy()
        node = store._node(node_type)
        batch[node_type].x = _torch(store.node_array(node_type, "x", ids))
        if node["y"] is not None:
            batch[node_type].y = _torch(store.node_array(node_type, "y", ids))
        for name in node["fields"]:
            batch[node_type][name] = _torch(store.node_array(node_type, name, ids))
        if node_type == target:
            _attach_store_split_masks(
                store,
                batch[node_type],
                ids,
                descriptor.active_split_tag,
            )
    for relation in batch.edge_types:
        ids = batch[relation].e_id.detach().cpu().numpy()
        if store._relation(relation)["edge_id"] is not None:
            batch[relation].edge_id = _torch(
                store.relation_field(relation, "edge_id", ids)
            )
        for name in store._relation(relation)["fields"]:
            batch[relation][name] = _torch(store.relation_field(relation, name, ids))


def _attach_neighbor_sample_counts(
    batch: HeteroData,
    *,
    target_node_type: str,
    seed_count: int,
    fanout: Mapping[CanonicalRelation, Sequence[int]],
) -> None:
    """Restore exact hop counts omitted by the installed PyG sampler backend."""
    node_present = {
        node_type: "num_sampled_nodes" in batch[node_type]
        for node_type in batch.node_types
    }
    edge_present = {
        relation: "num_sampled_edges" in batch[relation]
        for relation in batch.edge_types
    }
    native_node_counts = all(node_present.values())
    native_edge_counts = all(edge_present.values())
    if any(node_present.values()) and not native_node_counts:
        raise RuntimeError(
            "neighbor sampler returned inconsistent per-type node hop counts"
        )
    if any(edge_present.values()) and not native_edge_counts:
        raise RuntimeError(
            "neighbor sampler returned inconsistent per-type edge hop counts"
        )
    if native_node_counts and native_edge_counts:
        return

    hop_counts = {len(values) for values in fanout.values()}
    if len(hop_counts) != 1:
        raise RuntimeError("neighbor fanout must have one exact hop count")
    num_hops = next(iter(hop_counts))
    seen = {node_type: set() for node_type in batch.node_types}
    seen[target_node_type].update(range(seed_count))
    frontier = {node_type: set(values) for node_type, values in seen.items()}
    sampled = {
        node_type: [seed_count if node_type == target_node_type else 0]
        for node_type in batch.node_types
    }
    sampled_edges = {
        relation: [0] * num_hops for relation in batch.edge_types
    }
    relation_edges = {
        relation: batch[relation].edge_index.t().tolist()
        for relation in batch.edge_types
    }

    for hop in range(num_hops):
        candidates = {node_type: set() for node_type in batch.node_types}
        for relation, per_hop in fanout.items():
            if per_hop[hop] == 0 or relation not in batch.edge_types:
                continue
            source_type, _, destination_type = relation
            destination_frontier = frontier[destination_type]
            if not destination_frontier:
                continue
            for source_index, destination_index in relation_edges[relation]:
                if destination_index in destination_frontier:
                    candidates[source_type].add(source_index)
                    sampled_edges[relation][hop] += 1

        next_frontier: dict[str, set[int]] = {}
        for node_type in batch.node_types:
            discovered = candidates[node_type].difference(seen[node_type])
            sampled[node_type].append(len(discovered))
            seen[node_type].update(discovered)
            next_frontier[node_type] = discovered
        frontier = next_frontier

    for node_type in batch.node_types:
        node_count = len(batch[node_type].n_id)
        if len(seen[node_type]) != node_count:
            raise RuntimeError(
                "neighbor hop-count reconstruction did not cover sampled "
                f"node type {node_type!r}: covered={len(seen[node_type])}, "
                f"sampled={node_count}"
            )
        if not native_node_counts:
            batch[node_type].num_sampled_nodes = sampled[node_type]
    for relation in batch.edge_types:
        edge_count = batch[relation].edge_index.shape[1]
        reconstructed = sum(sampled_edges[relation])
        if reconstructed != edge_count:
            raise RuntimeError(
                "neighbor hop-count reconstruction did not cover sampled "
                f"relation {relation!r}: covered={reconstructed}, "
                f"sampled={edge_count}"
            )
        if not native_edge_counts:
            batch[relation].num_sampled_edges = sampled_edges[relation]


@dataclass(frozen=True, slots=True)
class _CanonicalFitView:
    """One exact canonical training-row identity over an admitted source."""

    source: SamplingSource
    context: FitContext
    target_node_type: str
    target_field: str
    train_ids: np.ndarray


def _fit_digest(value: object) -> str:
    return hashlib.sha256(_json(value).encode()).hexdigest()


def _fit_ids(
    values: object,
    *,
    row_count: int,
    active_split_tag: str,
) -> np.ndarray:
    source = np.asarray(values)
    if source.ndim != 1 or source.dtype.kind not in "iu":
        raise ValueError(
            f"canonical train IDs for split {active_split_tag!r} must be integral and one-dimensional"
        )
    identifiers = np.asarray(source, dtype=np.int64)
    if len(np.unique(identifiers)) != len(identifiers):
        raise ValueError(
            f"canonical train IDs for split {active_split_tag!r} contain duplicates"
        )
    if len(identifiers) and (
        int(identifiers.min()) < 0 or int(identifiers.max()) >= row_count
    ):
        raise ValueError(
            f"canonical train IDs for split {active_split_tag!r} are outside [0, {row_count})"
        )
    return np.sort(np.array(identifiers, copy=True))


def _fit_versions() -> tuple[tuple[str, str], ...]:
    return tuple(
        sorted(
            (
                ("numpy", np.__version__),
                ("torch", str(torch.__version__)),
                ("torch-geometric", _version("torch-geometric") or "unknown"),
                ("topobench", _version("topobench") or "unknown"),
            )
        )
    )


def _store_fit_view(
    source: TypedGraphStore,
    transform: FittableTransform,
    active_split_tag: str,
) -> _CanonicalFitView:
    if (
        source.output_kind == "heterogeneous"
        and transform.spec.target_node_type is None
    ):
        raise ValueError(
            "heterogeneous fitted transform requires explicit target_node_type"
        )
    target = transform.spec.target_node_type or _target(source)
    declared_target = _target(source)
    if target != declared_target:
        raise ValueError(
            f"fitted transform target {target!r} must match supervised target {declared_target!r}"
        )
    node = source._node(target)
    field = transform.spec.target_field
    if field == "x":
        record = node["x"]
    else:
        try:
            record = node["fields"][field]
        except KeyError as error:
            raise ValueError(
                f"fitted transform field {field!r} is unavailable on target {target!r}"
            ) from error
    shape = tuple(record["shape"])
    if len(shape) != 2 or shape[0] != node["count"] or shape[1] < 1:
        raise ValueError("fitted transform input must be a two-dimensional node field")
    train = source.split_ids(active_split_tag, "train")
    identifiers = _fit_ids(
        train,
        row_count=shape[0],
        active_split_tag=active_split_tag,
    )
    train_ids_sha = _fit_digest(
        {
            "dtype": identifiers.dtype.str,
            "shape": list(identifiers.shape),
            "values": identifiers.tolist(),
        }
    )
    split = source._manifest["splits"][active_split_tag]
    train_source_sha = _fit_digest(
        {
            "feature_sha256": record["sha256"],
            "split_fingerprint": split["fingerprint"],
            "train_ids_sha256": train_ids_sha,
            "train_phase_sha256": split["phases"]["train"]["sha256"],
        }
    )
    schema_sha = _fit_digest(
        {
            "node_type": target,
            "field": field,
            "dtype": record["dtype"],
            "shape": record["shape"],
            "finite": record["finite"],
            "role": record["role"],
        }
    )
    context = FitContext(
        content_sha256=source.content_sha256,
        active_split_tag=active_split_tag,
        train_ids_sha256=train_ids_sha,
        train_source_sha256=train_source_sha,
        target_node_type=target,
        target_field=field,
        input_shape=(len(identifiers), int(shape[1])),
        input_width=int(shape[1]),
        input_dtype=np.dtype(record["dtype"]).name,
        input_schema_sha256=schema_sha,
        package_versions=_fit_versions(),
        numeric_precision=transform.spec.accumulation_dtype,
    )
    return _CanonicalFitView(source, context, target, field, identifiers)


def _materialized_graph(
    source: MaterializedSource,
) -> Data | HeteroData:
    if isinstance(source, MaterializedHomogeneousPartition):
        return source._data
    return source


def _materialized_fit_view(
    source: MaterializedSource,
    content_sha256: str,
    transform: FittableTransform,
    active_split_tag: str,
) -> _CanonicalFitView:
    graph = _materialized_graph(source)
    if isinstance(graph, HeteroData):
        if transform.spec.target_node_type is None:
            raise ValueError(
                "heterogeneous fitted transform requires explicit target_node_type"
            )
        declared_target = _target(graph)
        target = transform.spec.target_node_type or declared_target
        if target != declared_target:
            raise ValueError(
                f"fitted transform target {target!r} must match supervised target {declared_target!r}"
            )
        storage = graph[target]
        identifiers = _fit_ids(
            _phase_ids(graph, target, active_split_tag, "train"),
            row_count=int(storage.num_nodes),
            active_split_tag=active_split_tag,
        )
    else:
        target = transform.spec.target_node_type or "node"
        mask_name = f"{active_split_tag}_train_mask"
        if mask_name not in graph:
            mask_name = "train_mask"
        if mask_name not in graph:
            raise ValueError(
                f"materialized source has no training mask for split {active_split_tag!r}"
            )
        storage = graph
        mask = storage[mask_name]
        if not isinstance(mask, Tensor) or mask.dtype is not torch.bool:
            raise ValueError("materialized training mask must be a boolean tensor")
        identifiers = _fit_ids(
            mask.nonzero(as_tuple=False).reshape(-1).cpu().numpy(),
            row_count=int(graph.num_nodes),
            active_split_tag=active_split_tag,
        )
    field = transform.spec.target_field
    if field not in storage:
        raise ValueError(
            f"fitted transform field {field!r} is unavailable on target {target!r}"
        )
    feature = storage[field]
    if (
        not isinstance(feature, Tensor)
        or feature.device.type != "cpu"
        or feature.ndim != 2
        or feature.shape[0] != int(storage.num_nodes)
    ):
        raise ValueError("fitted transform input must be a two-dimensional CPU node field")
    dtype = feature.detach().numpy().dtype
    train_ids_sha = _fit_digest(
        {
            "dtype": identifiers.dtype.str,
            "shape": list(identifiers.shape),
            "values": identifiers.tolist(),
        }
    )
    train_source_sha = _fit_digest(
        {
            "content_sha256": content_sha256,
            "target_node_type": target,
            "target_field": field,
            "train_ids_sha256": train_ids_sha,
        }
    )
    schema_sha = _fit_digest(
        {
            "node_type": target,
            "field": field,
            "dtype": dtype.str,
            "shape": [int(feature.shape[0]), int(feature.shape[1])],
        }
    )
    context = FitContext(
        content_sha256=content_sha256,
        active_split_tag=active_split_tag,
        train_ids_sha256=train_ids_sha,
        train_source_sha256=train_source_sha,
        target_node_type=target,
        target_field=field,
        input_shape=(len(identifiers), int(feature.shape[1])),
        input_width=int(feature.shape[1]),
        input_dtype=dtype.name,
        input_schema_sha256=schema_sha,
        package_versions=_fit_versions(),
        numeric_precision=transform.spec.accumulation_dtype,
    )
    return _CanonicalFitView(source, context, target, field, identifiers)


def _fit_view_features(view: _CanonicalFitView, rows: np.ndarray) -> np.ndarray:
    if isinstance(view.source, TypedGraphStore):
        return view.source.node_array(view.target_node_type, view.target_field, rows)
    graph = _materialized_graph(view.source)
    storage = graph[view.target_node_type] if isinstance(graph, HeteroData) else graph
    selected = torch.from_numpy(np.array(rows, copy=True))
    return storage[view.target_field].index_select(0, selected).detach().numpy()


def _fit_view_labels(view: _CanonicalFitView, rows: np.ndarray) -> np.ndarray:
    if isinstance(view.source, TypedGraphStore):
        return view.source.node_labels(view.target_node_type, rows)
    graph = _materialized_graph(view.source)
    storage = graph[view.target_node_type] if isinstance(graph, HeteroData) else graph
    if "y" not in storage:
        raise ValueError(
            f"supervised fitted transform target {view.target_node_type!r} has no labels"
        )
    selected = torch.from_numpy(np.array(rows, copy=True))
    return storage.y.index_select(0, selected).detach().numpy()


@dataclass(frozen=True, slots=True)
class _TensorWitness:
    reference: Tensor
    version: int
    dtype: torch.dtype
    shape: tuple[int, ...]
    device: torch.device


@dataclass(frozen=True, slots=True)
class _BatchWitness:
    graph_type: type[NativeBatch]
    store_schema: tuple[tuple[tuple[str, ...], tuple[str, ...]], ...]
    protected_tensors: dict[tuple[tuple[str, ...], str], _TensorWitness]
    metadata: dict[tuple[tuple[str, ...], str], object]
    target_location: tuple[tuple[str, ...], str]
    target_tensor: _TensorWitness


def _immutable_metadata(value: object) -> object:
    if is_dataclass(value):
        return (
            "dataclass",
            type(value).__module__,
            type(value).__qualname__,
            tuple(
                (field.name, _immutable_metadata(getattr(value, field.name)))
                for field in dataclass_fields(value)
            ),
        )
    if isinstance(value, Mapping):
        items = (
            (_immutable_metadata(key), _immutable_metadata(item))
            for key, item in value.items()
        )
        return ("mapping", tuple(sorted(items, key=repr)))
    if isinstance(value, tuple):
        return ("tuple", tuple(_immutable_metadata(item) for item in value))
    if isinstance(value, list):
        return ("list", tuple(_immutable_metadata(item) for item in value))
    if isinstance(value, np.generic):
        return _immutable_metadata(value.item())
    if value is None or type(value) in {bool, int, str}:
        return (type(value).__name__, value)
    if isinstance(value, float):
        return ("float", value.hex())
    raise RuntimeError(
        f"unsupported protected batch metadata type {type(value).__name__}"
    )


def _native_stores(
    batch: NativeBatch,
) -> tuple[tuple[tuple[str, ...], Any], ...]:
    if isinstance(batch, HeteroData):
        return (
            (("global",), batch._global_store),
            *(
                (("node", node_type), batch[node_type])
                for node_type in batch.node_types
            ),
            *(
                (("edge", *relation), batch[relation])
                for relation in batch.edge_types
            ),
        )
    return ((("data",), batch),)


def _tensor_witness(value: Tensor) -> _TensorWitness:
    return _TensorWitness(
        value,
        int(value._version),
        value.dtype,
        tuple(value.shape),
        value.device,
    )


def _snapshot_batch(
    batch: NativeBatch,
    transform: FittableTransform,
) -> _BatchWitness:
    if isinstance(batch, HeteroData) and transform.spec.target_node_type is None:
        raise ValueError(
            "heterogeneous fitted transform requires explicit target_node_type"
        )
    protected: dict[tuple[tuple[str, ...], str], _TensorWitness] = {}
    metadata: dict[tuple[tuple[str, ...], str], object] = {}
    schema: list[tuple[tuple[str, ...], tuple[str, ...]]] = []
    target_store = (
        ("node", transform.spec.target_node_type)
        if isinstance(batch, HeteroData)
        else ("data",)
    )
    target_location = (target_store, transform.spec.target_field)
    target_tensor: _TensorWitness | None = None
    for store_key, storage in _native_stores(batch):
        keys = tuple(sorted(storage.keys()))
        schema.append((store_key, keys))
        for name in keys:
            value = storage[name]
            location = (store_key, name)
            if isinstance(value, Tensor):
                witness = _tensor_witness(value)
                if location == target_location:
                    target_tensor = witness
                else:
                    protected[location] = witness
            else:
                metadata[location] = _immutable_metadata(value)
    if target_tensor is None:
        raise RuntimeError(
            "fitted transform target feature tensor is absent from native batch"
        )
    return _BatchWitness(
        type(batch),
        tuple(schema),
        protected,
        metadata,
        target_location,
        target_tensor,
    )


def _validate_witness(
    witness: _BatchWitness,
    output: NativeBatch,
    transform: FittableTransform,
) -> None:
    if type(output) is not witness.graph_type:
        raise RuntimeError("fitted transform changed native graph kind")
    stores = _native_stores(output)
    schema = tuple(
        (store_key, tuple(sorted(storage.keys())))
        for store_key, storage in stores
    )
    if schema != witness.store_schema:
        raise RuntimeError("fitted transform changed native store/key schema")
    current = {
        (store_key, name): storage[name]
        for store_key, storage in stores
        for name in storage
    }
    for location, expected in witness.metadata.items():
        value = current[location]
        if isinstance(value, Tensor) or _immutable_metadata(value) != expected:
            raise RuntimeError(
                f"fitted transform changed protected metadata {location[-1]}"
            )
    for location, expected in witness.protected_tensors.items():
        value = current[location]
        if expected.reference._version != expected.version:
            raise RuntimeError(
                f"fitted transform performed in-place protected tensor mutation at {location}"
            )
        if not isinstance(value, Tensor):
            raise RuntimeError(f"fitted transform replaced protected tensor {location}")
        if (
            value is expected.reference
            or
            value.dtype != expected.dtype
            or tuple(value.shape) != expected.shape
            or value.device != expected.device
            or not torch.equal(value, expected.reference)
        ):
            raise RuntimeError(f"fitted transform changed protected tensor {location}")
    target = current[witness.target_location]
    original = witness.target_tensor
    if original.reference._version != original.version:
        raise RuntimeError("fitted transform mutated target features in place")
    if not isinstance(target, Tensor) or target is original.reference:
        raise RuntimeError("fitted transform target features alias the source batch")
    if target.device.type != transform.spec.device or target.ndim != 2:
        raise RuntimeError("fitted transform target device/rank violates declaration")
    if target.shape[0] != original.shape[0]:
        raise RuntimeError("fitted transform changed target node identity/order")
    behavior = transform.spec.feature_width_behavior
    expected_width = (
        int(behavior.removeprefix("fixed:"))
        if behavior.startswith("fixed:")
        else original.shape[1]
    )
    if target.shape[1] != expected_width:
        raise RuntimeError("fitted transform output width violates declaration")
    expected_dtype = torch.from_numpy(
        np.empty((), dtype=np.dtype(transform.spec.output_dtype))
    ).dtype
    if target.dtype != expected_dtype:
        raise RuntimeError("fitted transform output dtype violates declaration")


def _apply_fitted_transform(
    batch: NativeBatch,
    transform: FittableTransform | None,
) -> NativeBatch:
    if transform is None:
        return batch
    witness = _snapshot_batch(batch, transform)
    output = transform.transform(batch)
    if output is batch:
        raise RuntimeError("fitted transform must return a native batch clone")
    _validate_witness(witness, output, transform)
    return output


class _LazyStore:
    """One validated path with one process-local lazily reopened store."""

    def __init__(self, source: str | Path | TypedGraphStore) -> None:
        if isinstance(source, TypedGraphStore):
            self.path = source.path
            self._state: TypedGraphStoreState | None = source.state()
            self._store: TypedGraphStore | None = source
            self._pid: int | None = os.getpid()
        else:
            self.path = Path(source)
            self._state = None
            self._store = None
            self._pid = None
        self._closed = False

    def get(self) -> TypedGraphStore:
        if self._closed:
            raise RuntimeError("disk graph store owner is closed")
        process_id = os.getpid()
        if self._pid != process_id:
            if self._store is not None:
                self._store.close()
            self._store, self._pid = None, process_id
        if self._store is None:
            self._store = (
                TypedGraphStore.open(self.path)
                if self._state is None
                else TypedGraphStore.from_state(self._state)
            )
            if self._state is None:
                self._state = self._store.state()
        return self._store

    def release_store(self) -> None:
        """Drop process-local maps while retaining qualified serializable state."""
        if self._store is not None:
            self._store.close()
            self._store = None

    def close(self) -> None:
        if not self._closed:
            self._closed = True
            if self._store is not None:
                self._store.close()
                self._store = None

    def __getstate__(self) -> dict[str, object]:
        return {
            "path": self.path,
            "_state": self._state,
            "_store": None,
            "_pid": None,
            "_closed": self._closed,
        }

    def __del__(self) -> None:
        self.close()


def _sequence_partition_identity(
    source: SamplingSource,
    strategy: GraphSamplingStrategy,
) -> str:
    """Return the content identity of the exact partition/sampling domain."""
    if isinstance(source, TypedGraphStore):
        return source.partition_book_identity
    declared = getattr(source, "partition_book_identity", None)
    if isinstance(declared, str):
        return _require_sha256(declared, "partition_book_identity")
    book = getattr(strategy, "partition_book", None)
    if isinstance(book, TypedPartitionBook):
        return _require_sha256(
            book.content_identity,
            "partition_book_identity",
        )
    if isinstance(source, MaterializedHomogeneousPartition):
        digest = hashlib.sha256()
        _tensor_frame(digest, "partition.node_perm", source.partition.node_perm)
        _tensor_frame(digest, "partition.partptr", source.partition.partptr)
        return digest.hexdigest()
    content = getattr(source, "content_sha256", None)
    if content is None:
        admission = getattr(strategy, "_admission", None)
        content = getattr(admission, "content_sha256", None)
    _require_sha256(content, "content_sha256")
    digest = hashlib.sha256()
    _frame(digest, "unpartitioned_graph", content.encode())
    return digest.hexdigest()


class _TrainSequenceSampler(Sampler[tuple[int, int]]):
    """Assign global sequence IDs in the parent while workers see immutable work."""

    def __init__(self, owner: DiskGraphDataModule) -> None:
        self.owner = owner

    def __iter__(self):
        state = self.owner.sequence_state
        start = (state.next_issue_id - 1) % state.descriptor_count
        for descriptor_index in range(start, state.descriptor_count):
            descriptor = self.owner.descriptors("train")[descriptor_index]
            sequence_id = self.owner.issue_descriptor(descriptor)
            yield sequence_id, descriptor_index

    def __len__(self) -> int:
        state = self.owner.sequence_state
        start = (state.next_issue_id - 1) % state.descriptor_count
        return state.descriptor_count - start


def _descriptor_record_count(descriptor: SamplingDescriptor) -> int | None:
    if descriptor.participant_counts:
        return sum(count for _, count in descriptor.participant_counts)
    if descriptor.target_seed_ids:
        return len(descriptor.target_seed_ids)
    return None



_WORKER_EVENT_ATTRIBUTE = "_execution_event_envelopes"
_MAX_WORKER_EVENTS_PER_ITEM = 3


@dataclass(frozen=True, slots=True)
class _WorkerEventEnvelope:
    operation: str
    phase: str
    split: str
    descriptor_sequence: int | None
    descriptor_digest: str
    duration_ns: int

    row_count: int | None
    unique_storage_bytes: int | None

def _drain_worker_events(
    batch: NativeBatch,
    monitor: InputMonitor | None,
) -> None:
    if _WORKER_EVENT_ATTRIBUTE not in batch:
        return
    raw = batch[_WORKER_EVENT_ATTRIBUTE]
    del batch[_WORKER_EVENT_ATTRIBUTE]
    if (
        not isinstance(raw, tuple)
        or len(raw) > _MAX_WORKER_EVENTS_PER_ITEM
        or any(not isinstance(item, _WorkerEventEnvelope) for item in raw)
    ):
        raise RuntimeError("worker execution event envelope is invalid")
    if monitor is None:
        return
    for item in raw:
        monitor.record(
            item.operation,
            phase=item.phase,
            split=item.split,
            descriptor_sequence=item.descriptor_sequence,
            descriptor_digest_value=item.descriptor_digest,
            duration_ns=item.duration_ns,
            evidence={"producer": "disk_graph_worker"},
            row_count=item.row_count,
            unique_storage_bytes=item.unique_storage_bytes,
        )


class _ExecutionEventLoader(DataLoader[NativeBatch]):
    """Drain bounded worker evidence in parent delivery order."""

    def __init__(
        self,
        owner: DiskGraphDataModule | object,
        dataset: _DescriptorDataset,
        **kwargs: object,
    ) -> None:
        self.event_owner = owner
        super().__init__(
            dataset,
            batch_size=None,
            shuffle=False,
            collate_fn=_identity,
            **kwargs,
        )

    def __iter__(self):
        for batch in super().__iter__():
            _drain_worker_events(
                batch,
                getattr(self.event_owner, "execution_monitor", None),
            )
            yield batch



class _TrainSequenceLoader(DataLoader[NativeBatch]):
    """Observe parent-side preparation/delivery as ordered batches arrive."""

    def __init__(
        self,
        owner: DiskGraphDataModule,
        dataset: _DescriptorDataset,
        defer_delivery: bool = False,
        **kwargs: object,
    ) -> None:
        self.sequence_owner = owner
        self.defer_delivery = _boolean(defer_delivery, "defer_delivery")
        super().__init__(
            dataset,
            batch_size=None,
            sampler=_TrainSequenceSampler(owner),
            shuffle=False,
            collate_fn=_identity,
            **kwargs,
        )

    def __iter__(self):
        self.sequence_owner._require_commit_callback()
        for batch in super().__iter__():
            _drain_worker_events(batch, self.sequence_owner.execution_monitor)
            sequence_id = _integer(
                getattr(batch, "sequence_id", None),
                "batch.sequence_id",
                minimum=1,
            )
            descriptor = self.sequence_owner.sequence_state.descriptor_for(
                sequence_id
            )
            self.sequence_owner.prepare_sequence(sequence_id, descriptor)
            if not self.defer_delivery:
                self.sequence_owner.deliver_sequence(sequence_id)
            monitor = self.sequence_owner.execution_monitor
            if monitor is not None and self.sequence_owner.prefetch_limits is None:
                prehashed = getattr(
                    batch,
                    "execution_descriptor_digest",
                    None,
                )
                if prehashed is None:
                    prehashed = descriptor_digest(descriptor)
                    batch.execution_descriptor_digest = prehashed
                monitor.mark_batch_ready(
                    phase="fit",
                    split="train",
                    descriptor_sequence=sequence_id,
                    descriptor_digest_value=prehashed,
                )
            yield batch


class _DescriptorDataset(Dataset[NativeBatch]):
    def __init__(
        self,
        source: SamplingSource | _LazyStore,
        strategy: GraphSamplingStrategy,
        descriptors: tuple[SamplingDescriptor, ...],
        fitted_transform: FittableTransform | None,
        phase: Phase,
        execution_monitor: InputMonitor | None,
        capture_worker_events: bool = False,
    ) -> None:
        self.source = source
        self.strategy = strategy
        self.descriptor_values = descriptors
        self.fitted_transform = fitted_transform
        self.phase = phase
        self.execution_monitor = execution_monitor
        self.capture_worker_events = _boolean(
            capture_worker_events,
            "capture_worker_events",
        )

    def __len__(self) -> int:
        return len(self.descriptor_values)

    def __getitem__(self, index: int | tuple[int, int]) -> NativeBatch:
        sequence_id: int | None
        if isinstance(index, tuple):
            sequence_id, descriptor_index = index
        else:
            sequence_id, descriptor_index = None, index
        source = (
            self.source.get()
            if isinstance(self.source, _LazyStore)
            else self.source
        )
        descriptor = self.descriptor_values[descriptor_index]
        record_count = _descriptor_record_count(descriptor)
        monitor = self.execution_monitor
        lifecycle_phase = (
            "fit" if self.phase in {"train", "val"} else "test"
        )
        worker_digest = (
            descriptor_digest(descriptor)
            if self.capture_worker_events
            else None
        )
        worker_events: list[_WorkerEventEnvelope] = []
        read_started_ns = (
            time.monotonic_ns() if self.capture_worker_events else None
        )
        read_token = (
            None
            if monitor is None
            else monitor.begin(
                ExecutionOperation.SELECTED_READ,
                phase=lifecycle_phase,
                split=self.phase,
                descriptor_sequence=sequence_id,
                descriptor_identity=descriptor,
            )
        )
        try:
            batch = self.strategy.materialize(source, descriptor)
        except BaseException:
            if read_token is not None:
                with suppress(Exception):
                    monitor.finish(
                        read_token,
                        status=ExecutionStatus.ERROR,
                        evidence={"failure_stage": "materialize"},
                    )
            raise
        if worker_digest is not None and read_started_ns is not None:
            batch.execution_descriptor_digest = worker_digest
            worker_duration_ns = max(
                0,
                time.monotonic_ns() - read_started_ns,
            )
            worker_events.extend(
                (
                    _WorkerEventEnvelope(
                        operation=ExecutionOperation.SELECTED_READ.value,
                        phase=lifecycle_phase,
                        split=self.phase,
                        descriptor_sequence=sequence_id,
                        descriptor_digest=worker_digest,
                        duration_ns=worker_duration_ns,
                        row_count=record_count,
                        unique_storage_bytes=None,
                    ),
                    _WorkerEventEnvelope(
                        operation=ExecutionOperation.NATIVE_ASSEMBLY.value,
                        phase=lifecycle_phase,
                        split=self.phase,
                        descriptor_sequence=sequence_id,
                        descriptor_digest=worker_digest,
                        duration_ns=worker_duration_ns,
                        row_count=record_count,
                        unique_storage_bytes=None,
                    ),
                )
            )
        if read_token is not None:
            batch.execution_descriptor_digest = read_token.descriptor_digest
            read_event = monitor.finish(
                read_token,
                row_count=record_count,
            )
            monitor.record(
                ExecutionOperation.NATIVE_ASSEMBLY,
                phase=lifecycle_phase,
                split=self.phase,
                descriptor_sequence=sequence_id,
                descriptor_digest_value=read_token.descriptor_digest,
                duration_ns=(
                    0 if read_event is None else read_event.duration_ns
                ),
                row_count=record_count,
            )
        transform_token = (
            None
            if monitor is None or self.fitted_transform is None
            else monitor.begin(
                ExecutionOperation.FITTED_TRANSFORM,
                phase="transform_apply",
                split=self.phase,
                descriptor_sequence=sequence_id,
                descriptor_digest_value=getattr(
                    batch,
                    "execution_descriptor_digest",
                    None,
                ),
            )
        )
        transform_started_ns = (
            time.monotonic_ns()
            if self.capture_worker_events and self.fitted_transform is not None
            else None
        )
        try:
            batch = _apply_fitted_transform(batch, self.fitted_transform)
        except BaseException:
            if transform_token is not None:
                with suppress(Exception):
                    monitor.finish(
                        transform_token,
                        status=ExecutionStatus.ERROR,
                        evidence={"failure_stage": "transform_apply"},
                    )
            raise
        if transform_token is not None:
            monitor.finish(transform_token)
        if (
            worker_digest is not None
            and transform_started_ns is not None
        ):
            batch.execution_descriptor_digest = worker_digest
            worker_events.append(
                _WorkerEventEnvelope(
                    operation=ExecutionOperation.FITTED_TRANSFORM.value,
                    phase="transform_apply",
                    split=self.phase,
                    descriptor_sequence=sequence_id,
                    descriptor_digest=worker_digest,
                    duration_ns=max(
                        0,
                        time.monotonic_ns() - transform_started_ns,
                    ),
                    row_count=record_count,
                    unique_storage_bytes=None,
                )
            )
        if sequence_id is not None:
            batch.sequence_id = sequence_id
            if isinstance(self.source, _LazyStore):
                batch.sampling_descriptor = descriptor
        if worker_events:
            if len(worker_events) > _MAX_WORKER_EVENTS_PER_ITEM:
                raise RuntimeError("worker execution event envelope bound exceeded")
            batch[_WORKER_EVENT_ATTRIBUTE] = tuple(worker_events)
        return batch


def _identity(value: NativeBatch) -> NativeBatch:
    return value


def _new_probe_transform(
    transform: FittableTransform | None,
) -> FittableTransform | None:
    """Construct an unfitted transform with the same canonical configuration."""
    if transform is None:
        return None
    config = dict(transform.canonical_config())
    config.pop("variance_edge_convention", None)
    signature = inspect.signature(type(transform))
    explicit = {
        name: value
        for name, value in config.items()
        if name in signature.parameters
        and signature.parameters[name].kind
        not in {
            inspect.Parameter.VAR_POSITIONAL,
            inspect.Parameter.VAR_KEYWORD,
        }
    }
    attempts = (explicit, config, {})
    last_error: TypeError | None = None
    clone: object | None = None
    for kwargs in attempts:
        try:
            clone = type(transform)(**kwargs)
            break
        except TypeError as error:
            last_error = error
    if not isinstance(clone, FittableTransform):
        raise TypeError(
            "fitted transform cannot construct an isolated probe instance"
        ) from last_error
    if clone.status is not FitStatus.UNFITTED:
        raise RuntimeError("isolated fitted transform must start unfitted")
    return clone


class DiskGraphDataModule(LightningDataModule):
    """Own one validated source and one strategy across all phase loaders."""

    def __init__(
        self,
        source: str | Path | SamplingSource,
        strategy: GraphSamplingStrategy,
        *,
        active_split_tag: str | None = None,
        num_workers: int = 0,
        persistent_workers: bool = False,
        train_shuffle: bool = True,
        fitted_transform: FittableTransform | None = None,
        fitted_state_root: str | Path | None = None,
        supervised_fit: bool = False,
        prefetch_limits: PrefetchLimits | None = None,
        prefetch_device: object = "cpu",
        execution_monitor: InputMonitor | None = None,
    ) -> None:
        super().__init__()
        if execution_monitor is not None and not isinstance(
            execution_monitor,
            InputMonitor,
        ):
            raise TypeError("execution_monitor must be InputMonitor or None")
        self.execution_monitor = execution_monitor
        if not isinstance(strategy, GraphSamplingStrategy):
            raise TypeError("strategy must implement GraphSamplingStrategy")
        if fitted_transform is not None and not isinstance(
            fitted_transform,
            FittableTransform,
        ):
            raise TypeError("fitted_transform must implement FittableTransform")
        if fitted_transform is None:
            if fitted_state_root is not None or supervised_fit:
                raise ValueError(
                    "fitted state location/capability requires fitted_transform"
                )
        elif fitted_state_root is None:
            raise ValueError("fitted_transform requires fitted_state_root")
        self.strategy = strategy
        self.fitted_transform = fitted_transform
        self.fitted_state_root = (
            None if fitted_state_root is None else Path(fitted_state_root)
        )
        self.supervised_fit = _boolean(supervised_fit, "supervised_fit")
        if fitted_transform is not None:
            _integer(
                getattr(fitted_transform, "max_batch_rows", None),
                "fitted_transform.max_batch_rows",
                minimum=1,
            )
            _integer(
                getattr(fitted_transform, "max_batch_bytes", None),
                "fitted_transform.max_batch_bytes",
                minimum=1,
            )
        self.num_workers = _integer(num_workers, "num_workers", minimum=0)
        self.persistent_workers = _boolean(
            persistent_workers,
            "persistent_workers",
        )
        if self.persistent_workers and not self.num_workers:
            raise ValueError(
                "persistent_workers requires num_workers greater than zero"
            )
        self.train_shuffle = _boolean(train_shuffle, "train_shuffle")
        if prefetch_limits is not None and not isinstance(
            prefetch_limits,
            PrefetchLimits,
        ):
            raise TypeError("prefetch_limits must be PrefetchLimits")
        self.prefetch_limits = prefetch_limits
        self._prefetch_capability = (
            None
            if prefetch_limits is None
            else PrefetchCapability.detect(prefetch_device)
        )
        self._fit_materialized: MaterializedSource | None = None
        self._fit_materialized_sha256: str | None = None
        if isinstance(source, (str, Path, TypedGraphStore)):
            self._owner: _LazyStore | None = _LazyStore(source)
            self._source: SamplingSource | _LazyStore = self._owner
            reference = self._owner.get()
            declared = _declared_tag(reference)
            if active_split_tag is None:
                assert declared is not None
                self.active_split_tag = declared
            else:
                self.active_split_tag = _active_tag(
                    reference,
                    active_split_tag,
                )
            self.strategy.validate_capabilities(reference)
        elif isinstance(source, (MaterializedHomogeneousPartition, HeteroData)):
            self._owner = None
            self._source = source
            declared = _declared_tag(source)
            if active_split_tag is None:
                if declared is None:
                    raise ValueError(
                        "materialized source requires an active_split_tag"
                    )
                self.active_split_tag = declared
            else:
                self.active_split_tag = _active_tag(source, active_split_tag)
            self.strategy.validate_capabilities(source)
            admission = getattr(self.strategy, "_admission", None)
            if not isinstance(admission, _MaterializedAdmission):
                raise RuntimeError(
                    "sampling strategy did not retain its admitted Task8 snapshot"
                )
            self._fit_materialized = admission.source
            self._fit_materialized_sha256 = admission.content_sha256
        else:
            raise TypeError(
                "source must be a graph reference, store, or store path"
            )
        self._descriptors: dict[Phase, tuple[SamplingDescriptor, ...]] = {}
        self._loaders: dict[
            Phase,
            DataLoader[NativeBatch] | DevicePrefetchLoader,
        ] = {}
        reference = (
            self._source
            if self._owner is None
            else self._owner.get()
        )
        self._partition_book_identity = _sequence_partition_identity(
            reference,
            self.strategy,
        )
        if self._owner is not None and not isinstance(source, TypedGraphStore):
            self._owner.release_store()
        self._sequence_state: SequenceState | None = None
        self._closed = False

    @property
    def closed(self) -> bool:
        return self._closed
    def set_execution_monitor(
        self,
        execution_monitor: InputMonitor | None,
    ) -> None:
        """Attach or detach the callback-owned optional monitor."""
        if execution_monitor is not None and not isinstance(
            execution_monitor,
            InputMonitor,
        ):
            raise TypeError("execution_monitor must be InputMonitor or None")
        self.execution_monitor = execution_monitor
        for loader in self._loaders.values():
            if isinstance(loader, DevicePrefetchLoader):
                loader.execution_monitor = execution_monitor
                base = loader.source
            else:
                base = loader
            dataset = getattr(base, "dataset", None)
            if isinstance(dataset, _DescriptorDataset):
                dataset.execution_monitor = (
                    execution_monitor if self.num_workers == 0 else None
                )
                dataset.capture_worker_events = (
                    execution_monitor is not None and self.num_workers > 0
                )


    def _materialized_reference(self) -> SamplingSource:
        if self._closed:
            raise RuntimeError("DiskGraphDataModule is closed")
        if self._owner is not None:
            raise RuntimeError("disk descriptors require a temporary store")
        return self._source  # type: ignore[return-value]

    def _fit_view(self, source: SamplingSource) -> _CanonicalFitView:
        assert self.fitted_transform is not None
        if isinstance(source, TypedGraphStore):
            return _store_fit_view(
                source,
                self.fitted_transform,
                self.active_split_tag,
            )
        if self._fit_materialized is None or self._fit_materialized_sha256 is None:
            raise RuntimeError("admitted materialized fit snapshot is unavailable")
        return _materialized_fit_view(
            self._fit_materialized,
            self._fit_materialized_sha256,
            self.fitted_transform,
            self.active_split_tag,
        )

    def _fit_missing_state(self, view: _CanonicalFitView) -> None:
        assert self.fitted_transform is not None
        assert self.fitted_state_root is not None
        transform = self.fitted_transform
        if not len(view.train_ids):
            raise ValueError(
                f"fitted transform has empty canonical train input for split {self.active_split_tag!r}"
            )
        if transform.spec.accesses_labels and not self.supervised_fit:
            raise PermissionError(
                "label-consuming fitted transform requires supervised_fit capability"
            )
        max_rows = _integer(
            transform.max_batch_rows,
            "fitted_transform.max_batch_rows",
            minimum=1,
        )
        max_bytes = _integer(
            transform.max_batch_bytes,
            "fitted_transform.max_batch_bytes",
            minimum=1,
        )
        bytes_per_row = view.context.input_width * max(
            np.dtype(view.context.input_dtype).itemsize,
            np.dtype(transform.spec.accumulation_dtype).itemsize,
        )
        chunk_rows = min(max_rows, max_bytes // bytes_per_row)
        if chunk_rows < 1:
            raise ValueError(
                "fitted transform byte bound cannot hold one canonical input row"
            )
        monitor = self.execution_monitor
        token = (
            None
            if monitor is None
            else monitor.begin(
                ExecutionOperation.FITTED_TRANSFORM,
                phase="transform_fit",
                split="train",
                descriptor_identity=(
                    view.context.content_sha256,
                    view.context.active_split_tag,
                    type(transform).__qualname__,
                ),
            )
        )
        try:
            transform.begin_fit(view.context)
            for start in range(0, len(view.train_ids), chunk_rows):
                rows = view.train_ids[start : start + chunk_rows]
                features = _fit_view_features(view, rows)
                labels = (
                    _fit_view_labels(view, rows)
                    if transform.spec.accesses_labels
                    else None
                )
                transform.update_fit(features, labels)
            transform.finalize_fit(self.fitted_state_root)
        except BaseException:
            if token is not None:
                with suppress(Exception):
                    monitor.finish(
                        token,
                        status=ExecutionStatus.ERROR,
                        row_count=len(view.train_ids),
                        evidence={"failure_stage": "transform_fit"},
                    )
            raise
        if token is not None:
            monitor.finish(token, row_count=len(view.train_ids))

    def _ensure_fitted(self, *, allow_fit: bool) -> None:
        transform = self.fitted_transform
        if transform is None:
            return
        if self._closed:
            raise RuntimeError("DiskGraphDataModule is closed")
        assert self.fitted_state_root is not None

        def ensure_view(view: _CanonicalFitView) -> None:
            expected_key = build_fit_state_key(view.context, transform)

            def validate_binding() -> None:
                actual_key = transform.state_key
                if transform.status is not FitStatus.FITTED or actual_key != expected_key:
                    raise FitStateError(
                        "fitted transform context identity mismatch: "
                        f"state={actual_key!r}, expected={expected_key!r}, "
                        f"content={view.context.content_sha256!r}, "
                        f"split={view.context.active_split_tag!r}"
                    )

            if transform.status is FitStatus.FITTED:
                validate_binding()
                return
            if transform.status is not FitStatus.UNFITTED:
                raise RuntimeError(
                    f"fitted transform is not reusable ({transform.status.value})"
                )
            try:
                transform.load_state(self.fitted_state_root, view.context)
            except FitStateNotFoundError as error:
                if not allow_fit:
                    raise FitStateError(
                        "validation/test setup requires an existing exact fitted state"
                    ) from error
                self._fit_missing_state(view)
            validate_binding()

        if self._owner is None:
            assert self._fit_materialized is not None
            ensure_view(self._fit_view(self._fit_materialized))
            return
        ensure_view(self._fit_view(self._owner.get()))

    def _setup_phase(self, phase: Phase) -> None:
        """Create one phase without evaluating unrelated split masks."""
        if phase in self._descriptors:
            return
        monitor = self.execution_monitor
        token = (
            None
            if monitor is None
            else monitor.begin(
                ExecutionOperation.PARTITION,
                phase="descriptor_setup",
                split=phase,
                descriptor_identity=(
                    self._partition_book_identity,
                    self.active_split_tag,
                    type(self.strategy).__qualname__,
                ),
            )
        )
        try:
            if self._owner is None:
                source = self._materialized_reference()
                descriptors = self.strategy.setup(
                    source,
                    phase=phase,
                    active_split_tag=self.active_split_tag,
                    shuffle=phase == "train" and self.train_shuffle,
                )
            else:
                descriptors = self.strategy.setup(
                    self._owner.get(),
                    phase=phase,
                    active_split_tag=self.active_split_tag,
                    shuffle=phase == "train" and self.train_shuffle,
                )
            self._descriptors[phase] = descriptors
            if phase == "train":
                self._ensure_sequence_state()
        except ValueError as error:
            if token is not None:
                with suppress(Exception):
                    monitor.finish(
                        token,
                        status=ExecutionStatus.ERROR,
                        evidence={"failure_stage": "descriptor_setup"},
                    )
            raise ValueError(f"{phase} phase setup failed: {error}") from error
        except BaseException:
            if token is not None:
                with suppress(Exception):
                    monitor.finish(
                        token,
                        status=ExecutionStatus.ERROR,
                        evidence={"failure_stage": "descriptor_setup"},
                    )
            raise
        if token is not None:
            monitor.finish(token, row_count=len(descriptors))

    def _ensure_sequence_state(self) -> SequenceState:
        if self._sequence_state is None:
            descriptors = self._descriptors.get("train")
            if descriptors is None:
                raise RuntimeError(
                    "train descriptors must exist before sequence state"
                )
            fitted_state_key = (
                None
                if self.fitted_transform is None
                else self.fitted_transform.state_key
            )
            identity = SequenceIdentity.from_descriptors(
                descriptors,
                partition_book_identity=self._partition_book_identity,
                fitted_transform_state_key=fitted_state_key,
                sampler_state=self.strategy.sampler_state(),
            )
            self._sequence_state = SequenceState(identity, descriptors)
        return self._sequence_state

    @property
    def sequence_state(self) -> SequenceState:
        """Return the parent-owned training sequence state."""
        self.descriptors("train")
        return self._ensure_sequence_state()

    def issue_descriptor(self, descriptor: SamplingDescriptor) -> int:
        """Issue the next exact immutable training descriptor."""
        return self.sequence_state.issue(descriptor)

    def prepare_sequence(
        self,
        sequence_id: int,
        descriptor: SamplingDescriptor,
    ) -> None:
        """Record one materialized descriptor without relaxing delivery order."""
        self.sequence_state.prepare(sequence_id, descriptor)

    def deliver_sequence(self, sequence_id: int) -> SamplingDescriptor:
        """Deliver a prepared training descriptor in exact sequence order."""
        descriptor = self.sequence_state.deliver(sequence_id)
        assert isinstance(descriptor, SamplingDescriptor)
        return descriptor

    def consume_sequence(self, sequence_id: int) -> None:
        """Append one delivered training sequence to the pending step group."""
        self.sequence_state.consume(sequence_id)

    def commit_optimizer_step(
        self,
        *,
        optimizer_succeeded: bool,
        model_global_step: int,
        evaluator_snapshot: Mapping[str, object],
        epoch: int,
    ) -> bool:
        """Atomically commit sampler/evaluator/global-step participants."""
        expected = {"sequence_id", "count", "state"}
        if set(evaluator_snapshot) != expected:
            raise ValueError(
                "evaluator snapshot keys must be exactly "
                f"{sorted(expected)!r}"
            )
        return self.sequence_state.commit(
            optimizer_succeeded=optimizer_succeeded,
            model_global_step=model_global_step,
            evaluator_sequence=evaluator_snapshot["sequence_id"],
            evaluator_count=evaluator_snapshot["count"],
            evaluator_state=evaluator_snapshot["state"],
            epoch=epoch,
        )

    def state_dict(self) -> dict[str, object]:
        """Serialize only the committed training sequence boundary."""
        return self.sequence_state.state_dict()

    def load_state_dict(self, state_dict: Mapping[str, object]) -> None:
        """Restore one exact committed boundary and discard transient work."""
        self.sequence_state.load_state_dict(state_dict)
        self._shutdown_loaders()

    def setup(self, stage: str | None = None) -> None:
        """Set up exactly the phases required by one Lightning stage."""
        phases_by_stage: dict[str | None, tuple[Phase, ...]] = {
            None: _PHASES,
            "fit": ("train", "val"),
            "validate": ("val",),
            "test": ("test",),
            "predict": ("test",),
        }
        try:
            phases = phases_by_stage[stage]
        except (KeyError, TypeError) as error:
            raise ValueError(f"unsupported setup stage: {stage!r}") from error
        self._ensure_fitted(allow_fit=stage in {None, "fit"})
        for phase in phases:
            self._setup_phase(phase)

    def descriptors(self, phase: Phase) -> tuple[SamplingDescriptor, ...]:
        """Return one phase's descriptors, setting up only that phase."""
        phase = _phase(phase)
        if self._closed:
            raise RuntimeError("DiskGraphDataModule is closed")
        self._ensure_fitted(allow_fit=phase == "train")
        self._setup_phase(phase)
        return self._descriptors[phase]

    def _require_commit_callback(self) -> None:
        """Reject attached training that cannot durably commit sequence state."""
        trainer = self.trainer
        if trainer is None:
            return
        from topobench.callbacks.dataloader_commit import (
            DataloaderCommitCallback,
        )

        count = sum(
            isinstance(callback, DataloaderCommitCallback)
            for callback in trainer.callbacks
        )
        if count != 1:
            raise RuntimeError(
                "DiskGraphDataModule training requires exactly one "
                "DataloaderCommitCallback; "
                f"found {count}"
            )


    def _deliver_prefetched(self, batch: NativeBatch) -> None:
        sequence_id = _integer(
            getattr(batch, "sequence_id", None),
            "batch.sequence_id",
            minimum=1,
        )
        self.deliver_sequence(sequence_id)

    def _abort_prefetched_training(self) -> None:
        durable = self.sequence_state.state_dict()
        self.sequence_state.load_state_dict(durable)

    def _loader(
        self,
        phase: Phase,
    ) -> DataLoader[NativeBatch] | DevicePrefetchLoader:
        if phase == "train":
            self._require_commit_callback()
        if self._closed:
            raise RuntimeError("DiskGraphDataModule is closed")
        if phase not in self._loaders:
            dataset = _DescriptorDataset(
                self._source,
                self.strategy,
                self.descriptors(phase),
                self.fitted_transform,
                phase,
                self.execution_monitor if self.num_workers == 0 else None,
                self.num_workers > 0 and self.execution_monitor is not None,
            )
            options = loader_worker_options(
                num_workers=self.num_workers,
                pin_memory=False,
                persistent_workers=self.persistent_workers,
            )
            base: DataLoader[NativeBatch] = (
                _TrainSequenceLoader(
                    self,
                    dataset,
                    defer_delivery=self.prefetch_limits is not None,
                    **options,
                )
                if phase == "train"
                else _ExecutionEventLoader(
                    self,
                    dataset,
                    **options,
                )
            )
            if self.prefetch_limits is None:
                self._loaders[phase] = base
            else:
                assert self._prefetch_capability is not None
                self._loaders[phase] = DevicePrefetchLoader(
                    base,
                    self.prefetch_limits,
                    capability=self._prefetch_capability,
                    owns_source=True,
                    on_yield=(
                        self._deliver_prefetched
                        if phase == "train"
                        else None
                    ),
                    on_abort=(
                        self._abort_prefetched_training
                        if phase == "train"
                        else None
                    ),
                    execution_monitor=self.execution_monitor,
                    monitor_phase=(
                        "fit" if phase in {"train", "val"} else "test"
                    ),
                    monitor_split=phase,
                )
        return self._loaders[phase]

    @contextmanager
    def noncommitting_probe_batches(
        self,
        phases: Sequence[str],
    ) -> Iterator[dict[str, NativeBatch]]:
        """Materialize representative batches through a disposable runtime."""
        requested = tuple(_phase(phase) for phase in phases)
        if not requested:
            raise ValueError("probe phases must not be empty")
        source: str | Path | SamplingSource
        if self._owner is None:
            source = _clone_materialized(self._materialized_reference())
        else:
            source = self._owner.path
        strategy = copy.deepcopy(self.strategy)
        if hasattr(strategy, "_admission"):
            strategy._admission = None
        transform = _new_probe_transform(self.fitted_transform)
        allow_fit = "train" in requested
        iterators: list[Iterator[NativeBatch]] = []
        probe: DiskGraphDataModule | None = None
        with tempfile.TemporaryDirectory(prefix="topobench-preflight-fit-") as root:
            try:
                probe = DiskGraphDataModule(
                    source,
                    strategy,
                    active_split_tag=self.active_split_tag,
                    num_workers=0,
                    persistent_workers=False,
                    train_shuffle=self.train_shuffle,
                    fitted_transform=transform,
                    fitted_state_root=(
                        (
                            Path(root) / "fitted"
                            if allow_fit
                            else self.fitted_state_root
                        )
                        if transform is not None
                        else None
                    ),
                    supervised_fit=self.supervised_fit,
                    prefetch_limits=None,
                    execution_monitor=None,
                )
                if transform is not None:
                    probe._ensure_fitted(allow_fit=allow_fit)
                batches: dict[str, NativeBatch] = {}
                for phase in requested:
                    loader = getattr(probe, f"{phase}_dataloader")()
                    iterator = iter(loader)
                    iterators.append(iterator)
                    try:
                        batches[phase] = next(iterator)
                    except StopIteration as error:
                        raise ValueError(
                            f"{phase} dataloader has no representative batch"
                        ) from error
                yield batches
            finally:
                for iterator in iterators:
                    close = getattr(iterator, "close", None)
                    if callable(close):
                        close()
                if probe is not None:
                    probe.close()

    def train_dataloader(self) -> DataLoader[NativeBatch] | DevicePrefetchLoader:
        return self._loader("train")

    def val_dataloader(self) -> DataLoader[NativeBatch] | DevicePrefetchLoader:
        return self._loader("val")

    def test_dataloader(self) -> DataLoader[NativeBatch] | DevicePrefetchLoader:
        return self._loader("test")

    def predict_dataloader(
        self,
    ) -> DataLoader[NativeBatch] | DevicePrefetchLoader:
        """Use the explicitly supported test split for prediction identity."""
        return self._loader("test")

    def _shutdown_loaders(self) -> None:
        for loader in self._loaders.values():
            close = getattr(loader, "close", None)
            if callable(close):
                close()
                continue
            iterator = getattr(loader, "_iterator", None)
            shutdown = getattr(iterator, "_shutdown_workers", None)
            if callable(shutdown):
                shutdown()
            if iterator is not None:
                loader._iterator = None
        self._loaders.clear()

    def close(self) -> None:
        if self._closed:
            return
        self._shutdown_loaders()
        if self._owner is not None:
            self._owner.close()
        self._closed = True

    def teardown(self, stage: str | None = None) -> None:
        """Release transient loaders/maps while keeping the module reusable."""
        if self._closed:
            return
        self._shutdown_loaders()
        if self._owner is not None:
            self._owner.release_store()


__all__ = [
    "DiskGraphDataModule",
    "GraphSamplingStrategy",
    "HeterogeneousClusterStrategy",
    "HeterogeneousNeighborStrategy",
    "HomogeneousClusterStrategy",
    "SamplingCapabilityError",
    "SamplingDescriptor",
]
