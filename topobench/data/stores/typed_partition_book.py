"""Immutable typed partition assignments, statistics, and qualification evidence."""
from __future__ import annotations

import hashlib
import json
from dataclasses import dataclass, field
from types import MappingProxyType
from typing import Any, Mapping, Sequence

import numpy as np

CanonicalRelation = tuple[str, str, str]


def _error(code: str, detail: str) -> Exception:
    from topobench.data.stores.typed_graph_ingestion import ArtifactValidationError
    return ArtifactValidationError(f"{code}: {detail}")


def _jsonable(value: Any) -> Any:
    if isinstance(value, Mapping):
        return {str(key): _jsonable(item) for key, item in value.items()}
    if isinstance(value, (list, tuple)):
        return [_jsonable(item) for item in value]
    if isinstance(value, np.generic):
        return value.item()
    return value


def _freeze(value: Any) -> Any:
    if isinstance(value, Mapping):
        return MappingProxyType({str(key): _freeze(item) for key, item in value.items()})
    if isinstance(value, (list, tuple)):
        return tuple(_freeze(item) for item in value)
    if isinstance(value, np.generic):
        return value.item()
    if value is None or isinstance(value, (str, int, float, bool)):
        return value
    raise TypeError(f"non-JSON evidence value {type(value).__name__}")


def _hash_json(value: Any) -> str:
    data = json.dumps(_jsonable(value), sort_keys=True, separators=(",", ":"), ensure_ascii=False).encode()
    return hashlib.sha256(data).hexdigest()


def _relation(
    value: Any,
    *,
    code: str = "PARTITION-ID-001",
) -> CanonicalRelation:
    if (
        not isinstance(value, tuple)
        or len(value) != 3
        or any(
            not isinstance(item, str) or not item
            for item in value
        )
    ):
        raise _error(code, "relation keys must be canonical triples")
    return value


def _limit_map(
    value: Mapping[Any, Any],
    role: str,
    *,
    code: str,
) -> Mapping[Any, int]:
    if not isinstance(value, Mapping):
        raise _error(code, f"{role} must be a mapping")
    result: dict[Any, int] = {}
    for key, limit in value.items():
        if (
            isinstance(limit, bool)
            or not isinstance(limit, int)
            or limit < 0
        ):
            raise _error(
                code,
                f"{role}[{key!r}] must be a non-negative integer",
            )
        result[key] = limit
    return MappingProxyType(result)


@dataclass(frozen=True, slots=True)
class PartitionQualificationLimits:
    """Optional absolute bounds; omitted limits are permissive."""
    max_nodes_per_type: Mapping[str, int] = field(
        default_factory=lambda: MappingProxyType({})
    )
    max_phase_nodes: Mapping[str, Mapping[str, int]] = field(
        default_factory=lambda: MappingProxyType({})
    )
    max_edges_per_relation: Mapping[CanonicalRelation, int] = field(
        default_factory=lambda: MappingProxyType({})
    )
    max_feature_bytes: int | None = None
    max_total_size_bytes: int | None = None
    max_cut_fraction: float | None = None
    min_locality: float | None = None

    def __post_init__(self) -> None:
        nodes = _limit_map(
            self.max_nodes_per_type,
            "max_nodes_per_type",
            code="PARTITION-TYPE-BALANCE-001",
        )
        if any(
            not isinstance(key, str) or not key
            for key in nodes
        ):
            raise _error(
                "PARTITION-TYPE-BALANCE-001",
                "node-type limit keys must be non-empty strings",
            )
        object.__setattr__(self, "max_nodes_per_type", nodes)
        phases: dict[str, Mapping[str, int]] = {}
        if not isinstance(self.max_phase_nodes, Mapping):
            raise _error(
                "PARTITION-PHASE-BALANCE-001",
                "max_phase_nodes must be a mapping",
            )
        for tag, values in self.max_phase_nodes.items():
            if (
                not isinstance(tag, str)
                or not tag
                or not isinstance(values, Mapping)
            ):
                raise _error(
                    "PARTITION-PHASE-BALANCE-001",
                    "phase limits must map non-empty tags to mappings",
                )
            unknown = set(values) - {"train", "val", "test"}
            if unknown:
                raise _error(
                    "PARTITION-PHASE-BALANCE-001",
                    f"unknown split phase {min(unknown)!r}",
                )
            phases[tag] = _limit_map(
                values,
                f"max_phase_nodes[{tag!r}]",
                code="PARTITION-PHASE-BALANCE-001",
            )
        object.__setattr__(
            self,
            "max_phase_nodes",
            MappingProxyType(phases),
        )
        if not isinstance(self.max_edges_per_relation, Mapping):
            raise _error(
                "PARTITION-RELATION-BALANCE-001",
                "max_edges_per_relation must be a mapping",
            )
        relations = {
            _relation(
                key,
                code="PARTITION-RELATION-BALANCE-001",
            ): value
            for key, value in self.max_edges_per_relation.items()
        }
        object.__setattr__(
            self,
            "max_edges_per_relation",
            _limit_map(
                relations,
                "max_edges_per_relation",
                code="PARTITION-RELATION-BALANCE-001",
            ),
        )
        for name in ("max_feature_bytes", "max_total_size_bytes"):
            value = getattr(self, name)
            if value is not None and (isinstance(value, bool) or not isinstance(value, int) or value < 0):
                raise ValueError(f"{name} must be a non-negative integer or None")
        for name in ("max_cut_fraction", "min_locality"):
            value = getattr(self, name)
            if value is not None and (isinstance(value, bool) or not isinstance(value, (int, float)) or not 0 <= float(value) <= 1):
                raise ValueError(f"{name} must be in [0, 1] or None")
            if value is not None:
                object.__setattr__(self, name, float(value))

    def as_record(self) -> dict[str, Any]:
        return {
            "max_nodes_per_type": dict(sorted(self.max_nodes_per_type.items())),
            "max_phase_nodes": {tag: dict(sorted(value.items())) for tag, value in sorted(self.max_phase_nodes.items())},
            "max_edges_per_relation": {json.dumps(key, separators=(",", ":")): value for key, value in sorted(self.max_edges_per_relation.items())},
            "max_feature_bytes": self.max_feature_bytes,
            "max_total_size_bytes": self.max_total_size_bytes,
            "max_cut_fraction": self.max_cut_fraction,
            "min_locality": self.min_locality,
        }

    @property
    def fingerprint(self) -> str:
        return _hash_json(self.as_record())


@dataclass(frozen=True, slots=True)
class QualificationCheck:
    check_id: str
    passed: bool
    observed: Mapping[str, Any]
    limit: Any = None
    detail: str = ""

    def __post_init__(self) -> None:
        object.__setattr__(self, "observed", _freeze(self.observed))
        object.__setattr__(self, "limit", _freeze(self.limit))

    def as_record(self) -> dict[str, Any]:
        return {"check_id": self.check_id, "passed": self.passed, "observed": _jsonable(self.observed), "limit": _jsonable(self.limit), "detail": self.detail}


@dataclass(frozen=True, slots=True)
class PartitionStatistics:
    node_counts: Mapping[str, tuple[int, ...]]
    phase_counts: Mapping[str, Mapping[str, tuple[int, ...]]]
    relation_edge_counts: Mapping[str, tuple[int, ...]]
    feature_bytes: tuple[int, ...]
    total_size_bytes: tuple[int, ...]
    owned_edges: tuple[int, ...]
    cut_edges: tuple[int, ...]
    cut_fraction: tuple[float, ...]
    locality: tuple[float, ...]

    def __post_init__(self) -> None:
        object.__setattr__(self, "node_counts", _freeze(self.node_counts))
        object.__setattr__(self, "phase_counts", _freeze(self.phase_counts))
        object.__setattr__(self, "relation_edge_counts", _freeze(self.relation_edge_counts))
        for name in ("feature_bytes", "total_size_bytes", "owned_edges", "cut_edges", "cut_fraction", "locality"):
            object.__setattr__(self, name, tuple(getattr(self, name)))
        lengths = {len(getattr(self, name)) for name in ("feature_bytes", "total_size_bytes", "owned_edges", "cut_edges", "cut_fraction", "locality")}
        lengths.update(len(value) for value in self.node_counts.values())
        lengths.update(len(value) for phases in self.phase_counts.values() for value in phases.values())
        lengths.update(len(value) for value in self.relation_edge_counts.values())
        if len(lengths) > 1:
            raise ValueError("statistics vectors have inconsistent partition counts")

    @classmethod
    def empty(cls, count: int) -> PartitionStatistics:
        zeros = (0,) * count
        return cls({}, {}, {}, zeros, zeros, zeros, zeros, (0.0,) * count, (1.0,) * count)

    def as_record(self) -> dict[str, Any]:
        return _jsonable({name: getattr(self, name) for name in self.__dataclass_fields__})


def _readonly(value: Any, role: str) -> np.ndarray:
    array = np.asarray(value)
    if array.dtype != np.dtype("int64") or array.ndim != 1:
        raise _error("PARTITION-ID-001", f"{role} must be one-dimensional int64")
    result = np.array(array, dtype=np.int64, copy=True)
    result.flags.writeable = False
    return result


def _frame(digest: Any, role: str, payload: bytes) -> None:
    name = role.encode()
    digest.update(len(name).to_bytes(8, "big")); digest.update(name)
    digest.update(len(payload).to_bytes(8, "big")); digest.update(payload)


def _array_frame(digest: Any, role: str, value: np.ndarray) -> None:
    array = np.asarray(value)
    if array.ndim != 1:
        raise ValueError(f"{role} must be one-dimensional")
    _frame(digest, role + ":dtype", b"<i8")
    _frame(digest, role + ":shape", json.dumps(list(array.shape)).encode())
    name = role.encode()
    digest.update(len(name).to_bytes(8, "big"))
    digest.update(name)
    digest.update((array.size * 8).to_bytes(8, "big"))
    elements_per_chunk = 128 * 1024
    for start in range(0, array.size, elements_per_chunk):
        chunk = np.asarray(
            array[start : start + elements_per_chunk],
            dtype=np.dtype("<i8"),
            order="C",
        )
        digest.update(memoryview(chunk).cast("B"))


def _identity(parts: int, fingerprint: str, assignments: Mapping[str, np.ndarray], ownership: Mapping[CanonicalRelation, np.ndarray], options: Mapping[str, Any]) -> str:
    digest = hashlib.sha256(); _frame(digest, "format", b"typed-partition-book-v1")
    _frame(digest, "parts", str(parts).encode()); _frame(digest, "topology", fingerprint.encode())
    for name, value in sorted(assignments.items()):
        _frame(digest, "node_type", name.encode()); _array_frame(digest, "assignment", value)
    for relation, value in sorted(ownership.items()):
        _frame(digest, "relation", json.dumps(relation, separators=(",", ":")).encode()); _array_frame(digest, "ownership", value)
    _frame(digest, "options", json.dumps(_jsonable(options), sort_keys=True, separators=(",", ":")).encode())
    return digest.hexdigest()


@dataclass(frozen=True, slots=True)
class TypedPartitionBook:
    num_partitions: int
    node_assignments: Mapping[str, np.ndarray]
    node_permutations: Mapping[str, np.ndarray]
    node_inverse_permutations: Mapping[str, np.ndarray]
    node_partptr: Mapping[str, np.ndarray]
    edge_ownership: Mapping[CanonicalRelation, np.ndarray]
    topology_fingerprint: str
    source_binding: Mapping[str, Any]
    backend: str
    backend_version: str
    options: Mapping[str, Any]
    provenance: Mapping[str, Any]
    estimated_resources: Mapping[str, Any]
    measured_resources: Mapping[str, Any]
    statistics: PartitionStatistics
    qualification_checks: tuple[QualificationCheck, ...]
    content_identity: str

    @classmethod
    def from_assignments(cls, *, num_partitions: int, node_assignments: Mapping[str, np.ndarray], edge_ownership: Mapping[CanonicalRelation, np.ndarray], topology_fingerprint: str, source_binding: Mapping[str, Any], backend: str, backend_version: str, options: Mapping[str, Any], provenance: Mapping[str, Any], estimated_resources: Mapping[str, Any], measured_resources: Mapping[str, Any], statistics: PartitionStatistics, qualification_checks: Sequence[QualificationCheck]) -> TypedPartitionBook:
        if isinstance(num_partitions, bool) or not isinstance(num_partitions, int) or num_partitions < 2:
            raise _error("PARTITION-ID-001", "num_partitions must exceed one")
        assignments: dict[str, np.ndarray] = {}
        permutations: dict[str, np.ndarray] = {}
        inverses: dict[str, np.ndarray] = {}
        pointers: dict[str, np.ndarray] = {}
        for name, raw in sorted(node_assignments.items()):
            assignment = _readonly(raw, f"assignment[{name!r}]")
            if not len(assignment) or np.any(assignment < 0) or np.any(assignment >= num_partitions):
                raise _error("PARTITION-ID-001", f"assignment[{name!r}] contains out-of-range IDs")
            ordinal = np.arange(len(assignment), dtype=np.int64)
            permutation = np.lexsort((ordinal, assignment)).astype(np.int64, copy=False)
            inverse = np.empty(len(assignment), dtype=np.int64); inverse[permutation] = ordinal
            pointer = np.concatenate(([0], np.cumsum(np.bincount(assignment, minlength=num_partitions)))).astype(np.int64)
            for value in (permutation, inverse, pointer): value.flags.writeable = False
            assignments[name] = assignment; permutations[name] = permutation; inverses[name] = inverse; pointers[name] = pointer
        if not assignments:
            raise _error("PARTITION-ID-001", "node assignments are empty")
        ownership: dict[CanonicalRelation, np.ndarray] = {}
        for key, raw in sorted(edge_ownership.items()):
            value = _readonly(raw, f"edge ownership[{key!r}]")
            if np.any(value < 0) or np.any(value >= num_partitions):
                raise _error("PARTITION-ID-001", f"edge ownership[{key!r}] is out of range")
            ownership[_relation(key)] = value
        if len(topology_fingerprint) != 64:
            raise _error("PARTITION-FINGERPRINT-001", "topology fingerprint is not SHA-256")
        content = _identity(num_partitions, topology_fingerprint, assignments, ownership, options)
        book = cls(num_partitions, MappingProxyType(assignments), MappingProxyType(permutations), MappingProxyType(inverses), MappingProxyType(pointers), MappingProxyType(ownership), topology_fingerprint, _freeze(source_binding), backend, backend_version, _freeze(options), _freeze(provenance), _freeze(estimated_resources), _freeze(measured_resources), statistics, tuple(qualification_checks), content)
        return validate_typed_partition_book(book)


def validate_typed_partition_book(book: TypedPartitionBook) -> TypedPartitionBook:
    if not isinstance(book, TypedPartitionBook):
        raise TypeError("book must be TypedPartitionBook")
    occupied = np.zeros(book.num_partitions, dtype=np.int64)
    for name, assignment in book.node_assignments.items():
        occupied += np.bincount(assignment, minlength=book.num_partitions)
        ordinal = np.arange(len(assignment), dtype=np.int64)
        permutation = np.lexsort((ordinal, assignment))
        pointer = np.concatenate(([0], np.cumsum(np.bincount(assignment, minlength=book.num_partitions))))
        if assignment.flags.writeable or not np.array_equal(book.node_permutations[name], permutation) or not np.array_equal(book.node_inverse_permutations[name][permutation], ordinal) or not np.array_equal(book.node_partptr[name], pointer):
            raise _error("PARTITION-ID-001", f"derived arrays for {name!r} are inconsistent")
    if np.any(occupied == 0):
        raise _error("PARTITION-EMPTY-001", "every partition must own a node")
    if book.content_identity != _identity(book.num_partitions, book.topology_fingerprint, book.node_assignments, book.edge_ownership, book.options):
        raise _error("PARTITION-FINGERPRINT-001", "book identity is inconsistent")
    return book


class _TopologyFingerprint:
    @staticmethod
    def from_components(*, node_counts: Sequence[tuple[str, int]], relations: Sequence[tuple[CanonicalRelation, np.ndarray, np.ndarray]] = ()) -> str:
        digest = hashlib.sha256(); _frame(digest, "format", b"typed-canonical-topology-v1")
        for name, count in node_counts:
            _frame(digest, "node_type", name.encode()); _frame(digest, "node_count", str(count).encode())
        for relation, colptr, row in relations:
            _frame(digest, "relation", json.dumps(relation, separators=(",", ":")).encode()); _array_frame(digest, "colptr", colptr); _array_frame(digest, "row", row)
        return digest.hexdigest()

    @classmethod
    def from_relation_build(cls, ingestor: Any, relation_build: Any, *, validate_binding: bool = True) -> str:
        if validate_binding and relation_build.stage_root != ingestor.stage_root(relation_build.inventory):
            raise _error("PARTITION-FINGERPRINT-001", "Task4 binding belongs to another ingestor")
        metadata_path = relation_build.artifact_root / "relations.json"
        if metadata_path.is_symlink() or not metadata_path.is_file():
            raise _error("PARTITION-FINGERPRINT-001", "unsafe Task4 metadata")
        try:
            metadata = json.loads(metadata_path.read_text())
        except (OSError, UnicodeDecodeError, json.JSONDecodeError) as error:
            raise _error("PARTITION-FINGERPRINT-001", "malformed Task4 metadata") from error
        counts = dict(relation_build.inventory.node_rows)
        nodes = tuple((node.name, counts[node.name]) for node in ingestor.source.spec.node_types)
        relations = []
        for key, record in sorted(metadata["relations"].items()):
            colptr_path = relation_build.artifact_root / record["colptr"]["relative_path"]
            row_path = relation_build.artifact_root / record["row"]["relative_path"]
            if colptr_path.is_symlink() or row_path.is_symlink():
                raise _error("PARTITION-FINGERPRINT-001", f"unsafe relation {key}")
            relations.append((tuple(record["relation"]), np.load(colptr_path, mmap_mode="r", allow_pickle=False), np.load(row_path, mmap_mode="r", allow_pickle=False)))
        return cls.from_components(node_counts=nodes, relations=relations)


topology_fingerprint = _TopologyFingerprint()

__all__ = ["CanonicalRelation", "PartitionQualificationLimits", "PartitionStatistics", "QualificationCheck", "TypedPartitionBook", "topology_fingerprint", "validate_typed_partition_book"]
