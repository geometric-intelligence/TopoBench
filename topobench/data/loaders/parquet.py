"""Immutable contracts for typed Parquet graph source declarations.

This module deliberately describes sources only.  It does not import a Parquet
engine, inspect schemas, scan rows, map identifiers, execute SQL, or construct a
PyG dataset.  Conversion and runtime storage belong to the ingestion layer.
"""

from __future__ import annotations

import re
import stat
from collections import Counter
from collections.abc import Mapping, Sequence
from dataclasses import dataclass, field
from pathlib import Path, PurePosixPath, PureWindowsPath
from typing import Any, Literal

OutputKind = Literal["homogeneous", "heterogeneous"]
IdDType = Literal[
    "int8",
    "int16",
    "int32",
    "int64",
    "uint8",
    "uint16",
    "uint32",
    "uint64",
    "string",
]
FeatureRepresentation = Literal["fixed_size_list", "scalar_columns"]
CoveragePolicy = Literal["complete", "partial"]
PartitionStrategy = Literal["cluster", "neighbor"]
PartitionBackend = Literal["pyg"]

_ID_DTYPES = frozenset(
    {
        "int8",
        "int16",
        "int32",
        "int64",
        "uint8",
        "uint16",
        "uint32",
        "uint64",
        "string",
    }
)
_FEATURE_DTYPES = frozenset(
    {
        "float16",
        "float32",
        "float64",
        "int8",
        "int16",
        "int32",
        "int64",
        "uint8",
        "uint16",
        "uint32",
        "uint64",
    }
)
_LABEL_DTYPES = _FEATURE_DTYPES | {"string"}
_SUPPORTED_SOURCE_MODES = frozenset(
    {
        ("homogeneous", "cluster", "pyg"),
        ("heterogeneous", "cluster", "pyg"),
        ("heterogeneous", "neighbor", "pyg"),
    }
)
_NAME_PATTERN = re.compile(r"[A-Za-z_][A-Za-z0-9_.-]*\Z")
_TAG_PATTERN = re.compile(r"[A-Za-z0-9][A-Za-z0-9_.-]*\Z")


def _mapping(value: object, *, context: str) -> Mapping[str, Any]:
    if not isinstance(value, Mapping):
        raise TypeError(f"{context} must be a mapping")
    if not all(isinstance(key, str) for key in value):
        raise TypeError(f"{context} keys must be strings")
    return value


def _sequence(value: object, *, context: str) -> Sequence[Any]:
    if isinstance(value, (str, bytes)) or not isinstance(value, Sequence):
        raise TypeError(f"{context} must be a sequence")
    return value


def _check_keys(
    value: Mapping[str, Any],
    *,
    allowed: frozenset[str],
    context: str,
) -> None:
    extras = sorted(set(value) - allowed)
    if extras:
        rendered = ", ".join(repr(key) for key in extras)
        raise ValueError(
            f"{context}: unsupported configuration key(s): {rendered}"
        )


def _required(
    value: Mapping[str, Any],
    key: str,
    *,
    context: str,
) -> Any:
    if key not in value or value[key] is None:
        raise ValueError(f"{context}.{key} is required")
    return value[key]


def _name(value: object, *, context: str) -> str:
    if not isinstance(value, str) or not _NAME_PATTERN.fullmatch(value):
        raise ValueError(f"{context} must be a non-empty canonical name")
    return value


def _column(value: object, *, context: str) -> str:
    return _name(value, context=context)


def _positive_int(value: object, *, context: str) -> int:
    if isinstance(value, bool) or not isinstance(value, int) or value <= 0:
        raise ValueError(f"{context} must be a positive integer")
    return value


def _nonnegative_int(value: object, *, context: str) -> int:
    if isinstance(value, bool) or not isinstance(value, int) or value < 0:
        raise ValueError(f"{context} must be a non-negative integer")
    return value


def _finite_nonnegative_float(value: object, *, context: str) -> float:
    if isinstance(value, bool) or not isinstance(value, (int, float)):
        raise TypeError(f"{context} must be numeric")
    result = float(value)
    if result < 0.0 or result == float("inf") or result != result:
        raise ValueError(f"{context} must be finite and non-negative")
    return result


def _relative_path(
    value: object,
    *,
    context: str,
    parquet_file: bool,
) -> str:
    if not isinstance(value, str) or not value or "\x00" in value:
        raise ValueError(f"{context} must be a safe relative path")
    if "${" in value or "\\" in value:
        raise ValueError(f"{context} must be a safe relative path")
    posix_path = PurePosixPath(value)
    windows_path = PureWindowsPath(value)
    if (
        posix_path.is_absolute()
        or windows_path.is_absolute()
        or bool(windows_path.drive)
        or ".." in posix_path.parts
    ):
        raise ValueError(f"{context} must be a safe relative path")
    normalized = posix_path.as_posix()
    if normalized in {"", "."}:
        raise ValueError(f"{context} must be a safe relative path")
    if parquet_file and posix_path.suffix != ".parquet":
        raise ValueError(f"{context} must name a .parquet file")
    return normalized


def _paths(value: object, *, context: str) -> tuple[str, ...]:
    items = _sequence(value, context=context)
    if not items:
        raise ValueError(f"{context} must contain at least one file")
    normalized = tuple(
        _relative_path(
            item,
            context=f"{context}[{index}]",
            parquet_file=True,
        )
        for index, item in enumerate(items)
    )
    if len(set(normalized)) != len(normalized):
        raise ValueError(f"{context} contains duplicate files")
    return tuple(sorted(normalized))


def _columns(
    value: object,
    *,
    context: str,
    sort: bool,
) -> tuple[str, ...]:
    items = _sequence(value, context=context)
    normalized = tuple(
        _column(item, context=f"{context}[{index}]")
        for index, item in enumerate(items)
    )
    if not normalized:
        raise ValueError(f"{context} must contain at least one column")
    if len(set(normalized)) != len(normalized):
        raise ValueError(f"{context} contains duplicate columns")
    return tuple(sorted(normalized)) if sort else normalized


def _confined_path(
    root: Path,
    relative: str,
    *,
    context: str,
) -> tuple[Path, tuple[int, int] | None]:
    candidate = root
    regular_file_identity: tuple[int, int] | None = None
    for component in PurePosixPath(relative).parts:
        candidate /= component
        try:
            metadata = candidate.lstat()
        except FileNotFoundError:
            regular_file_identity = None
            continue
        except OSError as error:
            raise ValueError(
                f"{context} could not inspect path component "
                f"{str(candidate)!r}"
            ) from error
        if stat.S_ISLNK(metadata.st_mode):
            raise ValueError(
                f"{context} contains symlink component {str(candidate)!r}"
            )
        regular_file_identity = (
            (metadata.st_dev, metadata.st_ino)
            if stat.S_ISREG(metadata.st_mode)
            else None
        )
    resolved = (root / relative).resolve(strict=False)
    try:
        resolved.relative_to(root)
    except ValueError as error:
        raise ValueError(f"{context} escapes source_root") from error
    return resolved, regular_file_identity


@dataclass(frozen=True, slots=True)
class NodeTypeSpec:
    """One node type's exact external-ID and fixed-width feature schema."""

    name: str
    paths: tuple[str, ...]
    id_column: str
    id_dtype: IdDType
    feature_columns: tuple[str, ...]
    feature_dtype: str
    feature_width: int
    feature_representation: FeatureRepresentation = "scalar_columns"

    def __post_init__(self) -> None:
        name = _name(self.name, context="node type name")
        context = f"node_types[{name}]"
        object.__setattr__(self, "name", name)
        object.__setattr__(
            self, "paths", _paths(self.paths, context=f"{context}.paths")
        )
        object.__setattr__(
            self,
            "id_column",
            _column(self.id_column, context=f"{context}.id_column"),
        )
        if self.id_dtype not in _ID_DTYPES:
            raise ValueError(
                f"{context}.id_dtype must be one of {sorted(_ID_DTYPES)!r}"
            )
        if self.feature_dtype not in _FEATURE_DTYPES:
            raise ValueError(
                f"{context}.features dtype {self.feature_dtype!r} is unsupported"
            )
        width = _positive_int(
            self.feature_width,
            context=f"{context}.features width",
        )
        columns = _columns(
            self.feature_columns,
            context=f"{context}.features columns",
            sort=False,
        )
        if self.feature_representation == "scalar_columns":
            if len(columns) != width:
                raise ValueError(
                    f"{context}.features scalar column count must equal width"
                )
        elif self.feature_representation == "fixed_size_list":
            if len(columns) != 1:
                raise ValueError(
                    f"{context}.features fixed_size_list requires one column"
                )
        else:
            raise ValueError(
                f"{context}.features representation must be fixed_size_list "
                "or scalar_columns"
            )
        if self.id_column in columns:
            raise ValueError(f"{context}.features may not reuse the ID column")
        object.__setattr__(self, "feature_columns", columns)
        object.__setattr__(self, "feature_width", width)


@dataclass(frozen=True, slots=True)
class RelationSpec:
    """One canonical directed relation and its source-file column roles."""

    relation: tuple[str, str, str]
    paths: tuple[str, ...]
    source_column: str
    destination_column: str
    edge_id_column: str | None = None
    edge_fields: tuple[str, ...] = ()

    def __post_init__(self) -> None:
        relation_value = self.relation
        if (
            isinstance(relation_value, (str, bytes))
            or not isinstance(relation_value, Sequence)
            or len(relation_value) != 3
            or any(
                not isinstance(component, str)
                or not _NAME_PATTERN.fullmatch(component)
                for component in relation_value
            )
        ):
            raise ValueError("relation must be a canonical triple of names")
        relation = tuple(relation_value)
        context = f"relations[{relation!r}]"
        object.__setattr__(self, "relation", relation)
        object.__setattr__(
            self, "paths", _paths(self.paths, context=f"{context}.paths")
        )
        source = _column(
            self.source_column,
            context=f"{context}.source_column",
        )
        destination = _column(
            self.destination_column,
            context=f"{context}.destination_column",
        )
        if source == destination:
            raise ValueError(
                f"{context} source and destination columns must be distinct"
            )
        edge_id = self.edge_id_column
        if edge_id is not None:
            edge_id = _column(edge_id, context=f"{context}.edge_id_column")
        fields = (
            ()
            if not self.edge_fields
            else _columns(
                self.edge_fields,
                context=f"{context}.edge_fields",
                sort=True,
            )
        )
        role_columns = (source, destination, edge_id)
        if any(field in role_columns for field in fields):
            raise ValueError(
                f"{context}.edge_fields may not reuse role columns"
            )
        if edge_id in {source, destination}:
            raise ValueError(f"{context}.edge_id_column must be a stable role")
        object.__setattr__(self, "source_column", source)
        object.__setattr__(self, "destination_column", destination)
        object.__setattr__(self, "edge_id_column", edge_id)
        object.__setattr__(self, "edge_fields", fields)


@dataclass(frozen=True, slots=True)
class SplitSetSpec:
    """One explicit train/validation/test source triplet for a named tag."""

    tag: str
    train: str
    val: str
    test: str
    coverage: CoveragePolicy
    qualified: bool = True

    def __post_init__(self) -> None:
        if not isinstance(self.tag, str) or not _TAG_PATTERN.fullmatch(
            self.tag
        ):
            raise ValueError(f"split tag {self.tag!r} is illegal")
        phase_paths = tuple(
            _relative_path(
                getattr(self, phase),
                context=f"split[{self.tag}].{phase}",
                parquet_file=True,
            )
            for phase in ("train", "val", "test")
        )
        if len(set(phase_paths)) != 3:
            raise ValueError(
                f"split[{self.tag}] phase filenames must be distinct"
            )
        if self.coverage not in {"complete", "partial"}:
            raise ValueError(
                f"split[{self.tag}].coverage must be complete or partial"
            )
        if not isinstance(self.qualified, bool):
            raise TypeError(f"split[{self.tag}].qualified must be boolean")
        object.__setattr__(self, "train", phase_paths[0])
        object.__setattr__(self, "val", phase_paths[1])
        object.__setattr__(self, "test", phase_paths[2])


@dataclass(frozen=True, slots=True)
class SplitRegistrySpec:
    """Canonical named split registry and mandatory ID-integrity policies."""

    active_tag: str
    sets: tuple[SplitSetSpec, ...]
    cross_tag_overlap: str = "allowed"
    within_phase_ids: str = "unique"
    within_tag_phases: str = "disjoint"
    target_id_resolution: str = "required"

    def __post_init__(self) -> None:
        split_sets = tuple(self.sets)
        if not split_sets:
            raise ValueError("split_registry.sets must not be empty")
        if not all(isinstance(split, SplitSetSpec) for split in split_sets):
            raise TypeError(
                "split_registry.sets must contain SplitSetSpec values"
            )
        tags = tuple(split.tag for split in split_sets)
        duplicate_tags = sorted(
            tag for tag, count in Counter(tags).items() if count > 1
        )
        if duplicate_tags:
            raise ValueError(
                f"split_registry has duplicate split tag {duplicate_tags[0]!r}"
            )
        if self.active_tag not in set(tags):
            raise ValueError(
                f"split_registry.active_tag {self.active_tag!r} is not registered"
            )
        all_phase_paths = tuple(
            path
            for split in split_sets
            for path in (split.train, split.val, split.test)
        )
        if len(set(all_phase_paths)) != len(all_phase_paths):
            raise ValueError(
                "split_registry phase filenames must be globally unique"
            )
        required_policies = {
            "cross_tag_overlap": "allowed",
            "within_phase_ids": "unique",
            "within_tag_phases": "disjoint",
            "target_id_resolution": "required",
        }
        for field_name, expected in required_policies.items():
            observed = getattr(self, field_name)
            if observed != expected:
                raise ValueError(
                    f"split_registry.{field_name} must be {expected!r}, "
                    f"got {observed!r}"
                )
        object.__setattr__(
            self,
            "sets",
            tuple(sorted(split_sets, key=lambda split: split.tag)),
        )


@dataclass(frozen=True, slots=True)
class SupervisionSpec:
    """Labels and named split ownership for exactly one target node type."""

    target_node_type: str
    label_column: str
    label_dtype: str
    split_registry: SplitRegistrySpec
    label_source: str = "nodes"
    label_paths: tuple[str, ...] = ()
    label_id_column: str | None = None

    def __post_init__(self) -> None:
        target = _name(
            self.target_node_type,
            context="supervision.target_node_type",
        )
        label_column = _column(
            self.label_column,
            context="supervision.labels.column",
        )
        if self.label_dtype not in _LABEL_DTYPES:
            raise ValueError(
                f"supervision.labels.dtype {self.label_dtype!r} is unsupported"
            )
        if not isinstance(self.split_registry, SplitRegistrySpec):
            raise TypeError(
                "supervision.split_registry must be a SplitRegistrySpec"
            )
        if self.label_source == "nodes":
            if self.label_paths or self.label_id_column is not None:
                raise ValueError(
                    "supervision node labels may not declare paths or node_id"
                )
            label_paths: tuple[str, ...] = ()
            label_id = None
        elif self.label_source == "dataset":
            label_paths = _paths(
                self.label_paths,
                context="supervision.labels.paths",
            )
            if self.label_id_column is None:
                raise ValueError(
                    "supervision.labels.node_id is required for dataset labels"
                )
            label_id = _column(
                self.label_id_column,
                context="supervision.labels.node_id",
            )
            if label_id == label_column:
                raise ValueError(
                    "supervision label and node ID columns must be distinct"
                )
        else:
            raise ValueError(
                "supervision.labels.source must be nodes or dataset"
            )
        object.__setattr__(self, "target_node_type", target)
        object.__setattr__(self, "label_column", label_column)
        object.__setattr__(self, "label_paths", label_paths)
        object.__setattr__(self, "label_id_column", label_id)


@dataclass(frozen=True, slots=True)
class PartitionSpec:
    """Partition-production capability and host-memory admission budget."""

    strategy: PartitionStrategy
    backend: PartitionBackend = "pyg"
    num_partitions: int = 1500
    recursive: bool = False
    memory_limit_bytes: int = 256 * 1024**3
    external_partition_map: str | None = None

    def __post_init__(self) -> None:
        if not isinstance(self.strategy, str) or not self.strategy:
            raise ValueError("partition.strategy must be a non-empty string")
        if not isinstance(self.backend, str) or not self.backend:
            raise ValueError("partition.backend must be a non-empty string")
        object.__setattr__(
            self,
            "num_partitions",
            _positive_int(
                self.num_partitions,
                context="partition.num_partitions",
            ),
        )
        if not isinstance(self.recursive, bool):
            raise TypeError("partition.recursive must be boolean")
        object.__setattr__(
            self,
            "memory_limit_bytes",
            _positive_int(
                self.memory_limit_bytes,
                context="partition.memory_limit_bytes",
            ),
        )
        if self.external_partition_map is not None:
            object.__setattr__(
                self,
                "external_partition_map",
                _relative_path(
                    self.external_partition_map,
                    context="partition.external_partition_map",
                    parquet_file=False,
                ),
            )


@dataclass(frozen=True, slots=True)
class FittedTransformSpec:
    """Non-executable declaration of a fitted feature transform."""

    name: str = "identity"
    fit_on: str = "train"
    state_path: str | None = None

    def __post_init__(self) -> None:
        if self.name not in {"identity", "pca"}:
            raise ValueError("fitted_transform.name must be identity or pca")
        if self.fit_on != "train":
            raise ValueError("fitted_transform.fit_on must be 'train'")
        if self.state_path is not None:
            object.__setattr__(
                self,
                "state_path",
                _relative_path(
                    self.state_path,
                    context="fitted_transform.state_path",
                    parquet_file=False,
                ),
            )


@dataclass(frozen=True, slots=True)
class ProfilingSpec:
    """Bounded structured-profiling emission policy."""

    enabled: bool = True
    sample_every_steps: int = 10
    emit_on_duration_delta: float = 0.10
    emit_on_memory_delta_bytes: int = 256 * 1024**2

    def __post_init__(self) -> None:
        if not isinstance(self.enabled, bool):
            raise TypeError("profiling.enabled must be boolean")
        object.__setattr__(
            self,
            "sample_every_steps",
            _positive_int(
                self.sample_every_steps,
                context="profiling.sample_every_steps",
            ),
        )
        object.__setattr__(
            self,
            "emit_on_duration_delta",
            _finite_nonnegative_float(
                self.emit_on_duration_delta,
                context="profiling.emit_on_duration_delta",
            ),
        )
        object.__setattr__(
            self,
            "emit_on_memory_delta_bytes",
            _nonnegative_int(
                self.emit_on_memory_delta_bytes,
                context="profiling.emit_on_memory_delta_bytes",
            ),
        )


@dataclass(frozen=True, slots=True)
class ReproducibilitySpec:
    """Reproducibility artifact policy for a typed source."""

    save_reproducibility_bundle: bool = True

    def __post_init__(self) -> None:
        if not isinstance(self.save_reproducibility_bundle, bool):
            raise TypeError(
                "reproducibility.save_reproducibility_bundle must be boolean"
            )


@dataclass(frozen=True, slots=True)
class IngestionLimits:
    """Explicit bounded-memory conversion limits, without runtime behavior."""

    record_batch_rows: int = 65_536
    memory_limit_bytes: int = 4 * 1024**3
    temp_directory: str = "tmp"

    def __post_init__(self) -> None:
        object.__setattr__(
            self,
            "record_batch_rows",
            _positive_int(
                self.record_batch_rows,
                context="ingestion.record_batch_rows",
            ),
        )
        object.__setattr__(
            self,
            "memory_limit_bytes",
            _positive_int(
                self.memory_limit_bytes,
                context="ingestion.memory_limit_bytes",
            ),
        )
        object.__setattr__(
            self,
            "temp_directory",
            _relative_path(
                self.temp_directory,
                context="ingestion.temp_directory",
                parquet_file=False,
            ),
        )


@dataclass(frozen=True, slots=True)
class ParquetTypedGraphSpec:
    """Canonical immutable semantic contract for one typed Parquet graph."""

    source_root: str | Path
    output_kind: OutputKind
    node_types: tuple[NodeTypeSpec, ...]
    relations: tuple[RelationSpec, ...]
    supervision: SupervisionSpec
    partition: PartitionSpec
    fitted_transform: FittedTransformSpec = FittedTransformSpec()
    profiling: ProfilingSpec = ProfilingSpec()
    reproducibility: ReproducibilitySpec = ReproducibilitySpec()
    ingestion: IngestionLimits = IngestionLimits()
    _files: tuple[Path, ...] = field(init=False, repr=False)

    def __post_init__(self) -> None:
        if not isinstance(self.source_root, (str, Path)):
            raise TypeError("source_root must be a path")
        if isinstance(self.source_root, str) and "${" in self.source_root:
            raise ValueError(
                "source_root contains an unresolved interpolation"
            )
        root = Path(self.source_root).expanduser().resolve(strict=False)
        nodes = tuple(self.node_types)
        relations = tuple(self.relations)
        if not nodes:
            raise ValueError("node_types must not be empty")
        if not relations:
            raise ValueError("relations must not be empty")
        if not all(isinstance(node, NodeTypeSpec) for node in nodes):
            raise TypeError("node_types must contain NodeTypeSpec values")
        if not all(
            isinstance(relation, RelationSpec) for relation in relations
        ):
            raise TypeError("relations must contain RelationSpec values")
        node_names = tuple(node.name for node in nodes)
        duplicate_nodes = sorted(
            name for name, count in Counter(node_names).items() if count > 1
        )
        if duplicate_nodes:
            raise ValueError(f"duplicate node type {duplicate_nodes[0]!r}")
        relation_keys = tuple(relation.relation for relation in relations)
        duplicate_relations = sorted(
            relation
            for relation, count in Counter(relation_keys).items()
            if count > 1
        )
        if duplicate_relations:
            raise ValueError(f"duplicate relation {duplicate_relations[0]!r}")
        nodes = tuple(sorted(nodes, key=lambda node: node.name))
        relations = tuple(
            sorted(relations, key=lambda relation: relation.relation)
        )
        if not isinstance(self.partition, PartitionSpec):
            raise TypeError("partition must be a PartitionSpec")
        source_mode = (
            self.output_kind,
            self.partition.strategy,
            self.partition.backend,
        )
        if source_mode not in _SUPPORTED_SOURCE_MODES:
            raise ValueError(
                "unsupported output/strategy/backend combination: "
                f"{source_mode!r}"
            )
        node_name_set = set(node_names)
        for relation in relations:
            unknown = sorted(
                endpoint
                for endpoint in (relation.relation[0], relation.relation[2])
                if endpoint not in node_name_set
            )
            if unknown:
                raise ValueError(
                    f"relation {relation.relation!r} references unknown node "
                    f"type {unknown[0]!r}"
                )
        if self.output_kind == "homogeneous":
            if len(nodes) != 1 or len(relations) != 1:
                raise ValueError(
                    "homogeneous output requires exactly one node type and "
                    "one relation"
                )
            expected = nodes[0].name
            if (
                relations[0].relation[0] != expected
                or relations[0].relation[2] != expected
            ):
                raise ValueError(
                    "homogeneous output requires one self-type relation"
                )
        elif len(nodes) < 2:
            raise ValueError(
                "heterogeneous output requires at least two node types"
            )
        if not isinstance(self.supervision, SupervisionSpec):
            raise TypeError("supervision must be a SupervisionSpec")
        if self.supervision.target_node_type not in node_name_set:
            raise ValueError(
                "supervision.target_node_type "
                f"{self.supervision.target_node_type!r} is missing from node_types"
            )
        target_node = next(
            node
            for node in nodes
            if node.name == self.supervision.target_node_type
        )
        if self.supervision.label_source == "nodes":
            label_column = self.supervision.label_column
            if label_column == target_node.id_column:
                raise ValueError(
                    f"supervision.labels.column {label_column!r} conflicts "
                    f"with node_types[{target_node.name}].id_column"
                )
            if label_column in target_node.feature_columns:
                raise ValueError(
                    f"supervision.labels.column {label_column!r} conflicts "
                    f"with node_types[{target_node.name}].feature_columns"
                )
        nested_specs = (
            ("fitted_transform", self.fitted_transform, FittedTransformSpec),
            ("profiling", self.profiling, ProfilingSpec),
            ("reproducibility", self.reproducibility, ReproducibilitySpec),
            ("ingestion", self.ingestion, IngestionLimits),
        )
        for context, observed, expected_type in nested_specs:
            if not isinstance(observed, expected_type):
                raise TypeError(
                    f"{context} must be a {expected_type.__name__}"
                )
        relative_files: list[tuple[str, str]] = []
        for node in nodes:
            relative_files.extend(
                (f"node_types[{node.name}].paths", path) for path in node.paths
            )
        for relation in relations:
            relative_files.extend(
                (f"relations[{relation.relation!r}].paths", path)
                for path in relation.paths
            )
        relative_files.extend(
            ("supervision.labels.paths", path)
            for path in self.supervision.label_paths
        )
        for split in self.supervision.split_registry.sets:
            relative_files.extend(
                (
                    f"supervision.splits[{split.tag}].{phase}",
                    getattr(split, phase),
                )
                for phase in ("train", "val", "test")
            )
        file_owners: dict[Path, str] = {}
        physical_file_owners: dict[tuple[int, int], tuple[Path, str]] = {}
        for context, path in relative_files:
            canonical, physical_identity = _confined_path(
                root,
                path,
                context=context,
            )
            previous_context = file_owners.get(canonical)
            if previous_context is not None:
                raise ValueError(
                    f"declared file {str(canonical)!r} is reused by "
                    f"{previous_context} and {context}"
                )
            if physical_identity is not None:
                previous_physical_owner = physical_file_owners.get(
                    physical_identity
                )
                if previous_physical_owner is not None:
                    previous_path, previous_role = previous_physical_owner
                    raise ValueError(
                        f"declared existing file {str(previous_path)!r} "
                        f"for {previous_role} and {str(canonical)!r} "
                        f"for {context} share physical inode "
                        f"(st_dev={physical_identity[0]}, "
                        f"st_ino={physical_identity[1]})"
                    )
                physical_file_owners[physical_identity] = (
                    canonical,
                    context,
                )
            file_owners[canonical] = context
        files = tuple(sorted(file_owners))
        extra_paths = (
            (
                "partition.external_partition_map",
                self.partition.external_partition_map,
            ),
            ("fitted_transform.state_path", self.fitted_transform.state_path),
            ("ingestion.temp_directory", self.ingestion.temp_directory),
        )
        for context, path in extra_paths:
            if path is not None:
                _confined_path(root, path, context=context)
        object.__setattr__(self, "source_root", root)
        object.__setattr__(self, "node_types", nodes)
        object.__setattr__(self, "relations", relations)
        object.__setattr__(self, "_files", files)

    @property
    def files(self) -> tuple[Path, ...]:
        """Return canonical declared Parquet paths without opening them."""
        return self._files


@dataclass(frozen=True, slots=True)
class ParquetTypedGraphSource:
    """Lightweight source descriptor consumed by later ingestion layers."""

    spec: ParquetTypedGraphSpec
    files: tuple[Path, ...] = field(init=False)

    def __post_init__(self) -> None:
        if not isinstance(self.spec, ParquetTypedGraphSpec):
            raise TypeError("source.spec must be a ParquetTypedGraphSpec")
        object.__setattr__(self, "files", self.spec.files)


@dataclass(frozen=True, slots=True, init=False)
class ParquetTypedGraphLoader:
    """Hydra construction boundary exposing only an immutable descriptor."""

    source: ParquetTypedGraphSource

    def __init__(
        self,
        parameters: Mapping[str, Any] | ParquetTypedGraphSpec,
    ) -> None:
        spec = (
            parameters
            if isinstance(parameters, ParquetTypedGraphSpec)
            else _spec_from_parameters(parameters)
        )
        object.__setattr__(self, "source", ParquetTypedGraphSource(spec))


def _spec_from_parameters(parameters: object) -> ParquetTypedGraphSpec:
    values = _mapping(parameters, context="loader.parameters")
    _check_keys(
        values,
        allowed=frozenset(
            {
                "data_domain",
                "data_type",
                "data_name",
                "source_root",
                "output_kind",
                "node_types",
                "edge_types",
                "supervision",
                "partition",
                "fitted_transform",
                "profiling",
                "reproducibility",
                "ingestion",
            }
        ),
        context="loader.parameters",
    )
    data_domain = values.get("data_domain")
    if data_domain not in {"graph", "heterogeneous"}:
        raise ValueError(
            "loader.parameters.data_domain must be graph or heterogeneous"
        )
    if values.get("data_type") != "parquet_typed":
        raise ValueError("loader.parameters.data_type must be 'parquet_typed'")
    if values.get("data_name") != "ParquetTypedGraph":
        raise ValueError(
            "loader.parameters.data_name must be 'ParquetTypedGraph'"
        )
    output_kind = _required(values, "output_kind", context="loader.parameters")
    expected_output = {
        "graph": "homogeneous",
        "heterogeneous": "heterogeneous",
    }[data_domain]
    if output_kind != expected_output:
        raise ValueError(
            "loader.parameters output_kind disagrees with data_domain: "
            f"expected {expected_output!r}"
        )

    node_values = _mapping(
        _required(values, "node_types", context="loader.parameters"),
        context="loader.parameters.node_types",
    )
    if not node_values:
        raise ValueError("loader.parameters.node_types must not be empty")
    nodes = tuple(
        _node_from_mapping(name, value) for name, value in node_values.items()
    )

    edge_values = _sequence(
        _required(values, "edge_types", context="loader.parameters"),
        context="loader.parameters.edge_types",
    )
    if not edge_values:
        raise ValueError("loader.parameters.edge_types must not be empty")
    relations = tuple(
        _relation_from_mapping(value, index=index)
        for index, value in enumerate(edge_values)
    )

    supervision = _supervision_from_mapping(
        _required(values, "supervision", context="loader.parameters")
    )
    partition = _partition_from_mapping(
        _required(values, "partition", context="loader.parameters")
    )
    return ParquetTypedGraphSpec(
        source_root=_required(
            values, "source_root", context="loader.parameters"
        ),
        output_kind=output_kind,
        node_types=nodes,
        relations=relations,
        supervision=supervision,
        partition=partition,
        fitted_transform=_fitted_transform_from_mapping(
            values.get("fitted_transform", {})
        ),
        profiling=_profiling_from_mapping(values.get("profiling", {})),
        reproducibility=_reproducibility_from_mapping(
            values.get("reproducibility", {})
        ),
        ingestion=_ingestion_from_mapping(values.get("ingestion", {})),
    )


def _node_from_mapping(name: str, value: object) -> NodeTypeSpec:
    context = f"loader.parameters.node_types[{name}]"
    node = _mapping(value, context=context)
    _check_keys(
        node,
        allowed=frozenset({"paths", "columns"}),
        context=context,
    )
    columns_context = f"{context}.columns"
    columns = _mapping(
        _required(node, "columns", context=context),
        context=columns_context,
    )
    _check_keys(
        columns,
        allowed=frozenset({"id", "id_dtype", "features"}),
        context=columns_context,
    )
    feature_context = f"{columns_context}.features"
    features = _mapping(
        _required(columns, "features", context=columns_context),
        context=feature_context,
    )
    _check_keys(
        features,
        allowed=frozenset(
            {"column", "columns", "dtype", "width", "representation"}
        ),
        context=feature_context,
    )
    representation = _required(
        features, "representation", context=feature_context
    )
    if representation == "fixed_size_list":
        feature_columns = (
            _required(features, "column", context=feature_context),
        )
        if "columns" in features:
            raise ValueError(
                f"{feature_context}.columns is invalid for fixed_size_list"
            )
    elif representation == "scalar_columns":
        feature_columns = _required(
            features,
            "columns",
            context=feature_context,
        )
        if "column" in features:
            raise ValueError(
                f"{feature_context}.column is invalid for scalar_columns"
            )
    else:
        feature_columns = ()
    return NodeTypeSpec(
        name=name,
        paths=_required(node, "paths", context=context),
        id_column=_required(columns, "id", context=columns_context),
        id_dtype=_required(columns, "id_dtype", context=columns_context),
        feature_columns=feature_columns,
        feature_dtype=_required(features, "dtype", context=feature_context),
        feature_width=_required(features, "width", context=feature_context),
        feature_representation=representation,
    )


def _relation_from_mapping(value: object, *, index: int) -> RelationSpec:
    context = f"loader.parameters.edge_types[{index}]"
    relation = _mapping(value, context=context)
    _check_keys(
        relation,
        allowed=frozenset({"type", "paths", "columns"}),
        context=context,
    )
    columns_context = f"{context}.columns"
    columns = _mapping(
        _required(relation, "columns", context=context),
        context=columns_context,
    )
    _check_keys(
        columns,
        allowed=frozenset({"source", "destination", "edge_id", "fields"}),
        context=columns_context,
    )
    return RelationSpec(
        relation=_required(relation, "type", context=context),
        paths=_required(relation, "paths", context=context),
        source_column=_required(columns, "source", context=columns_context),
        destination_column=_required(
            columns,
            "destination",
            context=columns_context,
        ),
        edge_id_column=columns.get("edge_id"),
        edge_fields=columns.get("fields", ()),
    )


def _supervision_from_mapping(value: object) -> SupervisionSpec:
    context = "loader.parameters.supervision"
    supervision = _mapping(value, context=context)
    _check_keys(
        supervision,
        allowed=frozenset({"target_node_type", "labels", "splits"}),
        context=context,
    )
    labels_context = f"{context}.labels"
    labels = _mapping(
        _required(supervision, "labels", context=context),
        context=labels_context,
    )
    _check_keys(
        labels,
        allowed=frozenset({"source", "column", "dtype", "paths", "node_id"}),
        context=labels_context,
    )
    label_paths_value = labels.get("paths")
    label_paths = () if label_paths_value is None else label_paths_value
    return SupervisionSpec(
        target_node_type=_required(
            supervision,
            "target_node_type",
            context=context,
        ),
        label_column=_required(labels, "column", context=labels_context),
        label_dtype=_required(labels, "dtype", context=labels_context),
        label_source=_required(labels, "source", context=labels_context),
        label_paths=label_paths,
        label_id_column=labels.get("node_id"),
        split_registry=_split_registry_from_mapping(
            _required(supervision, "splits", context=context)
        ),
    )


def _split_registry_from_mapping(value: object) -> SplitRegistrySpec:
    context = "loader.parameters.supervision.splits"
    registry = _mapping(value, context=context)
    _check_keys(
        registry,
        allowed=frozenset(
            {
                "active",
                "sets",
                "cross_tag_overlap",
                "within_phase_ids",
                "within_tag_phases",
                "target_id_resolution",
            }
        ),
        context=context,
    )
    sets_context = f"{context}.sets"
    sets = _mapping(
        _required(registry, "sets", context=context),
        context=sets_context,
    )
    split_specs: list[SplitSetSpec] = []
    for tag, value in sets.items():
        split_context = f"{sets_context}[{tag}]"
        split = _mapping(value, context=split_context)
        _check_keys(
            split,
            allowed=frozenset(
                {"train", "val", "test", "coverage", "qualified"}
            ),
            context=split_context,
        )
        qualified = split.get("qualified", True)
        if not isinstance(qualified, bool):
            raise TypeError(f"{split_context}.qualified must be boolean")
        split_specs.append(
            SplitSetSpec(
                tag=tag,
                train=_required(split, "train", context=split_context),
                val=_required(split, "val", context=split_context),
                test=_required(split, "test", context=split_context),
                coverage=_required(split, "coverage", context=split_context),
                qualified=qualified,
            )
        )
    return SplitRegistrySpec(
        active_tag=_required(registry, "active", context=context),
        sets=tuple(split_specs),
        cross_tag_overlap=registry.get("cross_tag_overlap", "allowed"),
        within_phase_ids=registry.get("within_phase_ids", "unique"),
        within_tag_phases=registry.get("within_tag_phases", "disjoint"),
        target_id_resolution=registry.get("target_id_resolution", "required"),
    )


def _partition_from_mapping(value: object) -> PartitionSpec:
    context = "loader.parameters.partition"
    partition = _mapping(value, context=context)
    _check_keys(
        partition,
        allowed=frozenset(
            {
                "strategy",
                "backend",
                "num_partitions",
                "recursive",
                "memory_limit_bytes",
                "external_partition_map",
            }
        ),
        context=context,
    )
    return PartitionSpec(
        strategy=_required(partition, "strategy", context=context),
        backend=partition.get("backend", "pyg"),
        num_partitions=partition.get("num_partitions", 1500),
        recursive=partition.get("recursive", False),
        memory_limit_bytes=partition.get(
            "memory_limit_bytes",
            256 * 1024**3,
        ),
        external_partition_map=partition.get("external_partition_map"),
    )


def _fitted_transform_from_mapping(value: object) -> FittedTransformSpec:
    context = "loader.parameters.fitted_transform"
    transform = _mapping(value, context=context)
    _check_keys(
        transform,
        allowed=frozenset({"name", "fit_on", "state_path"}),
        context=context,
    )
    return FittedTransformSpec(
        name=transform.get("name", "identity"),
        fit_on=transform.get("fit_on", "train"),
        state_path=transform.get("state_path"),
    )


def _profiling_from_mapping(value: object) -> ProfilingSpec:
    context = "loader.parameters.profiling"
    profiling = _mapping(value, context=context)
    _check_keys(
        profiling,
        allowed=frozenset(
            {
                "enabled",
                "sample_every_steps",
                "emit_on_duration_delta",
                "emit_on_memory_delta_bytes",
            }
        ),
        context=context,
    )
    return ProfilingSpec(
        enabled=profiling.get("enabled", True),
        sample_every_steps=profiling.get("sample_every_steps", 10),
        emit_on_duration_delta=profiling.get("emit_on_duration_delta", 0.10),
        emit_on_memory_delta_bytes=profiling.get(
            "emit_on_memory_delta_bytes",
            256 * 1024**2,
        ),
    )


def _reproducibility_from_mapping(value: object) -> ReproducibilitySpec:
    context = "loader.parameters.reproducibility"
    reproducibility = _mapping(value, context=context)
    _check_keys(
        reproducibility,
        allowed=frozenset({"save_reproducibility_bundle"}),
        context=context,
    )
    return ReproducibilitySpec(
        save_reproducibility_bundle=reproducibility.get(
            "save_reproducibility_bundle",
            True,
        )
    )


def _ingestion_from_mapping(value: object) -> IngestionLimits:
    context = "loader.parameters.ingestion"
    ingestion = _mapping(value, context=context)
    _check_keys(
        ingestion,
        allowed=frozenset(
            {"record_batch_rows", "memory_limit_bytes", "temp_directory"}
        ),
        context=context,
    )
    return IngestionLimits(
        record_batch_rows=ingestion.get("record_batch_rows", 65_536),
        memory_limit_bytes=ingestion.get("memory_limit_bytes", 4 * 1024**3),
        temp_directory=ingestion.get("temp_directory", "tmp"),
    )


__all__ = [
    "FittedTransformSpec",
    "IngestionLimits",
    "NodeTypeSpec",
    "ParquetTypedGraphLoader",
    "ParquetTypedGraphSource",
    "ParquetTypedGraphSpec",
    "PartitionSpec",
    "ProfilingSpec",
    "RelationSpec",
    "ReproducibilitySpec",
    "SplitRegistrySpec",
    "SplitSetSpec",
    "SupervisionSpec",
]
