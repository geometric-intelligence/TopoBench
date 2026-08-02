"""Bounded inventory and type-local external-ID mapping for Parquet graphs.

DuckDB and PyArrow are deliberately imported only inside explicit ingestion or
resume-validation calls. Conversion delegates ordering, distinctness, and
lookup-table construction to spill-capable DuckDB operators.
"""

from __future__ import annotations

import contextlib
import dataclasses
import functools
import hashlib
import json
import os
import shutil
import socket
import time
import uuid
from contextlib import contextmanager
from dataclasses import dataclass
from pathlib import Path, PurePosixPath
from typing import TYPE_CHECKING, Any, Iterator, Mapping

from topobench.data.stores.external_node_index import ExternalNodeIndex

if TYPE_CHECKING:
    from topobench.data.loaders.parquet import ParquetTypedGraphSource
    from topobench.data.stores.typed_graph_arrays import TypedGraphArrayBuild
    from topobench.data.stores.typed_graph_csc import TypedGraphRelationBuild
    from topobench.data.stores.pyg_partitioner import TypedPartitionBuild
    from topobench.data.stores.typed_partition_book import (
        PartitionQualificationLimits,
    )

_BEHAVIOR_VERSION = "typed-node-index-v3"
_SUPPORTED_ID_DTYPES = frozenset({"int64", "uint64", "string"})
_HASH_CHUNK_BYTES = 1024 * 1024


def _known_inventory_counts(
    result: object,
) -> tuple[int | None, int | None]:
    inventory = (
        result
        if hasattr(result, "total_rows") and hasattr(result, "total_bytes")
        else getattr(result, "inventory", None)
    )
    rows = getattr(inventory, "total_rows", None)
    byte_count = getattr(inventory, "total_bytes", None)
    return (
        rows if type(rows) is int and rows >= 0 else None,
        byte_count
        if type(byte_count) is int and byte_count >= 0
        else None,
    )



def _monitored_stage(
    operation: str,
    phase: str,
) -> Any:
    def decorate(method: Any) -> Any:
        @functools.wraps(method)
        def monitored(
            self: "ParquetTypedGraphIngestor",
            *args: object,
            **kwargs: object,
        ) -> object:
            monitor = self.execution_monitor
            token = (
                None
                if monitor is None
                else monitor.begin(
                    operation,
                    phase=phase,
                    evidence={"producer": "typed_parquet"},
                )
            )
            try:
                result = method(self, *args, **kwargs)
            except BaseException as error:
                if token is not None:
                    with contextlib.suppress(Exception):
                        monitor.finish(
                            token,
                            status="error",
                            evidence={
                                "failure_stage": phase,
                                "error_type": type(error).__name__,
                            },
                        )
                raise
            if token is not None:
                row_count, byte_count = _known_inventory_counts(result)
                monitor.finish(
                    token,
                    row_count=row_count,
                    unique_storage_bytes=byte_count,
                )
            return result

        return monitored

    return decorate



class ArtifactValidationError(RuntimeError):
    """An input, staging artifact, or completion record is not trustworthy."""


class SourceMutationError(ArtifactValidationError):
    """Inventoried source identity changed before mapping completed."""


class DiskAdmissionError(ArtifactValidationError):
    """Final and temporary storage cannot be simultaneously admitted."""


class ConcurrentBuildError(ArtifactValidationError):
    """A non-stale owner holds the content-addressed build lock."""


@dataclass(frozen=True, slots=True)
class FileInventory:
    """Exact evidence for one canonical source file."""

    relative_path: str
    absolute_path: Path
    byte_size: int
    row_count: int
    uncompressed_bytes: int
    sha256: str
    schema_fingerprint: str
    schema_serialized_hex: str


@dataclass(frozen=True, slots=True)
class SourceInventory:
    """Canonical source evidence and simultaneous disk-peak admission."""

    files: tuple[FileInventory, ...]
    total_bytes: int
    total_rows: int
    node_rows: tuple[tuple[str, int], ...]
    snapshot_bytes: int
    estimated_final_bytes: int
    estimated_temporary_bytes: int
    required_peak_bytes: int
    source_fingerprint: str
    config_fingerprint: str
    dependency_versions: tuple[tuple[str, str], ...]
    final_filesystem_path: Path
    temporary_filesystem_path: Path
    final_device: int
    temporary_device: int


@dataclass(frozen=True, slots=True)
class ExternalNodeIndexBuild:
    """Completed inventory plus independently namespaced type-local indexes."""

    inventory: SourceInventory
    stage_root: Path
    indexes: Mapping[str, ExternalNodeIndex]
    resumed: bool


class ParquetTypedGraphIngestor:
    """Inventory typed sources and build bounded external-ID indexes."""

    def __init__(
        self,
        source: ParquetTypedGraphSource,
        store_root: str | Path,
        *,
        disk_limit_bytes: int | None = None,
        threads: int = 1,
        lock_stale_seconds: float = 3600.0,
        execution_monitor: object | None = None,
    ) -> None:
        if not hasattr(source, "spec") or not hasattr(source, "files"):
            raise TypeError("source must expose the ParquetTypedGraphSource contract")
        if isinstance(disk_limit_bytes, bool) or (
            disk_limit_bytes is not None
            and (not isinstance(disk_limit_bytes, int) or disk_limit_bytes <= 0)
        ):
            raise ValueError("disk_limit_bytes must be a positive integer")
        if isinstance(threads, bool) or not isinstance(threads, int) or threads <= 0:
            raise ValueError("threads must be a positive integer")
        if lock_stale_seconds <= 0:
            raise ValueError("lock_stale_seconds must be positive")
        if execution_monitor is not None and (
            not callable(getattr(execution_monitor, "begin", None))
            or not callable(getattr(execution_monitor, "finish", None))
        ):
            raise TypeError(
                "execution_monitor must expose callable begin and finish methods"
            )
        self.source = source
        self.store_root = Path(store_root).expanduser().resolve(strict=False)
        self.disk_limit_bytes = disk_limit_bytes
        self.threads = threads
        self.lock_stale_seconds = float(lock_stale_seconds)
        self.execution_monitor = execution_monitor

    @_monitored_stage("conversion", "inventory")
    def inventory(self) -> SourceInventory:
        """Stream hashes and inspect exact Arrow metadata in canonical order."""
        pa, pq, duckdb = _parquet_dependencies()
        del duckdb
        spec = self.source.spec
        config_fingerprint = _sha256_json(self._behavior_configuration())
        dependencies = _dependency_versions(pa)
        entries = tuple(
            self._inventory_file(path, pa=pa, pq=pq)
            for path in sorted(self.source.files)
        )
        by_path = {entry.absolute_path: entry for entry in entries}
        node_rows: list[tuple[str, int]] = []
        for node in spec.node_types:
            if node.id_dtype not in _SUPPORTED_ID_DTYPES:
                raise ArtifactValidationError(
                    f"ID-DTYPE-001: node type {node.name!r} declares unsupported "
                    f"mapping domain {node.id_dtype!r}; expected int64, uint64, or string"
                )
            node_entries = tuple(
                by_path[(spec.source_root / relative).resolve(strict=True)]
                for relative in node.paths
            )
            fingerprints = {entry.schema_fingerprint for entry in node_entries}
            if len(fingerprints) != 1:
                raise ArtifactValidationError(
                    f"SCHEMA-DRIFT-001: node type {node.name!r} fragments have "
                    "different exact Arrow schemas"
                )
            expected_type = {
                "int64": pa.int64(),
                "uint64": pa.uint64(),
                "string": pa.string(),
            }[node.id_dtype]
            first_schema = pa.ipc.read_schema(
                pa.BufferReader(bytes.fromhex(node_entries[0].schema_serialized_hex))
            )
            field_index = first_schema.get_field_index(node.id_column)
            if field_index < 0:
                raise ArtifactValidationError(
                    f"ID-SCHEMA-001: node type {node.name!r} is missing ID column "
                    f"{node.id_column!r}"
                )
            observed_type = first_schema.field(field_index).type
            if not observed_type.equals(expected_type):
                raise ArtifactValidationError(
                    f"ID-SCHEMA-001: node type {node.name!r} ID column has exact "
                    f"Arrow type {observed_type}, expected {expected_type}"
                )
            node_rows.append(
                (node.name, sum(entry.row_count for entry in node_entries))
            )

        total_bytes = sum(entry.byte_size for entry in entries)
        total_rows = sum(entry.row_count for entry in entries)
        mapped_rows = sum(count for _, count in node_rows)
        node_paths = {
            (spec.source_root / relative).resolve(strict=True)
            for node in spec.node_types
            for relative in node.paths
        }
        node_uncompressed_bytes = sum(
            entry.uncompressed_bytes
            for entry in entries
            if entry.absolute_path in node_paths
        )
        # Cover two persistent exact-ID representations plus external-sort runs.
        # Parquet uncompressed page sizes prevent source compression from hiding
        # the bytes that conversion operators must encode.
        estimated_final = max(
            1,
            total_bytes + node_uncompressed_bytes * 2 + mapped_rows * 16,
        )
        estimated_temp = max(
            1,
            total_bytes * 3 + node_uncompressed_bytes * 3 + mapped_rows * 40,
        )
        required_peak = estimated_final + estimated_temp
        source_fingerprint = _sha256_json(
            {
                "behavior_version": _BEHAVIOR_VERSION,
                "config_fingerprint": config_fingerprint,
                "dependencies": dict(dependencies),
                "files": [_file_record(entry) for entry in entries],
            }
        )
        temporary_filesystem_path = self._temporary_filesystem_path()
        final_device, _, _ = _filesystem_capacity(self.store_root)
        temporary_device, _, _ = _filesystem_capacity(temporary_filesystem_path)
        inventory = SourceInventory(
            files=entries,
            total_bytes=total_bytes,
            total_rows=total_rows,
            node_rows=tuple(node_rows),
            snapshot_bytes=total_bytes,
            estimated_final_bytes=estimated_final,
            estimated_temporary_bytes=estimated_temp,
            required_peak_bytes=required_peak,
            source_fingerprint=source_fingerprint,
            config_fingerprint=config_fingerprint,
            dependency_versions=dependencies,
            final_filesystem_path=self.store_root,
            temporary_filesystem_path=temporary_filesystem_path,
            final_device=final_device,
            temporary_device=temporary_device,
        )
        self._admit_disk(inventory)
        return inventory

    @_monitored_stage("conversion", "publish")
    def build(self) -> ExternalNodeIndexBuild:
        """Inventory and build all type-local indexes."""
        inventory = self.inventory()
        return self.build_external_node_indexes(inventory)

    @_monitored_stage("conversion", "arrays")
    def build_arrays(
        self,
        index_build: ExternalNodeIndexBuild | None = None,
    ) -> TypedGraphArrayBuild:
        """Stream typed features, target labels, and every registered split."""
        if index_build is None:
            validated_indexes = self.build()
        else:
            if not isinstance(index_build, ExternalNodeIndexBuild):
                raise TypeError("index_build must be an ExternalNodeIndexBuild")
            expected_stage = self.stage_root(index_build.inventory)
            if index_build.stage_root != expected_stage:
                raise ArtifactValidationError(
                    "COMPLETION-EVIDENCE-001: external-ID index stage does not "
                    "belong to this ingestor"
                )
            validated_indexes = self.build_external_node_indexes(
                index_build.inventory
            )
        from topobench.data.stores.typed_graph_arrays import (
            TypedGraphArrayWriter,
        )

        return TypedGraphArrayWriter(self, validated_indexes).build()

    @_monitored_stage("conversion", "relations")
    def build_relations(
        self,
        index_build: ExternalNodeIndexBuild | None = None,
    ) -> TypedGraphRelationBuild:
        """Stream every canonical directed relation into one verified CSC subtree."""
        if index_build is None:
            validated_indexes = self.build()
        else:
            if not isinstance(index_build, ExternalNodeIndexBuild):
                raise TypeError("index_build must be an ExternalNodeIndexBuild")
            expected_stage = self.stage_root(index_build.inventory)
            if index_build.stage_root != expected_stage:
                raise ArtifactValidationError(
                    "COMPLETION-EVIDENCE-001: external-ID index stage does not "
                    "belong to this ingestor"
                )
            validated_indexes = self.build_external_node_indexes(
                index_build.inventory
            )
        arrays_completion = (
            validated_indexes.stage_root
            / "arrays"
            / "arrays.complete.json"
        )
        if arrays_completion.is_file() and not arrays_completion.is_symlink():
            from topobench.data.stores.typed_graph_arrays import (
                TypedGraphArrayWriter,
            )

            validated_arrays = TypedGraphArrayWriter(
                self,
                validated_indexes,
            )._open_validated(resumed=True)
        else:
            validated_arrays = self.build_arrays(validated_indexes)
        from topobench.data.stores.typed_graph_csc import (
            TypedGraphRelationWriter,
        )

        return TypedGraphRelationWriter(
            self,
            validated_indexes,
            validated_arrays,
        ).build()

    @_monitored_stage("partition", "partition")
    def build_partitions(
        self,
        *,
        limits: PartitionQualificationLimits | None = None,
    ) -> TypedPartitionBuild:
        """Generate, qualify, and atomically publish one typed partition book."""
        from topobench.data.stores.pyg_partitioner import (
            TopologyOnlyPyGPartitioner,
        )
        from topobench.data.stores.typed_partition_book import (
            PartitionQualificationLimits,
        )

        qualification_limits = (
            PartitionQualificationLimits() if limits is None else limits
        )
        if not isinstance(qualification_limits, PartitionQualificationLimits):
            raise TypeError("limits must be PartitionQualificationLimits")
        relations = self.build_relations()
        return TopologyOnlyPyGPartitioner(self, relations).build(
            qualification_limits
        )

    @_monitored_stage("conversion", "index")
    def build_external_node_indexes(
        self,
        inventory: SourceInventory,
    ) -> ExternalNodeIndexBuild:
        """Build or exactly validate one content/config-addressed stage."""
        if not isinstance(inventory, SourceInventory):
            raise TypeError("inventory must be a SourceInventory")
        pa, pq, duckdb = _parquet_dependencies()
        expected_config = _sha256_json(self._behavior_configuration())
        dependencies = _dependency_versions(pa)
        if inventory.config_fingerprint != expected_config:
            raise ArtifactValidationError(
                "CONFIG-EVIDENCE-001: inventory configuration does not match ingestor"
            )
        if inventory.dependency_versions != dependencies:
            raise ArtifactValidationError(
                "DEPENDENCY-EVIDENCE-001: inventory dependency versions changed"
            )
        self._admit_disk(inventory)
        self._validate_inventory_current(inventory, pa=pa, pq=pq)
        stage_root = self.stage_root(inventory)
        lock_path = self.lock_path(inventory)
        with self._build_lock(lock_path):
            self._validate_inventory_current(inventory, pa=pa, pq=pq)
            quarantine: Path | None = None
            if os.path.lexists(stage_root):
                if stage_root.is_symlink() or not stage_root.is_dir():
                    raise ArtifactValidationError(
                        "UNSAFE-STAGING-001: addressed stage is not a real directory"
                    )
                if (stage_root / "build.complete.json").is_file():
                    try:
                        return self._resume(
                            inventory,
                            stage_root,
                            pa=pa,
                            pq=pq,
                            duckdb=duckdb,
                        )
                    except ArtifactValidationError:
                        self._quarantine_stage(stage_root)
                        raise
                quarantine = self._quarantine_stage(stage_root)
            stage_root.mkdir(parents=True, exist_ok=False)
            fresh = True
            spill_root = self._new_ephemeral_root(
                inventory,
                purpose="build-spill",
            )
            spill_created = False
            snapshot_root: Path | None = None
            try:
                mappings_root = stage_root / "mappings"
                mappings_root.mkdir()
                spill_root.mkdir(parents=True, exist_ok=False)
                spill_created = True
                inventory_path = stage_root / "inventory.json"
                _atomic_json(inventory_path, _inventory_record(inventory))
                _atomic_json(
                    stage_root / "inventory.complete.json",
                    {
                        "stage": "inventory",
                        "behavior_version": _BEHAVIOR_VERSION,
                        "input_fingerprint": inventory.source_fingerprint,
                        "config_fingerprint": inventory.config_fingerprint,
                        "dependency_versions": dict(dependencies),
                        "outputs": {"inventory.json": _sha256_file(inventory_path)},
                    },
                )
                indexes: dict[str, ExternalNodeIndex] = {}
                with self._immutable_source_snapshot(
                    inventory,
                    purpose="source-snapshot",
                    pa=pa,
                    pq=pq,
                ) as snapshot_stage:
                    snapshot_paths, snapshot_root = snapshot_stage
                    for ordinal, node in enumerate(self.source.spec.node_types):
                        internal_key = f"n{ordinal:04d}"
                        indexes[node.name] = self._build_one_index(
                            inventory,
                            node=node,
                            internal_key=internal_key,
                            root=mappings_root / internal_key,
                            spill_root=spill_root,
                            snapshot_paths=snapshot_paths,
                            duckdb=duckdb,
                        )
                shutil.rmtree(spill_root)
                spill_created = False
                outputs = _stage_output_checksums(stage_root)
                _atomic_json(
                    stage_root / "build.complete.json",
                    {
                        "stage": "external_node_indexes",
                        "behavior_version": _BEHAVIOR_VERSION,
                        "input_fingerprint": inventory.source_fingerprint,
                        "config_fingerprint": inventory.config_fingerprint,
                        "dependency_versions": dict(dependencies),
                        "outputs": outputs,
                        "disk_admission": _disk_admission_record(inventory),
                        "spill_subtree": str(spill_root),
                        "snapshot_subtree": str(snapshot_root),
                        "indexes": {
                            node.name: {
                                "internal_key": f"n{ordinal:04d}",
                                "id_dtype": node.id_dtype,
                                "row_count": dict(inventory.node_rows)[node.name],
                            }
                            for ordinal, node in enumerate(self.source.spec.node_types)
                        },
                    },
                )
                if quarantine is not None:
                    shutil.rmtree(quarantine)
                    quarantine = None
                fresh = False
                return ExternalNodeIndexBuild(
                    inventory=inventory,
                    stage_root=stage_root,
                    indexes=indexes,
                    resumed=False,
                )
            finally:
                if spill_created and spill_root.exists():
                    shutil.rmtree(spill_root)
                if fresh and stage_root.exists():
                    shutil.rmtree(stage_root)

    def _new_ephemeral_root(
        self,
        inventory: SourceInventory,
        *,
        purpose: str,
    ) -> Path:
        fingerprint = self.stage_root(inventory).name
        return (
            inventory.temporary_filesystem_path
            / ".topobench-typed-graph-work"
            / fingerprint
            / f"{purpose}-{uuid.uuid4().hex}"
        )

    def _quarantine_stage(self, stage_root: Path) -> Path:
        if stage_root.is_symlink() or not stage_root.is_dir():
            raise ArtifactValidationError(
                "UNSAFE-STAGING-001: cannot quarantine non-directory stage"
            )
        quarantine = stage_root.with_name(
            f".{stage_root.name}.quarantine-{uuid.uuid4().hex}"
        )
        os.replace(stage_root, quarantine)
        _fsync_directory(stage_root.parent)
        return quarantine

    @contextmanager
    def _immutable_source_snapshot(
        self,
        inventory: SourceInventory,
        *,
        purpose: str,
        pa: Any,
        pq: Any,
    ) -> Iterator[tuple[Mapping[str, Path], Path]]:
        """Yield verified read-only source copies and always remove them."""
        snapshot_root = self._new_ephemeral_root(
            inventory,
            purpose=purpose,
        )
        self._validate_inventory_current(inventory, pa=pa, pq=pq)
        snapshot_root.mkdir(parents=True, exist_ok=False)
        try:
            snapshots = self._snapshot_sources(inventory, snapshot_root)
            self._verify_snapshot(inventory, snapshots)
            yield snapshots, snapshot_root
            self._verify_snapshot(inventory, snapshots)
            self._validate_inventory_current(inventory, pa=pa, pq=pq)
        finally:
            if snapshot_root.exists():
                shutil.rmtree(snapshot_root)

    def _snapshot_sources(
        self,
        inventory: SourceInventory,
        snapshot_root: Path,
    ) -> dict[str, Path]:
        snapshots: dict[str, Path] = {}
        for entry in inventory.files:
            relative = _validate_relative_artifact(entry.relative_path)
            destination = snapshot_root.joinpath(*relative.parts)
            destination.parent.mkdir(parents=True, exist_ok=True)
            _copy_inventory_file(entry, destination)
            snapshots[entry.relative_path] = destination
        return snapshots

    def _verify_snapshot(
        self,
        inventory: SourceInventory,
        snapshots: Mapping[str, Path],
    ) -> None:
        if set(snapshots) != {entry.relative_path for entry in inventory.files}:
            raise SourceMutationError(
                "SOURCE-MUTATION-001: immutable snapshot file set changed"
            )
        for entry in inventory.files:
            snapshot = snapshots[entry.relative_path]
            if (
                snapshot.is_symlink()
                or not snapshot.is_file()
                or snapshot.stat().st_size != entry.byte_size
                or snapshot.stat().st_mode & 0o222
                or _sha256_file(snapshot) != entry.sha256
            ):
                raise SourceMutationError(
                    f"SOURCE-MUTATION-001: snapshot changed for "
                    f"{entry.relative_path!r}"
                )

    def stage_root(self, inventory: SourceInventory) -> Path:
        """Return the content/config/dependency-addressed staging directory."""
        build_id = _sha256_json(
            {
                "source": inventory.source_fingerprint,
                "config": inventory.config_fingerprint,
                "dependencies": dict(inventory.dependency_versions),
                "behavior_version": _BEHAVIOR_VERSION,
            }
        )
        return self.store_root / ".staging" / build_id

    def lock_path(self, inventory: SourceInventory) -> Path:
        """Return the sibling lock path for an addressed staging directory."""
        root = self.stage_root(inventory)
        return root.parent / f"{root.name}.lock"

    def _inventory_file(self, path: Path, *, pa: Any, pq: Any) -> FileInventory:
        try:
            expected = path.resolve(strict=True)
        except (FileNotFoundError, NotADirectoryError) as error:
            raise ArtifactValidationError(
                f"SOURCE-MISSING-001: declared Parquet source is missing: {path}"
            ) from error
        if expected != path:
            raise SourceMutationError(
                f"SOURCE-PATH-001: canonical source path changed: {path}"
            )
        if not expected.is_relative_to(self.source.spec.source_root):
            raise SourceMutationError(
                f"SOURCE-PATH-001: source escaped configured root: {path}"
            )
        digest = hashlib.sha256()
        with expected.open("rb") as stream:
            before = os.fstat(stream.fileno())
            while chunk := stream.read(_HASH_CHUNK_BYTES):
                digest.update(chunk)
            stream.seek(0)
            parquet_file = pq.ParquetFile(stream)
            schema = parquet_file.schema_arrow
            rows = parquet_file.metadata.num_rows
            uncompressed_bytes = sum(
                parquet_file.metadata.row_group(row_group).column(column).total_uncompressed_size
                for row_group in range(parquet_file.metadata.num_row_groups)
                for column in range(parquet_file.metadata.num_columns)
            )
            after = os.fstat(stream.fileno())
        current = expected.stat()
        identities = (
            before.st_dev,
            before.st_ino,
            before.st_size,
            before.st_mtime_ns,
            before.st_ctime_ns,
        ), (
            after.st_dev,
            after.st_ino,
            after.st_size,
            after.st_mtime_ns,
            after.st_ctime_ns,
        ), (
            current.st_dev,
            current.st_ino,
            current.st_size,
            current.st_mtime_ns,
            current.st_ctime_ns,
        )
        if identities[0] != identities[1] or identities[1] != identities[2]:
            raise SourceMutationError(
                f"SOURCE-MUTATION-001: source changed while inventorying {path}"
            )
        serialized = schema.serialize().to_pybytes()
        return FileInventory(
            relative_path=expected.relative_to(self.source.spec.source_root).as_posix(),
            absolute_path=expected,
            byte_size=before.st_size,
            row_count=rows,
            sha256=digest.hexdigest(),
            schema_fingerprint=hashlib.sha256(serialized).hexdigest(),
            schema_serialized_hex=serialized.hex(),
            uncompressed_bytes=uncompressed_bytes,
        )

    def _validate_inventory_current(self, inventory: SourceInventory, *, pa: Any, pq: Any) -> None:
        observed = tuple(
            self._inventory_file(entry.absolute_path, pa=pa, pq=pq)
            for entry in inventory.files
        )
        if tuple(_file_record(entry) for entry in observed) != tuple(
            _file_record(entry) for entry in inventory.files
        ):
            raise SourceMutationError(
                "SOURCE-MUTATION-001: source bytes, schema, rows, or path changed "
                "between inventory and external-ID mapping"
            )

    def _temporary_filesystem_path(self) -> Path:
        root = self.source.spec.source_root
        candidate = (root / self.source.spec.ingestion.temp_directory).resolve(
            strict=False
        )
        if not candidate.is_relative_to(root):
            raise SourceMutationError(
                "SOURCE-PATH-001: ingestion temporary directory escaped source root"
            )
        return candidate

    def _admit_disk(self, inventory: SourceInventory) -> None:
        final_device, final_available, _ = _filesystem_capacity(
            inventory.final_filesystem_path
        )
        temporary_device, temporary_available, _ = _filesystem_capacity(
            inventory.temporary_filesystem_path
        )
        if (
            final_device != inventory.final_device
            or temporary_device != inventory.temporary_device
        ):
            raise DiskAdmissionError(
                "DISK-EVIDENCE-001: final or temporary filesystem changed "
                "between inventory and admission"
            )
        if self.disk_limit_bytes is not None:
            final_available = min(final_available, self.disk_limit_bytes)
            temporary_available = min(
                temporary_available,
                self.disk_limit_bytes,
            )
        if final_device == temporary_device:
            available = min(final_available, temporary_available)
            if inventory.required_peak_bytes > available:
                raise DiskAdmissionError(
                    "DISK-PREFLIGHT-001: shared final+temporary filesystem "
                    f"requires {inventory.required_peak_bytes} bytes, only "
                    f"{available} admitted"
                )
            return
        if inventory.estimated_final_bytes > final_available:
            raise DiskAdmissionError(
                "DISK-PREFLIGHT-001: final filesystem requires "
                f"{inventory.estimated_final_bytes} bytes, only "
                f"{final_available} admitted"
            )
        if inventory.estimated_temporary_bytes > temporary_available:
            raise DiskAdmissionError(
                "DISK-PREFLIGHT-001: temporary filesystem requires "
                f"{inventory.estimated_temporary_bytes} bytes, only "
                f"{temporary_available} admitted"
            )

    def _behavior_configuration(self) -> dict[str, Any]:
        spec = self.source.spec
        return {
            "behavior_version": _BEHAVIOR_VERSION,
            "output_kind": spec.output_kind,
            "node_types": [_normalized_dataclass(node) for node in spec.node_types],
            "relations": [_normalized_dataclass(value) for value in spec.relations],
            "supervision": _normalized_dataclass(spec.supervision),
            "partition": _normalized_dataclass(spec.partition),
            "fitted_transform": _normalized_dataclass(spec.fitted_transform),
            "profiling": _normalized_dataclass(spec.profiling),
            "reproducibility": _normalized_dataclass(spec.reproducibility),
            "ingestion": _normalized_dataclass(spec.ingestion),
            "threads": self.threads,
            "id_order": {
                "int64": "ascending signed numeric",
                "uint64": "ascending unsigned numeric",
                "string": "ascending unsigned UTF-8 bytes",
            },
        }

    def _build_one_index(
        self,
        inventory: SourceInventory,
        *,
        node: Any,
        internal_key: str,
        root: Path,
        spill_root: Path,
        snapshot_paths: Mapping[str, Path],
        duckdb: Any,
    ) -> ExternalNodeIndex:
        root.mkdir()
        temp_database = root / f"lookup.duckdb.tmp-{uuid.uuid4().hex}"
        temp_parquet = root / f"node_ids.parquet.tmp-{uuid.uuid4().hex}"
        final_database = root / "lookup.duckdb"
        final_parquet = root / "node_ids.parquet"
        connection = duckdb.connect(str(temp_database))
        try:
            memory_bytes = self.source.spec.ingestion.memory_limit_bytes
            connection.execute("SET preserve_insertion_order = false")
            escaped_spill = str(spill_root).replace("'", "''")
            connection.execute(f"SET memory_limit = '{memory_bytes}B'")
            connection.execute(f"SET temp_directory = '{escaped_spill}'")
            connection.execute(f"SET threads = {self.threads}")
            paths = [str(snapshot_paths[relative]) for relative in node.paths]
            relation = connection.read_parquet(paths)
            relation.project(
                f"{_quote_identifier(node.id_column)} AS external_id"
            ).create_view("source_ids")
            row_count, non_null_count, distinct_count = connection.execute(
                "SELECT COUNT(*), COUNT(external_id), "
                "COUNT(DISTINCT external_id) FROM source_ids"
            ).fetchone()
            if non_null_count != row_count:
                raise ArtifactValidationError(
                    f"ID-NULL-001: node type {node.name!r} contains "
                    f"{row_count - non_null_count} null external IDs"
                )
            if distinct_count != row_count:
                raise ArtifactValidationError(
                    f"ID-DUPLICATE-001: node type {node.name!r} has duplicate "
                    "external IDs, including across fragments"
                )
            order_expression = (
                "encode(external_id)" if node.id_dtype == "string" else "external_id"
            )
            connection.execute(
                "CREATE TABLE mapping AS SELECT external_id, "
                f"CAST(row_number() OVER (ORDER BY {order_expression}) - 1 AS BIGINT) "
                "AS local_ordinal FROM source_ids"
            )
            escaped_parquet = str(temp_parquet).replace("'", "''")
            row_group_size = max(2048, self.source.spec.ingestion.record_batch_rows)
            connection.execute(
                "COPY (SELECT local_ordinal, external_id FROM mapping "
                "ORDER BY local_ordinal) "
                f"TO '{escaped_parquet}' (FORMAT PARQUET, COMPRESSION ZSTD, "
                f"ROW_GROUP_SIZE {row_group_size})"
            )
            connection.execute("CHECKPOINT")
        finally:
            connection.close()
        os.replace(temp_database, final_database)
        os.replace(temp_parquet, final_parquet)
        _fsync_directory(root)
        completion_path = root / "mapping.complete.json"
        _atomic_json(
            completion_path,
            {
                "stage": "external_node_index",
                "node_type": node.name,
                "internal_key": internal_key,
                "id_dtype": node.id_dtype,
                "row_count": row_count,
                "behavior_version": _BEHAVIOR_VERSION,
                "input_fingerprint": inventory.source_fingerprint,
                "config_fingerprint": inventory.config_fingerprint,
                "dependency_versions": dict(inventory.dependency_versions),
                "outputs": {
                    "lookup.duckdb": _sha256_file(final_database),
                    "node_ids.parquet": _sha256_file(final_parquet),
                },
            },
        )
        return ExternalNodeIndex(
            node_type=node.name,
            id_dtype=node.id_dtype,
            row_count=row_count,
            root=root,
        )

    def _recover_incomplete_array_artifacts(
        self,
        stage_root: Path,
        *,
        completed_stage: str,
    ) -> bool:
        """Quarantine only uncommitted Task 3 paths under the shared lock."""
        if completed_stage != "external_node_indexes":
            return False
        candidates: list[Path] = []
        arrays_root = stage_root / "arrays"
        pending_complete = (
            not arrays_root.is_symlink()
            and arrays_root.is_dir()
            and not (arrays_root / "arrays.complete.json").is_symlink()
            and (arrays_root / "arrays.complete.json").is_file()
        )
        if os.path.lexists(arrays_root) and not pending_complete:
            candidates.append(arrays_root)
        for relative in (
            "nodes",
            "splits",
            "arrays.json",
            "arrays.complete.json",
        ):
            candidate = stage_root / relative
            if os.path.lexists(candidate):
                candidates.append(candidate)
        candidates.extend(
            path
            for path in stage_root.iterdir()
            if path.name.startswith(".arrays-tmp-")
        )
        if not candidates:
            return pending_complete
        quarantine = stage_root.parent / (
            f".{stage_root.name}.arrays-quarantine-{uuid.uuid4().hex}"
        )
        quarantine.mkdir(parents=False, exist_ok=False)
        for candidate in candidates:
            os.replace(candidate, quarantine / candidate.name)
        _fsync_directory(quarantine)
        _fsync_directory(stage_root)
        _fsync_directory(stage_root.parent)
        return pending_complete


    def _recover_incomplete_relation_artifacts(
        self,
        stage_root: Path,
    ) -> bool:
        """Quarantine uncommitted Task 4 paths while preserving exact indexes."""
        candidates: list[Path] = []
        relations_root = stage_root / "relations"
        pending_complete = (
            not relations_root.is_symlink()
            and relations_root.is_dir()
            and not (relations_root / "relations.complete.json").is_symlink()
            and (relations_root / "relations.complete.json").is_file()
        )
        if os.path.lexists(relations_root) and not pending_complete:
            candidates.append(relations_root)
        candidates.extend(
            path
            for path in stage_root.iterdir()
            if path.name.startswith(".relations-tmp-")
        )
        if not candidates:
            return pending_complete
        quarantine = stage_root.parent / (
            f".{stage_root.name}.relations-quarantine-{uuid.uuid4().hex}"
        )
        quarantine.mkdir(parents=False, exist_ok=False)
        for candidate in candidates:
            os.replace(candidate, quarantine / candidate.name)
        _fsync_directory(quarantine)
        _fsync_directory(stage_root)
        _fsync_directory(stage_root.parent)
        return pending_complete

    def _resume(
        self,
        inventory: SourceInventory,
        stage_root: Path,
        *,
        pa: Any,
        pq: Any,
        duckdb: Any,
    ) -> ExternalNodeIndexBuild:
        completion_path = stage_root / "build.complete.json"
        if not completion_path.is_file():
            raise ArtifactValidationError(
                "INCOMPLETE-ARTIFACT-001: staging directory lacks an atomic "
                "build completion record"
            )
        completion = _read_json(completion_path)
        _validate_common_evidence(completion, inventory)
        completed_stage = completion.get("stage")
        if completed_stage not in {
            "external_node_indexes",
            "typed_graph_arrays",
            "typed_graph_relations",
        }:
            raise ArtifactValidationError(
                "COMPLETION-EVIDENCE-001: unknown completed stage"
            )
        pending_complete_arrays = self._recover_incomplete_array_artifacts(
            stage_root,
            completed_stage=completed_stage,
        )
        pending_complete_relations = (
            self._recover_incomplete_relation_artifacts(stage_root)
        )
        if completion.get("disk_admission") != _disk_admission_record(inventory):
            raise ArtifactValidationError(
                "COMPLETION-EVIDENCE-001: filesystem admission evidence changed"
            )
        expected_work_parent = (
            inventory.temporary_filesystem_path
            / ".topobench-typed-graph-work"
            / stage_root.name
        )
        for field_name, prefix in (
            ("spill_subtree", "build-spill-"),
            ("snapshot_subtree", "source-snapshot-"),
        ):
            value = completion.get(field_name)
            if not isinstance(value, str):
                raise ArtifactValidationError(
                    f"COMPLETION-EVIDENCE-001: {field_name} evidence is missing"
                )
            subtree = Path(value)
            if (
                not subtree.is_absolute()
                or subtree.parent != expected_work_parent
                or not subtree.name.startswith(prefix)
                or subtree.exists()
            ):
                raise ArtifactValidationError(
                    f"COMPLETION-EVIDENCE-001: completed {field_name} "
                    "is unsafe or present"
                )
        outputs = _validated_output_map(completion.get("outputs"))
        core_outputs = {
            relative: checksum
            for relative, checksum in outputs.items()
            if not (
                pending_complete_relations
                and relative.startswith("relations/")
            )
        }
        expected_files = set(core_outputs) | {"build.complete.json"}
        observed_files: set[str] = set()
        for path in stage_root.rglob("*"):
            relative = path.relative_to(stage_root).as_posix()
            if (
                relative == "partitions"
                or relative.startswith("partitions/")
                or relative.startswith(".partitions-quarantine-")
                or relative.startswith(".partitions-tmp-")
                or relative.startswith(".pyg-partition-work/")
            ):
                # Task6 owns and fully validates these downstream subtrees
                # under this same content lock. They are not Task2-4 outputs.
                continue
            if (
                pending_complete_relations
                and relative.startswith("relations/")
            ):
                continue
            if path.is_symlink():
                raise ArtifactValidationError(
                    f"UNKNOWN-ARTIFACT-001: symlink in staging: {path}"
                )
            if pending_complete_arrays and relative.startswith("arrays/"):
                continue
            if path.is_file():
                observed_files.add(relative)
        unknown = sorted(observed_files - expected_files)
        missing = sorted(expected_files - observed_files)
        if unknown:
            raise ArtifactValidationError(
                f"UNKNOWN-ARTIFACT-001: undeclared staging artifact {unknown[0]!r}"
            )
        if missing:
            raise ArtifactValidationError(
                f"INCOMPLETE-ARTIFACT-001: missing staging artifact {missing[0]!r}"
            )
        for relative, expected_checksum in core_outputs.items():
            if _sha256_file(_safe_artifact_path(stage_root, relative)) != expected_checksum:
                raise ArtifactValidationError(
                    f"CHECKSUM-001: staging artifact checksum mismatch for {relative!r}"
                )

        index_evidence = completion.get("indexes")
        inventory_record = _read_json(stage_root / "inventory.json")
        if inventory_record != _inventory_record(inventory):
            raise ArtifactValidationError(
                "COMPLETION-EVIDENCE-001: inventory semantic evidence changed"
            )
        inventory_completion = _read_json(stage_root / "inventory.complete.json")
        _validate_common_evidence(inventory_completion, inventory)
        if (
            inventory_completion.get("stage") != "inventory"
            or inventory_completion.get("outputs")
            != {"inventory.json": _sha256_file(stage_root / "inventory.json")}
        ):
            raise ArtifactValidationError(
                "COMPLETION-EVIDENCE-001: inventory completion evidence changed"
            )
        if not isinstance(index_evidence, dict):
            raise ArtifactValidationError(
                "COMPLETION-EVIDENCE-001: index evidence must be an object"
            )
        indexes: dict[str, ExternalNodeIndex] = {}
        node_by_name = {node.name: node for node in self.source.spec.node_types}
        if set(index_evidence) != set(node_by_name):
            raise ArtifactValidationError(
                "COMPLETION-EVIDENCE-001: completed node-type set changed"
            )
        for node_name, evidence in index_evidence.items():
            if not isinstance(evidence, dict):
                raise ArtifactValidationError(
                    "COMPLETION-EVIDENCE-001: malformed node index evidence"
                )
            node = node_by_name[node_name]
            internal_key = evidence.get("internal_key")
            expected_rows = dict(inventory.node_rows)[node_name]
            if (
                not isinstance(internal_key, str)
                or evidence.get("id_dtype") != node.id_dtype
                or evidence.get("row_count") != expected_rows
            ):
                raise ArtifactValidationError(
                    f"COMPLETION-EVIDENCE-001: node evidence changed for {node_name!r}"
                )
            root = _safe_artifact_path(stage_root / "mappings", internal_key)
            index = ExternalNodeIndex(
                node_type=node_name,
                id_dtype=node.id_dtype,
                row_count=expected_rows,
                root=root,
            )
            record = _read_json(index.completion_path)
            _validate_common_evidence(record, inventory)
            if (
                record.get("stage") != "external_node_index"
                or record.get("node_type") != node_name
                or record.get("internal_key") != internal_key
                or record.get("id_dtype") != node.id_dtype
                or record.get("row_count") != expected_rows
            ):
                raise ArtifactValidationError(
                    f"COMPLETION-EVIDENCE-001: invalid per-type record for {node_name!r}"
                )
            per_outputs = _validated_output_map(record.get("outputs"))
            if set(per_outputs) != {"lookup.duckdb", "node_ids.parquet"}:
                raise ArtifactValidationError(
                    f"COMPLETION-EVIDENCE-001: invalid outputs for {node_name!r}"
                )
            for relative, checksum in per_outputs.items():
                if _sha256_file(_safe_artifact_path(root, relative)) != checksum:
                    raise ArtifactValidationError(
                        f"CHECKSUM-001: per-type checksum mismatch for {node_name!r}"
                    )
            parquet_file = pq.ParquetFile(index.node_ids_path)
            schema = parquet_file.schema_arrow
            expected_arrow_type = {
                "int64": pa.int64(),
                "uint64": pa.uint64(),
                "string": pa.string(),
            }[node.id_dtype]
            if (
                parquet_file.metadata.num_rows != expected_rows
                or schema.names != ["local_ordinal", "external_id"]
                or not schema.field("local_ordinal").type.equals(pa.int64())
                or not schema.field("external_id").type.equals(expected_arrow_type)
            ):
                raise ArtifactValidationError(
                    f"INDEX-BIJECTION-001: invalid node_ids Parquet for {node_name!r}"
                )
            audit_spill = self._new_ephemeral_root(
                inventory,
                purpose="resume-audit",
            )
            audit_spill.mkdir(parents=True, exist_ok=False)
            connection = None
            try:
                connection = duckdb.connect(
                    str(index.lookup_path),
                    read_only=True,
                )
                memory_bytes = self.source.spec.ingestion.memory_limit_bytes
                escaped_spill = str(audit_spill).replace("'", "''")
                connection.execute("SET preserve_insertion_order = false")
                connection.execute(f"SET memory_limit = '{memory_bytes}B'")
                connection.execute(f"SET temp_directory = '{escaped_spill}'")
                connection.execute(f"SET threads = {self.threads}")
                counts = connection.execute(
                    "SELECT COUNT(*), COUNT(external_id), "
                    "COUNT(DISTINCT external_id), COUNT(DISTINCT local_ordinal), "
                    "MIN(local_ordinal), MAX(local_ordinal) FROM mapping"
                ).fetchone()
                table_info = connection.execute(
                    "PRAGMA table_info('mapping')"
                ).fetchall()
                escaped_reverse = str(index.node_ids_path).replace("'", "''")
                reverse_mismatches = connection.execute(
                    "SELECT COUNT(*) FROM "
                    f"read_parquet('{escaped_reverse}') AS reverse "
                    "FULL OUTER JOIN mapping "
                    "ON reverse.local_ordinal = mapping.local_ordinal "
                    "WHERE reverse.local_ordinal IS NULL "
                    "OR mapping.local_ordinal IS NULL "
                    "OR reverse.external_id IS DISTINCT FROM mapping.external_id"
                ).fetchone()[0]
                order_expression = (
                    "encode(external_id)"
                    if node.id_dtype == "string"
                    else "external_id"
                )
                canonical_mismatches = connection.execute(
                    "SELECT COUNT(*) FROM ("
                    "SELECT local_ordinal, "
                    f"CAST(row_number() OVER (ORDER BY {order_expression}) - 1 "
                    "AS BIGINT) AS expected_ordinal FROM mapping"
                    ") WHERE local_ordinal != expected_ordinal"
                ).fetchone()[0]
            finally:
                if connection is not None:
                    connection.close()
                if audit_spill.exists():
                    shutil.rmtree(audit_spill)
            expected_sql_type = {
                "int64": "BIGINT",
                "uint64": "UBIGINT",
                "string": "VARCHAR",
            }[node.id_dtype]
            observed_sql_types = {
                row[1]: row[2].upper()
                for row in table_info
            }
            if observed_sql_types != {
                "external_id": expected_sql_type,
                "local_ordinal": "BIGINT",
            }:
                raise ArtifactValidationError(
                    f"INDEX-BIJECTION-001: lookup SQL types changed for {node_name!r}"
                )
            expected_bounds = (
                (None, None)
                if expected_rows == 0
                else (0, expected_rows - 1)
            )
            if counts[:4] != (expected_rows,) * 4 or counts[4:] != expected_bounds:
                raise ArtifactValidationError(
                    f"INDEX-BIJECTION-001: lookup state is not a bijection for {node_name!r}"
                )
            if reverse_mismatches:
                raise ArtifactValidationError(
                    f"INDEX-REVERSE-001: node_ids Parquet disagrees with lookup "
                    f"mapping for {node_name!r}"
                )
            if canonical_mismatches:
                raise ArtifactValidationError(
                    f"INDEX-CANONICAL-001: dense ordinals do not match canonical "
                    f"{node.id_dtype} ordering for {node_name!r}"
                )
            indexes[node_name] = index
        return ExternalNodeIndexBuild(
            inventory=inventory,
            stage_root=stage_root,
            indexes=indexes,
            resumed=True,
        )

    @contextmanager
    def _build_lock(self, path: Path) -> Iterator[None]:
        path.parent.mkdir(parents=True, exist_ok=True)
        token = uuid.uuid4().hex
        record = {
            "pid": os.getpid(),
            "hostname": socket.gethostname(),
            "created_ns": time.time_ns(),
            "token": token,
        }
        for attempt in range(2):
            try:
                descriptor = os.open(path, os.O_WRONLY | os.O_CREAT | os.O_EXCL, 0o600)
            except FileExistsError:
                if attempt or not self._remove_stale_lock(path):
                    raise ConcurrentBuildError(
                        f"BUILD-LOCK-001: active or unverifiable build owner at {path}"
                    ) from None
            else:
                with os.fdopen(descriptor, "w", encoding="utf-8") as stream:
                    json.dump(record, stream, sort_keys=True, separators=(",", ":"))
                    stream.flush()
                    os.fsync(stream.fileno())
                break
        try:
            yield
        finally:
            try:
                observed = _read_json(path)
            except (ArtifactValidationError, FileNotFoundError):
                observed = {}
            if observed.get("token") == token:
                path.unlink()
                _fsync_directory(path.parent)

    def _remove_stale_lock(self, path: Path) -> bool:
        try:
            inspected = path.stat(follow_symlinks=False)
            record = _read_json(path)
            pid = record["pid"]
            hostname = record["hostname"]
            created_ns = record["created_ns"]
            if (
                isinstance(pid, bool)
                or not isinstance(pid, int)
                or pid <= 0
                or not isinstance(hostname, str)
                or isinstance(created_ns, bool)
                or not isinstance(created_ns, int)
            ):
                return False
            age_seconds = (time.time_ns() - created_ns) / 1_000_000_000
            if age_seconds <= self.lock_stale_seconds:
                return False
            if hostname == socket.gethostname() and _pid_is_alive(pid):
                return False
            current = path.stat(follow_symlinks=False)
            inspected_identity = (
                inspected.st_dev,
                inspected.st_ino,
                inspected.st_size,
                inspected.st_mtime_ns,
                inspected.st_ctime_ns,
            )
            current_identity = (
                current.st_dev,
                current.st_ino,
                current.st_size,
                current.st_mtime_ns,
                current.st_ctime_ns,
            )
            if current_identity != inspected_identity:
                return False
            path.unlink()
            _fsync_directory(path.parent)
            return True
        except (FileNotFoundError, KeyError, OSError, ArtifactValidationError):
            return False


def _parquet_dependencies() -> tuple[Any, Any, Any]:
    try:
        import pyarrow as pa
        import pyarrow.parquet as pq
        import duckdb
    except ImportError as error:
        raise RuntimeError(
            "Typed Parquet ingestion requires the optional pyarrow and duckdb dependencies"
        ) from error
    return pa, pq, duckdb


def _dependency_versions(pa: Any) -> tuple[tuple[str, str], ...]:
    import duckdb

    return (("duckdb", duckdb.__version__), ("pyarrow", pa.__version__))


def _copy_inventory_file(entry: FileInventory, destination: Path) -> None:
    source = entry.absolute_path
    if source.resolve(strict=True) != source or source.is_symlink():
        raise SourceMutationError(
            f"SOURCE-MUTATION-001: source path changed for {entry.relative_path!r}"
        )
    digest = hashlib.sha256()
    destination_created = False
    try:
        with source.open("rb") as reader:
            descriptor = os.open(
                destination,
                os.O_WRONLY | os.O_CREAT | os.O_EXCL,
                0o600,
            )
            destination_created = True
            with os.fdopen(descriptor, "wb") as writer:
                before = os.fstat(reader.fileno())
                copied = 0
                while chunk := reader.read(_HASH_CHUNK_BYTES):
                    writer.write(chunk)
                    digest.update(chunk)
                    copied += len(chunk)
                writer.flush()
                os.fsync(writer.fileno())
                after = os.fstat(reader.fileno())
    except BaseException:
        if destination_created:
            try:
                destination.unlink()
            except FileNotFoundError:
                pass
        raise
    source_identity_before = (
        before.st_dev,
        before.st_ino,
        before.st_size,
        before.st_mtime_ns,
        before.st_ctime_ns,
    )
    source_identity_after = (
        after.st_dev,
        after.st_ino,
        after.st_size,
        after.st_mtime_ns,
        after.st_ctime_ns,
    )
    if (
        source_identity_before != source_identity_after
        or copied != entry.byte_size
        or digest.hexdigest() != entry.sha256
        or destination.stat().st_size != entry.byte_size
    ):
        raise SourceMutationError(
            f"SOURCE-MUTATION-001: snapshot copy differs for "
            f"{entry.relative_path!r}"
        )
    destination.chmod(0o400)


def _filesystem_capacity(path: Path) -> tuple[int, int, Path]:
    probe = path
    while not probe.exists() and probe != probe.parent:
        probe = probe.parent
    status = probe.stat()
    return status.st_dev, shutil.disk_usage(probe).free, probe


def _disk_admission_record(inventory: SourceInventory) -> dict[str, Any]:
    return {
        "final_filesystem_path": str(inventory.final_filesystem_path),
        "final_device": inventory.final_device,
        "temporary_filesystem_path": str(inventory.temporary_filesystem_path),
        "temporary_device": inventory.temporary_device,
        "estimated_final_bytes": inventory.estimated_final_bytes,
        "estimated_temporary_bytes": inventory.estimated_temporary_bytes,
        "required_peak_bytes": inventory.required_peak_bytes,
        "snapshot_bytes": inventory.snapshot_bytes,
    }


def _normalized_dataclass(value: Any) -> Any:
    if dataclasses.is_dataclass(value):
        return {
            field.name: _normalized_dataclass(getattr(value, field.name))
            for field in dataclasses.fields(value)
            if not field.name.startswith("_")
        }
    if isinstance(value, Path):
        return value.as_posix()
    if isinstance(value, tuple):
        return [_normalized_dataclass(item) for item in value]
    if isinstance(value, list):
        return [_normalized_dataclass(item) for item in value]
    if isinstance(value, dict):
        return {
            str(key): _normalized_dataclass(item)
            for key, item in sorted(value.items(), key=lambda pair: str(pair[0]))
        }
    return value


def _quote_identifier(value: str) -> str:
    return '"' + value.replace('"', '""') + '"'


def _sha256_json(value: Any) -> str:
    encoded = json.dumps(
        value,
        sort_keys=True,
        separators=(",", ":"),
        ensure_ascii=False,
    ).encode("utf-8")
    return hashlib.sha256(encoded).hexdigest()


def _sha256_file(path: Path) -> str:
    digest = hashlib.sha256()
    with path.open("rb") as stream:
        while chunk := stream.read(_HASH_CHUNK_BYTES):
            digest.update(chunk)
    return digest.hexdigest()


def _file_record(entry: FileInventory) -> dict[str, Any]:
    return {
        "relative_path": entry.relative_path,
        "byte_size": entry.byte_size,
        "row_count": entry.row_count,
        "uncompressed_bytes": entry.uncompressed_bytes,
        "sha256": entry.sha256,
        "schema_fingerprint": entry.schema_fingerprint,
        "schema_serialized_hex": entry.schema_serialized_hex,
    }


def _inventory_record(inventory: SourceInventory) -> dict[str, Any]:
    return {
        "behavior_version": _BEHAVIOR_VERSION,
        "files": [_file_record(entry) for entry in inventory.files],
        "total_bytes": inventory.total_bytes,
        "total_rows": inventory.total_rows,
        "node_rows": dict(inventory.node_rows),
        "snapshot_bytes": inventory.snapshot_bytes,
        "estimated_final_bytes": inventory.estimated_final_bytes,
        "estimated_temporary_bytes": inventory.estimated_temporary_bytes,
        "required_peak_bytes": inventory.required_peak_bytes,
        "source_fingerprint": inventory.source_fingerprint,
        "config_fingerprint": inventory.config_fingerprint,
        "dependency_versions": dict(inventory.dependency_versions),
        "final_filesystem_path": str(inventory.final_filesystem_path),
        "temporary_filesystem_path": str(inventory.temporary_filesystem_path),
        "final_device": inventory.final_device,
        "temporary_device": inventory.temporary_device,
    }


def _atomic_json(path: Path, value: Any) -> None:
    path.parent.mkdir(parents=True, exist_ok=True)
    temporary = path.parent / f".{path.name}.tmp-{uuid.uuid4().hex}"
    descriptor = os.open(temporary, os.O_WRONLY | os.O_CREAT | os.O_EXCL, 0o600)
    try:
        with os.fdopen(descriptor, "w", encoding="utf-8") as stream:
            json.dump(
                value,
                stream,
                sort_keys=True,
                separators=(",", ":"),
                ensure_ascii=False,
            )
            stream.write("\n")
            stream.flush()
            os.fsync(stream.fileno())
        os.replace(temporary, path)
        _fsync_directory(path.parent)
    except BaseException:
        try:
            temporary.unlink()
        except FileNotFoundError:
            pass
        raise


def _read_json(path: Path) -> dict[str, Any]:
    if path.is_symlink() or not path.is_file():
        raise ArtifactValidationError(f"COMPLETION-EVIDENCE-001: unsafe record {path}")
    try:
        value = json.loads(path.read_text(encoding="utf-8"))
    except (OSError, UnicodeError, json.JSONDecodeError) as error:
        raise ArtifactValidationError(
            f"COMPLETION-EVIDENCE-001: unreadable record {path}"
        ) from error
    if not isinstance(value, dict):
        raise ArtifactValidationError(
            f"COMPLETION-EVIDENCE-001: record is not an object: {path}"
        )
    return value


def _validated_output_map(value: Any) -> dict[str, str]:
    if not isinstance(value, dict) or not value:
        raise ArtifactValidationError(
            "COMPLETION-EVIDENCE-001: output checksums must be a non-empty object"
        )
    result: dict[str, str] = {}
    for relative, checksum in value.items():
        if (
            not isinstance(relative, str)
            or not isinstance(checksum, str)
            or len(checksum) != 64
        ):
            raise ArtifactValidationError(
                "COMPLETION-EVIDENCE-001: malformed output checksum evidence"
            )
        _validate_relative_artifact(relative)
        result[relative] = checksum
    return result


def _validate_common_evidence(record: Mapping[str, Any], inventory: SourceInventory) -> None:
    if (
        record.get("behavior_version") != _BEHAVIOR_VERSION
        or record.get("input_fingerprint") != inventory.source_fingerprint
        or record.get("config_fingerprint") != inventory.config_fingerprint
        or record.get("dependency_versions") != dict(inventory.dependency_versions)
    ):
        raise ArtifactValidationError(
            "COMPLETION-EVIDENCE-001: completion input/config/dependency evidence changed"
        )


def _validate_relative_artifact(relative: str) -> PurePosixPath:
    path = PurePosixPath(relative)
    if path.is_absolute() or not path.parts or any(part in {"", ".", ".."} for part in path.parts):
        raise ArtifactValidationError(
            f"COMPLETION-EVIDENCE-001: unsafe artifact path {relative!r}"
        )
    return path


def _safe_artifact_path(root: Path, relative: str) -> Path:
    parsed = _validate_relative_artifact(relative)
    candidate = root.joinpath(*parsed.parts)
    resolved_parent = candidate.parent.resolve(strict=True)
    resolved_root = root.resolve(strict=True)
    if not resolved_parent.is_relative_to(resolved_root):
        raise ArtifactValidationError(
            f"COMPLETION-EVIDENCE-001: artifact escapes staging {relative!r}"
        )
    return candidate


def _stage_output_checksums(stage_root: Path) -> dict[str, str]:
    outputs: dict[str, str] = {}
    for path in sorted(stage_root.rglob("*")):
        if path.is_symlink():
            raise ArtifactValidationError(
                f"UNKNOWN-ARTIFACT-001: symlink produced in staging: {path}"
            )
        if path.is_file():
            if path.name == "build.complete.json":
                continue
            relative = path.relative_to(stage_root).as_posix()
            outputs[relative] = _sha256_file(path)
    return outputs


def _fsync_directory(path: Path) -> None:
    descriptor = os.open(path, os.O_RDONLY)
    try:
        os.fsync(descriptor)
    finally:
        os.close(descriptor)


def _pid_is_alive(pid: int) -> bool:
    try:
        os.kill(pid, 0)
    except ProcessLookupError:
        return False
    except PermissionError:
        return True
    return True


__all__ = [
    "ArtifactValidationError",
    "ConcurrentBuildError",
    "DiskAdmissionError",
    "ExternalNodeIndexBuild",
    "FileInventory",
    "ParquetTypedGraphIngestor",
    "SourceInventory",
    "SourceMutationError",
]
