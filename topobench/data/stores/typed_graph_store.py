"""Atomic finalization and lazy access for immutable typed graph stores."""

from __future__ import annotations

import hashlib
import json
import os
import platform
import shutil
import stat
import sys
import uuid
from dataclasses import dataclass
from pathlib import Path
from types import MappingProxyType
from typing import TYPE_CHECKING, Any, Mapping, Sequence

import numpy as np

from topobench.data.stores.qualification_checks import (
    CONTENT_HASH_VERSION,
    REPORT_FORMAT_VERSION,
    STORE_FORMAT_VERSION,
    QualificationFailure,
    QualificationCheckResult,
    QualificationReport,
    ValidatedStore,
    _open_npy,
    _secure_descriptor,
    compute_content_identity,
    compute_metadata_binding,
    split_fingerprint,
    qualification_check_set_fingerprint,
    validate_store,
)

if TYPE_CHECKING:
    from topobench.data.stores.pyg_partitioner import TypedPartitionBuild
    from topobench.data.stores.typed_graph_ingestion import ParquetTypedGraphIngestor

_PHASES = ("train", "val", "test")


@dataclass(frozen=True, slots=True)
class TypedGraphStoreState:
    """Serializable qualification state without open files or connections."""

    root: Path
    manifest_json: bytes
    report_json: bytes
    file_identities: tuple[tuple[str, tuple[int, int, int, int, int]], ...]


@dataclass(frozen=True, slots=True)
class TypedGraphStoreBuild:
    """One validated content-addressed promotion result."""

    path: Path
    content_sha256: str
    cache_hit: bool
    quarantine_path: Path | None
    qualification_report: QualificationReport
    store: TypedGraphStore


class TypedGraphStoreWriter:
    """Finalize one exact Task1-6 build without mutating its recoverable stage."""

    def __init__(
        self,
        ingestor: ParquetTypedGraphIngestor,
        partition_build: TypedPartitionBuild,
        *,
        task_bindings: Mapping[str, Any] | None = None,
        execution_monitor: object | None = None,
    ) -> None:
        self.ingestor = ingestor
        self.partition_build = partition_build
        self.task_bindings = _normalized_json(task_bindings or {})
        if execution_monitor is not None and not callable(
            getattr(execution_monitor, "record_qualification", None)
        ):
            raise TypeError(
                "execution_monitor must expose "
                "record_qualification(result, report_path)"
            )
        self.execution_monitor = execution_monitor

    def build(self) -> TypedGraphStoreBuild:
        """Reopen Task1-6 under its lock, validate, and atomically promote."""
        inventory = self.partition_build.inventory
        expected_stage = self.ingestor.stage_root(inventory)
        if (
            self.partition_build.stage_root != expected_stage
            or self.partition_build.artifact_root != expected_stage / "partitions"
        ):
            raise _artifact_error(
                "STALE-BINDING-001: Task6 build does not belong to this ingestor"
            )
        staging_parent = self.ingestor.store_root / ".staging"
        staging_parent.mkdir(parents=True, exist_ok=True)
        candidate = staging_parent / f"finalize-{uuid.uuid4().hex}"
        quarantine: Path | None = None
        candidate.mkdir(parents=False, exist_ok=False)
        try:
            with self.ingestor._build_lock(self.ingestor.lock_path(inventory)):
                reopened = self._reopen_task1_6_locked()
                self._materialize_candidate(candidate, reopened)
            manifest_path = candidate / "manifest.json"
            manifest = _read_json(manifest_path)
            content_sha256 = compute_content_identity(manifest)
            manifest["content_sha256"] = content_sha256
            _write_json(manifest_path, manifest)
            validated = validate_store(
                candidate,
                expected_bindings=self.task_bindings,
                require_directory_identity=False,
            )
            target = self.ingestor.store_root / content_sha256
            if not _same_filesystem(candidate, self.ingestor.store_root):
                raise _artifact_error(
                    "PROMOTION-FILESYSTEM-001: final staging is not on the store filesystem"
                )
            if os.path.lexists(target):
                try:
                    existing = validate_store(
                        target,
                        expected_bindings=self.task_bindings,
                        require_directory_identity=True,
                        execution_monitor=self.execution_monitor,
                    )
                except QualificationFailure:
                    if target.is_symlink() or not target.is_dir():
                        raise _artifact_error(
                            "UNSAFE-STORE-001: colliding final path is not a real directory"
                        )
                    quarantine = self.ingestor.store_root / (
                        f".quarantine-{content_sha256}-{uuid.uuid4().hex}"
                    )
                    target.chmod(0o755)
                    try:
                        os.replace(target, quarantine)
                    except BaseException:
                        target.chmod(0o555)
                        raise
                    quarantine.chmod(0o555)
                    _fsync_directory(self.ingestor.store_root)
                else:
                    shutil.rmtree(candidate)
                    return TypedGraphStoreBuild(
                        target,
                        content_sha256,
                        True,
                        None,
                        existing.report,
                        TypedGraphStore(existing),
                    )
            _make_read_only(candidate, movable_root=True)
            _fsync_tree(candidate)
            os.replace(candidate, target)
            target.chmod(0o555)
            _fsync_directory(target)
            _fsync_directory(self.ingestor.store_root)
            promoted = validate_store(
                target,
                expected_bindings=self.task_bindings,
                require_directory_identity=True,
                execution_monitor=self.execution_monitor,
            )
            return TypedGraphStoreBuild(
                target,
                content_sha256,
                False,
                quarantine,
                promoted.report,
                TypedGraphStore(promoted),
            )
        finally:
            if candidate.exists():
                shutil.rmtree(candidate, ignore_errors=True)

    def _reopen_task1_6_locked(self) -> dict[str, Any]:
        from topobench.data.stores import pyg_partitioner as partitioner_module
        from topobench.data.stores import typed_graph_ingestion as ingestion_module
        from topobench.data.stores.pyg_partitioner import TopologyOnlyPyGPartitioner
        from topobench.data.stores.typed_graph_arrays import TypedGraphArrayWriter
        from topobench.data.stores.typed_graph_csc import TypedGraphRelationWriter

        inventory = self.partition_build.inventory
        pa, pq, duckdb = ingestion_module._parquet_dependencies()
        self.ingestor._validate_inventory_current(inventory, pa=pa, pq=pq)
        indexes = self.ingestor._resume(
            inventory,
            self.partition_build.stage_root,
            pa=pa,
            pq=pq,
            duckdb=duckdb,
        )
        arrays = TypedGraphArrayWriter(self.ingestor, indexes)._open_validated(
            resumed=True
        )
        relations = TypedGraphRelationWriter(
            self.ingestor,
            indexes,
            arrays,
        )._open_validated(
            resumed=True,
            pa=pa,
            pq=pq,
            duckdb=duckdb,
        )
        partitioner = TopologyOnlyPyGPartitioner(self.ingestor, relations)
        assignments, identity, limits = partitioner_module._read_subtree(
            partitioner.topology_context,
            self.partition_build.artifact_root,
        )
        book = partitioner_module._qualified_book(
            partitioner.topology_context,
            assignments,
            limits,
            backend=identity["backend"],
            estimated_resources=identity["estimated_resources"],
            measured_resources=identity["measured_resources"],
        )
        if book.content_identity != self.partition_build.book.content_identity:
            raise _artifact_error(
                "PARTITION-FINGERPRINT-001: Task6 identity changed before finalization"
            )
        return {
            "indexes": indexes,
            "arrays": arrays,
            "relations": relations,
            "partitioner": partitioner,
            "book": book,
            "partition_identity": identity,
            "limits": limits,
        }

    def _materialize_candidate(self, root: Path, reopened: Mapping[str, Any]) -> None:
        stage = self.partition_build.stage_root
        arrays_metadata = _read_json(stage / "arrays" / "arrays.json")
        relation_metadata = _read_json(stage / "relations" / "relations.json")
        book = reopened["book"]
        files: list[dict[str, Any]] = []
        nodes: dict[str, dict[str, Any]] = {}
        node_name_to_key: dict[str, str] = {}
        for key, source_record in arrays_metadata["nodes"].items():
            node_name = source_record["node_type"]
            node_name_to_key[node_name] = key
            x_relative = f"nodes/{key}/x.npy"
            self._copy_file(
                stage / "arrays" / source_record["relative_path"],
                root / x_relative,
            )
            x = _array_file_record(
                root / x_relative,
                x_relative,
                f"node-feature:{key}",
                finite=True,
            )
            files.append(x)
            mapping_record = _read_json(
                stage / "mappings" / key / "mapping.complete.json"
            )
            ids_relative = f"nodes/{key}/node_ids.parquet"
            self._copy_file(
                stage / "mappings" / key / "node_ids.parquet",
                root / ids_relative,
            )
            node_ids = _plain_file_record(
                root / ids_relative,
                ids_relative,
                f"external-node-ids:{key}",
                dtype=f"parquet:{mapping_record['id_dtype']}",
                shape=[mapping_record["row_count"], 2],
            )
            files.append(node_ids)
            nodes[key] = {
                "internal_key": key,
                "name": node_name,
                "id_dtype": source_record["id_dtype"],
                "count": source_record["count"],
                "feature_width": source_record["feature_width"],
                "x": x,
                "y": None,
                "node_ids": node_ids,
                "fields": {},
            }
        supervision = arrays_metadata["supervision"]
        target_key = supervision["target_internal_key"]
        y_relative = f"nodes/{target_key}/y.npy"
        self._copy_file(
            stage / "arrays" / supervision["relative_path"],
            root / y_relative,
        )
        y = _array_file_record(
            root / y_relative,
            y_relative,
            f"node-label:{target_key}",
            finite=supervision["task"] == "regression",
        )
        files.append(y)
        nodes[target_key]["y"] = y

        splits: dict[str, dict[str, Any]] = {}
        for tag, source_split in arrays_metadata["splits"].items():
            phases: dict[str, dict[str, Any]] = {}
            phase_arrays: dict[str, np.ndarray] = {}
            for phase in _PHASES:
                source_phase = source_split["phases"][phase]
                relative = f"splits/{tag}/{phase}_ids.npy"
                self._copy_file(
                    stage / "arrays" / source_phase["relative_path"],
                    root / relative,
                )
                record = _array_file_record(
                    root / relative,
                    relative,
                    f"split:{tag}:{phase}",
                )
                files.append(record)
                phases[phase] = record
                phase_arrays[phase] = np.load(
                    root / relative,
                    mmap_mode="r",
                    allow_pickle=False,
                )
            split = {
                "coverage": source_split["coverage"],
                "qualified": source_split["qualified"],
                "supervision_population": source_split["supervision_population"],
                "phases": phases,
            }
            split["fingerprint"] = split_fingerprint(tag, split, phase_arrays)
            splits[tag] = split
            phase_arrays.clear()

        relations: dict[str, dict[str, Any]] = {}
        relation_to_key: dict[tuple[str, str, str], str] = {}
        for key, source_relation in relation_metadata["relations"].items():
            triple = tuple(source_relation["relation"])
            relation_to_key[triple] = key
            relation = {
                "internal_key": key,
                "relation": list(triple),
                "source_internal_key": source_relation["source_internal_key"],
                "destination_internal_key": source_relation[
                    "destination_internal_key"
                ],
                "source_count": source_relation["source_count"],
                "destination_count": source_relation["destination_count"],
                "edge_count": source_relation["edge_count"],
                "canonical_order": source_relation["canonical_order"],
                "edge_id": None,
                "fields": {},
            }
            for array_name in ("colptr", "row"):
                source_array = source_relation[array_name]
                relative = f"relations/{key}/{array_name}.npy"
                self._copy_file(
                    stage / "relations" / source_array["relative_path"],
                    root / relative,
                )
                record = _array_file_record(
                    root / relative,
                    relative,
                    f"relation-{array_name}:{key}",
                )
                files.append(record)
                relation[array_name] = record
            if source_relation["edge_id"] is not None:
                source_edge_id = source_relation["edge_id"]
                relative = f"relations/{key}/edge_id.npy"
                self._copy_file(
                    stage / "relations" / source_edge_id["relative_path"],
                    root / relative,
                )
                edge_id = _array_file_record(
                    root / relative,
                    relative,
                    f"relation-edge-id:{key}",
                )
                files.append(edge_id)
                relation["edge_id"] = edge_id
            for field_name, source_field in source_relation["fields"].items():
                field_key = source_field["internal_key"]
                relative = f"relations/{key}/fields/{field_key}.npy"
                self._copy_file(
                    stage / "relations" / source_field["relative_path"],
                    root / relative,
                )
                field = _array_file_record(
                    root / relative,
                    relative,
                    f"relation-field:{key}:{field_key}",
                    finite=np.issubdtype(
                        np.dtype(source_field["storage_dtype"]), np.floating
                    ),
                )
                files.append(field)
                relation["fields"][field_name] = field
            relations[key] = relation

        partition_nodes: dict[str, dict[str, Any]] = {}
        for key, node in nodes.items():
            source_root = stage / "partitions" / "node_types" / key
            records: dict[str, Any] = {}
            for name, role in (
                ("assignment", "partition-assignment"),
                ("permutation", "partition-permutation"),
                ("inverse", "partition-inverse"),
                ("partptr", "partition-partptr"),
            ):
                relative = f"partitions/node_types/{key}/{name}.npy"
                self._copy_file(source_root / f"{name}.npy", root / relative)
                record = _array_file_record(
                    root / relative,
                    relative,
                    f"{role}:{key}",
                )
                files.append(record)
                records[name] = record
            partition_nodes[key] = records

        partition_relations: dict[str, dict[str, Any]] = {}
        for ordinal, triple in enumerate(sorted(relation_to_key)):
            key = relation_to_key[triple]
            relative = f"partitions/relations/{key}/edge_partition.npy"
            self._copy_file(
                stage
                / "partitions"
                / "relations"
                / f"r{ordinal:04d}"
                / "edge_partition.npy",
                root / relative,
            )
            record = _array_file_record(
                root / relative,
                relative,
                f"partition-edge-ownership:{key}",
            )
            files.append(record)
            partition_relations[key] = {"edge_partition": record}

        for source_name, role in (
            ("partition_book.json", "partition-book"),
            ("statistics.json", "partition-statistics"),
        ):
            relative = f"partitions/{source_name}"
            self._copy_file(
                stage / "partitions" / source_name,
                root / relative,
            )
            files.append(
                _plain_file_record(root / relative, relative, role, dtype="json", shape=[])
            )

        partition_identity = reopened["partition_identity"]
        source_binding = {
            "source_fingerprint": self.partition_build.inventory.source_fingerprint,
            "config_fingerprint": self.partition_build.inventory.config_fingerprint,
            "dependency_versions": dict(
                self.partition_build.inventory.dependency_versions
            ),
            "task3_content_sha256": arrays_metadata["content_sha256"],
            "task4_content_sha256": self.partition_build.book.source_binding[
                "task4_content_sha256"
            ],
            "task6_source_binding": _normalized_json(book.source_binding),
            "partition_book_identity": book.content_identity,
        }
        producer_checks = [
            {
                "check_id": check.check_id,
                "passed": check.passed,
                "observed": _normalized_json(check.observed),
                "expected": _normalized_json(check.limit),
                "limit": _normalized_json(check.limit),
                "evidence": {"detail": check.detail},
                "remediation": "none" if check.passed else "repartition",
            }
            for check in book.qualification_checks
        ]
        qualification_check_set = {
            "count": len(producer_checks),
            "sha256": qualification_check_set_fingerprint(producer_checks),
        }
        manifest = {
            "format_version": STORE_FORMAT_VERSION,
            "content_hash_version": CONTENT_HASH_VERSION,
            "content_sha256": "",
            "metadata_binding_sha256": "",
            "qualification_check_set": qualification_check_set,
            "output_kind": self.ingestor.source.spec.output_kind,
            "target_node_type": supervision["target_node_type"],
            "target_internal_key": target_key,
            "active_split_tag": arrays_metadata["active_split_tag"],
            "supported_capabilities": [
                "homogeneous-cluster",
                "heterogeneous-cluster",
                "heterogeneous-neighbor",
                "pyg-feature-store",
                "pyg-graph-store-csc",
            ],
            "task_bindings": self.task_bindings,
            "source_binding": source_binding,
            "nodes": nodes,
            "relations": relations,
            "splits": splits,
            "partition": {
                "format_version": partition_identity["format_version"],
                "num_partitions": book.num_partitions,
                "topology_fingerprint": book.topology_fingerprint,
                "source_binding": _normalized_json(book.source_binding),
                "backend": book.backend,
                "backend_version": book.backend_version,
                "options": _normalized_json(book.options),
                "content_identity": book.content_identity,
                "node_types": partition_nodes,
                "relations": partition_relations,
            },
            "build_environment": "build_environment.json",
            "qualification_report": "qualification_report.json",
            "files": [],
        }
        metadata_binding = compute_metadata_binding(manifest)
        manifest["metadata_binding_sha256"] = metadata_binding
        split_fingerprints = {
            tag: split["fingerprint"] for tag, split in splits.items()
        }
        environment = _build_environment(
            self.ingestor,
            book,
            metadata_binding=metadata_binding,
            task_bindings=self.task_bindings,
            split_fingerprints=split_fingerprints,
            dependency_versions=source_binding["dependency_versions"],
        )
        _write_json(root / "build_environment.json", environment)
        files.append(
            _plain_file_record(
                root / "build_environment.json",
                "build_environment.json",
                "build-environment",
                dtype="json",
                shape=[],
            )
        )
        producer_report = {
            "format_version": REPORT_FORMAT_VERSION,
            "passed": True,
            "report_path": "qualification_report.json",
            "metadata_binding_sha256": metadata_binding,
            "task_bindings": self.task_bindings,
            "partition_book_identity": book.content_identity,
            "split_fingerprints": split_fingerprints,
            "checks": producer_checks,
        }
        _write_json(root / "qualification_report.json", producer_report)
        files.append(
            _plain_file_record(
                root / "qualification_report.json",
                "qualification_report.json",
                "qualification-report",
                dtype="json",
                shape=[],
            )
        )
        manifest["files"] = sorted(
            files,
            key=lambda record: record["relative_path"],
        )
        _write_json(root / "manifest.json", manifest)
        _fsync_tree(root)

    def _copy_file(self, source: Path, destination: Path) -> None:
        """Copy one pinned regular file through descriptors, never links."""
        flags = os.O_RDONLY
        if hasattr(os, "O_NOFOLLOW"):
            flags |= os.O_NOFOLLOW
        source_descriptor = os.open(source, flags)
        try:
            source_lstat = source.lstat()
            before = os.fstat(source_descriptor)
            if (
                not stat.S_ISREG(source_lstat.st_mode)
                or not stat.S_ISREG(before.st_mode)
                or source_lstat.st_nlink != 1
                or before.st_nlink != 1
                or (source_lstat.st_dev, source_lstat.st_ino)
                != (before.st_dev, before.st_ino)
            ):
                raise _artifact_error(
                    f"ARTIFACT-TYPE-001: unsafe source artifact {source}"
                )
            destination.parent.mkdir(parents=True, exist_ok=True)
            destination_descriptor = os.open(
                destination,
                os.O_WRONLY | os.O_CREAT | os.O_EXCL,
                0o600,
            )
            try:
                copied = 0
                while True:
                    chunk = os.read(source_descriptor, 1024 * 1024)
                    if not chunk:
                        break
                    view = memoryview(chunk)
                    while view:
                        written = os.write(destination_descriptor, view)
                        view = view[written:]
                        copied += written
                os.fsync(destination_descriptor)
            finally:
                os.close(destination_descriptor)
            after = os.fstat(source_descriptor)
            if copied != before.st_size or _stat_identity(before) != _stat_identity(after):
                destination.unlink(missing_ok=True)
                raise _artifact_error(
                    f"SOURCE-MUTATION-001: source changed while copying {source}"
                )
        finally:
            os.close(source_descriptor)


class TypedGraphStore:
    """Validated lazy process-local read-only NumPy views of one store."""

    def __init__(self, validated: ValidatedStore) -> None:
        self.path = validated.root
        self._manifest = validated.manifest
        self.qualification_report = validated.report
        self._file_identities = validated.file_identities
        self._maps: dict[str, Any] = {}
        self._closed = False
        self._state: TypedGraphStoreState | None = None
        self._node_keys = {
            record["name"]: key for key, record in self._manifest["nodes"].items()
        }
        self._relation_keys = {
            tuple(record["relation"]): key
            for key, record in self._manifest["relations"].items()
        }

    @classmethod
    def open(
        cls,
        path: str | Path,
        *,
        expected_bindings: Mapping[str, Any] | None = None,
        execution_monitor: object | None = None,
    ) -> TypedGraphStore:
        return cls(
            validate_store(
                path,
                expected_bindings=expected_bindings,
                require_directory_identity=True,
                execution_monitor=execution_monitor,
            )
        )

    @classmethod
    def from_state(cls, state: TypedGraphStoreState) -> TypedGraphStore:
        """Rehydrate an already-qualified immutable view without rescanning."""
        if not isinstance(state, TypedGraphStoreState):
            raise TypeError("state must be a TypedGraphStoreState")
        manifest = json.loads(state.manifest_json)
        report_record = json.loads(state.report_json)
        if not isinstance(manifest, dict) or not isinstance(report_record, dict):
            raise TypeError("typed graph store state must contain JSON objects")
        root = Path(state.root)
        if (
            report_record.get("passed") is not True
            or report_record.get("store_path") != str(root)
        ):
            raise ValueError("typed graph store state is not a passed root binding")
        raw_checks = report_record.get("checks")
        if not isinstance(raw_checks, list):
            raise TypeError("typed graph store state has malformed checks")
        checks = tuple(
            QualificationCheckResult(
                check_id=record["check_id"],
                passed=record["passed"],
                observed=record.get("observed"),
                expected=record.get("expected"),
                limit=record.get("limit"),
                evidence=MappingProxyType(dict(record.get("evidence", {}))),
                remediation=record.get("remediation", ""),
            )
            for record in raw_checks
        )
        report = QualificationReport(
            passed=True,
            checks=checks,
            report_path=Path(report_record["report_path"]),
            store_path=root,
            format_version=report_record["format_version"],
        )
        store = cls(
            ValidatedStore(
                root=root,
                manifest=manifest,
                report=report,
                file_identities=MappingProxyType(dict(state.file_identities)),
            )
        )
        store._state = state
        return store

    def state(self) -> TypedGraphStoreState:
        """Capture immutable, picklable qualification evidence for workers."""
        self._ensure_open()
        if self._state is None:
            self._state = TypedGraphStoreState(
                root=self.path,
                manifest_json=json.dumps(
                    _normalized_json(self._manifest),
                    sort_keys=True,
                    separators=(",", ":"),
                    ensure_ascii=False,
                    allow_nan=False,
                ).encode("utf-8"),
                report_json=json.dumps(
                    self.qualification_report.as_record(),
                    sort_keys=True,
                    separators=(",", ":"),
                    ensure_ascii=False,
                    allow_nan=False,
                ).encode("utf-8"),
                file_identities=tuple(sorted(self._file_identities.items())),
            )
        return self._state

    def __enter__(self) -> TypedGraphStore:
        self._ensure_open()
        return self

    def __exit__(self, *_exc: Any) -> None:
        self.close()

    @property
    def content_sha256(self) -> str:
        return self._manifest["content_sha256"]

    @property
    def output_kind(self) -> str:
        return self._manifest["output_kind"]

    @property
    def active_split_tag(self) -> str:
        return self._manifest["active_split_tag"]

    @property
    def node_types(self) -> tuple[str, ...]:
        return tuple(record["name"] for record in self._manifest["nodes"].values())

    @property
    def relation_types(self) -> tuple[tuple[str, str, str], ...]:
        return tuple(
            tuple(record["relation"])
            for record in self._manifest["relations"].values()
        )

    @property
    def task_bindings(self) -> dict[str, Any]:
        return dict(self._manifest["task_bindings"])

    @property
    def mapped_paths(self) -> tuple[str, ...]:
        return tuple(self._maps)

    @property
    def num_partitions(self) -> int:
        return self._manifest["partition"]["num_partitions"]

    @property
    def partition_book_identity(self) -> str:
        return self._manifest["partition"]["content_identity"]

    def node_features(self, node_type: str, rows: Any = None) -> np.ndarray:
        node = self._node(node_type)
        return self._selected(self._map(node["x"]), rows)

    def node_labels(self, node_type: str, rows: Any = None) -> np.ndarray:
        node = self._node(node_type)
        if node["y"] is None:
            raise KeyError(f"node type {node_type!r} has no labels")
        return self._selected(self._map(node["y"]), rows)

    def node_array(self, node_type: str, attr_name: str, rows: Any = None) -> np.ndarray:
        if attr_name == "x":
            return self.node_features(node_type, rows)
        if attr_name == "y":
            return self.node_labels(node_type, rows)
        node = self._node(node_type)
        try:
            record = node["fields"][attr_name]
        except KeyError as error:
            raise KeyError((node_type, attr_name)) from error
        return self._selected(self._map(record), rows)

    def relation_csc(
        self,
        relation: Sequence[str],
    ) -> tuple[np.ndarray, np.ndarray]:
        record = self._relation(relation)
        return self._map(record["row"]), self._map(record["colptr"])

    def relation_field(
        self,
        relation: Sequence[str],
        field_name: str,
        rows: Any = None,
    ) -> np.ndarray:
        record = self._relation(relation)
        if field_name == "edge_id":
            field = record["edge_id"]
        else:
            field = record["fields"].get(field_name)
        if field is None:
            raise KeyError((tuple(relation), field_name))
        return self._selected(self._map(field), rows)

    def split_ids(self, tag: str, phase: str) -> np.ndarray:
        self._ensure_open()
        if phase not in _PHASES:
            raise KeyError((tag, phase))
        try:
            record = self._manifest["splits"][tag]["phases"][phase]
        except KeyError as error:
            raise KeyError((tag, phase)) from error
        return self._map(record)

    def partition_assignment(self, node_type: str) -> np.ndarray:
        return self._partition_node_array(node_type, "assignment")

    def partition_permutation(self, node_type: str) -> np.ndarray:
        return self._partition_node_array(node_type, "permutation")

    def partition_inverse_permutation(self, node_type: str) -> np.ndarray:
        return self._partition_node_array(node_type, "inverse")

    def partition_partptr(self, node_type: str) -> np.ndarray:
        return self._partition_node_array(node_type, "partptr")

    def relation_edge_partition(self, relation: Sequence[str]) -> np.ndarray:
        record = self._relation(relation)
        key = record["internal_key"]
        return self._map(
            self._manifest["partition"]["relations"][key]["edge_partition"]
        )

    def external_ids(self, node_type: str, rows: Any = None) -> list[int | str]:
        """Explicitly restore external IDs; imports PyArrow only on this call."""
        self._ensure_open()
        node = self._node(node_type)
        import pyarrow as pa
        import pyarrow.parquet as pq

        relative = node["node_ids"]["relative_path"]
        path = self.path / relative
        descriptor, before = _secure_descriptor(path)
        expected_identity = self._file_identities.get(relative)
        if (
            expected_identity is None
            or _stat_identity(before) != expected_identity
        ):
            os.close(descriptor)
            raise QualificationFailure(
                _external_id_failure(
                    node_type,
                    "artifact identity changed after store validation",
                    expected_identity,
                ),
                self.qualification_report.report_path,
            )
        stream = os.fdopen(descriptor, "rb", closefd=True)
        try:
            parquet = pq.ParquetFile(stream)
            restored = self._external_ids_from_parquet(
                node_type,
                node,
                rows,
                parquet,
                pa,
            )
            after = os.fstat(stream.fileno())
            if _stat_identity(before) != _stat_identity(after):
                raise QualificationFailure(
                    _external_id_failure(
                        node_type,
                        "artifact changed while restoring external IDs",
                        expected_identity,
                    ),
                    self.qualification_report.report_path,
                )
            return restored
        except QualificationFailure:
            raise
        except Exception as error:
            raise QualificationFailure(
                _external_id_failure(
                    node_type,
                    f"{type(error).__name__}: {error}",
                    "unchanged canonical Parquet external-ID map",
                ),
                self.qualification_report.report_path,
            ) from error
        finally:
            stream.close()

    def _external_ids_from_parquet(
        self,
        node_type: str,
        node: Mapping[str, Any],
        rows: Any,
        parquet: Any,
        pa: Any,
    ) -> list[int | str]:
        expected_type = {
            "int64": pa.int64(),
            "uint64": pa.uint64(),
            "string": pa.string(),
        }[node["id_dtype"]]
        schema = parquet.schema_arrow
        if (
            parquet.metadata.num_rows != node["count"]
            or schema.names != ["local_ordinal", "external_id"]
            or not schema.field("local_ordinal").type.equals(pa.int64())
            or not schema.field("external_id").type.equals(expected_type)
        ):
            raise QualificationFailure(
                _external_id_failure(
                    node_type,
                    parquet.metadata.num_rows,
                    node["count"],
                ),
                self.qualification_report.report_path,
            )
        indices = _row_indices(rows, node["count"])
        if indices is None:
            table = parquet.read(columns=["local_ordinal", "external_id"])
            ordinals = np.asarray(
                table.column("local_ordinal").to_pylist(),
                dtype=np.int64,
            )
            if not np.array_equal(
                ordinals,
                np.arange(node["count"], dtype=np.int64),
            ):
                raise QualificationFailure(
                    _external_id_failure(
                        node_type,
                        "non-canonical ordinals",
                        "0..count-1",
                    ),
                    self.qualification_report.report_path,
                )
            return table.column("external_id").to_pylist()
        requested = (
            np.arange(
                indices.start,
                indices.stop,
                indices.step,
                dtype=np.int64,
            )
            if isinstance(indices, slice)
            else np.asarray(indices, dtype=np.int64)
        )
        if len(requested) == 0:
            return []
        unique = np.unique(requested)
        restored: dict[int, int | str] = {}
        row_start = 0
        for row_group in range(parquet.metadata.num_row_groups):
            row_count = parquet.metadata.row_group(row_group).num_rows
            row_end = row_start + row_count
            selected_start = int(
                np.searchsorted(unique, row_start, side="left")
            )
            selected_end = int(
                np.searchsorted(unique, row_end, side="left")
            )
            if selected_start < selected_end:
                table = parquet.read_row_group(
                    row_group,
                    columns=["local_ordinal", "external_id"],
                )
                ordinals = np.asarray(
                    table.column("local_ordinal").to_pylist(),
                    dtype=np.int64,
                )
                expected_ordinals = np.arange(
                    row_start,
                    row_end,
                    dtype=np.int64,
                )
                if not np.array_equal(ordinals, expected_ordinals):
                    raise QualificationFailure(
                        _external_id_failure(
                            node_type,
                            {
                                "row_group": row_group,
                                "ordinals": ordinals.tolist(),
                            },
                            {
                                "first": row_start,
                                "exclusive_last": row_end,
                            },
                        ),
                        self.qualification_report.report_path,
                    )
                values = table.column("external_id").to_pylist()
                for ordinal in unique[selected_start:selected_end]:
                    local_offset = int(ordinal) - row_start
                    restored[int(ordinal)] = values[local_offset]
            row_start = row_end
        if row_start != node["count"] or len(restored) != len(unique):
            raise QualificationFailure(
                _external_id_failure(
                    node_type,
                    {
                        "restored": len(restored),
                        "row_count": row_start,
                    },
                    {
                        "selected": len(unique),
                        "row_count": node["count"],
                    },
                ),
                self.qualification_report.report_path,
            )
        return [restored[int(index)] for index in requested]

    def close(self) -> None:
        if self._closed:
            return
        self._closed = True
        handles = tuple(self._maps.values())
        self._maps.clear()
        for handle in handles:
            handle.close()

    def _ensure_open(self) -> None:
        if self._closed:
            raise RuntimeError("TypedGraphStore is closed")

    def _node(self, node_type: str) -> Mapping[str, Any]:
        self._ensure_open()
        try:
            return self._manifest["nodes"][self._node_keys[node_type]]
        except KeyError as error:
            raise KeyError(node_type) from error

    def _relation(self, relation: Sequence[str]) -> Mapping[str, Any]:
        self._ensure_open()
        if isinstance(relation, (str, bytes)):
            raise KeyError(relation)
        key_tuple = tuple(relation)
        try:
            return self._manifest["relations"][self._relation_keys[key_tuple]]
        except KeyError as error:
            raise KeyError(key_tuple) from error

    def _map(self, record: Mapping[str, Any]) -> np.ndarray:
        self._ensure_open()
        relative = record["relative_path"]
        handle = self._maps.get(relative)
        if handle is None:
            try:
                handle = _open_npy(
                    self.path / relative,
                    self._file_identities.get(relative),
                )
            except (OSError, ValueError) as error:
                raise QualificationFailure(
                    _lazy_array_failure(relative, str(error), record),
                    self.qualification_report.report_path,
                ) from error
            if (
                handle.array.dtype != np.dtype(record["dtype"])
                or handle.array.shape != tuple(record["shape"])
            ):
                observed = {
                    "dtype": handle.array.dtype.str,
                    "shape": list(handle.array.shape),
                }
                handle.close()
                raise QualificationFailure(
                    _lazy_array_failure(relative, observed, record),
                    self.qualification_report.report_path,
                )
            self._maps[relative] = handle
        return handle.array

    def _selected(self, array: np.ndarray, rows: Any) -> np.ndarray:
        indices = _row_indices(rows, array.shape[0])
        result = array if indices is None else array[indices]
        result.flags.writeable = False
        return result

    def _partition_node_array(self, node_type: str, name: str) -> np.ndarray:
        node = self._node(node_type)
        key = node["internal_key"]
        return self._map(self._manifest["partition"]["node_types"][key][name])


def _row_indices(rows: Any, count: int) -> np.ndarray | slice | None:
    if rows is None:
        return None
    if isinstance(rows, slice):
        start, stop, step = rows.indices(count)
        return slice(start, stop, step)
    if isinstance(rows, bool):
        raise TypeError("row selection cannot be boolean")
    if isinstance(rows, int):
        if rows < 0 or rows >= count:
            raise IndexError(rows)
        return np.array([rows], dtype=np.int64)
    if hasattr(rows, "detach") and hasattr(rows, "cpu"):
        rows = rows.detach().cpu().numpy()
    values = np.asarray(rows)
    if values.dtype.kind not in {"i", "u"} or values.ndim != 1:
        raise TypeError("row selection must be a one-dimensional integer index")
    if len(values) and (int(values.min()) < 0 or int(values.max()) >= count):
        raise IndexError("row selection is out of range")
    return values.astype(np.int64, copy=False)


def _plain_file_record(
    path: Path,
    relative: str,
    role: str,
    *,
    dtype: str | None,
    shape: list[int] | None,
) -> dict[str, Any]:
    return {
        "relative_path": relative,
        "role": role,
        "dtype": dtype,
        "shape": shape,
        "byte_size": path.stat().st_size,
        "sha256": _sha256_file(path),
        "finite": False,
    }


def _array_file_record(
    path: Path,
    relative: str,
    role: str,
    *,
    finite: bool = False,
) -> dict[str, Any]:
    array = np.load(path, mmap_mode="r", allow_pickle=False)
    record = _plain_file_record(
        path,
        relative,
        role,
        dtype=array.dtype.str,
        shape=list(array.shape),
    )
    record["finite"] = finite
    mmap = getattr(array, "_mmap", None)
    if mmap is not None:
        mmap.close()
    return record


def _build_environment(
    ingestor: Any,
    book: Any,
    *,
    metadata_binding: str,
    task_bindings: Mapping[str, Any],
    split_fingerprints: Mapping[str, str],
    dependency_versions: Mapping[str, str],
) -> dict[str, Any]:
    import torch
    import torch_geometric

    lock_path = Path(__file__).resolve().parents[3] / "uv.lock"
    cuda_available = torch.cuda.is_available()
    return {
        "format_version": "typed-graph-build-environment-v1",
        "python": platform.python_version(),
        "python_implementation": platform.python_implementation(),
        "pytorch": torch.__version__,
        "pyg": torch_geometric.__version__,
        "partition_backend": book.backend,
        "partition_backend_version": book.backend_version,
        "dependency_versions": dict(dependency_versions),
        "dependency_lock_sha256": (
            _sha256_file(lock_path) if lock_path.is_file() else None
        ),
        "source_state_sha256": book.source_binding["source_fingerprint"],
        "config_sha256": book.source_binding["config_fingerprint"],
        "os": platform.system(),
        "os_release": platform.release(),
        "architecture": platform.machine(),
        "processor": platform.processor(),
        "cpu_count": os.cpu_count(),
        "cuda_available": cuda_available,
        "cuda_version": torch.version.cuda,
        "cuda_device": torch.cuda.get_device_name(0) if cuda_available else None,
        "container_image": os.environ.get("CONTAINER_IMAGE_DIGEST"),
        "store_filesystem": str(ingestor.store_root),
        "metadata_binding_sha256": metadata_binding,
        "task_bindings": task_bindings,
        "partition_book_identity": book.content_identity,
        "split_fingerprints": dict(split_fingerprints),
    }


def _normalized_json(value: Any) -> Any:
    if isinstance(value, Mapping):
        return {str(key): _normalized_json(item) for key, item in sorted(value.items(), key=lambda item: str(item[0]))}
    if isinstance(value, (tuple, list)):
        return [_normalized_json(item) for item in value]
    if isinstance(value, np.generic):
        return value.item()
    if value is None or isinstance(value, (str, int, float, bool)):
        return value
    raise TypeError(f"task binding is not JSON data: {type(value).__name__}")


def _write_json(path: Path, value: Mapping[str, Any]) -> None:
    path.parent.mkdir(parents=True, exist_ok=True)
    temporary = path.with_name(f".{path.name}.{uuid.uuid4().hex}.tmp")
    with temporary.open("wb") as stream:
        stream.write(
            json.dumps(
                value,
                sort_keys=True,
                separators=(",", ":"),
                ensure_ascii=False,
                allow_nan=False,
            ).encode("utf-8")
            + b"\n"
        )
        stream.flush()
        os.fsync(stream.fileno())
    os.replace(temporary, path)
    _fsync_directory(path.parent)


def _read_json(path: Path) -> dict[str, Any]:
    if path.is_symlink() or not path.is_file():
        raise _artifact_error(f"MANIFEST-001: unsafe JSON artifact {path}")
    try:
        value = json.loads(path.read_text(encoding="utf-8"))
    except (OSError, UnicodeDecodeError, json.JSONDecodeError) as error:
        raise _artifact_error(f"MANIFEST-001: malformed JSON artifact {path}") from error
    if not isinstance(value, dict):
        raise _artifact_error(f"MANIFEST-001: JSON artifact {path} is not an object")
    return value


def _sha256_file(path: Path) -> str:
    digest = hashlib.sha256()
    with path.open("rb") as stream:
        while chunk := stream.read(1024 * 1024):
            digest.update(chunk)
    return digest.hexdigest()


def _stat_identity(value: os.stat_result) -> tuple[int, int, int, int, int]:
    return (
        value.st_dev,
        value.st_ino,
        value.st_size,
        value.st_mtime_ns,
        value.st_ctime_ns,
    )


def _same_filesystem(left: Path, right: Path) -> bool:
    return left.stat().st_dev == right.stat().st_dev


def _fsync_directory(path: Path) -> None:
    descriptor = os.open(path, os.O_RDONLY)
    try:
        os.fsync(descriptor)
    finally:
        os.close(descriptor)


def _fsync_tree(root: Path) -> None:
    for path in sorted(
        (item for item in root.rglob("*") if item.is_dir()),
        key=lambda item: len(item.parts),
        reverse=True,
    ):
        _fsync_directory(path)
    _fsync_directory(root)


def _make_read_only(root: Path, *, movable_root: bool = False) -> None:
    for path in root.rglob("*"):
        if path.is_symlink():
            raise _artifact_error("ARTIFACT-TYPE-001: store contains a symlink")
        path.chmod(0o555 if path.is_dir() else 0o444)
    root.chmod(0o755 if movable_root else 0o555)


def _artifact_error(message: str) -> Exception:
    from topobench.data.stores.typed_graph_ingestion import ArtifactValidationError

    return ArtifactValidationError(message)


def _external_id_failure(node_type: str, observed: Any, expected: Any) -> Any:
    from topobench.data.stores.qualification_checks import QualificationCheckResult

    return QualificationCheckResult(
        "EXTERNAL-ID-SCHEMA-001",
        False,
        observed,
        expected,
        evidence=MappingProxyType({"node_type": node_type}),
        remediation="restore the exact type-local node_ids.parquet artifact",
    )


def _lazy_array_failure(relative: str, observed: Any, expected: Any) -> Any:
    from topobench.data.stores.qualification_checks import QualificationCheckResult

    return QualificationCheckResult(
        "ARRAY-CHECKSUM-001",
        False,
        observed,
        expected,
        evidence=MappingProxyType({"relative_path": relative}),
        remediation="restore the exact checksummed NumPy artifact",
    )


__all__ = [
    "TypedGraphStore",
    "TypedGraphStoreBuild",
    "TypedGraphStoreState",
    "TypedGraphStoreWriter",
]
