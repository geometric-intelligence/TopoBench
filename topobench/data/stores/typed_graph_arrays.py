"""Bounded writers for ordinal-aligned typed features and target supervision.

PyArrow and DuckDB remain ingestion-only dependencies: this module imports
neither until :class:`TypedGraphArrayWriter` is explicitly invoked.  Every
source scan uses an immutable snapshot supplied by the ingestion stage, while
external-ID resolution reuses Task 2's exact per-type ``node_ids.parquet``
indexes.
"""

from __future__ import annotations

from dataclasses import dataclass, replace
import hashlib
import json
import os
from pathlib import Path
import shutil
from typing import TYPE_CHECKING, Any, Mapping
import uuid

import numpy as np

if TYPE_CHECKING:
    from topobench.data.stores.typed_graph_ingestion import (
        ExternalNodeIndexBuild,
        ParquetTypedGraphIngestor,
        SourceInventory,
    )

_ARRAY_BEHAVIOR_VERSION = "typed-graph-arrays-v1"
_PHASES = ("train", "val", "test")
_INTEGER_DTYPES = frozenset(
    {"int8", "int16", "int32", "int64", "uint8", "uint16", "uint32", "uint64"}
)
_FLOAT_DTYPES = frozenset({"float16", "float32", "float64"})


@dataclass(frozen=True, slots=True)
class TypedGraphArrayBuild:
    """A checksum-validated feature/supervision stage."""

    inventory: SourceInventory
    stage_root: Path
    artifact_root: Path
    content_sha256: str
    active_tag: str
    resumed: bool
    record_batch_rows: int
    max_record_batch_rows: int


class TypedGraphArrayWriter:
    """Write and reopen one bounded, atomically completed array stage."""

    def __init__(
        self,
        ingestor: ParquetTypedGraphIngestor,
        index_build: ExternalNodeIndexBuild,
    ) -> None:
        self.ingestor = ingestor
        self.index_build = index_build
        self._max_batch_rows = 0

    def _array_disk_requirements(self) -> dict[str, int | bool]:
        inventory = self.index_build.inventory
        node_counts = dict(inventory.node_rows)
        payload_bytes = sum(
            node_counts[node.name]
            * node.feature_width
            * _numpy_dtype(node.feature_dtype).itemsize
            for node in self.ingestor.source.spec.node_types
        )
        supervision = self.ingestor.source.spec.supervision
        target_count = node_counts[supervision.target_node_type]
        label_itemsize = (
            _numpy_dtype(supervision.label_dtype).itemsize
            if supervision.label_dtype in _INTEGER_DTYPES | _FLOAT_DTYPES
            else 8
        )
        payload_bytes += target_count * label_itemsize
        inventory_by_path = {
            entry.relative_path: entry for entry in inventory.files
        }
        split_rows = sum(
            inventory_by_path[getattr(split, phase)].row_count
            for split in supervision.split_registry.sets
            for phase in _PHASES
        )
        payload_bytes += split_rows * _numpy_dtype("int64").itemsize
        array_count = (
            len(self.ingestor.source.spec.node_types)
            + 1
            + 3 * len(supervision.split_registry.sets)
        )
        evidence_reserve = (
            1024**2
            + 4096
            * (
                array_count
                + len(inventory.files)
                + len(supervision.split_registry.sets)
            )
        )
        estimated_array_bytes = (
            payload_bytes + 512 * array_count + evidence_reserve
        )
        snapshot_bytes = inventory.snapshot_bytes
        spill_bytes = max(
            1,
            inventory.estimated_temporary_bytes - snapshot_bytes,
        )
        task2_final_bytes = inventory.estimated_final_bytes
        final_peak_bytes = task2_final_bytes + estimated_array_bytes
        temporary_peak_bytes = snapshot_bytes + spill_bytes
        same_filesystem = (
            inventory.final_device == inventory.temporary_device
        )
        return {
            "same_filesystem": same_filesystem,
            "task2_final_bytes": task2_final_bytes,
            "estimated_array_bytes": estimated_array_bytes,
            "array_payload_bytes": payload_bytes,
            "array_evidence_reserve_bytes": evidence_reserve,
            "snapshot_bytes": snapshot_bytes,
            "spill_bytes": spill_bytes,
            "final_peak_bytes": final_peak_bytes,
            "temporary_peak_bytes": temporary_peak_bytes,
            "shared_peak_bytes": final_peak_bytes + temporary_peak_bytes,
            "final_additional_bytes": estimated_array_bytes,
            "temporary_additional_bytes": temporary_peak_bytes,
            "shared_additional_bytes": (
                estimated_array_bytes + temporary_peak_bytes
            ),
        }

    def _admit_array_disk(
        self,
        requirements: Mapping[str, int | bool],
    ) -> dict[str, Any]:
        ingestion = _ingestion_module()
        inventory = self.index_build.inventory
        final_device, final_available, final_probe = (
            ingestion._filesystem_capacity(inventory.final_filesystem_path)
        )
        temporary_device, temporary_available, temporary_probe = (
            ingestion._filesystem_capacity(
                inventory.temporary_filesystem_path
            )
        )
        if (
            final_device != inventory.final_device
            or temporary_device != inventory.temporary_device
        ):
            raise ingestion.DiskAdmissionError(
                "DISK-EVIDENCE-001: final or temporary filesystem changed "
                "before Task 3 allocation"
            )
        same_filesystem = bool(requirements["same_filesystem"])
        if same_filesystem:
            available = min(final_available, temporary_available)
            required = int(requirements["shared_additional_bytes"])
            if required > available:
                raise ingestion.DiskAdmissionError(
                    "DISK-PREFLIGHT-001: shared final+temporary filesystem "
                    f"requires {required} additional bytes, only "
                    f"{available} available"
                )
            if (
                self.ingestor.disk_limit_bytes is not None
                and int(requirements["shared_peak_bytes"])
                > self.ingestor.disk_limit_bytes
            ):
                raise ingestion.DiskAdmissionError(
                    "DISK-PREFLIGHT-001: shared final+temporary filesystem "
                    f"peak requires {requirements['shared_peak_bytes']} bytes, "
                    f"limit is {self.ingestor.disk_limit_bytes}"
                )
        else:
            final_required = int(requirements["final_additional_bytes"])
            temporary_required = int(
                requirements["temporary_additional_bytes"]
            )
            if final_required > final_available:
                raise ingestion.DiskAdmissionError(
                    "DISK-PREFLIGHT-001: final filesystem requires "
                    f"{final_required} additional bytes, only "
                    f"{final_available} available"
                )
            if temporary_required > temporary_available:
                raise ingestion.DiskAdmissionError(
                    "DISK-PREFLIGHT-001: temporary filesystem requires "
                    f"{temporary_required} bytes, only "
                    f"{temporary_available} available"
                )
            if self.ingestor.disk_limit_bytes is not None:
                if (
                    int(requirements["final_peak_bytes"])
                    > self.ingestor.disk_limit_bytes
                ):
                    raise ingestion.DiskAdmissionError(
                        "DISK-PREFLIGHT-001: final filesystem peak requires "
                        f"{requirements['final_peak_bytes']} bytes, limit is "
                        f"{self.ingestor.disk_limit_bytes}"
                    )
                if (
                    int(requirements["temporary_peak_bytes"])
                    > self.ingestor.disk_limit_bytes
                ):
                    raise ingestion.DiskAdmissionError(
                        "DISK-PREFLIGHT-001: temporary filesystem peak "
                        f"requires {requirements['temporary_peak_bytes']} "
                        f"bytes, limit is {self.ingestor.disk_limit_bytes}"
                    )
        return {
            "requirements": dict(requirements),
            "observed": {
                "final_device": final_device,
                "temporary_device": temporary_device,
                "final_probe": str(final_probe),
                "temporary_probe": str(temporary_probe),
                "final_available_bytes": final_available,
                "temporary_available_bytes": temporary_available,
                "disk_limit_bytes": self.ingestor.disk_limit_bytes,
            },
        }

    def _validate_resume_disk(
        self,
        requirements: Mapping[str, int | bool],
        artifact_root: Path,
    ) -> None:
        ingestion = _ingestion_module()
        inventory = self.index_build.inventory
        limit = self.ingestor.disk_limit_bytes
        if isinstance(limit, bool) or (
            limit is not None
            and (not isinstance(limit, int) or limit <= 0)
        ):
            raise ValueError("disk_limit_bytes must be a positive integer")
        final_device, _, _ = ingestion._filesystem_capacity(
            inventory.final_filesystem_path
        )
        if (
            self.ingestor.store_root != inventory.final_filesystem_path
            or final_device != inventory.final_device
            or artifact_root.is_symlink()
            or not artifact_root.resolve(strict=True).is_relative_to(
                inventory.final_filesystem_path.resolve(strict=True)
            )
            or not artifact_root.is_dir()
        ):
            raise ingestion.DiskAdmissionError(
                "DISK-EVIDENCE-001: current final artifact filesystem does "
                "not match its admitted inventory"
            )
        if limit is None:
            return
        required_peak = (
            int(requirements["shared_peak_bytes"])
            if requirements["same_filesystem"]
            else max(
                int(requirements["final_peak_bytes"]),
                int(requirements["temporary_peak_bytes"]),
            )
        )
        if limit < required_peak:
            raise ingestion.DiskAdmissionError(
                "DISK-PREFLIGHT-001: current completed-stage disk limit "
                f"{limit} is below the recorded required peak "
                f"{required_peak}"
            )

    def build(self) -> TypedGraphArrayBuild:
        """Build once or validate and resume a completed addressed stage."""
        ingestion = _ingestion_module()
        inventory = self.index_build.inventory
        stage_root = self.index_build.stage_root
        artifact_root = stage_root / "arrays"
        pa, pq, duckdb = ingestion._parquet_dependencies()
        with self.ingestor._build_lock(self.ingestor.lock_path(inventory)):
            self.ingestor._validate_inventory_current(inventory, pa=pa, pq=pq)
            if (artifact_root / "arrays.complete.json").is_file():
                result = self._open_validated(resumed=True)
                self._finalize_top_completion(result)
                return result
            disk_requirements = self._array_disk_requirements()
            disk_admission = self._admit_array_disk(disk_requirements)
            for relative in (
                "arrays",
                "nodes",
                "splits",
                "arrays.json",
                "arrays.complete.json",
            ):
                if os.path.lexists(stage_root / relative):
                    raise ingestion.ArtifactValidationError(
                        f"INCOMPLETE-ARRAY-STAGE-001: uncompleted array artifact {relative!r}"
                    )

            temporary_root = stage_root / f".arrays-tmp-{uuid.uuid4().hex}"
            spill_root = self.ingestor._new_ephemeral_root(
                inventory,
                purpose="array-spill",
            )
            temporary_root.mkdir(parents=False, exist_ok=False)
            spill_created = False
            published = False
            try:
                spill_root.mkdir(parents=True, exist_ok=False)
                spill_created = True
                with self.ingestor._immutable_source_snapshot(
                    inventory,
                    purpose="array-source-snapshot",
                    pa=pa,
                    pq=pq,
                ) as snapshot_stage:
                    snapshots, snapshot_root = snapshot_stage
                    connection = duckdb.connect(str(spill_root / "arrays.duckdb"))
                    try:
                        self._configure_connection(connection, spill_root)
                        metadata = self._write_all(
                            temporary_root,
                            snapshots=snapshots,
                            connection=connection,
                            pa=pa,
                            pq=pq,
                        )
                    finally:
                        connection.close()
                    staged_array_bytes = _metadata_array_bytes(metadata)
                    metadata["resource_evidence"] = {
                        "record_batch_rows": self.ingestor.source.spec.ingestion.record_batch_rows,
                        "max_record_batch_rows": self._max_batch_rows,
                        "snapshot_bytes": inventory.snapshot_bytes,
                        "snapshot_subtree": "ephemeral/array-source-snapshot-*",
                        "snapshot_persisted": False,
                        "snapshot_bytes_accounted": True,
                        "staged_array_bytes": staged_array_bytes,
                        "disk_requirements": disk_requirements,
                        "duckdb_memory_limit_bytes": self.ingestor.source.spec.ingestion.memory_limit_bytes,
                        "bounded_memory": "O(record_batch_rows * max_feature_width) plus DuckDB spill state",
                    }
                    metadata["max_record_batch_rows"] = self._max_batch_rows
                    identity = {
                        "array_behavior_version": _ARRAY_BEHAVIOR_VERSION,
                        "input_fingerprint": inventory.source_fingerprint,
                        "config_fingerprint": inventory.config_fingerprint,
                        "active_split_tag": metadata["active_split_tag"],
                        "nodes": {
                            key: value["content_sha256"]
                            for key, value in metadata["nodes"].items()
                        },
                        "supervision": metadata["supervision"]["content_sha256"],
                        "splits": {
                            tag: {
                                phase: evidence["content_sha256"]
                                for phase, evidence in value["phases"].items()
                            }
                            for tag, value in metadata["splits"].items()
                        },
                    }
                    metadata["content_identity"] = identity
                    metadata["content_sha256"] = ingestion._sha256_json(identity)
                    ingestion._atomic_json(temporary_root / "arrays.json", metadata)
                    self._validate_array_tree(
                        temporary_root,
                        metadata,
                        check_completion=False,
                    )
                    outputs = self._array_outputs(temporary_root)
                    completion_record = {
                        "stage": "typed_graph_arrays",
                        "array_behavior_version": _ARRAY_BEHAVIOR_VERSION,
                        "behavior_version": ingestion._BEHAVIOR_VERSION,
                        "input_fingerprint": inventory.source_fingerprint,
                        "config_fingerprint": inventory.config_fingerprint,
                        "dependency_versions": dict(inventory.dependency_versions),
                        "content_sha256": metadata["content_sha256"],
                        "outputs": outputs,
                        "reopened_and_validated": True,
                        "disk_admission": disk_admission,
                        "prepared_array_stage_bytes": 0,
                    }
                    completion_path = (
                        temporary_root / "arrays.complete.json"
                    )
                    for _ in range(4):
                        ingestion._atomic_json(
                            completion_path,
                            completion_record,
                        )
                        prepared_bytes = _tree_bytes(temporary_root)
                        if (
                            completion_record["prepared_array_stage_bytes"]
                            == prepared_bytes
                        ):
                            break
                        completion_record["prepared_array_stage_bytes"] = (
                            prepared_bytes
                        )
                    else:
                        raise ingestion.ArtifactValidationError(
                            "DISK-EVIDENCE-001: prepared array byte evidence "
                            "did not stabilize"
                        )
                    prepared_bytes = _tree_bytes(temporary_root)
                    if prepared_bytes > int(
                        disk_requirements["estimated_array_bytes"]
                    ):
                        raise ingestion.DiskAdmissionError(
                            "DISK-ESTIMATE-001: prepared array stage exceeded "
                            "its preallocation admission estimate"
                        )
                if spill_created:
                    shutil.rmtree(spill_root)
                    spill_created = False
                self._publish(temporary_root, stage_root)
                published = True
                result = self._open_validated(resumed=False)
                self._finalize_top_completion(result)
                return result
            except BaseException:
                if published and artifact_root.exists():
                    shutil.rmtree(artifact_root)
                    ingestion._fsync_directory(stage_root)
                raise
            finally:
                if temporary_root.exists():
                    shutil.rmtree(temporary_root)
                if spill_created and spill_root.exists():
                    shutil.rmtree(spill_root)

    def _configure_connection(self, connection: Any, spill_root: Path) -> None:
        memory_bytes = self.ingestor.source.spec.ingestion.memory_limit_bytes
        escaped_spill = str(spill_root).replace("'", "''")
        connection.execute("SET preserve_insertion_order = false")
        connection.execute(f"SET memory_limit = '{memory_bytes}B'")
        connection.execute(f"SET temp_directory = '{escaped_spill}'")
        connection.execute(f"SET threads = {self.ingestor.threads}")

    def _preflight_label_ownership(
        self,
        *,
        snapshots: Mapping[str, Path],
        pq: Any,
    ) -> None:
        ingestion = _ingestion_module()
        spec = self.ingestor.source.spec
        supervision = spec.supervision
        if supervision.label_source != "nodes":
            return
        for node in spec.node_types:
            if node.name == supervision.target_node_type:
                continue
            paths = tuple(snapshots[relative] for relative in node.paths)
            for relative, path in zip(node.paths, paths, strict=True):
                schema = pq.ParquetFile(path).schema_arrow
                if schema.get_field_index(supervision.label_column) >= 0:
                    raise ingestion.ArtifactValidationError(
                        "TARGET-OWNERSHIP-001: non-target node type "
                        f"{node.name!r} source {relative!r} contains declared "
                        f"label column {supervision.label_column!r}"
                    )
            self._exact_role_schema(
                paths,
                pq=pq,
                context=f"non-target node type {node.name!r}",
            )

    def _write_all(
        self,
        root: Path,
        *,
        snapshots: Mapping[str, Path],
        connection: Any,
        pa: Any,
        pq: Any,
    ) -> dict[str, Any]:
        spec = self.ingestor.source.spec
        self._preflight_label_ownership(
            snapshots=snapshots,
            pq=pq,
        )
        nodes: dict[str, dict[str, Any]] = {}
        for ordinal, node in enumerate(spec.node_types):
            internal_key = f"n{ordinal:04d}"
            nodes[internal_key] = self._write_features(
                root,
                node=node,
                internal_key=internal_key,
                snapshots=snapshots,
                connection=connection,
                pa=pa,
                pq=pq,
            )
        target_ordinal, target = next(
            (ordinal, node)
            for ordinal, node in enumerate(spec.node_types)
            if node.name == spec.supervision.target_node_type
        )
        target_key = f"n{target_ordinal:04d}"
        supervision = self._write_supervision(
            root,
            target=target,
            target_key=target_key,
            snapshots=snapshots,
            connection=connection,
            pa=pa,
            pq=pq,
        )
        splits = self._write_splits(
            root,
            target=target,
            target_key=target_key,
            snapshots=snapshots,
            connection=connection,
            pa=pa,
            pq=pq,
        )
        return {
            "array_behavior_version": _ARRAY_BEHAVIOR_VERSION,
            "input_fingerprint": self.index_build.inventory.source_fingerprint,
            "config_fingerprint": self.index_build.inventory.config_fingerprint,
            "dependency_versions": dict(self.index_build.inventory.dependency_versions),
            "record_batch_rows": spec.ingestion.record_batch_rows,
            "max_record_batch_rows": self._max_batch_rows,
            "nodes": nodes,
            "supervision": supervision,
            "splits": splits,
            "active_split_tag": spec.supervision.split_registry.active_tag,
        }

    def _write_features(
        self,
        root: Path,
        *,
        node: Any,
        internal_key: str,
        snapshots: Mapping[str, Path],
        connection: Any,
        pa: Any,
        pq: Any,
    ) -> dict[str, Any]:
        ingestion = _ingestion_module()
        paths = tuple(snapshots[relative] for relative in node.paths)
        schema = self._exact_role_schema(paths, pq=pq, context=f"node type {node.name!r}")
        self._require_exact_id(schema, node.id_column, node.id_dtype, pa=pa, code="FEATURE-ID-CAST-001")
        expected_value_type = _arrow_type(pa, node.feature_dtype)
        if node.feature_representation == "fixed_size_list":
            field = self._field(schema, node.feature_columns[0], "FEATURE-SCHEMA-001", node.name)
            observed = field.type
            if not pa.types.is_fixed_size_list(observed):
                raise ingestion.ArtifactValidationError(
                    f"FEATURE-SCHEMA-001: node type {node.name!r} feature must be a fixed-size list"
                )
            if observed.list_size != node.feature_width:
                raise ingestion.ArtifactValidationError(
                    f"FEATURE-WIDTH-001: node type {node.name!r} has width {observed.list_size}, expected {node.feature_width}"
                )
            if not observed.value_type.equals(expected_value_type):
                raise ingestion.ArtifactValidationError(
                    f"FEATURE-CAST-001: node type {node.name!r} list value type {observed.value_type} does not exactly match {expected_value_type}"
                )
        else:
            for column in node.feature_columns:
                field = self._field(schema, column, "FEATURE-SCHEMA-001", node.name)
                if not field.type.equals(expected_value_type):
                    raise ingestion.ArtifactValidationError(
                        f"FEATURE-CAST-001: node type {node.name!r} column {column!r} type {field.type} does not exactly match {expected_value_type}"
                    )

        source_view = f"feature_source_{internal_key}"
        mapping_view = f"feature_mapping_{internal_key}"
        connection.read_parquet([str(path) for path in paths]).create_view(source_view)
        connection.read_parquet(str(self.index_build.indexes[node.name].node_ids_path)).create_view(mapping_view)
        columns = ", ".join(
            f"s.{_quote(column)}" for column in node.feature_columns
        )
        query = (
            f"SELECT m.local_ordinal, {columns} FROM {_quote(source_view)} s "
            f"INNER JOIN {_quote(mapping_view)} m ON s.{_quote(node.id_column)} "
            "IS NOT DISTINCT FROM m.external_id ORDER BY m.local_ordinal"
        )
        count = self.index_build.indexes[node.name].row_count
        path = root / "nodes" / internal_key / "x.npy"
        path.parent.mkdir(parents=True, exist_ok=False)
        array = np.lib.format.open_memmap(
            path,
            mode="w+",
            dtype=_numpy_dtype(node.feature_dtype),
            shape=(count, node.feature_width),
        )
        offset = 0
        try:
            reader = connection.execute(query).to_arrow_reader(
                self.ingestor.source.spec.ingestion.record_batch_rows
            )
            for batch in reader:
                rows = batch.num_rows
                self._observe_batch(rows)
                expected = np.arange(offset, offset + rows, dtype=np.int64)
                observed_ordinals = batch.column(0).to_numpy(zero_copy_only=False)
                if not np.array_equal(observed_ordinals, expected):
                    raise ingestion.ArtifactValidationError(
                        f"FEATURE-ALIGNMENT-001: node type {node.name!r} did not resolve to exact canonical ordinals"
                    )
                if node.feature_representation == "fixed_size_list":
                    values = batch.column(1)
                    flattened = values.flatten()
                    if values.null_count or flattened.null_count:
                        raise ingestion.ArtifactValidationError(
                            f"FEATURE-NULL-001: node type {node.name!r} contains null list features"
                        )
                    block = flattened.to_numpy(zero_copy_only=False).reshape(rows, node.feature_width)
                    self._validate_finite(block, code="FEATURE-FINITE-001", context=f"node type {node.name!r}")
                    array[offset : offset + rows] = block
                else:
                    for column_index, column_name in enumerate(node.feature_columns, start=1):
                        values = batch.column(column_index)
                        if values.null_count:
                            raise ingestion.ArtifactValidationError(
                                f"FEATURE-NULL-001: node type {node.name!r} column {column_name!r} contains nulls"
                            )
                        block = values.to_numpy(zero_copy_only=False)
                        self._validate_finite(block, code="FEATURE-FINITE-001", context=f"node type {node.name!r}")
                        array[offset : offset + rows, column_index - 1] = block
                offset += rows
            if offset != count:
                raise ingestion.ArtifactValidationError(
                    f"FEATURE-ALIGNMENT-001: node type {node.name!r} resolved {offset} rows, expected {count}"
                )
            array.flush()
        finally:
            del array
        _fsync_file(path)
        source_records = self._source_records(node.paths)
        return self._array_record(
            root,
            path,
            logical_dtype=node.feature_dtype,
            shape=(count, node.feature_width),
            count=count,
            extra={
                "node_type": node.name,
                "internal_key": internal_key,
                "representation": node.feature_representation,
                "feature_columns": list(node.feature_columns),
                "id_column": node.id_column,
                "id_dtype": node.id_dtype,
                "feature_width": node.feature_width,
                "arrow_dtype": _arrow_dtype_name(node.feature_dtype),
                "source_files": source_records,
                "source_sha256": _combined_source_sha(source_records),
            },
        )

    def _write_supervision(
        self,
        root: Path,
        *,
        target: Any,
        target_key: str,
        snapshots: Mapping[str, Path],
        connection: Any,
        pa: Any,
        pq: Any,
    ) -> dict[str, Any]:
        ingestion = _ingestion_module()
        supervision = self.ingestor.source.spec.supervision
        if supervision.label_dtype == "string":
            raise ingestion.ArtifactValidationError(
                "TARGET-DTYPE-001: string classification requires an explicit qualified vocabulary"
            )
        task = "classification" if supervision.label_dtype in _INTEGER_DTYPES else "regression"
        if supervision.label_dtype not in _INTEGER_DTYPES | _FLOAT_DTYPES:
            raise ingestion.ArtifactValidationError(
                f"TARGET-DTYPE-001: unsupported target dtype {supervision.label_dtype!r}"
            )
        if supervision.label_source == "nodes":
            paths = tuple(snapshots[relative] for relative in target.paths)
            id_column = target.id_column
        else:
            paths = tuple(snapshots[relative] for relative in supervision.label_paths)
            id_column = supervision.label_id_column
            if id_column is None:
                raise ingestion.ArtifactValidationError(
                    "SUPERVISION-POSITIONAL-001: keyed labels require an external-ID column"
                )
        schema = self._exact_role_schema(paths, pq=pq, context="target supervision")
        self._require_exact_id(
            schema,
            id_column,
            target.id_dtype,
            pa=pa,
            code="SUPERVISION-ID-CAST-001",
        )
        field = self._field(
            schema,
            supervision.label_column,
            "TARGET-SCHEMA-001",
            target.name,
        )
        expected = _arrow_type(pa, supervision.label_dtype)
        if not field.type.equals(expected):
            raise ingestion.ArtifactValidationError(
                f"TARGET-CAST-001: target column type {field.type} does not exactly match {expected}"
            )
        source_view = "supervision_source"
        mapping_view = "supervision_mapping"
        connection.read_parquet([str(path) for path in paths]).create_view(source_view)
        connection.read_parquet(str(self.index_build.indexes[target.name].node_ids_path)).create_view(mapping_view)
        row_count, non_null_ids, distinct_ids = connection.execute(
            f"SELECT COUNT(*), COUNT({_quote(id_column)}), COUNT(DISTINCT {_quote(id_column)}) FROM {_quote(source_view)}"
        ).fetchone()
        if non_null_ids != row_count:
            raise ingestion.ArtifactValidationError(
                "SUPERVISION-ID-NULL-001: target supervision contains null external IDs"
            )
        if distinct_ids != row_count:
            raise ingestion.ArtifactValidationError(
                "SUPERVISION-DUPLICATE-001: target supervision contains duplicate external IDs"
            )
        extra = connection.execute(
            f"SELECT COUNT(*) FROM {_quote(source_view)} s LEFT JOIN {_quote(mapping_view)} m "
            f"ON s.{_quote(id_column)} IS NOT DISTINCT FROM m.external_id WHERE m.local_ordinal IS NULL"
        ).fetchone()[0]
        if extra:
            raise ingestion.ArtifactValidationError(
                f"SUPERVISION-EXTRA-001: {extra} supervision IDs do not belong to target type {target.name!r}"
            )
        missing = connection.execute(
            f"SELECT COUNT(*) FROM {_quote(mapping_view)} m LEFT JOIN {_quote(source_view)} s "
            f"ON s.{_quote(id_column)} IS NOT DISTINCT FROM m.external_id WHERE s.{_quote(id_column)} IS NULL"
        ).fetchone()[0]
        if missing:
            raise ingestion.ArtifactValidationError(
                f"SUPERVISION-MISSING-001: {missing} target IDs have no supervision row"
            )
        non_null_labels = connection.execute(
            f"SELECT COUNT({_quote(supervision.label_column)}) FROM {_quote(source_view)}"
        ).fetchone()[0]
        if non_null_labels != row_count:
            raise ingestion.ArtifactValidationError(
                "TARGET-NULL-001: target supervision contains null labels"
            )
        vocabulary: dict[str, int | str] | None = None
        if task == "classification":
            minimum, maximum, distinct = connection.execute(
                f"SELECT MIN({_quote(supervision.label_column)}), MAX({_quote(supervision.label_column)}), "
                f"COUNT(DISTINCT {_quote(supervision.label_column)}) FROM {_quote(source_view)}"
            ).fetchone()
            if minimum != 0 or maximum is None or maximum + 1 != distinct:
                raise ingestion.ArtifactValidationError(
                    "TARGET-VOCABULARY-001: classification labels must form a qualified zero-based contiguous vocabulary"
                )
            vocabulary = {
                "kind": "zero_based_contiguous",
                "size": int(distinct),
                "minimum": int(minimum),
                "maximum": int(maximum),
            }
        query = (
            f"SELECT m.local_ordinal, s.{_quote(supervision.label_column)} FROM {_quote(source_view)} s "
            f"INNER JOIN {_quote(mapping_view)} m ON s.{_quote(id_column)} "
            "IS NOT DISTINCT FROM m.external_id ORDER BY m.local_ordinal"
        )
        count = self.index_build.indexes[target.name].row_count
        path = root / "nodes" / target_key / "y.npy"
        record = self._write_ordered_vector(
            root,
            path,
            query=query,
            connection=connection,
            logical_dtype=supervision.label_dtype,
            count=count,
            null_code="TARGET-NULL-001",
            finite_code="TARGET-FINITE-001" if task == "regression" else None,
            alignment_code="SUPERVISION-ALIGNMENT-001",
        )
        source_records = self._source_records(
            supervision.label_paths if supervision.label_source == "dataset" else target.paths
        )
        return {
            **record,
            "target_node_type": target.name,
            "target_internal_key": target_key,
            "task": task,
            "vocabulary": vocabulary,
            "source": supervision.label_source,
            "join": "external_id",
            "id_column": id_column,
            "label_column": supervision.label_column,
            "resolved_count": count,
            "source_files": source_records,
            "source_sha256": _combined_source_sha(source_records),
        }

    def _write_splits(
        self,
        root: Path,
        *,
        target: Any,
        target_key: str,
        snapshots: Mapping[str, Path],
        connection: Any,
        pa: Any,
        pq: Any,
    ) -> dict[str, Any]:
        ingestion = _ingestion_module()
        mapping_view = "split_target_mapping"
        connection.read_parquet(str(self.index_build.indexes[target.name].node_ids_path)).create_view(mapping_view)
        target_count = self.index_build.indexes[target.name].row_count
        result: dict[str, Any] = {}
        for tag_index, split in enumerate(self.ingestor.source.spec.supervision.split_registry.sets):
            phases: dict[str, dict[str, Any]] = {}
            tables: dict[str, str] = {}
            for phase_index, phase in enumerate(_PHASES):
                relative = getattr(split, phase)
                snapshot = snapshots[relative]
                schema = self._exact_role_schema((snapshot,), pq=pq, context=f"split {split.tag!r}/{phase}")
                self._require_exact_id(
                    schema,
                    target.id_column,
                    target.id_dtype,
                    pa=pa,
                    code="SPLIT-ID-CAST-001",
                )
                view = f"split_source_{tag_index}_{phase_index}"
                table = f"split_ordinals_{tag_index}_{phase_index}"
                connection.read_parquet(str(snapshot)).create_view(view)
                row_count, non_null, distinct = connection.execute(
                    f"SELECT COUNT(*), COUNT({_quote(target.id_column)}), "
                    f"COUNT(DISTINCT {_quote(target.id_column)}) FROM {_quote(view)}"
                ).fetchone()
                if non_null != row_count:
                    raise ingestion.ArtifactValidationError(
                        f"SPLIT-NULL-001: split {split.tag!r}/{phase} contains null IDs"
                    )
                if distinct != row_count:
                    raise ingestion.ArtifactValidationError(
                        f"SPLIT-DUPLICATE-001: split {split.tag!r}/{phase} contains duplicate IDs"
                    )
                unresolved = connection.execute(
                    f"SELECT COUNT(*) FROM {_quote(view)} s LEFT JOIN {_quote(mapping_view)} m "
                    f"ON s.{_quote(target.id_column)} IS NOT DISTINCT FROM m.external_id "
                    "WHERE m.local_ordinal IS NULL"
                ).fetchone()[0]
                if unresolved:
                    raise ingestion.ArtifactValidationError(
                        f"SPLIT-UNRESOLVED-001: split {split.tag!r}/{phase} has {unresolved} unresolved target IDs"
                    )
                connection.execute(
                    f"CREATE TABLE {_quote(table)} AS SELECT m.local_ordinal FROM {_quote(view)} s "
                    f"INNER JOIN {_quote(mapping_view)} m ON s.{_quote(target.id_column)} "
                    "IS NOT DISTINCT FROM m.external_id ORDER BY m.local_ordinal"
                )
                tables[phase] = table
                path = root / "splits" / split.tag / f"{phase}_ids.npy"
                record = self._write_ordered_vector(
                    root,
                    path,
                    query=f"SELECT local_ordinal, local_ordinal FROM {_quote(table)} ORDER BY local_ordinal",
                    connection=connection,
                    logical_dtype="int64",
                    count=int(row_count),
                    null_code="SPLIT-NULL-001",
                    finite_code=None,
                    alignment_code=None,
                )
                source_record = self._source_records((relative,))[0]
                phases[phase] = {
                    **record,
                    "source_relative_path": relative,
                    "source_sha256": source_record["sha256"],
                    "resolved_count": int(row_count),
                    "sorted": True,
                    "unique": True,
                    "target_internal_key": target_key,
                }
            for left_index, left in enumerate(_PHASES):
                for right in _PHASES[left_index + 1 :]:
                    overlap = connection.execute(
                        f"SELECT COUNT(*) FROM {_quote(tables[left])} l INNER JOIN {_quote(tables[right])} r "
                        "ON l.local_ordinal = r.local_ordinal"
                    ).fetchone()[0]
                    if overlap:
                        raise ingestion.ArtifactValidationError(
                            f"SPLIT-DISJOINT-001: split {split.tag!r} phases {left!r} and {right!r} overlap"
                        )
            union_count = sum(int(phases[phase]["count"]) for phase in _PHASES)
            if split.coverage == "complete" and union_count != target_count:
                raise ingestion.ArtifactValidationError(
                    f"SPLIT-COVERAGE-001: complete split {split.tag!r} covers {union_count} of {target_count} target IDs"
                )
            result[split.tag] = {
                "coverage": split.coverage,
                "qualified": split.qualified,
                "phases": phases,
                "union_count": union_count,
                "supervision_population": target_count,
                "coverage_complete": union_count == target_count,
                "pairwise_disjoint": True,
                "cross_tag_overlap": "allowed",
            }
        return result

    def _write_ordered_vector(
        self,
        root: Path,
        path: Path,
        *,
        query: str,
        connection: Any,
        logical_dtype: str,
        count: int,
        null_code: str,
        finite_code: str | None,
        alignment_code: str | None,
    ) -> dict[str, Any]:
        ingestion = _ingestion_module()
        path.parent.mkdir(parents=True, exist_ok=True)
        array = np.lib.format.open_memmap(
            path,
            mode="w+",
            dtype=_numpy_dtype(logical_dtype),
            shape=(count,),
        )
        offset = 0
        try:
            reader = connection.execute(query).to_arrow_reader(
                self.ingestor.source.spec.ingestion.record_batch_rows
            )
            for batch in reader:
                rows = batch.num_rows
                self._observe_batch(rows)
                ordinals = batch.column(0).to_numpy(zero_copy_only=False)
                if alignment_code is not None:
                    expected = np.arange(offset, offset + rows, dtype=np.int64)
                    if not np.array_equal(ordinals, expected):
                        raise ingestion.ArtifactValidationError(
                            f"{alignment_code}: resolved target rows are not exact canonical ordinals"
                        )
                values = batch.column(1)
                if values.null_count:
                    raise ingestion.ArtifactValidationError(f"{null_code}: array contains null values")
                block = values.to_numpy(zero_copy_only=False)
                if finite_code is not None:
                    self._validate_finite(block, code=finite_code, context="target supervision")
                array[offset : offset + rows] = block
                offset += rows
            if offset != count:
                code = alignment_code or "SPLIT-COUNT-001"
                raise ingestion.ArtifactValidationError(
                    f"{code}: wrote {offset} values, expected {count}"
                )
            array.flush()
        finally:
            del array
        _fsync_file(path)
        return self._array_record(
            root,
            path,
            logical_dtype=logical_dtype,
            shape=(count,),
            count=count,
            extra={},
        )

    def _array_record(
        self,
        root: Path,
        path: Path,
        *,
        logical_dtype: str,
        shape: tuple[int, ...],
        count: int,
        extra: Mapping[str, Any],
    ) -> dict[str, Any]:
        ingestion = _ingestion_module()
        return {
            **extra,
            "relative_path": path.relative_to(root).as_posix(),
            "dtype": logical_dtype,
            "storage_dtype": _numpy_dtype(logical_dtype).str,
            "shape": list(shape),
            "count": count,
            "byte_size": path.stat().st_size,
            "file_sha256": ingestion._sha256_file(path),
            "content_sha256": _array_content_sha(path, self.ingestor.source.spec.ingestion.record_batch_rows),
        }

    def _source_records(self, relatives: tuple[str, ...]) -> list[dict[str, Any]]:
        inventory_by_path = {
            entry.relative_path: entry for entry in self.index_build.inventory.files
        }
        return [
            {
                "relative_path": relative,
                "byte_size": inventory_by_path[relative].byte_size,
                "sha256": inventory_by_path[relative].sha256,
                "schema_fingerprint": inventory_by_path[relative].schema_fingerprint,
            }
            for relative in relatives
        ]

    def _exact_role_schema(self, paths: tuple[Path, ...], *, pq: Any, context: str) -> Any:
        ingestion = _ingestion_module()
        schemas = tuple(pq.ParquetFile(path).schema_arrow for path in paths)
        serialized = {hashlib.sha256(schema.serialize().to_pybytes()).hexdigest() for schema in schemas}
        if len(serialized) != 1:
            raise ingestion.ArtifactValidationError(
                f"SCHEMA-DRIFT-001: {context} fragments have different exact Arrow schemas"
            )
        return schemas[0]

    def _require_exact_id(self, schema: Any, column: str, dtype: str, *, pa: Any, code: str) -> None:
        ingestion = _ingestion_module()
        field = self._field(schema, column, code, "external-ID source")
        expected = _id_arrow_type(pa, dtype)
        if not field.type.equals(expected):
            raise ingestion.ArtifactValidationError(
                f"{code}: external-ID column {column!r} type {field.type} does not exactly match {expected}"
            )

    @staticmethod
    def _field(schema: Any, column: str, code: str, context: str) -> Any:
        ingestion = _ingestion_module()
        index = schema.get_field_index(column)
        if index < 0:
            raise ingestion.ArtifactValidationError(
                f"{code}: {context} is missing required column {column!r}"
            )
        return schema.field(index)

    @staticmethod
    def _validate_finite(array: np.ndarray, *, code: str, context: str) -> None:
        if np.issubdtype(array.dtype, np.floating) and not np.isfinite(array).all():
            ingestion = _ingestion_module()
            raise ingestion.ArtifactValidationError(
                f"{code}: {context} contains NaN or infinity"
            )

    def _observe_batch(self, rows: int) -> None:
        limit = self.ingestor.source.spec.ingestion.record_batch_rows
        if rows > limit:
            ingestion = _ingestion_module()
            raise ingestion.ArtifactValidationError(
                f"BATCH-BOUND-001: observed {rows} rows, configured maximum is {limit}"
            )
        self._max_batch_rows = max(self._max_batch_rows, rows)

    def _publish(self, temporary_root: Path, stage_root: Path) -> None:
        ingestion = _ingestion_module()
        destination = stage_root / "arrays"
        if os.path.lexists(destination):
            raise ingestion.ArtifactValidationError(
                "INCOMPLETE-ARRAY-STAGE-001: arrays destination already exists"
            )
        os.replace(temporary_root, destination)
        ingestion._fsync_directory(stage_root)

    def _finalize_top_completion(
        self,
        result: TypedGraphArrayBuild,
    ) -> None:
        ingestion = _ingestion_module()
        stage_root = self.index_build.stage_root
        previous = ingestion._read_json(stage_root / "build.complete.json")
        if (
            previous.get("stage") == "typed_graph_arrays"
            and previous.get("array_content_sha256") == result.content_sha256
        ):
            return
        previous["stage"] = "typed_graph_arrays"
        previous["array_behavior_version"] = _ARRAY_BEHAVIOR_VERSION
        previous["array_content_sha256"] = result.content_sha256
        previous["outputs"] = ingestion._stage_output_checksums(stage_root)
        ingestion._atomic_json(stage_root / "build.complete.json", previous)
        ingestion._fsync_directory(stage_root)

    def _array_outputs(self, root: Path) -> dict[str, str]:
        ingestion = _ingestion_module()
        outputs: dict[str, str] = {}
        for relative_root in ("nodes", "splits"):
            for path in sorted((root / relative_root).rglob("*")):
                if path.is_symlink():
                    raise ingestion.ArtifactValidationError(
                        f"UNKNOWN-ARTIFACT-001: symlink in array stage: {path}"
                    )
                if path.is_file():
                    outputs[path.relative_to(root).as_posix()] = ingestion._sha256_file(path)
        outputs["arrays.json"] = ingestion._sha256_file(root / "arrays.json")
        return outputs

    def _open_validated(self, *, resumed: bool) -> TypedGraphArrayBuild:
        ingestion = _ingestion_module()
        root = self.index_build.stage_root / "arrays"
        completion = ingestion._read_json(root / "arrays.complete.json")
        if (
            completion.get("stage") != "typed_graph_arrays"
            or completion.get("array_behavior_version") != _ARRAY_BEHAVIOR_VERSION
            or completion.get("behavior_version") != ingestion._BEHAVIOR_VERSION
            or completion.get("input_fingerprint") != self.index_build.inventory.source_fingerprint
            or completion.get("config_fingerprint") != self.index_build.inventory.config_fingerprint
            or completion.get("dependency_versions") != dict(self.index_build.inventory.dependency_versions)
            or completion.get("reopened_and_validated") is not True
        ):
            raise ingestion.ArtifactValidationError(
                "COMPLETION-EVIDENCE-001: array completion evidence changed"
            )
        disk_admission = completion.get("disk_admission")
        observed_disk = (
            disk_admission.get("observed")
            if isinstance(disk_admission, dict)
            else None
        )
        if (
            not isinstance(disk_admission, dict)
            or disk_admission.get("requirements")
            != self._array_disk_requirements()
            or not isinstance(observed_disk, dict)
            or observed_disk.get("final_device")
            != self.index_build.inventory.final_device
            or observed_disk.get("temporary_device")
            != self.index_build.inventory.temporary_device
        ):
            raise ingestion.ArtifactValidationError(
                "DISK-EVIDENCE-001: Task 3 completion is not bound to its "
                "preallocation admission"
            )
        self._validate_resume_disk(
            disk_admission["requirements"],
            root,
        )
        if completion.get("prepared_array_stage_bytes") != _tree_bytes(root):
            raise ingestion.ArtifactValidationError(
                "DISK-EVIDENCE-001: prepared array stage byte evidence changed"
            )
        outputs = ingestion._validated_output_map(completion.get("outputs"))
        observed = self._array_outputs(root)
        if outputs != observed:
            raise ingestion.ArtifactValidationError(
                "COMPLETION-EVIDENCE-001: array output set or checksums changed"
            )
        metadata = ingestion._read_json(root / "arrays.json")
        self._validate_array_tree(root, metadata, check_completion=True)
        if completion.get("content_sha256") != metadata.get("content_sha256"):
            raise ingestion.ArtifactValidationError(
                "COMPLETION-EVIDENCE-001: array content identity changed"
            )
        return TypedGraphArrayBuild(
            inventory=self.index_build.inventory,
            stage_root=self.index_build.stage_root,
            artifact_root=root,
            content_sha256=metadata["content_sha256"],
            active_tag=metadata["active_split_tag"],
            resumed=resumed,
            record_batch_rows=metadata["record_batch_rows"],
            max_record_batch_rows=metadata["max_record_batch_rows"],
        )

    def _validate_array_tree(
        self,
        root: Path,
        metadata: Mapping[str, Any],
        *,
        check_completion: bool,
    ) -> None:
        ingestion = _ingestion_module()
        if (
            metadata.get("array_behavior_version") != _ARRAY_BEHAVIOR_VERSION
            or metadata.get("input_fingerprint") != self.index_build.inventory.source_fingerprint
            or metadata.get("config_fingerprint") != self.index_build.inventory.config_fingerprint
            or metadata.get("dependency_versions") != dict(self.index_build.inventory.dependency_versions)
        ):
            raise ingestion.ArtifactValidationError(
                "COMPLETION-EVIDENCE-001: array semantic evidence changed"
            )
        resource_evidence = metadata.get("resource_evidence")
        if (
            not isinstance(resource_evidence, dict)
            or resource_evidence.get("disk_requirements")
            != self._array_disk_requirements()
        ):
            raise ingestion.ArtifactValidationError(
                "DISK-EVIDENCE-001: Task 3 metadata disk estimate changed"
            )
        if ingestion._sha256_json(metadata.get("content_identity")) != metadata.get("content_sha256"):
            raise ingestion.ArtifactValidationError(
                "CHECKSUM-001: array content identity checksum changed"
            )
        nodes = metadata.get("nodes")
        expected_nodes = {
            f"n{ordinal:04d}": node
            for ordinal, node in enumerate(self.ingestor.source.spec.node_types)
        }
        if not isinstance(nodes, dict) or set(nodes) != set(expected_nodes):
            raise ingestion.ArtifactValidationError(
                "COMPLETION-EVIDENCE-001: array node set changed"
            )
        for key, node in expected_nodes.items():
            record = nodes[key]
            source_files = self._source_records(node.paths)
            node_count = len(self.index_build.indexes[node.name])
            expected_feature = {
                "relative_path": f"nodes/{key}/x.npy",
                "node_type": node.name,
                "internal_key": key,
                "id_column": node.id_column,
                "id_dtype": node.id_dtype,
                "representation": node.feature_representation,
                "feature_columns": list(node.feature_columns),
                "feature_width": node.feature_width,
                "dtype": node.feature_dtype,
                "arrow_dtype": _arrow_dtype_name(node.feature_dtype),
                "storage_dtype": _numpy_dtype(node.feature_dtype).str,
                "shape": [node_count, node.feature_width],
                "count": node_count,
                "source_files": source_files,
                "source_sha256": _combined_source_sha(source_files),
            }
            if not isinstance(record, dict) or any(
                record.get(field) != value
                for field, value in expected_feature.items()
            ):
                raise ingestion.ArtifactValidationError(
                    f"FEATURE-EVIDENCE-001: feature array for node type "
                    f"{node.name!r} is not bound to its current "
                    "NodeTypeSpec/inventory/index contract"
                )
            self._validate_array_record(root, record, finite=True)
        supervision = metadata.get("supervision")
        supervision_spec = self.ingestor.source.spec.supervision
        target = next(
            node
            for node in self.ingestor.source.spec.node_types
            if node.name == supervision_spec.target_node_type
        )
        target_key = next(
            key for key, node in expected_nodes.items() if node.name == target.name
        )
        target_count = len(self.index_build.indexes[target.name])
        task = (
            "classification"
            if supervision_spec.label_dtype in _INTEGER_DTYPES
            else "regression"
        )
        id_column = (
            target.id_column
            if supervision_spec.label_source == "nodes"
            else supervision_spec.label_id_column
        )
        source_paths = (
            target.paths
            if supervision_spec.label_source == "nodes"
            else supervision_spec.label_paths
        )
        source_files = self._source_records(source_paths)
        expected_supervision = {
            "target_node_type": target.name,
            "target_internal_key": target_key,
            "task": task,
            "source": supervision_spec.label_source,
            "join": "external_id",
            "id_column": id_column,
            "label_column": supervision_spec.label_column,
            "dtype": supervision_spec.label_dtype,
            "storage_dtype": _numpy_dtype(supervision_spec.label_dtype).str,
            "shape": [target_count],
            "count": target_count,
            "resolved_count": target_count,
            "source_files": source_files,
            "source_sha256": _combined_source_sha(source_files),
        }
        if not isinstance(supervision, dict) or any(
            supervision.get(field) != value
            for field, value in expected_supervision.items()
        ):
            raise ingestion.ArtifactValidationError(
                "TARGET-EVIDENCE-001: supervision is not bound to the current "
                "target/source/schema contract"
            )
        self._validate_array_record(
            root,
            supervision,
            finite=task == "regression",
        )
        target_values = np.load(
            root / supervision["relative_path"],
            mmap_mode="r",
            allow_pickle=False,
        )
        if task == "classification":
            if check_completion:
                self._validate_classification_vocabulary(
                    target_values,
                    supervision.get("vocabulary"),
                )
            elif not isinstance(supervision.get("vocabulary"), dict):
                raise ingestion.ArtifactValidationError(
                    "TARGET-VOCABULARY-001: classification vocabulary "
                    "evidence is absent"
                )
        elif supervision.get("vocabulary") is not None:
            raise ingestion.ArtifactValidationError(
                "TARGET-EVIDENCE-001: regression target carries a vocabulary"
            )
        splits = metadata.get("splits")
        registry = self.ingestor.source.spec.supervision.split_registry
        if not isinstance(splits, dict) or set(splits) != {split.tag for split in registry.sets}:
            raise ingestion.ArtifactValidationError("COMPLETION-EVIDENCE-001: split registry evidence changed")
        target_count = len(self.index_build.indexes[target.name])
        for split_spec in registry.sets:
            split = splits[split_spec.tag]
            expected_split = {
                "coverage": split_spec.coverage,
                "qualified": split_spec.qualified,
                "supervision_population": target_count,
                "cross_tag_overlap": "allowed",
                "pairwise_disjoint": True,
            }
            if not isinstance(split, dict) or any(
                split.get(field) != value
                for field, value in expected_split.items()
            ):
                raise ingestion.ArtifactValidationError(
                    f"SPLIT-EVIDENCE-001: split {split_spec.tag!r} is not bound "
                    "to the current registry/target contract"
                )
            if set(split.get("phases", {})) != set(_PHASES):
                raise ingestion.ArtifactValidationError(
                    f"SPLIT-PHASES-001: split {split_spec.tag!r} lacks an exact phase triplet"
                )
            arrays: dict[str, np.ndarray] = {}
            inventory_by_path = {
                entry.relative_path: entry
                for entry in self.index_build.inventory.files
            }
            for phase in _PHASES:
                record = split["phases"][phase]
                source_relative = getattr(split_spec, phase)
                source_entry = inventory_by_path[source_relative]
                expected_phase = {
                    "source_relative_path": source_relative,
                    "source_sha256": source_entry.sha256,
                    "resolved_count": source_entry.row_count,
                    "count": source_entry.row_count,
                    "shape": [source_entry.row_count],
                    "dtype": "int64",
                    "storage_dtype": _numpy_dtype("int64").str,
                    "sorted": True,
                    "unique": True,
                    "target_internal_key": target_key,
                }
                if not isinstance(record, dict) or any(
                    record.get(field) != value
                    for field, value in expected_phase.items()
                ):
                    raise ingestion.ArtifactValidationError(
                        f"SPLIT-EVIDENCE-001: split "
                        f"{split_spec.tag!r}/{phase} is not bound to its "
                        "declared source/target contract"
                    )
                self._validate_array_record(root, record, finite=False)
                array = np.load(
                    root / record["relative_path"],
                    mmap_mode="r",
                    allow_pickle=False,
                )
                if array.ndim != 1 or not _strictly_increasing(
                    array,
                    self.ingestor.source.spec.ingestion.record_batch_rows,
                ):
                    raise ingestion.ArtifactValidationError(
                        f"SPLIT-DUPLICATE-001: split "
                        f"{split_spec.tag!r}/{phase} is not sorted and unique"
                    )
                if len(array) and (
                    int(array[0]) < 0 or int(array[-1]) >= target_count
                ):
                    raise ingestion.ArtifactValidationError(
                        f"SPLIT-RANGE-001: split {split_spec.tag!r}/{phase} "
                        "contains an ordinal outside the target population"
                    )
                arrays[phase] = array
            if any(
                _sorted_arrays_overlap(arrays[left], arrays[right])
                for index, left in enumerate(_PHASES)
                for right in _PHASES[index + 1 :]
            ):
                raise ingestion.ArtifactValidationError(
                    f"SPLIT-DISJOINT-001: split {split_spec.tag!r} phases overlap"
                )
            union_count = sum(len(array) for array in arrays.values())
            if (
                split.get("union_count") != union_count
                or split.get("coverage_complete") != (union_count == target_count)
            ):
                raise ingestion.ArtifactValidationError(
                    f"SPLIT-EVIDENCE-001: split {split_spec.tag!r} coverage "
                    "evidence changed"
                )
            if split_spec.coverage == "complete" and not _is_exact_population(
                arrays,
                target_count,
            ):
                raise ingestion.ArtifactValidationError(
                    f"SPLIT-COVERAGE-001: complete split "
                    f"{split_spec.tag!r} is not the exact target population"
                )
        expected_identity = {
            "array_behavior_version": _ARRAY_BEHAVIOR_VERSION,
            "input_fingerprint": self.index_build.inventory.source_fingerprint,
            "config_fingerprint": self.index_build.inventory.config_fingerprint,
            "active_split_tag": registry.active_tag,
            "nodes": {
                key: record["content_sha256"]
                for key, record in nodes.items()
            },
            "supervision": supervision["content_sha256"],
            "splits": {
                tag: {
                    phase: split["phases"][phase]["content_sha256"]
                    for phase in _PHASES
                }
                for tag, split in splits.items()
            },
        }
        if metadata.get("content_identity") != expected_identity:
            raise ingestion.ArtifactValidationError(
                "CHECKSUM-001: content identity is not bound to array records"
            )
        if metadata.get("active_split_tag") != registry.active_tag:
            raise ingestion.ArtifactValidationError(
                "COMPLETION-EVIDENCE-001: active split tag changed"
            )
        if check_completion and not (root / "arrays.complete.json").is_file():
            raise ingestion.ArtifactValidationError(
                "INCOMPLETE-ARRAY-STAGE-001: completion record is absent"
            )

    def _validate_array_record(
        self,
        root: Path,
        record: Mapping[str, Any],
        *,
        finite: bool,
    ) -> None:
        ingestion = _ingestion_module()
        relative = record.get("relative_path")
        if not isinstance(relative, str):
            raise ingestion.ArtifactValidationError("COMPLETION-EVIDENCE-001: array path missing")
        path = ingestion._safe_artifact_path(root, relative)
        array = np.load(path, mmap_mode="r", allow_pickle=False)
        expected_dtype = record.get("storage_dtype")
        expected_shape = tuple(record.get("shape", ()))
        if array.dtype.str != expected_dtype or array.shape != expected_shape or array.shape[0] != record.get("count"):
            raise ingestion.ArtifactValidationError(
                f"ARRAY-SHAPE-001: dtype/shape/count changed for {relative!r}"
            )
        if path.stat().st_size != record.get("byte_size") or ingestion._sha256_file(path) != record.get("file_sha256"):
            raise ingestion.ArtifactValidationError(
                f"CHECKSUM-001: file checksum changed for {relative!r}"
            )
        if _array_content_sha(path, self.ingestor.source.spec.ingestion.record_batch_rows) != record.get("content_sha256"):
            raise ingestion.ArtifactValidationError(
                f"CHECKSUM-001: array content checksum changed for {relative!r}"
            )
        if finite:
            rows = self.ingestor.source.spec.ingestion.record_batch_rows
            for start in range(0, array.shape[0], rows):
                self._validate_finite(np.asarray(array[start : start + rows]), code="ARRAY-FINITE-001", context=relative)


    def _validate_classification_vocabulary(
        self,
        values: np.ndarray,
        vocabulary: Any,
    ) -> None:
        ingestion = _ingestion_module()
        if not isinstance(vocabulary, dict):
            raise ingestion.ArtifactValidationError(
                "TARGET-VOCABULARY-001: qualified classification vocabulary "
                "evidence is absent"
            )
        size = vocabulary.get("size")
        expected = {
            "kind": "zero_based_contiguous",
            "size": size,
            "minimum": 0,
            "maximum": size - 1 if isinstance(size, int) else None,
        }
        if (
            isinstance(size, bool)
            or not isinstance(size, int)
            or size <= 0
            or vocabulary != expected
            or size > len(values)
        ):
            raise ingestion.ArtifactValidationError(
                "TARGET-VOCABULARY-001: classification vocabulary evidence "
                "is not qualified zero-based contiguous"
            )
        temporary_root = self.ingestor._new_ephemeral_root(
            self.index_build.inventory,
            purpose="array-vocabulary-validation",
        )
        temporary_root.mkdir(parents=True, exist_ok=False)
        seen: np.memmap | None = None
        rows = self.ingestor.source.spec.ingestion.record_batch_rows
        try:
            seen_path = temporary_root / "seen.npy"
            seen = np.lib.format.open_memmap(
                seen_path,
                mode="w+",
                dtype=np.uint8,
                shape=(size,),
            )
            seen[:] = 0
            for start in range(0, len(values), rows):
                block = np.asarray(values[start : start + rows])
                if (
                    len(block)
                    and (int(block.min()) < 0 or int(block.max()) >= size)
                ):
                    raise ingestion.ArtifactValidationError(
                        "TARGET-VOCABULARY-001: classification value is "
                        "outside the qualified vocabulary"
                    )
                seen[block] = 1
            for start in range(0, size, rows):
                if not np.all(seen[start : start + rows]):
                    raise ingestion.ArtifactValidationError(
                        "TARGET-VOCABULARY-001: classification labels do not "
                        "realize the qualified contiguous vocabulary"
                    )
        finally:
            if seen is not None:
                del seen
            shutil.rmtree(temporary_root)

def _ingestion_module() -> Any:
    from topobench.data.stores import typed_graph_ingestion

    return typed_graph_ingestion


def _arrow_type(pa: Any, dtype: str) -> Any:
    return {
        "float16": pa.float16(),
        "float32": pa.float32(),
        "float64": pa.float64(),
        "int8": pa.int8(),
        "int16": pa.int16(),
        "int32": pa.int32(),
        "int64": pa.int64(),
        "uint8": pa.uint8(),
        "uint16": pa.uint16(),
        "uint32": pa.uint32(),
        "uint64": pa.uint64(),
        "string": pa.string(),
    }[dtype]


def _arrow_dtype_name(dtype: str) -> str:
    return {
        "float16": "halffloat",
        "float32": "float",
        "float64": "double",
        "int8": "int8",
        "int16": "int16",
        "int32": "int32",
        "int64": "int64",
        "uint8": "uint8",
        "uint16": "uint16",
        "uint32": "uint32",
        "uint64": "uint64",
    }[dtype]


def _id_arrow_type(pa: Any, dtype: str) -> Any:
    return {"int64": pa.int64(), "uint64": pa.uint64(), "string": pa.string()}[dtype]


def _numpy_dtype(dtype: str) -> np.dtype[Any]:
    return np.dtype(dtype).newbyteorder("<")


def _quote(value: str) -> str:
    return '"' + value.replace('"', '""') + '"'


def _fsync_file(path: Path) -> None:
    with path.open("rb") as stream:
        os.fsync(stream.fileno())


def _array_content_sha(path: Path, batch_rows: int) -> str:
    array = np.load(path, mmap_mode="r", allow_pickle=False)
    digest = hashlib.sha256()
    for start in range(0, array.shape[0], batch_rows):
        block = np.ascontiguousarray(array[start : start + batch_rows])
        digest.update(memoryview(block).cast("B"))
    return digest.hexdigest()


def _combined_source_sha(records: list[dict[str, Any]]) -> str:
    if len(records) == 1:
        return records[0]["sha256"]
    encoded = json.dumps(records, sort_keys=True, separators=(",", ":")).encode()
    return hashlib.sha256(encoded).hexdigest()


def _strictly_increasing(array: np.ndarray, batch_rows: int) -> bool:
    if len(array) < 2:
        return True
    previous = int(array[0])
    for start in range(1, len(array), batch_rows):
        block = np.asarray(array[start : start + batch_rows])
        if int(block[0]) <= previous:
            return False
        if len(block) > 1 and not np.all(block[:-1] < block[1:]):
            return False
        previous = int(block[-1])
    return True


def _is_exact_population(
    arrays: Mapping[str, np.ndarray],
    population: int,
) -> bool:
    # The caller has already proved every phase sorted/unique, pairwise
    # disjoint, and in range. A union of ``population`` distinct members of
    # [0, population) is therefore exactly the full ordinal population.
    return (
        sum(len(array) for array in arrays.values()) == population
        and all(
            not len(array)
            or (
                int(array[0]) >= 0
                and int(array[-1]) < population
            )
            for array in arrays.values()
        )
    )


def _tree_bytes(root: Path) -> int:
    return sum(
        path.stat().st_size
        for path in root.rglob("*")
        if path.is_file() and not path.is_symlink()
    )


def _metadata_array_bytes(metadata: Mapping[str, Any]) -> int:
    node_bytes = sum(
        int(record["byte_size"]) for record in metadata["nodes"].values()
    )
    supervision_bytes = int(metadata["supervision"]["byte_size"])
    split_bytes = sum(
        int(phase["byte_size"])
        for split in metadata["splits"].values()
        for phase in split["phases"].values()
    )
    return node_bytes + supervision_bytes + split_bytes


def _sorted_arrays_overlap(left: np.ndarray, right: np.ndarray) -> bool:
    left_index = 0
    right_index = 0
    while left_index < len(left) and right_index < len(right):
        left_value = int(left[left_index])
        right_value = int(right[right_index])
        if left_value == right_value:
            return True
        if left_value < right_value:
            left_index += 1
        else:
            right_index += 1
    return False


__all__ = ["TypedGraphArrayBuild", "TypedGraphArrayWriter"]
