"""Bounded, exact typed-relation joins and destination-oriented CSC storage.

DuckDB and PyArrow are imported only when a relation build or semantic reopen is
requested. Mapped edges stream one configured record batch at a time into the
adjacency arrays; a scratch destination mmap and compact canonical permutation
replace DuckDB's otherwise blocking joined-edge sort.
"""

from __future__ import annotations

import hashlib
import io
import json
import os
import shutil
import uuid
from collections.abc import Mapping
from dataclasses import dataclass
from pathlib import Path
from typing import TYPE_CHECKING, Any

import numpy as np

if TYPE_CHECKING:
    from topobench.data.stores.typed_graph_arrays import TypedGraphArrayBuild
    from topobench.data.stores.typed_graph_ingestion import (
        ExternalNodeIndexBuild,
        ParquetTypedGraphIngestor,
        SourceInventory,
    )

_RELATION_BEHAVIOR_VERSION = "typed-graph-relations-v2"
_INTEGER_DTYPES = (
    "int8",
    "int16",
    "int32",
    "int64",
    "uint8",
    "uint16",
    "uint32",
    "uint64",
)
_FLOAT_DTYPES = ("float16", "float32", "float64")


@dataclass(frozen=True, slots=True)
class TypedGraphRelationBuild:
    """One checksum- and source-validated canonical relation subtree."""

    inventory: SourceInventory
    stage_root: Path
    artifact_root: Path
    content_sha256: str
    resumed: bool
    record_batch_rows: int
    max_record_batch_rows: int


class TypedGraphRelationWriter:
    """Externally join typed endpoints and atomically publish exact CSC arrays."""

    def __init__(
        self,
        ingestor: ParquetTypedGraphIngestor,
        index_build: ExternalNodeIndexBuild,
        array_build: TypedGraphArrayBuild,
    ) -> None:
        self.ingestor = ingestor
        self.index_build = index_build
        self.array_build = array_build
        self._max_batch_rows = 0

    def build(self) -> TypedGraphRelationBuild:
        """Build once or semantically reopen one addressed relations subtree."""
        ingestion = _ingestion_module()
        inventory = self.index_build.inventory
        stage_root = self.index_build.stage_root
        artifact_root = stage_root / "relations"
        pa, pq, duckdb = ingestion._parquet_dependencies()
        with self.ingestor._build_lock(self.ingestor.lock_path(inventory)):
            self.ingestor._validate_inventory_current(inventory, pa=pa, pq=pq)
            self._validate_task3_arrays()
            if (artifact_root / "relations.complete.json").is_file():
                try:
                    result = self._open_validated(
                        resumed=True,
                        pa=pa,
                        pq=pq,
                        duckdb=duckdb,
                    )
                except ingestion.ArtifactValidationError:
                    self._quarantine_relations(artifact_root)
                    raise
                self._finalize_top_completion(result)
                return result
            if os.path.lexists(artifact_root):
                raise ingestion.ArtifactValidationError(
                    "INCOMPLETE-RELATION-STAGE-001: uncompleted relations artifact"
                )
            if list(stage_root.glob(".relations-tmp-*")):
                raise ingestion.ArtifactValidationError(
                    "INCOMPLETE-RELATION-STAGE-001: uncommitted relation staging exists"
                )

            coarse_requirements = self._relation_disk_requirements()
            coarse_admission = self._admit_relation_disk(
                coarse_requirements,
                exact=False,
            )
            temporary_root = stage_root / f".relations-tmp-{uuid.uuid4().hex}"
            spill_root = self.ingestor._new_ephemeral_root(
                inventory,
                purpose="relation-spill",
            )
            temporary_root.mkdir(parents=False, exist_ok=False)
            spill_created = False
            try:
                spill_root.mkdir(parents=True, exist_ok=False)
                spill_created = True
                with self.ingestor._immutable_source_snapshot(
                    inventory,
                    purpose="relation-source-snapshot",
                    pa=pa,
                    pq=pq,
                ) as snapshot_stage:
                    snapshots, _ = snapshot_stage
                    connection = duckdb.connect(
                        str(spill_root / "relations.duckdb")
                    )
                    try:
                        self._configure_connection(connection, spill_root)
                        (
                            metadata,
                            exact_requirements,
                            exact_admission,
                        ) = self._write_all(
                            temporary_root,
                            snapshots=snapshots,
                            connection=connection,
                            pa=pa,
                            pq=pq,
                            coarse_requirements=coarse_requirements,
                        )
                        metadata["resource_evidence"] = {
                            "record_batch_rows": self.ingestor.source.spec.ingestion.record_batch_rows,
                            "max_record_batch_rows": self._max_batch_rows,
                            "duckdb_memory_limit_bytes": self.ingestor.source.spec.ingestion.memory_limit_bytes,
                            "duckdb_threads": self.ingestor.threads,
                            "snapshot_bytes": inventory.snapshot_bytes,
                            "snapshot_subtree": "ephemeral/relation-source-snapshot-*",
                            "snapshot_persisted": False,
                            "snapshot_bytes_accounted": True,
                            "spill_subtree": "ephemeral/relation-spill-*",
                            "disk_requirements": coarse_requirements,
                            "exact_disk_requirements": exact_requirements,
                            "canonical_permutation_bytes": exact_requirements[
                                "max_canonical_permutation_bytes"
                            ],
                            "bounded_memory": "O(record_batch_rows * declared_relation_width + edge_count * sizeof(intp)) plus DuckDB spill state",
                        }
                        self._audit_snapshot_streams(
                            temporary_root,
                            metadata,
                            snapshots=snapshots,
                            connection=connection,
                            pa=pa,
                            pq=pq,
                            view_prefix="prepublish",
                        )
                    finally:
                        connection.close()
                    metadata["max_record_batch_rows"] = self._max_batch_rows
                    identity = {
                        "relation_behavior_version": _RELATION_BEHAVIOR_VERSION,
                        "relations": {
                            key: {
                                "relation": value["relation"],
                                "semantic_sha256": value["semantic_sha256"],
                            }
                            for key, value in metadata["relations"].items()
                        },
                    }
                    metadata["content_identity"] = identity
                    metadata["content_sha256"] = ingestion._sha256_json(
                        identity
                    )
                    ingestion._atomic_json(
                        temporary_root / "relations.json", metadata
                    )
                    self._validate_relation_tree(
                        temporary_root,
                        metadata,
                        check_completion=False,
                    )
                    outputs = self._relation_outputs(temporary_root)
                    completion = {
                        "stage": "typed_graph_relations",
                        "relation_behavior_version": _RELATION_BEHAVIOR_VERSION,
                        "behavior_version": ingestion._BEHAVIOR_VERSION,
                        "input_fingerprint": inventory.source_fingerprint,
                        "config_fingerprint": inventory.config_fingerprint,
                        "dependency_versions": dict(
                            inventory.dependency_versions
                        ),
                        "content_sha256": metadata["content_sha256"],
                        "source_schema_sha256": metadata[
                            "source_schema_sha256"
                        ],
                        "index_bindings": metadata["index_bindings"],
                        "outputs": outputs,
                        "array_binding": metadata["array_binding"],
                        "reopened_and_validated": True,
                        "disk_admission": {
                            "coarse": coarse_admission,
                            "exact": exact_admission,
                        },
                        "prepared_relation_stage_bytes": 0,
                    }
                    completion_path = (
                        temporary_root / "relations.complete.json"
                    )
                    for _ in range(5):
                        ingestion._atomic_json(completion_path, completion)
                        prepared_bytes = _tree_bytes(temporary_root)
                        if (
                            completion["prepared_relation_stage_bytes"]
                            == prepared_bytes
                        ):
                            break
                        completion["prepared_relation_stage_bytes"] = (
                            prepared_bytes
                        )
                    else:
                        raise ingestion.ArtifactValidationError(
                            "DISK-EVIDENCE-001: prepared relation byte evidence did not stabilize"
                        )
                    if _tree_bytes(temporary_root) > int(
                        exact_requirements["estimated_relation_bytes"]
                    ):
                        raise ingestion.DiskAdmissionError(
                            "DISK-ESTIMATE-001: prepared relation stage exceeded its preallocation estimate"
                        )
                    self._fsync_prepared_tree(temporary_root)
                if spill_created:
                    shutil.rmtree(spill_root)
                    spill_created = False
                self._publish(temporary_root, stage_root)
                result = self._open_validated(
                    resumed=False,
                    pa=pa,
                    pq=pq,
                    duckdb=duckdb,
                )
                self._finalize_top_completion(result)
                return result
            finally:
                if temporary_root.exists():
                    shutil.rmtree(temporary_root)
                if spill_created and spill_root.exists():
                    shutil.rmtree(spill_root)

    def _validate_task3_arrays(self) -> None:
        ingestion = _ingestion_module()
        expected = self.array_build
        if (
            expected.inventory.source_fingerprint
            != self.index_build.inventory.source_fingerprint
            or expected.stage_root != self.index_build.stage_root
            or expected.artifact_root != self.index_build.stage_root / "arrays"
        ):
            raise ingestion.ArtifactValidationError(
                "ARRAY-BINDING-001: Task 3 build does not belong to these exact indexes"
            )
        from topobench.data.stores.typed_graph_arrays import (
            TypedGraphArrayWriter,
        )

        current = TypedGraphArrayWriter(
            self.ingestor,
            self.index_build,
        )._open_validated(resumed=True)
        if (
            current.content_sha256 != expected.content_sha256
            or current.active_tag != expected.active_tag
            or current.record_batch_rows != expected.record_batch_rows
        ):
            raise ingestion.ArtifactValidationError(
                "ARRAY-BINDING-001: Task 3 arrays changed between validation and relation lock"
            )
        self.array_build = current

    def _array_binding(self) -> dict[str, Any]:
        ingestion = _ingestion_module()
        root = self.array_build.artifact_root
        metadata = ingestion._read_json(root / "arrays.json")
        completion_path = root / "arrays.complete.json"
        completion = ingestion._read_json(completion_path)
        semantic_source_schema = {
            "nodes": _without_physical_array_evidence(metadata["nodes"]),
            "supervision": _without_physical_array_evidence(
                metadata["supervision"]
            ),
            "splits": _without_physical_array_evidence(metadata["splits"]),
        }
        index_bindings = self._index_bindings()
        return {
            "array_behavior_version": metadata["array_behavior_version"],
            "content_sha256": metadata["content_sha256"],
            "content_identity": metadata["content_identity"],
            "completion_sha256": ingestion._sha256_file(completion_path),
            "metadata_sha256": ingestion._sha256_file(root / "arrays.json"),
            "input_fingerprint": metadata["input_fingerprint"],
            "config_fingerprint": metadata["config_fingerprint"],
            "dependency_versions": metadata["dependency_versions"],
            "active_split_tag": metadata["active_split_tag"],
            "source_schema_sha256": ingestion._sha256_json(
                semantic_source_schema
            ),
            "index_bindings_sha256": ingestion._sha256_json(index_bindings),
            "completion_content_sha256": completion["content_sha256"],
        }

    def _relation_disk_requirements(self) -> dict[str, int | bool]:
        inventory = self.index_build.inventory
        inventory_by_path = {
            entry.relative_path: entry for entry in inventory.files
        }
        node_counts = dict(inventory.node_rows)
        relation_rows = 0
        relation_uncompressed = 0
        payload_bytes = 0
        array_count = 0
        for relation in self.ingestor.source.spec.relations:
            edge_count = sum(
                inventory_by_path[relative].row_count
                for relative in relation.paths
            )
            relation_rows += edge_count
            relation_bytes = sum(
                inventory_by_path[relative].uncompressed_bytes
                for relative in relation.paths
            )
            relation_uncompressed += relation_bytes
            destination_count = node_counts[relation.relation[2]]
            payload_bytes += (destination_count + 1) * 8 + edge_count * 8
            array_count += 2
            if relation.edge_id_column is not None:
                payload_bytes += max(edge_count * 8, relation_bytes)
                array_count += 1
            if relation.edge_fields:
                payload_bytes += max(
                    edge_count * 8 * len(relation.edge_fields),
                    relation_bytes,
                )
                array_count += len(relation.edge_fields)
        evidence_reserve = 1024**2 + 8192 * (
            array_count
            + len(self.ingestor.source.spec.relations)
            + len(inventory.files)
        )
        canonical_scratch_reserve = (
            max(payload_bytes, relation_uncompressed)
            + relation_rows * 32
            + 2 * 1024**2
        )
        estimated_relation = (
            payload_bytes
            + relation_uncompressed * 2
            + 512 * array_count
            + evidence_reserve
            + canonical_scratch_reserve
        )
        snapshot_bytes = inventory.snapshot_bytes
        spill_bytes = max(
            1,
            inventory.estimated_temporary_bytes - snapshot_bytes,
            relation_uncompressed * 3 + relation_rows * 40,
        )
        array_completion = _ingestion_module()._read_json(
            self.array_build.artifact_root / "arrays.complete.json"
        )
        task3_requirements = array_completion["disk_admission"]["requirements"]
        task3_final_bytes = int(task3_requirements["final_peak_bytes"])
        final_peak = task3_final_bytes + estimated_relation

        temporary_peak = snapshot_bytes + spill_bytes
        same_filesystem = inventory.final_device == inventory.temporary_device
        return {
            "same_filesystem": same_filesystem,
            "task2_final_bytes": inventory.estimated_final_bytes,
            "task2_required_peak_bytes": inventory.required_peak_bytes,
            "task3_final_bytes": task3_final_bytes,
            "task3_array_bytes": int(
                task3_requirements["estimated_array_bytes"]
            ),
            "estimated_relation_bytes": estimated_relation,
            "relation_payload_bytes": payload_bytes,
            "relation_evidence_reserve_bytes": evidence_reserve,
            "relation_rows": relation_rows,
            "snapshot_bytes": snapshot_bytes,
            "spill_bytes": spill_bytes,
            "final_peak_bytes": final_peak,
            "temporary_peak_bytes": temporary_peak,
            "shared_peak_bytes": final_peak + temporary_peak,
            "final_additional_bytes": estimated_relation,
            "temporary_additional_bytes": temporary_peak,
            "shared_additional_bytes": estimated_relation + temporary_peak,
        }

    def _exact_relation_disk_requirements(
        self,
        prepared: Mapping[str, tuple[Any, Mapping[str, Any]]],
        *,
        coarse_requirements: Mapping[str, int | bool],
    ) -> dict[str, Any]:
        arrays: dict[str, dict[str, Any]] = {}
        exact_array_bytes = 0
        exact_payload_bytes = 0
        max_canonical_permutation_bytes = 0
        max_canonical_scratch_file_bytes = 0
        for internal_key in sorted(prepared):
            _, context = prepared[internal_key]
            colptr = _npy_storage_sizing(
                np.dtype("<i8"),
                (context["destination_count"] + 1,),
            )
            row = _npy_storage_sizing(
                np.dtype("<i8"),
                (context["edge_count"],),
            )
            edge_id = None
            if context["edge_descriptor"] is not None:
                edge_id = _npy_storage_sizing(
                    np.dtype(context["edge_descriptor"]["storage_dtype"]),
                    (context["edge_count"],),
                )
            fields = {
                name: _npy_storage_sizing(
                    np.dtype(
                        context["field_descriptors"][name]["storage_dtype"]
                    ),
                    (
                        context["edge_count"],
                        *context["field_descriptors"][name]["value_shape"],
                    ),
                )
                for name in sorted(context["field_descriptors"])
            }
            relation_arrays = {
                "colptr": colptr,
                "row": row,
                "edge_id": edge_id,
                "fields": fields,
            }
            arrays[internal_key] = relation_arrays
            records = [colptr, row]
            if edge_id is not None:
                records.append(edge_id)
            records.extend(fields[name] for name in sorted(fields))
            exact_array_bytes += sum(
                int(record["file_bytes"]) for record in records
            )
            exact_payload_bytes += sum(
                int(record["payload_bytes"]) for record in records
            )
            permutation_scratch = _npy_storage_sizing(
                np.dtype("<i8"),
                (context["edge_count"],),
            )
            key_row_bytes = 3 * np.dtype("<i8").itemsize
            if context["edge_descriptor"] is not None:
                key_row_bytes += np.dtype(
                    context["edge_descriptor"]["storage_dtype"]
                ).itemsize
            key_row_groups = (
                context["edge_count"]
                + self.ingestor.source.spec.ingestion.record_batch_rows
                - 1
            ) // self.ingestor.source.spec.ingestion.record_batch_rows
            key_file_upper_bound = (
                context["edge_count"] * key_row_bytes
                + key_row_groups * 64 * 1024
                + 1024**2
            )
            reorder_records = [row]
            if edge_id is not None:
                reorder_records.append(edge_id)
            reorder_records.extend(fields[name] for name in sorted(fields))
            max_reorder_file_bytes = max(
                (int(record["file_bytes"]) for record in reorder_records),
                default=0,
            )
            max_canonical_scratch_file_bytes = max(
                max_canonical_scratch_file_bytes,
                int(permutation_scratch["file_bytes"])
                + max(key_file_upper_bound, max_reorder_file_bytes),
            )
            max_canonical_permutation_bytes = max(
                max_canonical_permutation_bytes,
                int(context["edge_count"]) * np.dtype(np.intp).itemsize,
            )
        evidence_reserve = int(
            coarse_requirements["relation_evidence_reserve_bytes"]
        )
        estimated_relation_bytes = exact_array_bytes + evidence_reserve
        task3_final_bytes = int(coarse_requirements["task3_final_bytes"])
        snapshot_bytes = int(coarse_requirements["snapshot_bytes"])
        spill_bytes = int(coarse_requirements["spill_bytes"])
        temporary_peak_bytes = snapshot_bytes + spill_bytes
        final_build_bytes = (
            estimated_relation_bytes + max_canonical_scratch_file_bytes
        )
        final_peak_bytes = task3_final_bytes + final_build_bytes
        same_filesystem = bool(coarse_requirements["same_filesystem"])
        return {
            "same_filesystem": same_filesystem,
            "task2_final_bytes": int(coarse_requirements["task2_final_bytes"]),
            "task2_required_peak_bytes": int(
                coarse_requirements["task2_required_peak_bytes"]
            ),
            "task3_final_bytes": task3_final_bytes,
            "task3_array_bytes": int(coarse_requirements["task3_array_bytes"]),
            "estimated_relation_bytes": estimated_relation_bytes,
            "exact_array_file_bytes": exact_array_bytes,
            "exact_array_payload_bytes": exact_payload_bytes,
            "relation_evidence_reserve_bytes": evidence_reserve,
            "relation_rows": int(coarse_requirements["relation_rows"]),
            "snapshot_bytes": snapshot_bytes,
            "spill_bytes": spill_bytes,
            "max_canonical_permutation_bytes": max_canonical_permutation_bytes,
            "max_canonical_scratch_file_bytes": max_canonical_scratch_file_bytes,
            "final_peak_bytes": final_peak_bytes,
            "temporary_peak_bytes": temporary_peak_bytes,
            "resume_temporary_additional_bytes": temporary_peak_bytes,
            "shared_peak_bytes": final_peak_bytes + temporary_peak_bytes,
            "final_additional_bytes": final_build_bytes,
            "temporary_additional_bytes": temporary_peak_bytes,
            "shared_additional_bytes": final_build_bytes
            + temporary_peak_bytes,
            "arrays": arrays,
        }

    def _admit_relation_disk(
        self,
        requirements: Mapping[str, Any],
        *,
        exact: bool,
    ) -> dict[str, Any]:
        ingestion = _ingestion_module()
        inventory = self.index_build.inventory
        final_device, final_available, final_probe = (
            ingestion._filesystem_capacity(inventory.final_filesystem_path)
        )
        temporary_device, temporary_available, temporary_probe = (
            ingestion._filesystem_capacity(inventory.temporary_filesystem_path)
        )
        if (
            final_device != inventory.final_device
            or temporary_device != inventory.temporary_device
        ):
            raise ingestion.DiskAdmissionError(
                "DISK-EVIDENCE-001: final or temporary filesystem changed before Task 4 allocation"
            )
        code = "DISK-PREFLIGHT-EXACT-001" if exact else "DISK-PREFLIGHT-001"
        limit = self.ingestor.disk_limit_bytes
        if requirements["same_filesystem"]:
            available = min(final_available, temporary_available)
            required = (
                int(requirements["final_additional_bytes"])
                + int(requirements["spill_bytes"])
                if exact
                else int(requirements["shared_additional_bytes"])
            )
            if required > available:
                raise ingestion.DiskAdmissionError(
                    f"{code}: shared final+temporary filesystem requires "
                    f"{required} additional bytes, only {available} available"
                )
            if (
                limit is not None
                and int(requirements["shared_peak_bytes"]) > limit
            ):
                raise ingestion.DiskAdmissionError(
                    f"{code}: shared final+temporary filesystem peak "
                    f"requires {requirements['shared_peak_bytes']} bytes, limit is "
                    f"{limit}"
                )
        else:
            final_required = int(requirements["final_additional_bytes"])
            temporary_required = (
                int(requirements["spill_bytes"])
                if exact
                else int(requirements["temporary_additional_bytes"])
            )
            if final_required > final_available:
                raise ingestion.DiskAdmissionError(
                    f"{code}: final filesystem requires "
                    f"{final_required} additional bytes, only {final_available} available"
                )
            if temporary_required > temporary_available:
                raise ingestion.DiskAdmissionError(
                    f"{code}: temporary filesystem requires "
                    f"{temporary_required} bytes, only {temporary_available} available"
                )
            if limit is not None and (
                int(requirements["final_peak_bytes"]) > limit
                or int(requirements["temporary_peak_bytes"]) > limit
            ):
                raise ingestion.DiskAdmissionError(
                    f"{code}: independent filesystem peak exceeds "
                    f"disk limit {limit}"
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
        requirements: Mapping[str, Any],
        artifact_root: Path,
    ) -> None:
        ingestion = _ingestion_module()
        inventory = self.index_build.inventory
        final_device, final_available, _ = ingestion._filesystem_capacity(
            inventory.final_filesystem_path
        )
        temporary_device, temporary_available, _ = (
            ingestion._filesystem_capacity(inventory.temporary_filesystem_path)
        )
        if (
            self.ingestor.store_root != inventory.final_filesystem_path
            or final_device != inventory.final_device
            or temporary_device != inventory.temporary_device
            or artifact_root.is_symlink()
            or not artifact_root.resolve(strict=True).is_relative_to(
                inventory.final_filesystem_path.resolve(strict=True)
            )
            or not artifact_root.is_dir()
        ):
            raise ingestion.DiskAdmissionError(
                "DISK-EVIDENCE-001: current relation filesystem differs from admission"
            )
        temporary_required = int(
            requirements["resume_temporary_additional_bytes"]
        )
        available = (
            min(final_available, temporary_available)
            if requirements["same_filesystem"]
            else temporary_available
        )
        if temporary_required > available:
            raise ingestion.DiskAdmissionError(
                "RESUME-DISK-PREFLIGHT-001: semantic audit temporary "
                f"storage requires {temporary_required} bytes, only "
                f"{available} available"
            )
        limit = self.ingestor.disk_limit_bytes
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
                "RESUME-DISK-PREFLIGHT-001: current relation disk limit is below "
                f"recorded required peak {required_peak}"
            )

    def _configure_connection(self, connection: Any, spill_root: Path) -> None:
        memory_bytes = self.ingestor.source.spec.ingestion.memory_limit_bytes
        escaped_spill = str(spill_root).replace("'", "''")
        connection.execute("SET preserve_insertion_order = false")
        connection.execute(f"SET memory_limit = '{memory_bytes}B'")
        connection.execute(f"SET temp_directory = '{escaped_spill}'")
        connection.execute(f"SET threads = {self.ingestor.threads}")

    def _write_all(
        self,
        root: Path,
        *,
        snapshots: Mapping[str, Path],
        connection: Any,
        pa: Any,
        pq: Any,
        coarse_requirements: Mapping[str, int | bool],
    ) -> tuple[dict[str, Any], dict[str, Any], dict[str, Any]]:
        prepared: dict[str, tuple[Any, dict[str, Any]]] = {}
        schema_evidence: dict[str, dict[str, Any]] = {}
        for ordinal, relation in enumerate(
            self.ingestor.source.spec.relations
        ):
            internal_key = f"r{ordinal:04d}"
            context = self._prepare_relation(
                relation=relation,
                internal_key=internal_key,
                snapshots=snapshots,
                connection=connection,
                pa=pa,
                pq=pq,
            )
            prepared[internal_key] = (relation, context)
            schema_evidence[internal_key] = context["schema_record"]
        exact_requirements = self._exact_relation_disk_requirements(
            prepared,
            coarse_requirements=coarse_requirements,
        )
        exact_admission = self._admit_relation_disk(
            exact_requirements,
            exact=True,
        )
        relations: dict[str, dict[str, Any]] = {}
        for internal_key, (relation, context) in prepared.items():
            record, _ = self._write_one_relation(
                root,
                relation=relation,
                internal_key=internal_key,
                context=context,
                connection=connection,
            )
            relations[internal_key] = record
        index_bindings = self._index_bindings()
        ingestion = _ingestion_module()
        metadata = {
            "relation_behavior_version": _RELATION_BEHAVIOR_VERSION,
            "input_fingerprint": self.index_build.inventory.source_fingerprint,
            "config_fingerprint": self.index_build.inventory.config_fingerprint,
            "dependency_versions": dict(
                self.index_build.inventory.dependency_versions
            ),
            "record_batch_rows": self.ingestor.source.spec.ingestion.record_batch_rows,
            "max_record_batch_rows": self._max_batch_rows,
            "index_bindings": index_bindings,
            "index_bindings_sha256": ingestion._sha256_json(index_bindings),
            "array_binding": self._array_binding(),
            "source_schemas": schema_evidence,
            "source_schema_sha256": ingestion._sha256_json(schema_evidence),
            "relations": relations,
        }
        return metadata, exact_requirements, exact_admission

    def _write_one_relation(
        self,
        root: Path,
        *,
        relation: Any,
        internal_key: str,
        context: Mapping[str, Any],
        connection: Any,
    ) -> tuple[dict[str, Any], dict[str, Any]]:
        edge_count = context["edge_count"]
        destination_count = context["destination_count"]
        relation_root = root / internal_key
        relation_root.mkdir(parents=True, exist_ok=False)
        colptr_path = relation_root / "colptr.npy"
        row_path = relation_root / "row.npy"
        int64_dtype = np.dtype("int64").newbyteorder("<")
        colptr = np.lib.format.open_memmap(
            colptr_path,
            mode="w+",
            dtype=int64_dtype,
            shape=(destination_count + 1,),
        )
        colptr[:] = 0
        row_output = _open_npy_output(
            row_path,
            dtype=int64_dtype,
            shape=(edge_count,),
        )
        edge_path: Path | None = None
        edge_output: Any = None
        if context["edge_descriptor"] is not None:
            edge_path = relation_root / "edge_id.npy"
            edge_output = _open_npy_output(
                edge_path,
                dtype=np.dtype(context["edge_descriptor"]["storage_dtype"]),
                shape=(edge_count,),
            )
        field_paths: dict[str, Path] = {}
        field_outputs: dict[str, Any] = {}
        for ordinal, (field_name, descriptor) in enumerate(
            context["field_descriptors"].items()
        ):
            path = relation_root / "fields" / f"f{ordinal:04d}.npy"
            path.parent.mkdir(parents=True, exist_ok=True)
            field_paths[field_name] = path
            field_outputs[field_name] = _open_npy_output(
                path,
                dtype=np.dtype(descriptor["storage_dtype"]),
                shape=(edge_count, *descriptor["value_shape"]),
            )
        key_path = relation_root / ".canonical_keys.parquet"
        pa, pq, _ = _ingestion_module()._parquet_dependencies()
        key_writer: Any = None
        outputs = [
            row_output,
            *([edge_output] if edge_output is not None else []),
            *field_outputs.values(),
        ]
        offset = 0
        try:
            reader = connection.execute(
                context["stream_query"]
            ).to_arrow_reader(
                self.ingestor.source.spec.ingestion.record_batch_rows
            )
            for batch in reader:
                rows = batch.num_rows
                self._observe_batch(rows)
                destinations = self._primitive_block(
                    batch.column(0),
                    context="destination ordinals",
                )
                sources = self._primitive_block(
                    batch.column(1),
                    context="source ordinals",
                )
                if (
                    np.any(destinations < 0)
                    or np.any(destinations >= destination_count)
                    or np.any(sources < 0)
                    or np.any(sources >= context["source_count"])
                ):
                    raise _artifact_error(
                        "CSC-BOUNDS-001: mapped relation endpoint is outside its exact type-local domain"
                    )
                key_arrays = [batch.column(0), batch.column(1)]
                key_names = ["destination_local", "source_local"]
                if context["edge_descriptor"] is not None:
                    key_arrays.append(batch.column(2))
                    key_names.append("stable_edge_id")
                key_arrays.append(
                    pa.array(
                        np.arange(offset, offset + rows, dtype=np.int64),
                        type=pa.int64(),
                    )
                )
                key_names.append("source_position")
                key_batch = pa.record_batch(key_arrays, names=key_names)
                if key_writer is None:
                    key_writer = pq.ParquetWriter(
                        key_path,
                        key_batch.schema,
                        compression="zstd",
                    )
                key_writer.write_batch(
                    key_batch,
                    row_group_size=self.ingestor.source.spec.ingestion.record_batch_rows,
                )
                _write_npy_block(row_output, sources, dtype=int64_dtype)
                np.add.at(colptr, destinations + 1, 1)
                column = 2
                if edge_output is not None:
                    descriptor = context["edge_descriptor"]
                    _write_npy_block(
                        edge_output,
                        self._value_block(
                            batch.column(column),
                            descriptor,
                            context="stable edge IDs",
                        ),
                        dtype=np.dtype(descriptor["storage_dtype"]),
                    )
                    column += 1
                for field_name, descriptor in context[
                    "field_descriptors"
                ].items():
                    _write_npy_block(
                        field_outputs[field_name],
                        self._value_block(
                            batch.column(column),
                            descriptor,
                            context=f"edge field {field_name!r}",
                        ),
                        dtype=np.dtype(descriptor["storage_dtype"]),
                    )
                    column += 1
                offset += rows
        finally:
            for output in outputs:
                output.close()
            if key_writer is not None:
                key_writer.close()
        permutation_path = relation_root / ".canonical_permutation.npy"
        try:
            if offset != edge_count:
                raise _artifact_error(
                    f"EDGE-CARDINALITY-001: wrote {offset} edges, expected {edge_count}"
                )
            np.cumsum(colptr, out=colptr)
            colptr.flush()
            _close_memmap(colptr)
            permutation_output = _open_npy_output(
                permutation_path,
                dtype=int64_dtype,
                shape=(edge_count,),
            )
            sorted_count = 0
            try:
                if edge_count:
                    key_view = f"canonical_keys_{uuid.uuid4().hex}"
                    connection.read_parquet(str(key_path)).create_view(
                        key_view
                    )
                    order = "destination_local, source_local"
                    if context["edge_descriptor"] is not None:
                        edge_order = (
                            "encode(stable_edge_id)"
                            if context["edge_descriptor"]["dtype"] == "string"
                            else "stable_edge_id"
                        )
                        order = f"{order}, {edge_order}"
                    reader = connection.execute(
                        f"SELECT source_position FROM {_quote(key_view)} ORDER BY {order}"
                    ).to_arrow_reader(
                        self.ingestor.source.spec.ingestion.record_batch_rows
                    )
                    for batch in reader:
                        rows = batch.num_rows
                        self._observe_batch(rows)
                        positions = self._primitive_block(
                            batch.column(0),
                            context="canonical source positions",
                        )
                        if np.any(positions < 0) or np.any(
                            positions >= edge_count
                        ):
                            raise _artifact_error(
                                "CSC-ORDER-001: canonical source position is outside the relation"
                            )
                        _write_npy_block(
                            permutation_output,
                            positions,
                            dtype=int64_dtype,
                        )
                        sorted_count += rows
                    del reader
                    connection.execute(f"DROP VIEW {_quote(key_view)}")
            finally:
                permutation_output.close()
            if sorted_count != edge_count:
                raise _artifact_error(
                    f"CSC-ORDER-001: canonicalized {sorted_count} edges, expected {edge_count}"
                )
            key_path.unlink(missing_ok=True)
            permutation = np.load(
                permutation_path,
                mmap_mode="r",
                allow_pickle=False,
            )
            self._reorder_array(row_path, permutation)
            if edge_path is not None:
                self._reorder_array(edge_path, permutation)
            for path in field_paths.values():
                self._reorder_array(path, permutation)
            _close_memmap(permutation)
        finally:
            key_path.unlink(missing_ok=True)
            permutation_path.unlink(missing_ok=True)
        for path in (colptr_path, row_path, edge_path, *field_paths.values()):
            if path is not None:
                _fsync_file(path)

        record: dict[str, Any] = {
            "internal_key": internal_key,
            "relation": list(relation.relation),
            "source_node_type": relation.relation[0],
            "destination_node_type": relation.relation[2],
            "source_internal_key": context["source_internal_key"],
            "destination_internal_key": context["destination_internal_key"],
            "source_column": relation.source_column,
            "destination_column": relation.destination_column,
            "source_id_dtype": context["source_id_dtype"],
            "destination_id_dtype": context["destination_id_dtype"],
            "source_count": context["source_count"],
            "destination_count": destination_count,
            "edge_count": edge_count,
            "canonical_order": [
                "destination_local",
                "source_local",
                *(
                    [relation.edge_id_column]
                    if relation.edge_id_column
                    else []
                ),
            ],
            "source_files": context["source_files"],
            "source_sha256": context["source_sha256"],
            "schema_fingerprint": context["schema_record"][
                "schema_fingerprint"
            ],
            "colptr": self._array_record(
                root,
                colptr_path,
                dtype="int64",
                shape=(destination_count + 1,),
                count=destination_count + 1,
                extra={},
            ),
            "row": self._array_record(
                root,
                row_path,
                dtype="int64",
                shape=(edge_count,),
                count=edge_count,
                extra={},
            ),
            "edge_id_column": relation.edge_id_column,
            "edge_id": None,
            "fields": {},
        }
        if context["edge_descriptor"] is not None and edge_path is not None:
            descriptor = context["edge_descriptor"]
            record["edge_id"] = self._array_record(
                root,
                edge_path,
                dtype=descriptor["dtype"],
                shape=(edge_count,),
                count=edge_count,
                extra={
                    "column": relation.edge_id_column,
                    "arrow_type": descriptor["arrow_type"],
                    "representation": "scalar",
                },
            )
        for ordinal, (field_name, descriptor) in enumerate(
            context["field_descriptors"].items()
        ):
            record["fields"][field_name] = self._array_record(
                root,
                field_paths[field_name],
                dtype=descriptor["dtype"],
                shape=(edge_count, *descriptor["value_shape"]),
                count=edge_count,
                extra={
                    "column": field_name,
                    "internal_key": f"f{ordinal:04d}",
                    "arrow_type": descriptor["arrow_type"],
                    "representation": descriptor["representation"],
                    "value_shape": list(descriptor["value_shape"]),
                },
            )
        record["semantic_sha256"] = _relation_semantic_sha256(
            root,
            record,
            self.ingestor.source.spec.ingestion.record_batch_rows,
        )
        return record, context["schema_record"]

    def _prepare_relation(
        self,
        *,
        relation: Any,
        internal_key: str,
        snapshots: Mapping[str, Path],
        connection: Any,
        pa: Any,
        pq: Any,
    ) -> dict[str, Any]:
        paths = tuple(snapshots[relative] for relative in relation.paths)
        schemas = tuple(pq.ParquetFile(path).schema_arrow for path in paths)
        schema_fingerprints = {
            hashlib.sha256(schema.serialize().to_pybytes()).hexdigest()
            for schema in schemas
        }
        if len(schema_fingerprints) != 1:
            raise _artifact_error(
                f"SCHEMA-DRIFT-001: relation {relation.relation!r} fragments have different exact Arrow schemas"
            )
        schema = schemas[0]
        source_node = self._node(relation.relation[0])
        destination_node = self._node(relation.relation[2])
        source_field = self._field(
            schema,
            relation.source_column,
            code="EDGE-SCHEMA-001",
            context=relation.relation,
        )
        destination_field = self._field(
            schema,
            relation.destination_column,
            code="EDGE-SCHEMA-001",
            context=relation.relation,
        )
        source_view = f"relation_source_{internal_key}"
        source_map_view = f"relation_source_map_{internal_key}"
        destination_map_view = f"relation_destination_map_{internal_key}"
        mapped_view = f"relation_mapped_{internal_key}"
        connection.read_parquet([str(path) for path in paths]).create_view(
            source_view
        )
        self._require_endpoint_type(
            connection,
            source_view=source_view,
            field=source_field,
            column=relation.source_column,
            expected_dtype=source_node.id_dtype,
            pa=pa,
        )
        self._require_endpoint_type(
            connection,
            source_view=source_view,
            field=destination_field,
            column=relation.destination_column,
            expected_dtype=destination_node.id_dtype,
            pa=pa,
        )
        connection.read_parquet(
            str(self.index_build.indexes[source_node.name].node_ids_path)
        ).create_view(source_map_view)
        connection.read_parquet(
            str(self.index_build.indexes[destination_node.name].node_ids_path)
        ).create_view(destination_map_view)

        edge_descriptor: dict[str, Any] | None = None
        if relation.edge_id_column is not None:
            edge_field = self._field(
                schema,
                relation.edge_id_column,
                code="EDGE-ID-SCHEMA-001",
                context=relation.relation,
            )
            edge_descriptor = self._edge_id_descriptor(edge_field.type, pa=pa)
        field_descriptors: dict[str, dict[str, Any]] = {}
        for field_name in relation.edge_fields:
            field = self._field(
                schema,
                field_name,
                code="EDGE-FIELD-SCHEMA-001",
                context=relation.relation,
            )
            field_descriptors[field_name] = self._field_descriptor(
                field.type, pa=pa
            )

        selections = [
            f"s.{_quote(relation.source_column)} AS source_external",
            f"s.{_quote(relation.destination_column)} AS destination_external",
            "source_map.local_ordinal AS source_local",
            "destination_map.local_ordinal AS destination_local",
        ]
        if relation.edge_id_column is not None:
            selections.append(
                f"s.{_quote(relation.edge_id_column)} AS stable_edge_id"
            )
        selections.extend(
            f"s.{_quote(field_name)} AS {_quote(f'field_{ordinal:04d}')}"
            for ordinal, field_name in enumerate(relation.edge_fields)
        )
        join_query = (
            f"SELECT {', '.join(selections)} FROM {_quote(source_view)} s "
            f"LEFT JOIN {_quote(source_map_view)} source_map ON "
            f"s.{_quote(relation.source_column)} IS NOT DISTINCT FROM source_map.external_id "
            f"LEFT JOIN {_quote(destination_map_view)} destination_map ON "
            f"s.{_quote(relation.destination_column)} IS NOT DISTINCT FROM destination_map.external_id"
        )
        connection.execute(
            f"CREATE TEMP VIEW {_quote(mapped_view)} AS {join_query}"
        )
        counts = connection.execute(
            f"SELECT COUNT(*), "
            "COUNT(*) FILTER (WHERE source_external IS NULL), "
            "COUNT(*) FILTER (WHERE destination_external IS NULL), "
            "COUNT(*) FILTER (WHERE source_external IS NOT NULL AND source_local IS NULL), "
            "COUNT(*) FILTER (WHERE destination_external IS NOT NULL AND destination_local IS NULL), "
            "COUNT(*) FILTER (WHERE source_local < 0 OR source_local >= ?), "
            "COUNT(*) FILTER (WHERE destination_local < 0 OR destination_local >= ?) "
            f"FROM {_quote(mapped_view)}",
            [
                self.index_build.indexes[source_node.name].row_count,
                self.index_build.indexes[destination_node.name].row_count,
            ],
        ).fetchone()
        edge_count = int(counts[0])
        if counts[1] or counts[2]:
            raise _artifact_error(
                f"EDGE-ENDPOINT-NULL-001: relation {relation.relation!r} contains null endpoints"
            )
        if counts[3]:
            raise _artifact_error(
                f"EDGE-SOURCE-UNRESOLVED-001: relation {relation.relation!r} has endpoints absent from source type {source_node.name!r}"
            )
        if counts[4]:
            raise _artifact_error(
                f"EDGE-DESTINATION-UNRESOLVED-001: relation {relation.relation!r} has endpoints absent from destination type {destination_node.name!r}"
            )
        if counts[5] or counts[6]:
            raise _artifact_error(
                "CSC-BOUNDS-001: joined endpoint ordinal exceeds its exact type-local domain"
            )
        if relation.edge_id_column is None:
            duplicate = connection.execute(
                f"SELECT 1 FROM {_quote(mapped_view)} GROUP BY destination_local, source_local HAVING COUNT(*) > 1 LIMIT 1"
            ).fetchone()
            if duplicate is not None:
                raise _artifact_error(
                    f"EDGE-DUPLICATE-ENDPOINT-001: relation {relation.relation!r} has ambiguous duplicate endpoints without a stable edge ID"
                )
        else:
            edge_column = _quote(relation.edge_id_column)
            edge_counts = connection.execute(
                f"SELECT COUNT(*) FILTER (WHERE {edge_column} IS NULL), "
                f"COUNT(DISTINCT {edge_column}) FROM {_quote(source_view)}"
            ).fetchone()
            if edge_counts[0]:
                raise _artifact_error(
                    f"EDGE-ID-NULL-001: relation {relation.relation!r} contains null stable edge IDs"
                )
            if edge_counts[1] != edge_count:
                raise _artifact_error(
                    f"EDGE-ID-DUPLICATE-001: relation {relation.relation!r} contains duplicate stable edge IDs"
                )
            if edge_descriptor["dtype"] == "string":
                string_stats = connection.execute(
                    "SELECT COUNT(*) FILTER "
                    f"(WHERE ends_with({edge_column}, chr(0))), "
                    f"COALESCE(MAX(length({edge_column})), 0) "
                    f"FROM {_quote(source_view)}"
                ).fetchone()
                if string_stats[0]:
                    raise _artifact_error(
                        "EDGE-STRING-TRAILING-NUL-001: stable edge IDs "
                        f"for relation {relation.relation!r} end in U+0000"
                    )
                edge_descriptor["storage_dtype"] = np.dtype(
                    f"<U{max(1, int(string_stats[1]))}"
                ).str
        for field_name, descriptor in field_descriptors.items():
            source_field = _quote(field_name)
            null_count = connection.execute(
                f"SELECT COUNT(*) FILTER (WHERE {source_field} IS NULL) "
                f"FROM {_quote(source_view)}"
            ).fetchone()[0]
            if null_count:
                raise _artifact_error(
                    f"EDGE-FIELD-NULL-001: relation {relation.relation!r} field {field_name!r} contains nulls"
                )
            if (
                descriptor["dtype"] in _FLOAT_DTYPES
                and descriptor["representation"] == "scalar"
            ):
                invalid = connection.execute(
                    f"SELECT COUNT(*) FILTER (WHERE NOT isfinite({source_field})) "
                    f"FROM {_quote(source_view)}"
                ).fetchone()[0]
                if invalid:
                    raise _artifact_error(
                        f"EDGE-FIELD-FINITE-001: relation {relation.relation!r} field {field_name!r} contains NaN or infinity"
                    )
            if descriptor["dtype"] == "string":
                if descriptor["representation"] == "scalar":
                    string_stats = connection.execute(
                        "SELECT 0, COUNT(*) FILTER "
                        f"(WHERE ends_with({source_field}, chr(0))), "
                        f"COALESCE(MAX(length({source_field})), 0) "
                        f"FROM {_quote(source_view)}"
                    ).fetchone()
                else:
                    string_stats = connection.execute(
                        "SELECT COUNT(*) FILTER (WHERE value IS NULL), "
                        "COUNT(*) FILTER "
                        "(WHERE ends_with(value, chr(0))), "
                        "COALESCE(MAX(length(value)), 0) "
                        f"FROM {_quote(source_view)}, "
                        f"UNNEST({source_field}) AS list_value(value)"
                    ).fetchone()
                if string_stats[0]:
                    raise _artifact_error(
                        "EDGE-FIELD-NULL-001: relation "
                        f"{relation.relation!r} field {field_name!r} "
                        "contains null list elements"
                    )
                if string_stats[1]:
                    raise _artifact_error(
                        "EDGE-STRING-TRAILING-NUL-001: relation "
                        f"{relation.relation!r} field {field_name!r} "
                        "contains UTF-8 values ending in U+0000"
                    )
                descriptor["storage_dtype"] = np.dtype(
                    f"<U{max(1, int(string_stats[2]))}"
                ).str

        projected = ["destination_local", "source_local"]
        if relation.edge_id_column is not None:
            projected.append("stable_edge_id")
        projected.extend(
            _quote(f"field_{ordinal:04d}")
            for ordinal in range(len(relation.edge_fields))
        )
        stream_query = (
            f"SELECT {', '.join(projected)} FROM {_quote(mapped_view)}"
        )
        source_files = self._source_records(relation.paths)
        schema_record = {
            "relation": list(relation.relation),
            "source_files": source_files,
            "schema_fingerprint": next(iter(schema_fingerprints)),
            "schema_serialized_hex": schema.serialize().to_pybytes().hex(),
        }
        return {
            "stream_query": stream_query,
            "edge_count": edge_count,
            "source_count": self.index_build.indexes[
                source_node.name
            ].row_count,
            "destination_count": self.index_build.indexes[
                destination_node.name
            ].row_count,
            "source_internal_key": self._node_key(source_node.name),
            "destination_internal_key": self._node_key(destination_node.name),
            "source_id_dtype": source_node.id_dtype,
            "destination_id_dtype": destination_node.id_dtype,
            "edge_descriptor": edge_descriptor,
            "field_descriptors": field_descriptors,
            "source_files": source_files,
            "source_sha256": _combined_source_sha(source_files),
            "schema_record": schema_record,
        }

    def _require_endpoint_type(
        self,
        connection: Any,
        *,
        source_view: str,
        field: Any,
        column: str,
        expected_dtype: str,
        pa: Any,
    ) -> None:
        expected = {
            "int64": pa.int64(),
            "uint64": pa.uint64(),
            "string": pa.string(),
        }[expected_dtype]
        if field.type.equals(expected):
            return
        overflow = False
        if expected_dtype == "int64" and pa.types.is_unsigned_integer(
            field.type
        ):
            overflow = bool(
                connection.execute(
                    f"SELECT COUNT(*) FROM {_quote(source_view)} WHERE {_quote(column)} > 9223372036854775807"
                ).fetchone()[0]
            )
        elif expected_dtype == "uint64" and pa.types.is_signed_integer(
            field.type
        ):
            overflow = bool(
                connection.execute(
                    f"SELECT COUNT(*) FROM {_quote(source_view)} WHERE {_quote(column)} < 0"
                ).fetchone()[0]
            )
        if overflow:
            raise _artifact_error(
                f"EDGE-ENDPOINT-OVERFLOW-001: endpoint column {column!r} exceeds declared {expected_dtype} domain"
            )
        raise _artifact_error(
            f"EDGE-ENDPOINT-TYPE-001: endpoint column {column!r} exact Arrow type {field.type} does not match {expected}"
        )

    def _edge_id_descriptor(
        self, arrow_type: Any, *, pa: Any
    ) -> dict[str, Any]:
        descriptor = self._primitive_descriptor(arrow_type, pa=pa)
        if (
            descriptor["dtype"] not in _INTEGER_DTYPES
            and descriptor["dtype"] != "string"
        ):
            raise _artifact_error(
                "EDGE-ID-SCHEMA-001: stable edge IDs must be exact integer or string scalars"
            )
        return descriptor

    def _field_descriptor(self, arrow_type: Any, *, pa: Any) -> dict[str, Any]:
        if pa.types.is_fixed_size_list(arrow_type):
            descriptor = self._primitive_descriptor(
                arrow_type.value_type,
                pa=pa,
            )
            descriptor["representation"] = "fixed_size_list"
            descriptor["value_shape"] = (arrow_type.list_size,)
            descriptor["arrow_type"] = str(arrow_type)
            return descriptor
        return self._primitive_descriptor(arrow_type, pa=pa)

    @staticmethod
    def _primitive_descriptor(arrow_type: Any, *, pa: Any) -> dict[str, Any]:
        dtype: str | None = None
        for candidate in _INTEGER_DTYPES + _FLOAT_DTYPES:
            factory = getattr(pa, candidate)
            if arrow_type.equals(factory()):
                dtype = candidate
                break
        if arrow_type.equals(pa.bool_()):
            dtype = "bool"
        elif arrow_type.equals(pa.string()):
            dtype = "string"
        if dtype is None:
            raise _artifact_error(
                f"EDGE-FIELD-SCHEMA-001: unsupported exact Arrow edge value type {arrow_type}"
            )
        storage_dtype = (
            np.dtype("<U1").str
            if dtype == "string"
            else _numpy_dtype(dtype).str
        )
        return {
            "dtype": dtype,
            "storage_dtype": storage_dtype,
            "arrow_type": str(arrow_type),
            "representation": "scalar",
            "value_shape": (),
        }

    @staticmethod
    def _field(schema: Any, column: str, *, code: str, context: Any) -> Any:
        index = schema.get_field_index(column)
        if index < 0:
            raise _artifact_error(
                f"{code}: relation {context!r} is missing configured column {column!r}"
            )
        return schema.field(index)

    def _node(self, name: str) -> Any:
        return next(
            node
            for node in self.ingestor.source.spec.node_types
            if node.name == name
        )

    def _node_key(self, name: str) -> str:
        return next(
            f"n{ordinal:04d}"
            for ordinal, node in enumerate(
                self.ingestor.source.spec.node_types
            )
            if node.name == name
        )

    def _index_bindings(self) -> dict[str, Any]:
        ingestion = _ingestion_module()
        bindings: dict[str, Any] = {}
        for ordinal, node in enumerate(self.ingestor.source.spec.node_types):
            index = self.index_build.indexes[node.name]
            completion = ingestion._read_json(index.completion_path)
            bindings[node.name] = {
                "internal_key": f"n{ordinal:04d}",
                "id_dtype": node.id_dtype,
                "row_count": index.row_count,
                "mapping_completion_sha256": ingestion._sha256_file(
                    index.completion_path
                ),
                "outputs": completion["outputs"],
            }
        return bindings

    def _source_records(
        self, relatives: tuple[str, ...]
    ) -> list[dict[str, Any]]:
        inventory_by_path = {
            entry.relative_path: entry
            for entry in self.index_build.inventory.files
        }
        return [
            {
                "relative_path": relative,
                "byte_size": inventory_by_path[relative].byte_size,
                "row_count": inventory_by_path[relative].row_count,
                "sha256": inventory_by_path[relative].sha256,
                "schema_fingerprint": inventory_by_path[
                    relative
                ].schema_fingerprint,
            }
            for relative in relatives
        ]

    def _array_record(
        self,
        root: Path,
        path: Path,
        *,
        dtype: str,
        shape: tuple[int, ...],
        count: int,
        extra: Mapping[str, Any],
    ) -> dict[str, Any]:
        ingestion = _ingestion_module()
        with path.open("rb") as stream:
            version = np.lib.format.read_magic(stream)
            if version == (1, 0):
                actual_shape, fortran_order, storage_dtype = (
                    np.lib.format.read_array_header_1_0(stream)
                )
            elif version == (2, 0):
                actual_shape, fortran_order, storage_dtype = (
                    np.lib.format.read_array_header_2_0(stream)
                )
            else:
                raise ValueError(
                    f"unsupported NumPy format version {version!r}"
                )
        if fortran_order or actual_shape != shape:
            raise ValueError(f"unexpected NumPy payload layout at {path}")
        return {
            **extra,
            "relative_path": path.relative_to(root).as_posix(),
            "dtype": dtype,
            "storage_dtype": storage_dtype.str,
            "shape": list(shape),
            "count": count,
            "byte_size": path.stat().st_size,
            "file_sha256": ingestion._sha256_file(path),
            "content_sha256": _array_content_sha256(
                path,
                self.ingestor.source.spec.ingestion.record_batch_rows,
            ),
        }

    @staticmethod
    def _primitive_block(values: Any, *, context: str) -> np.ndarray[Any, Any]:
        if values.null_count:
            raise _artifact_error(f"EDGE-NULL-001: {context} contains nulls")
        return values.to_numpy(zero_copy_only=False)

    def _value_block(
        self,
        values: Any,
        descriptor: Mapping[str, Any],
        *,
        context: str,
    ) -> np.ndarray[Any, Any]:
        if values.null_count:
            raise _artifact_error(
                f"EDGE-FIELD-NULL-001: {context} contains nulls"
            )
        if descriptor["representation"] == "fixed_size_list":
            flattened = values.flatten()
            if flattened.null_count:
                raise _artifact_error(
                    f"EDGE-FIELD-NULL-001: {context} contains null list values"
                )
            block = (
                np.asarray(
                    flattened.to_pylist(),
                    dtype=np.dtype(descriptor["storage_dtype"]),
                )
                if descriptor["dtype"] == "string"
                else flattened.to_numpy(zero_copy_only=False)
            )
            expected_values = len(values) * int(descriptor["value_shape"][0])
            if len(block) != expected_values:
                raise _artifact_error(
                    f"EDGE-FIELD-SCHEMA-001: {context} changed fixed width"
                )
            block = block.reshape(len(values), *descriptor["value_shape"])
        elif descriptor["dtype"] == "string":
            block = np.asarray(
                values.to_pylist(),
                dtype=np.dtype(descriptor["storage_dtype"]),
            )
        else:
            block = values.to_numpy(zero_copy_only=False)
        if descriptor["dtype"] != "string":
            expected_dtype = np.dtype(descriptor["storage_dtype"])

            if block.dtype != expected_dtype:
                raise _artifact_error(
                    f"EDGE-FIELD-CAST-001: {context} yielded {block.dtype}, "
                    f"expected exact {expected_dtype}"
                )
        if (
            descriptor["dtype"] in _FLOAT_DTYPES
            and not np.isfinite(block).all()
        ):
            raise _artifact_error(
                f"EDGE-FIELD-FINITE-001: {context} contains NaN or infinity"
            )
        return block

    def _reorder_array(
        self,
        path: Path,
        permutation: np.ndarray[Any, Any],
    ) -> None:
        batch_rows = self.ingestor.source.spec.ingestion.record_batch_rows
        scratch_path = path.with_name(
            f".{path.name}.canonical-{uuid.uuid4().hex}"
        )
        source: Any = None
        output: Any = None
        try:
            source = np.load(path, mmap_mode="r", allow_pickle=False)
            output = _open_npy_output(
                scratch_path,
                dtype=source.dtype,
                shape=source.shape,
            )
            for start in range(0, len(permutation), batch_rows):
                stop = min(len(permutation), start + batch_rows)
                _write_npy_block(
                    output,
                    source[permutation[start:stop]],
                    dtype=source.dtype,
                )
            output.close()
            output = None
            _close_memmap(source)
            source = None
            os.replace(scratch_path, path)
        finally:
            if output is not None:
                output.close()
            if source is not None:
                _close_memmap(source)
            scratch_path.unlink(missing_ok=True)

    def _observe_batch(self, rows: int) -> None:
        limit = self.ingestor.source.spec.ingestion.record_batch_rows
        if rows > limit:
            raise _artifact_error(
                f"BATCH-BOUND-001: observed {rows} relation rows, configured maximum is {limit}"
            )
        self._max_batch_rows = max(self._max_batch_rows, rows)

    def _fsync_prepared_tree(self, root: Path) -> None:
        ingestion = _ingestion_module()
        directories = [root]
        for path in root.rglob("*"):
            if path.is_symlink():
                raise ingestion.ArtifactValidationError(
                    f"UNKNOWN-ARTIFACT-001: symlink in prepared relation stage: {path}"
                )
            if path.is_dir():
                directories.append(path)
                continue
            if path.is_file():
                descriptor = os.open(path, os.O_RDONLY)
                try:
                    os.fsync(descriptor)
                finally:
                    os.close(descriptor)
        directories.sort(
            key=lambda path: (
                -len(path.relative_to(root).parts),
                path.as_posix(),
            )
        )
        for directory in directories:
            ingestion._fsync_directory(directory)

    def _publish(self, temporary_root: Path, stage_root: Path) -> None:
        destination = stage_root / "relations"
        if os.path.lexists(destination):
            raise _artifact_error(
                "INCOMPLETE-RELATION-STAGE-001: relations destination already exists"
            )
        os.replace(temporary_root, destination)
        _ingestion_module()._fsync_directory(stage_root)

    def _quarantine_relations(self, root: Path) -> Path:
        ingestion = _ingestion_module()
        if root.is_symlink() or not root.is_dir():
            raise ingestion.ArtifactValidationError(
                "UNSAFE-STAGING-001: cannot quarantine non-directory relations subtree"
            )
        stage_root = root.parent
        quarantine = stage_root.parent / (
            f".{stage_root.name}.relations-quarantine-{uuid.uuid4().hex}"
        )
        self._downgrade_top_completion_after_relation_quarantine(stage_root)
        os.replace(root, quarantine)
        ingestion._fsync_directory(stage_root)
        ingestion._fsync_directory(stage_root.parent)
        return quarantine

    def _downgrade_top_completion_after_relation_quarantine(
        self,
        stage_root: Path,
    ) -> None:
        ingestion = _ingestion_module()
        completion_path = stage_root / "build.complete.json"
        previous = ingestion._read_json(completion_path)
        previous["stage"] = "typed_graph_arrays"
        previous["array_content_sha256"] = self.array_build.content_sha256
        for field_name in (
            "relation_behavior_version",
            "relation_content_sha256",
            "relation_array_binding_sha256",
        ):
            previous.pop(field_name, None)
        validated_outputs = ingestion._validated_output_map(
            previous.get("outputs")
        )
        previous["outputs"] = {
            relative: checksum
            for relative, checksum in validated_outputs.items()
            if not relative.startswith("relations/")
        }
        ingestion._atomic_json(completion_path, previous)
        ingestion._fsync_directory(stage_root)
        ingestion._fsync_directory(stage_root.parent)

    def _finalize_top_completion(
        self, result: TypedGraphRelationBuild
    ) -> None:
        ingestion = _ingestion_module()
        stage_root = self.index_build.stage_root
        previous = ingestion._read_json(stage_root / "build.complete.json")
        array_binding = self._array_binding()
        array_binding_sha256 = ingestion._sha256_json(array_binding)
        if (
            previous.get("stage") == "typed_graph_relations"
            and previous.get("relation_content_sha256")
            == result.content_sha256
            and previous.get("relation_array_binding_sha256")
            == array_binding_sha256
        ):
            return
        previous["stage"] = "typed_graph_relations"
        previous["relation_behavior_version"] = _RELATION_BEHAVIOR_VERSION
        previous["relation_content_sha256"] = result.content_sha256
        previous["relation_array_binding_sha256"] = array_binding_sha256
        previous["outputs"] = ingestion._stage_output_checksums(stage_root)
        ingestion._atomic_json(stage_root / "build.complete.json", previous)
        ingestion._fsync_directory(stage_root)

    def _relation_outputs(self, root: Path) -> dict[str, str]:
        ingestion = _ingestion_module()
        outputs: dict[str, str] = {}
        for path in sorted(root.rglob("*")):
            if path.is_symlink():
                raise ingestion.ArtifactValidationError(
                    f"UNKNOWN-ARTIFACT-001: symlink in relation stage: {path}"
                )
            if path.is_file() and path.name != "relations.complete.json":
                outputs[path.relative_to(root).as_posix()] = (
                    ingestion._sha256_file(path)
                )
        return outputs

    def _open_validated(
        self,
        *,
        resumed: bool,
        pa: Any,
        pq: Any,
        duckdb: Any,
    ) -> TypedGraphRelationBuild:
        ingestion = _ingestion_module()
        root = self.index_build.stage_root / "relations"
        completion = ingestion._read_json(root / "relations.complete.json")
        inventory = self.index_build.inventory
        if (
            completion.get("stage") != "typed_graph_relations"
            or completion.get("relation_behavior_version")
            != _RELATION_BEHAVIOR_VERSION
            or completion.get("behavior_version")
            != ingestion._BEHAVIOR_VERSION
            or completion.get("input_fingerprint")
            != inventory.source_fingerprint
            or completion.get("config_fingerprint")
            != inventory.config_fingerprint
            or completion.get("dependency_versions")
            != dict(inventory.dependency_versions)
            or completion.get("reopened_and_validated") is not True
        ):
            raise ingestion.ArtifactValidationError(
                "COMPLETION-EVIDENCE-001: relation completion evidence changed"
            )
        disk_admission = completion.get("disk_admission")
        expected_requirements = self._relation_disk_requirements()
        coarse_admission = (
            disk_admission.get("coarse")
            if isinstance(disk_admission, dict)
            else None
        )
        exact_admission = (
            disk_admission.get("exact")
            if isinstance(disk_admission, dict)
            else None
        )
        coarse_observed = (
            coarse_admission.get("observed")
            if isinstance(coarse_admission, dict)
            else None
        )
        exact_observed = (
            exact_admission.get("observed")
            if isinstance(exact_admission, dict)
            else None
        )
        if (
            not isinstance(coarse_admission, dict)
            or coarse_admission.get("requirements") != expected_requirements
            or not isinstance(exact_admission, dict)
            or not isinstance(exact_admission.get("requirements"), dict)
            or not isinstance(coarse_observed, dict)
            or not isinstance(exact_observed, dict)
            or coarse_observed.get("final_device") != inventory.final_device
            or coarse_observed.get("temporary_device")
            != inventory.temporary_device
            or exact_observed.get("final_device") != inventory.final_device
            or exact_observed.get("temporary_device")
            != inventory.temporary_device
        ):
            raise ingestion.ArtifactValidationError(
                "DISK-EVIDENCE-001: Task 4 completion is not bound to both preallocation admissions"
            )
        exact_requirements = exact_admission["requirements"]
        if completion.get("prepared_relation_stage_bytes") != _tree_bytes(
            root
        ):
            raise ingestion.ArtifactValidationError(
                "DISK-EVIDENCE-001: prepared relation stage byte evidence changed"
            )
        outputs = ingestion._validated_output_map(completion.get("outputs"))
        if outputs != self._relation_outputs(root):
            raise ingestion.ArtifactValidationError(
                "COMPLETION-EVIDENCE-001: relation output set or checksums changed"
            )
        metadata = ingestion._read_json(root / "relations.json")
        self._validate_relation_tree(root, metadata, check_completion=True)
        resource = metadata.get("resource_evidence")
        if (
            not isinstance(resource, dict)
            or resource.get("exact_disk_requirements") != exact_requirements
        ):
            raise ingestion.ArtifactValidationError(
                "DISK-EVIDENCE-001: exact relation allocation evidence changed"
            )
        self._validate_resume_disk(exact_requirements, root)
        if (
            completion.get("content_sha256") != metadata.get("content_sha256")
            or completion.get("source_schema_sha256")
            != metadata.get("source_schema_sha256")
            or completion.get("index_bindings")
            != metadata.get("index_bindings")
            or completion.get("array_binding") != metadata.get("array_binding")
        ):
            raise ingestion.ArtifactValidationError(
                "COMPLETION-EVIDENCE-001: relation semantic binding changed"
            )
        self._audit_source_semantics(
            root,
            metadata,
            pa=pa,
            pq=pq,
            duckdb=duckdb,
        )
        return TypedGraphRelationBuild(
            inventory=inventory,
            stage_root=self.index_build.stage_root,
            artifact_root=root,
            content_sha256=metadata["content_sha256"],
            resumed=resumed,
            record_batch_rows=metadata["record_batch_rows"],
            max_record_batch_rows=metadata["max_record_batch_rows"],
        )

    def _validate_relation_tree(
        self,
        root: Path,
        metadata: Mapping[str, Any],
        *,
        check_completion: bool,
    ) -> None:
        ingestion = _ingestion_module()
        inventory = self.index_build.inventory
        if (
            metadata.get("relation_behavior_version")
            != _RELATION_BEHAVIOR_VERSION
            or metadata.get("input_fingerprint")
            != inventory.source_fingerprint
            or metadata.get("config_fingerprint")
            != inventory.config_fingerprint
            or metadata.get("dependency_versions")
            != dict(inventory.dependency_versions)
            or metadata.get("record_batch_rows")
            != self.ingestor.source.spec.ingestion.record_batch_rows
        ):
            raise ingestion.ArtifactValidationError(
                "COMPLETION-EVIDENCE-001: relation semantic evidence changed"
            )
        expected_bindings = self._index_bindings()
        if metadata.get("index_bindings") != expected_bindings or metadata.get(
            "index_bindings_sha256"
        ) != ingestion._sha256_json(expected_bindings):
            raise ingestion.ArtifactValidationError(
                "INDEX-BINDING-001: relation stage is not bound to exact Task 2 indexes"
            )
        expected_array_binding = self._array_binding()
        if metadata.get("array_binding") != expected_array_binding:
            raise ingestion.ArtifactValidationError(
                "ARRAY-BINDING-001: relation stage Task 3 arrays binding changed"
            )
        expected_schema = self._expected_schema_evidence()
        if metadata.get("source_schemas") != expected_schema or metadata.get(
            "source_schema_sha256"
        ) != ingestion._sha256_json(expected_schema):
            raise ingestion.ArtifactValidationError(
                "RELATION-SCHEMA-BINDING-001: source schema evidence changed"
            )
        resource = metadata.get("resource_evidence")
        if (
            not isinstance(resource, dict)
            or resource.get("disk_requirements")
            != self._relation_disk_requirements()
            or not isinstance(
                resource.get("exact_disk_requirements"),
                dict,
            )
        ):
            raise ingestion.ArtifactValidationError(
                "DISK-EVIDENCE-001: Task 4 metadata disk evidence changed"
            )
        relations = metadata.get("relations")
        expected_relations = {
            f"r{ordinal:04d}": relation
            for ordinal, relation in enumerate(
                self.ingestor.source.spec.relations
            )
        }
        if not isinstance(relations, dict) or set(relations) != set(
            expected_relations
        ):
            raise ingestion.ArtifactValidationError(
                "COMPLETION-EVIDENCE-001: completed relation set changed"
            )
        semantic_identity: dict[str, Any] = {
            "relation_behavior_version": _RELATION_BEHAVIOR_VERSION,
            "relations": {},
        }
        for key, relation in expected_relations.items():
            record = relations[key]
            if not isinstance(record, dict):
                raise ingestion.ArtifactValidationError(
                    "COMPLETION-EVIDENCE-001: malformed relation record"
                )
            self._validate_one_relation(
                root, record, relation=relation, key=key
            )
            semantic = _relation_semantic_sha256(
                root,
                record,
                self.ingestor.source.spec.ingestion.record_batch_rows,
            )
            if record.get("semantic_sha256") != semantic:
                raise ingestion.ArtifactValidationError(
                    f"RELATION-SEMANTIC-001: semantic digest changed for {relation.relation!r}"
                )
            semantic_identity["relations"][key] = {
                "relation": list(relation.relation),
                "semantic_sha256": semantic,
            }
        self._validate_exact_disk_evidence(resource, relations)
        if metadata.get(
            "content_identity"
        ) != semantic_identity or metadata.get(
            "content_sha256"
        ) != ingestion._sha256_json(semantic_identity):
            raise ingestion.ArtifactValidationError(
                "RELATION-SEMANTIC-001: relation content identity changed"
            )
        if (
            check_completion
            and not (root / "relations.complete.json").is_file()
        ):
            raise ingestion.ArtifactValidationError(
                "INCOMPLETE-RELATION-STAGE-001: relation completion is missing"
            )

    def _validate_one_relation(
        self,
        root: Path,
        record: Mapping[str, Any],
        *,
        relation: Any,
        key: str,
    ) -> None:
        source_node = self._node(relation.relation[0])
        destination_node = self._node(relation.relation[2])
        source_count = self.index_build.indexes[source_node.name].row_count
        destination_count = self.index_build.indexes[
            destination_node.name
        ].row_count
        source_files = self._source_records(relation.paths)
        expected = {
            "internal_key": key,
            "relation": list(relation.relation),
            "source_node_type": source_node.name,
            "destination_node_type": destination_node.name,
            "source_internal_key": self._node_key(source_node.name),
            "destination_internal_key": self._node_key(destination_node.name),
            "source_column": relation.source_column,
            "destination_column": relation.destination_column,
            "source_id_dtype": source_node.id_dtype,
            "destination_id_dtype": destination_node.id_dtype,
            "source_count": source_count,
            "destination_count": destination_count,
            "edge_id_column": relation.edge_id_column,
            "canonical_order": [
                "destination_local",
                "source_local",
                *(
                    [relation.edge_id_column]
                    if relation.edge_id_column
                    else []
                ),
            ],
            "source_files": source_files,
            "source_sha256": _combined_source_sha(source_files),
            "schema_fingerprint": self._expected_schema_evidence()[key][
                "schema_fingerprint"
            ],
        }
        if any(
            record.get(field) != value for field, value in expected.items()
        ):
            raise _artifact_error(
                f"RELATION-BINDING-001: relation {relation.relation!r} evidence changed"
            )
        edge_count = record.get("edge_count")
        expected_edge_count = sum(value["row_count"] for value in source_files)
        if (
            isinstance(edge_count, bool)
            or not isinstance(edge_count, int)
            or edge_count != expected_edge_count
        ):
            raise _artifact_error(
                f"EDGE-CARDINALITY-001: relation {relation.relation!r} cardinality changed"
            )
        colptr_record = record.get("colptr")
        row_record = record.get("row")
        if not isinstance(colptr_record, dict) or not isinstance(
            row_record, dict
        ):
            raise _artifact_error("CSC-SCHEMA-001: missing CSC array evidence")
        colptr = self._open_array(
            root,
            colptr_record,
            expected_dtype=np.dtype("int64").newbyteorder("<"),
            expected_shape=(destination_count + 1,),
            alignment_code="CSC-COLPTR-001",
        )
        row = self._open_array(
            root,
            row_record,
            expected_dtype=np.dtype("int64").newbyteorder("<"),
            expected_shape=(edge_count,),
            alignment_code="CSC-ROW-001",
        )
        colptr_path = _ingestion_module()._safe_artifact_path(
            root,
            colptr_record["relative_path"],
        )
        row_path = _ingestion_module()._safe_artifact_path(
            root,
            row_record["relative_path"],
        )
        if (
            len(colptr) == 0
            or int(colptr[0]) != 0
            or int(colptr[-1]) != edge_count
        ):
            raise _artifact_error(
                "CSC-COLPTR-001: colptr must begin at zero and end at edge count"
            )
        previous = 0
        batch_rows = self.ingestor.source.spec.ingestion.record_batch_rows
        for start in range(0, len(colptr), batch_rows):
            block = np.asarray(colptr[start : start + batch_rows])
            if len(block) and (
                int(block[0]) < previous or np.any(block[1:] < block[:-1])
            ):
                raise _artifact_error(
                    "CSC-COLPTR-001: colptr is not monotonic"
                )
            if len(block):
                previous = int(block[-1])
            del block
            if start + batch_rows < len(colptr):
                _close_memmap(colptr)
                colptr = np.load(
                    colptr_path,
                    mmap_mode="r",
                    allow_pickle=False,
                )
        for start in range(0, edge_count, batch_rows):
            block = np.asarray(row[start : start + batch_rows])
            if np.any(block < 0) or np.any(block >= source_count):
                raise _artifact_error(
                    "CSC-BOUNDS-001: row contains a source ordinal outside its type"
                )
            del block
            if start + batch_rows < edge_count:
                _close_memmap(row)
                row = np.load(row_path, mmap_mode="r", allow_pickle=False)
        _close_memmap(row)
        row = np.load(row_path, mmap_mode="r", allow_pickle=False)

        edge_record = record.get("edge_id")
        edge_array: np.ndarray[Any, Any] | None = None
        edge_path: Path | None = None
        if relation.edge_id_column is None:
            if edge_record is not None:
                raise _artifact_error(
                    "RELATION-BINDING-001: undeclared stable edge ID array"
                )
        else:
            if not isinstance(edge_record, dict):
                raise _artifact_error(
                    "EDGE-ID-ALIGNMENT-001: stable edge IDs are missing"
                )
            edge_array = self._open_array_from_record(
                root,
                edge_record,
                expected_shape=(edge_count,),
                alignment_code="EDGE-ID-ALIGNMENT-001",
            )
            edge_path = _ingestion_module()._safe_artifact_path(
                root,
                edge_record["relative_path"],
            )
            if edge_record.get("column") != relation.edge_id_column:
                raise _artifact_error(
                    "RELATION-BINDING-001: edge ID role changed"
                )
        fields = record.get("fields")
        if not isinstance(fields, dict) or set(fields) != set(
            relation.edge_fields
        ):
            raise _artifact_error(
                "EDGE-FIELD-ALIGNMENT-001: declared relation field set changed"
            )
        for ordinal, field_name in enumerate(relation.edge_fields):
            field_record = fields[field_name]
            if (
                not isinstance(field_record, dict)
                or field_record.get("column") != field_name
                or field_record.get("internal_key") != f"f{ordinal:04d}"
            ):
                raise _artifact_error(
                    f"RELATION-BINDING-001: field role changed for {field_name!r}"
                )
            shape = field_record.get("shape")
            if (
                not isinstance(shape, list)
                or not shape
                or shape[0] != edge_count
            ):
                raise _artifact_error(
                    f"EDGE-FIELD-ALIGNMENT-001: field {field_name!r} length differs from edge count"
                )
            array = self._open_array_from_record(
                root,
                field_record,
                expected_shape=tuple(shape),
                alignment_code="EDGE-FIELD-ALIGNMENT-001",
            )
            field_path = _ingestion_module()._safe_artifact_path(
                root,
                field_record["relative_path"],
            )
            if np.issubdtype(array.dtype, np.floating):
                for start in range(0, edge_count, batch_rows):
                    if not np.isfinite(
                        array[start : start + batch_rows]
                    ).all():
                        raise _artifact_error(
                            f"EDGE-FIELD-FINITE-001: field {field_name!r} contains NaN or infinity"
                        )
                    if start + batch_rows < edge_count:
                        _close_memmap(array)
                        array = np.load(
                            field_path,
                            mmap_mode="r",
                            allow_pickle=False,
                        )
            _close_memmap(array)

        resident_rows = 0
        resident_destinations = 0
        for destination in range(destination_count):
            start = int(colptr[destination])
            stop = int(colptr[destination + 1])
            previous_order: tuple[Any, ...] | None = None
            for chunk_start in range(start, stop, batch_rows):
                chunk_stop = min(stop, chunk_start + batch_rows)
                for position in range(chunk_start, chunk_stop):
                    source = int(row[position])
                    order = (
                        (source, _numpy_scalar(edge_array[position]))
                        if edge_array is not None
                        else (source,)
                    )
                    if previous_order is not None and order <= previous_order:
                        raise _artifact_error(
                            "CSC-ORDER-001: relation rows are not in canonical source/edge order"
                        )
                    previous_order = order
                    resident_rows += 1
                    if resident_rows >= batch_rows:
                        _close_memmap(row)
                        row = np.load(
                            row_path, mmap_mode="r", allow_pickle=False
                        )
                        if edge_array is not None:
                            assert edge_path is not None
                            _close_memmap(edge_array)
                            edge_array = np.load(
                                edge_path,
                                mmap_mode="r",
                                allow_pickle=False,
                            )
                        resident_rows = 0
            resident_destinations += 1
            if resident_destinations >= batch_rows:
                _close_memmap(colptr)
                colptr = np.load(
                    colptr_path,
                    mmap_mode="r",
                    allow_pickle=False,
                )
                resident_destinations = 0
        _close_memmap(colptr)
        _close_memmap(row)
        if edge_array is not None:
            _close_memmap(edge_array)
        self._validate_record_checksum(root, colptr_record)
        self._validate_record_checksum(root, row_record)
        if isinstance(edge_record, dict):
            self._validate_record_checksum(root, edge_record)
        for field_record in fields.values():
            self._validate_record_checksum(root, field_record)

    def _open_array(
        self,
        root: Path,
        record: Mapping[str, Any],
        *,
        expected_dtype: np.dtype[Any],
        expected_shape: tuple[int, ...],
        alignment_code: str,
    ) -> np.ndarray[Any, Any]:
        array = self._open_array_from_record(
            root,
            record,
            expected_shape=expected_shape,
            alignment_code=alignment_code,
        )
        if (
            array.dtype != expected_dtype
            or record.get("storage_dtype") != expected_dtype.str
        ):
            raise _artifact_error(
                f"{alignment_code}: array storage dtype is not exact little-endian {expected_dtype.str}"
            )
        return array

    def _open_array_from_record(
        self,
        root: Path,
        record: Mapping[str, Any],
        *,
        expected_shape: tuple[int, ...],
        alignment_code: str,
    ) -> np.ndarray[Any, Any]:
        ingestion = _ingestion_module()
        relative = record.get("relative_path")
        if not isinstance(relative, str):
            raise _artifact_error(f"{alignment_code}: array path is missing")
        path = ingestion._safe_artifact_path(root, relative)
        if path.is_symlink() or not path.is_file():
            raise _artifact_error(
                f"{alignment_code}: array artifact is unsafe or missing"
            )
        try:
            array = np.load(path, mmap_mode="r", allow_pickle=False)
        except (OSError, ValueError) as error:
            raise _artifact_error(
                f"{alignment_code}: malformed NumPy array"
            ) from error
        if array.shape != expected_shape:
            raise _artifact_error(
                f"{alignment_code}: array shape {array.shape} differs from {expected_shape}"
            )
        if (
            record.get("shape") != list(expected_shape)
            or record.get("count") != expected_shape[0]
            or record.get("storage_dtype") != array.dtype.str
        ):
            raise _artifact_error(f"{alignment_code}: array evidence changed")
        return array

    def _validate_exact_disk_evidence(
        self,
        resource: Mapping[str, Any],
        relations: Mapping[str, Any],
    ) -> None:
        ingestion = _ingestion_module()
        exact = resource["exact_disk_requirements"]
        arrays: dict[str, dict[str, Any]] = {}
        file_bytes = 0
        payload_bytes = 0
        max_canonical_permutation_bytes = 0
        max_canonical_scratch_file_bytes = 0
        for key in sorted(relations):
            record = relations[key]
            colptr = _npy_storage_sizing(
                np.dtype(record["colptr"]["storage_dtype"]),
                tuple(record["colptr"]["shape"]),
            )
            row = _npy_storage_sizing(
                np.dtype(record["row"]["storage_dtype"]),
                tuple(record["row"]["shape"]),
            )
            edge_id = None
            if record["edge_id"] is not None:
                edge_id = _npy_storage_sizing(
                    np.dtype(record["edge_id"]["storage_dtype"]),
                    tuple(record["edge_id"]["shape"]),
                )
            fields = {
                name: _npy_storage_sizing(
                    np.dtype(record["fields"][name]["storage_dtype"]),
                    tuple(record["fields"][name]["shape"]),
                )
                for name in sorted(record["fields"])
            }
            arrays[key] = {
                "colptr": colptr,
                "row": row,
                "edge_id": edge_id,
                "fields": fields,
            }
            records = [colptr, row]
            if edge_id is not None:
                records.append(edge_id)
            records.extend(fields[name] for name in sorted(fields))
            file_bytes += sum(int(item["file_bytes"]) for item in records)
            payload_bytes += sum(
                int(item["payload_bytes"]) for item in records
            )
            permutation_scratch = _npy_storage_sizing(
                np.dtype("<i8"),
                (record["edge_count"],),
            )
            key_row_bytes = 3 * np.dtype("<i8").itemsize
            if record["edge_id"] is not None:
                key_row_bytes += np.dtype(
                    record["edge_id"]["storage_dtype"]
                ).itemsize
            key_row_groups = (
                record["edge_count"]
                + self.ingestor.source.spec.ingestion.record_batch_rows
                - 1
            ) // self.ingestor.source.spec.ingestion.record_batch_rows
            key_file_upper_bound = (
                record["edge_count"] * key_row_bytes
                + key_row_groups * 64 * 1024
                + 1024**2
            )
            reorder_records = [row]
            if edge_id is not None:
                reorder_records.append(edge_id)
            reorder_records.extend(fields[name] for name in sorted(fields))
            max_reorder_file_bytes = max(
                (int(item["file_bytes"]) for item in reorder_records),
                default=0,
            )
            max_canonical_scratch_file_bytes = max(
                max_canonical_scratch_file_bytes,
                int(permutation_scratch["file_bytes"])
                + max(key_file_upper_bound, max_reorder_file_bytes),
            )
            max_canonical_permutation_bytes = max(
                max_canonical_permutation_bytes,
                int(record["edge_count"]) * np.dtype(np.intp).itemsize,
            )
        coarse = resource["disk_requirements"]
        evidence_reserve = int(coarse["relation_evidence_reserve_bytes"])
        estimated = file_bytes + evidence_reserve
        final_build_bytes = estimated + max_canonical_scratch_file_bytes
        final_peak = int(coarse["task3_final_bytes"]) + final_build_bytes
        temporary_peak = int(coarse["snapshot_bytes"]) + int(
            coarse["spill_bytes"]
        )
        expected = {
            "same_filesystem": bool(coarse["same_filesystem"]),
            "task2_final_bytes": int(coarse["task2_final_bytes"]),
            "task2_required_peak_bytes": int(
                coarse["task2_required_peak_bytes"]
            ),
            "task3_final_bytes": int(coarse["task3_final_bytes"]),
            "task3_array_bytes": int(coarse["task3_array_bytes"]),
            "estimated_relation_bytes": estimated,
            "exact_array_file_bytes": file_bytes,
            "exact_array_payload_bytes": payload_bytes,
            "relation_evidence_reserve_bytes": evidence_reserve,
            "relation_rows": int(coarse["relation_rows"]),
            "snapshot_bytes": int(coarse["snapshot_bytes"]),
            "spill_bytes": int(coarse["spill_bytes"]),
            "max_canonical_permutation_bytes": max_canonical_permutation_bytes,
            "max_canonical_scratch_file_bytes": max_canonical_scratch_file_bytes,
            "final_peak_bytes": final_peak,
            "temporary_peak_bytes": temporary_peak,
            "resume_temporary_additional_bytes": temporary_peak,
            "shared_peak_bytes": final_peak + temporary_peak,
            "final_additional_bytes": final_build_bytes,
            "temporary_additional_bytes": temporary_peak,
            "shared_additional_bytes": final_build_bytes + temporary_peak,
            "arrays": arrays,
        }
        if exact != expected:
            raise ingestion.ArtifactValidationError(
                "DISK-EVIDENCE-001: exact array allocation evidence changed"
            )
        if (
            resource.get("canonical_permutation_bytes")
            != max_canonical_permutation_bytes
        ):
            raise ingestion.ArtifactValidationError(
                "DISK-EVIDENCE-001: canonical permutation memory evidence changed"
            )

    def _validate_record_checksum(
        self,
        root: Path,
        record: Mapping[str, Any],
    ) -> None:
        ingestion = _ingestion_module()
        path = ingestion._safe_artifact_path(root, record["relative_path"])
        if (
            record.get("byte_size") != path.stat().st_size
            or record.get("file_sha256") != ingestion._sha256_file(path)
            or record.get("content_sha256")
            != _array_content_sha256(
                path,
                self.ingestor.source.spec.ingestion.record_batch_rows,
            )
        ):
            raise ingestion.ArtifactValidationError(
                f"CHECKSUM-001: relation array checksum mismatch for {record['relative_path']!r}"
            )

    def _expected_schema_evidence(self) -> dict[str, dict[str, Any]]:
        inventory_by_path = {
            entry.relative_path: entry
            for entry in self.index_build.inventory.files
        }
        evidence: dict[str, dict[str, Any]] = {}
        for ordinal, relation in enumerate(
            self.ingestor.source.spec.relations
        ):
            entries = [
                inventory_by_path[relative] for relative in relation.paths
            ]
            fingerprints = {entry.schema_fingerprint for entry in entries}
            serialized = {entry.schema_serialized_hex for entry in entries}
            if len(fingerprints) != 1 or len(serialized) != 1:
                raise _artifact_error(
                    f"SCHEMA-DRIFT-001: relation {relation.relation!r} fragments differ"
                )
            evidence[f"r{ordinal:04d}"] = {
                "relation": list(relation.relation),
                "source_files": self._source_records(relation.paths),
                "schema_fingerprint": next(iter(fingerprints)),
                "schema_serialized_hex": next(iter(serialized)),
            }
        return evidence

    def _audit_snapshot_streams(
        self,
        root: Path,
        metadata: Mapping[str, Any],
        *,
        snapshots: Mapping[str, Path],
        connection: Any,
        pa: Any,
        pq: Any,
        view_prefix: str,
    ) -> None:
        prepared: dict[str, tuple[Any, Mapping[str, Any]]] = {}
        for ordinal, relation in enumerate(
            self.ingestor.source.spec.relations
        ):
            key = f"r{ordinal:04d}"
            context = self._prepare_relation(
                relation=relation,
                internal_key=f"{view_prefix}_{key}",
                snapshots=snapshots,
                connection=connection,
                pa=pa,
                pq=pq,
            )
            prepared[key] = (relation, context)
        exact_requirements = self._exact_relation_disk_requirements(
            prepared,
            coarse_requirements=self._relation_disk_requirements(),
        )
        resource = metadata.get("resource_evidence")
        if (
            not isinstance(resource, Mapping)
            or resource.get("exact_disk_requirements") != exact_requirements
        ):
            raise _artifact_error(
                "DISK-EVIDENCE-001: exact source-derived relation allocation evidence changed"
            )
        for key in sorted(prepared):
            _, context = prepared[key]
            self._compare_source_stream(
                root,
                metadata["relations"][key],
                context,
                connection=connection,
            )

    def _audit_source_semantics(
        self,
        root: Path,
        metadata: Mapping[str, Any],
        *,
        pa: Any,
        pq: Any,
        duckdb: Any,
    ) -> None:
        inventory = self.index_build.inventory
        spill_root = self.ingestor._new_ephemeral_root(
            inventory,
            purpose="relation-resume-audit",
        )
        spill_created = False
        try:
            spill_root.mkdir(parents=True, exist_ok=False)
            spill_created = True
            with self.ingestor._immutable_source_snapshot(
                inventory,
                purpose="relation-audit-source-snapshot",
                pa=pa,
                pq=pq,
            ) as snapshot_stage:
                snapshots, _ = snapshot_stage
                connection = duckdb.connect(str(spill_root / "audit.duckdb"))
                try:
                    self._configure_connection(connection, spill_root)
                    self._audit_snapshot_streams(
                        root,
                        metadata,
                        snapshots=snapshots,
                        connection=connection,
                        pa=pa,
                        pq=pq,
                        view_prefix="resume",
                    )
                finally:
                    connection.close()
        finally:
            if spill_created and spill_root.exists():
                shutil.rmtree(spill_root)

    def _compare_source_stream(
        self,
        root: Path,
        record: Mapping[str, Any],
        context: Mapping[str, Any],
        *,
        connection: Any,
    ) -> None:
        ingestion = _ingestion_module()
        edge_descriptor = context["edge_descriptor"]
        edge_record = record["edge_id"]
        if (edge_descriptor is None) != (edge_record is None):
            raise _artifact_error(
                "RELATION-BINDING-001: stable edge ID role changed"
            )
        if edge_descriptor is not None:
            expected_edge = {
                "dtype": edge_descriptor["dtype"],
                "storage_dtype": edge_descriptor["storage_dtype"],
                "arrow_type": edge_descriptor["arrow_type"],
                "representation": "scalar",
                "shape": [context["edge_count"]],
            }
            if any(
                edge_record.get(field) != value
                for field, value in expected_edge.items()
            ):
                raise _artifact_error(
                    "RELATION-BINDING-001: stable edge ID schema changed"
                )
        for field_name, descriptor in context["field_descriptors"].items():
            field_record = record["fields"].get(field_name)
            expected_field = {
                "dtype": descriptor["dtype"],
                "storage_dtype": descriptor["storage_dtype"],
                "arrow_type": descriptor["arrow_type"],
                "representation": descriptor["representation"],
                "value_shape": list(descriptor["value_shape"]),
                "shape": [
                    context["edge_count"],
                    *descriptor["value_shape"],
                ],
            }
            if not isinstance(field_record, dict) or any(
                field_record.get(field) != value
                for field, value in expected_field.items()
            ):
                raise _artifact_error(
                    f"RELATION-BINDING-001: edge field {field_name!r} schema changed"
                )
        colptr_path = ingestion._safe_artifact_path(
            root,
            record["colptr"]["relative_path"],
        )
        row_path = ingestion._safe_artifact_path(
            root,
            record["row"]["relative_path"],
        )
        edge_path = (
            ingestion._safe_artifact_path(
                root,
                record["edge_id"]["relative_path"],
            )
            if record["edge_id"] is not None
            else None
        )
        field_paths = {
            name: ingestion._safe_artifact_path(root, value["relative_path"])
            for name, value in record["fields"].items()
        }
        colptr = np.load(colptr_path, mmap_mode="r", allow_pickle=False)
        row = np.load(row_path, mmap_mode="r", allow_pickle=False)
        edge = (
            np.load(edge_path, mmap_mode="r", allow_pickle=False)
            if edge_path is not None
            else None
        )
        fields = {
            name: np.load(path, mmap_mode="r", allow_pickle=False)
            for name, path in field_paths.items()
        }
        offset = 0
        page_bytes = os.sysconf("SC_PAGE_SIZE")
        lookup_arrays = 3 + int(edge is not None) + len(fields)
        audit_resident_bytes = min(
            4 * 1024**2,
            self.ingestor.source.spec.ingestion.memory_limit_bytes // 16,
        )
        audit_batch_rows = min(
            self.ingestor.source.spec.ingestion.record_batch_rows,
            max(1, audit_resident_bytes // (page_bytes * lookup_arrays)),
        )
        reader = connection.execute(context["stream_query"]).to_arrow_reader(
            audit_batch_rows
        )
        for batch in reader:
            rows = batch.num_rows
            self._observe_batch(rows)
            destinations = self._primitive_block(
                batch.column(0), context="audited destination ordinals"
            )
            sources = self._primitive_block(
                batch.column(1), context="audited source ordinals"
            )
            if (
                np.any(destinations < 0)
                or np.any(destinations >= context["destination_count"])
                or np.any(sources < 0)
                or np.any(sources >= context["source_count"])
            ):
                raise _artifact_error(
                    "RELATION-SEMANTIC-001: source relation endpoints exceed canonical bounds"
                )
            column = 2
            edge_block: np.ndarray[Any, Any] | None = None
            if edge is not None:
                edge_block = self._value_block(
                    batch.column(column),
                    context["edge_descriptor"],
                    context="audited stable edge IDs",
                )
                column += 1
            positions = np.empty(rows, dtype=np.int64)
            for local_index in range(rows):
                destination = int(destinations[local_index])
                source = int(sources[local_index])
                start = int(colptr[destination])
                stop = int(colptr[destination + 1])
                source_segment = row[start:stop]
                left = int(
                    np.searchsorted(source_segment, source, side="left")
                )
                right = int(
                    np.searchsorted(source_segment, source, side="right")
                )
                if left == right:
                    raise _artifact_error(
                        "RELATION-SEMANTIC-001: source edge is absent from canonical CSC"
                    )
                if edge is None:
                    if right - left != 1:
                        raise _artifact_error(
                            "RELATION-SEMANTIC-001: endpoint pair is ambiguous without stable edge IDs"
                        )
                    positions[local_index] = start + left
                    continue
                assert edge_block is not None
                expected_edge = _numpy_scalar(edge_block[local_index])
                edge_segment = edge[start + left : start + right]
                edge_offset = int(
                    np.searchsorted(edge_segment, expected_edge, side="left")
                )
                if (
                    edge_offset >= len(edge_segment)
                    or _numpy_scalar(edge_segment[edge_offset])
                    != expected_edge
                ):
                    raise _artifact_error(
                        "RELATION-SEMANTIC-001: stable edge ID is absent from canonical CSC"
                    )
                positions[local_index] = start + left + edge_offset
            if not np.array_equal(row[positions], sources):
                raise _artifact_error(
                    "RELATION-SEMANTIC-001: CSC row values differ from source relation"
                )
            if edge is not None:
                assert edge_block is not None
                if not _arrays_bitwise_equal(edge[positions], edge_block):
                    raise _artifact_error(
                        "RELATION-SEMANTIC-001: stable edge IDs differ from source relation"
                    )
            for field_name, descriptor in context["field_descriptors"].items():
                block = self._value_block(
                    batch.column(column),
                    descriptor,
                    context=f"audited edge field {field_name!r}",
                )
                if not _arrays_bitwise_equal(
                    fields[field_name][positions],
                    block,
                ):
                    raise _artifact_error(
                        f"RELATION-SEMANTIC-001: field {field_name!r} differs from source relation"
                    )
                column += 1
            del source_segment
            if edge is not None:
                del edge_segment
            offset += rows
            _close_memmap(row)
            if edge is not None:
                _close_memmap(edge)
            for field in fields.values():
                _close_memmap(field)
            if offset < context["edge_count"]:
                row = np.load(row_path, mmap_mode="r", allow_pickle=False)
                edge = (
                    np.load(edge_path, mmap_mode="r", allow_pickle=False)
                    if edge_path is not None
                    else None
                )
                fields = {
                    name: np.load(path, mmap_mode="r", allow_pickle=False)
                    for name, path in field_paths.items()
                }
        _close_memmap(colptr)
        _close_memmap(row)
        if edge is not None:
            _close_memmap(edge)
        for field in fields.values():
            _close_memmap(field)
        if offset != context["edge_count"]:
            raise _artifact_error(
                "RELATION-SEMANTIC-001: source relation cardinality changed"
            )


def _arrays_bitwise_equal(
    left: np.ndarray[Any, Any],
    right: np.ndarray[Any, Any],
) -> bool:
    return (
        left.shape == right.shape
        and left.dtype == right.dtype
        and np.array_equal(
            left.reshape(-1).view(np.uint8),
            right.reshape(-1).view(np.uint8),
        )
    )


def _relation_semantic_sha256(
    root: Path,
    record: Mapping[str, Any],
    batch_rows: int,
) -> str:
    contract = {
        "relation": record["relation"],
        "source_node_type": record["source_node_type"],
        "destination_node_type": record["destination_node_type"],
        "source_id_dtype": record["source_id_dtype"],
        "destination_id_dtype": record["destination_id_dtype"],
        "source_count": record["source_count"],
        "destination_count": record["destination_count"],
        "edge_count": record["edge_count"],
        "canonical_order": record["canonical_order"],
        "edge_id": (
            None
            if record["edge_id"] is None
            else {
                field: record["edge_id"][field]
                for field in (
                    "column",
                    "dtype",
                    "storage_dtype",
                    "arrow_type",
                    "representation",
                    "shape",
                )
            }
        ),
        "fields": {
            name: {
                field: record["fields"][name][field]
                for field in (
                    "column",
                    "dtype",
                    "storage_dtype",
                    "arrow_type",
                    "representation",
                    "value_shape",
                    "shape",
                )
            }
            for name in sorted(record["fields"])
        },
    }
    digest = hashlib.sha256()
    contract_bytes = json.dumps(
        contract,
        sort_keys=True,
        separators=(",", ":"),
        ensure_ascii=False,
    ).encode("utf-8")
    _start_digest_frame(
        digest, role="relation-contract", byte_length=len(contract_bytes)
    )
    digest.update(contract_bytes)
    components: list[tuple[str, str | None, Mapping[str, Any]]] = [
        ("colptr", None, record["colptr"]),
        ("row", None, record["row"]),
    ]
    if record["edge_id"] is not None:
        components.append(
            ("edge_id", record["edge_id"]["column"], record["edge_id"])
        )
    components.extend(
        ("field", name, record["fields"][name])
        for name in sorted(record["fields"])
    )

    for role, name, component in components:
        path = root / component["relative_path"]
        dtype = np.dtype(component["storage_dtype"])
        shape = tuple(component["shape"])
        byte_length = dtype.itemsize
        for extent in shape:
            byte_length *= extent
        header = json.dumps(
            {
                "role": role,
                "name": name,
                "logical_dtype": component["dtype"],
                "storage_dtype": component["storage_dtype"],
                "shape": component["shape"],
                "byte_length": byte_length,
            },
            sort_keys=True,
            separators=(",", ":"),
            ensure_ascii=False,
        ).encode("utf-8")
        _start_digest_frame(
            digest,
            role="component-header",
            byte_length=len(header),
        )
        digest.update(header)
        payload_role = role if name is None else f"{role}:{name}"
        _start_digest_frame(
            digest,
            role=f"component-bytes:{payload_role}",
            byte_length=byte_length,
        )
        with path.open("rb") as stream:
            version = np.lib.format.read_magic(stream)
            if version == (1, 0):
                actual_shape, fortran_order, actual_dtype = (
                    np.lib.format.read_array_header_1_0(stream)
                )
            elif version == (2, 0):
                actual_shape, fortran_order, actual_dtype = (
                    np.lib.format.read_array_header_2_0(stream)
                )
            else:
                raise ValueError(
                    f"unsupported NumPy format version {version!r}"
                )
            if fortran_order or actual_shape != shape or actual_dtype != dtype:
                raise ValueError(f"unexpected NumPy payload layout at {path}")
            row_bytes = dtype.itemsize
            for extent in shape[1:]:
                row_bytes *= extent
            remaining = byte_length
            read_bytes = max(row_bytes, batch_rows * row_bytes)
            while remaining:
                block = stream.read(min(remaining, read_bytes))
                if not block:
                    raise ValueError(
                        f"truncated NumPy array payload at {path}"
                    )
                digest.update(block)
                remaining -= len(block)
    return digest.hexdigest()


def _start_digest_frame(
    digest: Any,
    *,
    role: str,
    byte_length: int,
) -> None:
    role_bytes = role.encode("utf-8")
    digest.update(len(role_bytes).to_bytes(8, "big", signed=False))
    digest.update(role_bytes)
    digest.update(byte_length.to_bytes(8, "big", signed=False))


def _array_content_sha256(path: Path, batch_rows: int) -> str:
    with path.open("rb") as stream:
        version = np.lib.format.read_magic(stream)
        if version == (1, 0):
            shape, fortran_order, dtype = np.lib.format.read_array_header_1_0(
                stream
            )
        elif version == (2, 0):
            shape, fortran_order, dtype = np.lib.format.read_array_header_2_0(
                stream
            )
        else:
            raise ValueError(f"unsupported NumPy format version {version!r}")
        if not fortran_order:
            row_bytes = dtype.itemsize
            for extent in shape[1:]:
                row_bytes *= extent
            remaining = (shape[0] if shape else 1) * row_bytes
            read_bytes = max(row_bytes, batch_rows * row_bytes)
            digest = hashlib.sha256()
            while remaining:
                block = stream.read(min(remaining, read_bytes))
                if not block:
                    raise ValueError(
                        f"truncated NumPy array payload at {path}"
                    )
                digest.update(block)
                remaining -= len(block)
            return digest.hexdigest()
    array = np.load(path, mmap_mode="r", allow_pickle=False)
    digest = hashlib.sha256()
    for start in range(0, len(array), batch_rows):
        block = np.ascontiguousarray(array[start : start + batch_rows])
        digest.update(memoryview(block).cast("B"))
    return digest.hexdigest()


def _without_physical_array_evidence(value: Any) -> Any:
    if isinstance(value, Mapping):
        return {
            key: _without_physical_array_evidence(item)
            for key, item in value.items()
            if key
            not in {
                "relative_path",
                "byte_size",
                "file_sha256",
                "content_sha256",
            }
        }
    if isinstance(value, list):
        return [_without_physical_array_evidence(item) for item in value]
    return value


def _combined_source_sha(records: list[dict[str, Any]]) -> str:
    if len(records) == 1:
        return records[0]["sha256"]
    encoded = json.dumps(
        [
            {
                "relative_path": value["relative_path"],
                "sha256": value["sha256"],
            }
            for value in records
        ],
        sort_keys=True,
        separators=(",", ":"),
    ).encode("utf-8")
    return hashlib.sha256(encoded).hexdigest()


def _npy_storage_sizing(
    dtype: np.dtype[Any],
    shape: tuple[int, ...],
) -> dict[str, Any]:
    canonical_dtype = np.dtype(dtype)
    element_count = 1
    for extent in shape:
        element_count *= int(extent)
    payload_bytes = element_count * canonical_dtype.itemsize
    header = io.BytesIO()
    np.lib.format.write_array_header_1_0(
        header,
        {
            "descr": np.lib.format.dtype_to_descr(canonical_dtype),
            "fortran_order": False,
            "shape": shape,
        },
    )
    header_bytes = header.tell()
    return {
        "storage_dtype": canonical_dtype.str,
        "shape": list(shape),
        "element_count": element_count,
        "itemsize": canonical_dtype.itemsize,
        "header_bytes": header_bytes,
        "payload_bytes": payload_bytes,
        "file_bytes": header_bytes + payload_bytes,
    }


def _numpy_dtype(dtype: str) -> np.dtype[Any]:
    if dtype == "bool":
        return np.dtype("bool")
    return np.dtype(dtype).newbyteorder("<")


def _numpy_scalar(value: Any) -> Any:
    return value.item() if isinstance(value, np.generic) else value


def _close_memmap(array: np.ndarray[Any, Any]) -> None:
    mapping = getattr(array, "_mmap", None)
    if mapping is not None and not mapping.closed:
        mapping.close()


def _quote(value: str) -> str:
    return '"' + value.replace('"', '""') + '"'


def _open_npy_output(
    path: Path,
    *,
    dtype: np.dtype[Any],
    shape: tuple[int, ...],
) -> Any:
    element_count = 1
    for extent in shape:
        element_count *= extent
    with path.open("wb") as output:
        np.lib.format.write_array_header_1_0(
            output,
            {
                "descr": np.lib.format.dtype_to_descr(dtype),
                "fortran_order": False,
                "shape": shape,
            },
        )
        data_offset = output.tell()
        output.truncate(data_offset + element_count * dtype.itemsize)
    stream = path.open("r+b", buffering=0)
    stream.seek(data_offset)
    return stream


def _write_npy_block(
    stream: Any,
    block: np.ndarray[Any, Any],
    *,
    dtype: np.dtype[Any],
) -> None:
    contiguous = np.asarray(block, dtype=dtype, order="C")
    if not contiguous.flags.c_contiguous:
        contiguous = np.ascontiguousarray(contiguous)
    remaining = memoryview(contiguous).cast("B")
    while remaining:
        written = stream.write(remaining)
        if written is None or written <= 0:
            raise OSError("failed to write NumPy array block")
        remaining = remaining[written:]


def _fsync_file(path: Path) -> None:
    with path.open("rb") as stream:
        os.fsync(stream.fileno())


def _tree_bytes(root: Path) -> int:
    return sum(
        path.stat().st_size
        for path in root.rglob("*")
        if path.is_file() and not path.is_symlink()
    )


def _artifact_error(message: str) -> Any:
    return _ingestion_module().ArtifactValidationError(message)


def _ingestion_module() -> Any:
    from topobench.data.stores import typed_graph_ingestion

    return typed_graph_ingestion


__all__ = ["TypedGraphRelationBuild", "TypedGraphRelationWriter"]
