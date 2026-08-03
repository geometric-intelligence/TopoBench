"""Subprocess qualification for bounded typed-Parquet conversion and reads.

This is a release runner, not a pytest test.  Its thresholds are declared before
fixture generation and every missing Parquet/partition prerequisite is fatal.
Only aggregate resource and semantic evidence is emitted.
"""

from __future__ import annotations

import json
import math
import os
import re
import resource
import subprocess
import sys
import tempfile
from collections.abc import Mapping
from pathlib import Path
from typing import Any

_MIB = 1024**2
NODE_ROWS = int(os.environ.get("TOPOBENCH_TYPED_RSS_NODE_ROWS", "4000000"))
EDGE_ROWS = int(os.environ.get("TOPOBENCH_TYPED_RSS_EDGE_ROWS", "3500000"))
FEATURE_WIDTH = int(os.environ.get("TOPOBENCH_TYPED_RSS_FEATURE_WIDTH", "16"))
EDGE_FIELD_COUNT = int(
    os.environ.get("TOPOBENCH_TYPED_RSS_EDGE_FIELD_COUNT", "1")
)
EDGE_FIELDS = tuple(f"weight_{index:02d}" for index in range(EDGE_FIELD_COUNT))
RECORD_BATCH_ROWS = int(os.environ.get("TOPOBENCH_TYPED_RSS_BATCH_ROWS", "8192"))
RSS_DELTA_LIMIT_BYTES = int(
    os.environ.get("TOPOBENCH_TYPED_RSS_DELTA_LIMIT_BYTES", str(320 * _MIB))
)
PAYLOAD_TO_RSS_FACTOR = float(
    os.environ.get("TOPOBENCH_TYPED_RSS_PAYLOAD_FACTOR", "1.05")
)
DUCKDB_MEMORY_LIMIT_BYTES = int(
    os.environ.get("TOPOBENCH_TYPED_RSS_DUCKDB_BYTES", str(64 * _MIB))
)
PARTITION_MEMORY_LIMIT_BYTES = int(
    os.environ.get("TOPOBENCH_TYPED_PARTITION_RSS_BYTES", str(2 * 1024**3))
)
VALIDATION_WORKER_RSS_DELTA_LIMIT_BYTES = 320 * _MIB
NUM_PARTITIONS = int(os.environ.get("TOPOBENCH_TYPED_RSS_PARTITIONS", "8"))
WORKER_TIMEOUT_SECONDS = int(
    os.environ.get("TOPOBENCH_TYPED_RSS_TIMEOUT_SECONDS", "3600")
)


def _rss_bytes() -> int:
    observed = resource.getrusage(resource.RUSAGE_SELF).ru_maxrss
    return int(observed if sys.platform == "darwin" else observed * 1024)


def _directory_bytes(root: Path) -> int:
    return sum(path.stat().st_size for path in root.rglob("*") if path.is_file())


def _feature_batch(start: int, count: int, width: int) -> Any:
    import numpy as np
    import pyarrow as pa

    rows = np.arange(start, start + count, dtype=np.float32)[:, None]
    columns = np.arange(width, dtype=np.float32)[None, :]
    values = ((rows % 997.0) + (columns % 31.0)) / 997.0
    flat = pa.array(values.reshape(-1), type=pa.float32())
    return pa.FixedSizeListArray.from_arrays(flat, width)


def _write_node_files(
    root: Path,
    *,
    node_type: str,
    count: int,
    id_column: str,
    id_offset: int = 0,
    include_label: bool = True,
) -> tuple[str, ...]:
    import numpy as np
    import pyarrow as pa
    import pyarrow.parquet as pq

    paths: list[str] = []
    file_boundary = math.ceil(count / 2)
    for file_index, (begin, end) in enumerate(
        ((0, file_boundary), (file_boundary, count))
    ):
        relative = f"nodes/{node_type}-{file_index:02d}.parquet"
        path = root / relative
        path.parent.mkdir(parents=True, exist_ok=True)
        fields = [
            pa.field(id_column, pa.int64(), nullable=False),
            pa.field(
                "features",
                pa.list_(pa.float32(), FEATURE_WIDTH),
                nullable=False,
            ),
        ]
        if include_label:
            fields.append(pa.field("label", pa.int64(), nullable=False))
        schema = pa.schema(fields)
        with pq.ParquetWriter(path, schema, compression="zstd") as writer:
            for start in range(begin, end, RECORD_BATCH_ROWS):
                size = min(RECORD_BATCH_ROWS, end - start)
                identifiers = np.arange(
                    id_offset + start,
                    id_offset + start + size,
                    dtype=np.int64,
                )
                columns = [
                    pa.array(identifiers, type=pa.int64()),
                    _feature_batch(start, size, FEATURE_WIDTH),
                ]
                if include_label:
                    columns.append(pa.array(identifiers % 4, type=pa.int64()))
                writer.write_batch(pa.record_batch(columns, schema=schema))
        paths.append(relative)
    return tuple(paths)


def _write_edge_files(
    root: Path,
    *,
    name: str,
    count: int,
    source_count: int,
    destination_count: int,
    reverse: bool = False,
    edge_id_offset: int = 0,
) -> tuple[str, ...]:
    import numpy as np
    import pyarrow as pa
    import pyarrow.parquet as pq

    paths: list[str] = []
    boundary = math.ceil(count / 2)
    for file_index, (begin, end) in enumerate(((0, boundary), (boundary, count))):
        relative = f"relations/{name}-{file_index:02d}.parquet"
        path = root / relative
        path.parent.mkdir(parents=True, exist_ok=True)
        schema = pa.schema(
            [
                pa.field("src", pa.int64(), nullable=False),
                pa.field("dst", pa.int64(), nullable=False),
                pa.field("edge_id", pa.int64(), nullable=False),
                *(
                    pa.field(field, pa.float32(), nullable=False)
                    for field in EDGE_FIELDS
                ),
            ]
        )
        with pq.ParquetWriter(path, schema, compression="zstd") as writer:
            for start in range(begin, end, RECORD_BATCH_ROWS):
                size = min(RECORD_BATCH_ROWS, end - start)
                ordinal = np.arange(start, start + size, dtype=np.int64)
                source = ordinal % source_count
                destination = (ordinal * 7 + 3) % destination_count
                if reverse:
                    source, destination = destination, source
                edge_ids = edge_id_offset + ordinal
                writer.write_batch(
                    pa.record_batch(
                        [
                            pa.array(source, type=pa.int64()),
                            pa.array(destination, type=pa.int64()),
                            pa.array(edge_ids, type=pa.int64()),
                            *(
                                pa.array(
                                    ((edge_ids + field_index) % 101) / 10.0,
                                    type=pa.float32(),
                                )
                                for field_index in range(EDGE_FIELD_COUNT)
                            ),
                        ],
                        schema=schema,
                    )
                )
        paths.append(relative)
    return tuple(paths)


def _write_splits(root: Path, *, id_column: str, count: int) -> tuple[str, str, str]:
    import pyarrow as pa
    import pyarrow.parquet as pq

    values = (0, count // 2, count - 1)
    paths: list[str] = []
    for phase, identifier in zip(("train", "val", "test"), values, strict=True):
        relative = f"splits/primary-{phase}.parquet"
        path = root / relative
        path.parent.mkdir(parents=True, exist_ok=True)
        pq.write_table(
            pa.table({id_column: pa.array([identifier], type=pa.int64())}),
            path,
        )
        paths.append(relative)
    return paths[0], paths[1], paths[2]


def _write_fixture(root: Path, kind: str) -> int:
    if kind == "homogeneous":
        _write_node_files(
            root,
            node_type="node",
            count=NODE_ROWS,
            id_column="node_id",
        )
        _write_edge_files(
            root,
            name="links",
            count=EDGE_ROWS,
            source_count=NODE_ROWS,
            destination_count=NODE_ROWS,
        )
        _write_splits(root, id_column="node_id", count=NODE_ROWS)
        return NODE_ROWS * FEATURE_WIDTH * 4 + EDGE_ROWS * (
            8 * 3 + 4 * EDGE_FIELD_COUNT
        )

    author_count = NODE_ROWS // 2
    paper_count = NODE_ROWS - author_count
    _write_node_files(
        root,
        node_type="author",
        count=author_count,
        id_column="author_id",
    )
    _write_node_files(
        root,
        node_type="paper",
        count=paper_count,
        id_column="paper_id",
        include_label=False,
    )
    writes_count = EDGE_ROWS // 3
    cites_count = EDGE_ROWS - 2 * writes_count
    _write_edge_files(
        root,
        name="writes",
        count=writes_count,
        source_count=author_count,
        destination_count=paper_count,
    )
    _write_edge_files(
        root,
        name="written_by",
        count=writes_count,
        source_count=author_count,
        destination_count=paper_count,
        reverse=True,
        edge_id_offset=writes_count,
    )
    _write_edge_files(
        root,
        name="cites",
        count=cites_count,
        source_count=paper_count,
        destination_count=paper_count,
        edge_id_offset=2 * writes_count,
    )
    _write_splits(root, id_column="author_id", count=author_count)
    return NODE_ROWS * FEATURE_WIDTH * 4 + EDGE_ROWS * (
        8 * 3 + 4 * EDGE_FIELD_COUNT
    )


def _source(root: Path, kind: str) -> Any:
    from topobench.data.loaders.parquet import (
        IngestionLimits,
        NodeTypeSpec,
        ParquetTypedGraphSource,
        ParquetTypedGraphSpec,
        PartitionSpec,
        RelationSpec,
        SplitRegistrySpec,
        SplitSetSpec,
        SupervisionSpec,
    )

    split = SplitSetSpec(
        tag="primary",
        train="splits/primary-train.parquet",
        val="splits/primary-val.parquet",
        test="splits/primary-test.parquet",
        coverage="partial",
    )
    common = {
        "partition": PartitionSpec(
            strategy="cluster",
            num_partitions=NUM_PARTITIONS,
            memory_limit_bytes=PARTITION_MEMORY_LIMIT_BYTES,
        ),
        "ingestion": IngestionLimits(
            record_batch_rows=RECORD_BATCH_ROWS,
            memory_limit_bytes=DUCKDB_MEMORY_LIMIT_BYTES,
            temp_directory="duckdb-tmp",
        ),
    }
    if kind == "homogeneous":
        spec = ParquetTypedGraphSpec(
            source_root=root,
            output_kind="homogeneous",
            node_types=(
                NodeTypeSpec(
                    name="node",
                    paths=("nodes/node-00.parquet", "nodes/node-01.parquet"),
                    id_column="node_id",
                    id_dtype="int64",
                    feature_columns=("features",),
                    feature_dtype="float32",
                    feature_width=FEATURE_WIDTH,
                    feature_representation="fixed_size_list",
                ),
            ),
            relations=(
                RelationSpec(
                    relation=("node", "links", "node"),
                    paths=(
                        "relations/links-00.parquet",
                        "relations/links-01.parquet",
                    ),
                    source_column="src",
                    destination_column="dst",
                    edge_id_column="edge_id",
                    edge_fields=EDGE_FIELDS,
                ),
            ),
            supervision=SupervisionSpec(
                target_node_type="node",
                label_column="label",
                label_dtype="int64",
                split_registry=SplitRegistrySpec(active_tag="primary", sets=(split,)),
            ),
            **common,
        )
        return ParquetTypedGraphSource(spec)

    author_count = NODE_ROWS // 2
    writes_count = EDGE_ROWS // 3
    spec = ParquetTypedGraphSpec(
        source_root=root,
        output_kind="heterogeneous",
        node_types=(
            NodeTypeSpec(
                name="author",
                paths=("nodes/author-00.parquet", "nodes/author-01.parquet"),
                id_column="author_id",
                id_dtype="int64",
                feature_columns=("features",),
                feature_dtype="float32",
                feature_width=FEATURE_WIDTH,
                feature_representation="fixed_size_list",
            ),
            NodeTypeSpec(
                name="paper",
                paths=("nodes/paper-00.parquet", "nodes/paper-01.parquet"),
                id_column="paper_id",
                id_dtype="int64",
                feature_columns=("features",),
                feature_dtype="float32",
                feature_width=FEATURE_WIDTH,
                feature_representation="fixed_size_list",
            ),
        ),
        relations=(
            RelationSpec(
                relation=("author", "writes", "paper"),
                paths=("relations/writes-00.parquet", "relations/writes-01.parquet"),
                source_column="src",
                destination_column="dst",
                edge_id_column="edge_id",
                edge_fields=EDGE_FIELDS,
            ),
            RelationSpec(
                relation=("paper", "written_by", "author"),
                paths=(
                    "relations/written_by-00.parquet",
                    "relations/written_by-01.parquet",
                ),
                source_column="src",
                destination_column="dst",
                edge_id_column="edge_id",
                edge_fields=EDGE_FIELDS,
            ),
            RelationSpec(
                relation=("paper", "cites", "paper"),
                paths=("relations/cites-00.parquet", "relations/cites-01.parquet"),
                source_column="src",
                destination_column="dst",
                edge_id_column="edge_id",
                edge_fields=EDGE_FIELDS,
            ),
        ),
        supervision=SupervisionSpec(
            target_node_type="author",
            label_column="label",
            label_dtype="int64",
            split_registry=SplitRegistrySpec(active_tag="primary", sets=(split,)),
        ),
        **common,
    )
    assert author_count > 0 and writes_count > 0
    return ParquetTypedGraphSource(spec)


def _resource_metadata(stage_root: Path) -> tuple[dict[str, Any], dict[str, Any]]:
    arrays = json.loads((stage_root / "arrays/arrays.json").read_text(encoding="utf-8"))
    relations = json.loads(
        (stage_root / "relations/relations.json").read_text(encoding="utf-8")
    )
    return arrays["resource_evidence"], relations["resource_evidence"]


def _assert_semantics(store: Any, kind: str) -> dict[str, Any]:
    import numpy as np

    target = "node" if kind == "homogeneous" else "author"
    count = NODE_ROWS if kind == "homogeneous" else NODE_ROWS // 2
    selected_ids = np.array([0, count // 2, count - 1], dtype=np.int64)
    features = store.node_features(target, selected_ids)
    columns = np.arange(FEATURE_WIDTH, dtype=np.float32)
    expected = np.stack(
        [((float(identifier) % 997.0) + (columns % 31.0)) / 997.0 for identifier in selected_ids]
    ).astype(np.float32)
    np.testing.assert_array_equal(features, expected)
    np.testing.assert_array_equal(store.node_labels(target, selected_ids), selected_ids % 4)
    assert store.external_ids(target, selected_ids) == selected_ids.tolist()
    for phase, expected_id in zip(("train", "val", "test"), selected_ids, strict=True):
        np.testing.assert_array_equal(
            store.split_ids("primary", phase),
            np.array([expected_id], dtype=np.int64),
        )
    assert tuple(store.node_types) == (("node",) if kind == "homogeneous" else ("author", "paper"))
    expected_relations = (
        (("node", "links", "node"),)
        if kind == "homogeneous"
        else (
            ("author", "writes", "paper"),
            ("paper", "cites", "paper"),
            ("paper", "written_by", "author"),
        )
    )
    assert tuple(store.relation_types) == expected_relations
    relation_mmap_count = 0
    for relation in store.relation_types:
        row, colptr = store.relation_csc(relation)
        assert isinstance(row, np.memmap)
        assert isinstance(colptr, np.memmap)
        assert row.flags.writeable is False
        relation_edge_count = (
            EDGE_ROWS
            if kind == "homogeneous"
            else (
                EDGE_ROWS - 2 * (EDGE_ROWS // 3)
                if relation[1] == "cites"
                else EDGE_ROWS // 3
            )
        )
        assert len(row) == relation_edge_count
        assert len(colptr) == store._node(relation[2])["count"] + 1
        assert int(colptr[-1]) == relation_edge_count
        for edge_position in (0, 1, relation_edge_count - 1):
            if relation[1] == "writes":
                expected_source = edge_position % (NODE_ROWS // 2)
                expected_destination = (
                    edge_position * 7 + 3
                ) % (NODE_ROWS - NODE_ROWS // 2)
            elif relation[1] == "written_by":
                expected_source = (
                    edge_position * 7 + 3
                ) % (NODE_ROWS - NODE_ROWS // 2)
                expected_destination = edge_position % (NODE_ROWS // 2)
            elif kind == "heterogeneous":
                expected_source = edge_position % (NODE_ROWS - NODE_ROWS // 2)
                expected_destination = (
                    edge_position * 7 + 3
                ) % (NODE_ROWS - NODE_ROWS // 2)
            else:
                expected_source = edge_position % NODE_ROWS
                expected_destination = (edge_position * 7 + 3) % NODE_ROWS
            start = int(colptr[expected_destination])
            stop = int(colptr[expected_destination + 1])
            assert stop > start
            assert expected_source in row[start:stop]
            edge_id_offset = (
                0
                if kind == "homogeneous" or relation[1] == "writes"
                else (
                    EDGE_ROWS // 3
                    if relation[1] == "written_by"
                    else 2 * (EDGE_ROWS // 3)
                )
            )
            expected_edge_id = edge_id_offset + edge_position
            positions = np.arange(start, stop, dtype=np.int64)
            edge_ids = store.relation_field(relation, "edge_id", positions)
            matching = np.flatnonzero(edge_ids == expected_edge_id)
            assert len(matching) == 1
            csc_position = positions[int(matching[0])]
            for field_index, field in enumerate(EDGE_FIELDS):
                observed = store.relation_field(
                    relation,
                    field,
                    np.array([csc_position], dtype=np.int64),
                )
                expected_field = np.float32(
                    ((expected_edge_id + field_index) % 101) / 10.0
                )
                np.testing.assert_array_equal(
                    observed,
                    np.array([expected_field], dtype=np.float32),
                )
        relation_mmap_count += 2
    assert features.flags.writeable is False
    assert all(not Path(path).is_absolute() for path in store.mapped_paths)
    return {
        "selected_node_rows": len(selected_ids),
        "selected_feature_values": int(features.size),
        "node_type_count": len(store.node_types),
        "relation_count": len(store.relation_types),
        "active_split_tag_count": 1,
        "relation_mmap_count": relation_mmap_count,
    }

def _plain_evidence(value: Any) -> Any:
    if isinstance(value, Mapping):
        return {
            str(key): _plain_evidence(item)
            for key, item in sorted(value.items(), key=lambda item: str(item[0]))
        }
    if isinstance(value, (tuple, list)):
        return [_plain_evidence(item) for item in value]
    return value


def _worker(root: Path, kind: str, logical_payload_bytes: int) -> None:
    from topobench.data.stores.pyg_partitioner import (
        TopologyOnlyPyGPartitioner,
    )
    from topobench.data.stores.typed_graph_ingestion import (
        ParquetTypedGraphIngestor,
    )
    from topobench.data.stores.typed_graph_store import TypedGraphStoreWriter
    from topobench.data.stores.typed_partition_book import (
        PartitionQualificationLimits,
    )

    class _MeasuredStoreWriter(TypedGraphStoreWriter):
        """Expose bounded aggregate RSS at existing publication boundaries."""

        def __init__(self, *args: Any, phase_prefix: str) -> None:
            super().__init__(*args)
            self._phase_prefix = phase_prefix

        def _record_phase(self, phase: str) -> None:
            phase_peak_rss[f"{self._phase_prefix}_{phase}"] = _rss_bytes()

        def _recover_partition_build(self) -> Any:
            recovered = super()._recover_partition_build()
            self._record_phase("recovery")
            return recovered

        def _reopen_task1_6_locked(self) -> dict[str, Any]:
            reopened = super()._reopen_task1_6_locked()
            self._record_phase("reopen")
            return reopened

        def _materialize_candidate(
            self,
            candidate: Path,
            reopened: Mapping[str, Any],
        ) -> None:
            super()._materialize_candidate(candidate, reopened)
            self._record_phase("candidate_materialized")

        def _promote_candidate(self, candidate: Path) -> Any:
            promoted = super()._promote_candidate(candidate)
            self._record_phase("candidate_promoted")
            return promoted

    source = _source(root / "source", kind)
    baseline_rss = _rss_bytes()
    phase_peak_rss = {"baseline": baseline_rss}
    ingestor = ParquetTypedGraphIngestor(source, root / "stores", threads=1)
    relations = ingestor.build_relations()
    phase_peak_rss["relations"] = _rss_bytes()
    partition_build = TopologyOnlyPyGPartitioner(ingestor, relations).build(
        PartitionQualificationLimits()
    )
    phase_peak_rss["partition_adapter"] = _rss_bytes()
    del relations
    first = _MeasuredStoreWriter(
        ingestor,
        partition_build,
        phase_prefix="first_store",
    ).build()
    phase_peak_rss["first_store_publish"] = _rss_bytes()
    first.store.close()
    replay_partition = ingestor.build_partitions(
        limits=PartitionQualificationLimits()
    )
    phase_peak_rss["partition_replay"] = _rss_bytes()
    replay = _MeasuredStoreWriter(
        ingestor,
        replay_partition,
        phase_prefix="replay_store",
    ).build()
    phase_peak_rss["store_replay"] = _rss_bytes()
    assert replay.cache_hit is True
    assert replay.content_sha256 == first.content_sha256
    assert replay.path == first.path
    with replay.store as store:
        semantic = _assert_semantics(store, kind)
        mapped_file_count = len(store.mapped_paths)
    phase_peak_rss["semantic_reads"] = _rss_bytes()
    arrays_resource, relations_resource = _resource_metadata(partition_build.stage_root)
    assert arrays_resource["duckdb_memory_limit_bytes"] == DUCKDB_MEMORY_LIMIT_BYTES
    assert relations_resource["duckdb_memory_limit_bytes"] == DUCKDB_MEMORY_LIMIT_BYTES
    assert arrays_resource["max_record_batch_rows"] <= RECORD_BATCH_ROWS
    assert relations_resource["max_record_batch_rows"] <= RECORD_BATCH_ROWS
    assert arrays_resource["snapshot_persisted"] is False
    assert relations_resource["snapshot_persisted"] is False
    assert arrays_resource["snapshot_bytes_accounted"] is True
    assert relations_resource["snapshot_bytes_accounted"] is True
    assert relations_resource["spill_subtree"].startswith("ephemeral/")
    assert relations_resource["exact_disk_requirements"]
    assert "record_batch_rows" in arrays_resource["bounded_memory"]
    assert "record_batch_rows" in relations_resource["bounded_memory"]
    measured_partition = dict(partition_build.book.measured_resources)
    estimated_partition = dict(partition_build.book.estimated_resources)
    assert measured_partition["measurement_scope"] == "isolated-worker"
    assert measured_partition["peak_rss_bytes"] <= estimated_partition["peak_memory_bytes"]
    assert measured_partition["peak_rss_bytes"] <= PARTITION_MEMORY_LIMIT_BYTES
    assert measured_partition["temporary_disk_bytes"] <= estimated_partition["temporary_disk_bytes"]
    store_validation = {
        "first_store": _plain_evidence(first.validation_evidence),
        "replay_store": _plain_evidence(replay.validation_evidence),
    }
    for build_evidence in store_validation.values():
        for validation in build_evidence.values():
            memory = validation["memory"]
            assert validation["status"] == "passed"
            assert (
                memory["declared_peak_rss_delta_limit_bytes"]
                == VALIDATION_WORKER_RSS_DELTA_LIMIT_BYTES
            )
            assert (
                memory["peak_rss_delta_bytes"]
                <= VALIDATION_WORKER_RSS_DELTA_LIMIT_BYTES
            )
    peak_rss = _rss_bytes()
    evidence = {
        "schema_version": "typed-graph-rss-qualification-v1",
        "kind": kind,
        "thresholds": {
            "rss_delta_limit_bytes": RSS_DELTA_LIMIT_BYTES,
            "payload_to_rss_factor": PAYLOAD_TO_RSS_FACTOR,
            "duckdb_memory_limit_bytes": DUCKDB_MEMORY_LIMIT_BYTES,
            "partition_memory_limit_bytes": PARTITION_MEMORY_LIMIT_BYTES,
            "record_batch_rows": RECORD_BATCH_ROWS,
            "validation_worker_rss_delta_limit_bytes": (
                VALIDATION_WORKER_RSS_DELTA_LIMIT_BYTES
            ),
        },
        "fixture": {
            "node_rows": NODE_ROWS,
            "edge_rows": EDGE_ROWS,
            "feature_width": FEATURE_WIDTH,
            "edge_field_count": EDGE_FIELD_COUNT,
            "logical_feature_and_mapped_edge_bytes": logical_payload_bytes,
            "source_disk_bytes": _directory_bytes(root / "source"),
        },
        "memory": {
            "baseline_rss_bytes": baseline_rss,
            "peak_rss_bytes": peak_rss,
            "rss_delta_bytes": peak_rss - baseline_rss,
            "partition_estimated_peak_rss_bytes": estimated_partition["peak_memory_bytes"],
            "partition_measured_peak_rss_bytes": measured_partition["peak_rss_bytes"],
            "partition_measurement": _plain_evidence(measured_partition),
            "peak_rss_by_phase": {
                "conversion_and_store": peak_rss,
                "isolated_partition": measured_partition["peak_rss_bytes"],
            },
            "peak_rss_delta_by_phase": {
                phase: observed - baseline_rss
                for phase, observed in phase_peak_rss.items()
            },
            "isolated_store_validation": store_validation,
        },
        "disk": {
            "inventory_estimated_final_bytes": partition_build.inventory.estimated_final_bytes,
            "inventory_estimated_temporary_bytes": partition_build.inventory.estimated_temporary_bytes,
            "partition_estimated_temporary_bytes": estimated_partition["temporary_disk_bytes"],
            "partition_measured_temporary_bytes": measured_partition["temporary_disk_bytes"],
            "final_store_bytes": _directory_bytes(replay.path),
            "retained_stage_bytes": _directory_bytes(partition_build.stage_root),
        },
        "cache": {
            "content_sha256": replay.content_sha256,
            "validated_cache_hit": replay.cache_hit,
            "partition_book_identity": replay_partition.book.content_identity,
        },
        "bounded_execution": {
            "arrays": arrays_resource,
            "relations": relations_resource,
            "mapped_runtime_file_count": mapped_file_count,
            "ram_wide_feature_or_edge_table_excluded": (
                logical_payload_bytes > RSS_DELTA_LIMIT_BYTES
                and peak_rss - baseline_rss < RSS_DELTA_LIMIT_BYTES
                and semantic["relation_mmap_count"] >= 2
            ),
        },
        "provenance_input": {
            "store_fingerprint": replay.content_sha256,
            "source_fingerprint": partition_build.inventory.source_fingerprint,
            "partition_book_identity": replay_partition.book.content_identity,
            "schema_roles": {
                "output_kind": source.spec.output_kind,
                "target_node_type": source.spec.supervision.target_node_type,
                "node_type_count": len(source.spec.node_types),
                "relation_count": len(source.spec.relations),
                "active_split_tag": (
                    source.spec.supervision.split_registry.active_tag
                ),
            },
            "representation": "typed-csc-mmap",
            "strategy_state": {
                "strategy": source.spec.partition.strategy,
                "backend": partition_build.book.backend,
                "backend_version": partition_build.book.backend_version,
                "num_partitions": partition_build.book.num_partitions,
                "options": dict(partition_build.book.options),
            },
            "memory_summary": {
                "rss_delta_bytes": peak_rss - baseline_rss,
                "partition_peak_rss_bytes": (
                    measured_partition["peak_rss_bytes"]
                ),
            },
            "disk_summary": {
                "temporary_peak_bytes": (
                    measured_partition["temporary_disk_bytes"]
                ),
                "final_store_bytes": _directory_bytes(replay.path),
            },
        },
        "semantics": semantic,
        "status": "passed",
    }
    print(json.dumps(evidence, sort_keys=True))


def _bounded_worker_stderr(stderr: str) -> str:
    """Return useful bounded diagnostics without fixture external identifiers."""
    redacted_lines: list[str] = []
    for line in stderr.splitlines():
        if (
            "Out of Memory Error:" in line
            or "PARTITION-MEMORY-" in line
        ):
            safe_line = line
        else:
            safe_line = re.sub(
                r"(?<![\w.])[-+]?\d+(?:\.\d+)?(?![\w.])",
                "<number>",
                line,
            )
        redacted_lines.append(safe_line.replace(str(Path.home()), "$HOME"))
    redacted = "\n".join(redacted_lines)
    redacted = "".join(
        character
        for character in redacted
        if character in "\n\t" or ord(character) >= 32
    )
    return redacted[-8192:].strip() or "<no worker stderr>"


def _run_one(root: Path, kind: str, logical_payload_bytes: int) -> dict[str, Any]:
    completed = subprocess.run(
        [
            sys.executable,
            __file__,
            "--worker",
            str(root),
            kind,
            str(logical_payload_bytes),
        ],
        check=False,
        capture_output=True,
        text=True,
        timeout=WORKER_TIMEOUT_SECONDS,
    )
    if completed.returncode:
        raise SystemExit(
            f"{kind} RSS worker failed with exit {completed.returncode}:\n"
            f"{_bounded_worker_stderr(completed.stderr)}"
        )
    evidence = json.loads(completed.stdout.strip().splitlines()[-1])
    minimum_payload = math.ceil(RSS_DELTA_LIMIT_BYTES * PAYLOAD_TO_RSS_FACTOR)
    observed_payload = evidence["fixture"]["logical_feature_and_mapped_edge_bytes"]
    if observed_payload <= minimum_payload:
        raise SystemExit(
            "RSS fixture is too small: logical mapped-edge plus feature payload "
            f"{observed_payload} must exceed predeclared {minimum_payload}"
        )
    observed_rss = evidence["memory"]["rss_delta_bytes"]
    if observed_rss >= RSS_DELTA_LIMIT_BYTES:
        phase_deltas = evidence["memory"]["peak_rss_delta_by_phase"]
        partition_measurement = evidence["memory"]["partition_measurement"]
        raise SystemExit(
            f"{kind} RSS qualification failed: delta {observed_rss} >= "
            f"{RSS_DELTA_LIMIT_BYTES}; phase_deltas="
            f"{json.dumps(phase_deltas, sort_keys=True)}; "
            f"partition_measurement="
            f"{json.dumps(partition_measurement, sort_keys=True)}"
        )
    if evidence["bounded_execution"]["mapped_runtime_file_count"] < 1:
        raise SystemExit(f"{kind} qualification observed no selected runtime mmap read")
    if not evidence["bounded_execution"]["ram_wide_feature_or_edge_table_excluded"]:
        raise SystemExit(f"{kind} qualification did not prove bounded mmap execution")
    return evidence


def main() -> int:
    if len(sys.argv) == 5 and sys.argv[1] == "--worker":
        _worker(Path(sys.argv[2]), sys.argv[3], int(sys.argv[4]))
        return 0
    if len(sys.argv) > 1:
        raise SystemExit(
            "usage: qualify_typed_graph_rss.py (worker mode is internal)"
        )
    if NODE_ROWS < 4 or EDGE_ROWS < 4 or FEATURE_WIDTH < 1 or EDGE_FIELD_COUNT < 1:
        raise SystemExit("RSS fixture dimensions must be positive and non-trivial")
    if not 1.0 < PAYLOAD_TO_RSS_FACTOR <= 4.0:
        raise SystemExit("payload-to-RSS factor must be predeclared above one")
    aggregate: dict[str, Any] = {
        "schema_version": "typed-graph-rss-qualification-aggregate-v1",
        "thresholds": {
            "rss_delta_limit_bytes": RSS_DELTA_LIMIT_BYTES,
            "payload_to_rss_factor": PAYLOAD_TO_RSS_FACTOR,
            "duckdb_memory_limit_bytes": DUCKDB_MEMORY_LIMIT_BYTES,
            "partition_memory_limit_bytes": PARTITION_MEMORY_LIMIT_BYTES,
        },
        "runs": [],
    }
    with tempfile.TemporaryDirectory(prefix="topobench-typed-rss-") as directory:
        root = Path(directory)
        for kind in ("homogeneous", "heterogeneous"):
            run_root = root / kind
            logical_payload = _write_fixture(run_root / "source", kind)
            aggregate["runs"].append(_run_one(run_root, kind, logical_payload))
    aggregate["status"] = "passed"
    evidence_root = Path(
        os.environ.get("TOPOBENCH_QUALIFICATION_EVIDENCE_DIR", "qualification-evidence")
    )
    evidence_root.mkdir(parents=True, exist_ok=True)
    evidence_path = evidence_root / "typed-graph-rss-qualification.json"
    evidence_path.write_text(
        json.dumps(aggregate, indent=2, sort_keys=True) + "\n",
        encoding="utf-8",
    )
    print(json.dumps(aggregate, sort_keys=True))
    return 0


if __name__ == "__main__":
    raise SystemExit(main())
