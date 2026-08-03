"""Exact directed CSC round trips for typed Parquet relations."""

from __future__ import annotations

import json
from types import SimpleNamespace
from pathlib import Path
from typing import Any
import duckdb

import numpy as np
import pyarrow as pa
import pyarrow.parquet as pq

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
from topobench.data.stores.typed_graph_ingestion import ParquetTypedGraphIngestor
from topobench.data.stores import typed_graph_csc as csc_module
from topobench.data.stores.typed_graph_csc import TypedGraphRelationWriter


def _write_table(path: Path, columns: dict[str, pa.Array]) -> None:
    path.parent.mkdir(parents=True, exist_ok=True)
    pq.write_table(pa.table(columns), path)


def _split(root: Path, ids: tuple[str, ...] = ("alpha", "mu", "zeta")) -> SplitRegistrySpec:
    paths: dict[str, str] = {}
    for phase, node_id in zip(("train", "val", "test"), ids, strict=True):
        relative = f"splits/{phase}.parquet"
        _write_table(
            root / relative,
            {"author_id": pa.array([node_id], type=pa.string())},
        )
        paths[phase] = relative
    return SplitRegistrySpec(
        active_tag="default",
        sets=(
            SplitSetSpec(
                tag="default",
                train=paths["train"],
                val=paths["val"],
                test=paths["test"],
                coverage="complete",
            ),
        ),
    )


def _heterogeneous_source(
    root: Path,
    *,
    edge_layout: tuple[tuple[str, tuple[int, ...]], ...] = (
        ("relations/writes-b.parquet", (4, 0, 2)),
        ("relations/writes-a.parquet", (1, 3)),
    ),
    record_batch_rows: int = 2,
) -> ParquetTypedGraphSource:
    author_ids = ["zeta", "alpha", "mu"]
    _write_table(
        root / "nodes/authors.parquet",
        {
            "author_id": pa.array(author_ids, type=pa.string()),
            "feature": pa.array([9.0, 1.0, 5.0], type=pa.float32()),
            "label": pa.array([2, 0, 1], type=pa.int64()),
        },
    )
    paper_ids = [2**64 - 1, 0, 9, 7]
    _write_table(
        root / "nodes/papers.parquet",
        {
            "paper_id": pa.array(paper_ids, type=pa.uint64()),
            "feature": pa.array([4.0, 1.0, 3.0, 2.0], type=pa.float32()),
        },
    )

    source_ids = ["alpha", "alpha", "mu", "zeta", "mu"]
    destination_ids = [0, 0, 7, 7, 2**64 - 1]
    edge_ids = [10, 11, 30, 40, 50]
    weights = [1.0, 1.5, 3.0, 4.0, 5.0]
    attributes = [[10, -10], [11, -11], [30, -30], [40, -40], [50, -50]]
    for relative, indexes in edge_layout:
        _write_table(
            root / relative,
            {
                "writer": pa.array([source_ids[index] for index in indexes], type=pa.string()),
                "paper": pa.array([destination_ids[index] for index in indexes], type=pa.uint64()),
                "stable_id": pa.array([edge_ids[index] for index in indexes], type=pa.uint64()),
                "weight": pa.array([weights[index] for index in indexes], type=pa.float32()),
                "attribute": pa.array(
                    [attributes[index] for index in indexes],
                    type=pa.list_(pa.int16(), 2),
                ),
            },
        )
    _write_table(
        root / "relations/written-by.parquet",
        {
            "paper": pa.array([7, 0], type=pa.uint64()),
            "writer": pa.array(["zeta", "alpha"], type=pa.string()),
            "stable_id": pa.array([201, 200], type=pa.uint64()),
        },
    )
    split_registry = _split(root)
    spec = ParquetTypedGraphSpec(
        source_root=root,
        output_kind="heterogeneous",
        node_types=(
            NodeTypeSpec(
                name="paper",
                paths=("nodes/papers.parquet",),
                id_column="paper_id",
                id_dtype="uint64",
                feature_columns=("feature",),
                feature_dtype="float32",
                feature_width=1,
            ),
            NodeTypeSpec(
                name="author",
                paths=("nodes/authors.parquet",),
                id_column="author_id",
                id_dtype="string",
                feature_columns=("feature",),
                feature_dtype="float32",
                feature_width=1,
            ),
        ),
        relations=(
            RelationSpec(
                relation=("author", "writes", "paper"),
                paths=tuple(relative for relative, _ in edge_layout),
                source_column="writer",
                destination_column="paper",
                edge_id_column="stable_id",
                edge_fields=("weight", "attribute"),
            ),
            RelationSpec(
                relation=("paper", "written_by", "author"),
                paths=("relations/written-by.parquet",),
                source_column="paper",
                destination_column="writer",
                edge_id_column="stable_id",
            ),
        ),
        supervision=SupervisionSpec(
            target_node_type="author",
            label_column="label",
            label_dtype="int64",
            split_registry=split_registry,
        ),
        partition=PartitionSpec(strategy="cluster"),
        ingestion=IngestionLimits(
            record_batch_rows=record_batch_rows,
            memory_limit_bytes=64 * 1024**2,
            temp_directory="duckdb-tmp",
        ),
    )
    return ParquetTypedGraphSource(spec)


def _homogeneous_source(root: Path) -> ParquetTypedGraphSource:
    _write_table(
        root / "nodes/part.parquet",
        {
            "node_id": pa.array([100, -5, 9], type=pa.int64()),
            "feature": pa.array([10.0, -0.5, 0.9], type=pa.float32()),
            "label": pa.array([2, 0, 1], type=pa.int64()),
        },
    )
    _write_table(
        root / "relations/part.parquet",
        {
            "src": pa.array([9, 100, -5, 100], type=pa.int64()),
            "dst": pa.array([9, -5, 100, 100], type=pa.int64()),
        },
    )
    for phase, value in (("train", -5), ("val", 9), ("test", 100)):
        _write_table(
            root / f"splits/{phase}.parquet",
            {"node_id": pa.array([value], type=pa.int64())},
        )
    split = SplitSetSpec(
        tag="default",
        train="splits/train.parquet",
        val="splits/val.parquet",
        test="splits/test.parquet",
        coverage="complete",
    )
    return ParquetTypedGraphSource(
        ParquetTypedGraphSpec(
            source_root=root,
            output_kind="homogeneous",
            node_types=(
                NodeTypeSpec(
                    name="node",
                    paths=("nodes/part.parquet",),
                    id_column="node_id",
                    id_dtype="int64",
                    feature_columns=("feature",),
                    feature_dtype="float32",
                    feature_width=1,
                ),
            ),
            relations=(
                RelationSpec(
                    relation=("node", "links", "node"),
                    paths=("relations/part.parquet",),
                    source_column="src",
                    destination_column="dst",
                ),
            ),
            supervision=SupervisionSpec(
                target_node_type="node",
                label_column="label",
                label_dtype="int64",
                split_registry=SplitRegistrySpec(active_tag="default", sets=(split,)),
            ),
            partition=PartitionSpec(strategy="cluster"),
            ingestion=IngestionLimits(
                record_batch_rows=2,
                memory_limit_bytes=64 * 1024**2,
                temp_directory="duckdb-tmp",
            ),
        )
    )



def test_relation_writer_canonicalizes_an_unordered_bounded_stream(
    tmp_path: Path,
) -> None:
    connection = duckdb.connect()
    connection.execute(
        "CREATE TABLE mapped(destination_local BIGINT, source_local BIGINT, "
        "edge_id UBIGINT, weight FLOAT)"
    )
    connection.executemany(
        "INSERT INTO mapped VALUES (?, ?, ?, ?)",
        [
            (2, 1, 50, 5.0),
            (0, 2, 20, 2.0),
            (1, 0, 30, 3.0),
            (0, 1, 10, 1.0),
            (2, 0, 40, 4.0),
        ],
    )
    ingestor = SimpleNamespace(
        source=SimpleNamespace(
            spec=SimpleNamespace(
                ingestion=SimpleNamespace(record_batch_rows=2),
            )
        )
    )
    writer = TypedGraphRelationWriter(
        ingestor,
        SimpleNamespace(),
        SimpleNamespace(),
    )
    relation = SimpleNamespace(
        relation=("source", "links", "destination"),
        source_column="source_id",
        destination_column="destination_id",
        edge_id_column="edge_id",
    )
    context = {
        "stream_query": (
            "SELECT destination_local, source_local, edge_id, weight FROM mapped"
        ),
        "edge_count": 5,
        "source_count": 3,
        "destination_count": 3,
        "source_internal_key": "n0000",
        "destination_internal_key": "n0001",
        "source_id_dtype": "int64",
        "destination_id_dtype": "int64",
        "edge_descriptor": {
            "dtype": "uint64",
            "storage_dtype": "<u8",
            "arrow_type": "uint64",
            "representation": "scalar",
            "value_shape": (),
        },
        "field_descriptors": {
            "weight": {
                "dtype": "float32",
                "storage_dtype": "<f4",
                "arrow_type": "float",
                "representation": "scalar",
                "value_shape": (),
            }
        },
        "source_files": [],
        "source_sha256": "source-sha256",
        "schema_record": {"schema_fingerprint": "schema-fingerprint"},
    }
    try:
        record, _ = writer._write_one_relation(
            tmp_path,
            relation=relation,
            internal_key="r0000",
            context=context,
            connection=connection,
        )
    finally:
        connection.close()

    np.testing.assert_array_equal(
        np.load(tmp_path / record["colptr"]["relative_path"]),
        np.array([0, 2, 3, 5], dtype="<i8"),
    )
    np.testing.assert_array_equal(
        np.load(tmp_path / record["row"]["relative_path"]),
        np.array([1, 2, 0, 0, 1], dtype="<i8"),
    )
    np.testing.assert_array_equal(
        np.load(tmp_path / record["edge_id"]["relative_path"]),
        np.array([10, 20, 30, 40, 50], dtype="<u8"),
    )
    np.testing.assert_array_equal(
        np.load(tmp_path / record["fields"]["weight"]["relative_path"]),
        np.array([1, 2, 3, 4, 5], dtype="<f4"),
    )

def _metadata(result: Any) -> dict[str, Any]:
    return json.loads(
        (result.artifact_root / "relations.json").read_text(encoding="utf-8")
    )


def _relation_arrays(result: Any, key: str) -> dict[str, np.ndarray]:
    record = _metadata(result)["relations"][key]
    arrays: dict[str, np.ndarray] = {
        "colptr": np.load(
            result.artifact_root / record["colptr"]["relative_path"],
            mmap_mode="r",
            allow_pickle=False,
        ),
        "row": np.load(
            result.artifact_root / record["row"]["relative_path"],
            mmap_mode="r",
            allow_pickle=False,
        ),
    }
    if record["edge_id"] is not None:
        arrays["edge_id"] = np.load(
            result.artifact_root / record["edge_id"]["relative_path"],
            mmap_mode="r",
            allow_pickle=False,
        )
    for field_name, field in record["fields"].items():
        arrays[field_name] = np.load(
            result.artifact_root / field["relative_path"],
            mmap_mode="r",
            allow_pickle=False,
        )
    return arrays


def test_relation_build_closes_internal_validation_mappings(
    tmp_path: Path,
    monkeypatch: Any,
) -> None:
    source = _heterogeneous_source(tmp_path / "source")
    opened: list[np.memmap] = []
    original_load = np.load

    def tracked_load(*args: Any, **kwargs: Any) -> Any:
        array = original_load(*args, **kwargs)
        path = Path(args[0])
        if (
            isinstance(array, np.memmap)
            and "relations" in path.parts
        ):
            opened.append(array)
        return array

    monkeypatch.setattr(csc_module.np, "load", tracked_load)
    result = ParquetTypedGraphIngestor(
        source,
        tmp_path / "stores",
    ).build_relations()

    assert opened
    unclosed = [
        str(array.filename)
        for array in opened
        if not array._mmap.closed
    ]
    assert not unclosed, unclosed
    public = original_load(
        result.artifact_root / "r0000/row.npy",
        mmap_mode="r",
        allow_pickle=False,
    )
    assert public.flags.writeable is False
    assert public._mmap.closed is False
    public._mmap.close()


def test_writes_exact_heterogeneous_directed_relations_and_aligned_fields(
    tmp_path: Path,
) -> None:
    source = _heterogeneous_source(tmp_path / "source")
    result = ParquetTypedGraphIngestor(source, tmp_path / "stores").build_relations()
    metadata = _metadata(result)

    assert result.artifact_root == result.stage_root / "relations"
    assert list(metadata["relations"]) == ["r0000", "r0001"]
    writes = metadata["relations"]["r0000"]
    reverse = metadata["relations"]["r0001"]
    assert writes["relation"] == ["author", "writes", "paper"]
    assert writes["source_node_type"] == "author"
    assert writes["destination_node_type"] == "paper"
    assert writes["canonical_order"] == [
        "destination_local",
        "source_local",
        "stable_id",
    ]

    arrays = _relation_arrays(result, "r0000")
    np.testing.assert_array_equal(arrays["colptr"], np.array([0, 2, 4, 4, 5], dtype="<i8"))
    np.testing.assert_array_equal(arrays["row"], np.array([0, 0, 1, 2, 1], dtype="<i8"))
    np.testing.assert_array_equal(arrays["edge_id"], np.array([10, 11, 30, 40, 50], dtype="<u8"))
    np.testing.assert_array_equal(arrays["weight"], np.array([1.0, 1.5, 3.0, 4.0, 5.0], dtype="<f4"))
    np.testing.assert_array_equal(
        arrays["attribute"],
        np.array([[10, -10], [11, -11], [30, -30], [40, -40], [50, -50]], dtype="<i2"),
    )
    assert arrays["colptr"].dtype.str == "<i8"
    assert arrays["row"].dtype.str == "<i8"
    assert arrays["edge_id"].dtype.str == "<u8"
    assert arrays["weight"].dtype.str == "<f4"
    assert arrays["attribute"].shape == (5, 2)
    assert writes["fields"]["weight"]["representation"] == "scalar"
    assert writes["fields"]["attribute"]["representation"] == "fixed_size_list"

    reverse_arrays = _relation_arrays(result, "r0001")
    assert reverse["relation"] == ["paper", "written_by", "author"]
    np.testing.assert_array_equal(reverse_arrays["colptr"], np.array([0, 1, 1, 2], dtype="<i8"))
    np.testing.assert_array_equal(reverse_arrays["row"], np.array([0, 1], dtype="<i8"))
    np.testing.assert_array_equal(reverse_arrays["edge_id"], np.array([200, 201], dtype="<u8"))
    assert writes["edge_count"] == 5
    assert reverse["edge_count"] == 2
    assert metadata["max_record_batch_rows"] <= 2
    resource = metadata["resource_evidence"]
    assert resource["duckdb_memory_limit_bytes"] == 64 * 1024**2
    assert resource["snapshot_persisted"] is False
    assert resource["bounded_memory"].startswith("O(record_batch_rows")

    expected_files = {
        "relations.json",
        "relations.complete.json",
        "r0000/colptr.npy",
        "r0000/row.npy",
        "r0000/edge_id.npy",
        "r0000/fields/f0000.npy",
        "r0000/fields/f0001.npy",
        "r0001/colptr.npy",
        "r0001/row.npy",
        "r0001/edge_id.npy",
    }
    assert {
        path.relative_to(result.artifact_root).as_posix()
        for path in result.artifact_root.rglob("*")
        if path.is_file()
    } == expected_files


def test_homogeneous_csc_preserves_direction_self_loops_and_boundary_ordinals(
    tmp_path: Path,
) -> None:
    source = _homogeneous_source(tmp_path / "source")
    result = ParquetTypedGraphIngestor(source, tmp_path / "stores").build_relations()
    arrays = _relation_arrays(result, "r0000")

    # Canonical IDs are [-5, 9, 100].  Edges are 100->-5, 9->9,
    # -5->100 and 100->100.  The reverse  -5->100 is not inferred.
    np.testing.assert_array_equal(arrays["colptr"], np.array([0, 1, 2, 4], dtype="<i8"))
    np.testing.assert_array_equal(arrays["row"], np.array([2, 1, 0, 2], dtype="<i8"))
    assert "edge_id" not in arrays
    assert int(arrays["row"].min()) == 0
    assert int(arrays["row"].max()) == 2


def test_relation_semantic_digest_is_fragment_and_batch_layout_independent(
    tmp_path: Path,
) -> None:
    fragmented = _heterogeneous_source(tmp_path / "fragmented", record_batch_rows=2)
    consolidated = _heterogeneous_source(
        tmp_path / "consolidated",
        edge_layout=(("relations/all.parquet", (3, 1, 4, 0, 2)),),
        record_batch_rows=3,
    )

    first = ParquetTypedGraphIngestor(fragmented, tmp_path / "store-a").build_relations()
    second = ParquetTypedGraphIngestor(consolidated, tmp_path / "store-b").build_relations()

    assert first.inventory.source_fingerprint != second.inventory.source_fingerprint
    assert first.content_sha256 == second.content_sha256
    assert _metadata(first)["relations"]["r0000"]["semantic_sha256"] == _metadata(second)["relations"]["r0000"]["semantic_sha256"]
    for key in ("r0000", "r0001"):
        left = _relation_arrays(first, key)
        right = _relation_arrays(second, key)
        assert set(left) == set(right)
        for name in left:
            np.testing.assert_array_equal(left[name], right[name])


def test_semantic_digest_uses_canonical_framed_field_component_order(
    tmp_path: Path,
) -> None:
    from topobench.data.stores import typed_graph_csc as csc_module

    source = _heterogeneous_source(tmp_path / "source")
    assert source.spec.relations[0].edge_fields == ("attribute", "weight")
    ingestor = ParquetTypedGraphIngestor(source, tmp_path / "stores")
    result = ingestor.build_relations()
    record = _metadata(result)["relations"]["r0000"]
    reordered = json.loads(json.dumps(record))
    reordered["fields"] = {
        name: reordered["fields"][name]
        for name in reversed(tuple(reordered["fields"]))
    }

    assert csc_module._relation_semantic_sha256(
        result.artifact_root,
        reordered,
        result.record_batch_rows,
    ) == record["semantic_sha256"]
    reopened = ingestor.build_relations()
    assert _metadata(reopened)["relations"]["r0000"][
        "semantic_sha256"
    ] == record["semantic_sha256"]
    reopened_arrays = _relation_arrays(reopened, "r0000")
    np.testing.assert_array_equal(
        reopened_arrays["attribute"],
        np.array(
            [[10, -10], [11, -11], [30, -30], [40, -40], [50, -50]],
            dtype="<i2",
        ),
    )
    np.testing.assert_array_equal(
        reopened_arrays["weight"],
        np.array([1.0, 1.5, 3.0, 4.0, 5.0], dtype="<f4"),
    )


def test_relation_build_automatically_validates_and_binds_task3_arrays(
    tmp_path: Path,
) -> None:
    source = _heterogeneous_source(tmp_path / "source")
    ingestor = ParquetTypedGraphIngestor(source, tmp_path / "stores")
    index_build = ingestor.build()
    assert not (index_build.stage_root / "arrays").exists()

    result = ingestor.build_relations(index_build)

    arrays_root = index_build.stage_root / "arrays"
    assert (arrays_root / "arrays.complete.json").is_file()
    arrays_metadata = json.loads(
        (arrays_root / "arrays.json").read_text(encoding="utf-8")
    )
    relation_metadata = _metadata(result)
    binding = relation_metadata["array_binding"]
    assert binding["content_sha256"] == arrays_metadata["content_sha256"]
    assert binding["content_identity"] == arrays_metadata["content_identity"]
    assert binding["active_split_tag"] == "default"
    assert binding["input_fingerprint"] == result.inventory.source_fingerprint
    assert binding["index_bindings_sha256"] == relation_metadata[
        "index_bindings_sha256"
    ]
    assert len(binding["completion_sha256"]) == 64
    assert len(binding["metadata_sha256"]) == 64
    assert len(binding["source_schema_sha256"]) == 64
    completion = json.loads(
        (result.artifact_root / "relations.complete.json").read_text(
            encoding="utf-8"
        )
    )
    assert completion["array_binding"] == binding


def test_completed_relation_stage_reopens_with_full_binding_evidence(
    tmp_path: Path,
) -> None:
    source = _heterogeneous_source(tmp_path / "source")
    ingestor = ParquetTypedGraphIngestor(source, tmp_path / "stores")
    first = ingestor.build_relations()
    second = ingestor.build_relations()

    assert first.resumed is False
    assert second.resumed is True
    assert second.content_sha256 == first.content_sha256
    completion = json.loads(
        (first.artifact_root / "relations.complete.json").read_text(encoding="utf-8")
    )
    assert completion["reopened_and_validated"] is True
    assert completion["input_fingerprint"] == first.inventory.source_fingerprint
    assert completion["config_fingerprint"] == first.inventory.config_fingerprint
    assert completion["index_bindings"] == _metadata(first)["index_bindings"]
    assert completion["source_schema_sha256"] == _metadata(first)["source_schema_sha256"]
    assert completion["outputs"]
