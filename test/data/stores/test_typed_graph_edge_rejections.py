"""Hard rejection and recovery tests for typed relation CSC artifacts."""

from __future__ import annotations

from dataclasses import replace
import hashlib
import json
import os
from pathlib import Path
from typing import Any

import numpy as np
import pyarrow as pa
import pyarrow.parquet as pq
import pytest

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
import topobench.data.stores.typed_graph_ingestion as ingestion_module
from topobench.data.stores.typed_graph_ingestion import (
    ArtifactValidationError,
    ConcurrentBuildError,
    DiskAdmissionError,
    ParquetTypedGraphIngestor,
    SourceMutationError,
)
from test.data.stores.test_typed_graph_csc import (
    _heterogeneous_source,
    _homogeneous_source,
    _metadata,
    _write_table,
)
from test.data.stores.test_typed_graph_features import (
    _metadata as _array_metadata,
    _replace_feature_record,
    _reseal_feature_stage,
)


def _rewrite_writes(
    root: Path,
    columns: dict[str, pa.Array],
) -> tuple[tuple[str, tuple[int, ...]], ...]:
    relative = "relations/writes.parquet"
    return ((relative, tuple(range(len(next(iter(columns.values())))))),)


def _writes_columns(
    *,
    writers: list[str | None] | None = None,
    papers: list[int | None] | None = None,
    edge_ids: list[int | None] | None = None,
    weights: list[float | None] | None = None,
    attributes: list[list[int] | None] | None = None,
) -> dict[str, pa.Array]:
    writers = ["alpha", "mu"] if writers is None else writers
    papers = [0, 7] if papers is None else papers
    count = len(writers)
    edge_ids = list(range(100, 100 + count)) if edge_ids is None else edge_ids
    weights = [float(index + 1) for index in range(count)] if weights is None else weights
    attributes = [[index, -index] for index in range(count)] if attributes is None else attributes
    return {
        "writer": pa.array(writers, type=pa.string()),
        "paper": pa.array(papers, type=pa.uint64()),
        "stable_id": pa.array(edge_ids, type=pa.uint64()),
        "weight": pa.array(weights, type=pa.float32()),
        "attribute": pa.array(attributes, type=pa.list_(pa.int16(), 2)),
    }


def _relation_source_with_columns(
    root: Path,
    columns: dict[str, pa.Array],
    *,
    edge_id: bool = True,
) -> ParquetTypedGraphSource:
    layout = _rewrite_writes(root, columns)
    source = _heterogeneous_source(root, edge_layout=layout)
    _write_table(root / layout[0][0], columns)
    if not edge_id:
        relation = source.spec.relations[0]
        object.__setattr__(relation, "edge_id_column", None)
    return source


def _long_string_edge_source(
    root: Path,
    *,
    maximum_codepoints: int,
) -> ParquetTypedGraphSource:
    columns = _writes_columns()
    columns["stable_id"] = pa.array(
        ["é" * (maximum_codepoints - 1), "漢" * maximum_codepoints],
        type=pa.string(),
    )
    return _relation_source_with_columns(root, columns)


def _same_dtype_source(root: Path, *, swapped: bool) -> ParquetTypedGraphSource:
    _write_table(
        root / "nodes/authors.parquet",
        {
            "node_id": pa.array([1, 2], type=pa.int64()),
            "feature": pa.array([1.0, 2.0], type=pa.float32()),
            "label": pa.array([0, 1], type=pa.int64()),
        },
    )
    _write_table(
        root / "nodes/papers.parquet",
        {
            "node_id": pa.array([10, 20], type=pa.int64()),
            "feature": pa.array([10.0, 20.0], type=pa.float32()),
        },
    )
    src, dst = ((10, 1) if swapped else (1, 10))
    _write_table(
        root / "relations/edge.parquet",
        {
            "src": pa.array([src], type=pa.int64()),
            "dst": pa.array([dst], type=pa.int64()),
        },
    )
    for phase, values in (("train", [1]), ("val", [2]), ("test", [])):
        _write_table(
            root / f"splits/{phase}.parquet",
            {"node_id": pa.array(values, type=pa.int64())},
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
            output_kind="heterogeneous",
            node_types=(
                NodeTypeSpec(
                    name="paper",
                    paths=("nodes/papers.parquet",),
                    id_column="node_id",
                    id_dtype="int64",
                    feature_columns=("feature",),
                    feature_dtype="float32",
                    feature_width=1,
                ),
                NodeTypeSpec(
                    name="author",
                    paths=("nodes/authors.parquet",),
                    id_column="node_id",
                    id_dtype="int64",
                    feature_columns=("feature",),
                    feature_dtype="float32",
                    feature_width=1,
                ),
            ),
            relations=(
                RelationSpec(
                    relation=("author", "writes", "paper"),
                    paths=("relations/edge.parquet",),
                    source_column="src",
                    destination_column="dst",
                ),
            ),
            supervision=SupervisionSpec(
                target_node_type="author",
                label_column="label",
                label_dtype="int64",
                split_registry=SplitRegistrySpec(active_tag="default", sets=(split,)),
            ),
            partition=PartitionSpec(strategy="cluster"),
            ingestion=IngestionLimits(record_batch_rows=1, memory_limit_bytes=64 * 1024**2, temp_directory="duckdb-tmp"),
        )
    )


def _file_sha(path: Path) -> str:
    digest = hashlib.sha256()
    with path.open("rb") as stream:
        while chunk := stream.read(8192):
            digest.update(chunk)
    return digest.hexdigest()


def _relation_outputs(root: Path) -> dict[str, str]:
    outputs = {
        path.relative_to(root).as_posix(): _file_sha(path)
        for path in sorted(root.rglob("*"))
        if path.is_file() and path.name != "relations.complete.json"
    }
    return outputs


def _tree_bytes(root: Path) -> int:
    return sum(path.stat().st_size for path in root.rglob("*") if path.is_file())


def _reseal_checksums(result: Any, *, semantic_identity: bool = False) -> None:
    from topobench.data.stores import typed_graph_csc as csc_module

    metadata = _metadata(result)
    for relation in metadata["relations"].values():
        records = [relation["colptr"], relation["row"]]
        if relation["edge_id"] is not None:
            records.append(relation["edge_id"])
        records.extend(relation["fields"].values())
        for record in records:
            path = result.artifact_root / record["relative_path"]
            record["byte_size"] = path.stat().st_size
            record["file_sha256"] = _file_sha(path)
            record["content_sha256"] = csc_module._array_content_sha256(
                path,
                result.record_batch_rows,
            )
    if semantic_identity:
        for key, relation in metadata["relations"].items():
            relation["semantic_sha256"] = (
                csc_module._relation_semantic_sha256(
                    result.artifact_root,
                    relation,
                    result.record_batch_rows,
                )
            )
            metadata["content_identity"]["relations"][key][
                "semantic_sha256"
            ] = relation["semantic_sha256"]
        metadata["content_sha256"] = ingestion_module._sha256_json(
            metadata["content_identity"]
        )
    ingestion_module._atomic_json(
        result.artifact_root / "relations.json",
        metadata,
    )
    completion = ingestion_module._read_json(
        result.artifact_root / "relations.complete.json"
    )
    if semantic_identity:
        completion["content_sha256"] = metadata["content_sha256"]
    completion["outputs"] = _relation_outputs(result.artifact_root)
    for _ in range(5):
        ingestion_module._atomic_json(
            result.artifact_root / "relations.complete.json",
            completion,
        )
        size = _tree_bytes(result.artifact_root)
        if completion["prepared_relation_stage_bytes"] == size:
            break
        completion["prepared_relation_stage_bytes"] = size
    else:
        raise AssertionError("prepared byte evidence did not stabilize")
    top = ingestion_module._read_json(result.stage_root / "build.complete.json")
    if semantic_identity:
        top["relation_content_sha256"] = metadata["content_sha256"]
    top["outputs"] = ingestion_module._stage_output_checksums(result.stage_root)
    ingestion_module._atomic_json(
        result.stage_root / "build.complete.json",
        top,
    )


def test_rejects_null_and_unresolved_typed_endpoints_before_publish(tmp_path: Path) -> None:
    cases = (
        (_writes_columns(writers=[None], papers=[0]), "EDGE-ENDPOINT-NULL-001"),
        (_writes_columns(writers=["unknown"], papers=[0]), "EDGE-SOURCE-UNRESOLVED-001"),
        (_writes_columns(writers=["alpha"], papers=[123]), "EDGE-DESTINATION-UNRESOLVED-001"),
    )
    for ordinal, (columns, code) in enumerate(cases):
        root = tmp_path / f"source-{ordinal}"
        source = _relation_source_with_columns(root, columns)
        ingestor = ParquetTypedGraphIngestor(source, tmp_path / f"store-{ordinal}")
        with pytest.raises(ArtifactValidationError, match=code):
            ingestor.build_relations()
        stage = ingestor.stage_root(ingestor.inventory())
        assert not (stage / "relations").exists()
        assert not list(stage.glob(".relations-tmp-*"))


def test_rejects_source_destination_map_swap_and_implicit_reverse(tmp_path: Path) -> None:
    valid = _same_dtype_source(tmp_path / "valid", swapped=False)
    ParquetTypedGraphIngestor(valid, tmp_path / "valid-store").build_relations()

    swapped = _same_dtype_source(tmp_path / "swapped", swapped=True)
    with pytest.raises(ArtifactValidationError, match="EDGE-SOURCE-UNRESOLVED-001"):
        ParquetTypedGraphIngestor(swapped, tmp_path / "swapped-store").build_relations()


def test_rejects_endpoint_domain_overflow_instead_of_casting(tmp_path: Path) -> None:
    source = _homogeneous_source(tmp_path / "source")
    _write_table(
        source.spec.source_root / "relations/part.parquet",
        {
            "src": pa.array([9], type=pa.int64()),
            "dst": pa.array([2**64 - 1], type=pa.uint64()),
        },
    )
    with pytest.raises(ArtifactValidationError, match="EDGE-ENDPOINT-OVERFLOW-001"):
        ParquetTypedGraphIngestor(source, tmp_path / "store").build_relations()


def test_rejects_ambiguous_endpoint_duplicates_without_stable_identity(tmp_path: Path) -> None:
    columns = _writes_columns(writers=["alpha", "alpha"], papers=[0, 0])
    source = _relation_source_with_columns(tmp_path / "source", columns, edge_id=False)
    with pytest.raises(ArtifactValidationError, match="EDGE-DUPLICATE-ENDPOINT-001"):
        ParquetTypedGraphIngestor(source, tmp_path / "store").build_relations()


def test_rejects_duplicate_or_null_stable_edge_ids(tmp_path: Path) -> None:
    for ordinal, edge_ids in enumerate(([7, 7], [7, None])):
        source = _relation_source_with_columns(
            tmp_path / f"source-{ordinal}",
            _writes_columns(edge_ids=edge_ids),
        )
        code = "EDGE-ID-DUPLICATE-001" if edge_ids[1] == 7 else "EDGE-ID-NULL-001"
        with pytest.raises(ArtifactValidationError, match=code):
            ParquetTypedGraphIngestor(source, tmp_path / f"store-{ordinal}").build_relations()


@pytest.mark.parametrize(
    "case",
    ("stable_edge_id", "scalar_field", "fixed_list_field"),
)
def test_trailing_nul_utf8_relation_values_are_rejected_before_allocation(
    tmp_path: Path,
    monkeypatch: pytest.MonkeyPatch,
    case: str,
) -> None:
    from topobench.data.stores import typed_graph_csc as csc_module

    if case == "stable_edge_id":
        columns = _writes_columns(
            writers=["alpha", "alpha"],
            papers=[0, 0],
        )
        columns["stable_id"] = pa.array(
            ["a", "a\0"],
            type=pa.string(),
        )
    elif case == "scalar_field":
        columns = _writes_columns()
        columns["weight"] = pa.array(
            ["interior\0nul", "trailing\0"],
            type=pa.string(),
        )
    else:
        columns = _writes_columns()
        columns["attribute"] = pa.array(
            [["interior\0nul", "ok"], ["trailing\0", "ok"]],
            type=pa.list_(pa.string(), 2),
        )
    source = _relation_source_with_columns(tmp_path / "source", columns)
    ingestor = ParquetTypedGraphIngestor(source, tmp_path / "store")
    relation_allocations: list[Path] = []
    original_open_memmap = csc_module.np.lib.format.open_memmap

    def observe_allocation(*args: Any, **kwargs: Any) -> Any:
        path = Path(args[0])
        if any(
            part.startswith(".relations-tmp-")
            for part in path.parts
        ):
            relation_allocations.append(path)
        return original_open_memmap(*args, **kwargs)

    monkeypatch.setattr(
        csc_module.np.lib.format,
        "open_memmap",
        observe_allocation,
    )

    with pytest.raises(
        ArtifactValidationError,
        match="EDGE-STRING-TRAILING-NUL-001",
    ):
        ingestor.build_relations()

    assert relation_allocations == []
    assert not ingestor.stage_root(ingestor.inventory()).joinpath(
        "relations"
    ).exists()


def test_interior_nul_utf8_relation_values_remain_bit_exact(
    tmp_path: Path,
) -> None:
    columns = _writes_columns()
    columns["stable_id"] = pa.array(
        ["edge\0one", "edge\0two"],
        type=pa.string(),
    )
    columns["weight"] = pa.array(
        ["weight\0one", "weight\0two"],
        type=pa.string(),
    )
    columns["attribute"] = pa.array(
        [
            ["left\0one", "right\0one"],
            ["left\0two", "right\0two"],
        ],
        type=pa.list_(pa.string(), 2),
    )
    source = _relation_source_with_columns(tmp_path / "source", columns)
    result = ParquetTypedGraphIngestor(
        source,
        tmp_path / "store",
    ).build_relations()
    relation = _metadata(result)["relations"]["r0000"]

    edge_ids = np.load(
        result.artifact_root / relation["edge_id"]["relative_path"],
        allow_pickle=False,
    )
    weights = np.load(
        result.artifact_root
        / relation["fields"]["weight"]["relative_path"],
        allow_pickle=False,
    )
    attributes = np.load(
        result.artifact_root
        / relation["fields"]["attribute"]["relative_path"],
        allow_pickle=False,
    )

    assert edge_ids.tolist() == ["edge\0one", "edge\0two"]
    assert weights.tolist() == ["weight\0one", "weight\0two"]
    assert attributes.tolist() == [
        ["left\0one", "right\0one"],
        ["left\0two", "right\0two"],
    ]


@pytest.mark.parametrize(
    ("columns", "code"),
    [
        (_writes_columns(weights=[1.0, None]), "EDGE-FIELD-NULL-001"),
        (_writes_columns(weights=[1.0, float("nan")]), "EDGE-FIELD-FINITE-001"),
        (
            {
                **_writes_columns(),
                "attribute": pa.array([[1], [2, 3]], type=pa.list_(pa.int16())),
            },
            "EDGE-FIELD-SCHEMA-001",
        ),
    ],
)
def test_rejects_invalid_declared_edge_fields_before_publish(
    tmp_path: Path,
    columns: dict[str, pa.Array],
    code: str,
) -> None:
    source = _relation_source_with_columns(tmp_path / "source", columns)
    with pytest.raises(ArtifactValidationError, match=code):
        ParquetTypedGraphIngestor(source, tmp_path / "store").build_relations()


def test_resume_rejects_resealed_malformed_csc_and_quarantines_only_relations(
    tmp_path: Path,
) -> None:
    source = _heterogeneous_source(tmp_path / "source")
    ingestor = ParquetTypedGraphIngestor(source, tmp_path / "store")
    result = ingestor.build_relations()
    mapping_sha256 = _file_sha(
        result.stage_root / "mappings/n0000/lookup.duckdb"
    )
    arrays_completion_sha256 = _file_sha(
        result.stage_root / "arrays/arrays.complete.json"
    )
    metadata = _metadata(result)
    row_record = metadata["relations"]["r0000"]["row"]
    row_path = result.artifact_root / row_record["relative_path"]
    row = np.load(row_path, allow_pickle=False)
    row[0] = 99
    np.save(row_path, row, allow_pickle=False)
    _reseal_checksums(result)

    with pytest.raises(ArtifactValidationError, match="CSC-BOUNDS-001"):
        ingestor.build_relations()
    assert result.stage_root.exists()
    assert not result.artifact_root.exists()
    assert (result.stage_root / "mappings").is_dir()
    assert list(result.stage_root.parent.glob(f".{result.stage_root.name}.relations-quarantine-*"))
    top_completion = ingestion_module._read_json(
        result.stage_root / "build.complete.json"
    )
    assert top_completion["stage"] == "typed_graph_arrays"
    assert not any(
        relative.startswith("relations/")
        for relative in top_completion["outputs"]
    )

    rebuilt = ingestor.build_relations()

    assert rebuilt.resumed is False
    assert _file_sha(
        rebuilt.stage_root / "mappings/n0000/lookup.duckdb"
    ) == mapping_sha256
    assert _file_sha(
        rebuilt.stage_root / "arrays/arrays.complete.json"
    ) == arrays_completion_sha256
    assert not list(
        rebuilt.stage_root.parent.glob(
            f".{rebuilt.stage_root.name}.quarantine-*"
        )
    )


def test_nested_relation_symlink_is_quarantined_without_traversing_bad_subtree(
    tmp_path: Path,
) -> None:
    source = _heterogeneous_source(tmp_path / "source")
    ingestor = ParquetTypedGraphIngestor(source, tmp_path / "store")
    result = ingestor.build_relations()
    mapping_path = result.stage_root / "mappings/n0000/lookup.duckdb"
    arrays_completion_path = (
        result.stage_root / "arrays/arrays.complete.json"
    )
    mapping_sha256 = _file_sha(mapping_path)
    arrays_completion_sha256 = _file_sha(arrays_completion_path)
    nested = result.artifact_root / "r0000/nested"
    nested.mkdir()
    (nested / "bad-link").symlink_to(
        result.stage_root / "does-not-exist"
    )

    with pytest.raises(
        ArtifactValidationError,
        match="UNKNOWN-ARTIFACT-001",
    ):
        ingestor.build_relations()

    assert not result.artifact_root.exists()
    assert _file_sha(mapping_path) == mapping_sha256
    assert _file_sha(arrays_completion_path) == arrays_completion_sha256
    top_completion = ingestion_module._read_json(
        result.stage_root / "build.complete.json"
    )
    assert top_completion["stage"] == "typed_graph_arrays"
    assert not any(
        relative.startswith("relations/")
        for relative in top_completion["outputs"]
    )
    assert list(
        result.stage_root.parent.glob(
            f".{result.stage_root.name}.relations-quarantine-*"
        )
    )

    rebuilt = ingestor.build_relations()

    assert rebuilt.resumed is False
    assert _file_sha(mapping_path) == mapping_sha256
    assert _file_sha(arrays_completion_path) == arrays_completion_sha256


def test_resume_rejects_resealed_malformed_colptr_and_field_length(tmp_path: Path) -> None:
    for ordinal, artifact in enumerate(("colptr", "field")):
        source = _heterogeneous_source(tmp_path / f"source-{ordinal}")
        ingestor = ParquetTypedGraphIngestor(source, tmp_path / f"store-{ordinal}")
        result = ingestor.build_relations()
        metadata = _metadata(result)
        relation = metadata["relations"]["r0000"]
        if artifact == "colptr":
            path = result.artifact_root / relation["colptr"]["relative_path"]
            colptr = np.load(path, allow_pickle=False)
            colptr[2] = colptr[1] - 1
            np.save(path, colptr, allow_pickle=False)
            code = "CSC-COLPTR-001"
        else:
            path = result.artifact_root / relation["fields"]["weight"]["relative_path"]
            values = np.load(path, allow_pickle=False)[:-1]
            np.save(path, values, allow_pickle=False)
            code = "EDGE-FIELD-ALIGNMENT-001"
        _reseal_checksums(result)
        with pytest.raises(ArtifactValidationError, match=code):
            ingestor.build_relations()


def test_resume_semantic_audit_rejects_source_inconsistent_resealed_values(tmp_path: Path) -> None:
    source = _heterogeneous_source(tmp_path / "source")
    ingestor = ParquetTypedGraphIngestor(source, tmp_path / "store")
    result = ingestor.build_relations()
    relation = _metadata(result)["relations"]["r0000"]
    path = result.artifact_root / relation["fields"]["weight"]["relative_path"]
    values = np.load(path, allow_pickle=False)
    values[0] = 999.0
    np.save(path, values, allow_pickle=False)
    _reseal_checksums(result, semantic_identity=True)

    with pytest.raises(ArtifactValidationError, match="RELATION-SEMANTIC-001"):
        ingestor.build_relations()


def test_resume_source_audit_is_bit_exact_for_signed_zero_fields(
    tmp_path: Path,
) -> None:
    source = _relation_source_with_columns(
        tmp_path / "source",
        _writes_columns(weights=[0.0, 2.0]),
    )
    ingestor = ParquetTypedGraphIngestor(source, tmp_path / "store")
    result = ingestor.build_relations()
    relation = _metadata(result)["relations"]["r0000"]
    path = result.artifact_root / relation["fields"]["weight"][
        "relative_path"
    ]
    values = np.load(path, allow_pickle=False)
    assert not np.signbit(values[0])
    values[0] = np.float32(-0.0)
    np.save(path, values, allow_pickle=False)
    _reseal_checksums(result, semantic_identity=True)

    with pytest.raises(ArtifactValidationError, match="RELATION-SEMANTIC-001"):
        ingestor.build_relations()


def test_resume_rejects_fully_resealed_task3_array_content_change(
    tmp_path: Path,
) -> None:
    source = _heterogeneous_source(tmp_path / "source")
    ingestor = ParquetTypedGraphIngestor(source, tmp_path / "store")
    relation_result = ingestor.build_relations()
    arrays_result = ingestor.build_arrays()
    metadata = _array_metadata(arrays_result.stage_root)
    record = metadata["nodes"]["n0000"]
    values = np.load(
        arrays_result.artifact_root / record["relative_path"],
        allow_pickle=False,
    )
    changed = values.copy()
    changed[0, 0] = 999.0
    _replace_feature_record(
        arrays_result,
        metadata,
        "n0000",
        changed,
    )
    _reseal_feature_stage(arrays_result, metadata)
    _reseal_checksums(relation_result)

    with pytest.raises(ArtifactValidationError, match="ARRAY-BINDING-001"):
        ingestor.build_relations()


def test_relation_build_rejects_source_mutation_after_exact_indexes(tmp_path: Path) -> None:
    source = _heterogeneous_source(tmp_path / "source")
    ingestor = ParquetTypedGraphIngestor(source, tmp_path / "store")
    index_build = ingestor.build()
    path = source.spec.source_root / "relations/writes-a.parquet"
    table = pq.read_table(path)
    pq.write_table(table.replace_schema_metadata({b"mutated": b"yes"}), path)

    with pytest.raises(SourceMutationError, match="SOURCE-MUTATION-001"):
        ingestor.build_relations(index_build)
    assert not (index_build.stage_root / "relations").exists()


def test_relation_disk_limit_is_admitted_before_relation_allocation(tmp_path: Path) -> None:
    source = _heterogeneous_source(tmp_path / "source")
    baseline = ParquetTypedGraphIngestor(
        source,
        tmp_path / "baseline",
    ).build_relations()
    requirements = _metadata(baseline)["resource_evidence"][
        "exact_disk_requirements"
    ]
    limit = requirements["shared_peak_bytes"] - 1
    assert limit >= baseline.inventory.required_peak_bytes
    ingestor = ParquetTypedGraphIngestor(
        source,
        tmp_path / "limited",
        disk_limit_bytes=limit,
    )
    with pytest.raises(
        DiskAdmissionError,
        match="DISK-PREFLIGHT-001",
    ):
        ingestor.build_relations()
    inventory = replace(
        ingestor.inventory(),
        required_peak_bytes=limit,
    )
    assert not (ingestor.stage_root(inventory) / "relations").exists()


def test_coarse_relation_disk_limit_is_rejected_before_task4_side_effects(
    tmp_path: Path,
    monkeypatch: pytest.MonkeyPatch,
) -> None:
    from topobench.data.stores import typed_graph_csc as csc_module

    source = _heterogeneous_source(tmp_path / "source")
    baseline = ParquetTypedGraphIngestor(
        source,
        tmp_path / "baseline",
    ).build_relations()
    resource = _metadata(baseline)["resource_evidence"]
    coarse = resource["disk_requirements"]
    exact = resource["exact_disk_requirements"]
    assert exact["shared_peak_bytes"] <= coarse["shared_peak_bytes"]

    limited = ParquetTypedGraphIngestor(source, tmp_path / "limited")
    index_build = limited.build()
    limited.build_arrays(index_build)
    limited.disk_limit_bytes = coarse["shared_peak_bytes"] - 1
    work_parent = (
        Path(source.spec.ingestion.temp_directory)
        / ".topobench-typed-graph-work"
        / index_build.stage_root.name
    )
    before = set(work_parent.glob("relation-*"))

    def reject_relation_write(*args: Any, **kwargs: Any) -> Any:
        raise AssertionError("relation write reached before coarse admission")

    monkeypatch.setattr(
        csc_module.TypedGraphRelationWriter,
        "_write_all",
        reject_relation_write,
    )
    with pytest.raises(
        DiskAdmissionError,
        match="DISK-PREFLIGHT-001",
    ):
        limited.build_relations(index_build)

    assert not (index_build.stage_root / "relations").exists()
    assert not list(index_build.stage_root.glob(".relations-tmp-*"))
    after = set(work_parent.glob("relation-*"))
    assert after == before


def test_exact_long_string_disk_admission_precedes_all_relation_allocation(
    tmp_path: Path,
    monkeypatch: pytest.MonkeyPatch,
) -> None:
    from topobench.data.stores import typed_graph_csc as csc_module

    maximum_codepoints = 4096
    source = _long_string_edge_source(
        tmp_path / "source",
        maximum_codepoints=maximum_codepoints,
    )
    baseline = ParquetTypedGraphIngestor(
        source,
        tmp_path / "baseline",
    ).build_relations()
    metadata = _metadata(baseline)
    exact = metadata["resource_evidence"]["exact_disk_requirements"]
    coarse = metadata["resource_evidence"]["disk_requirements"]
    sizing = exact["arrays"]["r0000"]["edge_id"]
    assert sizing["storage_dtype"] == np.dtype(
        f"<U{maximum_codepoints}"
    ).str
    assert sizing["payload_bytes"] == 2 * 4 * maximum_codepoints
    edge_path = baseline.artifact_root / metadata["relations"]["r0000"][
        "edge_id"
    ]["relative_path"]
    assert np.load(
        edge_path,
        mmap_mode="r",
        allow_pickle=False,
    ).nbytes == sizing["payload_bytes"]
    assert edge_path.stat().st_size == sizing["file_bytes"]

    limited = ParquetTypedGraphIngestor(source, tmp_path / "limited")
    index_build = limited.build()
    limited.build_arrays(index_build)
    limited.disk_limit_bytes = exact["shared_peak_bytes"] - 1
    allocations: list[tuple[tuple[Any, ...], dict[str, Any]]] = []
    original_open_memmap = csc_module.np.lib.format.open_memmap

    def observe_allocation(*args: Any, **kwargs: Any) -> Any:
        path = Path(args[0])
        if any(
            part.startswith(".relations-tmp-")
            for part in path.parts
        ):
            allocations.append((args, kwargs))
        return original_open_memmap(*args, **kwargs)

    monkeypatch.setattr(
        csc_module.np.lib.format,
        "open_memmap",
        observe_allocation,
    )
    with pytest.raises(DiskAdmissionError, match="DISK-PREFLIGHT-001"):
        limited.build_relations(index_build)
    assert allocations == []
    assert not (index_build.stage_root / "relations").exists()

    monkeypatch.undo()
    admitted = ParquetTypedGraphIngestor(
        source,
        tmp_path / "admitted",
        disk_limit_bytes=coarse["shared_peak_bytes"],
    ).build_relations()
    admitted_metadata = _metadata(admitted)
    assert (
        admitted_metadata["resource_evidence"][
            "exact_disk_requirements"
        ]
        == exact
    )
    exact_edge_record = admitted_metadata["relations"]["r0000"][
        "edge_id"
    ]
    assert exact_edge_record["storage_dtype"] == np.dtype(
        f"<U{maximum_codepoints}"
    ).str


def test_relation_preflight_uses_actual_final_and_temporary_filesystems(
    tmp_path: Path,
    monkeypatch: pytest.MonkeyPatch,
) -> None:
    source = _heterogeneous_source(tmp_path / "source")
    ingestor = ParquetTypedGraphIngestor(source, tmp_path / "store")
    index_build = ingestor.build()
    real_capacity = ingestion_module._filesystem_capacity

    def capacity(path: Path) -> tuple[int, int, Path]:
        device, available, probe = real_capacity(path)
        if path == ingestor.store_root:
            return device, 1, probe
        return device, available, probe

    monkeypatch.setattr(ingestion_module, "_filesystem_capacity", capacity)
    with pytest.raises(DiskAdmissionError, match="DISK-PREFLIGHT-001"):
        ingestor.build_relations(index_build)
    assert not (index_build.stage_root / "relations").exists()


def test_resume_re_admits_temporary_space_before_snapshot_or_spill(
    tmp_path: Path,
    monkeypatch: pytest.MonkeyPatch,
) -> None:
    from topobench.data.stores import typed_graph_csc as csc_module

    source = _heterogeneous_source(tmp_path / "source")
    ingestor = ParquetTypedGraphIngestor(source, tmp_path / "store")
    initial = ingestor.build_relations()
    index_build = ingestor.build()
    array_build = ingestor.build_arrays(index_build)
    writer = csc_module.TypedGraphRelationWriter(
        ingestor,
        index_build,
        array_build,
    )
    exact = _metadata(initial)["resource_evidence"][
        "exact_disk_requirements"
    ]
    required = exact["resume_temporary_additional_bytes"]
    real_capacity = ingestion_module._filesystem_capacity
    temporary_root = initial.inventory.temporary_filesystem_path

    def capacity(path: Path) -> tuple[int, int, Path]:
        device, available, probe = real_capacity(path)
        if path == temporary_root:
            return device, required - 1, probe
        return device, available, probe

    work_parent = (
        temporary_root
        / ".topobench-typed-graph-work"
        / initial.stage_root.name
    )
    before = set(work_parent.glob("*")) if work_parent.exists() else set()
    monkeypatch.setattr(ingestion_module, "_filesystem_capacity", capacity)

    with pytest.raises(
        DiskAdmissionError,
        match="RESUME-DISK-PREFLIGHT-001",
    ):
        writer.build()

    after = set(work_parent.glob("*")) if work_parent.exists() else set()
    assert after == before


def test_prepared_relation_directories_are_fsynced_bottom_up_before_publish(
    tmp_path: Path,
    monkeypatch: pytest.MonkeyPatch,
) -> None:
    from topobench.data.stores import typed_graph_csc as csc_module

    source = _heterogeneous_source(tmp_path / "source")
    ingestor = ParquetTypedGraphIngestor(source, tmp_path / "store")
    index_build = ingestor.build()
    ingestor.build_arrays(index_build)
    observed: list[Path] = []
    real_fsync = ingestion_module._fsync_directory
    real_publish = csc_module.TypedGraphRelationWriter._publish

    def track_fsync(path: Path) -> None:
        observed.append(path)
        real_fsync(path)

    def verify_before_publish(
        self: Any,
        temporary_root: Path,
        stage_root: Path,
    ) -> None:
        expected = [
            temporary_root / "r0000/fields",
            temporary_root / "r0000",
            temporary_root / "r0001",
            temporary_root,
        ]
        positions = [
            max(
                index
                for index, observed_path in enumerate(observed)
                if observed_path == path
            )
            for path in expected
        ]
        assert positions == sorted(positions)
        real_publish(self, temporary_root, stage_root)

    monkeypatch.setattr(
        ingestion_module,
        "_fsync_directory",
        track_fsync,
    )
    monkeypatch.setattr(
        csc_module.TypedGraphRelationWriter,
        "_publish",
        verify_before_publish,
    )

    ingestor.build_relations(index_build)


def test_atomic_faults_never_publish_partial_subtrees_and_resume_verified_publish(
    tmp_path: Path,
    monkeypatch: pytest.MonkeyPatch,
) -> None:
    from topobench.data.stores import typed_graph_csc as csc_module

    source = _heterogeneous_source(tmp_path / "source")
    ingestor = ParquetTypedGraphIngestor(source, tmp_path / "store")
    index_build = ingestor.build()
    original_publish = csc_module.TypedGraphRelationWriter._publish

    def fail_before_publish(self: Any, temporary_root: Path, stage_root: Path) -> None:
        raise OSError("injected before atomic replace")

    monkeypatch.setattr(csc_module.TypedGraphRelationWriter, "_publish", fail_before_publish)
    with pytest.raises(OSError, match="injected"):
        ingestor.build_relations(index_build)
    assert not (index_build.stage_root / "relations").exists()
    assert not list(index_build.stage_root.glob(".relations-tmp-*"))

    monkeypatch.setattr(csc_module.TypedGraphRelationWriter, "_publish", original_publish)
    original_finalize = csc_module.TypedGraphRelationWriter._finalize_top_completion

    def fail_after_publish(self: Any, result: Any) -> None:
        raise OSError("injected after atomic replace")

    monkeypatch.setattr(csc_module.TypedGraphRelationWriter, "_finalize_top_completion", fail_after_publish)
    with pytest.raises(OSError, match="injected after"):
        ingestor.build_relations(index_build)
    assert (index_build.stage_root / "relations/relations.complete.json").is_file()

    monkeypatch.setattr(csc_module.TypedGraphRelationWriter, "_finalize_top_completion", original_finalize)
    resumed = ingestor.build_relations(index_build)
    assert resumed.resumed is True


def test_relation_writer_reuses_content_lock_for_concurrent_builds(tmp_path: Path) -> None:
    source = _heterogeneous_source(tmp_path / "source")
    ingestor = ParquetTypedGraphIngestor(source, tmp_path / "store")
    index_build = ingestor.build()
    lock_path = ingestor.lock_path(index_build.inventory)
    lock_path.write_text(
        json.dumps(
            {
                "pid": os.getpid(),
                "hostname": __import__("socket").gethostname(),
                "created_ns": __import__("time").time_ns(),
                "token": "active",
            }
        ),
        encoding="utf-8",
    )
    with pytest.raises(ConcurrentBuildError, match="BUILD-LOCK-001"):
        ingestor.build_relations(index_build)
