"""Behavioral tests for bounded, ordinal-aligned typed feature arrays."""

from __future__ import annotations

import hashlib
import json
from dataclasses import replace
from pathlib import Path
from typing import Any

import numpy as np
import pyarrow as pa
import pyarrow.parquet as pq
import pytest

import topobench.data.stores.typed_graph_arrays as arrays_module
import topobench.data.stores.typed_graph_ingestion as ingestion_module
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
from topobench.data.stores.typed_graph_ingestion import (
    ArtifactValidationError,
    DiskAdmissionError,
    ParquetTypedGraphIngestor,
    SourceMutationError,
)


def _write_table(path: Path, columns: dict[str, pa.Array]) -> None:
    path.parent.mkdir(parents=True, exist_ok=True)
    pq.write_table(pa.table(columns), path)


def _feature_source(
    root: Path,
    *,
    author_features: pa.Array | None = None,
    paper_columns: tuple[pa.Array, pa.Array, pa.Array] | None = None,
    author_feature_dtype: str = "float32",
    author_feature_width: int = 2,
    paper_feature_dtype: str = "float64",
) -> ParquetTypedGraphSource:
    author_ids = ["zeta", "alpha", "mu"]
    if author_features is None:
        author_features = pa.array(
            [[9.0, 90.0], [1.0, 10.0], [5.0, 50.0]],
            type=pa.list_(pa.float32(), 2),
        )
    _write_table(
        root / "nodes/authors.parquet",
        {
            "author_id": pa.array(author_ids, type=pa.string()),
            "embedding": author_features,
            "label": pa.array([2, 0, 1], type=pa.int64()),
        },
    )

    paper_ids = [2**64 - 1, 0, 7]
    if paper_columns is None:
        paper_columns = (
            pa.array([30.0, 10.0, 20.0], type=pa.float64()),
            pa.array([31.0, 11.0, 21.0], type=pa.float64()),
            pa.array([32.0, 12.0, 22.0], type=pa.float64()),
        )
    _write_table(
        root / "nodes/papers.parquet",
        {
            "paper_id": pa.array(paper_ids, type=pa.uint64()),
            "f0": paper_columns[0],
            "f1": paper_columns[1],
            "f2": paper_columns[2],
        },
    )
    _write_table(
        root / "relations/writes.parquet",
        {
            "src": pa.array(["alpha", "mu"], type=pa.string()),
            "dst": pa.array([0, 7], type=pa.uint64()),
        },
    )
    for phase, ids in (
        ("train", ["mu"]),
        ("val", ["zeta"]),
        ("test", ["alpha"]),
    ):
        _write_table(
            root / f"splits/{phase}.parquet",
            {"author_id": pa.array(ids, type=pa.string())},
        )

    split = SplitSetSpec(
        tag="default",
        train="splits/train.parquet",
        val="splits/val.parquet",
        test="splits/test.parquet",
        coverage="complete",
    )
    spec = ParquetTypedGraphSpec(
        source_root=root,
        output_kind="heterogeneous",
        node_types=(
            NodeTypeSpec(
                name="paper",
                paths=("nodes/papers.parquet",),
                id_column="paper_id",
                id_dtype="uint64",
                feature_columns=("f0", "f1", "f2"),
                feature_dtype=paper_feature_dtype,
                feature_width=3,
            ),
            NodeTypeSpec(
                name="author",
                paths=("nodes/authors.parquet",),
                id_column="author_id",
                id_dtype="string",
                feature_columns=("embedding",),
                feature_dtype=author_feature_dtype,
                feature_width=author_feature_width,
                feature_representation="fixed_size_list",
            ),
        ),
        relations=(
            RelationSpec(
                relation=("author", "writes", "paper"),
                paths=("relations/writes.parquet",),
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
        ingestion=IngestionLimits(
            record_batch_rows=2,
            memory_limit_bytes=64 * 1024**2,
            temp_directory="duckdb-tmp",
        ),
    )
    return ParquetTypedGraphSource(spec)


def _content_sha(path: Path, *, batch_rows: int = 2) -> str:
    array = np.load(path, mmap_mode="r", allow_pickle=False)
    digest = hashlib.sha256()
    for start in range(0, array.shape[0], batch_rows):
        block = np.ascontiguousarray(array[start : start + batch_rows])
        digest.update(memoryview(block).cast("B"))
    return digest.hexdigest()


def _file_sha(path: Path) -> str:
    digest = hashlib.sha256()
    with path.open("rb") as stream:
        while chunk := stream.read(8192):
            digest.update(chunk)
    return digest.hexdigest()


def _metadata(stage_root: Path) -> dict[str, object]:
    artifact_root = (
        stage_root / "arrays"
        if (stage_root / "arrays").is_dir()
        else stage_root
    )
    return json.loads(
        (artifact_root / "arrays.json").read_text(encoding="utf-8")
    )


def _array_outputs(root: Path) -> dict[str, str]:
    outputs = {
        path.relative_to(root).as_posix(): _file_sha(path)
        for subtree in ("nodes", "splits")
        for path in sorted((root / subtree).rglob("*"))
        if path.is_file()
    }
    outputs["arrays.json"] = _file_sha(root / "arrays.json")
    return outputs


def _reseal_feature_stage(
    result: arrays_module.TypedGraphArrayBuild,
    metadata: dict[str, Any],
) -> None:
    ingestion_module._atomic_json(result.artifact_root / "arrays.json", metadata)
    completion = ingestion_module._read_json(
        result.artifact_root / "arrays.complete.json"
    )
    completion["content_sha256"] = metadata["content_sha256"]
    completion["outputs"] = _array_outputs(result.artifact_root)
    for _ in range(4):
        ingestion_module._atomic_json(
            result.artifact_root / "arrays.complete.json",
            completion,
        )
        prepared_bytes = arrays_module._tree_bytes(result.artifact_root)
        if completion["prepared_array_stage_bytes"] == prepared_bytes:
            break
        completion["prepared_array_stage_bytes"] = prepared_bytes
    else:
        raise AssertionError("prepared byte evidence did not stabilize")
    top_completion = ingestion_module._read_json(
        result.stage_root / "build.complete.json"
    )
    top_completion["array_content_sha256"] = metadata["content_sha256"]
    top_completion["outputs"] = ingestion_module._stage_output_checksums(
        result.stage_root
    )
    ingestion_module._atomic_json(
        result.stage_root / "build.complete.json",
        top_completion,
    )


def _replace_feature_record(
    result: arrays_module.TypedGraphArrayBuild,
    metadata: dict[str, Any],
    key: str,
    values: np.ndarray,
) -> dict[str, Any]:
    record = metadata["nodes"][key]
    path = result.artifact_root / record["relative_path"]
    np.save(path, values, allow_pickle=False)
    record["shape"] = list(values.shape)
    record["count"] = len(values)
    record["byte_size"] = path.stat().st_size
    record["file_sha256"] = _file_sha(path)
    record["content_sha256"] = arrays_module._array_content_sha(
        path,
        result.record_batch_rows,
    )
    metadata["content_identity"]["nodes"][key] = record["content_sha256"]
    metadata["content_sha256"] = ingestion_module._sha256_json(
        metadata["content_identity"]
    )
    return record


def test_array_build_closes_internal_validation_mappings(
    tmp_path: Path,
    monkeypatch: pytest.MonkeyPatch,
) -> None:
    source = _feature_source(tmp_path / "source")
    opened: list[np.memmap] = []
    original_load = np.load

    def tracked_load(*args: Any, **kwargs: Any) -> Any:
        array = original_load(*args, **kwargs)
        path = Path(args[0])
        if (
            isinstance(array, np.memmap)
            and "arrays" in path.parts
        ):
            opened.append(array)
        return array

    monkeypatch.setattr(arrays_module.np, "load", tracked_load)
    result = ParquetTypedGraphIngestor(
        source,
        tmp_path / "stores",
    ).build_arrays()

    assert opened
    assert all(array._mmap.closed for array in opened)
    public = original_load(
        result.artifact_root / "nodes/n0000/x.npy",
        mmap_mode="r",
        allow_pickle=False,
    )
    assert public.flags.writeable is False
    assert public._mmap.closed is False
    public._mmap.close()


def test_streams_distinct_fixed_list_and_scalar_features_by_exact_ordinal(
    tmp_path: Path,
) -> None:
    source = _feature_source(tmp_path / "source")
    result = ParquetTypedGraphIngestor(source, tmp_path / "stores").build_arrays()
    assert result.artifact_root == result.stage_root / "arrays"

    metadata = _metadata(result.stage_root)
    author = metadata["nodes"]["n0000"]
    paper = metadata["nodes"]["n0001"]
    author_path = result.artifact_root / author["relative_path"]
    paper_path = result.artifact_root / paper["relative_path"]
    author_x = np.load(author_path, mmap_mode="r", allow_pickle=False)
    paper_x = np.load(paper_path, mmap_mode="r", allow_pickle=False)

    np.testing.assert_array_equal(
        author_x,
        np.array([[1.0, 10.0], [5.0, 50.0], [9.0, 90.0]], dtype="<f4"),
    )
    np.testing.assert_array_equal(
        paper_x,
        np.array(
            [[10.0, 11.0, 12.0], [20.0, 21.0, 22.0], [30.0, 31.0, 32.0]],
            dtype="<f8",
        ),
    )
    assert author_x.dtype.str == "<f4"
    assert paper_x.dtype.str == "<f8"
    assert author_x.shape == (3, 2)
    assert paper_x.shape == (3, 3)
    assert author["representation"] == "fixed_size_list"
    assert paper["representation"] == "scalar_columns"
    assert author["content_sha256"] == _content_sha(author_path)
    assert author["source_sha256"] == _file_sha(
        source.spec.source_root / "nodes/authors.parquet"
    )
    assert paper["source_sha256"] == _file_sha(
        source.spec.source_root / "nodes/papers.parquet"
    )
    assert paper["content_sha256"] == _content_sha(paper_path)
    assert author["count"] == paper["count"] == 3
    assert metadata["record_batch_rows"] == 2
    assert metadata["max_record_batch_rows"] <= 2
    resource = metadata["resource_evidence"]
    assert resource["snapshot_bytes"] == result.inventory.snapshot_bytes
    assert resource["snapshot_persisted"] is False
    assert resource["snapshot_bytes_accounted"] is True
    requirements = resource["disk_requirements"]
    assert requirements["snapshot_bytes"] == resource["snapshot_bytes"]
    assert requirements["final_peak_bytes"] == (
        requirements["task2_final_bytes"]
        + requirements["estimated_array_bytes"]
    )
    assert requirements["temporary_peak_bytes"] == (
        requirements["snapshot_bytes"] + requirements["spill_bytes"]
    )
    assert not list(result.artifact_root.rglob("*mask*"))
    assert not (result.artifact_root / "nodes/n0001/y.npy").exists()


@pytest.mark.parametrize(
    ("author_features", "author_dtype", "author_width", "paper_columns", "message"),
    [
        (
            pa.array([[1.0, 2.0], [3.0], [4.0, 5.0]], type=pa.list_(pa.float32())),
            "float32",
            2,
            None,
            "FEATURE-SCHEMA-001",
        ),
        (
            pa.array(
                [[1.0, 2.0], [3.0, 4.0], [5.0, 6.0]],
                type=pa.list_(pa.float32(), 2),
            ),
            "float32",
            3,
            None,
            "FEATURE-WIDTH-001",
        ),
        (
            pa.array(
                [[1.0, 2.0], None, [5.0, 6.0]],
                type=pa.list_(pa.float32()),
            ),
            "float32",
            2,
            None,
            "FEATURE-SCHEMA-001",
        ),
        (
            pa.array(
                [[1.0, 2.0], [3.0, None], [5.0, 6.0]],
                type=pa.list_(pa.float32(), 2),
            ),
            "float32",
            2,
            None,
            "FEATURE-NULL-001",
        ),
        (
            pa.array(
                [[1.0, 2.0], [3.0, 4.0], [5.0, 6.0]],
                type=pa.list_(pa.float64(), 2),
            ),
            "float32",
            2,
            None,
            "FEATURE-CAST-001",
        ),
        (
            pa.array(
                [[1.0, 2.0], [3.0, float("nan")], [5.0, 6.0]],
                type=pa.list_(pa.float32(), 2),
            ),
            "float32",
            2,
            None,
            "FEATURE-FINITE-001",
        ),
        (
            None,
            "float32",
            2,
            (
                pa.array([30.0, 10.0, 20.0], type=pa.float64()),
                pa.array([31.0, None, 21.0], type=pa.float64()),
                pa.array([32.0, 12.0, 22.0], type=pa.float64()),
            ),
            "FEATURE-NULL-001",
        ),
        (
            None,
            "float32",
            2,
            (
                pa.array([30.0, 10.0, 20.0], type=pa.float64()),
                pa.array([31.0, float("inf"), 21.0], type=pa.float64()),
                pa.array([32.0, 12.0, 22.0], type=pa.float64()),
            ),
            "FEATURE-FINITE-001",
        ),
    ],
    ids=(
        "variable-list",
        "wrong-fixed-width",
        "null-variable-list",
        "null-list-element",
        "coercive-cast",
        "non-finite-fixed-list",
        "null-scalar",
        "non-finite-scalar",
    ),
)
def test_rejects_malformed_feature_arrays_without_partial_publish(
    tmp_path: Path,
    author_features: pa.Array | None,
    author_dtype: str,
    author_width: int,
    paper_columns: tuple[pa.Array, pa.Array, pa.Array] | None,
    message: str,
) -> None:
    source = _feature_source(
        tmp_path / "source",
        author_features=author_features,
        paper_columns=paper_columns,
        author_feature_dtype=author_dtype,
        author_feature_width=author_width,
    )
    ingestor = ParquetTypedGraphIngestor(source, tmp_path / "stores")

    with pytest.raises(ArtifactValidationError, match=message):
        ingestor.build_arrays()

    inventory = ingestor.inventory()
    stage_root = ingestor.stage_root(inventory)
    assert (stage_root / "build.complete.json").is_file()
    assert not (stage_root / "arrays.complete.json").exists()
    assert not (stage_root / "nodes").exists()
    assert not (stage_root / "splits").exists()
    assert not (stage_root / "arrays").exists()
    assert not list(stage_root.glob(".arrays-tmp-*"))


def test_array_stage_resumes_only_after_reopen_and_checksum_validation(
    tmp_path: Path,
) -> None:
    source = _feature_source(tmp_path / "source")
    ingestor = ParquetTypedGraphIngestor(source, tmp_path / "stores")

    first = ingestor.build_arrays()
    second = ingestor.build_arrays()

    assert first.resumed is False
    assert second.resumed is True
    assert first.content_sha256 == second.content_sha256
    assert first.stage_root == second.stage_root
    completion = json.loads(
        (first.artifact_root / "arrays.complete.json").read_text(encoding="utf-8")
    )
    assert completion["reopened_and_validated"] is True
    assert completion["content_sha256"] == first.content_sha256
    assert completion["input_fingerprint"] == first.inventory.source_fingerprint

    metadata = _metadata(first.stage_root)
    feature_path = first.artifact_root / metadata["nodes"]["n0000"]["relative_path"]
    feature_path.chmod(0o600)
    with feature_path.open("r+b") as stream:
        stream.seek(-1, 2)
        last = stream.read(1)
        stream.seek(-1, 2)
        stream.write(bytes([last[0] ^ 1]))

    rebuilt = ingestor.build_arrays()
    assert rebuilt.resumed is False
    assert rebuilt.stage_root == first.stage_root
    assert rebuilt.content_sha256 == first.content_sha256
    assert rebuilt.stage_root.exists()
    assert list(first.stage_root.parent.glob(f".{first.stage_root.name}.quarantine-*"))


def test_array_stage_rejects_source_mutation_after_index_snapshot(
    tmp_path: Path,
) -> None:
    source = _feature_source(tmp_path / "source")
    ingestor = ParquetTypedGraphIngestor(source, tmp_path / "stores")
    index_build = ingestor.build()
    path = source.spec.source_root / "nodes/authors.parquet"
    table = pq.read_table(path)
    pq.write_table(table.replace_schema_metadata({b"mutated": b"yes"}), path)

    with pytest.raises(SourceMutationError, match="SOURCE-MUTATION-001"):
        ingestor.build_arrays(index_build)
    assert not (index_build.stage_root / "arrays").exists()


@pytest.mark.parametrize(
    "legacy_boundary",
    ("nodes", "splits", "metadata", "completion"),
)
def test_incomplete_legacy_array_boundaries_are_quarantined_and_rebuilt(
    tmp_path: Path,
    legacy_boundary: str,
) -> None:
    source = _feature_source(tmp_path / "source")
    ingestor = ParquetTypedGraphIngestor(source, tmp_path / "stores")
    index_build = ingestor.build()
    mapping_path = index_build.indexes["author"].lookup_path
    mapping_sha = _file_sha(mapping_path)
    stage_root = index_build.stage_root
    ordered_boundaries = ("nodes", "splits", "metadata", "completion")
    boundary_index = ordered_boundaries.index(legacy_boundary)
    if boundary_index >= 0:
        (stage_root / "nodes/n0000").mkdir(parents=True)
        (stage_root / "nodes/n0000/partial.npy").write_bytes(b"partial")
    if boundary_index >= 1:
        (stage_root / "splits/default").mkdir(parents=True)
        (stage_root / "splits/default/partial.npy").write_bytes(b"partial")
    if boundary_index >= 2:
        (stage_root / "arrays.json").write_text("{}", encoding="utf-8")
    if boundary_index >= 3:
        (stage_root / "arrays.complete.json").write_text("{}", encoding="utf-8")

    result = ingestor.build_arrays(index_build)

    assert result.artifact_root == stage_root / "arrays"
    assert result.resumed is False
    assert _file_sha(mapping_path) == mapping_sha
    quarantines = list(
        stage_root.parent.glob(f".{stage_root.name}.arrays-quarantine-*")
    )
    assert len(quarantines) == 1
    assert (quarantines[0] / "nodes").exists()
    assert (result.artifact_root / "arrays.complete.json").is_file()


@pytest.mark.parametrize(
    "former_boundary",
    ("nodes", "splits", "metadata", "completion"),
)
def test_fault_before_atomic_array_rename_never_exposes_partial_stage(
    tmp_path: Path,
    monkeypatch: pytest.MonkeyPatch,
    former_boundary: str,
) -> None:
    source = _feature_source(tmp_path / "source")
    ingestor = ParquetTypedGraphIngestor(source, tmp_path / "stores")
    index_build = ingestor.build()
    mapping_path = index_build.indexes["author"].lookup_path
    mapping_sha = _file_sha(mapping_path)
    real_publish = arrays_module.TypedGraphArrayWriter._publish
    attempts = 0

    def fail_once(
        writer: arrays_module.TypedGraphArrayWriter,
        temporary_root: Path,
        stage_root: Path,
    ) -> None:
        nonlocal attempts
        attempts += 1
        if attempts == 1:
            raise RuntimeError(f"fault after former {former_boundary} boundary")
        real_publish(writer, temporary_root, stage_root)

    monkeypatch.setattr(
        arrays_module.TypedGraphArrayWriter,
        "_publish",
        fail_once,
    )
    with pytest.raises(RuntimeError, match=f"former {former_boundary}"):
        ingestor.build_arrays(index_build)

    assert _file_sha(mapping_path) == mapping_sha
    assert not (index_build.stage_root / "arrays").exists()
    assert not (index_build.stage_root / "nodes").exists()
    assert not (index_build.stage_root / "splits").exists()
    assert not list(index_build.stage_root.glob(".arrays-tmp-*"))

    recovered = ingestor.build_arrays(index_build)
    assert recovered.artifact_root == index_build.stage_root / "arrays"
    assert (recovered.artifact_root / "arrays.complete.json").is_file()


def test_task3_shared_filesystem_admission_is_exact_and_precedes_allocation(
    tmp_path: Path,
    monkeypatch: pytest.MonkeyPatch,
) -> None:
    source = _feature_source(tmp_path / "source")
    ingestor = ParquetTypedGraphIngestor(source, tmp_path / "stores")
    index_build = ingestor.build()
    writer = arrays_module.TypedGraphArrayWriter(ingestor, index_build)
    requirements = writer._array_disk_requirements()
    assert requirements["same_filesystem"] is True
    assert requirements["shared_peak_bytes"] == (
        requirements["task2_final_bytes"]
        + requirements["estimated_array_bytes"]
        + requirements["snapshot_bytes"]
        + requirements["spill_bytes"]
    )

    device = index_build.inventory.final_device
    shared_additional = requirements["shared_additional_bytes"]
    available = shared_additional - 1

    def shared_capacity(path: Path) -> tuple[int, int, Path]:
        return device, available, path

    monkeypatch.setattr(ingestion_module, "_filesystem_capacity", shared_capacity)
    allocations: list[object] = []
    original_open_memmap = np.lib.format.open_memmap

    def record_allocation(*args: object, **kwargs: object) -> np.memmap:
        allocations.append(args[0])
        return original_open_memmap(*args, **kwargs)

    monkeypatch.setattr(np.lib.format, "open_memmap", record_allocation)
    with pytest.raises(DiskAdmissionError, match="DISK-PREFLIGHT-001"):
        writer.build()
    assert allocations == []
    assert not (index_build.stage_root / "arrays").exists()
    assert not list(index_build.stage_root.glob(".arrays-tmp-*"))

    available = shared_additional
    ingestor.disk_limit_bytes = requirements["shared_peak_bytes"] - 1
    with pytest.raises(DiskAdmissionError, match="DISK-PREFLIGHT-001"):
        writer.build()
    assert allocations == []

    ingestor.disk_limit_bytes = requirements["shared_peak_bytes"]
    result = writer.build()
    completion = json.loads(
        (result.artifact_root / "arrays.complete.json").read_text(
            encoding="utf-8"
        )
    )
    evidence = completion["disk_admission"]
    assert evidence["requirements"] == requirements
    assert evidence["observed"]["final_available_bytes"] == shared_additional
    assert evidence["observed"]["temporary_available_bytes"] == shared_additional


def test_task3_distinct_filesystems_charge_final_and_temporary_boundaries(
    tmp_path: Path,
    monkeypatch: pytest.MonkeyPatch,
) -> None:
    source = _feature_source(tmp_path / "source")
    ingestor = ParquetTypedGraphIngestor(source, tmp_path / "stores")
    index_build = ingestor.build()
    inventory = replace(
        index_build.inventory,
        final_device=101,
        temporary_device=202,
    )
    separated_build = replace(index_build, inventory=inventory)
    writer = arrays_module.TypedGraphArrayWriter(ingestor, separated_build)
    requirements = writer._array_disk_requirements()
    assert requirements["same_filesystem"] is False
    assert requirements["final_peak_bytes"] == (
        requirements["task2_final_bytes"]
        + requirements["estimated_array_bytes"]
    )
    assert requirements["temporary_peak_bytes"] == (
        requirements["snapshot_bytes"] + requirements["spill_bytes"]
    )
    final_available = requirements["final_additional_bytes"] - 1
    temporary_available = requirements["temporary_additional_bytes"]

    def separated_capacity(path: Path) -> tuple[int, int, Path]:
        if path == inventory.final_filesystem_path:
            return 101, final_available, path
        return 202, temporary_available, path

    monkeypatch.setattr(
        ingestion_module,
        "_filesystem_capacity",
        separated_capacity,
    )
    with pytest.raises(DiskAdmissionError, match="final filesystem requires"):
        writer.build()
    assert not (index_build.stage_root / "arrays").exists()

    final_available = requirements["final_additional_bytes"]
    temporary_available = requirements["temporary_additional_bytes"] - 1
    with pytest.raises(
        DiskAdmissionError,
        match="temporary filesystem requires",
    ):
        writer.build()
    assert not (index_build.stage_root / "arrays").exists()

    temporary_available = requirements["temporary_additional_bytes"]
    result = writer.build()
    assert result.artifact_root.is_dir()


@pytest.mark.parametrize(
    ("field", "value"),
    [
        ("node_type", "wrong"),
        ("internal_key", "n9999"),
        ("id_column", "wrong_id"),
        ("id_dtype", "uint64"),
        ("representation", "scalar_columns"),
        ("feature_columns", ["wrong_feature"]),
        ("feature_width", 99),
        ("dtype", "float64"),
        ("arrow_dtype", "double"),
        ("storage_dtype", "<f8"),
        ("shape", [3, 99]),
        ("count", 2),
        ("source_files", []),
        ("source_sha256", "0" * 64),
    ],
)
def test_resume_rejects_resealed_feature_contract_field_tampering(
    tmp_path: Path,
    field: str,
    value: Any,
) -> None:
    source = _feature_source(tmp_path / "source")
    ingestor = ParquetTypedGraphIngestor(source, tmp_path / "stores")
    result = ingestor.build_arrays()
    metadata = _metadata(result.stage_root)
    metadata["nodes"]["n0000"][field] = value
    _reseal_feature_stage(result, metadata)

    with pytest.raises(ArtifactValidationError, match="FEATURE-EVIDENCE-001"):
        ingestor.build_arrays()


def test_resume_rejects_resealed_actual_wrong_feature_dtype(
    tmp_path: Path,
) -> None:
    source = _feature_source(tmp_path / "source")
    ingestor = ParquetTypedGraphIngestor(source, tmp_path / "stores")
    result = ingestor.build_arrays()
    metadata = _metadata(result.stage_root)
    path = result.artifact_root / metadata["nodes"]["n0000"]["relative_path"]
    values = np.asarray(np.load(path, mmap_mode="r"), dtype="<f8")
    record = _replace_feature_record(result, metadata, "n0000", values)
    record["dtype"] = "float64"
    record["arrow_dtype"] = "double"
    record["storage_dtype"] = "<f8"
    _reseal_feature_stage(result, metadata)

    with pytest.raises(ArtifactValidationError, match="FEATURE-EVIDENCE-001"):
        ingestor.build_arrays()


def test_resume_rejects_resealed_scalar_feature_column_substitution(
    tmp_path: Path,
) -> None:
    source = _feature_source(tmp_path / "source")
    ingestor = ParquetTypedGraphIngestor(source, tmp_path / "stores")
    result = ingestor.build_arrays()
    metadata = _metadata(result.stage_root)
    path = result.artifact_root / metadata["nodes"]["n0001"]["relative_path"]
    values = np.array(np.load(path, mmap_mode="r"), copy=True)
    values = values[:, ::-1]
    record = _replace_feature_record(result, metadata, "n0001", values)
    record["feature_columns"] = list(
        reversed(record["feature_columns"])
    )
    _reseal_feature_stage(result, metadata)

    with pytest.raises(ArtifactValidationError, match="FEATURE-EVIDENCE-001"):
        ingestor.build_arrays()


@pytest.mark.parametrize("build_limit", [None, 10**12], ids=("none", "high"))
def test_resume_rechecks_current_disk_limit_against_recorded_peak(
    tmp_path: Path,
    build_limit: int | None,
) -> None:
    source = _feature_source(tmp_path / "source")
    store_root = tmp_path / "stores"
    original = ParquetTypedGraphIngestor(
        source,
        store_root,
        disk_limit_bytes=build_limit,
    )
    index_build = original.build()
    built = arrays_module.TypedGraphArrayWriter(
        original,
        index_build,
    ).build()
    completion = ingestion_module._read_json(
        built.artifact_root / "arrays.complete.json"
    )
    requirements = completion["disk_admission"]["requirements"]
    required_peak = (
        requirements["shared_peak_bytes"]
        if requirements["same_filesystem"]
        else max(
            requirements["final_peak_bytes"],
            requirements["temporary_peak_bytes"],
        )
    )

    too_low = ParquetTypedGraphIngestor(
        source,
        store_root,
        disk_limit_bytes=required_peak - 1,
    )
    with pytest.raises(DiskAdmissionError, match="DISK-PREFLIGHT-001"):
        arrays_module.TypedGraphArrayWriter(too_low, index_build).build()

    exact = ParquetTypedGraphIngestor(
        source,
        store_root,
        disk_limit_bytes=required_peak,
    )
    resumed = arrays_module.TypedGraphArrayWriter(exact, index_build).build()
    assert resumed.resumed is True
    assert resumed.content_sha256 == built.content_sha256
