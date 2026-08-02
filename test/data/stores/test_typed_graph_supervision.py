"""Behavioral tests for exact target supervision and named split triplets."""

from __future__ import annotations

import hashlib
import json
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
import topobench.data.stores.typed_graph_arrays as arrays_module
import topobench.data.stores.typed_graph_ingestion as ingestion_module
from topobench.data.stores.typed_graph_ingestion import ArtifactValidationError, ParquetTypedGraphIngestor

_TARGET_IDS = (2**64 - 1, 0, 7, 42, 99)


def _write_table(path: Path, columns: dict[str, pa.Array]) -> None:
    path.parent.mkdir(parents=True, exist_ok=True)
    pq.write_table(pa.table(columns), path)


def _sha256(path: Path) -> str:
    digest = hashlib.sha256()
    with path.open("rb") as stream:
        while chunk := stream.read(8192):
            digest.update(chunk)
    return digest.hexdigest()


def _write_generated_triplet(
    root: Path, *, tag: str, ids: tuple[int, ...], counts: tuple[int, int, int], seed: int
) -> tuple[SplitSetSpec, dict[str, Any]]:
    generator = np.random.Generator(np.random.PCG64(seed))
    shuffled = np.asarray(ids, dtype=np.uint64)[generator.permutation(len(ids))]
    boundaries = np.cumsum((0, *counts))
    phase_ids: dict[str, list[int]] = {}
    paths: dict[str, str] = {}
    for index, phase in enumerate(("train", "val", "test")):
        values = [int(value) for value in shuffled[boundaries[index] : boundaries[index + 1]]]
        relative = f"splits/{phase}_{tag}.parquet"
        _write_table(root / relative, {"target_id": pa.array(values, type=pa.uint64())})
        phase_ids[phase] = values
        paths[phase] = relative
    split = SplitSetSpec(
        tag=tag,
        train=paths["train"],
        val=paths["val"],
        test=paths["test"],
        coverage="complete" if sum(counts) == len(_TARGET_IDS) else "partial",
    )
    return split, {
        "algorithm": "numpy.random.PCG64",
        "version": np.__version__,
        "seed": seed,
        "phase_ids": phase_ids,
    }


def _supervision_source(
    root: Path,
    *,
    label_source: str = "nodes",
    label_dtype: str = "int64",
    node_labels: pa.Array | None = None,
    keyed_ids: pa.Array | None = None,
    keyed_labels: pa.Array | None = None,
    active_tag: str = "alpha",
    label_on_non_target: bool = False,
) -> tuple[ParquetTypedGraphSource, dict[str, dict[str, Any]]]:
    if node_labels is None and label_source == "nodes":
        node_labels = pa.array([2, 0, 1, 1, 2], type=pa.int64())
    target_columns: dict[str, pa.Array] = {
        "target_id": pa.array(_TARGET_IDS, type=pa.uint64()),
        "feature": pa.array([50, 10, 20, 30, 40], type=pa.float32()),
    }
    if label_source == "nodes":
        assert node_labels is not None
        target_columns["label"] = node_labels
    _write_table(root / "nodes/targets.parquet", target_columns)

    other_columns: dict[str, pa.Array] = {
        "other_id": pa.array(["b", "a"], type=pa.string()),
        "feature": pa.array([2.0, 1.0], type=pa.float32()),
    }
    if label_on_non_target:
        other_columns["label"] = pa.array([0, 1], type=pa.int64())
    _write_table(root / "nodes/others.parquet", other_columns)
    _write_table(
        root / "relations/links.parquet",
        {"src": pa.array(["a"], type=pa.string()), "dst": pa.array([0], type=pa.uint64())},
    )

    alpha, alpha_provenance = _write_generated_triplet(
        root, tag="alpha", ids=_TARGET_IDS, counts=(2, 1, 2), seed=20260802
    )
    beta, beta_provenance = _write_generated_triplet(
        root, tag="beta", ids=(0, 7, 42), counts=(1, 1, 1), seed=17
    )

    label_paths: tuple[str, ...] = ()
    label_id_column: str | None = None
    if label_source == "dataset":
        if keyed_ids is None:
            keyed_ids = pa.array([42, 2**64 - 1, 0, 99, 7], type=pa.uint64())
        if keyed_labels is None:
            keyed_labels = pa.array([4.25, 9.5, 0.5, 9.9, 0.75], type=pa.float32())
        _write_table(root / "labels/targets.parquet", {"target_id": keyed_ids, "target": keyed_labels})
        label_paths = ("labels/targets.parquet",)
        label_id_column = "target_id"

    spec = ParquetTypedGraphSpec(
        source_root=root,
        output_kind="heterogeneous",
        node_types=(
            NodeTypeSpec(
                name="target", paths=("nodes/targets.parquet",), id_column="target_id",
                id_dtype="uint64", feature_columns=("feature",), feature_dtype="float32",
                feature_width=1,
            ),
            NodeTypeSpec(
                name="other", paths=("nodes/others.parquet",), id_column="other_id",
                id_dtype="string", feature_columns=("feature",), feature_dtype="float32",
                feature_width=1,
            ),
        ),
        relations=(RelationSpec(
            relation=("other", "links", "target"), paths=("relations/links.parquet",),
            source_column="src", destination_column="dst",
        ),),
        supervision=SupervisionSpec(
            target_node_type="target", label_column="label" if label_source == "nodes" else "target",
            label_dtype=label_dtype,
            split_registry=SplitRegistrySpec(active_tag=active_tag, sets=(beta, alpha)),
            label_source=label_source, label_paths=label_paths, label_id_column=label_id_column,
        ),
        partition=PartitionSpec(strategy="cluster"),
        ingestion=IngestionLimits(record_batch_rows=2, memory_limit_bytes=64 * 1024**2, temp_directory="duckdb-tmp"),
    )
    return ParquetTypedGraphSource(spec), {"alpha": alpha_provenance, "beta": beta_provenance}


def _metadata(stage_root: Path) -> dict[str, Any]:
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
        path.relative_to(root).as_posix(): _sha256(path)
        for subtree in ("nodes", "splits")
        for path in sorted((root / subtree).rglob("*"))
        if path.is_file()
    }
    outputs["arrays.json"] = _sha256(root / "arrays.json")
    return outputs


def _reseal_array_stage(
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


def _replace_array_record(
    result: arrays_module.TypedGraphArrayBuild,
    metadata: dict[str, Any],
    record: dict[str, Any],
    values: np.ndarray,
    *,
    identity_path: tuple[str, ...],
) -> None:
    path = result.artifact_root / record["relative_path"]
    np.save(path, values, allow_pickle=False)
    record["shape"] = list(values.shape)
    record["count"] = len(values)
    record["byte_size"] = path.stat().st_size
    record["file_sha256"] = _sha256(path)
    record["content_sha256"] = arrays_module._array_content_sha(
        path,
        result.record_batch_rows,
    )
    identity: dict[str, Any] = metadata["content_identity"]
    for component in identity_path[:-1]:
        identity = identity[component]
    identity[identity_path[-1]] = record["content_sha256"]
    metadata["content_sha256"] = ingestion_module._sha256_json(
        metadata["content_identity"]
    )
    _reseal_array_stage(result, metadata)


def test_node_classification_and_all_explicit_tags_are_exact_and_compact(tmp_path: Path) -> None:
    source, provenance = _supervision_source(tmp_path / "source")
    result = ParquetTypedGraphIngestor(source, tmp_path / "stores").build_arrays()
    metadata = _metadata(result.stage_root)

    target = metadata["supervision"]
    y = np.load(result.artifact_root / target["relative_path"], mmap_mode="r")
    np.testing.assert_array_equal(y, np.array([0, 1, 1, 2, 2], dtype="<i8"))
    assert y.shape == (5,)
    assert y.dtype.str == "<i8"
    assert target["target_node_type"] == "target"
    assert target["task"] == "classification"
    assert target["vocabulary"] == {
        "kind": "zero_based_contiguous",
        "size": 3,
        "minimum": 0,
        "maximum": 2,
    }
    assert target["source"] == "nodes"
    assert metadata["active_split_tag"] == "alpha"
    assert set(metadata["splits"]) == {"alpha", "beta"}
    seen_by_tag: dict[str, set[int]] = {}
    for tag, split in metadata["splits"].items():
        phase_sets: list[set[int]] = []
        for phase in ("train", "val", "test"):
            phase_metadata = split["phases"][phase]
            ids = np.load(result.artifact_root / phase_metadata["relative_path"], mmap_mode="r")
            assert ids.dtype.str == "<i8"
            assert ids.ndim == 1
            assert np.all(ids[:-1] < ids[1:])
            assert phase_metadata["source_sha256"] == _sha256(
                source.spec.source_root / phase_metadata["source_relative_path"]
            )
            phase_sets.append(set(int(value) for value in ids))
        assert phase_sets[0].isdisjoint(phase_sets[1])
        assert phase_sets[0].isdisjoint(phase_sets[2])
        assert phase_sets[1].isdisjoint(phase_sets[2])
        seen_by_tag[tag] = set.union(*phase_sets)
        assert split["pairwise_disjoint"] is True
        assert split["union_count"] == sum(len(values) for values in phase_sets)
    assert len(seen_by_tag["alpha"]) == len(_TARGET_IDS)
    assert len(seen_by_tag["beta"]) == 3
    assert seen_by_tag["alpha"] & seen_by_tag["beta"]
    assert metadata["splits"]["alpha"]["coverage"] == "complete"
    assert metadata["splits"]["beta"]["coverage"] == "partial"
    assert provenance["alpha"]["algorithm"] == "numpy.random.PCG64"
    assert provenance["alpha"]["version"] == np.__version__
    assert provenance["alpha"]["seed"] == 20260802
    assert not list(result.stage_root.rglob("*mask*"))


def test_keyed_regression_is_joined_by_uint64_id_not_row_position(tmp_path: Path) -> None:
    source, _ = _supervision_source(tmp_path / "source", label_source="dataset", label_dtype="float32")
    result = ParquetTypedGraphIngestor(source, tmp_path / "stores").build_arrays()
    target = _metadata(result.stage_root)["supervision"]
    y = np.load(result.artifact_root / target["relative_path"], mmap_mode="r")
    np.testing.assert_array_equal(y, np.array([0.5, 0.75, 4.25, 9.9, 9.5], dtype="<f4"))
    assert target["task"] == "regression"
    assert target["source"] == "dataset"
    assert target["join"] == "external_id"
    assert target["resolved_count"] == len(_TARGET_IDS)
    assert target["source_sha256"] == _sha256(source.spec.source_root / "labels/targets.parquet")


def test_locally_generated_split_fixtures_are_deterministic_without_global_rng(tmp_path: Path) -> None:
    before = np.random.get_state()
    source_a, provenance_a = _supervision_source(tmp_path / "a/source")
    middle = np.random.get_state()
    source_b, provenance_b = _supervision_source(tmp_path / "b/source")
    after = np.random.get_state()
    assert before[0] == middle[0] == after[0]
    assert np.array_equal(before[1], middle[1]) and np.array_equal(before[1], after[1])
    assert before[2:] == middle[2:] == after[2:]
    assert provenance_a == provenance_b
    result_a = ParquetTypedGraphIngestor(source_a, tmp_path / "a/stores").build_arrays()
    result_b = ParquetTypedGraphIngestor(source_b, tmp_path / "b/stores").build_arrays()
    assert result_a.inventory.source_fingerprint == result_b.inventory.source_fingerprint
    assert result_a.content_sha256 == result_b.content_sha256
    assert _metadata(result_a.stage_root) == _metadata(result_b.stage_root)


def test_active_tag_is_stable_authoritative_content_identity(tmp_path: Path) -> None:
    source_a, _ = _supervision_source(tmp_path / "a/source", active_tag="alpha")
    source_b, _ = _supervision_source(tmp_path / "b/source", active_tag="beta")
    result_a = ParquetTypedGraphIngestor(source_a, tmp_path / "a/stores").build_arrays()
    result_b = ParquetTypedGraphIngestor(source_b, tmp_path / "b/stores").build_arrays()
    assert result_a.active_tag == "alpha" and result_b.active_tag == "beta"
    assert result_a.content_sha256 != result_b.content_sha256
    assert result_a.inventory.config_fingerprint != result_b.inventory.config_fingerprint


@pytest.mark.parametrize(
    ("keyed_ids", "keyed_labels", "label_dtype", "message"),
    [
        (pa.array([42, 2**64 - 1, 0, 99, 99], type=pa.uint64()), pa.array([4.25, 9.5, 0.5, 9.9, 8.8], type=pa.float32()), "float32", "SUPERVISION-DUPLICATE-001"),
        (pa.array([42, 0, 99, 7], type=pa.uint64()), pa.array([4.25, 0.5, 9.9, 0.75], type=pa.float32()), "float32", "SUPERVISION-MISSING-001"),
        (pa.array([42, 2**64 - 1, 0, 99, 7, 1234], type=pa.uint64()), pa.array([4.25, 9.5, 0.5, 9.9, 0.75, 1.0], type=pa.float32()), "float32", "SUPERVISION-EXTRA-001"),
        (pa.array([42, 43, 0, 99, 7], type=pa.int64()), pa.array([4.25, 9.5, 0.5, 9.9, 0.75], type=pa.float32()), "float32", "SUPERVISION-ID-CAST-001"),
        (pa.array([42, 2**64 - 1, 0, 99, 7], type=pa.uint64()), pa.array([4.25, 9.5, None, 9.9, 0.75], type=pa.float32()), "float32", "TARGET-NULL-001"),
        (pa.array([42, 2**64 - 1, 0, 99, 7], type=pa.uint64()), pa.array([4.25, 9.5, float("nan"), 9.9, 0.75], type=pa.float32()), "float32", "TARGET-FINITE-001"),
        (pa.array([42, 2**64 - 1, 0, 99, 7], type=pa.uint64()), pa.array([4.25, 9.5, float("inf"), 9.9, 0.75], type=pa.float32()), "float32", "TARGET-FINITE-001"),
        (pa.array([42, 2**64 - 1, 0, 99, 7], type=pa.uint64()), pa.array([4, 9, 0, 9, 0], type=pa.int64()), "float32", "TARGET-CAST-001"),
    ],
    ids=("duplicate", "missing", "extra", "uint64-coercion", "null-regression", "nan-regression", "inf-regression", "target-cast"),
)
def test_rejects_invalid_keyed_supervision_without_partial_arrays(
    tmp_path: Path, keyed_ids: pa.Array, keyed_labels: pa.Array, label_dtype: str, message: str
) -> None:
    source, _ = _supervision_source(tmp_path / "source", label_source="dataset", label_dtype=label_dtype, keyed_ids=keyed_ids, keyed_labels=keyed_labels)
    ingestor = ParquetTypedGraphIngestor(source, tmp_path / "stores")
    with pytest.raises(ArtifactValidationError, match=message):
        ingestor.build_arrays()
    stage_root = ingestor.stage_root(ingestor.inventory())
    assert (stage_root / "build.complete.json").is_file()
    assert not (stage_root / "arrays.complete.json").exists()
    assert not (stage_root / "nodes").exists() and not (stage_root / "splits").exists()
    assert not (stage_root / "arrays").exists()


@pytest.mark.parametrize(
    ("labels", "label_dtype", "message"),
    [
        (pa.array([0, -1, 1, 2, 2], type=pa.int64()), "int64", "TARGET-VOCABULARY-001"),
        (pa.array([0, 2, 2, 2, 2], type=pa.int64()), "int64", "TARGET-VOCABULARY-001"),
        (pa.array([0, 1, None, 2, 2], type=pa.int64()), "int64", "TARGET-NULL-001"),
        (pa.array([0.0, 1.0, 1.0, 2.0, 2.0], type=pa.float64()), "int64", "TARGET-CAST-001"),
        (pa.array(["a", "b", "b", "c", "c"], type=pa.string()), "string", "TARGET-DTYPE-001"),
    ],
    ids=("negative-class", "gapped-class", "null-class", "non-integer-class", "unqualified-string"),
)
def test_rejects_malformed_node_classification_targets(tmp_path: Path, labels: pa.Array, label_dtype: str, message: str) -> None:
    source, _ = _supervision_source(tmp_path / "source", node_labels=labels, label_dtype=label_dtype)
    with pytest.raises(ArtifactValidationError, match=message):
        ParquetTypedGraphIngestor(source, tmp_path / "stores").build_arrays()


def test_labels_are_accepted_only_from_the_declared_target_type(
    tmp_path: Path,
    monkeypatch: pytest.MonkeyPatch,
) -> None:
    source, _ = _supervision_source(
        tmp_path / "source",
        label_on_non_target=True,
    )

    def reject_any_array_write(*args: Any, **kwargs: Any) -> None:
        raise AssertionError("label ownership must be preflighted before writes")

    monkeypatch.setattr(
        np.lib.format,
        "open_memmap",
        reject_any_array_write,
    )
    with pytest.raises(
        ArtifactValidationError,
        match="TARGET-OWNERSHIP-001.*other.*label",
    ):
        ParquetTypedGraphIngestor(
            source,
            tmp_path / "stores",
        ).build_arrays()


def test_dataset_labels_cannot_request_an_implicit_positional_join() -> None:
    split = SplitSetSpec(tag="default", train="train.parquet", val="val.parquet", test="test.parquet", coverage="partial")
    with pytest.raises(ValueError, match="node_id is required"):
        SupervisionSpec(
            target_node_type="target", label_column="label", label_dtype="int64",
            label_source="dataset", label_paths=("labels.parquet",), label_id_column=None,
            split_registry=SplitRegistrySpec(active_tag="default", sets=(split,)),
        )


@pytest.mark.parametrize(
    ("phase", "values", "dtype", "message"),
    [
        ("train", [0, 0], pa.uint64(), "SPLIT-DUPLICATE-001"),
        ("val", [0], pa.uint64(), "SPLIT-DISJOINT-001"),
        ("test", [1234], pa.uint64(), "SPLIT-UNRESOLVED-001"),
        ("test", [99], pa.uint64(), "SPLIT-COVERAGE-001"),
        ("val", [None], pa.uint64(), "SPLIT-NULL-001"),
        ("val", [0], pa.int64(), "SPLIT-ID-CAST-001"),
    ],
    ids=("duplicate", "phase-overlap", "unresolved", "incomplete-complete", "null-id", "id-cast"),
)
def test_rejects_invalid_explicit_split_phase(tmp_path: Path, phase: str, values: list[int | None], dtype: pa.DataType, message: str) -> None:
    source, _ = _supervision_source(tmp_path / "source")
    path = source.spec.source_root / f"splits/{phase}_alpha.parquet"
    _write_table(path, {"target_id": pa.array(values, type=dtype)})
    with pytest.raises(ArtifactValidationError, match=message):
        ParquetTypedGraphIngestor(source, tmp_path / "stores").build_arrays()


def test_missing_registered_split_phase_is_a_hard_failure(tmp_path: Path) -> None:
    source, _ = _supervision_source(tmp_path / "source")
    (source.spec.source_root / "splits/test_alpha.parquet").unlink()
    with pytest.raises(ArtifactValidationError, match="SOURCE-MISSING-001"):
        ParquetTypedGraphIngestor(source, tmp_path / "stores").build_arrays()


@pytest.mark.parametrize(
    ("field", "value"),
    [
        ("target_node_type", "other"),
        ("target_internal_key", "n0000"),
        ("task", "regression"),
        ("source", "dataset"),
        ("join", "positional"),
        ("id_column", "wrong_id"),
        ("label_column", "wrong_label"),
        ("dtype", "float32"),
    ],
)
def test_resume_rejects_self_consistent_supervision_binding_tampering(
    tmp_path: Path,
    field: str,
    value: str,
) -> None:
    source, _ = _supervision_source(tmp_path / "source")
    ingestor = ParquetTypedGraphIngestor(source, tmp_path / "stores")
    result = ingestor.build_arrays()
    metadata = _metadata(result.stage_root)
    metadata["supervision"][field] = value
    _reseal_array_stage(result, metadata)

    with pytest.raises(ArtifactValidationError, match="TARGET-EVIDENCE-001"):
        ingestor.build_arrays()


def test_resume_revalidates_classification_vocabulary_from_target_values(
    tmp_path: Path,
) -> None:
    source, _ = _supervision_source(tmp_path / "source")
    ingestor = ParquetTypedGraphIngestor(source, tmp_path / "stores")
    result = ingestor.build_arrays()
    metadata = _metadata(result.stage_root)
    record = metadata["supervision"]
    _replace_array_record(
        result,
        metadata,
        record,
        np.array([0, 2, 2, 2, 2], dtype="<i8"),
        identity_path=("supervision",),
    )

    with pytest.raises(ArtifactValidationError, match="TARGET-VOCABULARY-001"):
        ingestor.build_arrays()


def test_resume_rejects_self_consistent_incomplete_target_shape(
    tmp_path: Path,
) -> None:
    source, _ = _supervision_source(tmp_path / "source")
    ingestor = ParquetTypedGraphIngestor(source, tmp_path / "stores")
    result = ingestor.build_arrays()
    metadata = _metadata(result.stage_root)
    record = metadata["supervision"]
    _replace_array_record(
        result,
        metadata,
        record,
        np.array([0, 1, 1, 2], dtype="<i8"),
        identity_path=("supervision",),
    )

    with pytest.raises(ArtifactValidationError, match="TARGET-EVIDENCE-001"):
        ingestor.build_arrays()


@pytest.mark.parametrize(
    ("scope", "field", "value"),
    [
        ("tag", "coverage", "partial"),
        ("tag", "qualified", False),
        ("tag", "supervision_population", 999),
        ("tag", "union_count", 999),
        ("phase", "source_relative_path", "splits/test_beta.parquet"),
        ("phase", "source_sha256", "0" * 64),
        ("phase", "resolved_count", 999),
        ("phase", "dtype", "uint64"),
        ("phase", "target_internal_key", "n0000"),
    ],
)
def test_resume_rejects_self_consistent_split_binding_tampering(
    tmp_path: Path,
    scope: str,
    field: str,
    value: Any,
) -> None:
    source, _ = _supervision_source(tmp_path / "source")
    ingestor = ParquetTypedGraphIngestor(source, tmp_path / "stores")
    result = ingestor.build_arrays()
    metadata = _metadata(result.stage_root)
    target = (
        metadata["splits"]["alpha"]
        if scope == "tag"
        else metadata["splits"]["alpha"]["phases"]["train"]
    )
    target[field] = value
    _reseal_array_stage(result, metadata)

    with pytest.raises(ArtifactValidationError, match="SPLIT-EVIDENCE-001"):
        ingestor.build_arrays()


def test_resume_rejects_out_of_range_complete_split_with_resealed_checksums(
    tmp_path: Path,
) -> None:
    source, _ = _supervision_source(tmp_path / "source")
    ingestor = ParquetTypedGraphIngestor(source, tmp_path / "stores")
    result = ingestor.build_arrays()
    metadata = _metadata(result.stage_root)
    record = metadata["splits"]["alpha"]["phases"]["train"]
    path = result.artifact_root / record["relative_path"]
    values = np.array(np.load(path, mmap_mode="r"), dtype="<i8")
    values[-1] = len(_TARGET_IDS) + 10
    values.sort()
    _replace_array_record(
        result,
        metadata,
        record,
        values,
        identity_path=("splits", "alpha", "train"),
    )

    with pytest.raises(ArtifactValidationError, match="SPLIT-RANGE-001"):
        ingestor.build_arrays()
