"""Hard typed-partition qualification, fallback, and atomic resume."""

from __future__ import annotations

import hashlib
import json
from pathlib import Path
import stat

import numpy as np
import pytest
from topobench.data.stores import pyg_partitioner as partitioner_module

from topobench.data.stores.pyg_partitioner import TopologyOnlyPyGPartitioner
from topobench.data.stores.typed_graph_ingestion import ArtifactValidationError, ParquetTypedGraphIngestor
from topobench.data.stores.typed_partition_book import PartitionQualificationLimits
from test.data.stores.test_topology_only_pyg_partitioner import asymmetric_typed_source

GOOD_ASSIGNMENTS = {"author": np.array([0, 0, 1, 1], dtype=np.int64), "paper": np.array([0, 1, 0, 1, 1], dtype=np.int64)}


def _partitioner(tmp_path: Path) -> TopologyOnlyPyGPartitioner:
    source = asymmetric_typed_source(tmp_path / "source")
    ingestor = ParquetTypedGraphIngestor(source, tmp_path / "stores")
    return TopologyOnlyPyGPartitioner(
        ingestor,
        ingestor.build_relations(),
    )


def _reseal_json(
    artifact_root: Path,
    relative_path: str,
    value: object,
) -> None:
    path = artifact_root / relative_path
    path.write_text(
        json.dumps(value, sort_keys=True),
        encoding="utf-8",
    )
    completion_path = artifact_root / "partitions.complete.json"
    completion = json.loads(completion_path.read_text(encoding="utf-8"))
    completion["outputs"][relative_path] = hashlib.sha256(
        path.read_bytes()
    ).hexdigest()
    completion_path.write_text(
        json.dumps(completion, sort_keys=True),
        encoding="utf-8",
    )


@pytest.mark.parametrize(("limits", "code"), [
    (PartitionQualificationLimits(max_nodes_per_type={"paper": 2}), "PARTITION-TYPE-BALANCE-001"),
    (PartitionQualificationLimits(max_phase_nodes={"primary": {"train": 1}}), "PARTITION-PHASE-BALANCE-001"),
    (PartitionQualificationLimits(max_edges_per_relation={("author", "writes", "paper"): 2}), "PARTITION-RELATION-BALANCE-001"),
    (PartitionQualificationLimits(max_feature_bytes=79), "PARTITION-FEATURE-BYTES-001"),
    (PartitionQualificationLimits(max_total_size_bytes=1), "PARTITION-TOTAL-SIZE-001"),
    (PartitionQualificationLimits(max_cut_fraction=0.0), "PARTITION-CUT-001"),
    (PartitionQualificationLimits(min_locality=1.0), "PARTITION-CUT-001"),
])
def test_rejects_every_configured_absolute_limit(tmp_path: Path, limits: PartitionQualificationLimits, code: str) -> None:
    with pytest.raises(ArtifactValidationError, match=code):
        _partitioner(tmp_path).adapt_assignments(GOOD_ASSIGNMENTS, limits, backend="external")


def test_empty_partition_is_always_rejected(tmp_path: Path) -> None:
    assignments = {key: np.zeros(len(value), dtype=np.int64) for key, value in GOOD_ASSIGNMENTS.items()}
    with pytest.raises(ArtifactValidationError, match="PARTITION-EMPTY-001"):
        _partitioner(tmp_path).adapt_assignments(assignments, PartitionQualificationLimits(), backend="external")


def test_unqualified_split_tag_is_not_used_for_hard_phase_balance(tmp_path: Path) -> None:
    book = _partitioner(tmp_path).adapt_assignments(GOOD_ASSIGNMENTS, PartitionQualificationLimits(max_phase_nodes={"diagnostic": {"train": 0}}), backend="external")
    check = next(item for item in book.qualification_checks if item.check_id == "PARTITION-PHASE-BALANCE-001")
    assert check.passed and "diagnostic" not in check.observed


def test_memory_and_current_temp_disk_preflight_fail_before_materialization(tmp_path: Path) -> None:
    partitioner = _partitioner(tmp_path)
    with pytest.raises(ArtifactValidationError, match="PARTITION-MEMORY-001"):
        partitioner.preflight(memory_limit_bytes=1)
    with pytest.raises(ArtifactValidationError, match="PARTITION-TEMP-DISK-001"):
        partitioner.preflight(temp_available_bytes=1)
    assert partitioner.materialization_count == 0


def test_backend_is_hard_qualified(tmp_path: Path) -> None:
    with pytest.raises(ArtifactValidationError, match="PARTITION-BACKEND-001"):
        _partitioner(tmp_path).adapt_assignments(GOOD_ASSIGNMENTS, PartitionQualificationLimits(), backend="random")


def _write_external_map(partitioner: TopologyOnlyPyGPartitioner, *, fingerprint: str | None = None) -> None:
    root = partitioner.ingestor.source.spec.source_root
    path = root / "external/assignment.npy"
    path.parent.mkdir(parents=True, exist_ok=True)
    np.save(path, np.concatenate([GOOD_ASSIGNMENTS["author"], GOOD_ASSIGNMENTS["paper"]]), allow_pickle=False)
    manifest = {
        "format_version": "typed-external-partition-map-v1", "topology_fingerprint": fingerprint or partitioner.topology_fingerprint,
        "num_partitions": 2, "node_type_offsets": {"n0000": [0, 4], "n0001": [4, 9]},
        "assignment": {"relative_path": "external/assignment.npy", "sha256": hashlib.sha256(path.read_bytes()).hexdigest(), "dtype": "int64", "shape": [9]},
    }
    (root / "external/manifest.json").write_text(json.dumps(manifest), encoding="utf-8")


def test_overbudget_candidate_uses_checksum_pinned_external_map(tmp_path: Path) -> None:
    source = asymmetric_typed_source(tmp_path / "source", memory_limit_bytes=1, external_partition_map="external/manifest.json")
    ingestor = ParquetTypedGraphIngestor(source, tmp_path / "stores")
    partitioner = TopologyOnlyPyGPartitioner(ingestor, ingestor.build_relations())
    _write_external_map(partitioner)
    result = ingestor.build_partitions(limits=PartitionQualificationLimits())
    assert result.book.backend == "external"
    assert partitioner.materialization_count == 0


def test_external_map_rejects_fingerprint_and_executable_payload(tmp_path: Path) -> None:
    source = asymmetric_typed_source(tmp_path / "source", memory_limit_bytes=1, external_partition_map="external/manifest.json")
    ingestor = ParquetTypedGraphIngestor(source, tmp_path / "stores")
    partitioner = TopologyOnlyPyGPartitioner(ingestor, ingestor.build_relations())
    _write_external_map(partitioner, fingerprint="f" * 64)
    with pytest.raises(ArtifactValidationError, match="PARTITION-FINGERPRINT-001"):
        ingestor.build_partitions(limits=PartitionQualificationLimits())
    manifest_path = source.spec.source_root / "external/manifest.json"
    manifest = json.loads(manifest_path.read_text(encoding="utf-8"))
    manifest["topology_fingerprint"] = partitioner.topology_fingerprint
    manifest["assignment"]["relative_path"] = "external/assignment.pt"
    manifest_path.write_text(json.dumps(manifest), encoding="utf-8")
    (source.spec.source_root / "external/assignment.pt").write_bytes(b"pickle")
    with pytest.raises(ArtifactValidationError, match="PARTITION-EXTERNAL-MAP-001"):
        ingestor.build_partitions(limits=PartitionQualificationLimits())


def test_partition_subtree_publishes_resumes_and_revalidates_limits(tmp_path: Path) -> None:
    source = asymmetric_typed_source(tmp_path / "source", memory_limit_bytes=1, external_partition_map="external/manifest.json")
    ingestor = ParquetTypedGraphIngestor(source, tmp_path / "stores")
    partitioner = TopologyOnlyPyGPartitioner(ingestor, ingestor.build_relations())
    _write_external_map(partitioner)
    first = ingestor.build_partitions(limits=PartitionQualificationLimits())
    second = ingestor.build_partitions(limits=PartitionQualificationLimits())
    assert first.resumed is False and second.resumed is True
    assert first.artifact_root == first.stage_root / "partitions"
    assert (first.artifact_root / "partitions.complete.json").is_file()
    assert not list(first.stage_root.glob(".partitions-tmp-*"))
    with pytest.raises(ArtifactValidationError, match="PARTITION-TYPE-BALANCE-001"):
        ingestor.build_partitions(limits=PartitionQualificationLimits(max_nodes_per_type={"paper": 2}))


def test_tampered_partition_subtree_is_quarantined_and_rebuilt_only(tmp_path: Path) -> None:
    source = asymmetric_typed_source(tmp_path / "source", memory_limit_bytes=1, external_partition_map="external/manifest.json")
    ingestor = ParquetTypedGraphIngestor(source, tmp_path / "stores")
    partitioner = TopologyOnlyPyGPartitioner(ingestor, ingestor.build_relations())
    _write_external_map(partitioner)
    first = ingestor.build_partitions(limits=PartitionQualificationLimits())
    mapping_path = first.stage_root / "mappings/n0000/lookup.duckdb"
    before = hashlib.sha256(mapping_path.read_bytes()).hexdigest()
    (first.artifact_root / "node_types/n0000/assignment.npy").write_bytes(b"tampered")
    rebuilt = ingestor.build_partitions(limits=PartitionQualificationLimits())
    assert rebuilt.resumed is False
    assert hashlib.sha256(mapping_path.read_bytes()).hexdigest() == before
    assert list(first.stage_root.glob(".partitions-quarantine-*"))


def test_partition_publication_reopens_task4_inside_its_lock(
    tmp_path: Path,
) -> None:
    source = asymmetric_typed_source(
        tmp_path / "source",
        memory_limit_bytes=1,
        external_partition_map="external/manifest.json",
    )
    ingestor = ParquetTypedGraphIngestor(source, tmp_path / "stores")
    relations = ingestor.build_relations()
    partitioner = TopologyOnlyPyGPartitioner(ingestor, relations)
    _write_external_map(partitioner)
    metadata = json.loads(
        (relations.artifact_root / "relations.json").read_text(
            encoding="utf-8"
        )
    )
    field = metadata["relations"]["r0000"]["fields"]["weight"]
    (relations.artifact_root / field["relative_path"]).write_bytes(b"tampered")

    with pytest.raises(
        ArtifactValidationError,
        match=r"(?:CHECKSUM|DISK-EVIDENCE)-001",
    ):
        partitioner.build(PartitionQualificationLimits())


def test_resume_rejects_omitted_required_partition_output(
    tmp_path: Path,
) -> None:
    source = asymmetric_typed_source(
        tmp_path / "source",
        memory_limit_bytes=1,
        external_partition_map="external/manifest.json",
    )
    ingestor = ParquetTypedGraphIngestor(source, tmp_path / "stores")
    partitioner = TopologyOnlyPyGPartitioner(
        ingestor,
        ingestor.build_relations(),
    )
    _write_external_map(partitioner)
    first = ingestor.build_partitions(limits=PartitionQualificationLimits())
    completion_path = first.artifact_root / "partitions.complete.json"
    completion = json.loads(completion_path.read_text(encoding="utf-8"))
    completion["outputs"].pop("relations/r0000/edge_partition.npy")
    completion_path.write_text(json.dumps(completion), encoding="utf-8")

    rebuilt = ingestor.build_partitions(limits=PartitionQualificationLimits())

    assert rebuilt.resumed is False
    assert list(first.stage_root.glob(".partitions-quarantine-*"))


def test_resume_rejects_resealed_inconsistent_permutation(
    tmp_path: Path,
) -> None:
    source = asymmetric_typed_source(
        tmp_path / "source",
        memory_limit_bytes=1,
        external_partition_map="external/manifest.json",
    )
    ingestor = ParquetTypedGraphIngestor(source, tmp_path / "stores")
    partitioner = TopologyOnlyPyGPartitioner(
        ingestor,
        ingestor.build_relations(),
    )
    _write_external_map(partitioner)
    first = ingestor.build_partitions(limits=PartitionQualificationLimits())
    permutation_path = (
        first.artifact_root / "node_types/n0000/permutation.npy"
    )
    permutation = np.load(permutation_path, allow_pickle=False)
    np.save(
        permutation_path,
        np.roll(permutation, 1),
        allow_pickle=False,
    )
    completion_path = first.artifact_root / "partitions.complete.json"
    completion = json.loads(completion_path.read_text(encoding="utf-8"))
    completion["outputs"]["node_types/n0000/permutation.npy"] = hashlib.sha256(
        permutation_path.read_bytes()
    ).hexdigest()
    completion_path.write_text(json.dumps(completion), encoding="utf-8")

    rebuilt = ingestor.build_partitions(limits=PartitionQualificationLimits())

    assert rebuilt.resumed is False
    assert list(first.stage_root.glob(".partitions-quarantine-*"))


def test_stale_task6_publish_work_does_not_poison_task4_resume(
    tmp_path: Path,
) -> None:
    source = asymmetric_typed_source(
        tmp_path / "source",
        memory_limit_bytes=1,
        external_partition_map="external/manifest.json",
    )
    ingestor = ParquetTypedGraphIngestor(source, tmp_path / "stores")
    partitioner = TopologyOnlyPyGPartitioner(
        ingestor,
        ingestor.build_relations(),
    )
    _write_external_map(partitioner)
    stale = partitioner.relation_build.stage_root / ".partitions-tmp-dead"
    stale.mkdir()
    (stale / "partial.npy").write_bytes(b"partial")

    result = ingestor.build_partitions(
        limits=PartitionQualificationLimits()
    )

    assert result.book.backend == "external"
    assert not stale.exists()


def test_external_assignment_mutation_during_snapshot_is_rejected(
    tmp_path: Path,
    monkeypatch: pytest.MonkeyPatch,
) -> None:
    source = asymmetric_typed_source(
        tmp_path / "source",
        memory_limit_bytes=1,
        external_partition_map="external/manifest.json",
    )
    ingestor = ParquetTypedGraphIngestor(source, tmp_path / "stores")
    partitioner = TopologyOnlyPyGPartitioner(
        ingestor,
        ingestor.build_relations(),
    )
    _write_external_map(partitioner)
    assignment_path = source.spec.source_root / "external/assignment.npy"
    original_read = partitioner_module.os.read
    mutated = False

    def mutate_during_descriptor_copy(
        descriptor: int,
        size: int,
    ) -> bytes:
        nonlocal mutated
        if not mutated:
            writable = np.load(
                assignment_path,
                mmap_mode="r+",
                allow_pickle=False,
            )
            writable[0] = 1 - writable[0]
            writable.flush()
            del writable
            mutated = True
        return original_read(descriptor, size)

    monkeypatch.setattr(
        partitioner_module.os,
        "read",
        mutate_during_descriptor_copy,
    )

    with pytest.raises(
        ArtifactValidationError,
        match="PARTITION-EXTERNAL-MAP-001",
    ):
        partitioner_module._external_assignments(
            partitioner.topology_context,
            source.spec.source_root,
            "external/manifest.json",
        )


@pytest.mark.parametrize(
    ("limits", "code"),
    [
        (
            PartitionQualificationLimits(
                max_nodes_per_type={"ghost": 1}
            ),
            "PARTITION-TYPE-BALANCE-001",
        ),
        (
            PartitionQualificationLimits(
                max_edges_per_relation={
                    ("ghost", "links", "paper"): 1
                }
            ),
            "PARTITION-RELATION-BALANCE-001",
        ),
        (
            PartitionQualificationLimits(
                max_phase_nodes={"ghost": {"train": 1}}
            ),
            "PARTITION-PHASE-BALANCE-001",
        ),
    ],
)
def test_unknown_limit_keys_have_stable_check_ids(
    tmp_path: Path,
    limits: PartitionQualificationLimits,
    code: str,
) -> None:
    with pytest.raises(ArtifactValidationError, match=code):
        _partitioner(tmp_path).adapt_assignments(
            GOOD_ASSIGNMENTS,
            limits,
            backend="external",
        )


def test_mistyped_phase_limit_has_stable_check_id() -> None:
    with pytest.raises(
        ArtifactValidationError,
        match="PARTITION-PHASE-BALANCE-001",
    ):
        PartitionQualificationLimits(
            max_phase_nodes={"primary": {"trian": 1}}
        )


def test_external_assignment_is_loaded_from_a_pinned_descriptor(
    tmp_path: Path,
    monkeypatch: pytest.MonkeyPatch,
) -> None:
    source = asymmetric_typed_source(
        tmp_path / "source",
        memory_limit_bytes=1,
        external_partition_map="external/manifest.json",
    )
    ingestor = ParquetTypedGraphIngestor(source, tmp_path / "stores")
    partitioner = TopologyOnlyPyGPartitioner(
        ingestor,
        ingestor.build_relations(),
    )
    _write_external_map(partitioner)
    assignment_path = (
        source.spec.source_root / "external/assignment.npy"
    )
    original_load = np.load

    def forbid_path_reopen(
        file: object,
        *args: object,
        **kwargs: object,
    ) -> np.ndarray:
        if isinstance(file, (str, Path)) and Path(file) == assignment_path:
            raise AssertionError("external assignment reopened by pathname")
        return original_load(file, *args, **kwargs)

    monkeypatch.setattr(partitioner_module.np, "load", forbid_path_reopen)

    assignments = partitioner_module._external_assignments(
        partitioner.topology_context,
        source.spec.source_root,
        "external/manifest.json",
    )

    assert set(assignments) == {"author", "paper"}


@pytest.mark.parametrize("nested", [False, True])
def test_external_manifest_rejects_unknown_keys(
    tmp_path: Path,
    nested: bool,
) -> None:
    source = asymmetric_typed_source(
        tmp_path / "source",
        memory_limit_bytes=1,
        external_partition_map="external/manifest.json",
    )
    ingestor = ParquetTypedGraphIngestor(source, tmp_path / "stores")
    partitioner = TopologyOnlyPyGPartitioner(
        ingestor,
        ingestor.build_relations(),
    )
    _write_external_map(partitioner)
    manifest_path = source.spec.source_root / "external/manifest.json"
    manifest = json.loads(manifest_path.read_text(encoding="utf-8"))
    if nested:
        manifest["assignment"]["unexpected"] = True
    else:
        manifest["unexpected"] = True
    manifest_path.write_text(json.dumps(manifest), encoding="utf-8")

    with pytest.raises(
        ArtifactValidationError,
        match="PARTITION-EXTERNAL-MAP-001",
    ):
        partitioner_module._external_assignments(
            partitioner.topology_context,
            source.spec.source_root,
            "external/manifest.json",
        )


def test_partition_publication_fsyncs_arrays_and_directories(
    tmp_path: Path,
    monkeypatch: pytest.MonkeyPatch,
) -> None:
    source = asymmetric_typed_source(
        tmp_path / "source",
        memory_limit_bytes=1,
        external_partition_map="external/manifest.json",
    )
    ingestor = ParquetTypedGraphIngestor(source, tmp_path / "stores")
    partitioner = TopologyOnlyPyGPartitioner(
        ingestor,
        ingestor.build_relations(),
    )
    _write_external_map(partitioner)
    original_fsync = partitioner_module.os.fsync
    regular_calls = 0
    directory_calls = 0

    def observe_fsync(descriptor: int) -> None:
        nonlocal regular_calls, directory_calls
        mode = partitioner_module.os.fstat(descriptor).st_mode
        regular_calls += int(stat.S_ISREG(mode))
        directory_calls += int(stat.S_ISDIR(mode))
        original_fsync(descriptor)

    monkeypatch.setattr(partitioner_module.os, "fsync", observe_fsync)

    ingestor.build_partitions(limits=PartitionQualificationLimits())

    assert regular_calls >= 15
    assert directory_calls >= 8


@pytest.mark.parametrize(
    ("field", "value"),
    [
        ("backend", []),
        ("estimated_resources", "malformed"),
    ],
)
def test_malformed_resealed_partition_identity_is_quarantined(
    tmp_path: Path,
    field: str,
    value: object,
) -> None:
    source = asymmetric_typed_source(
        tmp_path / "source",
        memory_limit_bytes=1,
        external_partition_map="external/manifest.json",
    )
    ingestor = ParquetTypedGraphIngestor(source, tmp_path / "stores")
    partitioner = TopologyOnlyPyGPartitioner(
        ingestor,
        ingestor.build_relations(),
    )
    _write_external_map(partitioner)
    first = ingestor.build_partitions(limits=PartitionQualificationLimits())
    identity = json.loads(
        (first.artifact_root / "partition_book.json").read_text(
            encoding="utf-8"
        )
    )
    identity[field] = value
    _reseal_json(first.artifact_root, "partition_book.json", identity)

    rebuilt = ingestor.build_partitions(
        limits=PartitionQualificationLimits()
    )

    assert rebuilt.resumed is False
    assert list(first.stage_root.glob(".partitions-quarantine-*"))


def test_resealed_qualification_evidence_is_quarantined(
    tmp_path: Path,
) -> None:
    source = asymmetric_typed_source(
        tmp_path / "source",
        memory_limit_bytes=1,
        external_partition_map="external/manifest.json",
    )
    ingestor = ParquetTypedGraphIngestor(source, tmp_path / "stores")
    partitioner = TopologyOnlyPyGPartitioner(
        ingestor,
        ingestor.build_relations(),
    )
    _write_external_map(partitioner)
    first = ingestor.build_partitions(limits=PartitionQualificationLimits())
    qualification = json.loads(
        (first.artifact_root / "qualification.json").read_text(
            encoding="utf-8"
        )
    )
    qualification["checks"][0]["observed"] = {"tampered": True}
    _reseal_json(
        first.artifact_root,
        "qualification.json",
        qualification,
    )

    rebuilt = ingestor.build_partitions(
        limits=PartitionQualificationLimits()
    )

    assert rebuilt.resumed is False
    assert list(first.stage_root.glob(".partitions-quarantine-*"))


def _subtree_bytes(root: Path) -> dict[str, bytes]:
    return {
        path.relative_to(root).as_posix(): path.read_bytes()
        for path in root.rglob("*")
        if path.is_file()
    }


def test_stricter_limit_failure_preserves_prior_subtree_byte_for_byte(
    tmp_path: Path,
) -> None:
    source = asymmetric_typed_source(
        tmp_path / "source",
        memory_limit_bytes=1,
        external_partition_map="external/manifest.json",
    )
    ingestor = ParquetTypedGraphIngestor(source, tmp_path / "stores")
    partitioner = TopologyOnlyPyGPartitioner(
        ingestor,
        ingestor.build_relations(),
    )
    _write_external_map(partitioner)
    first = ingestor.build_partitions(limits=PartitionQualificationLimits())
    before = _subtree_bytes(first.artifact_root)

    with pytest.raises(
        ArtifactValidationError,
        match="PARTITION-TYPE-BALANCE-001",
    ):
        ingestor.build_partitions(
            limits=PartitionQualificationLimits(
                max_nodes_per_type={"paper": 2}
            )
        )

    assert _subtree_bytes(first.artifact_root) == before
    assert not list(first.stage_root.glob(".partitions-quarantine-*"))


def test_passing_limit_change_reuses_assignments_and_republishes_evidence(
    tmp_path: Path,
) -> None:
    source = asymmetric_typed_source(
        tmp_path / "source",
        memory_limit_bytes=1,
        external_partition_map="external/manifest.json",
    )
    ingestor = ParquetTypedGraphIngestor(source, tmp_path / "stores")
    partitioner = TopologyOnlyPyGPartitioner(
        ingestor,
        ingestor.build_relations(),
    )
    _write_external_map(partitioner)
    first = ingestor.build_partitions(limits=PartitionQualificationLimits())
    changed_limits = PartitionQualificationLimits(
        max_nodes_per_type={"paper": 3}
    )

    changed = ingestor.build_partitions(limits=changed_limits)

    assert changed.resumed is True
    assert changed.book.content_identity == first.book.content_identity
    identity = json.loads(
        (changed.artifact_root / "partition_book.json").read_text(
            encoding="utf-8"
        )
    )
    qualification = json.loads(
        (changed.artifact_root / "qualification.json").read_text(
            encoding="utf-8"
        )
    )
    assert identity["limits_fingerprint"] == changed_limits.fingerprint
    assert qualification["limits"] == changed_limits.as_record()
    assert not list(first.stage_root.glob(".partitions-quarantine-*"))


def test_resume_rejects_npz_disguised_as_npy_and_closes_it(
    tmp_path: Path,
    monkeypatch: pytest.MonkeyPatch,
) -> None:
    source = asymmetric_typed_source(
        tmp_path / "source",
        memory_limit_bytes=1,
        external_partition_map="external/manifest.json",
    )
    ingestor = ParquetTypedGraphIngestor(source, tmp_path / "stores")
    partitioner = TopologyOnlyPyGPartitioner(
        ingestor,
        ingestor.build_relations(),
    )
    _write_external_map(partitioner)
    first = ingestor.build_partitions(limits=PartitionQualificationLimits())
    assignment_path = (
        first.artifact_root / "node_types/n0000/assignment.npy"
    )
    with assignment_path.open("wb") as stream:
        np.savez(stream, payload=np.arange(4, dtype=np.int64))
    completion_path = first.artifact_root / "partitions.complete.json"
    completion = json.loads(completion_path.read_text(encoding="utf-8"))
    completion["outputs"][
        "node_types/n0000/assignment.npy"
    ] = hashlib.sha256(assignment_path.read_bytes()).hexdigest()
    completion_path.write_text(json.dumps(completion), encoding="utf-8")
    closed = False
    original_load = partitioner_module.np.load

    def observe_npz_close(*args: object, **kwargs: object) -> object:
        nonlocal closed
        value = original_load(*args, **kwargs)
        if isinstance(value, np.lib.npyio.NpzFile):
            original_close = value.close

            def close() -> None:
                nonlocal closed
                closed = True
                original_close()

            value.close = close
        return value

    monkeypatch.setattr(
        partitioner_module.np,
        "load",
        observe_npz_close,
    )

    rebuilt = ingestor.build_partitions(
        limits=PartitionQualificationLimits()
    )

    assert rebuilt.resumed is False
    assert closed is True
    assert list(first.stage_root.glob(".partitions-quarantine-*"))
