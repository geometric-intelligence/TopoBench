"""End-to-end crash, invalidation, locking, and promotion qualification."""

from __future__ import annotations

import multiprocessing
from dataclasses import replace
import json
import hashlib
import os
from pathlib import Path
import threading
from typing import Any

import pyarrow as pa
import pyarrow.parquet as pq
import pytest

import topobench.data.stores.typed_graph_ingestion as ingestion_module
import topobench.data.stores.typed_graph_store as store_module
from test.data.stores.test_topology_only_pyg_partitioner import (
    asymmetric_typed_source,
    homogeneous_source,
)
from test.data.stores.test_typed_graph_store import _external_partition_map
from topobench.data.loaders.parquet import (
    FittedTransformSpec,
    ParquetTypedGraphSource,
)
from topobench.data.stores.pyg_partitioner import TopologyOnlyPyGPartitioner
from topobench.data.stores.typed_graph_ingestion import (
    ArtifactValidationError,
    ConcurrentBuildError,
    ParquetTypedGraphIngestor,
)
from topobench.data.stores.typed_graph_store import (
    TypedGraphStore,
    TypedGraphStoreBuild,
    TypedGraphStoreWriter,
)
from topobench.data.stores.typed_partition_book import (
    PartitionQualificationLimits,
)


_COMMITTED_SUBTREES = {
    "index": ("mappings",),
    "arrays": ("mappings", "arrays"),
    "relations": ("mappings", "arrays", "relations"),
    "partition": ("mappings", "arrays", "relations", "partitions"),
}
_INVALIDATION_AXES = (
    "source",
    "schema",
    "dependency",
    "split",
    "partition",
    "transform",
    "output",
    "strategy",
)


def _sha256(path: Path) -> str:
    digest = hashlib.sha256()
    with path.open("rb") as stream:
        while chunk := stream.read(1024 * 1024):
            digest.update(chunk)
    return digest.hexdigest()


def _snapshot(root: Path, subtrees: tuple[str, ...]) -> dict[str, str]:
    return {
        path.relative_to(root).as_posix(): _sha256(path)
        for subtree in subtrees
        for path in sorted((root / subtree).rglob("*"))
        if path.is_file() and not path.is_symlink()
    }


def _source(root: Path) -> ParquetTypedGraphSource:
    return asymmetric_typed_source(
        root,
        num_partitions=3,
        memory_limit_bytes=1,
        external_partition_map="external/manifest.json",
    )


def _resume_through_promotion(
    ingestor: ParquetTypedGraphIngestor,
) -> tuple[Any, Any, Any, Any, TypedGraphStoreBuild]:
    inventory = ingestor.inventory()
    indexes = ingestor.build_external_node_indexes(inventory)
    arrays = ingestor.build_arrays(indexes)
    relations = ingestor.build_relations(indexes)
    partitioner = TopologyOnlyPyGPartitioner(ingestor, relations)
    _external_partition_map(partitioner)
    partitions = ingestor.build_partitions(limits=PartitionQualificationLimits())
    store_build = TypedGraphStoreWriter(ingestor, partitions).build()
    return indexes, arrays, relations, partitions, store_build


_ABRUPT_EXIT_CODE = 86


def _publish_then_exit(
    source: ParquetTypedGraphSource,
    store_root: Path,
    boundary: str,
) -> None:
    """Publish one full atomic boundary, then exit without Python cleanup."""
    ingestor = ParquetTypedGraphIngestor(source, store_root)
    indexes = ingestor.build()
    if boundary == "index":
        os._exit(_ABRUPT_EXIT_CODE)
    arrays = ingestor.build_arrays(indexes)
    if boundary == "arrays":
        os._exit(_ABRUPT_EXIT_CODE)
    relations = ingestor.build_relations(indexes)
    if boundary == "relations":
        os._exit(_ABRUPT_EXIT_CODE)
    partitioner = TopologyOnlyPyGPartitioner(ingestor, relations)
    _external_partition_map(partitioner)
    partitions = ingestor.build_partitions(
        limits=PartitionQualificationLimits()
    )
    if boundary == "partition":
        os._exit(_ABRUPT_EXIT_CODE)
    TypedGraphStoreWriter(ingestor, partitions).build()
    os._exit(_ABRUPT_EXIT_CODE)

def _materialize_candidate_then_exit(
    source: ParquetTypedGraphSource,
    store_root: Path,
) -> None:
    """Die with the content lock held after materializing a final candidate."""
    ingestor = ParquetTypedGraphIngestor(source, store_root)
    relations = ingestor.build_relations()
    partitioner = TopologyOnlyPyGPartitioner(ingestor, relations)
    _external_partition_map(partitioner)
    partitions = ingestor.build_partitions(
        limits=PartitionQualificationLimits()
    )
    original = TypedGraphStoreWriter._materialize_candidate

    def materialize_then_exit(
        writer: TypedGraphStoreWriter,
        root: Path,
        reopened: Any,
    ) -> None:
        original(writer, root, reopened)
        os._exit(_ABRUPT_EXIT_CODE)

    TypedGraphStoreWriter._materialize_candidate = materialize_then_exit
    TypedGraphStoreWriter(ingestor, partitions).build()
    raise AssertionError("candidate materialization did not terminate the process")


def _partial_final_copy_then_exit(
    source: ParquetTypedGraphSource,
    store_root: Path,
) -> None:
    """Die during candidate copying without unwinding writer cleanup."""
    ingestor = ParquetTypedGraphIngestor(source, store_root)
    relations = ingestor.build_relations()
    partitioner = TopologyOnlyPyGPartitioner(ingestor, relations)
    _external_partition_map(partitioner)
    partitions = ingestor.build_partitions(
        limits=PartitionQualificationLimits()
    )
    writer = TypedGraphStoreWriter(ingestor, partitions)
    real_copy = writer._copy_file
    copied = 0

    def copy_then_exit(source_path: Path, destination: Path) -> None:
        nonlocal copied
        real_copy(source_path, destination)
        copied += 1
        if copied == 3:
            os._exit(_ABRUPT_EXIT_CODE)

    writer._copy_file = copy_then_exit
    writer.build()
    raise AssertionError("partial candidate copy did not terminate the process")


@pytest.mark.parametrize("boundary", tuple(_COMMITTED_SUBTREES))
def test_clean_restart_reuses_every_checksum_validated_conversion_boundary(
    tmp_path: Path,
    boundary: str,
) -> None:
    """A process may stop after any committed stage without replaying that stage."""
    source = _source(tmp_path / "source")
    store_root = tmp_path / "stores"
    first = ParquetTypedGraphIngestor(source, store_root)
    inventory = first.inventory()
    indexes = first.build_external_node_indexes(inventory)
    stage_root = indexes.stage_root
    result_by_boundary: dict[str, Any] = {"index": indexes}
    if boundary in {"arrays", "relations", "partition"}:
        result_by_boundary["arrays"] = first.build_arrays(indexes)
    if boundary in {"relations", "partition"}:
        result_by_boundary["relations"] = first.build_relations(indexes)
    if boundary == "partition":
        partitioner = TopologyOnlyPyGPartitioner(
            first,
            result_by_boundary["relations"],
        )
        _external_partition_map(partitioner)
        result_by_boundary["partition"] = first.build_partitions(
            limits=PartitionQualificationLimits()
        )

    before = _snapshot(stage_root, _COMMITTED_SUBTREES[boundary])
    restarted = ParquetTypedGraphIngestor(source, store_root)
    resumed = _resume_through_promotion(restarted)
    indexes2, arrays2, relations2, partitions2, published = resumed
    resumed_by_boundary = {
        "index": indexes2,
        "arrays": arrays2,
        "relations": relations2,
        "partition": partitions2,
    }

    assert resumed_by_boundary[boundary].resumed is True
    assert _snapshot(stage_root, _COMMITTED_SUBTREES[boundary]) == before
    assert published.path.name == published.content_sha256
    assert published.cache_hit is False
    with TypedGraphStore.open(published.path) as reopened:
        assert reopened.content_sha256 == published.content_sha256
    published.store.close()

@pytest.mark.parametrize(
    "boundary",
    ("index", "arrays", "relations", "partition", "promotion"),
)
def test_abrupt_process_death_preserves_and_reuses_atomic_boundary(
    tmp_path: Path,
    boundary: str,
) -> None:
    source = _source(tmp_path / "source")
    store_root = tmp_path / "stores"
    ingestor = ParquetTypedGraphIngestor(source, store_root)
    inventory = ingestor.inventory()
    stage_root = ingestor.stage_root(inventory)

    if boundary in {"arrays", "relations", "partition", "promotion"}:
        indexes = ingestor.build_external_node_indexes(inventory)
    if boundary in {"relations", "partition", "promotion"}:
        ingestor.build_arrays(indexes)
    if boundary in {"partition", "promotion"}:
        relations = ingestor.build_relations(indexes)
        partitioner = TopologyOnlyPyGPartitioner(ingestor, relations)
        _external_partition_map(partitioner)
    if boundary == "promotion":
        ingestor.build_partitions(limits=PartitionQualificationLimits())

    context = multiprocessing.get_context("spawn")
    process = context.Process(
        target=_publish_then_exit,
        args=(source, store_root, boundary),
    )
    process.start()
    process.join(timeout=30)
    if process.is_alive():
        process.terminate()
        process.join(timeout=10)
    exit_code = process.exitcode
    process.close()
    assert exit_code == _ABRUPT_EXIT_CODE

    subtrees = _COMMITTED_SUBTREES[
        "partition" if boundary == "promotion" else boundary
    ]
    before = _snapshot(stage_root, subtrees)
    assert before
    indexes2, arrays2, relations2, partitions2, published = (
        _resume_through_promotion(
            ParquetTypedGraphIngestor(source, store_root)
        )
    )
    resumed = {
        "index": indexes2,
        "arrays": arrays2,
        "relations": relations2,
        "partition": partitions2,
    }

    assert _snapshot(stage_root, subtrees) == before
    if boundary == "promotion":
        assert published.cache_hit is True
    else:
        assert resumed[boundary].resumed is True
        assert published.cache_hit is False
    assert published.path.name == published.content_sha256
    assert not list((store_root / ".staging").glob("finalize-*"))
    published.store.close()


def _mutated_source(
    axis: str,
    source: ParquetTypedGraphSource,
    tmp_path: Path,
    monkeypatch: pytest.MonkeyPatch,
) -> ParquetTypedGraphSource:
    spec = source.spec
    if axis == "source":
        path = spec.files[0]
        table = pq.read_table(path)
        column = table.column_names[0]
        values = table[column].to_pylist()
        if pa.types.is_integer(table[column].type):
            values[0] = int(values[0]) + 10_000
        elif pa.types.is_floating(table[column].type):
            values[0] = float(values[0]) + 10_000.0
        else:
            values[0] = str(values[0]) + "-changed"
        pq.write_table(
            table.set_column(
                table.schema.get_field_index(column),
                column,
                pa.array(values, type=table[column].type),
            ),
            path,
        )
        return ParquetTypedGraphSource(spec)
    if axis == "schema":
        changed = next(
            node for node in spec.node_types if node.feature_width > 1
        )
        nodes = tuple(
            replace(
                node,
                feature_columns=tuple(reversed(node.feature_columns)),
            )
            if node.name == changed.name
            else node
            for node in spec.node_types
        )
        return ParquetTypedGraphSource(replace(spec, node_types=nodes))
    if axis == "dependency":
        real_versions = ingestion_module._dependency_versions

        def changed_versions(pyarrow: Any) -> tuple[tuple[str, str], ...]:
            return (*real_versions(pyarrow), ("qualification-probe", "2"))

        monkeypatch.setattr(
            ingestion_module,
            "_dependency_versions",
            changed_versions,
        )
        return source
    if axis == "split":
        registry = spec.supervision.split_registry
        renamed = replace(registry.sets[0], tag="alternate")
        supervision = replace(
            spec.supervision,
            split_registry=replace(
                registry,
                active_tag="alternate",
                sets=(renamed,),
            ),
        )
        return ParquetTypedGraphSource(replace(spec, supervision=supervision))
    if axis == "partition":
        return ParquetTypedGraphSource(
            replace(
                spec,
                partition=replace(spec.partition, num_partitions=2),
            )
        )
    if axis == "transform":
        return ParquetTypedGraphSource(
            replace(spec, fitted_transform=FittedTransformSpec(name="pca"))
        )
    if axis == "strategy":
        return ParquetTypedGraphSource(
            replace(
                spec,
                partition=replace(spec.partition, strategy="neighbor"),
            )
        )
    if axis == "output":
        return homogeneous_source(tmp_path / "homogeneous-source")
    raise AssertionError(f"unknown invalidation axis {axis!r}")


@pytest.mark.parametrize("axis", _INVALIDATION_AXES)
def test_exact_identity_change_never_reuses_an_addressed_stage(
    tmp_path: Path,
    monkeypatch: pytest.MonkeyPatch,
    axis: str,
) -> None:
    baseline_source = _source(tmp_path / "source")
    store_root = tmp_path / "stores"
    baseline_ingestor = ParquetTypedGraphIngestor(baseline_source, store_root)
    baseline_inventory = baseline_ingestor.inventory()
    baseline_stage = baseline_ingestor.stage_root(baseline_inventory)
    baseline_ingestor.build_external_node_indexes(baseline_inventory)

    changed_source = _mutated_source(axis, baseline_source, tmp_path, monkeypatch)
    changed_ingestor = ParquetTypedGraphIngestor(changed_source, store_root)
    changed_inventory = changed_ingestor.inventory()
    changed_stage = changed_ingestor.stage_root(changed_inventory)

    assert changed_stage != baseline_stage
    assert changed_stage.name != baseline_stage.name
    assert not changed_stage.exists()
    changed = changed_ingestor.build_external_node_indexes(changed_inventory)
    assert changed.resumed is False
    assert baseline_stage.is_dir()


def test_failed_final_copy_is_invisible_and_exact_retry_promotes_once(
    tmp_path: Path,
    monkeypatch: pytest.MonkeyPatch,
) -> None:
    ingestor = ParquetTypedGraphIngestor(_source(tmp_path / "source"), tmp_path / "stores")
    indexes = ingestor.build()
    relations = ingestor.build_relations(indexes)
    partitioner = TopologyOnlyPyGPartitioner(ingestor, relations)
    _external_partition_map(partitioner)
    partitions = ingestor.build_partitions(limits=PartitionQualificationLimits())
    stage_before = _snapshot(partitions.stage_root, _COMMITTED_SUBTREES["partition"])
    writer = TypedGraphStoreWriter(ingestor, partitions)
    real_copy = writer._copy_file
    copied = 0

    def crash_during_copy(source: Path, destination: Path) -> None:
        nonlocal copied
        copied += 1
        if copied == 3:
            raise OSError("qualification crash during final copy")
        real_copy(source, destination)

    monkeypatch.setattr(writer, "_copy_file", crash_during_copy)
    with pytest.raises(OSError, match="qualification crash"):
        writer.build()

    assert _snapshot(partitions.stage_root, _COMMITTED_SUBTREES["partition"]) == stage_before
    assert not list(ingestor.store_root.glob("[0-9a-f]" * 64))
    assert not list((ingestor.store_root / ".staging").glob("finalize-*"))

    promoted = TypedGraphStoreWriter(ingestor, partitions).build()
    repeated = TypedGraphStoreWriter(ingestor, partitions).build()
    assert promoted.cache_hit is False
    assert repeated.cache_hit is True
    assert repeated.path == promoted.path
    assert repeated.content_sha256 == promoted.content_sha256
    promoted.store.close()
    repeated.store.close()


def test_final_promotion_lock_covers_validation_collision_and_atomic_replace(
    tmp_path: Path,
    monkeypatch: pytest.MonkeyPatch,
) -> None:
    """A second publisher cannot enter after materialization but before rename."""
    ingestor = ParquetTypedGraphIngestor(_source(tmp_path / "source"), tmp_path / "stores")
    indexes = ingestor.build()
    relations = ingestor.build_relations(indexes)
    partitioner = TopologyOnlyPyGPartitioner(ingestor, relations)
    _external_partition_map(partitioner)
    partitions = ingestor.build_partitions(limits=PartitionQualificationLimits())
    first_writer = TypedGraphStoreWriter(ingestor, partitions)
    second_writer = TypedGraphStoreWriter(ingestor, partitions)
    real_validate = store_module.validate_store
    first_candidate_ready = threading.Event()
    release_first = threading.Event()
    paused = False

    def pause_first_candidate(path: str | Path, **kwargs: Any) -> Any:
        nonlocal paused
        candidate = Path(path)
        if candidate.name.startswith("finalize-") and not paused:
            paused = True
            first_candidate_ready.set()
            assert release_first.wait(timeout=10)
        return real_validate(path, **kwargs)

    monkeypatch.setattr(store_module, "validate_store", pause_first_candidate)
    first_result: list[TypedGraphStoreBuild] = []
    first_errors: list[BaseException] = []

    def publish_first() -> None:
        try:
            first_result.append(first_writer.build())
        except BaseException as error:
            first_errors.append(error)

    thread = threading.Thread(target=publish_first, daemon=True)
    thread.start()
    assert first_candidate_ready.wait(timeout=10)
    try:
        with pytest.raises(ConcurrentBuildError, match="BUILD-LOCK-001"):
            second_writer.build()
    finally:
        release_first.set()
        thread.join(timeout=10)

    assert not thread.is_alive()
    assert first_errors == []
    assert len(first_result) == 1
    assert first_result[0].path.is_dir()
    assert not list((ingestor.store_root / ".staging").glob("finalize-*"))
    first_result[0].store.close()


@pytest.mark.parametrize("entry_point", ("arrays", "relations", "finalization"))
def test_every_public_stage_entry_quarantines_checksum_failure_and_rebinds(
    tmp_path: Path,
    entry_point: str,
) -> None:
    ingestor = ParquetTypedGraphIngestor(
        _source(tmp_path / "source"),
        tmp_path / "stores",
    )
    indexes = ingestor.build()
    partitions = None
    if entry_point == "relations":
        ingestor.build_arrays(indexes)
    elif entry_point == "finalization":
        relations = ingestor.build_relations(indexes)
        partitioner = TopologyOnlyPyGPartitioner(ingestor, relations)
        _external_partition_map(partitioner)
        partitions = ingestor.build_partitions(
            limits=PartitionQualificationLimits()
        )
    mapping = next((indexes.stage_root / "mappings").rglob("node_ids.parquet"))
    os.chmod(mapping, 0o600)
    mapping.write_bytes(mapping.read_bytes() + b"corrupt")

    if entry_point == "arrays":
        result = ingestor.build_arrays(indexes)
    elif entry_point == "relations":
        result = ingestor.build_relations(indexes)
    else:
        assert partitions is not None
        with pytest.raises(
            ArtifactValidationError,
            match="PARTITION-FINGERPRINT-001",
        ):
            TypedGraphStoreWriter(ingestor, partitions).build()
        recovered = ingestor.build_partitions(limits=partitions.limits)
        assert recovered.binding != partitions.binding
        result = TypedGraphStoreWriter(ingestor, recovered).build()
        result.store.close()

    if entry_point != "finalization":
        assert result.stage_root == indexes.stage_root
    quarantines = tuple(
        indexes.stage_root.parent.glob(f".{indexes.stage_root.name}.quarantine-*")
    )
    assert len(quarantines) == 1
    assert not mapping.read_bytes().endswith(b"corrupt")


def test_finalization_validates_task6_evidence_before_reusing_trusted_limits(
    tmp_path: Path,
    monkeypatch: pytest.MonkeyPatch,
) -> None:
    ingestor = ParquetTypedGraphIngestor(
        _source(tmp_path / "source"),
        tmp_path / "stores",
    )
    indexes = ingestor.build()
    relations = ingestor.build_relations(indexes)
    partitioner = TopologyOnlyPyGPartitioner(ingestor, relations)
    _external_partition_map(partitioner)
    trusted_limits = PartitionQualificationLimits()
    partitions = ingestor.build_partitions(limits=trusted_limits)
    qualification_path = partitions.artifact_root / "qualification.json"
    qualification = json.loads(qualification_path.read_text(encoding="utf-8"))
    qualification["limits"]["max_total_size_bytes"] = 10**12
    qualification_path.write_text(
        json.dumps(qualification, sort_keys=True, separators=(",", ":")) + "\n",
        encoding="utf-8",
    )
    observed_limits: list[PartitionQualificationLimits] = []
    real_build_partitions = ingestor.build_partitions

    def capture_limits(
        *,
        limits: PartitionQualificationLimits,
    ) -> Any:
        observed_limits.append(limits)
        return real_build_partitions(limits=limits)

    monkeypatch.setattr(ingestor, "build_partitions", capture_limits)

    promoted = TypedGraphStoreWriter(ingestor, partitions).build()

    assert observed_limits == [trusted_limits]
    assert len(
        tuple(
            partitions.stage_root.glob(".partitions-quarantine-*")
        )
    ) == 1
    promoted.store.close()


@pytest.mark.parametrize("changed_field", ("binding", "evidence"))
def test_finalization_rejects_complete_task6_metadata_mismatch(
    tmp_path: Path,
    monkeypatch: pytest.MonkeyPatch,
    changed_field: str,
) -> None:
    ingestor = ParquetTypedGraphIngestor(
        _source(tmp_path / "source"),
        tmp_path / "stores",
    )
    indexes = ingestor.build()
    relations = ingestor.build_relations(indexes)
    partitioner = TopologyOnlyPyGPartitioner(ingestor, relations)
    _external_partition_map(partitioner)
    partitions = ingestor.build_partitions(
        limits=PartitionQualificationLimits()
    )
    changed = dict(getattr(partitions, changed_field))
    changed["task14_mismatch"] = True
    mismatched = replace(partitions, **{changed_field: changed})
    mapping = next(
        (indexes.stage_root / "mappings").rglob("node_ids.parquet")
    )
    mapping.write_bytes(mapping.read_bytes() + b"corrupt")
    monkeypatch.setattr(
        ingestor,
        "build_partitions",
        lambda *, limits: mismatched,
    )

    with pytest.raises(
        ArtifactValidationError,
        match="PARTITION-FINGERPRINT-001",
    ):
        TypedGraphStoreWriter(ingestor, partitions).build()


@pytest.mark.parametrize(
    "prefix",
    (
        ".partitions-quarantine-probe",
        ".partitions-tmp-probe",
        ".pyg-partition-work/probe",
    ),
)
def test_task6_owned_stage_prefixes_never_become_core_resume_outputs(
    tmp_path: Path,
    prefix: str,
) -> None:
    ingestor = ParquetTypedGraphIngestor(
        _source(tmp_path / "source"),
        tmp_path / "stores",
    )
    indexes = ingestor.build()
    downstream = indexes.stage_root / prefix / "owned.bin"
    downstream.parent.mkdir(parents=True)
    downstream.write_bytes(b"Task6")
    ingestor.build_arrays(indexes)

    resumed = ingestor.build()

    assert resumed.resumed is True
    assert downstream.read_bytes() == b"Task6"


@pytest.mark.parametrize(
    "crash_target",
    (_materialize_candidate_then_exit, _partial_final_copy_then_exit),
    ids=("materialized", "partial-copy"),
)
def test_dead_same_host_lock_and_final_candidate_are_recovered_immediately(
    tmp_path: Path,
    crash_target: Any,
) -> None:
    source = _source(tmp_path / "source")
    store_root = tmp_path / "stores"
    ingestor = ParquetTypedGraphIngestor(source, store_root)
    _, _, _, partitions, initial = _resume_through_promotion(ingestor)
    initial.store.close()
    context = multiprocessing.get_context("spawn")
    process = context.Process(
        target=crash_target,
        args=(source, store_root),
    )
    process.start()
    process.join(timeout=30)
    if process.is_alive():
        process.terminate()
        process.join(timeout=10)
    exit_code = process.exitcode
    process.close()
    assert exit_code == _ABRUPT_EXIT_CODE
    lock_path = ingestor.lock_path(partitions.inventory)
    assert lock_path.is_file()
    candidates = tuple(
        path
        for path in (store_root / ".staging").rglob("finalize-*")
        if path.is_dir()
    )
    assert len(candidates) == 1

    resumed = TypedGraphStoreWriter(ingestor, partitions).build()

    assert resumed.cache_hit is True
    assert not lock_path.exists()
    assert not tuple(
        path
        for path in (store_root / ".staging").rglob("finalize-*")
        if path.is_dir()
    )


def test_corrupt_completed_stage_is_quarantined_without_reusing_bad_bytes(
    tmp_path: Path,
) -> None:
    ingestor = ParquetTypedGraphIngestor(_source(tmp_path / "source"), tmp_path / "stores")
    first = ingestor.build()
    mapping = next((first.stage_root / "mappings").rglob("node_ids.parquet"))
    os.chmod(mapping, 0o600)
    mapping.write_bytes(mapping.read_bytes() + b"corrupt")

    rebuilt = ingestor.build()

    assert rebuilt.resumed is False
    assert rebuilt.stage_root == first.stage_root
    assert rebuilt.indexes
    assert list(first.stage_root.parent.glob(f".{first.stage_root.name}.quarantine-*"))
    assert not mapping.read_bytes().endswith(b"corrupt")
