"""Behavioral tests for bounded typed-Parquet source inventory and staging."""

from __future__ import annotations

import hashlib
from dataclasses import replace
import json
import os
import socket
import subprocess
import sys
import time
from pathlib import Path

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
from topobench.dataloader.input_monitor import InputMonitor
from topobench.profiling.execution_events import ExecutionOperation


def _write_table(path: Path, columns: dict[str, pa.Array]) -> None:
    path.parent.mkdir(parents=True, exist_ok=True)
    pq.write_table(pa.table(columns), path)


def _make_source(
    root: Path,
    *,
    user_fragments: tuple[tuple[str, pa.DataType, list[object | None]], ...] = (
        ("users/b.parquet", pa.int64(), [9, -4]),
        ("users/a.parquet", pa.int64(), [3]),
    ),
    item_fragments: tuple[tuple[str, pa.DataType, list[object | None]], ...] = (
        ("items/z.parquet", pa.uint64(), [2**64 - 1, 7]),
        ("items/a.parquet", pa.uint64(), [0]),
    ),
    memory_limit_bytes: int = 64 * 1024**2,
) -> ParquetTypedGraphSource:
    for relative, dtype, ids in user_fragments:
        _write_table(
            root / relative,
            {
                "node_id": pa.array(ids, type=dtype),
                "feature": pa.array(range(len(ids)), type=pa.float32()),
                "label": pa.array(range(len(ids)), type=pa.int64()),
            },
        )
    for relative, dtype, ids in item_fragments:
        _write_table(
            root / relative,
            {
                "node_id": pa.array(ids, type=dtype),
                "feature": pa.array(range(len(ids)), type=pa.float32()),
            },
        )
    _write_table(
        root / "edges/part.parquet",
        {
            "src": pa.array([-4], type=pa.int64()),
            "dst": pa.array([0], type=pa.uint64()),
        },
    )
    for phase, values in (("train", [-4]), ("val", [3]), ("test", [9])):
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
    spec = ParquetTypedGraphSpec(
        source_root=root,
        output_kind="heterogeneous",
        node_types=(
            NodeTypeSpec(
                name="item",
                paths=tuple(relative for relative, _, _ in item_fragments),
                id_column="node_id",
                id_dtype="uint64",
                feature_columns=("feature",),
                feature_dtype="float32",
                feature_width=1,
            ),
            NodeTypeSpec(
                name="user",
                paths=tuple(relative for relative, _, _ in user_fragments),
                id_column="node_id",
                id_dtype="int64",
                feature_columns=("feature",),
                feature_dtype="float32",
                feature_width=1,
            ),
        ),
        relations=(
            RelationSpec(
                relation=("user", "buys", "item"),
                paths=("edges/part.parquet",),
                source_column="src",
                destination_column="dst",
            ),
        ),
        supervision=SupervisionSpec(
            target_node_type="user",
            label_column="label",
            label_dtype="int64",
            split_registry=SplitRegistrySpec(active_tag="default", sets=(split,)),
        ),
        partition=PartitionSpec(strategy="cluster"),
        ingestion=IngestionLimits(
            record_batch_rows=2,
            memory_limit_bytes=memory_limit_bytes,
            temp_directory="duckdb-tmp",
        ),
    )
    return ParquetTypedGraphSource(spec)


def _sha256(path: Path) -> str:
    digest = hashlib.sha256()
    with path.open("rb") as stream:
        while chunk := stream.read(8192):
            digest.update(chunk)
    return digest.hexdigest()


def test_ingestor_emits_bounded_conversion_stage_events(tmp_path: Path) -> None:
    source = _make_source(tmp_path / "private-source")
    monitor = InputMonitor(event_capacity=16, pending_cuda_capacity=1)
    result = ParquetTypedGraphIngestor(
        source,
        tmp_path / "stores",
        execution_monitor=monitor,
    ).build()
    result.indexes["user"].close()
    result.indexes["item"].close()

    events = monitor.drain()
    assert [
        (event.operation, event.phase)
        for event in events
    ] == [
        (ExecutionOperation.CONVERSION, "inventory"),
        (ExecutionOperation.CONVERSION, "index"),
        (ExecutionOperation.CONVERSION, "publish"),
    ]
    assert all(
        event.row_count == result.inventory.total_rows
        for event in events
    )
    assert all(
        event.unique_storage_bytes == result.inventory.total_bytes
        for event in events
    )
    assert "private-source" not in repr(events)



def test_inventory_is_canonical_exact_and_resource_admitted(tmp_path: Path) -> None:
    """Inventory records stable paths, bytes, rows, exact schemas and peak disk."""
    source = _make_source(tmp_path / "source")
    ingestor = ParquetTypedGraphIngestor(source, tmp_path / "stores")

    inventory = ingestor.inventory()

    assert [entry.relative_path for entry in inventory.files] == sorted(
        str(path.relative_to(source.spec.source_root)) for path in source.files
    )
    assert inventory.total_rows == sum(entry.row_count for entry in inventory.files)
    assert inventory.total_bytes == sum(entry.byte_size for entry in inventory.files)
    assert all(len(entry.schema_fingerprint) == 64 for entry in inventory.files)
    assert all(entry.sha256 == _sha256(entry.absolute_path) for entry in inventory.files)
    assert len({entry.schema_fingerprint for entry in inventory.files}) >= 3
    assert inventory.estimated_final_bytes > 0
    assert inventory.estimated_temporary_bytes > 0
    assert inventory.required_peak_bytes == (
        inventory.estimated_final_bytes + inventory.estimated_temporary_bytes
    )
    assert len(inventory.source_fingerprint) == 64
    assert len(inventory.config_fingerprint) == 64


def test_inventory_rejects_schema_drift_and_unsupported_id_domains(
    tmp_path: Path,
) -> None:
    """All fragments of a type have one exact schema and a supported ID domain."""
    drifted = _make_source(
        tmp_path / "drift",
        user_fragments=(
            ("users/a.parquet", pa.int64(), [1]),
            ("users/b.parquet", pa.int32(), [2]),
        ),
    )
    with pytest.raises(ArtifactValidationError, match="SCHEMA-DRIFT-001"):
        ParquetTypedGraphIngestor(drifted, tmp_path / "stores-a").inventory()

    unsupported = _make_source(tmp_path / "unsupported")
    object.__setattr__(unsupported.spec.node_types[1], "id_dtype", "int32")
    with pytest.raises(ArtifactValidationError, match="ID-DTYPE-001"):
        ParquetTypedGraphIngestor(unsupported, tmp_path / "stores-b").inventory()


def test_inventory_preflight_requires_simultaneous_final_and_temp_space(
    tmp_path: Path,
) -> None:
    """Declared usable disk must admit final output and temporary peak together."""
    source = _make_source(tmp_path / "source")
    baseline = ParquetTypedGraphIngestor(source, tmp_path / "baseline").inventory()
    ingestor = ParquetTypedGraphIngestor(
        source,
        tmp_path / "stores",
        disk_limit_bytes=baseline.required_peak_bytes - 1,
    )

    with pytest.raises(DiskAdmissionError, match="DISK-PREFLIGHT-001"):
        ingestor.inventory()


def test_inventory_owns_configured_spill_subtree_and_records_filesystems(
    tmp_path: Path,
) -> None:
    """DuckDB spills only below the resolved configured temporary directory."""
    source = _make_source(tmp_path / "source")
    ingestor = ParquetTypedGraphIngestor(source, tmp_path / "stores")

    inventory = ingestor.inventory()
    result = ingestor.build_external_node_indexes(inventory)

    assert inventory.final_filesystem_path == ingestor.store_root
    assert inventory.temporary_filesystem_path == (
        source.spec.source_root / "duckdb-tmp"
    )
    assert inventory.snapshot_bytes == inventory.total_bytes
    completion = json.loads((result.stage_root / "build.complete.json").read_text())
    assert completion["disk_admission"] == {
        "final_filesystem_path": str(inventory.final_filesystem_path),
        "final_device": inventory.final_device,
        "temporary_filesystem_path": str(inventory.temporary_filesystem_path),
        "temporary_device": inventory.temporary_device,
        "estimated_final_bytes": inventory.estimated_final_bytes,
        "estimated_temporary_bytes": inventory.estimated_temporary_bytes,
        "required_peak_bytes": inventory.required_peak_bytes,
        "snapshot_bytes": inventory.snapshot_bytes,
    }
    spill_subtree = Path(completion["spill_subtree"])
    assert spill_subtree.is_relative_to(inventory.temporary_filesystem_path)
    assert not spill_subtree.exists()
    snapshot_subtree = Path(completion["snapshot_subtree"])
    assert snapshot_subtree.is_relative_to(inventory.temporary_filesystem_path)
    assert not snapshot_subtree.exists()


@pytest.mark.parametrize("same_filesystem", [True, False])
def test_preflight_checks_actual_final_and_temporary_filesystems(
    tmp_path: Path,
    monkeypatch: pytest.MonkeyPatch,
    same_filesystem: bool,
) -> None:
    """Admission applies one shared peak or independent per-filesystem limits."""
    source = _make_source(tmp_path / "source")
    ingestor = ParquetTypedGraphIngestor(source, tmp_path / "stores")
    inventory = ingestor.inventory()
    final_device = 101
    temporary_device = final_device if same_filesystem else 202
    inventory = replace(
        inventory,
        final_device=final_device,
        temporary_device=temporary_device,
    )

    def capacity(path: Path) -> tuple[int, int, Path]:
        if path == ingestor.store_root:
            available = (
                inventory.required_peak_bytes - 1
                if same_filesystem
                else inventory.estimated_final_bytes - 1
            )
            return final_device, available, path
        return temporary_device, inventory.estimated_temporary_bytes, path

    monkeypatch.setattr(
        ingestion_module,
        "_filesystem_capacity",
        capacity,
        raising=False,
    )
    with pytest.raises(DiskAdmissionError, match="DISK-PREFLIGHT-001"):
        ingestor.build_external_node_indexes(inventory)


def test_disk_estimate_accounts_for_uncompressed_string_id_payload(
    tmp_path: Path,
) -> None:
    """Compressed sources do not hide the external-sort and output disk peak."""
    values = [f"{index:08d}-" + "repeated-value-" * 80 for index in range(200)]
    source = _make_source(
        tmp_path / "source",
        user_fragments=(("users/a.parquet", pa.string(), values),),
    )
    object.__setattr__(source.spec.node_types[1], "id_dtype", "string")

    inventory = ParquetTypedGraphIngestor(source, tmp_path / "stores").inventory()

    payload_bytes = sum(len(value.encode("utf-8")) for value in values)
    assert inventory.estimated_final_bytes >= payload_bytes
    assert inventory.estimated_temporary_bytes >= payload_bytes * 2


def test_mapping_rejects_source_mutation_after_inventory(tmp_path: Path) -> None:
    """Mapping revalidates every inventoried source byte before consuming IDs."""
    source = _make_source(tmp_path / "source")
    ingestor = ParquetTypedGraphIngestor(source, tmp_path / "stores")
    inventory = ingestor.inventory()
    user_path = source.spec.source_root / "users/a.parquet"
    original = pq.read_table(user_path)
    pq.write_table(original.append_column("extra", pa.array([1])), user_path)

    with pytest.raises(SourceMutationError, match="SOURCE-MUTATION-001"):
        ingestor.build_external_node_indexes(inventory)


def test_resume_reopens_checksums_and_rejects_unknown_artifacts(
    tmp_path: Path,
) -> None:
    """A resume is an evidence validation, never a directory-exists cache hit."""
    source = _make_source(tmp_path / "source")
    ingestor = ParquetTypedGraphIngestor(source, tmp_path / "stores")
    inventory = ingestor.inventory()
    first = ingestor.build_external_node_indexes(inventory)
    second = ingestor.build_external_node_indexes(inventory)

    assert second.resumed is True
    assert second.stage_root == first.stage_root
    completion = json.loads((first.stage_root / "build.complete.json").read_text())
    assert completion["behavior_version"]
    assert completion["input_fingerprint"] == inventory.source_fingerprint
    assert completion["config_fingerprint"] == inventory.config_fingerprint
    assert completion["dependency_versions"]["duckdb"]
    assert completion["dependency_versions"]["pyarrow"]
    assert completion["outputs"]
    for relative, expected in completion["outputs"].items():
        assert _sha256(first.stage_root / relative) == expected

    (first.stage_root / "unexpected.bin").write_bytes(b"not declared")
    with pytest.raises(ArtifactValidationError, match="UNKNOWN-ARTIFACT-001"):
        ingestor.build_external_node_indexes(inventory)
    assert not first.stage_root.exists()
    quarantines = list(
        first.stage_root.parent.glob(f".{first.stage_root.name}.quarantine-*")
    )
    assert len(quarantines) == 1

    rebuilt = ingestor.build_external_node_indexes(inventory)
    node_ids = rebuilt.indexes["user"].node_ids_path
    node_ids.write_bytes(node_ids.read_bytes() + b"corrupt")
    with pytest.raises(ArtifactValidationError, match="CHECKSUM-001"):
        ingestor.build_external_node_indexes(inventory)
    assert not rebuilt.stage_root.exists()
    assert len(
        list(rebuilt.stage_root.parent.glob(f".{rebuilt.stage_root.name}.quarantine-*"))
    ) == 2


def test_interrupted_incomplete_stage_is_quarantined_and_rebuilt(
    tmp_path: Path,
) -> None:
    """A killed owner cannot leave an addressed stage blocking every retry."""
    source = _make_source(tmp_path / "source")
    ingestor = ParquetTypedGraphIngestor(source, tmp_path / "stores")
    inventory = ingestor.inventory()
    stage_root = ingestor.stage_root(inventory)
    stage_root.mkdir(parents=True)
    (stage_root / "partial.tmp").write_bytes(b"interrupted")

    result = ingestor.build_external_node_indexes(inventory)

    assert result.resumed is False
    assert (stage_root / "build.complete.json").is_file()
    assert not list(stage_root.parent.glob(f".{stage_root.name}.quarantine-*"))


def test_resume_revalidates_semantic_completion_evidence(tmp_path: Path) -> None:
    """Coordinated checksum edits cannot make changed stage semantics resumable."""
    source = _make_source(tmp_path / "source")
    ingestor = ParquetTypedGraphIngestor(source, tmp_path / "stores")
    result = ingestor.build()
    record_path = result.indexes["user"].completion_path
    record = json.loads(record_path.read_text())
    record["row_count"] += 1
    record_path.write_text(json.dumps(record))
    build_path = result.stage_root / "build.complete.json"
    build = json.loads(build_path.read_text())
    relative = record_path.relative_to(result.stage_root).as_posix()
    build["outputs"][relative] = _sha256(record_path)
    build_path.write_text(json.dumps(build))

    with pytest.raises(ArtifactValidationError, match="COMPLETION-EVIDENCE-001"):
        ingestor.build_external_node_indexes(result.inventory)


def test_resume_audit_cleans_only_its_owned_spill_on_connection_failure(
    tmp_path: Path,
    monkeypatch: pytest.MonkeyPatch,
) -> None:
    """Failed resume setup leaves no run-owned spill subtree behind."""
    import duckdb

    source = _make_source(tmp_path / "source")
    ingestor = ParquetTypedGraphIngestor(source, tmp_path / "stores")
    result = ingestor.build()
    real_connect = duckdb.connect

    def failing_connect(*args, **kwargs):
        if kwargs.get("read_only"):
            raise RuntimeError("resume audit connection failed")
        return real_connect(*args, **kwargs)

    monkeypatch.setattr(duckdb, "connect", failing_connect)
    with pytest.raises(RuntimeError, match="resume audit connection failed"):
        ingestor.build_external_node_indexes(result.inventory)
    owned_parent = (
        result.inventory.temporary_filesystem_path
        / ".topobench-typed-graph-work"
        / result.stage_root.name
    )
    assert not list(owned_parent.iterdir())


def test_lock_rejects_live_owner_and_recovers_only_dead_stale_owner(
    tmp_path: Path,
) -> None:
    """A keyed build lock distinguishes a live owner from dead stale evidence."""
    source = _make_source(tmp_path / "source")
    ingestor = ParquetTypedGraphIngestor(
        source,
        tmp_path / "stores",
        lock_stale_seconds=1.0,
    )
    inventory = ingestor.inventory()
    lock_path = ingestor.lock_path(inventory)
    lock_path.parent.mkdir(parents=True, exist_ok=True)
    stage_root = ingestor.stage_root(inventory)
    stage_root.mkdir(parents=True)
    partial = stage_root / "partial.tmp"
    partial.write_bytes(b"owned by live process")
    lock_path.write_text(
        json.dumps(
            {
                "pid": os.getpid(),
                "hostname": socket.gethostname(),
                "created_ns": time.time_ns() - 10_000_000_000,
                "token": "live",
            }
        )
    )
    with pytest.raises(ConcurrentBuildError, match="BUILD-LOCK-001"):
        ingestor.build_external_node_indexes(inventory)
    assert partial.read_bytes() == b"owned by live process"

    lock_path.write_text(
        json.dumps(
            {
                "pid": 2**30,
                "hostname": socket.gethostname(),
                "created_ns": time.time_ns() - 10_000_000_000,
                "token": "dead",
            }
        )
    )
    result = ingestor.build_external_node_indexes(inventory)
    assert result.stage_root.is_dir()
    assert not lock_path.exists()
    assert not list(stage_root.parent.glob(f".{stage_root.name}.quarantine-*"))


def test_stale_lock_recovery_never_unlinks_a_replacement_owner(
    tmp_path: Path,
    monkeypatch: pytest.MonkeyPatch,
) -> None:
    """Stale-owner inspection cannot race an atomic replacement lock."""
    source = _make_source(tmp_path / "source")
    ingestor = ParquetTypedGraphIngestor(
        source,
        tmp_path / "stores",
        lock_stale_seconds=1.0,
    )
    inventory = ingestor.inventory()
    lock_path = ingestor.lock_path(inventory)
    lock_path.parent.mkdir(parents=True, exist_ok=True)
    lock_path.write_text(
        json.dumps(
            {
                "pid": 2**30,
                "hostname": socket.gethostname(),
                "created_ns": time.time_ns() - 10_000_000_000,
                "token": "stale",
            }
        )
    )

    def replace_owner(_: int) -> bool:
        lock_path.write_text(
            json.dumps(
                {
                    "pid": os.getpid(),
                    "hostname": socket.gethostname(),
                    "created_ns": time.time_ns(),
                    "token": "replacement",
                }
            )
        )
        return False

    monkeypatch.setattr(ingestion_module, "_pid_is_alive", replace_owner)
    with pytest.raises(ConcurrentBuildError, match="BUILD-LOCK-001"):
        ingestor.build_external_node_indexes(inventory)
    assert json.loads(lock_path.read_text())["token"] == "replacement"


def test_store_package_keeps_parquet_dependencies_lazy() -> None:
    """Importing the store surface does not import optional Parquet engines."""
    code = (
        "import sys, topobench; before=set(sys.modules); "
        "import topobench.data.stores; "
        "print(int('duckdb' in sys.modules), "
        "int(any(name.startswith(('duckdb', 'pyarrow')) "
        "for name in set(sys.modules)-before)))"
    )
    output = subprocess.check_output([sys.executable, "-c", code], text=True).strip()
    assert output == "0 0"
