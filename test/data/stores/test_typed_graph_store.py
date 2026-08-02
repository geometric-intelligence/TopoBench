"""Immutable typed-store publication and lazy read behavior."""

from __future__ import annotations

import hashlib
import json
import os
import shutil
import subprocess
import sys
from dataclasses import dataclass, replace
from pathlib import Path
from typing import Any

import numpy as np
import pytest

from topobench.data.loaders.parquet import ParquetTypedGraphSource
from topobench.data.stores.pyg_partitioner import TopologyOnlyPyGPartitioner
from topobench.data.stores.typed_graph_ingestion import ParquetTypedGraphIngestor
from topobench.data.stores.qualification_checks import QualificationFailure
from topobench.data.stores.typed_graph_store import (
    TypedGraphStore,
    TypedGraphStoreBuild,
    TypedGraphStoreWriter,
)
from topobench.data.stores.typed_partition_book import PartitionQualificationLimits
from test.data.stores.test_topology_only_pyg_partitioner import (
    asymmetric_typed_source,
    homogeneous_source,
)


@dataclass(frozen=True, slots=True)
class QualifiedStoreFixture:
    source: ParquetTypedGraphSource
    ingestor: ParquetTypedGraphIngestor
    partition_build: Any
    store_build: TypedGraphStoreBuild


def _renamed_heterogeneous_source(root: Path) -> ParquetTypedGraphSource:
    source = asymmetric_typed_source(
        root,
        memory_limit_bytes=1,
        external_partition_map="external/manifest.json",
    )
    aliases = {"paper": "entity-kind", "author": "entity.kind"}
    spec = source.spec
    nodes = tuple(
        replace(node, name=aliases[node.name]) for node in spec.node_types
    )
    relations = tuple(
        replace(
            relation,
            relation=(
                aliases[relation.relation[0]],
                relation.relation[1],
                aliases[relation.relation[2]],
            ),
        )
        for relation in spec.relations
    )
    supervision = replace(
        spec.supervision,
        target_node_type=aliases[spec.supervision.target_node_type],
    )
    return ParquetTypedGraphSource(
        replace(
            spec,
            node_types=nodes,
            relations=relations,
            supervision=supervision,
        )
    )


def _external_partition_map(partitioner: TopologyOnlyPyGPartitioner) -> None:
    topology = partitioner.topology_context
    values: list[np.ndarray] = []
    offsets: dict[str, list[int]] = {}
    offset = 0
    for internal_key, node_name in topology.internal_node_types.items():
        count = topology.node_counts[node_name]
        assignment = np.arange(count, dtype=np.int64) % topology.num_partitions
        values.append(assignment)
        offsets[internal_key] = [offset, offset + count]
        offset += count
    combined = np.concatenate(values)
    root = partitioner.ingestor.source.spec.source_root
    assignment_path = root / "external/assignment.npy"
    assignment_path.parent.mkdir(parents=True, exist_ok=True)
    np.save(assignment_path, combined, allow_pickle=False)
    manifest = {
        "format_version": "typed-external-partition-map-v1",
        "topology_fingerprint": topology.fingerprint,
        "num_partitions": topology.num_partitions,
        "node_type_offsets": offsets,
        "assignment": {
            "relative_path": "external/assignment.npy",
            "sha256": hashlib.sha256(assignment_path.read_bytes()).hexdigest(),
            "dtype": "int64",
            "shape": list(combined.shape),
        },
    }
    (root / "external/manifest.json").write_text(
        json.dumps(manifest, sort_keys=True),
        encoding="utf-8",
    )


def _build_qualified_store(
    source: ParquetTypedGraphSource,
    store_root: Path,
) -> QualifiedStoreFixture:
    ingestor = ParquetTypedGraphIngestor(source, store_root)
    partitioner = TopologyOnlyPyGPartitioner(
        ingestor,
        ingestor.build_relations(),
    )
    _external_partition_map(partitioner)
    partition_build = ingestor.build_partitions(
        limits=PartitionQualificationLimits()
    )
    store_build = TypedGraphStoreWriter(ingestor, partition_build).build()
    return QualifiedStoreFixture(source, ingestor, partition_build, store_build)


@pytest.fixture(scope="session")
def qualified_stores(
    tmp_path_factory: pytest.TempPathFactory,
) -> dict[str, QualifiedStoreFixture]:
    root = tmp_path_factory.mktemp("task7-qualified-stores")
    heterogeneous = _build_qualified_store(
        _renamed_heterogeneous_source(root / "heterogeneous-source"),
        root / "heterogeneous-stores",
    )
    homogeneous_source_value = homogeneous_source(root / "homogeneous-source")
    homogeneous_source_value = ParquetTypedGraphSource(
        replace(
            homogeneous_source_value.spec,
            partition=replace(
                homogeneous_source_value.spec.partition,
                num_partitions=2,
                memory_limit_bytes=1,
                external_partition_map="external/manifest.json",
            ),
        )
    )
    homogeneous = _build_qualified_store(
        homogeneous_source_value,
        root / "homogeneous-stores",
    )
    return {"heterogeneous": heterogeneous, "homogeneous": homogeneous}


def test_opens_homogeneous_and_heterogeneous_content_addressed_stores(
    qualified_stores: dict[str, QualifiedStoreFixture],
) -> None:
    for output_kind, fixture in qualified_stores.items():
        build = fixture.store_build
        assert build.path == fixture.ingestor.store_root / build.content_sha256
        assert build.path.name == build.content_sha256
        assert build.cache_hit is False
        assert fixture.partition_build.stage_root.is_dir()
        assert not list(build.path.glob("*.pt"))
        assert not list(build.path.glob("*.pkl"))

        with TypedGraphStore.open(build.path) as store:
            assert store.output_kind == output_kind
            assert store.content_sha256 == build.content_sha256
            assert store.mapped_paths == ()
            assert tuple(store.node_types) == tuple(
                node.name for node in fixture.source.spec.node_types
            )
            assert tuple(store.relation_types) == tuple(
                relation.relation for relation in fixture.source.spec.relations
            )
            assert store.active_split_tag == (
                fixture.source.spec.supervision.split_registry.active_tag
            )


def test_selected_reads_are_bounded_read_only_and_csc_stays_memory_mapped(
    qualified_stores: dict[str, QualifiedStoreFixture],
) -> None:
    fixture = qualified_stores["heterogeneous"]
    store = TypedGraphStore.open(fixture.store_build.path)
    node_type = "entity-kind"
    selected = store.node_features(node_type, np.array([4, 0], dtype=np.int64))
    np.testing.assert_array_equal(
        selected,
        np.array([[5.0, 15.0, 25.0], [1.0, 11.0, 21.0]], dtype=np.float64),
    )
    assert selected.shape == (2, 3)
    assert selected.flags.writeable is False
    full = store.node_features(node_type)
    assert isinstance(full, np.memmap)
    assert full.flags.writeable is False

    relation = ("entity.kind", "writes", "entity-kind")
    row, colptr = store.relation_csc(relation)
    assert isinstance(row, np.memmap)
    assert isinstance(colptr, np.memmap)
    assert row.flags.writeable is False and colptr.flags.writeable is False
    assert colptr[-1] == len(row) == 5
    mapped = store.mapped_paths
    assert all(not Path(path).is_absolute() for path in mapped)
    mmap_objects = [full._mmap, row._mmap, colptr._mmap]
    store.close()
    assert all(item.closed for item in mmap_objects)
    with pytest.raises(RuntimeError, match="closed"):
        store.node_features(node_type)


def test_external_ids_splits_and_partition_identity_round_trip_per_type(
    qualified_stores: dict[str, QualifiedStoreFixture],
    monkeypatch: pytest.MonkeyPatch,
) -> None:
    fixture = qualified_stores["heterogeneous"]
    with TypedGraphStore.open(fixture.store_build.path) as store:
        import pyarrow.parquet as pq

        parquet_file = pq.ParquetFile
        calls = {"row_groups": 0}

        class TrackingParquetFile:
            def __init__(self, *args: object, **kwargs: object) -> None:
                self._inner = parquet_file(*args, **kwargs)

            def __getattr__(self, name: str) -> object:
                return getattr(self._inner, name)

            def read(self, *args: object, **kwargs: object) -> object:
                raise AssertionError("selected external IDs must not read all row groups")

            def read_row_group(self, *args: object, **kwargs: object) -> object:
                calls["row_groups"] += 1
                return self._inner.read_row_group(*args, **kwargs)

        monkeypatch.setattr(pq, "ParquetFile", TrackingParquetFile)
        assert store.external_ids("entity.kind", [3, 0, 3, 2]) == [
            "d",
            "a",
            "d",
            "c",
        ]
        assert calls["row_groups"] > 0
        assert store.external_ids("entity-kind", [4, 0]) == [50, 10]
        np.testing.assert_array_equal(
            store.split_ids("primary", "train"),
            np.array([0, 1], dtype=np.int64),
        )
        assignment = store.partition_assignment("entity.kind")
        permutation = store.partition_permutation("entity.kind")
        inverse = store.partition_inverse_permutation("entity.kind")
        partptr = store.partition_partptr("entity.kind")
        np.testing.assert_array_equal(inverse[permutation], np.arange(len(assignment)))
        np.testing.assert_array_equal(
            np.diff(partptr),
            np.bincount(assignment, minlength=store.num_partitions),
        )
        ownership = store.relation_edge_partition(
            ("entity.kind", "writes", "entity-kind")
        )
        assert len(ownership) == 5
        assert store.partition_book_identity == fixture.partition_build.book.content_identity


def test_lazy_external_ids_reject_same_size_rewrite_after_store_open(
    qualified_stores: dict[str, QualifiedStoreFixture],
    tmp_path: Path,
) -> None:
    fixture = qualified_stores["heterogeneous"]
    root = tmp_path / fixture.store_build.path.name
    shutil.copytree(fixture.store_build.path, root)
    for path in root.rglob("*"):
        if not path.is_symlink():
            os.chmod(path, 0o700 if path.is_dir() else 0o600)
    os.chmod(root, 0o700)
    store = TypedGraphStore.open(root)
    manifest = json.loads((root / "manifest.json").read_text(encoding="utf-8"))
    node = next(
        record
        for record in manifest["nodes"].values()
        if record["name"] == "entity.kind"
    )
    path = root / node["node_ids"]["relative_path"]
    before = path.stat()
    with path.open("r+b") as stream:
        stream.seek(-1, 2)
        original = stream.read(1)
        stream.seek(-1, 2)
        stream.write(bytes([original[0] ^ 1]))
        stream.flush()
        os.fsync(stream.fileno())
    os.utime(path, ns=(before.st_atime_ns, before.st_mtime_ns))
    assert path.stat().st_ctime_ns != before.st_ctime_ns
    with pytest.raises(QualificationFailure) as captured:
        store.external_ids("entity.kind", [0])
    assert captured.value.check_id == "EXTERNAL-ID-SCHEMA-001"
    store.close()


def test_lazy_array_map_rejects_same_size_rewrite_after_store_open(
    qualified_stores: dict[str, QualifiedStoreFixture],
    tmp_path: Path,
) -> None:
    fixture = qualified_stores["homogeneous"]
    root = tmp_path / fixture.store_build.path.name
    shutil.copytree(fixture.store_build.path, root)
    for path in root.rglob("*"):
        if not path.is_symlink():
            os.chmod(path, 0o700 if path.is_dir() else 0o600)
    os.chmod(root, 0o700)
    store = TypedGraphStore.open(root)
    manifest = json.loads((root / "manifest.json").read_text(encoding="utf-8"))
    node = next(iter(manifest["nodes"].values()))
    path = root / node["x"]["relative_path"]
    before = path.stat()
    array = np.load(path, mmap_mode="r+")
    array.flat[0] = array.flat[0] + 1
    array.flush()
    del array
    os.utime(path, ns=(before.st_atime_ns, before.st_mtime_ns))
    assert path.stat().st_ctime_ns != before.st_ctime_ns
    with pytest.raises(QualificationFailure) as captured:
        store.node_features(node["name"], [0])
    assert captured.value.check_id == "ARRAY-CHECKSUM-001"
    store.close()


def test_arbitrary_names_are_aliases_not_filesystem_paths(
    qualified_stores: dict[str, QualifiedStoreFixture],
) -> None:
    fixture = qualified_stores["heterogeneous"]
    manifest = json.loads(
        (fixture.store_build.path / "manifest.json").read_text(encoding="utf-8")
    )
    assert {
        value["name"] for value in manifest["nodes"].values()
    } == {"entity-kind", "entity.kind"}
    assert all(key.startswith("n") for key in manifest["nodes"])
    assert all(key.startswith("r") for key in manifest["relations"])
    assert not (fixture.store_build.path / "nodes/entity-kind").exists()
    assert not (fixture.store_build.path / "nodes/entity.kind").exists()


def test_clean_import_does_not_load_parquet_or_duckdb() -> None:
    script = """
import sys
import topobench.data.stores
from topobench.data.stores.typed_graph_store import TypedGraphStore
from topobench.data.stores.pyg_store import PyGTypedFeatureStore, PyGTypedGraphStore
assert 'duckdb' not in sys.modules
assert 'pyarrow' not in sys.modules
assert TypedGraphStore and PyGTypedFeatureStore and PyGTypedGraphStore
"""
    completed = subprocess.run(
        [sys.executable, "-c", script],
        check=False,
        capture_output=True,
        text=True,
    )
    assert completed.returncode == 0, completed.stderr
