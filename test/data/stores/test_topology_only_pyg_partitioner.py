"""Topology-only PyG partition generation and trusted artifact adaptation."""

from __future__ import annotations

import json
from pathlib import Path
from typing import Any

import numpy as np
import pyarrow as pa
import pyarrow.parquet as pq
import pytest
import torch
from torch_geometric.data import HeteroData
from torch_geometric.distributed import Partitioner
from torch_geometric.distributed.utils import as_str

from topobench.data.loaders.parquet import (
    IngestionLimits, NodeTypeSpec, ParquetTypedGraphSource, ParquetTypedGraphSpec,
    PartitionSpec, RelationSpec, SplitRegistrySpec, SplitSetSpec, SupervisionSpec,
)
from topobench.data.stores.pyg_partitioner import (
    CanonicalRelationTopology,
    PyGPartitionArtifactAdapter,
    TopologyOnlyPyGPartitioner,
    _partition_worker,
)
from topobench.data.stores.typed_graph_ingestion import ArtifactValidationError, ParquetTypedGraphIngestor
from topobench.data.stores.typed_partition_book import PartitionQualificationLimits


def _write_table(path: Path, columns: dict[str, pa.Array]) -> None:
    path.parent.mkdir(parents=True, exist_ok=True)
    pq.write_table(pa.table(columns), path)


def asymmetric_typed_source(
    root: Path, *, memory_limit_bytes: int = 256 * 1024**3,
    external_partition_map: str | None = None, num_partitions: int = 2,
) -> ParquetTypedGraphSource:
    """Create unequal-width typed data with exact and missing reverse topology."""
    _write_table(root / "nodes/authors.parquet", {
        "author_id": pa.array(["d", "a", "c", "b"], type=pa.string()),
        "f0": pa.array([4.0, 1.0, 3.0, 2.0], type=pa.float32()),
        "label": pa.array([3, 0, 2, 1], type=pa.int64()),
    })
    _write_table(root / "nodes/papers.parquet", {
        "paper_id": pa.array([50, 10, 40, 20, 30], type=pa.int64()),
        "f0": pa.array([5, 1, 4, 2, 3], type=pa.float64()),
        "f1": pa.array([15, 11, 14, 12, 13], type=pa.float64()),
        "f2": pa.array([25, 21, 24, 22, 23], type=pa.float64()),
    })
    writes_src = ["a", "a", "b", "c", "d"]
    writes_dst = [10, 10, 20, 40, 50]
    _write_table(root / "relations/writes.parquet", {
        "src": pa.array(writes_src, type=pa.string()),
        "dst": pa.array(writes_dst, type=pa.int64()),
        "edge_id": pa.array([101, 102, 103, 104, 105], type=pa.int64()),
        "weight": pa.array([1, 2, 3, 4, 5], type=pa.float32()),
        "vector": pa.array([[1, -1], [2, -2], [3, -3], [4, -4], [5, -5]], type=pa.list_(pa.int16(), 2)),
    })
    _write_table(root / "relations/written_by.parquet", {
        "src": pa.array(list(reversed(writes_dst)), type=pa.int64()),
        "dst": pa.array(list(reversed(writes_src)), type=pa.string()),
        "edge_id": pa.array([205, 204, 203, 202, 201], type=pa.int64()),
    })
    _write_table(root / "relations/cites.parquet", {
        "src": pa.array([10, 20, 20, 40], type=pa.int64()),
        "dst": pa.array([20, 10, 20, 50], type=pa.int64()),
        "edge_id": pa.array([301, 302, 303, 304], type=pa.int64()),
    })
    split_sets: list[SplitSetSpec] = []
    split_ids = {
        "primary": {"train": ["a", "b"], "val": ["c"], "test": ["d"]},
        "diagnostic": {"train": ["d"], "val": ["a"], "test": ["b"]},
    }
    for tag, phases in split_ids.items():
        paths: dict[str, str] = {}
        for phase, ids in phases.items():
            relative = f"splits/{tag}-{phase}.parquet"
            _write_table(root / relative, {"author_id": pa.array(ids, type=pa.string())})
            paths[phase] = relative
        split_sets.append(SplitSetSpec(
            tag=tag, train=paths["train"], val=paths["val"], test=paths["test"],
            coverage="complete" if tag == "primary" else "partial", qualified=tag == "primary",
        ))
    return ParquetTypedGraphSource(ParquetTypedGraphSpec(
        source_root=root, output_kind="heterogeneous",
        node_types=(
            NodeTypeSpec(name="paper", paths=("nodes/papers.parquet",), id_column="paper_id", id_dtype="int64", feature_columns=("f0", "f1", "f2"), feature_dtype="float64", feature_width=3),
            NodeTypeSpec(name="author", paths=("nodes/authors.parquet",), id_column="author_id", id_dtype="string", feature_columns=("f0",), feature_dtype="float32", feature_width=1),
        ),
        relations=(
            RelationSpec(relation=("author", "writes", "paper"), paths=("relations/writes.parquet",), source_column="src", destination_column="dst", edge_id_column="edge_id", edge_fields=("weight", "vector")),
            RelationSpec(relation=("paper", "written_by", "author"), paths=("relations/written_by.parquet",), source_column="src", destination_column="dst", edge_id_column="edge_id"),
            RelationSpec(relation=("paper", "cites", "paper"), paths=("relations/cites.parquet",), source_column="src", destination_column="dst", edge_id_column="edge_id"),
        ),
        supervision=SupervisionSpec(target_node_type="author", label_column="label", label_dtype="int64", split_registry=SplitRegistrySpec(active_tag="primary", sets=tuple(split_sets))),
        partition=PartitionSpec(strategy="cluster", num_partitions=num_partitions, memory_limit_bytes=memory_limit_bytes, external_partition_map=external_partition_map),
        ingestion=IngestionLimits(record_batch_rows=2, memory_limit_bytes=64 * 1024**2, temp_directory="duckdb-tmp"),
    ))


def homogeneous_source(root: Path) -> ParquetTypedGraphSource:
    from test.data.stores.test_typed_graph_csc import _homogeneous_source
    return _homogeneous_source(root)


def _build_partitioner(tmp_path: Path) -> tuple[TopologyOnlyPyGPartitioner, Any]:
    source = asymmetric_typed_source(tmp_path / "source")
    ingestor = ParquetTypedGraphIngestor(source, tmp_path / "stores")
    relations = ingestor.build_relations()
    return TopologyOnlyPyGPartitioner(ingestor, relations), relations


def test_materializes_only_internal_topology_and_confined_synthetic_reverse(tmp_path: Path) -> None:
    partitioner, relations = _build_partitioner(tmp_path)
    data = partitioner.materialize_topology()
    assert isinstance(data, HeteroData)
    assert data.node_types == ["n0000", "n0001"]
    assert all(set(data[node_type].keys()) == {"num_nodes"} for node_type in data.node_types)
    assert all(set(data[edge_type].keys()) == {"edge_index"} for edge_type in data.edge_types)
    assert not any(key in data for key in ("x", "y", "train_mask", "edge_attr"))
    assert set(partitioner.canonical_edge_types) == {("n0000", "r0000", "n0001"), ("n0001", "r0001", "n0001"), ("n0001", "r0002", "n0000")}
    assert set(partitioner.synthetic_edge_types) == {("n0001", "srev_r0001", "n0001")}
    assert not list(relations.artifact_root.glob("*srev*"))

    synthetic_edges = data[("n0001", "srev_r0001", "n0001")].edge_index
    assert synthetic_edges.shape == (2, 1)
    assert synthetic_edges[:, 0].tolist() == [4, 3]

def test_real_pyg_generation_adapts_and_cleans_trusted_output(tmp_path: Path) -> None:
    partitioner, _ = _build_partitioner(tmp_path)
    book = partitioner.generate(PartitionQualificationLimits())
    assert book.num_partitions == 2
    assert book.backend == "pyg"
    assert set(book.node_assignments) == {"author", "paper"}
    assert set(book.edge_ownership) == {("author", "writes", "paper"), ("paper", "cites", "paper"), ("paper", "written_by", "author")}
    assert all(check.passed for check in book.qualification_checks)
    assert partitioner.last_work_root is not None and not partitioner.last_work_root.exists()
    assert not (
        partitioner.relation_build.stage_root / ".pyg-partition-work"
    ).exists()


def test_trusted_adapter_streams_csc_and_mmaps_tensor_storage(
    tmp_path: Path,
    monkeypatch: pytest.MonkeyPatch,
) -> None:
    partitioner, _ = _build_partitioner(tmp_path)
    original_load = torch.load
    mmap_modes: list[Any] = []

    def checked_load(*args: Any, **kwargs: Any) -> Any:
        mmap_modes.append(kwargs.get("mmap"))
        return original_load(*args, **kwargs)

    def destination_must_not_materialize(
        _relation: CanonicalRelationTopology,
    ) -> np.ndarray:
        raise AssertionError("adapter materialized full CSC destinations")

    monkeypatch.setattr(torch, "load", checked_load)
    monkeypatch.setattr(
        CanonicalRelationTopology,
        "destination",
        property(destination_must_not_materialize),
    )
    book = partitioner.generate(PartitionQualificationLimits())
    assert mmap_modes
    assert all(mode is True for mode in mmap_modes)
    book.close()
def test_bounded_worker_matches_pyg_typed_maps_and_permutation(
    tmp_path: Path,
) -> None:
    partitioner, _ = _build_partitioner(tmp_path)
    topology = partitioner.topology_context
    work_root = tmp_path / "bounded" / topology.fingerprint
    work_root.mkdir(parents=True)
    request_path = work_root / "request.json"
    response_path = work_root / "response.json"
    request_path.write_text(
        json.dumps(
            {
                "format_version": "topology-only-pyg-worker-v1",
                "fingerprint": topology.fingerprint,
                "num_partitions": topology.num_partitions,
                "recursive": topology.recursive,
                "output_root": str(work_root / "output"),
                "nodes": [
                    {
                        "internal_key": internal,
                        "count": topology.node_counts[node_type],
                    }
                    for internal, node_type
                    in topology.internal_node_types.items()
                ],
                "relations": [
                    {
                        "relation": list(relation),
                        "internal_key": item.internal_key,
                        "source_internal_key": item.source_internal_key,
                        "destination_internal_key": item.destination_internal_key,
                        "source_count": item.source_count,
                        "destination_count": item.destination_count,
                        "edge_count": item.edge_count,
                        "colptr_path": str(item.colptr_path),
                        "row_path": str(item.row_path),
                    }
                    for relation, item in topology.relations.items()
                ],
            }
        ),
        encoding="utf-8",
    )
    _partition_worker(str(request_path), str(response_path))
    response = json.loads(response_path.read_text(encoding="utf-8"))
    assert response["status"] == "ok", response

    oracle_root = tmp_path / "oracle"
    Partitioner(
        partitioner.materialize_topology(),
        num_parts=topology.num_partitions,
        root=str(oracle_root),
        recursive=topology.recursive,
    ).generate_partition()
    actual_root = work_root / "output"
    assert json.loads(
        (actual_root / "META.json").read_text(encoding="utf-8")
    ) == json.loads((oracle_root / "META.json").read_text(encoding="utf-8"))

    metadata = json.loads(
        (oracle_root / "META.json").read_text(encoding="utf-8")
    )
    for internal in metadata["node_types"]:
        actual = torch.load(
            actual_root / "node_map" / f"{internal}.pt",
            weights_only=True,
        )
        expected = torch.load(
            oracle_root / "node_map" / f"{internal}.pt",
            weights_only=True,
        )
        assert torch.equal(actual, expected)
    for edge_type in metadata["edge_types"]:
        filename = f"{as_str(tuple(edge_type))}.pt"
        actual = torch.load(
            actual_root / "edge_map" / filename,
            weights_only=True,
        )
        expected = torch.load(
            oracle_root / "edge_map" / filename,
            weights_only=True,
        )
        assert torch.equal(actual, expected)
    for partition in range(topology.num_partitions):
        actual = torch.load(
            actual_root / f"part_{partition}" / "node_feats.pt",
            weights_only=True,
        )
        expected = torch.load(
            oracle_root / f"part_{partition}" / "node_feats.pt",
            weights_only=True,
        )
        assert actual.keys() == expected.keys()
        for internal in actual:
            assert torch.equal(
                actual[internal]["global_id"],
                expected[internal]["global_id"],
            )
            assert torch.equal(
                actual[internal]["id"],
                expected[internal]["id"],
            )
            assert actual[internal]["feats"] == expected[internal]["feats"]
        actual_graph = torch.load(
            actual_root / f"part_{partition}" / "graph.pt",
            weights_only=True,
        )
        expected_graph = torch.load(
            oracle_root / f"part_{partition}" / "graph.pt",
            weights_only=True,
        )
        assert actual_graph.keys() == expected_graph.keys()
        for edge_type in actual_graph:
            assert actual_graph[edge_type]["size"] == expected_graph[
                edge_type
            ]["size"]
            for field in ("edge_id", "row", "col"):
                observed = actual_graph[edge_type][field]
                reference = expected_graph[edge_type][field]
                assert torch.equal(observed, reference), (
                    partition,
                    edge_type,
                    field,
                    observed,
                    reference,
                )
        actual_edge_features = torch.load(
            actual_root / f"part_{partition}" / "edge_feats.pt",
            weights_only=False,
        )
        expected_edge_features = torch.load(
            oracle_root / f"part_{partition}" / "edge_feats.pt",
            weights_only=False,
        )
        assert actual_edge_features == expected_edge_features


def test_adapter_cross_checks_part_membership_against_node_map(
    tmp_path: Path,
) -> None:
    partitioner, _ = _build_partitioner(tmp_path)
    artifact = tmp_path / "trusted" / partitioner.topology_fingerprint
    Partitioner(
        partitioner.materialize_topology(),
        num_parts=2,
        root=str(artifact),
    ).generate_partition()
    records = [
        torch.load(
            artifact / f"part_{partition}/node_feats.pt",
            map_location="cpu",
            weights_only=True,
        )
        for partition in range(2)
    ]
    selected: tuple[str, int, int] | None = None
    for internal in ("n0000", "n0001"):
        if (
            len(records[0][internal]["id"]) > 0
            and len(records[1][internal]["id"]) > 0
        ):
            selected = (internal, 0, 1)
            break
    assert selected is not None
    internal, left, right = selected
    left_id = records[left][internal]["id"][0].clone()
    right_id = records[right][internal]["id"][0].clone()
    records[left][internal]["id"][0] = right_id
    records[right][internal]["id"][0] = left_id
    torch.save(records[left], artifact / f"part_{left}/node_feats.pt")
    torch.save(records[right], artifact / f"part_{right}/node_feats.pt")

    with pytest.raises(ArtifactValidationError, match="PARTITION-ID-001"):
        partitioner.artifact_adapter(artifact).adapt(
            PartitionQualificationLimits()
        )




def test_trusted_adapter_rejects_malformed_metadata_and_maps(tmp_path: Path) -> None:
    partitioner, _ = _build_partitioner(tmp_path)
    artifact = tmp_path / "trusted" / partitioner.topology_fingerprint
    Partitioner(partitioner.materialize_topology(), num_parts=2, root=str(artifact)).generate_partition()
    metadata_path = artifact / "META.json"
    metadata = json.loads(metadata_path.read_text(encoding="utf-8"))
    metadata["num_parts"] = 3
    metadata_path.write_text(json.dumps(metadata), encoding="utf-8")
    adapter = partitioner.artifact_adapter(artifact)
    with pytest.raises(ArtifactValidationError, match="PARTITION-OUTPUT-001"):
        adapter.adapt(PartitionQualificationLimits())
    metadata["num_parts"] = 2
    metadata_path.write_text(json.dumps(metadata), encoding="utf-8")
    torch.save(torch.tensor([0, 0, 9, 1], dtype=torch.int64), artifact / "node_map" / "n0000.pt")
    with pytest.raises(ArtifactValidationError, match="PARTITION-ID-001"):
        adapter.adapt(PartitionQualificationLimits())


def test_adapter_rejects_symlink_without_loading_it(tmp_path: Path) -> None:
    partitioner, _ = _build_partitioner(tmp_path)
    artifact = tmp_path / "trusted" / partitioner.topology_fingerprint
    artifact.mkdir(parents=True)
    outside = tmp_path / "outside.pt"
    torch.save(torch.zeros(4, dtype=torch.int64), outside)
    (artifact / "META.json").write_text("{}", encoding="utf-8")
    (artifact / "node_map").mkdir()
    (artifact / "node_map" / "n0000.pt").symlink_to(outside)
    with pytest.raises(ArtifactValidationError, match="PARTITION-OUTPUT-001"):
        partitioner.artifact_adapter(artifact).adapt(PartitionQualificationLimits())


def test_homogeneous_one_type_generation_has_destination_ownership(tmp_path: Path) -> None:
    source = homogeneous_source(tmp_path / "source")
    ingestor = ParquetTypedGraphIngestor(source, tmp_path / "stores")
    partitioner = TopologyOnlyPyGPartitioner(ingestor, ingestor.build_relations(), num_partitions=2)
    book = partitioner.generate(PartitionQualificationLimits())
    relation = partitioner.canonical_relations[("node", "links", "node")]
    np.testing.assert_array_equal(book.edge_ownership[("node", "links", "node")], book.node_assignments["node"][relation.destination])


def test_adapter_type_rejects_untrusted_root(tmp_path: Path) -> None:
    partitioner, _ = _build_partitioner(tmp_path)
    with pytest.raises(ArtifactValidationError, match="PARTITION-OUTPUT-001"):
        PyGPartitionArtifactAdapter(partitioner.topology_context, tmp_path / "not-fingerprint-owned", trusted_parent=tmp_path / "other")


def test_synthetic_edge_types_preflights_before_scanning(
    tmp_path: Path,
    monkeypatch: pytest.MonkeyPatch,
) -> None:
    source = asymmetric_typed_source(
        tmp_path / "source",
        memory_limit_bytes=1,
    )
    ingestor = ParquetTypedGraphIngestor(source, tmp_path / "stores")
    partitioner = TopologyOnlyPyGPartitioner(
        ingestor,
        ingestor.build_relations(),
    )
    monkeypatch.setattr(
        partitioner,
        "_scoring_edges",
        lambda: pytest.fail("scoring scan ran before preflight"),
    )

    with pytest.raises(ArtifactValidationError, match="PARTITION-MEMORY-001"):
        _ = partitioner.synthetic_edge_types


def test_empty_synthetic_edge_type_result_is_cached(
    tmp_path: Path,
    monkeypatch: pytest.MonkeyPatch,
) -> None:
    partitioner, _ = _build_partitioner(tmp_path)
    calls = 0

    def empty_scoring_view() -> dict[tuple[str, str, str], np.ndarray]:
        nonlocal calls
        calls += 1
        partitioner._synthetic_edge_types = ()
        return {}

    monkeypatch.setattr(partitioner, "_scoring_edges", empty_scoring_view)

    assert partitioner.synthetic_edge_types == ()
    assert partitioner.synthetic_edge_types == ()
    assert calls == 1


@pytest.mark.parametrize("override", [False, True, 0, 1])
def test_partition_override_rejects_bool_and_values_below_two(
    tmp_path: Path,
    override: bool | int,
) -> None:
    source = asymmetric_typed_source(tmp_path / f"source-{override!s}")
    ingestor = ParquetTypedGraphIngestor(
        source,
        tmp_path / f"stores-{override!s}",
    )
    relations = ingestor.build_relations()

    with pytest.raises((TypeError, ValueError)):
        TopologyOnlyPyGPartitioner(
            ingestor,
            relations,
            num_partitions=override,
        )



def test_declared_partition_count_below_two_is_rejected_before_worker(
    tmp_path: Path,
) -> None:
    source = asymmetric_typed_source(tmp_path / "source", num_partitions=1)
    ingestor = ParquetTypedGraphIngestor(source, tmp_path / "stores")
    relations = ingestor.build_relations()

    with pytest.raises(ValueError, match="num_partitions must be at least 2"):
        TopologyOnlyPyGPartitioner(ingestor, relations)

def test_partition_worker_reports_nonempty_context_for_assertion_failures(
    tmp_path: Path,
) -> None:
    work_root = tmp_path / "fingerprint"
    work_root.mkdir()
    request_path = work_root / "request.json"
    response_path = work_root / "response.json"
    request_path.write_text(
        json.dumps(
            {
                "format_version": "topology-only-pyg-worker-v1",
                "fingerprint": work_root.name,
                "output_root": str(work_root / "output"),
                "nodes": [{"internal_key": "node", "count": 1}],
                "relations": [],
                "num_partitions": 1,
                "recursive": False,
            }
        ),
        encoding="utf-8",
    )

    _partition_worker(str(request_path), str(response_path))

    response = json.loads(response_path.read_text(encoding="utf-8"))
    assert response["status"] == "error"
    assert response["error_type"] == "AssertionError"
    assert response["detail"] == "AssertionError()"
