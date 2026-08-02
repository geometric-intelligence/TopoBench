"""Immutable typed partition-book invariants and collision-safe identity."""

from __future__ import annotations

import tracemalloc
from pathlib import Path

import numpy as np
import pytest

from topobench.data.stores.typed_graph_ingestion import ArtifactValidationError, ParquetTypedGraphIngestor
from topobench.data.stores.typed_partition_book import (
    PartitionStatistics,
    TypedPartitionBook,
    topology_fingerprint,
    validate_typed_partition_book,
)
from test.data.stores.test_topology_only_pyg_partitioner import asymmetric_typed_source


def _book(assignments: dict[str, np.ndarray]) -> TypedPartitionBook:
    return TypedPartitionBook.from_assignments(
        num_partitions=2,
        node_assignments=assignments,
        edge_ownership={
            ("author", "writes", "paper"): np.array([0, 1, 0], dtype=np.int64),
        },
        topology_fingerprint="a" * 64,
        source_binding={"task4_content_sha256": "b" * 64, "active_split_tag": "primary"},
        backend="pyg",
        backend_version="2.8.0",
        options={"recursive": False},
        provenance={"producer": "unit"},
        estimated_resources={"peak_memory_bytes": 1, "temporary_disk_bytes": 1},
        measured_resources={"peak_rss_bytes": 1, "temporary_disk_bytes": 1},
        statistics=PartitionStatistics.empty(2),
        qualification_checks=(),
    )


def test_assignments_derive_stable_permutation_inverse_and_partptr() -> None:
    book = _book({
        "author": np.array([1, 0, 1, 0], dtype=np.int64),
        "paper": np.array([0, 1, 1], dtype=np.int64),
    })
    np.testing.assert_array_equal(book.node_permutations["author"], [1, 3, 0, 2])
    np.testing.assert_array_equal(book.node_inverse_permutations["author"][book.node_permutations["author"]], np.arange(4))
    np.testing.assert_array_equal(book.node_partptr["author"], [0, 2, 4])
    for array_map in (book.node_assignments, book.node_permutations, book.node_inverse_permutations, book.node_partptr, book.edge_ownership):
        assert all(not array.flags.writeable for array in array_map.values())
    with pytest.raises(ValueError):
        book.node_assignments["author"][0] = 0


@pytest.mark.parametrize(
    "assignment",
    [
        np.array([0, 2], dtype=np.int64),
        np.array([0.0, 1.0], dtype=np.float64),
        np.array([[0, 1]], dtype=np.int64),
    ],
)
def test_validator_rejects_malformed_typed_assignments(assignment: np.ndarray) -> None:
    with pytest.raises(ArtifactValidationError, match="PARTITION-ID-001"):
        _book({"author": assignment, "paper": np.array([0, 1], dtype=np.int64)})


def test_assignment_and_options_are_content_identity_not_assumed_deterministic() -> None:
    first = _book({"author": np.array([0, 1]), "paper": np.array([0, 1, 0])})
    second = _book({"author": np.array([1, 0]), "paper": np.array([0, 1, 0])})
    changed_options = TypedPartitionBook.from_assignments(
        num_partitions=2,
        node_assignments=first.node_assignments,
        edge_ownership=first.edge_ownership,
        topology_fingerprint=first.topology_fingerprint,
        source_binding=first.source_binding,
        backend="pyg", backend_version="2.8.0", options={"recursive": True},
        provenance=first.provenance, estimated_resources=first.estimated_resources,
        measured_resources=first.measured_resources, statistics=first.statistics,
        qualification_checks=(),
    )
    assert len({first.content_identity, second.content_identity, changed_options.content_identity}) == 3
    assert validate_typed_partition_book(first) is first


def test_topology_fingerprint_is_canonical_framed_and_feature_independent(tmp_path: Path) -> None:
    source = asymmetric_typed_source(tmp_path / "source")
    ingestor = ParquetTypedGraphIngestor(source, tmp_path / "stores")
    relation_build = ingestor.build_relations()
    first = topology_fingerprint.from_relation_build(ingestor, relation_build)

    arrays = relation_build.stage_root / "arrays" / "nodes" / "n0000" / "x.npy"
    features = np.load(arrays, allow_pickle=False)
    changed = np.array(features, copy=True)
    changed[0, 0] += 100
    np.save(arrays, changed, allow_pickle=False)
    assert topology_fingerprint.from_relation_build(ingestor, relation_build, validate_binding=False) == first


def test_topology_fingerprint_distinguishes_ambiguous_unframed_components() -> None:
    left = topology_fingerprint.from_components(
        node_counts=(("ab", 1), ("c", 2)), relations=(),
    )
    right = topology_fingerprint.from_components(
        node_counts=(("a", 1), ("bc", 2)), relations=(),
    )
    assert left != right


def test_topology_fingerprint_hashes_memmaps_in_bounded_chunks(
    tmp_path: Path,
) -> None:
    row = np.memmap(
        tmp_path / "large-row.bin",
        mode="w+",
        dtype=np.int64,
        shape=(2_000_000,),
    )
    row[:] = 1
    row.flush()
    colptr = np.array([0, len(row)], dtype=np.int64)

    tracemalloc.start()
    topology_fingerprint.from_components(
        node_counts=(("node", len(row)),),
        relations=((("node", "links", "node"), colptr, row),),
    )
    _, peak = tracemalloc.get_traced_memory()
    tracemalloc.stop()

    assert peak < 4 * 1024**2
