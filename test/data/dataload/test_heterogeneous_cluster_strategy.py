"""Exact typed materialized and disk cluster-union strategy tests."""

from __future__ import annotations

import numpy as np
import pytest
import torch
from torch_geometric.data import HeteroData

from test.data.dataload.test_disk_graph_datamodule import (
    assert_heterogeneous_exact,
    materialized_heterogeneous_reference,
    task8_stores,
)
from test.data.stores.test_typed_graph_store import QualifiedStoreFixture
from topobench.data.stores.typed_graph_store import TypedGraphStore
from topobench.dataloader.disk_graph import HeterogeneousClusterStrategy


def _subsets(
    store: TypedGraphStore,
    partition_ids: tuple[int, ...],
) -> dict[str, torch.Tensor]:
    """Return canonical type-local members in partition/permutation order."""
    subsets: dict[str, torch.Tensor] = {}
    for node_type in store.node_types:
        permutation = store.partition_permutation(node_type)
        partptr = store.partition_partptr(node_type)
        selected = np.concatenate(
            [
                permutation[
                    int(partptr[part_id]) : int(partptr[part_id + 1])
                ]
                for part_id in partition_ids
            ]
        )
        subsets[node_type] = torch.from_numpy(np.array(selected, copy=True))
    return subsets


def test_materialized_and_disk_unions_match_heterodata_subgraph_exactly(
    task8_stores: dict[str, QualifiedStoreFixture],
    monkeypatch: pytest.MonkeyPatch,
) -> None:
    """Single, noncontiguous, and all typed unions share the PyG oracle."""
    fixture = task8_stores["heterogeneous"]
    with TypedGraphStore.open(fixture.store_build.path) as store:
        reference = materialized_heterogeneous_reference(store)
        groups = ((0,), (2, 0), (2, 1, 0))
        materialized = HeterogeneousClusterStrategy(
            partition_book=fixture.partition_build.book,
            partition_groups=groups,
            clusters_per_batch=1,
            seed=29,
        )
        memory_descriptors = materialized.setup(
            reference,
            phase="train",
            active_split_tag=store.active_split_tag,
            shuffle=False,
        )
        assert tuple(d.partition_ids for d in memory_descriptors) == (
            (0,),
            (0, 2),
            (0, 1, 2),
        )

        disk = HeterogeneousClusterStrategy(
            partition_groups=groups,
            clusters_per_batch=1,
            seed=29,
        )
        node_reads: list[tuple[str, str, object]] = []
        relation_reads: list[tuple[tuple[str, str, str], str, object]] = []
        original_node_array = store.node_array
        original_relation_field = store.relation_field

        def observed_node_array(
            node_type: str,
            field_name: str,
            rows: object = None,
        ) -> np.ndarray:
            node_reads.append((node_type, field_name, rows))
            return original_node_array(node_type, field_name, rows)

        def observed_relation_field(
            relation: tuple[str, str, str],
            field_name: str,
            rows: object = None,
        ) -> np.ndarray:
            relation_reads.append((relation, field_name, rows))
            return original_relation_field(relation, field_name, rows)

        monkeypatch.setattr(store, "node_array", observed_node_array)
        monkeypatch.setattr(store, "relation_field", observed_relation_field)
        disk_descriptors = disk.setup(
            store,
            phase="train",
            active_split_tag=store.active_split_tag,
            shuffle=False,
        )

        for memory_descriptor, disk_descriptor in zip(
            memory_descriptors,
            disk_descriptors,
            strict=True,
        ):
            subset_dict = _subsets(store, memory_descriptor.partition_ids)
            oracle = reference.subgraph(subset_dict)
            memory_batch = materialized.materialize(
                reference,
                memory_descriptor,
            )
            disk_batch = disk.materialize(store, disk_descriptor)
            assert_heterogeneous_exact(memory_batch, oracle)
            assert_heterogeneous_exact(disk_batch, oracle)
            assert memory_batch.participant_counts == {
                node_type: len(subset_dict[node_type])
                for node_type in reference.node_types
            }
            assert disk_batch.participant_counts == memory_batch.participant_counts
            assert tuple(disk_batch.edge_types) == tuple(reference.edge_types)
            assert torch.equal(
                disk_batch[reference.target_node_type].supervised_mask,
                disk_batch[reference.target_node_type].train_mask,
            )

    assert node_reads and all(rows is not None for _, _, rows in node_reads)
    assert relation_reads and all(rows is not None for _, _, rows in relation_reads)


def test_heterogeneous_cluster_rejects_partition_book_identity_mismatch(
    task8_stores: dict[str, QualifiedStoreFixture],
) -> None:
    """A materialized reference cannot be paired with another partition identity."""
    fixture = task8_stores["heterogeneous"]
    with TypedGraphStore.open(fixture.store_build.path) as store:
        reference = materialized_heterogeneous_reference(store)
        reference.partition_book_identity = "f" * 64
        strategy = HeterogeneousClusterStrategy(
            partition_book=fixture.partition_build.book,
            clusters_per_batch=1,
        )
        with pytest.raises(ValueError, match="partition book identity mismatch"):
            strategy.setup(
                reference,
                phase="train",
                active_split_tag=store.active_split_tag,
                shuffle=False,
            )
        del reference.partition_book_identity
        missing_identity = HeterogeneousClusterStrategy(
            partition_book=fixture.partition_build.book,
            clusters_per_batch=1,
        )
        with pytest.raises(
            ValueError,
            match="partition book identity is required",
        ):
            missing_identity.setup(
                reference,
                phase="train",
                active_split_tag=store.active_split_tag,
                shuffle=False,
            )


def test_heterogeneous_cluster_requires_native_heterodata_reference() -> None:
    """Materialized typed cluster loading never accepts duck-typed graph objects."""
    strategy = HeterogeneousClusterStrategy(clusters_per_batch=1)
    with pytest.raises(TypeError, match="HeteroData or TypedGraphStore"):
        strategy.setup(
            object(),
            phase="train",
            active_split_tag="primary",
            shuffle=False,
        )
