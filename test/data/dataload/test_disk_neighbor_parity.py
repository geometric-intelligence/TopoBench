"""Ordered materialized/disk PyG NeighborLoader parity qualification."""

from __future__ import annotations

import shutil
from pathlib import Path

import numpy as np
import pytest
import torch

from test.data.dataload.test_disk_graph_datamodule import (
    assert_heterogeneous_exact,
    materialized_heterogeneous_reference,
    task8_stores,
)
from test.data.stores.test_typed_graph_store import QualifiedStoreFixture
from topobench.data.stores.typed_graph_store import TypedGraphStore
from topobench.dataloader.disk_graph import HeterogeneousNeighborStrategy


def _asymmetric_two_hop_fanout(
    relations: tuple[tuple[str, str, str], ...],
) -> dict[tuple[str, str, str], list[int]]:
    """Use exhaustive incoming reverse arcs and asymmetric zero fanout."""
    return {
        relation: (
            [-1, -1]
            if relation[1] == "written_by"
            else [0, -1]
            if relation[1] == "cites"
            else [0, 0]
        )
        for relation in relations
    }


def test_materialized_and_disk_neighbor_batches_are_ordered_exact_parity(
    task8_stores: dict[str, QualifiedStoreFixture],
    monkeypatch: pytest.MonkeyPatch,
) -> None:
    """Seeds, hops, duplicate edges, fields, masks, and participants all agree."""
    fixture = task8_stores["heterogeneous"]
    with TypedGraphStore.open(fixture.store_build.path) as store:
        reference = materialized_heterogeneous_reference(store)
        fanout = _asymmetric_two_hop_fanout(store.relation_types)
        memory_strategy = HeterogeneousNeighborStrategy(
            batch_size=2,
            num_neighbors=fanout,
            seed=53,
        )
        disk_strategy = HeterogeneousNeighborStrategy(
            batch_size=2,
            num_neighbors=fanout,
            seed=53,
        )
        memory_descriptors = memory_strategy.setup(
            reference,
            phase="train",
            active_split_tag=store.active_split_tag,
            shuffle=False,
        )
        disk_descriptors = disk_strategy.setup(
            store,
            phase="train",
            active_split_tag=store.active_split_tag,
            shuffle=False,
        )
        assert tuple(
            descriptor.target_seed_ids for descriptor in disk_descriptors
        ) == tuple(
            descriptor.target_seed_ids for descriptor in memory_descriptors
        )
        assert all(
            disk.content_sha256 != memory.content_sha256
            for memory, disk in zip(
                memory_descriptors,
                disk_descriptors,
                strict=True,
            )
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
        for memory_descriptor, disk_descriptor in zip(
            memory_descriptors,
            disk_descriptors,
            strict=True,
        ):
            memory_batch = memory_strategy.materialize(
                reference,
                memory_descriptor,
            )
            disk_batch = disk_strategy.materialize(store, disk_descriptor)
            assert_heterogeneous_exact(disk_batch, memory_batch)
            target = reference.target_node_type
            assert disk_batch[target].batch_size == len(
                disk_descriptor.target_seed_ids
            )
            assert tuple(
                disk_batch[target].n_id[: disk_batch[target].batch_size]
            ) == disk_descriptor.target_seed_ids
            assert disk_batch.participant_counts == memory_batch.participant_counts
            for node_type in disk_batch.node_types:
                assert torch.equal(
                    disk_batch[node_type].n_id,
                    memory_batch[node_type].n_id,
                )
                assert (
                    disk_batch[node_type].num_sampled_nodes
                    == memory_batch[node_type].num_sampled_nodes
                )
            for relation in disk_batch.edge_types:
                assert torch.equal(
                    disk_batch[relation].e_id,
                    memory_batch[relation].e_id,
                )
                assert (
                    disk_batch[relation].num_sampled_edges
                    == memory_batch[relation].num_sampled_edges
                )

    assert node_reads and all(rows is not None for _, _, rows in node_reads)
    assert relation_reads and all(rows is not None for _, _, rows in relation_reads)


def test_disk_neighbor_reload_and_move_reproduce_the_same_batch(
    task8_stores: dict[str, QualifiedStoreFixture],
    tmp_path: Path,
) -> None:
    """Content identity and local descriptor seeds survive close and relocation."""
    source_path = task8_stores["heterogeneous"].store_build.path
    moved_path = tmp_path / source_path.name
    shutil.copytree(source_path, moved_path)

    def load_one(path: Path) -> tuple[object, object]:
        with TypedGraphStore.open(path) as store:
            strategy = HeterogeneousNeighborStrategy(
                batch_size=2,
                num_neighbors=_asymmetric_two_hop_fanout(
                    store.relation_types
                ),
                seed=59,
            )
            descriptor = strategy.setup(
                store,
                phase="train",
                active_split_tag=store.active_split_tag,
                shuffle=False,
            )[0]
            return descriptor, strategy.materialize(store, descriptor)

    first_descriptor, first_batch = load_one(source_path)
    reloaded_descriptor, reloaded_batch = load_one(source_path)
    moved_descriptor, moved_batch = load_one(moved_path)
    assert first_descriptor == reloaded_descriptor == moved_descriptor
    assert_heterogeneous_exact(reloaded_batch, first_batch)
    assert_heterogeneous_exact(moved_batch, first_batch)


def test_disk_neighbor_rejects_relation_fanout_identity_mismatch(
    task8_stores: dict[str, QualifiedStoreFixture],
) -> None:
    """Every canonical relation, including explicit reverse, needs one fanout."""
    path = task8_stores["heterogeneous"].store_build.path
    with TypedGraphStore.open(path) as store:
        incomplete = {
            relation: [-1]
            for relation in store.relation_types
            if relation[1] != "written_by"
        }
        strategy = HeterogeneousNeighborStrategy(
            batch_size=1,
            num_neighbors=incomplete,
        )
        with pytest.raises(
            ValueError,
            match="fanout keys.*missing=.*written_by",
        ):
            strategy.setup(
                store,
                phase="train",
                active_split_tag=store.active_split_tag,
                shuffle=False,
            )
