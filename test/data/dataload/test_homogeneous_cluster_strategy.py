"""Exact homogeneous materialized and disk cluster-union strategy tests."""

from __future__ import annotations

from typing import Any

import numpy as np
import pytest
import torch
from torch_geometric.utils import subgraph

from test.data.dataload.test_disk_graph_datamodule import (
    materialized_homogeneous_reference,
    task8_stores,
)
from test.data.stores.test_materialized_homogeneous_partition import (
    _directed_graph,
)
from test.data.stores.test_typed_graph_store import QualifiedStoreFixture
from topobench.data.stores.materialized_partition import (
    MaterializedHomogeneousPartition,
)
from topobench.data.stores.typed_graph_store import TypedGraphStore
from topobench.dataloader.disk_graph import HomogeneousClusterStrategy


def _assert_tensor_fields(actual: Any, expected: Any) -> None:
    """Require ordered equality for every tensor emitted by the Task5 oracle."""
    for field_name, expected_value in expected.items():
        if isinstance(expected_value, torch.Tensor):
            assert field_name in actual
            assert torch.equal(actual[field_name], expected_value), field_name


def _disk_expected(
    store: TypedGraphStore,
    partition_ids: tuple[int, ...],
    phase: str,
) -> Any:
    """Apply the original directed induced-subgraph oracle to selected members."""
    source = materialized_homogeneous_reference(store)
    node_type = store.node_types[0]
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
    global_nid = torch.from_numpy(np.array(selected, copy=True))
    edge_index, _, edge_mask = subgraph(
        global_nid,
        source.edge_index,
        relabel_nodes=True,
        num_nodes=source.num_nodes,
        return_edge_mask=True,
    )
    expected = source.clone()
    expected.edge_index = edge_index
    for field_name in ("x", "y", "train_mask", "val_mask", "test_mask"):
        expected[field_name] = source[field_name].index_select(0, global_nid)
    for field_name, value in tuple(source.items()):
        if (
            isinstance(value, torch.Tensor)
            and value.dim() > 0
            and value.size(0) == source.edge_index.size(1)
            and field_name not in {"edge_index"}
        ):
            expected[field_name] = value[edge_mask]
    expected.num_nodes = len(global_nid)
    expected.global_nid = global_nid
    expected.selected_partition_ids = torch.tensor(partition_ids)
    expected.num_selected_partitions = len(partition_ids)
    expected.supervised_mask = expected[f"{phase}_mask"].clone()
    return expected


def test_materialized_single_noncontiguous_and_all_unions_match_task5() -> None:
    """The shared strategy delegates exact homogeneous semantics to Task5."""
    source = _directed_graph()
    source.content_sha256 = "1" * 64
    source.active_split_tag = "default"
    partitioned = MaterializedHomogeneousPartition(source, num_parts=3)
    groups = ((0,), (2, 0), (2, 1, 0))
    strategy = HomogeneousClusterStrategy(
        clusters_per_batch=1,
        partition_groups=groups,
        seed=19,
    )

    descriptors = strategy.setup(
        partitioned,
        phase="train",
        active_split_tag="default",
        shuffle=False,
    )
    assert tuple(descriptor.partition_ids for descriptor in descriptors) == (
        (0,),
        (0, 2),
        (0, 1, 2),
    )
    assert descriptors == strategy.setup(
        partitioned,
        phase="train",
        active_split_tag="default",
        shuffle=False,
    )

    for descriptor in descriptors:
        actual = strategy.materialize(partitioned, descriptor)
        expected = partitioned.materialize(
            descriptor.partition_ids,
            phase="train",
        )
        _assert_tensor_fields(actual, expected)
        assert actual.participant_counts == {"node": actual.num_nodes}
        assert actual.sampling_descriptor == descriptor
        assert torch.equal(actual.supervised_mask, actual.train_mask)


def test_disk_union_reads_only_selected_rows_and_exact_relation_spans(
    task8_stores: dict[str, QualifiedStoreFixture],
    monkeypatch: pytest.MonkeyPatch,
) -> None:
    """Disk cluster unions never request complete feature or edge-field arrays."""
    path = task8_stores["homogeneous"].store_build.path
    with TypedGraphStore.open(path) as store:
        node_reads: list[tuple[str, object]] = []
        relation_reads: list[tuple[str, object]] = []
        csc_reads: list[tuple[str, str, str]] = []
        original_node_array = store.node_array
        original_relation_field = store.relation_field
        original_relation_csc = store.relation_csc

        def observed_node_array(
            node_type: str,
            field_name: str,
            rows: object = None,
        ) -> np.ndarray:
            node_reads.append((field_name, rows))
            return original_node_array(node_type, field_name, rows)

        def observed_relation_csc(
            relation: tuple[str, str, str],
        ) -> tuple[np.ndarray, np.ndarray]:
            csc_reads.append(relation)
            return original_relation_csc(relation)

        def observed_relation_field(
            relation: tuple[str, str, str],
            field_name: str,
            rows: object = None,
        ) -> np.ndarray:
            relation_reads.append((field_name, rows))
            return original_relation_field(relation, field_name, rows)

        monkeypatch.setattr(store, "node_array", observed_node_array)
        monkeypatch.setattr(store, "relation_field", observed_relation_field)
        monkeypatch.setattr(store, "relation_csc", observed_relation_csc)
        strategy = HomogeneousClusterStrategy(
            partition_groups=((2, 0),),
            clusters_per_batch=2,
            seed=23,
        )
        descriptor = strategy.setup(
            store,
            phase="val",
            active_split_tag=store.active_split_tag,
            shuffle=False,
        )[0]
        actual = strategy.materialize(store, descriptor)
        actual_csc_reads = tuple(csc_reads)
        csc_reads.clear()
        expected = _disk_expected(store, (0, 2), "val")

    _assert_tensor_fields(actual, expected)
    assert actual.participant_counts == {"node": actual.num_nodes}
    assert node_reads and all(rows is not None for _, rows in node_reads)
    assert actual_csc_reads == (("node", "links", "node"),)
    assert all(rows is not None for _, rows in relation_reads)


def test_partition_identity_validation_rejects_duplicates_and_bad_scalars() -> None:
    """Partition selection never silently coerces or de-duplicates identities."""
    with pytest.raises(ValueError, match="duplicate partition"):
        HomogeneousClusterStrategy(
            partition_groups=((0, 0),),
            clusters_per_batch=1,
        )
    with pytest.raises(TypeError, match="partition.*non-boolean integers"):
        HomogeneousClusterStrategy(
            partition_groups=((True,),),
            clusters_per_batch=1,
        )
    with pytest.raises(TypeError, match="clusters_per_batch"):
        HomogeneousClusterStrategy(clusters_per_batch=True)
