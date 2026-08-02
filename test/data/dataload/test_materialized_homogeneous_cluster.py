"""Characterize exact native ``Data`` unions over PyG partitions."""

import torch
from torch_geometric.data import Data

from topobench.data.stores.materialized_partition import (
    MaterializedHomogeneousPartition,
)

from test.data.stores.test_materialized_homogeneous_partition import (
    partitioned_graph,
)


def _canonical_members(partition, part_ids: list[int]) -> torch.Tensor:
    perm_rows = torch.cat(
        [
            torch.arange(
                int(partition.partptr[part_id]),
                int(partition.partptr[part_id + 1]),
            )
            for part_id in sorted(set(part_ids))
        ]
    )
    return partition.node_perm[perm_rows]


def _induced_expectation(
    source: Data,
    global_nid: torch.Tensor,
) -> tuple[torch.Tensor, torch.Tensor]:
    to_local = torch.full((source.num_nodes,), -1, dtype=torch.long)
    to_local[global_nid] = torch.arange(global_nid.numel())
    relabeled = to_local[source.edge_index]
    edge_mask = (relabeled >= 0).all(dim=0)
    return relabeled[:, edge_mask], edge_mask


def test_global_nid_uses_nonidentity_node_perm_not_permuted_row_numbers(
    partitioned_graph: tuple[Data, MaterializedHomogeneousPartition],
) -> None:
    source, materialized = partitioned_graph
    partition = materialized.partition

    assert partition.__class__.__module__ == "torch_geometric.loader.cluster"
    assert not torch.equal(partition.node_perm, torch.arange(source.num_nodes))
    start = int(partition.partptr[0])
    end = int(partition.partptr[1])
    canonical = partition.node_perm[start:end]
    assert not torch.equal(canonical, torch.arange(start, end))

    output = materialized.materialize([0])

    assert torch.equal(output.global_nid, canonical)
    assert torch.equal(output.x, source.x[canonical])
    assert torch.equal(output.y, source.y[canonical])


def test_split_selected_partition_union_preserves_all_masks_and_exact_edges(
    partitioned_graph: tuple[Data, MaterializedHomogeneousPartition],
) -> None:
    source, materialized = partitioned_graph
    partition = materialized.partition
    expected_parts = [
        part_id
        for part_id in range(materialized.num_parts)
        if bool(source.train_mask[_canonical_members(partition, [part_id])].any())
    ]
    assert len(expected_parts) == 2

    selected_parts = materialized.partition_ids_for_phase("train")
    output = materialized.materialize(selected_parts, phase="train")
    expected_global = _canonical_members(partition, expected_parts)
    expected_edge_index, edge_mask = _induced_expectation(source, expected_global)

    assert selected_parts == expected_parts
    assert torch.equal(output.selected_partition_ids, torch.tensor(expected_parts))
    assert output.num_selected_partitions == len(expected_parts)
    assert torch.equal(output.global_nid, expected_global)
    assert torch.equal(output.edge_index, expected_edge_index)
    assert torch.equal(output.edge_id, source.edge_id[edge_mask])
    assert torch.equal(output.edge_attr, source.edge_attr[edge_mask])
    assert torch.equal(output.edge_weight, source.edge_weight[edge_mask])
    assert torch.equal(output.train_mask, source.train_mask[expected_global])
    assert torch.equal(output.val_mask, source.val_mask[expected_global])
    assert torch.equal(output.test_mask, source.test_mask[expected_global])
    assert torch.equal(output.supervised_mask, output.train_mask)


def test_noncontiguous_union_is_input_order_independent_and_exactly_induced(
    partitioned_graph: tuple[Data, MaterializedHomogeneousPartition],
) -> None:
    source, materialized = partitioned_graph
    expected_global = _canonical_members(materialized.partition, [0, 2])
    expected_edge_index, edge_mask = _induced_expectation(source, expected_global)

    reversed_with_duplicate = materialized.materialize([2, 0, 2])
    normalized = materialized.materialize([0, 2])

    assert torch.equal(
        reversed_with_duplicate.selected_partition_ids,
        torch.tensor([0, 2]),
    )
    assert torch.equal(reversed_with_duplicate.global_nid, expected_global)
    assert torch.equal(reversed_with_duplicate.edge_index, expected_edge_index)
    assert torch.equal(reversed_with_duplicate.edge_id, source.edge_id[edge_mask])
    for key in reversed_with_duplicate.keys():
        left = reversed_with_duplicate[key]
        right = normalized[key]
        if isinstance(left, torch.Tensor):
            assert torch.equal(left, right)
        else:
            assert left == right


def test_all_partition_union_retains_isolated_node_self_loops_and_parallel_edges(
    partitioned_graph: tuple[Data, MaterializedHomogeneousPartition],
) -> None:
    source, materialized = partitioned_graph
    output = materialized.materialize(range(materialized.num_parts))
    expected_global = materialized.perm_to_global
    expected_edge_index, edge_mask = _induced_expectation(source, expected_global)

    assert output.num_nodes == source.num_nodes
    assert torch.equal(output.global_nid, expected_global)
    assert 9 in output.global_nid.tolist()
    isolated_local = int((output.global_nid == 9).nonzero().item())
    assert not bool((output.edge_index == isolated_local).any())
    assert torch.equal(output.edge_index, expected_edge_index)
    assert torch.equal(output.edge_id, source.edge_id[edge_mask])

    global_edges = output.global_nid[output.edge_index]
    self_loop_ids = output.edge_id[global_edges[0] == global_edges[1]]
    assert torch.equal(self_loop_ids, torch.tensor([1014, 1018]))
    parallel_ids = output.edge_id[
        (global_edges[0] == 0) & (global_edges[1] == 2)
    ]
    assert torch.equal(parallel_ids, torch.tensor([1000, 1016, 1017]))
