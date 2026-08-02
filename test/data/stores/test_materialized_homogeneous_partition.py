"""Characterization tests for exact in-memory homogeneous partitions."""

from collections.abc import Iterable

import pytest
import torch
from torch_geometric.data import Data, HeteroData
from torch_geometric.utils import to_undirected

import topobench.data.stores.materialized_partition as materialized_partition
from topobench.data.stores.materialized_partition import (
    MaterializedHomogeneousPartition,
)


def _directed_graph() -> Data:
    edge_index = torch.tensor(
        [
            [0, 2, 4, 0, 2, 4, 1, 3, 5, 1, 3, 5, 6, 7, 7, 6, 0, 0, 8],
            [2, 4, 0, 4, 0, 2, 3, 5, 1, 5, 1, 3, 7, 6, 7, 7, 2, 2, 8],
        ],
        dtype=torch.long,
    )
    num_nodes = 10
    num_edges = edge_index.size(1)
    train_mask = torch.zeros(num_nodes, dtype=torch.bool)
    train_mask[[4, 8]] = True
    val_mask = torch.zeros(num_nodes, dtype=torch.bool)
    val_mask[[3, 9]] = True
    test_mask = torch.zeros(num_nodes, dtype=torch.bool)
    test_mask[[0, 5, 7]] = True
    return Data(
        x=torch.arange(num_nodes * 3, dtype=torch.float64).view(num_nodes, 3),
        y=torch.arange(num_nodes, dtype=torch.int16),
        node_code=(100 + torch.arange(num_nodes, dtype=torch.int32)),
        edge_index=edge_index,
        edge_attr=torch.stack(
            [
                torch.arange(num_edges, dtype=torch.int32),
                -torch.arange(num_edges, dtype=torch.int32),
            ],
            dim=1,
        ),
        edge_weight=torch.linspace(0.25, 4.75, num_edges, dtype=torch.float64),
        edge_id=1000 + torch.arange(num_edges, dtype=torch.long),
        train_mask=train_mask,
        val_mask=val_mask,
        test_mask=test_mask,
        graph_tensor=torch.tensor([[501, 502]], dtype=torch.int64),
        graph_name="independent-directed-fixture",
        num_nodes=num_nodes,
    )

def _equal_domain_graph() -> Data:
    return Data(
        edge_index=torch.tensor(
            [[0, 1, 2, 0], [1, 2, 0, 2]],
            dtype=torch.long,
        ),
        x=torch.arange(8, dtype=torch.float32).view(4, 2),
        weight=torch.tensor([0.5, 1.5, 2.5, 3.5], dtype=torch.float64),
        custom_signal=torch.tensor([10, 20, 30, 40], dtype=torch.int32),
        num_nodes=4,
    )


@pytest.fixture(scope="session")
def partitioned_graph() -> tuple[Data, MaterializedHomogeneousPartition]:
    source = _directed_graph()
    return source, MaterializedHomogeneousPartition(source, num_parts=3)


def _global_nodes(partition, part_ids: Iterable[int]) -> torch.Tensor:
    rows = [
        torch.arange(
            int(partition.partptr[part_id]),
            int(partition.partptr[part_id + 1]),
            dtype=torch.long,
        )
        for part_id in sorted(set(part_ids))
    ]
    return partition.node_perm[torch.cat(rows)]


def _expected_induced(
    data: Data,
    global_nid: torch.Tensor,
) -> tuple[torch.Tensor, torch.Tensor]:
    global_to_local = torch.full((data.num_nodes,), -1, dtype=torch.long)
    global_to_local[global_nid] = torch.arange(global_nid.numel())
    relabeled = global_to_local[data.edge_index]
    edge_mask = (relabeled >= 0).all(dim=0)
    return relabeled[:, edge_mask], edge_mask


def _assert_exact_materialization(
    source: Data,
    partition,
    part_ids: list[int],
    output: Data,
) -> None:
    global_nid = _global_nodes(partition, part_ids)
    expected_edge_index, edge_mask = _expected_induced(source, global_nid)

    assert type(output) is Data
    assert torch.equal(output.global_nid, global_nid)
    assert torch.equal(output.edge_index, expected_edge_index)
    assert torch.equal(output.x, source.x[global_nid])
    assert torch.equal(output.y, source.y[global_nid])
    assert torch.equal(output.node_code, source.node_code[global_nid])
    assert torch.equal(output.train_mask, source.train_mask[global_nid])
    assert torch.equal(output.val_mask, source.val_mask[global_nid])
    assert torch.equal(output.test_mask, source.test_mask[global_nid])
    assert torch.equal(output.edge_attr, source.edge_attr[edge_mask])
    assert torch.equal(output.edge_weight, source.edge_weight[edge_mask])
    assert torch.equal(output.edge_id, source.edge_id[edge_mask])
    assert torch.equal(output.graph_tensor, source.graph_tensor)
    assert output.graph_name == source.graph_name
    assert output.x.dtype == source.x.dtype
    assert output.y.dtype == source.y.dtype
    assert output.edge_attr.dtype == source.edge_attr.dtype
    assert output.edge_weight.dtype == source.edge_weight.dtype


def test_constructor_retains_real_pyg_partition_and_canonical_identity(
    partitioned_graph: tuple[Data, MaterializedHomogeneousPartition],
) -> None:
    source, materialized = partitioned_graph

    assert materialized.partition.__class__.__module__ == "torch_geometric.loader.cluster"
    assert not torch.equal(
        materialized.partition.node_perm,
        torch.arange(source.num_nodes),
    )
    assert materialized.num_parts == 3
    assert torch.equal(materialized.perm_to_global, materialized.partition.node_perm)

def test_constructor_partitions_only_a_temporary_undirected_topology(
    monkeypatch: pytest.MonkeyPatch,
) -> None:
    source = _directed_graph()
    captured: dict[str, Data] = {}
    real_cluster_data = materialized_partition.ClusterData

    def recording_cluster_data(data: Data, *args, **kwargs):
        captured["input"] = data
        cluster_data = real_cluster_data(data, *args, **kwargs)
        captured["retained"] = cluster_data.data
        return cluster_data

    monkeypatch.setattr(
        materialized_partition,
        "ClusterData",
        recording_cluster_data,
    )

    store = MaterializedHomogeneousPartition(source, num_parts=3)
    scoring = captured["input"]
    expected_scoring_edges = to_undirected(
        source.edge_index,
        num_nodes=source.num_nodes,
    )

    assert scoring is not source
    assert set(scoring.keys()) == {"edge_index", "num_nodes"}
    assert torch.equal(scoring.edge_index, expected_scoring_edges)
    assert set(captured["retained"].keys()) == {"num_nodes"}

    output = store.materialize(range(store.num_parts))
    assert torch.equal(output.global_nid[output.edge_index], source.edge_index)
    assert torch.equal(output.edge_id, source.edge_id)
    assert torch.equal(output.edge_attr, source.edge_attr)

def test_edge_index_only_singleton_union_retains_its_node_count() -> None:
    source = Data(
        edge_index=torch.tensor([[0], [1]], dtype=torch.long),
    )
    store = MaterializedHomogeneousPartition(source, num_parts=2)
    assert torch.equal(
        store.partition.partptr[1:] - store.partition.partptr[:-1],
        torch.ones(2, dtype=torch.long),
    )

    output = store.materialize([0])

    assert output.global_nid.numel() == 1
    assert output.edge_index.numel() == 0
    assert output.num_nodes == 1


def test_equal_node_and_edge_counts_reject_undeclared_ambiguous_fields() -> None:
    with pytest.raises(ValueError, match="attribute.*weight.*ambiguous"):
        MaterializedHomogeneousPartition(_equal_domain_graph(), num_parts=2)


def test_explicit_roles_slice_custom_node_and_edge_fields_exactly() -> None:
    source = _equal_domain_graph()
    store = MaterializedHomogeneousPartition(
        source,
        num_parts=2,
        attribute_roles={
            "weight": "edge",
            "custom_signal": "node",
        },
    )
    output = store.materialize([0])
    expected_global = _global_nodes(store.partition, [0])
    _, edge_mask = _expected_induced(source, expected_global)

    assert torch.equal(output.custom_signal, source.custom_signal[expected_global])
    assert output.custom_signal.dtype == source.custom_signal.dtype
    assert torch.equal(output.weight, source.weight[edge_mask])
    assert output.weight.dtype == source.weight.dtype
    assert output.is_node_attr("custom_signal")
    assert output.is_edge_attr("weight")


def test_attribute_roles_validate_mapping_keys_roles_shapes_and_reserved_fields() -> None:
    with pytest.raises(TypeError, match="attribute_roles.*mapping"):
        MaterializedHomogeneousPartition(
            _equal_domain_graph(),
            num_parts=2,
            attribute_roles=[("weight", "edge")],
        )
    with pytest.raises(ValueError, match="attribute_roles.*missing"):
        MaterializedHomogeneousPartition(
            _equal_domain_graph(),
            num_parts=2,
            attribute_roles={"missing": "node"},
        )
    with pytest.raises(ValueError, match="attribute_roles.*weight.*role"):
        MaterializedHomogeneousPartition(
            _equal_domain_graph(),
            num_parts=2,
            attribute_roles={"weight": "invalid"},
        )
    with pytest.raises(ValueError, match="attribute_roles.*reserved.*edge_index"):
        MaterializedHomogeneousPartition(
            _equal_domain_graph(),
            num_parts=2,
            attribute_roles={"edge_index": "edge"},
        )
    with pytest.raises(ValueError, match="attribute_roles.*node_code.*edge.*shape"):
        MaterializedHomogeneousPartition(
            _directed_graph(),
            num_parts=3,
            attribute_roles={"node_code": "edge"},
        )

@pytest.mark.parametrize(
    ("key", "role"),
    [
        ("edge_attr", "node"),
        ("edge_attr", "graph"),
        ("train_mask", "edge"),
        ("train_mask", "graph"),
    ],
)
def test_attribute_roles_cannot_override_intrinsic_domains(
    key: str,
    role: str,
) -> None:
    topology = _equal_domain_graph().edge_index
    source = Data(
        edge_index=topology,
        edge_attr=torch.arange(4, dtype=torch.int16),
        train_mask=torch.tensor([True, False, True, False]),
        num_nodes=4,
    )

    with pytest.raises(ValueError, match=f"attribute_roles.*{key}.*intrinsic"):
        MaterializedHomogeneousPartition(
            source,
            num_parts=2,
            attribute_roles={key: role},
        )


@pytest.mark.parametrize("part_ids", [[1], [0, 2], [0, 1, 2]])
def test_materialize_preserves_exact_induced_graph_in_pyg_row_order(
    part_ids: list[int],
    partitioned_graph: tuple[Data, MaterializedHomogeneousPartition],
) -> None:
    source, materialized = partitioned_graph
    output = materialized.materialize(part_ids)

    _assert_exact_materialization(source, materialized.partition, part_ids, output)
    assert torch.equal(
        output.selected_partition_ids,
        torch.tensor(part_ids, dtype=torch.long),
    )
    assert output.num_selected_partitions == len(part_ids)
    assert output.selected_partition_ids.shape == (len(part_ids),)
    assert output.selected_partition_ids.dtype == torch.long
    assert not output.is_node_attr("selected_partition_ids")
    assert not output.is_edge_attr("selected_partition_ids")
    assert type(output.num_selected_partitions) is int


def test_materialize_normalizes_ids_and_sets_phase_supervision(
    partitioned_graph: tuple[Data, MaterializedHomogeneousPartition],
) -> None:
    _, materialized = partitioned_graph
    first = materialized.materialize([2, 0, 2], phase="train")
    second = materialized.materialize([0, 2], phase="train")

    assert torch.equal(first.selected_partition_ids, torch.tensor([0, 2]))
    assert first.num_selected_partitions == 2
    for key in first.keys():
        first_value = first[key]
        second_value = second[key]
        if isinstance(first_value, torch.Tensor):
            assert torch.equal(first_value, second_value)
        else:
            assert first_value == second_value
    assert torch.equal(first.supervised_mask, first.train_mask)
    assert "val_mask" in first and "test_mask" in first


def test_materialized_data_is_writable_and_tensor_independent(
    partitioned_graph: tuple[Data, MaterializedHomogeneousPartition],
) -> None:
    source, materialized = partitioned_graph
    originals = {
        key: value.clone()
        for key, value in source.items()
        if isinstance(value, torch.Tensor)
    }
    output = materialized.materialize([0, 1, 2], phase="val")

    for key, source_value in source.items():
        output_value = output.get(key)
        if isinstance(source_value, torch.Tensor) and isinstance(output_value, torch.Tensor):
            assert not torch._C._is_alias_of(source_value, output_value)
    assert not torch._C._is_alias_of(output.val_mask, output.supervised_mask)

    output.x.fill_(-1)
    output.y.fill_(-1)
    output.train_mask.logical_not_()
    output.edge_index.fill_(0)
    output.edge_attr.fill_(-1)
    output.edge_weight.fill_(-1)
    output.graph_tensor.fill_(-1)
    output.supervised_mask.logical_not_()

    for key, original in originals.items():
        assert torch.equal(source[key], original), key
    assert "x_0" not in output
    assert "batch_0" not in output
    assert "cell_statistics" not in output


def test_partition_ids_for_phase_uses_canonical_partition_membership(
    partitioned_graph: tuple[Data, MaterializedHomogeneousPartition],
) -> None:
    source, materialized = partitioned_graph
    partition = materialized.partition
    expected = []
    for part_id in range(materialized.num_parts):
        members = _global_nodes(partition, [part_id])
        if bool(source.train_mask[members].any()):
            expected.append(part_id)

    actual = materialized.partition_ids_for_phase("train")

    assert actual == expected
    assert actual == sorted(actual)
    assert len(actual) == 2


@pytest.mark.parametrize(
    ("data", "message"),
    [
        (HeteroData(), "homogeneous.*Data"),
        (Data(num_nodes=3), "edge_index"),
        (
            Data(edge_index=torch.tensor([[0], [1]], dtype=torch.int32), num_nodes=2),
            "edge_index.*torch.long",
        ),
        (
            Data(edge_index=torch.tensor([[0, 1, 2]], dtype=torch.long), num_nodes=3),
            "edge_index.*shape",
        ),
        (
            Data(edge_index=torch.tensor([[0], [3]], dtype=torch.long), num_nodes=3),
            "edge_index.*range",
        ),
        (Data(edge_index=torch.empty((2, 0), dtype=torch.long), num_nodes=0), "node"),
    ],
)
def test_constructor_rejects_invalid_graphs_contextually(
    data: Data,
    message: str,
) -> None:
    with pytest.raises((TypeError, ValueError), match=message):
        MaterializedHomogeneousPartition(data, num_parts=1)


@pytest.mark.parametrize("num_parts", [True, 1.0, 0, -1, 11])
def test_constructor_rejects_invalid_partition_counts(num_parts: object) -> None:
    with pytest.raises((TypeError, ValueError), match="num_parts"):
        MaterializedHomogeneousPartition(_directed_graph(), num_parts=num_parts)


def test_constructor_rejects_non_boolean_recursive_flag() -> None:
    with pytest.raises(TypeError, match="recursive.*bool"):
        MaterializedHomogeneousPartition(
            _directed_graph(),
            num_parts=3,
            recursive=1,
        )


@pytest.mark.parametrize(
    ("mask", "message"),
    [
        (torch.zeros(10, dtype=torch.int64), "train_mask.*bool"),
        (torch.zeros((10, 1), dtype=torch.bool), "train_mask.*shape"),
        (torch.zeros(9, dtype=torch.bool), "train_mask.*10"),
    ],
)
def test_constructor_rejects_invalid_masks_contextually(
    mask: torch.Tensor,
    message: str,
) -> None:
    source = _directed_graph()
    source.train_mask = mask

    with pytest.raises((TypeError, ValueError), match=message):
        MaterializedHomogeneousPartition(source, num_parts=3)


@pytest.mark.parametrize(
    "part_ids",
    [[], [True], [1.0], ["1"], [-1], [3], [0, False], 1],
)
def test_materialize_rejects_invalid_partition_ids(
    part_ids: object,
    partitioned_graph: tuple[Data, MaterializedHomogeneousPartition],
) -> None:
    _, materialized = partitioned_graph

    with pytest.raises((TypeError, ValueError), match="partition_ids"):
        materialized.materialize(part_ids)


@pytest.mark.parametrize("phase", [1, "", "missing"])
def test_phase_selection_rejects_invalid_phase_or_missing_mask(
    phase: object,
    partitioned_graph: tuple[Data, MaterializedHomogeneousPartition],
) -> None:
    _, materialized = partitioned_graph

    with pytest.raises((TypeError, ValueError), match="phase"):
        materialized.partition_ids_for_phase(phase)


def test_materialize_rejects_invalid_phase_before_selecting_supervision(
    partitioned_graph: tuple[Data, MaterializedHomogeneousPartition],
) -> None:
    _, materialized = partitioned_graph

    with pytest.raises(ValueError, match="phase.*missing_mask"):
        materialized.materialize([0], phase="missing_mask")
