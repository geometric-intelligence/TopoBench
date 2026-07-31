"""Unit tests for the batched cell-complex HGT backbone."""

from __future__ import annotations

import pytest
import torch
from torch_geometric.data import Data

from topobench.dataloader.utils import collate_fn
from topobench.nn.backbones.combinatorial.hgt import CellHGT

NEIGHBORHOODS = [
    "up_incidence-0",
    "down_incidence-1",
    "up_incidence-1",
    "down_incidence-2",
]


def make_complex(
    *,
    num_faces: int = 1,
    feature_dims: tuple[int, int, int] = (8, 8, 8),
    feature_shift: float = 0.0,
) -> Data:
    """Build a three-vertex, two-edge synthetic cell complex."""
    x_0 = (
        torch.arange(3 * feature_dims[0], dtype=torch.float32).reshape(
            3, feature_dims[0]
        )
        + feature_shift
    )
    x_1 = (
        torch.arange(2 * feature_dims[1], dtype=torch.float32).reshape(
            2, feature_dims[1]
        )
        + feature_shift
    )
    x_2 = (
        torch.arange(num_faces * feature_dims[2], dtype=torch.float32).reshape(
            num_faces, feature_dims[2]
        )
        + feature_shift
    )

    incidence_1 = torch.sparse_coo_tensor(
        torch.tensor([[0, 1, 1, 2], [0, 0, 1, 1]]),
        torch.ones(4),
        size=(3, 2),
    ).coalesce()

    if num_faces:
        incidence_2_indices = torch.tensor([[0, 1], [0, 0]])
        incidence_2_values = torch.ones(2)
    else:
        incidence_2_indices = torch.empty((2, 0), dtype=torch.long)
        incidence_2_values = torch.empty(0)
    incidence_2 = torch.sparse_coo_tensor(
        incidence_2_indices,
        incidence_2_values,
        size=(2, num_faces),
    ).coalesce()

    data = Data(x_0=x_0, x_1=x_1, x_2=x_2, y=torch.tensor([0]))
    data["incidence_1"] = incidence_1
    data["incidence_2"] = incidence_2
    data["up_incidence-0"] = incidence_1.t().coalesce()
    data["down_incidence-1"] = incidence_1
    data["up_incidence-1"] = incidence_2.t().coalesce()
    data["down_incidence-2"] = incidence_2
    data["shape"] = torch.tensor([[3, 2, num_faces]])
    return data


def make_model(
    *,
    neighborhoods: list[str] | None = None,
    dropout: float = 0.0,
) -> CellHGT:
    """Build the smallest useful test model."""
    return CellHGT(
        hidden_channels=8,
        num_layers=2,
        heads=2,
        neighborhoods=neighborhoods or NEIGHBORHOODS,
        max_rank=2,
        dropout=dropout,
        activation="relu",
    )


def loader_item(data: Data):
    """Adapt a data object to the input format expected by ``collate_fn``."""
    keys = list(data.keys())
    return ([data[key] for key in keys], keys)


def test_to_heterogeneous_inputs_preserves_types_and_direction():
    batch = make_complex()
    model = make_model()

    x_dict, edge_index_dict = model.to_heterogeneous_inputs(batch)

    assert list(x_dict) == ["rank_0", "rank_1", "rank_2"]
    assert x_dict["rank_0"] is batch.x_0
    assert x_dict["rank_1"] is batch.x_1
    assert x_dict["rank_2"] is batch.x_2

    expected_up_01 = torch.tensor([[0, 1, 1, 2], [0, 0, 1, 1]])
    expected_down_10 = torch.tensor([[0, 0, 1, 1], [0, 1, 1, 2]])
    expected_up_12 = torch.tensor([[0, 1], [0, 0]])
    expected_down_21 = torch.tensor([[0, 0], [0, 1]])

    torch.testing.assert_close(
        edge_index_dict[("rank_0", "up_incidence-0", "rank_1")],
        expected_up_01,
    )
    torch.testing.assert_close(
        edge_index_dict[("rank_1", "down_incidence-1", "rank_0")],
        expected_down_10,
    )
    torch.testing.assert_close(
        edge_index_dict[("rank_1", "up_incidence-1", "rank_2")],
        expected_up_12,
    )
    torch.testing.assert_close(
        edge_index_dict[("rank_2", "down_incidence-2", "rank_1")],
        expected_down_21,
    )


def test_constructor_rejects_invalid_hyperparameters():
    with pytest.raises(ValueError, match="divisible"):
        CellHGT(
            hidden_channels=10,
            num_layers=1,
            heads=4,
            neighborhoods=NEIGHBORHOODS,
        )

    with pytest.raises(ValueError, match="incidence"):
        CellHGT(
            hidden_channels=8,
            num_layers=1,
            heads=2,
            neighborhoods=["up_adjacency-0"],
        )


def test_constructor_rejects_nonpositive_heads():
    with pytest.raises(ValueError, match="heads must be.*positive"):
        CellHGT(
            hidden_channels=8,
            num_layers=1,
            heads=0,
            neighborhoods=NEIGHBORHOODS,
        )


def test_constructor_rejects_nonpositive_hidden_channels():
    for hidden_channels in (0, -8):
        with pytest.raises(
            ValueError, match="hidden_channels must be.*positive"
        ):
            CellHGT(
                hidden_channels=hidden_channels,
                num_layers=1,
                heads=2,
                neighborhoods=NEIGHBORHOODS,
            )


def test_state_dict_preserves_exact_historical_cell_hgt_keys():
    """Extraction keeps existing one-layer CellHGT checkpoint names."""
    model = CellHGT(
        hidden_channels=8,
        num_layers=1,
        heads=2,
        neighborhoods=NEIGHBORHOODS,
    )

    assert model.internal_metadata == (
        ["rank_0", "rank_1", "rank_2"],
        sorted(model.edge_types),
    )
    assert set(model.state_dict()) == {
        "convs.0.kqv_lin.lins.rank_0.weight",
        "convs.0.kqv_lin.lins.rank_0.bias",
        "convs.0.kqv_lin.lins.rank_1.weight",
        "convs.0.kqv_lin.lins.rank_1.bias",
        "convs.0.kqv_lin.lins.rank_2.weight",
        "convs.0.kqv_lin.lins.rank_2.bias",
        "convs.0.out_lin.lins.rank_0.weight",
        "convs.0.out_lin.lins.rank_0.bias",
        "convs.0.out_lin.lins.rank_1.weight",
        "convs.0.out_lin.lins.rank_1.bias",
        "convs.0.out_lin.lins.rank_2.weight",
        "convs.0.out_lin.lins.rank_2.bias",
        "convs.0.k_rel.weight",
        "convs.0.v_rel.weight",
        "convs.0.skip.rank_0",
        "convs.0.skip.rank_1",
        "convs.0.skip.rank_2",
        "convs.0.p_rel.rank_0__up_incidence-0__rank_1",
        "convs.0.p_rel.rank_1__down_incidence-1__rank_0",
        "convs.0.p_rel.rank_1__up_incidence-1__rank_2",
        "convs.0.p_rel.rank_2__down_incidence-2__rank_1",
        "norms.0.rank_0.weight",
        "norms.0.rank_0.bias",
        "norms.0.rank_1.weight",
        "norms.0.rank_1.bias",
        "norms.0.rank_2.weight",
        "norms.0.rank_2.bias",
    }


def test_all_present_empty_relations_match_legacy_cell_hgt_forward():
    """Explicit empty CellHGT relations still execute the historical layer."""
    batch = make_complex()
    model = make_model().eval()
    for neighborhood in NEIGHBORHOODS:
        matrix = batch[neighborhood]
        batch[neighborhood] = torch.sparse_coo_tensor(
            torch.empty((2, 0), dtype=torch.long),
            torch.empty(0),
            size=matrix.size(),
        ).coalesce()

    x_dict, edge_index_dict = model.to_heterogeneous_inputs(batch)
    legacy = dict(x_dict)
    for conv, norms in zip(model.convs, model.norms, strict=True):
        previous = legacy
        messages = conv(previous, edge_index_dict)
        legacy = {
            node_type: (
                old_features
                if messages.get(node_type) is None
                else model.dropout(
                    model.activation(norms[node_type](messages[node_type]))
                )
            )
            for node_type, old_features in previous.items()
        }

    output = model(batch)

    for rank in range(model.max_rank + 1):
        torch.testing.assert_close(output[rank], legacy[f"rank_{rank}"])
    assert any(
        not torch.equal(output[rank], batch[f"x_{rank}"])
        for rank in range(model.max_rank + 1)
    )


def test_constructor_rejects_negative_route_rank():
    with pytest.raises(ValueError, match="between 0 and max_rank"):
        CellHGT(
            hidden_channels=8,
            num_layers=1,
            heads=2,
            neighborhoods=["down_incidence-0"],
        )


def test_conversion_requires_every_configured_field():
    batch = make_complex()
    del batch["up_incidence-1"]

    with pytest.raises(KeyError, match="up_incidence-1"):
        make_model().to_heterogeneous_inputs(batch)


def test_forward_preserves_every_rank_shape_without_mutating_batch():
    batch = make_complex()
    original = {rank: batch[f"x_{rank}"].clone() for rank in range(3)}

    output = make_model()(batch)

    assert set(output) == {0, 1, 2}
    for rank in range(3):
        assert output[rank].shape == original[rank].shape
        torch.testing.assert_close(batch[f"x_{rank}"], original[rank])


def test_backward_produces_finite_hgt_gradients():
    batch = make_complex()
    model = make_model()

    output = model(batch)
    loss = sum(value.square().mean() for value in output.values())
    loss.backward()

    hgt_gradients = [
        parameter.grad
        for name, parameter in model.named_parameters()
        if "convs" in name and parameter.requires_grad
    ]
    assert hgt_gradients
    for gradient in hgt_gradients:
        assert gradient is not None
        assert torch.isfinite(gradient).all()


def test_eval_mode_is_deterministic():
    batch = make_complex()
    model = make_model(dropout=0.25).eval()

    first = model(batch)
    second = model(batch)

    for rank in range(3):
        torch.testing.assert_close(first[rank], second[rank])


def test_forward_handles_an_empty_rank_two():
    batch = make_complex(num_faces=0)

    output = make_model()(batch)

    assert output[0].shape == (3, 8)
    assert output[1].shape == (2, 8)
    assert output[2].shape == (0, 8)
    assert all(torch.isfinite(value).all() for value in output.values())


def test_rank_without_destination_relation_is_carried_forward():
    batch = make_complex()
    model = make_model(neighborhoods=["down_incidence-1"])

    output = model(batch)

    torch.testing.assert_close(output[1], batch.x_1)
    torch.testing.assert_close(output[2], batch.x_2)


def test_collated_relations_never_cross_graph_boundaries():
    graph_a = make_complex(num_faces=1)
    graph_b = make_complex(num_faces=1, feature_shift=100.0)
    batch = collate_fn([loader_item(graph_a), loader_item(graph_b)])

    _, edge_index_dict = make_model().to_heterogeneous_inputs(batch)
    memberships = {f"rank_{rank}": batch[f"batch_{rank}"] for rank in range(3)}

    for (
        source_type,
        _,
        destination_type,
    ), edge_index in edge_index_dict.items():
        if edge_index.numel() == 0:
            continue
        source_membership = memberships[source_type][edge_index[0]]
        destination_membership = memberships[destination_type][edge_index[1]]
        assert torch.equal(source_membership, destination_membership)

    face_edge_types = [
        ("rank_1", "up_incidence-1", "rank_2"),
        ("rank_2", "down_incidence-2", "rank_1"),
    ]
    for source_type, relation, destination_type in face_edge_types:
        edge_index = edge_index_dict[(source_type, relation, destination_type)]
        assert torch.any(memberships[source_type][edge_index[0]] == 1)
        assert torch.any(memberships[destination_type][edge_index[1]] == 1)


@pytest.mark.parametrize("num_faces_b", [0, 1])
def test_eval_output_is_equal_alone_and_in_a_batch(num_faces_b):
    graph_a = make_complex(num_faces=1)
    graph_b = make_complex(num_faces=num_faces_b, feature_shift=100.0)
    graph_a_batch = collate_fn([loader_item(graph_a)])
    graph_b_batch = collate_fn([loader_item(graph_b)])
    combined_batch = collate_fn([loader_item(graph_a), loader_item(graph_b)])
    model = make_model().eval()

    output_a = model(graph_a_batch)
    output_b = model(graph_b_batch)
    combined_output = model(combined_batch)

    counts_a = [graph_a[f"x_{rank}"].shape[0] for rank in range(3)]
    counts_b = [graph_b[f"x_{rank}"].shape[0] for rank in range(3)]
    for rank, (count_a, count_b) in enumerate(
        zip(counts_a, counts_b, strict=True)
    ):
        assert combined_output[rank].shape[0] == count_a + count_b
        torch.testing.assert_close(
            combined_output[rank][:count_a],
            output_a[rank],
            rtol=1e-5,
            atol=1e-6,
        )
        torch.testing.assert_close(
            combined_output[rank][count_a : count_a + count_b],
            output_b[rank],
            rtol=1e-5,
            atol=1e-6,
        )


def test_batch_with_no_faces_in_any_graph_is_supported():
    graph_a = make_complex(num_faces=0)
    graph_b = make_complex(num_faces=0, feature_shift=100.0)
    batch = collate_fn([loader_item(graph_a), loader_item(graph_b)])

    output = make_model()(batch)

    assert output[0].shape == (6, 8)
    assert output[1].shape == (4, 8)
    assert output[2].shape == (0, 8)
