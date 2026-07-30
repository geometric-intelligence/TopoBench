"""Unit tests for the batched cell-complex HGT backbone."""

from __future__ import annotations

import pytest
import torch
from torch_geometric.data import Data

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
    assert any(gradient is not None for gradient in hgt_gradients)
    assert all(
        torch.isfinite(gradient).all()
        for gradient in hgt_gradients
        if gradient is not None
    )


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
