"""Tests for the SMCN combinatorial backbone skeleton."""

import pytest
import torch
from torch_geometric.data import Data

from topobench.nn.backbones.combinatorial.smcn import SMCN


def test_smcn_forward_returns_rank_dict():
    """SMCN should return updated rank-wise features."""
    in_channels = 8
    hidden_channels = 16

    batch = Data(
        x_0=torch.randn(5, in_channels),
        x_1=torch.randn(7, in_channels),
        x_2=torch.randn(3, in_channels),
    )

    model = SMCN(
        in_channels=in_channels,
        hidden_channels=hidden_channels,
        neighborhoods=[
            "up_adjacency-1",
            "up_incidence-0",
            "down_incidence-2",
        ],
        layers=1,
        activation="relu",
    )

    out = model(batch)

    assert set(out.keys()) == {0, 1, 2}
    assert out[0].shape == (5, hidden_channels)
    assert out[1].shape == (7, hidden_channels)
    assert out[2].shape == (3, hidden_channels)


def test_smcn_supports_multiple_layers():
    """SMCN should support more than one placeholder update layer."""
    in_channels = 8
    hidden_channels = 16

    batch = Data(
        x_0=torch.randn(5, in_channels),
        x_1=torch.randn(7, in_channels),
        x_2=torch.randn(3, in_channels),
    )

    model = SMCN(
        in_channels=in_channels,
        hidden_channels=hidden_channels,
        layers=2,
        activation="relu",
    )

    out = model(batch)

    assert out[0].shape == (5, hidden_channels)
    assert out[1].shape == (7, hidden_channels)
    assert out[2].shape == (3, hidden_channels)


def test_smcn_rejects_unknown_activation():
    """SMCN should fail clearly for unsupported activations."""
    with pytest.raises(ValueError, match="Unsupported activation"):
        SMCN(
            in_channels=8,
            hidden_channels=16,
            activation="not_an_activation",
        )


def test_smcn_builds_binary_rank02_incidence():
    """SMCN should compose rank 0-to-2 incidence from incidences 0-to-1 and 1-to-2."""
    incidence_1 = torch.tensor(
        [
            [1.0, 0.0, 1.0],
            [1.0, 1.0, 0.0],
            [0.0, 1.0, 1.0],
        ]
    ).to_sparse()
    incidence_2 = torch.tensor([[1.0], [1.0], [1.0]]).to_sparse()
    batch = Data(
        x_0=torch.randn(3, 8),
        x_2=torch.randn(1, 8),
        incidence_1=incidence_1,
        incidence_2=incidence_2,
    )
    model = SMCN(in_channels=8, hidden_channels=16)

    subcomplex = model.build_rank02_subcomplex(batch)
    incidence_0_2 = subcomplex["incidence_0_2"]

    assert incidence_0_2.is_sparse
    assert incidence_0_2.shape == (3, 1)
    assert torch.equal(
        incidence_0_2.to_dense(),
        torch.ones(3, 1),
    )
    assert torch.equal(subcomplex["low_indices"], torch.tensor([0, 1, 2]))
    assert torch.equal(subcomplex["high_indices"], torch.tensor([0, 0, 0]))
    assert torch.equal(subcomplex["binary_marking"], torch.ones(3))
    assert subcomplex["tuple_features"].shape == (3, 16)


def test_smcn_builds_empty_rank02_subcomplex_when_no_rank2_cells():
    """SMCN should handle batches without rank-2 cells."""
    incidence_1 = torch.tensor(
        [
            [1.0, 0.0, 1.0],
            [1.0, 1.0, 0.0],
            [0.0, 1.0, 1.0],
        ]
    ).to_sparse()
    incidence_2 = torch.sparse_coo_tensor(size=(3, 0)).coalesce()
    batch = Data(
        x_0=torch.randn(3, 8),
        x_2=torch.empty(0, 8),
        incidence_1=incidence_1,
        incidence_2=incidence_2,
    )
    model = SMCN(in_channels=8, hidden_channels=16)

    subcomplex = model.build_rank02_subcomplex(batch)
    incidence_0_2 = subcomplex["incidence_0_2"]

    assert incidence_0_2.is_sparse
    assert incidence_0_2.shape == (3, 0)
    assert incidence_0_2._nnz() == 0
    assert torch.equal(subcomplex["low_indices"], torch.empty(0, dtype=torch.long))
    assert torch.equal(subcomplex["high_indices"], torch.empty(0, dtype=torch.long))
    assert torch.equal(subcomplex["binary_marking"], torch.empty(0))
    assert subcomplex["tuple_features"].shape == (0, 16)


def test_smcn_pools_rank02_tuple_features_to_rank0():
    """SMCN should sum tuple features back onto their rank-0 cells."""
    subcomplex = {
        "tuple_features": torch.tensor(
            [
                [1.0, 2.0],
                [3.0, 4.0],
                [5.0, 6.0],
            ]
        ),
        "low_indices": torch.tensor([0, 1, 0]),
    }

    pooled = SMCN.pool_rank02_to_rank0(subcomplex, num_low_cells=3)

    assert torch.equal(
        pooled,
        torch.tensor(
            [
                [6.0, 8.0],
                [3.0, 4.0],
                [0.0, 0.0],
            ]
        ),
    )


def test_smcn_builds_and_pools_rank02_subcomplex():
    """SMCN should build rank-0/2 tuples and pool them back to rank 0."""
    incidence_1 = torch.tensor(
        [
            [1.0, 0.0, 1.0],
            [1.0, 1.0, 0.0],
            [0.0, 1.0, 1.0],
        ]
    ).to_sparse()
    incidence_2 = torch.tensor([[1.0], [1.0], [1.0]]).to_sparse()
    batch = Data(
        x_0=torch.ones(3, 8),
        x_2=2 * torch.ones(1, 8),
        incidence_1=incidence_1,
        incidence_2=incidence_2,
    )
    model = SMCN(in_channels=8, hidden_channels=16)

    subcomplex = model.build_rank02_subcomplex(batch)
    pooled = model.pool_rank02_to_rank0(
        subcomplex,
        num_low_cells=batch.x_0.size(0),
    )

    assert pooled.shape == (3, 16)


def test_smcn_filters_rank02_tuples_across_batched_graphs():
    """SMCN should not create node-face tuples across different graphs."""
    incidence_1 = torch.eye(4).to_sparse()
    incidence_2 = torch.tensor(
        [
            [1.0, 0.0],
            [1.0, 0.0],
            [0.0, 1.0],
            [0.0, 1.0],
        ]
    ).to_sparse()
    batch = Data(
        x_0=torch.ones(4, 8),
        x_2=2 * torch.ones(2, 8),
        incidence_1=incidence_1,
        incidence_2=incidence_2,
        batch_0=torch.tensor([0, 0, 1, 1]),
        batch_2=torch.tensor([0, 1]),
    )
    model = SMCN(in_channels=8, hidden_channels=16)

    subcomplex = model.build_rank02_subcomplex(batch)

    assert torch.equal(subcomplex["low_indices"], torch.tensor([0, 1, 2, 3]))
    assert torch.equal(subcomplex["high_indices"], torch.tensor([0, 0, 1, 1]))
    assert torch.equal(subcomplex["binary_marking"], torch.ones(4))
    assert subcomplex["tuple_features"].shape == (4, 16)


def test_smcn_pools_empty_rank02_tuple_features_to_rank0():
    """SMCN should return zeros when there are no rank-0/2 tuples."""
    subcomplex = {
        "tuple_features": torch.empty(0, 2),
        "low_indices": torch.empty(0, dtype=torch.long),
    }

    pooled = SMCN.pool_rank02_to_rank0(subcomplex, num_low_cells=3)

    assert torch.equal(pooled, torch.zeros(3, 2))
