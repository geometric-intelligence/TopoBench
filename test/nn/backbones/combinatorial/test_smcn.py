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