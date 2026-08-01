"""Unit tests for native graph DGM feature encoding."""

import pytest
import torch
from torch_geometric.data import Data

from topobench.nn.encoders import DGMStructureFeatureEncoder


@pytest.fixture
def sample_data() -> Data:
    """Return one graph using only native homogeneous fields."""
    return Data(
        x=torch.randn(10, 5),
        batch=torch.zeros(10, dtype=torch.long),
    )


def test_initialization_uses_scalar_channels() -> None:
    encoder = DGMStructureFeatureEncoder(in_channels=5, out_channels=64)

    assert encoder.in_channels == 5
    assert encoder.out_channels == 64
    assert encoder.encoder.base_enc.linear.in_features == 5
    assert encoder.encoder.base_enc.linear.out_features == 64


def test_rank_lists_are_not_supported() -> None:
    with pytest.raises(TypeError, match="in_channels must be an integer"):
        DGMStructureFeatureEncoder(in_channels=[5], out_channels=64)


def test_forward_pass_replaces_native_features(sample_data: Data) -> None:
    encoder = DGMStructureFeatureEncoder(in_channels=5, out_channels=64)

    output_data = encoder(sample_data)

    assert output_data is sample_data
    assert output_data.x.shape == (10, 64)
    assert output_data.dgm_aux.shape == (10, 64)
    assert "dgm_edge_index" in output_data
    assert "dgm_logprobs" in output_data


def test_dropout_configuration() -> None:
    encoder = DGMStructureFeatureEncoder(
        in_channels=5,
        out_channels=64,
        proj_dropout=0.5,
    )

    assert encoder.encoder.base_enc.dropout.p == 0.5
    assert encoder.encoder.embed_f.dropout.p == 0.5
