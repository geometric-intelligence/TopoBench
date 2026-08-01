"""Behavioral contract for native homogeneous graph feature encoding."""

import pytest
import torch
from torch.nn.parameter import UninitializedParameter
from torch_geometric.data import Batch, Data, HeteroData
from torch_geometric.nn import GraphNorm

from topobench.nn.encoders import GraphNodeFeatureEncoder


def test_parameters_are_created_eagerly() -> None:
    encoder = GraphNodeFeatureEncoder(3, 8, dropout=0.25)

    assert encoder.projection.weight.shape == (8, 3)
    assert encoder.norm.weight.shape == (3,)
    assert not any(
        isinstance(parameter, UninitializedParameter)
        for parameter in encoder.parameters()
    )


@pytest.mark.parametrize(
    ("in_channels", "out_channels", "message"),
    [
        ([3], 8, "in_channels must be an integer"),
        (3, [8], "out_channels must be an integer"),
    ],
)
def test_rank_lists_are_not_supported(
    in_channels: object,
    out_channels: object,
    message: str,
) -> None:
    with pytest.raises(TypeError, match=message):
        GraphNodeFeatureEncoder(in_channels, out_channels)


def test_requires_homogeneous_data() -> None:
    encoder = GraphNodeFeatureEncoder(3, 8)

    with pytest.raises(
        TypeError, match="GraphNodeFeatureEncoder requires homogeneous Data"
    ):
        encoder(HeteroData())


@pytest.mark.parametrize(
    ("x", "error_type", "message"),
    [
        (torch.ones(4), ValueError, "data.x must be a rank-2 tensor"),
        (
            torch.ones(4, 3, 1),
            ValueError,
            "data.x must be a rank-2 tensor",
        ),
        (
            torch.ones(4, 3, dtype=torch.long),
            TypeError,
            "data.x must have a floating dtype",
        ),
    ],
)
def test_validates_feature_rank_and_dtype(
    x: torch.Tensor,
    error_type: type[Exception],
    message: str,
) -> None:
    encoder = GraphNodeFeatureEncoder(3, 8)

    with pytest.raises(error_type, match=message):
        encoder(Data(x=x))


def test_graph_norm_uses_native_batch_assignments() -> None:
    graph_a = Data(x=torch.tensor([[1.0, 2.0], [3.0, 6.0]]))
    graph_b = Data(x=torch.tensor([[20.0, 5.0], [24.0, 9.0]]))
    data = Batch.from_data_list([graph_a, graph_b])
    original_x = data.x.clone()
    encoder = GraphNodeFeatureEncoder(2, 2, dropout=0.0)
    encoder.eval()
    with torch.no_grad():
        encoder.projection.weight.copy_(torch.eye(2))
        encoder.projection.bias.zero_()
        expected = torch.relu(GraphNorm(2)(original_x, batch=data.batch))

    encoder(data)

    torch.testing.assert_close(data.x, expected)


def test_missing_batch_uses_single_graph_fallback() -> None:
    data = Data(x=torch.tensor([[1.0, 2.0], [3.0, 6.0]]))
    original_x = data.x.clone()
    encoder = GraphNodeFeatureEncoder(2, 2, dropout=0.0)
    encoder.eval()
    with torch.no_grad():
        encoder.projection.weight.copy_(torch.eye(2))
        encoder.projection.bias.zero_()
        expected = torch.relu(GraphNorm(2)(original_x))

    encoder(data)

    torch.testing.assert_close(data.x, expected)


def test_graph_node_encoder_replaces_native_features() -> None:
    graph_a = Data(x=torch.randn(3, 3))
    graph_b = Data(x=torch.randn(5, 3))
    data = Batch.from_data_list([graph_a, graph_b])
    original_x = data.x
    encoder = GraphNodeFeatureEncoder(3, 8, dropout=0.0)

    result = encoder(data)

    assert result is data
    assert result.x.shape == (graph_a.num_nodes + graph_b.num_nodes, 8)
    assert result.x is not original_x
