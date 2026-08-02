"""Behavioral contract for native homogeneous graph feature encoding."""

import pytest
import torch
from torch.nn.parameter import UninitializedParameter
from omegaconf import OmegaConf
from torch_geometric.data import Batch, Data, HeteroData
from torch_geometric.nn import GraphNorm

from topobench.data.features import (
    OGB_ATOM_FEATURE_CARDINALITIES,
    encode_categorical_columns,
)
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

    result = encoder(data)

    torch.testing.assert_close(result.x, expected)
    torch.testing.assert_close(data.x, original_x)


def test_missing_batch_uses_single_graph_fallback() -> None:
    data = Data(x=torch.tensor([[1.0, 2.0], [3.0, 6.0]]))
    original_x = data.x.clone()
    encoder = GraphNodeFeatureEncoder(2, 2, dropout=0.0)
    encoder.eval()
    with torch.no_grad():
        encoder.projection.weight.copy_(torch.eye(2))
        encoder.projection.bias.zero_()
        expected = torch.relu(GraphNorm(2)(original_x))

    result = encoder(data)

    torch.testing.assert_close(result.x, expected)
    torch.testing.assert_close(data.x, original_x)


def test_graph_node_encoder_replaces_native_features() -> None:
    graph_a = Data(x=torch.randn(3, 3))
    graph_b = Data(x=torch.randn(5, 3))
    data = Batch.from_data_list([graph_a, graph_b])
    original_x = data.x
    encoder = GraphNodeFeatureEncoder(3, 8, dropout=0.0)

    result = encoder(data)

    assert result is not data
    assert data.x is original_x
    assert result.x.shape == (graph_a.num_nodes + graph_b.num_nodes, 8)
    assert result.x is not original_x


def test_categorical_batch_encoding_matches_canonical_one_hot_path() -> None:
    categories = torch.tensor(
        [
            [0, 0, 0, 0, 0, 0, 0, 0, 0],
            [118, 4, 11, 11, 9, 5, 5, 1, 1],
            [6, 0, 3, 1, 2, 0, 4, 1, 0],
        ],
        dtype=torch.long,
    )
    canonical = encode_categorical_columns(
        categories,
        OGB_ATOM_FEATURE_CARDINALITIES,
    )
    compact_input = categories.clone()
    categorical_data = Data(x=compact_input)
    canonical_data = Data(x=canonical)
    categorical_encoder = GraphNodeFeatureEncoder(
        9,
        174,
        dropout=0.0,
        encoding_mode="categorical_one_hot",
        categorical_cardinalities=OGB_ATOM_FEATURE_CARDINALITIES,
    )
    continuous_encoder = GraphNodeFeatureEncoder(174, 174, dropout=0.0)
    continuous_encoder.load_state_dict(categorical_encoder.state_dict())
    categorical_encoder.eval()
    continuous_encoder.eval()

    categorical_result = categorical_encoder(categorical_data)
    canonical_result = continuous_encoder(canonical_data)

    assert categorical_result.x.shape == (3, 174)
    torch.testing.assert_close(categorical_result.x, canonical_result.x)
    torch.testing.assert_close(categories, compact_input)


def test_categorical_prefix_encoding_preserves_continuous_suffix() -> None:
    categories = torch.tensor(
        [
            [0, 0, 0, 0, 0, 0, 0, 0, 0],
            [118, 4, 11, 11, 9, 5, 5, 1, 1],
        ],
        dtype=torch.float,
    )
    suffix = torch.tensor([[0.25, -1.5], [2.0, 3.25]])
    mixed = torch.cat((categories, suffix), dim=1)
    canonical = torch.cat(
        (
            encode_categorical_columns(
                categories.to(dtype=torch.long),
                OGB_ATOM_FEATURE_CARDINALITIES,
            ),
            suffix,
        ),
        dim=1,
    )
    categorical_encoder = GraphNodeFeatureEncoder(
        11,
        8,
        dropout=0.0,
        encoding_mode="categorical_one_hot",
        categorical_cardinalities=OGB_ATOM_FEATURE_CARDINALITIES,
    )
    continuous_encoder = GraphNodeFeatureEncoder(176, 8, dropout=0.0)
    continuous_encoder.load_state_dict(categorical_encoder.state_dict())
    categorical_encoder.eval()
    continuous_encoder.eval()

    categorical_result = categorical_encoder(Data(x=mixed))
    canonical_result = continuous_encoder(Data(x=canonical))

    assert categorical_encoder.encoded_in_channels == 176
    torch.testing.assert_close(categorical_result.x, canonical_result.x)
    torch.testing.assert_close(mixed[:, 9:], suffix)


def test_categorical_encoder_allocates_only_for_current_batch() -> None:
    categories = torch.zeros((5, 9), dtype=torch.long)
    encoder = GraphNodeFeatureEncoder(
        9,
        8,
        encoding_mode="categorical_one_hot",
        categorical_cardinalities=OGB_ATOM_FEATURE_CARDINALITIES,
    )
    assert encoder.in_channels == 9
    assert encoder.encoded_in_channels == 174
    assert encoder.norm.weight.shape == (174,)
    assert encoder.projection.weight.shape == (8, 174)

    result = encoder(Data(x=categories))

    assert categories.shape == (5, 9)
    assert categories.dtype == torch.long
    assert result.x.shape == (5, 8)


@pytest.mark.parametrize(
    ("x", "error_type", "message"),
    [
        (
            torch.zeros((2, 9), dtype=torch.bool),
            TypeError,
            "integral dtype",
        ),
        (
            torch.tensor(
                [[0.5, 0, 0, 0, 0, 0, 0, 0, 0]],
                dtype=torch.float,
            ),
            ValueError,
            "exact integral categories",
        ),
        (
            torch.zeros((2, 8), dtype=torch.long),
            ValueError,
            "exactly 9.*columns",
        ),
        (
            torch.zeros((2, 9, 1), dtype=torch.long),
            ValueError,
            "rank-2",
        ),
        (
            torch.tensor(
                [[-1, 0, 0, 0, 0, 0, 0, 0, 0]],
                dtype=torch.long,
            ),
            ValueError,
            "column 0.*range",
        ),
        (
            torch.tensor(
                [[0, 5, 0, 0, 0, 0, 0, 0, 0]],
                dtype=torch.long,
            ),
            ValueError,
            "column 1.*range",
        ),
    ],
)
def test_categorical_encoder_rejects_invalid_categories_before_projection(
    x: torch.Tensor,
    error_type: type[Exception],
    message: str,
) -> None:
    encoder = GraphNodeFeatureEncoder(
        9,
        8,
        encoding_mode="categorical_one_hot",
        categorical_cardinalities=OGB_ATOM_FEATURE_CARDINALITIES,
    )
    projection_called = False

    def mark_projection_called(_module, _inputs):
        nonlocal projection_called
        projection_called = True

    handle = encoder.projection.register_forward_pre_hook(mark_projection_called)
    data = Data(x=x)
    with pytest.raises(error_type, match=message):
        encoder(data)
    handle.remove()

    assert not projection_called
    assert data.x is x


def test_categorical_encoder_rejects_nonfinite_continuous_suffix() -> None:
    x = torch.zeros((2, 10), dtype=torch.float)
    x[0, 9] = torch.nan
    encoder = GraphNodeFeatureEncoder(
        10,
        8,
        encoding_mode="categorical_one_hot",
        categorical_cardinalities=OGB_ATOM_FEATURE_CARDINALITIES,
    )

    with pytest.raises(ValueError, match="continuous suffix.*finite"):
        encoder(Data(x=x))


def test_categorical_encoder_accepts_hydra_cardinality_sequence() -> None:
    cardinalities = OmegaConf.create(
        list(OGB_ATOM_FEATURE_CARDINALITIES)
    )

    encoder = GraphNodeFeatureEncoder(
        9,
        8,
        encoding_mode="categorical_one_hot",
        categorical_cardinalities=cardinalities,
    )

    assert encoder.categorical_cardinalities == (
        OGB_ATOM_FEATURE_CARDINALITIES
    )


@pytest.mark.parametrize(
    ("kwargs", "error_type", "message"),
    [
        (
            {"encoding_mode": "embedding"},
            ValueError,
            "encoding_mode",
        ),
        (
            {
                "encoding_mode": "categorical_one_hot",
                "categorical_cardinalities": [119, True],
            },
            ValueError,
            "positive integers",
        ),
        (
            {
                "encoding_mode": "categorical_one_hot",
                "categorical_cardinalities": [2] * 10,
            },
            ValueError,
            "may not exceed in_channels",
        ),
        (
            {
                "encoding_mode": "categorical_one_hot",
                "categorical_cardinalities": "123456789",
            },
            TypeError,
            "ordered sequence",
        ),
        (
            {
                "encoding_mode": "categorical_one_hot",
                "categorical_cardinalities": {
                    index: cardinality
                    for index, cardinality in enumerate(
                        OGB_ATOM_FEATURE_CARDINALITIES,
                        start=1,
                    )
                },
            },
            TypeError,
            "ordered sequence",
        ),
    ],
)
def test_categorical_encoder_validates_declared_contract(
    kwargs: dict[str, object],
    error_type: type[Exception],
    message: str,
) -> None:
    with pytest.raises(error_type, match=message):
        GraphNodeFeatureEncoder(9, 8, **kwargs)
