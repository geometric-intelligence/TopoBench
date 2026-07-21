"""Unit tests for COSIMO."""

import pytest
import torch

from topobench.nn.backbones.simplicial.cosimo import COSIMO, COSIMOLayer
from topobench.transforms.liftings.graph2simplicial import (
    SimplicialCliqueLifting,
)


def _simplicial_inputs(graph):
    lifting = SimplicialCliqueLifting(complex_dim=3, signed=True)
    data = lifting(graph)
    x_all = (data.x, data.x_1, data.x_2)
    laplacian_all = (
        data.hodge_laplacian_0,
        data.down_laplacian_1,
        data.up_laplacian_1,
        data.down_laplacian_2,
        data.up_laplacian_2,
    )
    incidence_all = (data.incidence_1, data.incidence_2)
    return x_all, laplacian_all, incidence_all


def test_cosimo_forward_shapes(simple_graph_1):
    """Test COSIMO output shapes on a lifted clique complex."""
    x_all, laplacian_all, incidence_all = _simplicial_inputs(simple_graph_1)
    model = COSIMO(
        in_channels_all=tuple(x.shape[1] for x in x_all),
        hidden_channels_all=(4, 4, 4),
        n_layers=2,
        t_init=0.5,
        num_branches=3,
        diffusion_method="taylor",
        taylor_order=4,
    )

    out = model(x_all, laplacian_all, incidence_all)

    assert len(out) == 3
    assert out[0].shape == (x_all[0].shape[0], 4)
    assert out[1].shape == (x_all[1].shape[0], 4)
    assert out[2].shape == (x_all[2].shape[0], 4)
    assert all(torch.isfinite(x).all() for x in out)


def test_cosimo_exact_diffusion_is_deterministic(simple_graph_1):
    """Test exact heat diffusion branch on a small complex."""
    x_all, laplacian_all, incidence_all = _simplicial_inputs(simple_graph_1)
    torch.manual_seed(0)
    model = COSIMO(
        in_channels_all=tuple(x.shape[1] for x in x_all),
        hidden_channels_all=(3, 3, 3),
        n_layers=1,
        diffusion_method="exact",
        update_func=None,
    )

    out_1 = model(x_all, laplacian_all, incidence_all)
    out_2 = model(
        tuple(x.clone() for x in x_all), laplacian_all, incidence_all
    )

    for tensor_1, tensor_2 in zip(out_1, out_2, strict=True):
        assert torch.equal(tensor_1, tensor_2)


def test_cosimo_supports_rank_specific_hidden_channels(simple_graph_1):
    """Test COSIMO when hidden widths differ across simplex ranks."""
    x_all, laplacian_all, incidence_all = _simplicial_inputs(simple_graph_1)
    model = COSIMO(
        in_channels_all=tuple(x.shape[1] for x in x_all),
        hidden_channels_all=(2, 3, 4),
        n_layers=2,
        num_branches=2,
    )

    out = model(x_all, laplacian_all, incidence_all)

    assert out[0].shape == (x_all[0].shape[0], 2)
    assert out[1].shape == (x_all[1].shape[0], 3)
    assert out[2].shape == (x_all[2].shape[0], 4)


def test_cosimo_accepts_four_laplacian_tuple(simple_graph_1):
    """Test fallback when only a lower 2-Laplacian is provided."""
    x_all, laplacian_all, incidence_all = _simplicial_inputs(simple_graph_1)
    model = COSIMO(
        in_channels_all=tuple(x.shape[1] for x in x_all),
        hidden_channels_all=(2, 2, 2),
        n_layers=1,
    )

    out = model(x_all, laplacian_all[:4], incidence_all)

    assert out[2].shape == (x_all[2].shape[0], 2)


def test_cosimo_layer_uses_independent_branch_times():
    """Test that each COSIMO branch has its own learnable receptive fields."""
    layer = COSIMOLayer(
        in_channels=(2, 2, 2),
        out_channels=(2, 2, 2),
        num_branches=3,
    )

    assert layer.num_branches == 3
    assert len(layer.raw_times) == 3 * len(layer.branch_names())
    assert "0_x1_lower" in layer.raw_times
    assert "2_x1_lower" in layer.raw_times


@pytest.mark.parametrize(
    ("kwargs", "match"),
    [
        ({"diffusion_method": "bad"}, "diffusion_method"),
        ({"update_func": "bad"}, "update_func"),
        ({"t_init": 0.0}, "t_init"),
        ({"num_branches": 0}, "num_branches"),
        ({"taylor_order": 0}, "taylor_order"),
    ],
)
def test_cosimo_layer_validation(kwargs, match):
    """Test validation of COSIMO layer options."""
    with pytest.raises(ValueError, match=match):
        COSIMOLayer(
            in_channels=(2, 2, 2),
            out_channels=(2, 2, 2),
            **kwargs,
        )
