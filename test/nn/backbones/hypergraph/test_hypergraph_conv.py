"""Focused native-incidence tests for the PyG HypergraphConv backbone."""

from __future__ import annotations

import pytest
import torch

from topobench.nn.backbones.hypergraph.hypergraph_conv import (
    HypergraphConvBackbone,
)


def _incidence() -> torch.Tensor:
    return torch.tensor(
        [[0, 1, 1, 2, 3, 4], [0, 0, 1, 1, 2, 2]],
        dtype=torch.long,
    )


def test_hypergraph_conv_returns_node_embeddings_with_gradients() -> None:
    """The PyG baseline is differentiable over the native dense contract."""
    x = torch.randn(5, 3, requires_grad=True)
    model = HypergraphConvBackbone(
        in_channels=3,
        hidden_channels=7,
        num_layers=2,
        dropout=0.0,
    )

    output = model(x, _incidence())
    output.square().mean().backward()

    assert output.shape == (x.size(0), 7)
    assert x.grad is not None
    assert torch.isfinite(x.grad).all()
    assert all(parameter.grad is not None for parameter in model.parameters())


@pytest.mark.parametrize(
    ("kwargs", "message"),
    [
        ({"in_channels": 0, "hidden_channels": 4, "num_layers": 1}, "in_channels"),
        ({"in_channels": 3, "hidden_channels": 0, "num_layers": 1}, "hidden_channels"),
        ({"in_channels": 3, "hidden_channels": 4, "num_layers": 0}, "num_layers"),
        (
            {
                "in_channels": 3,
                "hidden_channels": 4,
                "num_layers": 1,
                "dropout": -0.1,
            },
            "dropout",
        ),
        (
            {
                "in_channels": 3,
                "hidden_channels": 4,
                "num_layers": 1,
                "dropout": 1.0,
            },
            "dropout",
        ),
    ],
)
def test_hypergraph_conv_validates_constructor(
    kwargs: dict[str, int | float], message: str
) -> None:
    """Invalid widths, depth, and dropout fail during construction."""
    with pytest.raises((TypeError, ValueError), match=message):
        HypergraphConvBackbone(**kwargs)


@pytest.mark.parametrize(
    "hyperedge_index",
    [
        torch.tensor([0, 1, 2], dtype=torch.long),
        torch.tensor([[0.0, 1.0], [0.0, 0.0]]),
        torch.tensor([[0, 5], [0, 0]], dtype=torch.long),
        torch.tensor([[0, 1], [0, -1]], dtype=torch.long),
        torch.tensor([[0, 1], [0, 2]], dtype=torch.long),
    ],
)
def test_hypergraph_conv_rejects_malformed_dense_incidence(
    hyperedge_index: torch.Tensor,
) -> None:
    """Malformed incidence is rejected before entering PyG message passing."""
    model = HypergraphConvBackbone(3, 4, 1)

    with pytest.raises((TypeError, ValueError), match="hyperedge_index"):
        model(torch.randn(3, 3), hyperedge_index)
