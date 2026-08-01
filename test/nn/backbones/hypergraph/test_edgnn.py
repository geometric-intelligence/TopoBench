"""Focused native-incidence contract tests for EDGNN."""

from __future__ import annotations

import pytest
import torch

from topobench.nn.backbones.hypergraph.edgnn import EDGNN


def _incidence() -> torch.Tensor:
    return torch.tensor(
        [[0, 1, 1, 2, 3, 4], [0, 0, 1, 1, 2, 2]],
        dtype=torch.long,
    )


def test_edgnn_returns_native_node_embeddings_with_gradients() -> None:
    """Dense incidence produces one differentiable embedding per node."""
    hidden_channels = 4
    x = torch.randn(5, hidden_channels, requires_grad=True)
    model = EDGNN(
        num_features=hidden_channels,
        input_dropout=0.0,
        dropout=0.0,
        MLP_num_layers=1,
        All_num_layers=2,
        aggregate="sum",
    )

    output = model(x, _incidence())
    output.square().mean().backward()

    assert isinstance(output, torch.Tensor)
    assert output.shape == (x.size(0), hidden_channels)
    assert x.grad is not None
    assert torch.isfinite(x.grad).all()
    assert any(parameter.grad is not None for parameter in model.parameters())


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
def test_edgnn_rejects_malformed_dense_incidence(
    hyperedge_index: torch.Tensor,
) -> None:
    """Only bounded, contiguous dense long incidence reaches EquivSet."""
    model = EDGNN(num_features=3, All_num_layers=1)

    with pytest.raises((TypeError, ValueError), match="hyperedge_index"):
        model(torch.randn(3, 3), hyperedge_index)


def test_edgnn_rejects_sparse_incidence() -> None:
    """Sparse rank-era incidence has no compatibility path."""
    dense = _incidence()
    sparse = torch.sparse_coo_tensor(
        dense,
        torch.ones(dense.size(1)),
        size=(5, 3),
    )

    with pytest.raises((TypeError, ValueError), match="dense"):
        EDGNN(num_features=4)(torch.randn(5, 4), sparse)
