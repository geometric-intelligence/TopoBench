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


def _model(*, input_dropout: float = 0.0, edconv_type: str = "EquivSet") -> EDGNN:
    return EDGNN(
        num_features=4,
        input_dropout=input_dropout,
        dropout=0.0,
        MLP_num_layers=1,
        All_num_layers=1,
        edconv_type=edconv_type,
        aggregate="sum",
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


def test_edgnn_input_dropout_is_seeded_and_training_only() -> None:
    """Input dropout is repeatable in training and disabled in evaluation."""
    x = torch.arange(20, dtype=torch.float).reshape(5, 4)
    with torch.random.fork_rng(devices=[]):
        torch.manual_seed(17)
        reference = _model(input_dropout=0.0)
        regularized = _model(input_dropout=0.5)
        regularized.load_state_dict(reference.state_dict())

        reference.train()
        regularized.train()
        torch.manual_seed(23)
        reference_output = reference(x, _incidence())
        torch.manual_seed(23)
        regularized_output = regularized(x, _incidence())
        torch.manual_seed(23)
        repeated_output = regularized(x, _incidence())

        assert not torch.equal(regularized_output, reference_output)
        torch.testing.assert_close(regularized_output, repeated_output)

        reference.eval()
        regularized.eval()
        torch.testing.assert_close(
            regularized(x, _incidence()),
            reference(x, _incidence()),
            rtol=0.0,
            atol=0.0,
        )


def test_edgnn_input_dropout_preserves_gradients() -> None:
    """Regularized input features retain a finite autograd path."""
    with torch.random.fork_rng(devices=[]):
        torch.manual_seed(29)
        x = torch.randn(5, 4, requires_grad=True)
        model = _model(input_dropout=0.5)
        model.train()

        torch.manual_seed(31)
        model(x, _incidence()).square().mean().backward()
    assert x.grad is not None
    assert torch.isfinite(x.grad).all()
    assert torch.count_nonzero(x.grad) > 0
    assert any(
        parameter.grad is not None
        and torch.isfinite(parameter.grad).all()
        and torch.count_nonzero(parameter.grad) > 0
        for parameter in model.parameters()
    )


@pytest.mark.parametrize(
    ("node_ids", "isolated_nodes"),
    [
        ([0, 1, 2, 3, 4], []),
        ([0, 1, 3, 4], [2]),
        ([0, 1, 2, 3], [4]),
    ],
    ids=("none", "interior", "trailing"),
)
def test_edgnn_meandeg_returns_full_finite_output_with_isolated_nodes(
    node_ids: list[int],
    isolated_nodes: list[int],
) -> None:
    """MeanDeg represents zero degree by log(clamp(degree, min=1)) == 0."""
    x = torch.arange(20, dtype=torch.float).reshape(5, 4)
    incidence = torch.tensor(
        [node_ids, [0, 0, *([1] * (len(node_ids) - 2))]],
        dtype=torch.long,
    )
    original_x = x.clone()
    original_incidence = incidence.clone()

    output = _model(edconv_type="MeanDeg")(x, incidence)

    assert output.shape == x.shape
    assert torch.isfinite(output).all()
    assert torch.equal(x, original_x)
    assert torch.equal(incidence, original_incidence)
    if isolated_nodes:
        assert torch.isfinite(output[isolated_nodes]).all()


@pytest.mark.parametrize("isolated_position", [2, 4], ids=("interior", "trailing"))
def test_edgnn_meandeg_isolation_does_not_change_nonisolated_output(
    isolated_position: int,
) -> None:
    """Adding an isolated node preserves every existing node embedding."""
    base_x = torch.arange(16, dtype=torch.float).reshape(4, 4)
    base_incidence = torch.tensor(
        [[0, 1, 2, 3], [0, 0, 1, 1]],
        dtype=torch.long,
    )
    isolated_x = torch.full((1, 4), 101.0)
    extended_x = torch.cat(
        [base_x[:isolated_position], isolated_x, base_x[isolated_position:]]
    )
    extended_incidence = base_incidence.clone()
    extended_incidence[0, extended_incidence[0] >= isolated_position] += 1
    model = _model(edconv_type="MeanDeg")
    model.eval()

    base_output = model(base_x, base_incidence)
    extended_output = model(extended_x, extended_incidence)
    nonisolated = [
        node_id
        for node_id in range(extended_x.size(0))
        if node_id != isolated_position
    ]

    assert extended_output.shape == extended_x.shape
    assert torch.isfinite(extended_output).all()
    torch.testing.assert_close(extended_output[nonisolated], base_output)


def test_edgnn_rejects_empty_incidence_contextually() -> None:
    """EDGNN rejects an incidence-free hypergraph at its public boundary."""
    with pytest.raises(
        ValueError,
        match="EDGNN hyperedge_index must contain at least one incidence",
    ):
        _model(edconv_type="MeanDeg")(
            torch.randn(3, 4),
            torch.empty((2, 0), dtype=torch.long),
        )


@pytest.mark.parametrize(
    "hyperedge_index",
    [
        torch.tensor([0, 1, 2], dtype=torch.long),
        torch.tensor([[0.0, 1.0], [0.0, 0.0]]),
    ],
)
def test_edgnn_rejects_malformed_dense_incidence(
    hyperedge_index: torch.Tensor,
) -> None:
    """Only structurally valid dense long incidence reaches EquivSet."""
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
