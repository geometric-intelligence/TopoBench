"""Native batch-isolation gates for the GCN-DGM candidate."""

import pytest
import torch

from topobench.nn.backbones.graph import GCNDGM


def test_gcn_dgm_returns_only_tensor_embeddings_and_isolated_edges() -> None:
    torch.manual_seed(5)
    batch = torch.tensor([0, 0, 0, 1, 1, 1, 1], dtype=torch.long)
    model = GCNDGM(
        in_channels=4,
        hidden_channels=8,
        num_layers=2,
        k=3,
    )

    output = model(
        torch.randn(7, 4),
        torch.empty((2, 0), dtype=torch.long),
        batch=batch,
    )

    assert isinstance(output, torch.Tensor)
    assert output.shape == (7, 8)
    assert model.last_auxiliary_edge_index is not None
    source, target = model.last_auxiliary_edge_index
    assert torch.equal(batch[source], batch[target])


def test_gcn_dgm_large_graph_is_invariant_to_singleton_cobatch() -> None:
    torch.manual_seed(7)
    model = GCNDGM(
        in_channels=4,
        hidden_channels=8,
        num_layers=2,
        k=3,
    )
    large_x = torch.randn(5, 4)

    large_output = model(
        large_x,
        torch.empty((2, 0), dtype=torch.long),
    )
    assert model.last_auxiliary_edge_index is not None
    assert model.last_auxiliary_logprobs is not None
    large_edges = model.last_auxiliary_edge_index.clone()
    large_logprobs = model.last_auxiliary_logprobs.clone()

    batched_output = model(
        torch.cat((large_x, torch.randn(1, 4))),
        torch.empty((2, 0), dtype=torch.long),
        batch=torch.tensor([0, 0, 0, 0, 0, 1]),
    )

    assert model.last_auxiliary_edge_index is not None
    assert model.last_auxiliary_logprobs is not None
    assert model.last_auxiliary_logprobs.ndim == 1
    assert model.last_auxiliary_logprobs.numel() == 16
    torch.testing.assert_close(
        model.last_auxiliary_edge_index[:, :15],
        large_edges,
    )
    torch.testing.assert_close(
        model.last_auxiliary_logprobs[:15],
        large_logprobs,
    )
    torch.testing.assert_close(batched_output[:5], large_output)
    assert torch.equal(
        model.last_auxiliary_edge_index[:, -1],
        torch.tensor([5, 5]),
    )


def test_gcn_dgm_structure_parameters_receive_supervised_gradient() -> None:
    torch.manual_seed(9)
    model = GCNDGM(
        in_channels=4,
        hidden_channels=8,
        num_layers=1,
        k=2,
    )
    output = model(
        torch.randn(6, 4),
        torch.empty((2, 0), dtype=torch.long),
        batch=torch.tensor([0, 0, 0, 1, 1, 1]),
    )

    output.square().mean().backward()

    gradient = model.structure_encoder.linear.weight.grad
    assert gradient is not None
    assert torch.count_nonzero(gradient)


@pytest.mark.parametrize("k", (2, 3))
def test_gcn_dgm_supported_k_trains_structure_and_temperature(k: int) -> None:
    torch.manual_seed(13)
    model = GCNDGM(
        in_channels=2,
        hidden_channels=4,
        num_layers=1,
        k=k,
    )
    x = torch.tensor(
        [[0.0, 0.0], [0.5, 1.0], [2.0, -0.5], [4.0, 2.0], [9.0, -1.0]]
    )
    structure = model.structure_encoder(x)
    _, weights, _ = model._learned_edges(
        structure,
        torch.zeros(x.size(0), dtype=torch.long),
    )

    coefficients = torch.linspace(
        0.25,
        2.0,
        weights.numel(),
        dtype=weights.dtype,
    )
    (weights * coefficients).sum().backward()

    structure_gradient = model.structure_encoder.linear.weight.grad
    temperature_gradient = model.log_temperature.grad
    assert structure_gradient is not None
    assert torch.count_nonzero(structure_gradient)
    assert temperature_gradient is not None
    assert torch.count_nonzero(temperature_gradient)


def test_gcn_dgm_edges_are_selected_neighbor_to_query_and_incoming_normalized() -> None:
    model = GCNDGM(
        in_channels=1,
        hidden_channels=2,
        num_layers=1,
        k=2,
    )
    structure = torch.tensor([[0.0], [1.0], [4.0], [10.0]])

    edge_index, weights, logprobs = model._learned_edges(
        structure,
        torch.zeros(4, dtype=torch.long),
    )

    expected_neighbors = torch.tensor(
        [[1, 2], [0, 2], [1, 0], [2, 1]],
        dtype=torch.long,
    )
    expected_queries = torch.arange(4).unsqueeze(1).expand(-1, 2)
    torch.testing.assert_close(
        edge_index[0].reshape(4, 2).sort(dim=1).values,
        expected_neighbors.sort(dim=1).values,
    )
    torch.testing.assert_close(
        edge_index[1].reshape(4, 2),
        expected_queries,
    )
    incoming_weights = weights.reshape(4, 2)
    torch.testing.assert_close(
        incoming_weights.sum(dim=1),
        torch.ones(4),
    )
    torch.testing.assert_close(logprobs.exp(), weights)


@pytest.mark.parametrize("k", (True, False, 2.5, "2"))
def test_gcn_dgm_k_requires_an_integer_not_bool(k: object) -> None:
    with pytest.raises(TypeError, match="k must be an integer"):
        GCNDGM(
            in_channels=4,
            hidden_channels=8,
            num_layers=1,
            k=k,  # type: ignore[arg-type]
        )


@pytest.mark.parametrize("k", (-1, 0, 1))
def test_gcn_dgm_k_requires_at_least_two(k: int) -> None:
    with pytest.raises(ValueError, match="k must be at least 2"):
        GCNDGM(
            in_channels=4,
            hidden_channels=8,
            num_layers=1,
            k=k,
        )


def test_gcn_dgm_requires_valid_batch_membership() -> None:
    model = GCNDGM(
        in_channels=4,
        hidden_channels=8,
        num_layers=1,
        k=2,
    )

    with pytest.raises(ValueError, match="batch must be contiguous"):
        model(
            torch.randn(4, 4),
            torch.empty((2, 0), dtype=torch.long),
            batch=torch.tensor([0, 0, 2, 2]),
        )
