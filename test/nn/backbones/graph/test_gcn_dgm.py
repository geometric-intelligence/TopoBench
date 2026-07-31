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
