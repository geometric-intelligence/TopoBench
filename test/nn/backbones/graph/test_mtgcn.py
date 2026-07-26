"""Tests for the Multi-Track Graph Convolutional Network backbone."""

import torch
from torch.nn import functional as F
from torch_geometric.data import Batch, Data

from topobench.nn.backbones.graph.mtgcn import MTGCNEncoder, _prepare_graph


def _graph() -> Data:
    """Create a small bidirectional graph with non-uniform features."""
    x = torch.tensor(
        [
            [1.0, 0.0, -1.0],
            [0.0, 2.0, 1.0],
            [-1.0, 1.0, 0.5],
            [2.0, -1.0, 0.0],
        ]
    )
    edge_index = torch.tensor(
        [[0, 1, 1, 2, 2, 3], [1, 0, 2, 1, 3, 2]], dtype=torch.long
    )
    return Data(x=x, edge_index=edge_index)


def _model(**kwargs) -> MTGCNEncoder:
    """Create a deterministic compact MTGCN encoder."""
    torch.manual_seed(7)
    return MTGCNEncoder(
        in_channels=3,
        hidden_channels=5,
        num_layers=2,
        num_tracks=3,
        num_heads=2,
        dropout=0.0,
        output_norm="none",
        **kwargs,
    )


def _prepared_for(model: MTGCNEncoder, data: Data):
    """Prepare the graph exactly as the model forward does."""
    return _prepare_graph(
        data.edge_index,
        None,
        data.num_nodes,
        data.x.dtype,
        data.x.device,
        model.normalization,
        model.add_self_loops,
        model.make_undirected,
    )


def test_forward_shape():
    """Return node embeddings with the configured hidden size."""
    data = _graph()
    output = _model()(data.x, data.edge_index)
    assert output.shape == (data.num_nodes, 5)
    assert torch.isfinite(output).all()


def test_wrapper_kwargs():
    """Accept wrapper arguments, batches, and scalar edge attributes."""
    data = _graph()
    edge_attr = torch.linspace(0.5, 1.5, data.num_edges)
    output = _model()(
        data.x,
        data.edge_index,
        edge_attr=edge_attr,
        batch=torch.zeros(data.num_nodes, dtype=torch.long),
        model_state="train",
    )
    assert output.shape == (data.num_nodes, 5)


def test_edge_weight_support():
    """Use supplied edge weights during every sparse propagation."""
    data = _graph()
    model = _model().eval()
    unweighted = model(data.x, data.edge_index)
    edge_weight = torch.tensor([0.1, 0.1, 1.0, 1.0, 3.0, 3.0])
    weighted = model(data.x, data.edge_index, edge_weight=edge_weight)
    assert torch.isfinite(weighted).all()
    assert not torch.allclose(unweighted, weighted)


def test_batched_graphs():
    """Keep propagation local to each graph in a disjoint PyG batch."""
    first = _graph()
    second = _graph()
    second.x = second.x.flip(0)
    model = _model().eval()

    expected = torch.cat(
        (model(first.x, first.edge_index), model(second.x, second.edge_index))
    )
    batched = Batch.from_data_list([first, second])
    actual = model(batched.x, batched.edge_index, batch=batched.batch)
    assert torch.allclose(actual, expected, atol=1e-6)


def test_affiliations_sum_to_one():
    """Produce finite normalized head-specific affiliations."""
    data = _graph()
    model = _model()
    graph = _prepared_for(model, data)
    affiliations = model._compute_affiliations(
        model._affiliation_embeddings(data.x, graph)
    )
    diagnostics = model.affiliation_diagnostics(data.x, data.edge_index)

    assert affiliations.shape == (data.num_nodes, 2, 3)
    assert torch.allclose(affiliations.sum(-1), torch.ones(data.num_nodes, 2))
    assert all(torch.isfinite(value) for value in diagnostics.values())


def test_affiliation_sources():
    """Support raw-feature, auxiliary-GCN, and hybrid affiliation inputs."""
    data = _graph()
    outputs = []
    for source in ("features", "auxiliary", "hybrid"):
        model = _model(affiliation_source=source).eval()
        outputs.append(model(data.x, data.edge_index))
    assert all(output.shape == (data.num_nodes, 5) for output in outputs)
    assert not torch.allclose(outputs[0], outputs[1])


def test_head_specific_acquiring():
    """Acquire tracks with each head's own affiliation weights before fusion."""
    torch.manual_seed(3)
    model = _model().eval()
    tracks = torch.randn(4, 3, 2, 5)
    affiliations = F.softmax(torch.randn(4, 2, 3), dim=-1)

    acquired = model._acquire_messages(tracks, affiliations)
    mean_affiliation = affiliations.mean(dim=1)
    averaged = (mean_affiliation[..., None, None] * tracks).sum(dim=1)
    averaged = model.head_fusion(averaged.transpose(1, 2)).squeeze(-1)
    assert acquired.shape == (4, 5)
    assert not torch.allclose(acquired, averaged)


def test_entropy_sharpening_modes():
    """Keep direct affiliations by default and normalized sharpened weights on request."""
    affiliations = torch.tensor([[[0.25, 0.25, 0.50], [0.20, 0.30, 0.50]]])
    direct = _model(use_entropy_sharpening=False)._track_weights(affiliations)
    sharpened = _model(
        use_entropy_sharpening=True, sharpening_power=2.0
    )._track_weights(affiliations)

    assert torch.allclose(direct, affiliations)
    assert torch.allclose(sharpened.sum(-1), torch.ones(1, 2))
    assert sharpened[..., -1].gt(affiliations[..., -1]).all()


def test_output_residual():
    """Add the projected input once after acquisition when enabled."""
    data = _graph()
    model = MTGCNEncoder(
        in_channels=3,
        hidden_channels=3,
        num_layers=1,
        num_tracks=2,
        num_heads=1,
        dropout=0.0,
        use_output_residual=True,
        output_norm="none",
    )
    for name, parameter in model.named_parameters():
        if not name.startswith("output_residual"):
            parameter.data.zero_()
    assert torch.allclose(model(data.x, data.edge_index), data.x)


def test_make_undirected():
    """Match graph preparation for directed and already-bidirectional inputs."""
    x = torch.randn(3, 4)
    one_way = torch.tensor([[0, 1], [1, 2]], dtype=torch.long)
    two_way = torch.tensor([[0, 1, 1, 2], [1, 0, 2, 1]], dtype=torch.long)
    left = _prepare_graph(
        one_way, None, 3, x.dtype, x.device, "symmetric", True, True
    )
    right = _prepare_graph(
        two_way, None, 3, x.dtype, x.device, "symmetric", True, True
    )
    assert torch.allclose(left.matmul(x), right.matmul(x), atol=1e-6)


def test_multiple_tracks_active():
    """Assign nonzero probability and gradients to multiple tracks."""
    data = _graph()
    model = _model()
    graph = _prepared_for(model, data)
    affiliations = model._compute_affiliations(
        model._affiliation_embeddings(data.x, graph)
    )

    assert (affiliations.sum(dim=(0, 1)) > 0).sum() == model.num_tracks
    model(data.x, data.edge_index).square().sum().backward()
    prototype_grad = model.track_prototypes.grad.abs().sum(dim=1)
    assert (prototype_grad > 0).sum() > 1


def test_finite_backward_gradients():
    """Backpropagate finite gradients through loading and propagation."""
    data = _graph()
    data.x.requires_grad_(True)
    model = _model()
    model(data.x, data.edge_index).mean().backward()

    assert data.x.grad is not None
    assert torch.isfinite(data.x.grad).all()
    gradients = [
        parameter.grad
        for parameter in model.parameters()
        if parameter.requires_grad
    ]
    assert all(gradient is not None for gradient in gradients)
    assert all(torch.isfinite(gradient).all() for gradient in gradients)


def test_not_equivalent_to_single_mean_aggregation():
    """Produce behavior beyond one ordinary mean-aggregation layer."""
    data = _graph()
    model = _model().eval()
    output = model(data.x, data.edge_index)

    mean_graph = _prepare_graph(
        data.edge_index,
        None,
        data.num_nodes,
        data.x.dtype,
        data.x.device,
        "row",
        True,
        False,
    )
    ordinary_mean = mean_graph.matmul(data.x)
    projected_mean = F.pad(ordinary_mean, (0, 2))
    assert not torch.allclose(output, projected_mean)
