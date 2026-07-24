"""Unit tests for AdvDIFFormer."""

from copy import deepcopy

import pytest
import torch
from torch_geometric.data import Batch

from topobench.nn.backbones.graph.advdifformer import (
    AdvDIFFormerEncoder,
    AdvDIFFormerLayer,
    _normalized_adjacency_matmul,
)


def _reference_series(layer, x, edge_index, batch, edge_weight=None):
    """Direct implementation of the original series equations."""
    _, canonical = torch.unique(batch, sorted=True, return_inverse=True)
    num_graphs = canonical.max().item() + 1
    outputs = []
    for query, key, output in zip(
        layer.query, layer.key, layer.output, strict=True
    ):
        q = torch.nn.functional.normalize(query(x), dim=-1, eps=1e-12)
        k = torch.nn.functional.normalize(key(x), dim=-1, eps=1e-12)
        states = [x]
        current = x
        for _ in range(layer.propagation_steps):
            attentive = torch.zeros_like(current)
            for graph_id in range(num_graphs):
                idx = torch.nonzero(canonical == graph_id, as_tuple=True)[0]
                q_graph = q[idx]
                k_graph = k[idx]
                values = current[idx]
                similarity = 1 + q_graph @ k_graph.T
                attentive[idx] = (
                    similarity @ values
                    / similarity.sum(dim=-1, keepdim=True).clamp_min(1e-12)
                )
            current = attentive + layer.beta * _normalized_adjacency_matmul(
                current,
                edge_index,
                edge_weight=edge_weight,
                make_undirected=layer.make_undirected,
            )
            states.append(current)
        outputs.append(output(torch.cat(states, dim=-1)))
    result = torch.stack(outputs).sum(dim=0)
    return result / layer.heads if layer.head_aggregation == "mean" else result


class TestAdvDIFFormerEncoder:
    """Test AdvDIFFormerEncoder."""

    def setup_method(self):
        """Set up test fixtures."""
        self.input_dim = 8
        self.hidden_dim = 16

    def _features(self, num_nodes, feat_dim=None):
        """Create random node features."""
        return torch.randn(num_nodes, feat_dim or self.input_dim)

    def test_initialization_default(self):
        """Test default initialization."""
        model = AdvDIFFormerEncoder(
            input_dim=self.input_dim,
            hidden_dim=self.hidden_dim,
        )

        assert model.input_dim == self.input_dim
        assert model.hidden_dim == self.hidden_dim
        assert model.out_channels == self.hidden_dim
        assert model.num_layers == 2
        assert model.variant == "series"
        assert len(model.layers) == 2

    def test_invalid_variant(self):
        """Test that invalid variants fail clearly."""
        with pytest.raises(ValueError, match="variant"):
            AdvDIFFormerEncoder(
                input_dim=self.input_dim,
                hidden_dim=self.hidden_dim,
                variant="bad",
            )

    def test_variant_aliases(self):
        """Test short variant aliases."""
        model = AdvDIFFormerEncoder(
            input_dim=self.input_dim,
            hidden_dim=self.hidden_dim,
            variant="i",
        )

        assert model.variant == "inverse"

    def test_row_normalized_advection(self):
        """Test D^{-1} A aggregation used by the official implementation."""
        edge_index = torch.tensor([[0, 1, 2], [1, 2, 2]])
        x = torch.tensor([[1.0], [2.0], [4.0]])

        out = _normalized_adjacency_matmul(
            x,
            edge_index,
            make_undirected=False,
        )

        expected = torch.tensor([[0.0], [1.0], [3.0]])
        assert torch.allclose(out, expected)

    def test_forward_scalable(self, simple_graph_0):
        """Test AdvDIFFormer-S forward pass."""
        model = AdvDIFFormerEncoder(
            input_dim=self.input_dim,
            hidden_dim=self.hidden_dim,
            num_layers=2,
            heads=2,
            variant="s",
            propagation_steps=2,
        )

        x = self._features(simple_graph_0.num_nodes)
        out = model(x=x, edge_index=simple_graph_0.edge_index)

        assert out.shape == (simple_graph_0.num_nodes, self.hidden_dim)
        assert not torch.isnan(out).any()
        assert not torch.isinf(out).any()

    def test_forward_inverse(self, simple_graph_0):
        """Test AdvDIFFormer-I forward pass."""
        model = AdvDIFFormerEncoder(
            input_dim=self.input_dim,
            hidden_dim=self.hidden_dim,
            num_layers=1,
            heads=1,
            variant="i",
            theta=1.0,
        )

        x = self._features(simple_graph_0.num_nodes)
        out = model(x=x, edge_index=simple_graph_0.edge_index)

        assert out.shape == (simple_graph_0.num_nodes, self.hidden_dim)
        assert not torch.isnan(out).any()
        assert not torch.isinf(out).any()

    def test_forward_batched_graphs(self, simple_graph_0, simple_graph_1):
        """Test that batched graphs preserve graph boundaries."""
        batch_data = Batch.from_data_list([simple_graph_0, simple_graph_1])
        model = AdvDIFFormerEncoder(
            input_dim=self.input_dim,
            hidden_dim=self.hidden_dim,
            num_layers=1,
            heads=2,
            variant="s",
        )

        x = self._features(batch_data.num_nodes)
        out = model(
            x=x,
            edge_index=batch_data.edge_index,
            batch=batch_data.batch,
        )

        assert out.shape == (batch_data.num_nodes, self.hidden_dim)

    def test_forward_with_edge_weight(self, simple_graph_0):
        """Test optional edge weights."""
        model = AdvDIFFormerEncoder(
            input_dim=self.input_dim,
            hidden_dim=self.hidden_dim,
            num_layers=1,
        )
        edge_weight = torch.ones(simple_graph_0.edge_index.shape[1])

        out = model(
            x=self._features(simple_graph_0.num_nodes),
            edge_index=simple_graph_0.edge_index,
            edge_weight=edge_weight,
        )

        assert out.shape == (simple_graph_0.num_nodes, self.hidden_dim)

    def test_backward_pass(self, simple_graph_0):
        """Test gradient flow."""
        model = AdvDIFFormerEncoder(
            input_dim=self.input_dim,
            hidden_dim=self.hidden_dim,
            num_layers=1,
        )
        x = self._features(simple_graph_0.num_nodes).requires_grad_(True)

        out = model(x=x, edge_index=simple_graph_0.edge_index)
        loss = out.mean()
        loss.backward()

        assert x.grad is not None
        assert any(
            param.grad is not None
            for param in model.parameters()
            if param.requires_grad
        )

    def test_series_matches_reference_outputs_and_gradients(self):
        """Optimized series propagation must preserve values and gradients."""
        torch.manual_seed(7)
        layer = AdvDIFFormerLayer(
            hidden_dim=5,
            heads=2,
            propagation_steps=2,
            beta=0.3,
            dropout=0.0,
            head_aggregation="mean",
            make_undirected=True,
        )
        reference_layer = deepcopy(layer)
        edge_index = torch.tensor([[0, 1, 2, 3, 4], [1, 0, 3, 4, 2]])
        edge_weight = torch.tensor([0.5, 1.5, 2.0, 0.75, 1.25])
        batch = torch.tensor([10, 10, 3, 3, 3])
        x = torch.randn(5, 5, requires_grad=True)
        x_reference = x.detach().clone().requires_grad_(True)

        actual = layer(x, edge_index, batch, edge_weight)
        expected = _reference_series(
            reference_layer,
            x_reference,
            edge_index,
            batch,
            edge_weight,
        )
        actual.square().sum().backward()
        expected.square().sum().backward()

        assert torch.allclose(actual, expected, atol=2e-6, rtol=2e-5)
        assert torch.allclose(x.grad, x_reference.grad, atol=2e-6, rtol=2e-5)
        for parameter, reference_parameter in zip(
            layer.parameters(), reference_layer.parameters(), strict=True
        ):
            assert torch.allclose(
                parameter.grad,
                reference_parameter.grad,
                atol=2e-6,
                rtol=2e-5,
            )

    def test_interleaved_non_contiguous_batch_ids(self):
        """Graph-local attention supports unusual batch IDs and node ordering."""
        torch.manual_seed(11)
        layer = AdvDIFFormerLayer(hidden_dim=4, propagation_steps=1)
        x = torch.randn(6, 4)
        batch = torch.tensor([9, 2, 9, 2, 9, 2])
        edge_index = torch.empty((2, 0), dtype=torch.long)

        actual = layer(x, edge_index, batch)
        expected = _reference_series(layer, x, edge_index, batch)

        assert torch.allclose(actual, expected, atol=1e-6, rtol=1e-5)

    @pytest.mark.parametrize("heads", [1, 2, 4])
    def test_different_heads(self, simple_graph_0, heads):
        """Test different numbers of heads."""
        model = AdvDIFFormerEncoder(
            input_dim=self.input_dim,
            hidden_dim=self.hidden_dim,
            num_layers=1,
            heads=heads,
            head_aggregation="mean",
        )

        out = model(
            x=self._features(simple_graph_0.num_nodes),
            edge_index=simple_graph_0.edge_index,
        )

        assert out.shape == (simple_graph_0.num_nodes, self.hidden_dim)
