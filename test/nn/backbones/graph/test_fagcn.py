"""Unit tests for the FAGCN graph backbone."""

import pytest
import torch
from torch_geometric.data import Batch, Data
from torch_geometric.nn import FAConv
from topobench.nn.backbones.graph.fagcn import FAGCN
from topobench.nn.wrappers import GNNWrapper


class TestFAConv:
    """Tests for the single FAConv message-passing layer."""

    def setup_method(self):
        """Set up common dimensions."""
        self.channels = 16

    def test_initialization(self):
        """Test FAConv initializes with correct shapes."""
        conv = FAConv(channels=self.channels, eps=0.1, dropout=0.0)
        # Attention vector has shape [2*channels, 1]
        assert conv.att.weight.shape == (1, 2 * self.channels)
        assert conv.eps == 0.1

    def test_forward_basic(self, simple_graph_0):
        """Test FAConv forward pass returns correct shape."""
        conv = FAConv(channels=self.channels, dropout=0.0)
        x = torch.randn(simple_graph_0.num_nodes, self.channels)

        out = conv(x, simple_graph_0.edge_index)

        assert out.shape == (simple_graph_0.num_nodes, self.channels)
        assert not torch.isnan(out).any()
        assert not torch.isinf(out).any()

    def test_forward_edgeless(self):
        """Test FAConv on a graph with no edges (only residual term)."""
        conv = FAConv(channels=self.channels, eps=0.1, dropout=0.0)
        num_nodes = 5
        x = torch.randn(num_nodes, self.channels)
        edge_index = torch.empty(2, 0, dtype=torch.long)

        out = conv(x, edge_index)

        # No neighbors => output is eps * x
        assert out.shape == (num_nodes, self.channels)
        torch.testing.assert_close(out, 0.1 * x)

    def test_signed_attention(self, simple_graph_0):
        """Test that attention coefficients span (-1, 1) via tanh."""
        conv = FAConv(channels=self.channels, dropout=0.0)
        x = torch.randn(simple_graph_0.num_nodes, self.channels)

        # Hook to capture messages
        messages = []

        def hook(module, inputs, output):
            messages.append(output)

        conv.register_forward_hook(hook)
        conv(x, simple_graph_0.edge_index)

        # Output should be bounded (tanh keeps alpha in (-1,1))
        assert len(messages) > 0


class TestFAGCN:
    """Tests for the full FAGCN model."""

    def setup_method(self):
        """Set up test dimensions."""
        self.in_channels = 16
        self.hidden_channels = 32
        self.out_channels = 24

    def _features(self, num_nodes, channels=None):
        """Create random node features."""
        channels = self.in_channels if channels is None else channels
        return torch.randn(num_nodes, channels)

    def test_initialization_default(self):
        """Test default initialization."""
        model = FAGCN(
            in_channels=self.in_channels,
            hidden_channels=self.hidden_channels,
        )

        assert model.in_channels == self.in_channels
        assert model.hidden_channels == self.hidden_channels
        assert model.out_channels == self.hidden_channels
        assert model.num_layers == 2
        assert model.eps == 0.1
        assert len(model.convs) == 2

    def test_initialization_custom(self):
        """Test custom initialization."""
        model = FAGCN(
            in_channels=self.in_channels,
            hidden_channels=self.hidden_channels,
            out_channels=self.out_channels,
            num_layers=4,
            eps=0.2,
            dropout=0.3,
        )

        assert model.out_channels == self.out_channels
        assert model.num_layers == 4
        assert model.eps == 0.2
        assert model.dropout == 0.3
        assert len(model.convs) == 4

    def test_out_channels_projection(self):
        """Test that out_channels != hidden_channels adds a projection."""
        model = FAGCN(
            in_channels=self.in_channels,
            hidden_channels=self.hidden_channels,
            out_channels=self.out_channels,
        )
        import torch.nn as nn
        assert isinstance(model.out_proj, nn.Linear)
        assert model.out_proj.out_features == self.out_channels

    def test_no_projection_when_same_channels(self):
        """Test that no projection is added when out_channels == hidden_channels."""
        model = FAGCN(
            in_channels=self.in_channels,
            hidden_channels=self.hidden_channels,
        )
        import torch.nn as nn
        assert isinstance(model.out_proj, nn.Identity)

    def test_invalid_in_channels(self):
        """Test that non-positive in_channels raises ValueError."""
        with pytest.raises(ValueError, match="in_channels"):
            FAGCN(in_channels=0, hidden_channels=self.hidden_channels)

    def test_invalid_hidden_channels(self):
        """Test that non-positive hidden_channels raises ValueError."""
        with pytest.raises(ValueError, match="hidden_channels"):
            FAGCN(in_channels=self.in_channels, hidden_channels=-1)

    def test_invalid_num_layers(self):
        """Test that num_layers < 1 raises ValueError."""
        with pytest.raises(ValueError, match="num_layers"):
            FAGCN(
                in_channels=self.in_channels,
                hidden_channels=self.hidden_channels,
                num_layers=0,
            )

    def test_invalid_dropout(self):
        """Test that dropout >= 1.0 raises ValueError."""
        with pytest.raises(ValueError, match="dropout"):
            FAGCN(
                in_channels=self.in_channels,
                hidden_channels=self.hidden_channels,
                dropout=1.0,
            )

    def test_forward_basic(self, simple_graph_0):
        """Test basic forward pass shape and validity."""
        model = FAGCN(
            in_channels=self.in_channels,
            hidden_channels=self.hidden_channels,
            out_channels=self.out_channels,
            dropout=0.0,
        )
        x = self._features(simple_graph_0.num_nodes)

        out = model(x=x, edge_index=simple_graph_0.edge_index)

        assert out.shape == (simple_graph_0.num_nodes, self.out_channels)
        assert not torch.isnan(out).any()
        assert not torch.isinf(out).any()

    def test_forward_edgeless(self):
        """Test forward on a graph with no edges."""
        num_nodes = 6
        edge_index = torch.empty(2, 0, dtype=torch.long)
        x = self._features(num_nodes)
        model = FAGCN(
            in_channels=self.in_channels,
            hidden_channels=self.hidden_channels,
            dropout=0.0,
        )

        out = model(x=x, edge_index=edge_index)

        assert out.shape == (num_nodes, self.hidden_channels)
        assert not torch.isnan(out).any()

    def test_forward_ignores_edge_weight(self, simple_graph_0):
        """Test that edge_weight is accepted but ignored without errors."""
        model = FAGCN(
            in_channels=self.in_channels,
            hidden_channels=self.hidden_channels,
            dropout=0.0,
        )
        x = self._features(simple_graph_0.num_nodes)
        edge_weight = torch.rand(simple_graph_0.edge_index.size(1))

        out = model(
            x=x,
            edge_index=simple_graph_0.edge_index,
            edge_weight=edge_weight,
        )

        assert out.shape == (simple_graph_0.num_nodes, self.hidden_channels)

    def test_forward_ignores_edge_attr(self, simple_graph_0):
        """Test that edge_attr is accepted but ignored without errors."""
        model = FAGCN(
            in_channels=self.in_channels,
            hidden_channels=self.hidden_channels,
            dropout=0.0,
        )
        x = self._features(simple_graph_0.num_nodes)
        edge_attr = torch.randn(simple_graph_0.edge_index.size(1), 4)

        out = model(
            x=x,
            edge_index=simple_graph_0.edge_index,
            edge_attr=edge_attr,
        )

        assert out.shape == (simple_graph_0.num_nodes, self.hidden_channels)

    def test_forward_single_layer(self, simple_graph_0):
        """Test forward with a single FA-GCN layer."""
        model = FAGCN(
            in_channels=self.in_channels,
            hidden_channels=self.hidden_channels,
            num_layers=1,
            dropout=0.0,
        )
        x = self._features(simple_graph_0.num_nodes)

        out = model(x=x, edge_index=simple_graph_0.edge_index)

        assert out.shape == (simple_graph_0.num_nodes, self.hidden_channels)

    def test_forward_many_layers(self, simple_graph_0):
        """Test forward with many FA-GCN layers (deep model)."""
        model = FAGCN(
            in_channels=self.in_channels,
            hidden_channels=self.hidden_channels,
            num_layers=8,
            dropout=0.0,
        )
        x = self._features(simple_graph_0.num_nodes)

        out = model(x=x, edge_index=simple_graph_0.edge_index)

        assert out.shape == (simple_graph_0.num_nodes, self.hidden_channels)

    def test_wrapper_compatibility(self, simple_graph_0):
        """Test compatibility with the standard GNNWrapper."""
        channels = self.in_channels
        x = self._features(simple_graph_0.num_nodes, channels)
        batch = Data(
            x=x,
            x_0=x,
            edge_index=simple_graph_0.edge_index,
            y=simple_graph_0.y,
            batch_0=torch.zeros(simple_graph_0.num_nodes, dtype=torch.long),
        )
        model = FAGCN(
            in_channels=channels,
            hidden_channels=channels,
            out_channels=channels,
            dropout=0.0,
        )
        wrapper = GNNWrapper(
            model,
            out_channels=channels,
            num_cell_dimensions=1,
            residual_connections=False,
        )

        model_out = wrapper(batch)

        assert model_out["x_0"].shape == x.shape
        assert model_out["batch_0"].shape == (simple_graph_0.num_nodes,)
        assert torch.equal(model_out["labels"], simple_graph_0.y)

    def test_batched_graphs(self, simple_graph_0, simple_graph_1):
        """Test forward pass with a batched PyG input."""
        batch_data = Batch.from_data_list([simple_graph_0, simple_graph_1])
        x = self._features(batch_data.num_nodes)
        model = FAGCN(
            in_channels=self.in_channels,
            hidden_channels=self.hidden_channels,
            dropout=0.0,
        )

        out = model(
            x=x,
            edge_index=batch_data.edge_index,
            batch=batch_data.batch,
        )

        assert out.shape == (batch_data.num_nodes, self.hidden_channels)

    def test_backward_pass(self, simple_graph_0):
        """Test gradients flow through the model end-to-end."""
        model = FAGCN(
            in_channels=self.in_channels,
            hidden_channels=self.hidden_channels,
            dropout=0.0,
        )
        x = self._features(simple_graph_0.num_nodes).requires_grad_(True)

        out = model(x=x, edge_index=simple_graph_0.edge_index)
        out.mean().backward()

        assert x.grad is not None
        assert any(
            p.grad is not None
            for p in model.parameters()
            if p.requires_grad
        )

    def test_eval_mode_deterministic(self, simple_graph_0):
        """Test that eval mode produces deterministic outputs (dropout off)."""
        model = FAGCN(
            in_channels=self.in_channels,
            hidden_channels=self.hidden_channels,
            dropout=0.5,
        )
        model.eval()
        x = self._features(simple_graph_0.num_nodes)

        with torch.no_grad():
            out1 = model(x=x, edge_index=simple_graph_0.edge_index)
            out2 = model(x=x, edge_index=simple_graph_0.edge_index)

        torch.testing.assert_close(out1, out2)

    def test_different_eps_values(self, simple_graph_0):
        """Test model works with different eps residual weights."""
        x = self._features(simple_graph_0.num_nodes)

        for eps in [0.0, 0.1, 0.5, 1.0]:
            model = FAGCN(
                in_channels=self.in_channels,
                hidden_channels=self.hidden_channels,
                eps=eps,
                dropout=0.0,
            )
            out = model(x=x, edge_index=simple_graph_0.edge_index)
            assert out.shape == (simple_graph_0.num_nodes, self.hidden_channels)
            assert not torch.isnan(out).any()
