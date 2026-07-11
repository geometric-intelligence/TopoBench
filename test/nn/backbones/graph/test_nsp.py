"""Unit tests for the Neural Sheaf Propagation (NSP) backbone."""

import pytest
import torch

import topobench.nn.backbones as backbones
from topobench.nn.backbones.graph.nsd_utils.inductive_discrete_models import (
    InductiveDiscreteDiagSheafPropagation,
)
from topobench.nn.backbones.graph.nsp import NSPEncoder


def _graph():
    """Build the canonical 8-node test graph edge_index.

    Returns
    -------
    torch.Tensor
        Edge indices of shape [2, num_edges].
    """
    edges = [
        [0, 1],
        [0, 2],
        [0, 4],
        [2, 3],
        [5, 2],
        [5, 6],
        [6, 3],
        [2, 7],
    ]
    return torch.tensor(edges).t().long()


class TestNSPEncoder:
    """Test the NSPEncoder backbone."""

    def setup_method(self):
        """Set up test fixtures before each test method."""
        self.input_dim = 16
        self.hidden_dim = 32

    def _features(self, num_nodes):
        """Create random features of shape [num_nodes, input_dim].

        Parameters
        ----------
        num_nodes : int
            Number of nodes.

        Returns
        -------
        torch.Tensor
            Random feature tensor.
        """
        return torch.randn(num_nodes, self.input_dim)

    def test_initialization_default(self):
        """Test default initialization."""
        model = NSPEncoder(input_dim=self.input_dim, hidden_dim=self.hidden_dim)
        assert model.input_dim == self.input_dim
        assert model.hidden_dim == self.hidden_dim
        assert model.num_layers == 2
        assert model.d == 2
        assert model.step_size == 0.5
        assert model.sheaf_propagation_model is not None

    def test_initialization_custom(self):
        """Test custom initialization."""
        model = NSPEncoder(
            input_dim=self.input_dim,
            hidden_dim=self.hidden_dim,
            num_layers=4,
            d=3,
            step_size=0.2,
        )
        assert model.num_layers == 4
        assert model.d == 3
        assert model.step_size == 0.2

    def test_invalid_d(self):
        """A stalk dimension below 1 must raise."""
        with pytest.raises(AssertionError):
            NSPEncoder(
                input_dim=self.input_dim, hidden_dim=self.hidden_dim, d=0
            )

    def test_uses_propagation_model(self):
        """The backbone must use the diagonal propagation model."""
        model = NSPEncoder(input_dim=self.input_dim, hidden_dim=self.hidden_dim)
        assert isinstance(
            model.sheaf_propagation_model,
            InductiveDiscreteDiagSheafPropagation,
        )

    def test_auto_discovered(self):
        """NSPEncoder is auto-exported under topobench.nn.backbones."""
        assert hasattr(backbones, "NSPEncoder")

    def test_forward_basic(self, simple_graph_0):
        """Test basic forward pass.

        Parameters
        ----------
        simple_graph_0 : torch_geometric.data.Data
            Test graph fixture.
        """
        x = self._features(simple_graph_0.num_nodes)
        model = NSPEncoder(
            input_dim=self.input_dim, hidden_dim=self.hidden_dim, num_layers=2
        )
        out = model(
            x=x,
            edge_index=simple_graph_0.edge_index,
            batch=torch.zeros(simple_graph_0.num_nodes, dtype=torch.long),
        )
        assert out.shape == (simple_graph_0.num_nodes, self.hidden_dim)
        assert torch.isfinite(out).all()

    def test_forward_no_batch(self, simple_graph_0):
        """Test forward pass without a batch vector.

        Parameters
        ----------
        simple_graph_0 : torch_geometric.data.Data
            Test graph fixture.
        """
        x = self._features(simple_graph_0.num_nodes)
        model = NSPEncoder(input_dim=self.input_dim, hidden_dim=self.hidden_dim)
        out = model(x=x, edge_index=simple_graph_0.edge_index)
        assert out.shape == (simple_graph_0.num_nodes, self.hidden_dim)

    def test_forward_ignores_edge_attr(self, simple_graph_0):
        """Edge attributes/weights must be ignored without error.

        Parameters
        ----------
        simple_graph_0 : torch_geometric.data.Data
            Test graph fixture.
        """
        x = self._features(simple_graph_0.num_nodes)
        model = NSPEncoder(input_dim=self.input_dim, hidden_dim=self.hidden_dim)
        n_edges = simple_graph_0.edge_index.shape[1]
        out = model(
            x=x,
            edge_index=simple_graph_0.edge_index,
            edge_attr=torch.randn(n_edges, 4),
            edge_weight=torch.randn(n_edges),
        )
        assert out.shape == (simple_graph_0.num_nodes, self.hidden_dim)

    def test_forward_d1(self, simple_graph_0):
        """Test forward pass with stalk dimension d=1.

        Parameters
        ----------
        simple_graph_0 : torch_geometric.data.Data
            Test graph fixture.
        """
        x = self._features(simple_graph_0.num_nodes)
        model = NSPEncoder(
            input_dim=self.input_dim, hidden_dim=self.hidden_dim, d=1
        )
        out = model(x=x, edge_index=simple_graph_0.edge_index)
        assert out.shape == (simple_graph_0.num_nodes, self.hidden_dim)

    def test_get_sheaf_propagation_model(self):
        """get_sheaf_propagation_model returns the underlying model."""
        model = NSPEncoder(input_dim=self.input_dim, hidden_dim=self.hidden_dim)
        assert (
            model.get_sheaf_propagation_model()
            is model.sheaf_propagation_model
        )


class TestNSPFlags:
    """Test the NSP-specific knobs: step_size, second_linear, dynamic Delta."""

    def _run(self, x, ei, **kw):
        """Run an NSPEncoder forward pass with the given kwargs.

        Parameters
        ----------
        x : torch.Tensor
            Node features.
        ei : torch.Tensor
            Edge indices.
        **kw : dict
            NSPEncoder keyword arguments.

        Returns
        -------
        tuple
            (model, output).
        """
        torch.manual_seed(0)
        model = NSPEncoder(input_dim=16, hidden_dim=32, **kw)
        return model, model(x, ei)

    def test_forward_finite_at_default(self):
        """Config-default step size stays numerically stable (finite)."""
        ei = _graph()
        _, out = self._run(torch.randn(8, 16), ei, num_layers=2, d=2)
        assert torch.isfinite(out).all()

    def test_step_size_controls_magnitude(self):
        """Larger step size drives a larger-magnitude deep wave (CFL control)."""
        ei = _graph()
        x = torch.randn(8, 16)
        _, small = self._run(x, ei, num_layers=16, d=2, step_size=0.1)
        _, large = self._run(x, ei, num_layers=16, d=2, step_size=1.0)
        assert small.abs().max() < large.abs().max()

    def test_deep_network_is_finite(self):
        """A deep (16-layer) wave stays finite at the default step size."""
        ei = _graph()
        _, out = self._run(torch.randn(8, 16), ei, num_layers=16, d=2)
        assert torch.isfinite(out).all()

    def test_second_linear_creates_extra_projection(self):
        """second_linear adds the lin12 layer and extra parameters."""
        off = NSPEncoder(input_dim=16, hidden_dim=32, second_linear=False)
        on = NSPEncoder(input_dim=16, hidden_dim=32, second_linear=True)
        assert not hasattr(off.sheaf_propagation_model, "lin_second")
        assert hasattr(on.sheaf_propagation_model, "lin_second")
        assert sum(p.numel() for p in on.parameters()) > sum(
            p.numel() for p in off.parameters()
        )

    def test_new_laplacian_each_step_learner_count(self):
        """Dynamic geometry keeps one learner per layer; fixed keeps one."""
        dyn = NSPEncoder(
            input_dim=16, hidden_dim=32, num_layers=3,
            new_laplacian_each_step=True,
        )
        fixed = NSPEncoder(
            input_dim=16, hidden_dim=32, num_layers=3,
            new_laplacian_each_step=False,
        )
        assert len(dyn.sheaf_propagation_model.sheaf_learners) == 3
        assert len(fixed.sheaf_propagation_model.sheaf_learners) == 1

    @pytest.mark.parametrize(
        "flags",
        [
            {},
            {"second_linear": True},
            {"new_laplacian_each_step": False},
            {"second_linear": True, "new_laplacian_each_step": False},
        ],
    )
    def test_no_dead_parameters(self, flags):
        """Every learnable parameter receives a gradient (no dead params).

        Parameters
        ----------
        flags : dict
            The flag combination to enable.
        """
        ei = _graph()
        # dropout=0 so the gradient check is deterministic, not flaky.
        model = NSPEncoder(
            input_dim=16, hidden_dim=32, num_layers=3, d=2,
            dropout=0.0, input_dropout=0.0, **flags,
        )
        model(torch.randn(8, 16), ei).sum().backward()
        dead = [
            n
            for n, p in model.named_parameters()
            if p.requires_grad and (p.grad is None or p.grad.abs().sum() == 0)
        ]
        assert dead == []

    def test_no_dead_epsilons(self):
        """NSP must not carry the diffusion epsilons parameter."""
        model = NSPEncoder(input_dim=16, hidden_dim=32)
        names = [n for n, _ in model.named_parameters()]
        assert not any("epsilon" in n for n in names)
