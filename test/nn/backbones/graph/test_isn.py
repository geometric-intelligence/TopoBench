"""Unit tests for the Identity Sheaf Network (ISN) backbone and metrics."""

import pytest
import torch

from topobench.nn.backbones.graph.identity_sheaf import ISNEncoder
from topobench.nn.backbones.graph.nsd_utils.inductive_discrete_models import (
    InductiveDiscreteIdentitySheafDiffusion,
)
from topobench.nn.backbones.graph.nsd_utils.sheaf_metrics import (
    class_gain,
    heterophily_category,
    neighbor_class_matrix,
    rayleigh_quotient,
)
from topobench.nn.backbones.graph.nsd_utils.sheaf_models import (
    IdentitySheafLearner,
)


class TestIdentitySheafLearner:
    """Test the parameter-free IdentitySheafLearner."""

    def test_diagonal_maps_are_ones(self):
        """Diagonal identity maps must be all ones of shape [E, d]."""
        learner = IdentitySheafLearner(in_channels=8, out_shape=(3,))
        x = torch.randn(5, 8)
        edge_index = torch.tensor([[0, 1, 2], [1, 2, 3]])
        maps = learner(x, edge_index)
        assert maps.shape == (3, 3)
        assert torch.all(maps == 1)

    def test_general_maps_are_identity(self):
        """General identity maps must be d x d identity per edge."""
        learner = IdentitySheafLearner(in_channels=8, out_shape=(2, 2))
        x = torch.randn(5, 8)
        edge_index = torch.tensor([[0, 1, 2], [1, 2, 3]])
        maps = learner(x, edge_index)
        assert maps.shape == (3, 2, 2)
        for e in range(3):
            assert torch.allclose(maps[e], torch.eye(2))

    def test_no_learnable_parameters(self):
        """The identity learner must have zero learnable parameters."""
        learner = IdentitySheafLearner(in_channels=8, out_shape=(3,))
        assert sum(p.numel() for p in learner.parameters()) == 0

    def test_invalid_out_shape(self):
        """An out_shape with more than 2 dims must raise."""
        with pytest.raises(AssertionError):
            IdentitySheafLearner(in_channels=8, out_shape=(2, 2, 2))

    def test_maps_match_input_dtype(self):
        """Returned maps must follow the input dtype."""
        learner = IdentitySheafLearner(in_channels=4, out_shape=(2,))
        x = torch.randn(3, 4, dtype=torch.float64)
        edge_index = torch.tensor([[0, 1], [1, 2]])
        maps = learner(x, edge_index)
        assert maps.dtype == torch.float64


class TestISNEncoder:
    """Test the ISNEncoder backbone."""

    def setup_method(self):
        """Set up test fixtures before each test method."""
        self.input_dim = 16
        self.hidden_dim = 32

    def _prepare_features(self, num_nodes, feat_dim=None):
        """Create random features.

        Parameters
        ----------
        num_nodes : int
            Number of nodes.
        feat_dim : int, optional
            Feature dimension. Defaults to self.input_dim.

        Returns
        -------
        torch.Tensor
            Random feature tensor of shape [num_nodes, feat_dim].
        """
        if feat_dim is None:
            feat_dim = self.input_dim
        return torch.randn(num_nodes, feat_dim)

    def test_initialization_default(self):
        """Test default initialization."""
        model = ISNEncoder(input_dim=self.input_dim, hidden_dim=self.hidden_dim)
        assert model.input_dim == self.input_dim
        assert model.hidden_dim == self.hidden_dim
        assert model.num_layers == 2
        assert model.d == 2
        assert model.sheaf_model is not None

    def test_initialization_custom(self):
        """Test custom initialization."""
        model = ISNEncoder(
            input_dim=self.input_dim,
            hidden_dim=self.hidden_dim,
            num_layers=3,
            d=4,
            dropout=0.2,
            input_dropout=0.1,
        )
        assert model.num_layers == 3
        assert model.d == 4

    def test_invalid_d(self):
        """A stalk dimension below 1 must raise."""
        with pytest.raises(AssertionError):
            ISNEncoder(
                input_dim=self.input_dim, hidden_dim=self.hidden_dim, d=0
            )

    def test_uses_identity_diffusion(self):
        """The backbone must use the identity diffusion model."""
        model = ISNEncoder(input_dim=self.input_dim, hidden_dim=self.hidden_dim)
        assert isinstance(
            model.sheaf_model, InductiveDiscreteIdentitySheafDiffusion
        )

    def test_no_learnable_sheaf_parameters(self):
        """ISN must not learn any restriction-map parameters."""
        model = ISNEncoder(input_dim=self.input_dim, hidden_dim=self.hidden_dim)
        learner_params = sum(
            p.numel()
            for learner in model.sheaf_model.sheaf_learners
            for p in learner.parameters()
        )
        assert learner_params == 0

    def test_forward_basic(self, simple_graph_0):
        """Test basic forward pass.

        Parameters
        ----------
        simple_graph_0 : torch_geometric.data.Data
            Test graph fixture.
        """
        x = self._prepare_features(simple_graph_0.num_nodes)
        model = ISNEncoder(
            input_dim=self.input_dim,
            hidden_dim=self.hidden_dim,
            num_layers=2,
        )
        out = model(
            x=x,
            edge_index=simple_graph_0.edge_index,
            batch=torch.zeros(simple_graph_0.num_nodes, dtype=torch.long),
        )
        assert out.shape == (simple_graph_0.num_nodes, self.hidden_dim)
        assert not torch.isnan(out).any()
        assert not torch.isinf(out).any()

    def test_forward_no_batch(self, simple_graph_0):
        """Test forward pass without a batch vector.

        Parameters
        ----------
        simple_graph_0 : torch_geometric.data.Data
            Test graph fixture.
        """
        x = self._prepare_features(simple_graph_0.num_nodes)
        model = ISNEncoder(
            input_dim=self.input_dim, hidden_dim=self.hidden_dim
        )
        out = model(x=x, edge_index=simple_graph_0.edge_index)
        assert out.shape == (simple_graph_0.num_nodes, self.hidden_dim)

    def test_forward_ignores_edge_attr(self, simple_graph_0):
        """Edge attributes/weights must be ignored without error.

        Parameters
        ----------
        simple_graph_0 : torch_geometric.data.Data
            Test graph fixture.
        """
        x = self._prepare_features(simple_graph_0.num_nodes)
        model = ISNEncoder(
            input_dim=self.input_dim, hidden_dim=self.hidden_dim
        )
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
        x = self._prepare_features(simple_graph_0.num_nodes)
        model = ISNEncoder(
            input_dim=self.input_dim, hidden_dim=self.hidden_dim, d=1
        )
        out = model(x=x, edge_index=simple_graph_0.edge_index)
        assert out.shape == (simple_graph_0.num_nodes, self.hidden_dim)

    def test_get_sheaf_model(self):
        """get_sheaf_model must return the underlying diffusion model."""
        model = ISNEncoder(
            input_dim=self.input_dim, hidden_dim=self.hidden_dim
        )
        assert model.get_sheaf_model() is model.sheaf_model


class TestClassGain:
    """Test the class-gain heterophily metric."""

    def _graph(self):
        """Build the canonical 8-node test graph and labels.

        Returns
        -------
        tuple
            (edge_index, labels) for the test graph.
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
        edge_index = torch.tensor(edges).t().long()
        y = torch.tensor([0, 1, 1, 1, 0, 0, 0, 0])
        return edge_index, y

    def test_neighbor_class_matrix_shape(self):
        """Neighbor-class counts and sizes must have the right shape."""
        edge_index, y = self._graph()
        counts, sizes = neighbor_class_matrix(edge_index, y, num_classes=2)
        assert counts.shape == (2, 2)
        assert sizes.shape == (2,)
        assert sizes.sum().item() == 8

    def test_neighbor_class_matrix_infers_classes(self):
        """num_classes is inferred from labels when omitted."""
        edge_index, y = self._graph()
        counts, _ = neighbor_class_matrix(edge_index, y)
        assert counts.shape == (2, 2)

    def test_class_gain_properties(self):
        """Gain matrix is square, symmetric, with zero diagonal."""
        edge_index, y = self._graph()
        gain = class_gain(edge_index, y, num_classes=2)
        assert gain.shape == (2, 2)
        assert torch.allclose(gain, gain.t())
        assert torch.allclose(gain.diag(), torch.zeros(2))

    def test_class_gain_nonnegative(self):
        """All gains must be non-negative (L1 norm)."""
        edge_index, y = self._graph()
        gain = class_gain(edge_index, y, num_classes=2)
        assert torch.all(gain >= 0)

    def test_heterophily_good(self):
        """A large off-diagonal gain yields the 'good' category."""
        gain = torch.tensor([[0.0, 0.8], [0.8, 0.0]])
        assert heterophily_category(gain, threshold=0.2) == "good"

    def test_heterophily_bad(self):
        """A small off-diagonal gain yields the 'bad' category."""
        gain = torch.tensor([[0.0, 0.1], [0.1, 0.0]])
        assert heterophily_category(gain, threshold=0.2) == "bad"

    def test_heterophily_mixed(self):
        """A spread of gains across the threshold yields 'mixed'."""
        gain = torch.tensor(
            [
                [0.0, 0.1, 0.9],
                [0.1, 0.0, 0.5],
                [0.9, 0.5, 0.0],
            ]
        )
        assert heterophily_category(gain, threshold=0.2) == "mixed"

    def test_heterophily_single_class(self):
        """A single-class gain matrix is reported as 'mixed'."""
        gain = torch.zeros(1, 1)
        assert heterophily_category(gain) == "mixed"


class TestRayleighQuotient:
    """Test the Rayleigh quotient / normalized Dirichlet energy."""

    def _path_laplacian(self):
        """Build the Laplacian of a 3-node path graph.

        Returns
        -------
        torch.Tensor
            Dense [3, 3] graph Laplacian.
        """
        adjacency = torch.tensor(
            [[0.0, 1, 0], [1, 0, 1], [0, 1, 0]]
        )
        return torch.diag(adjacency.sum(1)) - adjacency

    def test_matches_closed_form_vector(self):
        """Single-column input matches x^T L x / x^T x exactly."""
        lap = self._path_laplacian()
        x = torch.randn(3, 1)
        expected = (x.t() @ lap @ x / (x.t() @ x)).item()
        assert abs(rayleigh_quotient(x, lap).item() - expected) < 1e-5

    def test_constant_signal_is_zero(self):
        """A constant signal has zero Dirichlet energy."""
        lap = self._path_laplacian()
        x = torch.ones(3, 4)
        assert abs(rayleigh_quotient(x, lap).item()) < 1e-6

    def test_psd_nonnegative(self):
        """The quotient of a PSD operator is non-negative."""
        lap = self._path_laplacian()
        x = torch.randn(3, 8)
        assert rayleigh_quotient(x, lap).item() >= 0

    def test_one_dimensional_input(self):
        """A 1-D input vector is handled (reshaped internally)."""
        lap = self._path_laplacian()
        x = torch.randn(3)
        out = rayleigh_quotient(x, lap)
        assert out.dim() == 0

    def test_sparse_tuple_matches_dense(self):
        """Sparse (indices, values) input matches the dense result."""
        lap = self._path_laplacian()
        indices = lap.nonzero().t()
        values = lap[indices[0], indices[1]]
        x = torch.randn(3, 5)
        dense = rayleigh_quotient(x, lap).item()
        sparse = rayleigh_quotient(
            x, (indices, values), num_nodes=3
        ).item()
        assert abs(dense - sparse) < 1e-5

    def test_sparse_tuple_infers_num_nodes(self):
        """Sparse input infers num_nodes from x when it is omitted."""
        lap = self._path_laplacian()
        indices = lap.nonzero().t()
        values = lap[indices[0], indices[1]]
        x = torch.randn(3, 5)
        dense = rayleigh_quotient(x, lap).item()
        sparse = rayleigh_quotient(x, (indices, values)).item()
        assert abs(dense - sparse) < 1e-5
