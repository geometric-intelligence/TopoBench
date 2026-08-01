"""Unit tests for GREAD."""

from copy import deepcopy

import pytest
import torch
from torch_geometric.data import Batch

from topobench.nn.backbones.graph.gread import (
    GREADEncoder,
    _prepare_edges,
)


class TestGREADEncoder:
    """Test GREADEncoder."""

    def setup_method(self):
        """Set up test dimensions."""
        self.input_dim = 8
        self.hidden_dim = 8

    def _model(self, **kwargs):
        """Build a small deterministic GREAD model."""
        defaults = {
            "input_dim": self.input_dim,
            "hidden_dim": self.hidden_dim,
            "adjacency": "original",
            "solver": "euler",
            "integration_time": 1.0,
            "step_size": 1.0,
            "make_undirected": False,
            "self_loop_weight": 0.0,
            "input_dropout": 0.0,
            "output_dropout": 0.0,
        }
        defaults.update(kwargs)
        return GREADEncoder(**defaults)

    def test_initialization_defaults(self):
        """Test paper-oriented default settings."""
        model = GREADEncoder(self.input_dim, self.hidden_dim)

        assert model.out_channels == self.hidden_dim
        assert model.reaction_term == "blurring_sharpening"
        assert model.adjacency == "soft"
        assert model.solver == "rk4"
        assert model.self_loop_weight == 0.0
        assert not model.dynamic_adjacency
        assert model.beta.shape == (self.hidden_dim,)

    def test_forward_shape(self, simple_graph_0):
        """Test the public encoder output shape."""
        model = self._model()
        x = torch.randn(simple_graph_0.num_nodes, self.input_dim)

        out = model(x, simple_graph_0.edge_index)

        assert out.shape == (simple_graph_0.num_nodes, self.hidden_dim)

    def test_accepts_wrapper_kwargs(self, simple_graph_0):
        """Test arguments supplied by the TopoBench graph wrapper."""
        model = self._model()
        x = torch.randn(simple_graph_0.num_nodes, self.input_dim)
        batch = torch.zeros(simple_graph_0.num_nodes, dtype=torch.long)

        out = model(
            x,
            simple_graph_0.edge_index,
            batch=batch,
            unused_wrapper_value=True,
        )

        assert out.shape == (simple_graph_0.num_nodes, self.hidden_dim)

    @pytest.mark.parametrize(
        ("alias", "reaction"),
        [
            ("bspm", "blurring_sharpening"),
            ("f", "fisher"),
            ("ac", "allen_cahn"),
            ("z", "zeldovich"),
            ("st", "source"),
            ("fb", "filter_bank"),
            ("fb*", "filter_bank_star"),
        ],
    )
    def test_reaction_aliases(self, alias, reaction):
        """Test aliases used by the paper and official implementation."""
        model = self._model(reaction_term=alias)

        assert model.reaction_term == reaction

    @pytest.mark.parametrize(
        "option",
        [
            {"reaction_term": "unknown"},
            {"adjacency": "dense"},
            {"normalization": "none"},
            {"solver": "dopri5"},
            {"heads": 0},
        ],
    )
    def test_invalid_options(self, option):
        """Test that unsupported settings fail clearly."""
        with pytest.raises(ValueError):
            self._model(**option)

    def test_original_adjacency_euler_equation(self):
        """Test one Euler step against dH/dt = alpha(AH-H)."""
        model = self._model(
            reaction_term="source",
            alpha_mode="scalar",
            beta_mode="scalar",
            constrain_alpha=False,
            constrain_beta=False,
            integration_time=0.25,
            step_size=0.25,
        )
        model.alpha.data.fill_(0.75)
        model.beta.data.zero_()
        x = torch.tensor([[1.0] * 8, [3.0] * 8])
        edge_index = torch.tensor([[0, 1], [1, 0]])

        out = model(x, edge_index)
        adjacency_x = x.flip(0)
        expected = x + 0.25 * 0.75 * (adjacency_x - x)

        assert torch.allclose(out, expected)

    @pytest.mark.parametrize(
        "reaction_term",
        [
            "blurring_sharpening",
            "fisher",
            "allen_cahn",
            "zeldovich",
            "source",
            "filter_bank",
            "filter_bank_star",
        ],
    )
    def test_all_reactions_are_finite(self, simple_graph_0, reaction_term):
        """Test every reaction term from Eq. (10)."""
        model = self._model(reaction_term=reaction_term)
        x = torch.randn(simple_graph_0.num_nodes, self.input_dim)

        out = model(x, simple_graph_0.edge_index)

        assert out.shape == (simple_graph_0.num_nodes, self.hidden_dim)
        assert torch.isfinite(out).all()

    def test_blurring_sharpening_equation(self):
        """Test r(H) = (A - A^2)H from the paper."""
        model = self._model(
            reaction_term="bs",
            alpha_mode="scalar",
            beta_mode="scalar",
            constrain_alpha=False,
            constrain_beta=False,
        )
        model.alpha.data.zero_()
        model.beta.data.fill_(1.0)
        x = torch.tensor(
            [
                [1.0] * self.input_dim,
                [2.0] * self.input_dim,
                [4.0] * self.input_dim,
            ]
        )
        edge_index = torch.tensor([[0, 1, 2], [1, 2, 0]])
        ax = x[[2, 0, 1]]
        a2x = x[[1, 2, 0]]

        out = model(x, edge_index)

        assert torch.allclose(out, x + ax - a2x)

    def test_soft_adjacency_is_target_normalized(self):
        """Test learned adjacency normalization over incoming edges."""
        model = GREADEncoder(
            self.input_dim,
            self.hidden_dim,
            heads=2,
            self_loop_weight=1.0,
        )
        x = torch.randn(4, self.input_dim)
        edge_index = torch.tensor([[0, 1, 2, 3], [1, 0, 3, 2]])
        edges = _prepare_edges(
            edge_index,
            None,
            x.shape[0],
            x.dtype,
            x.device,
            False,
            1.0,
        )

        operator = model._soft_operator(x, edges)
        target_sums = torch.zeros(x.shape[0]).index_add(
            0, operator.target, operator.weight
        )

        assert torch.allclose(target_sums, torch.ones_like(target_sums))

    def test_soft_adjacency_initialization_is_nonuniform(self):
        """Test Xavier projections produce nonuniform learned edge weights."""
        torch.manual_seed(23)
        model = GREADEncoder(
            self.input_dim,
            self.hidden_dim,
            heads=1,
            self_loop_weight=0.0,
        )
        x = torch.randn(3, self.input_dim)
        edge_index = torch.tensor([[0, 1], [2, 2]])
        edges = _prepare_edges(
            edge_index, None, 3, x.dtype, x.device, False, 0.0
        )

        operator = model._soft_operator(x, edges)
        incoming = operator.weight[operator.target == 2]
        uniform = torch.full_like(incoming, 1 / incoming.numel())

        assert incoming.numel() == 2
        assert not torch.allclose(incoming, uniform)

    def test_directed_message_flow(self):
        """Test that edge 0 -> 1 propagates source features to target 1."""
        model = self._model(
            reaction_term="source",
            alpha_mode="scalar",
            beta_mode="scalar",
            constrain_alpha=False,
            constrain_beta=False,
        )
        model.alpha.data.fill_(1.0)
        model.beta.data.zero_()
        x = torch.zeros(3, self.input_dim)
        x[0] = 2.0
        edge_index = torch.tensor([[0], [1]])

        out = model(x, edge_index)
        expected = torch.zeros_like(x)
        expected[1] = 2.0

        assert torch.allclose(out, expected)

    @pytest.mark.parametrize("self_loop_weight", [0.0, 1.0])
    def test_self_loop_modes(self, self_loop_weight):
        """Test propagation with disabled and explicit self-loops."""
        model = self._model(
            reaction_term="source",
            alpha_mode="scalar",
            beta_mode="scalar",
            constrain_alpha=False,
            constrain_beta=False,
            self_loop_weight=self_loop_weight,
        )
        model.alpha.data.fill_(1.0)
        model.beta.data.zero_()
        x = torch.randn(2, self.input_dim)
        edge_index = torch.empty((2, 0), dtype=torch.long)

        out = model(x, edge_index)
        expected = torch.zeros_like(x) if self_loop_weight == 0 else x

        assert torch.allclose(out, expected)

    def test_existing_self_loops_are_not_duplicated(self):
        """Test coalescing when the graph already contains a self-loop."""
        edge_index = torch.tensor([[0], [0]])
        edge_weight = torch.tensor([2.0])

        edges = _prepare_edges(
            edge_index,
            edge_weight,
            2,
            torch.float32,
            edge_index.device,
            False,
            1.0,
        )

        assert edges.source.numel() == 2
        assert torch.equal(edges.source, edges.target)
        assert torch.allclose(
            edges.edge_weight.sort().values, torch.tensor([1.0, 2.0])
        )

    @pytest.mark.parametrize("dynamic_adjacency", [False, True])
    def test_dynamic_soft_adjacency_backward(
        self, simple_graph_0, dynamic_adjacency
    ):
        """Test fixed and dynamic soft adjacency shapes and gradients."""
        model = GREADEncoder(
            self.input_dim,
            self.hidden_dim,
            heads=2,
            solver="rk4",
            dynamic_adjacency=dynamic_adjacency,
        )
        x = torch.randn(
            simple_graph_0.num_nodes, self.input_dim, requires_grad=True
        )

        out = model(x, simple_graph_0.edge_index)
        out.square().mean().backward()

        assert out.shape == (simple_graph_0.num_nodes, self.hidden_dim)
        assert x.grad is not None
        assert torch.isfinite(x.grad).all()

    @pytest.mark.parametrize(
        ("dynamic_adjacency", "expected_calls"), [(False, 1), (True, 8)]
    )
    def test_dynamic_adjacency_projection_calls(
        self, simple_graph_0, dynamic_adjacency, expected_calls
    ):
        """Test dynamic RK4 adjacency at every derivative evaluation."""
        model = GREADEncoder(
            self.input_dim,
            self.hidden_dim,
            heads=2,
            solver="rk4",
            integration_time=1.0,
            step_size=0.5,
            dynamic_adjacency=dynamic_adjacency,
        )
        query_calls = []
        key_calls = []
        query_handle = model.query.register_forward_hook(
            lambda module, inputs, output: query_calls.append(output)
        )
        key_handle = model.key.register_forward_hook(
            lambda module, inputs, output: key_calls.append(output)
        )

        model(
            torch.randn(simple_graph_0.num_nodes, self.input_dim),
            simple_graph_0.edge_index,
        )
        query_handle.remove()
        key_handle.remove()

        assert len(query_calls) == expected_calls
        assert len(key_calls) == expected_calls

    def test_original_adjacency_ignores_dynamic_flag(self, simple_graph_0):
        """Test that original adjacency remains fixed in both modes."""
        model = self._model(dynamic_adjacency=False)
        dynamic_model = deepcopy(model)
        dynamic_model.dynamic_adjacency = True
        x = torch.randn(simple_graph_0.num_nodes, self.input_dim)

        fixed = model(x, simple_graph_0.edge_index)
        dynamic = dynamic_model(x, simple_graph_0.edge_index)

        assert torch.allclose(fixed, dynamic)

    def test_beta_initialization_range(self):
        """Test stable signed initialization of unconstrained beta."""
        torch.manual_seed(5)
        model = self._model(beta_mode="channel", constrain_beta=False)

        assert torch.all(model.beta >= -0.1)
        assert torch.all(model.beta <= 0.1)

    def test_rk4_solver_reference(self):
        """Test one RK4 step for the linear ODE dx/dt = -x."""
        model = self._model(
            reaction_term="source",
            solver="rk4",
            alpha_mode="scalar",
            beta_mode="scalar",
            constrain_alpha=False,
            constrain_beta=False,
        )
        model.alpha.data.fill_(1.0)
        model.beta.data.zero_()
        x = torch.randn(2, self.input_dim)
        edge_index = torch.empty((2, 0), dtype=torch.long)

        out = model(x, edge_index)

        assert torch.allclose(out, 0.375 * x)

    def test_batched_graphs_match_separate_forwards(
        self, simple_graph_0, simple_graph_1
    ):
        """Test that sparse soft adjacency does not mix batched graphs."""
        torch.manual_seed(12)
        model = GREADEncoder(
            self.input_dim,
            self.hidden_dim,
            heads=2,
            solver="euler",
            output_dropout=0.0,
        )
        separate_model = deepcopy(model)
        batch_data = Batch.from_data_list([simple_graph_0, simple_graph_1])
        x0 = torch.randn(simple_graph_0.num_nodes, self.input_dim)
        x1 = torch.randn(simple_graph_1.num_nodes, self.input_dim)

        batched = model(
            torch.cat((x0, x1)),
            batch_data.edge_index,
            batch=batch_data.batch,
        )
        separate = torch.cat(
            (
                separate_model(x0, simple_graph_0.edge_index),
                separate_model(x1, simple_graph_1.edge_index),
            )
        )

        assert torch.allclose(batched, separate, atol=1e-6, rtol=1e-5)

    def test_weighted_edges_and_input_projection(self, simple_graph_0):
        """Test weighted propagation when input and hidden dimensions differ."""
        model = GREADEncoder(
            input_dim=5,
            hidden_dim=self.hidden_dim,
            adjacency="original",
            solver="euler",
            make_undirected=True,
        )
        edge_weight = torch.linspace(
            0.5, 1.5, simple_graph_0.edge_index.shape[1]
        )

        out = model(
            torch.randn(simple_graph_0.num_nodes, 5),
            simple_graph_0.edge_index,
            edge_weight=edge_weight,
        )

        assert out.shape == (simple_graph_0.num_nodes, self.hidden_dim)
        assert torch.isfinite(out).all()

    def test_rk4_backward_pass(self, simple_graph_0):
        """Test gradient flow through soft adjacency and RK4 integration."""
        model = GREADEncoder(
            self.input_dim,
            self.hidden_dim,
            heads=2,
            integration_time=1.0,
            step_size=0.5,
            solver="rk4",
        )
        x = torch.randn(
            simple_graph_0.num_nodes, self.input_dim, requires_grad=True
        )

        model(x, simple_graph_0.edge_index).square().mean().backward()

        assert x.grad is not None
        assert torch.isfinite(x.grad).all()
        assert all(
            parameter.grad is not None
            for parameter in model.parameters()
            if parameter.requires_grad
        )
