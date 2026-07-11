"""Unit tests for Deep Neural Sheaf Diffusion (DNSD)."""

import pytest
import torch

from topobench.nn.backbones.graph.dnsd import DNSDEncoder
from topobench.nn.backbones.graph.nsd_utils.dnsd_models import (
    DeepNeuralSheafDiffusion,
    DiagonalRestrictionBuilder,
    DNSDConv,
    FullRestrictionBuilder,
    Orthogonal,
    OrthogonalRestrictionBuilder,
    _scale_linear_,
)


def _graph(num_nodes=12, num_edges=40, feat_dim=16):
    """Build a random directed graph for testing.

    Parameters
    ----------
    num_nodes : int
        Number of nodes.
    num_edges : int
        Number of directed edges.
    feat_dim : int
        Node feature dimension.

    Returns
    -------
    tuple of torch.Tensor
        Node features [num_nodes, feat_dim] and edge_index [2, num_edges].
    """
    x = torch.randn(num_nodes, feat_dim)
    edge_index = torch.randint(0, num_nodes, (2, num_edges))
    return x, edge_index


class TestDNSDEncoder:
    """Test the DNSDEncoder backbone wrapper."""

    def setup_method(self):
        """Set up common dimensions."""
        self.input_dim = 16
        self.hidden_dim = 32
        self.d = 4
        self.num_nodes = 12

    def test_default_forward_shape(self):
        """Default encoder returns [num_nodes, hidden_dim] embeddings."""
        x, ei = _graph(self.num_nodes, 40, self.input_dim)
        model = DNSDEncoder(self.input_dim, self.hidden_dim, d=self.d)
        out = model(x, ei)
        assert out.shape == (self.num_nodes, self.hidden_dim)
        assert torch.isfinite(out).all()

    def test_defaults_are_full_dnsd(self):
        """Encoder defaults switch on the DNSD adjacency operator and odd act."""
        model = DNSDEncoder(self.input_dim, self.hidden_dim, d=self.d)
        assert model.dnsd.convs[0].directional is True
        assert model.dnsd.convs[0].odd is True

    @pytest.mark.parametrize("map_type", ["diagonal", "full", "orthogonal"])
    def test_map_types(self, map_type):
        """All restriction-map families run end to end."""
        x, ei = _graph(self.num_nodes, 40, self.input_dim)
        model = DNSDEncoder(
            self.input_dim, self.hidden_dim, d=self.d, map_type=map_type
        )
        out = model(x, ei)
        assert out.shape == (self.num_nodes, self.hidden_dim)
        assert torch.isfinite(out).all()

    def test_extra_kwargs_ignored(self):
        """Unknown kwargs are accepted and ignored."""
        model = DNSDEncoder(
            self.input_dim, self.hidden_dim, d=self.d, something_extra=123
        )
        assert isinstance(model, DNSDEncoder)

    def test_edge_attr_and_batch_ignored(self):
        """edge_attr / edge_weight / batch are accepted but unused."""
        x, ei = _graph(self.num_nodes, 40, self.input_dim)
        model = DNSDEncoder(self.input_dim, self.hidden_dim, d=self.d)
        out = model(
            x,
            ei,
            edge_attr=torch.randn(ei.size(1), 3),
            edge_weight=torch.randn(ei.size(1)),
            batch=torch.zeros(self.num_nodes, dtype=torch.long),
        )
        assert out.shape == (self.num_nodes, self.hidden_dim)

    def test_undirected_conversion(self):
        """A single directed edge is symmetrized before diffusion."""
        x = torch.randn(4, self.input_dim)
        ei = torch.tensor([[0], [1]])  # one directed edge 0->1
        model = DNSDEncoder(self.input_dim, self.hidden_dim, d=self.d)
        out = model(x, ei)
        assert out.shape == (4, self.hidden_dim)
        assert torch.isfinite(out).all()

    def test_backprop(self):
        """Gradients flow to input-projection parameters."""
        x, ei = _graph(self.num_nodes, 40, self.input_dim)
        model = DNSDEncoder(self.input_dim, self.hidden_dim, d=self.d)
        model(x, ei).sum().backward()
        assert model.dnsd.lin_in.weight.grad is not None


class TestDNSDFlags:
    """Test the individual DNSD ablation flags and their wiring."""

    def setup_method(self):
        """Set up common dimensions."""
        self.hidden_dim = 32
        self.d = 4

    def test_directional_differs_from_laplacian(self):
        """The sheaf-adjacency operator (DNSD's defining change) must produce a
        different update than the Laplacian, with all else held equal.

        The two convs share weights (via state_dict) and see distinct
        source/target maps, so any difference is due solely to the
        ``directional`` flag — not to random init or a degenerate F_s == F_t.
        """
        dir_conv = DNSDConv(self.hidden_dim, self.d, directional=True)
        lap_conv = DNSDConv(self.hidden_dim, self.d, directional=False)
        lap_conv.load_state_dict(dir_conv.state_dict())

        x = torch.randn(6, self.hidden_dim)
        ei = torch.randint(0, 6, (2, 15))
        f_source = torch.randn(15, self.d, self.d)
        f_target = torch.randn(15, self.d, self.d)  # distinct from f_source

        out_dir = dir_conv(x, ei, f_source, f_target)
        out_lap = lap_conv(x, ei, f_source, f_target)
        assert out_dir.shape == (6, self.hidden_dim)
        assert not torch.allclose(out_dir, out_lap)

    def test_message_matches_oracle_both_operators(self):
        """The per-edge message equals the hand-computed sheaf operator.

        Validated directly on ``DNSDConv.message`` (a pure per-edge function),
        so there is no PyG aggregation/flow-direction ambiguity to reconstruct.
        directional=True  -> norm * F_source^T (F_target x_j)
        directional=False -> norm * F_source^T (F_source x_i - F_target x_j)
        """
        s, h = self.d, self.hidden_dim // self.d
        e = 15
        x_i = torch.randn(e, self.hidden_dim)
        x_j = torch.randn(e, self.hidden_dim)
        f_source = torch.randn(e, s, s)
        f_target = torch.randn(e, s, s)
        norm = torch.rand(e)

        fx_t = torch.bmm(f_target, x_j.view(e, s, h))
        fx_s = torch.bmm(f_source, x_i.view(e, s, h))
        ft = f_source.transpose(1, 2)

        dir_conv = DNSDConv(self.hidden_dim, self.d, directional=True)
        oracle_dir = (torch.bmm(ft, fx_t) * norm.view(-1, 1, 1)).view(e, -1)
        assert torch.allclose(
            dir_conv.message(x_i, x_j, f_source, f_target, norm),
            oracle_dir,
            atol=1e-5,
        )

        lap_conv = DNSDConv(self.hidden_dim, self.d, directional=False)
        oracle_lap = (
            torch.bmm(ft, fx_s - fx_t) * norm.view(-1, 1, 1)
        ).view(e, -1)
        assert torch.allclose(
            lap_conv.message(x_i, x_j, f_source, f_target, norm),
            oracle_lap,
            atol=1e-5,
        )

    def test_odd_flag_changes_activation(self):
        """odd=True (tanh) must yield a different, bounded update than relu.

        Shared weights isolate the nonlinearity; a large-magnitude input makes
        the tanh bound visible where relu is unbounded.
        """
        odd_conv = DNSDConv(self.hidden_dim, self.d, odd=True)
        relu_conv = DNSDConv(self.hidden_dim, self.d, odd=False)
        relu_conv.load_state_dict(odd_conv.state_dict())

        x = torch.randn(6, self.hidden_dim) * 1e3  # large -> saturates tanh
        ei = torch.randint(0, 6, (2, 15))
        f = torch.randn(15, self.d, self.d)

        out_odd = odd_conv(x, ei, f, f)
        out_relu = relu_conv(x, ei, f, f)
        assert not torch.allclose(out_odd, out_relu)

    def test_stalk_gate_adds_params(self):
        """use_stalk_gate=True registers the gate linear layer."""
        conv = DNSDConv(self.hidden_dim, self.d, use_stalk_gate=True)
        assert hasattr(conv, "gate_lin")
        # Zero-init keeps the untrained gate symmetric at sigmoid(0)=0.5.
        assert torch.equal(
            conv.gate_lin.weight, torch.zeros_like(conv.gate_lin.weight)
        )
        assert torch.equal(
            conv.gate_lin.bias, torch.zeros_like(conv.gate_lin.bias)
        )

    def test_stalk_gate_absent_by_default(self):
        """Without the flag there is no gate layer."""
        conv = DNSDConv(self.hidden_dim, self.d)
        assert not hasattr(conv, "gate_lin")

    def test_stalk_dropout_is_identity_in_eval(self):
        """stalk_dropout is a no-op outside training mode."""
        conv = DNSDConv(self.hidden_dim, self.d, stalk_dropout=0.5)
        conv.eval()
        x = torch.randn(6, self.hidden_dim)
        ei = torch.randint(0, 6, (2, 15))
        f = torch.randn(15, self.d, self.d)
        a = conv(x, ei, f, f)
        b = conv(x, ei, f, f)
        assert torch.allclose(a, b)

    def test_stalk_dropout_active_in_train(self):
        """stalk_dropout perturbs the update while training."""
        torch.manual_seed(0)
        conv = DNSDConv(self.hidden_dim, self.d, stalk_dropout=0.9)
        conv.train()
        x = torch.randn(6, self.hidden_dim)
        ei = torch.randint(0, 6, (2, 15))
        f = torch.randn(15, self.d, self.d)
        a = conv(x, ei, f, f)
        b = conv(x, ei, f, f)
        assert not torch.allclose(a, b)

    def test_gate_and_dropout_combined(self):
        """Gate and per-stalk dropout compose in one forward pass."""
        conv = DNSDConv(
            self.hidden_dim,
            self.d,
            use_stalk_gate=True,
            stalk_dropout=0.3,
            odd=True,
            directional=True,
        )
        conv.train()
        x = torch.randn(6, self.hidden_dim)
        ei = torch.randint(0, 6, (2, 15))
        f = torch.randn(15, self.d, self.d)
        out = conv(x, ei, f, f)
        assert out.shape == (6, self.hidden_dim)


class TestRestrictionBuilders:
    """Test the restriction-map builders."""

    def setup_method(self):
        """Set up common dimensions."""
        self.hidden_dim = 8
        self.d = 3
        self.num_edges = 15

    def _inputs(self):
        """Create node features and edges for a builder.

        Returns
        -------
        tuple of torch.Tensor
            Node features and edge_index.
        """
        x = torch.randn(6, self.hidden_dim)
        ei = torch.randint(0, 6, (2, self.num_edges))
        return x, ei

    def test_full_builder_shape_and_range(self):
        """Full builder returns [E, d, d] maps bounded by tanh."""
        b = FullRestrictionBuilder(self.hidden_dim, self.d)
        x, ei = self._inputs()
        maps = b(x, ei)
        assert maps.shape == (self.num_edges, self.d, self.d)
        assert maps.abs().max() <= 1.0

    def test_diagonal_builder_is_diagonal(self):
        """Diagonal builder returns diagonal maps only."""
        b = DiagonalRestrictionBuilder(self.hidden_dim, self.d)
        x, ei = self._inputs()
        maps = b(x, ei)
        assert maps.shape == (self.num_edges, self.d, self.d)
        # Off-diagonal entries must be exactly zero.
        off_diag = maps - torch.diag_embed(
            torch.diagonal(maps, dim1=-2, dim2=-1)
        )
        assert torch.count_nonzero(off_diag) == 0

    def test_orthogonal_builder_is_orthogonal(self):
        """Orthogonal builder returns maps with Q^T Q ≈ I."""
        b = OrthogonalRestrictionBuilder(self.hidden_dim, self.d)
        x, ei = self._inputs()
        maps = b(x, ei)
        assert maps.shape == (self.num_edges, self.d, self.d)
        eye = torch.eye(self.d).expand(self.num_edges, self.d, self.d)
        qtq = torch.bmm(maps.transpose(1, 2), maps)
        assert torch.allclose(qtq, eye, atol=1e-4)

    def test_init_scale_shrinks_weights(self):
        """init_scale rescales the builder's linear weights by exactly the
        factor. Both builders are seeded identically, and ``_scale_linear_``
        runs *after* ``nn.Linear`` init, so the scaled params must equal the
        unscaled params times 0.5 to the bit. This fails if ``_scale_linear_``
        is a no-op or is removed.
        """
        torch.manual_seed(0)
        b_default = FullRestrictionBuilder(self.hidden_dim, self.d)
        torch.manual_seed(0)
        b_scaled = FullRestrictionBuilder(
            self.hidden_dim, self.d, init_scale=0.5
        )
        assert torch.allclose(b_scaled.lin.weight, b_default.lin.weight * 0.5)
        assert torch.allclose(b_scaled.lin.bias, b_default.lin.bias * 0.5)

    def test_scale_linear_noop_when_one(self):
        """_scale_linear_ leaves parameters unchanged when scale is 1.0."""
        lin = torch.nn.Linear(4, 4)
        w0 = lin.weight.clone()
        _scale_linear_(lin, 1.0)
        assert torch.equal(lin.weight, w0)

    def test_scale_linear_bias_none(self):
        """_scale_linear_ handles a bias-free linear layer."""
        lin = torch.nn.Linear(4, 4, bias=False)
        w0 = lin.weight.clone()
        _scale_linear_(lin, 0.5)
        assert torch.allclose(lin.weight, w0 * 0.5)

    def test_base_builder_raises(self):
        """The abstract base builder forward raises NotImplementedError."""
        from topobench.nn.backbones.graph.nsd_utils.dnsd_models import (
            _RestrictionBuilder,
        )

        with pytest.raises(NotImplementedError):
            _RestrictionBuilder().forward(None, None)


class TestOrthogonal:
    """Test the orthogonal-map parameterizations."""

    @pytest.mark.parametrize(
        "method", ["cayley", "matrix_exp", "householder", "qr"]
    )
    def test_methods_produce_orthogonal(self, method):
        """Every method yields matrices with Q^T Q ≈ I."""
        d, e = 4, 7
        ortho = Orthogonal(d, method=method)
        params = torch.randn(e, d * d)
        q = ortho(params)
        assert q.shape == (e, d, d)
        eye = torch.eye(d).expand(e, d, d)
        assert torch.allclose(
            torch.bmm(q.transpose(1, 2), q), eye, atol=1e-4
        )

    def test_unknown_method_raises(self):
        """An unsupported method raises ValueError."""
        with pytest.raises(ValueError):
            Orthogonal(3, method="nope")


class TestDeepNeuralSheafDiffusion:
    """Test the full DNSD model directly."""

    def setup_method(self):
        """Set up common dimensions."""
        self.input_dim = 16
        self.hidden_dim = 32
        self.output_dim = 5
        self.d = 4

    def test_forward_shape(self):
        """Model maps [N, input_dim] to [N, output_dim]."""
        x, ei = _graph(10, 30, self.input_dim)
        model = DeepNeuralSheafDiffusion(
            self.input_dim, self.hidden_dim, self.output_dim, num_stalks=self.d
        )
        out = model(x, ei)
        assert out.shape == (10, self.output_dim)

    def test_non_divisible_hidden_raises(self):
        """hidden_dim not divisible by num_stalks raises ValueError."""
        with pytest.raises(ValueError, match="divisible"):
            DeepNeuralSheafDiffusion(
                self.input_dim, 30, self.output_dim, num_stalks=4
            )

    def test_unknown_map_type_raises(self):
        """Unknown map_type raises ValueError."""
        with pytest.raises(ValueError, match="map_type"):
            DeepNeuralSheafDiffusion(
                self.input_dim,
                self.hidden_dim,
                self.output_dim,
                num_stalks=self.d,
                map_type="banana",
            )

    @pytest.mark.parametrize("mode", ["embedding", "stalk", None, False])
    def test_layer_norm_modes(self, mode):
        """All valid LayerNorm modes run and produce finite output."""
        x, ei = _graph(10, 30, self.input_dim)
        model = DeepNeuralSheafDiffusion(
            self.input_dim,
            self.hidden_dim,
            self.output_dim,
            num_stalks=self.d,
            layer_norm_mode=mode,
        )
        out = model(x, ei)
        assert out.shape == (10, self.output_dim)
        assert torch.isfinite(out).all()
        if mode in (None, False):
            assert model.norms is None
        else:
            assert model.norms is not None

    def test_invalid_layer_norm_mode_raises(self):
        """An invalid LayerNorm mode raises ValueError."""
        with pytest.raises(ValueError, match="layer_norm_mode"):
            DeepNeuralSheafDiffusion(
                self.input_dim,
                self.hidden_dim,
                self.output_dim,
                num_stalks=self.d,
                layer_norm_mode="weird",
            )

    def test_input_dropout_noop_in_eval(self):
        """input_dropout does not perturb the deterministic eval forward."""
        x, ei = _graph(10, 30, self.input_dim)
        model = DeepNeuralSheafDiffusion(
            self.input_dim,
            self.hidden_dim,
            self.output_dim,
            num_stalks=self.d,
            input_dropout=0.5,
        )
        model.eval()
        assert torch.allclose(model(x, ei), model(x, ei))

    def test_deep_network_is_finite(self):
        """A deep DNSD stack stays finite — the model's design goal."""
        x, ei = _graph(10, 30, self.input_dim)
        model = DeepNeuralSheafDiffusion(
            self.input_dim,
            self.hidden_dim,
            self.output_dim,
            num_layers=16,
            num_stalks=self.d,
            directional=True,
            odd=True,
        )
        model.eval()
        out = model(x, ei)
        assert torch.isfinite(out).all()

    def test_orthogonal_uses_ortho_method(self):
        """ortho_method is forwarded to the orthogonal builders."""
        model = DeepNeuralSheafDiffusion(
            self.input_dim,
            self.hidden_dim,
            self.output_dim,
            num_stalks=self.d,
            map_type="orthogonal",
            ortho_method="matrix_exp",
        )
        assert model.source_builders[0].ortho.method == "matrix_exp"
