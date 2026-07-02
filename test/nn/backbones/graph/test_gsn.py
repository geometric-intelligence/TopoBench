"""Unit tests for the GSN message-passing layers and models."""

import pytest
import torch
from torch_geometric.data import Batch, Data

# NOTE: import from the submodule, not the package. The backbones package
# auto-discovers modules and re-imports them under a synthetic name
# ("__init__.gsn"), which breaks PyG's MessagePassing signature inspector
# (it looks the class module up in sys.modules).
from topobench.nn.backbones.graph.gsn import (
    GSNGINconcatBaseLayer,
    GSNGINconcatELayer,
    GSNGINconcatModel,
    GSNGINconcatVLayer,
    GSNGINVirtualNodeLayerV,
    GSNGINVirtualNodeModule,
    GSNMPNNBaselineVLayer,
    mlp_builder,
    mlp_dimension_builder,
)

# --- small shared inputs -------------------------------------------------

EDGE_INDEX = torch.tensor([[0, 1, 1, 2], [1, 0, 2, 1]], dtype=torch.long)
NUM_NODES = 3
IN = 4
GSN = 5
OUT = 6


def node_data():
    """A 3-node graph carrying node-level GSN encodings."""
    return Data(
        x=torch.randn(NUM_NODES, IN),
        edge_index=EDGE_INDEX,
        node_gsn_encodings=torch.randn(NUM_NODES, GSN),
    )


def edge_data(edge_dim=0):
    """A 3-node graph carrying edge-level GSN encodings (2E directed rows)."""
    data = Data(
        x=torch.randn(NUM_NODES, IN),
        edge_index=EDGE_INDEX,
        edge_gsn_encodings=torch.randn(EDGE_INDEX.size(1), GSN),
    )
    if edge_dim > 0:
        data.edge_attr = torch.randn(EDGE_INDEX.size(1), edge_dim)
    return data


class TestMLPBuilders:
    """Tests for the MLP-construction helpers."""

    def test_dimension_builder(self):
        """A flat arch becomes consecutive (in, out) pairs."""
        assert mlp_dimension_builder([4, 8, 6]) == [(4, 8), (8, 6)]

    def test_builder_inserts_activation_between_layers(self):
        """A 2-layer MLP has 2 Linears + 1 ReLU, no trailing activation."""
        mlp = mlp_builder(mlp_dimension_builder([4, 8, 6]))
        n_linear = sum(1 for m in mlp if isinstance(m, torch.nn.Linear))
        n_relu = sum(1 for m in mlp if isinstance(m, torch.nn.ReLU))
        assert n_linear == 2
        assert n_relu == 1
        # last module is a Linear (unbounded output)
        assert isinstance(mlp[-1], torch.nn.Linear)


class TestConcatLayers:
    """Tests for the GIN-concat V/E layers and their abstract base."""

    def test_base_layer_is_abstract(self):
        """The abstract base cannot be instantiated directly."""
        with pytest.raises(TypeError):
            GSNGINconcatBaseLayer(IN, GSN, OUT)

    def test_v_layer_forward_shape(self):
        """Node-mode layer maps to (N, out_channels)."""
        layer = GSNGINconcatVLayer(IN, GSN, OUT)
        data = node_data()
        out = layer(data.edge_index, data.x, data.node_gsn_encodings)
        assert out.shape == (NUM_NODES, OUT)

    def test_v_layer_rejects_edge_dim(self):
        """Node-mode layer forbids edge_dim > 0."""
        with pytest.raises(ValueError):
            GSNGINconcatVLayer(IN, GSN, OUT, edge_dim=2)

    def test_default_update_is_two_layer_mlp(self):
        """The default UP update is a genuine 2-layer MLP with a nonlinearity."""
        layer = GSNGINconcatVLayer(IN, GSN, OUT)
        n_linear = sum(
            1 for m in layer.UP if isinstance(m, torch.nn.Linear)
        )
        n_activation = sum(
            1 for m in layer.UP if isinstance(m, torch.nn.ReLU)
        )
        assert n_linear == 2
        # an activation between the two linears keeps the update non-linear
        assert n_activation == 1

    def test_eps_buffer_vs_parameter(self):
        """eps is a Parameter iff train_eps, else a registered buffer."""
        trainable = GSNGINconcatVLayer(IN, GSN, OUT, train_eps=True)
        fixed = GSNGINconcatVLayer(IN, GSN, OUT, train_eps=False)
        assert isinstance(trainable.eps, torch.nn.Parameter)
        assert "eps" in dict(fixed.named_buffers())

    def test_eps_receives_gradient(self):
        """A trainable eps accumulates a gradient after backward."""
        layer = GSNGINconcatELayer(IN, GSN, OUT, train_eps=True)
        data = edge_data()
        out = layer(data.edge_index, data.x, data.edge_gsn_encodings)
        out.sum().backward()
        assert layer.eps.grad is not None

    def test_e_layer_ignores_edge_attr_when_edge_dim_zero(self):
        """edge_dim=0 + edge_attr present is tolerated (attr ignored)."""
        layer = GSNGINconcatELayer(IN, GSN, OUT, edge_dim=0)
        out = layer(
            EDGE_INDEX,
            torch.randn(NUM_NODES, IN),
            torch.randn(EDGE_INDEX.size(1), GSN),
            edge_attr=torch.randn(EDGE_INDEX.size(1), 2),
        )
        assert out.shape == (NUM_NODES, OUT)

    def test_e_layer_correct_edge_dim(self):
        """A matching edge_dim / edge_attr runs."""
        layer = GSNGINconcatELayer(IN, GSN, OUT, edge_dim=2)
        data = edge_data(edge_dim=2)
        out = layer(
            data.edge_index,
            data.x,
            data.edge_gsn_encodings,
            edge_attr=data.edge_attr,
        )
        assert out.shape == (NUM_NODES, OUT)

    def test_e_layer_width_mismatch_raises(self):
        """A wrong edge_attr width raises a clear ValueError."""
        layer = GSNGINconcatELayer(IN, GSN, OUT, edge_dim=2)
        with pytest.raises(ValueError):
            layer(
                EDGE_INDEX,
                torch.randn(NUM_NODES, IN),
                torch.randn(EDGE_INDEX.size(1), GSN),
                edge_attr=torch.randn(EDGE_INDEX.size(1), 5),
            )

    def test_e_layer_missing_edge_attr_raises(self):
        """edge_dim > 0 with no edge_attr raises a clear ValueError."""
        layer = GSNGINconcatELayer(IN, GSN, OUT, edge_dim=2)
        with pytest.raises(ValueError):
            layer(
                EDGE_INDEX,
                torch.randn(NUM_NODES, IN),
                torch.randn(EDGE_INDEX.size(1), GSN),
            )


class TestConcatModel:
    """Tests for the multi-layer GSNGINconcatModel."""

    @pytest.mark.parametrize("num_layers", [1, 3])
    def test_node_mode_forward(self, num_layers):
        """Node-mode model maps to (N, out_channels) for any depth."""
        model = GSNGINconcatModel(
            "node", num_layers, IN, GSN, OUT, width=8
        )
        assert model(node_data()).shape == (NUM_NODES, OUT)

    def test_edge_mode_forward(self):
        """Edge-mode model with edge features maps to (N, out_channels)."""
        model = GSNGINconcatModel("edge", 2, IN, GSN, OUT, width=8, edge_dim=2)
        assert model(edge_data(edge_dim=2)).shape == (NUM_NODES, OUT)

    def test_custom_gsn_keyword(self):
        """A custom gsn_keyword is read from the Data object."""
        model = GSNGINconcatModel(
            "node", 2, IN, GSN, OUT, width=8, gsn_keyword="my_key"
        )
        data = Data(
            x=torch.randn(NUM_NODES, IN),
            edge_index=EDGE_INDEX,
            my_key=torch.randn(NUM_NODES, GSN),
        )
        assert model(data).shape == (NUM_NODES, OUT)

    def test_invalid_num_layers(self):
        """num_layers < 1 is rejected."""
        with pytest.raises(ValueError):
            GSNGINconcatModel("node", 0, IN, GSN, OUT, width=8)

    def test_invalid_mode(self):
        """An unknown mode is rejected."""
        with pytest.raises(ValueError):
            GSNGINconcatModel("cell", 2, IN, GSN, OUT, width=8)

    def test_node_mode_rejects_edge_dim(self):
        """Node mode with edge_dim > 0 is rejected."""
        with pytest.raises(ValueError):
            GSNGINconcatModel("node", 2, IN, GSN, OUT, width=8, edge_dim=2)


class TestMPNNBaseline:
    """Tests for the general MPNN baseline layer."""

    def test_forward_without_edge_attr(self):
        """Runs with no edge features when edge_dim == 0."""
        layer = GSNMPNNBaselineVLayer(IN, OUT, GSN)
        data = node_data()
        out = layer(data.edge_index, data.x, data.node_gsn_encodings)
        assert out.shape == (NUM_NODES, OUT)

    def test_forward_with_edge_attr(self):
        """Runs with edge features when edge_dim > 0."""
        layer = GSNMPNNBaselineVLayer(IN, OUT, GSN, edge_dim=2)
        data = node_data()
        out = layer(
            data.edge_index,
            data.x,
            data.node_gsn_encodings,
            edge_attr=torch.randn(EDGE_INDEX.size(1), 2),
        )
        assert out.shape == (NUM_NODES, OUT)

    def test_edge_attr_presence_mismatch_raises(self):
        """Passing edge_attr while edge_dim == 0 raises."""
        layer = GSNMPNNBaselineVLayer(IN, OUT, GSN, edge_dim=0)
        data = node_data()
        with pytest.raises(ValueError):
            layer(
                data.edge_index,
                data.x,
                data.node_gsn_encodings,
                edge_attr=torch.randn(EDGE_INDEX.size(1), 2),
            )


class TestVirtualNode:
    """Tests for the GIN+VN layer and module."""

    def test_layer_forward_with_and_without_edge_attr(self):
        """The VN layer runs with and without edge embeddings."""
        layer = GSNGINVirtualNodeLayerV(OUT, common_embedding_dim=IN)
        x = torch.randn(NUM_NODES, IN)
        assert layer(EDGE_INDEX, x).shape == (NUM_NODES, OUT)

        layer_e = GSNGINVirtualNodeLayerV(
            OUT, common_embedding_dim=IN, edge_dim=IN
        )
        edge_attr = torch.randn(EDGE_INDEX.size(1), IN)
        assert layer_e(EDGE_INDEX, x, edge_attr).shape == (NUM_NODES, OUT)

    @pytest.mark.parametrize("num_layers", [1, 3])
    def test_module_forward(self, num_layers):
        """Node-mode module maps to (N, out_channels) for any depth."""
        module = GSNGINVirtualNodeModule(
            "node", num_layers, IN, OUT, GSN, common_embedding_dim=8
        )
        assert module(node_data()).shape == (NUM_NODES, OUT)

    def test_module_edge_dim_path(self):
        """The edge-feature path projects and runs."""
        module = GSNGINVirtualNodeModule(
            "node", 2, IN, OUT, GSN, common_embedding_dim=8, edge_dim=2
        )
        data = node_data()
        data.edge_attr = torch.randn(EDGE_INDEX.size(1), 2)
        assert module(data).shape == (NUM_NODES, OUT)

    def test_module_no_batch_fallback(self):
        """A single graph without data.batch is treated as one graph."""
        module = GSNGINVirtualNodeModule(
            "node", 2, IN, OUT, GSN, common_embedding_dim=8
        )
        data = node_data()
        assert getattr(data, "batch", None) is None
        assert module(data).shape == (NUM_NODES, OUT)

    def test_edge_mode_not_implemented(self):
        """Edge mode is not implemented and raises."""
        with pytest.raises(NotImplementedError):
            GSNGINVirtualNodeModule("edge", 2, IN, OUT, GSN)

    def test_edge_attr_with_edge_dim_zero_raises(self):
        """edge_attr present while edge_dim == 0 raises a RuntimeError."""
        module = GSNGINVirtualNodeModule(
            "node", 2, IN, OUT, GSN, common_embedding_dim=8
        )
        data = node_data()
        data.edge_attr = torch.randn(EDGE_INDEX.size(1), 2)
        with pytest.raises(RuntimeError):
            module(data)

    def test_per_graph_independence(self):
        """The virtual node stays per-graph: batching must not mix graphs."""
        torch.manual_seed(0)
        g1 = Data(
            x=torch.randn(3, IN),
            edge_index=torch.tensor([[0, 1, 1, 2], [1, 0, 2, 1]]),
            node_gsn_encodings=torch.randn(3, GSN),
        )
        g2 = Data(
            x=torch.randn(2, IN),
            edge_index=torch.tensor([[0, 1], [1, 0]]),
            node_gsn_encodings=torch.randn(2, GSN),
        )
        module = GSNGINVirtualNodeModule(
            "node", 3, IN, OUT, GSN, common_embedding_dim=8
        )
        module.eval()
        batched = module(Batch.from_data_list([g1, g2]))
        solo = module(Batch.from_data_list([g1]))
        assert torch.allclose(batched[:3], solo, atol=1e-5)


# --- helpers for the batch-norm / dropout tests --------------------------


def _count(module, cls):
    """Count direct children of ``module`` that are instances of ``cls``."""
    return sum(1 for m in module if isinstance(m, cls))


def two_graph_batch():
    """A batch of two 3-node graphs (>1 sample so BatchNorm works in train)."""
    g1 = Data(
        x=torch.randn(NUM_NODES, IN),
        edge_index=EDGE_INDEX,
        node_gsn_encodings=torch.randn(NUM_NODES, GSN),
    )
    g2 = Data(
        x=torch.randn(NUM_NODES, IN),
        edge_index=EDGE_INDEX,
        node_gsn_encodings=torch.randn(NUM_NODES, GSN),
    )
    return Batch.from_data_list([g1, g2])


class TestBatchNormAndDropout:
    """Tests for the optional BatchNorm / Dropout support."""

    def test_mlp_builder_inserts_bn_and_dropout_on_hidden_only(self):
        """BN + Dropout land on hidden layers; the output Linear stays raw."""
        # arch with two hidden layers -> three Linears
        mlp = mlp_builder(
            mlp_dimension_builder([4, 8, 8, 6]),
            batch_norm=True,
            dropout=0.3,
        )
        assert _count(mlp, torch.nn.Linear) == 3
        # one BN + one Dropout per non-final linear (i.e. per hidden layer)
        assert _count(mlp, torch.nn.BatchNorm1d) == 2
        assert _count(mlp, torch.nn.Dropout) == 2
        # nothing is appended after the final linear
        assert isinstance(mlp[-1], torch.nn.Linear)
        # BN precedes its activation; dropout follows it
        kinds = [type(m).__name__ for m in mlp]
        assert kinds[:4] == ["Linear", "BatchNorm1d", "ReLU", "Dropout"]

    def test_mlp_builder_defaults_add_nothing(self):
        """Without the flags the MLP is unchanged (no BN, no Dropout)."""
        mlp = mlp_builder(mlp_dimension_builder([4, 8, 6]))
        assert _count(mlp, torch.nn.BatchNorm1d) == 0
        assert _count(mlp, torch.nn.Dropout) == 0

    def test_zero_dropout_adds_no_dropout_layer(self):
        """dropout=0.0 must not insert Dropout modules (even with BN on)."""
        mlp = mlp_builder(
            mlp_dimension_builder([4, 8, 6]), batch_norm=True, dropout=0.0
        )
        assert _count(mlp, torch.nn.Dropout) == 0
        assert _count(mlp, torch.nn.BatchNorm1d) == 1

    def test_v_layer_threads_flags_into_update_mlp(self):
        """The concat V layer forwards BN/Dropout into its UP MLP."""
        layer = GSNGINconcatVLayer(
            IN, GSN, OUT, batch_norm=True, dropout=0.2
        )
        assert _count(layer.UP, torch.nn.BatchNorm1d) == 1
        assert _count(layer.UP, torch.nn.Dropout) == 1

    def test_e_layer_threads_flags_into_update_mlp(self):
        """The concat E layer forwards BN/Dropout into its UP MLP."""
        layer = GSNGINconcatELayer(
            IN, GSN, OUT, batch_norm=True, dropout=0.2
        )
        assert _count(layer.UP, torch.nn.BatchNorm1d) == 1
        assert _count(layer.UP, torch.nn.Dropout) == 1

    def test_mpnn_baseline_threads_flags_into_both_mlps(self):
        """The MPNN baseline layer applies BN/Dropout to inner and outer MLPs."""
        layer = GSNMPNNBaselineVLayer(
            IN, OUT, GSN, batch_norm=True, dropout=0.2
        )
        assert _count(layer.inner_mlp, torch.nn.BatchNorm1d) == 1
        assert _count(layer.inner_mlp, torch.nn.Dropout) == 1
        assert _count(layer.outer_mlp, torch.nn.BatchNorm1d) == 1
        assert _count(layer.outer_mlp, torch.nn.Dropout) == 1

    def test_vn_layer_threads_flags_into_update_mlp(self):
        """The VN layer forwards BN/Dropout into its UP MLP."""
        layer = GSNGINVirtualNodeLayerV(
            OUT, common_embedding_dim=IN, batch_norm=True, dropout=0.2
        )
        assert _count(layer.UP, torch.nn.BatchNorm1d) == 1
        assert _count(layer.UP, torch.nn.Dropout) == 1

    def test_concat_model_between_layer_norms(self):
        """Intermediate layers get BatchNorm; the final layer gets Identity."""
        model = GSNGINconcatModel(
            "node", 3, IN, GSN, OUT, width=8, batch_norm=True, dropout=0.2
        )
        norms = list(model._between_layer_norms)
        assert len(norms) == 3
        assert all(
            isinstance(n, torch.nn.BatchNorm1d) for n in norms[:-1]
        )
        # last layer output is left raw for the downstream head
        assert isinstance(norms[-1], torch.nn.Identity)

    def test_concat_model_defaults_are_identity(self):
        """With the flags off, every between-layer norm is an Identity."""
        model = GSNGINconcatModel("node", 3, IN, GSN, OUT, width=8)
        assert all(
            isinstance(n, torch.nn.Identity)
            for n in model._between_layer_norms
        )

    def test_vn_module_between_layer_norms_and_g_updaters(self):
        """VN module: intermediate BN, Identity on the last, BN in G-updaters."""
        module = GSNGINVirtualNodeModule(
            "node",
            3,
            IN,
            OUT,
            GSN,
            common_embedding_dim=8,
            batch_norm=True,
            dropout=0.2,
        )
        norms = list(module._between_layer_norms)
        assert all(isinstance(n, torch.nn.BatchNorm1d) for n in norms[:-1])
        assert isinstance(norms[-1], torch.nn.Identity)
        # the virtual-node update MLPs also pick up the flags
        assert all(
            _count(mlp, torch.nn.BatchNorm1d) == 1
            for mlp in module.G_updater_MLPs
        )

    def test_concat_model_forward_train_mode_with_batch(self):
        """A BN/Dropout concat model runs in train mode on a real batch."""
        model = GSNGINconcatModel(
            "node", 3, IN, GSN, OUT, width=8, batch_norm=True, dropout=0.2
        )
        model.train()
        out = model(two_graph_batch())
        assert out.shape == (2 * NUM_NODES, OUT)

    def test_vn_module_forward_train_mode_with_batch(self):
        """A BN/Dropout VN module runs in train mode on a real batch."""
        module = GSNGINVirtualNodeModule(
            "node",
            3,
            IN,
            OUT,
            GSN,
            common_embedding_dim=8,
            batch_norm=True,
            dropout=0.2,
        )
        module.train()
        out = module(two_graph_batch())
        assert out.shape == (2 * NUM_NODES, OUT)

    def test_dropout_active_in_train_inactive_in_eval(self):
        """Dropout perturbs outputs in train mode but is a no-op in eval."""
        torch.manual_seed(0)
        model = GSNGINconcatModel(
            "node", 3, IN, GSN, OUT, width=8, dropout=0.5
        )
        batch = two_graph_batch()

        # eval: deterministic across repeated calls
        model.eval()
        assert torch.allclose(model(batch), model(batch))

        # train: two forward passes differ because dropout masks differ
        model.train()
        assert not torch.allclose(model(batch), model(batch))
