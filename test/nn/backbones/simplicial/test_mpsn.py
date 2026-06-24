"""Unit tests for the MPSN (Message Passing Simplicial Network) backbone.

MPSN: Bodnar et al., "Weisfeiler and Lehman Go Topological: Message Passing
Simplicial Networks", ICML 2021, arXiv:2103.03212.

These tests assert behaviour on tiny, hand-built simplicial complexes (NO
GraphUniverse loading). The central capability being verified is the one the
model exists for: a complex *with* a filled triangle (2-simplex) yields a
different rank-2 signal / graph-level readout than one *without* it, because the
2-cell becomes a first-class object that boundary + upper-adjacency message
passing can read. Plain 1-WL GNNs provably cannot do this.
"""

import torch

from topobench.nn.backbones.simplicial.mpsn import MPSN


# --------------------------------------------------------------------------- #
# Helpers: tiny hand-built complexes (unsigned 0/1 incidences, as the clique  #
# lifting produces with signed=False).                                         #
# --------------------------------------------------------------------------- #
def _triangle_complex():
    """A single filled triangle on nodes {0,1,2}.

    Nodes: 0,1,2 ; Edges: (0,1),(0,2),(1,2) ; Triangles: (0,1,2).

    Returns
    -------
    tuple
        ((x_0, x_1, x_2), (incidence_1, incidence_2)).
    """
    # incidence_1: nodes x edges  (node belongs to edge -> 1)
    #   edge0=(0,1), edge1=(0,2), edge2=(1,2)
    incidence_1 = torch.tensor(
        [
            [1.0, 1.0, 0.0],  # node 0 in e0, e1
            [1.0, 0.0, 1.0],  # node 1 in e0, e2
            [0.0, 1.0, 1.0],  # node 2 in e1, e2
        ]
    )
    # incidence_2: edges x triangles  (edge belongs to triangle -> 1)
    #   tri0 = (0,1,2) -> all three edges
    incidence_2 = torch.tensor([[1.0], [1.0], [1.0]])

    x_0 = torch.arange(1, 4, dtype=torch.float).unsqueeze(1)  # (3, 1)
    x_1 = torch.arange(1, 4, dtype=torch.float).unsqueeze(1)  # (3, 1)
    x_2 = torch.ones(1, 1)  # (1, 1)
    return (x_0, x_1, x_2), (incidence_1, incidence_2)


def _open_triad_complex():
    """The SAME three nodes / three edges but the triangle is NOT filled.

    Identical 1-skeleton to ``_triangle_complex`` but with zero 2-simplices, so
    the only structural difference is the presence of the 2-cell.

    Returns
    -------
    tuple
        ((x_0, x_1, x_2), (incidence_1, incidence_2)).
    """
    incidence_1 = torch.tensor(
        [
            [1.0, 1.0, 0.0],
            [1.0, 0.0, 1.0],
            [0.0, 1.0, 1.0],
        ]
    )
    # No triangles: edges x 0-triangles.
    incidence_2 = torch.zeros(3, 0)

    x_0 = torch.arange(1, 4, dtype=torch.float).unsqueeze(1)
    x_1 = torch.arange(1, 4, dtype=torch.float).unsqueeze(1)
    x_2 = torch.zeros(0, 1)
    return (x_0, x_1, x_2), (incidence_1, incidence_2)


def _make_model(in_channels_all=(1, 1, 1), hidden_dim=8, n_layers=3):
    """Construct an MPSN backbone with fixed seed for reproducibility.

    Parameters
    ----------
    in_channels_all : tuple of int
        Input channels on (nodes, edges, triangles).
    hidden_dim : int
        Hidden width (shared across ranks).
    n_layers : int
        Number of MPSN layers.

    Returns
    -------
    MPSN
        The constructed backbone.
    """
    torch.manual_seed(0)
    return MPSN(
        in_channels_all=in_channels_all,
        hidden_dim=hidden_dim,
        n_layers=n_layers,
    )


# --------------------------------------------------------------------------- #
# Tests                                                                        #
# --------------------------------------------------------------------------- #
def test_per_rank_output_shapes():
    """Forward returns one tensor per rank with hidden_dim width."""
    hidden_dim = 8
    model = _make_model(hidden_dim=hidden_dim)
    x_all, inc_all = _triangle_complex()

    x_0, x_1, x_2 = model(x_all, inc_all)

    assert x_0.shape == (3, hidden_dim)
    assert x_1.shape == (3, hidden_dim)
    assert x_2.shape == (1, hidden_dim)


def test_forward_runs_end_to_end_and_is_finite():
    """The forward pass runs on a toy complex and produces finite outputs."""
    model = _make_model()
    x_all, inc_all = _triangle_complex()

    outs = model(x_all, inc_all)

    assert len(outs) == 3
    assert all(torch.isfinite(o).all() for o in outs)


def test_learnable_epsilon_and_rank_specific_mlps_exist():
    """Each layer has a learnable per-rank epsilon (init 0) and 3 distinct MLPs."""
    n_layers = 3
    model = _make_model(n_layers=n_layers)

    assert len(model.layers) == n_layers
    for layer in model.layers:
        # learnable epsilons, one per rank, init 0.0
        for r in range(3):
            eps = getattr(layer, f"eps_{r}")
            assert isinstance(eps, torch.nn.Parameter)
            assert eps.requires_grad
            assert eps.shape == torch.Size([])
            assert float(eps) == 0.0
        # one distinct MLP per rank
        mlp_ids = {id(getattr(layer, f"mlp_{r}")) for r in range(3)}
        assert len(mlp_ids) == 3


def test_triangle_presence_changes_rank2_and_readout():
    """A filled triangle yields a different rank-2 signal and graph readout
    than the identical open 1-skeleton without the 2-cell.

    This is the SWL > 1-WL capability: the 2-cell is read by the model. Two
    complexes with identical node/edge structure but differing only in whether
    the triangle is filled MUST produce different higher-rank signals; a 1-WL
    GNN sees no difference.
    """
    model = _make_model()

    x_tri, inc_tri = _triangle_complex()
    x_open, inc_open = _open_triad_complex()

    out_tri = model(x_tri, inc_tri)
    out_open = model(x_open, inc_open)

    # Rank-2 signal exists only when the triangle is filled.
    assert out_tri[2].shape[0] == 1
    assert out_open[2].shape[0] == 0

    # Graph-level sum readout over all ranks differs because the 2-cell feeds in.
    def _graph_readout(outs):
        return sum(o.sum() for o in outs)

    assert not torch.allclose(_graph_readout(out_tri), _graph_readout(out_open))

    # The edge (rank-1) signal also differs: upper-adjacency among edges is
    # routed through the shared triangle, present only in the filled complex.
    assert not torch.allclose(out_tri[1], out_open[1])


def test_upper_adjacency_uses_shared_coface_feature():
    """The rank-1 upper message reads the shared triangle's feature.

    Two complexes identical except for the *feature value* on the filled
    triangle must produce different edge embeddings, proving the upper message
    incorporates h_{shared coface} (paper's M_up term), not just the
    edge-edge adjacency pattern.
    """
    model = _make_model()

    x_all, inc_all = _triangle_complex()
    (x_0, x_1, x_2), (inc_1, inc_2) = x_all, inc_all

    out_a = model((x_0, x_1, x_2), (inc_1, inc_2))
    # Same topology, different triangle feature.
    x_2_b = x_2 + 5.0
    out_b = model((x_0, x_1, x_2_b), (inc_1, inc_2))

    # Edge embeddings depend on the triangle feature via the upper message.
    assert not torch.allclose(out_a[1], out_b[1])


def test_node_upper_adjacency_through_shared_edge():
    """Node (rank-0) embeddings depend on edge features via upper-adjacency.

    Nodes have no boundary; their only neighbourhood is upper-adjacency through
    a shared edge whose feature the message reads. Changing an edge feature must
    change node embeddings.
    """
    model = _make_model()
    (x_0, x_1, x_2), (inc_1, inc_2) = _triangle_complex()

    out_a = model((x_0, x_1, x_2), (inc_1, inc_2))
    x_1_b = x_1 + 3.0
    out_b = model((x_0, x_1_b, x_2), (inc_1, inc_2))

    assert not torch.allclose(out_a[0], out_b[0])


def test_n_layers_configurable():
    """n_layers controls the number of stacked MPSN layers."""
    assert len(_make_model(n_layers=1).layers) == 1
    assert len(_make_model(n_layers=4).layers) == 4
