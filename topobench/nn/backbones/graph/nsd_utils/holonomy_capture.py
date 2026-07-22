"""Capture learned per-edge transport maps from the sheaf diffusion/propagation models.

The diagonal / bundle / general sheaf models rebuild their Laplacian on every
forward pass and stash the *un-normalised* off-diagonal blocks ``-F_a^T F_b`` on
each sheaf learner (via ``SheafLearner.set_L``). For O(d) (bundle) restriction
maps this block is exactly the orthogonal parallel-transport matrix, so composing
it around a cycle yields the sheaf holonomy consumed by
:mod:`topobench.nn.backbones.graph.nsd_utils.sheaf_holonomy`.

This module reads those blocks back out and packages them into the
``{(dst, src): matrix}`` transport dict that the holonomy readouts expect. The
typical flow is to run one forward pass and then read the transports off the
propagation model, enumerate the triangles, and batch the readouts::

    enc = NSPEncoder(input_dim=..., hidden_dim=..., d=2, sheaf_type="bundle")
    enc(x, edge_index)                       # populates sheaf_learners[*].L
    prop = enc.get_sheaf_propagation_model()
    transports = sheaf_transports(prop, edge_index)
    readouts = triangle_holonomies(transports, enumerate_triangles(edge_index))

For diagonal restriction maps the block stores only the ``d`` diagonal entries
(the connection is abelian / flat up to a sign), so the returned matrices are
``torch.diag`` of those entries and the meaningful cycle readout is the Z2
frustration sign rather than a rotation angle.
"""

import torch
from torch_geometric.utils import to_undirected

from .laplace import compute_left_right_map_index


def blocks_to_transports(blocks, vertex_tril_idx):
    """
    Package saved off-diagonal Laplacian blocks into a transport dict.

    Parameters
    ----------
    blocks : torch.Tensor
        The ``-F_a^T F_b`` blocks stored by a Laplacian builder: shape
        ``[E, d]`` for diagonal maps (only the diagonal is stored) or
        ``[E, d, d]`` for bundle / general maps.
    vertex_tril_idx : torch.Tensor
        Long tensor of shape ``[2, E]`` whose column ``i`` holds the ``(a, b)``
        node pair of ``blocks[i]``, as returned by
        :func:`compute_left_right_map_index`.

    Returns
    -------
    dict
        Maps a directed edge ``(dst, src)`` to its ``[d, d]`` transport matrix
        ``F_a^T F_b`` (which carries a stalk vector ``b -> a``). Both
        orientations of every stored edge are inserted -- the reverse is the
        transpose -- so any cycle walk can resolve every step.
    """
    a_nodes = vertex_tril_idx[0].tolist()
    b_nodes = vertex_tril_idx[1].tolist()
    transports = {}
    for i, (a, b) in enumerate(zip(a_nodes, b_nodes, strict=True)):
        block = blocks[i]
        mat = torch.diag(block) if block.dim() == 1 else block
        transport = -mat  # -(-F_a^T F_b) = F_a^T F_b : carries b -> a
        transports[(a, b)] = transport
        transports[(b, a)] = transport.transpose(-1, -2)
    return transports


def sheaf_transports(propagation_model, edge_index, layer=-1):
    """
    Read the per-edge transport maps a sheaf model learned on its last forward.

    Parameters
    ----------
    propagation_model : torch.nn.Module
        An ``InductiveDiscrete*Sheaf{Diffusion,Propagation}`` instance whose
        ``forward`` has already been called, so ``sheaf_learners[*].L`` is set.
    edge_index : torch.Tensor
        The *same* edge index passed to the encoder. It is symmetrised with
        :func:`torch_geometric.utils.to_undirected` here to reproduce the
        builder's node ordering.
    layer : int, optional
        Which sheaf learner's transports to read. Default ``-1`` (the last,
        most-evolved layer). Fixed-geometry models keep a single learner.

    Returns
    -------
    dict
        The ``{(dst, src): matrix}`` transport dict for the requested layer.

    Raises
    ------
    RuntimeError
        If the model has not been run (no stored ``L``), or the stored block
        count does not match the graph's undirected edge count (a sign the
        wrong ``edge_index`` was supplied).
    """
    learner = propagation_model.sheaf_learners[layer]
    blocks = learner.L
    if blocks is None:
        raise RuntimeError(
            "propagation_model has no stored transports; "
            "run a forward pass first"
        )
    undirected = to_undirected(edge_index)
    _, vertex_tril_idx = compute_left_right_map_index(undirected)
    if vertex_tril_idx.size(1) != blocks.size(0):
        raise RuntimeError(
            f"stored block count ({blocks.size(0)}) does not match the number "
            f"of undirected edges ({vertex_tril_idx.size(1)}); pass the same "
            "edge_index the model was called with"
        )
    return blocks_to_transports(blocks, vertex_tril_idx)
