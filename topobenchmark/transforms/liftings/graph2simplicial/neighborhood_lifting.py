r"""This module implements the neighborhood/Dowker lifting.

This lifting constructs a neighborhood simplicial complex as it is
`usually defined <https://mathworld.wolfram.com/NeighborhoodComplex.html>`_
in the field of topological combinatorics.
In this lifting, for each vertex in the original graph, its neighborhood is
the subset of adjacent vertices, and the simplices are these subsets.

That is, if :math:`G = (V, E)` is a graph, then its neighborhood complex
:math:`N(G)` is a simplicial complex with the vertex set :math:`V` and
simplices given by subsets :math:`A \subseteq V` such, that
:math:`\forall a \in A ; \exists v:(a, v) \in E`.
That is, say, 3 vertices form a simplex iff there's another vertex which
is adjacent to each of these 3 vertices.

This construction differs from
`another lifting <https://github.com/pyt-team/challenge-icml-2024/pull/5>`_
with the similar naming.
The difference is, for example, that in this construction the edges of an
original graph doesn't present as the edges in the simplicial complex.

This lifting is a
`Dowker construction <https://ncatlab.org/nlab/show/Dowker%27s+theorem>`_
since an edge between two vertices in the graph can be considered as a
symmetric binary relation between these vertices.
"""

import torch
import torch_geometric
from toponetx.classes import SimplicialComplex

from topobenchmark.transforms.liftings.base import LiftingMap


class NeighborhoodComplexLifting(LiftingMap):
    r"""Lifts graphs to simplicial complex domain by constructing the neighborhood complex."""

    def lift(self, domain):
        r"""Lift the topology to simplicial complex domain.

        Parameters
        ----------
        domain : torch_geometric.data.Data
            The input data to be lifted.

        Returns
        -------
        toponetx.SimplicialComplex
            Lifted simplicial complex.
        """
        undir_edge_index = torch_geometric.utils.to_undirected(
            domain.edge_index
        )

        simplices = [
            set(
                undir_edge_index[1, j].tolist()
                for j in torch.nonzero(undir_edge_index[0] == i).squeeze()
            )
            for i in torch.unique(undir_edge_index[0])
        ]

        node_features = {i: domain.x[i, :] for i in range(domain.x.shape[0])}

        simplicial_complex = SimplicialComplex(simplices)
        simplicial_complex.set_simplex_attributes(node_features, name="x")

        return simplicial_complex
