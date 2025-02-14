"""This module implements the CliqueLifting class, which lifts graphs to simplicial complexes."""

from itertools import combinations

import networkx as nx
from toponetx.classes import SimplicialComplex

from topobenchmark.transforms.liftings.base import LiftingMap


class SimplicialCliqueLifting(LiftingMap):
    r"""Lift graphs to simplicial complex domain.

    The algorithm creates simplices by identifying the cliques
    and considering them as simplices of the same dimension.

    Parameters
    ----------
    complex_dim : int
        Dimension of the subcomplex.
    """

    def __init__(self, complex_dim=2):
        super().__init__()
        self.complex_dim = complex_dim

    def lift(self, domain):
        r"""Lift the topology of a graph to a simplicial complex.

        Parameters
        ----------
        domain : nx.Graph
            Graph to be lifted.

        Returns
        -------
        toponetx.SimplicialComplex
            Lifted simplicial complex.
        """
        graph = domain

        simplicial_complex = SimplicialComplex(graph)
        cliques = nx.find_cliques(graph)
        simplices = [set() for _ in range(2, self.complex_dim + 1)]
        for clique in cliques:
            for i in range(2, self.complex_dim + 1):
                for c in combinations(clique, i + 1):
                    simplices[i - 2].add(tuple(c))

        for set_k_simplices in simplices:
            simplicial_complex.add_simplices_from(list(set_k_simplices))

        # because ComplexData pads unexisting dimensions with empty matrices
        simplicial_complex.practical_dim = self.complex_dim

        return simplicial_complex
