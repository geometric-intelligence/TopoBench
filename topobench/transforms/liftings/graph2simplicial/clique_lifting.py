"""This module implements the CliqueLifting class, which lifts graphs to simplicial complexes."""

from itertools import combinations
from typing import Any

import networkx as nx
import torch_geometric
from toponetx.classes import SimplicialComplex

from topobench.transforms.liftings.graph2simplicial import (
    Graph2SimplicialLifting,
)


class SimplicialCliqueLifting(Graph2SimplicialLifting):
    r"""Lift graphs to simplicial complex domain.

    The algorithm creates simplices by identifying the cliques and considering them as simplices of the same dimension.

    Parameters
    ----------
    **kwargs : optional
        Additional arguments for the class.
    """

    def __init__(self, **kwargs):
        super().__init__(**kwargs)

    def lift_topology(self, data: torch_geometric.data.Data) -> dict:
        r"""Lift the topology of a graph to a simplicial complex.

        Parameters
        ----------
        data : torch_geometric.data.Data
            The input data to be lifted.

        Returns
        -------
        dict
            The lifted topology.
        """
        graph = self._generate_graph_from_data(data)
        simplicial_complex = SimplicialComplex(graph)
        cliques = nx.find_cliques(graph)
        simplices: list[set[tuple[Any, ...]]] = [
            set() for _ in range(2, self.complex_dim + 1)
        ]
        for clique in cliques:
            for i in range(2, self.complex_dim + 1):
                for c in combinations(clique, i + 1):
                    simplices[i - 2].add(tuple(c))

        for set_k_simplices in simplices:
            simplicial_complex.add_simplices_from(list(set_k_simplices))

        return self._get_lifted_topology(simplicial_complex, graph)
