"""This module implements the SimplicialGraphInducedLifting class, which lifts graphs to simplicial complexes."""

from itertools import combinations

import networkx as nx
import torch_geometric
from toponetx.classes import SimplicialComplex

from topobench.transforms.liftings.graph2simplicial.base import (
    Graph2SimplicialLifting,
)


class SimplicialGraphInducedLifting(Graph2SimplicialLifting):
    r"""Lift graphs to simplicial complex domain by identifying connected subgraphs as simplices.

    Parameters
    ----------
    **kwargs : optional
        Additional arguments for the class.
    """

    def __init__(self, **kwargs):
        super().__init__(**kwargs)

    def lift_topology(self, data: torch_geometric.data.Data) -> dict:
        r"""Lift the topology of a graph to a simplicial complex by identifying connected subgraphs as simplices.

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
        all_nodes = list(graph.nodes)
        simplices = [set() for _ in range(2, self.complex_dim + 1)]

        for k in range(2, self.complex_dim + 1):
            for combination in combinations(all_nodes, k + 1):
                subgraph = graph.subgraph(combination)
                if nx.is_connected(subgraph):
                    simplices[k - 2].add(tuple(sorted(combination)))

        for set_k_simplices in simplices:
            simplicial_complex.add_simplices_from(list(set_k_simplices))

        return self._get_lifted_topology(simplicial_complex, graph)
